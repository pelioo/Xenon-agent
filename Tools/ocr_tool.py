"""
OCR 工具 - 基于 PaddleOCR 3.x / PP-OCRv6 的离线文字识别
=========================================================
支持单张图片、批量图片、目录扫描的 OCR 识别。

依赖: pip install paddlepaddle==3.2.2 paddleocr
首次运行会自动从 ModelScope 下载模型到本地缓存，之后完全离线。

用法（代码中）:
    from Tools.ocr_tool import OCRToolManager
    ocr = OCRToolManager()
    result = ocr.ocr_image("图片路径")

用法（命令行）:
    python ocr_tool.py ocr_image <图片路径>
    python ocr_tool.py ocr_images '[路径1, 路径2]'
    python ocr_tool.py ocr_directory <目录路径> [--recursive]
    python ocr_tool.py ocr_text <图片路径>
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# ---------------------------------------------------------------------------
# PaddleX 缓存目录 — 必须在 import paddleocr/paddlex 之前设置
# ---------------------------------------------------------------------------
# PaddleX 默认把模型缓存写到 ~/.paddlex（C 盘用户目录）。这里统一重定向到
# <项目根>/models/paddlex_cache/（由本文件位置推导，换电脑/换目录自动适配）。
# 本地模型齐全时不会触发下载；万一有漏网模型，也只下载到项目内，不碰 C 盘。
_PDX_CACHE_HOME = Path(__file__).resolve().parent.parent / "models" / "paddlex_cache"
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(_PDX_CACHE_HOME))
# 跳过 PaddleX 启动时的模型源连通性检查（可离线、加快启动）
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# ---------------------------------------------------------------------------
# 依赖检查
# ---------------------------------------------------------------------------
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
    ".webp",
}
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
DEFAULT_OCR_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "ocr"
DEFAULT_OCR_TABLE_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "ocr_tables"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_OCR_PYTHON = PROJECT_ROOT / "venv_ocr" / "Scripts" / "python.exe"
GLOBAL_PYTHON_313 = Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python313" / "python.exe"
_PYTHON_BACKEND_CACHE: Dict[str, Dict[str, Any]] = {}
PADDLEOCR_LANG_MAP = {
    "ch": "ch",                     # 中文简体
    "en": "en",                     # 英文
    "chinese_cht": "chinese_cht",   # 中文繁体
    "japan": "japan",               # 日文
    "korean": "korean",             # 韩文
    "fr": "fr",                     # 法文
    "de": "de",                     # 德文
}

# ── 本地模型目录（<项目根>/models/ 下，可移植，不依赖绝对路径）──
# PaddleOCR 各子模型的目录名 → PaddleOCR 构造参数名
OCR_MODEL_DIR_MAP = {
    "text_detection_model_dir": "PP-OCRv6_medium_det",
    "text_recognition_model_dir": "PP-OCRv6_medium_rec",
    "doc_orientation_classify_model_dir": "PP-LCNet_x1_0_doc_ori",
    "doc_unwarping_model_dir": "UVDoc",
    "textline_orientation_model_dir": "PP-LCNet_x1_0_textline_ori",
}

# PP-StructureV3 表格识别附加模型（目录名 → PPStructureV3 构造参数名）
TABLE_MODEL_DIR_MAP = {
    "layout_detection_model_dir": "PP-DocLayout_plus-L",
    "table_classification_model_dir": "PP-LCNet_x1_0_table_cls",
    # 表格方向分类：PaddleX 默认模型就是 doc_ori，本地直接复用
    "table_orientation_classify_model_dir": "PP-LCNet_x1_0_doc_ori",
    "wired_table_structure_recognition_model_dir": "SLANeXt_wired",
    "wireless_table_structure_recognition_model_dir": "SLANet_plus",
    "wired_table_cells_detection_model_dir": "RT-DETR-L_wired_table_cell_det",
    "wireless_table_cells_detection_model_dir": "RT-DETR-L_wireless_table_cell_det",
}


def _resolve_ocr_model_kwargs() -> Dict[str, str]:
    """解析本地 OCR 模型目录，返回可传给 PaddleOCR 的 model_dir 参数。

    模型统一放在 <项目根>/models/paddleocr/ 下（由本文件位置推导，
    换电脑/换目录自动适配）。若本地模型缺失，返回空 dict →
    PaddleX 回退到默认逻辑（自动下载/缓存）。
    """
    return _resolve_paddle_model_kwargs(OCR_MODEL_DIR_MAP)


def _resolve_table_model_kwargs() -> Dict[str, str]:
    """解析 PP-StructureV3 表格识别所需的全部本地模型目录。

    包含基础 OCR 模型（det/rec/cls/unwarping/ori）和表格专用模型
    （版面检测/表格方向/有线无线表格结构/单元格检测）。
    模型统一放在 <项目根>/models/paddleocr/ 下，可移植、无绝对路径。
    缺失时跳过该项，由 PaddleX 回退默认逻辑。
    """
    return _resolve_paddle_model_kwargs(
        {**OCR_MODEL_DIR_MAP, **TABLE_MODEL_DIR_MAP}
    )


def _resolve_paddle_model_kwargs(model_map: Dict[str, str]) -> Dict[str, str]:
    """按 model_map（参数名 → 目录名）解析本地模型目录。"""
    base = Path(__file__).resolve().parent.parent / "models"
    ocr_dir = base / "paddleocr"
    # 优先 models/paddleocr/，兼容旧位置 models/ 根目录
    search_dirs = [ocr_dir] if ocr_dir.is_dir() else [base]
    kwargs: Dict[str, str] = {}
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for param, folder in model_map.items():
            if param in kwargs:
                continue
            model_dir = search_dir / folder
            if model_dir.is_dir() and (model_dir / "inference.yml").exists():
                kwargs[param] = str(model_dir)
    return kwargs

# ── 异步 OCR 默认配置 ──
OCR_STATE_FILE = Path("Memory/ocr_tasks.json")
MAX_OCR_TASK_HISTORY = 50
MAX_OCR_THREADS = 4                 # OCR 是 CPU/GPU 密集型，不宜过多


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def is_image_file(file_path: str) -> bool:
    """判断是否为支持的图片文件"""
    return Path(file_path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def _safe_output_stem(file_path: str) -> str:
    """生成适合保存 OCR 结果的文件名前缀"""
    stem = Path(file_path).stem or "image"
    for ch in '<>:"/\\|?*':
        stem = stem.replace(ch, "_")
    return stem.strip() or "image"


def _extract_result_text(results: Dict[str, Any]) -> str:
    if isinstance(results.get("result"), dict):
        return str(results["result"].get("text", ""))
    if isinstance(results.get("text"), str):
        return results["text"]
    if isinstance(results.get("results"), list):
        text_parts = []
        for item in results["results"]:
            if isinstance(item, dict):
                text_parts.append(str(item.get("text", "")))
        return "\n\n".join(part for part in text_parts if part)
    return ""


def _save_ocr_auto_output(results: Dict[str, Any], image_path: str,
                          output_dir: Optional[str] = None) -> Dict[str, Any]:
    """自动将 OCR 结果写入指定目录（默认 output/ocr），避免大段返回内容被上层拦截后丢失。

    Args:
        results: OCR 结果字典
        image_path: 图片路径（用于生成文件名前缀）
        output_dir: 自定义保存目录；为 None 时使用默认 output/ocr/
    """
    target_dir = Path(output_dir) if output_dir else DEFAULT_OCR_OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = _safe_output_stem(image_path)
    json_path = target_dir / f"{stem}_{timestamp}.json"
    txt_path = target_dir / f"{stem}_{timestamp}.txt"
    latest_json_path = DEFAULT_OUTPUT_DIR / "latest_ocr_result.json"
    latest_txt_path = DEFAULT_OUTPUT_DIR / "latest_ocr_result.txt"

    text = _extract_result_text(results)
    json_text = json.dumps(results, ensure_ascii=False, indent=2, default=str)
    json_path.write_text(json_text, encoding="utf-8")
    txt_path.write_text(text, encoding="utf-8")
    latest_json_path.write_text(json_text, encoding="utf-8")
    latest_txt_path.write_text(text, encoding="utf-8")

    return {
        "json": str(json_path),
        "txt": str(txt_path),
        "latest_json": str(latest_json_path),
        "latest_txt": str(latest_txt_path),
    }


def _attach_auto_output(results: Dict[str, Any], image_path: str,
                        output_dir: Optional[str] = None) -> Dict[str, Any]:
    if not results.get("success"):
        return results
    try:
        results["output_files"] = _save_ocr_auto_output(results, image_path, output_dir=output_dir)
    except Exception as e:
        results["output_save_error"] = f"保存 OCR 结果失败: {str(e)}"
    return results


# ---------------------------------------------------------------------------
# 表格识别 — 结果标准化 / HTML / Excel 导出
# ---------------------------------------------------------------------------
def _parse_table_html_to_matrix(html_str: str) -> List[List[str]]:
    """解析表格 HTML 为二维矩阵（处理 rowspan/colspan 展开）。"""
    from html.parser import HTMLParser

    class _TableParser(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.rows = []
            self.cur_row = None
            self.cur_cell = None
            self.in_cell = False
            self.colspan = 1
            self.rowspan = 1
            self.cell_texts = []

        def handle_starttag(self, tag, attrs):
            tag = tag.lower()
            if tag == "tr":
                self.cur_row = []
            elif tag in ("td", "th"):
                self.in_cell = True
                self.cell_texts = []
                d = dict(attrs)
                try:
                    self.colspan = max(1, int(d.get("colspan", 1)))
                except ValueError:
                    self.colspan = 1
                try:
                    self.rowspan = max(1, int(d.get("rowspan", 1)))
                except ValueError:
                    self.rowspan = 1

        def handle_data(self, data):
            if self.in_cell:
                self.cell_texts.append(data)

        def handle_endtag(self, tag):
            tag = tag.lower()
            if tag in ("td", "th") and self.cur_row is not None:
                text = "".join(self.cell_texts).replace("\xa0", " ").strip()
                self.cur_row.append({
                    "text": text, "colspan": self.colspan, "rowspan": self.rowspan,
                })
                self.in_cell = False
            elif tag == "tr" and self.cur_row is not None:
                self.rows.append(self.cur_row)
                self.cur_row = None

    parser = _TableParser()
    try:
        parser.feed(html_str)
    except Exception:
        return []

    # 展开 rowspan/colspan 为完整矩阵
    grid: List[List[str]] = []
    occupancy: Dict[int, int] = {}  # 列号 -> 剩余占用行数
    for row in parser.rows:
        out_row: List[str] = []
        col = 0
        for cell in row:
            while occupancy.get(col, 0) > 0:
                out_row.append("")
                occupancy[col] -= 1
                col += 1
            out_row.append(cell["text"])
            if cell["colspan"] > 1:
                for _ in range(cell["colspan"] - 1):
                    out_row.append("")
                    col += 1
            if cell["rowspan"] > 1:
                for c in range(col, col + cell["colspan"]):
                    occupancy[c] = max(occupancy.get(c, 0), cell["rowspan"] - 1)
            col += 1
        # 行尾补齐被占用的列
        while occupancy.get(col, 0) > 0:
            out_row.append("")
            occupancy[col] -= 1
            col += 1
        grid.append(out_row)
    return grid


def _merge_table_htmls(html_parts: List[str]) -> str:
    """合并多个表格 HTML 为一个完整页面。"""
    import re

    tables = []
    for h in html_parts:
        m = re.search(r"(<table.*?</table>)", h, re.S | re.I)
        tables.append(m.group(1) if m else h)
    return (
        '<html><head><meta charset="utf-8"></head><body>'
        + "\n<br/><br/>\n".join(tables)
        + "</body></html>"
    )


def _matrices_to_excel(out_path: str, matrices: List[List[List[str]]]) -> None:
    """多个二维矩阵写入 Excel，每个矩阵一个工作表。"""
    import openpyxl

    wb = openpyxl.Workbook()
    for i, matrix in enumerate(matrices, 1):
        ws = wb.active if i == 1 else wb.create_sheet()
        ws.title = f"表格{i}"
        for row in matrix:
            ws.append([str(c) if c is not None else "" for c in row])
    wb.save(out_path)


def _normalize_table_result(paddle_result: List[dict]) -> Dict[str, Any]:
    """将 PPStructureV3 的 predict() 结果标准化。

    PaddleX 返回每个元素为 dict，关键字段:
        table_res_list: [{pred_html, table_bbox_list/bbox, ...}, ...]
        rec_texts / rec_scores / dt_polys: 整页 OCR 文本

    标准化输出:
        {
            "file": str,
            "table_count": int,
            "tables": [{"index", "rows", "cols", "matrix", "html", "preview"}],
            "text": 整页纯文本（用于对照）,
            "line_count": int,
            "confidence": float,
        }
    """
    if not paddle_result or not isinstance(paddle_result, list):
        return {"file": "", "table_count": 0, "tables": [], "text": "",
                "line_count": 0, "confidence": 0.0}

    page = paddle_result[0] if isinstance(paddle_result[0], dict) else {}
    table_res_list = page.get("table_res_list") or []
    tables = []
    for i, tr in enumerate(table_res_list, 1):
        pred_html = str(tr.get("pred_html", "") or "")
        matrix = _parse_table_html_to_matrix(pred_html) if pred_html else []
        rows = len(matrix)
        cols = max((len(r) for r in matrix), default=0)
        tables.append({
            "index": i,
            "rows": rows,
            "cols": cols,
            "matrix": matrix,
            "html": pred_html,
            "preview": "\n".join(" | ".join(r) for r in matrix[:8]),
        })

    # 整页 OCR 文本（表格外的文字，用于对照）
    rec_texts = page.get("rec_texts") or []
    rec_scores = page.get("rec_scores") or []
    confidences = [float(s) for s in rec_scores if s is not None]
    return {
        "file": str(page.get("input_path", "")),
        "table_count": len(tables),
        "tables": tables,
        "text": "\n".join(str(t) for t in rec_texts),
        "line_count": len(rec_texts),
        "confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
    }


def _save_table_auto_output(result: Dict[str, Any], image_path: str,
                            output_dir: Optional[str] = None) -> Dict[str, Any]:
    """将表格识别结果保存（默认 output/ocr_tables/，可指定目录），html + xlsx + txt + json。"""
    target_dir = Path(output_dir) if output_dir else DEFAULT_OCR_TABLE_OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_output_stem(image_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    html_path = target_dir / f"{stem}_{timestamp}.html"
    xlsx_path = target_dir / f"{stem}_{timestamp}.xlsx"
    txt_path = target_dir / f"{stem}_{timestamp}.txt"
    json_path = target_dir / f"{stem}_{timestamp}.json"

    tables = result.get("tables", [])
    html_parts = [t["html"] for t in tables if t.get("html", "").strip()]
    matrices = [t["matrix"] for t in tables if t.get("matrix")]

    files = {}
    if html_parts:
        html_path.write_text(_merge_table_htmls(html_parts), encoding="utf-8")
        files["html"] = str(html_path)
    if matrices:
        try:
            _matrices_to_excel(str(xlsx_path), matrices)
            files["xlsx"] = str(xlsx_path)
        except Exception as e:
            files["xlsx_error"] = f"Excel 导出失败: {str(e)}"
    txt_path.write_text(result.get("text", ""), encoding="utf-8")
    files["txt"] = str(txt_path)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    files["json"] = str(json_path)
    return files


def _compact_tool_output(result: Dict[str, Any], max_preview: int = 500) -> Dict[str, Any]:
    """压缩工具返回结果：完整文本落盘，上下文只留摘要。

    识别结果可能很大（尤其整页文字 + 每个字的 bbox），直接返回会
    挤爆上下文。这里把 text/lines/matrix 替换为摘要，并保留
    output_files 路径，调用方需要全文时按路径读取。
    """
    if not isinstance(result, dict) or not result.get("success"):
        return result
    if "gpu_info" in result or "available_languages" in result:
        return result

    compact = {k: v for k, v in result.items() if k not in ("result", "results")}

    # 单图结果
    if isinstance(result.get("result"), dict):
        r = result["result"]
        text = str(r.get("text", ""))
        compact["result"] = {
            "file": r.get("file"),
            "line_count": r.get("line_count"),
            "confidence": r.get("confidence"),
            "text_preview": text[:max_preview],
            "text_total_chars": len(text),
        }
        # 表格识别结果：保留表格概要
        if "table_count" in r:
            compact["result"]["table_count"] = r["table_count"]
            compact["result"]["tables"] = [
                {
                    "index": t.get("index"),
                    "rows": t.get("rows"),
                    "cols": t.get("cols"),
                    "preview": (t.get("preview") or "")[:max_preview],
                }
                for t in r.get("tables", [])
            ]
            out_files = result.get("output_files") or {}
            save_hint = out_files.get("html") or out_files.get("xlsx") or out_files.get("json")
            compact["message"] = (
                f"表格识别完成，共 {r.get('table_count', 0)} 个表格，"
                f"结果已保存: {save_hint or 'output/ocr_tables/'}"
            )
    # 批量结果
    elif isinstance(result.get("results"), list):
        compact["results"] = [
            {
                "file": r.get("file"),
                "line_count": r.get("line_count"),
                "confidence": r.get("confidence"),
                "text_preview": str(r.get("text", ""))[:max_preview],
            }
            for r in result["results"] if isinstance(r, dict)
        ]
        compact["result_count"] = len(compact["results"])

    # 纯文本接口
    if isinstance(result.get("text"), str):
        text = result["text"]
        compact["text"] = text[:max_preview]
        compact["text_total_chars"] = len(text)

    return compact


def _compact_cli_output(results: Dict[str, Any]) -> Dict[str, Any]:
    """命令行默认只返回摘要，完整文本请读取 output 中的落盘文件"""
    if not isinstance(results, dict) or not results.get("success"):
        return results
    if "gpu_info" in results or "available_languages" in results:
        return results
    # 已被 _compact_tool_output 压缩过（工具层摘要）→ 直接透传
    if isinstance(results.get("result"), dict) and "text_preview" in results["result"]:
        return results
    if isinstance(results.get("results"), list) and results["results"] and \
            all(isinstance(r, dict) and "text_preview" in r for r in results["results"]):
        return results

    compact = {
        "success": True,
        "message": results.get("message", "OCR 完成，完整结果已保存到 output"),
    }
    for key in (
        "output_files", "output_save_error", "subprocess_python",
        "subprocess_backend", "total", "success_count", "error_count", "errors",
    ):
        if key in results:
            compact[key] = results[key]

    if isinstance(results.get("result"), dict):
        result = results["result"]
        compact["file"] = result.get("file")
        compact["confidence"] = result.get("confidence")
        compact["line_count"] = result.get("line_count")
        text = str(result.get("text", ""))
        compact["text_preview"] = text[:300]
    elif "text" in results:
        text = str(results.get("text", ""))
        compact["confidence"] = results.get("confidence")
        compact["line_count"] = results.get("line_count")
        compact["text_preview"] = text[:300]
    elif isinstance(results.get("results"), list):
        compact["result_count"] = len(results["results"])

    return compact


def _print_cli_result(results: Dict[str, Any]) -> None:
    if "--full" not in sys.argv:
        results = _compact_cli_output(results)
    print(json.dumps(results, ensure_ascii=False, default=str))


def _patch_paddlex_paddle_dependency_check() -> None:
    """兼容 paddlepaddle-gpu/元数据异常导致 PaddleX 误判未安装 paddlepaddle"""
    try:
        import paddle  # noqa: F401
    except Exception:
        return

    try:
        import paddlex.utils.deps as deps
    except Exception:
        return

    original = getattr(deps, "is_dep_available", None)
    if not callable(original) or getattr(original, "_xenon_ocr_patch", False):
        patched = original
    else:
        def patched(dep, /, check_version=False):
            if dep == "paddlepaddle":
                try:
                    import paddle  # noqa: F401
                    return True
                except Exception:
                    return False
            return original(dep, check_version=check_version)

        patched._xenon_ocr_patch = True
        deps.is_dep_available = patched

    # 部分 PaddleX 模块用 `from paddlex.utils.deps import is_dep_available`
    # 复制了函数引用，需要同步替换。
    for module_name in (
        "paddlex.inference.models.engines.paddle",
        "paddlex.utils.env",
        "paddlex.utils.device",
    ):
        try:
            module = __import__(module_name, fromlist=["is_dep_available"])
            if patched:
                module.is_dep_available = patched
        except Exception:
            pass


def _candidate_ocr_pythons() -> List[Path]:
    candidates = [GLOBAL_PYTHON_313, VENV_OCR_PYTHON]
    discovered = shutil.which("python")
    if discovered:
        candidates.append(Path(discovered))
    candidates.append(Path(sys.executable))

    unique = []
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        key = str(resolved).lower()
        if key not in seen and resolved.exists():
            unique.append(resolved)
            seen.add(key)
    unique.sort(key=_python_backend_sort_key, reverse=True)
    return unique


def _probe_python_backend(python_exe: Path) -> Dict[str, Any]:
    try:
        resolved = python_exe.resolve()
    except Exception:
        resolved = python_exe
    cache_key = str(resolved).lower()
    if cache_key in _PYTHON_BACKEND_CACHE:
        return _PYTHON_BACKEND_CACHE[cache_key]

    code = r'''
import json
import sys
out = {
    "executable": sys.executable,
    "python_version": sys.version.split()[0],
    "paddle_import": False,
    "compiled_cuda": False,
    "cuda_device_count": 0,
}
try:
    import paddle
    out["paddle_import"] = True
    out["paddle_version"] = getattr(paddle, "__version__", None)
    out["compiled_cuda"] = bool(getattr(paddle, "is_compiled_with_cuda", lambda: False)())
    if out["compiled_cuda"]:
        try:
            out["cuda_device_count"] = int(paddle.device.cuda.device_count())
            out["cuda_device_name"] = paddle.device.cuda.get_device_name()
        except Exception as e:
            out["cuda_error"] = str(e)
except Exception as e:
    out["error"] = repr(e)
print("__OCR_PROBE__" + json.dumps(out, ensure_ascii=False))
'''
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        result = subprocess.run(
            [str(resolved), "-X", "utf8", "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=30,
        )
        info = {
            "executable": str(resolved),
            "returncode": result.returncode,
            "compiled_cuda": False,
            "cuda_device_count": 0,
        }
        for line in reversed((result.stdout or "").splitlines()):
            if line.startswith("__OCR_PROBE__"):
                info.update(json.loads(line.removeprefix("__OCR_PROBE__")))
                break
        else:
            info["error"] = (result.stderr or result.stdout or "probe produced no marker")[-1000:]
    except Exception as e:
        info = {
            "executable": str(resolved),
            "compiled_cuda": False,
            "cuda_device_count": 0,
            "error": str(e),
        }

    _PYTHON_BACKEND_CACHE[cache_key] = info
    return info


def _python_backend_sort_key(python_exe: Path) -> tuple:
    info = _probe_python_backend(python_exe)
    return (
        1 if info.get("compiled_cuda") else 0,
        int(info.get("cuda_device_count") or 0),
        1 if info.get("paddle_import") else 0,
    )


def _has_gpu_capable_subprocess() -> bool:
    current = Path(sys.executable).resolve()
    for python_exe in _candidate_ocr_pythons():
        if python_exe == current:
            continue
        if _probe_python_backend(python_exe).get("compiled_cuda"):
            return True
    return False


def _parse_worker_result(stdout: str) -> Dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("__OCR_RESULT__"):
            return json.loads(line.removeprefix("__OCR_RESULT__"))
    raise json.JSONDecodeError("OCR worker result marker not found", stdout, 0)


# ---------------------------------------------------------------------------
# GPU 加速检测 — 自动识别可用后端，无需手动配置
# ---------------------------------------------------------------------------
def _detect_gpu_backend() -> str:
    """检测当前环境可用的 GPU 加速后端

    检测优先级: NVIDIA CUDA → AMD ROCm → 自定义设备 → CPU

    Returns:
        "nvidia_cuda" : NVIDIA GPU (CUDA) 可用
        "amd_rocm"    : AMD GPU (ROCm) 可用
        "custom_xxx"  : 其他自定义设备（如 DirectML）
        "cpu"         : 无可用的 GPU 加速
    """
    try:
        import paddle
        if hasattr(paddle, 'is_compiled_with_cuda') and paddle.is_compiled_with_cuda():
            return "nvidia_cuda"
        if hasattr(paddle, 'is_compiled_with_rocm') and paddle.is_compiled_with_rocm():
            return "amd_rocm"
        # 检查自定义设备（Intel Arc、昇腾、昆仑芯等）
        if hasattr(paddle, 'device') and hasattr(paddle.device, 'get_all_custom_device_type'):
            custom_devices = paddle.device.get_all_custom_device_type()
            if custom_devices:
                return f"custom_{custom_devices[0]}"
    except ImportError:
        pass
    except Exception:
        pass
    return "cpu"


def get_gpu_info() -> Dict[str, Any]:
    """获取当前环境的 GPU 加速能力详情"""
    backend = _detect_gpu_backend()
    info = {
        "backend": backend,
        "gpu_available": backend != "cpu",
        "python_executable": sys.executable,
        "description": {
            "nvidia_cuda": "NVIDIA GPU（CUDA）加速 ✓",
            "amd_rocm": "AMD GPU（ROCm）加速 ✓",
            "cpu": "CPU 模式（未检测到 GPU 加速）",
        }.get(backend, f"自定义设备加速: {backend}"),
    }
    try:
        info["ocr_python_candidates"] = [
            _probe_python_backend(candidate)
            for candidate in _candidate_ocr_pythons()
        ]
    except Exception as e:
        info["ocr_python_candidates_error"] = str(e)
    if backend == "nvidia_cuda":
        try:
            import paddle
            info["gpu_count"] = paddle.device.cuda.device_count()
            info["current_device"] = paddle.device.cuda.get_device_name()
        except Exception:
            pass
    elif backend == "amd_rocm":
        try:
            import paddle
            info["gpu_count"] = paddle.device.cuda.device_count()  # ROCm 也走 cuda 接口
        except Exception:
            pass
    return info


def _normalize_v3_result(paddle_result: List[dict]) -> Dict[str, Any]:
    """将 PaddleOCR 3.x 的 predict() 结果标准化

    PaddleOCR 3.x 返回格式（每个元素是一个 dict）:
        {
            'input_path': str,
            'page_index': None,
            'dt_polys': [array(4x2), ...],
            'rec_texts': ['文本1', '文本2'],
            'rec_scores': [0.99, 0.98],
            'rec_polys': [array(4x2), ...],
            'rec_boxes': array(N, 4),
            'textline_orientation_angles': [0, ...],
        }

    标准化输出:
        {
            "file": str,
            "text": "识别的纯文本\\n拼接",
            "lines": [{"text": str, "confidence": float, "bbox": [[x1,y1],...]}, ...],
            "confidence": float,
            "line_count": int,
        }
    """
    if not paddle_result or not isinstance(paddle_result, list):
        return {"file": "", "text": "", "lines": [], "confidence": 0.0, "line_count": 0}

    lines = []
    confidences = []

    for page_data in paddle_result:
        if not isinstance(page_data, dict):
            continue
        rec_texts = page_data.get("rec_texts") or []
        rec_scores = page_data.get("rec_scores") or []
        dt_polys = page_data.get("dt_polys") or []

        for i, text in enumerate(rec_texts):
            confidence = rec_scores[i] if i < len(rec_scores) else 0.0
            bbox = (dt_polys[i].tolist()
                    if i < len(dt_polys) and hasattr(dt_polys[i], "tolist")
                    else [])
            lines.append({
                "text": str(text),
                "confidence": round(float(confidence), 4),
                "bbox": bbox,
            })
            confidences.append(float(confidence))

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    full_text = "\n".join(l["text"] for l in lines)

    return {
        "file": paddle_result[0].get("input_path", "") if paddle_result else "",
        "text": full_text,
        "lines": lines,
        "confidence": round(avg_confidence, 4),
        "line_count": len(lines),
    }


# ---------------------------------------------------------------------------
# OCR 引擎 — 懒惰初始化单例
# ---------------------------------------------------------------------------
class _OCREngine:
    """PaddleOCR 引擎单例，避免重复加载模型"""

    _instance = None
    _ocr = None
    _lang = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_ocr(cls, lang: str = "ch") -> Optional[Any]:
        """获取（或创建）PaddleOCR 实例

        自动检测 GPU 加速能力：
        - NVIDIA CUDA → 自动启用 GPU
        - AMD ROCm    → 自动启用 GPU
        - 无 GPU      → CPU 模式
        """
        if cls._ocr is not None and cls._lang == lang:
            return cls._ocr
        if not PADDLEOCR_AVAILABLE:
            return None
        cls._lang = lang

        # 主动检测可用 GPU 后端，决定是否启用 GPU 加速
        backend = _detect_gpu_backend()
        use_gpu = backend != "cpu"
        if backend in {"nvidia_cuda", "amd_rocm"}:
            device = "gpu"
        elif backend.startswith("custom_"):
            device = backend.removeprefix("custom_")
        else:
            device = "cpu"
        _patch_paddlex_paddle_dependency_check()
        cls._ocr = PaddleOCR(lang=lang, device=device, **_resolve_ocr_model_kwargs())

        # 日志输出 GPU 状态
        gpu_info = get_gpu_info()
        print(f"[OCR] 加速模式: {gpu_info['description']}")
        if use_gpu:
            try:
                import paddle
                if backend == "nvidia_cuda":
                    gpu_name = paddle.device.cuda.get_device_name()
                    gpu_count = paddle.device.cuda.device_count()
                    print(f"[OCR] 检测到 {gpu_count} 个 NVIDIA GPU: {gpu_name}")
                elif backend == "amd_rocm":
                    gpu_count = paddle.device.cuda.device_count()
                    print(f"[OCR] 检测到 {gpu_count} 个 AMD GPU (ROCm)")
            except Exception:
                pass

        return cls._ocr


def _direct_ocr_image(image_path: str, lang: str = "ch") -> Dict[str, Any]:
    """不做子进程回退的 OCR 入口，供 worker 使用"""
    path = Path(image_path)
    if not path.exists():
        return {"success": False, "error": f"文件不存在: {image_path}"}
    if not is_image_file(image_path):
        return {
            "success": False,
            "error": f"不支持的图片格式: {path.suffix}，"
                     f"支持的格式: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}",
        }
    if not PADDLEOCR_AVAILABLE:
        return {
            "success": False,
            "error": "未安装 PaddleOCR，请安装 paddlepaddle 和 paddleocr",
        }

    try:
        backend = _detect_gpu_backend()
        if backend in {"nvidia_cuda", "amd_rocm"}:
            device = "gpu"
        elif backend.startswith("custom_"):
            device = backend.removeprefix("custom_")
        else:
            device = "cpu"

        _patch_paddlex_paddle_dependency_check()
        ocr = PaddleOCR(lang=lang, device=device, **_resolve_ocr_model_kwargs())
        raw = ocr.predict(str(path))
        if not raw or not isinstance(raw, list) or len(raw) == 0:
            return {
                "success": True,
                "result": {
                    "file": str(path), "text": "", "lines": [],
                    "confidence": 0.0, "line_count": 0,
                },
                "message": "未识别到文字",
            }

        normalized = _normalize_v3_result(raw)
        normalized["file"] = str(path)
        return {
            "success": True,
            "result": normalized,
            "message": f"识别完成，共 {normalized['line_count']} 行文字，"
                       f"平均置信度 {normalized['confidence']:.1%}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"OCR worker 识别失败: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def _direct_ocr_table(image_path: str, lang: str = "ch") -> Dict[str, Any]:
    """不做子进程回退的表格识别入口，供 worker 使用。

    使用 PPStructureV3（PaddleOCR 3.x 自带），全部模型走本地
    <项目根>/models/paddleocr/，无绝对路径、可移植。
    """
    path = Path(image_path)
    if not path.exists():
        return {"success": False, "error": f"文件不存在: {image_path}"}
    if not is_image_file(image_path):
        return {
            "success": False,
            "error": f"不支持的图片格式: {path.suffix}，"
                     f"支持的格式: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}",
        }
    if not PADDLEOCR_AVAILABLE:
        return {
            "success": False,
            "error": "未安装 PaddleOCR，请安装 paddlepaddle 和 paddleocr",
        }

    try:
        backend = _detect_gpu_backend()
        if backend in {"nvidia_cuda", "amd_rocm"}:
            device = "gpu"
        elif backend.startswith("custom_"):
            device = backend.removeprefix("custom_")
        else:
            device = "cpu"

        _patch_paddlex_paddle_dependency_check()
        from paddleocr import PPStructureV3

        kwargs = _resolve_table_model_kwargs()
        # 本地已有 det/rec 模型时，同时指定 model_name，防止 PaddleX
        # 按默认名（PP-OCRv5 系列）联网下载
        if "text_detection_model_dir" in kwargs:
            kwargs.setdefault("text_detection_model_name", "PP-OCRv6_medium_det")
        if "text_recognition_model_dir" in kwargs:
            kwargs.setdefault("text_recognition_model_name", "PP-OCRv6_medium_rec")

        engine = PPStructureV3(
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_textline_orientation=True,
            use_table_recognition=True,
            use_seal_recognition=False,
            use_formula_recognition=False,
            use_chart_recognition=False,
            use_region_detection=False,
            device=device,
            **kwargs,
        )
        raw = engine.predict(str(path))
        if not raw or not isinstance(raw, list) or len(raw) == 0:
            return {
                "success": True,
                "result": {
                    "file": str(path), "table_count": 0, "tables": [],
                    "text": "", "line_count": 0, "confidence": 0.0,
                },
                "message": "未识别到内容",
            }

        normalized = _normalize_table_result(raw)
        normalized["file"] = str(path)
        return {
            "success": True,
            "result": normalized,
            "message": f"表格识别完成，共 {normalized['table_count']} 个表格",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"表格识别 worker 失败: {str(e)}",
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# OCRHandler — 核心逻辑
# ---------------------------------------------------------------------------
class OCRHandler:
    """OCR 核心处理器"""

    def __init__(self, lang: str = "ch"):
        self.lang = lang
        if not PADDLEOCR_AVAILABLE:
            self._init_error = (
                "未安装 PaddleOCR，请执行:\n"
                "  pip install paddlepaddle==3.2.2\n"
                "  pip install paddleocr"
            )
        else:
            self._init_error = None

    def _ensure_engine(self):
        if self._init_error:
            return None
        return _OCREngine.get_ocr(lang=self.lang)

    def _should_fallback_to_subprocess(self, error: Any) -> bool:
        text = str(error)
        markers = (
            "paddle_static",
            "paddle_dynamic",
            "paddlepaddle",
            "PaddleOCR",
            "paddleocr",
            "No module named 'paddle'",
            "No module named \"paddle\"",
        )
        return any(marker in text for marker in markers)

    def _run_via_subprocess(self, image_path: str, require_gpu: bool = False,
                            output_dir: Optional[str] = None) -> Dict[str, Any]:
        last_error = ""
        script_path = Path(__file__).resolve()
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        for python_exe in _candidate_ocr_pythons():
            if python_exe == Path(sys.executable).resolve():
                continue
            backend_info = _probe_python_backend(python_exe)
            if require_gpu and not backend_info.get("compiled_cuda"):
                continue
            try:
                result = subprocess.run(
                    [
                        str(python_exe),
                        "-X", "utf8",
                        str(script_path),
                        "__ocr_worker",
                        str(image_path),
                        self.lang,
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    timeout=180,
                )
                if result.returncode != 0:
                    last_error = (
                        f"{python_exe} 退出码 {result.returncode}: "
                        f"{(result.stderr or result.stdout)[-1000:]}"
                    )
                    continue

                worker_result = _parse_worker_result(result.stdout)
                if worker_result.get("success"):
                    worker_result["subprocess_python"] = str(python_exe)
                    worker_result["subprocess_backend"] = backend_info
                    return _attach_auto_output(worker_result, image_path, output_dir=output_dir)
                last_error = f"{python_exe}: {worker_result.get('error', '未知错误')}"
            except subprocess.TimeoutExpired:
                last_error = f"{python_exe}: OCR 子进程超时"
            except Exception as e:
                last_error = f"{python_exe}: {str(e)}"

        return {
            "success": False,
            "error": f"OCR 子进程回退失败: {last_error or '没有可用的 Python 环境'}",
        }

    def ocr_image(self, image_path: str,
                  output_dir: Optional[str] = None) -> Dict[str, Any]:
        """识别单张图片

        Args:
            image_path: 图片路径
            output_dir: 识别结果保存目录；None 时保存到默认 output/ocr/
        """
        path = Path(image_path)
        if not path.exists():
            return {"success": False, "error": f"文件不存在: {image_path}"}
        if not is_image_file(image_path):
            return {
                "success": False,
                "error": f"不支持的图片格式: {path.suffix}，"
                         f"支持的格式: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}",
            }

        if _detect_gpu_backend() == "cpu" and _has_gpu_capable_subprocess():
            gpu_result = self._run_via_subprocess(image_path, require_gpu=True, output_dir=output_dir)
            if gpu_result.get("success"):
                return gpu_result

        try:
            ocr = self._ensure_engine()
        except Exception as e:
            if self._should_fallback_to_subprocess(e):
                return self._run_via_subprocess(image_path, output_dir=output_dir)
            return {
                "success": False,
                "error": f"OCR 引擎初始化失败: {str(e)}",
                "traceback": traceback.format_exc(),
            }
        if not ocr:
            fallback = self._run_via_subprocess(image_path, output_dir=output_dir)
            if fallback.get("success"):
                return fallback
            return {"success": False, "error": self._init_error or fallback.get("error")}

        try:
            # PaddleOCR 3.x 推荐使用 predict()
            raw = ocr.predict(str(path))
            if not raw or not isinstance(raw, list) or len(raw) == 0:
                return _attach_auto_output({
                    "success": True,
                    "result": {
                        "file": str(path), "text": "", "lines": [],
                        "confidence": 0.0, "line_count": 0,
                    },
                    "message": "未识别到文字",
                }, str(path), output_dir=output_dir)

            normalized = _normalize_v3_result(raw)
            normalized["file"] = str(path)
            return _attach_auto_output({
                "success": True,
                "result": normalized,
                "message": f"识别完成，共 {normalized['line_count']} 行文字，"
                          f"平均置信度 {normalized['confidence']:.1%}",
            }, str(path), output_dir=output_dir)
        except Exception as e:
            if self._should_fallback_to_subprocess(e):
                return self._run_via_subprocess(image_path, output_dir=output_dir)
            return {
                "success": False,
                "error": f"OCR 识别失败: {str(e)}",
                "traceback": traceback.format_exc(),
            }

    def ocr_images(self, image_paths: List[str],
                   output_dir: Optional[str] = None) -> Dict[str, Any]:
        """批量识别多张图片"""
        results = []
        errors = []
        output_files = []
        success_count = 0
        for img_path in image_paths:
            res = self.ocr_image(img_path, output_dir=output_dir)
            if res["success"]:
                results.append(res["result"])
                if res.get("output_files"):
                    output_files.append({"file": img_path, **res["output_files"]})
                success_count += 1
            else:
                errors.append({"file": img_path, "error": res.get("error", "未知错误")})

        summary = {
            "success": True,
            "results": results,
            "total": len(image_paths),
            "success_count": success_count,
            "error_count": len(errors),
            "message": f"批量识别完成: {success_count}/{len(image_paths)} 成功",
        }
        if errors:
            summary["errors"] = errors
        if output_files:
            summary["output_files"] = output_files
        return summary

    def ocr_directory(self, dir_path: str, recursive: bool = False,
                      extensions: Optional[List[str]] = None,
                      output_dir: Optional[str] = None) -> Dict[str, Any]:
        """扫描目录并识别所有图片"""
        path = Path(dir_path)
        if not path.exists() or not path.is_dir():
            return {"success": False, "error": f"目录不存在: {dir_path}"}

        if extensions is None:
            extensions = list(SUPPORTED_IMAGE_EXTENSIONS)

        image_files = []
        for ext in extensions:
            pattern = f"*{ext}"
            if recursive:
                image_files.extend(path.rglob(pattern))
                image_files.extend(path.rglob(pattern.upper()))
            else:
                image_files.extend(path.glob(pattern))
                image_files.extend(path.glob(pattern.upper()))

        image_files = sorted(set(str(f) for f in image_files))
        if not image_files:
            return {
                "success": True, "results": [], "total": 0,
                "success_count": 0, "error_count": 0,
                "message": f"目录 '{dir_path}' 中未找到支持的图片文件",
            }
        return self.ocr_images(image_files, output_dir=output_dir)

    def ocr_image_to_text(self, image_path: str,
                          output_dir: Optional[str] = None) -> Dict[str, Any]:
        """简化接口：仅返回纯文本"""
        result = self.ocr_image(image_path, output_dir=output_dir)
        if not result["success"]:
            return {"success": False, "error": result.get("error", "识别失败")}
        payload = {
            "success": True,
            "text": result["result"]["text"],
            "confidence": result["result"]["confidence"],
            "line_count": result["result"]["line_count"],
        }
        if result.get("output_files"):
            payload["output_files"] = result["output_files"]
        for key in ("subprocess_python", "subprocess_backend"):
            if key in result:
                payload[key] = result[key]
        return payload

    def get_language_info(self) -> Dict[str, Any]:
        return {
            "success": True,
            "current_lang": self.lang,
            "available_languages": list(PADDLEOCR_LANG_MAP.keys()),
            "paddleocr_available": PADDLEOCR_AVAILABLE,
        }

    def get_gpu_info(self) -> Dict[str, Any]:
        """获取当前 GPU 加速能力信息"""
        return {
            "success": True,
            "gpu_info": get_gpu_info(),
        }

    # ── 表格识别（PP-StructureV3）─────────────────────────────────────────

    def ocr_table(self, image_path: str,
                  output_dir: Optional[str] = None) -> Dict[str, Any]:
        """识别图片中的表格（PP-StructureV3，GPU 优先，自动回退子进程）。

        Args:
            image_path: 图片路径
            output_dir: 结果保存目录（html/xlsx/txt/json）；None 时
                        保存到默认 output/ocr_tables/
        """
        path = Path(image_path)
        if not path.exists():
            return {"success": False, "error": f"文件不存在: {image_path}"}
        if not is_image_file(image_path):
            return {
                "success": False,
                "error": f"不支持的图片格式: {path.suffix}，"
                         f"支持的格式: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}",
            }

        if _detect_gpu_backend() == "cpu" and _has_gpu_capable_subprocess():
            gpu_result = self._run_table_via_subprocess(image_path, require_gpu=True,
                                                        output_dir=output_dir)
            if gpu_result.get("success"):
                return gpu_result

        try:
            if not PADDLEOCR_AVAILABLE:
                raise RuntimeError("未安装 PaddleOCR")
            result = _direct_ocr_table(image_path, lang=self.lang)
            if not result.get("success"):
                raise RuntimeError(result.get("error", "表格识别失败"))
            result["output_files"] = _save_table_auto_output(
                result["result"], image_path, output_dir=output_dir)
            return result
        except Exception as e:
            if self._should_fallback_to_subprocess(e):
                return self._run_table_via_subprocess(image_path, output_dir=output_dir)
            return {
                "success": False,
                "error": f"表格识别失败: {str(e)}",
                "traceback": traceback.format_exc(),
            }

    def _run_table_via_subprocess(self, image_path: str, require_gpu: bool = False,
                                  output_dir: Optional[str] = None) -> Dict[str, Any]:
        """表格识别子进程回退（GPU Python 环境优先）。"""
        last_error = ""
        script_path = Path(__file__).resolve()
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        for python_exe in _candidate_ocr_pythons():
            if python_exe == Path(sys.executable).resolve():
                continue
            backend_info = _probe_python_backend(python_exe)
            if require_gpu and not backend_info.get("compiled_cuda"):
                continue
            try:
                result = subprocess.run(
                    [
                        str(python_exe), "-X", "utf8",
                        str(script_path), "__table_worker",
                        str(image_path), self.lang,
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    timeout=600,
                )
                if result.returncode != 0:
                    last_error = (
                        f"{python_exe} 退出码 {result.returncode}: "
                        f"{(result.stderr or result.stdout)[-1000:]}"
                    )
                    continue

                worker_result = _parse_worker_result(result.stdout)
                if worker_result.get("success"):
                    worker_result["subprocess_python"] = str(python_exe)
                    worker_result["subprocess_backend"] = backend_info
                    worker_result["output_files"] = _save_table_auto_output(
                        worker_result["result"], image_path, output_dir=output_dir)
                    return worker_result
                last_error = f"{python_exe}: {worker_result.get('error', '未知错误')}"
            except subprocess.TimeoutExpired:
                last_error = f"{python_exe}: 表格识别子进程超时"
            except Exception as e:
                last_error = f"{python_exe}: {str(e)}"

        return {
            "success": False,
            "error": f"表格识别子进程回退失败: {last_error or '没有可用的 Python 环境'}",
        }


# ---------------------------------------------------------------------------
# OCRToolManager — 外部接口
# ---------------------------------------------------------------------------
class OCRToolManager:
    """OCR 工具管理器 — 同步 + 异步 OCR。

    同步方法（阻塞）:
        ocr_image / ocr_images / ocr_directory / ocr_image_to_text

    异步方法（线程池 + 轮询池，不阻塞）:
        ocr_image_async / ocr_images_async / ocr_status / ocr_wait

    所有异步任务完成后自动推送结果到 MessagePollingPool。
    """

    def __init__(self, lang: str = "ch"):
        self.handler = OCRHandler(lang=lang)
        # ── 异步状态 ──
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._state_file = OCR_STATE_FILE
        self._cancel_flags: Dict[str, threading.Event] = {}
        self._executor = ThreadPoolExecutor(max_workers=MAX_OCR_THREADS)
        self._futures: Dict[str, Any] = {}
        self._load_state()

    # ═══════════════════════════════════════════════════════════ #
    #  异步 OCR — 线程池 + 轮询池
    # ═══════════════════════════════════════════════════════════ #

    def ocr_image_async(
        self, image_path: str, background: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """异步 OCR 识别单张图片（默认不阻塞）。

        后台模式 (background=True)：OCR 在线程池中执行，立即返回 task_id。
        结果完成后自动推送到 MessagePollingPool（推送的是压缩摘要，
        完整结果落盘后按 output_files 读取）。

        Args:
            image_path: 图片文件路径
            background: True → 后台执行，立即返回 task_id
                        False → 同步执行（阻塞等待）
            output_dir: 结果保存目录；None 时保存到默认 output/ocr/

        Returns:
            后台模式: {"success": True, "task_id": "...", "status": "pending"}
            同步模式: 压缩后的 OCR 结果字典

        💡 调用建议：单张图片且不急于获取结果时，优先使用此异步方法。
           大量图片请用 ocr_images_async 批量异步。
        """
        path = Path(image_path)
        if not path.exists():
            return {"success": False, "error": f"文件不存在: {image_path}"}
        if not is_image_file(image_path):
            return {
                "success": False,
                "error": f"不支持的图片格式: {path.suffix}，"
                         f"支持的格式: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}",
            }

        task_id = uuid.uuid4().hex[:8]

        if not background:
            result = self.handler.ocr_image(image_path, output_dir=output_dir)
            result["task_id"] = task_id
            return _compact_tool_output(result)

        # 后台模式：提交到线程池
        self._register_task(task_id, {
            "type": "ocr_single",
            "image_path": str(path),
            "status": "pending",
            "created_at": time.time(),
        })

        future = self._executor.submit(
            self._ocr_worker, task_id, image_path, output_dir)
        self._futures[task_id] = future

        return {
            "success": True,
            "task_id": task_id,
            "status": "pending",
            "message": f"OCR 任务已提交: {task_id}",
        }

    def ocr_images_async(
        self, image_paths: List[str], background: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """异步批量 OCR（默认不阻塞，线程池并行处理）。

        每张图片独立提交到线程池，MAX_OCR_THREADS 控制并发度。

        Args:
            image_paths: 图片路径列表
            background: True → 批量并行后台执行
                        False → 同步批量（阻塞）
            output_dir: 结果保存目录；None 时保存到默认 output/ocr/

        Returns:
            后台模式: {"success": True, "task_ids": [...], "total": N, "status": "pending"}
            同步模式: 完整批量结果字典

        💡 调用建议：大量图片（≥3张）请优先使用此异步方法，线程池并行处理不阻塞。
        """
        if not image_paths:
            return {"success": False, "error": "image_paths 不能为空"}

        if not background:
            return _compact_tool_output(
                self.handler.ocr_images(image_paths, output_dir=output_dir))

        task_ids = []
        for img_path in image_paths:
            r = self.ocr_image_async(img_path, background=True, output_dir=output_dir)
            if r.get("success"):
                task_ids.append(r["task_id"])
            else:
                task_ids.append(None)

        valid_count = len([t for t in task_ids if t])
        return {
            "success": True,
            "task_ids": task_ids,
            "total": len(image_paths),
            "success_count": valid_count,
            "error_count": len(image_paths) - valid_count,
            "status": "pending",
            "message": f"已提交 {valid_count}/{len(image_paths)} 个 OCR 任务",
        }

    def ocr_status(self, task_id: str) -> Dict[str, Any]:
        """查询 OCR 异步任务状态。

        Args:
            task_id: 任务 ID

        Returns:
            {"success": True, "data": {"task_id": ..., "status": "pending|running|completed|failed", ...}}
        """
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return {"success": False, "error": f"任务不存在: {task_id}"}
        return {"success": True, "data": dict(task)}

    def ocr_wait(
        self, task_id: str, timeout: Optional[float] = 300
    ) -> Dict[str, Any]:
        """等待 OCR 异步任务完成。

        Args:
            task_id: 任务 ID
            timeout: 超时秒数，默认 300；传 None 表示无限等待（不推荐）

        Returns:
            任务结果，超时时返回当前状态
        """
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return {"success": False, "error": f"任务不存在: {task_id}"}

        future = self._futures.get(task_id)
        if not future:
            return {"success": True, "data": dict(task)}

        try:
            future.result(timeout=timeout)
        except Exception:
            pass  # 超时或取消，返回当前状态

        with self._lock:
            task = self._tasks.get(task_id, {})
        return {"success": True, "data": dict(task)}

    # ═══════════════════════════════════════════════════════════ #
    #  内部 — OCR worker
    # ═══════════════════════════════════════════════════════════ #

    def _ocr_worker(self, task_id: str, image_path: str,
                    output_dir: Optional[str] = None) -> None:
        """OCR 工作线程：执行同步 OCR 并推送结果。

        在线程池中运行，完成后自动：
          1. 更新任务状态为 completed/failed
          2. 推送结果（压缩摘要）到 MessagePollingPool
        """
        try:
            self._update_task(task_id, {"status": "running"})
            result = self.handler.ocr_image(image_path, output_dir=output_dir)
            result["task_id"] = task_id
            self._update_task(task_id, {
                "status": "completed",
                "result": result,
                "completed_at": time.time(),
            })
            self._push_to_pool(task_id, result)
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"OCR 任务异常: {str(e)}",
                "task_id": task_id,
                "traceback": traceback.format_exc(),
            }
            self._update_task(task_id, {
                "status": "failed",
                "error": str(e),
                "failed_at": time.time(),
            })
            self._push_to_pool(task_id, error_result)

    # ═══════════════════════════════════════════════════════════ #
    #  表格识别 — 同步 + 异步（PP-StructureV3）
    # ═══════════════════════════════════════════════════════════ #

    def ocr_table(self, image_path: str,
                  output_dir: Optional[str] = None) -> Dict[str, Any]:
        """识别图片中的表格，结果自动保存（默认 output/ocr_tables/）。

        使用 PP-StructureV3（GPU 优先），输出 html + xlsx + txt + json。
        返回压缩摘要（tables 概要 + output_files），完整表格矩阵在
        output_files.json / .xlsx 中。

        Args:
            image_path: 图片文件路径
            output_dir: 结果保存目录（含 html/xlsx/txt/json）；
                        None 时保存到默认 output/ocr_tables/

        Returns:
            Dict with keys: success, result (table_count/tables 概要/text 预览),
            output_files (html/xlsx/txt/json 路径), message
        """
        try:
            result = self.handler.ocr_table(image_path, output_dir=output_dir)
            return _compact_tool_output(result)
        except Exception as e:
            return {"success": False, "error": f"表格识别失败: {str(e)}"}

    def ocr_table_async(self, image_path: str, background: bool = True,
                        output_dir: Optional[str] = None) -> Dict[str, Any]:
        """异步表格识别（默认不阻塞，结果完成后推送到 MessagePollingPool）。

        Args:
            image_path: 图片文件路径
            background: True → 后台执行，立即返回 task_id
                        False → 同步执行（阻塞等待）
            output_dir: 结果保存目录；None 时保存到默认 output/ocr_tables/

        Returns:
            后台模式: {"success": True, "task_id": "...", "status": "pending"}
            同步模式: 压缩后的表格识别结果字典
        """
        path = Path(image_path)
        if not path.exists():
            return {"success": False, "error": f"文件不存在: {image_path}"}
        if not is_image_file(image_path):
            return {
                "success": False,
                "error": f"不支持的图片格式: {path.suffix}，"
                         f"支持的格式: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}",
            }

        task_id = uuid.uuid4().hex[:8]

        if not background:
            result = self.handler.ocr_table(image_path, output_dir=output_dir)
            result["task_id"] = task_id
            return _compact_tool_output(result)

        self._register_task(task_id, {
            "type": "ocr_table",
            "image_path": str(path),
            "status": "pending",
            "created_at": time.time(),
        })
        future = self._executor.submit(
            self._ocr_table_worker, task_id, image_path, output_dir)
        self._futures[task_id] = future

        return {
            "success": True,
            "task_id": task_id,
            "status": "pending",
            "message": f"表格识别任务已提交: {task_id}",
        }

    def _ocr_table_worker(self, task_id: str, image_path: str,
                          output_dir: Optional[str] = None) -> None:
        """表格识别工作线程：执行并推送结果到轮询池。"""
        try:
            self._update_task(task_id, {"status": "running"})
            result = self.handler.ocr_table(image_path, output_dir=output_dir)
            result["task_id"] = task_id
            self._update_task(task_id, {
                "status": "completed",
                "result": result,
                "completed_at": time.time(),
            })
            self._push_to_pool(task_id, result)
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"表格识别任务异常: {str(e)}",
                "task_id": task_id,
                "traceback": traceback.format_exc(),
            }
            self._update_task(task_id, {
                "status": "failed",
                "error": str(e),
                "failed_at": time.time(),
            })
            self._push_to_pool(task_id, error_result)

    # ═══════════════════════════════════════════════════════════ #
    #  内部 — 轮询池推送
    # ═══════════════════════════════════════════════════════════ #

    def _push_to_pool(self, task_id: str, result: Dict[str, Any]) -> None:
        """将 OCR 任务结果（压缩摘要）推送到消息轮询池。"""
        try:
            from xenon_core.polling_pool import get_pool, PoolMessage

            pool = get_pool()
            is_success = result.get("success", False)
            # 推送压缩摘要，避免大段识别文本进入上下文
            payload = _compact_tool_output(result)
            payload["task_id"] = task_id
            pool.push(
                PoolMessage(
                    source="ocr_tool",
                    scenario="ocr",
                    msg_type="result",
                    payload=payload,
                    priority=2 if not is_success else 1,
                    ttl=3600,  # 1 小时后过期
                )
            )
        except ImportError:
            pass  # 轮询池未初始化（如在独立脚本中运行）
        except Exception:
            pass  # 推送失败非致命

    # ═══════════════════════════════════════════════════════════ #
    #  内部 — 任务状态管理
    # ═══════════════════════════════════════════════════════════ #

    def _register_task(self, task_id: str, task_data: Dict[str, Any]) -> None:
        """注册新任务并持久化。"""
        with self._lock:
            task_data["task_id"] = task_id
            self._tasks[task_id] = task_data
            self._trim_history()
            self._save_state()

    def _update_task(self, task_id: str, updates: Dict[str, Any]) -> None:
        """更新任务状态并持久化。"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.update(updates)
                task["updated_at"] = time.time()
            self._save_state()

    def _trim_history(self) -> None:
        """超出上限时清理最旧的任务。"""
        if len(self._tasks) <= MAX_OCR_TASK_HISTORY:
            return
        sorted_tasks = sorted(
            self._tasks.items(),
            key=lambda kv: kv[1].get("created_at", 0),
            reverse=True,
        )
        keep_ids = {tid for tid, _ in sorted_tasks[:MAX_OCR_TASK_HISTORY]}
        for tid in list(self._tasks.keys()):
            if tid not in keep_ids:
                del self._tasks[tid]

    def _load_state(self) -> None:
        """从磁盘恢复任务状态。"""
        try:
            if self._state_file.exists():
                with open(self._state_file, "r", encoding="utf-8") as f:
                    self._tasks = json.load(f)
        except Exception:
            pass

    def _save_state(self) -> None:
        """持久化任务状态到磁盘。"""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with open(self._state_file, "w", encoding="utf-8") as f:
                    json.dump(self._tasks, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass

    def ocr_image(self, image_path: str,
                  output_dir: Optional[str] = None) -> Dict[str, Any]:
        """识别单张图片，结果自动保存（默认 output/ocr/，json + txt）

        返回的是压缩摘要（text_preview + 元数据 + output_files 路径），
        完整识别文本按 output_files 中的 json/txt 读取，避免占用上下文。

        返回结果中 output_files 包含：
          - json / txt: 本次识别结果文件路径
          - latest_json / latest_txt: 最近一次识别结果（覆盖更新）

        Args:
            image_path: 图片文件路径（支持 jpg/png/bmp/tiff/webp 等）
            output_dir: 结果保存目录（可指定为图片所在文件夹等）；
                        None 时保存到默认 output/ocr/

        Returns:
            Dict with keys: success, result (含 text_preview/confidence/line_count),
            output_files (自动保存的文件路径), message

        💡 调用建议：单张或少量图片用此同步方法即可；
           大量图片（≥3张）请改用 ocr_images_async 异步方法，避免长时间阻塞。
        """
        try:
            result = self.handler.ocr_image(image_path, output_dir=output_dir)
            return _compact_tool_output(result)
        except Exception as e:
            return {"success": False, "error": f"OCR 识别失败: {str(e)}"}

    def ocr_images(self, image_paths: List[str],
                   output_dir: Optional[str] = None) -> Dict[str, Any]:
        """批量识别多张图片，每张图片结果自动保存（默认 output/ocr/）

        Args:
            image_paths: 图片路径列表
            output_dir: 结果保存目录（可指定为图片所在文件夹等）；
                        None 时保存到默认 output/ocr/

        Returns:
            Dict with keys: success, results (压缩摘要列表), total, success_count,
            error_count, output_files (各图片的保存路径)

        💡 调用建议：少量图片可用此同步方法；
           大量图片（≥3张）请改用 ocr_images_async 异步方法（线程池并行，不阻塞）。
        """
        try:
            result = self.handler.ocr_images(image_paths, output_dir=output_dir)
            return _compact_tool_output(result)
        except Exception as e:
            return {"success": False, "error": f"批量 OCR 失败: {str(e)}"}

    def ocr_directory(self, dir_path: str, recursive: bool = False,
                      extensions: Optional[List[str]] = None,
                      output_dir: Optional[str] = None) -> Dict[str, Any]:
        """扫描目录并识别所有图片，结果自动保存（默认 output/ocr/）

        Args:
            dir_path: 目标目录路径
            recursive: 是否递归子目录（默认 False）
            extensions: 要识别的图片扩展名列表（默认为所有支持的格式）
            output_dir: 结果保存目录（可指定为图片所在文件夹等）；
                        None 时保存到默认 output/ocr/

        Returns:
            Dict with keys: success, results (压缩摘要列表), total, success_count, error_count
        """
        try:
            result = self.handler.ocr_directory(
                dir_path, recursive, extensions, output_dir=output_dir)
            return _compact_tool_output(result)
        except Exception as e:
            return {"success": False, "error": f"目录 OCR 失败: {str(e)}"}

    def ocr_image_to_text(self, image_path: str,
                          output_dir: Optional[str] = None) -> Dict[str, Any]:
        """简化接口：识别图片并返回纯文本（结果自动保存，默认 output/ocr/）

        返回的 text 为前 500 字预览，完整文本见 output_files.txt。

        Args:
            image_path: 图片文件路径
            output_dir: 结果保存目录；None 时保存到默认 output/ocr/

        Returns:
            Dict with keys: success, text (预览), confidence, line_count,
            output_files (自动保存的文件路径)
        """
        try:
            result = self.handler.ocr_image_to_text(image_path, output_dir=output_dir)
            return _compact_tool_output(result)
        except Exception as e:
            return {"success": False, "error": f"文本提取失败: {str(e)}"}

    def get_language_info(self) -> Dict[str, Any]:
        try:
            return self.handler.get_language_info()
        except Exception as e:
            return {"success": False, "error": f"获取语言信息失败: {str(e)}"}

    def get_gpu_info(self) -> Dict[str, Any]:
        """获取当前 GPU 加速能力信息"""
        try:
            return self.handler.get_gpu_info()
        except Exception as e:
            return {"success": False, "error": f"获取 GPU 信息失败: {str(e)}"}

    def list_images(self, base_path: str = ".", recursive: bool = False) -> Dict[str, Any]:
        """列出目录中的图片文件"""
        try:
            from pathlib import Path
            base = Path(base_path)
            if not base.exists():
                return {"success": False, "error": f"路径不存在: {base_path}"}
            extensions = set(ext.lower() for ext in SUPPORTED_IMAGE_EXTENSIONS)
            if recursive:
                files = sorted(str(p) for p in base.rglob("*") if p.suffix.lower() in extensions and p.is_file())
            else:
                files = sorted(str(p) for p in base.glob("*") if p.suffix.lower() in extensions and p.is_file())
            return {"success": True, "files": files, "total_files": len(files), "current_path": str(base),
                    "message": f"找到 {len(files)} 个图片文件"}
        except Exception as e:
            return {"success": False, "error": f"列出图片文件失败: {str(e)}"}

    def save_results(self, results: Dict[str, Any], output_path: str,
                     format: str = "json") -> Dict[str, Any]:
        """保存 OCR 结果到文件"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if format == "txt":
                text_parts = []
                if "result" in results and isinstance(results["result"], dict):
                    text_parts.append(results["result"].get("text", ""))
                elif "results" in results:
                    for res in results["results"]:
                        if isinstance(res, dict):
                            text_parts.append(res.get("text", ""))
                output_path.write_text("\n".join(text_parts), encoding="utf-8")
            else:
                output_path.write_text(
                    json.dumps(results, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
            return {"success": True, "file": str(output_path),
                    "message": f"结果已保存到 {output_path}"}
        except Exception as e:
            return {"success": False, "error": f"保存结果失败: {str(e)}"}


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------
def create_ocr_tool_manager(lang: str = "ch") -> OCRToolManager:
    """创建 OCR 工具管理器实例"""
    return OCRToolManager(lang=lang)


def _run_worker_main() -> None:
    image_path = sys.argv[2] if len(sys.argv) >= 3 else ""
    lang = sys.argv[3] if len(sys.argv) >= 4 else "ch"
    result = _direct_ocr_image(image_path, lang=lang)
    print("__OCR_RESULT__" + json.dumps(result, ensure_ascii=False, default=str))


def _run_table_worker_main() -> None:
    image_path = sys.argv[2] if len(sys.argv) >= 3 else ""
    lang = sys.argv[3] if len(sys.argv) >= 4 else "ch"
    result = _direct_ocr_table(image_path, lang=lang)
    print("__OCR_RESULT__" + json.dumps(result, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "__ocr_worker":
        _run_worker_main()
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "__table_worker":
        _run_table_worker_main()
        return

    if len(sys.argv) < 3:
        print(json.dumps({
            "success": False, "error": "参数不足",
            "usage": [
                "python ocr_tool.py ocr_image <图片路径> [output_dir]",
                "python ocr_tool.py ocr_images '[路径1, 路径2]' [output_dir]",
                "python ocr_tool.py ocr_directory <目录路径> [--recursive] [output_dir]",
                "python ocr_tool.py ocr_text <图片路径> [output_dir]",
                "python ocr_tool.py ocr_table <图片路径> [output_dir]",
                "python ocr_tool.py get_gpu_info dummy",
                "python ocr_tool.py get_language_info dummy",
                "默认会把完整结果保存到 output；如需打印完整 JSON，可追加 --full",
            ],
        }, ensure_ascii=False))
        sys.exit(1)

    action = sys.argv[1]
    manager = create_ocr_tool_manager()

    def _optional_output_dir(idx: int) -> Optional[str]:
        """解析可选 output_dir 参数（跳过 --full 等开关）"""
        if idx < len(sys.argv) and not sys.argv[idx].startswith("--"):
            return sys.argv[idx]
        return None

    if action == "ocr_image" and len(sys.argv) >= 3:
        result = manager.ocr_image(sys.argv[2], output_dir=_optional_output_dir(3))
        _print_cli_result(result)
    elif action == "ocr_images" and len(sys.argv) >= 3:
        result = manager.ocr_images(json.loads(sys.argv[2]), output_dir=_optional_output_dir(3))
        _print_cli_result(result)
    elif action == "ocr_directory" and len(sys.argv) >= 3:
        recursive = "--recursive" in sys.argv
        idx = 3 if len(sys.argv) >= 4 and not sys.argv[3].startswith("--") else None
        result = manager.ocr_directory(sys.argv[2], recursive=recursive,
                                       output_dir=_optional_output_dir(3))
        _print_cli_result(result)
    elif action == "ocr_text" and len(sys.argv) >= 3:
        result = manager.ocr_image_to_text(sys.argv[2], output_dir=_optional_output_dir(3))
        _print_cli_result(result)
    elif action == "ocr_table" and len(sys.argv) >= 3:
        result = manager.ocr_table(sys.argv[2], output_dir=_optional_output_dir(3))
        _print_cli_result(result)
    elif action == "list_images" and len(sys.argv) >= 3:
        recursive = "--recursive" in sys.argv
        result = manager.list_images(sys.argv[2], recursive=recursive)
        _print_cli_result(result)
    elif action == "get_gpu_info":
        result = manager.get_gpu_info()
        _print_cli_result(result)
    elif action == "get_language_info":
        result = manager.get_language_info()
        _print_cli_result(result)
    elif action == "save_results" and len(sys.argv) >= 4:
        results = json.loads(sys.argv[2])
        output_path = sys.argv[3]
        fmt = json.loads(sys.argv[4]) if len(sys.argv) >= 5 else "json"
        result = manager.save_results(results, output_path, format=fmt)
        _print_cli_result(result)
    else:
        print(json.dumps({
            "success": False, "error": f"未知操作或参数不足: {action}",
        }, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
