# -*- coding: utf-8 -*-
"""共享媒体编码器（原生多模态改造 Phase 0，2026-10）。

职责：把本地媒体文件编码为 base64 / data URI，并估算其 token 成本。

本模块从 ``Tools/vision_tool.py`` 抽取（“抽取而非重写”，行为保持一致），
由以下两方共享，保证编码策略只存在一份（方案 D3 / §3.2a）：

- ``Tools/vision_tool.py`` —— 带外视觉工具（方法保留为薄委托，签名不变）；
- ``xenon_core/media_payload.py`` —— 原生多模态主管线（聊天附图）。

依赖约束：本模块不 import 任何项目内其它模块，可被 Tools/ 与 xenon_core/
双向引用，避免循环依赖。
"""
from __future__ import annotations

import base64
import io
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from PIL import Image, ImageFilter

    PILLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on local runtime
    PILLOW_AVAILABLE = False


# ── 图片压缩参数（自 vision_tool 原样迁移；禁止在别处复制第二份）──────
DEFAULT_MAX_DIM = 1568          # 最大边长（像素），超过则等比缩放
DEFAULT_JPEG_QUALITY = 85       # JPEG 压缩质量
MAX_BASE64_CHARS = 4 * 1024 * 1024  # base64 字符串上限（约 3MB 原图），超过拒绝发送
# 智能放大：中尺寸小图（界面截图/K线/文档）直接发送时文字太小模型看不清，
# 模型看不清小字会“瞎猜”（幻觉）。放大+锐化后文字清晰，识别准确率大幅提升。
SMART_UPSCALE_FACTOR = 1.5      # 放大倍数
SMART_UPSCALE_MIN_DIM = 512     # 宽高均 >= 512 才放大（避免放大无意义的小缩略图）
SMART_UPSCALE_MAX_DIM = 2048    # 放大后最大边长上限（控制成本）
# 文字增强（text_boost）：裁剪顶部区域放大后【不压缩】直接发送，
# 让文字像素足够大，稳定跨过模型识别能力边界（约 18px，压缩会压回边界内）
TEXT_BOOST_TOP_RATIO = 0.14     # 顶部区域占比（窗口截图标题栏通常在此范围内）
TEXT_BOOST_SCALE = 2.5          # 顶部区域放大倍数
# 估算单价（元/百万 token 输入，仅用于日志估算，可按实际价格调整）
VISION_PRICE_PER_MTOKEN = 0.6

# ── 支持的媒体扩展名（附件识别 / WebUI 上传校验共用）──────────────────
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".avi", ".mkv")

# ── 图片 token 估算参数（OpenAI tile 规则：85 + 170×tiles，tile=512px）──
IMAGE_TOKEN_BASE = 85
IMAGE_TOKEN_PER_TILE = 170
IMAGE_TILE_SIZE = 512
# 远程 URL 无法探测尺寸/体积时的保守估计（约 2×2 tiles，宁可高估）
REMOTE_IMAGE_TOKEN_FALLBACK = IMAGE_TOKEN_BASE + IMAGE_TOKEN_PER_TILE * 4


def _print_notice(message: str) -> None:
    """与 vision_tool 旧实现保持一致的终端提示（保持行为零变化）。"""
    print(message)


def encode_image_to_base64(
    image_path: str,
    max_dim: int = None,
    data_uri: bool = False,
    quality: int = DEFAULT_JPEG_QUALITY,
    smart_upscale: bool = True,
) -> Optional[str]:
    """将图片编码为 base64（可压缩 + 可加 MIME 前缀）。

    Args:
        image_path: 图片文件路径
        max_dim: 最大边长，超过则等比缩放（None=不缩放，保持原样）
        data_uri: 是否返回 data:image/...;base64, 前缀形式（OpenAI 标准协议要求）
        quality: JPEG 压缩质量（仅在需要重编码时生效）
        smart_upscale: 中尺寸小图自动放大+锐化（截图/文档类小图文字太小，
                       模型看不清会幻觉，放大后显著提升识别准确率）
    """
    try:
        original_size = os.path.getsize(image_path)
        mime = "image/png"
        suffix = Path(image_path).suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        elif suffix == ".gif":
            mime = "image/gif"
        elif suffix == ".webp":
            mime = "image/webp"
        elif suffix == ".bmp":
            mime = "image/bmp"

        # Pillow 可用时：缩放/智能放大 + 转 JPEG/PNG，控制体积同时提升小图文字可读性
        if PILLOW_AVAILABLE and max_dim and max_dim > 0:
            with Image.open(image_path) as im:
                im.load()
                w, h = im.size
                need_scale = max(w, h) > max_dim
                # 智能放大判定：中尺寸小图（截图/K线/文档）文字太小，
                # 模型看不清会瞎猜，放大+锐化后识别准确率显著提升
                do_upscale = (smart_upscale and not need_scale
                              and min(w, h) >= SMART_UPSCALE_MIN_DIM
                              and max(w, h) < max_dim)
                # 小图（无需缩放/放大且体积小）：直接原样编码，省去压缩开销
                if not need_scale and not do_upscale and original_size < 300 * 1024:
                    with open(image_path, "rb") as img_file:
                        img_base = base64.b64encode(img_file.read()).decode("utf-8")
                    mime = "image/png" if suffix == ".png" else mime
                else:
                    if need_scale:
                        scale = max_dim / float(max(w, h))
                        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
                    elif do_upscale:
                        scale = min(SMART_UPSCALE_FACTOR,
                                    SMART_UPSCALE_MAX_DIM / float(max(w, h)))
                        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
                        # 轻微锐化：增强文字边缘清晰度，抵消放大造成的模糊
                        im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))
                    # 统一转 RGB（PNG 大图压成 JPEG 体积可降 10 倍以上）
                    if im.mode in ("RGBA", "LA", "P"):
                        rgba = im.convert("RGBA")
                        bg = Image.new("RGB", rgba.size, (255, 255, 255))
                        bg.paste(rgba, mask=rgba.split()[-1])
                        im = bg
                    else:
                        im = im.convert("RGB")
                    # JPEG 编码
                    buf = tempfile.SpooledTemporaryFile(max_size=10 * 1024 * 1024)
                    im.save(buf, format="JPEG", quality=quality, optimize=True)
                    buf.seek(0)
                    jpeg_base = base64.b64encode(buf.read()).decode("utf-8")
                    buf.close()
                    # 处理后图再编码 PNG，与 JPEG 取体积小者（线条图/截图 PNG 往往更小）
                    buf2 = tempfile.SpooledTemporaryFile(max_size=10 * 1024 * 1024)
                    im.save(buf2, format="PNG", optimize=True)
                    buf2.seek(0)
                    png_base = base64.b64encode(buf2.read()).decode("utf-8")
                    buf2.close()
                    mime = "image/png"
                    if len(png_base) < len(jpeg_base):
                        img_base = png_base
                        final_size = len(png_base) // 4 * 3
                        _print_notice(f"[vision] 图片处理: {w}x{h} -> {im.size[0]}x{im.size[1]} | "
                                      f"{original_size/1024:.0f}KB -> {final_size/1024:.0f}KB (PNG)")
                    else:
                        img_base = jpeg_base
                        mime = "image/jpeg"
                        final_size = len(jpeg_base) // 4 * 3
                        _print_notice(f"[vision] 图片压缩: {w}x{h} -> {im.size[0]}x{im.size[1]} | "
                                      f"{original_size/1024:.0f}KB -> {final_size/1024:.0f}KB (JPEG)")
        else:
            with open(image_path, "rb") as img_file:
                img_base = base64.b64encode(img_file.read()).decode("utf-8")

        if data_uri:
            return f"data:{mime};base64,{img_base}"
        return img_base
    except Exception as e:
        _print_notice(f"编码图片失败: {str(e)}")
        return None


def encode_region_to_base64(
    image_path: str,
    box_ratio: tuple = (0, 0, 1, TEXT_BOOST_TOP_RATIO),
    scale: float = TEXT_BOOST_SCALE,
    max_dim: int = None,
) -> Optional[str]:
    """裁剪图片指定区域并放大编码（文字增强用）。

    与 encode_image_to_base64 的关键区别：默认【不压缩】，
    保证区域文字像素足够大（压缩会把放大后的文字压回模型能力边界内，
    导致“看不清→幻觉”）。区域本身不大，不压缩也不会明显增加成本。

    Args:
        image_path: 图片文件路径
        box_ratio: 裁剪区域相对坐标 (left, top, right, bottom)，0~1
        scale: 放大倍数
        max_dim: 可选最大边长限制（None=不限制）
    """
    if not PILLOW_AVAILABLE:
        return None
    try:
        with Image.open(image_path) as im:
            im.load()
            w, h = im.size
            left = int(w * box_ratio[0])
            top = int(h * box_ratio[1])
            right = int(w * box_ratio[2])
            bottom = int(h * box_ratio[3])
            if right <= left or bottom <= top:
                return None
            region = im.crop((left, top, right, bottom))
            if max_dim and max(region.size) > max_dim:
                s = max_dim / float(max(region.size))
                region = region.resize(
                    (max(1, int(region.width * s)), max(1, int(region.height * s))), Image.LANCZOS)
            if scale and scale > 1.0:
                region = region.resize(
                    (max(1, int(region.width * scale)), max(1, int(region.height * scale))), Image.LANCZOS)
                region = region.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))
            if region.mode in ("RGBA", "LA", "P"):
                rgba = region.convert("RGBA")
                bg = Image.new("RGB", rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.split()[-1])
                region = bg
            else:
                region = region.convert("RGB")
            buf = tempfile.SpooledTemporaryFile(max_size=10 * 1024 * 1024)
            region.save(buf, format="JPEG", quality=DEFAULT_JPEG_QUALITY, optimize=True)
            buf.seek(0)
            jpeg_b = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            buf2 = tempfile.SpooledTemporaryFile(max_size=10 * 1024 * 1024)
            region.save(buf2, format="PNG", optimize=True)
            buf2.seek(0)
            png_b = base64.b64encode(buf2.read()).decode("utf-8")
            buf2.close()
            if len(png_b) < len(jpeg_b):
                return f"data:image/png;base64,{png_b}"
            return f"data:image/jpeg;base64,{jpeg_b}"
    except Exception as e:
        _print_notice(f"区域编码失败: {str(e)}")
        return None


def encode_video_to_base64(video_path: str) -> Optional[str]:
    """视频原样 base64 编码（不做转码；转码由调用方按需先行处理）。"""
    try:
        with open(video_path, "rb") as video_file:
            video_base = base64.b64encode(video_file.read()).decode("utf-8")
        return video_base
    except Exception as e:
        _print_notice(f"编码视频失败: {str(e)}")
        return None


def encode_audio_to_base64(audio_path: str) -> Optional[str]:
    """音频原样 base64 编码（不做转码/压缩）。"""
    try:
        with open(audio_path, "rb") as audio_file:
            audio_base = base64.b64encode(audio_file.read()).decode("utf-8")
        return audio_base
    except Exception as e:
        _print_notice(f"编码音频失败: {str(e)}")
        return None


def encode_to_data_uri(path: str) -> Optional[str]:
    """统一编码入口：按扩展名分派，返回 data URI；失败返回 None。

    - 图片：走压缩链路（max_dim=DEFAULT_MAX_DIM，smart_upscale 开启）；
    - 音频/视频：原样 base64 + MIME 前缀（协议透传，压缩由调用方决定）。
    """
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return encode_image_to_base64(path, max_dim=DEFAULT_MAX_DIM, data_uri=True)
    if suffix in VIDEO_EXTENSIONS:
        return _wrap_data_uri(encode_video_to_base64(path), path)
    if suffix in AUDIO_EXTENSIONS:
        return _wrap_data_uri(encode_audio_to_base64(path), path)
    # 未知扩展名：按图片尝试（Pillow 能打开则成功，否则返回 None）
    return encode_image_to_base64(path, max_dim=DEFAULT_MAX_DIM, data_uri=True)


def _wrap_data_uri(raw_base64: Optional[str], path: str) -> Optional[str]:
    if not raw_base64:
        return None
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return f"data:{mime};base64,{raw_base64}"


def estimate_vision_cost(base64_len: int, width: int = None, height: int = None) -> Dict[str, Any]:
    """估算视觉请求的 token 与费用（按 OpenAI 图片分块规则粗估）。"""
    if width and height:
        tiles = max(1, -(-width // 512)) * max(1, -(-height // 512))
        est_tokens = 85 + 170 * tiles
    else:
        est_tokens = max(1, base64_len // 1000)  # 未知尺寸时按体积粗估
    est_cost = est_tokens / 1_000_000 * VISION_PRICE_PER_MTOKEN
    return {
        "est_tokens": est_tokens,
        "est_cost_yuan": round(est_cost, 4),
        "base64_chars": base64_len,
        "base64_mb": round(base64_len / 1024 / 1024, 2),
    }


# ── 图片 token 估算（供上下文计量使用，高频调用带缓存）────────────────
_TOKEN_CACHE: Dict[str, int] = {}
_TOKEN_CACHE_LIMIT = 256
# 只取 data URI 前 64K 字符（≈48KB 图像数据）解码读尺寸，避免全量解码开销
_HEADER_PROBE_CHARS = 65536


def _token_cache_key(data_uri: str) -> str:
    return f"{len(data_uri)}|{data_uri[:24]}|{data_uri[-120:]}"


def estimate_image_tokens_from_data_uri(data_uri: str) -> int:
    """把图片 data URI / URL 估算为 token 数。

    优先按图片真实分辨率走 tile 公式（85 + 170 × tiles，tile=512px）；
    无法解析尺寸时回退为按 base64 体积的启发式（与 vision_tool 老逻辑一致）。
    结果带缓存（key=长度+头尾指纹），供上下文计量高频调用。
    """
    if not isinstance(data_uri, str) or not data_uri:
        return 0
    key = _token_cache_key(data_uri)
    cached = _TOKEN_CACHE.get(key)
    if cached is not None:
        return cached

    tokens = _estimate_image_tokens_uncached(data_uri)
    if len(_TOKEN_CACHE) >= _TOKEN_CACHE_LIMIT:
        _TOKEN_CACHE.clear()
    _TOKEN_CACHE[key] = tokens
    return tokens


def _estimate_image_tokens_uncached(data_uri: str) -> int:
    if not data_uri.startswith("data:"):
        # 远程 URL：无法探测尺寸/体积，按保守常量估计（宁可高估）
        return REMOTE_IMAGE_TOKEN_FALLBACK
    size = _decode_image_size_from_data_uri(data_uri)
    if size:
        width, height = size
        tiles = (
            max(1, -(-width // IMAGE_TILE_SIZE))
            * max(1, -(-height // IMAGE_TILE_SIZE))
        )
        return IMAGE_TOKEN_BASE + IMAGE_TOKEN_PER_TILE * tiles
    payload_len = len(data_uri) - data_uri.find(",") - 1
    return max(1, payload_len // 1000)


def _decode_image_size_from_data_uri(data_uri: str) -> Optional[Tuple[int, int]]:
    """从 data URI 前缀快速解析图片分辨率；解析失败返回 None。"""
    if not PILLOW_AVAILABLE:
        return None
    try:
        comma = data_uri.find(",")
        if comma < 0:
            return None
        payload = data_uri[comma + 1:].strip()
        probe = payload[:_HEADER_PROBE_CHARS]
        # base64 只解码 4 字符对齐的前缀（图像尺寸头总在前 48KB 内）
        probe = probe[: len(probe) - (len(probe) % 4)]
        if len(probe) < 16:
            return None
        raw = base64.b64decode(probe, validate=False)
        with Image.open(io.BytesIO(raw)) as im:
            width, height = im.size
        if width > 0 and height > 0:
            return int(width), int(height)
    except Exception:
        return None
    return None


def clear_token_cache() -> None:
    """清空 token 估算缓存（测试用）。"""
    _TOKEN_CACHE.clear()
