#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Terminal/Shell 操作工具 (优化版)
支持执行命令行命令，包括 cmd、PowerShell 和其他终端命令
修复了乱码、进程泄露和命令注入问题
"""

import json
import sys
import subprocess
import os
import platform
import base64
import shutil
import signal
import threading
import time
import uuid
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor


# 项目根目录：轮询池持久化需要 project_root，否则结果不落盘、进程重启即丢
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 异步任务注册表保留上限（防止无限增长）
_ASYNC_TASK_MAX_KEEP = 50


class TerminalHandler:
    # ─── 命令分类规则（同步/异步路由） ────────────────────────────────
    # 异步关键词：命中即走后台。仅保留「常驻服务 / 持续输出监控」类；
    # 有限时长任务（pip install / git clone / npm install 等）改回同步执行，
    # 由 timeout 保护，避免快速命令被误判转后台、浪费一轮轮询。
    _ASYNC_KEYWORDS = (
        # 服务类（常驻进程）
        "uvicorn", "gunicorn", "waitress", "daphne", "hypercorn",
        "flask run", "flask --app", "flask app",
        "node server", "node app.js", "node index.js", "npm run dev", "npm start",
        "npm run serve", "yarn dev", "yarn start", "pnpm dev", "pnpm start",
        "python -m http.server", "python -m simplehttpserver",
        "streamlit run", "gradio", "jupyter", "jupyter lab", "jupyter notebook",
        "ngrok", "minio server", "redis-server", "mongod", "dockerd",
        "nginx", "apachectl", "caddy", "vite", "webpack serve", "webpack-dev-server",
        "next dev", "nuxt dev", "astro dev", "vue-cli-service serve", "quasar dev",
        "flutter run", "expo start", "meteor", "rails server", "php artisan serve",
        "manage.py runserver", "django runserver", "air", "go run", "nodemon",
        "docker compose up", "docker-compose up", "docker run -it",
        # 持续输出 / 监控类
        "tail -f", "tail --follow", "tail -F", "ping -t", "ping /t",
        "tcpdump", "tshark", "kubectl logs -f", "docker logs -f", "docker logs --follow",
        "watch", "watch -n", "inotifywait", "fswatch",
    )
    # 同步关键词：明确快速返回的命令。当前默认即 sync，此表为显式声明保留
    # （未来若调整默认策略为 async，可用作例外清单）。
    _SYNC_KEYWORDS = (
        "dir", "cd", "echo", "type", "copy", "move", "del", "mkdir",
        "rmdir", "ren", "cls", "clear", "pwd", "ls", "cat", "head", "grep",
        "git status", "git log", "git diff", "git branch", "git remote",
        "git add", "git commit", "git push", "git pull", "git fetch",
        "git checkout", "git merge", "git stash",
        "python -c", "python --version", "python -V", "--version", "-v", "--help",
        "where", "which", "tasklist", "netstat", "ipconfig", "systeminfo",
        "ver", "hostname", "date", "time", "set", "get", "ping -n", "ping -c",
        "tracert", "nslookup", "powershell get-", "sfc /scannow",
    )

    def __init__(self, sandbox_context=None):
        # Phase 4: 支持沙箱隔离执行
        self.sandbox_context = sandbox_context
        if sandbox_context is not None:
            self.current_directory = str(sandbox_context.sandbox_dir)
        else:
            self.current_directory = os.getcwd()
        self.platform_system = platform.system().lower()

        # 异步任务支持：任务注册表 + 后台线程池（最多 3 并发）
        self._async_tasks: Dict[str, Dict[str, Any]] = {}
        self._async_tasks_lock = threading.Lock()
        self._async_executor = ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="term_async"
        )

    def _terminate_process_tree(self, process: subprocess.Popen) -> None:
        """尽量终止整个进程树，避免 shell 包裹命令超时后留下子进程。"""
        if not process:
            return

        try:
            if self.platform_system == 'windows':
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False
                )
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def _decode_output(self, byte_data: bytes) -> str:
        """
        智能解码输出，解决乱码问题
        优先尝试 UTF-8，失败后尝试 GBK (Windows默认)，最后使用 replace 忽略错误
        """
        if not byte_data:
            return ""
        
        # 1. 优先尝试 UTF-8 (现代工具和 Windows 新版控制台常用)
        try:
            return byte_data.decode('utf-8')
        except UnicodeDecodeError:
            pass
        
        # 2. 尝试 GBK (中文 Windows 传统默认)
        try:
            return byte_data.decode('gbk')
        except UnicodeDecodeError:
            pass
            
        # 3. 最终回退：使用系统默认编码，忽略无法解码的字符
        try:
            return byte_data.decode(sys.getdefaultencoding(), errors='replace')
        except:
            return byte_data.decode('latin1', errors='replace')

    def _execute_internal(self, exec_cmd: Union[str, list], timeout: int, working_dir: Optional[str], is_powershell: bool = False, on_process_started=None) -> Dict[str, Any]:
        """
        核心执行逻辑，统一处理超时和流读取

        Args:
            exec_cmd: 要执行的命令（字符串或参数列表）
            timeout: 超时秒数；None 或 <=0 时按默认 300 处理（防止 communicate 永久挂起）
            working_dir: 工作目录
            is_powershell: 已废弃（保留参数仅为兼容旧调用方，内部不再使用）
            on_process_started: 可选回调，Popen 创建后立即调用（异步任务用它注册进程句柄以支持取消）
        """
        # 防御：timeout 为 None/0 时 communicate(timeout=None) 会永久挂起
        if not timeout or timeout <= 0:
            timeout = 300

        # 用墙上时钟计时（os.times().elapsed 是进程累计 CPU 时间，
        # IO 密集命令如 ping/网络请求 CPU 时间趋近 0，导致耗时恒为 0.0）
        start_time = time.monotonic()
        
        # 确定工作目录
        exec_working_dir = str(Path(working_dir).resolve()) if working_dir else self.current_directory
        
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        # 强制某些 Windows 工具输出 UTF-8 (可选，不仅限于 Python)
        if self.platform_system == 'windows':
            env['PYTHONUTF8'] = '1'

        popen_kwargs = {}
        if self.platform_system == 'windows':
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        # Windows + shell=True 多行命令修复
        # cmd.exe 会把换行符解析为命令分隔符，导致多行 python -c "..." 等命令被截断
        # 方案：检测 python -c 多行命令 → Base64 编码代码块 → 单行执行
        _tmp_script_file = None
        if self.platform_system == 'windows' and isinstance(exec_cmd, str) and '\n' in exec_cmd:
            import re as _re
            # 用 DOTALL 让 . 匹配换行，匹配到结束引号为止，保留代码中的换行符
            py_match = _re.match(
                r'^python\s+-c\s+(["\'])(.*)\1',
                exec_cmd,
                _re.DOTALL
            )
            if py_match:
                code = py_match.group(2)
                import base64
                encoded = base64.b64encode(code.encode('utf-8')).decode('ascii')
                exec_cmd = f'python -c "import base64;exec(base64.b64decode(\'{encoded}\'))"'
            else:
                # 非 python 多行命令：写入临时 .py 文件（换行符在 .py 文件中合法）
                import tempfile, uuid
                _tmp_script_file = os.path.join(
                    tempfile.gettempdir(),
                    f"xenon_tmp_{uuid.uuid4().hex[:8]}.py"
                )
                try:
                    with open(_tmp_script_file, 'w', encoding='utf-8') as _f:
                        _f.write(exec_cmd + '\n')
                    exec_cmd = f'python "{_tmp_script_file}"'
                except Exception:
                    if os.path.exists(_tmp_script_file):
                        try: os.remove(_tmp_script_file)
                        except: pass
                    _tmp_script_file = None

        process = None
        try:
            # 使用 Popen 以便手动控制流和超时
            # 注意：不指定 encoding 和 text，直接读取 bytes
            process = subprocess.Popen(
                exec_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE, # 提供 stdin 防止部分命令挂起等待输入
                cwd=exec_working_dir,
                env=env,
                **popen_kwargs
            )
            
            # 注册进程句柄（异步任务取消 / 监控用）
            if on_process_started:
                try:
                    on_process_started(process)
                except Exception:
                    pass
            
            # 等待进程结束，设置超时
            # communicate 返回的是 bytes
            stdout_bytes, stderr_bytes = process.communicate(timeout=timeout)
            
            # 智能解码
            stdout = self._decode_output(stdout_bytes)
            stderr = self._decode_output(stderr_bytes)
            
            returncode = process.returncode
            
            # 计算耗时（墙上时钟）
            end_time = time.monotonic()
            
            return {
                "success": True,
                "exit_code": returncode,
                "stdout": stdout,
                "stderr": stderr,
                "execution_time": round(end_time - start_time, 2),
                "working_dir": exec_working_dir
            }

        except subprocess.TimeoutExpired:
            # 关键修复：超时必须杀死进程
            if process:
                try:
                    self._terminate_process_tree(process)
                except Exception:
                    pass
                # 第二次 communicate 必须带超时，不能永久阻塞：
                # 若命令通过 start /b 等启动后台进程，Windows 句柄继承会让后台进程
                # 持有 stdout/stderr 管道写端（>nul 重定向无法剥离已继承的句柄）；
                # 而 cmd 可能已先于超时退出，taskkill /T 对已死 PID 无法遍历进程树，
                # 后台进程漏杀 -> 管道永不 EOF -> 无超时的 communicate 会永久挂起智能体
                try:
                    process.communicate(timeout=5) # 清理缓冲区避免僵尸进程
                except Exception:
                    pass
            
            # 返回超时错误信息（解码尽量尝试）
            err_msg = f"命令执行超时 ({timeout}秒)"
            # 尝试读取超时前产生的输出（如果有）
            # 注意：TimeoutExpired 异常对象中可能包含部分 output，但比较复杂，这里简化处理
            return {
                "success": False,
                "error": err_msg,
                "exit_code": -1,
                "stdout": "",
                "stderr": err_msg
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"执行异常: {str(e)}",
                "exit_code": -1
            }
            
        finally:
            # 清理多行命令创建的临时脚本文件
            if _tmp_script_file and os.path.exists(_tmp_script_file):
                try:
                    os.remove(_tmp_script_file)
                except Exception:
                    pass


    @staticmethod
    def _smart_truncate(text: str, budget: int, label: str = "输出") -> Tuple[str, bool]:
        """智能截断：优先压缩重复内容，否则头尾保留。

        返回 (截断后文本, 是否发生截断)，输出长度严格 ≤ budget。
        - 重复内容：单字符占比 >= 90% 或短模式（1~16 字符）重复 >= 5 次时，压缩为摘要
          （如 'x' × 30000 → 'x'×30000，'abc' × 10000 → 'abc'×10000）
        - 普通文本：保留头部 60% + 尾部 40%，中间插入省略提示
        """
        if len(text) <= budget:
            return text, False

        # 1) 重复内容压缩：避免 'x'*30000 / 'abc'*10000 这类输出浪费上下文
        summary = TerminalHandler._repetition_summary(text, label)
        if summary is not None:
            if len(summary) <= budget:
                return summary, True
            # 预算太小放不下完整摘要：截断摘要本体（信息优先于重复内容，仍严格 ≤ budget）
            return summary[:max(1, budget)], True

        # 2) 头尾保留式截断：头部 60% + 尾部 40%，中间插入省略提示
        notice = f"\n... [截断: {label}超过 {budget} 字符限制，已省略中间 {len(text) - budget} 字符] ...\n"
        head_budget = max(1, int(budget * 0.6))
        tail_budget = budget - head_budget - len(notice)
        if tail_budget < 1:
            # 预算太小放不下提示，退化为纯头部截断（不带提示，严格 ≤ budget）
            return text[:budget], True
        return text[:head_budget] + notice + text[-tail_budget:], True

    @staticmethod
    def _repetition_summary(text: str, label: str) -> Optional[str]:
        """检测重复内容并生成压缩摘要；非重复内容返回 None。

        - 单字符重复：占比 >= 90% 且长度 >= 200（如 'x'*30000）
        - 短模式重复：1~16 字符单元重复 >= 5 次（如 'abc'*10000、重复日志行）
        """
        if len(text) < 200:
            return None

        head = text[:20].replace('\n', '\\n')
        tail = text[-20:].replace('\n', '\\n')

        # 单字符重复（一次遍历，最快路径）
        from collections import Counter
        _char, _count = Counter(text).most_common(1)[0]
        if _count / len(text) >= 0.9:
            return (f"[{label}为重复字符 '{_char}' × {len(text)}，已压缩省略；"
                    f"头部样本: {head!r}，尾部样本: {tail!r}]")

        # 短模式重复：找最短重复单元（1~16 字符），纯字符串比较，无正则回溯风险
        for unit_len in range(1, 17):
            if len(text) < unit_len * 5:
                break
            unit = text[:unit_len]
            full, rem = divmod(len(text), unit_len)
            if text == unit * full + unit[:rem]:
                return (f"[{label}为重复模式 {unit[:8]!r} × {full}，已压缩省略；"
                        f"头部样本: {head!r}，尾部样本: {tail!r}]")
        return None

    def _format_result(
        self,
        result: Dict[str, Any],
        command: str,
        command_type: str,
        max_output_lines: int = 100,
        max_output_chars: int = 10000,
    ) -> Dict[str, Any]:
        """统一格式化输出结果，支持行数和字符数截断。

        max_output_chars 为 stdout + stderr 的【总预算】（默认 10000，即 10KB）：
        stdout 优先使用预算，stderr 保留最多 2KB（或总预算的 20%）用于错误可见性。
        截断采用智能策略：重复内容优先压缩为摘要（'x'×30000、'abc'×10000 不再原样搬运），
        普通超长文本保留头部 + 尾部并提示省略量，避免无意义字符浪费上下文。
        """
        if not result.get("success"):
            return {
                **result,
                "command": command,
                "message": f"{command_type} 执行失败: {result.get('error', 'Unknown error')}"
            }

        stdout_raw = result.get("stdout", "")
        stderr_raw = result.get("stderr", "")
        exit_code = result.get("exit_code", -1)
        
        stdout = self._clean_output(stdout_raw)
        stderr = self._clean_output(stderr_raw)
        
        # ---- 截断逻辑 ----
        # 总预算：stdout + stderr 合计 ≤ max_output_chars（stdout 优先，stderr 保留错误可见性）
        truncated = False

        _content_budget = max(0, max_output_chars)
        _stderr_budget = min(2000, max(0, _content_budget // 5))
        _stdout_budget = max(0, _content_budget - _stderr_budget)

        # 智能截断：重复字符压缩 / 头尾保留（每流严格 ≤ 各自预算，提示已计入）
        if max_output_chars > 0:
            stdout, _t1 = self._smart_truncate(stdout, _stdout_budget, "输出")
            stderr, _t2 = self._smart_truncate(stderr, _stderr_budget, "错误输出")
            truncated = _t1 or _t2

        output_lines = stdout.split('\n') if stdout else []
        error_lines = stderr.split('\n') if stderr else []

        # 再按行数截断（头尾保留，中间省略提示）
        if max_output_lines > 0 and len(output_lines) > max_output_lines:
            truncated = True
            _head_lines = max(1, int(max_output_lines * 0.6))
            _tail_lines = max(1, max_output_lines - _head_lines - 1)
            if _tail_lines < 1:
                output_lines = output_lines[:max_output_lines] + [f"... [截断: 输出超过 {max_output_lines} 行限制]"]
            else:
                output_lines = (output_lines[:_head_lines]
                                + [f"... [截断: 共 {len(output_lines)} 行，仅显示头部 {_head_lines} 行 + 尾部 {_tail_lines} 行] ..."]
                                + output_lines[-_tail_lines:])

        if max_output_lines > 0 and len(error_lines) > max_output_lines:
            truncated = True
            _head_lines = max(1, int(max_output_lines * 0.6))
            _tail_lines = max(1, max_output_lines - _head_lines - 1)
            if _tail_lines < 1:
                error_lines = error_lines[:max_output_lines] + [f"... [截断: 错误输出超过 {max_output_lines} 行限制]"]
            else:
                error_lines = (error_lines[:_head_lines]
                               + [f"... [截断: 共 {len(error_lines)} 行，仅显示头部 {_head_lines} 行 + 尾部 {_tail_lines} 行] ..."]
                               + error_lines[-_tail_lines:])
        
        command_succeeded = exit_code == 0
        status_msg = "成功" if command_succeeded else f"失败(退出码: {exit_code})"
        
        return {
            "success": command_succeeded,
            "command": command,
            "exit_code": exit_code,
            "output": output_lines,
            "errors": error_lines,
            "working_dir": result.get("working_dir"),
            "execution_time": result.get("execution_time", 0),
            "output_lines": len(output_lines),
            "error_lines": len(error_lines),
            "truncated": truncated,
            "message": f"{command_type}执行{status_msg} | 耗时: {result.get('execution_time', 0):.2f}秒"
        }

    def _clean_output(self, text: str) -> str:
        """清理输出文本：移除多余空白和换行符"""
        if not text:
            return ""
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = text.split('\n')
        cleaned_lines = []
        prev_empty = False
        for line in lines:
            stripped = line.rstrip()
            is_empty = not stripped
            if is_empty:
                if not prev_empty and cleaned_lines:
                    prev_empty = True
                    cleaned_lines.append("")
            else:
                prev_empty = False
                if ' : ' in stripped or ' :' in stripped:
                    stripped = self._clean_table_line(stripped)
                cleaned_lines.append(stripped)
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()
        return '\n'.join(cleaned_lines)

    def _clean_table_line(self, line: str) -> str:
        """清理表格行中的多余空格"""
        import re
        match = re.match(r'^(\S.*?)\s+:\s*(.*)$', line)
        if match:
            key = match.group(1).rstrip()
            value = match.group(2).strip()
            return f"{key} : {value}" if value else f"{key} :"
        return line.rstrip()

    # ═══════════════════════════════════════════════════════════════ #
    #  同步/异步 自动路由
    # ═══════════════════════════════════════════════════════════════ #

    @staticmethod
    def _split_first_command(command: str) -> str:
        """取第一个命令段：识别引号/转义，在 && / || / | / ; 处切分。

        避免 `python -c "print('a|b')"` 这类引号内的分隔符被误切。
        """
        quote = None
        escape = False
        for i, ch in enumerate(command):
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if quote:
                if ch == quote:
                    quote = None
                continue
            if ch in ('"', "'"):
                quote = ch
                continue
            if ch in ('&', '|', ';'):
                return command[:i].strip()
        return command.strip()

    def _extract_command_head(self, command: str) -> str:
        """提取命令首段「可执行部分」，剥离包裹层与参数内容。

        修复全文子串误判：python -c "print('pip install x')" 这类命令，
        参数/代码内容不应参与关键词匹配。
        """
        first = self._split_first_command(command or "")
        if not first:
            return ""
        lower = first.lower()
        # python -c "..." → 只看 "python -c"（-c 之后是代码，不参与匹配）
        m = re.match(r'^(python\w*(?:\.exe)?)\s+-c\b', lower)
        if m:
            return f"{m.group(1)} -c"
        # echo 之后全是参数
        if re.match(r'^echo\b', lower):
            return "echo"
        # cmd /c xxx / cmd /k xxx → 递归剥离
        m = re.match(r'^cmd\s*/\s*[cd]\s+(.+)$', lower)
        if m:
            return self._extract_command_head(m.group(1))
        # powershell / pwsh [-Option ...] xxx → 递归剥离
        m = re.match(r'^(?:powershell(?:\.exe)?|pwsh(?:\.exe)?)\s+(.*)$', lower)
        if m:
            rest = re.sub(r'^(?:\s*-[a-z]+\s+)+', '', m.group(1))
            return self._extract_command_head(rest) if rest else "powershell"
        return lower

    @staticmethod
    def _head_matches(head: str, kw: str) -> bool:
        """关键词匹配命令首段：kw 必须出现在 head 开头且带词边界。

        如 "node server" 匹配 "node server.js"（边界字符 '.'），
        但不匹配 "node server2"（'2' 不是边界字符）。
        """
        kw = kw.strip()
        if not kw or not head:
            return False
        if head == kw:
            return True
        if head.startswith(kw):
            nxt = head[len(kw)]
            return nxt in (" ", ".", "/", "\\", "-")
        return False

    def classify_command(self, command: str) -> str:
        """命令分类：根据「命令首段可执行名」判断应走同步还是异步（纯函数，可单测）。

        规则优先级：
            1. 异步关键词命中（仅匹配首段可执行名）→ 'async'（常驻服务 / 持续输出监控）
            2. 否则 → 'sync'（保守默认；有限时长命令由 timeout 保护）

        Args:
            command: 原始命令字符串

        Returns:
            'sync' 或 'async'
        """
        if not command or not command.strip():
            return 'sync'
        head = self._extract_command_head(command)
        # 1. 异步词优先（常驻/持续输出，阻塞代价高，宁可后台也不卡主流程）
        for kw in self._ASYNC_KEYWORDS:
            if self._head_matches(head, kw):
                return 'async'
        # 2. 其余默认同步（快速命令 + 有限时长命令均由 timeout 保护）
        return 'sync'

    def execute_command_async(self, command: str, timeout: int = 300, working_dir: Optional[str] = None, non_interactive: bool = True, max_output_lines: int = 100, max_output_chars: int = 10000) -> Dict[str, Any]:
        """异步执行终端命令：立即返回 task_id，不阻塞调用方。

        后台线程执行命令，完成后：
            - 任务注册表状态更新为 done/failed
            - 结果推入全局轮询池（source='terminal', scenario='cmd_result'），
              调用方下一回合 peek() 即可取到结果

        Returns:
            {success, task_id, status: 'running', message}
        """
        self._prune_async_tasks()  # 清理已完成旧任务，防止注册表无限增长
        task_id = f"t-{uuid.uuid4().hex[:8]}"
        task = {
            "task_id": task_id,
            "command": command,
            "status": "running",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "result": None,
            "_process": None,  # 内部字段：后台进程句柄（取消用），不对外暴露
        }
        with self._async_tasks_lock:
            self._async_tasks[task_id] = task

        self._async_executor.submit(
            self._run_async_task,
            task_id, command, timeout, working_dir,
            non_interactive, max_output_lines, max_output_chars,
        )
        return {
            "success": True,
            "task_id": task_id,
            "status": "running",
            "message": f"命令已提交后台执行 (task_id={task_id})，完成后将推入轮询池",
        }

    def _run_async_task(self, task_id: str, command: str, timeout: int, working_dir: Optional[str], non_interactive: bool, max_output_lines: int, max_output_chars: int) -> None:
        """后台线程执行体：跑命令 → 更新注册表 → 推轮询池。"""
        task = None
        with self._async_tasks_lock:
            task = self._async_tasks.get(task_id)
            # 提交后可能已被 cancel（进程尚未创建时），直接放弃执行
            if task is not None and task["status"] == "cancelled":
                return

        def _register_process(proc):
            with self._async_tasks_lock:
                if task_id in self._async_tasks:
                    self._async_tasks[task_id]["_process"] = proc

        try:
            exec_cmd = command
            if non_interactive and command.lower().endswith('.bat'):
                exec_cmd = f'echo. | {command}'
            raw_result = self._execute_internal(
                exec_cmd, timeout, working_dir,
                on_process_started=_register_process,
            )
            formatted = self._format_result(raw_result, command, "命令", max_output_lines, max_output_chars)

            if task:
                with self._async_tasks_lock:
                    # 若已被 cancel，不再用完成状态覆盖
                    if task["status"] != "cancelled":
                        task["status"] = "done"
                        task["result"] = formatted

            # 推入全局轮询池（失败不影响任务本身；传 project_root 保证持久化落盘）
            try:
                from xenon_core.polling_pool import get_pool
                pool = get_pool(_PROJECT_ROOT)
                pool.push_result(
                    source="terminal",
                    scenario="cmd_result",
                    result={
                        "task_id": task_id,
                        "command": command,
                        **formatted,
                    },
                    priority=1,
                    ttl=600,
                )
            except Exception as e:
                if task:
                    with self._async_tasks_lock:
                        if task["status"] == "done":
                            task["result"] = dict(task.get("result") or {})
                            task["result"]["pool_push_warning"] = str(e)
        except Exception as e:
            if task:
                with self._async_tasks_lock:
                    if task["status"] != "cancelled":
                        task["status"] = "failed"
                        task["result"] = {
                            "success": False,
                            "error": f"异步执行异常: {str(e)}",
                            "command": command,
                        }

    def _prune_async_tasks(self, max_keep: int = _ASYNC_TASK_MAX_KEEP) -> None:
        """清理已完成任务，防止注册表无限增长（仅清理终态任务，运行中保留）。"""
        with self._async_tasks_lock:
            if len(self._async_tasks) <= max_keep:
                return
            finished = [
                tid for tid, t in self._async_tasks.items()
                if t["status"] in ("done", "failed", "cancelled")
            ]
            # 按创建时间从旧到新清理
            finished.sort(key=lambda tid: self._async_tasks[tid]["created_at"])
            for tid in finished[: len(self._async_tasks) - max_keep]:
                self._async_tasks.pop(tid, None)

    def list_async_tasks(self, include_done: bool = True, include_results: bool = False) -> Dict[str, Any]:
        """列出所有后台终端任务状态。

        Args:
            include_done: 是否包含已完成任务（默认 True）
            include_results: 是否附带完整结果内容（默认 False，省 token；
                             需要结果时优先用 get_async_task 单独取）

        Returns:
            {success, count, tasks: [{task_id, command, status, created_at, has_result, (result)}]}
        """
        with self._async_tasks_lock:
            tasks = []
            for tid, t in self._async_tasks.items():
                if not include_done and t["status"] in ("done", "failed", "cancelled"):
                    continue
                item = {
                    "task_id": tid,
                    "command": t["command"],
                    "status": t["status"],
                    "created_at": t["created_at"],
                    "has_result": t["result"] is not None,
                }
                if include_results:
                    item["result"] = t["result"]
                tasks.append(item)
        tasks.sort(key=lambda x: x["created_at"], reverse=True)
        return {"success": True, "count": len(tasks), "tasks": tasks}

    def get_async_task(self, task_id: str) -> Dict[str, Any]:
        """查询单个后台任务的完整信息（含执行结果）。

        Args:
            task_id: execute_command_async 返回的任务 ID

        Returns:
            {success, task_id, command, status, created_at, result}
        """
        if not task_id:
            return {"success": False, "error": "task_id 不能为空"}
        with self._async_tasks_lock:
            task = self._async_tasks.get(task_id)
            if task is None:
                return {"success": False, "error": f"任务不存在或已被清理: {task_id}", "task_id": task_id}
            info = {k: v for k, v in task.items() if not k.startswith("_")}
        return {"success": True, **info}

    def cancel_async_task(self, task_id: str) -> Dict[str, Any]:
        """取消正在运行的后台任务（终止其进程树，释放线程池 worker）。

        Args:
            task_id: execute_command_async 返回的任务 ID

        Returns:
            {success, task_id, status}
        """
        if not task_id:
            return {"success": False, "error": "task_id 不能为空"}
        proc = None
        with self._async_tasks_lock:
            task = self._async_tasks.get(task_id)
            if task is None:
                return {"success": False, "error": f"任务不存在或已被清理: {task_id}", "task_id": task_id}
            proc = task.get("_process")
            if task["status"] == "running":
                task["status"] = "cancelled"
        if proc is not None:
            self._terminate_process_tree(proc)
        return {"success": True, "task_id": task_id, "status": "cancelled"}

    def execute_command(self, command: str, timeout: int = 300, working_dir: Optional[str] = None, non_interactive: bool = True, max_output_lines: int = 100, max_output_chars: int = 10000, mode: str = 'auto') -> Dict[str, Any]:
        """
        执行通用终端命令

        Args:
            command: 要执行的命令
            timeout: 超时秒数（默认 300）
            working_dir: 工作目录
            non_interactive: 非交互模式
            max_output_lines: 输出最大行数
            max_output_chars: 输出最大字符数
            mode: 'auto' → 自动分类路由（默认）；'sync' → 强制同步；'async' → 强制后台
        """
        # 同步/异步路由：async 直接转后台，立即返回 task_id
        if mode == 'async' or (mode == 'auto' and self.classify_command(command) == 'async'):
            return self.execute_command_async(
                command, timeout, working_dir, non_interactive,
                max_output_lines, max_output_chars,
            )

        exec_cmd = command
        
        # 简单的非交互处理
        if non_interactive and command.lower().endswith('.bat'):
             # echo. | command 确保有输入，防止 bat 暂停
            exec_cmd = f'echo. | {command}'

        raw_result = self._execute_internal(exec_cmd, timeout, working_dir)
        return self._format_result(raw_result, command, "命令", max_output_lines, max_output_chars)

    def execute_cmd_command(self, command: str, timeout: int = 300, working_dir: Optional[str] = None, non_interactive: bool = True, max_output_lines: int = 100, max_output_chars: int = 10000) -> Dict[str, Any]:
        """
        执行 CMD 命令 (Windows)
        """
        if self.platform_system != 'windows':
            return {
                "success": False,
                "error": "CMD 命令仅在 Windows 系统上可用",
                "command": command
            }

        exec_cmd = command
        if not command.strip().lower().startswith('cmd'):
            # 设置 Code Page 为 65001 (UTF-8) 以支持更好的一致性，
            # 虽然 _decode_output 会自动处理，但设置 CP65001 有助于某些内置命令输出 UTF8
            exec_cmd = f'cmd /c chcp 65001 >nul && {command}'
        
        if non_interactive and '.bat' in command.lower():
            exec_cmd = f'echo. | {exec_cmd}'

        raw_result = self._execute_internal(exec_cmd, timeout, working_dir)
        return self._format_result(raw_result, command, "CMD", max_output_lines, max_output_chars)

    def execute_powershell_command(self, command: str, timeout: int = 300, working_dir: Optional[str] = None, max_output_lines: int = 100, max_output_chars: int = 10000) -> Dict[str, Any]:
        """
        执行 PowerShell 命令
        使用 Base64 编码避免引号和转义地狱
        """
        powershell_exe = shutil.which("powershell") or shutil.which("pwsh")
        if not powershell_exe:
            return {
                "success": False,
                "error": "未找到 powershell 或 pwsh 可执行文件",
                "command": command
            }

        # 构造 PowerShell 命令
        # 强制内部输出编码为 UTF-8
        ps_script = f"""
$ProgressPreference = 'SilentlyContinue'
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
{command}
"""
        # 必须使用 UTF-16LE 编码转换为 bytes，PowerShell -EncodedCommand 要求如此
        script_bytes = ps_script.strip().encode('utf-16-le')
        encoded_cmd = base64.b64encode(script_bytes).decode('ascii')
        
        # 构造最终命令
        # -NonInteractive 不显示交互式提示
        # -NoProfile 不加载用户配置文件(加快启动)
        exec_cmd = f'"{powershell_exe}" -NonInteractive -NoProfile -ExecutionPolicy Bypass -EncodedCommand {encoded_cmd}'
        
        raw_result = self._execute_internal(exec_cmd, timeout, working_dir)
        return self._format_result(raw_result, command, "PowerShell", max_output_lines, max_output_chars)

    def get_system_info(self) -> Dict[str, Any]:
        try:
            return {
                "platform": platform.system(),
                "version": platform.version(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "current_directory": os.getcwd(),
                "success": True
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def check_command_exists(self, command: str) -> Dict[str, Any]:
        try:
            command = command.strip()
            if not command:
                return {"success": False, "error": "命令不能为空", "command": command}

            cmd = ["where", command] if self.platform_system == 'windows' else ["which", command]
            result = subprocess.run(cmd, shell=False, capture_output=True)
            
            if result.returncode == 0:
                path = self._decode_output(result.stdout).strip().splitlines()[0]
                return {"success": True, "exists": True, "path": path, "command": command}
            else:
                return {"success": True, "exists": False, "command": command}
        except Exception as e:
            return {"success": False, "error": str(e), "command": command}

# --- 保持原有的管理器和 Main 接口以兼容调用方 ---

class TerminalToolManager:
    def __init__(self):
        self.handler = TerminalHandler()

    def execute_command(self, command: str, timeout: int = 300, working_dir: Optional[str] = None, non_interactive: bool = True, max_output_lines: int = 100, max_output_chars: int = 10000, mode: str = 'auto') -> Dict[str, Any]:
        return self.handler.execute_command(command, timeout, working_dir, non_interactive, max_output_lines, max_output_chars, mode)

    def execute_command_async(self, command: str, timeout: int = 300, working_dir: Optional[str] = None, non_interactive: bool = True, max_output_lines: int = 100, max_output_chars: int = 10000) -> Dict[str, Any]:
        return self.handler.execute_command_async(command, timeout, working_dir, non_interactive, max_output_lines, max_output_chars)

    def list_async_tasks(self, include_done: bool = True, include_results: bool = False) -> Dict[str, Any]:
        return self.handler.list_async_tasks(include_done, include_results)

    def get_async_task(self, task_id: str) -> Dict[str, Any]:
        return self.handler.get_async_task(task_id)

    def cancel_async_task(self, task_id: str) -> Dict[str, Any]:
        return self.handler.cancel_async_task(task_id)

    def execute_cmd_command(self, command: str, timeout: int = 300, working_dir: Optional[str] = None, non_interactive: bool = True, max_output_lines: int = 100, max_output_chars: int = 10000) -> Dict[str, Any]:
        return self.handler.execute_cmd_command(command, timeout, working_dir, non_interactive, max_output_lines, max_output_chars)

    def execute_powershell_command(self, command: str, timeout: int = 30, working_dir: Optional[str] = None, max_output_lines: int = 100, max_output_chars: int = 10000) -> Dict[str, Any]:
        return self.handler.execute_powershell_command(command, timeout, working_dir, max_output_lines, max_output_chars)
        
    def get_system_info(self) -> Dict[str, Any]:
        return self.handler.get_system_info()

    def check_command_exists(self, command: str) -> Dict[str, Any]:
        return self.handler.check_command_exists(command)

def create_terminal_tool_manager():
    return TerminalToolManager()

def main():
    # 简化 main 入口，保持原有参数解析逻辑
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "缺少参数"}, ensure_ascii=False))
        sys.exit(1)

    action = sys.argv[1]
    manager = TerminalToolManager()

    try:
        if action in ["execute", "execute_cmd", "execute_powershell"]:
            if len(sys.argv) < 3:
                print(json.dumps({"success": False, "error": "缺少命令参数"}, ensure_ascii=False))
                sys.exit(1)
            
            cmd_arg = sys.argv[2]
            timeout_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 300
            workdir_arg = sys.argv[4] if len(sys.argv) > 4 else None
            
            if action == "execute":
                res = manager.execute_command(cmd_arg, timeout_arg, workdir_arg)
            elif action == "execute_cmd":
                res = manager.execute_cmd_command(cmd_arg, timeout_arg, workdir_arg)
            else:
                res = manager.execute_powershell_command(cmd_arg, timeout_arg, workdir_arg)
            
            print(json.dumps(res, ensure_ascii=False, indent=2))

        elif action == "get_system_info":
            print(json.dumps(manager.get_system_info(), ensure_ascii=False, indent=2))
        
        elif action == "check_command":
            if len(sys.argv) < 3: raise ValueError("Missing command")
            print(json.dumps(manager.check_command_exists(sys.argv[2]), ensure_ascii=False, indent=2))
        
        else:
            print(json.dumps({"success": False, "error": f"未知操作: {action}"}, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({"success": False, "error": f"执行时发生错误: {str(e)}"}, ensure_ascii=False))

if __name__ == "__main__":
    main()
