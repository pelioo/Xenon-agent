#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发网页搜索工具模块
提供网页并发访问、内容提取和网站状态检查功能
支持批量并发请求，显著提升搜索速度
"""

import requests
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
import time
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import asyncio
from playwright.async_api import async_playwright, Browser

# 网页解析和学术搜索
import xml.etree.ElementTree as ET
import urllib.parse
import json

# DuckDuckGo 搜索引擎（免费，0 API key，0 配置）
from ddgs import DDGS


class WebSearchTool:
    """网页搜索工具类（核心实现）"""

    def __init__(self, max_workers: int = 10, headless: bool = True, chrome_path: Optional[str] = None):
        """
        初始化网页搜索工具

        :param max_workers: 最大并发线程数，默认10
        :param headless: 是否使用无头浏览器模式，默认True
        :param chrome_path: Chrome浏览器路径，默认None（使用Playwright内置浏览器）
        """
        self.max_workers = max_workers
        self.headless = headless
        self.chrome_path = chrome_path or r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        self.local = threading.local()
        self._init_session()
        self._playwright = None
        self._browser = None
        self._browser_lock = threading.Lock()

    def _init_session(self):
        """初始化线程局部的请求会话"""
        self.local.session = requests.Session()
        self.local.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def _get_session(self) -> requests.Session:
        """获取当前线程的会话"""
        if not hasattr(self.local, 'session') or self.local.session is None:
            self._init_session()
        return self.local.session

    async def _get_browser(self) -> Browser:
        """获取或创建浏览器实例（线程安全）"""
        if self._browser is None:
            with self._browser_lock:
                if self._browser is None:
                    self._playwright = await async_playwright().start()
                    self._browser = await self._playwright.chromium.launch(
                        headless=self.headless,
                        executable_path=self.chrome_path
                    )
        return self._browser

    async def _close_browser(self):
        """关闭浏览器实例"""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def browse_website_with_js(
        self,
        url: str,
        wait_time: float = 2.0,
        timeout: int = 30000,
        wait_for_selector: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        使用JavaScript渲染浏览指定网站（支持SPA应用）

        :param url: 目标网址
        :param wait_time: 页面加载后等待时间（秒），默认2秒
        :param timeout: 页面加载超时时间（毫秒），默认30000
        :param wait_for_selector: 等待特定选择器出现后再提取内容
        :return: 包含页面内容和状态的字典
        """
        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {
                    "success": False,
                    "error": "无效的URL格式",
                    "url": url,
                    "message": "提供的URL格式不正确"
                }

            browser = await self._get_browser()
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            page = await context.new_page()

            start_time = time.time()

            try:
                await page.goto(url, timeout=timeout, wait_until='domcontentloaded')

                if wait_for_selector:
                    try:
                        await page.wait_for_selector(wait_for_selector, timeout=timeout)
                    except:
                        pass

                await asyncio.sleep(wait_time)

                title = await page.title()

                description = await page.evaluate('''() => {
                    const meta = document.querySelector('meta[name="description"]');
                    return meta ? meta.getAttribute('content') : '';
                }''')

                body_text = await page.evaluate('''() => {
                    const body = document.body;
                    if (body) {
                        return body.innerText || body.textContent || '';
                    }
                    return document.body ? document.body.innerText : '';
                }''')

                text_content = re.sub(r'\s+', ' ', body_text).strip()

                max_content_length = 2000
                if len(text_content) > max_content_length:
                    text_content = text_content[:max_content_length] + "...(内容已截断)"

                elapsed_time = time.time() - start_time

                return {
                    "success": True,
                    "url": url,
                    "status_code": 200,
                    "title": title,
                    "description": description,
                    "content": text_content,
                    "content_length": len(text_content),
                    "elapsed_time": elapsed_time,
                    "method": "javascript_rendering",
                    "message": f"成功访问网站 (JS渲染): {title if title else url}"
                }

            finally:
                await context.close()

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "message": f"浏览网站时发生错误: {str(e)}"
            }

    def browse_website_with_js_sync(
        self,
        url: str,
        wait_time: float = 2.0,
        timeout: int = 30000,
        wait_for_selector: Optional[str] = None,
        extract_text_only: bool = True,
        max_content_length: int = 50000
    ) -> Dict[str, Any]:
        """
        使用JavaScript渲染浏览指定网站（同步版本）
        每次调用创建独立的浏览器实例，确保线程安全

        :param url: 目标网址
        :param wait_time: 页面加载后等待时间（秒），默认2秒
        :param timeout: 页面加载超时时间（毫秒），默认30000
        :param wait_for_selector: 等待特定选择器出现后再提取内容
        :param extract_text_only: 是否只提取可见文本（默认True，避免HTML过大）
        :param max_content_length: 最大内容长度（字符数），超过则截断
        :return: 包含页面内容和状态的字典
        """
        async def _browse_with_isolated_browser():
            playwright = None
            browser = None
            try:
                playwright = await async_playwright().start()
                browser = await playwright.chromium.launch(
                    headless=self.headless,
                    executable_path=self.chrome_path
                )
                context = await browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                )
                page = await context.new_page()
                start_time = time.time()
                
                try:
                    await page.goto(url, timeout=timeout, wait_until='domcontentloaded')
                    
                    if wait_for_selector:
                        await page.wait_for_selector(wait_for_selector, timeout=timeout)
                    else:
                        await asyncio.sleep(wait_time)
                    
                    title = await page.title()
                    
                    if extract_text_only:
                        content = await page.locator('body').inner_text()
                        content = ' '.join(content.split())
                    else:
                        content = await page.content()
                    
                    original_length = len(content)
                    if len(content) > max_content_length:
                        content = content[:max_content_length] + "\n... [内容已截断]"
                    
                    response_time = time.time() - start_time
                    
                    return {
                        "success": True,
                        "url": url,
                        "title": title,
                        "content": content,
                        "content_length": len(content),
                        "original_length": original_length,
                        "truncated": original_length > max_content_length,
                        "response_time": response_time,
                        "method": "javascript_rendering",
                        "extract_mode": "text_only" if extract_text_only else "full_html",
                        "message": f"成功获取页面内容 (JS渲染, {'纯文本' if extract_text_only else '完整HTML'})"
                    }
                finally:
                    await context.close()
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "url": url,
                    "message": f"浏览网站时发生错误: {str(e)}"
                }
            finally:
                if browser:
                    await browser.close()
                if playwright:
                    await playwright.stop()

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(_browse_with_isolated_browser())
            finally:
                loop.close()
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "message": f"浏览网站时发生错误: {str(e)}"
            }

    async def check_website_status_with_js(
        self,
        url: str,
        wait_time: float = 1.0,
        timeout: int = 30000
    ) -> Dict[str, Any]:
        """
        使用JavaScript渲染检查网站状态（支持SPA应用）

        :param url: 目标网址
        :param wait_time: 页面加载后等待时间（秒），默认1秒
        :param timeout: 页面加载超时时间（毫秒），默认30000
        :return: 包含网站状态信息的字典
        """
        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {
                    "success": False,
                    "error": "无效的URL格式",
                    "url": url,
                    "message": "提供的URL格式不正确"
                }

            browser = await self._get_browser()
            context = await browser.new_context()
            page = await context.new_page()

            start_time = time.time()

            try:
                await page.goto(url, timeout=timeout, wait_until='domcontentloaded')
                await asyncio.sleep(wait_time)

                response_time = time.time() - start_time

                return {
                    "success": True,
                    "url": url,
                    "status_code": 200,
                    "is_reachable": True,
                    "response_time": response_time,
                    "method": "javascript_rendering",
                    "message": f"网站状态: 在线 (JS渲染)"
                }

            finally:
                await context.close()

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "message": f"检查网站状态时发生错误: {str(e)}"
            }

    def check_website_status_with_js_sync(
        self,
        url: str,
        wait_time: float = 1.0,
        timeout: int = 30000
    ) -> Dict[str, Any]:
        """
        使用JavaScript渲染检查网站状态（同步版本）
        每次调用创建独立的浏览器实例，确保线程安全

        :param url: 目标网址
        :param wait_time: 页面加载后等待时间（秒），默认1秒
        :param timeout: 页面加载超时时间（毫秒），默认30000
        :return: 包含网站状态信息的字典
        """
        async def _check_with_isolated_browser():
            playwright = None
            browser = None
            try:
                playwright = await async_playwright().start()
                browser = await playwright.chromium.launch(
                    headless=self.headless,
                    executable_path=self.chrome_path
                )
                context = await browser.new_context()
                page = await context.new_page()
                start_time = time.time()
                
                try:
                    await page.goto(url, timeout=timeout, wait_until='domcontentloaded')
                    await asyncio.sleep(wait_time)
                    response_time = time.time() - start_time
                    
                    return {
                        "success": True,
                        "url": url,
                        "status_code": 200,
                        "is_reachable": True,
                        "response_time": response_time,
                        "method": "javascript_rendering",
                        "message": f"网站状态: 在线 (JS渲染)"
                    }
                finally:
                    await context.close()
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "url": url,
                    "message": f"检查网站状态时发生错误: {str(e)}"
                }
            finally:
                if browser:
                    await browser.close()
                if playwright:
                    await playwright.stop()

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(_check_with_isolated_browser())
            finally:
                loop.close()
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "message": f"检查网站状态时发生错误: {str(e)}"
            }

    async def search_web_content_with_js(
        self,
        url: str,
        keyword: str,
        wait_time: float = 2.0,
        timeout: int = 30000
    ) -> Dict[str, Any]:
        """
        在网页中搜索特定关键词（支持JavaScript渲染）

        :param url: 目标网址
        :param keyword: 要搜索的关键词
        :param wait_time: 页面加载后等待时间（秒），默认2秒
        :param timeout: 页面加载超时时间（毫秒），默认30000
        :return: 包含搜索结果的字典
        """
        try:
            browse_result = await self.browse_website_with_js(url, wait_time, timeout)

            if not browse_result["success"]:
                return browse_result

            content = browse_result["content"].lower()
            keyword_lower = keyword.lower()

            positions = []
            start = 0
            while True:
                pos = content.find(keyword_lower, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1

            contexts = []
            for pos in positions[:5]:
                original_start = max(0, pos - 50)
                original_end = min(len(browse_result["content"]), pos + len(keyword_lower) + 50)
                context_text = browse_result["content"][original_start:original_end]

                contexts.append({
                    "position": pos,
                    "context": context_text,
                    "highlighted": context_text.replace(keyword, f"[{keyword}]")
                })

            return {
                "success": True,
                "url": url,
                "keyword": keyword,
                "match_count": len(positions),
                "positions": positions[:5],
                "contexts": contexts,
                "title": browse_result.get("title", ""),
                "method": "javascript_rendering",
                "message": f"在网页中找到 {len(positions)} 个 '{keyword}' 的匹配项 (JS渲染)"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "keyword": keyword,
                "message": f"搜索网页内容时发生错误: {str(e)}"
            }

    def search_web_content_with_js_sync(
        self,
        url: str,
        keyword: str,
        wait_time: float = 2.0,
        timeout: int = 30000,
        extract_text_only: bool = True
    ) -> Dict[str, Any]:
        """
        在网页中搜索特定关键词（同步版本，支持JavaScript渲染）
        每次调用创建独立的浏览器实例，确保线程安全

        :param url: 目标网址
        :param keyword: 要搜索的关键词
        :param wait_time: 页面加载后等待时间（秒），默认2秒
        :param timeout: 页面加载超时时间（毫秒），默认30000
        :param extract_text_only: 是否只提取可见文本（默认True，避免HTML过大）
        :return: 包含搜索结果的字典
        """
        async def _search_with_isolated_browser():
            playwright = None
            browser = None
            try:
                playwright = await async_playwright().start()
                browser = await playwright.chromium.launch(
                    headless=self.headless,
                    executable_path=self.chrome_path
                )
                context = await browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                )
                page = await context.new_page()
                
                try:
                    await page.goto(url, timeout=timeout, wait_until='domcontentloaded')
                    await asyncio.sleep(wait_time)
                    
                    title = await page.title()
                    
                    if extract_text_only:
                        content = await page.locator('body').inner_text()
                        content = ' '.join(content.split())
                    else:
                        content = await page.content()
                    
                    content_lower = content.lower()
                    keyword_lower = keyword.lower()
                    
                    positions = []
                    start = 0
                    while True:
                        pos = content_lower.find(keyword_lower, start)
                        if pos == -1:
                            break
                        positions.append(pos)
                        start = pos + 1
                    
                    contexts = []
                    for pos in positions[:5]:
                        original_start = max(0, pos - 50)
                        original_end = min(len(content), pos + len(keyword_lower) + 50)
                        context_text = content[original_start:original_end]
                        contexts.append({
                            "position": pos,
                            "context": context_text,
                            "highlighted": context_text.replace(keyword, f"[{keyword}]")
                        })
                    
                    return {
                        "success": True,
                        "url": url,
                        "keyword": keyword,
                        "match_count": len(positions),
                        "positions": positions[:5],
                        "contexts": contexts,
                        "title": title,
                        "method": "javascript_rendering",
                        "extract_mode": "text_only" if extract_text_only else "full_html",
                        "message": f"在网页中找到 {len(positions)} 个 '{keyword}' 的匹配项 (JS渲染)"
                    }
                finally:
                    await context.close()
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "url": url,
                    "keyword": keyword,
                    "message": f"搜索网页内容时发生错误: {str(e)}"
                }
            finally:
                if browser:
                    await browser.close()
                if playwright:
                    await playwright.stop()

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(_search_with_isolated_browser())
            finally:
                loop.close()
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "keyword": keyword,
                "message": f"搜索网页内容时发生错误: {str(e)}"
            }

    def __del__(self):
        """析构函数，清理浏览器资源"""
        try:
            if self._browser or self._playwright:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self._close_browser())
        except:
            pass

    # ============================================================
    # 搜索引擎搜索（多引擎并发竞速 — Bing / 360 / DuckDuckGo）
    # ============================================================

    def search_web(
        self,
        query: str,
        max_results: int = 10,
        timeout: int = 15,
    ) -> Dict[str, Any]:
        """
        多引擎并发搜索互联网（Bing / 360 / DuckDuckGo 竞速）

        同时发起 Bing、360、DuckDuckGo 三个引擎的搜索请求，
        谁先成功返回就用谁的结果（winner-takes-all），
        总延迟约等于最快引擎的延迟，不再被单个慢引擎拖累。
        DuckDuckGo 使用短超时（3s），网络不可达时快速失败，不影响整体。

        :param query: 搜索关键词
        :param max_results: 最大结果数，默认10
        :param timeout: 总超时预算（秒），默认15
        :return: 包含搜索结果的字典（含 engine_used 标记实际命中的引擎）
        """
        try:
            request_timeout = max(1, min(int(timeout), 60))
        except (TypeError, ValueError):
            request_timeout = 15

        # 多引擎并发竞速：Bing（实测最快）+ 360（国内兜底）+ DDG（短超时）
        engines = [
            ("bing", lambda: self._search_bing(query, max_results, request_timeout)),
            ("so360", lambda: self._search_so360(query, max_results, request_timeout)),
            ("duckduckgo", lambda: self._search_ddg(query, max_results, min(request_timeout, 3))),
        ]

        errors = []
        executor = ThreadPoolExecutor(max_workers=len(engines))
        try:
            futures = {executor.submit(fn): name for name, fn in engines}
            try:
                for fut in as_completed(futures, timeout=request_timeout):
                    name = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as e:
                        errors.append(f"{name}: {e}")
                        continue
                    if result and result.get("success"):
                        return result
                    errors.append(f"{name}: 未返回结果")
            except TimeoutError:
                errors.append(f"整体超时（>{request_timeout}s）")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        return {
            "success": False,
            "query": query,
            "error": "; ".join(errors) if errors else "所有引擎均失败",
            "message": f"搜索失败（多引擎竞速，{len(engines)} 个引擎均未返回有效结果）",
        }

    def _search_bing(
        self,
        query: str,
        max_results: int,
        timeout: int,
    ) -> Optional[Dict[str, Any]]:
        """Bing 搜索引擎（实测最快，主力引擎），失败时抛出异常由竞速层捕获"""
        session = self._get_session()
        url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}&count={min(max_results, 20)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        resp = session.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        results = []
        for li in soup.select('li.b_algo, .b_algo'):
            title_el = li.select_one('h2 a')
            snippet_el = li.select_one('.b_caption p, .b_lineclamp2')
            cite_el = li.select_one('cite')
            if title_el is None:
                continue
            results.append({
                "title": title_el.get_text(strip=True),
                "url": title_el.get('href', ''),
                "snippet": snippet_el.get_text(strip=True) if snippet_el else '',
                "display_url": cite_el.get_text(strip=True) if cite_el else '',
                "source": "bing",
            })
            if len(results) >= max_results:
                break
        if not results:
            return None
        return {
            "success": True,
            "query": query,
            "total_results": len(results),
            "results": results,
            "method": "bing",
            "engine_used": "bing",
            "message": f"搜索完成（bing），找到 {len(results)} 条结果",
        }

    def _search_so360(
        self,
        query: str,
        max_results: int,
        timeout: int,
    ) -> Optional[Dict[str, Any]]:
        """360 搜索（国内可用，兜底引擎），失败时抛出异常由竞速层捕获"""
        session = self._get_session()
        url = f"https://www.so.com/s?q={urllib.parse.quote(query)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        resp = session.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        results = []
        for li in soup.select('li.res-list'):
            title_el = li.select_one('h3.res-title a, h3.g-title a, h3 a')
            snippet_el = li.select_one('p.res-desc, .res-desc, p')
            cite_el = li.select_one('cite')
            if title_el is None:
                continue
            results.append({
                "title": title_el.get_text(strip=True),
                "url": title_el.get('href', ''),
                "snippet": snippet_el.get_text(strip=True) if snippet_el else '',
                "display_url": cite_el.get_text(strip=True) if cite_el else '',
                "source": "so360",
            })
            if len(results) >= max_results:
                break
        if not results:
            return None
        return {
            "success": True,
            "query": query,
            "total_results": len(results),
            "results": results,
            "method": "so360",
            "engine_used": "so360",
            "message": f"搜索完成（so360），找到 {len(results)} 条结果",
        }

    def _search_ddg(
        self,
        query: str,
        max_results: int,
        timeout: int,
    ) -> Optional[Dict[str, Any]]:
        """DuckDuckGo 搜索引擎（ddgs 库，短超时快速失败），失败时抛出异常由竞速层捕获"""
        ddgs = DDGS(timeout=timeout)
        raw_results = list(ddgs.text(query, max_results=max_results))
        if not raw_results:
            return None
        results = []
        for r in raw_results:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
                "source": "duckduckgo",
            })
        return {
            "success": True,
            "query": query,
            "total_results": len(results),
            "results": results,
            "method": "duckduckgo",
            "engine_used": "duckduckgo",
            "message": f"搜索完成（duckduckgo），找到 {len(results)} 条结果",
        }

    # ============================================================
    # 学术论文搜索（arXiv / Semantic Scholar — 免费 API）
    # ============================================================

    def search_academic(
        self,
        query: str,
        max_results: int = 10,
        source: str = "arxiv",
    ) -> Dict[str, Any]:
        """
        搜索学术论文（免费API，无需API key）

        :param query: 搜索关键词
        :param max_results: 最大结果数，默认10
        :param source: 数据源，支持 'arxiv'、'semantic_scholar'
        :return: 包含论文列表的字典
        """
        if source == "arxiv":
            return self._search_arxiv(query, max_results)
        elif source == "semantic_scholar":
            return self._search_semantic_scholar(query, max_results)
        else:
            return {
                "success": False,
                "query": query,
                "message": f"不支持的学术数据源: {source}，支持: arxiv, semantic_scholar"
            }

    def _search_arxiv(
        self,
        query: str,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """
        使用 arXiv API 搜索学术论文（通过 requests 库）
        """
        try:
            session = self._get_session()
            encoded_query = urllib.parse.quote(query)
            url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"

            resp = session.get(url, headers={'User-Agent': 'Xenon/1.0 (Academic Search)'}, timeout=15)
            resp.raise_for_status()

            root = ET.fromstring(resp.content)

            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom',
            }

            results = []
            for entry in root.findall('atom:entry', ns):
                title_el = entry.find('atom:title', ns)
                summary_el = entry.find('atom:summary', ns)
                published_el = entry.find('atom:published', ns)
                updated_el = entry.find('atom:updated', ns)
                id_el = entry.find('atom:id', ns)

                authors = []
                for author in entry.findall('atom:author', ns):
                    name_el = author.find('atom:name', ns)
                    if name_el is not None:
                        authors.append(name_el.text)

                results.append({
                    "title": (title_el.text or "").strip().replace('\n', ' ') if title_el is not None else "",
                    "authors": authors,
                    "summary": (summary_el.text or "").strip().replace('\n', ' ')[:500] if summary_el is not None else "",
                    "published": published_el.text if published_el is not None else "",
                    "updated": updated_el.text if updated_el is not None else "",
                    "url": (id_el.text or "").strip() if id_el is not None else "",
                    "source": "arxiv",
                })

            return {
                "success": True,
                "query": query,
                "total_results": len(results),
                "results": results,
                "source": "arxiv",
                "message": f"arXiv 搜索完成，找到 {len(results)} 篇论文"
            }
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "source": "arxiv",
                "message": f"arXiv 搜索失败: {str(e)}"
            }

    def _search_semantic_scholar(
        self,
        query: str,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """
        使用 Semantic Scholar API 搜索学术论文（通过 requests 库）
        """
        try:
            session = self._get_session()
            encoded_query = urllib.parse.quote(query)
            url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&limit={min(max_results, 20)}&fields=title,authors,year,externalIds,url,abstract,citationCount"

            resp = session.get(
                url,
                headers={
                    'User-Agent': 'Xenon/1.0 (Scholar Search; mail@xenon.local)',
                    'Accept': 'application/json',
                },
                timeout=15,
            )

            if resp.status_code == 429:
                return {
                    "success": False,
                    "query": query,
                    "message": "Semantic Scholar API 请求过于频繁，请稍后再试",
                    "source": "semantic_scholar",
                    "results": [],
                }

            resp.raise_for_status()
            data = resp.json()

            results = []
            for paper in data.get('data', []):
                authors = []
                for author in paper.get('authors', []):
                    authors.append(author.get('name', ''))

                results.append({
                    "title": paper.get('title', ''),
                    "authors": authors,
                    "year": paper.get('year', ''),
                    "abstract": (paper.get('abstract') or '')[:500],
                    "url": paper.get('url', ''),
                    "citation_count": paper.get('citationCount', 0),
                    "source": "semantic_scholar",
                })

            return {
                "success": True,
                "query": query,
                "total_results": len(results),
                "results": results,
                "source": "semantic_scholar",
                "message": f"Semantic Scholar 搜索完成，找到 {len(results)} 篇论文"
            }
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "source": "semantic_scholar",
                "message": f"Semantic Scholar 搜索失败: {str(e)}"
            }

    def browse_website(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        """
        浏览指定网站并返回页面内容

        :param url: 目标网址
        :param timeout: 请求超时时间（秒），默认10秒
        :return: 包含页面内容和状态的字典
        """
        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {
                    "success": False,
                    "error": "无效的URL格式",
                    "url": url,
                    "message": "提供的URL格式不正确"
                }

            session = self._get_session()
            response = session.get(url, timeout=timeout)

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"HTTP错误: {response.status_code}",
                    "status_code": response.status_code,
                    "url": url,
                    "message": f"访问网站失败，HTTP状态码: {response.status_code}"
                }

            soup = BeautifulSoup(response.content, 'html.parser')

            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else ""

            desc_tag = soup.find('meta', attrs={'name': 'description'})
            description = desc_tag.get('content', '') if desc_tag else ""

            body_content = soup.find('body')
            if body_content:
                text_content = re.sub(r'\s+', ' ', body_content.get_text()).strip()
            else:
                text_content = re.sub(r'\s+', ' ', soup.get_text()).strip()

            max_content_length = 2000
            if len(text_content) > max_content_length:
                text_content = text_content[:max_content_length] + "...(内容已截断)"

            return {
                "success": True,
                "url": url,
                "status_code": response.status_code,
                "title": title,
                "description": description,
                "content": text_content,
                "content_length": len(text_content),
                "headers": dict(response.headers),
                "encoding": response.encoding,
                "elapsed_time": response.elapsed.total_seconds(),
                "message": f"成功访问网站: {title if title else url}"
            }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "请求超时",
                "url": url,
                "timeout": timeout,
                "message": f"访问网站超时，超过{timeout}秒"
            }

        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "连接错误",
                "url": url,
                "message": "无法连接到目标网站，请检查网络连接或网址是否正确"
            }

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "message": f"请求过程中发生错误: {str(e)}"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "message": f"浏览网站时发生未知错误: {str(e)}"
            }

    def check_website_status(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        """
        检查网站状态

        :param url: 目标网址
        :param timeout: 请求超时时间（秒），默认10秒
        :return: 包含网站状态信息的字典
        """
        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {
                    "success": False,
                    "error": "无效的URL格式",
                    "url": url,
                    "message": "提供的URL格式不正确"
                }

            session = self._get_session()
            response = session.head(url, timeout=timeout, allow_redirects=True)

            return {
                "success": True,
                "url": url,
                "status_code": response.status_code,
                "is_reachable": response.status_code == 200,
                "server": response.headers.get('Server', ''),
                "content_type": response.headers.get('Content-Type', ''),
                "content_length": response.headers.get('Content-Length', ''),
                "response_time": response.elapsed.total_seconds(),
                "final_url": response.url,
                "message": f"网站状态: {'在线' if response.status_code == 200 else f'状态码 {response.status_code}'}"
            }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "请求超时",
                "url": url,
                "timeout": timeout,
                "message": f"检查网站状态超时，超过{timeout}秒"
            }

        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "连接错误",
                "url": url,
                "message": "无法连接到目标网站"
            }

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "message": f"请求过程中发生错误: {str(e)}"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "message": f"检查网站状态时发生未知错误: {str(e)}"
            }

    def search_web_content(self, url: str, keyword: str, timeout: int = 10) -> Dict[str, Any]:
        """
        在网页中搜索特定关键词

        :param url: 目标网址
        :param keyword: 要搜索的关键词
        :param timeout: 请求超时时间（秒），默认10秒
        :return: 包含搜索结果的字典
        """
        try:
            browse_result = self.browse_website(url, timeout)

            if not browse_result["success"]:
                return browse_result

            content = browse_result["content"].lower()
            keyword_lower = keyword.lower()

            positions = []
            start = 0
            while True:
                pos = content.find(keyword_lower, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1

            contexts = []
            for pos in positions[:5]:
                original_start = max(0, pos - 50)
                original_end = min(len(browse_result["content"]), pos + len(keyword_lower) + 50)
                context_text = browse_result["content"][original_start:original_end]

                contexts.append({
                    "position": pos,
                    "context": context_text,
                    "highlighted": context_text.replace(keyword, f"[{keyword}]")
                })

            return {
                "success": True,
                "url": url,
                "keyword": keyword,
                "match_count": len(positions),
                "positions": positions[:5],
                "contexts": contexts,
                "title": browse_result.get("title", ""),
                "message": f"在网页中找到 {len(positions)} 个 '{keyword}' 的匹配项"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "keyword": keyword,
                "message": f"搜索网页内容时发生错误: {str(e)}"
            }

    def _browse_single_url(self, url_timeout_tuple: tuple) -> Dict[str, Any]:
        """辅助方法：并发浏览单个URL"""
        url, timeout = url_timeout_tuple
        result = self.browse_website(url, timeout)
        result["url"] = url
        return result

    def _check_single_status(self, url_timeout_tuple: tuple) -> Dict[str, Any]:
        """辅助方法：并发检查单个网站状态"""
        url, timeout = url_timeout_tuple
        result = self.check_website_status(url, timeout)
        result["url"] = url
        return result

    def _search_single_url(self, url_keyword_timeout_tuple: tuple) -> Dict[str, Any]:
        """辅助方法：并发搜索单个URL"""
        url, keyword, timeout = url_keyword_timeout_tuple
        result = self.search_web_content(url, keyword, timeout)
        result["url"] = url
        return result

    def concurrent_browse_websites(
        self,
        urls: List[str],
        timeout: int = 10,
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        并发浏览多个网站

        :param urls: 目标网址列表
        :param timeout: 请求超时时间（秒），默认10秒
        :param max_workers: 最大并发线程数，默认使用实例配置
        :return: 包含所有网站浏览结果的字典
        """
        if not urls:
            return {
                "success": False,
                "error": "URL列表为空",
                "message": "请提供有效的URL列表"
            }

        workers = max_workers if max_workers is not None else self.max_workers
        workers = min(workers, len(urls))

        results = []
        start_time = time.time()

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_url = {
                    executor.submit(self._browse_single_url, (url, timeout)): url
                    for url in urls
                }

                for future in as_completed(future_to_url):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        url = future_to_url[future]
                        results.append({
                            "success": False,
                            "error": str(e),
                            "url": url,
                            "message": f"处理URL时发生错误: {str(e)}"
                        })

            successful = sum(1 for r in results if r.get("success"))
            failed = len(results) - successful
            elapsed_time = time.time() - start_time

            return {
                "success": True,
                "total_urls": len(urls),
                "successful": successful,
                "failed": failed,
                "total_time": elapsed_time,
                "avg_time_per_url": elapsed_time / len(urls) if urls else 0,
                "throughput": len(urls) / elapsed_time if elapsed_time > 0 else 0,
                "results": results,
                "message": f"成功浏览 {successful}/{len(urls)} 个网站，耗时 {elapsed_time:.2f}秒"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"并发浏览网站时发生错误: {str(e)}"
            }

    def concurrent_check_status(
        self,
        urls: List[str],
        timeout: int = 10,
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        并发检查多个网站状态

        :param urls: 目标网址列表
        :param timeout: 请求超时时间（秒），默认10秒
        :param max_workers: 最大并发线程数，默认使用实例配置
        :return: 包含所有网站状态检查结果的字典
        """
        if not urls:
            return {
                "success": False,
                "error": "URL列表为空",
                "message": "请提供有效的URL列表"
            }

        workers = max_workers if max_workers is not None else self.max_workers
        workers = min(workers, len(urls))

        results = []
        start_time = time.time()

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_url = {
                    executor.submit(self._check_single_status, (url, timeout)): url
                    for url in urls
                }

                for future in as_completed(future_to_url):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        url = future_to_url[future]
                        results.append({
                            "success": False,
                            "error": str(e),
                            "url": url,
                            "message": f"检查状态时发生错误: {str(e)}"
                        })

            online_count = sum(1 for r in results if r.get("is_reachable"))
            elapsed_time = time.time() - start_time

            return {
                "success": True,
                "total_urls": len(urls),
                "online": online_count,
                "offline": len(urls) - online_count,
                "total_time": elapsed_time,
                "avg_time_per_url": elapsed_time / len(urls) if urls else 0,
                "throughput": len(urls) / elapsed_time if elapsed_time > 0 else 0,
                "results": results,
                "message": f"成功检查 {len(urls)} 个网站状态，{online_count} 个在线，耗时 {elapsed_time:.2f}秒"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"并发检查网站状态时发生错误: {str(e)}"
            }

    def concurrent_search_multiple_urls(
        self,
        urls: List[str],
        keyword: str,
        timeout: int = 10,
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        并发在多个URL中搜索关键词

        :param urls: 目标网址列表
        :param keyword: 要搜索的关键词
        :param timeout: 请求超时时间（秒），默认10秒
        :param max_workers: 最大并发线程数，默认使用实例配置
        :return: 包含所有搜索结果的字典
        """
        if not urls:
            return {
                "success": False,
                "error": "URL列表为空",
                "message": "请提供有效的URL列表"
            }

        workers = max_workers if max_workers is not None else self.max_workers
        workers = min(workers, len(urls))

        results = []
        start_time = time.time()

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_url = {
                    executor.submit(self._search_single_url, (url, keyword, timeout)): url
                    for url in urls
                }

                for future in as_completed(future_to_url):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        url = future_to_url[future]
                        results.append({
                            "success": False,
                            "error": str(e),
                            "url": url,
                            "keyword": keyword,
                            "message": f"搜索URL时发生错误: {str(e)}"
                        })

            urls_with_matches = sum(1 for r in results if r.get("success") and r.get("match_count", 0) > 0)
            total_matches = sum(r.get("match_count", 0) for r in results if r.get("success"))
            elapsed_time = time.time() - start_time

            return {
                "success": True,
                "total_urls": len(urls),
                "keyword": keyword,
                "urls_with_matches": urls_with_matches,
                "total_matches": total_matches,
                "total_time": elapsed_time,
                "avg_time_per_url": elapsed_time / len(urls) if urls else 0,
                "throughput": len(urls) / elapsed_time if elapsed_time > 0 else 0,
                "results": results,
                "message": f"在 {urls_with_matches}/{len(urls)} 个URL中找到 '{keyword}'，共 {total_matches} 处匹配，耗时 {elapsed_time:.2f}秒"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"并发搜索时发生错误: {str(e)}"
            }

    def _browse_single_url_with_js(self, url_wait_timeout_tuple: tuple) -> Dict[str, Any]:
        """辅助方法：并发浏览单个URL（支持JavaScript渲染）"""
        url, wait_time, timeout = url_wait_timeout_tuple
        result = self.browse_website_with_js_sync(url, wait_time, timeout)
        result["url"] = url
        return result

    def _check_single_status_with_js(self, url_wait_timeout_tuple: tuple) -> Dict[str, Any]:
        """辅助方法：并发检查单个网站状态（支持JavaScript渲染）"""
        url, wait_time, timeout = url_wait_timeout_tuple
        result = self.check_website_status_with_js_sync(url, wait_time, timeout)
        result["url"] = url
        return result

    def _search_single_url_with_js(self, url_keyword_wait_timeout_tuple: tuple) -> Dict[str, Any]:
        """辅助方法：并发搜索单个URL（支持JavaScript渲染）"""
        url, keyword, wait_time, timeout = url_keyword_wait_timeout_tuple
        result = self.search_web_content_with_js_sync(url, keyword, wait_time, timeout)
        result["url"] = url
        return result

    def concurrent_browse_websites_with_js(
        self,
        urls: List[str],
        wait_time: float = 2.0,
        timeout: int = 30000,
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        并发浏览多个网站（支持JavaScript渲染）

        :param urls: 目标网址列表
        :param wait_time: 页面加载后等待时间（秒），默认2秒
        :param timeout: 页面加载超时时间（毫秒），默认30000
        :param max_workers: 最大并发线程数，默认使用实例配置
        :return: 包含所有网站浏览结果的字典
        """
        if not urls:
            return {
                "success": False,
                "error": "URL列表为空",
                "message": "请提供有效的URL列表"
            }

        workers = max_workers if max_workers is not None else self.max_workers
        workers = min(workers, len(urls))

        results = []
        start_time = time.time()

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_url = {
                    executor.submit(self._browse_single_url_with_js, (url, wait_time, timeout)): url
                    for url in urls
                }

                for future in as_completed(future_to_url):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        url = future_to_url[future]
                        results.append({
                            "success": False,
                            "error": str(e),
                            "url": url,
                            "message": f"处理URL时发生错误: {str(e)}"
                        })

            successful = sum(1 for r in results if r.get("success"))
            failed = len(results) - successful
            elapsed_time = time.time() - start_time

            return {
                "success": True,
                "total_urls": len(urls),
                "successful": successful,
                "failed": failed,
                "total_time": elapsed_time,
                "avg_time_per_url": elapsed_time / len(urls) if urls else 0,
                "throughput": len(urls) / elapsed_time if elapsed_time > 0 else 0,
                "results": results,
                "method": "javascript_rendering",
                "message": f"成功浏览 {successful}/{len(urls)} 个网站 (JS渲染)，耗时 {elapsed_time:.2f}秒"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"并发浏览网站时发生错误: {str(e)}"
            }

    def concurrent_check_status_with_js(
        self,
        urls: List[str],
        wait_time: float = 1.0,
        timeout: int = 30000,
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        并发检查多个网站状态（支持JavaScript渲染）

        :param urls: 目标网址列表
        :param wait_time: 页面加载后等待时间（秒），默认1秒
        :param timeout: 页面加载超时时间（毫秒），默认30000
        :param max_workers: 最大并发线程数，默认使用实例配置
        :return: 包含所有网站状态检查结果的字典
        """
        if not urls:
            return {
                "success": False,
                "error": "URL列表为空",
                "message": "请提供有效的URL列表"
            }

        workers = max_workers if max_workers is not None else self.max_workers
        workers = min(workers, len(urls))

        results = []
        start_time = time.time()

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_url = {
                    executor.submit(self._check_single_status_with_js, (url, wait_time, timeout)): url
                    for url in urls
                }

                for future in as_completed(future_to_url):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        url = future_to_url[future]
                        results.append({
                            "success": False,
                            "error": str(e),
                            "url": url,
                            "message": f"检查状态时发生错误: {str(e)}"
                        })

            online_count = sum(1 for r in results if r.get("is_reachable"))
            elapsed_time = time.time() - start_time

            return {
                "success": True,
                "total_urls": len(urls),
                "online": online_count,
                "offline": len(urls) - online_count,
                "total_time": elapsed_time,
                "avg_time_per_url": elapsed_time / len(urls) if urls else 0,
                "throughput": len(urls) / elapsed_time if elapsed_time > 0 else 0,
                "results": results,
                "method": "javascript_rendering",
                "message": f"成功检查 {len(urls)} 个网站状态 (JS渲染)，{online_count} 个在线，耗时 {elapsed_time:.2f}秒"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"并发检查网站状态时发生错误: {str(e)}"
            }

    def concurrent_search_multiple_urls_with_js(
        self,
        urls: List[str],
        keyword: str,
        wait_time: float = 2.0,
        timeout: int = 30000,
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        并发在多个URL中搜索关键词（支持JavaScript渲染）

        :param urls: 目标网址列表
        :param keyword: 要搜索的关键词
        :param wait_time: 页面加载后等待时间（秒），默认2秒
        :param timeout: 页面加载超时时间（毫秒），默认30000
        :param max_workers: 最大并发线程数，默认使用实例配置
        :return: 包含所有搜索结果的字典
        """
        if not urls:
            return {
                "success": False,
                "error": "URL列表为空",
                "message": "请提供有效的URL列表"
            }

        workers = max_workers if max_workers is not None else self.max_workers
        workers = min(workers, len(urls))

        results = []
        start_time = time.time()

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_url = {
                    executor.submit(self._search_single_url_with_js, (url, keyword, wait_time, timeout)): url
                    for url in urls
                }

                for future in as_completed(future_to_url):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        url = future_to_url[future]
                        results.append({
                            "success": False,
                            "error": str(e),
                            "url": url,
                            "keyword": keyword,
                            "message": f"搜索URL时发生错误: {str(e)}"
                        })

            urls_with_matches = sum(1 for r in results if r.get("success") and r.get("match_count", 0) > 0)
            total_matches = sum(r.get("match_count", 0) for r in results if r.get("success"))
            elapsed_time = time.time() - start_time

            return {
                "success": True,
                "total_urls": len(urls),
                "keyword": keyword,
                "urls_with_matches": urls_with_matches,
                "total_matches": total_matches,
                "total_time": elapsed_time,
                "avg_time_per_url": elapsed_time / len(urls) if urls else 0,
                "throughput": len(urls) / elapsed_time if elapsed_time > 0 else 0,
                "results": results,
                "method": "javascript_rendering",
                "message": f"在 {urls_with_matches}/{len(urls)} 个URL中找到 '{keyword}' (JS渲染)，共 {total_matches} 处匹配，耗时 {elapsed_time:.2f}秒"
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"并发搜索时发生错误: {str(e)}"
            }


class WebSearchToolManager:
    """并发网页搜索工具管理器"""

    def __init__(self, max_workers: int = 10, headless: bool = True, chrome_path: Optional[str] = None):
        """
        初始化工具管理器

        :param max_workers: 最大并发线程数，默认10
        :param headless: 是否使用无头浏览器模式，默认True
        :param chrome_path: Chrome浏览器路径，默认None（使用Playwright内置浏览器）
        """
        self.tool = WebSearchTool(max_workers=max_workers, headless=headless, chrome_path=chrome_path)

    def browse_website(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        """
        浏览指定网站并返回页面内容

        :param url: 目标网址
        :param timeout: 请求超时时间（秒），默认10秒
        :return: 包含页面内容和状态的字典
        """
        try:
            return self.tool.browse_website(url, timeout)
        except Exception as e:
            return {"success": False, "error": f"浏览网站失败: {str(e)}"}

    def check_website_status(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        """
        检查网站状态

        :param url: 目标网址
        :param timeout: 请求超时时间（秒），默认10秒
        :return: 包含网站状态信息的字典
        """
        try:
            return self.tool.check_website_status(url, timeout)
        except Exception as e:
            return {"success": False, "error": f"检查网站状态失败: {str(e)}"}

    def search_web_content(self, url: str, keyword: str, timeout: int = 10) -> Dict[str, Any]:
        """
        在网页中搜索特定关键词

        :param url: 目标网址
        :param keyword: 要搜索的关键词
        :param timeout: 请求超时时间（秒），默认10秒
        :return: 包含搜索结果的字典
        """
        try:
            return self.tool.search_web_content(url, keyword, timeout)
        except Exception as e:
            return {"success": False, "error": f"搜索网页内容失败: {str(e)}"}

    def concurrent_browse_websites(
        self,
        urls: List[str],
        timeout: int = 10,
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        并发浏览多个网站

        :param urls: 目标网址列表
        :param timeout: 请求超时时间（秒），默认10秒
        :param max_workers: 最大并发线程数，默认使用实例配置（10）
        :return: 包含所有网站浏览结果的字典
        """
        try:
            return self.tool.concurrent_browse_websites(urls, timeout, max_workers)
        except Exception as e:
            return {"success": False, "error": f"并发浏览网站失败: {str(e)}"}

    def concurrent_check_status(
        self,
        urls: List[str],
        timeout: int = 10,
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        并发检查多个网站状态

        :param urls: 目标网址列表
        :param timeout: 请求超时时间（秒），默认10秒
        :param max_workers: 最大并发线程数，默认使用实例配置（10）
        :return: 包含所有网站状态检查结果的字典
        """
        try:
            return self.tool.concurrent_check_status(urls, timeout, max_workers)
        except Exception as e:
            return {"success": False, "error": f"并发检查网站状态失败: {str(e)}"}

    def concurrent_search_multiple_urls(
        self,
        urls: List[str],
        keyword: str,
        timeout: int = 10,
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        并发在多个URL中搜索关键词

        :param urls: 目标网址列表
        :param keyword: 要搜索的关键词
        :param timeout: 请求超时时间（秒），默认10秒
        :param max_workers: 最大并发线程数，默认使用实例配置（10）
        :return: 包含所有搜索结果的字典
        """
        try:
            return self.tool.concurrent_search_multiple_urls(urls, keyword, timeout, max_workers)
        except Exception as e:
            return {"success": False, "error": f"并发搜索失败: {str(e)}"}

    def browse_website_with_js(
        self,
        url: str,
        wait_time: float = 2.0,
        timeout: int = 30000,
        wait_for_selector: Optional[str] = None,
        extract_text_only: bool = True,
        max_content_length: int = 50000
    ) -> Dict[str, Any]:
        """
        使用JavaScript渲染浏览指定网站（支持SPA应用）

        :param url: 目标网址
        :param wait_time: 页面加载后等待时间（秒），默认2秒
        :param timeout: 页面加载超时时间（毫秒），默认30000
        :param wait_for_selector: 等待特定选择器出现后再提取内容
        :param extract_text_only: 是否只提取可见文本（默认True，避免HTML过大）
        :param max_content_length: 最大内容长度（字符数），超过则截断
        :return: 包含页面内容和状态的字典
        """
        try:
            return self.tool.browse_website_with_js_sync(url, wait_time, timeout, wait_for_selector, extract_text_only, max_content_length)
        except Exception as e:
            return {"success": False, "error": f"浏览网站失败: {str(e)}"}

    def check_website_status_with_js(
        self,
        url: str,
        wait_time: float = 1.0,
        timeout: int = 30000
    ) -> Dict[str, Any]:
        """
        使用JavaScript渲染检查网站状态（支持SPA应用）

        :param url: 目标网址
        :param wait_time: 页面加载后等待时间（秒），默认1秒
        :param timeout: 页面加载超时时间（毫秒），默认30000
        :return: 包含网站状态信息的字典
        """
        try:
            return self.tool.check_website_status_with_js_sync(url, wait_time, timeout)
        except Exception as e:
            return {"success": False, "error": f"检查网站状态失败: {str(e)}"}

    def search_web_content_with_js(
        self,
        url: str,
        keyword: str,
        wait_time: float = 2.0,
        timeout: int = 30000,
        extract_text_only: bool = True
    ) -> Dict[str, Any]:
        """
        在网页中搜索特定关键词（支持JavaScript渲染）

        :param url: 目标网址
        :param keyword: 要搜索的关键词
        :param wait_time: 页面加载后等待时间（秒），默认2秒
        :param timeout: 页面加载超时时间（毫秒），默认30000
        :param extract_text_only: 是否只提取可见文本（默认True，避免HTML过大）
        :return: 包含搜索结果的字典
        """
        try:
            return self.tool.search_web_content_with_js_sync(url, keyword, wait_time, timeout, extract_text_only)
        except Exception as e:
            return {"success": False, "error": f"搜索网页内容失败: {str(e)}"}

    def concurrent_browse_websites_with_js(
        self,
        urls: List[str],
        wait_time: float = 2.0,
        timeout: int = 30000,
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        并发浏览多个网站（支持JavaScript渲染）

        :param urls: 目标网址列表
        :param wait_time: 页面加载后等待时间（秒），默认2秒
        :param timeout: 页面加载超时时间（毫秒），默认30000
        :param max_workers: 最大并发线程数，默认使用实例配置（10）
        :return: 包含所有网站浏览结果的字典
        """
        try:
            return self.tool.concurrent_browse_websites_with_js(urls, wait_time, timeout, max_workers)
        except Exception as e:
            return {"success": False, "error": f"并发浏览网站失败: {str(e)}"}

    def concurrent_check_status_with_js(
        self,
        urls: List[str],
        wait_time: float = 1.0,
        timeout: int = 30000,
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        并发检查多个网站状态（支持JavaScript渲染）

        :param urls: 目标网址列表
        :param wait_time: 页面加载后等待时间（秒），默认1秒
        :param timeout: 页面加载超时时间（毫秒），默认30000
        :param max_workers: 最大并发线程数，默认使用实例配置（10）
        :return: 包含所有网站状态检查结果的字典
        """
        try:
            return self.tool.concurrent_check_status_with_js(urls, wait_time, timeout, max_workers)
        except Exception as e:
            return {"success": False, "error": f"并发检查网站状态失败: {str(e)}"}

    def concurrent_search_multiple_urls_with_js(
        self,
        urls: List[str],
        keyword: str,
        wait_time: float = 2.0,
        timeout: int = 30000,
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        并发在多个URL中搜索关键词（支持JavaScript渲染）

        :param urls: 目标网址列表
        :param keyword: 要搜索的关键词
        :param wait_time: 页面加载后等待时间（秒），默认2秒
        :param timeout: 页面加载超时时间（毫秒），默认30000
        :param max_workers: 最大并发线程数，默认使用实例配置（10）
        :return: 包含所有搜索结果的字典
        """
        try:
            return self.tool.concurrent_search_multiple_urls_with_js(urls, keyword, wait_time, timeout, max_workers)
        except Exception as e:
            return {"success": False, "error": f"并发搜索失败: {str(e)}"}

    # ============================================================
    # 搜索引擎搜索（包装方法）
    # ============================================================

    def search_web(
        self,
        query: str,
        max_results: int = 10,
        timeout: int = 15,
    ) -> Dict[str, Any]:
        """
        多引擎并发搜索互联网（Bing / 360 / DuckDuckGo 竞速，谁快用谁）

        :param query: 搜索关键词
        :param max_results: 最大结果数，默认10
        :param timeout: 总超时预算（秒），默认15
        :return: 包含搜索结果的字典
        """
        try:
            return self.tool.search_web(query, max_results, timeout)
        except Exception as e:
            return {"success": False, "error": f"搜索失败: {str(e)}"}

    def search_academic(
        self,
        query: str,
        max_results: int = 10,
        source: str = "arxiv",
    ) -> Dict[str, Any]:
        """
        搜索学术论文（免费API，无需API key）

        :param query: 搜索关键词
        :param max_results: 最大结果数，默认10
        :param source: 数据源，支持 'arxiv'、'semantic_scholar'
        :return: 包含论文列表的字典
        """
        try:
            return self.tool.search_academic(query, max_results, source)
        except Exception as e:
            return {"success": False, "error": f"学术搜索失败: {str(e)}"}
