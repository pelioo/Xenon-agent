# -*- coding: utf-8 -*-
"""Xenon 异常定义（P5：从 Xenon.py 迁出，供 agent_runtime 等核心模块使用）。"""


class InterruptedException(Exception):
    """自定义中断异常"""
    pass
