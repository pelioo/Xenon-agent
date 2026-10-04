# -*- coding: utf-8 -*-
"""Xenon 内置插件包（P1 第一批）。

每个模块遵循加载器约定：
- 模块级 PLUGIN dict：{id, name, inject: [服务key], ...}
- activate(ctx) / deactivate(ctx) 可选（缺省视为无生命周期动作）
- 激活只做"注册服务/幂等启动"，不得改变现有调用路径（默认装配 = 现状行为）
"""
