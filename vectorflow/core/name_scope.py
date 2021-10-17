# -*- coding: utf-8 -*-
"""
计算图的名称空间

Authors: dongrenguang(dongrenguang@163.com)
Date: 2021/10/17
"""

from .graph import default_graph

class NameScope(object):
    """计算图的名称空间类
    """

    def __init__(self, name_scope):
        self.name_scope = name_scope
    
    def __enter__(self):
        default_graph.name_scope = self.name_scope
        return self
    
    def __exit__(self):
        default_graph.name_scope = None
