# -*- coding: utf-8 -*-
"""
基础函数

Authors: dongrenguang(dongrenguang@163.com)
Date: 2021/10/17
"""

from .graph import default_graph
from .variable import Variable

def get_node_from_graph(node_name, name_scope=None, graph=None):
    """根据节点名和名称空间，从计算图中获取节点
    """
    if graph is None:
        graph = default_graph
    if name_scope:
        node_name = "{}/{}".format(name_scope, node_name)
    for node in graph.nodes:
        if node.name == node_name:
            return node
    return None

def get_trainable_variables_from_graph(graph=None):
    """根据节点名和名称空间，从计算图中获取可训练的变量节点
    """
    if graph is None:
        graph = default_graph
    return [node for node in graph.nodes if (isinstance(node, Variable) and node.trainable)]
