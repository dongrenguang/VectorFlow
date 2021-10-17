# -*- coding: utf-8 -*-
"""
计算图

Authors: dongrenguang(dongrenguang@163.com)
Date: 2021/10/16
"""

class Graph(object):
    """计算图类
    """

    def __init__(self, ):
        self.nodes = [] # 计算图内的所有节点列表
        self.name_scope = None # 名称空间

    def add_node(self, node):
        """添加节点
        """
        self.nodes.append(node)

    def reset_value(self):
        """重置所有节点的值
        """
        for node in self.nodes:
            node.reset_value(False)

    def clear_jacobi(self):
        """清除所有节点的雅可比矩阵
        """
        for node in self.nodes:
            node.clear_jacobi()
    
    def node_count(self):
        """获取计算图中的节点数量
        """
        return len(self.nodes)
    
# 全局默认计算图
default_graph = Graph()