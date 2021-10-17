# -*- coding: utf-8 -*-
"""
计算图节点基类

Authors: dongrenguang(dongrenguang@163.com)
Date: 2021/10/16
"""

import abc
import numpy as np
from .graph import default_graph

class Node(object):
    """
    计算图节点基类
    """
    def __init__(self, *parents, **kargs):
        self.graph = kargs.get('graph', default_graph)
        self.need_save = kargs.get('need_save', True)
        
        self.name = self.gen_node_name(**kargs) # 节点名称
        self.parents = list(parents) # 父节点列表
        self.children = [] # 子节点列表
        self.value = None # 本节点的值
        self.jacobi = None # 结果节点对本节点的雅可比矩阵

        for parent in parents:
            parent.children.append(self)
        
        self.graph.add_node(self)

    def get_parents(self):
        """获取本节点的父节点列表
        """
        return self.parents

    def get_children(self):
        """获取本节点的子节点列表
        """
        return self.children

    @abc.abstractmethod
    def compute(self):
        """抽象方法，根据父节点的值计算本节点的值
        """

    def reset_value(self, recursive=True):
        """重置本节点的值，并递归重置本节点的后代节点的值
        """
        self.value = None
        if recursive:
            for child in self.children:
                child.reset_value(True)

    def forward(self):
        """前向传播，计算本节点的值
        """
        for parent in self.parents:
            if parent.value is None:
                parent.forward()
        self.compute()

    def backward(self, result_node):
        """反向传播，计算结果节点对本节点的雅可比矩阵
        """
        if self.jacobi is not None:
            return self.jacobi

        if result_node is self:
            return np.mat(np.eye(self.dimension()))
        else:
            self.jacobi = np.mat(
                np.zeros((result_node.dimension, self.dimension()))
            )
            for child in self.children:
                self.jacobi += child.backward() * child.get_jacobi_with_parent(self)
        return self.jacobi

    @abc.abstractmethod
    def get_jacobi_with_parent(self, parent):
        """抽象方法，计算本节点对某个父节点的雅可比矩阵
        """

    def clear_jacobi(self):
        """清空结果节点对本节点的雅可比矩阵
        """
        self.jacobi = None

    def dimension(self):
        """获取本节点的值展开成向量后的维数
        """
        return self.value.shape()[0] * self.value.shape()[1]
    
    def shape(self):
        """获取本节点的值做为矩阵的形状
        """
        return self.value.shape()

    def _gen_node_name(self, **kargs):
        """生成节点名称
        """
        name = kargs.get('name', '{}:{}'.format(self.__class__.__name__, self.graph.node_count()))
        if self.graph.name_scope:
            name = '{}/{}'.format(self.graph.name_scope, name)
        return name
