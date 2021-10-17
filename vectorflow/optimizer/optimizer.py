# -*- coding: utf-8 -*-
"""
优化器

Authors: dongrenguang(dongrenguang@163.com)
Date: 2021/10/17
"""

import abc
from ..core import get_trainable_variables_from_graph

class Optimizer(object):
    """优化器抽象类
    """

    def __init__(self, graph, target, learning_rate=0.01):
        self.graph = graph # 计算图
        self.target = target # 目标节点
        self.learning_rate = learning_rate # 学习率

        self.acc_gradient = dict() # 为每个参与训练的节点累加一个 Mini Batch 的全部样本的梯度
        self.acc_no = 0 # 当前累加的节点数

    def one_step(self):
        """计算并累加样本的梯度
        """
        self.forward_backward()
        self.acc_no += 1

    def forward_backward(self):
        """执行一次前向传播和反向传播：前向传播计算结果节点的值，反向传播计算结果节点对各个可训练的变量节点的雅可比矩阵
        """
        self.target.forward() # 前向传播计算结果节点的值

        self.graph.clear_jacobi() # 清除计算图中所有节点的雅可比矩阵
        trainable_variables = get_trainable_variables_from_graph(self.graph)
        # 反向传播计算结果节点对各个可训练的变量节点的雅可比矩阵
        for node in trainable_variables:
            node.backward(self.target)

            # 最终结果（标量）对节点值的雅可比是一个行向量，其转置是梯度（列向量）。
            # 这里将梯度 reshape 成与节点值相同的形状，好对节点值进行更新。
            gradient = node.jacobi.T.reshape(node.shape())
            if node not in self.acc_gradient:
                self.acc_gradient[node] = gradient
            else:
                self.acc_gradient[node] += gradient

    def update(self):
        """更新各个可训练的变量节点的值
        """
        self._update()

        # 清空累加器
        self.acc_gradient.clear()
        self.acc_no = 0

    def _update(self):
        """抽象方法，执行具体的梯度更新算法
        """

    def get_gradient(self, node):
        """返回节点的平均梯度
        """
        assert node in self.acc_gradient
        return self.acc_gradient[node] / self.acc_no
