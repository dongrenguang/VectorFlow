# -*- coding: utf-8 -*-
"""
朴素梯度下降优化器

Authors: dongrenguang(dongrenguang@163.com)
Date: 2021/10/18
"""

from .optimizer import Optimizer
from ..core import get_trainable_variables_from_graph

class GradientDescent(Optimizer):
    """朴素梯度下降优化器
    """

    def __init__(self, graph, target, learning_rate=0.01):
        Optimizer.__init__(self, graph, target, learning_rate)

    def _update(self):
        """执行具体的梯度更新算法
        """
        trainable_variables = get_trainable_variables_from_graph(self.graph)
        for node in trainable_variables:
            gradient = self.get_gradient(node) # 该节点在当前 Mini Batch 的平均梯度
            node.set_value(node.value - self.learning_rate * gradient)
