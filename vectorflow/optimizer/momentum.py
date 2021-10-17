# -*- coding: utf-8 -*-
"""
冲量优化器

Authors: dongrenguang(dongrenguang@163.com)
Date: 2021/10/18
"""

from .optimizer import Optimizer
from ..core import get_trainable_variables_from_graph

class Momentum(Optimizer):
    """冲量优化器
    """

    def __init__(self, graph, target, learning_rate=0.01, decay_rate=0.9):
        Optimizer.__init__(self, graph, target, learning_rate)
        self.decay_rate = decay_rate # 动量衰减率
        self.momentum_dict = dict() # 累积历史动量的词典

    def _update(self):
        """执行具体的梯度更新算法
        """
        trainable_variables = get_trainable_variables_from_graph(self.graph)
        for node in trainable_variables:
            gradient = self.get_gradient(node) # 该节点在当前 Mini Batch 的平均梯度
            if node not in self.momentum_dict:
                self.momentum_dict[node] =  gradient
            else:
                self.momentum_dict[node] = self.decay_rate * self.momentum_dict[node] + gradient
            node.set_value(node.value - self.learning_rate * self.momentum_dict[node])
