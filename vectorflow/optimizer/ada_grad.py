# -*- coding: utf-8 -*-
"""
AdaGrad 优化器

Authors: dongrenguang(dongrenguang@163.com)
Date: 2021/10/18
"""

import numpy as np
from .optimizer import Optimizer
from ..core import get_trainable_variables_from_graph

class AdaGrad(Optimizer):
    """AdaGrad 优化器
    """

    def __init__(self, graph, target, learning_rate=0.01):
        Optimizer.__init__(self, graph, target, learning_rate)
        self.acc_sqr_gradient_dict = dict() # 累积历史梯度平方的词典

    def _update(self):
        """执行具体的梯度更新算法
        """
        trainable_variables = get_trainable_variables_from_graph(self.graph)
        for node in trainable_variables:
            gradient = self.get_gradient(node) # 该节点在当前 Mini Batch 的平均梯度
            # 累积梯度各分量的平方和
            if node not in self.acc_sqr_gradient_dict:
                self.acc_sqr_gradient_dict[node] =  np.power(gradient, 2)
            else:
                self.acc_sqr_gradient_dict[node] += np.power(gradient, 2)
            node.set_value(node.value - self.learning_rate * gradient / np.sqrt(self.acc_sqr_gradient_dict[node] + 1e-10))
