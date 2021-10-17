# -*- coding: utf-8 -*-
"""
变量节点

Authors: dongrenguang(dongrenguang@163.com)
Date: 2021/10/16
"""

import numpy as np
from .node import Node

class Variable(Node):
    """变量结点
    """

    def __init__(self, dim, init=False, trainable=True, **kargs):
        Node.__init__(self, **kargs) # 变量结点没有父节点
        self.dim = dim # 变量的形状
        self.trainable = trainable # 变量结点是否参与训练
        if init:
            # 按正态分布随机初始化变量的值
            self.value = np.mat(np.random.normal(0, 0.001, dim))
    
    def set_value(self, value):
        """为变量赋值
        """
        assert isinstance(value, np.matrix) and value.shape == self.dim
        self.reset_value(True) # 亦重置后代节点的值
        self.value = value
