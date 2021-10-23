# -*- coding: utf-8 -*-
"""
损失函数

Authors: dongrenguang(dongrenguang@163.com)
Date: 2021/10/17
"""

import numpy as np
from ..core import Node
from ..operator import SoftMax

class LossFunction(Node):
    """损失函数抽象类
    """
    pass

class PerceptionLoss(LossFunction):
    """感知机损失
    """

    def compute(self):
        # 输入为正时为0，输入为负时为输入的相反数
        self.value = np.mat(np.where(
            self.parents[0].value >= 0.0, 0.0, -self.parents[0].value))

    def get_jacobi_with_parent(self, parent):
        # 雅可比矩阵为对角阵，每个对角线元素对应一个父节点元素。若父节点元素大于0，则相应对角线元素（偏导数）为0，否则为-1。
        diag = np.where(parent.value >= 0.0, 0.0, -1)
        return np.diag(diag.ravel())

class LogLoss(LossFunction):
    """LogLoss
    """

    def compute(self):
        assert len(self.parents) == 1
        x = self.parents[0].value
        self.value = np.log(1 + np.power(np.e, np.where(-x > 1e2, 1e2, -x)))

    def get_jacobi_with_parent(self, parent):
        x = parent.value
        diag = -1 / (1 + np.power(np.e, np.where(x > 1e2, 1e2, x)))
        return np.diag(diag.ravel())

class CrossEntropyWithSoftMax(LossFunction):
    """对第一个父节点施加SoftMax之后，再以第二个父节点为标签One-Hot向量计算交叉熵
    """

    def compute(self):
        prob = SoftMax.softmax(self.parents[0].value)
        self.value = np.mat(-np.sum(np.multiply(self.parents[1].value, np.log(prob + 1e-10))))

    def get_jacobi_with_parent(self, parent):
        prob = SoftMax.softmax(self.parents[0].value)
        if parent is self.parents[0]:
            return (prob - self.parents[1].value).T
        else:
            return (-np.log(prob)).T
