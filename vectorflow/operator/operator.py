# -*- coding: utf-8 -*-
"""
操作符

Authors: dongrenguang(dongrenguang@163.com)
Date: 2021/10/17
"""

import numpy as np
from ..core import Node

class Operator(Node):
    """操作符抽象类
    """
    pass

class Add(Operator):
    """矩阵加法
    """

    def compute(self):
        """根据父节点的值计算本节点的值
        """
        self.value = np.mat(np.zeros(self.parents[0].shape()))
        for parent in self.parents:
            self.value += parent.value

    def get_jacobi_with_parent(self, parent):
        """计算本节点对某个父节点的雅可比矩阵
        """
        return np.mat(np.eye(self.dimension())) # 矩阵之和对其中任一个矩阵的雅可比矩阵是单位矩阵

class Multiply(Operator):
    """两个父节点的值是相同形状的矩阵，将它们对应位置的值相乘
    """

    def compute(self):
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi_with_parent(self, parent):
        if parent is self.parents[0]:
            return np.diag(self.parents[1].value.A1)
        else:
            return np.diag(self.parents[0].value.A1)

class MatMul(Operator):
    """矩阵乘法
    """

    def compute(self):
        """根据父节点的值计算本节点的值
        """
        assert len(self.parents) == 2 and self.parents[0].shape()[1] == self.parents[1].shape()[0]
        self.value = self.parents[0].value * self.parents[1].value
    
    def get_jacobi_with_parent(self, parent):
        """计算本节点对某个父节点的雅可比矩阵
        """
        zeros = np.mat(np.zeros((self.dimension(), parent.dimension())))
        if parent is self.parents[0]:
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            row_sort = np.arange(self.dimension()).reshape(self.shape()[::-1]).T.ravel()
            col_sort = np.arange(parent.dimension()).reshape(parent.shape()[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]

class Step(Operator):
    """阶跃函数
    """

    def compute(self):
        """根据父节点的值计算本节点的值
        """
        self.value = np.mat(np.where(self.parents[0].value >= 0.0, 1.0, 0.0))

    def get_jacobi_with_parent(self, parent):
        """计算本节点对某个父节点的雅可比矩阵
        """
        return np.zeros(np.where(self.parents[0].value.A1 >= 0.0, 0.0, -1.0))

class Logistic(Operator):
    """Logistic函数
    """

    def compute(self):
        x = self.parents[0].value
        self.value = np.mat(1.0 / (1.0 + np.power(np.e, np.where(-x > 1e2, 1e2, -x))))

    def get_jacobi_with_parent(self, parent):
        return np.diag(np.mat(np.multiply(self.value, 1 - self.value)).A1)

class SoftMax(Operator):
    """SoftMax函数
    """

    @staticmethod
    def softmax(a):
        a[a > 1e2] = 1e2 # 防止溢出
        ep = np.power(np.e, a)
        return ep / np.sum(ep)

    def compute(self):
        self.value = SoftMax.softmax(self.parents[0].value)

    def get_jacobi_with_parent(self, parent):
        """我们不实现SoftMax节点的get_jacobi函数，训练时使用CrossEntropyWithSoftMax节点
        """
        raise NotImplementedError("Don't use SoftMax's get_jacobi_with_parent")

class ReLU(Operator):
    """LeakyReLU函数
    """

    nslope = 0.1 # 负半轴的斜率

    def compute(self):
        self.value = np.mat(np.where(
            self.parents[0].value > 0.0,
            self.parents[0].value,
            self.nslope * self.parents[0].value)
        )

    def get_jacobi_with_parent(self, parent):
        return np.diag(np.where(self.parents[0].value.A1 > 0.0, 1.0, self.nslope))

def fill_diagonal(to_be_filled, filler):
    """将 filler 矩阵填充在 to_be_filled 的对角线上
    """
    assert to_be_filled.shape[0] / filler.shape[0] == to_be_filled.shape[1] / filler.shape[1]
    n = int(to_be_filled.shape[0] / filler.shape[0])

    r, c = filler.shape
    for i in range(n):
        to_be_filled[i * r:(i + 1) * r, i * c:(i + 1) * c] = filler

    return to_be_filled
