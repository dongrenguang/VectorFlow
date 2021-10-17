# -*- coding: utf-8 -*-
"""
损失函数

Authors: dongrenguang(dongrenguang@163.com)
Date: 2021/10/17
"""

import numpy as np
from ..core import Node

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
