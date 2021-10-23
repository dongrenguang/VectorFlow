# -*- coding: utf-8 -*-
"""
层

Authors: dongrenguang(dongrenguang@163.com)
Date: 2021/10/23
"""

from ..core import Variable
from ..operator import Add, MatMul, Logistic, ReLU

def fc(input, input_size, size, activation):
    """全连接层

    Args:
        input: 输入向量
        input_size: 输入向量的维度
        size: 该层神经元的数量，也就是输出向量的维数
        activation: 激活函数

    Returns:
        施加了激活函数后的输出向量
    """

    weights = Variable((size, input_size), init=True, trainable=True)
    bias = Variable((size, 1), init=True, trainable=True)
    affine = Add(MatMul(weights, input), bias)

    if activation == "ReLU":
        return ReLU(affine)
    elif activation == "Logistic":
        return Logistic(affine)
    else:
        return affine
