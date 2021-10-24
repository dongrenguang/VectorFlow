# -*- coding: utf-8 -*-
"""
训练器

Authors: dongrenguang(dongrenguang@163.com)
Date: 2021/10/24
"""

import abc
import time
import numpy as np

class Trainer(object):
    """训练器
    """

    def __init__(self, input_x, input_y,
                 loss_op, optimizer,
                 epoches, batch_size=8,
                 eval_on_train=False, metrics_ops=None, *args, **kargs):
        self.inputs = input_x # 计算图的输入节点，可以有多个，类型是list
        self.input_y = input_y # 计算图的标签节点
        self.loss_op = loss_op # 损失函数
        self.optimizer = optimizer # 优化器
        self.epoches = epoches # 训练执行的迭代轮数
        self.epoch = 0
        self.batch_size = batch_size # 批大小
        self.eval_on_train = eval_on_train # 是否在训练迭代中进行评估
        self.metrics_ops = metrics_ops # 评估指标列表
        self.print_iteration_interval = kargs.get('print_iteration_interval', 100)

    def train_and_eval(self, train_x, train_y, test_x=None, test_y=None):
        """开始训练（评估）流程
        """
        self._variable_weights_init()
        self.main_loop(train_x, train_y, test_x, test_y)

    def main_loop(self, train_x, train_y, test_x, test_y):
        """训练（评估）的主循环
        """
        for self.epoch in range(self.epoches):
            self.train(train_x, train_y)
            if self.eval_on_train and test_x is not None and test_y is not None:
                self.eval(test_x, test_y)

    def train(self, train_x, train_y):
        """训练
        """
        print('- Epoch [{}] train start, batch size: {}, train data size: {}'.format(
            self.epoch + 1, self.batch_size, len(train_x)))
        start_time = time.time()
        last_iter_start_time = time.time()
        # 遍历训练数据集
        for i in range(len(list(train_x.values())[0])):
            self.one_step(self._get_input_values(train_x, i), train_y[i]) # 使用一个样本，执行一次前向传播和反向传播

            if (i+1) % self.print_iteration_interval == 0:
                print('-- iteration [{}] finished, time cost: {:.2f}  and loss value: {:4f}'.format(
                    i, time.time() - last_iter_start_time, float(self.loss_op.value)))
                last_iter_start_time = time.time()

            # 如果次数超过批大小，执行一次更新
            if (i+1) % self.batch_size == 0:
                last_batch_end_time = time.time()
                self._optimizer_update()

        print('- Epoch [{}] train finished, time cost: {:.2f}'.format(
            self.epoch + 1, time.time() - start_time))

    def eval(self, test_x, test_y):
        """使用测试集进行模型评估
        """
        for metrics_op in self.metrics_ops:
            metrics_op.reset()

        for i in range(len(list(test_x.values())[0])):
            self.one_step(self._get_input_values(test_x, i), test_y[i], is_eval=True)
            for metrics_op in self.metrics_ops:
                metrics_op.forward()

        metrics_str = 'Epoch [{}] evaluation metrics '.format(self.epoch + 1)
        for metrics_op in self.metrics_ops:
            metrics_str += metrics_op.value_str()

        print(metrics_str)

    def _get_input_values(self, x, index):
        """x是dict类型的数据集，需要取出第index个样本
        """
        input_values = dict()
        for input_node_name in x.keys():
            input_values[input_node_name] = x[input_node_name][index]

        return input_values

    def one_step(self, data_x, data_y, is_eval=False):
        """执行一次前向计算和一次后向计算(可能)
        """
        for i in range(len(self.inputs)):
            # 根据输入节点的名称，从输入数据dict中找到对应数据
            input_value = data_x.get(self.inputs[i].name)
            self.inputs[i].set_value(np.mat(input_value).T)
        
        self.input_y.set_value(np.mat(data_y).T)

        # 只有在训练阶段才执行优化器
        if not is_eval:
            self.optimizer.one_step()

    @abc.abstractmethod
    def _variable_weights_init(self):
        """权值变量初始化，具体的初始化操作由子类完成
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _optimizer_update(self):
        """
        调用优化器执行参数更新
        """
        raise NotImplementedError()
