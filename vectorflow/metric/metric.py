# -*- coding: utf-8 -*-
"""
模型评估

Authors: dongrenguang(dongrenguang@163.com)
Date: 2021/10/24
"""

import abc
import numpy as np
from ..core import Node

class Metric(Node):
    """评估指标算子抽象基类
    """

    def __init__(self, *parents, **kargs):
        kargs['need_save'] = kargs.get('need_save', False)
        Node.__init__(self, *parents, **kargs)
        self.init()

    def reset(self):
        self.reset_value()
        self.init()

    @abc.abstractmethod
    def init(self):
        # 初始化节点，由具体子类实现
        pass

    @staticmethod
    def prob_to_label(prob, threshold=0.5):
        """将预估值转化为标签
        """
        if prob.shape[0] > 1:
            # 多分类，预测类别为概率最大的类别
            labels = np.zeros((prob.shape[0], 1))
            labels[np.argmax(prob, axis=0)] = 1
        else:
            # 二分类，以thresholds为概率阈值判断类别
            labels = np.where(prob >= threshold, 1, -1)

        return labels

    def get_jacobi_with_parent(self, parent):
        # 对于评估指标节点，计算雅可比无意义
        raise NotImplementedError()

    def value_str(self):
        return "{}: {:.4f} ".format(self.__class__.__name__, self.value)


class Accuracy(Metric):
    """正确率
    """

    def __init(self, *parents, **kargs):
        Metric.__init__(self, *parents, **kargs)

    def init(self):
        self.correct_num = 0
        self.total_num = 0

    def compute(self):
        """Accrucy: (TP + TN) / TOTAL
        这里假设第一个父节点是预测值（概率），第二个父节点是标签
        """
        pred = Metric.prob_to_label(self.parents[0].value)
        gt = self.parents[1].value
        assert len(pred) == len(gt)

        # TODO:
        if pred.shape[0] > 1:
            self.correct_num += np.sum(np.multiply(pred, gt))
            self.total_num += pred.shape[1]
        else:
            self.correct_num += np.sum(pred == gt)
            self.total_num += len(pred)

        self.value = 0
        if self.total_num != 0:
            self.value = float(self.correct_num) / self.total_num


class Precision(Metric):
    """查准率
    """

    def __init__(self, *parents, **kargs):
        Metric.__init__(self, *parents, **kargs)

    def init(self):
        self.true_pos_num = 0
        self.pred_pos_num = 0

    def compute(self):
        """Precision： TP / (TP + FP)
        """
        pred = Metric.prob_to_label(self.parents[0].value)
        gt = self.parents[1].value
        self.true_pos_num += np.sum(pred == gt and pred == 1)
        self.pred_pos_num += np.sum(pred == 1)
        self.value = 0
        if self.pred_pos_num != 0:
            self.value = float(self.true_pos_num) / self.pred_pos_num


class Recall(Metric):
    """查全率
    """

    def __init__(self, *parents, **kargs):
        Metric.__init__(self, *parents, **kargs)

    def init(self):
        self.true_pos_num = 0
        self.gt_pos_num = 0

    def compute(self):
        """Recall： TP / (TP + FN)
        """
        pred = Metric.prob_to_label(self.parents[0].value)
        gt = self.parents[1].value
        self.true_pos_num += np.sum(pred == gt and pred == 1)
        self.gt_pos_num += np.sum(gt == 1)
        self.value = 0
        if self.gt_pos_num != 0:
            self.value = float(self.true_pos_num) / self.gt_pos_num


class ROC(Metric):
    """ROC曲线
    """

    def __init__(self, *parents, **kargs):
        Metric.__init__(self, *parents, **kargs)

    def init(self):
        self.count = 100
        self.gt_pos_num = 0
        self.gt_neg_num = 0
        self.true_pos_num = np.array([0] * self.count)
        self.false_pos_num = np.array([0] * self.count)
        self.tpr = np.array([0] * self.count) # 真阳率
        self.fpr = np.array([0] * self.count) # 假阳率

    def compute(self):
        prob = self.parents[0].value
        gt = self.parents[1].value
        self.gt_pos_num += np.sum(gt == 1)
        self.gt_neg_num += np.sum(gt == -1)

        thresholds = list(np.arange(0.01, 1.0, 0.01)) # 最小值0.01，最大值0.99，步长0.01，生成99个阈值
        # 分别使用多个阈值产生类别预测，与标签比较
        for index in range(0, len(thresholds)):
            pred = Metric.prob_to_label(prob, thresholds[index])
            self.true_pos_num[index] += np.sum(pred == gt and pred == 1)
            self.false_pos_num[index] += np.sum(pred != gt and pred == 1)

        if self.gt_pos_num != 0 and self.gt_neg_num != 0:
            self.tpr = self.true_pos_num / self.gt_pos_num
            self.fpr = self.false_pos_num / self.gt_neg_num

    def value_str(self):
        # 绘制 ROC 曲线
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('TkAgg')
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.plot(self.fpr, self.tpr)
        plt.show()
        return ''

class ROC_AUC(Metric):
    """ROC AUC
    """

    def __init__(self, *parents, **kargs):
        Metric.__init__(self, *parents, **kargs)

    def init(self):
        self.gt_pos_preds = []
        self.gt_neg_preds = []

    def compute(self):
        prob = self.parents[0].value
        gt = self.parents[1].value

        # 简单起见，假设只有一个元素
        if gt[0, 0] == 1:
            self.gt_pos_preds.append(prob)
        else:
            self.gt_neg_preds.append(prob)

        self.total = len(self.gt_pos_preds) * len(self.gt_neg_preds)

    def value_str(self):
        count = 0

        # 遍历m x n个样本对，计算正类概率大于负类概率的数量
        for gt_pos_pred in self.gt_pos_preds:
            for gt_neg_pred in self.gt_neg_preds:
                if gt_pos_pred > gt_neg_pred:
                    count += 1
        # 使用这个数量，除以m x n
        self.value = float(count) / self.total
        return "{}: {:.4f} ".format(self.__class__.__name__, self.value)


class F1Score(Metric):
    """F1 Score
    """

    def __init__(self, *parents, **kargs):
        Metric.__init__(self, *parents, **kargs)

    def init(self):
        self.true_pos_num = 0
        self.pred_pos_num = 0
        self.gt_pos_num = 0

    def compute(self):
        """F1 Score: (2 * pre * recall) / (pre + recall)
        """
        prob = self.parents[0].value
        gt = self.parents[1].value
        pred = Metric.prob_to_label(prob)
        self.true_pos_num += np.sum(pred == gt and pred == 1)
        self.pred_pos_num += np.sum(pred == 1)
        self.gt_pos_num += np.sum(gt == 1)

        precision = 0
        recall = 0
        if self.pred_pos_num != 0:
            precision = float(self.true_pos_num) / self.pred_pos_num
        if self.gt_pos_num != 0:
            recall = float(self.true_pos_num) / self.gt_pos_num
        
        self.value = 0
        if precision + recall != 0:
            self.value = 2 * np.multiply(precision, recall) / (precision + recall)
