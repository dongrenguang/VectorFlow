# -*- coding: utf-8 -*-
"""
辅助工具

Authors: dongrenguang(dongrenguang@163.com)
Date: 2021/10/24
"""

class ClassMining(object):
    @classmethod
    def get_subclass_list(cls, model):
        """获取子类（包含所有的子孙类）列表
        """
        subclass_list = []
        for subclass in model.__subclasses__():
            subclass_list.append(subclass)
            subclass_list.extend(cls.get_subclass_list(subclass))
        return subclass_list

    @classmethod
    def get_subclass_dict(cls, model):
        """获取字典形式的子类列表
        """
        subclass_list = cls.get_subclass_list(model=model)
        return {k: k.__name__ for k in subclass_list}

    @classmethod
    def get_subclass_names(cls, model):
        """获取子类名称列表
        """
        subclass_list = cls.get_subclass_list(model=model)
        return [k.__name__ for k in subclass_list]

    @classmethod
    def get_instance_by_subclass_name(cls, model, name):
        """根据子类名返回子类
        """
        for subclass in model.__subclasses__():
            if subclass.__name__ == name:
                return subclass
            instance = cls.get_instance_by_subclass_name(subclass, name)
            if instance:
                return instance
