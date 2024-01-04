import torch
import pandas as pd
from torch.utils.data.dataset import Dataset
import numpy as np
import random

# 定义一个Z分数归一化的函数
def z_score_normalization(wave):
    # 将list转换为numpy数组
    wave = np.array(wave)
    # 计算均值和标准差
    mean_value = np.mean(wave)
    std_value = np.std(wave)
    # 按照公式进行归一化
    wave_normalized = (wave - mean_value) / std_value
    # 返回归一化后的数组
    return wave_normalized

class dataset_spectra():
    '''
        随机抽取
    '''
    def __init__(self, path_xlsx, fold, train_test_ratio, seed=42):
        '''
        args:
            path_xlsx:xxx
            fold:...
            train_test_ratio:训练/测试
        '''
        self.df = pd.read_excel(path_xlsx, sheet_name="CalSetA1")
        refl = self.df.iloc[:, 2:]
        gt = self.df["Protein"]

        # 将反射率和ground truth转换为list
        refl_list = refl.values.tolist()
        gt_list = gt.values.tolist()

        # z-score 预处理
        for d in refl_list:
            d = z_score_normalization(d)

        # 创建一个空的tuple list
        tuple_list = []

        # 遍历反射率和ground truth列表，将每个元素组合成一个tuple，并添加到tuple list中
        for i in range(len(refl_list)):
            tuple_list.append((refl_list[i], gt_list[i]))

        tuple_list = sorted(tuple_list, key=lambda x: x[1])

        self.training_set = []
        self.test_set = []

        for i in range(len(tuple_list)):
            if i % (train_test_ratio+1) == 0:
                self.test_set.append(tuple_list[i])
            else:
                self.training_set.append(tuple_list[i])
        
        # folds
        subfolds = []
        for i in range(fold):
            subfolds.append([])

        for i in range(len(self.training_set)):
            subfolds[i%fold].append(self.training_set[i])

        self.folds = []
        for i in range(fold):
            self.folds.append([[], []])

        for i in range(fold):
            for j in range(fold):
                if i == j:
                    self.folds[i][1] = subfolds[j]
                else:
                    self.folds[i][0] += subfolds[j]


    def random_batch(fold, batch_size):
        # 从self.folds[fold][0]这个列表中随机抽取batch_size条数据返回一个新的list
        pass
        
    def random_batch(self, fold, batch_size):
        data = self.folds[fold][0]  # 获取数据列表
        random_batch = random.sample(data, batch_size)  # 随机抽取batch_size条数据
        return random_batch

    