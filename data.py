from __future__ import print_function

from torchtools import *
import torch.utils.data as data
import random
import os
import numpy as np
import pickle
import pandas as pd


class CreditLoader_train(data.Dataset):
    def __init__(self, train_data=None):
        super(CreditLoader_train, self).__init__()
        self.data_size = [198]
        self.data_train = train_data

    def get_task_batch(self,
                       num_tasks=5,
                       num_ways=20,
                       num_shots=1,
                       num_queries=1):

        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size, dtype='float32')
            label = np.zeros(shape=[num_tasks], dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] + self.data_size, dtype='float32')
            label = np.zeros(shape=[num_tasks], dtype='float32')
            query_data.append(data)
            query_label.append(label)

        # for each task
        for t_idx in range(num_tasks):
            for c_idx in range(num_ways):
                class_data_list = random.sample(self.data_train[c_idx], num_shots + num_queries)

                for i_idx in range(num_shots):
                    support_data[i_idx + c_idx * num_shots][t_idx] = class_data_list[i_idx]
                    support_label[i_idx + c_idx * num_shots][t_idx] = c_idx

                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = class_data_list[num_shots + i_idx]
                    query_label[i_idx + c_idx * num_queries][t_idx] = c_idx

        # convert to tensor (num_tasks x (num_ways * (num_supports + num_queries)) x ...)
        support_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in query_label], 1)

        return [support_data, support_label, query_data, query_label]


class CreditLoader_test(data.Dataset):
    def __init__(self, train_data=None, test_data=None):
        super(CreditLoader_test, self).__init__()
        self.data_size = [198]
        self.data_train = train_data
        data_test_0 = [np.append(i, [0]) for i in test_data[0]]
        data_test_1 = [np.append(i, [1]) for i in test_data[1]]
        self.data_test = data_test_0 + data_test_1
        random.shuffle(self.data_test)

    def get_task_batch(self,
                       num_tasks=5,
                       num_ways=20,
                       num_shots=1,
                       num_queries=1,
                       iter=0):

        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            query_data.append(data)
            query_label.append(label)

        # for each task
        for t_idx in range(num_tasks):
            for c_idx in range(num_ways):
                class_data_list = random.sample(self.data_train[c_idx], num_shots)

                for i_idx in range(num_shots):
                    support_data[i_idx + c_idx * num_shots][t_idx] = class_data_list[i_idx]
                    support_label[i_idx + c_idx * num_shots][t_idx] = c_idx

                query_data[c_idx][t_idx] = self.data_test[iter * num_tasks * num_ways + t_idx * num_ways + c_idx][:-1]
                query_label[c_idx][t_idx] = self.data_test[iter * num_tasks * num_ways + t_idx * num_ways + c_idx][-1]

        # convert to tensor (num_tasks x (num_ways * (num_supports + num_queries)) x ...)
        support_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in query_label], 1)

        return [support_data, support_label, query_data, query_label]


def dataset_import(root):
    data_train = {0: [],
                  1: []}
    data_test = {0: [],
                 1: []}

    dataset_path = os.path.join(root, '../credit/bank_198_scale.pickle')
    with open(dataset_path, 'rb') as handle:
        data_all = pickle.load(handle)

    random.shuffle(data_all[0])
    random.shuffle(data_all[1])

    data_train[0] = data_all[0][:int(tt.arg.num_sample / 2)]
    data_train[1] = data_all[1][:int(tt.arg.num_sample / 2)]
    data_test[0] = data_all[0][int(tt.arg.num_sample / 2):]
    data_test[1] = data_all[1][int(tt.arg.num_sample / 2):]

    return data_train, data_test
