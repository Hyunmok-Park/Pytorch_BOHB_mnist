import os
import glob
import time

import torch
import pickle
import numpy as np
from collections import defaultdict

from struct import *
from torch.utils.data import Dataset

class DataSet(Dataset):
    def __init__(self, split='train'):
        if split == "train":
            fp_image = open('train-images-idx3-ubyte','rb')
            fp_label = open('train-labels-idx1-ubyte','rb')
        else:
            fp_image = open('t10k-images-idx3-ubyte','rb')
            fp_label = open('t10k-labels-idx1-ubyte','rb')

        s = fp_image.read(16)
        l = fp_label.read(8)
        k = 0

        lbl = [ [],[],[],[],[],[],[],[],[],[] ]
        # print("LOADING DATA")
        while True:
            s = fp_image.read(784) #784바이트씩 읽음
            l = fp_label.read(1) #1바이트씩 읽음
            if not s:
                break;
            if not l:
                break;
            index = int(l[0])
            #unpack
            img = np.reshape(unpack(len(s)*'B',s), (28,28))
            lbl[index].append(img) #각 숫자영역별로 해당이미지를 추가

        self.data_list = []

        if split == "train":
            for label, datas in enumerate(lbl):
                for _ in range(10):
                    data = {}
                    data['input'] = datas[_]
                    data['target'] = label
                    self.data_list.append(data)
                    k += 1
        else:
            for label, datas in enumerate(lbl):
                for _ in range(1):
                    data = {}
                    data['input'] = datas[_]
                    data['target'] = label
                    self.data_list.append(data)
                    k += 1

        self.num_data = k
        # print("DATALOAD COMPLETE")
        # print(len(self.data_list))

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return self.num_data

    def collate_fn(self, batch):
        data = {}
        data['input'] = torch.from_numpy(np.concatenate([[bch['input']] for bch in batch], axis=0).reshape(-1,1,28,28)).float()
        data['target'] = torch.from_numpy(np.concatenate([[bch['target']] for bch in batch], axis=0))
        return data

if __name__ == '__main__':
    train_dataset = DataSet()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=5,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        drop_last=False)

    for i in train_loader:
        print(i['input'].shape)
        print(i['target'].shape)


