import numpy as np
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from dataset.dataset import *
from model.net import *

class NeuralNetworkRunner():
    def __init__(self):
        self.config = 0

    def train(self, lr, batch_size, epoch):
        self.lr = lr
        self.epoch = epoch
        self.shuffle = True
        self.batch_size = batch_size

        TIK = time.time()
        torch.cuda.empty_cache()

        train_dataset = DataSet(split='train')
        val_dataset = DataSet(split='val')

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=train_dataset.collate_fn,
            drop_last=False)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
            drop_last=False)

        # create models
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cuda:0')
        model = cnn()
        model.to(device)

        # create optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(params, lr=self.lr)

        # reset gradient
        optimizer.zero_grad()
        model.train()
        train_loss = []
        val_loss = []
        # for epo in tqdm(range(self.epoch), desc="EPOCH"):
        for epo in range(self.epoch):
            for idx, data in enumerate(train_loader):
                optimizer.zero_grad()
                x, t_loss = model(data['input'].to(device), data['target'].to(device))
                t_loss.backward()
                optimizer.step()
                train_loss.append(t_loss.item())

            model.eval()
            for idx, data in enumerate(val_loader):
                x, v_loss = model(data['input'].to(device), data['target'].to(device))
                val_loss.append(v_loss.item())

        mean_train_loss = np.mean(train_loss)
        mean_val_loss = np.mean(val_loss)

        print("lr : {} / batch_size : {} / epoch : {} / loss : {}--{}".format(self.lr, self.batch_size, self.epoch, mean_train_loss, mean_val_loss))
        return mean_val_loss


if __name__ == "__main__":
    model = NeuralNetworkRunner().train()



