import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from Model.GCN import GCN
from Model.GCN_GRU import GCN_GRU
from Setting import param
from util.AVG_Meter import AverageMeter
from util.DataLoader import Loader
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import os


class trainer(param):
    def __init__(self):
        super(trainer, self).__init__()

    def weighted_mse_loss(self, pred_, label, weight):
        num = pred_.shape[0] * pred_.shape[1]
        return torch.sum(weight * (pred_ - label) ** 2) / num

    def run(self):
        model = GCN_GRU(K=self.K, aggregator="CNN")
        model.cuda()
        criterion = self.weighted_mse_loss
        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.LR, weight_decay=0.0)

        eval_loss_increas_step = 0

        train_losses = []
        eval_losses = []
        train_inputs, train_adj, train_labels = Loader()(type=0)
        # public_inputs, private_inputs, public_adj, private_adj = Loader()(type=1)

        train_dataset = TensorDataset(train_inputs, train_adj, train_labels)

        print('STARTING TRAIN!!!!!')
        for epoch in range(self.EPOCH):
            train_loader = DataLoader(train_dataset, batch_size=self.BATCHSZ, shuffle=True)
            # test_loader = DataLoader(test_dataset, batch_size=self.BATCHSZ, shuffle=True)

            print('--------------------------------------------------------------------------------------')
            print(f'|| [now epoch : {epoch}/{self.EPOCH}] ||')
            print('--------------------------------------------------------------------------------------')
            model.train()
            model.zero_grad()
            train_loss = AverageMeter()

            for index, (input_, adj, label) in enumerate(train_loader):
                input_ = input_.cuda()
                adj = adj.cuda()
                label = label.cuda()
                preds = model(input_, adj)

                loss = criterion(preds, label, self.LOSS_WEIGHT)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss.update(loss.item())

                if index % 100 == 0:
                    print(f'|| [now step : {index}/{len(train_loader)}] --> LOSS : {train_loss.avg} ||')

            avg_loss = train_loss.avg
            print(f'|| [now epoch : {epoch}/{self.EPOCH}] || --> {avg_loss}')
            print('--------------------------------------------------------------------------------------')
            train_losses.append(avg_loss)
            os.makedirs(f"{self.CKP_PATH}", exist_ok=True)
            torch.save(model.state_dict(), f'{self.CKP_PATH}/{epoch}.pt')
