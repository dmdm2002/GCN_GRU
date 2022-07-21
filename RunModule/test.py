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
import pandas as pd
import numpy as np


class tester(param):
    def __init__(self):
        super(tester, self).__init__()

    def weighted_mse_loss(self, pred_, label, weight):
        num = pred_.shape[0] * pred_.shape[1]
        return torch.sum(weight * (pred_ - label) ** 2) / num

    def run(self):
        print('STARTING TESTING!!!!!')
        pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
        submission = pd.read_csv(self.submission_path)
        public_df, private_df, public_inputs, private_inputs, public_adj, private_adj = Loader()(type=1)
        model_short = GCN_GRU(seq_len=107, pred_len=107, K=self.K, aggregator='CNN')
        model_long = GCN_GRU(seq_len=130, pred_len=130, K=self.K, aggregator='CNN')

        list_public_preds = []
        list_private_preds = []

        for epoch in range(0, 200):
            ckp_path = f'{self.CKP_PATH}/{epoch}.pt'
            model_short.load_state_dict(torch.load(ckp_path))
            model_long.load_state_dict(torch.load(ckp_path))
            model_short.cuda()
            model_long.cuda()

            model_short.eval()
            model_long.eval()

            public_preds = model_short(public_inputs.cuda(), public_adj.cuda())
            private_preds = model_long(private_inputs.cuda(), private_adj.cuda())
            public_preds = public_preds.cpu().detach().numpy()
            private_preds = private_preds.cpu().detach().numpy()

            list_public_preds.append(public_preds)
            list_private_preds.append(private_preds)

            public_preds = np.mean(list_public_preds, axis=0)
            private_preds = np.mean(list_private_preds, axis=0)

            preds_ls = []
            submission = pd.read_csv(self.submission_path)

            for df, preds in [(public_df, public_preds), (private_df, private_preds)]:
                for i, uid in enumerate(df.id):
                    single_pred = preds[i]

                    single_df = pd.DataFrame(single_pred, columns=pred_cols)
                    single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

                    preds_ls.append(single_df)

            preds_df = pd.concat(preds_ls)

            submission = submission[['id_seqpos']].merge(preds_df, on=['id_seqpos'])

            submission.to_csv(f'./submission_{epoch}.csv', index=False)