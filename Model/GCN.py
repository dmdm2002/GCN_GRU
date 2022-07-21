import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, aggregator='CNN'):
        super(GCN, self).__init__()
        self.aggregator = aggregator

        linear_input_dim = input_dim
        # self.GRU_hidden = 64
        # linear_input_dim = input_dim + self.GRU_hidden
        # self.GRU_agg = nn.GRU(input_dim, self.GRU_hidden, num_layers=1, batch_first=1)

        self.linear_gcn = nn.Linear(in_features=linear_input_dim, out_features=output_dim)

    def forward(self, input_, adj_matrix):
        idx = torch.arange(0, adj_matrix.shape[-1], out=torch.LongTensor())
        adj_matrix[:, idx, idx] = 1

        adj_matrix = adj_matrix.type(torch.float32)
        sum_adj = torch.sum(adj_matrix, axis=2)
        sum_adj[sum_adj == 0] = 1

        feature_agg = torch.bmm(adj_matrix, input_)
        feature_agg = feature_agg / sum_adj.unsqueeze(dim=2)

        # feature_agg = torch.zeros(input_.shape[0], input_.shape[1], self.GRU_agg).cuda()
        #
        # for i in range(adj_matrix.shape[1]):
        #     neighbors = adj_matrix[:, i, :].unsqueeze(2) * input_
        #     _, hn = self.lstm_agg(neighbors)
        #     feature_agg[:, i, :] = torch.squeeze(hn[0],0)

        feature_cat = feature_agg
        feature = torch.sigmoid((self.linear_gcn(feature_cat)))
        feature = feature / torch.norm(feature, p=2, dim=2).unsqueeze(dim=2)

        return feature