import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from Model.GCN import GCN


class GCN_GRU(nn.Module):
    def __init__(self, num_embedding=14, seq_len=107, pred_len=68, dropout=0.5,
                 embed_dim=100, hidden_dim=128, K=1, aggregator='CNN'):
        
        super(GCN_GRU, self).__init__()

        print(f'|| aggregate function is {aggregator} ||')

        self.pred_len = pred_len
        self.embedding_layer = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embed_dim)

        self.GCN_Module = nn.ModuleList([GCN(3 * embed_dim, 3 * embed_dim, aggregator=aggregator) for i in range(K)])
        self.GRU_Layer = nn.GRU(input_size=3 * embed_dim, hidden_size=hidden_dim, num_layers=3, batch_first=True,
                                dropout=dropout, bidirectional=True)

        self.linear_layer = nn.Linear(in_features=2 * hidden_dim, out_features=5)

    def forward(self, input_, adj_matrix):
        embedding = self.embedding_layer(input_)
        embedding = torch.reshape(embedding, (-1, embedding.shape[1], embedding.shape[2] * embedding.shape[3]))

        # gcn
        gcn_feature = embedding
        for gcn_layer in self.GCN_Module:
            gcn_feature = gcn_layer(gcn_feature, adj_matrix)

        # gru
        gru_output, gru_hidden = self.GRU_Layer(gcn_feature)
        truncated = gru_output[:, :self.pred_len]

        output = self.linear_layer(truncated)

        return output