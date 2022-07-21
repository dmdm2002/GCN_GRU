import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
from Setting import param
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
import torch


class Loader(param):
    def __init__(self):
        super(Loader, self).__init__()

        self.pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
        self.token2int = {x: i for i, x in enumerate('().ACGUBEHIMSX')}
        self.train = pd.read_json(self.train_path, lines=True)
        self.test = pd.read_json(self.test_path, lines=True)
        self.submission = pd.read_csv(self.submission_path)

    def get_couples(self, structure):
        """
        For each closing parenthesis, I find the matching opening one and store their index in the couples list.
        The assigned list is used to keep track of the assigned opening parenthesis
        """
        opened = [idx for idx, i in enumerate(structure) if i == '(']
        closed = [idx for idx, i in enumerate(structure) if i == ')']

        assert len(opened) == len(closed)
        assigned = []
        couples = []

        for close_idx in closed:
            for open_idx in opened:
                if open_idx < close_idx:
                    if open_idx not in assigned:
                        candidate = open_idx
                else:
                    break
            assigned.append(candidate)
            couples.append([candidate, close_idx])

        assert len(couples) == len(opened)

        return couples

    def build_matrix(self, couples, size):
        mat = np.zeros((size, size))

        for i in range(size):  # neigbouring bases are linked as well
            if i < size - 1:
                mat[i, i + 1] = 1
            if i > 0:
                mat[i, i - 1] = 1

        for i, j in couples:
            mat[i, j] = 1
            mat[j, i] = 1

        return mat

    def convert_to_adj(self, structure):
        couples = self.get_couples(structure)
        mat = self.build_matrix(couples, len(structure))
        return mat

    def preprocess_inputs(self, df, cols=['sequence', 'structure', 'predicted_loop_type']):
        inputs = np.transpose(
            np.array(df[cols].applymap(lambda seq: [self.token2int[x] for x in seq]).values.tolist()), (0, 2, 1))

        adj_matrix = np.array(df['structure'].apply(self.convert_to_adj).values.tolist())

        return inputs, adj_matrix

    def __call__(self, type=0):
        if type == 0:
            train_inputs, train_adj = self.preprocess_inputs(self.train)

            train_labels = np.array(self.train[self.pred_cols].values.tolist()).transpose((0, 2, 1))

            train_inputs = torch.tensor(train_inputs, dtype=torch.long)
            train_adj = torch.tensor(train_adj, dtype=torch.long)
            train_labels = torch.tensor(train_labels, dtype=torch.float32)

            return train_inputs, train_adj, train_labels

        else:
            public_df = self.test.query("seq_length == 107").copy()
            private_df = self.test.query("seq_length == 130").copy()

            public_inputs, public_adj = self.preprocess_inputs(public_df)
            private_inputs, private_adj = self.preprocess_inputs(private_df)

            public_inputs = torch.tensor(public_inputs, dtype=torch.long)
            private_inputs = torch.tensor(private_inputs, dtype=torch.long)
            public_adj = torch.tensor(public_adj, dtype=torch.long)
            private_adj = torch.tensor(private_adj, dtype=torch.long)

            return public_df, private_df, public_inputs, private_inputs, public_adj, private_adj