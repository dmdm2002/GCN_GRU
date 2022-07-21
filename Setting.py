import torch


class param(object):
    def __init__(self):
        self.root = 'D:/Datasets/Covid_vaccine_data'
        self.train_path = f'{self.root}/train.json'
        self.test_path = f'{self.root}/test.json'
        self.submission_path = f'{self.root}/sample_submission.csv'

        self.CKP_PATH = f'{self.root}/ckp'

        """ Training Option """
        self.EPOCH = 200
        self.BATCHSZ = 10
        self.LR = 1e-3
        self.K = 1
        self.N_SPLIT = 2
        self.SEED = 1234
        self.LOSS_WEIGHT = torch.tensor([0.3,0.3,0.05,0.3,0.05]).cuda()
        self.PATIENCE = 15

        """ 0 : train / 1 : test"""
        self.RunType = 0