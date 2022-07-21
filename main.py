from RunModule.train import trainer
from RunModule.test import tester
from Setting import param


class handler(param):
    def __init__(self):
        super(handler, self).__init__()

        if self.RunType == 0:
            trainer().run()

        elif self.RunType == 1:
            tester().run()