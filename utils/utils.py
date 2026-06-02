import logging
import time
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0

def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')

def Logger(logger_name, level=logging.DEBUG):

    logging.basicConfig(level=level, format='%(message)s')
    logger = logging.getLogger(logger_name)
    file = logging.FileHandler(logger_name)
    logger.addHandler(file)

    return logger

class Timer:

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()
    
class Loss_Accumulator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.l = []
        self.x = []

    def update(self, l_x):
        l, x = l_x
        l, x = float(l), float(x)
        self.l.append(l)
        self.x.append(x)

    def compute(self):
        loss = [l*x for l, x in zip(self.l, self.x)]
        return sum(loss)/sum(self.x)