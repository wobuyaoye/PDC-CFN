import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torch.fft as fft


class Load_Dataset(Dataset):
    def __init__(self, dataset, mode, percent=1,args=None):
        super().__init__()
        self.mode = mode
        if mode == 'finetune':
            num = int(dataset[0].size(0) * percent)
            x_data = dataset[0][:num]
            y_data = dataset[1][:num]


        else:
            x_data = dataset[0]
            y_data = dataset[1]

        if isinstance(x_data, np.ndarray):
            self.t_data = torch.from_numpy(x_data).float()
            self.y_data = torch.from_numpy(y_data).long()
        else:
            self.t_data = x_data.float()
            self.y_data = y_data.long()


        '''Frequency domain'''
        self.f_data = fft.fft(self.t_data).abs()  # /(window_length)
        """Augmentation"""

    def __getitem__(self, index):
        return self.t_data[index], self.f_data[index], self.y_data[index]

    def __len__(self):
        return self.t_data.shape[0]


def Data_Loader(path, args, mode):
    if mode == 'train':
        data = torch.load(os.path.join(path, 'train.pt'))
        dataset = Load_Dataset(data, mode,args=args)
        loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
    elif mode == 'finetune':
        data = torch.load(os.path.join(path, 'val.pt')) #finetune.pt
        # data = sample_one_signal_per_class(data,args.classes)  # one-shot
        dataset = Load_Dataset(data, mode, args.percent)
        loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
    else:
        data = torch.load(os.path.join(path, 'test.pt'))
        dataset = Load_Dataset(data, mode)
        loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False)

    return loader




def sample_one_signal_per_class(data, num_classes=13):
    signals = data[0]
    labels = data[1]
    new_signals = []
    new_labels = []

    for c in range(num_classes):

        indices = (labels == c).nonzero(as_tuple=True)[0]
        rand_index = indices[torch.randint(len(indices), (1,))].item()
        new_signals.append(signals[rand_index])
        new_labels.append(labels[rand_index])
    new_signals = torch.stack(new_signals)
    new_labels = torch.tensor(new_labels)
    perm = torch.randperm(num_classes)
    new_signals = new_signals[perm]
    new_labels = new_labels[perm]
    return new_signals, new_labels