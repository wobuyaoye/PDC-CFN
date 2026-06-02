import torch
import torch.nn.functional as F
from torch import nn

from model.baseModels import base_Model



class TFencoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.Dropout1 = nn.Dropout(p=configs.dropout)
        self.model = base_Model(configs)

    def forward(self, x):
        x=self.model(x)
        x=self.Dropout1(x)
        return x

'''Cross-domain interaction encoder'''

class Cross_encoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.layer = configs.T_decoder_layer
        self.T_decoder = nn.ModuleList([nn.TransformerDecoderLayer(configs.length, dim_feedforward=configs.ndim,
                                                                   nhead=configs.nhead, batch_first=True) for _ in
                                        range(configs.T_decoder_layer)])
        self.F_encoder = nn.ModuleList([nn.TransformerEncoderLayer(configs.length, dim_feedforward=configs.ndim,
                                                                   nhead=configs.nhead, batch_first=True) for _ in
                                        range(configs.F_encoder_layer)])


    def forward(self, t, f):
        for i in range(self.layer):
            f = self.F_encoder[i](f)
            t = self.T_decoder[i](t, f)
        return t


class AttentionLayer(nn.Module):
    def __init__(self, configs):
        super(AttentionLayer, self).__init__()
        self._latent_dim = configs.length
        self.mlp = nn.Sequential(
            nn.Linear(self._latent_dim * 2, self._latent_dim * 2),
            nn.BatchNorm1d(self._latent_dim * 2),
            nn.PReLU(),
            nn.Linear(self._latent_dim * 2, self._latent_dim * 2),
            nn.BatchNorm1d(self._latent_dim * 2),
            nn.PReLU()
        )
        self.output_layer = nn.Linear(self._latent_dim * 2, 2, bias=True)

        self.tau=configs.Scaling

    def forward(self, h1, h2):
        h = torch.cat((h1, h2), dim=1)
        act = self.output_layer(self.mlp(h))
        act = F.sigmoid(act) /  self.tau
        e = F.softmax(act, dim=1)
        # weights = torch.mean(e, dim=0)
        # h = weights[0] * h1 + weights[1] * h2
        h = e[:, 0].unsqueeze(1) * h1 + e[:, 1].unsqueeze(1) * h2
        return h