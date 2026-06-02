import torch
from torch import nn

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8,
                      stride=4, bias=False, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class Projection(nn.Module):
    def __init__(self, latent_dim,hidden_dim):
        super(Projection, self).__init__()
        self._latent_dim = latent_dim
        self._hidden_dim = hidden_dim
        self.instance_projector = nn.Sequential(
            nn.Linear(self._latent_dim, self._hidden_dim),
            nn.BatchNorm1d(self._hidden_dim),
            nn.PReLU(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
            nn.BatchNorm1d(self._hidden_dim)
        )

    def forward(self, x):
        return self.instance_projector(x)


class ClusterProject(nn.Module):
    def __init__(self, latent_dim, n_clusters):
        super(ClusterProject, self).__init__()
        self._latent_dim = latent_dim
        self._n_clusters = n_clusters
        self.cluster_projector = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.BatchNorm1d(self._latent_dim),
            nn.ReLU(),
        )
        self.cluster = nn.Sequential(
            nn.Linear(self._latent_dim, self._n_clusters),
            # nn.BatchNorm1d(self._n_clusters),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        z = self.cluster_projector(x)
        y = self.cluster(z)
        return y
