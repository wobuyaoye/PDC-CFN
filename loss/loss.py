import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class APDCL(nn.Module):
    """自适应原型驱动对比"""
    def __init__(self, class_num, temperature, device):
        super(APDCL, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j, weight=None,alpha=1.0):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        total_loss=loss + alpha * ne_loss
        total_loss = weight * total_loss if weight is not None else total_loss
        return total_loss


class NTXentLoss_poly(torch.nn.Module):
    """归一化温度交叉熵"""
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss_poly, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        """Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0. """
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        CE = self.criterion(logits, labels)

        onehot_label = torch.cat((torch.ones(2 * self.batch_size, 1),torch.zeros(2 * self.batch_size, negatives.shape[-1])),dim=-1).to(self.device).long()
        # Add poly loss
        pt = torch.mean(onehot_label* torch.nn.functional.softmax(logits,dim=-1))

        epsilon = self.batch_size
        # loss = CE/ (2 * self.batch_size) + epsilon*(1-pt) # replace 1 by 1/self.batch_size
        loss = CE / (2 * self.batch_size) + epsilon * (1/self.batch_size - pt)
        # loss = CE / (2 * self.batch_size)

        return loss


class DPCCL(torch.nn.Module):
    """双视角互补对比"""
    def __init__(self, device, batch_size, margin=0.1):
        super(DPCCL, self).__init__()
        self.batch_size = batch_size
        # self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        # self.margin = margin

        self.log_margin = torch.nn.Parameter(torch.tensor(np.log(2.0)).float())

        self.pair_distance = torch.nn.PairwiseDistance(p=2)

    def _get_correlated_mask(self):
        diag = np.eye(N=self.batch_size, M=3 * self.batch_size)
        l_1 = np.eye(N=self.batch_size, M=3 * self.batch_size, k=self.batch_size)
        l_2 = np.eye(N=self.batch_size, M=3 * self.batch_size, k=self.batch_size * 2)
        mask = torch.from_numpy((diag + l_1 + l_2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    def forward(self, feature_total, feature_1, feature_2):
        representations = torch.cat([feature_total, feature_1, feature_2], dim=0)
        expanded_tensor1 = feature_total.unsqueeze(1)
        expanded_tensor2 = representations.unsqueeze(0)

        distance_matrix = torch.cdist(expanded_tensor1, expanded_tensor2, p=2.0).view(feature_total.shape[0], -1)
        distance_matrix = F.relu(distance_matrix)

        distance_matrix_2 = self.pair_distance(feature_1, feature_2)

        l_pos_1 = torch.diag(distance_matrix, self.batch_size)
        l_pos_2 = torch.diag(distance_matrix, self.batch_size*2)
        positives = (l_pos_1 + l_pos_2 + distance_matrix_2).view(self.batch_size, 1)

        positive_value, positive_index = torch.max(torch.stack([l_pos_1, l_pos_2]), dim=0)
        positive_value = positive_value.view(-1, 1)

        negatives = distance_matrix[self.mask_samples_from_same_repr].view(self.batch_size, -1)
        negative_value, negative_index = torch.min(negatives, dim=-1)
        negatives = negative_value.view(-1, 1)

        margin = torch.exp(self.log_margin.clamp(min=np.log(0.1), max=np.log(10.0)))
        triplet_loss = positives - negatives + margin
        triplet_loss = F.relu(triplet_loss)

        # triplet_loss += positive_value
        triplet_loss = triplet_loss + positive_value

        triplet_loss = torch.sum(triplet_loss)

        return triplet_loss / (self.batch_size)


def compute_view_value(rs, H, view):
    N = H.shape[0]
    w = []
    # all features are normalized
    global_sim = torch.matmul(H,H.t())
    for v in range(view):
        view_sim = torch.matmul(rs[v],rs[v].t())
        related_sim = torch.matmul(rs[v],H.t())
        # The implementation of MMD
        w_v = (torch.sum(view_sim) + torch.sum(global_sim) - 2 * torch.sum(related_sim)) / (N*N)
        w.append(torch.exp(-w_v))
    w = torch.stack(w)
    w = w / torch.sum(w)
    return w.squeeze()

