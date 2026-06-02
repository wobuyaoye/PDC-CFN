import torch
import torch.nn.functional as F
from torch import nn

from loss.loss import NTXentLoss_poly, compute_view_value, DPCCL, APDCL
from model.baseModels import Projection, ClusterProject
from model.encoder import TFencoder, Cross_encoder, AttentionLayer


class train_model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.device=configs.device
        self.batch_size=configs.batch_size

        self.encoder_t = TFencoder(configs)
        self.encoder_f = TFencoder(configs)

        self.cross_encoder = Cross_encoder(configs)

        self.fusion = AttentionLayer(configs)

        self.T_projection = Projection(configs.length, configs.hidden_dim)
        self.F_projection = Projection(configs.length, configs.hidden_dim)

        self.T_cluster = ClusterProject(configs.length, configs.num_class)
        self.cluster = ClusterProject(configs.length, configs.num_class)


        self.align_temperature=configs.Align_T
        self.criterion = NTXentLoss_poly(self.device, self.batch_size, temperature=self.align_temperature,
                                            use_cosine_similarity=True).to(self.device)

        self.margin_ = configs.DPC_T
        self.drc_loss = DPCCL(self.device, self.batch_size,self.margin_).to(self.device)

        self.apd_temperature = configs.APD_T
        self.criterion_cluster = APDCL(configs.num_class, temperature=self.apd_temperature, device=self.device).to(self.device)

        self.lamda = configs.lamda
        self.lamda2 = configs.lamda2

    def forward(self, x_t, x_f):
        #Encoder
        h_t = self.encoder_t(x_t)
        h_f = self.encoder_f(x_f)


        #Time-frequency alignment
        q_t = F.normalize(self.T_projection(h_t))
        q_f = F.normalize(self.F_projection(h_f))
        loss1=self.criterion(q_t,q_f)
        # Feature fusion
        H = self.fusion(h_t, h_f)

        # Cross-domain interaction transformer
        z_t = self.cross_encoder(h_t, h_f) #torch.Size([192, 9, 128])
        z_f = self.cross_encoder(h_f, h_t) #torch.Size([192, 9, 128])

        #########################################
        # Collaborative contrastive loss
        loss2 = self.drc_loss(H, z_t, z_f)    # DPCCL

        y1 = self.T_cluster(z_t)
        y2 = self.T_cluster(z_f)
        Y = self.cluster(H)
        ys=[y1,y2]
        with torch.no_grad():
            w = compute_view_value(ys, Y, 2)

        loss_list = []
        for v in range(2):
            loss_list.append(self.criterion_cluster(Y, ys[v], w[v]))
        loss3 = sum(loss_list)     # APDCL
        #########################################
        total_loss=self.lamda*loss1+self.lamda2*(loss2+loss3)

        return total_loss



class finetune_model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.encoder_t = TFencoder(configs)
        self.encoder_f = TFencoder(configs)

        self.fusion = AttentionLayer(configs)

        self.fc = ClusterProject(configs.length, configs.num_class)


    def forward(self, x_t, x_f,return_features=False):
        h_t = self.encoder_t(x_t)
        h_f = self.encoder_f(x_f)
        H = self.fusion(h_t, h_f)
        y = self.fc(H)

        if return_features:
            return y, H
        return y

