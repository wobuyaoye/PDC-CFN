import torch


class Config(object):
    def __init__(self):
        '''feature length & seq length'''
        self.length = 128

        self.num_embed = 2

        '''feedforward dim'''
        self.hidden_dim = 512
        self.ndim = 512
        self.logit_dim = 512

        '''head number'''
        self.nhead = 8

        '''temperature'''
        self.T = 0.2

        '''layer number'''
        self.TF_nlayer = 6
        self.T_decoder_layer = 6
        self.F_encoder_layer = 6
        self.num_projection = 2
        self.num_prediction = 1

        '''dropout'''
        self.dropout = 0.1

        '''num_classes'''

        self.num_class = 13

        # self.device = torch.device("cuda:1")
        self.device=torch.device("cuda")
