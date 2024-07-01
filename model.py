import torch.nn as nn
import torch
import torch.nn.functional as F

class TRAFFICModel(nn.Module):
    def __init__(self, n_skill, n_cat, nout, max_seq=100, embed_dim=128, pos_encode='LSTM', nlayers=2, rnnlayers=3,
                 dropout=0.1, nheads=8):
        super(TRAFFICModel, self).__init__()
        self.embedding = nn.Linear(n_skill, embed_dim)
        self.pos_encoder = nn.ModuleList([ResidualLSTM(embed_dim) for _ in range(rnnlayers)])
        self.layer_normal = nn.LayerNorm(embed_dim)
        encoder_layers = [nn.TransformerEncoderLayer(embed_dim, nheads, embed_dim*4, dropout) for _ in range(nlayers)]
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        self.pred = nn.Linear(embed_dim, nout)

    def forward(self, numerical_features):
        x = self.embedding(numerical_features)
        x = x.permute(1, 0, 2)
        for lstm in self.pos_encoder:
            lstm.LSTM.flatten_parameters()
            x = lstm(x)
        x = x.permute(1, 0, 2)
        x = self.pred(x)
        return x.squeeze(-1)

class ResidualLSTM(nn.Module):
    def __init__(self, d_model):
        super(ResidualLSTM, self).__init__()
        self.LSTM = nn.LSTM(d_model, d_model, num_layers=1, bidirectional=True)
        self.linear1 = nn.Linear(d_model*2, d_model*4)
        self.linear2 = nn.Linear(d_model*4, d_model)

    def forward(self, x):
        res = x
        x, _ = self.LSTM(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return res + x
