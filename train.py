import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from data_loader import generate_data
from model import TRAFFICModel
import numpy as np
import pandas as pd

class TRAFICDataset(Dataset):
    def __init__(self, features, targets, masks):
        super(TRAFICDataset, self).__init__()
        self.features = features
        self.targets = targets
        self.masks = masks

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index].astype('float32'), self.targets[index].astype('float32'), self.masks[index].astype('bool')

def train():
    data_path = "data/pems-bay.h5"
    seq_len = 15
    pre_len = 1
    pre_sens_num = 1

    train_data, train_w, train_l, test_data, test_w, test_l, med, min_val = generate_data(data_path, seq_len, pre_len, pre_sens_num)

    train_masks = np.ones((train_data.shape[0], train_data.shape[1]))
    train_dataset = TRAFICDataset(train_data, train_l, train_masks)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = TRAFFICModel(n_skill=train_data.shape[-1], n_cat=10, nout=1, embed_dim=128, pos_encode='LSTM', max_seq=64, nlayers=3, rnnlayers=3, dropout=0.1, nheads=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(100):
        model.train()
        train_loss = 0
        for features, targets, mask in train_dataloader:
            features, targets, mask = features.to(device), targets.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch [{epoch+1}/100], Loss: {train_loss/len(train_dataloader):.4f}")

if __name__ == "__main__":
    train()
