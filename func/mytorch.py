import torch.nn as nn
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import numpy as np

def get_xyodds(df, odds_cols, target):
    x = df.drop(columns=target).astype(np.float32).values
    y = df[target].astype(np.float32).values.reshape(-1, 1)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    odds = df[odds_cols].astype(np.float32).values

    return x, y, odds


class BettingDataset(Dataset):
    def __init__(self, x, y, odds):
        self.x = x
        self.y = y
        self.odds = odds
    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx], self.odds[idx]]

    def n_features(self):
        return self.x.shape[1]

class NN(torch.nn.Module):
    def __init__(self, input_dim, hidden_nodes, hidden_layers, dropout_rate):
        super(NN, self).__init__()

        self.n_hidden_layers = hidden_layers

        if self.n_hidden_layers == 0:
            self.l1 = nn.Linear(input_dim, 1)
            nn.init.xavier_normal_(self.l1.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
        else:
            self.l1 = nn.Linear(input_dim, hidden_nodes)
            nn.init.xavier_normal_(self.l1.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
            self.hidden_layers = []
            for i in range(0, hidden_layers - 1):
                self.hidden_layers.append(nn.Linear(hidden_nodes, hidden_nodes))
                nn.init.xavier_normal_(self.hidden_layers[-1].weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))

            self.hidden_layers.append(nn.Linear(hidden_nodes, 1))
            nn.init.xavier_normal_(self.hidden_layers[-1].weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.tanh = nn.Tanh()
        self.leaky = nn.LeakyReLU(0.2)
        self.silu = nn.SiLU()

    def forward(self, x):
        if self.n_hidden_layers == 0:
            # x = self.silu(x)
            x = self.l1(x)
            # x = self.dropout(x)
            x = self.sigmoid(x)
        else:
            #x = self.silu(x)
            x = self.l1(x)
            x = self.silu(x)
            x = self.dropout(x)
            for l in self.hidden_layers:
                x = l(x)
                x = self.dropout(x)
                #x = self.relu(x)

            x = self.sigmoid(x)

        return x