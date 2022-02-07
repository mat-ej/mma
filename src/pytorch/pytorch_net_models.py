import torch
import torch.nn as nn


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.l1 = torch.nn.Linear(input_dim, 1)
        nn.init.xavier_normal_(self.l1.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = self.l1(x)
        x = torch.sigmoid(x)
        return x


class OneHiddenLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_nodes, dropout_rate):
        super(OneHiddenLayer, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_nodes)
        nn.init.xavier_normal_(self.l1.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
        self.l2 = nn.Linear(hidden_nodes, 1)
        nn.init.xavier_normal_(self.l2.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.tanh = nn.Tanh()
        self.leaky = nn.LeakyReLU(0.2)
        self.silu = nn.SiLU()

    def forward(self, x):
        #x = self.silu(x)
        x = self.l1(x)
        x = self.silu(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.dropout(x)
        #x = self.relu(x)
        x = self.sigmoid(x)
        return x
