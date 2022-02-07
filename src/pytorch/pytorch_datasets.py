import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from src.preprocessing.features_extraction.feature_variables import *


class BasicDataset(Dataset):
    def __init__(self, df):
        self.y = df['WINNER'].astype(np.float32).values.reshape(-1, 1)
        self.x = df[red_fighter + red_extra_features + red_stats + blue_fighter + blue_extra_features + blue_stats + ['R_ODDS', 'B_ODDS']].drop(['R_NAME', 'B_NAME', 'R_WEIGHT', 'B_WEIGHT', 'R_STYLE', 'B_STYLE'], axis=1).astype(np.float32).values
        self.odds = df[['R_ODDS', 'B_ODDS']].astype(np.float32).values
        scaler = StandardScaler()
        self.x = scaler.fit_transform(self.x)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx], self.odds[idx]]

    def n_features(self):
        return self.x.shape[1]
