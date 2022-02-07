import torch
import torch.nn as nn
import torch.nn.functional as f


class BCEDecorrelationLoss(nn.Module):
    def __init__(self, c):
        super(BCEDecorrelationLoss, self).__init__()
        self.cross_entropy = nn.BCELoss(reduction='none')
        self.decorrelation_BCE = nn.BCELoss(reduction='none')
        self.c = c

    def forward(self, output, y, odds):
        result_distances = self.cross_entropy(output, y)
        booker_probabilities = f.normalize(1 / odds, p=1, dim=1)[:, 0]
        booker_probabilities = torch.reshape(booker_probabilities, output.shape)
        booker_distances = self.decorrelation_BCE(output, booker_probabilities)
        return torch.mean(result_distances - self.c * booker_distances)


class MSEDecorrelationLoss(nn.Module):
    def __init__(self, c):
        super(MSEDecorrelationLoss, self).__init__()
        self.cross_entropy = nn.BCELoss(reduction='none')
        self.c = c

    def forward(self, output, y, odds):
        n = y.shape[0]
        result_distances = self.cross_entropy(output, y)
        probabilities = torch.zeros(size=(n, 2))
        probabilities[:, 0] = output[:, 0] + 0.001
        probabilities[:, 1] = 1 - output[:, 0] + 0.001
        probabilities = f.normalize(probabilities, p=1, dim=1)[:, 0].reshape(n, 1)
        booker_probabilities = f.normalize(1 / odds, p=1, dim=1)[:, 0].reshape(n, 1)
        pwr = torch.pow(probabilities - booker_probabilities, 2)

        return torch.mean(result_distances - self.c * pwr)


class KLDecorrelationLoss(nn.Module):
    def __init__(self, c):
        super(KLDecorrelationLoss, self).__init__()
        self.cross_entropy = nn.BCELoss(reduction='none')
        self.c = c

    def forward(self, output, y, odds):
        n = y.shape[0]
        result_distances = self.cross_entropy(output, y)

        probabilities = torch.zeros(size=(n, 2))
        probabilities[:, 0] = output[:, 0] + 0.001
        probabilities[:, 1] = 1 - output[:, 0] + 0.001
        probabilities = f.normalize(probabilities, p=1, dim=1)
        booker_probabilities = f.normalize(1 / odds, p=1, dim=1)
        kl_distances = torch.sum(booker_probabilities * torch.log(booker_probabilities / probabilities), dim=1).reshape(n, 1)
        if torch.isnan(kl_distances).any():
            print(kl_distances)
        losses = result_distances - self.c * kl_distances

        return torch.mean(losses)


class JSDecorrelationLoss(nn.Module):
    def __init__(self, c, device=None):
        super(JSDecorrelationLoss, self).__init__()
        self.cross_entropy = nn.BCELoss(reduction='none')
        self.c = c
        self.device = device

    def forward(self, output, y, odds):
        n = y.shape[0]
        result_distances = self.cross_entropy(output, y)

        if self.device:
            probabilities = torch.zeros(size=(n, 2), device=self.device)
        else:
            probabilities = torch.zeros(size=(n, 2))

        probabilities[:, 0] = output[:, 0] + 0.001
        probabilities[:, 1] = 1 - output[:, 0] + 0.001
        probabilities = f.normalize(probabilities, p=1, dim=1)
        booker_probabilities = f.normalize(1 / odds, p=1, dim=1)
        pointwise_mean = 1/2 * (probabilities + booker_probabilities)

        kl_prediction_distances = torch.sum(probabilities * torch.log(probabilities / pointwise_mean), dim=1).reshape(n, 1)
        kl_booker_distances = torch.sum(booker_probabilities * torch.log(booker_probabilities / pointwise_mean), dim=1).reshape(n, 1)

        distances_sum = (kl_prediction_distances + kl_booker_distances) / 2
        losses = result_distances - self.c * distances_sum

        return torch.mean(losses)
