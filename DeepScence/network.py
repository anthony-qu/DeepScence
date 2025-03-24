import torch
from torch import nn
import torch.nn.functional as F
from scipy.sparse import issparse
from scipy.stats import pearsonr
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        size_factors,
        batchnorm=False,
        dropout=None,
        lambda_ortho=None,
    ):
        super(Autoencoder, self).__init__()
        self.lambda_ortho = lambda_ortho
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.encoder = (None,)
        self.bottleneck = (None,)
        self.decoder = (None,)
        self.abs_cosine = None
        if self.lambda_ortho is None:
            self.middle_size = 1
        else:
            self.middle_size = 2

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(self.dropout)
        )
        self.bottleneck = nn.Sequential(nn.Linear(64, self.middle_size), nn.Tanh())
        self.decoder = nn.Sequential(
            nn.Linear(self.middle_size, 64), nn.ReLU(), nn.Linear(64, input_size)
        )
        self.init_weights()

    def forward(self, inputs):
        x, sf = inputs

        encoded = self.encoder(x)
        encoded_scores = self.bottleneck(encoded)
        self.encoded_scores = encoded_scores
        decoded = self.decoder(encoded_scores)
        mu = torch.clamp(torch.exp(decoded), 1e-5, 1e6)
        sf = sf.to(mu.device).unsqueeze(1)
        # mu = mu * sf
        output = mu
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def predict(self, adata, device=None):
        if device is None:
        # If no device is specified, grab it from the model parameters
            device = next(self.parameters()).device
        if issparse(adata.X):
            X_input = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        else:
            X_input = torch.tensor(adata.X, dtype=torch.float32)

        X_input = X_input.to(device)

        # Forward pass
        encoded = self.encoder(X_input)
        bottleneck = self.bottleneck[0](encoded)

        # Convert result back to CPU + NumPy
        encoded_scores = bottleneck.detach().cpu().numpy()

        # if self.lambda_ortho is not None:
        #     cos = np.dot(encoded_scores[:, 0], encoded_scores[:, 1]) / (
        #         np.linalg.norm(encoded_scores[:, 0])
        #         * np.linalg.norm(encoded_scores[:, 1])
        #     )
        #     self.abs_cosine = np.abs(cos)
        return encoded_scores

    def loss(self, y_true, output):
        mse = nn.MSELoss()
        mse_loss = mse(output, y_true)

        # ortho penalty
        if self.lambda_ortho is not None:
            encoded_scores = self.encoded_scores
            s1 = encoded_scores[:, 0]
            s2 = encoded_scores[:, 1]
            ortho_loss = self.lambda_ortho * (
                1 * cosine_loss(s1, s2) + 0 * decorrelation_loss(s1, s2)
            )

            result = mse_loss + ortho_loss
            return result
        else:
            return mse_loss


class ZINBAutoencoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        size_factors,
        batchnorm=False,
        dropout=None,
        lambda_ortho=None,
        lambda_mmd=None,
        batch_matrix=None,
    ):
        super(ZINBAutoencoder, self).__init__()
        self.lambda_ortho = lambda_ortho
        self.lambda_mmd = lambda_mmd
        self.batch_matrix = batch_matrix
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.encoder = (None,)
        self.bottleneck = (None,)
        self.decoder = (None,)
        self.corrs = []
        if self.lambda_ortho is None:
            self.middle_size = 1
        else:
            self.middle_size = 2

        encoder = []
        prev_size = input_size
        for size in hidden_sizes:
            encoder.append(nn.Linear(prev_size, size))
            if self.batchnorm:
                encoder.append(nn.BatchNorm1d(size))
            encoder.append(nn.ReLU())
            # if self.dropout is not None:
            #     encoder.append(nn.Dropout(self.dropout))
            prev_size = size

        bottleneck = nn.Sequential(nn.Linear(prev_size, self.middle_size), nn.Tanh())

        prev_size = self.middle_size

        decoder = []

        for size in hidden_sizes[::-1]:
            decoder.append(nn.Linear(prev_size, size))
            if self.batchnorm:
                decoder.append(nn.BatchNorm1d(size))
            decoder.append(nn.ReLU())

            prev_size = size
        decoder.append(nn.Linear(prev_size, input_size * 3))

        self.encoder = nn.Sequential(*encoder)
        self.bottleneck = bottleneck
        self.decoder = nn.Sequential(*decoder)
        self.init_weights()

    def forward(self, inputs):
        x, sf, batch_matrix = inputs

        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)

        # record pre-act scores
        encoded_scores = self.bottleneck[0](encoded)
        # encoded_scores = self.bottleneck(encoded)
        self.encoded_scores = encoded_scores

        decoded = self.decoder(bottleneck)

        pi, mu, theta = torch.split(decoded, x.shape[1], dim=1)
        pi = torch.sigmoid(pi)  # dropout

        mu = torch.exp(mu)  # mean (n_sample, n_feature) , PLEASE CAP!!!
        mu = torch.clamp(mu, 1e-5, 1e6)
        sf = sf.to(mu.device).unsqueeze(1)
        mu = mu * sf

        theta = torch.exp(theta)  # disp, PLEASE CAP!!!
        theta = torch.clamp(theta, 1e-4, 1e4)

        output = [pi, mu, theta, encoded_scores, batch_matrix]

        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def predict(self, adata):
        if issparse(adata.X):
            X_input = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        else:
            X_input = torch.tensor(adata.X, dtype=torch.float32)

        encoded = self.encoder(X_input)
        bottleneck = self.bottleneck[0](encoded)  # preactivated scores
        # bottleneck = self.bottleneck(encoded)

        # Convert the output back to a NumPy array
        encoded_scores = bottleneck.detach().cpu().numpy()
        return encoded_scores

    def loss(self, y_true, output):
        pi, mean, theta, encoded_scores, batch_matrix = output
        self.eps = 1e-8
        theta = torch.clamp(theta, max=1e6)

        # Negative Binomial part
        t1 = (
            torch.lgamma(theta + self.eps)
            + torch.lgamma(y_true + 1.0)
            - torch.lgamma(y_true + theta + self.eps)
        )
        t2 = (theta + y_true) * torch.log1p(mean / (theta + self.eps)) + (
            y_true * (torch.log(theta + self.eps) - torch.log(mean + self.eps))
        )

        nb_case = t1 + t2 - torch.log(1.0 - pi + self.eps)

        # Zero-inflated part
        zero_nb = torch.pow(theta / (theta + mean + self.eps), theta)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + self.eps)

        result = torch.where(y_true < 1e-8, zero_case, nb_case)

        loss = torch.mean(result)

        # ortho penalty
        if self.lambda_ortho is not None:
            s1 = encoded_scores[:, 0]
            s2 = encoded_scores[:, 1]
            ortho_loss = pearson_cor(s1, s2)

            loss += self.lambda_ortho * ortho_loss

        # mmd loss
        if self.lambda_mmd is not None:
            mmd_penalty = mmd_loss(encoded_scores, batch_matrix)
            loss += self.lambda_mmd * mmd_penalty

        return loss


def mmd_loss(encoded_scores, batch_matrix):
    """
    Compute MMD loss to penalize batch effects in the latent space.

    Args:
        encoded_scores: A tensor of shape (n_samples, n_features) representing the latent space.
        batch_matrix: A one-hot encoded tensor of shape (n_samples, n_batches).

    Returns:
        mmd_loss: The computed MMD loss.
    """
    if isinstance(batch_matrix, np.ndarray):
        batch_matrix = torch.tensor(batch_matrix, dtype=torch.float32)
    n_batches = batch_matrix.shape[1]
    batch_means = []

    for i in range(n_batches):
        batch_mask = batch_matrix[:, i].bool()
        batch_group = encoded_scores[batch_mask]

        if batch_group.size(0) > 0:
            batch_means.append(batch_group.mean(dim=0))

    mmd = 0.0
    for i in range(len(batch_means)):
        for j in range(i + 1, len(batch_means)):
            mean_diff = batch_means[i] - batch_means[j]
            mmd += torch.sum(mean_diff**2)

    return mmd / (len(batch_means) * (len(batch_means) - 1))


def correlation_loss(x, y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean
    covariance = torch.mean(x_centered * y_centered)
    return torch.abs(covariance)


def pearson_cor(x, y):
    stacked = torch.stack((x, y))
    corr_matrix = torch.corrcoef(stacked)
    corr = corr_matrix[0, 1]
    return corr**2


def cosine_loss(x, y):
    cosine_similarity = torch.nn.functional.cosine_similarity(x, y, dim=0)
    return cosine_similarity**2
