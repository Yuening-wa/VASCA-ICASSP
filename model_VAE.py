import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Linear_positive(nn.Linear):
    def __init__(self, input_dim, output_dim):
        super(Linear_positive, self).__init__(input_dim, output_dim, bias=False)

    def forward(self, input):
        return F.linear(input, self.weight.exp())
    

class VAE(nn.Module):
    def __init__(self, input_size=100, h1_dim=64, h2_dim=32, h3_dim=32, h4_dim=16, \
                  h5_dim = 5, h6_dim = 5, z_dim=5, A_init=None, num_samples=1):
        super(VAE, self).__init__()
        self.num_samples = num_samples
        # encoder
        self.fc11 = nn.Linear(input_size, h1_dim)
        self.fc21 = nn.Linear(h1_dim, h2_dim)
        self.fc31 = nn.Linear(h2_dim, h3_dim)
        self.fc41 = nn.Linear(h3_dim, h4_dim)
        self.fc51 = nn.Linear(h4_dim, z_dim-1)
        # self.fc51 = nn.Linear(h4_dim, h5_dim)
        # self.fc61 = nn.Linear(h5_dim, h6_dim)
        # self.fc71 = nn.Linear(h6_dim, z_dim-1)

        self.fc12 = nn.Linear(input_size, h1_dim)
        self.fc22 = nn.Linear(h1_dim, h2_dim)
        self.fc32 = nn.Linear(h2_dim, h3_dim)
        self.fc42 = nn.Linear(h3_dim, h4_dim)
        self.fc52 = nn.Linear(h4_dim, z_dim-1)
        # self.fc52 = nn.Linear(h4_dim, h5_dim)
        # self.fc62 = nn.Linear(h5_dim, h6_dim)
        # self.fc72 = nn.Linear(h6_dim, z_dim-1)

        self.bn11 = nn.BatchNorm1d(h1_dim)
        self.bn12 = nn.BatchNorm1d(h1_dim)
        self.bn21 = nn.BatchNorm1d(h2_dim)
        self.bn22 = nn.BatchNorm1d(h2_dim)
        self.bn31 = nn.BatchNorm1d(h3_dim)
        self.bn32 = nn.BatchNorm1d(h3_dim)
        self.bn41 = nn.BatchNorm1d(h4_dim)
        self.bn42 = nn.BatchNorm1d(h4_dim)

        # self.bn51 = nn.BatchNorm1d(h5_dim)
        # self.bn52 = nn.BatchNorm1d(h5_dim)
        # self.bn61 = nn.BatchNorm1d(h6_dim)
        # self.bn62 = nn.BatchNorm1d(h6_dim)
        # decoder
        self.fc_decoder = Linear_positive(z_dim, input_size)
        if A_init is not None:
            assert A_init.shape == self.fc_decoder.weight.shape, "Dimensions of A and fc4 weights must match"
            self.fc_decoder.weight = nn.Parameter(torch.log(A_init))
            # self.fc_decoder.weight.requires_grad = False

    def encode(self, x):
        # layers for mu
        h1 = F.relu(self.bn11(self.fc11(x)))
        # if torch.isnan(h1).any():
        #     print('h1 is nan')
        h1 = F.relu(self.bn21(self.fc21(h1)))
        h1 = F.relu(self.bn31(self.fc31(h1)))
        h1 = F.relu(self.bn41(self.fc41(h1)))

        # h1 = F.relu(self.bn51(self.fc51(h1)))
        # h1 = F.relu(self.bn61(self.fc61(h1)))
        # layers for log_var
        h2 = F.relu(self.bn12(self.fc12(x)))
        h2 = F.relu(self.bn22(self.fc22(h2)))
        h2 = F.relu(self.bn32(self.fc32(h2)))
        h2 = F.relu(self.bn42(self.fc42(h2)))

        # h2 = F.relu(self.bn52(self.fc52(h2)))
        # h2 = F.relu(self.bn62(self.fc62(h2)))
        return self.fc51(h1), self.fc52(h2)
        # return self.fc71(h1), self.fc72(h2)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        if self.num_samples == 1:
            eps = torch.randn_like(std)
            samples = mu + eps * std
            samples = torch.cat((samples, torch.zeros(std.shape[0], 1, device=mu.device)), dim=1)
        else:
            eps = torch.randn(self.num_samples, *std.shape, device=mu.device)
            samples = mu.unsqueeze(0) + eps * std.unsqueeze(0)
            samples = torch.cat((samples, torch.zeros(samples.shape[0], samples.shape[1], 1, device=mu.device)), dim=2)
        return F.softmax(samples, dim=-1)

    def decode(self, z):
        x = self.fc_decoder(z)
        return x, self.fc_decoder.weight.exp()

    def forward(self, x):
        mu, log_var = self.encode(x) # N-1 dimension
        z = self.reparameterize(mu, log_var)
        x_reconst, A = self.decode(z)
        return x_reconst, A, mu, log_var, z

