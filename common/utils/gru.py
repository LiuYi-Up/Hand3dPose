import torch
import torch.nn as nn


class GRU(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        # self.h0 = torch.zeros(1, 16, 256, device='cuda')
        self.rnn = nn.GRU(input_size, output_size, batch_first=True)

    def forward(self, x):
        # if x.shape[0] != self.h0.shape[1]:
        #     self.h0 = self.h0[:, :x.shape[0], :]
        # x, self.h0 = self.rnn(x, self.h0)
        # self.h0 = self.h0.data
        x, h0 = self.rnn(x)
        return x
