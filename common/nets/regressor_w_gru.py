import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.hand_head import hand_regHead, hand_Encoder
from utils.ChebConv import _ResChebGC, adj_mx_from_edges, ChebConv
from utils.gru import GRU


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.conv = nn.Conv2d( 256, 21, kernel_size=1, stride=1, padding=0)

        self.bn = nn.BatchNorm2d(21)
        self.bn1d = nn.BatchNorm1d(21)
        self.avgpool = nn.AdaptiveAvgPool2d((16, 16))
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(0.1)

        edges = torch.tensor([[0, 13], [0, 1], [0, 4], [0, 10], [0, 7],
                                [13, 14], [14, 15], [15, 16],
                                [1, 2], [2, 3], [3, 17],
                                [4, 5], [5, 6], [6, 18], 
                                [10, 11], [11, 12], [12, 19], 
                                [7, 8], [8, 9], [9, 20]], dtype=torch.long)
        self.adj = adj_mx_from_edges(num_pts=21, edges=edges, sparse=False).cuda()
        
        self.gru = GRU(235, 256)
        self.encoeder = _ResChebGC(adj=self.adj, input_dim=256, output_dim=256, hid_dim=256, p_dropout=0.1)
        self.gconv_output = ChebConv(in_c=256, out_c=3, K=2)

        
    
    def forward(self, feature_s, target_s):
        """
        feats: [b, c, h, w], feature
        target:, [b, 21, h, w], 21 joints heatmap
        """

        query_embed_list = []
        for feature, target in zip(feature_s, target_s):
            # temp = target.sum(dim=-1).sum(dim=-1, keepdims=True)
            # temp = temp[:, :, None]
            # target = target / (temp + 1e-8)
            query_embed = target.flatten(1) @ feature.flatten(1).permute(1, 0)
            query_embed_list.append(query_embed)
        query_embed = torch.stack(query_embed_list, dim=0)

        query_embed = self.bn1d(query_embed)
        query_embed = self.leakyrelu(query_embed)

        query_embed = self.gru(query_embed)
        # x = self.conv(feats)
        # x = self.bn(x)
        # x = self.relu(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 2)

        # x = x.view(-1, 21, 256)
        x = self.encoeder(query_embed)
        x = self.gconv_output(x, self.adj)

        return x

