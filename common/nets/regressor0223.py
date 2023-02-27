import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from nets.hand_head import hand_regHead, hand_Encoder
from utils.ChebConv import _ResChebGC, adj_mx_from_edges, ChebConv


"""
被编辑于: 2023-2-23
更改内容: 尝试使用GAT作为regressor, 替换调ChebConv
"""

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()

        self.bn1d = nn.BatchNorm1d(21)
        self.leakyrelu = nn.LeakyReLU(0.1)

        edges = torch.tensor([[0, 13], [0, 1], [0, 4], [0, 10], [0, 7],
                                [13, 14], [14, 15], [15, 16],
                                [1, 2], [2, 3], [3, 17],
                                [4, 5], [5, 6], [6, 18], 
                                [10, 11], [11, 12], [12, 19], 
                                [7, 8], [8, 9], [9, 20]], dtype=torch.long)
        self.adj = adj_mx_from_edges(num_pts=21, edges=edges, sparse=False).cuda()
        self.edg = torch.tensor([[0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                 [13, 1, 4, 10, 7, 2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]], dtype=torch.long).cuda()
        self.encoeder = GATConv(235, 32, 4)
        self.gconv_output = ChebConv(in_c=128, out_c=3, K=2)
        
    
    def forward(self, feature_s, target_s):
        """
        feats: [b, c, h, w], feature
        target:, [b, 21, h, w], 21 joints heatmap
        """

        query_embed_list = []
        for feature, target in zip(feature_s, target_s):
            query_embed = target.flatten(1) @ feature.flatten(1).permute(1, 0)
            query_embed_list.append(query_embed)
        query_embed = torch.stack(query_embed_list, dim=0)

        query_embed = self.bn1d(query_embed)
        query_embed = self.leakyrelu(query_embed)
        B, N = query_embed.shape[0:2]

        query_embed = list(query_embed)
        query_embed = [Data(x=q, edge_index=self.edg) for q in query_embed]
        batch = Batch.from_data_list(query_embed)


        x = self.encoeder(batch.x, batch.edge_index).reshape(B, N, -1)
        x = self.gconv_output(x, self.adj)

        return x

