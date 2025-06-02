import torch
import torch.nn as nn
import pdb


class AttentionWeight(nn.Module):

    def __init__(self):
        super(AttentionWeight, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, host_q, vice_q):

        b, c, h, w = host_q.size()

        host_q = self.avg_pool(host_q)  # B * C * 1 * 1
        vice_q = self.avg_pool(vice_q)  # B * C * 1 * 1
        atten_q = vice_q.view(b, c, 1).permute(0, 2, 1)
        atten_k = host_q.view(b, c, 1)
        # pdb.set_trace()

        attention = torch.bmm(atten_q, atten_k).view(b)  # B * 1 * 1

        return attention
