import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register

@register('pm')
class PatchMatch(nn.Module):
    def __init__(self, x=1):
        super(PatchMatch, self).__init__()
        self.all_weights = None
        self.top_3_index = None
    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, x):
        # x = [B, C, H, W]
        B, C, H, W = x.shape
        # x_unfold = [B, C*9, H*W]
        x_unfold  = F.unfold(x, kernel_size=(3, 3), padding=1)
        x_unfold = F.normalize(x_unfold, dim=2)
        R = torch.bmm(x_unfold.permute(0, 2, 1), x_unfold)
        # values = [B, H*W, 3]
        values, indices = torch.topk(R, k=3, dim=2)
        unfold_1 = self.bis(x_unfold, 2, indices[:, :, 0]).permute(0, 2, 1).reshape(B, H*W, -1, C)
        unfold_2 = self.bis(x_unfold, 2, indices[:, :, 1]).permute(0, 2, 1).reshape(B, H*W, -1, C)
        unfold_3 = self.bis(x_unfold, 2, indices[:, :, 2]).permute(0, 2, 1).reshape(B, H*W, -1, C)
        patch_stack = torch.cat((unfold_1, unfold_2, unfold_3), dim=2)
        # attention
        Q = x.reshape(B, C, H*W).permute(0, 2, 1) # [B, N, C]
        K = patch_stack # [B, N, K, C]
        V = K # [B, N, k, C]
        attention_scores = torch.einsum('bnc,bnkc->bnk', Q, K) / (C ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=2)
        output = torch.einsum('bnk,bnkc->bnc', attention_weights, V)
        return output.reshape(B, H, W, C).permute(0, 3, 1, 2)
    
    @torch.no_grad()
    def get_attention_map(self, x):
        B, C, H, W = x.shape
        x_unfold  = F.unfold(x, kernel_size=(3, 3), padding=1)
        x_unfold = F.normalize(x_unfold, dim=2)
        R = torch.bmm(x_unfold.permute(0, 2, 1), x_unfold)
        _, indices = torch.topk(R, k=3, dim=2)
        return R, indices