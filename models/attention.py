import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from models import register
class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=2, bias=True,
        bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, stride, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)
def default_conv(in_channels, out_channels, kernel_size,stride=2, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

@register("nla")
class NonLocalAttention(nn.Module):
    def __init__(self, in_dim=3, K=4, scale=4, dims=[6, 9]):
        super(NonLocalAttention, self).__init__()
        # kernel_size = 3, stride = 2, padding = 1, /2
        # kernel_size = 5, strid = 4, padding = 0, /4
        self.block1 = BasicBlock(default_conv, in_channels=in_dim, out_channels=dims[0], 
                                 kernel_size=5, stride=scale, bn=True, act=nn.ReLU())
        self.block2 = BasicBlock(default_conv, in_channels=dims[0], out_channels=dims[1], 
                                 kernel_size=5, stride=scale, bn=True, act=nn.ReLU())
        # self.conv1 = nn.Conv2d(in_dim, dims[0], 3, 2, 1, bias=False)
        # self.conv2 = nn.Conv2d(dims[0], dims[1], 3, 2, 1, bias=False)
        self.x1 = None
        self.x2 = None
        self.x3 = None
        self.K = K
        self.scale = scale
        self.q = None
        self.v = None
        # 画attention map图的数据
        self.indices_3 = None
        self.indices = None
        self.attention_weights = None
        # 第3层使用的proj
        self.proj_q3 = nn.Conv2d(dims[1], dims[1], 1, 1, 0, bias=False)
        self.proj_k3 = nn.Conv2d(dims[1], dims[1], 1, 1, 0, bias=False)
        # self.proj_v3 = nn.Conv2d(dims[1], dims[1], 1, 1, 0, bias=False)
        # 第2层使用的proj
        self.proj_q2 = nn.Conv2d(dims[0], dims[0], 1, 1, 0, bias=False)
        self.proj_k2 = nn.Conv2d(dims[0], dims[0], 1, 1, 0, bias=False)
        # self.proj_v2 = nn.Conv2d(dims[1], dims[1], 1, 1, 0, bias=False)
        # 第1层使用的proj
        self.proj_q1 = nn.Conv2d(in_dim, in_dim, 1, 1, 0, bias=False)
        self.proj_k1 = nn.Conv2d(in_dim, in_dim, 1, 1, 0, bias=False)
        # self.proj_v1 = nn.Conv2d(dims[1], dims[1], 1, 1, 0, bias=False)
    def gen_feature(self, x):
        self.x1 = x
        self.x2 = self.block1(x)
        self.x3 = self.block2(self.x2)

    def expand_indices(self, topK_indices, scale, now_shape):
        '''
        topK_indices: [B, C, H*W], 每个点topK个近似的点
        scale : int, 放大倍数
        now_shape : [H, W], 放大前的shape
        '''
        B, N, K = topK_indices.shape
        H, W = now_shape
        # 假设某个点的序列号为n1，则扩大scale倍后的序列号n2是这样计算的
        # n1 % W = i, n - i = j
        # n2 = j * scale * scale + i * scale
        new_topK = torch.zeros_like(topK_indices)
        for k in range(K):
            i = topK_indices[:, :, k] % W
            j = topK_indices[:, :, k] - i
            new_topK[:, :, k] = j * scale * scale + i * scale
        # 先把水平方向上的重复一遍
        new_topK = new_topK.repeat_interleave(scale, dim=2)
        for k in range(K):
            new_topK[:, :, k * scale : (k + 1) * scale] += torch.arange(scale).view(1, 1, -1).to(new_topK)
        # 得到了每个点第一行的排列，将每一行重复scale次
        # 就得到了每个点扩展成的(scale, scale)的大小
        # 无需考虑顺序问题，因为索引大小表示前后，计算出值排序即可
        new_topK = new_topK.repeat(1, 1, scale)
        for s in range(scale):
            new_topK[:, :, s * scale * K : (s + 1) * scale * K] += (s * scale * W)

        new_topK, _ = torch.sort(new_topK, dim=2)
        # 接着需要注意顺序，先每个点重复scale次，接着分组重复scale次
        new_topK = new_topK.repeat_interleave(scale, dim=1)
        # 每一行有W * scale个元素，所以按照这个分割每一行
        split_tensors = torch.split(new_topK, W * scale, dim=1)
        # 将每一行重复scale次
        repeated_segments = [segment.repeat(1, scale, 1) for segment in split_tensors]
        # result = [B, N*scale*scale, K*scale*scale]
        result = torch.cat(repeated_segments, dim=1)
        return result
    def attention(self, x, indices, layer=2):
        '''
        x: [B, C, H, W], 是待求张量
        indices: [B, H*W, K], K个索引
        '''
        B, C, H, W = x.shape
        if layer == 2:
            q = self.proj_q2(x).reshape(B, C, H*W)
            k = self.proj_k2(x).reshape(B, C, H*W)
            # v = self.proj_v2(x).reshape(B, C, H*W)
        else :
            q = self.proj_q1(x).reshape(B, C, H*W)
            k = self.proj_k1(x).reshape(B, C, H*W)
            # v = self.proj_v1(x).reshape(B, C, H*W)
            self.q = q
            self.k = k
            # self.v = v
        indices = indices.unsqueeze(1).expand(-1, C, -1, -1)
        k_ = [None] * (H*W)
        # 对于每个采样点取样
        for i in range(H*W):
            k_[i] = torch.gather(k, dim=2, index=indices[:, :, i, :])
        k = torch.stack(k_, dim=2) # [B, C, N, K]
        # 将q, k, v转置，通道放在最后面
        # q=[B, N, C], k,v =[B, N, K, C]
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 3, 1)
        # scores = [B, N, K]
        attention_scores = torch.einsum('bnc,bnkc->bnk', q, k)
        d_k = C
        attention_scores = attention_scores / (d_k ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=2)
        return attention_weights
    def non_local_attention(self):
        '''
        第一阶段 在最小的形状上
        '''
        x = self.x3
        B3, C3, H3, W3 = x.shape
        q = self.proj_q3(x).reshape(B3, C3, H3*W3)
        k = self.proj_k3(x).reshape(B3, C3, H3*W3)
        # v = self.proj_v3(x).reshape(B, C, H*W)
        attention_3 = torch.einsum('b c m, b c n -> b m n', q, k)
        d_k_3 = C3 ** 0.5
        attention_3 = attention_3 / d_k_3
        attention_3 = F.softmax(attention_3, dim=2)
        # 取出前self.K个most similar的
        _, topK_indices_3 = torch.topk(attention_3, self.K, dim=2)
        '''
        第二阶段
        '''
        # 将indices扩展
        topK_indices_3_expand = self.expand_indices(topK_indices_3, self.scale, (H3, W3))
        # 利用indices采样
        attention2 = self.attention(self.x2, topK_indices_3_expand, 2)
        _, temp_indices_2 = torch.topk(attention2, self.K, dim=2)
        # temp_indices_2 = [B, H2*W2, K]
        # 从topK_indices_3_expand中选择K个索引，得到在第二阶段的实际索引值
        topK_indices_2 = torch.gather(topK_indices_3_expand, dim=2, index=temp_indices_2)
        '''
        # 第三阶段
        # '''
        topK_indices_2_expand = self.expand_indices(topK_indices_2, self.scale, 
                                            (self.x2.shape[-2], self.x2.shape[-1]))
        attention1 = self.attention(self.x1, topK_indices_2_expand, 1)

        # 在K*scale*scale中得到前K个most similar
        _, temp_indices_1 = torch.topk(attention1, self.K, dim=2)
        # 这里才是实际的索引值
        topK_indices_1 = torch.gather(topK_indices_2_expand, dim=2, index=temp_indices_1)
        '''
        最后输出阶段
        '''
        B1, C1, H1, W1 = self.x1.shape
        Q = self.q
        # 重复匹配通道
        topK_indices_1_expand = topK_indices_1.clone().unsqueeze(1).expand(-1, C1, -1, -1)
        K_ = [None] * (H1 * W1)
        for i in range(H1 * W1):
            K_[i] = torch.gather(self.k, dim=2, index=topK_indices_1_expand[:, :, i, :])
        
        K = torch.stack(K_, dim=2)
        V = K
        Q = Q.permute(0, 2, 1) # [B, N, C]
        K = K.permute(0, 2, 3, 1) # [B, N, K, C]
        V = V.permute(0, 2, 3, 1) # [B, N, k, C]
        attention_scores = torch.einsum('bnc,bnkc->bnk', Q, K)
        d_k = C1
        attention_scores = attention_scores / (d_k ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=2) # [B, N, K]
        output = torch.einsum('bnk,bnkc->bnc', attention_weights, V) # [B, N, C]
        # 用于attention map的数据
        self.attention_weights = attention_weights
        self.indices = topK_indices_1
        self.indices_3 = topK_indices_3
        return output
    def get_attention_map(self, x):
        self.gen_feature(x)
        self.non_local_attention()

        B, C, H, W = x.shape
        # attention_weights=[B, H*W/scale, H*W/scale]
        # indices = [B, H*W, K]
        attention_weights = self.attention_weights,
        indices = self.attention_weights 
        indices_3 = self.indices_3
        # indices = [B, H*W, K, 2] 2代表一个坐标[H, W], K个坐标
        indices = indices.unsqueeze(-1).repeat(1, 1, 1, 2)
        indices[:, :, :, 0] = torch.div(indices[:, :, :, 0], H, rounding_mode='trunc')
         
        indices[:, :, :, 1] = torch.remainder(indices[:, :, :, 1], W)
        indices_3 = indices.unsqueeze(-1).repeat(1, 1, 1, 2)
        indices_3[:, :, :, 0] = torch.div(indices_3[:, :, :, 0], H // self.scale // self.scale, rounding_mode='trunc')
        indices_3[:, :, :, 1] = torch.remainder(indices_3[:, :, :, 1], H // self.scale // self.scale)
        return attention_weights, indices, indices_3

    def forward(self, x):
        self.gen_feature(x)
        self.non_local_attention()
        # print(f"self.x1.shape = {self.x1.shape}")
        # print(f"self.x2.shape = {self.x2.shape}")
        # print(f"self.x3.shape = {self.x3.shape}")
        B, C, H, W = x.shape
        output = self.non_local_attention()
        output = output.permute(0, 2, 1).reshape(B, C, H, W)
        print(f"output.shape = {output.shape}")
        return x