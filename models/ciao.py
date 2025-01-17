import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_coord
import models
from models import register
import time
class LocalImplicitSRNet(nn.Module):
    """
    The subclasses should define `generator` with `encoder` and `imnet`,
        and overwrite the function `gen_feature`.
    If `encoder` does not contain `mid_channels`, `__init__` should be
        overwrite.

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        feat_unfold (bool): Whether to use feature unfold. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 encoder_spec,
                 imnet_q_spec,
                 imnet_k_spec,
                 imnet_v_spec,
                 nla_spec,
                 query_mlp,
                 key_mlp,
                 value_mlp,
                 local_size=2,
                 feat_unfold=True,
                 eval_bsize=None,
                 non_local_attn=True,
                 multi_scale=[4],
                 softmax_scale=1,
                 ):
        super().__init__()

        self.feat_unfold = feat_unfold
        self.eval_bsize = eval_bsize
        self.local_size = local_size
        self.non_local_attn = non_local_attn
        self.multi_scale = multi_scale
        self.softmax_scale = softmax_scale

        # imnet
        self.encoder = models.make(encoder_spec)
        # 输入通道
        imnet_dim = self.encoder.mid_channels
        if self.feat_unfold:
            imnet_q_spec['args']['in_dim'] = imnet_dim * 9
            imnet_k_spec['args']['in_dim'] = imnet_k_spec['args']['out_dim'] = imnet_dim * 9
            imnet_v_spec['args']['in_dim'] = imnet_v_spec['args']['out_dim'] = imnet_dim * 9
        else:
            imnet_q_spec['args']['in_dim'] = imnet_dim
            imnet_k_spec['args']['in_dim'] = imnet_k_spec['args']['out_dim'] = imnet_dim
            imnet_v_spec['args']['in_dim'] = imnet_v_spec['args']['out_dim'] = imnet_dim
        # coord and cell
        imnet_k_spec['args']['in_dim'] += 4
        imnet_v_spec['args']['in_dim'] += 4
        # 这里是多尺度的个数，为1
        if self.non_local_attn:
            imnet_q_spec['args']['in_dim'] += 3*len(multi_scale)
            imnet_v_spec['args']['in_dim'] += 3*len(multi_scale)
            imnet_v_spec['args']['out_dim'] += 3*len(multi_scale)
        # imnet_q['in_dim'] *= 2
        self.imnet_q = models.make(imnet_q_spec) 
        self.imnet_k = models.make(imnet_k_spec) 
        self.imnet_v = models.make(imnet_v_spec) 
        
        if self.non_local_attn:
            self.cs_attn = models.make(nla_spec)    
            

    def forward(self, x, coord, cell, test_mode=False):
        """Forward function.

        Args:
            x: input tensor.
            coord (Tensor): coordinates tensor.
            cell (Tensor): cell tensor.
            test_mode (bool): Whether in test mode or not. Default: False.

        Returns:
            pred (Tensor): output of model.
        """
        feature = self.gen_feature(x)

        if self.eval_bsize is None or not test_mode:
            pred = self.query_rgb(feature, x, coord, cell)
        else:
            pred = self.batched_predict(feature, x, coord, cell)

        pred += F.grid_sample(x, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                    padding_mode='border', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

        return pred


    def query_rgb(self, features, x, coord, scale=None):
        """Query RGB value of GT.

        Copyright (c) 2020, Yinbo Chen, under BSD 3-Clause License.

        Args:
            feature (Tensor): encoded feature.
            coord (Tensor): coord tensor, shape (BHW, 2).

        Returns:
            result (Tensor): (part of) output.
        """

        res_features = []
        for feature in features:
            
            B, C, H, W = feature.shape      #[16, 64, 48, 48]

            if self.feat_unfold:
                feat_q = F.unfold(feature, 3, padding=1).view(B, C*9, H, W)         #[16, 576, 48, 48]
                feat_k = F.unfold(feature, 3, padding=1).view(B, C*9, H, W)         #[16, 576, 48, 48]
                if self.non_local_attn:
                    start_time = time.time()
                    non_local_feat_v = self.cs_attn(x)                        #[16, 64, 48, 48]
                    time_diff = time.time() - start_time
                    # print(f"nla elapse = {int((time_diff % 60)):02}.{int((time_diff % 1) * 100):02}")
                    feat_v = F.unfold(feature, 3, padding=1).view(B, C*9, H, W)     #[16, 576, 48, 48]
                    feat_v = torch.cat([feat_v, non_local_feat_v], dim=1)           #[16, 576+64, 48, 48]
                else:
                    feat_v = F.unfold(feature, 3, padding=1).view(B, C*9, H, W)     #[16, 576, 48, 48]
            else:
                feat_q = feat_k = feat_v = feature


            # query
            query = F.grid_sample(feat_q, coord.flip(-1).unsqueeze(1), mode='nearest', 
                        align_corners=False).permute(0, 3, 2, 1).contiguous()       #[16, 2304, 1, 576]

            feat_coord = make_coord(feature.shape[-2:], flatten=False).permute(2, 0, 1) \
                            .unsqueeze(0).expand(B, 2, *feature.shape[-2:])         #[16, 2, 48, 48]
            feat_coord = feat_coord.to(coord)

            if self.local_size == 1:
                v_lst = [(0, 0)]
            else:
                # 就是[-1, -1]到[1, 1]
                v_lst = [(i,j) for i in range(-1, 2, 4-self.local_size) for j in range(-1, 2, 4-self.local_size)]
            eps_shift = 1e-6
            preds_k, preds_v = [], []
            
            for v in v_lst:
                vx, vy = v[0], v[1]
                # project to LR field
                # tx, ty是归一化后的H,W
                tx = ((H - 1) / (1 - scale[:,0,0])).view(B,  1)     # [16, 1]
                ty = ((W - 1) / (1 - scale[:,0,1])).view(B,  1)     # [16, 1]
                rx = (2*abs(vx) -1) / tx if vx != 0 else 0          # [16, 1]
                ry = (2*abs(vy) -1) / ty if vy != 0 else 0          # [16, 1]
                
                bs, q = coord.shape[:2]     
                coord_ = coord.clone()  # [16, 2304, 2]
                if vx != 0:
                    coord_[:, :, 0] += vx /abs(vx) * rx + eps_shift  # [16, 2304]
                if vy != 0:
                    coord_[:, :, 1] += vy /abs(vy) * ry + eps_shift  # [16, 2304]
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # key and value
                key = F.grid_sample(feat_k, coord_.flip(-1).unsqueeze(1), mode='nearest', 
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()          #[16, 2304, 64*9]
                value = F.grid_sample(feat_v, coord_.flip(-1).unsqueeze(1), mode='nearest', 
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()          #[16, 2304, 64*9+3]

                #Interpolate K to HR resolution
                coord_k = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)       #[16, 2304, 2]

                Q, K = coord, coord_k   #[16, 2304, 2]
                rel = Q - K             #[16, 2304, 2]
                rel[:, :, 0] *= feature.shape[-2]   # without mul
                rel[:, :, 1] *= feature.shape[-1]
                inp = rel   #[16, 2304, 2]

                scale_ = scale.clone()      #[16, 2304, 2]
                scale_[:, :, 0] *= feature.shape[-2]
                scale_[:, :, 1] *= feature.shape[-1]

                inp_v = torch.cat([value, inp, scale_], dim=-1)   #[16, 2304, 64*9+3+4]
                inp_k = torch.cat([key, inp, scale_], dim=-1)     #[16, 2304, 64*9+4]
                    

                inp_k = inp_k.contiguous().view(bs * q, -1)
                inp_v = inp_v.contiguous().view(bs * q, -1)
                weight_k = self.imnet_k(inp_k).view(bs, q, -1).contiguous()   #[16, 2304, 576]
                # print(f"key.shape = {key.shape}, weight_k.shape = {weight_k.shape}")
                pred_k = (key * weight_k).view(bs, q, -1)                     #[16, 2304, 576]
                
                weight_v = self.imnet_v(inp_v).view(bs, q, -1).contiguous()   #[16, 2304, 576]
                pred_v = (value * weight_v).view(bs, q, -1)                   #[16, 2304, 576]

                preds_v.append(pred_v)
                preds_k.append(pred_k)

            preds_k = torch.stack(preds_k, dim=-1)                      # [16, 2304, 576, 4]
            preds_v = torch.stack(preds_v, dim=-2)                      # [16, 2304, 4, 576]

            attn = (query @ preds_k)                                    # [16, 2304, 1, 4]
            x = ((attn/self.softmax_scale).softmax(dim=-1) @ preds_v)   # [16, 2304, 1, 576]
            x = x.view(bs*q, -1)                                        # [16*2304, 576]

            res_features.append(x) 

        result = torch.cat(res_features, dim=-1)    # [36864, 640]
        result = self.imnet_q(result)               # [36864, 3]
        # print(f"result.shape2 = {result.shape}")
        result = result.view(bs, q, -1)

        return result

    def batched_predict(self, x, coord, cell):
        """Batched predict.

        Args:
            x (Tensor): Input tensor.
            coord (Tensor): coord tensor.
            cell (Tensor): cell tensor.

        Returns:
            pred (Tensor): output of model.
        """
        with torch.no_grad(): 
            n = coord.shape[1]
            left = 0
            preds = []
            while left < n:
                right = min(left + self.eval_bsize, n)
                pred = self.query_rgb(x, coord[:, left:right, :],
                                      cell[:, left:right, :])
                preds.append(pred)
                left = right
            pred = torch.cat(preds, dim=1)
        return pred
@register("ciao")
class LocalImplicitSREDSR(LocalImplicitSRNet):
    """LocalImplicitSR based on EDSR.

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        local_ensemble (bool): Whether to use local ensemble. Default: True.
        feat_unfold (bool): Whether to use feature unfold. Default: True.
        cell_decode (bool): Whether to use cell decode. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self,
                 encoder_spec,
                 imnet_q_spec,
                 imnet_k_spec,
                 imnet_v_spec,
                 nla_spec,
                 query_mlp=None,
                 key_mlp=None,
                 value_mlp=None,
                 local_size=2,
                 feat_unfold=True,
                 eval_bsize=None,
                 non_local_attn=True,
                 multi_scale=[2],
                 softmax_scale=1,
                 ):
        super().__init__(
            encoder_spec=encoder_spec,
            imnet_q_spec=imnet_q_spec,
            imnet_k_spec=imnet_k_spec,
            imnet_v_spec=imnet_v_spec,
            nla_spec=nla_spec,
            query_mlp=query_mlp,
            key_mlp=key_mlp,
            value_mlp=value_mlp,
            local_size=local_size,
            feat_unfold=feat_unfold,
            eval_bsize=eval_bsize,
            non_local_attn=non_local_attn,
            multi_scale=multi_scale,
            softmax_scale=softmax_scale,
            )
        self.encoder = models.make(encoder_spec)
        self.conv_first = self.encoder.conv_first
        self.body = self.encoder.body
        self.conv_after_body = self.encoder.conv_after_body
        del self.encoder

    def gen_feature(self, x):
        """Generate feature.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        x = self.conv_first(x)
        res = self.body(x)
        res = self.conv_after_body(res)
        res += x

        return [res]