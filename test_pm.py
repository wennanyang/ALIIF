import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils
from utils import make_coord
def clip_test(img_lq, model, scale):
    
    all_flops = []
    sf = scale
    b, c, h, w = img_lq.size()

    tile = min(144, h, w)
    tile_overlap = 32
    
    stride = tile - tile_overlap
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
    W = torch.zeros_like(E)
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            
            target_size = (round(in_patch.shape[-2]*sf), 
                            round(in_patch.shape[-1]*sf))

            hr_coord = make_coord(target_size[-2:]).unsqueeze(0).expand(b, -1, 2).to(img_lq)  #.cuda() TODO
            cell = torch.ones_like(hr_coord).to(img_lq)  #.cuda()   TODO
            cell[:, :, 0] *= 2 / target_size[-2]
            cell[:, :, 1] *= 2 / target_size[-1]

            out_patch = model(in_patch, hr_coord, cell, test_mode=True) 
            
            ih, iw = in_patch.shape[-2:]
            shape = [in_patch.shape[0], round(ih * sf), round(iw * sf), 3]
            out_patch = out_patch.view(*shape).permute(0, 3, 1, 2).contiguous()

            out_patch_mask = torch.ones_like(out_patch)

            E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
            W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
    output = E.div_(W)
    output = output.view(b, 3, -1).permute(0,2,1).contiguous()
    return output

def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False, scale=1):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        if eval_bsize is None:
            with torch.no_grad():
                # pred = model(inp, batch['coord'], batch['cell'])
                pred = clip_test(inp, model, scale)
                
        else:
            pred = clip_test(inp, model, scale)
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)
        if eval_type is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0, 1, 2')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True,
        scale=config.get('scale'))
    print('result: {:.4f}'.format(res))
