import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict
import torch.nn.functional as F
import cv2
from pathlib import Path
ROOT=Path("test_picture")

def gen_picture(model_path, input, output, scale):
    '''
    model_path : str, 
    input : str,
    output : str
    scale : int
    out : save picture in output name 
    '''
    # 转换为张量 [C, H, W]
    img = transforms.ToTensor()(Image.open(input).convert('RGB'))
    model = models.make(torch.load(model_path)['model'], load_sd=True).cuda()
    # 切片
    # ih, iw = 240, 240
    # start, end = 400, 500
    # img = img[:, start:start + ih, end:end + iw]
    _, H, W = img.shape
    if scale == 1:
        transforms.ToPILImage()(img).save(output)
        return
    h, w = H * scale, W * scale
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred).save(output)

def gen_picture_bicubic_pytorch(input, scale=4, output='bicubic_pytorch.png'):
    '''
    model_path : str, 
    input : str,
    output : str
    scale : int
    out : save picture in output name 
    '''
    # 转换为张量 [C, H, W]
    img = transforms.ToTensor()(Image.open(input).convert('RGB'))
    # 切片
    # ih, iw = 240, 240
    # start, end = 400, 500
    # img = img[:, start:start + ih, end:end + iw]
    _, H, W = img.shape
    if scale == 1:
        transforms.ToPILImage()(img).save(output)
        return
    h, w = H * scale, W * scale
    x_resized = F.interpolate(img.unsqueeze(0), size=(h, w), mode='bicubic', 
                              align_corners=False)
    transforms.ToPILImage()(x_resized[0]).save(output)

def gen_picture_bicubic_opencv(input, scale=4, output='bicubic_opencv.png'):
    img = cv2.imread(input)

    height, width = img.shape[:2]
    H, W = height * scale, width * scale
    resized_img = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output, resized_img)

def get_liif_pic(path, shape, output=Path('liif_pic.png')):
    '''
    path : image path
    shape : (W, H)
    '''
    img = cv2.imread(path)
    cv2.imwrite(output.parent.joinpath("origin.png"), img)
    H, W, _ = img.shape
    print(f"H, W = {H, W}")
    h_start, w_start = 310, 690
    h, w = 128, 128
    img = img[h_start:h_start+h, w_start:w_start+w, :]
    
    resized_img = cv2.resize(img, shape, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output, resized_img)
def main():
    if not ROOT.exists():
        os.mkdir(ROOT)
    path = Path("/home/ywn/graduate/DATASET/benchmark/Urban100/HR/img004.png")

    parent_path = ROOT.joinpath(str(path.name).split('.')[0])

    if not parent_path.exists():
        os.mkdir(parent_path)

    get_liif_pic(path, (128, 128), parent_path.joinpath('liif_pic.png'))
    input_img = parent_path.joinpath('liif_pic.png')
    # gen_picture("/home/ywn/graduate/liif/save/_ywn/epoch-700.pth", input_img, parent_path.joinpath("attention.png"), 8)
    gen_picture('/home/ywn/refsr/LIIF/save/div2k_1014/epoch-best.pth', input_img, parent_path.joinpath("liif-best.png"), 8)
    gen_picture('/home/ywn/refsr/LIIF/save/div2k_1014/epoch-100.pth', input_img, parent_path.joinpath("liif-200.png"), 8)
    gen_picture_bicubic_opencv(input_img, 8, parent_path.joinpath('bicubic_opencv.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/home/ywn/graduate/liif/liif_pic.png')
    parser.add_argument('--model', default='/home/ywn/refsr/LIIF/save/div2k_1014/epoch-best.pth')
    parser.add_argument('--scale', default=4)
    parser.add_argument('--output', default='attention.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main()
    # model_spec = torch.load(args.model)['model']
    # print(model_spec)
    








    
