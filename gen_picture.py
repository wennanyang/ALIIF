import argparse
import os
from PIL import Image
import math
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import models
from utils import make_coord
from test import batched_predict
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pathlib import Path
from utils import psnr, ssim
import random
ROOT=Path("test_picture")
def calc_metrics(path_img1, path_img2):
    img1 = cv2.imread(path_img1)
    img2 = cv2.imread(path_img2)
    return psnr(img1, img2), ssim(img1, img2)
def gen_attention_map(input, model_path, output):
    img = transforms.ToTensor()(Image.open(input).convert('RGB')).cuda()

    model = models.make(torch.load(model_path)['model'], load_sd=True).cuda()
    model_ = nn.parallel.DataParallel(model)
    model = model_.module
    # 
    feature = model.gen_feature(img.unsqueeze(0))[0]   # [1, 64, 48, 48]
    R, index = model.pm.get_attention_map(feature)
    B, N, _ = R.shape
    H = W = int(math.sqrt(N))
    weights = R.reshape(B, N, H, W)[0].detach().cpu().numpy()
    random_index = np.random.randint(1, N, size=(1))[0]
    index = index[0].cpu().numpy()
    # 第一个
    plt.figure(figsize=(8, 8))
    data = weights[random_index, :, :]
    plt.imshow(cv2.imread(input))
    sns.heatmap(data, cmap='rainbow', alpha=0.8,cbar=False)
    points = index[random_index, :]
    x = points % H
    y = points // W
    plt.scatter(x, y, marker='o', s=20, color='white')
    q_x = random_index % H
    q_y = random_index // W
    plt.scatter(q_x + 0.5, q_y + 0.5, marker='*',s=50 ,color='black')
    plt.axis('off')
    plt.tight_layout(pad=3)
    plt.title(f"qx = {q_x}, qy = {q_y}")
    plt.savefig(output)
    plt.close()
def gen_picture_pm(input, scale, model_path, output):
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
    model_ = nn.parallel.DataParallel(model)
    model = model_.module
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
    # feature = model.gen_feature(((img - 0.5) / 0.5).cuda().unsqueeze(0))
    # pred = model.batched_predict(feature, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
    #     coord.unsqueeze(0), cell.unsqueeze(0))[0]
    pred = model(((img - 0.5) / 0.5).cuda().unsqueeze(0), coord.unsqueeze(0),
                 cell.unsqueeze(0), test_mode=True)
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred).save(output)
def gen_picture_liif(input, scale, model_path, output):
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
    model_ = nn.parallel.DataParallel(model)
    model = model_.module
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
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=3000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred).save(output)
def gen_picture_bicubic_opencv(input, scale=4, output='bicubic_opencv.png'):
    img = cv2.imread(input)

    height, width = img.shape[:2]
    H, W = height * scale, width * scale
    resized_img = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output, resized_img)
def gen_crop(path, hs, ws, h, w, output):
    img = cv2.imread(path)
    img = img[hs : hs + h, ws : ws + w, :]
    cv2.imwrite(img, output)
def get_LR_HR(LR_path, HR_path, hs, ws, h, w, scale : int, parent : Path):

    '''
    path : image path
    shape : (W, H)
    '''
    LR_IMG = cv2.imread(LR_path)
    HR_IMG = cv2.imread(HR_path)
    H, W, _ = LR_IMG.shape
    # print(LR_IMG.shape)
    # print(HR_IMG.shape)
    if h is None:
        h = H
    if w is None:
        w = W
    crop_h, crop_w = int(h * scale), int(w * scale)
    crop_hs, crop_ws =int(hs * scale), int(ws * scale)
    GT = HR_IMG[crop_hs:crop_hs+crop_h, crop_ws:crop_ws+crop_w, :]
    LR = LR_IMG[hs : hs+h, ws : ws + w, :]
    cv2.imwrite(parent.joinpath("aGT.png"), GT)
    cv2.imwrite(parent.joinpath("aLR.png"), LR)
    cv2.imwrite(parent.joinpath("aHR.png"), HR_IMG)
    
def save_picture(hr_path, lr_path, hs, ws, scale):
    # monarch [130 : 130+292, 250 : 250+292, :]
    # zebra [30: 30+450, 100 : 100+450, :]
    img_name = str(hr_path.name).split('.')[0]
    PARENT = ROOT.joinpath(f"{img_name} {hs} {ws} x{scale}")
    if not PARENT.exists():
        os.mkdir(PARENT)
    get_LR_HR(lr_path, hr_path, hs=hs, ws=ws, h=48, w=48, scale=scale, parent=PARENT)



    PM_MODEL = Path('/home/ywn/graduate/ALIIF/save/pm_4/epoch-best.pth')
    LIIF_MODEL = Path('/home/ywn/refsr/LIIF/save/div2k_1014/epoch-best.pth')
    CIAO_MODEL = Path('/home/ywn/graduate/ALIIF/archive_models/ciaosr/ciaosr_2/epoch-best.pth')

    # cv2.imwrite(PARENT.joinpath("LR.png"), cv2.imread(LR_path))
    # cv2.imwrite(PARENT.joinpath("HR.png"), cv2.imread(HR_PATH))
    
    GT      = PARENT.joinpath("aGT.png")
    LR      = PARENT.joinpath("aLR.png")
    BICUBIC = PARENT.joinpath("bicubic.png")
    PM      = PARENT.joinpath("pm.png")
    LIIF    = PARENT.joinpath("liif.png")
    CIAO    = PARENT.joinpath("ciao.png")
    MAP     = PARENT.joinpath("zmap.png")
    # cv2.imwrite("/home/ywn/graduate/DATASET/TEST/HR/" + NAME + ".png", cv2.imread(GT))
    # cv2.imwrite("/home/ywn/graduate/DATASET/TEST/LR_bicubic/" + NAME + ".png", cv2.imread(LR))
    # cv2.imwrite("/home/ywn/graduate/DATASET/TEST/LRbicx4/" + NAME + ".png", cv2.imread(LR))
    
    gen_picture_bicubic_opencv(LR, scale, BICUBIC)
    gen_picture_pm(LR, scale, PM_MODEL, PM)
    gen_picture_liif(LR, scale, LIIF_MODEL, LIIF)
    gen_picture_pm(LR, scale, CIAO_MODEL, CIAO)
    gen_attention_map(LR, PM_MODEL, MAP)
    PSNR1, SSIM1 = calc_metrics(PM, GT)
    PSNR2, SSIM2 = calc_metrics(BICUBIC, GT)
    PSNR3, SSIM3 = calc_metrics(LIIF, GT)
    PSNR4, SSIM4 = calc_metrics(CIAO, GT)
    # print(f"bicubic PSNR = {PSNR2:.2f}, SSIM = {SSIM2:.2f}")
    # print(f"pm PSNR = {PSNR1:.2f}, SSIM = {SSIM1:.2f}")
    # print(f"liif PSNR = {PSNR3:.2f}, SSIM = {SSIM3:.2f}")
    # print(f"ciao PSNR = {PSNR4:.2f}, SSIM = {SSIM4:.2f}")
    with open(PARENT.joinpath("psnr_ssim.txt"), 'w+') as f:
        f.write(f"pm PSNR = {PSNR1:.2f}, SSIM = {SSIM1:.2f}\n")
        f.write(f"bicubic PSNR = {PSNR2:.2f}, SSIM = {SSIM2:.2f}\n")
        f.write(f"liif PSNR = {PSNR3:.2f}, SSIM = {SSIM3:.2f}\n")
        f.write(f"ciao PSNR = {PSNR4:.2f}, SSIM = {SSIM4:.2f}\n")
    # monarch: hs=130, ws=250, h=292, w=292
    # zebra: hs=30, ws=100, h=432, w=432
    # img004: hs=190, ws=800, h=216, w=216

def main():
    if not ROOT.exists():
        os.mkdir(ROOT)
    # monarch [130 : 130+292, 250 : 250+292, :]
    # zebra [30: 30+450, 100 : 100+450, :]
    scale = 4
    # NAME = "img008"
    # Urban100
    # HR_PATH = Path(f"/home/ywn/graduate/DATASET/benchmark/Urban100/HR/{NAME}.png")
    # LR_PATH = Path(f"/home/ywn/graduate/DATASET/benchmark/Urban100/LR_bicubic/X{str(scale)}/{NAME}x{str(scale)}.png")
    # DIV2K
    # HR_PATH = Path(f"/home/ywn/graduate/DATASET/DIV2K/DIV2K_valid_HR/{NAME}.png")
    # LR_PATH = Path(f"/home/ywn/graduate/DATASET/DIV2K/DIV2K_valid_LR_bicubic/X{str(scale)}/{NAME}x{str(scale)}.png")
    HR_Path= Path("/home/ywn/graduate/DATASET/benchmark/Urban100/HR")
    LR_Path = Path(f"/home/ywn/graduate/DATASET/benchmark/Urban100/LR_bicubic/X{str(scale)}")
    Images = list(HR_Path.glob("*.png"))
    for hr in Images:
        lr = LR_Path.joinpath(f"{hr.name.split('.')[0]}x{scale}.png")
        H, W, _ = cv2.imread(lr).shape
        hs = random.randint(0, H-48)
        ws = random.randint(0, W-48)
        save_picture(hr, lr, hs, ws, scale)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./monarch.png')
    parser.add_argument('--model', default='./archive_models/pm0219/pm_1/epoch-best.pth')
    parser.add_argument('--scale', default=4)
    parser.add_argument('--output', default='monarch.png')
    parser.add_argument('--gpu', default='0, 1, 2')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main()
    # model_spec = torch.load(args.model)['model']
    # print(model_spec)
    








    
