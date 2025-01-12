from matplotlib import pyplot as plt
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import seaborn as sns
import models
import math
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
cmap = LinearSegmentedColormap.from_list(
    "transparent_cmap", 
    [(1, 0, 0, 0), (1, 0, 0, 0.5), (1, 0, 0, 1)],  # RGBA 透明度渐变
    N=256
)
torch.manual_seed(43)
def index_mapping(index, scale, edge):
    '''
    index : 小的索引
    scale : 放大倍数
    edge : 大图的边长
    '''
    h = index // edge
    w = index % edge
    nh, nw = h // scale, w // scale
    return nh * (edge // scale) + nw

def plt_and_save_map(model_path, img_path):
    # 加载图片得到输入
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图片转换为 Tensor，值归一化到 [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]（可选）
    ])
    image_tensor = transform(image)
    input = image_tensor.unsqueeze(0).cuda()
    # 加载模型并得到attention和index
    model_spec = torch.load(model_path)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    # attention_weights=[B, H*W/(scale*scale), H*W/(scale*scale)]
    # index=[B, H*W, K, 2]
    attention_weights, index, index_3 = model.nla.get_attention_map(input)
    B, N, _, _ = index.shape
    scale  = int(math.sqrt(N // attention_weights.shape[1]))
    H = int(math.sqrt(N))
    W = H
    weights = attention_weights.reshape(B, attention_weights.shape[1], H // scale, W // scale)
    weights = F.interpolate(weights, (H, W), mode='bicubic')[0].cpu().detach().numpy()
    random_index = np.random.randint(1, N, size=(1))[0]
    index = index.cpu().detach().numpy()
    # 第一个
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    data = weights[index_mapping(random_index, scale, H), :, :]
    plt.imshow(image)
    sns.heatmap(data, cmap='rainbow', alpha=0.8,cbar=False)
    x = index[0, random_index, :, 1]
    y = index[0, random_index, :, 0]
    plt.scatter(x, y, marker='o', s=20, color='white')
    q_x = random_index % H
    q_y = random_index // W
    plt.scatter(q_x, q_y, marker='*',s=50 ,color='black')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.title(f"qx = {q_x}, qy = {q_y}")
    # plt.savefig(f"./map/attention_map{q_x, q_y}.png")
    # 不插值情况下的检查
    index3 = index_mapping(random_index, scale, H)
    
    h , w = H // scale, W // scale
    data1 = attention_weights[0][index3, :].reshape(h, w).cpu().detach().numpy()
    # _, topK = torch.topk(attention_weights, 4, dim=2, largest=True)
    # print(topK.shape)
    # topK = topK[0].cpu().detach().numpy()
    
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    sns.heatmap(data1, cmap='rainbow', alpha=0.8, cbar=False)
    x = index_3[0, index3, :, 1].cpu().detach().numpy()
    y = index_3[0, index3, :, 0].cpu().detach().numpy()
    plt.scatter(x, y, marker='o', s=20, color='white')
    q_x = index3 % w
    q_y = index3 // h

    plt.scatter(q_x, q_y, marker='*',s=50 ,color='black')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.title(f"qx = {q_x}, qy = {q_y}")
    print(f"random_index = {random_index}")
    
    plt.savefig(f"./map/attention_map{random_index}.png")
def main(model_path, img_path):
    plt_and_save_map(model_path, img_path)

if __name__ == '__main__':
    main("/home/ywn/graduate/ALIIF/archive_models/div2k0110/epoch-best.pth", 
         "/home/ywn/graduate/ALIIF/liif_pic.png")