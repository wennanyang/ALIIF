from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
import torch
import seaborn as sns
import models
import math
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import cv2
from pathlib import Path
cmap = LinearSegmentedColormap.from_list(
    "transparent_cmap", 
    [(1, 0, 0, 0), (1, 0, 0, 0.5), (1, 0, 0, 1)],  # RGBA 透明度渐变
    N=256
)

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
    '''
    临时方案
    '''
    model_spec['name']='ciao_pm'

    model = models.make(model_spec, load_sd=True).cuda()
    # 
    feature = model.gen_feature(input)[0]   # [1, 64, 48, 48]
    R, index = model.pm.get_attention_map(feature)
    B, N, _ = R.shape
    H = W = int(math.sqrt(N))
    weights = R.reshape(B, N, H, W)[0].detach().cpu().numpy()
    random_index = np.random.randint(1, N, size=(1))[0]
    index = index[0].cpu().numpy()
    # 第一个
    plt.figure(figsize=(8, 8))
    data = weights[random_index, :, :]
    plt.imshow(image)
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
    print(np.array_equal(data, weights[0, :, :]))
    plt.savefig(f"./map/{CROP.parent.name}/attention_map{random_index}.png")

if __name__ == '__main__':
    # monarch [130 : 130+292, 250 : 250+292, :]
    # zebra [30: 30+450, 100 : 100+450, :]
    IMG = Path("../DATASET/benchmark/Set14/HR/zebra.png")
    PARENT = Path("./map").joinpath(IMG.name.split('.')[0])
    if not PARENT.exists():
        PARENT.mkdir()
    img = cv2.imread(IMG)
    CROP = PARENT.joinpath("origin.png")
    img = cv2.resize(img[30: 30+450, 100 : 100+450, :], (48, 48), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(PARENT.joinpath("origin.png"), img)
    plt_and_save_map("./archive_models/pm_2/epoch-best.pth", CROP)

