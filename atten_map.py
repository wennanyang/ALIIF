from matplotlib import pyplot as plt
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import seaborn as sns
class AttentionModule(nn.Module):
    def __init__(self, input_dim, attention_dim):
        """
        初始化 Attention 模块
        :param input_dim: 输入特征的维度
        :param attention_dim: Attention 中间特征的维度
        """
        super(AttentionModule, self).__init__()
        
        # 定义线性变换
        self.query_layer = nn.Linear(input_dim, attention_dim)
        self.key_layer = nn.Linear(input_dim, attention_dim)
        self.value_layer = nn.Linear(input_dim, attention_dim)
        
        # 归一化的 Softmax 层
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，形状为 (batch_size, seq_length, input_dim)
        :return: 加权后的输出，形状为 (batch_size, seq_length, attention_dim)
        """
        # 计算 Query、Key、Value
        query = self.query_layer(x)  # (batch_size, seq_length, attention_dim)
        key = self.key_layer(x)      # (batch_size, seq_length, attention_dim)
        value = self.value_layer(x)  # (batch_size, seq_length, attention_dim)
        
        # 计算注意力分数 (scaled dot-product attention)
        attention_scores = torch.bmm(query, key.transpose(1, 2))  # (batch_size, seq_length, seq_length)
        attention_scores = attention_scores / (query.size(-1) ** 0.5)  # 缩放
        
        # 归一化得到注意力权重
        attention_weights = self.softmax(attention_scores)  # (batch_size, seq_length, seq_length)
        
        # 使用注意力权重加权 Value
        output = torch.bmm(attention_weights, value)  # (batch_size, seq_length, attention_dim)
        
        return output, attention_weights
torch.manual_seed(42)

def img2tensor(path):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图片转换为 Tensor，值归一化到 [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]（可选）
    ])
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)
def attention(path):
    img = img2tensor(path)
    B, C, H, W = img.shape
    K = 4
    attention_weights = torch.rand((B, H * W, K))
    attention_weights = F.softmax(attention_weights, dim=2)
    attention_map = attention_weights.mean(dim=-1)
    attention_map = attention_map[0].reshape(H, W).detach().numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(img[0].permute(1, 2, 0).detach().numpy())
    plt.imshow(attention_map, cmap='rainbow', alpha=0.5)
    plt.title("Attention Map")
    plt.axis("off")
    plt.savefig("output_image.png")
def main(path):
    attention(path)

if __name__ == '__main__':
    main("/home/ywn/graduate/liif/img004/liif_pic.png")