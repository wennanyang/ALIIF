from models.attention import NonLocalAttention
import torch 


torch.manual_seed(42)
# yes!! 证明是正确的！！！
def dot_multiply():
    B, C, H, W = 1, 3, 4, 4
    batch = torch.randint(0, 5, (B, C, H, W))
    batch = batch.reshape(B, C, H*W)
    K = 5
    index = torch.randint(0, H*W, (B, H*W, K))
    index = index.unsqueeze(1).expand(-1, C, -1, -1)
    k_ = [None] * (H*W)
    for i in range(H*W):
        k_[i] = torch.gather(batch, dim=2, index=index[:, :, i, :])
    k = torch.stack(k_, dim=2)
    k = k.permute(0, 2, 3, 1)
    batch = batch.permute(0, 2, 1)
    scores = torch.einsum('bnc,bnkc->bnk', batch, k)
    print(f"fisrt q data = {batch[0, 0, :]}")
    print(f"fisrt k data = {k[0, 0, :, :]}")
    print(f"first v data = {scores[0, 0, :]}")
    print(scores.shape)
def main():
    device = "cuda:0"
    batch = torch.ones((16, 3, 128, 128), device=device, dtype=torch.float)
    model = NonLocalAttention()
    model.to(device=device)
    model(batch)
    # tensor = torch.tensor([[1, 2], [3, 4]])
    # tensor = tensor.repeat(1, 2)
    # print(tensor)

if __name__ == '__main__':
    dot_multiply()