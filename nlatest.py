from models.attention import NonLocalAttention

import torch 

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
    main()