import torch
from models.patch_match import PatchMatch
def main():
    pt = PatchMatch()
    input = torch.randn((16, 64, 48, 48), dtype=torch.float)
    pt(input)
if __name__ == '__main__':
    main()