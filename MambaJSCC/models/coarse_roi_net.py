

### 2026/4/21 wyj：粗ROI识别：搭建轻量粗 ROI 网络

import torch
import torch.nn as nn


# class ConvBNReLU(nn.Module):
#     def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.block(x)
### 2026/4/23 wyj:改进
class ConvGNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, num_groups: int = 4):
        super().__init__()
        groups = min(num_groups, out_ch)
        while out_ch % groups != 0 and groups > 1:
            groups -= 1

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LightCoarseROINet(nn.Module):
    """
    输入:
        x: [B, 3, 896, 1024]

    输出:
        logits: [B, 1, 7, 8]
    """
    def __init__(self, base_ch: int = 16, grid_h: int = 7, grid_w: int = 8):
        super().__init__()

        self.stem = ConvGNReLU(3, base_ch, stride=2)              # 896x1024 -> 448x512
        self.stage1 = ConvGNReLU(base_ch, base_ch * 2, stride=2)  # -> 224x256
        self.stage2 = ConvGNReLU(base_ch * 2, base_ch * 4, 2)     # -> 112x128
        self.stage3 = ConvGNReLU(base_ch * 4, base_ch * 6, 2)     # -> 56x64

        self.refine = nn.Sequential(
            ConvGNReLU(base_ch * 6, base_ch * 6, 1),
            ConvGNReLU(base_ch * 6, base_ch * 6, 1),
        )

        self.pool = nn.AdaptiveAvgPool2d((grid_h, grid_w))
        self.head = nn.Conv2d(base_ch * 6, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.refine(x)
        x = self.pool(x)
        logits = self.head(x)   # [B, 1, 7, 8]
        return logits


if __name__ == "__main__":
    x = torch.randn(2, 3, 896, 1024)
    model = LightCoarseROINet(base_ch=16)
    y = model(x)
    print("input :", x.shape)
    print("output:", y.shape)   # 应该是 [2,1,7,8]