from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 5),  # input_size:[3,32,32]     out_size: [16,28,28]
            nn.Sigmoid(),
            nn.AvgPool2d(2),  # input_size: [16,28,28]   out_size: [16,14,14]
            nn.Conv2d(16, 32, 5),  # input_size: [16,14,14]    out_size: [32,10,10]
            nn.Sigmoid(),
            nn.AvgPool2d(2),  # input_size: [32,10,10]   out_size: [32,5,5]
            nn.Flatten(),  # 矩阵展开
            nn.Linear(32 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x
