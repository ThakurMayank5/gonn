import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

model = MLP()
dummy = torch.randn(1,4)

torch.onnx.export(model, dummy, "mlp.onnx")