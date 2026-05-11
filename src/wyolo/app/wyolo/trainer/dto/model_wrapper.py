import torch


class MLflowYOLOModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def predict(self, data):
        return self.model(data)
