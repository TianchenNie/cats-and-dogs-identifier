
import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
from data_utils import DataLoader
import numpy as np

class CatOrDog(nn.Module):
    def __init__(self):
        super(CatOrDog, self).__init__()
        self.name = "TunTunAI"
        alexnet = torchvision.models.alexnet(pretrained=True)
        self.feature_extractor = alexnet.features
        self.model = nn.Sequential(OrderedDict([
            ("fully_connected_1", nn.Linear(6*6*256, 2048)),
            ("relu1", nn.ReLU6()),
            ("fully_connected_2", nn.Linear(2048, 128)),
            ("relu2", nn.ReLU6()),
            ("fully_connected_3", nn.Linear(128, 32)),
            ("relu3", nn.ReLU6()),
            ('full_connected_4', nn.Linear(32,1))
        ]))

    def forward(self, images):
        features = self.feature_extractor(images)
        features = features.view(-1, 256*6*6)
        prediction_values = self.model(features)
        return prediction_values.reshape(prediction_values.shape[0],-1)



if __name__ == "__main__":
    model = CatOrDog()
    data_loader = DataLoader()
    data_loader.load()
    for i in data_loader.data:
        images, labels = i
        output = model(images)
        print(output.shape)
        # print(output.squeeze().long() != labels)
        # print(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]).long())
        break

