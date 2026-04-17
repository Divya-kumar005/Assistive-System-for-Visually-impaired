import torch
import torchvision.models as models

class EncoderCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, images):
        features = self.feature_extractor(images)
        return features.view(features.size(0), -1)