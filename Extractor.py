import torch
from torch import tensor
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision import transforms

from PIL import Image


class Extractor:
    def __init__(self, weights_path: str = 'weights/model_weights.pth', device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 438)

        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

    def get_features(self, image: Image) -> tensor:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image).unsqueeze(0)
        features = self.model(image)

        return features
