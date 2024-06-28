import math
from numpy import ndarray

import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.models import resnet50


class SimpleCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, 68*2)

    def forward(self, x):
        output = self.model(x)
        return output


class Normalizer:
    def __init__(self, model_path: str = 'models/model_normalizer.pth'):
        self.model = torch.load(model_path)

    def normalize_img(self, image: ndarray) -> ndarray:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((220, 220))
        ])

        image = torch.unsqueeze(transform(image) * 255, 0)

        keypoints = self.model(image)
        keypoints = keypoints.detach().numpy().reshape(-1, 2)

        '''
        Точки под этими индексами будут использоваться для поворота лица.
        Точка под индексом:
        - 36 - уголок правого глаза
        - 45 - уголок левого глаза
        '''

        num_keypoints = [36, 45]
        x, y = keypoints[num_keypoints, 0], keypoints[num_keypoints, 1]

        # Вычисляем разность координат между уголками глаз, а затем угл поворота
        delta_x = abs(x[0] - x[1])
        delta_y = y[0] - y[1]
        angle = math.degrees(math.acos(delta_x / math.sqrt(delta_x ** 2 + delta_y ** 2)))

        # Определяем в какую сторону будет поворот
        is_right_turn = y[0] > y[1]
        if is_right_turn:
            angle = -angle

        img = transforms.functional.rotate(img=image, angle=angle)
        img = img[0].permute(1, 2, 0).int()
        img = img.numpy()

        return img
