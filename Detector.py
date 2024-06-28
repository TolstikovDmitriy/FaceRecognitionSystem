import torch
from facenet_pytorch import MTCNN

from numpy import ndarray


class FaceDetector:
    def __init__(self, weights_path: str = 'weights/mtcnn_weights.pth',
                 min_face_size: int | float = 20, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.min_face_size = min_face_size

        self.model = MTCNN(keep_all=True, device=self.device)
        self.load_weights(weights_path)

    def load_weights(self, weights_path: str) -> None:
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def detect_faces(self, image: ndarray) -> list:
        boxes, _ = self.model.detect(image)

        if boxes is not None:
            filtered_boxes = []

            for box in boxes:
                width = box[2] - box[0]
                height = box[3] - box[1]

                if width >= self.min_face_size and height >= self.min_face_size:
                    box_img = [box[0], box[1], width, height]
                    filtered_boxes.append(box_img)

            boxes = filtered_boxes

        return boxes

    def face_areas(self, image: ndarray) -> list:
        boxes = self.detect_faces(image)
        face_list = []

        for box in boxes:
            x1, y1 = int(box[0]), int(box[1])
            x2, y2 = int(x1 + box[2]), int(y1 + box[3])

            face = image[y1:y2, x1:x2]
            face_list.append(face)

        return face_list
