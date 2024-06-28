from Detector import FaceDetector
from Normalizer import Normalizer
from Extractor import Extractor

import torch

import numpy as np
from numpy import ndarray

from PIL import Image


class FaceRecognition:
    def face_preprocessing(self, image: ndarray) -> list[ndarray]:
        face_list = FaceDetector().face_areas(image)
        norm_face_list = []

        for face in face_list:
            norm_face = Normalizer().normalize_img(face)
            norm_face[:, :, [0, 1, 2]] = norm_face[:, :, [2, 1, 0]]
            norm_face = norm_face.astype(np.uint8)

            norm_face_list.append(norm_face)

        return norm_face_list

    def face_extraction(self, image: ndarray) -> ndarray:
        norm_face_list = self.face_preprocessing(image)

        norm_img = norm_face_list[0]
        norm_img = Image.fromarray(norm_img, 'RGB')

        feature = Extractor().get_features(norm_img)

        return feature

    def similarity_score(self, image1: ndarray, image2: ndarray) -> float:
        features1, features2 = self.face_extraction(image1), self.face_extraction(image2)

        euclidean_dist = torch.norm(features1 - features2, p=2)
        similarity = round(float(torch.exp(-0.5 * euclidean_dist)), 2)

        return similarity
