from FaceRecognition import FaceRecognition
from Normalizer import SimpleCNNModel
import cv2

img_path1 = 'imgs/johny_depp_1.jpg'
img_path2 = 'imgs/johny_depp_2.jpg'

img1 = cv2.imread(img_path1)
img2 = cv2.imread(img_path2)

similarity = FaceRecognition().similarity_score(img1, img2)

print(similarity)
