
import cv2


class HaarcascadeDetector:
    def __init__(self, pretrain_path):
        self.detector = cv2.CascadeClassifier(pretrain_path)

    def detect_faces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray)
        return faces
