import torch
import cv2
import numpy as np

class YOLOv5Detector:
    def __init__(self, model_path):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    def detect(self, image):
        results = self.model(image)
        return results

def process_image(file_path):
    img = cv2.imread(file_path)  # Загрузка изображения
    return img

def main(image_path):
    detector = YOLOv5Detector('best.pt')  # Укажите путь к вашим весам
    img = process_image(image_path)
    results = detector.detect(img)

    # Вывод результатов
    results.print()  # Печатает результаты в консоль
    results.show()  # Показывает изображение с детекцией
    results.save()  # Сохраняет изображение с детекцией

if __name__ == "__main__":
    main('img.png')  # Укажите путь к изображению
