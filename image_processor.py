import os
from models import YOLOSeg
import cv2


class ImageProcessor:
    """
    Класс для обработки изображений или видео.
    """

    def __init__(self, model_path: str = 'yolov8n-seg.pt'):
        """
        Args:
            model_path (str): Путь к файлу модели.
        """
        self.model = YOLOSeg(model_path)

    def process_image(self, image_path: str) -> None:
        """
        Обработка изображения и визуализация результатов
        
        Args:
            image_path (str): Путь к изображению
        """
        # Чтение изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        results = self.yolo_segmentation.segment_batch([image])[0]
        self.yolo_segmentation.visualize_results(image, results)

    def process_batch(self, image_paths: list[str]) -> None:
        """
        Обработка батча изображений и визуализация результатов
        
        Args:
            image_paths (list[str]): Список путей к изображениям для сегментации.
        """
        images = [cv2.imread(image_path) for image_path in image_paths]
        if any(image is None for image in images):
            raise ValueError("Некоторые изображения не удалось загрузить.")

        results_batch = self.model.segment_batch(images)

        for image, results in zip(images, results_batch):
            img = self.model.visualize_results(image, results)[:,:,::-1]
            cv2.imshow('frame', img)
            if cv2.waitKey(3000) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def process_video(self, video_path: str) -> None:
        """
        Обработка видео и отображение результатов в реальном времени.
        
        Args:
            video_path (str): Путь к видеофайлу.
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Видео отсутствует по пути: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Выполнение сегментации для текущего кадра
            results = self.model.segment_batch([frame])[0]

            # Визуализация результатов
            img = self.model.visualize_results(frame, results)[:,:,::-1]
            cv2.imshow('frame', img)

            # Прерывание при нажатии клавиши 'q'
            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Пример использования
    processor = ImageProcessor()

    # Обработка изображений
    imgpath = "data"
    images = [
        os.path.join(imgpath, img) for img in os.listdir(imgpath) 
        if img.lower().endswith(('jpg', 'png', 'tiff', 'bmp', 'jpeg'))
    ]
    processor.process_batch(images)

    # Обработка видео
    processor.process_video("data/lp_1.mp4")