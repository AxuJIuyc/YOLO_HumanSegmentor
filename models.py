import cv2
from ultralytics import YOLO
import numpy as np
import torch

class YOLOSeg:
    """
    Сегментация людей YOLOv8.
    
    Attributes:
        model_path (str): Путь к предобученной модели YOLO.
        model (YOLO): vодель YOLO для сегментации.
    """

    def __init__(self, model_path: str = 'yolov8n-seg.pt'):
        """
        Инициализация модели YOLO.
        
        Args:
            model_path (str): Путь к файлу модели. 
                Default: 'yolov8n-seg.pt'.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        self.ns = 640 # normal size
        self.aim_classes = [0] # person

    def preprocess_batch(self, images: list[np.ndarray]) -> torch.Tensor:
        """
        Подготавливает батч изображений для обработки (преобразует в тензор).
        
        Args:
            images (list[np.ndarray]): Список изображений в формате numpy (BGR).
            
        Returns:
            torch.Tensor: Тензор с изображениями.
        """
        # Изменение размера всех изображений и их нормализация
        resized_images = [cv2.resize(img, (self.ns, self.ns)) for img in images]
        # Преобразование в тензор, нормализация и перемещение на устройство (GPU или CPU)
        tensor_images = torch.stack([torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 for img in resized_images])
        return tensor_images.to(self.device)
    
    def postprocess_results(self, results, original_sizes: list[tuple]) -> list[dict]:
        """
        Постобработка результатов: изменение размера масок и боксов до исходных размеров изображений.
        
        Args:
            results (list): Сырые результаты модели.
            original_sizes (list[tuple]): Список оригинальных размеров изображений.
            
        Returns:
            list[dict]: Список обработанных результатов (маски, боксы, классы и вероятности).
        """
        processed_results = []

        for result, (orig_width, orig_height) in zip(results, original_sizes):
            boxes = result.boxes.data.cpu()
            if len(boxes) == 0:
                continue
            masks = result.masks.data.cpu().numpy()
            scores = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.tolist()

            # Масштабирование масок до оригинального размера изображения
                # masks = np.array([cv2.resize(mask, (orig_width, orig_height)) for mask in masks])
            masks = [cv2.resize(mask, (orig_width, orig_height)) for mask in masks]

            # Масштабирование боксов до оригинального размера изображения
            if boxes is not None:
                boxes = [
                    [
                        int(box[0] * orig_width / self.ns),  # x1
                        int(box[1] * orig_height / self.ns),  # y1
                        int(box[2] * orig_width / self.ns),  # x2
                        int(box[3] * orig_height / self.ns),  # y2
                        *box[4:]
                    ] for box in boxes
                ]

            processed_results.append({
                "masks": masks,
                "boxes": boxes,
                "scores": scores,
                "classes": classes
            })

        return processed_results    

    def cls_filter(self, results: list[dict]) -> list[dict]:
        """
        Оставляет только целевые классы

        Args:
            results (list[dict]): Список обработанных результатов (маски, боксы, классы и вероятности).
        Returns:
            list[dict]: Список обработанных результатов (маски, боксы, классы и вероятности).
        """
        for j, result in enumerate(results):
            deletes = []
            for i, cls in enumerate(result["classes"]):
                if cls not in self.aim_classes:
                    deletes.append(i)

            for key, values in result.items():
                for i in deletes[::-1]:
                    results[j][key].pop(i)
        return results

    def segment_batch(self, images: list[np.ndarray]) -> list[dict]:
        """
        Выполняет сегментацию для батча изображений.
        
        Args:
            images (list[np.ndarray]): Список изображений в формате numpy (BGR).
            
        Returns:
            list[dict]: Список с результатами сегментации для каждого изображения.
        """
        # Препроцессинг: преобразование батча изображений в тензор
        original_sizes = [(img.shape[1], img.shape[0]) for img in images]  # (width, height)
        batch_tensor = self.preprocess_batch(images)

        # Одновременная обработка батча изображений
        results = self.model(batch_tensor)

        # Постобработка результатов для приведения к исходным размерам изображений
        processed_results = self.postprocess_results(results, original_sizes)
        processed_results = self.cls_filter(processed_results)

        return processed_results

    def visualize_results(self, image: np.ndarray, results: dict) -> np.ndarray:
        """
        Отображение результатов сегментации.
        
        Args:
            image (np.ndarray): Исходное изображение.
            results (dict): Результаты сегментации (маски, боксы, классы и вероятности).

        Reurns:
            np.ndarray: Изображение с отрисованными детекцями
        """
        masks = results.get("masks")
        boxes = results.get("boxes")
        scores = results.get("scores")
        classes = results.get("classes")

        # Преобразуем изображение в формат RGB для визуализации
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if masks is not None:
            for mask in masks:
                # Создаем маску с прозрачностью только для областей, где маска > 0.5
                alpha = 0.5
                mask_area = mask > 0.5  # Области, где активна маска
                # Копируем оригинальное изображение, чтобы не изменять его полностью
                img_copy = img_rgb.copy()                
                # Красим только активные области маски в красный цвет
                img_copy[mask_area] = [255, 0, 0]  # Красный цвет для маски                
                # Накладываем полупрозрачную маску на изображение
                img_rgb[mask_area] = cv2.addWeighted(img_copy, alpha, img_rgb, 1 - alpha, 0)[mask_area]

        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                label = f"{classes[int(box[5])]} {scores[i]:.2f}"
                img_rgb = cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                img_rgb = cv2.putText(img_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img_rgb