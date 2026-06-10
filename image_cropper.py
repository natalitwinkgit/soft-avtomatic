"""
Автоматичне обрізання зображень з детектуванням меж контенту.
Реалізація функцій для пошуку фактичних меж зображення та його обрізання.
"""

from typing import Tuple, Optional
import numpy as np
from PIL import Image
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageCropper:
    """
    Клас для автоматичного обрізання зображень з пошуком реальних меж контенту.
    """

    def __init__(self, pixel_tolerance: int = 0):
        """
        Ініціалізація обрізувача зображень.
        
        Args:
            pixel_tolerance: Допуск при пошуку непрозорих пікселів (за замовчуванням 0)
        """
        self.pixel_tolerance = pixel_tolerance

    def find_content_bounds(self, image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """
        Пошук фактичних меж контенту на зображенні.
        
        Аналізує всі непрозорі пікселі та визначає крайні координати.
        
        Args:
            image: PIL Image об'єкт
            
        Returns:
            Кортеж (лева, верхня, права, нижня) або None якщо контенту не знайдено
        """
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        img_array = np.array(image)
        alpha_channel = img_array[:, :, 3]
        
        non_transparent = alpha_channel > self.pixel_tolerance
        
        if not np.any(non_transparent):
            logger.warning("Не знайдено непрозорих пікселів на зображенні")
            return None
        
        rows = np.any(non_transparent, axis=1)
        cols = np.any(non_transparent, axis=0)
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return (int(x_min), int(y_min), int(x_max) + 1, int(y_max) + 1)

    def crop_image(self, image: Image.Image, bounds: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
        """
        Обрізання зображення за вказаними межами.
        
        Args:
            image: PIL Image об'єкт
            bounds: Кортеж (лева, верхня, права, нижня) або None для автоматичного пошуку
            
        Returns:
            Обрізане зображення
        """
        if bounds is None:
            bounds = self.find_content_bounds(image)
        
        if bounds is None:
            logger.warning("Не вдалося визначити межи. Повертаємо оригінальне зображення")
            return image
        
        cropped = image.crop(bounds)
        logger.info(f"Зображення обрізано до розміру: {cropped.size}")
        
        return cropped

    def export_pixel_perfect(self, image: Image.Image, output_path: str, crop: bool = True) -> None:
        """
        Експорт зображення в PNG з режимом Pixel Perfect.
        
        Args:
            image: PIL Image об'єкт для експорту
            output_path: Шлях для збереження файлу
            crop: Чи обрізати зображення перед експортом
        """
        if image.mode != 'RGBA':
            export_image = image.convert('RGBA')
        else:
            export_image = image.copy()
        
        if crop:
            bounds = self.find_content_bounds(export_image)
            if bounds is not None:
                export_image = export_image.crop(bounds)
        
        export_image.save(output_path, 'PNG', optimize=False)
        logger.info(f"Зображення експортовано в: {output_path}")
