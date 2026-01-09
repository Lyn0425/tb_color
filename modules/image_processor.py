# modules/image_processor.py

import cv2
import numpy as np
import logging
from typing import Dict, Tuple, Any
from functools import lru_cache
from .config import COLOR_SCHEMES
from PIL import Image

logger = logging.getLogger(__name__)

class MedicalImageProcessor:
    """医学图像处理核心类"""
    
    # 缓存CLAHE对象，避免重复创建
    _clahe_cache = {}
    
    @staticmethod
    def _get_clahe(clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> cv2.CLAHE:
        """获取或创建CLAHE对象"""
        cache_key = (clip_limit, tile_grid_size)
        if cache_key not in MedicalImageProcessor._clahe_cache:
            MedicalImageProcessor._clahe_cache[cache_key] = cv2.createCLAHE(
                clipLimit=clip_limit, 
                tileGridSize=tile_grid_size
            )
        return MedicalImageProcessor._clahe_cache[cache_key]
    
    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """将图像转换为灰度图（辅助方法）"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy() if image.dtype != np.uint8 else image
    
    @staticmethod
    def preprocess_image(image: np.ndarray, apply_clahe: bool = True, 
                         contrast: float = 1.0, brightness: int = 0) -> np.ndarray:
        """图像预处理"""
        try:
            gray = MedicalImageProcessor._to_gray(image)
            
            # 使用缓存的CLAHE对象
            if apply_clahe:
                clahe = MedicalImageProcessor._get_clahe(clip_limit=2.0, tile_grid_size=(8, 8))
                gray = clahe.apply(gray)
            
            # 调整对比度和亮度
            gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)
            
            logger.debug(f"图像预处理完成，尺寸: {gray.shape}, apply_clahe: {apply_clahe}, contrast: {contrast}, brightness: {brightness}")
            return gray
        except cv2.error as e:
            logger.error(f"图像预处理失败: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"图像预处理时发生未知错误: {e}", exc_info=True)
            raise
    
    @staticmethod
    def enhance_pseudocolor(image: np.ndarray, color_scheme: str = "标准") -> np.ndarray:
        """胸片灰度分层伪彩色增强"""
        try:
            gray = MedicalImageProcessor._to_gray(image)
            
            color_img = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
            layers = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES["标准"])
            
            for min_val, max_val, color in layers:
                mask = (gray >= min_val) & (gray < max_val)
                color_img[mask] = color
            
            logger.debug(f"伪彩色增强完成，尺寸: {color_img.shape}, 颜色方案: {color_scheme}")
            return color_img
        except cv2.error as e:
            logger.error(f"伪彩色增强失败: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"伪彩色增强时发生未知错误: {e}", exc_info=True)
            raise
    
    @staticmethod
    def calculate_image_stats(image: np.ndarray) -> Dict[str, float]:
        """计算图像统计信息"""
        try:
            gray = MedicalImageProcessor._to_gray(image)
            
            stats = {
                "min": float(np.min(gray)),
                "max": float(np.max(gray)),
                "mean": float(np.mean(gray)),
                "std": float(np.std(gray)),
                "width": float(gray.shape[1]),
                "height": float(gray.shape[0]),
                "median": float(np.median(gray))
            }
            
            logger.debug(f"图像统计计算完成，尺寸: {gray.shape}")
            return stats
        except cv2.error as e:
            logger.error(f"图像统计计算失败: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"图像统计计算时发生未知错误: {e}", exc_info=True)
            raise
    
    @staticmethod
    def convert_to_pil(image: np.ndarray) -> 'Image':
        try:
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
            logger.debug(f"图像转换为PIL格式完成，尺寸: {image.shape}")
            return Image.fromarray(rgb_image)
        except cv2.error as e:
            logger.error(f"图像转换为PIL格式失败: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"图像转换为PIL格式时发生未知错误: {e}", exc_info=True)
            raise
    
    @staticmethod
    def resize_image(image: np.ndarray, max_size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
        """调整图像大小（保持宽高比）"""
        try:
            height, width = image.shape[:2]
            
            if width > max_size[0] or height > max_size[1]:
                scale = min(max_size[0] / width, max_size[1] / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                logger.debug(f"图像调整大小完成，原尺寸: ({width}, {height}), 新尺寸: ({new_width}, {new_height})")
            else:
                logger.debug(f"图像尺寸无需调整，当前尺寸: ({width}, {height})")
            
            return image
        except cv2.error as e:
            logger.error(f"图像调整大小失败: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"图像调整大小时发生未知错误: {e}", exc_info=True)
            raise

    @staticmethod
    def generate_legend(color_scheme: str = "标准") -> np.ndarray:
        try:
            layers = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES["标准"])
            legend = np.zeros((40, 256, 3), dtype=np.uint8)
            for min_val, max_val, color in layers:
                legend[:, min_val:max_val] = color
            
            logger.debug(f"图例生成完成，颜色方案: {color_scheme}")
            return legend
        except cv2.error as e:
            logger.error(f"图例生成失败: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"图例生成时发生未知错误: {e}", exc_info=True)
            raise

    @staticmethod
    def compute_histogram(image: np.ndarray) -> np.ndarray:
        try:
            gray = MedicalImageProcessor._to_gray(image)
            counts = np.bincount(gray.flatten(), minlength=256)
            
            logger.debug(f"直方图计算完成，尺寸: {gray.shape}")
            return counts
        except cv2.error as e:
            logger.error(f"直方图计算失败: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"直方图计算时发生未知错误: {e}", exc_info=True)
            raise
