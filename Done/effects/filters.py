# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║              XR HUD 影像濾鏡引擎 — Filter Engine              ║
║  支援 8 種濾鏡模式 + 亮度/對比度即時調整。                    ║
║  所有濾鏡操作基於 OpenCV 色彩空間轉換與 NumPy 向量運算。      ║
╚══════════════════════════════════════════════════════════════╝
"""
import cv2
import numpy as np

# 將上層目錄加入搜尋路徑以匯入中央設定
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import *


class FilterEngine:
    """
    影像濾鏡引擎。

    提供 8 種濾鏡模式：
      0 - B&W       (黑白)
      1 - NORMAL    (正常 / 無濾鏡)
      2 - VIVID     (鮮豔 / 飽和度增強)
      3 - EDGE      (邊緣偵測 / 科幻線框)
      4 - COOL      (冷色調)
      5 - WARM      (暖色調)
      6 - NEGATIVE  (負片反轉)
      7 - POSTERIZE (海報化 / 色階壓縮)

    額外支援亮度 (-100 ~ +100) 與對比度 (0.5 ~ 2.0) 即時調節。
    """

    def __init__(self):
        """初始化濾鏡引擎，預設模式為 NORMAL (索引 1)。"""
        self.current_mode = 1     # 預設：NORMAL
        self.brightness = 0.0     # 亮度偏移 (-100 ~ +100)
        self.contrast = 1.0       # 對比度倍率 (0.5 ~ 2.0)

        # ── 濾鏡模式對應的處理函式映射表 ──
        # 使用字典映射避免冗長的 if-elif 鏈
        self._filter_dispatch = {
            0: self._apply_bw,         # B&W
            1: self._apply_normal,     # NORMAL
            2: self._apply_vivid,      # VIVID
            3: self._apply_edge,       # EDGE
            4: self._apply_cool,       # COOL
            5: self._apply_warm,       # WARM
            6: self._apply_negative,   # NEGATIVE
            7: self._apply_posterize,  # POSTERIZE
        }

    # ============================================================
    #  模式與參數設定
    # ============================================================
    def set_mode(self, mode_index):
        """
        設定當前濾鏡模式。

        Args:
            mode_index: 模式索引 (0 ~ len(FILTER_MODES)-1)，
                        超出範圍時自動裁剪。
        """
        self.current_mode = max(0, min(mode_index, len(FILTER_MODES) - 1))

    def set_brightness(self, value):
        """
        設定亮度偏移量。

        Args:
            value: 亮度值 (-100 ~ +100)，超出範圍時自動裁剪。
        """
        self.brightness = max(-100.0, min(100.0, float(value)))

    def set_contrast(self, value):
        """
        設定對比度倍率。

        Args:
            value: 對比度值 (0.5 ~ 2.0)，超出範圍時自動裁剪。
        """
        self.contrast = max(0.5, min(2.0, float(value)))

    # ============================================================
    #  主要處理流程
    # ============================================================
    def apply(self, frame):
        """
        對輸入影像幀套用當前濾鏡模式與亮度/對比度調整。

        處理順序：
          1. 套用亮度/對比度 (如果有偏移)
          2. 套用模式特定濾鏡

        Args:
            frame: 輸入影像幀 (BGR, uint8)

        Returns:
            處理後的影像幀 (BGR, uint8)
        """
        result = frame.copy()

        # 步驟 1: 亮度/對比度調整
        # 只在數值偏離預設時才執行，節省運算
        if abs(self.brightness) > 0.5 or abs(self.contrast - 1.0) > 0.01:
            result = self._apply_brightness_contrast(result)

        # 步驟 2: 模式特定濾鏡
        filter_func = self._filter_dispatch.get(self.current_mode,
                                                  self._apply_normal)
        result = filter_func(result)

        return result

    # ============================================================
    #  亮度/對比度調整
    # ============================================================
    def _apply_brightness_contrast(self, frame):
        """
        套用亮度與對比度調整。

        公式: output = clip(frame * contrast + brightness, 0, 255)

        使用 float32 中間運算避免 uint8 溢位。

        Args:
            frame: 輸入影像幀

        Returns:
            調整後的影像幀
        """
        # 轉換為 float32 進行運算，避免 uint8 溢位/下溢
        adjusted = frame.astype(np.float32)

        # 對比度以 128 為中心進行縮放 (保持中灰不變)
        adjusted = (adjusted - 128.0) * self.contrast + 128.0

        # 加上亮度偏移
        adjusted += self.brightness

        # 裁剪並轉回 uint8
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    # ============================================================
    #  濾鏡模式實作
    # ============================================================

    def _apply_bw(self, frame):
        """
        黑白濾鏡 — 將影像轉為灰階後轉回 BGR 格式。

        使用 cv2.COLOR_BGR2GRAY 進行加權灰階轉換
        (0.114*B + 0.587*G + 0.299*R)，比單純取平均更符合人眼感知。
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _apply_normal(self, frame):
        """正常模式 — 不做任何處理，直接返回原始影像。"""
        return frame

    def _apply_vivid(self, frame):
        """
        鮮豔模式 — 在 HSV 色彩空間中提升飽和度至 2.5 倍。

        處理步驟：
          1. BGR → HSV 轉換
          2. 將 S (飽和度) 通道乘以 FILTER_MODES 中定義的倍率
          3. 裁剪至合法範圍 [0, 255]
          4. HSV → BGR 轉回
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

        # 取得此模式的飽和度倍率 (預設 2.5)
        sat_factor = FILTER_MODES[2].get('sat', 2.5)

        # 提升飽和度通道
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _apply_edge(self, frame):
        """
        邊緣偵測濾鏡 — 使用 Canny 演算法提取邊緣，
        以青色線條呈現於深色背景上，營造科幻全息風格。

        處理步驟：
          1. 轉灰階
          2. 高斯模糊降噪
          3. Canny 邊緣偵測 (閾值 50/150)
          4. 將白色邊緣著色為青色
          5. 疊加至深色背景
        """
        # 轉灰階
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 高斯模糊 — 降低雜訊以獲得更乾淨的邊緣
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

        # Canny 邊緣偵測
        edges = cv2.Canny(blurred, 50, 150)

        # 建立深色背景
        result = np.zeros_like(frame)

        # 將邊緣著色為青色 (科幻風格)
        # edges 是單通道遮罩，將其擴展至 3 通道
        edge_mask = edges > 0
        result[edge_mask] = CLR_CYAN  # BGR: (255, 255, 0) 青色

        # 加入輕微的原始影像作為底色 (增添層次感)
        dim_original = (frame * 0.08).astype(np.uint8)
        result = cv2.add(result, dim_original)

        return result

    def _apply_cool(self, frame):
        """
        冷色調濾鏡 — 提升藍色通道，降低紅色通道。

        使用 numpy 直接操作通道值：
          - B 通道: +30 (裁剪至 255)
          - R 通道: -20 (裁剪至 0)
        """
        result = frame.copy().astype(np.int16)

        # 提升藍色通道 (BGR 索引 0)
        result[:, :, 0] = np.clip(result[:, :, 0] + 30, 0, 255)

        # 降低紅色通道 (BGR 索引 2)
        result[:, :, 2] = np.clip(result[:, :, 2] - 20, 0, 255)

        # 輕微提升綠色通道以增添清冷感
        result[:, :, 1] = np.clip(result[:, :, 1] + 10, 0, 255)

        return result.astype(np.uint8)

    def _apply_warm(self, frame):
        """
        暖色調濾鏡 — 提升紅色通道，降低藍色通道。

        使用 numpy 直接操作通道值：
          - R 通道: +30 (裁剪至 255)
          - B 通道: -20 (裁剪至 0)
        """
        result = frame.copy().astype(np.int16)

        # 提升紅色通道 (BGR 索引 2)
        result[:, :, 2] = np.clip(result[:, :, 2] + 30, 0, 255)

        # 降低藍色通道 (BGR 索引 0)
        result[:, :, 0] = np.clip(result[:, :, 0] - 20, 0, 255)

        # 輕微提升綠色通道以增添溫暖感
        result[:, :, 1] = np.clip(result[:, :, 1] + 8, 0, 255)

        return result.astype(np.uint8)

    def _apply_negative(self, frame):
        """
        負片濾鏡 — 反轉所有色彩。

        公式: output = 255 - input
        使用 numpy 廣播實現高效運算。
        """
        return cv2.bitwise_not(frame)

    def _apply_posterize(self, frame):
        """
        海報化濾鏡 — 壓縮色階產生漫畫/海報風格效果。

        將每個通道的 256 個色階量化為約 8 個等級。
        公式: output = round(input / 32) * 32

        同時對邊緣進行強化以增添漫畫線條感。
        """
        # 色階量化 — 每個通道壓縮為 8 級 (256 / 32 = 8)
        levels = 8
        divisor = 256 // levels  # = 32

        # 使用整數除法進行量化 (高效)
        quantized = (frame // divisor) * divisor

        # 加入輕微邊緣強化 — 增添海報風格的黑色輪廓
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 使用自適應二值化提取邊緣
        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=9,
            C=2
        )
        # 將邊緣轉為 3 通道遮罩
        edge_mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # 用邊緣遮罩壓暗量化影像的輪廓
        result = cv2.bitwise_and(quantized, edge_mask)

        return result

    # ============================================================
    #  查詢方法
    # ============================================================
    def get_mode_name(self):
        """
        取得當前模式的顯示名稱。

        Returns:
            str: 模式名稱 (例如 'NORMAL', 'VIVID')
        """
        return FILTER_MODES[self.current_mode]['name']

    def get_mode_icon(self):
        """
        取得當前模式的圖示字元。

        Returns:
            str: 圖示字元 (例如 '◉', '◈')
        """
        return FILTER_MODES[self.current_mode]['icon']

    def get_mode_index(self):
        """
        取得當前模式索引。

        Returns:
            int: 模式索引 (0 ~ 7)
        """
        return self.current_mode

    def get_total_modes(self):
        """
        取得可用濾鏡模式總數。

        Returns:
            int: 模式總數
        """
        return len(FILTER_MODES)

    def next_mode(self):
        """切換至下一個濾鏡模式 (循環)。"""
        self.current_mode = (self.current_mode + 1) % len(FILTER_MODES)

    def prev_mode(self):
        """切換至上一個濾鏡模式 (循環)。"""
        self.current_mode = (self.current_mode - 1) % len(FILTER_MODES)

    def reset(self):
        """重置所有濾鏡參數至預設值。"""
        self.current_mode = 1    # NORMAL
        self.brightness = 0.0
        self.contrast = 1.0

    def get_info(self):
        """
        取得當前濾鏡狀態的摘要資訊。

        Returns:
            dict: 包含模式、亮度、對比度等資訊
        """
        return {
            'mode_index': self.current_mode,
            'mode_name': self.get_mode_name(),
            'mode_icon': self.get_mode_icon(),
            'brightness': self.brightness,
            'contrast': self.contrast,
        }
