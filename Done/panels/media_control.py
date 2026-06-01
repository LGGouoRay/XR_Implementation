# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║          XR HUD — 媒體控制面板 (Media Control Panel)          ║
║  影像調整與濾鏡控制：                                         ║
║    • 亮度滑桿 (-100 ~ +100)                                  ║
║    • 對比度滑桿 (0.5 ~ 2.0)                                  ║
║    • 濾鏡模式顯示                                             ║
║    • +/- 調整按鈕                                             ║
║    • 重設按鈕                                                  ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import math
import time

from panels.base_panel import Panel3D, Button3D
from config import *


class MediaControlPanel(Panel3D):
    """
    媒體控制面板 — 影像參數調整介面。

    功能：
      - 亮度調整（-100 ~ +100）
      - 對比度調整（0.5 ~ 2.0）
      - 當前濾鏡模式名稱顯示
      - +/- 微調按鈕
      - RESET 重設按鈕
    """

    def __init__(self):
        super().__init__(
            *MEDIA_SIZE_3D,
            *MEDIA_CANVAS_SIZE,
            MEDIA_INITIAL_POS,
            panel_id="media_ctrl",
        )

        # ── 影像參數 ──
        self.brightness = 0.0       # 亮度偏移 [-100, +100]
        self.contrast = 1.0         # 對比度倍率 [0.5, 2.0]
        self.current_filter = 1     # 當前濾鏡模式索引（對應 FILTER_MODES）

        # ── 建立控制按鈕 ──
        self._create_buttons()

    # ────────────────────────────────────────────────────────────
    #  _create_buttons — 建立調整按鈕
    # ────────────────────────────────────────────────────────────
    def _create_buttons(self):
        """
        建立亮度 +/-、對比度 +/-、重設共 5 個按鈕。
        """
        btn_w = 36
        btn_h = 24
        margin_x = 12

        # ── 亮度 +/- 按鈕位置（在滑桿右側）──
        bri_btn_y = 68
        slider_end_x = self.canvas_w - 50

        btn_bri_minus = Button3D(
            label="-",
            rect=(slider_end_x, bri_btn_y, btn_w, btn_h),
            action_id="bri_minus",
        )
        btn_bri_plus = Button3D(
            label="+",
            rect=(slider_end_x + btn_w + 4, bri_btn_y, btn_w, btn_h),
            action_id="bri_plus",
        )

        # ── 對比度 +/- 按鈕位置 ──
        con_btn_y = 112
        btn_con_minus = Button3D(
            label="-",
            rect=(slider_end_x, con_btn_y, btn_w, btn_h),
            action_id="con_minus",
        )
        btn_con_plus = Button3D(
            label="+",
            rect=(slider_end_x + btn_w + 4, con_btn_y, btn_w, btn_h),
            action_id="con_plus",
        )

        # ── RESET 按鈕（底部中央）──
        reset_w = 80
        reset_h = 26
        reset_x = (self.canvas_w - reset_w) // 2
        reset_y = self.canvas_h - reset_h - 12

        btn_reset = Button3D(
            label="RESET",
            rect=(reset_x, reset_y, reset_w, reset_h),
            action_id="reset",
            icon=None,
            toggle=False,
        )

        self.buttons = [
            btn_bri_minus, btn_bri_plus,
            btn_con_minus, btn_con_plus,
            btn_reset,
        ]

    # ────────────────────────────────────────────────────────────
    #  draw_canvas — 繪製媒體控制畫布
    # ────────────────────────────────────────────────────────────
    def draw_canvas(self):
        """
        繪製媒體控制面板的完整畫布。

        佈局：
          1. 標題「MEDIA CONTROL」
          2. 亮度滑桿 + 數值標籤 + +/- 按鈕
          3. 對比度滑桿 + 數值標籤 + +/- 按鈕
          4. 當前濾鏡模式名稱
          5. RESET 重設按鈕

        Returns
        -------
        np.ndarray
            畫布影像 (canvas_h, canvas_w, 3)。
        """
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        # ── 科幻基底背景 ──
        self._draw_base_background(canvas)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # ── 標題 ──
        title = "MEDIA CONTROL"
        title_scale = 0.45
        title_thick = 1
        title_size, _ = cv2.getTextSize(title, font, title_scale, title_thick)
        title_x = (self.canvas_w - title_size[0]) // 2
        cv2.putText(canvas, title, (title_x, 20), font, title_scale,
                    CLR_HUD_TITLE, title_thick, cv2.LINE_AA)

        # ── 亮度滑桿 ──
        self._draw_slider(
            canvas,
            label="BRIGHTNESS",
            value=self.brightness,
            min_val=-100.0,
            max_val=100.0,
            y=55,
            color=CLR_YELLOW,
            fmt="{:+.0f}",
        )

        # ── 對比度滑桿 ──
        self._draw_slider(
            canvas,
            label="CONTRAST",
            value=self.contrast,
            min_val=0.5,
            max_val=2.0,
            y=100,
            color=CLR_CYAN,
            fmt="{:.2f}",
        )

        # ── 當前濾鏡模式顯示 ──
        self._draw_filter_display(canvas, y=142)

        # ── 所有按鈕 ──
        for btn in self.buttons:
            btn.draw(canvas)

        return canvas

    # ────────────────────────────────────────────────────────────
    #  _draw_slider — 繪製滑桿
    # ────────────────────────────────────────────────────────────
    def _draw_slider(self, canvas, label, value, min_val, max_val, y, color, fmt="{:.1f}"):
        """
        繪製帶標籤和數值的水平滑桿。

        Parameters
        ----------
        canvas : np.ndarray
            目標畫布。
        label : str
            滑桿標籤。
        value : float
            當前值。
        min_val : float
            最小值。
        max_val : float
            最大值。
        y : int
            滑桿 Y 位置。
        color : tuple
            滑桿填充 BGR 顏色。
        fmt : str
            數值格式化字串。
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        margin = 12
        slider_x = margin
        slider_w = self.canvas_w - 120   # 為右側按鈕留空間
        slider_h = 8

        # ── 標籤文字 ──
        cv2.putText(canvas, label, (slider_x, y), font, 0.30,
                    CLR_CYAN_DIM, 1, cv2.LINE_AA)

        # ── 數值文字（標籤右側）──
        value_text = fmt.format(value)
        val_size, _ = cv2.getTextSize(value_text, font, 0.32, 1)
        val_x = slider_x + slider_w - val_size[0]
        cv2.putText(canvas, value_text, (val_x, y), font, 0.32,
                    CLR_HUD_DATA, 1, cv2.LINE_AA)

        # ── 滑桿軌道 ──
        track_y = y + 6
        cv2.rectangle(canvas,
                      (slider_x, track_y),
                      (slider_x + slider_w, track_y + slider_h),
                      CLR_DARK_OVERLAY, -1)
        cv2.rectangle(canvas,
                      (slider_x, track_y),
                      (slider_x + slider_w, track_y + slider_h),
                      CLR_CYAN_DIM, 1)

        # ── 計算填充比例 ──
        val_range = max_val - min_val
        if val_range > 0:
            ratio = (value - min_val) / val_range
        else:
            ratio = 0.0
        ratio = min(max(ratio, 0.0), 1.0)

        fill_w = int(ratio * slider_w)

        # ── 填充條 ──
        if fill_w > 1:
            cv2.rectangle(canvas,
                          (slider_x + 1, track_y + 1),
                          (slider_x + fill_w - 1, track_y + slider_h - 1),
                          color, -1)

        # ── 滑桿把手（小方塊）──
        handle_x = slider_x + fill_w
        handle_w = 4
        handle_h_ext = 3  # 上下延伸
        cv2.rectangle(canvas,
                      (handle_x - handle_w // 2, track_y - handle_h_ext),
                      (handle_x + handle_w // 2, track_y + slider_h + handle_h_ext),
                      CLR_WHITE, -1)

        # ── 中線標記（代表預設值位置）──
        if min_val < 0 < max_val:
            # 零值位置
            zero_ratio = (0.0 - min_val) / val_range
            zero_x = slider_x + int(zero_ratio * slider_w)
            cv2.line(canvas, (zero_x, track_y - 2), (zero_x, track_y + slider_h + 2),
                     CLR_CYAN_DIM, 1)

    # ────────────────────────────────────────────────────────────
    #  _draw_filter_display — 當前濾鏡模式顯示
    # ────────────────────────────────────────────────────────────
    def _draw_filter_display(self, canvas, y):
        """
        顯示當前選中的濾鏡模式名稱。

        Parameters
        ----------
        canvas : np.ndarray
            目標畫布。
        y : int
            顯示 Y 位置。
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 濾鏡標籤
        cv2.putText(canvas, "FILTER:", (12, y), font, 0.30,
                    CLR_CYAN_DIM, 1, cv2.LINE_AA)

        # 模式名稱
        if 0 <= self.current_filter < len(FILTER_MODES):
            mode = FILTER_MODES[self.current_filter]
            mode_text = f"{mode['icon']} {mode['name']}"
        else:
            mode_text = "---"

        cv2.putText(canvas, mode_text, (75, y), font, 0.38,
                    CLR_YELLOW, 1, cv2.LINE_AA)

        # 分隔線
        sep_y = y + 8
        cv2.line(canvas, (12, sep_y), (self.canvas_w - 12, sep_y),
                 CLR_DARK_GRID, 1)

    # ────────────────────────────────────────────────────────────
    #  adjust_brightness — 調整亮度
    # ────────────────────────────────────────────────────────────
    def adjust_brightness(self, delta):
        """
        調整亮度值。

        Parameters
        ----------
        delta : float
            亮度變化量（正值增亮，負值減暗）。
            最終值限制在 [-100, +100] 範圍內。
        """
        self.brightness = max(-100.0, min(100.0, self.brightness + delta))
        self.mark_dirty()

    # ────────────────────────────────────────────────────────────
    #  adjust_contrast — 調整對比度
    # ────────────────────────────────────────────────────────────
    def adjust_contrast(self, delta):
        """
        調整對比度值。

        Parameters
        ----------
        delta : float
            對比度變化量。
            最終值限制在 [0.5, 2.0] 範圍內。
        """
        self.contrast = max(0.5, min(2.0, self.contrast + delta))
        self.mark_dirty()

    # ────────────────────────────────────────────────────────────
    #  reset — 重設為預設值
    # ────────────────────────────────────────────────────────────
    def reset(self):
        """將亮度和對比度重設為預設值。"""
        self.brightness = 0.0
        self.contrast = 1.0
        self.mark_dirty()

    # ────────────────────────────────────────────────────────────
    #  handle_button_press — 處理按鈕按壓事件
    # ────────────────────────────────────────────────────────────
    def handle_button_press(self, button_id):
        """
        處理按鈕按壓回呼。

        Parameters
        ----------
        button_id : str
            按鈕的 action_id。
        """
        if button_id == "bri_plus":
            self.adjust_brightness(5.0)
        elif button_id == "bri_minus":
            self.adjust_brightness(-5.0)
        elif button_id == "con_plus":
            self.adjust_contrast(0.1)
        elif button_id == "con_minus":
            self.adjust_contrast(-0.1)
        elif button_id == "reset":
            self.reset()

    # ────────────────────────────────────────────────────────────
    #  set_filter — 設定濾鏡模式
    # ────────────────────────────────────────────────────────────
    def set_filter(self, filter_index):
        """
        設定當前濾鏡模式。

        Parameters
        ----------
        filter_index : int
            FILTER_MODES 列表索引。
        """
        if 0 <= filter_index < len(FILTER_MODES):
            self.current_filter = filter_index
            self.mark_dirty()

    # ────────────────────────────────────────────────────────────
    #  get_adjustments — 取得當前調整值
    # ────────────────────────────────────────────────────────────
    def get_adjustments(self):
        """
        回傳當前亮度和對比度調整值。

        Returns
        -------
        dict
            {"brightness": float, "contrast": float, "filter_index": int}
        """
        return {
            "brightness": self.brightness,
            "contrast": self.contrast,
            "filter_index": self.current_filter,
        }
