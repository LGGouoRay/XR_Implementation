# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║          XR HUD — 主 HUD 面板 (Main HUD Panel)              ║
║  原 SciFiScreen3D 的增強版本：                                ║
║    • 8 種濾鏡模式按鈕（2 行 × 4 列佈局）                      ║
║    • 即時遙測數據（YAW / PITCH / ROLL / DEPTH）               ║
║    • 動態正弦波裝飾動畫                                       ║
║    • 當前模式指示器                                           ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import math
import time

from panels.base_panel import Panel3D, Button3D
from config import *


class MainHUDPanel(Panel3D):
    """
    主 HUD 面板 — 系統的核心控制介面。

    功能：
      - 顯示「X.R. PROJECTION MATRIX v2.0」標題
      - 即時遙測數據（YAW, PITCH, ROLL, DEPTH）
      - 8 個濾鏡模式切換按鈕（2×4 排列）
      - 動態正弦波裝飾
      - 當前濾鏡模式指示器
    """

    def __init__(self):
        super().__init__(
            *MAIN_HUD_SIZE_3D,
            *MAIN_HUD_CANVAS_SIZE,
            MAIN_HUD_INITIAL_POS,
            panel_id="main_hud",
        )

        # ── 濾鏡模式追蹤 ──
        self.current_filter_mode = 1  # 預設 NORMAL（索引 1）

        # ── 遙測數據（由外部更新）──
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.depth = 0.0

        # ── 建立濾鏡模式按鈕（2 行 × 4 列）──
        self._create_filter_buttons()

    # ────────────────────────────────────────────────────────────
    #  _create_filter_buttons — 建立 8 個濾鏡模式按鈕
    # ────────────────────────────────────────────────────────────
    def _create_filter_buttons(self):
        """
        根據 config.FILTER_MODES 建立 8 個按鈕。
        排列方式：2 行 × 4 列，位於畫布下半部。
        """
        btn_w = 90         # 按鈕寬度
        btn_h = 30         # 按鈕高度
        gap_x = 8          # 水平間距
        gap_y = 6          # 垂直間距
        cols = 4            # 每行列數
        rows = 2            # 行數

        # 按鈕區域起始位置（畫布下半部）
        total_w = cols * btn_w + (cols - 1) * gap_x
        start_x = (self.canvas_w - total_w) // 2
        start_y = self.canvas_h - rows * (btn_h + gap_y) - 30

        self.buttons = []
        for idx, mode in enumerate(FILTER_MODES):
            row = idx // cols
            col = idx % cols

            x = start_x + col * (btn_w + gap_x)
            y = start_y + row * (btn_h + gap_y)

            btn = Button3D(
                label=mode["name"],
                rect=(x, y, btn_w, btn_h),
                action_id=f"filter_{idx}",
                icon=mode.get("icon", None),
                toggle=True,
            )

            # 預設啟用 NORMAL 模式
            if idx == self.current_filter_mode:
                btn.is_active = True

            self.buttons.append(btn)

    # ────────────────────────────────────────────────────────────
    #  draw_canvas — 繪製主 HUD 畫布
    # ────────────────────────────────────────────────────────────
    def draw_canvas(self):
        """
        繪製主 HUD 面板的完整畫布。

        內容由上到下：
          1. 科幻基底背景
          2. 標題「X.R. PROJECTION MATRIX v2.0」
          3. 遙測數據區（YAW / PITCH / ROLL / DEPTH）
          4. 動態正弦波裝飾
          5. 8 個濾鏡模式按鈕
          6. 當前模式指示器

        Returns
        -------
        np.ndarray
            畫布影像 (canvas_h, canvas_w, 3)。
        """
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        # ── 科幻基底背景 ──
        self._draw_base_background(canvas)

        # ── 標題 ──
        title = "X.R. PROJECTION MATRIX v2.0"
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = 0.50
        title_thick = 1

        title_size, _ = cv2.getTextSize(title, font, title_scale, title_thick)
        title_x = (self.canvas_w - title_size[0]) // 2
        title_y = 20

        cv2.putText(canvas, title, (title_x, title_y), font, title_scale,
                    CLR_HUD_TITLE, title_thick, cv2.LINE_AA)

        # 標題底線裝飾
        underline_y = title_y + 6
        cv2.line(canvas, (title_x, underline_y),
                 (title_x + title_size[0], underline_y),
                 CLR_CYAN_DIM, 1, cv2.LINE_AA)

        # ── 遙測數據區 ──
        self._draw_telemetry(canvas, y_start=42)

        # ── 動態正弦波裝飾 ──
        self._draw_waveform(canvas, y_center=110, amplitude=12, wave_w=self.canvas_w - 20)

        # ── 濾鏡模式按鈕 ──
        for btn in self.buttons:
            btn.draw(canvas)

        # ── 當前模式指示器 ──
        self._draw_mode_indicator(canvas)

        return canvas

    # ────────────────────────────────────────────────────────────
    #  _draw_telemetry — 繪製遙測數據
    # ────────────────────────────────────────────────────────────
    def _draw_telemetry(self, canvas, y_start=42):
        """
        繪製 4 項遙測數據：YAW, PITCH, ROLL, DEPTH。
        以水平排列方式顯示。

        Parameters
        ----------
        canvas : np.ndarray
            目標畫布。
        y_start : int
            繪製起始 Y 位置。
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_scale = 0.32
        value_scale = 0.42
        thick = 1

        # 遙測項目定義
        items = [
            ("YAW",   f"{self.yaw:+6.1f}", CLR_CYAN),
            ("PITCH", f"{self.pitch:+6.1f}", CLR_CYAN),
            ("ROLL",  f"{self.roll:+6.1f}", CLR_CYAN),
            ("DEPTH", f"{self.depth:6.0f}", CLR_YELLOW_DIM),
        ]

        # 每項的寬度和起始 X 位置
        item_w = self.canvas_w // len(items)
        for i, (label, value, color) in enumerate(items):
            cx = i * item_w + item_w // 2

            # 標籤（上方）
            lbl_size, _ = cv2.getTextSize(label, font, label_scale, thick)
            lbl_x = cx - lbl_size[0] // 2
            cv2.putText(canvas, label, (lbl_x, y_start), font, label_scale,
                        CLR_CYAN_DIM, thick, cv2.LINE_AA)

            # 數值（下方）
            val_size, _ = cv2.getTextSize(value, font, value_scale, thick)
            val_x = cx - val_size[0] // 2
            cv2.putText(canvas, value, (val_x, y_start + 20), font, value_scale,
                        color, thick, cv2.LINE_AA)

        # 分隔線
        sep_y = y_start + 30
        cv2.line(canvas, (10, sep_y), (self.canvas_w - 10, sep_y),
                 CLR_DARK_GRID, 1)

    # ────────────────────────────────────────────────────────────
    #  _draw_waveform — 動態正弦波裝飾
    # ────────────────────────────────────────────────────────────
    def _draw_waveform(self, canvas, y_center=110, amplitude=12, wave_w=420):
        """
        繪製動態正弦波裝飾線。
        使用時間偏移實現流動效果。

        Parameters
        ----------
        canvas : np.ndarray
            目標畫布。
        y_center : int
            波形的垂直中心位置。
        amplitude : int
            波形振幅 (px)。
        wave_w : int
            波形寬度 (px)。
        """
        t = time.time()
        start_x = (self.canvas_w - wave_w) // 2
        num_points = wave_w

        # 生成波形點
        pts = []
        for i in range(num_points):
            x = start_x + i
            # 疊加兩個頻率的正弦波
            phase1 = (i / wave_w) * 4 * math.pi + t * 2.0
            phase2 = (i / wave_w) * 8 * math.pi + t * 3.5
            y = int(y_center + amplitude * math.sin(phase1) * 0.7 +
                    amplitude * math.sin(phase2) * 0.3)
            pts.append([x, y])

        if len(pts) > 1:
            pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_arr], False, CLR_CYAN_DIM, 1, cv2.LINE_AA)

            # 波形高亮中心段
            mid_start = len(pts) // 3
            mid_end = 2 * len(pts) // 3
            if mid_end > mid_start:
                mid_pts = np.array(pts[mid_start:mid_end],
                                   dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [mid_pts], False, CLR_CYAN, 1, cv2.LINE_AA)

    # ────────────────────────────────────────────────────────────
    #  _draw_mode_indicator — 當前模式指示器
    # ────────────────────────────────────────────────────────────
    def _draw_mode_indicator(self, canvas):
        """
        在按鈕區上方顯示當前選中的濾鏡模式名稱。

        Parameters
        ----------
        canvas : np.ndarray
            目標畫布。
        """
        if 0 <= self.current_filter_mode < len(FILTER_MODES):
            mode = FILTER_MODES[self.current_filter_mode]
            mode_text = f"[ {mode['name']} ]"
        else:
            mode_text = "[ --- ]"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.40
        thick = 1

        text_size, _ = cv2.getTextSize(mode_text, font, font_scale, thick)
        tx = (self.canvas_w - text_size[0]) // 2

        # 按鈕區域上方
        # 找到第一個按鈕的 y 位置
        if self.buttons:
            indicator_y = self.buttons[0].rect[1] - 12
        else:
            indicator_y = self.canvas_h - 90

        # 模式名稱文字
        cv2.putText(canvas, mode_text, (tx, indicator_y), font, font_scale,
                    CLR_YELLOW, thick, cv2.LINE_AA)

        # 裝飾點
        dot_y = indicator_y - text_size[1] // 2
        cv2.circle(canvas, (tx - 10, dot_y), 3, CLR_GREEN, -1, cv2.LINE_AA)

    # ────────────────────────────────────────────────────────────
    #  get_current_filter_mode — 取得當前濾鏡模式
    # ────────────────────────────────────────────────────────────
    def get_current_filter_mode(self):
        """
        回傳當前選中的濾鏡模式索引。

        Returns
        -------
        int
            FILTER_MODES 列表中的索引。
        """
        return self.current_filter_mode

    # ────────────────────────────────────────────────────────────
    #  handle_button_press — 處理按鈕按壓事件
    # ────────────────────────────────────────────────────────────
    def handle_button_press(self, button_id):
        """
        處理按鈕按壓回呼。

        Parameters
        ----------
        button_id : str
            按鈕的 action_id，格式為 'filter_{idx}'。
        """
        if not button_id.startswith("filter_"):
            return

        try:
            idx = int(button_id.split("_")[1])
        except (IndexError, ValueError):
            return

        if 0 <= idx < len(FILTER_MODES):
            # 更新當前模式
            old_mode = self.current_filter_mode
            self.current_filter_mode = idx

            # 更新按鈕啟用狀態（互斥切換）
            for i, btn in enumerate(self.buttons):
                btn.is_active = (i == idx)

            self.mark_dirty()

    # ────────────────────────────────────────────────────────────
    #  update_telemetry — 更新遙測數據
    # ────────────────────────────────────────────────────────────
    def update_telemetry(self, yaw=None, pitch=None, roll=None, depth=None):
        """
        更新遙測顯示數據。

        Parameters
        ----------
        yaw : float or None
            偏航角。
        pitch : float or None
            俯仰角。
        roll : float or None
            滾轉角。
        depth : float or None
            深度值 (mm)。
        """
        if yaw is not None:
            self.yaw = yaw
        if pitch is not None:
            self.pitch = pitch
        if roll is not None:
            self.roll = roll
        if depth is not None:
            self.depth = depth
        self.mark_dirty()
