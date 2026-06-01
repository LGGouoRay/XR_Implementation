# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║          XR HUD — 系統監控面板 (System Monitor Panel)         ║
║  即時系統資源監控：                                            ║
║    • FPS 滾動折線圖                                           ║
║    • CPU / 記憶體圓弧儀表盤                                    ║
║    • 追蹤信心度條                                              ║
║    • 幀計數器                                                  ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import math
import time
from collections import deque

from panels.base_panel import Panel3D
from config import *

# ── 選用 psutil（若未安裝則降級）──
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class SystemMonitorPanel(Panel3D):
    """
    系統監控面板 — 顯示即時效能指標。

    功能：
      - FPS 滾動折線圖（最近 120 幀）
      - CPU 使用率圓弧儀表盤
      - 記憶體使用率圓弧儀表盤
      - 追蹤信心度進度條
      - 幀計數器
    """

    def __init__(self):
        super().__init__(
            *SYSMON_SIZE_3D,
            *SYSMON_CANVAS_SIZE,
            SYSMON_INITIAL_POS,
            panel_id="sys_monitor",
        )

        # ── FPS 歷史紀錄 ──
        self.fps_history = deque(maxlen=FPS_HISTORY_SIZE)  # 最近 120 幀

        # ── 系統資源數據 ──
        self.cpu_usage = 0.0          # CPU 使用率 (0-100)
        self.mem_usage = 0.0          # 記憶體使用率 (0-100)
        self.tracking_confidence = 0.0  # 追蹤信心度 (0-1)
        self.frame_count = 0          # 累計幀數

        # ── 更新頻率控制 ──
        self.last_update_time = time.time()
        self.last_sys_update = 0      # 每 30 幀才查詢一次 psutil
        self.has_psutil = _HAS_PSUTIL

        # ── 初始化 FPS 歷史為 0 ──
        for _ in range(FPS_HISTORY_SIZE):
            self.fps_history.append(0.0)

    # ────────────────────────────────────────────────────────────
    #  update_stats — 更新效能統計（每幀呼叫）
    # ────────────────────────────────────────────────────────────
    def update_stats(self, fps, tracking_conf=0.0):
        """
        更新效能統計數據。應在每幀被呼叫。

        Parameters
        ----------
        fps : float
            當前幀率。
        tracking_conf : float
            手部追蹤信心度 (0.0 ~ 1.0)。
        """
        self.fps_history.append(fps)
        self.tracking_confidence = tracking_conf
        self.frame_count += 1

        # ── 每 30 幀查詢系統資源 ──
        if self.has_psutil and (self.frame_count - self.last_sys_update) >= 30:
            self.last_sys_update = self.frame_count
            try:
                self.cpu_usage = psutil.cpu_percent(interval=0)
                mem = psutil.virtual_memory()
                self.mem_usage = mem.percent
            except Exception:
                pass  # 查詢失敗時保留舊值

        self.mark_dirty()

    # ────────────────────────────────────────────────────────────
    #  draw_canvas — 繪製系統監控畫布
    # ────────────────────────────────────────────────────────────
    def draw_canvas(self):
        """
        繪製系統監控面板的完整畫布。

        佈局（由上到下）：
          1. 標題「SYSTEM MONITOR」
          2. FPS 滾動折線圖
          3. CPU / 記憶體圓弧儀表盤（左右並排）
          4. 追蹤信心度進度條
          5. 幀計數器

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
        title = "SYSTEM MONITOR"
        title_scale = 0.45
        title_thick = 1
        title_size, _ = cv2.getTextSize(title, font, title_scale, title_thick)
        title_x = (self.canvas_w - title_size[0]) // 2
        cv2.putText(canvas, title, (title_x, 20), font, title_scale,
                    CLR_HUD_TITLE, title_thick, cv2.LINE_AA)

        # ── FPS 折線圖 ──
        fps_rect = (10, 35, self.canvas_w - 20, 55)
        self._draw_fps_graph(canvas, fps_rect)

        # ── CPU 和記憶體圓弧儀表盤 ──
        gauge_y = 115
        gauge_r = 28
        cpu_center = (self.canvas_w // 4, gauge_y)
        mem_center = (3 * self.canvas_w // 4, gauge_y)

        self._draw_arc_gauge(canvas, cpu_center, gauge_r,
                             self.cpu_usage, 100.0, "CPU", CLR_CYAN)
        self._draw_arc_gauge(canvas, mem_center, gauge_r,
                             self.mem_usage, 100.0, "MEM", CLR_MAGENTA)

        # ── psutil 不可用提示 ──
        if not self.has_psutil:
            hint_text = "(psutil N/A)"
            hint_size, _ = cv2.getTextSize(hint_text, font, 0.28, 1)
            hint_x = (self.canvas_w - hint_size[0]) // 2
            cv2.putText(canvas, hint_text, (hint_x, gauge_y + gauge_r + 18),
                        font, 0.28, CLR_DARK_GRID, 1, cv2.LINE_AA)

        # ── 追蹤信心度條 ──
        bar_y = 155
        self._draw_confidence_bar(canvas, bar_y)

        # ── 幀計數器 ──
        frame_text = f"FRAMES: {self.frame_count}"
        cv2.putText(canvas, frame_text, (10, self.canvas_h - 12), font, 0.32,
                    CLR_CYAN_DIM, 1, cv2.LINE_AA)

        # ── 當前 FPS 數字顯示 ──
        if self.fps_history:
            current_fps = self.fps_history[-1]
            fps_text = f"FPS: {current_fps:.0f}"
            fps_color = CLR_HUD_OK if current_fps > 25 else (
                CLR_HUD_WARN if current_fps > 15 else CLR_HUD_ERR
            )
            fps_size, _ = cv2.getTextSize(fps_text, font, 0.35, 1)
            fps_x = self.canvas_w - fps_size[0] - 10
            cv2.putText(canvas, fps_text, (fps_x, self.canvas_h - 12),
                        font, 0.35, fps_color, 1, cv2.LINE_AA)

        return canvas

    # ────────────────────────────────────────────────────────────
    #  _draw_fps_graph — FPS 滾動折線圖
    # ────────────────────────────────────────────────────────────
    def _draw_fps_graph(self, canvas, rect):
        """
        繪製 FPS 滾動折線圖。

        顏色依 FPS 值分段：
          • 綠色 (> 25 FPS)
          • 黃色 (15-25 FPS)
          • 紅色 (< 15 FPS)

        Parameters
        ----------
        canvas : np.ndarray
            目標畫布。
        rect : tuple(int, int, int, int)
            圖表區域 (x, y, w, h)。
        """
        rx, ry, rw, rh = rect
        data = list(self.fps_history)
        max_fps = 60.0  # Y 軸最大值

        # ── 圖表背景框 ──
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh),
                      CLR_DARK_OVERLAY, -1)
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh),
                      CLR_CYAN_DIM, 1)

        # ── 水平參考線 ──
        for ref_fps in [15, 30, 45]:
            ref_y = ry + rh - int((ref_fps / max_fps) * rh)
            if ry < ref_y < ry + rh:
                cv2.line(canvas, (rx, ref_y), (rx + rw, ref_y),
                         CLR_DARK_GRID, 1)
                # 參考線標籤
                cv2.putText(canvas, str(ref_fps), (rx + 2, ref_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.22, CLR_CYAN_DIM, 1)

        # ── 繪製折線 ──
        if len(data) < 2:
            return

        num_pts = len(data)
        pts = []
        for i, fps_val in enumerate(data):
            x = rx + int(i / max(num_pts - 1, 1) * rw)
            clamped = min(max(fps_val, 0), max_fps)
            y = ry + rh - int((clamped / max_fps) * rh)
            pts.append([x, y])

        # ── 分段著色 ──
        for i in range(len(pts) - 1):
            fps_val = data[i + 1]
            if fps_val > 25:
                seg_color = CLR_HUD_OK
            elif fps_val > 15:
                seg_color = CLR_HUD_WARN
            else:
                seg_color = CLR_HUD_ERR

            cv2.line(canvas,
                     (pts[i][0], pts[i][1]),
                     (pts[i + 1][0], pts[i + 1][1]),
                     seg_color, 1, cv2.LINE_AA)

        # ── 最新值的小圓點 ──
        if pts:
            last_pt = pts[-1]
            cv2.circle(canvas, (last_pt[0], last_pt[1]), 3, CLR_WHITE, -1, cv2.LINE_AA)

    # ────────────────────────────────────────────────────────────
    #  _draw_line_graph — 通用折線圖（備用）
    # ────────────────────────────────────────────────────────────
    def _draw_line_graph(self, canvas, data, rect, color, max_val):
        """
        通用折線圖繪製器。

        Parameters
        ----------
        canvas : np.ndarray
            目標畫布。
        data : list or deque
            數據序列。
        rect : tuple(int, int, int, int)
            圖表區域 (x, y, w, h)。
        color : tuple
            線條 BGR 顏色。
        max_val : float
            Y 軸最大值。
        """
        rx, ry, rw, rh = rect
        if len(data) < 2 or max_val <= 0:
            return

        num_pts = len(data)
        pts = []
        for i, val in enumerate(data):
            x = rx + int(i / max(num_pts - 1, 1) * rw)
            clamped = min(max(val, 0), max_val)
            y = ry + rh - int((clamped / max_val) * rh)
            pts.append([x, y])

        pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts_arr], False, color, 1, cv2.LINE_AA)

    # ────────────────────────────────────────────────────────────
    #  _draw_arc_gauge — 圓弧儀表盤
    # ────────────────────────────────────────────────────────────
    def _draw_arc_gauge(self, canvas, center, radius, value, max_val, label, color):
        """
        繪製圓弧儀表盤 (0 ~ 270 度)。

        儀表盤從左下方 (135°) 開始，順時針到右下方 (405° / 45°)。
        中央顯示百分比數值，下方顯示標籤。

        Parameters
        ----------
        canvas : np.ndarray
            目標畫布。
        center : tuple(int, int)
            圓心座標。
        radius : int
            儀表盤半徑。
        value : float
            當前值。
        max_val : float
            最大值。
        label : str
            標籤文字。
        color : tuple
            主色調 BGR。
        """
        cx, cy = center
        arc_range = 270    # 圓弧角度範圍
        start_angle = 135  # 起始角度（從左下方開始）

        # ── 背景弧（暗色）──
        cv2.ellipse(canvas, (cx, cy), (radius, radius),
                    0, start_angle, start_angle + arc_range,
                    CLR_DARK_GRID, 2, cv2.LINE_AA)

        # ── 數值弧（依百分比填充）──
        ratio = min(max(value / max_val, 0.0), 1.0) if max_val > 0 else 0.0
        value_angle = int(ratio * arc_range)

        if value_angle > 0:
            # 依百分比選色（低：綠、中：黃、高：紅）
            if ratio < 0.5:
                arc_color = CLR_HUD_OK
            elif ratio < 0.8:
                arc_color = CLR_HUD_WARN
            else:
                arc_color = CLR_HUD_ERR

            cv2.ellipse(canvas, (cx, cy), (radius, radius),
                        0, start_angle, start_angle + value_angle,
                        arc_color, 3, cv2.LINE_AA)

        # ── 中央數值文字 ──
        font = cv2.FONT_HERSHEY_SIMPLEX
        pct_text = f"{value:.0f}%"
        pct_size, _ = cv2.getTextSize(pct_text, font, 0.32, 1)
        pct_x = cx - pct_size[0] // 2
        pct_y = cy + pct_size[1] // 2
        cv2.putText(canvas, pct_text, (pct_x, pct_y), font, 0.32,
                    CLR_HUD_DATA, 1, cv2.LINE_AA)

        # ── 下方標籤 ──
        lbl_size, _ = cv2.getTextSize(label, font, 0.28, 1)
        lbl_x = cx - lbl_size[0] // 2
        lbl_y = cy + radius + 14
        cv2.putText(canvas, label, (lbl_x, lbl_y), font, 0.28,
                    color, 1, cv2.LINE_AA)

    # ────────────────────────────────────────────────────────────
    #  _draw_confidence_bar — 追蹤信心度進度條
    # ────────────────────────────────────────────────────────────
    def _draw_confidence_bar(self, canvas, y):
        """
        繪製追蹤信心度水平進度條。

        Parameters
        ----------
        canvas : np.ndarray
            目標畫布。
        y : int
            進度條 Y 位置。
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        margin = 12
        bar_x = margin
        bar_w = self.canvas_w - 2 * margin
        bar_h = 10

        # ── 標籤 ──
        label = "TRACKING"
        cv2.putText(canvas, label, (bar_x, y), font, 0.28,
                    CLR_CYAN_DIM, 1, cv2.LINE_AA)

        # ── 進度條背景 ──
        bar_top = y + 4
        cv2.rectangle(canvas, (bar_x, bar_top), (bar_x + bar_w, bar_top + bar_h),
                      CLR_DARK_OVERLAY, -1)
        cv2.rectangle(canvas, (bar_x, bar_top), (bar_x + bar_w, bar_top + bar_h),
                      CLR_CYAN_DIM, 1)

        # ── 進度條填充 ──
        ratio = min(max(self.tracking_confidence, 0.0), 1.0)
        fill_w = int(ratio * bar_w)
        if fill_w > 0:
            # 顏色梯度：低=紅, 中=黃, 高=綠
            if ratio > 0.7:
                bar_color = CLR_HUD_OK
            elif ratio > 0.4:
                bar_color = CLR_HUD_WARN
            else:
                bar_color = CLR_HUD_ERR

            cv2.rectangle(canvas, (bar_x + 1, bar_top + 1),
                          (bar_x + fill_w - 1, bar_top + bar_h - 1),
                          bar_color, -1)

        # ── 百分比文字 ──
        pct_text = f"{ratio * 100:.0f}%"
        pct_size, _ = cv2.getTextSize(pct_text, font, 0.28, 1)
        pct_x = bar_x + bar_w - pct_size[0]
        cv2.putText(canvas, pct_text, (pct_x, y), font, 0.28,
                    CLR_HUD_DATA, 1, cv2.LINE_AA)
