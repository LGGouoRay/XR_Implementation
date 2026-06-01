# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║             XR HUD 覆蓋層 — HUD Overlay Effects              ║
║  在最終渲染之上疊加科幻風格 HUD 元素：掃描線、角落裝飾、     ║
║  系統資訊、追蹤品質、狀態列、迷你雷達、呼吸邊緣、截圖閃光。 ║
╚══════════════════════════════════════════════════════════════╝
"""
import cv2
import numpy as np
import math
import time
from datetime import datetime

# 將上層目錄加入搜尋路徑以匯入中央設定
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import *


class HUDOverlay:
    """
    HUD 覆蓋層渲染器。

    在主畫面之上繪製科幻風格的疊加元素，包含：
      - 掃描線效果 (scrolling scanlines)
      - 角落裝飾 (corner brackets)
      - 系統資訊顯示 (FPS / 時間 / 版本)
      - 追蹤品質指示器 (tracking quality bar)
      - 底部狀態滾動字幕 (status ticker)
      - 迷你雷達 (mini radar)
      - 螢幕邊緣呼吸光暈 (breathing edges)
      - 截圖閃光動畫 (screenshot flash)
    """

    def __init__(self, screen_w, screen_h):
        """
        初始化 HUD 覆蓋層。

        Args:
            screen_w: 螢幕寬度 (像素)
            screen_h: 螢幕高度 (像素)
        """
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.frame_count = 0
        self.start_time = time.time()

        # ── 掃描線狀態 ──
        self.scanline_offset = 0.0  # 垂直捲動偏移量

        # ── 預先生成掃描線覆蓋層以提升效能 ──
        self._scanline_overlay = None
        self._generate_scanline_overlay()

        # ── 截圖閃光狀態 ──
        self.flash_active = False
        self.flash_counter = 0

        # ── 狀態訊息佇列 ──
        # 格式: (訊息字串, 時間戳, BGR 顏色)
        self.status_messages = []

        # ── 追蹤品質 ──
        self.tracking_quality = 0.0  # 0.0 ~ 1.0

        # ── FPS 計算 ──
        self._fps_time_history = []
        self._current_fps = 0.0

        # ── 雷達掃描角度 ──
        self._radar_angle = 0.0

        # ── 滾動字幕偏移 ──
        self._ticker_offset = 0.0

        # ── 預設狀態訊息 ──
        self.add_status_message("XR-HUD SYSTEM INITIALIZED", CLR_CYAN)
        self.add_status_message("HAND TRACKING ACTIVE", CLR_GREEN)

    # ============================================================
    #  掃描線覆蓋層預生成
    # ============================================================
    def _generate_scanline_overlay(self):
        """
        預先計算掃描線圖案覆蓋層。
        水平線條每隔 HUD_SCANLINE_SPACING 像素繪製一條。
        此覆蓋層會被快取以避免每幀重新建立。
        高度為兩倍螢幕高度，捲動時循環使用。
        """
        # 建立雙倍高度的覆蓋層，以實現無縫捲動
        h = self.screen_h * 2
        self._scanline_overlay = np.zeros((h, self.screen_w, 3), dtype=np.uint8)

        # 每隔 HUD_SCANLINE_SPACING 像素繪製一條暗色水平線
        for y in range(0, h, HUD_SCANLINE_SPACING):
            self._scanline_overlay[y, :] = (15, 15, 8)  # 極暗的青色調

    # ============================================================
    #  狀態更新 — 每幀呼叫
    # ============================================================
    def update(self, tracking_quality=0.0):
        """
        更新 HUD 內部狀態。

        Args:
            tracking_quality: 手部追蹤品質 [0.0 ~ 1.0]
        """
        self.frame_count += 1
        self.tracking_quality = tracking_quality

        # 掃描線捲動偏移
        self.scanline_offset = (self.scanline_offset + HUD_SCANLINE_SCROLL_SPEED) % self.screen_h

        # 截圖閃光倒計時
        if self.flash_active:
            self.flash_counter -= 1
            if self.flash_counter <= 0:
                self.flash_active = False

        # FPS 計算 — 使用滑動視窗
        now = time.time()
        self._fps_time_history.append(now)
        # 保留最近 30 幀的時間戳
        if len(self._fps_time_history) > 30:
            self._fps_time_history = self._fps_time_history[-30:]
        if len(self._fps_time_history) >= 2:
            dt = self._fps_time_history[-1] - self._fps_time_history[0]
            if dt > 0:
                self._current_fps = (len(self._fps_time_history) - 1) / dt

        # 雷達掃描角度 — 每幀旋轉 3 度
        self._radar_angle = (self._radar_angle + 3) % 360

        # 滾動字幕偏移
        self._ticker_offset += 1.5

        # 清理過期的狀態訊息 (超過 8 秒自動移除)
        cutoff = now - 8.0
        self.status_messages = [
            (msg, ts, clr) for msg, ts, clr in self.status_messages
            if ts > cutoff
        ]

    # ============================================================
    #  主渲染流程
    # ============================================================
    def render(self, frame, panels_info=None, hand_positions=None):
        """
        在提供的影像幀上繪製所有 HUD 覆蓋元素。

        渲染順序 (由底到頂)：
          1. 掃描線
          2. 角落裝飾
          3. 系統資訊 (左上)
          4. 追蹤品質 (右上)
          5. 底部狀態列
          6. 迷你雷達 (右下)
          7. 呼吸邊緣光暈
          8. 截圖閃光 (如果啟用)

        Args:
            frame: 輸入影像幀 (BGR, uint8)
            panels_info: 面板資訊 (可選，用於雷達顯示)
            hand_positions: 手部 3D 位置列表 (可選)

        Returns:
            繪製完成的影像幀
        """
        # 1. 掃描線
        self._draw_scanlines(frame)

        # 2. 角落裝飾
        self._draw_corner_decorations(frame)

        # 3. 系統資訊
        self._draw_system_info(frame)

        # 4. 追蹤品質指示器
        self._draw_tracking_indicator(frame)

        # 5. 底部狀態列
        self._draw_status_bar(frame)

        # 6. 迷你雷達
        self._draw_mini_radar(frame, hand_positions)

        # 7. 呼吸邊緣光暈
        self._draw_breathing_edges(frame)

        # 8. 截圖閃光
        if self.flash_active:
            self._draw_flash(frame)

        return frame

    # ============================================================
    #  掃描線效果
    # ============================================================
    def _draw_scanlines(self, frame):
        """
        在畫面上疊加捲動掃描線效果。
        使用預先計算的覆蓋層，根據 scanline_offset 進行垂直位移。
        以極低的 alpha 值混合，產生微妙的 CRT 螢幕質感。
        """
        if self._scanline_overlay is None:
            return

        offset = int(self.scanline_offset)

        # 從預計算覆蓋層中裁切出當前可見區域
        scanlines_roi = self._scanline_overlay[offset:offset + self.screen_h, :self.screen_w]

        # 確保尺寸匹配
        if scanlines_roi.shape[0] != frame.shape[0] or scanlines_roi.shape[1] != frame.shape[1]:
            return

        # 以極低 alpha 進行加法混合
        cv2.addWeighted(scanlines_roi, HUD_SCANLINE_ALPHA, frame, 1.0, 0, frame)

    # ============================================================
    #  角落裝飾 (Sci-Fi Corner Brackets)
    # ============================================================
    def _draw_corner_decorations(self, frame):
        """
        在螢幕四個角落繪製科幻風格的 L 形括弧裝飾。
        每個角落包含：
          - L 形線段 (水平 + 垂直)
          - 角落頂點的小圓點
          - 呼吸脈衝動畫效果
        """
        size = HUD_CORNER_SIZE
        t = HUD_CORNER_THICKNESS
        breathe = self._get_breathe_alpha()
        margin = 8  # 距離螢幕邊緣的留白

        # 根據呼吸值調整顏色亮度
        intensity = int(180 * breathe + 75)
        color = (intensity, intensity, 0)         # 青色調
        dot_color = (intensity, intensity, 40)     # 角點顏色稍帶暖調

        w = self.screen_w
        h = self.screen_h

        # ── 左上角 ──
        cv2.line(frame, (margin, margin), (margin + size, margin), color, t, cv2.LINE_AA)
        cv2.line(frame, (margin, margin), (margin, margin + size), color, t, cv2.LINE_AA)
        cv2.circle(frame, (margin, margin), 3, dot_color, -1, cv2.LINE_AA)

        # ── 右上角 ──
        cv2.line(frame, (w - margin, margin), (w - margin - size, margin), color, t, cv2.LINE_AA)
        cv2.line(frame, (w - margin, margin), (w - margin, margin + size), color, t, cv2.LINE_AA)
        cv2.circle(frame, (w - margin, margin), 3, dot_color, -1, cv2.LINE_AA)

        # ── 左下角 ──
        cv2.line(frame, (margin, h - margin), (margin + size, h - margin), color, t, cv2.LINE_AA)
        cv2.line(frame, (margin, h - margin), (margin, h - margin - size), color, t, cv2.LINE_AA)
        cv2.circle(frame, (margin, h - margin), 3, dot_color, -1, cv2.LINE_AA)

        # ── 右下角 ──
        cv2.line(frame, (w - margin, h - margin), (w - margin - size, h - margin), color, t, cv2.LINE_AA)
        cv2.line(frame, (w - margin, h - margin), (w - margin, h - margin - size), color, t, cv2.LINE_AA)
        cv2.circle(frame, (w - margin, h - margin), 3, dot_color, -1, cv2.LINE_AA)

        # ── 中間短線裝飾 (每條 L 形線的中段加一個小刻度) ──
        tick = size // 2
        tick_len = 6
        cv2.line(frame, (margin + tick, margin), (margin + tick, margin + tick_len), color, 1, cv2.LINE_AA)
        cv2.line(frame, (margin, margin + tick), (margin + tick_len, margin + tick), color, 1, cv2.LINE_AA)

        cv2.line(frame, (w - margin - tick, margin), (w - margin - tick, margin + tick_len), color, 1, cv2.LINE_AA)
        cv2.line(frame, (w - margin, margin + tick), (w - margin - tick_len, margin + tick), color, 1, cv2.LINE_AA)

        cv2.line(frame, (margin + tick, h - margin), (margin + tick, h - margin - tick_len), color, 1, cv2.LINE_AA)
        cv2.line(frame, (margin, h - margin - tick), (margin + tick_len, h - margin - tick), color, 1, cv2.LINE_AA)

        cv2.line(frame, (w - margin - tick, h - margin), (w - margin - tick, h - margin - tick_len), color, 1, cv2.LINE_AA)
        cv2.line(frame, (w - margin, h - margin - tick), (w - margin - tick_len, h - margin - tick), color, 1, cv2.LINE_AA)

    # ============================================================
    #  系統資訊 (左上區域)
    # ============================================================
    def _draw_system_info(self, frame):
        """
        在左上角顯示系統狀態資訊：
          - XR-HUD 版本標題
          - 當前日期時間
          - FPS 計數器 (色碼：綠=良好, 黃=普通, 紅=低)
          - 累計幀數
        """
        x_base = 20
        y_base = 28
        line_h = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_small = 0.40
        font_title = 0.50

        # ── 半透明背景面板 ──
        panel_w = 190
        panel_h = 95
        overlay = frame[y_base - 18:y_base + panel_h - 18,
                        x_base - 8:x_base + panel_w].copy()
        if overlay.size > 0:
            dark_panel = np.zeros_like(overlay)
            cv2.addWeighted(overlay, 0.6, dark_panel, 0.4, 0, overlay)
            frame[y_base - 18:y_base + panel_h - 18,
                  x_base - 8:x_base + panel_w] = overlay

        # ── 標題 ──
        breathe = self._get_breathe_alpha()
        title_intensity = int(200 * breathe + 55)
        title_color = (title_intensity, title_intensity, 0)
        cv2.putText(frame, "XR-HUD v2.0", (x_base, y_base),
                    font, font_title, title_color, 1, cv2.LINE_AA)

        # ── 分隔線 ──
        cv2.line(frame, (x_base, y_base + 5), (x_base + 140, y_base + 5),
                 (80, 80, 0), 1, cv2.LINE_AA)

        # ── 日期時間 ──
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        y = y_base + line_h + 5
        cv2.putText(frame, f"DATE {date_str}", (x_base, y),
                    font, font_small, CLR_HUD_DATA, 1, cv2.LINE_AA)

        y += line_h
        cv2.putText(frame, f"TIME {time_str}", (x_base, y),
                    font, font_small, CLR_HUD_DATA, 1, cv2.LINE_AA)

        # ── FPS (色碼判定) ──
        fps = self._current_fps
        if fps >= TARGET_FPS * 0.9:
            fps_color = CLR_HUD_OK       # 綠色 — 良好
        elif fps >= FPS_LOW_THRESHOLD:
            fps_color = CLR_HUD_WARN     # 橙色 — 普通
        else:
            fps_color = CLR_HUD_ERR      # 紅色 — 過低

        y += line_h
        cv2.putText(frame, f"FPS  {fps:.1f}", (x_base, y),
                    font, font_small, fps_color, 1, cv2.LINE_AA)

        # ── 幀計數 ──
        frame_str = f"#{self.frame_count:06d}"
        cv2.putText(frame, frame_str, (x_base + 110, y),
                    font, font_small * 0.85, CLR_CYAN_DIM, 1, cv2.LINE_AA)

    # ============================================================
    #  追蹤品質指示器 (右上區域)
    # ============================================================
    def _draw_tracking_indicator(self, frame):
        """
        在右上角顯示手部追蹤品質指示器：
          - 'TRACKING' 標籤
          - 色碼品質條 (綠=良好, 黃=普通, 紅=差)
          - 數值百分比
        """
        x_base = self.screen_w - 180
        y_base = 28
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_small = 0.40

        # ── 半透明背景面板 ──
        panel_w = 168
        panel_h = 55
        py1 = max(0, y_base - 18)
        py2 = min(self.screen_h, y_base + panel_h - 18)
        px1 = max(0, x_base - 8)
        px2 = min(self.screen_w, x_base + panel_w)
        overlay = frame[py1:py2, px1:px2].copy()
        if overlay.size > 0:
            dark = np.zeros_like(overlay)
            cv2.addWeighted(overlay, 0.6, dark, 0.4, 0, overlay)
            frame[py1:py2, px1:px2] = overlay

        # ── TRACKING 標籤 ──
        cv2.putText(frame, "TRACKING", (x_base, y_base),
                    font, font_small, CLR_CYAN, 1, cv2.LINE_AA)

        # ── 品質條 ──
        bar_x = x_base
        bar_y = y_base + 10
        bar_w = 140
        bar_h = 10
        quality = self.tracking_quality

        # 品質色碼
        if quality >= 0.7:
            bar_color = CLR_HUD_OK       # 綠色
        elif quality >= 0.4:
            bar_color = CLR_YELLOW        # 黃色
        else:
            bar_color = CLR_HUD_ERR      # 紅色

        # 外框
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      CLR_CYAN_DIM, 1, cv2.LINE_AA)

        # 填充部分
        fill_w = int(bar_w * quality)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x + 1, bar_y + 1),
                          (bar_x + fill_w, bar_y + bar_h - 1),
                          bar_color, -1, cv2.LINE_AA)

        # ── 百分比數值 ──
        pct_str = f"{quality * 100:.0f}%"
        cv2.putText(frame, pct_str, (bar_x + bar_w + 5, bar_y + bar_h - 1),
                    font, font_small * 0.9, bar_color, 1, cv2.LINE_AA)

    # ============================================================
    #  底部狀態列
    # ============================================================
    def _draw_status_bar(self, frame):
        """
        在螢幕底部繪製狀態列：
          - 細水平分隔線
          - 滾動狀態文字字幕
          - 模式指示圖標
        """
        y_bar = self.screen_h - 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_small = 0.35

        # ── 半透明底部面板 ──
        panel_y1 = max(0, y_bar - 5)
        overlay = frame[panel_y1:self.screen_h, :].copy()
        if overlay.size > 0:
            dark = np.zeros_like(overlay)
            cv2.addWeighted(overlay, 0.65, dark, 0.35, 0, overlay)
            frame[panel_y1:self.screen_h, :] = overlay

        # ── 分隔線 ──
        cv2.line(frame, (10, y_bar - 5), (self.screen_w - 10, y_bar - 5),
                 CLR_CYAN_DIM, 1, cv2.LINE_AA)

        # ── 滾動字幕 ──
        if self.status_messages:
            # 組合所有狀態訊息為一條長字串
            ticker_text = "  ◆  ".join(
                msg for msg, ts, clr in self.status_messages
            )
            # 計算滾動位置
            text_width = len(ticker_text) * 8  # 粗略估算
            offset = int(self._ticker_offset) % (text_width + self.screen_w)
            draw_x = self.screen_w - offset

            # 使用最近一條訊息的顏色
            text_color = self.status_messages[-1][2] if self.status_messages else CLR_CYAN

            cv2.putText(frame, ticker_text, (draw_x, y_bar + 12),
                        font, font_small, text_color, 1, cv2.LINE_AA)
        else:
            # 預設待機文字
            cv2.putText(frame, "SYSTEM STANDBY", (20, y_bar + 12),
                        font, font_small, CLR_CYAN_DIM, 1, cv2.LINE_AA)

        # ── 左側模式指示器 ──
        mode_indicators = ["SYS", "TRK", "VFX"]
        for i, label in enumerate(mode_indicators):
            ix = 12 + i * 45
            iy = self.screen_h - 8
            # 小方框
            cv2.rectangle(frame, (ix, iy - 10), (ix + 35, iy + 2),
                          CLR_CYAN_DIM, 1, cv2.LINE_AA)
            cv2.putText(frame, label, (ix + 4, iy),
                        font, font_small * 0.85, CLR_CYAN, 1, cv2.LINE_AA)

    # ============================================================
    #  迷你雷達 (右下角)
    # ============================================================
    def _draw_mini_radar(self, frame, hand_positions=None):
        """
        在右下角繪製圓形迷你雷達顯示器：
          - 同心圓表示深度區域
          - 旋轉掃描線
          - 手部位置點 (從 3D 空間映射到雷達平面)
          - 中心十字準星
        """
        radius = HUD_RADAR_SIZE // 2
        cx = self.screen_w - HUD_RADAR_POS[0]
        cy = self.screen_h - HUD_RADAR_POS[1]

        # ── 半透明圓形背景 ──
        # 建立圓形遮罩
        mask_y1 = max(0, cy - radius - 5)
        mask_y2 = min(self.screen_h, cy + radius + 5)
        mask_x1 = max(0, cx - radius - 5)
        mask_x2 = min(self.screen_w, cx + radius + 5)

        roi = frame[mask_y1:mask_y2, mask_x1:mask_x2].copy()
        if roi.size > 0:
            # 建立圓形遮罩
            local_cx = cx - mask_x1
            local_cy = cy - mask_y1
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (local_cx, local_cy), radius + 2, 255, -1)

            # 暗化圓形區域
            dark = (roi * 0.3).astype(np.uint8)
            roi_masked = np.where(mask[:, :, None] > 0, dark, roi)
            frame[mask_y1:mask_y2, mask_x1:mask_x2] = roi_masked

        # ── 同心圓 (深度區域) ──
        for r in [radius, radius * 2 // 3, radius // 3]:
            cv2.circle(frame, (cx, cy), r, CLR_DARK_GRID, 1, cv2.LINE_AA)

        # ── 外框 ──
        cv2.circle(frame, (cx, cy), radius, CLR_CYAN_DIM, 1, cv2.LINE_AA)

        # ── 十字線 ──
        cross_len = radius - 5
        cv2.line(frame, (cx - cross_len, cy), (cx + cross_len, cy),
                 CLR_DARK_GRID, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - cross_len), (cx, cy + cross_len),
                 CLR_DARK_GRID, 1, cv2.LINE_AA)

        # ── 旋轉掃描線 ──
        angle_rad = math.radians(self._radar_angle)
        sweep_x = int(cx + radius * math.cos(angle_rad))
        sweep_y = int(cy + radius * math.sin(angle_rad))
        cv2.line(frame, (cx, cy), (sweep_x, sweep_y),
                 CLR_CYAN, 1, cv2.LINE_AA)

        # 掃描線尾跡 — 繪製多條漸淡的線段形成扇形尾跡效果
        for i in range(1, 6):
            trail_angle = math.radians(self._radar_angle - i * 5)
            trail_x = int(cx + radius * math.cos(trail_angle))
            trail_y = int(cy + radius * math.sin(trail_angle))
            trail_intensity = max(10, 60 - i * 12)
            trail_color = (trail_intensity, trail_intensity, 0)
            cv2.line(frame, (cx, cy), (trail_x, trail_y),
                     trail_color, 1, cv2.LINE_AA)

        # ── 中心點 ──
        cv2.circle(frame, (cx, cy), 2, CLR_CYAN_BRIGHT, -1, cv2.LINE_AA)

        # ── 手部位置映射 ──
        if hand_positions:
            for hp in hand_positions:
                # hp 預期格式: (x_norm, y_norm, z) — 歸一化座標 + 深度
                # 將 3D 位置映射到雷達 2D 平面
                if len(hp) >= 3:
                    # X/Y 映射到雷達範圍 (歸一化座標 0~1 映射到 -radius~+radius)
                    rx = int(cx + (hp[0] - 0.5) * 2 * (radius - 8))
                    ry = int(cy + (hp[1] - 0.5) * 2 * (radius - 8))

                    # 深度映射到點的大小 (越近越大)
                    dot_size = max(2, min(6, int(5 - abs(hp[2]) * 3)))

                    # 確保點在雷達範圍內
                    dist = math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
                    if dist < radius - 3:
                        cv2.circle(frame, (rx, ry), dot_size,
                                   CLR_YELLOW, -1, cv2.LINE_AA)
                        # 小圓環
                        cv2.circle(frame, (rx, ry), dot_size + 3,
                                   CLR_YELLOW_DIM, 1, cv2.LINE_AA)

        # ── 雷達標籤 ──
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "RADAR", (cx - radius, cy - radius - 6),
                    font, 0.30, CLR_CYAN_DIM, 1, cv2.LINE_AA)

    # ============================================================
    #  呼吸邊緣光暈
    # ============================================================
    def _draw_breathing_edges(self, frame):
        """
        在螢幕四邊繪製微妙的呼吸光暈效果。
        Alpha 值隨時間以正弦波振盪，營造活躍感。
        使用漸層條帶而非單色線條，提升視覺品質。
        """
        breathe = self._get_breathe_alpha()
        edge_width = 3         # 邊緣光暈寬度
        intensity = int(50 * breathe)
        if intensity < 5:
            return

        color = np.array([intensity, intensity, 0], dtype=np.uint8)

        # ── 頂邊 ──
        if frame.shape[0] > edge_width and frame.shape[1] > 0:
            glow = np.full((edge_width, frame.shape[1], 3), color, dtype=np.uint8)
            frame[:edge_width] = cv2.add(frame[:edge_width], glow)

        # ── 底邊 ──
        if frame.shape[0] > edge_width:
            glow = np.full((edge_width, frame.shape[1], 3), color, dtype=np.uint8)
            frame[-edge_width:] = cv2.add(frame[-edge_width:], glow)

        # ── 左邊 ──
        if frame.shape[1] > edge_width:
            glow = np.full((frame.shape[0], edge_width, 3), color, dtype=np.uint8)
            frame[:, :edge_width] = cv2.add(frame[:, :edge_width], glow)

        # ── 右邊 ──
        if frame.shape[1] > edge_width:
            glow = np.full((frame.shape[0], edge_width, 3), color, dtype=np.uint8)
            frame[:, -edge_width:] = cv2.add(frame[:, -edge_width:], glow)

    # ============================================================
    #  截圖閃光動畫
    # ============================================================
    def trigger_flash(self):
        """觸發截圖閃光動畫。"""
        self.flash_active = True
        self.flash_counter = SCREENSHOT_FLASH_DURATION

    def _draw_flash(self, frame):
        """
        繪製截圖閃光效果：
          - 全螢幕白色閃光 (快速衰減)
          - 縮小的邊框矩形動畫
        """
        if self.flash_counter <= 0:
            return

        # 閃光強度隨幀數遞減
        progress = self.flash_counter / SCREENSHOT_FLASH_DURATION  # 1.0 → 0.0
        flash_alpha = progress * 0.7  # 最高 70% 不透明度

        # ── 全螢幕白色閃光 ──
        white_overlay = np.full_like(frame, 255, dtype=np.uint8)
        cv2.addWeighted(white_overlay, flash_alpha, frame, 1.0 - flash_alpha, 0, frame)

        # ── 縮小的邊框矩形動畫 ──
        # 矩形從螢幕邊緣向內收縮
        shrink = int((1.0 - progress) * 50)  # 0 → 50 px
        border_color = CLR_WHITE if progress > 0.5 else CLR_CYAN_BRIGHT
        cv2.rectangle(frame,
                      (shrink, shrink),
                      (self.screen_w - shrink - 1, self.screen_h - shrink - 1),
                      border_color, 2, cv2.LINE_AA)

    # ============================================================
    #  狀態訊息管理
    # ============================================================
    def add_status_message(self, message, color=None):
        """
        新增一條狀態訊息到底部滾動字幕。

        Args:
            message: 訊息文字
            color: BGR 顏色 (預設 CLR_CYAN)
        """
        if color is None:
            color = CLR_CYAN
        self.status_messages.append((message, time.time(), color))

        # 限制訊息數量 (最多保留 10 條)
        if len(self.status_messages) > 10:
            self.status_messages = self.status_messages[-10:]

    # ============================================================
    #  呼吸 Alpha 計算
    # ============================================================
    def _get_breathe_alpha(self):
        """
        計算當前呼吸動畫的 alpha 值。
        在 HUD_BREATHE_MIN_ALPHA 和 HUD_BREATHE_MAX_ALPHA 之間
        以正弦波振盪。

        Returns:
            float: 當前 alpha 值
        """
        t = time.time() - self.start_time
        return (HUD_BREATHE_MIN_ALPHA +
                (HUD_BREATHE_MAX_ALPHA - HUD_BREATHE_MIN_ALPHA) *
                (0.5 + 0.5 * math.sin(t * HUD_BREATHE_SPEED * 2 * math.pi)))

    # ============================================================
    #  公用工具
    # ============================================================
    def set_screen_size(self, w, h):
        """
        動態更新螢幕尺寸並重新生成掃描線覆蓋層。

        Args:
            w: 新寬度
            h: 新高度
        """
        self.screen_w = w
        self.screen_h = h
        self._generate_scanline_overlay()

    def get_fps(self):
        """回傳當前計算的 FPS 值。"""
        return self._current_fps

    def get_frame_count(self):
        """回傳累計幀數。"""
        return self.frame_count
