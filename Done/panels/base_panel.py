# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║          XR HUD — 3D 面板基礎類別 (Base Panel)               ║
║  提供所有 3D 浮動面板的公共功能：                              ║
║    • 3D 姿態管理（位移、旋轉、EMA 平滑）                      ║
║    • 2D 畫布繪製 → 3D 透視投影（Homography Warp）             ║
║    • 拖曳互動、觸控檢測                                       ║
║    • 按鈕元件（Button3D）                                     ║
║    • 科幻風格視覺效果                                         ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import math
import time

from core.math3d import (
    get_rotation_matrix,
    project_point,
    euler_to_quaternion,
    quaternion_slerp,
    quaternion_to_euler,
    lerp_3d,
)
from config import *


# ════════════════════════════════════════════════════════════════
#  Button3D — 面板畫布上的可重用按鈕元件
# ════════════════════════════════════════════════════════════════
class Button3D:
    """
    可重用的 3D 面板按鈕元件。
    按鈕繪製在 2D 畫布上，透過面板的觸控系統接收互動事件。

    Parameters
    ----------
    label : str
        按鈕文字標籤。
    rect : tuple(int, int, int, int)
        按鈕在畫布上的位置和大小 (x, y, w, h)。
    action_id : str
        按鈕動作的字串識別符，用於回呼。
    icon : str or None
        選用的圖示字元（將繪製在標籤左側）。
    toggle : bool
        若為 True，按鈕為切換模式（點擊切換 active 狀態）。
    """

    def __init__(self, label, rect, action_id, icon=None, toggle=False):
        self.label = label
        self.rect = rect                # (x, y, w, h) — 畫布座標
        self.action_id = action_id
        self.icon = icon
        self.toggle = toggle

        # ── 互動狀態 ──
        self.is_hovered = False         # 手指懸停中
        self.is_pressed = False         # 正在被按下
        self.is_active = False          # 切換按鈕的啟用狀態
        self.press_anim = 0.0           # 按壓動畫計數器 [0.0 ~ 1.0]

    # ────────────────────────────────────────────────────────────
    #  draw — 在畫布上繪製按鈕
    # ────────────────────────────────────────────────────────────
    def draw(self, canvas):
        """
        在指定畫布上繪製此按鈕。
        根據當前互動狀態（normal / hover / pressed / active）
        改變外觀。
        """
        x, y, w, h = self.rect

        # ── 按壓動畫衰減 ──
        if self.press_anim > 0.01:
            self.press_anim *= 0.85     # 指數衰減

        # ── 計算按壓縮放偏移 ──
        shrink = int(self.press_anim * 4)   # 最多內縮 4px
        bx, by = x + shrink, y + shrink
        bw, bh = w - shrink * 2, h - shrink * 2

        # ── 依狀態選色 ──
        if self.is_active and self.toggle:
            # 啟用狀態：填滿底色
            bg_color = CLR_BTN_ACTIVE
            border_color = CLR_BTN_ACTIVE
            text_color = CLR_BTN_TEXT        # 深色文字
            fill = True
        elif self.is_pressed:
            # 按壓狀態：紅色高亮
            bg_color = CLR_BTN_PRESSED
            border_color = CLR_BTN_PRESSED
            text_color = CLR_WHITE
            fill = True
        elif self.is_hovered:
            # 懸停狀態：亮邊框 + 微光
            bg_color = None
            border_color = CLR_BTN_HOVER
            text_color = CLR_BTN_HOVER
            fill = False
        else:
            # 正常狀態：暗邊框
            bg_color = None
            border_color = CLR_BTN_NORMAL
            text_color = CLR_BTN_NORMAL
            fill = False

        # ── 繪製圓角矩形 ──
        radius = 6
        if fill and bg_color is not None:
            # 填充背景
            cv2.rectangle(canvas, (bx + radius, by), (bx + bw - radius, by + bh), bg_color, -1)
            cv2.rectangle(canvas, (bx, by + radius), (bx + bw, by + bh - radius), bg_color, -1)
            cv2.circle(canvas, (bx + radius, by + radius), radius, bg_color, -1)
            cv2.circle(canvas, (bx + bw - radius, by + radius), radius, bg_color, -1)
            cv2.circle(canvas, (bx + radius, by + bh - radius), radius, bg_color, -1)
            cv2.circle(canvas, (bx + bw - radius, by + bh - radius), radius, bg_color, -1)

        # 繪製邊框
        _draw_rounded_rect_static(canvas, (bx, by), (bx + bw, by + bh),
                                  border_color, 1, radius)

        # ── 懸停微光效果 ──
        if self.is_hovered and not fill:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), border_color, -1)
            cv2.addWeighted(overlay, 0.08, canvas, 0.92, 0, canvas)

        # ── 繪製標籤文字（置中）──
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        thickness = 1
        text = self.label

        # 若有圖示，加在前面
        if self.icon:
            text = f"{self.icon} {text}"

        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        tw, th = text_size
        tx = bx + (bw - tw) // 2
        ty = by + (bh + th) // 2

        cv2.putText(canvas, text, (tx, ty), font, font_scale,
                    text_color, thickness, cv2.LINE_AA)

        return canvas

    # ────────────────────────────────────────────────────────────
    #  hit_test — 碰撞檢測
    # ────────────────────────────────────────────────────────────
    def hit_test(self, canvas_x, canvas_y):
        """
        判斷畫布座標 (canvas_x, canvas_y) 是否在按鈕範圍內。

        Returns
        -------
        bool
            True 表示座標在按鈕區域內。
        """
        x, y, w, h = self.rect
        return (x <= canvas_x <= x + w) and (y <= canvas_y <= y + h)


# ════════════════════════════════════════════════════════════════
#  輔助函式
# ════════════════════════════════════════════════════════════════
def _draw_rounded_rect_static(canvas, pt1, pt2, color, thickness=1, radius=8):
    """
    繪製圓角矩形（靜態函式版本）。
    使用 cv2 直線和圓弧基元實現。

    Parameters
    ----------
    canvas : np.ndarray
        目標畫布。
    pt1 : tuple(int, int)
        左上角 (x1, y1)。
    pt2 : tuple(int, int)
        右下角 (x2, y2)。
    color : tuple
        BGR 顏色。
    thickness : int
        線條粗細，-1 為填滿。
    radius : int
        圓角半徑。
    """
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    # 四條直線段
    cv2.line(canvas, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)

    # 四個圓角弧
    cv2.ellipse(canvas, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


# ════════════════════════════════════════════════════════════════
#  Panel3D — 所有 3D 浮動面板的基礎類別
# ════════════════════════════════════════════════════════════════
class Panel3D:
    """
    3D XR 空間中浮動面板的基礎類別。

    核心職責：
      1. 管理面板的 3D 姿態（位置 T、旋轉 rot）
      2. 提供 EMA 平滑以避免抖動
      3. 將 2D 畫布透過 Homography 投影到攝影機視圖
      4. 處理拖曳和觸控互動
      5. 繪製科幻風格的基底畫布

    子類需覆寫 draw_canvas() 以繪製自訂內容。
    """

    def __init__(self, w_3d, h_3d, canvas_w, canvas_h, initial_pos, panel_id="panel"):
        """
        Parameters
        ----------
        w_3d : float
            面板在 3D 空間中的寬度 (mm)。
        h_3d : float
            面板在 3D 空間中的高度 (mm)。
        canvas_w : int
            2D 畫布寬度 (px)。
        canvas_h : int
            2D 畫布高度 (px)。
        initial_pos : np.ndarray
            初始 3D 位置 [x, y, z]。
        panel_id : str
            面板識別符。
        """
        self.panel_id = panel_id
        self.w_3d = w_3d
        self.h_3d = h_3d
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h

        # ── 3D 姿態 ──
        self.T = initial_pos.copy().astype(float)       # 當前位置
        self.rot = np.array([0.0, 0.0, 0.0])            # 當前歐拉角 (yaw, pitch, roll)
        self.target_T = self.T.copy()                    # 目標位置（平滑插值）
        self.target_rot = self.rot.copy()                # 目標旋轉

        # ── 平滑參數 ──
        self.smooth_alpha_pos = PANEL_SMOOTH_ALPHA_POS
        self.smooth_alpha_rot = PANEL_SMOOTH_ALPHA_ROT

        # ── 拖曳狀態 ──
        self.is_dragging = False
        self.drag_start_hand_pos = None       # 拖曳開始時的手部 3D 位置
        self.drag_start_screen_pos = None     # 拖曳開始時的面板位置
        self.drag_start_hand_rot = None       # 拖曳開始時的手部旋轉
        self.drag_start_screen_rot = None     # 拖曳開始時的面板旋轉

        # ── 可見性與動畫 ──
        self.visible = True
        self.opacity = PANEL_OPACITY
        self.show_anim = 1.0                  # 0.0 = 完全隱藏, 1.0 = 完全可見
        self.target_show_anim = 1.0

        # ── 畫布快取 ──
        self._canvas_cache = None
        self._canvas_dirty = True

        # ── 按鈕列表 ──
        self.buttons = []                     # List[Button3D]

        # ── 包圍球半徑（快速碰撞檢測）──
        self.bounding_radius = math.sqrt((w_3d / 2) ** 2 + (h_3d / 2) ** 2) + 30

        # ── 時間追蹤 ──
        self.creation_time = time.time()

        # ── 內部動畫計數器 ──
        self._anim_tick = 0                   # 掃描線動畫用

    # ────────────────────────────────────────────────────────────
    #  3D 幾何：取得本地角點
    # ────────────────────────────────────────────────────────────
    def get_local_corners(self):
        """
        回傳面板在本地座標系中的四個角點。
        原點在面板中心，Z = 0。

        Returns
        -------
        np.ndarray, shape (4, 3)
            四個角點：左上、右上、右下、左下。
        """
        hw = self.w_3d / 2.0
        hh = self.h_3d / 2.0
        return np.array([
            [-hw, -hh, 0.0],   # 左上
            [ hw, -hh, 0.0],   # 右上
            [ hw,  hh, 0.0],   # 右下
            [-hw,  hh, 0.0],   # 左下
        ], dtype=float)

    # ────────────────────────────────────────────────────────────
    #  3D 幾何：取得世界座標角點
    # ────────────────────────────────────────────────────────────
    def get_world_corners(self, R=None):
        """
        將本地角點轉換到世界座標。

        Parameters
        ----------
        R : np.ndarray or None
            3x3 旋轉矩陣。若為 None，則依 self.rot 計算。

        Returns
        -------
        np.ndarray, shape (4, 3)
            世界座標角點。
        """
        if R is None:
            R = get_rotation_matrix(*self.rot)
        local = self.get_local_corners()
        world = (R @ local.T).T + self.T
        return world

    # ────────────────────────────────────────────────────────────
    #  姿態更新：EMA 平滑
    # ────────────────────────────────────────────────────────────
    def update_pose(self):
        """
        使用指數移動平均 (EMA) 平滑更新位置和旋轉。
        同時推進顯示/隱藏動畫。
        """
        a_pos = self.smooth_alpha_pos
        a_rot = self.smooth_alpha_rot

        # ── 位置平滑 ──
        self.T = self.T * (1.0 - a_pos) + self.target_T * a_pos

        # ── 旋轉平滑 ──
        self.rot = self.rot * (1.0 - a_rot) + self.target_rot * a_rot

        # ── 顯示/隱藏動畫 ──
        anim_speed = 0.08
        if abs(self.show_anim - self.target_show_anim) > 0.005:
            self.show_anim += (self.target_show_anim - self.show_anim) * anim_speed
        else:
            self.show_anim = self.target_show_anim

        # 動畫到 0 時設為不可見
        if self.show_anim < 0.01 and self.target_show_anim <= 0.0:
            self.visible = False
            self.show_anim = 0.0

        # ── 動畫計數器 ──
        self._anim_tick += 1

    # ────────────────────────────────────────────────────────────
    #  畫布管理
    # ────────────────────────────────────────────────────────────
    def mark_dirty(self):
        """標記畫布需要重繪。"""
        self._canvas_dirty = True

    def get_canvas(self):
        """
        回傳畫布內容。若畫布已髒或尚未建立，則重新繪製。

        Returns
        -------
        np.ndarray
            畫布影像 (canvas_h, canvas_w, 3)，BGR 格式。
        """
        if self._canvas_dirty or self._canvas_cache is None:
            self._canvas_cache = self.draw_canvas()
            self._canvas_dirty = False
        return self._canvas_cache

    # ────────────────────────────────────────────────────────────
    #  draw_canvas — 子類覆寫入口
    # ────────────────────────────────────────────────────────────
    def draw_canvas(self):
        """
        繪製 2D 畫布內容。子類應覆寫此方法。
        基礎實作：繪製科幻格線背景 + 邊框。

        Returns
        -------
        np.ndarray
            畫布影像 (canvas_h, canvas_w, 3)。
        """
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        self._draw_base_background(canvas)
        return canvas

    # ────────────────────────────────────────────────────────────
    #  _draw_base_background — 科幻風格基底背景
    # ────────────────────────────────────────────────────────────
    def _draw_base_background(self, canvas):
        """
        在畫布上繪製科幻風格的基底背景，包含：
          • 深色底板
          • 格線圖案
          • 動態水平掃描線
          • 外框與角落裝飾
          • 標題欄區域
        """
        h, w = canvas.shape[:2]

        # ── 深色底板 ──
        canvas[:] = CLR_DARK_PANEL

        # ── 格線圖案 ──
        grid_spacing = 20
        for gx in range(0, w, grid_spacing):
            cv2.line(canvas, (gx, 0), (gx, h), CLR_DARK_GRID, 1)
        for gy in range(0, h, grid_spacing):
            cv2.line(canvas, (0, gy), (w, gy), CLR_DARK_GRID, 1)

        # ── 動態水平掃描線（每幀向下移動）──
        scan_y = (self._anim_tick * 2) % h
        scan_color = tuple(int(c * 0.3) for c in CLR_CYAN)
        cv2.line(canvas, (0, scan_y), (w, scan_y), scan_color, 1, cv2.LINE_AA)
        # 掃描線上下方的漸淡效果
        for offset in range(1, 6):
            alpha = 0.3 * (1.0 - offset / 6.0)
            fade_color = tuple(int(c * alpha) for c in CLR_CYAN)
            if 0 <= scan_y - offset < h:
                cv2.line(canvas, (0, scan_y - offset), (w, scan_y - offset),
                         fade_color, 1)
            if 0 <= scan_y + offset < h:
                cv2.line(canvas, (0, scan_y + offset), (w, scan_y + offset),
                         fade_color, 1)

        # ── 外邊框 ──
        border = PANEL_BORDER_THICKNESS
        cv2.rectangle(canvas, (border, border), (w - border - 1, h - border - 1),
                      CLR_CYAN_DIM, border)

        # ── 角落裝飾 ──
        corner_len = PANEL_CORNER_BRACKET_LEN
        corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
        for cx, cy in corners:
            dx = 1 if cx == 0 else -1
            dy = 1 if cy == 0 else -1
            cv2.line(canvas, (cx, cy), (cx + dx * corner_len, cy),
                     CLR_CYAN, 2, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy), (cx, cy + dy * corner_len),
                     CLR_CYAN, 2, cv2.LINE_AA)

        # ── 標題欄分隔線 ──
        title_y = 28
        cv2.line(canvas, (5, title_y), (w - 5, title_y), CLR_CYAN_DIM, 1, cv2.LINE_AA)

    # ────────────────────────────────────────────────────────────
    #  _draw_rounded_rect — 圓角矩形繪製
    # ────────────────────────────────────────────────────────────
    def _draw_rounded_rect(self, canvas, pt1, pt2, color, thickness=1, radius=8):
        """
        繪製圓角矩形。

        Parameters
        ----------
        canvas : np.ndarray
            目標畫布。
        pt1 : tuple(int, int)
            左上角。
        pt2 : tuple(int, int)
            右下角。
        color : tuple
            BGR 顏色。
        thickness : int
            線條粗細。
        radius : int
            圓角半徑。
        """
        _draw_rounded_rect_static(canvas, pt1, pt2, color, thickness, radius)

    # ════════════════════════════════════════════════════════════
    #  render_to_world — 核心渲染方法
    #  將 2D 畫布透過 Homography 投影到攝影機的 3D 透視視圖
    # ════════════════════════════════════════════════════════════
    def render_to_world(self, frame, f, cx, cy):
        """
        將面板的 2D 畫布投影到 3D 透視攝影機視圖上。

        工作流程：
          1. 取得面板的 3D 世界座標角點
          2. 透過針孔攝影機模型投影到 2D 螢幕
          3. 計算畫布角點 → 投影角點的 Homography
          4. 透視變形 (warpPerspective) 畫布到螢幕空間
          5. 使用遮罩進行 alpha 混合
          6. 繪製科幻風格角括弧和軸向指示器

        Parameters
        ----------
        frame : np.ndarray
            攝影機影像幀 (H, W, 3)。
        f : float
            焦距 (px)。
        cx : float
            主點 X (px)。
        cy : float
            主點 Y (px)。

        Returns
        -------
        np.ndarray
            修改後的影像幀。
        """
        # ── 不可見時跳過 ──
        if not self.visible or self.show_anim < 0.01:
            return frame

        frame_h, frame_w = frame.shape[:2]

        # ── 取得旋轉矩陣與世界角點 ──
        R = get_rotation_matrix(*self.rot)
        world_corners = self.get_world_corners(R)

        # ── 投影四角到 2D 螢幕空間 ──
        pts_2d = []
        all_visible = True
        for pt3d in world_corners:
            pt2d = project_point(pt3d, f, cx, cy)
            if pt2d is None or pt3d[2] <= 0:
                all_visible = False
                break
            pts_2d.append(pt2d)

        if not all_visible or len(pts_2d) != 4:
            return frame

        dst_pts = np.array(pts_2d, dtype=np.float32)

        # ── 檢查投影面積是否過小或退化 ──
        # 使用 Shoelace 公式計算多邊形面積
        def _poly_area(pts):
            n = len(pts)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += pts[i][0] * pts[j][1]
                area -= pts[j][0] * pts[i][1]
            return abs(area) / 2.0

        proj_area = _poly_area(dst_pts)
        if proj_area < 100:
            return frame  # 面積太小，不繪製

        # ── 畫布角點（來源）──
        src_pts = np.array([
            [0, 0],
            [self.canvas_w - 1, 0],
            [self.canvas_w - 1, self.canvas_h - 1],
            [0, self.canvas_h - 1],
        ], dtype=np.float32)

        # ── 計算 Homography 矩陣 ──
        H, status = cv2.findHomography(src_pts, dst_pts)
        if H is None:
            return frame

        # ── 取得畫布 ──
        canvas = self.get_canvas()
        if canvas is None:
            return frame

        # ── 透視變形 ──
        warped = cv2.warpPerspective(
            canvas, H, (frame_w, frame_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # ── 建立遮罩：非零像素區域 ──
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # ── 計算有效混合不透明度 ──
        # 結合面板基礎不透明度和顯示/隱藏動畫
        effective_alpha = self.opacity * self.show_anim

        # ── Alpha 混合 ──
        # 方法：在遮罩區域內進行加權混合
        mask_3ch = cv2.merge([mask, mask, mask])
        mask_float = mask_3ch.astype(np.float32) / 255.0

        # 混合公式: output = frame * (1 - alpha * mask) + warped * alpha * mask
        blended = frame.astype(np.float32) * (1.0 - effective_alpha * mask_float) + \
                  warped.astype(np.float32) * effective_alpha * mask_float

        frame = np.clip(blended, 0, 255).astype(np.uint8)

        # ── 繪製科幻角括弧 ──
        bracket_len = PANEL_CORNER_BRACKET_LEN
        bracket_color = CLR_CYAN
        # 根據動畫調整亮度
        breathe = 0.7 + 0.3 * math.sin(time.time() * 2.0)
        bracket_color = tuple(
            int(c * breathe * self.show_anim) for c in bracket_color
        )

        for i in range(4):
            p1 = tuple(dst_pts[i].astype(int))
            p2 = tuple(dst_pts[(i + 1) % 4].astype(int))

            # 計算邊向量
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            edge_len = max(math.sqrt(dx ** 2 + dy ** 2), 1e-6)
            ndx = dx / edge_len
            ndy = dy / edge_len

            # 從角點出發，沿兩條邊繪製短線段
            bracket_end_x = int(p1[0] + ndx * bracket_len)
            bracket_end_y = int(p1[1] + ndy * bracket_len)
            cv2.line(frame, p1, (bracket_end_x, bracket_end_y),
                     bracket_color, 2, cv2.LINE_AA)

            # 反向邊的括弧（從 p2 出發回頭）
            prev_i = (i - 1) % 4
            p_prev = tuple(dst_pts[prev_i].astype(int))
            dx2 = p_prev[0] - p1[0]
            dy2 = p_prev[1] - p1[1]
            edge_len2 = max(math.sqrt(dx2 ** 2 + dy2 ** 2), 1e-6)
            ndx2 = dx2 / edge_len2
            ndy2 = dy2 / edge_len2

            bracket_end_x2 = int(p1[0] + ndx2 * bracket_len)
            bracket_end_y2 = int(p1[1] + ndy2 * bracket_len)
            cv2.line(frame, p1, (bracket_end_x2, bracket_end_y2),
                     bracket_color, 2, cv2.LINE_AA)

        # ── 拖曳中的邊緣發光效果 ──
        if self.is_dragging:
            glow_color = CLR_YELLOW
            for i in range(4):
                p1 = tuple(dst_pts[i].astype(int))
                p2 = tuple(dst_pts[(i + 1) % 4].astype(int))
                cv2.line(frame, p1, p2, glow_color, 2, cv2.LINE_AA)

        # ── XYZ 軸向指示器（面板中心）──
        center_3d = self.T.copy()
        axis_len = min(self.w_3d, self.h_3d) * 0.15

        axes = [
            (np.array([axis_len, 0, 0]), CLR_RED),    # X 軸 — 紅
            (np.array([0, axis_len, 0]), CLR_GREEN),   # Y 軸 — 綠
            (np.array([0, 0, axis_len]), CLR_BLUE),    # Z 軸 — 藍
        ]

        center_2d = project_point(center_3d, f, cx, cy)
        if center_2d is not None:
            for axis_vec, axis_color in axes:
                end_3d = center_3d + R @ axis_vec
                end_2d = project_point(end_3d, f, cx, cy)
                if end_2d is not None:
                    c2d = tuple(int(v) for v in center_2d)
                    e2d = tuple(int(v) for v in end_2d)
                    # 根據 show_anim 調整軸線透明度
                    dim_color = tuple(
                        int(c * 0.6 * self.show_anim) for c in axis_color
                    )
                    cv2.line(frame, c2d, e2d, dim_color, 1, cv2.LINE_AA)

        return frame

    # ════════════════════════════════════════════════════════════
    #  拖曳互動
    # ════════════════════════════════════════════════════════════
    def start_drag(self, hand_pos, hand_rot=None):
        """
        開始拖曳操作。記錄起始位置作為偏移參考。

        Parameters
        ----------
        hand_pos : np.ndarray
            手部 3D 位置。
        hand_rot : np.ndarray or None
            手部歐拉角旋轉（選用）。
        """
        self.is_dragging = True
        self.drag_start_hand_pos = hand_pos.copy()
        self.drag_start_screen_pos = self.target_T.copy()
        self.drag_start_hand_rot = hand_rot.copy() if hand_rot is not None else None
        self.drag_start_screen_rot = self.target_rot.copy()

    def update_drag(self, hand_pos, hand_rot=None):
        """
        拖曳期間更新面板的目標位置和旋轉。

        Parameters
        ----------
        hand_pos : np.ndarray
            當前手部 3D 位置。
        hand_rot : np.ndarray or None
            當前手部歐拉角旋轉。
        """
        if not self.is_dragging or self.drag_start_hand_pos is None:
            return

        # 位置差量
        delta_pos = hand_pos - self.drag_start_hand_pos
        self.target_T = self.drag_start_screen_pos + delta_pos

        # 旋轉差量（若可用）
        if hand_rot is not None and self.drag_start_hand_rot is not None:
            delta_rot = hand_rot - self.drag_start_hand_rot
            self.target_rot = self.drag_start_screen_rot + delta_rot * 0.5  # 旋轉敏感度降低

        self.mark_dirty()

    def end_drag(self):
        """結束拖曳操作。"""
        self.is_dragging = False
        self.drag_start_hand_pos = None
        self.drag_start_screen_pos = None
        self.drag_start_hand_rot = None
        self.drag_start_screen_rot = None

    # ════════════════════════════════════════════════════════════
    #  觸控檢測
    # ════════════════════════════════════════════════════════════
    def check_touch(self, finger_tip_3d):
        """
        檢查手指尖是否觸碰到面板。

        工作流程：
          1. 將手指尖轉換到面板的本地座標系
          2. 檢查 XY 是否在面板範圍內（含容錯）
          3. 檢查 Z 深度決定是懸停還是點擊
          4. 對按鈕進行碰撞檢測

        Parameters
        ----------
        finger_tip_3d : np.ndarray
            手指尖的世界 3D 座標。

        Returns
        -------
        tuple(np.ndarray or None, bool, str or None)
            (local_pos, is_active_click, hit_button_action_id)
            若不在範圍內則回傳 (None, False, None)。
        """
        if not self.visible or self.show_anim < 0.3:
            return None, False, None

        # ── 快速包圍球排除 ──
        dist = np.linalg.norm(finger_tip_3d - self.T)
        if dist > self.bounding_radius:
            return None, False, None

        # ── 轉換到本地座標 ──
        R = get_rotation_matrix(*self.rot)
        local_pos = R.T @ (finger_tip_3d - self.T)

        hw = self.w_3d / 2.0 + TOUCH_BOUNDARY_PADDING
        hh = self.h_3d / 2.0 + TOUCH_BOUNDARY_PADDING

        # ── 檢查 XY 邊界 ──
        if abs(local_pos[0]) > hw or abs(local_pos[1]) > hh:
            # 重置所有按鈕懸停狀態
            for btn in self.buttons:
                btn.is_hovered = False
            return None, False, None

        # ── 檢查 Z 深度 ──
        z = local_pos[2]
        hover_near, hover_far = TOUCH_HOVER_RANGE
        click_deep, click_shallow = TOUCH_CLICK_RANGE

        is_hover = hover_near <= z <= hover_far
        is_click = click_deep <= z <= click_shallow

        # ── 將本地 XY 映射到畫布座標 ──
        canvas_x = int((local_pos[0] + self.w_3d / 2.0) / self.w_3d * self.canvas_w)
        canvas_y = int((local_pos[1] + self.h_3d / 2.0) / self.h_3d * self.canvas_h)
        canvas_x = max(0, min(canvas_x, self.canvas_w - 1))
        canvas_y = max(0, min(canvas_y, self.canvas_h - 1))

        # ── 按鈕碰撞檢測 ──
        hit_button_id = None
        for btn in self.buttons:
            if btn.hit_test(canvas_x, canvas_y):
                btn.is_hovered = True
                if is_click and not btn.is_pressed:
                    btn.is_pressed = True
                    btn.press_anim = 1.0
                    if btn.toggle:
                        btn.is_active = not btn.is_active
                    hit_button_id = btn.action_id
                    self.mark_dirty()
                elif not is_click:
                    btn.is_pressed = False
            else:
                btn.is_hovered = False
                btn.is_pressed = False

        if is_hover or is_click:
            self.mark_dirty()
            return local_pos, is_click, hit_button_id

        return None, False, None

    # ════════════════════════════════════════════════════════════
    #  可見性控制
    # ════════════════════════════════════════════════════════════
    def show(self):
        """顯示面板（帶淡入動畫）。"""
        self.visible = True
        self.target_show_anim = 1.0

    def hide(self):
        """隱藏面板（帶淡出動畫）。"""
        self.target_show_anim = 0.0
        # 當 show_anim 衰減到 0 時，update_pose() 會自動設 visible = False

    def toggle_visibility(self):
        """切換面板可見性。"""
        if self.visible and self.target_show_anim > 0.5:
            self.hide()
        else:
            self.show()

    # ════════════════════════════════════════════════════════════
    #  深度排序
    # ════════════════════════════════════════════════════════════
    def get_depth(self):
        """
        回傳面板的 Z 深度，用於畫家演算法排序。
        深度越大（越遠）先繪製。

        Returns
        -------
        float
            面板的 Z 座標。
        """
        return self.T[2]

    # ════════════════════════════════════════════════════════════
    #  序列化
    # ════════════════════════════════════════════════════════════
    def to_dict(self):
        """
        將面板狀態序列化為字典，用於儲存佈局。

        Returns
        -------
        dict
            面板的位置、旋轉、可見性等狀態。
        """
        return {
            "panel_id": self.panel_id,
            "position": self.T.tolist(),
            "rotation": self.rot.tolist(),
            "visible": self.visible,
            "opacity": self.opacity,
        }

    def from_dict(self, data):
        """
        從字典恢復面板狀態。

        Parameters
        ----------
        data : dict
            先前由 to_dict() 產生的狀態字典。
        """
        if "position" in data:
            pos = np.array(data["position"], dtype=float)
            self.T = pos.copy()
            self.target_T = pos.copy()
        if "rotation" in data:
            rot = np.array(data["rotation"], dtype=float)
            self.rot = rot.copy()
            self.target_rot = rot.copy()
        if "visible" in data:
            self.visible = data["visible"]
            self.show_anim = 1.0 if self.visible else 0.0
            self.target_show_anim = self.show_anim
        if "opacity" in data:
            self.opacity = data["opacity"]
