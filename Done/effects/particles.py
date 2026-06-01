# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║               XR HUD 粒子系統 — Particle System              ║
║  包含環境漂浮粒子、手指軌跡殘影、按鈕爆發粒子三大子系統。    ║
║  使用 NumPy 批量運算優化位置/速度更新，以維持高幀率。        ║
╚══════════════════════════════════════════════════════════════╝
"""
import cv2
import numpy as np
import math
import time
from collections import deque

# 將上層目錄加入搜尋路徑以匯入中央設定
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import *


# ================================================================
#  單一粒子資料結構 — 使用 __slots__ 減少記憶體開銷
# ================================================================
class Particle:
    """
    單個粒子實體。
    儲存位置 (x, y)、速度 (vx, vy)、生命週期、尺寸、顏色與透明度。
    particle_type 可為 'ambient'（環境）或 'burst'（爆發）。
    """
    __slots__ = [
        'x', 'y',               # 目前位置 (像素)
        'vx', 'vy',             # 速度向量 (像素/幀)
        'life', 'max_life',     # 剩餘壽命 / 最大壽命 (幀)
        'size',                 # 粒子半徑 (像素)
        'color',                # BGR 色彩元組
        'alpha',                # 當前透明度 [0.0 ~ 1.0]
        'particle_type',        # 類型標籤: 'ambient' | 'burst'
        'phase',                # 呼吸動畫相位偏移 (僅 ambient 使用)
    ]

    def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0,
                 life=100, size=2, color=(255, 255, 0),
                 alpha=1.0, particle_type='ambient', phase=0.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.size = size
        self.color = color
        self.alpha = alpha
        self.particle_type = particle_type
        self.phase = phase


# ================================================================
#  粒子系統主類別
# ================================================================
class ParticleSystem:
    """
    管理所有粒子子系統的核心控制器。

    子系統包含：
      1. Ambient（環境粒子）— 持續漂浮，營造科幻氛圍
      2. Trail（軌跡殘影）— 跟隨手指移動產生拖尾效果
      3. Burst（爆發粒子）— 事件觸發（按鈕點擊等）時向外噴射
    """

    def __init__(self, screen_w, screen_h):
        """
        初始化粒子系統。

        Args:
            screen_w: 螢幕寬度 (像素)
            screen_h: 螢幕高度 (像素)
        """
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.particles = []            # 存放所有粒子物件的列表
        self.start_time = time.time()  # 系統啟動時間戳，用於動畫計時

        # ── 初始化環境漂浮粒子 ──
        self._init_ambient_particles()

        # ── 手指軌跡子系統 ──
        self.trail_positions = deque(maxlen=PARTICLE_TRAIL_LENGTH)
        self.trail_active = False      # 軌跡是否正在繪製

        # ── 用於批量運算的 NumPy 陣列快取 ──
        # 每幀更新時重新構建，避免逐粒子迭代的 Python 瓶頸
        self._np_positions = None
        self._np_velocities = None

    # ============================================================
    #  環境粒子初始化
    # ============================================================
    def _init_ambient_particles(self):
        """
        在螢幕範圍內隨機散佈 PARTICLE_AMBIENT_COUNT 個環境粒子。
        每個粒子具有：
          - 隨機位置 (均勻分佈於螢幕範圍)
          - 緩慢隨機速度 (PARTICLE_AMBIENT_SPEED 為上限)
          - 隨機尺寸 (介於 MIN 和 MAX 之間)
          - 以青色為基調，帶有輕微隨機色相偏移的顏色
          - 隨機相位偏移，用於呼吸動畫的去同步化
        """
        for _ in range(PARTICLE_AMBIENT_COUNT):
            # 隨機位置 — 均勻散佈於整個螢幕
            x = np.random.uniform(0, self.screen_w)
            y = np.random.uniform(0, self.screen_h)

            # 隨機速度 — 方向任意，大小受 PARTICLE_AMBIENT_SPEED 約束
            angle = np.random.uniform(0, 2 * math.pi)
            speed = np.random.uniform(0.2, PARTICLE_AMBIENT_SPEED)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed

            # 隨機尺寸
            size = np.random.randint(PARTICLE_AMBIENT_MIN_SIZE,
                                     PARTICLE_AMBIENT_MAX_SIZE + 1)

            # 色彩 — 以青色 (255, 255, 0) 為基底，加入隨機偏移
            # BGR 格式：B=200~255, G=200~255, R=0~60
            b = np.random.randint(200, 256)
            g = np.random.randint(200, 256)
            r = np.random.randint(0, 61)
            color = (int(b), int(g), int(r))

            # 隨機相位偏移 — 使各粒子的呼吸動畫不同步
            phase = np.random.uniform(0, 2 * math.pi)

            # 環境粒子壽命設為極大值（永不消亡）
            p = Particle(
                x=x, y=y, vx=vx, vy=vy,
                life=999999, size=size, color=color,
                alpha=np.random.uniform(0.3, 0.8),
                particle_type='ambient',
                phase=phase
            )
            self.particles.append(p)

    # ============================================================
    #  爆發粒子生成
    # ============================================================
    def spawn_burst(self, x, y, count=None, color=None):
        """
        在指定位置 (x, y) 產生一次粒子爆發。

        Args:
            x: 爆發中心 X 座標
            y: 爆發中心 Y 座標
            count: 粒子數量，預設 PARTICLE_BURST_COUNT
            color: 粒子顏色 (BGR)，預設從青/黃/白中隨機選取
        """
        if count is None:
            count = PARTICLE_BURST_COUNT

        # 預設可選顏色池 — 科幻風格的青、黃、白
        color_pool = [CLR_CYAN, CLR_CYAN_BRIGHT, CLR_YELLOW, CLR_WHITE]

        for _ in range(count):
            # 徑向隨機方向
            angle = np.random.uniform(0, 2 * math.pi)
            # 速度帶有 ±30% 隨機變異
            speed = PARTICLE_BURST_SPEED * np.random.uniform(0.7, 1.3)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed

            # 顏色：使用指定色或從色池隨機選取
            c = color if color else color_pool[np.random.randint(0, len(color_pool))]

            # 尺寸：爆發粒子稍大一些 (2~5 px)
            size = np.random.randint(2, 6)

            p = Particle(
                x=float(x), y=float(y),
                vx=vx, vy=vy,
                life=PARTICLE_BURST_LIFETIME,
                size=size, color=c,
                alpha=1.0,
                particle_type='burst',
                phase=0.0
            )
            self.particles.append(p)

    # ============================================================
    #  手指軌跡更新
    # ============================================================
    def update_trail(self, finger_x, finger_y, active=True):
        """
        更新手指軌跡位置佇列。

        Args:
            finger_x: 手指 X 座標
            finger_y: 手指 Y 座標
            active: 是否正在追蹤 (False 時軌跡自然消散)
        """
        self.trail_active = active

        if active:
            # 記錄位置及時間戳，用於後續淡出計算
            self.trail_positions.append((finger_x, finger_y, time.time()))
        # 如果非活躍狀態，不再新增點位，既有軌跡會在渲染時自然縮短

    # ============================================================
    #  物理更新 — 每幀呼叫
    # ============================================================
    def update(self):
        """
        更新所有粒子的物理狀態。

        - Ambient: 移動位置，螢幕邊緣環繞，更新呼吸透明度
        - Burst: 移動位置，施加重力，遞減壽命，移除死亡粒子
        - Trail: 由 deque maxlen 自動裁剪
        """
        current_time = time.time() - self.start_time
        alive_particles = []

        # ── 分離環境粒子與爆發粒子以進行批量處理 ──
        ambient_list = []
        burst_list = []

        for p in self.particles:
            if p.particle_type == 'ambient':
                ambient_list.append(p)
            else:
                burst_list.append(p)

        # ── 批量更新環境粒子 (NumPy 向量化) ──
        if ambient_list:
            n = len(ambient_list)

            # 提取位置與速度到 NumPy 陣列
            pos = np.array([[p.x, p.y] for p in ambient_list], dtype=np.float32)
            vel = np.array([[p.vx, p.vy] for p in ambient_list], dtype=np.float32)
            phases = np.array([p.phase for p in ambient_list], dtype=np.float32)

            # 批量位置更新
            pos += vel

            # 螢幕邊緣環繞 — 從一側消失後從另一側出現
            pos[:, 0] = np.mod(pos[:, 0], self.screen_w)
            pos[:, 1] = np.mod(pos[:, 1], self.screen_h)

            # 批量計算呼吸透明度
            # alpha 在 0.15 ~ 0.7 之間以正弦波振盪
            breath = 0.15 + 0.55 * (0.5 + 0.5 * np.sin(
                current_time * PARTICLE_AMBIENT_FADE_SPEED * 2 * math.pi + phases
            ))

            # 將計算結果寫回粒子物件
            for i, p in enumerate(ambient_list):
                p.x = float(pos[i, 0])
                p.y = float(pos[i, 1])
                p.alpha = float(breath[i])

            alive_particles.extend(ambient_list)

        # ── 批量更新爆發粒子 (NumPy 向量化) ──
        if burst_list:
            n = len(burst_list)

            pos = np.array([[p.x, p.y] for p in burst_list], dtype=np.float32)
            vel = np.array([[p.vx, p.vy] for p in burst_list], dtype=np.float32)
            lives = np.array([p.life for p in burst_list], dtype=np.float32)
            max_lives = np.array([p.max_life for p in burst_list], dtype=np.float32)

            # 施加重力 — Y 軸正方向為下
            vel[:, 1] += PARTICLE_BURST_GRAVITY

            # 位置更新
            pos += vel

            # 壽命遞減
            lives -= 1

            # 計算存活率 (用於 alpha 淡出與尺寸縮小)
            life_ratio = np.clip(lives / max_lives, 0.0, 1.0)

            # 將計算結果寫回並篩選存活粒子
            for i, p in enumerate(burst_list):
                if lives[i] <= 0:
                    continue  # 跳過已死亡的粒子
                p.x = float(pos[i, 0])
                p.y = float(pos[i, 1])
                p.vx = float(vel[i, 0])
                p.vy = float(vel[i, 1])
                p.life = int(lives[i])
                p.alpha = float(life_ratio[i])
                alive_particles.append(p)

        # ── 更新主粒子列表 ──
        self.particles = alive_particles

        # ── 軌跡裁剪 — 移除過期的位置點 ──
        if not self.trail_active and len(self.trail_positions) > 0:
            # 非活躍時，每幀移除最舊的一個點以產生消散效果
            if self.trail_positions:
                self.trail_positions.popleft()

    # ============================================================
    #  渲染 — 將粒子繪製到影像幀上
    # ============================================================
    def render(self, frame):
        """
        將所有粒子效果繪製到提供的影像幀上。

        渲染順序：
          1. 環境粒子 (小圓點 + 柔和光暈)
          2. 爆發粒子 (隨壽命縮小 + alpha 淡出)
          3. 手指軌跡 (連接線段 + 漸變粗細與透明度)

        使用加法混合 (additive blending) 實現發光效果。

        Args:
            frame: 輸入影像幀 (BGR, uint8)

        Returns:
            修改後的影像幀
        """
        # ── 繪製環境粒子 ──
        for p in self.particles:
            if p.particle_type == 'ambient':
                self._draw_glow_circle(
                    frame,
                    int(p.x), int(p.y),
                    p.size, p.color, p.alpha
                )

        # ── 繪製爆發粒子 ──
        for p in self.particles:
            if p.particle_type == 'burst':
                # 尺寸隨壽命比例縮小
                life_ratio = p.life / p.max_life if p.max_life > 0 else 0
                current_size = max(1, int(p.size * life_ratio))
                self._draw_glow_circle(
                    frame,
                    int(p.x), int(p.y),
                    current_size, p.color, p.alpha
                )

        # ── 繪製手指軌跡 ──
        self._render_trail(frame)

        return frame

    # ============================================================
    #  軌跡渲染
    # ============================================================
    def _render_trail(self, frame):
        """
        繪製手指軌跡殘影。
        線段從最新點到最舊點，粗細遞減、透明度逐漸降低。
        使用加法混合產生光暈效果。
        """
        if len(self.trail_positions) < 2:
            return

        trail_list = list(self.trail_positions)
        n = len(trail_list)

        for i in range(n - 1):
            # 計算此段線的透明度和粗細 — 越新的越亮越粗
            ratio = (i + 1) / n  # 0.0 (最舊) → 1.0 (最新)
            alpha = ratio * PARTICLE_TRAIL_FADE
            thickness = max(1, int(ratio * 6))  # 粗細 1~6

            # 顏色隨位置漸變：青色 → 白色
            b = int(200 + 55 * ratio)
            g = int(200 + 55 * ratio)
            r = int(50 * ratio)
            color = (b, g, r)

            pt1 = (int(trail_list[i][0]), int(trail_list[i][1]))
            pt2 = (int(trail_list[i + 1][0]), int(trail_list[i + 1][1]))

            # 建立線段的 ROI 覆蓋區，使用加法混合
            overlay = frame.copy()
            cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)

            # alpha 混合 — 使用加法模式增強發光感
            # 為了效能，直接使用 addWeighted 而非逐像素操作
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    # ============================================================
    #  光暈圓形繪製 (ROI 優化)
    # ============================================================
    def _draw_glow_circle(self, frame, x, y, radius, color, alpha):
        """
        繪製帶有柔和光暈效果的圓形粒子。

        為了效能，只在粒子周圍的 ROI (Region of Interest) 區域
        進行 alpha 混合，而非整張影像。

        Args:
            frame: 目標影像幀
            x, y: 圓心座標
            radius: 圓半徑
            color: BGR 色彩
            alpha: 透明度 [0.0 ~ 1.0]
        """
        if alpha < 0.01:
            return  # 完全透明，無需繪製

        # 光暈半徑為粒子半徑的 2.5 倍
        glow_radius = max(radius * 2 + 4, 6)

        # 計算 ROI 邊界 (裁剪至螢幕範圍內)
        x1 = max(0, x - glow_radius)
        y1 = max(0, y - glow_radius)
        x2 = min(self.screen_w, x + glow_radius + 1)
        y2 = min(self.screen_h, y + glow_radius + 1)

        # 如果 ROI 太小或超出範圍，跳過
        if x2 <= x1 or y2 <= y1:
            return

        # 提取 ROI 區域
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return

        # 在 ROI 上建立覆蓋層
        overlay = roi.copy()

        # 相對於 ROI 的圓心座標
        cx = x - x1
        cy = y - y1

        # 繪製外層光暈（較大、較暗）
        glow_color = (
            int(color[0] * 0.4),
            int(color[1] * 0.4),
            int(color[2] * 0.4)
        )
        cv2.circle(overlay, (cx, cy), glow_radius, glow_color, -1, cv2.LINE_AA)

        # 繪製核心粒子（較小、較亮）
        cv2.circle(overlay, (cx, cy), radius, color, -1, cv2.LINE_AA)

        # 使用加法混合 — 讓發光效果與背景疊加
        # 先進行 alpha 縮放
        glow_contribution = (overlay.astype(np.float32) * alpha).astype(np.uint8)

        # 加法混合：background + glow (裁剪至 255)
        frame[y1:y2, x1:x2] = cv2.add(roi, glow_contribution)

    # ============================================================
    #  公用工具方法
    # ============================================================
    def get_particle_count(self):
        """回傳當前存活粒子總數 (不含軌跡點)。"""
        return len(self.particles)

    def get_trail_length(self):
        """回傳當前軌跡點數量。"""
        return len(self.trail_positions)

    def clear_all(self):
        """清除所有粒子並重新初始化環境粒子。"""
        self.particles.clear()
        self.trail_positions.clear()
        self._init_ambient_particles()

    def clear_bursts(self):
        """僅清除爆發粒子，保留環境粒子。"""
        self.particles = [p for p in self.particles
                          if p.particle_type == 'ambient']

    def set_screen_size(self, w, h):
        """
        動態更新螢幕尺寸 (用於視窗調整大小時)。

        Args:
            w: 新寬度
            h: 新高度
        """
        self.screen_w = w
        self.screen_h = h
