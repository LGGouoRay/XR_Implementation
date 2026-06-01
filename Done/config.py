# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║            XR HUD SYSTEM — CENTRAL CONFIGURATION            ║
║  All tunable constants, colors, thresholds, and settings    ║
║  are centralized here for easy maintenance and tweaking.    ║
╚══════════════════════════════════════════════════════════════╝
"""
import numpy as np


# ============================================================
#  攝影機設定
# ============================================================
CAMERA_ID = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# ============================================================
#  MediaPipe 手部追蹤設定
# ============================================================
MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.7

# ============================================================
#  3D 空間設定
# ============================================================
INITIAL_PANEL_DEPTH = 450.0          # 面板初始 Z 深度 (mm)
DEPTH_CALIBRATION_WIDTH = 120.0      # 深度校正基準寬度 (px)
DEPTH_CALIBRATION_DIST = 400.0       # 深度校正基準距離 (mm)
DEPTH_Z_FACTOR = 1.2                 # MediaPipe z 值對物理深度的倍率

# ============================================================
#  手勢辨識門檻
# ============================================================
PINCH_THRESHOLD = 0.25               # 捏合判定閾值 (比例)
GRAB_RANGE = 250.0                   # 抓取觸發距離 (mm)
FIST_CURL_THRESHOLD = 0.6            # 握拳判定 — 手指彎曲比例
OPEN_HAND_EXTEND_THRESHOLD = 0.8     # 張手判定 — 手指伸直比例
THUMBS_UP_ANGLE_THRESHOLD = 45.0     # 比讚判定 — 拇指朝上角度 (度)
SWIPE_VELOCITY_THRESHOLD = 0.15      # 揮手速度閾值 (歸一化/幀)
GESTURE_COOLDOWN_FRAMES = 15         # 手勢觸發冷卻幀數
GESTURE_CONFIRM_FRAMES = 3           # 手勢確認所需連續幀數

# ============================================================
#  面板共用設定
# ============================================================
PANEL_SMOOTH_ALPHA_POS = 0.25        # 位置平滑係數 (EMA)
PANEL_SMOOTH_ALPHA_ROT = 0.20        # 旋轉平滑係數 (EMA)
PANEL_OPACITY = 0.75                 # 面板不透明度
PANEL_CORNER_BRACKET_LEN = 25        # 角括弧長度 (px)
PANEL_BORDER_THICKNESS = 2           # 框線粗細

# 面板碰觸檢測
TOUCH_HOVER_RANGE = (0.0, 80.0)      # 懸停距離範圍 (mm) — [近, 遠]
TOUCH_CLICK_RANGE = (-60.0, -5.0)    # 點擊穿透範圍 (mm) — [穿透深, 穿透淺]
TOUCH_BOUNDARY_PADDING = 20.0        # 面板邊界碰觸容錯 (mm)

# ============================================================
#  面板預設尺寸與位置
# ============================================================
# 主 HUD 面板
MAIN_HUD_SIZE_3D = (300, 180)        # (寬, 高) in 3D mm
MAIN_HUD_CANVAS_SIZE = (440, 280)    # (寬, 高) in 2D px
MAIN_HUD_INITIAL_POS = np.array([0.0, -30.0, 450.0])

# 系統監控面板
SYSMON_SIZE_3D = (220, 160)
SYSMON_CANVAS_SIZE = (320, 220)
SYSMON_INITIAL_POS = np.array([-280.0, -80.0, 500.0])

# 媒體控制面板
MEDIA_SIZE_3D = (240, 140)
MEDIA_CANVAS_SIZE = (360, 200)
MEDIA_INITIAL_POS = np.array([280.0, -80.0, 500.0])

# ============================================================
#  色彩系統 (BGR 格式)
# ============================================================
# 主色系 — 青色為主調
CLR_CYAN = (255, 255, 0)
CLR_CYAN_DIM = (128, 128, 0)
CLR_CYAN_BRIGHT = (255, 255, 100)

# 強調色
CLR_YELLOW = (0, 255, 255)
CLR_YELLOW_DIM = (0, 180, 180)
CLR_ORANGE = (0, 165, 255)
CLR_MAGENTA = (255, 0, 255)

# 基礎色
CLR_WHITE = (255, 255, 255)
CLR_RED = (0, 0, 255)
CLR_GREEN = (0, 255, 0)
CLR_BLUE = (255, 0, 0)

# 暗色系
CLR_DARK_BG = (15, 15, 15)
CLR_DARK_GRID = (25, 25, 10)
CLR_DARK_PANEL = (10, 10, 10)
CLR_DARK_OVERLAY = (20, 20, 20)

# 漸層色 (用於進度條/儀表盤)
CLR_GRADIENT_START = (255, 200, 0)   # 青色偏亮
CLR_GRADIENT_END = (200, 0, 200)     # 洋紅色

# 按鈕狀態配色
CLR_BTN_NORMAL = CLR_CYAN
CLR_BTN_HOVER = CLR_CYAN_BRIGHT
CLR_BTN_ACTIVE = CLR_YELLOW
CLR_BTN_PRESSED = CLR_RED
CLR_BTN_TEXT = (0, 0, 0)

# HUD 元素配色
CLR_HUD_TITLE = CLR_CYAN
CLR_HUD_DATA = CLR_WHITE
CLR_HUD_WARN = CLR_ORANGE
CLR_HUD_OK = CLR_GREEN
CLR_HUD_ERR = CLR_RED

# ============================================================
#  粒子系統設定
# ============================================================
PARTICLE_AMBIENT_COUNT = 60          # 環境漂浮粒子數量
PARTICLE_AMBIENT_MIN_SIZE = 1        # 最小粒子半徑
PARTICLE_AMBIENT_MAX_SIZE = 3        # 最大粒子半徑
PARTICLE_AMBIENT_SPEED = 0.8         # 漂浮速度
PARTICLE_AMBIENT_FADE_SPEED = 0.02   # 呼吸淡入淡出速度

PARTICLE_BURST_COUNT = 30            # 爆發粒子數量 (按鈕點擊)
PARTICLE_BURST_SPEED = 4.0           # 爆發速度
PARTICLE_BURST_LIFETIME = 25         # 爆發粒子壽命 (幀)
PARTICLE_BURST_GRAVITY = 0.15        # 爆發粒子重力

PARTICLE_TRAIL_LENGTH = 12           # 手指軌跡殘影長度
PARTICLE_TRAIL_FADE = 0.85           # 軌跡淡出係數

# ============================================================
#  HUD 覆蓋層效果設定
# ============================================================
HUD_SCANLINE_ALPHA = 0.08            # 掃描線不透明度
HUD_SCANLINE_SPACING = 3             # 掃描線間距 (px)
HUD_SCANLINE_SCROLL_SPEED = 1.5      # 掃描線捲動速度

HUD_BREATHE_SPEED = 0.03             # 呼吸燈脈搏速度
HUD_BREATHE_MIN_ALPHA = 0.6          # 呼吸燈最低亮度
HUD_BREATHE_MAX_ALPHA = 1.0          # 呼吸燈最高亮度

HUD_CORNER_SIZE = 40                 # 角落裝飾尺寸 (px)
HUD_CORNER_THICKNESS = 2             # 角落裝飾線條粗細

HUD_RADAR_SIZE = 100                 # 迷你雷達直徑 (px)
HUD_RADAR_POS = (70, 70)             # 雷達中心位置 (相對右下角)

# ============================================================
#  影像濾鏡設定
# ============================================================
FILTER_MODES = [
    {"name": "B&W",         "sat": 0.0,  "icon": "◐"},
    {"name": "NORMAL",      "sat": 1.0,  "icon": "◉"},
    {"name": "VIVID",       "sat": 2.5,  "icon": "◈"},
    {"name": "EDGE",        "sat": 1.0,  "icon": "▦"},
    {"name": "COOL",        "sat": 1.0,  "icon": "❄"},
    {"name": "WARM",        "sat": 1.0,  "icon": "☀"},
    {"name": "NEGATIVE",    "sat": 1.0,  "icon": "⊘"},
    {"name": "POSTERIZE",   "sat": 1.0,  "icon": "▧"},
]

# ============================================================
#  效能與自適應品質
# ============================================================
TARGET_FPS = 30
ADAPTIVE_QUALITY = True
FPS_LOW_THRESHOLD = 20               # FPS 低於此值啟動降級
FPS_HISTORY_SIZE = 120               # FPS 歷史記錄長度

# ============================================================
#  截圖系統
# ============================================================
SCREENSHOT_DIR = "screenshots"
SCREENSHOT_FLASH_DURATION = 8        # 截圖閃光動畫持續幀數

# ============================================================
#  面板佈局記憶
# ============================================================
LAYOUT_SAVE_FILE = "panel_layout.json"
