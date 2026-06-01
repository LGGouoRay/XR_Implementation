# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║       GESTURE RECOGNIZER — 進階手勢辨識系統                   ║
║  支援 8 種手勢，包含確認緩衝、冷卻計時與回呼機制              ║
╚══════════════════════════════════════════════════════════════╝

手勢類型:
  PINCH      — 拇指+食指捏合 (抓取/拖曳)
  FIST       — 握拳 (關閉/最小化面板)
  OPEN_HAND  — 五指張開 (顯示所有面板)
  THUMBS_UP  — 比讚 (截圖)
  PEACE      — 比 V (切換面板)
  SWIPE_LEFT — 向左揮手
  SWIPE_RIGHT— 向右揮手
  POINT      — 食指指向 (觸碰/懸停)

辨識流程:
  1. 每幀計算五指的伸直/彎曲狀態
  2. 依優先順序檢查各手勢
  3. 確認緩衝: 手勢需連續 N 幀穩定才觸發
  4. 冷卻機制: 觸發後一段時間內不再重複觸發
  5. 回呼系統: 觸發時自動呼叫已註冊的處理函式
"""

import math
import numpy as np
from enum import Enum
from collections import deque
from config import (
    PINCH_THRESHOLD,
    FIST_CURL_THRESHOLD,
    OPEN_HAND_EXTEND_THRESHOLD,
    THUMBS_UP_ANGLE_THRESHOLD,
    SWIPE_VELOCITY_THRESHOLD,
    GESTURE_COOLDOWN_FRAMES,
    GESTURE_CONFIRM_FRAMES,
)


# ═════════════════════════════════════════════════════════════
#  手勢列舉
# ═════════════════════════════════════════════════════════════

class GestureType(Enum):
    """
    所有可辨識的手勢類型。

    數值順序同時代表辨識優先順序 (數值小 = 優先級高)。
    """
    NONE = 0             # 無手勢 / 未辨識
    PINCH = 1            # 捏合 — 拇指尖(4) 與食指尖(8) 靠近
    FIST = 2             # 握拳 — 五指皆彎曲
    OPEN_HAND = 3        # 張手 — 五指皆伸直
    THUMBS_UP = 4        # 比讚 — 拇指朝上，其餘四指彎曲
    PEACE = 5            # 比 V — 食指+中指伸直，無名指+小指彎曲
    SWIPE_LEFT = 6       # 向左揮手 — 手腕快速左移
    SWIPE_RIGHT = 7      # 向右揮手 — 手腕快速右移
    POINT = 8            # 指向 — 僅食指伸直


# ═════════════════════════════════════════════════════════════
#  手指定義常數
# ═════════════════════════════════════════════════════════════

# 手指名稱對應
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# MediaPipe 手部關節索引映射表
# 每根手指: [MCP/CMC, PIP/MCP, DIP/IP, TIP]
# 拇指結構特殊: [CMC(1), MCP(2), IP(3), TIP(4)]
# 其他手指:     [MCP, PIP, DIP, TIP]
FINGER_LANDMARKS = {
    0: [1, 2, 3, 4],       # 拇指: CMC → MCP → IP → TIP
    1: [5, 6, 7, 8],       # 食指: MCP → PIP → DIP → TIP
    2: [9, 10, 11, 12],    # 中指: MCP → PIP → DIP → TIP
    3: [13, 14, 15, 16],   # 無名指: MCP → PIP → DIP → TIP
    4: [17, 18, 19, 20],   # 小指: MCP → PIP → DIP → TIP
}


class GestureRecognizer:
    """
    進階手勢辨識器。

    設計原則:
      - 穩定性優先: 使用確認緩衝避免誤觸發
      - 低延遲: 確認幀數設定為較小值 (預設 3 幀)
      - 可擴展: 回呼機制允許外部模組訂閱手勢事件
      - 抗干擾: 冷卻計時器防止連續觸發
    """

    def __init__(self):
        """
        初始化手勢辨識器的所有內部狀態。
        """
        # ── 冷卻計時器 ────────────────────────────────────────
        # 每種手勢獨立的冷卻倒數 (幀)
        # 當值 > 0 時，該手勢不會被觸發
        self._cooldown_timers: dict[GestureType, int] = {
            gt: 0 for gt in GestureType
        }

        # ── 確認緩衝 ─────────────────────────────────────────
        # 儲存最近 N 幀的偵測結果
        # 手勢必須在連續 GESTURE_CONFIRM_FRAMES 幀中被偵測到才會確認
        self._confirm_buffer: deque[GestureType] = deque(
            maxlen=GESTURE_CONFIRM_FRAMES
        )

        # ── 揮手偵測: 位置歷史 ───────────────────────────────
        # 儲存手腕的歸一化 X 座標歷史，用於計算速度
        self._swipe_history: deque[float] = deque(maxlen=10)

        # ── 回呼註冊表 ───────────────────────────────────────
        # GestureType → [callable, callable, ...]
        self._callbacks: dict[GestureType, list] = {
            gt: [] for gt in GestureType
        }

        # ── 上一幀確認的手勢 (避免重複觸發回呼) ──────────────
        self._last_confirmed = GestureType.NONE

    # ═════════════════════════════════════════════════════════
    #  公共 API
    # ═════════════════════════════════════════════════════════

    def update(self, landmarks, pts_3d: np.ndarray = None) -> GestureType:
        """
        每幀呼叫的主更新函式。

        辨識流程:
          1. 計算五指伸直/彎曲狀態
          2. 依優先順序檢查各手勢候選
          3. 將候選手勢加入確認緩衝
          4. 檢查緩衝是否連續穩定
          5. 確認後檢查冷卻狀態
          6. 觸發回呼

        Args:
            landmarks: MediaPipe NormalizedLandmarkList (21 個關節)
            pts_3d:    可選的 3D 座標 (shape 21×3)，部分手勢可利用 3D 資訊

        Returns:
            本幀確認的 GestureType (若無觸發則為 NONE)
        """
        lm = landmarks.landmark

        # ── 遞減所有冷卻計時器 ────────────────────────────────
        for gt in self._cooldown_timers:
            if self._cooldown_timers[gt] > 0:
                self._cooldown_timers[gt] -= 1

        # ── 步驟 1: 偵測候選手勢 ─────────────────────────────
        # 依優先順序檢查 (優先順序高的手勢先判定)
        candidate = self._detect_candidate(lm, pts_3d)

        # ── 步驟 2: 確認緩衝 ─────────────────────────────────
        self._confirm_buffer.append(candidate)

        confirmed = GestureType.NONE

        # 檢查緩衝是否全部為同一手勢 (且不為 NONE)
        if (
            len(self._confirm_buffer) == GESTURE_CONFIRM_FRAMES
            and candidate != GestureType.NONE
        ):
            # 所有幀都是同一個候選手勢
            if all(g == candidate for g in self._confirm_buffer):
                # ── 步驟 3: 冷卻檢查 ─────────────────────────
                if self._cooldown_timers[candidate] <= 0:
                    confirmed = candidate
                    # 設定冷卻 (防止連續觸發)
                    self._cooldown_timers[candidate] = GESTURE_COOLDOWN_FRAMES
                    # 清空確認緩衝 (避免同一手勢持續觸發)
                    self._confirm_buffer.clear()

        # ── 步驟 4: 觸發回呼 ─────────────────────────────────
        if confirmed != GestureType.NONE:
            self._fire_callbacks(confirmed)

        self._last_confirmed = confirmed
        return confirmed

    def register_callback(self, gesture_type: GestureType, callback: callable) -> None:
        """
        註冊手勢觸發回呼函式。

        當指定手勢被確認觸發時，所有已註冊的回呼函式會依序被呼叫。

        Args:
            gesture_type: 要訂閱的手勢類型
            callback:     回呼函式 (無參數)

        用法範例:
            recognizer.register_callback(GestureType.THUMBS_UP, take_screenshot)
        """
        if gesture_type in self._callbacks:
            self._callbacks[gesture_type].append(callback)

    def get_finger_states(self, landmarks) -> dict:
        """
        取得所有手指的狀態資訊。

        用於 HUD 顯示手指狀態指示器。

        Args:
            landmarks: MediaPipe NormalizedLandmarkList

        Returns:
            dict: {
                "thumb":  (is_extended: bool, curl_ratio: float),
                "index":  (is_extended: bool, curl_ratio: float),
                "middle": (is_extended: bool, curl_ratio: float),
                "ring":   (is_extended: bool, curl_ratio: float),
                "pinky":  (is_extended: bool, curl_ratio: float),
            }
        """
        lm = landmarks.landmark
        states = {}
        for finger_id, name in enumerate(FINGER_NAMES):
            extended = self._is_finger_extended(lm, finger_id)
            curl = self._compute_finger_curl(lm, finger_id)
            states[name] = (extended, curl)
        return states

    def get_pinch_data(self, landmarks) -> tuple[tuple[float, float], float]:
        """
        取得捏合相關數據，供互動系統使用。

        Returns:
            tuple:
              - midpoint: (x, y) 拇指尖與食指尖的中點 (歸一化座標)
              - pinch_ratio: 捏合距離比例 (0 = 完全捏合, 1+ = 完全張開)
        """
        return self._get_pinch_data(landmarks.landmark)

    # ═════════════════════════════════════════════════════════
    #  手勢候選偵測 (優先順序判定)
    # ═════════════════════════════════════════════════════════

    def _detect_candidate(self, lm, pts_3d: np.ndarray = None) -> GestureType:
        """
        依優先順序偵測候選手勢。

        優先順序設計邏輯:
          1. PINCH — 最常用的互動手勢，需即時回應
          2. SWIPE — 動態手勢，需在靜態手勢之前偵測
          3. FIST — 明確的靜態手勢
          4. THUMBS_UP — 需要拇指朝上的特定姿勢
          5. PEACE — 兩指伸直的特定組合
          6. POINT — 單指伸直
          7. OPEN_HAND — 所有手指伸直 (最不特殊的手勢)

        Args:
            lm: 關節列表 (21 個 NormalizedLandmark)
            pts_3d: 可選 3D 座標

        Returns:
            偵測到的候選 GestureType
        """
        # ── 1. 捏合偵測 (最高優先) ───────────────────────────
        is_pinch, pinch_strength = self._detect_pinch(lm)
        if is_pinch:
            return GestureType.PINCH

        # ── 2. 揮手偵測 (動態手勢) ───────────────────────────
        swipe = self._detect_swipe(lm)
        if swipe is not None:
            return swipe

        # ── 3. 握拳偵測 ─────────────────────────────────────
        if self._detect_fist(lm):
            return GestureType.FIST

        # ── 4. 比讚偵測 ─────────────────────────────────────
        if self._detect_thumbs_up(lm):
            return GestureType.THUMBS_UP

        # ── 5. 比 V 偵測 ────────────────────────────────────
        if self._detect_peace(lm):
            return GestureType.PEACE

        # ── 6. 指向偵測 ─────────────────────────────────────
        if self._detect_point(lm):
            return GestureType.POINT

        # ── 7. 張手偵測 (最低優先) ───────────────────────────
        if self._detect_open_hand(lm):
            return GestureType.OPEN_HAND

        return GestureType.NONE

    # ═════════════════════════════════════════════════════════
    #  手指狀態分析
    # ═════════════════════════════════════════════════════════

    def _is_finger_extended(self, lm, finger_id: int) -> bool:
        """
        判斷手指是否伸直。

        生物力學原理:
          - 當手指伸直時，指尖 (TIP) 會遠離掌根 (MCP)，
            且指尖 Y 座標低於 PIP 關節 (影像座標系中 Y 向下)
          - 拇指結構特殊: 需比較指尖到 IP 的距離與 MCP 到手腕的距離

        拇指判定 (finger_id == 0):
          由於拇指的運動軸與其他四指不同 (對掌運動)，
          我們比較拇指尖(4) 到食指MCP(5) 的距離
          與拇指IP(3) 到食指MCP(5) 的距離。
          當拇指伸直時，拇指尖會比 IP 更遠離手掌中心。

        其他手指判定 (finger_id 1-4):
          比較指尖 (TIP) 的 Y 座標與 PIP 關節的 Y 座標。
          在影像座標系中，Y 值越小代表位置越高。
          當手指伸直時，TIP.y < PIP.y (指尖高於 PIP)。

        Args:
            lm:        關節列表
            finger_id: 0=拇指, 1=食指, 2=中指, 3=無名指, 4=小指

        Returns:
            True = 伸直, False = 彎曲
        """
        indices = FINGER_LANDMARKS[finger_id]
        # indices: [base, pip_or_mcp, dip_or_ip, tip]

        if finger_id == 0:
            # ── 拇指特殊處理 ──────────────────────────────────
            # 拇指的伸直判定需考慮對掌運動
            # 比較 TIP(4) 到食指MCP(5) 的距離 vs IP(3) 到食指MCP(5) 的距離
            # 拇指伸直時，TIP 距離食指MCP 更遠
            tip = lm[indices[3]]      # TIP = landmark 4
            ip = lm[indices[2]]       # IP  = landmark 3
            ref = lm[5]              # 食指 MCP 作為參考點

            dist_tip = math.hypot(tip.x - ref.x, tip.y - ref.y)
            dist_ip = math.hypot(ip.x - ref.x, ip.y - ref.y)

            return dist_tip > dist_ip
        else:
            # ── 其他四指 ──────────────────────────────────────
            # 伸直判定: 指尖 Y 座標 < PIP Y 座標 (影像中更高)
            # 同時檢查 TIP 到 MCP 的距離是否大於 PIP 到 MCP 的距離
            # (雙重驗證避免手掌朝下時的誤判)
            tip = lm[indices[3]]      # TIP
            pip = lm[indices[1]]      # PIP
            mcp = lm[indices[0]]      # MCP

            # 主要判定: 指尖高於 PIP
            tip_above_pip = tip.y < pip.y

            # 輔助判定: TIP 到 MCP 距離 > PIP 到 MCP 距離
            dist_tip_mcp = math.hypot(tip.x - mcp.x, tip.y - mcp.y)
            dist_pip_mcp = math.hypot(pip.x - mcp.x, pip.y - mcp.y)
            tip_farther = dist_tip_mcp > dist_pip_mcp

            # 兩個條件都滿足時判定為伸直
            return tip_above_pip and tip_farther

    def _compute_finger_curl(self, lm, finger_id: int) -> float:
        """
        計算手指彎曲程度 (curl ratio)。

        生物力學原理:
          手指由三個指骨 (phalanx) 組成，各指骨間的角度決定彎曲程度。
          完全伸直時，三個指骨近似共線 (角度 ≈ 180°)。
          完全彎曲時，指尖靠近掌心 (角度 ≈ 0-60°)。

          我們使用指尖 (TIP) 到掌根 (MCP/CMC) 的距離
          與完全伸直時的最大距離之比來估算彎曲度。

        計算方式:
          curl = 1.0 - (tip_to_base_dist / max_possible_dist)

          其中 max_possible_dist 近似為手指長度
          (各指骨長度之和 = base→pip + pip→dip + dip→tip)

        Args:
            lm:        關節列表
            finger_id: 0-4

        Returns:
            0.0 = 完全伸直, 1.0 = 完全彎曲
        """
        indices = FINGER_LANDMARKS[finger_id]
        base = lm[indices[0]]   # 掌根 (MCP 或 CMC)
        pip = lm[indices[1]]    # PIP 或 MCP (拇指)
        dip = lm[indices[2]]    # DIP 或 IP (拇指)
        tip = lm[indices[3]]    # 指尖

        # 指尖到掌根的直線距離 (歸一化座標)
        tip_to_base = math.hypot(tip.x - base.x, tip.y - base.y)

        # 手指完全伸直時的近似長度 (各段之和)
        seg1 = math.hypot(pip.x - base.x, pip.y - base.y)
        seg2 = math.hypot(dip.x - pip.x, dip.y - pip.y)
        seg3 = math.hypot(tip.x - dip.x, tip.y - dip.y)
        max_length = seg1 + seg2 + seg3

        if max_length < 1e-6:
            return 0.0

        # 伸直度: tip_to_base / max_length
        # 彎曲度: 1 - 伸直度
        straightness = tip_to_base / max_length
        curl = 1.0 - min(straightness, 1.0)

        return curl

    # ═════════════════════════════════════════════════════════
    #  各手勢偵測器
    # ═════════════════════════════════════════════════════════

    def _detect_pinch(self, lm) -> tuple[bool, float]:
        """
        偵測拇指與食指的捏合手勢。

        生物力學原理:
          捏合 (precision grip) 是人類最精細的手部動作之一，
          透過拇指尖 (landmark 4) 與食指尖 (landmark 8) 的距離判定。

          為了消除手掌大小的影響，我們將捏合距離除以
          食指MCP(5) 到手腕(0) 的距離 (掌長) 進行歸一化。

        判定邏輯:
          pinch_ratio = (拇指尖到食指尖距離) / (掌長)
          if pinch_ratio < PINCH_THRESHOLD → 捏合中

        Args:
            lm: 關節列表

        Returns:
            (is_pinching, pinch_strength)
            pinch_strength: 0.0 (未捏合) → 1.0 (完全捏合)
        """
        # 拇指尖(4) 到食指尖(8) 的距離
        thumb_tip = lm[4]
        index_tip = lm[8]
        pinch_dist = math.hypot(
            thumb_tip.x - index_tip.x,
            thumb_tip.y - index_tip.y,
        )

        # 掌長: 食指MCP(5) 到手腕(0)，作為歸一化基準
        # 這個距離在各手勢中相對穩定
        palm_ref = math.hypot(
            lm[5].x - lm[0].x,
            lm[5].y - lm[0].y,
        )

        # 防止除以零
        if palm_ref < 1e-6:
            return False, 0.0

        # 歸一化捏合距離
        pinch_ratio = pinch_dist / palm_ref

        # 判定捏合
        is_pinching = pinch_ratio < PINCH_THRESHOLD

        # 捏合強度: 從 PINCH_THRESHOLD 到 0 線性映射為 0 到 1
        if is_pinching:
            pinch_strength = 1.0 - (pinch_ratio / PINCH_THRESHOLD)
        else:
            pinch_strength = 0.0

        return is_pinching, pinch_strength

    def _detect_fist(self, lm) -> bool:
        """
        偵測握拳手勢。

        生物力學原理:
          握拳時五指全部彎曲，指尖靠近掌心。
          我們檢查所有手指的彎曲度 (curl ratio) 是否超過閾值。

          注意: 拇指在握拳時通常包裹在外側或內側，
          其彎曲方式與其他四指不同，但 curl ratio 仍然會升高。

        判定邏輯:
          所有 5 指的 curl_ratio > FIST_CURL_THRESHOLD

        Args:
            lm: 關節列表

        Returns:
            True = 握拳
        """
        for finger_id in range(5):
            curl = self._compute_finger_curl(lm, finger_id)
            if curl < FIST_CURL_THRESHOLD:
                return False
        return True

    def _detect_open_hand(self, lm) -> bool:
        """
        偵測五指張開手勢。

        生物力學原理:
          五指張開時所有手指充分伸展。
          我們使用 "伸直度" (1 - curl) 來判定。

          張手判定比握拳稍難，因為日常中手指很少完全伸直，
          所以閾值設定要考慮自然手勢的舒適範圍。

        判定邏輯:
          所有 5 指的 (1 - curl_ratio) > OPEN_HAND_EXTEND_THRESHOLD
          即 curl_ratio < (1 - OPEN_HAND_EXTEND_THRESHOLD)

        Args:
            lm: 關節列表

        Returns:
            True = 五指張開
        """
        max_curl_allowed = 1.0 - OPEN_HAND_EXTEND_THRESHOLD

        for finger_id in range(5):
            curl = self._compute_finger_curl(lm, finger_id)
            if curl > max_curl_allowed:
                return False
        return True

    def _detect_thumbs_up(self, lm) -> bool:
        """
        偵測比讚手勢 (👍)。

        生物力學原理:
          比讚需要:
            1. 拇指伸直且朝上 — TIP.y 明顯低於 MCP.y (影像座標)
            2. 其餘四指彎曲 — 握住掌心

          額外檢查拇指方向: 拇指尖(4) 的 Y 座標需低於
          拇指MCP(2) 的 Y 座標至少一定角度，
          確保拇指確實朝上而非朝側邊。

        Args:
            lm: 關節列表

        Returns:
            True = 比讚
        """
        # ── 檢查拇指伸直 ─────────────────────────────────────
        if not self._is_finger_extended(lm, 0):
            return False

        # ── 檢查拇指朝上 ─────────────────────────────────────
        # 在影像座標系中，Y 向下為正
        # 拇指朝上 = tip.y < mcp.y (tip 在 mcp 上方)
        thumb_tip = lm[4]
        thumb_mcp = lm[2]

        # 計算拇指方向角度 (相對垂直方向)
        # 使用 atan2 計算拇指向量相對於垂直線的偏移角度
        dx = thumb_tip.x - thumb_mcp.x
        dy = thumb_tip.y - thumb_mcp.y  # 負值代表朝上

        # 拇指向量角度 (度)，0° = 正上方
        angle_from_up = abs(math.degrees(math.atan2(dx, -dy)))

        if angle_from_up > THUMBS_UP_ANGLE_THRESHOLD:
            return False

        # ── 檢查其餘四指彎曲 ─────────────────────────────────
        for finger_id in range(1, 5):
            curl = self._compute_finger_curl(lm, finger_id)
            if curl < FIST_CURL_THRESHOLD:
                return False

        return True

    def _detect_peace(self, lm) -> bool:
        """
        偵測比 V 手勢 (✌️)。

        生物力學原理:
          比 V 手勢需要:
            1. 食指 (index) 與中指 (middle) 伸直
            2. 無名指 (ring) 與小指 (pinky) 彎曲
            3. 拇指可以自由 (通常貼著彎曲的無名指)

          這是較為「特化」的手勢，誤判率較低。

        Args:
            lm: 關節列表

        Returns:
            True = 比 V
        """
        # 食指 & 中指必須伸直
        if not self._is_finger_extended(lm, 1):
            return False
        if not self._is_finger_extended(lm, 2):
            return False

        # 無名指 & 小指必須彎曲
        ring_curl = self._compute_finger_curl(lm, 3)
        pinky_curl = self._compute_finger_curl(lm, 4)

        if ring_curl < FIST_CURL_THRESHOLD:
            return False
        if pinky_curl < FIST_CURL_THRESHOLD:
            return False

        return True

    def _detect_point(self, lm) -> bool:
        """
        偵測食指指向手勢 (☝️)。

        生物力學原理:
          指向手勢是最基本的指示動作:
            1. 食指伸直 — 指向目標
            2. 拇指彎曲 — 與握拳時相同
            3. 中指、無名指、小指彎曲 — 握住掌心

          與「比 V」的區別在於中指必須彎曲。
          與「張手」的區別在於只有食指伸直。

        Args:
            lm: 關節列表

        Returns:
            True = 指向
        """
        # 食指必須伸直
        if not self._is_finger_extended(lm, 1):
            return False

        # 拇指必須彎曲 (排除「OK」手勢的可能性)
        thumb_curl = self._compute_finger_curl(lm, 0)
        if thumb_curl < FIST_CURL_THRESHOLD:
            return False

        # 中指、無名指、小指必須彎曲
        for finger_id in [2, 3, 4]:
            curl = self._compute_finger_curl(lm, finger_id)
            if curl < FIST_CURL_THRESHOLD:
                return False

        return True

    def _detect_swipe(self, lm) -> GestureType | None:
        """
        偵測揮手手勢 (水平方向)。

        運動學原理:
          揮手是動態手勢，需要追蹤手腕位置隨時間的變化。
          我們記錄手腕 (landmark 0) 的歸一化 X 座標歷史，
          計算最近若干幀的水平速度。

          速度計算: v_x = x_current - x_history[-N]
          當 |v_x| > SWIPE_VELOCITY_THRESHOLD 時判定為揮手。

          方向判定:
            v_x < 0 → SWIPE_LEFT  (在影像中向左移動)
            v_x > 0 → SWIPE_RIGHT (在影像中向右移動)

          注意: 影像座標的 X 軸方向取決於攝影機是否鏡像。
          此處假設未鏡像 (左右與真實世界一致)。

        Args:
            lm: 關節列表

        Returns:
            GestureType.SWIPE_LEFT, SWIPE_RIGHT, or None
        """
        wrist_x = lm[0].x
        self._swipe_history.append(wrist_x)

        # 需要至少 5 幀歷史才能可靠判斷速度
        if len(self._swipe_history) < 5:
            return None

        # 計算速度: 當前位置 - 5 幀前的位置
        velocity_x = wrist_x - self._swipe_history[-5]

        if abs(velocity_x) > SWIPE_VELOCITY_THRESHOLD:
            if velocity_x < 0:
                return GestureType.SWIPE_LEFT
            else:
                return GestureType.SWIPE_RIGHT

        return None

    # ═════════════════════════════════════════════════════════
    #  捏合輔助數據
    # ═════════════════════════════════════════════════════════

    def _get_pinch_data(self, lm) -> tuple[tuple[float, float], float]:
        """
        計算捏合中點與距離比例。

        用途:
          互動系統需要知道捏合發生的位置 (用於拖曳面板)
          以及捏合的距離 (用於判斷抓取狀態)。

        中點計算:
          midpoint = (thumb_tip + index_tip) / 2
          以歸一化座標表示 (0-1 範圍)

        距離比例:
          與 _detect_pinch 相同的歸一化距離

        Args:
            lm: 關節列表

        Returns:
            ((mid_x, mid_y), pinch_ratio)
        """
        thumb_tip = lm[4]
        index_tip = lm[8]

        # 中點
        mid_x = (thumb_tip.x + index_tip.x) / 2.0
        mid_y = (thumb_tip.y + index_tip.y) / 2.0

        # 捏合距離
        pinch_dist = math.hypot(
            thumb_tip.x - index_tip.x,
            thumb_tip.y - index_tip.y,
        )

        # 掌長歸一化
        palm_ref = math.hypot(
            lm[5].x - lm[0].x,
            lm[5].y - lm[0].y,
        )

        pinch_ratio = pinch_dist / max(palm_ref, 1e-6)

        return (mid_x, mid_y), pinch_ratio

    # ═════════════════════════════════════════════════════════
    #  回呼系統
    # ═════════════════════════════════════════════════════════

    def _fire_callbacks(self, gesture: GestureType) -> None:
        """
        觸發指定手勢的所有已註冊回呼函式。

        錯誤處理:
          每個回呼獨立執行，單一回呼的異常不會影響其他回呼。
          異常會被捕獲並印出警告 (不中斷主程式)。

        Args:
            gesture: 已確認的手勢類型
        """
        for cb in self._callbacks.get(gesture, []):
            try:
                cb()
            except Exception as e:
                print(f"[GestureRecognizer] 回呼執行失敗 ({gesture.name}): {e}")
