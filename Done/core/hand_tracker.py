# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║         HAND TRACKER 3D — MediaPipe 手部追蹤與3D重建          ║
║  負責從攝影機畫面中偵測手部，並將 2D 關節點重建為              ║
║  3D 攝影機空間座標，包含深度估計與手部旋轉計算                ║
╚══════════════════════════════════════════════════════════════╝

核心改進:
  - 多特徵深度估計 (掌寬、掌高、手指展開度)
  - 每隻手獨立的卡爾曼濾波器穩定深度
  - 正交基底手部旋轉矩陣計算
"""

import cv2
import mediapipe as mp
import numpy as np
import math
from config import (
    MAX_HANDS,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE,
    DEPTH_CALIBRATION_WIDTH,
    DEPTH_CALIBRATION_DIST,
    DEPTH_Z_FACTOR,
    CLR_WHITE,
    CLR_CYAN,
)


class HandTracker3D:
    """
    MediaPipe 手部追蹤器，支援 3D 空間座標重建。

    功能:
      1. 使用 MediaPipe Hands 偵測手部 21 個關節點
      2. 透過多特徵加權平均估算手部深度 (Z)
      3. 利用卡爾曼濾波器穩定深度估計值
      4. 從掌面向量計算手部旋轉 (Euler angles)
    """

    # ── MediaPipe 手部關節索引參考 ──────────────────────────
    # 0: WRIST (手腕)
    # 1-4: THUMB (拇指)     CMC / MCP / IP / TIP
    # 5-8: INDEX (食指)     MCP / PIP / DIP / TIP
    # 9-12: MIDDLE (中指)   MCP / PIP / DIP / TIP
    # 13-16: RING (無名指)  MCP / PIP / DIP / TIP
    # 17-20: PINKY (小指)   MCP / PIP / DIP / TIP

    def __init__(self):
        """
        初始化 MediaPipe 手部追蹤模組與卡爾曼濾波器狀態。

        卡爾曼濾波器參數:
          - kalman_z: 當前深度估計值 (首次量測時初始化)
          - kalman_P: 估計誤差共變異數
          - kalman_Q: 過程噪聲共變異數 (模型不確定性)
          - kalman_R: 量測噪聲共變異數 (感測器噪聲)
        """
        # ── MediaPipe 初始化 ──────────────────────────────────
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,           # 連續影片模式，啟用追蹤優化
            max_num_hands=MAX_HANDS,            # 最多追蹤手數
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )

        # ── 卡爾曼濾波器狀態 (每隻手獨立) ────────────────────
        # 使用 dict 以 hand_index 為鍵，儲存各手的濾波器狀態
        # 每個條目為: {"z": float, "P": float}
        self._kalman_states: dict[int, dict] = {}

        # 卡爾曼濾波器超參數 (所有手共用)
        self._kalman_Q = 0.1    # 過程噪聲 — 越大追蹤越靈敏但越不穩定
        self._kalman_R = 5.0    # 量測噪聲 — 越大濾波越強但延遲越大

        # ── 深度估計特徵權重 ──────────────────────────────────
        # [掌寬, 掌高, 手指展開度]
        # 掌寬與掌高較為穩定，給予較高權重
        # 手指展開度受手勢影響較大，權重較低
        self._depth_feature_weights = np.array([0.4, 0.4, 0.2])

    # ═════════════════════════════════════════════════════════
    #  公共 API
    # ═════════════════════════════════════════════════════════

    def process_frame(self, rgb_frame: np.ndarray):
        """
        執行 MediaPipe 手部偵測。

        Args:
            rgb_frame: RGB 格式影像 (H, W, 3)，注意 MediaPipe 需要 RGB 而非 BGR

        Returns:
            mediapipe.framework.formats.landmark_pb2 結果物件
            包含 multi_hand_landmarks 與 multi_handedness
        """
        # MediaPipe 要求輸入為不可寫的 NumPy 陣列可以提升效能
        # 但為相容性保留可寫狀態
        results = self.hands.process(rgb_frame)
        return results

    def extract_hand_3d(
        self,
        landmarks,
        img_w: int,
        img_h: int,
        f: float,
        cx: float,
        cy: float,
        hand_index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        從 MediaPipe 2D 關節點重建 3D 攝影機空間座標。

        改進策略:
          使用四個特徵進行深度估計 (原始版本僅用掌寬):
            1. 掌寬: landmarks[17] ↔ landmarks[5] (小指MCP ↔ 食指MCP)
            2. 掌高: landmarks[9] ↔ landmarks[0]  (中指MCP ↔ 手腕)
            3. 手指展開度: landmarks[20] ↔ landmarks[8] (小指尖 ↔ 食指尖)
          多特徵加權平均可降低單一特徵受手勢影響的波動。

        Args:
            landmarks: MediaPipe 手部關節點 (NormalizedLandmarkList)
            img_w:     影像寬度 (像素)
            img_h:     影像高度 (像素)
            f:         攝影機焦距 (像素)
            cx:        光學中心 X (像素)
            cy:        光學中心 Y (像素)
            hand_index: 手部索引 (用於區分左右手的卡爾曼狀態)

        Returns:
            tuple:
              - pts_3d: shape (21, 3) 的 numpy 陣列，各關節的 3D 座標 [x, y, z] (mm)
              - hand_rot: shape (3,) 的 numpy 陣列，手部旋轉角度 [pitch, yaw, roll] (弧度)
        """
        lm = landmarks.landmark

        # ─────────────────────────────────────────────────────
        # 步驟 1: 多特徵深度距離計算
        # ─────────────────────────────────────────────────────
        # 特徵 1: 掌寬 — 食指 MCP(5) 到小指 MCP(17) 的像素距離
        # 這是最穩定的特徵，因為掌骨長度在各手勢中幾乎不變
        palm_width = self._landmark_dist_2d(lm[17], lm[5], img_w, img_h)

        # 特徵 2: 掌高 — 中指 MCP(9) 到手腕(0) 的像素距離
        # 同樣穩定，但在手腕彎曲時會有些許變化
        palm_height = self._landmark_dist_2d(lm[9], lm[0], img_w, img_h)

        # 特徵 3: 手指展開度 — 小指尖(20) 到食指尖(8) 的像素距離
        # 受手勢影響最大 (握拳時會大幅縮小)，因此權重最低
        finger_spread = self._landmark_dist_2d(lm[20], lm[8], img_w, img_h)

        # 加權平均距離
        features = np.array([palm_width, palm_height, finger_spread])
        dist_2d = float(np.dot(self._depth_feature_weights, features))

        # ─────────────────────────────────────────────────────
        # 步驟 2: 原始深度計算 (透視投影反推)
        # ─────────────────────────────────────────────────────
        # 原理: 物體在影像中的投影大小與距離成反比
        # Z = (已知實際大小 / 影像投影大小) × 校正距離
        # DEPTH_CALIBRATION_WIDTH: 在 DEPTH_CALIBRATION_DIST 距離時，
        #   手掌在影像中的像素寬度 (校正基準)
        Z_raw = (DEPTH_CALIBRATION_WIDTH / max(dist_2d, 1.0)) * DEPTH_CALIBRATION_DIST

        # ─────────────────────────────────────────────────────
        # 步驟 3: 卡爾曼濾波穩定深度
        # ─────────────────────────────────────────────────────
        # 單幀的深度估計會因關節抖動而不穩定
        # 卡爾曼濾波器以物理模型預測下一幀深度，
        # 再融合量測值，輸出更平滑且準確的深度估計
        Z_hand = self._kalman_update(Z_raw, hand_index)

        # ─────────────────────────────────────────────────────
        # 步驟 4: 計算每個關節的 3D 物理座標
        # ─────────────────────────────────────────────────────
        # 使用針孔攝影機模型反投影:
        #   x_phys = (u - cx) × (z / f)
        #   y_phys = (v - cy) × (z / f)
        #   z_phys = Z_hand + lm.z × Z_hand × DEPTH_Z_FACTOR
        #
        # MediaPipe 的 lm.z 代表相對於手腕的深度偏移
        # (比手腕更靠近攝影機為負值)
        # 我們將其縮放後加到基礎深度 Z_hand 上

        pts_3d = np.zeros((21, 3), dtype=np.float64)

        for i in range(21):
            # 各關節相對手腕的深度偏移
            z_phys = Z_hand + lm[i].z * Z_hand * DEPTH_Z_FACTOR

            # 影像座標 (像素)
            u = lm[i].x * img_w
            v = lm[i].y * img_h

            # 反投影到 3D 空間 (mm)
            x_phys = (u - cx) * (z_phys / f)
            y_phys = (v - cy) * (z_phys / f)

            pts_3d[i] = [x_phys, y_phys, z_phys]

        # ─────────────────────────────────────────────────────
        # 步驟 5: 計算手部旋轉 (正交基底法)
        # ─────────────────────────────────────────────────────
        hand_rot = self._compute_hand_rotation(pts_3d)

        return pts_3d, hand_rot

    def draw_hand_landmarks(self, frame: np.ndarray, hand_landmarks) -> None:
        """
        在畫面上繪製風格化手部骨架。

        使用白色圓點標示關節，青色線條連接骨骼，
        營造科技感 HUD 風格。

        Args:
            frame:          BGR 影像 (會被原地修改)
            hand_landmarks: MediaPipe 手部關節點
        """
        # 關節點樣式 — 白色實心圓
        joint_spec = self.mp_draw.DrawingSpec(
            color=CLR_WHITE,      # BGR: 白色
            thickness=2,
            circle_radius=3,
        )

        # 骨骼連線樣式 — 青色線條
        bone_spec = self.mp_draw.DrawingSpec(
            color=CLR_CYAN,       # BGR: 青色 (科技風)
            thickness=2,
            circle_radius=1,
        )

        self.mp_draw.draw_landmarks(
            image=frame,
            landmark_list=hand_landmarks,
            connections=self.mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=joint_spec,
            connection_drawing_spec=bone_spec,
        )

    def release(self) -> None:
        """
        釋放 MediaPipe 資源。

        應在程式結束時呼叫，確保底層 C++ 模組正確清理。
        """
        self.hands.close()
        self._kalman_states.clear()

    # ═════════════════════════════════════════════════════════
    #  內部方法
    # ═════════════════════════════════════════════════════════

    def _kalman_update(self, z_measured: float, hand_index: int) -> float:
        """
        一維卡爾曼濾波器更新。

        卡爾曼濾波器數學原理:
          預測步驟 (Predict):
            x̂⁻ = x̂ₖ₋₁          (假設等速模型，預測 = 上一次估計)
            P⁻  = Pₖ₋₁ + Q      (預測共變異數增加過程噪聲)

          更新步驟 (Update):
            K  = P⁻ / (P⁻ + R)   (卡爾曼增益)
            x̂  = x̂⁻ + K × (z - x̂⁻)  (融合量測值)
            P  = (1 - K) × P⁻    (更新共變異數)

        Args:
            z_measured: 當幀原始深度量測值 (mm)
            hand_index: 手部索引 (用於獨立濾波)

        Returns:
            濾波後的深度估計值 (mm)
        """
        state = self._kalman_states.get(hand_index)

        if state is None:
            # 首次量測 — 直接使用量測值初始化
            self._kalman_states[hand_index] = {
                "z": z_measured,
                "P": 1.0,       # 初始共變異數 (不確定性較高)
            }
            return z_measured

        # ── 預測步驟 ──────────────────────────────────────────
        z_pred = state["z"]                    # 等速模型: 預測 = 前一估計
        P_pred = state["P"] + self._kalman_Q   # 預測不確定性增加

        # ── 更新步驟 ──────────────────────────────────────────
        K = P_pred / (P_pred + self._kalman_R)  # 卡爾曼增益
        z_est = z_pred + K * (z_measured - z_pred)  # 融合量測值
        P_est = (1.0 - K) * P_pred              # 更新不確定性

        # 儲存狀態
        state["z"] = z_est
        state["P"] = P_est

        return z_est

    def _landmark_dist_2d(self, lm_a, lm_b, img_w: int, img_h: int) -> float:
        """
        計算兩個歸一化關節點在像素空間的歐幾里得距離。

        MediaPipe 的關節座標為歸一化 [0, 1]，需乘以影像尺寸轉為像素。

        Args:
            lm_a, lm_b: MediaPipe NormalizedLandmark
            img_w, img_h: 影像寬高 (像素)

        Returns:
            像素距離 (float)
        """
        dx = (lm_a.x - lm_b.x) * img_w
        dy = (lm_a.y - lm_b.y) * img_h
        return math.sqrt(dx * dx + dy * dy)

    def _compute_hand_rotation(self, pts_3d: np.ndarray) -> np.ndarray:
        """
        從 3D 關節座標計算手部旋轉角度。

        使用掌面兩個向量建立正交基底:
          v_up:      手腕(0) → 中指MCP(9)  (掌面向上方向)
          v_right:   食指MCP(5) → 小指MCP(17) (掌面向右方向)
          v_forward: v_right × v_up  (掌面法線，指向觀察者)

        Gram-Schmidt 正交化後建構旋轉矩陣，
        再從旋轉矩陣提取 Euler 角 (XYZ 順序)。

        生物力學解釋:
          - Pitch (俯仰): 手掌上下傾斜 (如按下按鈕)
          - Yaw (偏航):   手掌左右轉動 (如揮手)
          - Roll (翻轉):  手掌繞前臂軸旋轉 (如翻掌)

        Args:
            pts_3d: shape (21, 3) 的 3D 關節座標

        Returns:
            shape (3,) 的 Euler 角 [pitch, yaw, roll] (弧度)
        """
        # ── 建立掌面向量 ──────────────────────────────────────
        v_up = pts_3d[9] - pts_3d[0]       # 手腕 → 中指MCP
        v_right = pts_3d[17] - pts_3d[5]   # 食指MCP → 小指MCP

        # ── 計算法線向量 (叉積) ───────────────────────────────
        v_forward = np.cross(v_right, v_up)

        # ── Gram-Schmidt 正交化 ──────────────────────────────
        # 確保三個軸互相垂直，形成正交基底
        def _normalize(v):
            n = np.linalg.norm(v)
            return v / n if n > 1e-8 else np.array([0.0, 0.0, 1.0])

        # 以 v_forward (法線) 為基準軸
        z_axis = _normalize(v_forward)

        # 重新計算 v_up 使其垂直於 z_axis
        # y' = y - (y·z)z
        y_axis = v_up - np.dot(v_up, z_axis) * z_axis
        y_axis = _normalize(y_axis)

        # x_axis = y × z (右手定則)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = _normalize(x_axis)

        # ── 建構旋轉矩陣 ─────────────────────────────────────
        # R = [x_axis | y_axis | z_axis] (各軸為列向量)
        R_hand = np.array([x_axis, y_axis, z_axis])  # 3×3

        # ── 提取 Euler 角 (XYZ 順序) ─────────────────────────
        # 從旋轉矩陣提取 pitch, yaw, roll
        # 參考: https://www.geometrictools.com/Documentation/EulerAngles.pdf
        #
        # R = Rz(roll) × Ry(yaw) × Rx(pitch)
        # R[2,0] = -sin(yaw)
        # R[2,1] = cos(yaw) × sin(pitch)
        # R[2,2] = cos(yaw) × cos(pitch)
        # R[0,0] = cos(yaw) × cos(roll)
        # R[1,0] = cos(yaw) × sin(roll)

        pitch, yaw, roll = self._rotation_matrix_to_euler(R_hand)

        return np.array([pitch, yaw, roll])

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
        """
        從 3×3 旋轉矩陣提取 XYZ 順序的 Euler 角。

        處理 gimbal lock (萬向鎖) 的邊界情況:
          當 |R[2,0]| ≈ 1 時，yaw ≈ ±90°，pitch 與 roll 退化。

        Args:
            R: 3×3 旋轉矩陣

        Returns:
            (pitch, yaw, roll) 以弧度為單位
        """
        # 安全限制 sin(yaw) 在 [-1, 1] 範圍內
        sy = -R[2, 0]
        sy = np.clip(sy, -1.0, 1.0)

        yaw = math.asin(sy)

        if abs(sy) < 0.99999:
            # 一般情況 — 無萬向鎖
            pitch = math.atan2(R[2, 1], R[2, 2])
            roll = math.atan2(R[1, 0], R[0, 0])
        else:
            # 萬向鎖 — yaw ≈ ±90°
            # pitch 與 roll 退化，只能求出 pitch - roll 或 pitch + roll
            pitch = math.atan2(-R[1, 2], R[1, 1])
            roll = 0.0

        return pitch, yaw, roll
