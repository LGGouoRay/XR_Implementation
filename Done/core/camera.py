# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║          XR HUD SYSTEM — CAMERA MANAGER                     ║
║  攝影機初始化、影像擷取、內參管理                            ║
║  Camera initialization, frame capture, intrinsics mgmt      ║
╚══════════════════════════════════════════════════════════════╝

本模組封裝 OpenCV VideoCapture，提供：
  - 攝影機初始化與解析度設定
  - 鏡像翻轉影像擷取
  - 針孔相機內參計算 (焦距、光心)
  - 安全的資源釋放
"""

import cv2
import numpy as np

# 匯入全域設定常數
# Import global configuration constants
from config import (
    CAMERA_ID,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)


class CameraManager:
    """
    攝影機管理器 — 負責影像擷取與相機內參管理。
    Camera Manager — handles frame capture and camera intrinsics.

    使用 OpenCV VideoCapture 進行影像擷取，並基於針孔相機模型
    (pinhole camera model) 計算內參數。

    針孔相機模型簡介:
      在針孔模型中，焦距 f 決定了三維世界到二維影像平面的投影比例。
      當缺乏真實標定數據時，常用的近似值為 f ≈ 影像寬度 (以像素為單位)，
      這大致對應於水平視場角 (FOV) ≈ 53°，適用於大多數消費級網路攝影機。

      光心 (cx, cy) 為影像中心點，假設鏡頭無畸變時恰好在影像正中央。

    Attributes
    ----------
    camera_id : int
        攝影機裝置 ID / Camera device ID.
    width : int
        影像寬度 (px) / Frame width in pixels.
    height : int
        影像高度 (px) / Frame height in pixels.
    focal_length : float
        焦距 (px)，針孔模型近似值 / Focal length (px), pinhole approximation.
    cx : float
        光心 x 座標 (px) / Principal point x-coordinate.
    cy : float
        光心 y 座標 (px) / Principal point y-coordinate.
    cap : cv2.VideoCapture
        OpenCV 影像擷取物件 / OpenCV video capture object.
    """

    def __init__(self, camera_id: int = CAMERA_ID,
                 width: int = FRAME_WIDTH,
                 height: int = FRAME_HEIGHT):
        """
        初始化攝影機管理器。
        Initialize the Camera Manager.

        步驟:
          1. 建立 VideoCapture 連線
          2. 設定解析度 (寬 × 高)
          3. 計算針孔相機內參 (焦距、光心)

        Parameters
        ----------
        camera_id : int, optional
            攝影機裝置編號，預設從 config 取得。
            Camera device index, defaults to CAMERA_ID from config.
        width : int, optional
            影像寬度 (px)，預設從 config 取得。
            Frame width, defaults to FRAME_WIDTH from config.
        height : int, optional
            影像高度 (px)，預設從 config 取得。
            Frame height, defaults to FRAME_HEIGHT from config.
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height

        # ── 初始化 VideoCapture ──────────────────────────────
        # 使用 DirectShow 後端 (Windows) 可加速初始化
        # Use DirectShow backend on Windows for faster init
        self.cap = cv2.VideoCapture(camera_id)

        # 設定攝影機解析度
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # 讀取實際解析度 (攝影機可能不支援請求的解析度)
        # Read back actual resolution (camera may not support requested)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != width or actual_h != height:
            # 更新為攝影機實際支援的解析度
            # Update to actually supported resolution
            self.width = actual_w
            self.height = actual_h

        # ── 計算針孔相機內參 ─────────────────────────────────
        # 焦距近似值: f ≈ width (像素)
        # 這對應於 FOV ≈ 2·arctan(w/(2f)) ≈ 53° 的水平視場角
        # Focal length approximation: f ≈ width (pixels)
        # Corresponds to horizontal FOV ≈ 2·arctan(w/(2f)) ≈ 53°
        self.focal_length = float(self.width)

        # 光心位於影像中心 (假設無鏡頭畸變)
        # Principal point at image center (assuming no lens distortion)
        self.cx = self.width  / 2.0
        self.cy = self.height / 2.0

    def read_frame(self) -> np.ndarray | None:
        """
        讀取一幀影像並進行水平鏡像翻轉。
        Read a single frame and apply horizontal flip (mirror).

        鏡像翻轉使使用者看到的畫面如同照鏡子，手部動作方向與螢幕一致，
        提升手勢操作的直覺性。
        Mirror flip makes the view intuitive for gesture interaction,
        so hand movements match on-screen direction.

        Returns
        -------
        np.ndarray or None
            BGR 格式影像 (H × W × 3)，讀取失敗時回傳 None。
            BGR frame (H × W × 3), or None if capture failed.
        """
        ret, frame = self.cap.read()

        if not ret or frame is None:
            # 擷取失敗 — 可能是攝影機斷線或忙碌
            # Capture failed — camera may be disconnected or busy
            return None

        # 水平翻轉 (flipCode=1): 左右鏡像
        # Horizontal flip (flipCode=1): mirror left-right
        frame = cv2.flip(frame, 1)

        return frame

    def get_intrinsics(self) -> tuple[float, float, float]:
        """
        取得相機內參數。
        Get camera intrinsic parameters.

        回傳針孔相機模型的三個核心參數：
          - focal_length: 焦距，決定投影縮放比例
          - cx: 光心 x，投影原點水平偏移
          - cy: 光心 y，投影原點垂直偏移

        Returns
        -------
        tuple[float, float, float]
            (focal_length, cx, cy) — 焦距與光心座標。
        """
        return (self.focal_length, self.cx, self.cy)

    def get_dimensions(self) -> tuple[int, int]:
        """
        取得影像尺寸。
        Get frame dimensions.

        Returns
        -------
        tuple[int, int]
            (width, height) — 影像寬度與高度 (px)。
        """
        return (self.width, self.height)

    def release(self) -> None:
        """
        釋放攝影機資源。
        Release camera resources.

        應在程式結束或不再需要攝影機時呼叫，以釋放裝置佔用。
        Call when shutting down or when camera is no longer needed.
        """
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
