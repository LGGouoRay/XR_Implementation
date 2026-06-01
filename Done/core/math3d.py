# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║          XR HUD SYSTEM — 3D MATHEMATICS ENGINE              ║
║  Rotation matrices, quaternions, projection, interpolation  ║
║  All core spatial math operations for the XR HUD pipeline   ║
╚══════════════════════════════════════════════════════════════╝

本模組提供 XR HUD 系統所需的所有三維數學運算，包括：
  - 歐拉角旋轉矩陣 (Y-X-Z 順序)
  - 針孔相機投影
  - 四元數表示法及轉換 (避免萬向鎖)
  - 球面線性插值 (SLERP) 實現平滑旋轉動畫
  - 三維線性插值與碰撞包圍球計算
"""

import numpy as np
import math


# ============================================================
#  旋轉矩陣 / Rotation Matrix
# ============================================================

def get_rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """
    從歐拉角計算 3×3 旋轉矩陣 (Y-X-Z 內旋順序)。
    Compute a 3×3 rotation matrix from Euler angles in Y-X-Z intrinsic order.

    數學推導 / Mathematical derivation:
      R = Ry(yaw) · Rx(pitch) · Rz(roll)

      其中各軸旋轉矩陣定義為:
      Rx(θ) = [[1,  0,      0    ],    繞 X 軸 (俯仰 pitch)
               [0,  cos θ, -sin θ],
               [0,  sin θ,  cos θ]]

      Ry(θ) = [[ cos θ, 0, sin θ],    繞 Y 軸 (偏航 yaw)
               [ 0,     1, 0    ],
               [-sin θ, 0, cos θ]]

      Rz(θ) = [[cos θ, -sin θ, 0],    繞 Z 軸 (翻滾 roll)
               [sin θ,  cos θ, 0],
               [0,      0,     1]]

    Parameters
    ----------
    pitch : float
        繞 X 軸旋轉角度 (弧度) / Rotation about X-axis in radians.
    yaw : float
        繞 Y 軸旋轉角度 (弧度) / Rotation about Y-axis in radians.
    roll : float
        繞 Z 軸旋轉角度 (弧度) / Rotation about Z-axis in radians.

    Returns
    -------
    np.ndarray
        3×3 旋轉矩陣 (float64) / 3×3 rotation matrix.
    """
    # 預計算三角函數值以避免重複計算
    # Pre-compute trig values to avoid redundant calls
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    cr, sr = math.cos(roll),  math.sin(roll)

    # 展開矩陣乘積 R = Ry · Rx · Rz，直接填入最終元素
    # Expanded product R = Ry · Rx · Rz, elements written directly
    #
    # R[0,0] = cy*cr + sy*sp*sr       R[0,1] = -cy*sr + sy*sp*cr      R[0,2] = sy*cp
    # R[1,0] = cp*sr                  R[1,1] = cp*cr                  R[1,2] = -sp
    # R[2,0] = -sy*cr + cy*sp*sr      R[2,1] = sy*sr + cy*sp*cr       R[2,2] = cy*cp
    return np.array([
        [cy * cr + sy * sp * sr,   -cy * sr + sy * sp * cr,   sy * cp],
        [cp * sr,                   cp * cr,                  -sp    ],
        [-sy * cr + cy * sp * sr,   sy * sr + cy * sp * cr,    cy * cp],
    ], dtype=np.float64)


# ============================================================
#  針孔相機投影 / Pinhole Camera Projection
# ============================================================

def project_point(pt: np.ndarray, f: float, cx: float, cy: float):
    """
    將三維點投影至二維螢幕座標 (針孔相機模型)。
    Project a 3D point onto 2D screen coordinates using the pinhole camera model.

    投影公式 / Projection formula:
      px = f * (X / Z) + cx
      py = f * (Y / Z) + cy

    其中:
      f  = 焦距 (focal length, pixels)
      cx = 光心 x 偏移 (principal point x)
      cy = 光心 y 偏移 (principal point y)

    若 Z <= 10.0，表示點在相機平面之後或過近，回傳 None 避免除零。
    If Z <= 10.0, the point is behind or too close to the camera; returns None.

    Parameters
    ----------
    pt : np.ndarray
        三維座標 [X, Y, Z] / 3D point as [X, Y, Z].
    f : float
        焦距 (像素) / Focal length in pixels.
    cx : float
        主點 x 座標 / Principal point x-coordinate.
    cy : float
        主點 y 座標 / Principal point y-coordinate.

    Returns
    -------
    tuple[int, int] or None
        投影後的螢幕座標 (px, py)，或 None (深度不足)。
        Projected screen coordinates (px, py) as integers, or None.
    """
    z = pt[2]

    # 深度檢查：避免投影到相機背面或產生數值爆炸
    # Depth guard: prevent projection behind camera or numerical explosion
    if z <= 10.0:
        return None

    # 針孔投影核心公式
    # Core pinhole projection equations
    inv_z = 1.0 / z          # 倒數取一次，減少除法運算
    px = f * pt[0] * inv_z + cx
    py = f * pt[1] * inv_z + cy

    return (int(px), int(py))


# ============================================================
#  歐拉角 ↔ 四元數 / Euler ↔ Quaternion Conversions
# ============================================================

def euler_to_quaternion(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """
    將歐拉角轉換為四元數 [w, x, y, z]。
    Convert Euler angles (Y-X-Z intrinsic) to a unit quaternion [w, x, y, z].

    此函數用於取代舊版 EMA 歐拉角插值，以避免萬向鎖 (gimbal lock) 問題。
    Replaces legacy EMA Euler interpolation to avoid gimbal lock.

    轉換公式 / Conversion formula:
      每個軸的半角:
        hp = pitch / 2,  hy = yaw / 2,  hr = roll / 2

      Y-X-Z 順序組合:
        w = cy*cp*cr + sy*sp*sr
        x = cy*sp*cr + sy*cp*sr
        y = sy*cp*cr - cy*sp*sr
        z = cy*cp*sr - sy*sp*cr

      其中 c_ = cos(半角), s_ = sin(半角)

    Parameters
    ----------
    pitch : float
        俯仰角 (弧度) / Pitch in radians.
    yaw : float
        偏航角 (弧度) / Yaw in radians.
    roll : float
        翻滾角 (弧度) / Roll in radians.

    Returns
    -------
    np.ndarray
        單位四元數 [w, x, y, z] (float64) / Unit quaternion.
    """
    # 半角預計算 / Half-angle pre-computation
    hp = pitch * 0.5
    hy = yaw   * 0.5
    hr = roll  * 0.5

    cp, sp = math.cos(hp), math.sin(hp)
    cy, sy = math.cos(hy), math.sin(hy)
    cr, sr = math.cos(hr), math.sin(hr)

    # Y-X-Z 內旋順序的四元數合成
    # Quaternion composition for Y-X-Z intrinsic rotation order
    w = cy * cp * cr + sy * sp * sr
    x = cy * sp * cr + sy * cp * sr
    y = sy * cp * cr - cy * sp * sr
    z = cy * cp * sr - sy * sp * cr

    return np.array([w, x, y, z], dtype=np.float64)


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """
    將四元數轉換回歐拉角 [pitch, yaw, roll]。
    Convert a unit quaternion [w, x, y, z] back to Euler angles [pitch, yaw, roll].

    反向轉換公式 (Y-X-Z 順序):
      sinP = 2(wx - yz)
      pitch = arcsin(clamp(sinP, -1, 1))

      若 |sinP| ≈ 1 (萬向鎖):
        yaw = 2 * atan2(y, w)
        roll = 0
      否則:
        yaw   = atan2(2(wy + xz), 1 - 2(x² + y²))
        roll  = atan2(2(wz + xy), 1 - 2(x² + z²))

    Parameters
    ----------
    q : np.ndarray
        單位四元數 [w, x, y, z] / Unit quaternion.

    Returns
    -------
    np.ndarray
        歐拉角 [pitch, yaw, roll] (弧度) / Euler angles in radians.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    # 計算 pitch 的 sin 值 (可能超出 [-1,1] 因浮點誤差)
    # Compute sin(pitch), may exceed [-1,1] due to floating point
    sin_pitch = 2.0 * (w * x - y * z)

    # 鉗制到 [-1, 1] 防止 arcsin 返回 NaN
    # Clamp to [-1, 1] to prevent arcsin returning NaN
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
    pitch = math.asin(sin_pitch)

    # 萬向鎖檢測：當 pitch ≈ ±90° 時退化為二自由度
    # Gimbal lock detection: degenerates to 2-DOF when pitch ≈ ±90°
    if abs(sin_pitch) > 0.99999:
        # 萬向鎖情況：roll 設為 0，yaw 吸收所有自由度
        yaw  = 2.0 * math.atan2(y, w)
        roll = 0.0
    else:
        # 正常情況 / Normal case
        yaw  = math.atan2(2.0 * (w * y + x * z),
                          1.0 - 2.0 * (x * x + y * y))
        roll = math.atan2(2.0 * (w * z + x * y),
                          1.0 - 2.0 * (x * x + z * z))

    return np.array([pitch, yaw, roll], dtype=np.float64)


# ============================================================
#  四元數球面線性插值 / Quaternion SLERP
# ============================================================

def quaternion_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    兩個四元數之間的球面線性插值 (SLERP)。
    Spherical Linear Interpolation between two quaternions.

    用於產生平滑的旋轉動畫，在 SO(3) 旋轉空間中以恆定角速度插值。
    Produces smooth rotation animation at constant angular velocity in SO(3).

    SLERP 公式:
      cos θ = q1 · q2  (四元數點積)

      若 cos θ < 0，翻轉 q2 以走短弧:
        q2 ← -q2,  cos θ ← -cos θ

      若 cos θ ≈ 1 (角度極小)，退化為線性插值:
        result = (1-t)*q1 + t*q2  (正規化)

      否則:
        θ = arccos(cos θ)
        result = sin((1-t)θ)/sin(θ) * q1 + sin(tθ)/sin(θ) * q2

    Parameters
    ----------
    q1 : np.ndarray
        起始四元數 [w, x, y, z] / Start quaternion.
    q2 : np.ndarray
        結束四元數 [w, x, y, z] / End quaternion.
    t : float
        插值因子 [0, 1]，0 = q1, 1 = q2 / Interpolation factor.

    Returns
    -------
    np.ndarray
        插值後的單位四元數 / Interpolated unit quaternion.
    """
    # 四元數點積 = cos(半角差)
    # Dot product = cos(half-angle difference)
    dot = np.dot(q1, q2)

    # 確保走最短弧 (短弧插值)
    # Ensure shortest-arc interpolation
    q2_adj = q2.copy()
    if dot < 0.0:
        q2_adj = -q2_adj
        dot = -dot

    # 鉗制避免 arccos 數值域外
    # Clamp to prevent arccos domain error
    dot = min(dot, 1.0)

    # 角度極小時退化為正規化線性插值 (NLERP)
    # Near-parallel: fall back to normalized linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2_adj - q1)
        # 正規化確保仍為單位四元數
        return result / np.linalg.norm(result)

    # 標準 SLERP 公式
    # Standard SLERP formula
    theta = math.acos(dot)           # 四元數夾角
    sin_theta = math.sin(theta)      # sin(θ) 作為分母

    # 計算插值權重
    # Compute interpolation weights
    w1 = math.sin((1.0 - t) * theta) / sin_theta
    w2 = math.sin(t * theta) / sin_theta

    return w1 * q1 + w2 * q2_adj


# ============================================================
#  四元數乘法 / Quaternion Multiplication
# ============================================================

def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    四元數乘法 (Hamilton 積)。
    Quaternion multiplication (Hamilton product).

    代表旋轉的複合：先施加 q2 的旋轉，再施加 q1 的旋轉。
    Represents rotation composition: apply q2 first, then q1.

    Hamilton 積公式:
      (a₁ + b₁i + c₁j + d₁k)(a₂ + b₂i + c₂j + d₂k) =
        w = a₁a₂ - b₁b₂ - c₁c₂ - d₁d₂
        x = a₁b₂ + b₁a₂ + c₁d₂ - d₁c₂
        y = a₁c₂ - b₁d₂ + c₁a₂ + d₁b₂
        z = a₁d₂ + b₁c₂ - c₁b₂ + d₁a₂

    Parameters
    ----------
    q1 : np.ndarray
        左四元數 [w, x, y, z] / Left quaternion.
    q2 : np.ndarray
        右四元數 [w, x, y, z] / Right quaternion.

    Returns
    -------
    np.ndarray
        乘積四元數 [w, x, y, z] / Product quaternion.
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,   # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,   # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,   # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,   # z
    ], dtype=np.float64)


# ============================================================
#  旋轉矩陣 ↔ 四元數 / Rotation Matrix ↔ Quaternion
# ============================================================

def quaternion_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """
    從 3×3 旋轉矩陣提取四元數 (Shepperd 方法)。
    Extract quaternion from a 3×3 rotation matrix using Shepperd's method.

    此演算法根據矩陣跡 (trace) 選擇最穩定的計算路徑，
    避免在特定旋轉角度下除以接近零的值。
    This algorithm selects the most numerically stable computation path
    based on the matrix trace, avoiding near-zero divisions.

    Shepperd 方法:
      trace = R[0,0] + R[1,1] + R[2,2]

      Case 1 (trace > 0):
        s = 2 * √(1 + trace)
        w = s/4,  x = (R[2,1]-R[1,2])/s, ...

      Case 2-4: 選擇對角線最大元素對應的分支
                Select branch corresponding to largest diagonal element

    Parameters
    ----------
    R : np.ndarray
        3×3 旋轉矩陣 / 3×3 rotation matrix.

    Returns
    -------
    np.ndarray
        單位四元數 [w, x, y, z] / Unit quaternion.
    """
    # 矩陣跡 = 對角線元素之和
    # Matrix trace = sum of diagonal elements
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0.0:
        # 最穩定路徑：trace > 0 表示旋轉角度 < 120°
        # Most stable path: trace > 0 means rotation angle < 120°
        s = 2.0 * math.sqrt(1.0 + trace)  # s = 4w
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s

    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        # X 軸分量最大
        # X component is largest
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])  # s = 4x
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s

    elif R[1, 1] > R[2, 2]:
        # Y 軸分量最大
        # Y component is largest
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])  # s = 4y
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s

    else:
        # Z 軸分量最大
        # Z component is largest
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])  # s = 4z
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)

    # 正規化以補償浮點累積誤差
    # Normalize to compensate for accumulated floating-point error
    return q / np.linalg.norm(q)


def rotation_matrix_from_quaternion(q: np.ndarray) -> np.ndarray:
    """
    從四元數計算 3×3 旋轉矩陣。
    Convert a unit quaternion [w, x, y, z] to a 3×3 rotation matrix.

    轉換公式:
      令 q = [w, x, y, z]，預先計算二次項:
        xx = x², yy = y², zz = z²
        xy = x·y, xz = x·z, yz = y·z
        wx = w·x, wy = w·y, wz = w·z

      R = [[1-2(yy+zz),  2(xy-wz),    2(xz+wy)  ],
           [2(xy+wz),    1-2(xx+zz),  2(yz-wx)  ],
           [2(xz-wy),    2(yz+wx),    1-2(xx+yy)]]

    Parameters
    ----------
    q : np.ndarray
        單位四元數 [w, x, y, z] / Unit quaternion.

    Returns
    -------
    np.ndarray
        3×3 旋轉矩陣 (float64) / 3×3 rotation matrix.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    # 預計算二次項以減少乘法次數
    # Pre-compute quadratic terms to reduce multiplications
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # 直接組裝旋轉矩陣
    # Directly assemble the rotation matrix
    return np.array([
        [1.0 - 2.0 * (yy + zz),   2.0 * (xy - wz),         2.0 * (xz + wy)        ],
        [2.0 * (xy + wz),         1.0 - 2.0 * (xx + zz),   2.0 * (yz - wx)        ],
        [2.0 * (xz - wy),         2.0 * (yz + wx),         1.0 - 2.0 * (xx + yy)  ],
    ], dtype=np.float64)


# ============================================================
#  三維線性插值 / 3D Linear Interpolation
# ============================================================

def lerp_3d(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """
    三維位置向量的線性插值 (LERP)。
    Linear interpolation for 3D position vectors.

    用於面板位置的平滑過渡。
    Used for smooth panel position transitions.

    公式 / Formula:
      result = a + t * (b - a)
             = (1 - t) * a + t * b

    Parameters
    ----------
    a : np.ndarray
        起始位置 [x, y, z] / Start position.
    b : np.ndarray
        結束位置 [x, y, z] / End position.
    t : float
        插值因子 [0, 1]，0 = a, 1 = b / Interpolation factor.

    Returns
    -------
    np.ndarray
        插值後的位置向量 / Interpolated position vector.
    """
    # 使用 a + t*(b-a) 形式比 (1-t)*a + t*b 少一次乘法
    # a + t*(b-a) form uses one fewer multiplication than (1-t)*a + t*b
    return a + t * (b - a)


# ============================================================
#  碰撞檢測輔助 / Collision Detection Helper
# ============================================================

def compute_bounding_sphere(center: np.ndarray, half_w: float, half_h: float) -> float:
    """
    計算軸對齊矩形面板的包圍球半徑，用於快速碰撞粗篩。
    Compute bounding sphere radius for an axis-aligned rectangular panel,
    used for fast broad-phase collision culling.

    包圍球是包含整個面板的最小球體，其半徑為面板對角線的一半。
    The bounding sphere is the smallest sphere enclosing the entire panel;
    its radius equals half the panel diagonal.

    公式 / Formula:
      radius = √(half_w² + half_h²)

    在碰撞檢測流程中，先用球體測試 (O(1)) 排除明顯不相交的物件，
    通過粗篩後再進行精確的矩形碰撞測試。
    In collision detection, sphere tests (O(1)) quickly reject obviously
    non-intersecting objects before performing precise rectangle tests.

    Parameters
    ----------
    center : np.ndarray
        面板中心 3D 座標 [x, y, z] (本函數未使用，保留供呼叫端一致性)。
        Panel center 3D position (unused here, kept for caller consistency).
    half_w : float
        面板半寬 (mm) / Half-width of the panel in mm.
    half_h : float
        面板半高 (mm) / Half-height of the panel in mm.

    Returns
    -------
    float
        包圍球半徑 (mm) / Bounding sphere radius in mm.
    """
    # 勾股定理求對角線的一半 = 包圍球半徑
    # Pythagorean theorem: half-diagonal = bounding sphere radius
    return math.sqrt(half_w * half_w + half_h * half_h)
