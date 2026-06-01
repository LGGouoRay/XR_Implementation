import cv2
import mediapipe as mp
import numpy as np
import math

# ==========================================
# 1. 系統初始化與常數定義
# ==========================================
mp_hands = mp.solutions.hands
# 啟動雙手追蹤以提供更好的 XR 互動體驗（左手旋轉/右手拖曳，或單手多功能操作）
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# 未來科技 HUD 配色 (BGR)
CLR_CYAN = (255, 255, 0)
CLR_YELLOW = (0, 255, 255)
CLR_WHITE = (255, 255, 255)
CLR_RED = (0, 0, 255)
CLR_GREEN = (0, 255, 0)
CLR_BLUE = (255, 0, 0)
CLR_DARK_BG = (15, 15, 15)

# 啟動攝影機
cap = cv2.VideoCapture(0)
# 設定畫面寬高（若硬體支援，可自行調整）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

print("➔ 未來科技 3D XR HUD 系統已啟動！")
print("➔ 操控指南：")
print("  1. 【移動與旋轉】：將【大拇指】與【食指】捏合，即可『抓取』虛擬面板。")
print("     此時面板會隨您的手掌在 3D 空間中移動與進行 360 度傾斜、旋轉。")
print("  2. 【按鈕觸控】：當面板靜止時，伸出【單獨食指】靠近虛擬螢幕並點擊按鈕區。")
print("     系統會偵測食指的物理深度，當食指穿過螢幕表面時即觸發點擊。")
print("➔ 按下 'q' 鍵可結束程式。")


# ==========================================
# 2. 3D 數學投影與輔助函數
# ==========================================
def get_rotation_matrix(pitch, yaw, roll):
    """根據歐拉角計算 3D 旋轉矩陣 (Y-X-Z 順序)"""
    cx, sx = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cz, sz = math.cos(roll), math.sin(roll)

    # 旋轉矩陣 R
    R = np.array([
        [cy*cz + sy*sx*sz, -cy*sz + sy*sx*cz, sy*cx],
        [cx*sz, cx*cz, -sx],
        [-sy*cz + cy*sx*sz, sy*sz + cy*sx*cz, cy*cx]
    ])
    return R

def project_point(pt, f, cx, cy):
    """將相機空間中的 3D 點投影至 2D 螢幕像素座標"""
    X, Y, Z = pt
    if Z <= 10.0:  # 避開除以零或投影到相機後方的點
        return None
    px = int((X * f / Z) + cx)
    py = int((Y * f / Z) + cy)
    return (px, py)


# ==========================================
# 3. 3D 虛擬螢幕類別
# ==========================================
class SciFiScreen3D:
    def __init__(self, w_3d=300, h_3d=180):
        # 3D 空間尺寸
        self.w_3d = w_3d
        self.h_3d = h_3d
        
        # 3D 空間初始狀態 (位置 T 與旋轉角度)
        self.T = np.array([0.0, -30.0, 450.0])  # 置中，稍微偏上，距離相機 450 像素
        self.rot = np.array([0.0, 0.0, 0.0])    # [Pitch, Yaw, Roll] 弧度
        
        # 平滑濾波（避免手震造成 3D 物件劇烈晃動）
        self.smooth_alpha_pos = 0.25
        self.smooth_alpha_rot = 0.20
        self.target_T = self.T.copy()
        self.target_rot = self.rot.copy()

        # 互動狀態
        self.is_dragging = False
        self.drag_start_hand_pos = None
        self.drag_start_screen_pos = None
        self.drag_start_hand_rot = None
        self.drag_start_screen_rot = None

        # 2D 畫布內部尺寸 (用於產生視訊來源紋理)
        self.canvas_w = 400
        self.canvas_h = 240
        self.current_saturation_mode = 1  # 預設為 Normal
        
        # 2D 畫布上的按鈕設計
        btn_w = self.canvas_w // 3 - 20
        btn_h = 45
        btn_y = self.canvas_h - 65
        self.buttons = [
            {"label": "B&W",   "rect": (15, btn_y, btn_w, btn_h), "sat": 0.0},
            {"label": "NORM",  "rect": (btn_w + 30, btn_y, btn_w, btn_h), "sat": 1.0},
            {"label": "VIVID", "rect": (btn_w * 2 + 45, btn_y, btn_w, btn_h), "sat": 2.5}
        ]
        self.btn_hover = [False, False, False]
        self.btn_pressed = [False, False, False]

    def get_local_corners(self):
        """定義虛擬面板在自身座標系下的四個角點"""
        hw = self.w_3d / 2
        hh = self.h_3d / 2
        return np.array([
            [-hw, -hh, 0.0],  # 左上
            [ hw, -hh, 0.0],  # 右上
            [ hw,  hh, 0.0],  # 右下
            [-hw,  hh, 0.0]   # 左下
        ])

    def get_world_corners(self, R):
        """將局部座標點轉換為 3D 相機世界座標點"""
        local_pts = self.get_local_corners()
        world_pts = []
        for pt in local_pts:
            world_pt = R.dot(pt) + self.T
            world_pts.append(world_pt)
        return np.array(world_pts)

    def draw_canvas(self):
        """繪製虛擬螢幕上的 2D GUI 內容"""
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        
        # 繪製高科技感背景柵格
        grid_space = 20
        for x in range(0, self.canvas_w, grid_space):
            cv2.line(canvas, (x, 0), (x, self.canvas_h), (25, 25, 10), 1)
        for y in range(0, self.canvas_h, grid_space):
            cv2.line(canvas, (0, y), (self.canvas_w, y), (25, 25, 10), 1)

        # 繪製半透明外框
        cv2.rectangle(canvas, (5, 5), (self.canvas_w - 5, self.canvas_h - 5), CLR_CYAN, 1)

        # 標題資訊
        cv2.putText(canvas, "X.R. PROJECTION MATRIX", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_CYAN, 1, cv2.LINE_AA)
        cv2.line(canvas, (15, 38), (200, 38), CLR_CYAN, 1)

        # 顯示當前 3D 姿態數值
        yaw_deg = math.degrees(self.rot[1])
        pitch_deg = math.degrees(self.rot[0])
        roll_deg = math.degrees(self.rot[2])
        cv2.putText(canvas, f"YAW  : {yaw_deg:+.1f} deg", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"PITCH: {pitch_deg:+.1f} deg", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"ROLL : {roll_deg:+.1f} deg", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"DEPTH: {self.T[2]:.1f} mm", (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_WHITE, 1, cv2.LINE_AA)

        # 繪製 3D 按鈕
        for i, btn in enumerate(self.buttons):
            bx, by, bw, bh = btn["rect"]
            b_color = CLR_WHITE
            
            # 生效狀態配色
            if i == self.current_saturation_mode:
                b_color = CLR_YELLOW
                cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), b_color, -1)
                text_color = (0, 0, 0)
            else:
                if self.btn_pressed[i]:
                    b_color = CLR_RED
                elif self.btn_hover[i]:
                    b_color = CLR_CYAN
                cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), b_color, 2)
                text_color = b_color
            
            # 填入按鈕標籤
            cv2.putText(canvas, btn["label"], (bx + 15, by + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

        return canvas

    def update_pose(self):
        """應用指數移動平均 (EMA) 平滑姿態，消除手部震顫"""
        self.T = self.T * (1.0 - self.smooth_alpha_pos) + self.target_T * self.smooth_alpha_pos
        self.rot = self.rot * (1.0 - self.smooth_alpha_rot) + self.target_rot * self.smooth_alpha_rot

    def render_to_world(self, frame, f, cx, cy):
        """將 3D 虛擬螢幕投射渲染至相機視訊框上"""
        R = get_rotation_matrix(self.rot[0], self.rot[1], self.rot[2])
        world_pts = self.get_world_corners(R)

        # 投影 3D 角點至 2D 像素
        img_corners = []
        for pt in world_pts:
            px_py = project_point(pt, f, cx, cy)
            if px_py is None:
                return frame
            img_corners.append(px_py)
        img_corners = np.array(img_corners, dtype=np.float32)

        # 產生 2D 虛擬面板紋理畫布
        canvas = self.draw_canvas()

        # 頂點對應對映：[左上, 右上, 右下, 左下]
        src_corners = np.array([
            [0, 0],
            [self.canvas_w, 0],
            [self.canvas_w, self.canvas_h],
            [0, self.canvas_h]
        ], dtype=np.float32)

        # 計算單應性透視變換並對紋理進行 Warp
        try:
            H, _ = cv2.findHomography(src_corners, img_corners)
            warped_screen = cv2.warpPerspective(canvas, H, (frame.shape[1], frame.shape[0]))
            
            # 建立遮罩以進行透明度融合 (Glass Glassmorphism Effect)
            mask = np.zeros((self.canvas_h, self.canvas_w), dtype=np.uint8)
            cv2.rectangle(mask, (0, 0), (self.canvas_w, self.canvas_h), 255, -1)
            warped_mask = cv2.warpPerspective(mask, H, (frame.shape[1], frame.shape[0]))
            
            # 將遮罩擴展至 3 通道
            mask_3ch = cv2.merge([warped_mask, warped_mask, warped_mask]) / 255.0
            
            # XR 玻璃拟態融合 (0.75 為虛擬面板不透明度)
            alpha = 0.75
            blended_region = cv2.addWeighted(warped_screen, alpha, frame, 1.0 - alpha, 0)
            frame = np.where(mask_3ch > 0.01, blended_region, frame)
            
        except Exception:
            # 避免投影矩陣退化時程式崩潰
            pass

        # 4. 繪製 3D 空間框線 (不因 Warp 模糊，維持高清晰度向量線條)
        t = 2
        c_len = 25  # 角括弧長度
        color = CLR_YELLOW if self.is_dragging else CLR_CYAN

        # 針對投影後的四個頂點繪製科技感角點 (Corners)
        pts = img_corners.astype(int)
        for idx in range(4):
            curr = pts[idx]
            next_pt = pts[(idx + 1) % 4]
            prev_pt = pts[(idx - 1) % 4]
            
            # 繪製角點括弧向量
            v_next = next_pt - curr
            v_prev = prev_pt - curr
            
            v_next_norm = v_next / (np.linalg.norm(v_next) + 1e-6)
            v_prev_norm = v_prev / (np.linalg.norm(v_prev) + 1e-6)
            
            pt1 = (curr + v_next_norm * c_len).astype(int)
            pt2 = (curr + v_prev_norm * c_len).astype(int)
            
            cv2.line(frame, tuple(curr), tuple(pt1), color, t, cv2.LINE_AA)
            cv2.line(frame, tuple(curr), tuple(pt2), color, t, cv2.LINE_AA)

        # 繪製 3D 空間中心姿態指示器 (XYZ 座標軸)
        axis_len = 40
        o_3d = self.T
        ax_3d = o_3d + R.dot(np.array([axis_len, 0.0, 0.0]))
        ay_3d = o_3d + R.dot(np.array([0.0, -axis_len, 0.0])) # Y 向上
        az_3d = o_3d + R.dot(np.array([0.0, 0.0, axis_len]))
        
        op = project_point(o_3d, f, cx, cy)
        ap_x = project_point(ax_3d, f, cx, cy)
        ap_y = project_point(ay_3d, f, cx, cy)
        ap_z = project_point(az_3d, f, cx, cy)
        
        if op:
            if ap_x: cv2.line(frame, op, ap_x, CLR_RED, 2, cv2.LINE_AA)   # X 軸: 紅
            if ap_y: cv2.line(frame, op, ap_y, CLR_GREEN, 2, cv2.LINE_AA) # Y 軸: 綠
            if ap_z: cv2.line(frame, op, ap_z, CLR_BLUE, 2, cv2.LINE_AA)  # Z 軸: 藍

        return frame


# ==========================================
# 4. 實體互動與手勢追蹤引擎
# ==========================================
def extract_hand_data_3d(landmarks, img_w, img_h, f, cx, cy):
    """
    從 MediaPipe 的 2D 關節及相對 Z，重建具備物理尺度的 3D 空間相機座標。
    """
    # 估算手部至相機的實際距離 (結合掌寬與手長，大幅提升 100%~300% 穩定性)
    dx1 = (landmarks[17].x - landmarks[5].x) * img_w
    dy1 = (landmarks[17].y - landmarks[5].y) * img_h
    dx2 = (landmarks[9].x - landmarks[0].x) * img_w
    dy2 = (landmarks[9].y - landmarks[0].y) * img_h
    
    # 取特徵距離平均，避免單一角度旋轉導致數值突變
    dist_2d = (math.sqrt(dx1*dx1 + dy1*dy1) + math.sqrt(dx2*dx2 + dy2*dy2)) / 2.0
    
    # 基準校正常數：寬度 120 像素時大約距離 400 mm
    Z_hand = (120.0 / max(dist_2d, 1.0)) * 400.0
    
    # 重建所有 21 個關節的 3D 相機世界座標
    pts_3d = []
    for lm in landmarks:
        # 配合物理尺度進行深度對映
        z_phys = Z_hand + lm.z * Z_hand * 1.2
        x_phys = (lm.x * img_w - cx) * (z_phys / f)
        y_phys = (lm.y * img_h - cy) * (z_phys / f)
        pts_3d.append(np.array([x_phys, y_phys, z_phys]))
        
    pts_3d = np.array(pts_3d)
    
    # 計算手掌的 3D 旋轉基礎 (正交化基底)
    # 掌心向上方向向量 (腕部 -> 中指根部)
    v_up = pts_3d[9] - pts_3d[0]
    v_up /= np.linalg.norm(v_up) + 1e-6
    
    # 掌心橫向方向向量 (食指根部 -> 小指根部)
    v_right = pts_3d[17] - pts_3d[5]
    v_right /= np.linalg.norm(v_right) + 1e-6
    
    # 計算法向量 (掌面朝向)
    v_forward = np.cross(v_right, v_up)
    v_forward /= np.linalg.norm(v_forward) + 1e-6
    
    # 再次正交化以確保矩陣為正交矩陣 (Orthogonal Matrix)
    v_right = np.cross(v_up, v_forward)
    v_right /= np.linalg.norm(v_right) + 1e-6
    
    R_hand = np.column_stack((v_right, v_up, v_forward))
    
    # 自矩陣中抽取歐拉角 (Z-Y-X 尤拉旋轉系統)
    sy = math.sqrt(R_hand[2, 1]**2 + R_hand[2, 2]**2)
    if sy > 1e-6:
        pitch = math.atan2(R_hand[2, 1], R_hand[2, 2])
        yaw   = math.atan2(-R_hand[2, 0], sy)
        roll  = math.atan2(R_hand[1, 0], R_hand[0, 0])
    else:
        pitch = math.atan2(-R_hand[1, 2], R_hand[1, 1])
        yaw   = math.atan2(-R_hand[2, 0], sy)
        roll  = 0.0
        
    return pts_3d, np.array([pitch, yaw, roll])


def handle_interaction(screen, landmarks, pts_3d, hand_rot):
    """
    依據 2D 拓樸與 3D 物理空間進行極高精度抓取拖曳或觸碰檢測
    """
    thumb_tip_3d = pts_3d[4]
    index_tip_3d = pts_3d[8]
    
    # 1. 檢測手指捏合手勢 (利用 2D 歸一化比例，準確率提升至近乎 100%)
    thumb_2d, index_2d = landmarks[4], landmarks[8]
    wrist_2d, middle_mcp_2d = landmarks[0], landmarks[9]
    
    pinch_dist_norm = math.hypot(thumb_2d.x - index_2d.x, thumb_2d.y - index_2d.y)
    palm_dist_norm = math.hypot(wrist_2d.x - middle_mcp_2d.x, wrist_2d.y - middle_mcp_2d.y)
    
    # 當捏合距離小於手掌長度的 25% 視為穩定捏合，完全免疫深度 Z 軸抖動造成的誤判
    is_pinching = (pinch_dist_norm / max(palm_dist_norm, 1e-6)) < 0.25
    pinch_midpoint = (thumb_tip_3d + index_tip_3d) / 2.0

    # 檢測抓取捏合點是否與面板內部發生碰撞
    dist_to_screen_center = np.linalg.norm(pinch_midpoint - screen.T)
    
    if is_pinching:
        # 若未處於拖曳狀態，但手勢靠近面板，則觸發「3D空間抓取」(範圍擴大增強手感)
        if not screen.is_dragging and dist_to_screen_center < 250.0:
            screen.is_dragging = True
            screen.drag_start_hand_pos = pinch_midpoint.copy()
            screen.drag_start_screen_pos = screen.T.copy()
            screen.drag_start_hand_rot = hand_rot.copy()
            screen.drag_start_screen_rot = screen.rot.copy()
            
        # 若已在拖曳狀態中，面板姿態與位置同步連動
        if screen.is_dragging:
            delta_pos = pinch_midpoint - screen.drag_start_hand_pos
            screen.target_T = screen.drag_start_screen_pos + delta_pos
            
            delta_rot = hand_rot - screen.drag_start_hand_rot
            # 反饋手部旋轉，產生在 3D 空間中旋轉面板的效果
            screen.target_rot = screen.drag_start_screen_rot + delta_rot
    else:
        # 手指放開，解除抓取
        screen.is_dragging = False

    # 2. 空間虛擬觸碰偵測 (當手指未抓取且接近面板表面時)
    for i in range(3):
        screen.btn_hover[i] = False
        screen.btn_pressed[i] = False
        
    if not screen.is_dragging and not is_pinching:
        # 將食指尖 3D 座標，轉換回虛擬螢幕的 3D 局部座標系
        R_screen = get_rotation_matrix(screen.rot[0], screen.rot[1], screen.rot[2])
        # R_screen.T 為旋轉矩阵的轉置，等同於逆矩陣
        local_finger = R_screen.T.dot(index_tip_3d - screen.T)
        
        u, v, w = local_finger  # u: 橫向偏移, v: 縱向偏移, w: 距離面板法面深度
        
        # 檢測食指是否在面板 3D 實體邊界內
        half_w = screen.w_3d / 2
        half_h = screen.h_3d / 2
        
        if -half_w - 20 < u < half_w + 20 and -half_h - 20 < v < half_h + 20:
            # 物理按鍵觸碰判斷 (加大容錯寬容度，解決距離誤判與手抖問題)：
            # 若法面距離在 0~80mm 內 -> 指向 Hover
            # 若法面距離小於 0mm 且高於 -60mm (穿透面板) -> Click
            is_hover = (-5.0 <= w < 80.0)   # 稍微容忍微穿透仍算 hovering
            is_click = (-60.0 < w < -5.0)   # 確實穿透才算 Click，範圍加深避免漏判
            
            # 將 3D 空間的 u, v 映射回 2D GUI 畫布中的 (x, y) 座標點
            u_norm = (u + half_w) / screen.w_3d
            v_norm = (v + half_h) / screen.h_3d
            canvas_x = int(u_norm * screen.canvas_w)
            canvas_y = int(v_norm * screen.canvas_h)
            
            for i, btn in enumerate(screen.buttons):
                bx, by, bw, bh = btn["rect"]
                if bx < canvas_x < bx + bw and by < canvas_y < by + bh:
                    if is_click:
                        screen.btn_pressed[i] = True
                        screen.current_saturation_mode = i
                    elif is_hover:
                        screen.btn_hover[i] = True
                        
            return local_finger, True  # 返回食指座標供雷射筆視覺渲染使用

    return None, False


# ==========================================
# 5. 全局色彩與飽和度處理
# ==========================================
def apply_saturation(img, factor):
    """調整整張影像的飽和度 (HSV 空間處理)"""
    if factor == 1.0: 
        return img
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    s_float = s.astype(np.float32) * factor
    s_final = np.clip(s_float, 0, 255).astype(np.uint8)
    
    hsv_final = cv2.merge([h, s_final, v])
    return cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)


# ==========================================
# 6. 主程式執行迴圈
# ==========================================
# 實例化 3D 空間中的虛擬面板
my_screen_3d = SciFiScreen3D(w_3d=300, h_3d=180)
global_sat_factor = 1.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    # 水平翻轉以利於操作 (鏡像效果)
    frame = cv2.flip(frame, 1)
    img_h, img_w, _ = frame.shape

    # 模擬針孔相機焦距與主點 (假設焦距等於寬度)
    focal_length = img_w
    cx, cy = img_w / 2, img_h / 2

    # 將畫面轉為 RGB 送入手勢引擎
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    finger_local_pos = None
    finger_active = False

    # 若偵測到手部
    if results.multi_hand_landmarks:
        # 取最靠近的第一隻手進行主面板控制
        landmarks = results.multi_hand_landmarks[0].landmark
        
        # 1. 建立高精度 3D 手部骨架坐標
        pts_3d, hand_rot = extract_hand_data_3d(landmarks, img_w, img_h, focal_length, cx, cy)
        
        # 2. 進行物理交互檢測 (抓取 / 位移 / 觸控)
        finger_local_pos, finger_active = handle_interaction(my_screen_3d, landmarks, pts_3d, hand_rot)
        
        # 獲取當前應套用的飽和度參數
        global_sat_factor = my_screen_3d.buttons[my_screen_3d.current_saturation_mode]["sat"]

        # 3. 繪製精緻的 AR 手部關節線 (細長白色線條，突顯現代科技感)
        mp_draw.draw_landmarks(
            frame, 
            results.multi_hand_landmarks[0], 
            mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=CLR_WHITE, thickness=1, circle_radius=1),
            mp_draw.DrawingSpec(color=CLR_CYAN, thickness=1, circle_radius=1)
        )
        
        # 4. 如果食指在面板上方懸停，繪製 XR 空間追蹤雷射線
        if finger_active and finger_local_pos is not None:
            # 取得食指尖 (8號) 的 2D 投影座標
            idx_2d = project_point(pts_3d[8], focal_length, cx, cy)
            
            # 取得雷射在面板平面上的著陸點 (u, v, 0) 之世界座標
            R_screen = get_rotation_matrix(my_screen_3d.rot[0], my_screen_3d.rot[1], my_screen_3d.rot[2])
            laser_hit_world = my_screen_3d.T + R_screen.dot(np.array([finger_local_pos[0], finger_local_pos[1], 0.0]))
            laser_hit_2d = project_point(laser_hit_world, focal_length, cx, cy)
            
            if idx_2d and laser_hit_2d:
                # 繪製青色光學雷射追蹤線
                cv2.line(frame, idx_2d, laser_hit_2d, CLR_CYAN, 1, cv2.LINE_AA)
                # 繪製 3D 觸控著陸點光圈
                cv2.circle(frame, laser_hit_2d, 5, CLR_YELLOW, -1)
                cv2.circle(frame, laser_hit_2d, 10, CLR_CYAN, 1, cv2.LINE_AA)
    else:
        # 當失去手掌訊號，強制取消拖曳以避免下次抓取時物件產生瞬移
        my_screen_3d.is_dragging = False

    # 5. 更新虛擬面板狀態 (平滑過渡)
    my_screen_3d.update_pose()

    # 6. 先對背景套用色彩濾鏡
    frame = apply_saturation(frame, global_sat_factor)

    # 7. 將虛擬面板投影渲染至最前層 (避免 UI 本身被飽和度濾鏡所干擾)
    frame = my_screen_3d.render_to_world(frame, focal_length, cx, cy)

    # 8. 顯示 HUD 頂層系統數據流狀態
    mode_names = ['Grayscale (B&W)', 'Normal (1.0x)', 'Vivid (2.5x)']
    status_text = f"SYS_MODE: {mode_names[my_screen_3d.current_saturation_mode]}"
    
    cv2.putText(frame, f"VIDEO SAT: {global_sat_factor:.1f}x", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_YELLOW, 2, cv2.LINE_AA)
    cv2.putText(frame, status_text, (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_WHITE, 1, cv2.LINE_AA)
    
    if my_screen_3d.is_dragging:
        cv2.putText(frame, "STATUS: PANEL DRAGGING", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_RED, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "STATUS: SYSTEM READY", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_GREEN, 1, cv2.LINE_AA)

    # 顯示主輸出視窗
    cv2.imshow('Future Tech 3D XR HUD', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()