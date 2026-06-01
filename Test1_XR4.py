import cv2
import mediapipe as mp
import numpy as np
import math
import time

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

# 臉部追蹤初始化 (用於 Face-Follow 模式)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    refine_landmarks=True
)

# 姿勢追蹤初始化 (用於偵測手臂交叉)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# 未來科技 HUD 配色 (BGR)
CLR_CYAN = (255, 255, 0)
CLR_YELLOW = (0, 255, 255)
CLR_WHITE = (255, 255, 255)
CLR_RED = (0, 0, 255)
CLR_GREEN = (0, 255, 0)
CLR_BLUE = (255, 0, 0)
CLR_DARK_BG = (15, 15, 15)
CLR_ORANGE = (0, 165, 255)
CLR_MAGENTA = (255, 0, 255)
CLR_PURPLE = (180, 50, 200)

# 啟動攝影機
cap = cv2.VideoCapture(0)
# 設定畫面寬高
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

print("➔ 未來科技 3D XR HUD 系統已啟動！")
print("➔ 操控指南：")
print("  1. 【移動與旋轉】：將【大拇指】與【食指】捏合，即可『抓取』虛擬面板。")
print("     此時面板會隨您的手掌在 3D 空間中移動與進行 360 度傾斜、旋轉。")
print("  2. 【按鈕觸控】：將光標（食指指尖）對準虛擬面板上的按鈕。")
print("     只要將【食指指尖與中指指尖並攏】，即可精準觸發點擊！")
print("  3. 【模式切換】：雙手交叉比『叉叉』(X) 持續 2 秒即可切換模式。")
print("     模式循環: 預設模式 → 臉部跟隨模式 → 鍵盤模式 → 預設模式")
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
    if Z <= 10.0:  
        return None
    px = int((X * f / Z) + cx)
    py = int((Y * f / Z) + cy)
    return (px, py)


# ==========================================
# 3. 交叉手勢 (叉叉) 偵測器
# ==========================================
class CrossGestureDetector:
    def __init__(self, hold_duration=1.2):
        self.hold_duration = hold_duration
        self.cross_start_time = None
        self.is_crossing = False
        self.progress = 0.0       
        self.just_triggered = False  
        self.last_seen_time = 0      

    def _segments_intersect_2d(self, p1, p2, p3, p4):
        def cross2d(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        d1 = cross2d(p3, p4, p1)
        d2 = cross2d(p3, p4, p2)
        d3 = cross2d(p1, p2, p3)
        d4 = cross2d(p1, p2, p4)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def detect(self, pose_landmarks, img_w, img_h):
        self.just_triggered = False

        if pose_landmarks is None:
            return self._handle_absence()

        lm = pose_landmarks.landmark
        
        vis_thresh = 0.3
        if (lm[13].visibility < vis_thresh or lm[14].visibility < vis_thresh or 
            lm[15].visibility < vis_thresh or lm[16].visibility < vis_thresh):
            return self._handle_absence()

        elbow_l = np.array([lm[13].x * img_w, lm[13].y * img_h])
        wrist_l = np.array([lm[15].x * img_w, lm[15].y * img_h])
        elbow_r = np.array([lm[14].x * img_w, lm[14].y * img_h])
        wrist_r = np.array([lm[16].x * img_w, lm[16].y * img_h])

        vec_l = wrist_l - elbow_l
        vec_r = wrist_r - elbow_r

        wrist_l_ext = elbow_l + vec_l * 1.5
        elbow_l_ext = elbow_l - vec_l * 0.2
        wrist_r_ext = elbow_r + vec_r * 1.5
        elbow_r_ext = elbow_r - vec_r * 0.2

        if (wrist_l[1] > elbow_l[1] + 80) and (wrist_r[1] > elbow_r[1] + 80):
            return self._handle_absence()

        segments_cross = self._segments_intersect_2d(elbow_l_ext, wrist_l_ext, elbow_r_ext, wrist_r_ext)

        dot_val = np.dot(vec_l, vec_r)
        mag_l = np.linalg.norm(vec_l)
        mag_r = np.linalg.norm(vec_r)
        cos_angle = dot_val / (mag_l * mag_r + 1e-6)
        angle_deg = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))

        angle_ok = 30.0 < angle_deg < 160.0

        avg_arm_len = (mag_l + mag_r) / 2.0
        wrist_dist = np.linalg.norm(wrist_l - wrist_r)
        wrists_close = wrist_dist < avg_arm_len * 0.8
        
        is_crossing_now = (segments_cross or wrists_close) and angle_ok

        if is_crossing_now:
            now = time.time()
            if not self.is_crossing:
                self.cross_start_time = now
                self.is_crossing = True
            
            self.last_seen_time = now
            elapsed = now - self.cross_start_time
            self.progress = min(elapsed / self.hold_duration, 1.0)

            if self.progress >= 1.0:
                self.just_triggered = True
                self._reset()
                return True, 1.0

            return False, self.progress
        else:
            return self._handle_absence()

    def _handle_absence(self):
        now = time.time()
        if self.is_crossing and (now - self.last_seen_time < 0.3):
            return False, self.progress
        else:
            self._reset()
            return False, 0.0

    def _reset(self):
        self.cross_start_time = None
        self.is_crossing = False
        self.progress = 0.0
        self.last_seen_time = 0


# ==========================================
# 4. 臉部位置追蹤器 (高精度 3D 向量重構版 v2)
# ==========================================
class FaceTracker:
    """使用 MediaPipe Face Mesh 精準追蹤臉部位置與朝向。
    
    核心改進:
    - 以 Landmark 168 (印堂/雙眼中心) 作為 3D 錨點
    - 使用臉頰寬度 (Landmark 234/454) 計算穩定深度 (假設人臉寬 140mm)
    - 精確法向量計算 + 正交化 (Gram-Schmidt)
    - 360 度正向顯示 (billboard 修正)
    - Alpha=0.15 低通濾波消除高頻抖動
    """
    AVERAGE_FACE_WIDTH_MM = 140.0  # 人臉平均寬度 (mm)
    SCREEN_DISTANCE_MM = 150.0     # 螢幕投射距離 (mm)，大約 15cm

    def __init__(self):
        # 位置平滑 (3D 世界座標, mm)
        self.smooth_center = np.array([0.0, 0.0, 500.0])
        
        # 朝向平滑 (歐拉角, radians)
        self.smooth_pitch = 0.0
        self.smooth_yaw = 0.0
        self.smooth_roll = 0.0
        
        # 平滑的前方向量
        self.smooth_v_forward = np.array([0.0, 0.0, 1.0])
        
        # 低通濾波係數 (越小越平滑，但延遲越大)
        self.alpha = 0.15
        
        self.detected = False
        self._initialized = False

    def _landmark_to_3d(self, lm, img_w, img_h, f, cx, cy, Z_base):
        """將單一 landmark 轉換為相機空間 3D 座標 (mm)"""
        # MediaPipe 的 lm.z 是相對深度 (以臉寬為尺度)，乘上基準深度得到物理偏移
        z_phys = Z_base + lm.z * Z_base * 0.5
        x_phys = (lm.x * img_w - cx) * (z_phys / f)
        y_phys = (lm.y * img_h - cy) * (z_phys / f)
        return np.array([x_phys, y_phys, z_phys])

    def update(self, face_results, img_w, img_h, f, cx, cy):
        if not face_results.multi_face_landmarks:
            self.detected = False
            return
        
        face_lm = face_results.multi_face_landmarks[0].landmark
        
        # ── Step 1: 穩定深度估算 (使用臉頰寬度) ──
        # Landmark 234 = 左臉頰, Landmark 454 = 右臉頰
        left_cheek = face_lm[234]
        right_cheek = face_lm[454]
        
        # 在 2D 影像上的臉頰距離 (pixels)
        cheek_dx = (left_cheek.x - right_cheek.x) * img_w
        cheek_dy = (left_cheek.y - right_cheek.y) * img_h
        cheek_dist_px = math.sqrt(cheek_dx * cheek_dx + cheek_dy * cheek_dy)
        
        # 透過針孔相機模型反推深度: Z = (real_width * focal_length) / pixel_width
        Z_base = (self.AVERAGE_FACE_WIDTH_MM * f) / max(cheek_dist_px, 1.0)
        
        # ── Step 2: 取得關鍵 3D 點 ──
        # Landmark 168 = 印堂 (雙眼之間, 鼻樑上方) — 作為中心錨點
        # Landmark 10  = 額頭頂部
        # Landmark 152 = 下巴底部
        # Landmark 234 = 左臉頰
        # Landmark 454 = 右臉頰
        key_indices = [168, 10, 152, 234, 454]
        pts = {}
        for idx in key_indices:
            pts[idx] = self._landmark_to_3d(face_lm[idx], img_w, img_h, f, cx, cy, Z_base)
        
        # 雙眼中心 (印堂) 作為螢幕投射的起始點
        eye_center = pts[168]
        
        # ── Step 3: 建立正交基底 (臉部與螢幕座標系) ──
        # v_up: 從下巴指向額頭 (臉的上方向)
        v_up_raw = pts[10] - pts[152]
        
        # v_right: 從左到右在鏡像畫面中 (由 234 指向 454)
        v_right_raw = pts[454] - pts[234]
        
        # 使用 Gram-Schmidt 正交化確保基底正交
        # 1) 先正規化 v_right
        v_right = v_right_raw / (np.linalg.norm(v_right_raw) + 1e-8)
        
        # 2) v_up 去除 v_right 分量後正規化
        v_up = v_up_raw - np.dot(v_up_raw, v_right) * v_right
        v_up = v_up / (np.linalg.norm(v_up) + 1e-8)
        
        # 3) v_forward = v_right × v_up (右手定則: 指向臉部前方，朝向相機方向)
        v_forward = np.cross(v_right, v_up)
        v_forward = v_forward / (np.linalg.norm(v_forward) + 1e-8)
        
        # ── Step 4: 建立螢幕的 3D 旋轉矩陣 R_screen ──
        # 當人臉朝向正前方 (v_right=[1,0,0], v_up=[0,-1,0], v_forward=[0,0,-1]) 時，
        # 螢幕應為 Identity，其基底分別為 v_right, -v_up, -v_forward
        R_screen = np.column_stack((v_right, -v_up, -v_forward))
        
        # ── Step 5: 從 R_screen 提取與 get_rotation_matrix() 一致的歐拉角 ──
        # 與 get_rotation_matrix() 中的 Y-X-Z 順序一致：
        # R[1, 2] = -sx, R[1, 0] = cx*sz, R[1, 1] = cx*cz
        # R[0, 2] = sy*cx, R[2, 2] = cy*cx
        cx = math.sqrt(R_screen[1, 0]**2 + R_screen[1, 1]**2)
        if cx > 1e-6:
            screen_pitch = math.atan2(-R_screen[1, 2], cx)
            screen_yaw   = math.atan2(R_screen[0, 2], R_screen[2, 2])
            screen_roll  = math.atan2(R_screen[1, 0], R_screen[1, 1])
        else:
            screen_pitch = math.atan2(-R_screen[1, 2], 0.0)
            screen_yaw   = math.atan2(-R_screen[2, 0], R_screen[0, 0])
            screen_roll  = 0.0
        
        # ── Step 6: 低通濾波平滑 ──
        if not self._initialized:
            # 第一次偵測到臉部時，直接設定初始值 (不做平滑)
            self.smooth_center = eye_center.copy()
            self.smooth_pitch = screen_pitch
            self.smooth_yaw = screen_yaw
            self.smooth_roll = screen_roll
            self.smooth_v_forward = v_forward.copy()
            self._initialized = True
        else:
            a = self.alpha
            self.smooth_center = self.smooth_center * (1.0 - a) + eye_center * a
            self.smooth_v_forward = self.smooth_v_forward * (1.0 - a) + v_forward * a
            self.smooth_v_forward /= np.linalg.norm(self.smooth_v_forward) + 1e-8
            
            # 角度平滑 (注意角度環繞問題，使用 atan2 差值)
            def smooth_angle(old, new, alpha):
                diff = math.atan2(math.sin(new - old), math.cos(new - old))
                return old + alpha * diff
            
            self.smooth_pitch = smooth_angle(self.smooth_pitch, screen_pitch, a)
            self.smooth_yaw = smooth_angle(self.smooth_yaw, screen_yaw, a)
            self.smooth_roll = smooth_angle(self.smooth_roll, screen_roll, a)
        
        self.detected = True

    def get_screen_target_pos_and_rot(self, f, cx, cy):
        """計算虛擬螢幕的目標 3D 位置與旋轉角度。
        
        螢幕位於雙眼中心正前方 150mm 處，面向使用者。
        """
        if not self.detected:
            return None, None
        
        # 沿平滑後的前方向量投射 150mm
        screen_pos = self.smooth_center + self.smooth_v_forward * self.SCREEN_DISTANCE_MM
        
        # 確保深度不會太小 (防止投影崩潰)
        screen_pos[2] = max(screen_pos[2], 80.0)
        
        target_rot = np.array([self.smooth_pitch, self.smooth_yaw, self.smooth_roll])
        
        return screen_pos, target_rot


# ==========================================
# 5. 虛擬鍵盤類別
# ==========================================
class VirtualKeyboard:
    def __init__(self):
        self.canvas_w = 600
        self.canvas_h = 280
        self.w_3d = 360
        self.h_3d = 168

        self.rows = [
            list("QWERTYUIOP"),
            list("ASDFGHJKL"),
            list("ZXCVBNM"),
            ["SPACE", "DEL", "ENTER"]
        ]

        self.typed_text = ""
        self.cursor_blink = 0

        self.T = np.array([0.0, 0.0, 480.0])
        self.rot = np.array([0.0, 0.0, 0.0]) 

        self.target_T = self.T.copy()
        self.target_rot = self.rot.copy()
        self.smooth_alpha = 0.5

        self.key_cooldown = {}
        self.cooldown_time = 0.4 

        self.hover_key = None
        self.pressed_key = None
        self.press_flash_time = 0

    def get_key_rects(self):
        rects = []
        margin_x = 8
        margin_y = 70  
        key_h = 42
        gap = 5

        for row_idx, row in enumerate(self.rows):
            if row_idx == 3:  
                total_w = self.canvas_w - margin_x * 2
                space_w = total_w // 2 - gap
                del_w = total_w // 4 - gap
                enter_w = total_w // 4 - gap
                y = margin_y + row_idx * (key_h + gap)
                x = margin_x
                rects.append((x, y, space_w, key_h, "SPACE"))
                x += space_w + gap
                rects.append((x, y, del_w, key_h, "DEL"))
                x += del_w + gap
                rects.append((x, y, enter_w, key_h, "ENTER"))
            else:
                n_keys = len(row)
                total_w = self.canvas_w - margin_x * 2
                key_w = (total_w - gap * (n_keys - 1)) // n_keys
                offset_x = margin_x + (row_idx % 2) * 10 
                y = margin_y + row_idx * (key_h + gap)
                for k_idx, key_label in enumerate(row):
                    x = offset_x + k_idx * (key_w + gap)
                    rects.append((x, y, key_w, key_h, key_label))
        return rects

    def draw_canvas(self):
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        for y in range(self.canvas_h):
            alpha = y / self.canvas_h
            val = int(10 + alpha * 15)
            canvas[y, :] = (val, val + 2, val)

        cv2.rectangle(canvas, (2, 2), (self.canvas_w - 2, self.canvas_h - 2), CLR_MAGENTA, 1)

        cv2.rectangle(canvas, (10, 10), (self.canvas_w - 10, 60), (30, 30, 30), -1)
        cv2.rectangle(canvas, (10, 10), (self.canvas_w - 10, 60), CLR_CYAN, 1)
        
        cursor = "_" if int(time.time() * 2) % 2 == 0 else ""
        disp_txt = self.typed_text if len(self.typed_text) < 30 else "..." + self.typed_text[-27:]
        cv2.putText(canvas, f"INPUT> {disp_txt}{cursor}", (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_WHITE, 1, cv2.LINE_AA)

        key_rects = self.get_key_rects()
        now = time.time()

        for (x, y, w, h, label) in key_rects:
            color = (40, 40, 40)
            text_color = CLR_WHITE
            border_color = (80, 80, 80)

            if self.pressed_key == label and (now - self.press_flash_time) < 0.15:
                color = (0, 200, 200)  
                text_color = (0, 0, 0)
                border_color = CLR_YELLOW
            elif self.hover_key == label:
                color = (50, 60, 60)
                border_color = CLR_CYAN

            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, 1)

            font_scale = 0.35 if len(label) > 1 else 0.45
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            tx = x + (w - text_size[0]) // 2
            ty = y + (h + text_size[1]) // 2
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, text_color, 1, cv2.LINE_AA)

        return canvas

    def handle_touch(self, local_finger, is_im_pinched):
        if local_finger is None:
            self.hover_key = None
            return

        u, v, w = local_finger
        half_w = self.w_3d / 2
        half_h = self.h_3d / 2

        u_norm = (u + half_w) / self.w_3d
        v_norm = (v + half_h) / self.h_3d
        canvas_x = int(u_norm * self.canvas_w)
        canvas_y = int(v_norm * self.canvas_h)

        in_range = (-30.0 <= w <= 80.0)
        is_click = in_range and is_im_pinched
        is_hover = in_range and not is_im_pinched

        self.hover_key = None
        key_rects = self.get_key_rects()
        now = time.time()

        for (kx, ky, kw, kh, label) in key_rects:
            if kx < canvas_x < kx + kw and ky < canvas_y < ky + kh:
                if is_hover or is_click:
                    self.hover_key = label
                if is_click:
                    last_press = self.key_cooldown.get(label, 0)
                    if now - last_press > self.cooldown_time:
                        self.key_cooldown[label] = now
                        self.pressed_key = label
                        self.press_flash_time = now
                        self._process_key(label)
                break

    def _process_key(self, label):
        if label == "SPACE":
            self.typed_text += " "
        elif label == "DEL":
            self.typed_text = self.typed_text[:-1]
        elif label == "ENTER":
            self.typed_text += "\n"
        else:
            self.typed_text += label

        if len(self.typed_text) > 200:
            self.typed_text = self.typed_text[-200:]

    def update_pose(self, screen_T):
        self.target_T = np.array([0.0, 30.0, 480.0])
        self.target_rot = np.array([0.0, 0.0, 0.0])

        self.T = self.T * (1.0 - self.smooth_alpha) + self.target_T * self.smooth_alpha
        self.rot = self.rot * (1.0 - self.smooth_alpha) + self.target_rot * self.smooth_alpha

    def render_to_world(self, frame, f, cx, cy):
        R = get_rotation_matrix(self.rot[0], self.rot[1], self.rot[2])
        local_pts = np.array([
            [-self.w_3d / 2, -self.h_3d / 2, 0.0],
            [ self.w_3d / 2, -self.h_3d / 2, 0.0],
            [ self.w_3d / 2,  self.h_3d / 2, 0.0],
            [-self.w_3d / 2,  self.h_3d / 2, 0.0]
        ])

        world_pts = []
        for pt in local_pts:
            world_pt = R.dot(pt) + self.T
            world_pts.append(world_pt)

        img_corners = []
        for pt in world_pts:
            px_py = project_point(pt, f, cx, cy)
            if px_py is None:
                return frame
            img_corners.append(px_py)
        img_corners = np.array(img_corners, dtype=np.float32)

        canvas = self.draw_canvas()

        src_corners = np.array([
            [0, 0],
            [self.canvas_w, 0],
            [self.canvas_w, self.canvas_h],
            [0, self.canvas_h]
        ], dtype=np.float32)

        try:
            H, _ = cv2.findHomography(src_corners, img_corners)
            warped = cv2.warpPerspective(canvas, H, (frame.shape[1], frame.shape[0]))

            mask = np.zeros((self.canvas_h, self.canvas_w), dtype=np.uint8)
            cv2.rectangle(mask, (0, 0), (self.canvas_w, self.canvas_h), 255, -1)
            warped_mask = cv2.warpPerspective(mask, H, (frame.shape[1], frame.shape[0]))

            mask_3ch = cv2.merge([warped_mask, warped_mask, warped_mask]) / 255.0

            alpha = 0.80
            blended = cv2.addWeighted(warped, alpha, frame, 1.0 - alpha, 0)
            frame = np.where(mask_3ch > 0.01, blended, frame)

            pts = img_corners.astype(int)
            for idx in range(4):
                curr = pts[idx]
                next_pt = pts[(idx + 1) % 4]
                cv2.line(frame, tuple(curr), tuple(next_pt), CLR_MAGENTA, 1, cv2.LINE_AA)

        except Exception:
            pass

        return frame

    def get_local_finger_for_keyboard(self, index_tip_3d):
        R_kb = get_rotation_matrix(self.rot[0], self.rot[1], self.rot[2])
        local = R_kb.T.dot(index_tip_3d - self.T)
        return local


# ==========================================
# 6. 3D 虛擬螢幕類別
# ==========================================
class SciFiScreen3D:
    def __init__(self, w_3d=300, h_3d=180):
        self.w_3d = w_3d
        self.h_3d = h_3d
        
        self.T = np.array([0.0, -30.0, 450.0])  
        self.rot = np.array([0.0, 0.0, 0.0])    
        
        self.default_T = self.T.copy()
        self.default_rot = self.rot.copy()
        
        self.smooth_alpha_pos = 0.25
        self.smooth_alpha_rot = 0.20
        self.target_T = self.T.copy()
        self.target_rot = self.rot.copy()

        self.is_dragging = False
        self.drag_start_hand_pos = None
        self.drag_start_screen_pos = None
        self.drag_start_hand_rot = None
        self.drag_start_screen_rot = None

        self.canvas_w = 400
        self.canvas_h = 240
        self.current_saturation_mode = 1  
        
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

        self.keyboard_text = ""

    def save_current_as_default(self):
        self.default_T = self.T.copy()
        self.default_rot = self.rot.copy()

    def restore_default(self):
        self.target_T = self.default_T.copy()
        self.target_rot = self.default_rot.copy()

    def get_local_corners(self):
        hw = self.w_3d / 2
        hh = self.h_3d / 2
        return np.array([
            [-hw, -hh, 0.0],  
            [ hw, -hh, 0.0],  
            [ hw,  hh, 0.0],  
            [-hw,  hh, 0.0]   
        ])

    def get_world_corners(self, R):
        local_pts = self.get_local_corners()
        world_pts = []
        for pt in local_pts:
            world_pt = R.dot(pt) + self.T
            world_pts.append(world_pt)
        return np.array(world_pts)

    def draw_canvas(self, current_mode=0, keyboard_text=""):
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        
        grid_space = 20
        for x in range(0, self.canvas_w, grid_space):
            cv2.line(canvas, (x, 0), (x, self.canvas_h), (25, 25, 10), 1)
        for y in range(0, self.canvas_h, grid_space):
            cv2.line(canvas, (0, y), (self.canvas_w, y), (25, 25, 10), 1)

        cv2.rectangle(canvas, (5, 5), (self.canvas_w - 5, self.canvas_h - 5), CLR_CYAN, 1)

        cv2.putText(canvas, "X.R. PROJECTION MATRIX", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_CYAN, 1, cv2.LINE_AA)
        cv2.line(canvas, (15, 38), (200, 38), CLR_CYAN, 1)

        mode_labels = ["DEFAULT", "FACE FOLLOW", "KEYBOARD"]
        mode_colors = [CLR_GREEN, CLR_ORANGE, CLR_MAGENTA]
        mode_label = mode_labels[current_mode]
        mode_color = mode_colors[current_mode]
        cv2.putText(canvas, f"MODE: {mode_label}", (220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, mode_color, 1, cv2.LINE_AA)

        if current_mode == 2:
            cv2.putText(canvas, "KEYBOARD INPUT:", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_MAGENTA, 1, cv2.LINE_AA)
            cv2.line(canvas, (15, 68), (self.canvas_w - 15, 68), CLR_MAGENTA, 1)

            lines = keyboard_text.split("\n")
            display_lines = lines[-5:] if len(lines) > 5 else lines
            for li, line_text in enumerate(display_lines):
                y_pos = 90 + li * 22
                if y_pos > self.canvas_h - 80:
                    break
                cursor = "|" if (int(time.time() * 3) % 2 == 0 and li == len(display_lines) - 1) else ""
                display = line_text + cursor
                if len(display) > 35:
                    display = "..." + display[-32:]
                cv2.putText(canvas, display, (20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_WHITE, 1, cv2.LINE_AA)

            for i, btn in enumerate(self.buttons):
                bx, by, bw, bh = btn["rect"]
                b_color = CLR_WHITE
                if i == self.current_saturation_mode:
                    b_color = CLR_YELLOW
                    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), b_color, -1)
                    text_color = (0, 0, 0)
                else:
                    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), b_color, 2)
                    text_color = b_color
                cv2.putText(canvas, btn["label"], (bx + 15, by + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
        else:
            yaw_deg = math.degrees(self.rot[1])
            pitch_deg = math.degrees(self.rot[0])
            roll_deg = math.degrees(self.rot[2])
            cv2.putText(canvas, f"YAW  : {yaw_deg:+.1f} deg", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_WHITE, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"PITCH: {pitch_deg:+.1f} deg", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_WHITE, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"ROLL : {roll_deg:+.1f} deg", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_WHITE, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"DEPTH: {self.T[2]:.1f} mm", (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_WHITE, 1, cv2.LINE_AA)

            for i, btn in enumerate(self.buttons):
                bx, by, bw, bh = btn["rect"]
                b_color = CLR_WHITE
                
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
                
                cv2.putText(canvas, btn["label"], (bx + 15, by + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

        return canvas

    def update_pose(self):
        self.T = self.T * (1.0 - self.smooth_alpha_pos) + self.target_T * self.smooth_alpha_pos
        self.rot = self.rot * (1.0 - self.smooth_alpha_rot) + self.target_rot * self.smooth_alpha_rot

    def render_to_world(self, frame, f, cx, cy, current_mode=0, keyboard_text=""):
        R = get_rotation_matrix(self.rot[0], self.rot[1], self.rot[2])
        world_pts = self.get_world_corners(R)

        img_corners = []
        for pt in world_pts:
            px_py = project_point(pt, f, cx, cy)
            if px_py is None:
                return frame
            img_corners.append(px_py)
        img_corners = np.array(img_corners, dtype=np.float32)

        canvas = self.draw_canvas(current_mode, keyboard_text)

        src_corners = np.array([
            [0, 0],
            [self.canvas_w, 0],
            [self.canvas_w, self.canvas_h],
            [0, self.canvas_h]
        ], dtype=np.float32)

        try:
            H, _ = cv2.findHomography(src_corners, img_corners)
            warped_screen = cv2.warpPerspective(canvas, H, (frame.shape[1], frame.shape[0]))
            
            mask = np.zeros((self.canvas_h, self.canvas_w), dtype=np.uint8)
            cv2.rectangle(mask, (0, 0), (self.canvas_w, self.canvas_h), 255, -1)
            warped_mask = cv2.warpPerspective(mask, H, (frame.shape[1], frame.shape[0]))
            
            mask_3ch = cv2.merge([warped_mask, warped_mask, warped_mask]) / 255.0
            
            alpha = 0.75
            blended_region = cv2.addWeighted(warped_screen, alpha, frame, 1.0 - alpha, 0)
            frame = np.where(mask_3ch > 0.01, blended_region, frame)
            
        except Exception:
            pass

        t = 2
        c_len = 25  
        mode_border_colors = [CLR_CYAN, CLR_ORANGE, CLR_MAGENTA]
        base_color = mode_border_colors[current_mode]
        color = CLR_YELLOW if self.is_dragging else base_color

        pts = img_corners.astype(int)
        for idx in range(4):
            curr = pts[idx]
            next_pt = pts[(idx + 1) % 4]
            prev_pt = pts[(idx - 1) % 4]
            
            v_next = next_pt - curr
            v_prev = prev_pt - curr
            
            v_next_norm = v_next / (np.linalg.norm(v_next) + 1e-6)
            v_prev_norm = v_prev / (np.linalg.norm(v_prev) + 1e-6)
            
            pt1 = (curr + v_next_norm * c_len).astype(int)
            pt2 = (curr + v_prev_norm * c_len).astype(int)
            
            cv2.line(frame, tuple(curr), tuple(pt1), color, t, cv2.LINE_AA)
            cv2.line(frame, tuple(curr), tuple(pt2), color, t, cv2.LINE_AA)

        axis_len = 40
        o_3d = self.T
        ax_3d = o_3d + R.dot(np.array([axis_len, 0.0, 0.0]))
        ay_3d = o_3d + R.dot(np.array([0.0, -axis_len, 0.0])) 
        az_3d = o_3d + R.dot(np.array([0.0, 0.0, axis_len]))
        
        op = project_point(o_3d, f, cx, cy)
        ap_x = project_point(ax_3d, f, cx, cy)
        ap_y = project_point(ay_3d, f, cx, cy)
        ap_z = project_point(az_3d, f, cx, cy)
        
        if op:
            if ap_x: cv2.line(frame, op, ap_x, CLR_RED, 2, cv2.LINE_AA)   
            if ap_y: cv2.line(frame, op, ap_y, CLR_GREEN, 2, cv2.LINE_AA) 
            if ap_z: cv2.line(frame, op, ap_z, CLR_BLUE, 2, cv2.LINE_AA)  

        return frame


# ==========================================
# 7. 實體互動與手勢追蹤引擎
# ==========================================
def extract_hand_data_3d(landmarks, img_w, img_h, f, cx, cy):
    dx1 = (landmarks[17].x - landmarks[5].x) * img_w
    dy1 = (landmarks[17].y - landmarks[5].y) * img_h
    dx2 = (landmarks[9].x - landmarks[0].x) * img_w
    dy2 = (landmarks[9].y - landmarks[0].y) * img_h
    
    dist_2d = (math.sqrt(dx1*dx1 + dy1*dy1) + math.sqrt(dx2*dx2 + dy2*dy2)) / 2.0
    Z_hand = (120.0 / max(dist_2d, 1.0)) * 400.0
    
    pts_3d = []
    for lm in landmarks:
        z_phys = Z_hand + lm.z * Z_hand * 1.2
        x_phys = (lm.x * img_w - cx) * (z_phys / f)
        y_phys = (lm.y * img_h - cy) * (z_phys / f)
        pts_3d.append(np.array([x_phys, y_phys, z_phys]))
        
    pts_3d = np.array(pts_3d)
    
    v_up = pts_3d[9] - pts_3d[0]
    v_up /= np.linalg.norm(v_up) + 1e-6
    
    v_right = pts_3d[17] - pts_3d[5]
    v_right /= np.linalg.norm(v_right) + 1e-6
    
    v_forward = np.cross(v_right, v_up)
    v_forward /= np.linalg.norm(v_forward) + 1e-6
    
    v_right = np.cross(v_up, v_forward)
    v_right /= np.linalg.norm(v_right) + 1e-6
    
    R_hand = np.column_stack((v_right, v_up, v_forward))
    
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
    thumb_tip_3d = pts_3d[4]
    index_tip_3d = pts_3d[8]
    
    thumb_2d, index_2d = landmarks[4], landmarks[8]
    wrist_2d, middle_mcp_2d = landmarks[0], landmarks[9]
    
    pinch_dist_norm = math.hypot(thumb_2d.x - index_2d.x, thumb_2d.y - index_2d.y)
    palm_dist_norm = math.hypot(wrist_2d.x - middle_mcp_2d.x, wrist_2d.y - middle_mcp_2d.y)
    
    is_pinching = (pinch_dist_norm / max(palm_dist_norm, 1e-6)) < 0.25
    pinch_midpoint = (thumb_tip_3d + index_tip_3d) / 2.0

    dist_to_screen_center = np.linalg.norm(pinch_midpoint - screen.T)
    
    if is_pinching:
        if not screen.is_dragging and dist_to_screen_center < 250.0:
            screen.is_dragging = True
            screen.drag_start_hand_pos = pinch_midpoint.copy()
            screen.drag_start_screen_pos = screen.T.copy()
            screen.drag_start_hand_rot = hand_rot.copy()
            screen.drag_start_screen_rot = screen.rot.copy()
            
        if screen.is_dragging:
            delta_pos = pinch_midpoint - screen.drag_start_hand_pos
            screen.target_T = screen.drag_start_screen_pos + delta_pos
            
            delta_rot = hand_rot - screen.drag_start_hand_rot
            screen.target_rot = screen.drag_start_screen_rot + delta_rot
    else:
        screen.is_dragging = False

    for i in range(3):
        screen.btn_hover[i] = False
        screen.btn_pressed[i] = False
        
    if not screen.is_dragging and not is_pinching:
        R_screen = get_rotation_matrix(screen.rot[0], screen.rot[1], screen.rot[2])
        local_finger = R_screen.T.dot(index_tip_3d - screen.T)
        
        u, v, w = local_finger  
        
        half_w = screen.w_3d / 2
        half_h = screen.h_3d / 2
        
        if -half_w - 20 < u < half_w + 20 and -half_h - 20 < v < half_h + 20:
            idx_2d = landmarks[8]
            mid_2d = landmarks[12]
            click_pinch_dist_norm = math.hypot(idx_2d.x - mid_2d.x, idx_2d.y - mid_2d.y)
            is_im_pinched = (click_pinch_dist_norm / max(palm_dist_norm, 1e-6)) < 0.22  
            
            in_range = (-30.0 <= w <= 80.0)
            is_click = in_range and is_im_pinched
            is_hover = in_range and not is_im_pinched
            
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
                        
            return local_finger, True  

    return None, False


# ==========================================
# 8. 進度條 / 進度環繪製器
# ==========================================
def draw_mode_switch_progress(frame, progress, current_mode, next_mode):
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    mode_names = ["DEFAULT", "FACE FOLLOW", "KEYBOARD"]
    mode_colors_bgr = [CLR_GREEN, CLR_ORANGE, CLR_MAGENTA]

    next_color = mode_colors_bgr[next_mode]

    overlay = frame.copy()
    cv2.rectangle(overlay, (center_x - 160, center_y - 100),
                  (center_x + 160, center_y + 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.rectangle(frame, (center_x - 160, center_y - 100),
                  (center_x + 160, center_y + 100), next_color, 2)

    cv2.putText(frame, "MODE SWITCH", (center_x - 70, center_y - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR_WHITE, 1, cv2.LINE_AA)

    from_text = mode_names[current_mode]
    to_text = mode_names[next_mode]
    cv2.putText(frame, f"{from_text}", (center_x - 130, center_y - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_colors_bgr[current_mode], 1, cv2.LINE_AA)
    cv2.putText(frame, ">>>", (center_x - 15, center_y - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{to_text}", (center_x + 30, center_y - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, next_color, 1, cv2.LINE_AA)

    ring_radius = 30
    ring_center = (center_x, center_y + 10)
    angle_end = int(progress * 360)

    cv2.ellipse(frame, ring_center, (ring_radius, ring_radius),
                -90, 0, 360, (40, 40, 40), 3, cv2.LINE_AA)
    if angle_end > 0:
        cv2.ellipse(frame, ring_center, (ring_radius, ring_radius),
                    -90, 0, angle_end, next_color, 4, cv2.LINE_AA)

    pct_text = f"{int(progress * 100)}%"
    text_size = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(frame, pct_text,
                (ring_center[0] - text_size[0] // 2, ring_center[1] + text_size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_WHITE, 1, cv2.LINE_AA)

    bar_x1 = center_x - 130
    bar_x2 = center_x + 130
    bar_y = center_y + 60
    bar_h = 12

    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (40, 40, 40), -1)
    fill_w = int((bar_x2 - bar_x1) * progress)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + fill_w, bar_y + bar_h), next_color, -1)
    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), next_color, 1)

    remaining = max(0.0, 2.0 * (1.0 - progress))
    cv2.putText(frame, f"{remaining:.1f}s", (center_x - 15, bar_y + bar_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, next_color, 1, cv2.LINE_AA)

    return frame


def draw_cross_indicator(frame, pose_landmarks, img_w, img_h):
    if pose_landmarks is None:
        return frame

    lm = pose_landmarks.landmark
    vis_thresh = 0.3
    if (lm[13].visibility < vis_thresh or lm[14].visibility < vis_thresh or 
        lm[15].visibility < vis_thresh or lm[16].visibility < vis_thresh):
        return frame

    elbow_l = (int(lm[13].x * img_w), int(lm[13].y * img_h))
    wrist_l = (int(lm[15].x * img_w), int(lm[15].y * img_h))
    elbow_r = (int(lm[14].x * img_w), int(lm[14].y * img_h))
    wrist_r = (int(lm[16].x * img_w), int(lm[16].y * img_h))

    cv2.line(frame, elbow_l, wrist_l, CLR_ORANGE, 5, cv2.LINE_AA)
    cv2.line(frame, elbow_r, wrist_r, CLR_ORANGE, 5, cv2.LINE_AA)
    
    cv2.circle(frame, wrist_l, 8, CLR_YELLOW, -1)
    cv2.circle(frame, wrist_r, 8, CLR_YELLOW, -1)

    return frame


# ==========================================
# 9. 全局色彩與飽和度處理
# ==========================================
def apply_saturation(img, factor):
    if factor == 1.0: 
        return img
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    s_float = s.astype(np.float32) * factor
    s_final = np.clip(s_float, 0, 255).astype(np.uint8)
    
    hsv_final = cv2.merge([h, s_final, v])
    return cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)


# ==========================================
# 10. 主程式執行迴圈
# ==========================================
my_screen_3d = SciFiScreen3D(w_3d=300, h_3d=180)
global_sat_factor = 1.0

current_mode = 0
cross_detector = CrossGestureDetector(hold_duration=1.2)
face_tracker = FaceTracker()
virtual_keyboard = VirtualKeyboard()

mode_switch_cooldown = 0
MODE_SWITCH_COOLDOWN_TIME = 1.0  

mode_transition_alpha = 0.0
mode_transition_active = False
mode_transition_start = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    frame = cv2.flip(frame, 1)
    img_h, img_w, _ = frame.shape

    focal_length = img_w
    cx, cy = img_w / 2, img_h / 2

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)
    
    # 執行新版 3D Face 追蹤演算法
    face_results = face_mesh.process(rgb_frame)
    face_tracker.update(face_results, img_w, img_h, focal_length, cx, cy)

    finger_local_pos = None
    finger_active = False

    # ========== 叉叉手勢偵測 ==========
    now = time.time()
    cross_triggered = False
    cross_progress = 0.0

    if now - mode_switch_cooldown > MODE_SWITCH_COOLDOWN_TIME:
        cross_triggered, cross_progress = cross_detector.detect(
            pose_results.pose_landmarks if pose_results.pose_landmarks else None,
            img_w, img_h
        )
    else:
        cross_detector._reset()

    if cross_triggered:
        old_mode = current_mode
        current_mode = (current_mode + 1) % 3
        mode_switch_cooldown = now
        mode_transition_active = True
        mode_transition_start = now

        print(f"➔ 模式切換: {['預設', '臉部跟隨', '鍵盤'][old_mode]} → {['預設', '臉部跟隨', '鍵盤'][current_mode]}")

        if current_mode == 0:
            my_screen_3d.restore_default()
            my_screen_3d.is_dragging = False
        elif current_mode == 1:
            my_screen_3d.save_current_as_default()
            my_screen_3d.target_rot = np.array([0.0, 0.0, 0.0])
            my_screen_3d.is_dragging = False
        elif current_mode == 2:
            my_screen_3d.is_dragging = False
            virtual_keyboard.typed_text = ""  

    if mode_transition_active:
        elapsed = now - mode_transition_start
        if elapsed < 0.5:
            mode_transition_alpha = 1.0 - (elapsed / 0.5)
        else:
            mode_transition_active = False
            mode_transition_alpha = 0.0

    # ========== 根據模式處理互動 ==========

    if results.multi_hand_landmarks:
        if cross_progress < 0.1:
            landmarks = results.multi_hand_landmarks[0].landmark

            pts_3d, hand_rot = extract_hand_data_3d(landmarks, img_w, img_h, focal_length, cx, cy)

            if current_mode == 0:
                finger_local_pos, finger_active = handle_interaction(my_screen_3d, landmarks, pts_3d, hand_rot)

            elif current_mode == 1:
                finger_local_pos, finger_active = handle_interaction(my_screen_3d, landmarks, pts_3d, hand_rot)

            elif current_mode == 2:
                index_tip_3d = pts_3d[8]
                kb_local = virtual_keyboard.get_local_finger_for_keyboard(index_tip_3d)

                idx_2d = landmarks[8]
                mid_2d = landmarks[12]
                wrist_2d = landmarks[0]
                middle_mcp_2d = landmarks[9]
                palm_dist_norm = math.hypot(wrist_2d.x - middle_mcp_2d.x, wrist_2d.y - middle_mcp_2d.y)
                click_pinch_dist_norm = math.hypot(idx_2d.x - mid_2d.x, idx_2d.y - mid_2d.y)
                is_im_pinched = (click_pinch_dist_norm / max(palm_dist_norm, 1e-6)) < 0.22

                virtual_keyboard.handle_touch(kb_local, is_im_pinched)

                my_screen_3d.is_dragging = False
                finger_active = True
                finger_local_pos = kb_local

            global_sat_factor = my_screen_3d.buttons[my_screen_3d.current_saturation_mode]["sat"]

            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=CLR_WHITE, thickness=1, circle_radius=1),
                    mp_draw.DrawingSpec(color=CLR_CYAN, thickness=1, circle_radius=1)
                )

            if finger_active and finger_local_pos is not None:
                idx_2d = project_point(pts_3d[8], focal_length, cx, cy)
                
                if current_mode == 2:
                    R_kb = get_rotation_matrix(virtual_keyboard.rot[0], virtual_keyboard.rot[1], virtual_keyboard.rot[2])
                    laser_hit_world = virtual_keyboard.T + R_kb.dot(np.array([finger_local_pos[0], finger_local_pos[1], 0.0]))
                else:
                    R_screen = get_rotation_matrix(my_screen_3d.rot[0], my_screen_3d.rot[1], my_screen_3d.rot[2])
                    laser_hit_world = my_screen_3d.T + R_screen.dot(np.array([finger_local_pos[0], finger_local_pos[1], 0.0]))
                
                laser_hit_2d = project_point(laser_hit_world, focal_length, cx, cy)

                if idx_2d and laser_hit_2d:
                    cv2.line(frame, idx_2d, laser_hit_2d, CLR_CYAN, 1, cv2.LINE_AA)
                    cv2.circle(frame, laser_hit_2d, 5, CLR_YELLOW, -1)
                    cv2.circle(frame, laser_hit_2d, 10, CLR_CYAN, 1, cv2.LINE_AA)
    else:
        my_screen_3d.is_dragging = False

    # ========== 模式特殊邏輯更新 ==========

    if current_mode == 1:
        target_pos, target_rot = face_tracker.get_screen_target_pos_and_rot(focal_length, cx, cy)
        if target_pos is not None:
            my_screen_3d.target_T = target_pos
            if not my_screen_3d.is_dragging:
                my_screen_3d.target_rot = target_rot

    if current_mode == 2:
        virtual_keyboard.update_pose(my_screen_3d.T)

    my_screen_3d.update_pose()
    frame = apply_saturation(frame, global_sat_factor)

    if current_mode == 2:
        frame = virtual_keyboard.render_to_world(frame, focal_length, cx, cy)
    else:
        frame = my_screen_3d.render_to_world(frame, focal_length, cx, cy, current_mode, "")

    # ========== HUD 狀態顯示 ==========
    mode_names_display = ['DEFAULT', 'FACE FOLLOW', 'KEYBOARD']
    mode_colors_hud = [CLR_GREEN, CLR_ORANGE, CLR_MAGENTA]

    # 【核心修正】：已將 CL_YELLOW 修正為 CLR_YELLOW
    cv2.putText(frame, f"VIDEO SAT: {global_sat_factor:.1f}x", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_YELLOW, 2, cv2.LINE_AA)

    cv2.putText(frame, f"MODE: {mode_names_display[current_mode]}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_colors_hud[current_mode], 1, cv2.LINE_AA)

    if my_screen_3d.is_dragging:
        cv2.putText(frame, "STATUS: PANEL DRAGGING", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_RED, 1, cv2.LINE_AA)
    else:
        status_msg = "STATUS: SYSTEM READY"
        if current_mode == 1:
            if face_tracker.detected:
                status_msg = "STATUS: FACE TRACKING ACTIVE"
            else:
                status_msg = "STATUS: SEARCHING FACE..."
        elif current_mode == 2:
            status_msg = "STATUS: KEYBOARD ACTIVE"
        cv2.putText(frame, status_msg, (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_GREEN, 1, cv2.LINE_AA)

    cv2.putText(frame, "X CROSS HANDS 2s -> SWITCH MODE", (20, img_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

    # ========== 叉叉進度條 ==========
    if cross_progress > 0.01:
        next_mode = (current_mode + 1) % 3
        frame = draw_mode_switch_progress(frame, cross_progress, current_mode, next_mode)
        frame = draw_cross_indicator(frame, pose_results.pose_landmarks if pose_results.pose_landmarks else None, img_w, img_h)

    # ========== 模式切換過渡閃光 ==========
    if mode_transition_active and mode_transition_alpha > 0:
        flash_color = mode_colors_hud[current_mode]
        flash_overlay = np.full_like(frame, flash_color, dtype=np.uint8)
        alpha_val = mode_transition_alpha * 0.3
        frame = cv2.addWeighted(flash_overlay, alpha_val, frame, 1.0 - alpha_val, 0)

    cv2.imshow('Future Tech 3D XR HUD', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()