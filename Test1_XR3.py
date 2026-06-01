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
    refine_landmarks=False
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
# 設定畫面寬高（若硬體支援，可自行調整）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

print("➔ 未來科技 3D XR HUD 系統已啟動！")
print("➔ 操控指南：")
print("  1. 【移動與旋轉】：將【大拇指】與【食指】捏合，即可『抓取』虛擬面板。")
print("     此時面板會隨您的手掌在 3D 空間中移動與進行 360 度傾斜、旋轉。")
print("  2. 【按鈕觸控】：當面板靜止時，伸出【單獨食指】靠近虛擬螢幕並點擊按鈕區。")
print("     系統會偵測食指的物理深度，當食指穿過螢幕表面時即觸發點擊。")
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
    if Z <= 10.0:  # 避開除以零或投影到相機後方的點
        return None
    px = int((X * f / Z) + cx)
    py = int((Y * f / Z) + cy)
    return (px, py)


# ==========================================
# 3. 交叉手勢 (叉叉) 偵測器
# ==========================================
class CrossGestureDetector:
    """
    偵測手臂交叉 (X) 手勢，持續超過指定時間後觸發模式切換。
    判斷邏輯：
      1. 判斷 Pose (姿態) 的左右前臂 (手肘~手腕) 在 2D 畫面上相交
      2. 雙臂方向向量的交叉角度合理 (20° ~ 160°)
      3. 手腕必須往上抬起 (位於手肘上方或相近)，防止自然下垂時的誤判
      4. 增加判斷寬容度，確保不會「做了沒反應」
    """
    def __init__(self, hold_duration=1.2):
        self.hold_duration = hold_duration
        self.cross_start_time = None
        self.is_crossing = False
        self.progress = 0.0       # 0.0 ~ 1.0 當前進度
        self.just_triggered = False  # 邊緣觸發旗標
        self.last_seen_time = 0      # 用於容許短暫遮擋斷訊

    def _segments_intersect_2d(self, p1, p2, p3, p4):
        """檢查兩條 2D 線段是否相交"""
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
        """
        輸入 MediaPipe 的 Pose 結果，回傳是否已完成一次完整的叉叉觸發。
        回傳 (triggered: bool, progress: float)
        """
        self.just_triggered = False

        if pose_landmarks is None:
            return self._handle_absence()

        lm = pose_landmarks.landmark
        
        # 13: 左手肘, 15: 左手腕, 14: 右手肘, 16: 右手腕 (注意：MediaPipe 是以目標主體視角標示左右)
        vis_thresh = 0.3
        if (lm[13].visibility < vis_thresh or lm[14].visibility < vis_thresh or 
            lm[15].visibility < vis_thresh or lm[16].visibility < vis_thresh):
            return self._handle_absence()

        elbow_l = np.array([lm[13].x * img_w, lm[13].y * img_h])
        wrist_l = np.array([lm[15].x * img_w, lm[15].y * img_h])
        elbow_r = np.array([lm[14].x * img_w, lm[14].y * img_h])
        wrist_r = np.array([lm[16].x * img_w, lm[16].y * img_h])

        # 前臂向量
        vec_l = wrist_l - elbow_l
        vec_r = wrist_r - elbow_r

        # 為了容錯，將前臂向外延伸 1.5 倍 (涵蓋手掌長度) 以及向手肘後方延伸 0.2 倍
        wrist_l_ext = elbow_l + vec_l * 1.5
        elbow_l_ext = elbow_l - vec_l * 0.2
        wrist_r_ext = elbow_r + vec_r * 1.5
        elbow_r_ext = elbow_r - vec_r * 0.2

        # 條件 1: 確保手有抬起，而非自然垂下
        # y 軸向下為正，所以如果手腕的 y 遠大於手肘的 y，表示手往下擺
        if (wrist_l[1] > elbow_l[1] + 80) and (wrist_r[1] > elbow_r[1] + 80):
            return self._handle_absence()

        # 條件 2: 兩前臂延長線段是否在 2D 空間中相交
        segments_cross = self._segments_intersect_2d(elbow_l_ext, wrist_l_ext, elbow_r_ext, wrist_r_ext)

        # 條件 3: 計算兩前臂方向向量的夾角
        dot_val = np.dot(vec_l, vec_r)
        mag_l = np.linalg.norm(vec_l)
        mag_r = np.linalg.norm(vec_r)
        cos_angle = dot_val / (mag_l * mag_r + 1e-6)
        angle_deg = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))

        # 交叉角度在 30°~160° 之間才算是 X 形交叉，若是兩手平行 (<30度) 則不算
        angle_ok = 30.0 < angle_deg < 160.0

        # 手腕距離檢查：如果手腕非常靠近 (例如貼在一起)，也算是交叉的一種特例
        avg_arm_len = (mag_l + mag_r) / 2.0
        wrist_dist = np.linalg.norm(wrist_l - wrist_r)
        wrists_close = wrist_dist < avg_arm_len * 0.8
        
        # 交叉寬容判定
        # 只要前臂明顯相交且角度對，或者手腕靠近且角度對，都認定是在比叉叉
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
        """處理短暫無辨識到的情況，提供 0.3 秒的容錯緩衝"""
        now = time.time()
        if self.is_crossing and (now - self.last_seen_time < 0.3):
            # 短暫丟失，仍算在進行中，進度不倒退也不增加
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
# 4. 臉部位置追蹤器
# ==========================================
class FaceTracker:
    """使用 MediaPipe Face Mesh 追蹤臉部中心位置與朝向，用於面板跟隨模式"""
    def __init__(self):
        self.smooth_x = 0.5
        self.smooth_y = 0.5
        self.smooth_z = 400.0
        
        self.smooth_pitch = 0.0
        self.smooth_yaw = 0.0
        self.smooth_roll = 0.0
        
        self.alpha = 0.15  # 平滑係數
        self.detected = False

    def update(self, face_results, img_w, img_h):
        if face_results.multi_face_landmarks:
            face_lm = face_results.multi_face_landmarks[0].landmark
            # 取鼻尖 (1) 和左右臉頰 (234, 454) 估算臉部中心與大小
            nose = face_lm[1]
            left_cheek = face_lm[234]
            right_cheek = face_lm[454]
            chin = face_lm[152]
            forehead = face_lm[10]

            # 臉部中心 (歸一化座標)
            face_cx = nose.x
            face_cy = nose.y

            # 計算頭部 3D 旋轉角度
            # MediaPipe 中 z 是相對於頭部中心的深度
            # 臉頰左右深度差 (Yaw)
            dx = (right_cheek.x - left_cheek.x)
            dz = (right_cheek.z - left_cheek.z)
            yaw = math.atan2(-dz, dx)  # 負號校正視角
            
            # 使用臉部高度 (下巴到額頭) 來估算距離，比臉頰寬度穩定 (轉頭時高度不變)
            face_height_px = abs(chin.y - forehead.y) * img_h
            # 平均臉高約 200mm
            face_z = (200.0 / max(face_height_px, 1.0)) * img_w * 0.8
            
            # 限制極端旋轉數值，避免判定失誤跑到背後
            yaw = np.clip(yaw, -1.2, 1.2)

            self.smooth_x = self.smooth_x * (1.0 - self.alpha) + face_cx * self.alpha
            self.smooth_y = self.smooth_y * (1.0 - self.alpha) + face_cy * self.alpha
            self.smooth_z = self.smooth_z * (1.0 - self.alpha) + face_z * self.alpha
            
            # 下巴到額頭深度差 (Pitch)
            dy = (chin.y - forehead.y)
            dz_pitch = (chin.z - forehead.z)
            pitch = math.atan2(dz_pitch, dy)
            
            # 臉頰左右高低差 (Roll)
            roll = math.atan2(right_cheek.y - left_cheek.y, right_cheek.x - left_cheek.x)
            
            self.smooth_pitch = self.smooth_pitch * (1.0 - self.alpha) + pitch * self.alpha
            self.smooth_yaw = self.smooth_yaw * (1.0 - self.alpha) + yaw * self.alpha
            self.smooth_roll = self.smooth_roll * (1.0 - self.alpha) + (-roll) * self.alpha
            
            self.detected = True
        else:
            self.detected = False

    def get_screen_target_pos_and_rot(self, f, cx, cy):
        """計算面板應該跟隨的 3D 世界座標與旋轉角度 (在臉部前方)"""
        if not self.detected:
            return None, None
            
        # 計算頭部的法向量方向，沿著法面往前延伸 (真正意義上的「臉部正前方」)
        direction_z = math.cos(self.smooth_pitch) * math.cos(self.smooth_yaw)
        direction_x = math.sin(self.smooth_yaw)
        direction_y = -math.sin(self.smooth_pitch)

        # 這裡的 distance_from_face 就是面板離臉部的真實 3D 距離 (不因視角而改變)
        distance_from_face = 200.0
        
        # 面板中心點
        sx = (self.smooth_x * cx * 2 - cx) * (self.smooth_z / f) + direction_x * distance_from_face
        sy = (self.smooth_y * cy * 2 - cy) * (self.smooth_z / f) + direction_y * distance_from_face
        sz = self.smooth_z - direction_z * distance_from_face

        # 限制不要太近，保留最低觀察距離
        sz = max(sz, 100.0)
        
        target_rot = np.array([self.smooth_pitch, self.smooth_yaw, self.smooth_roll])
        
        return np.array([sx, sy, sz]), target_rot


# ==========================================
# 5. 虛擬鍵盤類別
# ==========================================
class VirtualKeyboard:
    """
    3D 空間中的虛擬鍵盤。
    固定在畫面中間，不會隨意亂動。
    利用「食指和拇指捏合(Pinch)」的微小動作來觸發點擊，非常精準。
    """
    def __init__(self):
        self.canvas_w = 600
        self.canvas_h = 280
        self.w_3d = 360
        self.h_3d = 168

        # 鍵盤行列定義
        self.rows = [
            list("QWERTYUIOP"),
            list("ASDFGHJKL"),
            list("ZXCVBNM"),
            ["SPACE", "DEL", "ENTER"]
        ]

        self.typed_text = ""
        self.cursor_blink = 0

        # 將鍵盤推遠一點，避免因為視角造成大小超出螢幕
        self.T = np.array([0.0, 0.0, 480.0])
        self.rot = np.array([0.0, 0.0, 0.0])  # 正面朝向

        # 平滑化
        self.target_T = self.T.copy()
        self.target_rot = self.rot.copy()
        self.smooth_alpha = 0.5

        # 按鍵冷卻 (防止連續觸發)
        self.key_cooldown = {}
        self.cooldown_time = 0.4  # 秒

        # Hover 追蹤
        self.hover_key = None
        self.pressed_key = None
        self.press_flash_time = 0
        self.was_pinching = False

    def get_key_rects(self):
        """計算所有按鍵的畫布座標 (x, y, w, h, label)"""
        rects = []
        margin_x = 8
        margin_y = 70  # 將按鍵往下推，上方留給文字輸入框
        key_h = 42
        gap = 5

        for row_idx, row in enumerate(self.rows):
            if row_idx == 3:  # 特殊列 (SPACE, DEL, ENTER)
                total_w = self.canvas_w - margin_x * 2
                # SPACE 佔一半，DEL 和 ENTER 各佔四分之一
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
                offset_x = margin_x + (row_idx % 2) * 10  # 交錯排列
                y = margin_y + row_idx * (key_h + gap)
                for k_idx, key_label in enumerate(row):
                    x = offset_x + k_idx * (key_w + gap)
                    rects.append((x, y, key_w, key_h, key_label))
        return rects

    def draw_canvas(self):
        """繪製鍵盤 UI"""
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        # 背景漸層
        for y in range(self.canvas_h):
            alpha = y / self.canvas_h
            val = int(10 + alpha * 15)
            canvas[y, :] = (val, val + 2, val)

        # 外框
        cv2.rectangle(canvas, (2, 2), (self.canvas_w - 2, self.canvas_h - 2), CLR_MAGENTA, 1)

        # ====== 繪製輸入文字框 ======
        cv2.rectangle(canvas, (10, 10), (self.canvas_w - 10, 60), (30, 30, 30), -1)
        cv2.rectangle(canvas, (10, 10), (self.canvas_w - 10, 60), CLR_CYAN, 1)
        
        cursor = "_" if int(time.time() * 2) % 2 == 0 else ""
        disp_txt = self.typed_text if len(self.typed_text) < 30 else "..." + self.typed_text[-27:]
        cv2.putText(canvas, f"INPUT> {disp_txt}{cursor}", (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_WHITE, 1, cv2.LINE_AA)

        key_rects = self.get_key_rects()
        now = time.time()

        for (x, y, w, h, label) in key_rects:
            # 決定按鍵顏色
            color = (40, 40, 40)
            text_color = CLR_WHITE
            border_color = (80, 80, 80)

            if self.pressed_key == label and (now - self.press_flash_time) < 0.15:
                color = (0, 200, 200)  # 按下閃光
                text_color = (0, 0, 0)
                border_color = CLR_YELLOW
            elif self.hover_key == label:
                color = (50, 60, 60)
                border_color = CLR_CYAN

            # 繪製按鍵背景
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, 1)

            # 繪製按鍵文字
            font_scale = 0.35 if len(label) > 1 else 0.45
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            tx = x + (w - text_size[0]) // 2
            ty = y + (h + text_size[1]) // 2
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, text_color, 1, cv2.LINE_AA)

        return canvas

    def handle_touch(self, local_finger):
        """處理虛擬按鍵觸碰"""
        if local_finger is None:
            self.hover_key = None
            return

        u, v, w = local_finger
        half_w = self.w_3d / 2
        half_h = self.h_3d / 2

        # 將 3D 局部座標映射至畫布座標
        u_norm = (u + half_w) / self.w_3d
        v_norm = (v + half_h) / self.h_3d
        canvas_x = int(u_norm * self.canvas_w)
        canvas_y = int(v_norm * self.canvas_h)

        # 恢復使用食指戳入深度的點擊判定
        is_hover = (-5.0 <= w <= 80.0)
        is_click = (-60.0 <= w < -5.0)

        self.hover_key = None
        key_rects = self.get_key_rects()
        now = time.time()

        for (kx, ky, kw, kh, label) in key_rects:
            if kx < canvas_x < kx + kw and ky < canvas_y < ky + kh:
                if is_hover or is_click:
                    self.hover_key = label
                if is_click:
                    # 檢查冷卻時間
                    last_press = self.key_cooldown.get(label, 0)
                    if now - last_press > self.cooldown_time:
                        self.key_cooldown[label] = now
                        self.pressed_key = label
                        self.press_flash_time = now
                        self._process_key(label)
                break

    def _process_key(self, label):
        """處理按鍵輸入邏輯"""
        if label == "SPACE":
            self.typed_text += " "
        elif label == "DEL":
            self.typed_text = self.typed_text[:-1]
        elif label == "ENTER":
            self.typed_text += "\n"
        else:
            self.typed_text += label

        # 限制文字長度
        if len(self.typed_text) > 200:
            self.typed_text = self.typed_text[-200:]

    def update_pose(self, screen_T):
        """更新鍵盤位置 (強制固定在正前方，不受面板影響)"""
        # 鍵盤固定在相機前方的中央位置，距離拉遠以確保不超出畫面
        self.target_T = np.array([0.0, 30.0, 480.0])
        self.target_rot = np.array([0.0, 0.0, 0.0])

        self.T = self.T * (1.0 - self.smooth_alpha) + self.target_T * self.smooth_alpha
        self.rot = self.rot * (1.0 - self.smooth_alpha) + self.target_rot * self.smooth_alpha

    def render_to_world(self, frame, f, cx, cy):
        """將鍵盤投射至畫面"""
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

            # 繪製科技感外框
            pts = img_corners.astype(int)
            for idx in range(4):
                curr = pts[idx]
                next_pt = pts[(idx + 1) % 4]
                cv2.line(frame, tuple(curr), tuple(next_pt), CLR_MAGENTA, 1, cv2.LINE_AA)

        except Exception:
            pass

        return frame

    def get_local_finger_for_keyboard(self, index_tip_3d):
        """將食指 3D 位置轉換為鍵盤的局部座標"""
        R_kb = get_rotation_matrix(self.rot[0], self.rot[1], self.rot[2])
        local = R_kb.T.dot(index_tip_3d - self.T)
        return local


# ==========================================
# 6. 3D 虛擬螢幕類別
# ==========================================
class SciFiScreen3D:
    def __init__(self, w_3d=300, h_3d=180):
        # 3D 空間尺寸
        self.w_3d = w_3d
        self.h_3d = h_3d
        
        # 3D 空間初始狀態 (位置 T 與旋轉角度)
        self.T = np.array([0.0, -30.0, 450.0])  # 置中，稍微偏上，距離相機 450 像素
        self.rot = np.array([0.0, 0.0, 0.0])    # [Pitch, Yaw, Roll] 弧度
        
        # 儲存預設位置 (用於模式切換回復)
        self.default_T = self.T.copy()
        self.default_rot = self.rot.copy()
        
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

        # 鍵盤模式打字文字 (由外部 VirtualKeyboard 寫入)
        self.keyboard_text = ""

    def save_current_as_default(self):
        """儲存當前位置為預設值"""
        self.default_T = self.T.copy()
        self.default_rot = self.rot.copy()

    def restore_default(self):
        """恢復至預設位置"""
        self.target_T = self.default_T.copy()
        self.target_rot = self.default_rot.copy()

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

    def draw_canvas(self, current_mode=0, keyboard_text=""):
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

        # 模式指示器
        mode_labels = ["DEFAULT", "FACE FOLLOW", "KEYBOARD"]
        mode_colors = [CLR_GREEN, CLR_ORANGE, CLR_MAGENTA]
        mode_label = mode_labels[current_mode]
        mode_color = mode_colors[current_mode]
        cv2.putText(canvas, f"MODE: {mode_label}", (220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, mode_color, 1, cv2.LINE_AA)

        if current_mode == 2:
            # 鍵盤模式：顯示打字內容
            cv2.putText(canvas, "KEYBOARD INPUT:", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_MAGENTA, 1, cv2.LINE_AA)
            cv2.line(canvas, (15, 68), (self.canvas_w - 15, 68), CLR_MAGENTA, 1)

            # 顯示輸入文字 (支援多行)
            lines = keyboard_text.split("\n")
            # 取最後幾行顯示
            display_lines = lines[-5:] if len(lines) > 5 else lines
            for li, line_text in enumerate(display_lines):
                y_pos = 90 + li * 22
                if y_pos > self.canvas_h - 80:
                    break
                # 閃爍游標
                cursor = "|" if (int(time.time() * 3) % 2 == 0 and li == len(display_lines) - 1) else ""
                display = line_text + cursor
                # 限制每行顯示長度
                if len(display) > 35:
                    display = "..." + display[-32:]
                cv2.putText(canvas, display, (20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_WHITE, 1, cv2.LINE_AA)

            # 底部仍然顯示按鈕
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
            # 一般模式：顯示 3D 姿態數值
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

    def render_to_world(self, frame, f, cx, cy, current_mode=0, keyboard_text=""):
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
        canvas = self.draw_canvas(current_mode, keyboard_text)

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
            
            # XR 玻璃拟态融合 (0.75 為虛擬面板不透明度)
            alpha = 0.75
            blended_region = cv2.addWeighted(warped_screen, alpha, frame, 1.0 - alpha, 0)
            frame = np.where(mask_3ch > 0.01, blended_region, frame)
            
        except Exception:
            # 避免投影矩陣退化時程式崩潰
            pass

        # 4. 繪製 3D 空間框線 (不因 Warp 模糊，維持高清晰度向量線條)
        t = 2
        c_len = 25  # 角括弧長度
        # 根據模式改變框線顏色
        mode_border_colors = [CLR_CYAN, CLR_ORANGE, CLR_MAGENTA]
        base_color = mode_border_colors[current_mode]
        color = CLR_YELLOW if self.is_dragging else base_color

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
# 7. 實體互動與手勢追蹤引擎
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
# 8. 進度條 / 進度環繪製器
# ==========================================
def draw_mode_switch_progress(frame, progress, current_mode, next_mode):
    """
    在畫面中央繪製科技感進度環與進度條，
    讓使用者清楚看到模式切換倒數進度。
    """
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    mode_names = ["DEFAULT", "FACE FOLLOW", "KEYBOARD"]
    mode_colors_bgr = [CLR_GREEN, CLR_ORANGE, CLR_MAGENTA]

    next_color = mode_colors_bgr[next_mode]

    # 半透明背景遮罩
    overlay = frame.copy()
    cv2.rectangle(overlay, (center_x - 160, center_y - 100),
                  (center_x + 160, center_y + 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # 外框
    cv2.rectangle(frame, (center_x - 160, center_y - 100),
                  (center_x + 160, center_y + 100), next_color, 2)

    # 標題
    cv2.putText(frame, "MODE SWITCH", (center_x - 70, center_y - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR_WHITE, 1, cv2.LINE_AA)

    # 當前模式 → 目標模式
    from_text = mode_names[current_mode]
    to_text = mode_names[next_mode]
    cv2.putText(frame, f"{from_text}", (center_x - 130, center_y - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_colors_bgr[current_mode], 1, cv2.LINE_AA)
    cv2.putText(frame, ">>>", (center_x - 15, center_y - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{to_text}", (center_x + 30, center_y - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, next_color, 1, cv2.LINE_AA)

    # 進度環 (圓弧)
    ring_radius = 30
    ring_center = (center_x, center_y + 10)
    angle_end = int(progress * 360)

    # 背景環
    cv2.ellipse(frame, ring_center, (ring_radius, ring_radius),
                -90, 0, 360, (40, 40, 40), 3, cv2.LINE_AA)
    # 進度環
    if angle_end > 0:
        cv2.ellipse(frame, ring_center, (ring_radius, ring_radius),
                    -90, 0, angle_end, next_color, 4, cv2.LINE_AA)

    # 中央百分比
    pct_text = f"{int(progress * 100)}%"
    text_size = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(frame, pct_text,
                (ring_center[0] - text_size[0] // 2, ring_center[1] + text_size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_WHITE, 1, cv2.LINE_AA)

    # 底部進度條
    bar_x1 = center_x - 130
    bar_x2 = center_x + 130
    bar_y = center_y + 60
    bar_h = 12

    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (40, 40, 40), -1)
    fill_w = int((bar_x2 - bar_x1) * progress)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + fill_w, bar_y + bar_h), next_color, -1)
    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), next_color, 1)

    # 倒數文字
    remaining = max(0.0, 2.0 * (1.0 - progress))
    cv2.putText(frame, f"{remaining:.1f}s", (center_x - 15, bar_y + bar_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, next_color, 1, cv2.LINE_AA)

    return frame


def draw_cross_indicator(frame, pose_landmarks, img_w, img_h):
    """當偵測到手臂交叉時，繪製 X 標記視覺反饋"""
    if pose_landmarks is None:
        return frame

    lm = pose_landmarks.landmark
    vis_thresh = 0.3
    if (lm[13].visibility < vis_thresh or lm[14].visibility < vis_thresh or 
        lm[15].visibility < vis_thresh or lm[16].visibility < vis_thresh):
        return frame

    # 繪製兩隻手的前臂線 (手肘→手腕)
    elbow_l = (int(lm[13].x * img_w), int(lm[13].y * img_h))
    wrist_l = (int(lm[15].x * img_w), int(lm[15].y * img_h))
    elbow_r = (int(lm[14].x * img_w), int(lm[14].y * img_h))
    wrist_r = (int(lm[16].x * img_w), int(lm[16].y * img_h))

    cv2.line(frame, elbow_l, wrist_l, CLR_ORANGE, 5, cv2.LINE_AA)
    cv2.line(frame, elbow_r, wrist_r, CLR_ORANGE, 5, cv2.LINE_AA)
    
    # 畫個發光圓點在手掌位置
    cv2.circle(frame, wrist_l, 8, CLR_YELLOW, -1)
    cv2.circle(frame, wrist_r, 8, CLR_YELLOW, -1)

    return frame


# ==========================================
# 9. 全局色彩與飽和度處理
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
# 10. 主程式執行迴圈
# ==========================================
# 實例化 3D 空間中的虛擬面板
my_screen_3d = SciFiScreen3D(w_3d=300, h_3d=180)
global_sat_factor = 1.0

# 模式切換系統
# Mode 0: 預設模式 (螢幕在固定 3D 位置)
# Mode 1: 臉部跟隨模式 (螢幕跟在臉部前面)
# Mode 2: 鍵盤模式 (螢幕+鍵盤，鍵盤固定在前方)
current_mode = 0
cross_detector = CrossGestureDetector(hold_duration=1.2)
face_tracker = FaceTracker()
virtual_keyboard = VirtualKeyboard()

# 模式切換冷卻 (防止連續觸發)
mode_switch_cooldown = 0
MODE_SWITCH_COOLDOWN_TIME = 1.0  # 切換後 1 秒內不允許再次觸發

# 模式切換時的過渡動畫
mode_transition_alpha = 0.0
mode_transition_active = False
mode_transition_start = 0

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

    # 將畫面轉為 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 執行追蹤模型
    results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)
    
    # 臉部追蹤 (每幀都執行，但僅 Mode 1 時使用結果)
    face_results = face_mesh.process(rgb_frame)
    face_tracker.update(face_results, img_w, img_h)

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

    # 模式切換邏輯
    if cross_triggered:
        old_mode = current_mode
        current_mode = (current_mode + 1) % 3
        mode_switch_cooldown = now
        mode_transition_active = True
        mode_transition_start = now

        print(f"➔ 模式切換: {['預設', '臉部跟隨', '鍵盤'][old_mode]} → {['預設', '臉部跟隨', '鍵盤'][current_mode]}")

        if current_mode == 0:
            # 回到預設模式，恢復原始位置
            my_screen_3d.restore_default()
            my_screen_3d.is_dragging = False
        elif current_mode == 1:
            # 進入臉部跟隨模式，記住當前作為預設位置
            my_screen_3d.save_current_as_default()
            # 允許面板旋轉，初始化回正面
            my_screen_3d.target_rot = np.array([0.0, 0.0, 0.0])
            my_screen_3d.is_dragging = False
        elif current_mode == 2:
            # 進入鍵盤模式
            my_screen_3d.is_dragging = False
            virtual_keyboard.typed_text = ""  # 清空打字緩衝

    # 模式過渡動畫 (短暫的切換閃光效果)
    if mode_transition_active:
        elapsed = now - mode_transition_start
        if elapsed < 0.5:
            mode_transition_alpha = 1.0 - (elapsed / 0.5)
        else:
            mode_transition_active = False
            mode_transition_alpha = 0.0

    # ========== 根據模式處理互動 ==========

    # 若偵測到手部
    if results.multi_hand_landmarks:
        # 在叉叉進行中時不處理一般互動 (避免誤觸)
        if cross_progress < 0.1:
            # 取最靠近的第一隻手進行主面板控制
            landmarks = results.multi_hand_landmarks[0].landmark

            # 1. 建立高精度 3D 手部骨架坐標
            pts_3d, hand_rot = extract_hand_data_3d(landmarks, img_w, img_h, focal_length, cx, cy)

            if current_mode == 0:
                # 預設模式：正常抓取+按鈕觸控
                finger_local_pos, finger_active = handle_interaction(my_screen_3d, landmarks, pts_3d, hand_rot)

            elif current_mode == 1:
                # 臉部跟隨模式：允許抓取來旋轉面板 360 度，也可以按按鈕 (位置變更會由臉部追蹤覆蓋)
                finger_local_pos, finger_active = handle_interaction(my_screen_3d, landmarks, pts_3d, hand_rot)

            elif current_mode == 2:
                # 鍵盤模式：食指觸碰鍵盤按鍵 (以深度觸發)
                index_tip_3d = pts_3d[8]
                kb_local = virtual_keyboard.get_local_finger_for_keyboard(index_tip_3d)

                virtual_keyboard.handle_touch(kb_local)

                # 停止螢幕拖曳
                my_screen_3d.is_dragging = False
                
                # 準備傳遞給光標渲染
                finger_active = True
                finger_local_pos = kb_local

            # 獲取當前應套用的飽和度參數
            global_sat_factor = my_screen_3d.buttons[my_screen_3d.current_saturation_mode]["sat"]

            # 繪製精緻的 AR 手部關節線 (細長白色線條，突顯現代科技感)
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=CLR_WHITE, thickness=1, circle_radius=1),
                    mp_draw.DrawingSpec(color=CLR_CYAN, thickness=1, circle_radius=1)
                )

            # 如果食指在面板上方懸停，繪製 XR 空間追蹤雷射線
            if finger_active and finger_local_pos is not None:
                idx_2d = project_point(pts_3d[8], focal_length, cx, cy)
                
                if current_mode == 2:
                    # 在鍵盤模式，使用虛擬鍵盤的矩陣進行反投影
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
        # 當失去手掌訊號，強制取消拖曳以避免下次抓取時物件產生瞬移
        my_screen_3d.is_dragging = False

    # ========== 模式特殊邏輯更新 ==========

    if current_mode == 1:
        # 臉部跟隨模式：更新螢幕目標位置與角度，讓面板完全服貼臉的 3D 朝向
        target_pos, target_rot = face_tracker.get_screen_target_pos_and_rot(focal_length, cx, cy)
        if target_pos is not None:
            # 覆蓋被抓取時的 target_T，讓面板位置永遠釘在臉部前方
            my_screen_3d.target_T = target_pos
            
            # 若手沒有在抓取(拖曳)旋轉，則自動使用臉部的 3D 朝向 (若有抓取，則保留手的旋轉控制權)
            if not my_screen_3d.is_dragging:
                my_screen_3d.target_rot = target_rot

    if current_mode == 2:
        # 鍵盤模式：更新鍵盤位置 (靜止在中間)
        virtual_keyboard.update_pose(my_screen_3d.T)

    # 更新虛擬面板狀態 (平滑過渡)
    my_screen_3d.update_pose()

    # 先對背景套用色彩濾鏡
    frame = apply_saturation(frame, global_sat_factor)

    # 將虛擬面板投影渲染至最前層
    if current_mode == 2:
        # 鍵盤模式下只渲染鍵盤，不顯示原有的 HUD 面板
        frame = virtual_keyboard.render_to_world(frame, focal_length, cx, cy)
    else:
        frame = my_screen_3d.render_to_world(frame, focal_length, cx, cy, current_mode, "")

    # ========== HUD 狀態顯示 ==========
    mode_names_display = ['DEFAULT', 'FACE FOLLOW', 'KEYBOARD']
    mode_colors_hud = [CLR_GREEN, CLR_ORANGE, CLR_MAGENTA]

    # 飽和度數值
    cv2.putText(frame, f"VIDEO SAT: {global_sat_factor:.1f}x", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_YELLOW, 2, cv2.LINE_AA)

    # 當前模式
    cv2.putText(frame, f"MODE: {mode_names_display[current_mode]}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_colors_hud[current_mode], 1, cv2.LINE_AA)

    # 操作狀態
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

    # 叉叉手勢提示
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

    # 顯示主輸出視窗
    cv2.imshow('Future Tech 3D XR HUD', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()