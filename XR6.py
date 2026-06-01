import cv2
import mediapipe as mp
import numpy as np
import math
import time

# ==========================================
# 1. 系統初始化與常數定義
# ==========================================
mp_hands, mp_draw, mp_face, mp_pose = mp.solutions.hands, mp.solutions.drawing_utils, mp.solutions.face_mesh, mp.solutions.pose
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.6, min_tracking_confidence=0.6, refine_landmarks=True)
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# 配色定義 (BGR)
C_CYAN, C_YELLOW, C_WHITE, C_RED, C_GREEN, C_BLUE, C_ORANGE, C_MAGENTA, C_PURPLE = \
    (255, 255, 0), (0, 255, 255), (255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 165, 255), (255, 0, 255), (180, 50, 200)

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

print("➔ 未來科技 3D XR HUD 系統已啟動！\n➔ 操控指南：\n  1. 【移動與旋轉】：大拇指與食指捏合即可拖曳/旋轉虛擬面板。\n  2. 【按鈕觸控】：食指尖與中指尖並攏即可點擊按鈕。\n  3. 【模式切換】：雙手交叉持續 2 秒即可切換模式。\n➔ 按下 'q' 鍵可結束程式。")

# ==========================================
# 2. 3D 數學投影與輔助函數
# ==========================================
def get_rotation_matrix(p, y, r):
    cx, sx, cy, sy, cz, sz = math.cos(p), math.sin(p), math.cos(y), math.sin(y), math.cos(r), math.sin(r)
    return np.array([
        [cy*cz + sy*sx*sz, -cy*sz + sy*sx*cz, sy*cx],
        [cx*sz, cx*cz, -sx],
        [-sy*cz + cy*sx*sz, sy*sz + cy*sx*cz, cy*cx]
    ])

def project_point(pt, f, cx, cy):
    return (int((pt[0] * f / pt[2]) + cx), int((pt[1] * f / pt[2]) + cy)) if pt is not None and pt[2] > 10.0 else None

def intersect(p1, p2, p3, p4):
    c = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    return (c(p3, p4, p1) * c(p3, p4, p2) < 0) and (c(p1, p2, p3) * c(p1, p2, p4) < 0)

class OneEuroFilter:
    def __init__(self, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self.x_prev = self.dx_prev = None

    def __call__(self, x, dt):
        x = np.array(x, dtype=np.float32)
        if self.x_prev is None:
            self.x_prev, self.dx_prev = x.copy(), np.zeros_like(x)
            return x
        a_d = 1.0 / (1.0 + 1.0 / (2 * math.pi * self.dcutoff * dt))
        self.dx_prev = a_d * ((x - self.x_prev) / dt) + (1.0 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * math.hypot(*self.dx_prev)
        a = 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * dt))
        self.x_prev = a * x + (1.0 - a) * self.x_prev
        return self.x_prev

# ==========================================
# 3. 交叉手勢 (叉叉) 偵測器
# ==========================================
class CrossGestureDetector:
    def __init__(self, hold_duration=1.2):
        self.hold_duration = hold_duration
        self._reset()

    def detect(self, pose_landmarks, img_w, img_h):
        self.just_triggered = False
        if not pose_landmarks: return self._handle_absence()
        lm = pose_landmarks.landmark
        if any(lm[i].visibility < 0.5 for i in [11, 12, 15, 16]): return self._handle_absence()
        
        p1, p2, p3, p4 = [(lm[i].x * img_w, lm[i].y * img_h) for i in [11, 15, 12, 16]]
        if intersect(p1, p2, p3, p4) and math.hypot(p2[0]-p4[0], p2[1]-p4[1]) < math.hypot(p1[0]-p3[0], p1[1]-p3[1]) * 0.85:
            now = time.time()
            if not self.is_crossing:
                self.cross_start_time, self.is_crossing = now, True
            self.last_seen_time = now
            self.progress = min((now - self.cross_start_time) / self.hold_duration, 1.0)
            if self.progress >= 1.0:
                self.just_triggered = True
                self._reset()
                return True, 1.0
            return False, self.progress
        return self._handle_absence()

    def _handle_absence(self):
        now = time.time()
        if self.is_crossing and (now - self.last_seen_time < 0.3): return False, self.progress
        self._reset()
        return False, 0.0

    def _reset(self):
        self.cross_start_time = self.last_seen_time = 0
        self.is_crossing = False
        self.progress = 0.0

# ==========================================
# 4. 臉部位置追蹤器 (高精度 3D 向量重隔版 v2)
# ==========================================
class FaceTracker:
    AVERAGE_FACE_WIDTH_MM, SCREEN_DISTANCE_MM = 140.0, 150.0

    def __init__(self):
        self.smooth_center = np.array([0.0, 0.0, 500.0])
        self.smooth_pitch = self.smooth_yaw = self.smooth_roll = 0.0
        self.smooth_v_forward = np.array([0.0, 0.0, 1.0])
        self.alpha = 0.15
        self.detected = self._initialized = False

    def update(self, face_results, img_w, img_h, f, cx, cy):
        if not face_results or not face_results.multi_face_landmarks:
            self.detected = False
            return
        
        lm = face_results.multi_face_landmarks[0].landmark
        cheek_dist = math.hypot((lm[234].x - lm[454].x) * img_w, (lm[234].y - lm[454].y) * img_h)
        Z_base = (self.AVERAGE_FACE_WIDTH_MM * f) / max(cheek_dist, 1.0)
        
        pts = {}
        for idx in [168, 10, 152, 234, 454]:
            l = lm[idx]
            z = Z_base * (1.0 + l.z * 0.5)
            pts[idx] = np.array([(l.x * img_w - cx) * z / f, (l.y * img_h - cy) * z / f, z])
        
        v_up_raw, v_right_raw = pts[10] - pts[152], pts[454] - pts[234]
        v_right = v_right_raw / (math.hypot(*v_right_raw) + 1e-8)
        v_up = v_up_raw - np.dot(v_up_raw, v_right) * v_right
        v_up /= (math.hypot(*v_up) + 1e-8)
        v_forward = np.cross(v_right, v_up)
        
        cx_val = math.sqrt(v_right[1]**2 + v_up[1]**2)
        p, y, r = (math.atan2(v_forward[1], cx_val), math.atan2(-v_forward[0], -v_forward[2]), math.atan2(v_right[1], -v_up[1])) if cx_val > 1e-6 else (math.atan2(v_forward[1], 0.0), math.atan2(-v_right[2], v_right[0]), 0.0)
            
        if not self._initialized:
            self.smooth_center = pts[168].copy()
            self.smooth_pitch, self.smooth_yaw, self.smooth_roll = p, y, r
            self.smooth_v_forward, self._initialized = v_forward.copy(), True
        else:
            a = self.alpha
            self.smooth_center = self.smooth_center * (1.0 - a) + pts[168] * a
            self.smooth_v_forward = self.smooth_v_forward * (1.0 - a) + v_forward * a
            self.smooth_v_forward /= math.hypot(*self.smooth_v_forward) + 1e-8
            
            smooth_ang = lambda old, new: old + a * math.atan2(math.sin(new - old), math.cos(new - old))
            self.smooth_pitch, self.smooth_yaw, self.smooth_roll = smooth_ang(self.smooth_pitch, p), smooth_ang(self.smooth_yaw, y), smooth_ang(self.smooth_roll, r)
        self.detected = True

    def get_screen_target_pos_and_rot(self, f, cx, cy):
        if not self.detected: return None, None
        screen_pos = self.smooth_center + self.smooth_v_forward * self.SCREEN_DISTANCE_MM
        screen_pos[2] = max(screen_pos[2], 80.0)
        return screen_pos, np.array([self.smooth_pitch, self.smooth_yaw, self.smooth_roll])

# ==========================================
# 5. 3D 空間四邊形繪製輔助函數
# ==========================================
def render_quad_3d(frame, canvas, T, rot, w_3d, h_3d, f, cx, cy, color, draw_style='full', alpha=0.75, draw_axes=False):
    R = get_rotation_matrix(*rot)
    hw, hh = w_3d / 2, h_3d / 2
    local_pts = np.array([[-hw, -hh, 0.0], [hw, -hh, 0.0], [hw, hh, 0.0], [-hw, hh, 0.0]])
    
    pts_3d = local_pts @ R.T + T
    if np.any(pts_3d[:, 2] <= 10.0): return frame
    
    img_corners = np.column_stack((
        (pts_3d[:, 0] * f / pts_3d[:, 2] + cx).astype(int),
        (pts_3d[:, 1] * f / pts_3d[:, 2] + cy).astype(int)
    )).astype(np.float32)
    
    ch, cw = canvas.shape[:2]
    src_corners = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float32)
    
    try:
        if not cv2.isContourConvex(img_corners.astype(np.int32)): return frame
        H, _ = cv2.findHomography(src_corners, img_corners)
        warped = cv2.warpPerspective(canvas, H, (frame.shape[1], frame.shape[0]))
        mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.rectangle(mask, (0, 0), (cw, ch), 255, -1)
        warped_mask = cv2.warpPerspective(mask, H, (frame.shape[1], frame.shape[0]))
        
        blended = cv2.addWeighted(warped, alpha, frame, 1.0 - alpha, 0)
        np.copyto(frame, blended, where=warped_mask[:, :, None] >= 3)
    except:
        pass
        
    pts = img_corners.astype(np.int32)
    if draw_style == 'full':
        cv2.polylines(frame, [pts], True, color, 1, cv2.LINE_AA)
    elif draw_style == 'corners':
        c_len = 25
        for idx in range(4):
            curr, nxt, prv = pts[idx], pts[(idx + 1) % 4], pts[(idx - 1) % 4]
            d_nxt = math.hypot(*(nxt - curr)) + 1e-6
            d_prv = math.hypot(*(prv - curr)) + 1e-6
            cv2.line(frame, tuple(curr), tuple((curr + (nxt - curr) * (c_len / d_nxt)).astype(int)), color, 2, cv2.LINE_AA)
            cv2.line(frame, tuple(curr), tuple((curr + (prv - curr) * (c_len / d_prv)).astype(int)), color, 2, cv2.LINE_AA)
            
    if draw_axes:
        op = project_point(T, f, cx, cy)
        if op:
            for col_idx, scale, col in [(0, 40, C_RED), (1, -40, C_GREEN), (2, 40, C_BLUE)]:
                pt_2d = project_point(T + R[:, col_idx] * scale, f, cx, cy)
                if pt_2d: cv2.line(frame, op, pt_2d, col, 2, cv2.LINE_AA)
    return frame

# ==========================================
# 6. 虛擬鍵盤類別
# ==========================================
class VirtualKeyboard:
    def __init__(self):
        self.canvas_w, self.canvas_h, self.w_3d, self.h_3d = 600, 280, 360, 168
        self.rows = [list("QWERTYUIOP"), list("ASDFGHJKL"), list("ZXCVBNM"), ["SPACE", "DEL", "ENTER"]]
        self.typed_text = ""
        self.T, self.rot = np.array([0.0, 0.0, 480.0]), np.array([0.0, 0.0, 0.0])
        self.smooth_alpha = 0.5
        self.key_cooldown = {}
        self.cooldown_time = 0.4
        self.hover_key = self.pressed_key = None
        self.press_flash_time = 0
        
        # 預先計算漸層背景
        self.bg_canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        for y in range(self.canvas_h):
            val = int(10 + (y / self.canvas_h) * 15)
            self.bg_canvas[y, :] = (val, val + 2, val)
            
        # 預先計算鍵盤按鍵範圍
        self.key_rects = []
        margin_x, margin_y, key_h, gap = 8, 70, 42, 5
        total_w = self.canvas_w - margin_x * 2
        for r_idx, row in enumerate(self.rows):
            y = margin_y + r_idx * (key_h + gap)
            if r_idx == 3:
                sw, dw, ew = total_w // 2 - gap, total_w // 4 - gap, total_w // 4 - gap
                self.key_rects.extend([
                    (margin_x, y, sw, key_h, "SPACE"),
                    (margin_x + sw + gap, y, dw, key_h, "DEL"),
                    (margin_x + sw + gap + dw + gap, y, ew, key_h, "ENTER")
                ])
            else:
                n = len(row)
                kw = (total_w - gap * (n - 1)) // n
                ox = margin_x + (r_idx % 2) * 10
                for k_idx, lbl in enumerate(row):
                    self.key_rects.append((ox + k_idx * (kw + gap), y, kw, key_h, lbl))

    def draw_canvas(self):
        canvas = self.bg_canvas.copy()
        cv2.rectangle(canvas, (2, 2), (self.canvas_w - 2, self.canvas_h - 2), C_MAGENTA, 1)
        cv2.rectangle(canvas, (10, 10), (self.canvas_w - 10, 60), (30, 30, 30), -1)
        cv2.rectangle(canvas, (10, 10), (self.canvas_w - 10, 60), C_CYAN, 1)
        
        cursor = "_" if int(time.time() * 2) % 2 == 0 else ""
        disp = self.typed_text if len(self.typed_text) < 30 else "..." + self.typed_text[-27:]
        cv2.putText(canvas, f"INPUT> {disp}{cursor}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_WHITE, 1, cv2.LINE_AA)
        
        now = time.time()
        for x, y, w, h, lbl in self.key_rects:
            bg, fg, border = (40, 40, 40), C_WHITE, (80, 80, 80)
            if self.pressed_key == lbl and (now - self.press_flash_time) < 0.15:
                bg, fg, border = (0, 200, 200), (0, 0, 0), C_YELLOW
            elif self.hover_key == lbl:
                bg, border = (50, 60, 60), C_CYAN
                
            cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 1)
            
            f_scale = 0.35 if len(lbl) > 1 else 0.45
            ts = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, f_scale, 1)[0]
            cv2.putText(canvas, lbl, (x + (w - ts[0]) // 2, y + (h + ts[1]) // 2), cv2.FONT_HERSHEY_SIMPLEX, f_scale, fg, 1, cv2.LINE_AA)
        return canvas

    def handle_touch(self, local_finger, is_im_pinched):
        if local_finger is None: return
        u, v, w = local_finger
        cx_x = int(((u + self.w_3d / 2) / self.w_3d) * self.canvas_w)
        cx_y = int(((v + self.h_3d / 2) / self.h_3d) * self.canvas_h)
        
        self.hover_key = None
        now = time.time()
        
        if -30.0 <= w <= 80.0:
            for x, y, width, height, lbl in self.key_rects:
                if x < cx_x < x + width and y < cx_y < y + height:
                    self.hover_key = lbl
                    if is_im_pinched and (now - self.key_cooldown.get(lbl, 0) > self.cooldown_time):
                        self.key_cooldown[lbl] = now
                        self.pressed_key, self.press_flash_time = lbl, now
                        if lbl == "SPACE": self.typed_text += " "
                        elif lbl == "DEL": self.typed_text = self.typed_text[:-1]
                        elif lbl == "ENTER": self.typed_text += "\n"
                        else: self.typed_text += lbl
                        if len(self.typed_text) > 200: self.typed_text = self.typed_text[-200:]
                    break

    def update_pose(self):
        target_T, target_rot = np.array([0.0, 0.0, 480.0]), np.array([0.0, 0.0, 0.0])
        self.T = self.T * (1.0 - self.smooth_alpha) + target_T * self.smooth_alpha
        self.rot = self.rot * (1.0 - self.smooth_alpha) + target_rot * self.smooth_alpha

    def get_local_finger_for_keyboard(self, index_tip_3d):
        return get_rotation_matrix(*self.rot).T.dot(index_tip_3d - self.T)

# ==========================================
# 7. 3D 虛擬螢幕類別
# ==========================================
class SciFiScreen3D:
    def __init__(self, w_3d=300, h_3d=180):
        self.w_3d, self.h_3d = w_3d, h_3d
        self.T = np.array([0.0, -30.0, 450.0])
        self.rot = np.array([0.0, 0.0, 0.0])
        self.default_T, self.default_rot = self.T.copy(), self.rot.copy()
        self.target_T, self.target_rot = self.T.copy(), self.rot.copy()
        self.filter_T = OneEuroFilter(mincutoff=0.8, beta=0.03)
        self.filter_rot = OneEuroFilter(mincutoff=0.5, beta=0.01)
        self.last_time = time.time()
        
        self.is_dragging = False
        self.drag_start_hand_pos = self.drag_start_screen_pos = None
        self.drag_start_hand_rot = self.drag_start_screen_rot = None
        
        self.canvas_w, self.canvas_h = 400, 240
        self.current_saturation_mode = 1
        
        # 預先計算網格背景
        self.bg_canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        for x in range(0, self.canvas_w, 20): cv2.line(self.bg_canvas, (x, 0), (x, self.canvas_h), (25, 25, 10), 1)
        for y in range(0, self.canvas_h, 20): cv2.line(self.bg_canvas, (0, y), (self.canvas_w, y), (25, 25, 10), 1)
        
        btn_w = self.canvas_w // 3 - 20
        btn_h, btn_y = 45, self.canvas_h - 65
        self.buttons = [
            {"label": "B&W",   "rect": (15, btn_y, btn_w, btn_h), "sat": 0.0},
            {"label": "NORM",  "rect": (btn_w + 30, btn_y, btn_w, btn_h), "sat": 1.0},
            {"label": "VIVID", "rect": (btn_w * 2 + 45, btn_y, btn_w, btn_h), "sat": 2.5}
        ]
        self.btn_hover = [False] * 3
        self.btn_pressed = [False] * 3

    def save_current_as_default(self):
        self.default_T, self.default_rot = self.T.copy(), self.rot.copy()

    def restore_default(self):
        self.target_T, self.target_rot = self.default_T.copy(), self.default_rot.copy()

    def draw_canvas(self, current_mode=0):
        canvas = self.bg_canvas.copy()
            
        cv2.rectangle(canvas, (5, 5), (self.canvas_w - 5, self.canvas_h - 5), C_CYAN, 1)
        cv2.putText(canvas, "X.R. PROJECTION MATRIX", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_CYAN, 1, cv2.LINE_AA)
        cv2.line(canvas, (15, 38), (200, 38), C_CYAN, 1)
        
        lbls, cols = ["DEFAULT", "FACE FOLLOW"], [C_GREEN, C_ORANGE]
        cv2.putText(canvas, f"MODE: {lbls[current_mode]}", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cols[current_mode], 1, cv2.LINE_AA)
        
        yaw_d, pitch_d, roll_d = map(math.degrees, self.rot)
        stats = [f"YAW  : {yaw_d:+.1f} deg", f"PITCH: {pitch_d:+.1f} deg", f"ROLL : {roll_d:+.1f} deg", f"DEPTH: {self.T[2]:.1f} mm"]
        for i, txt in enumerate(stats): cv2.putText(canvas, txt, (15, 65 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_WHITE, 1, cv2.LINE_AA)
                
        for i, btn in enumerate(self.buttons):
            bx, by, bw, bh = btn["rect"]
            b_col = C_YELLOW if i == self.current_saturation_mode else (C_RED if self.btn_pressed[i] else (C_CYAN if self.btn_hover[i] else C_WHITE))
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), b_col, -1 if i == self.current_saturation_mode else 2)
            cv2.putText(canvas, btn["label"], (bx + 15, by + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0) if i == self.current_saturation_mode else b_col, 1, cv2.LINE_AA)
            
        return canvas

    def update_pose(self):
        now = time.time()
        dt = max(now - self.last_time, 1e-4)
        self.last_time = now
        self.T = self.filter_T(self.target_T, dt)
        diff_rot = np.arctan2(np.sin(self.target_rot - self.rot), np.cos(self.target_rot - self.rot))
        self.rot = self.filter_rot(self.rot + diff_rot, dt)

# ==========================================
# 8. 實體互動與手勢追蹤引擎
# ==========================================
def extract_hand_data_3d(landmarks, img_w, img_h, f, cx, cy):
    dx1, dy1 = (landmarks[17].x - landmarks[5].x) * img_w, (landmarks[17].y - landmarks[5].y) * img_h
    dx2, dy2 = (landmarks[9].x - landmarks[0].x) * img_w, (landmarks[9].y - landmarks[0].y) * img_h
    Z_hand = (120.0 / max((math.hypot(dx1, dy1) + math.hypot(dx2, dy2)) / 2.0, 1.0)) * 400.0
    
    lms_arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    pts_3d = np.empty((21, 3))
    pts_3d[:, 2] = Z_hand * (1.0 + lms_arr[:, 2] * 1.2)
    pts_3d[:, 0] = (lms_arr[:, 0] * img_w - cx) * pts_3d[:, 2] / f
    pts_3d[:, 1] = (lms_arr[:, 1] * img_h - cy) * pts_3d[:, 2] / f
    
    v_up = pts_3d[9] - pts_3d[0]
    v_up /= (math.hypot(*v_up) + 1e-6)
    v_right = pts_3d[17] - pts_3d[5]
    v_right /= (math.hypot(*v_right) + 1e-6)
    v_forward = np.cross(v_right, v_up)
    v_forward /= (math.hypot(*v_forward) + 1e-6)
    v_right = np.cross(v_up, v_forward)
    
    sy = math.sqrt(v_up[2]**2 + v_forward[2]**2)
    pitch, yaw, roll = (math.atan2(v_up[2], v_forward[2]), math.atan2(-v_right[2], sy), math.atan2(v_right[1], v_right[0])) if sy > 1e-6 else (math.atan2(-v_forward[1], v_up[1]), math.atan2(-v_right[2], sy), 0.0)
        
    return pts_3d, np.array([pitch, yaw, roll])

def handle_interaction(screen, landmarks, pts_3d, hand_rot):
    thumb_tip, index_tip = pts_3d[4], pts_3d[8]
    t_2d, idx_2d, w_2d, m_2d = landmarks[4], landmarks[8], landmarks[0], landmarks[9]
    pinch_dist = math.hypot(t_2d.x - idx_2d.x, t_2d.y - idx_2d.y)
    palm_dist = math.hypot(w_2d.x - m_2d.x, w_2d.y - m_2d.y)
    is_pinching = (pinch_dist / max(palm_dist, 1e-6)) < 0.25
    pinch_mid = (thumb_tip + index_tip) / 2.0
    
    if is_pinching:
        if not screen.is_dragging and math.hypot(*(pinch_mid - screen.T)) < 250.0:
            screen.is_dragging = True
            screen.drag_start_hand_pos, screen.drag_start_screen_pos = pinch_mid.copy(), screen.T.copy()
            screen.drag_start_hand_rot, screen.drag_start_screen_rot = hand_rot.copy(), screen.rot.copy()
        if screen.is_dragging:
            screen.target_T = screen.drag_start_screen_pos + (pinch_mid - screen.drag_start_hand_pos)
            screen.target_rot = screen.drag_start_screen_rot + (hand_rot - screen.drag_start_hand_rot)
    else:
        screen.is_dragging = False

    for i in range(3): screen.btn_hover[i] = screen.btn_pressed[i] = False
        
    if not screen.is_dragging and not is_pinching:
        local = get_rotation_matrix(*screen.rot).T.dot(index_tip - screen.T)
        u, v, w = local
        hw, hh = screen.w_3d / 2, screen.h_3d / 2
        if -hw - 20 < u < hw + 20 and -hh - 20 < v < hh + 20:
            click_pinch = math.hypot(idx_2d.x - landmarks[12].x, idx_2d.y - landmarks[12].y)
            is_click = (-30.0 <= w <= 80.0) and (click_pinch / max(palm_dist, 1e-6) < 0.22)
            is_hover = (-30.0 <= w <= 80.0) and not is_click
            cx_x = int(((u + hw) / screen.w_3d) * screen.canvas_w)
            cx_y = int(((v + hh) / screen.h_3d) * screen.canvas_h)
            for i, btn in enumerate(screen.buttons):
                bx, by, bw, bh = btn["rect"]
                if bx < cx_x < bx + bw and by < cx_y < by + bh:
                    if is_click:
                        screen.btn_pressed[i], screen.current_saturation_mode = True, i
                    elif is_hover:
                        screen.btn_hover[i] = True
            return local, True
    return None, False

# ==========================================
# 9. 進度條 / 進度環繪製器
# ==========================================
def draw_mode_switch_progress(frame, progress, current_mode, next_mode):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    names = ["DEFAULT", "FACE FOLLOW", "KEYBOARD"]
    colors = [C_GREEN, C_ORANGE, C_MAGENTA]
    ncol = colors[next_mode]
    
    roi = frame[cy - 100:cy + 100, cx - 160:cx + 160]
    frame[cy - 100:cy + 100, cx - 160:cx + 160] = roi >> 1
    
    cv2.rectangle(frame, (cx - 160, cy - 100), (cx + 160, cy + 100), ncol, 2)
    cv2.putText(frame, "MODE SWITCH", (cx - 70, cy - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_WHITE, 1, cv2.LINE_AA)
    
    cv2.putText(frame, names[current_mode], (cx - 130, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[current_mode], 1, cv2.LINE_AA)
    cv2.putText(frame, ">>>", (cx - 15, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, names[next_mode], (cx + 30, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ncol, 1, cv2.LINE_AA)
    
    cv2.ellipse(frame, (cx, cy + 10), (30, 30), -90, 0, 360, (40, 40, 40), 3, cv2.LINE_AA)
    if progress > 0:
        cv2.ellipse(frame, (cx, cy + 10), (30, 30), -90, 0, int(progress * 360), ncol, 4, cv2.LINE_AA)
        
    pct = f"{int(progress * 100)}%"
    ts = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(frame, pct, (cx - ts[0] // 2, cy + 10 + ts[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WHITE, 1, cv2.LINE_AA)
    
    cv2.rectangle(frame, (cx - 130, cy + 60), (cx + 130, cy + 72), (40, 40, 40), -1)
    if progress > 0:
        cv2.rectangle(frame, (cx - 130, cy + 60), (cx - 130 + int(260 * progress), cy + 72), ncol, -1)
    cv2.rectangle(frame, (cx - 130, cy + 60), (cx + 130, cy + 72), ncol, 1)
    
    cv2.putText(frame, f"{max(0.0, 2.0 * (1.0 - progress)):.1f}s", (cx - 15, cy + 97), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ncol, 1, cv2.LINE_AA)
    return frame

def draw_cross_indicator(frame, pose_landmarks, img_w, img_h):
    if not pose_landmarks: return frame
    lm = pose_landmarks.landmark
    if any(lm[i].visibility < 0.5 for i in [11, 12, 15, 16]): return frame
    
    pts = [(int(lm[i].x * img_w), int(lm[i].y * img_h)) for i in [11, 15, 12, 16]]
    cv2.line(frame, pts[0], pts[1], C_ORANGE, 5, cv2.LINE_AA)
    cv2.line(frame, pts[2], pts[3], C_ORANGE, 5, cv2.LINE_AA)
    cv2.circle(frame, pts[1], 8, C_YELLOW, -1)
    cv2.circle(frame, pts[3], 8, C_YELLOW, -1)
    return frame

# ==========================================
# 10. 全局色彩與飽和度處理
# ==========================================
def apply_saturation(img, factor):
    if factor == 1.0: return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=factor)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# ==========================================
# 11. 主程式執行迴圈
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

frame_count = 0
last_pose_landmarks = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    img_h, img_w, _ = frame.shape
    focal_length = img_w
    cx, cy = img_w / 2, img_h / 2
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    
    frame_count += 1
    if frame_count % 3 == 0:
        pose_results = pose.process(rgb_frame)
        last_pose_landmarks = pose_results.pose_landmarks if pose_results else None
        
    face_results = None
    if current_mode == 1:
        face_results = face_mesh.process(rgb_frame)
        face_tracker.update(face_results, img_w, img_h, focal_length, cx, cy)
    else:
        face_tracker.detected = False

    finger_local_pos = None
    finger_active = False
    now = time.time()
    cross_triggered, cross_progress = False, 0.0

    if now - mode_switch_cooldown > MODE_SWITCH_COOLDOWN_TIME:
        cross_triggered, cross_progress = cross_detector.detect(last_pose_landmarks, img_w, img_h)
    else:
        cross_detector._reset()

    if cross_triggered:
        old_mode, current_mode = current_mode, (current_mode + 1) % 3
        mode_switch_cooldown = now
        mode_transition_active, mode_transition_start = True, now

        print(f"➔ 模式切換: {['預設', '臉部跟隨', '鍵盤'][old_mode]} → {['預設', '臉部跟隨', '鍵盤'][current_mode]}")

        my_screen_3d.is_dragging = False
        if current_mode == 0:
            my_screen_3d.restore_default()
        elif current_mode == 1:
            my_screen_3d.save_current_as_default()
            my_screen_3d.target_rot = np.array([0.0, 0.0, 0.0])
        elif current_mode == 2:
            virtual_keyboard.typed_text = ""

    if mode_transition_active:
        elapsed = now - mode_transition_start
        mode_transition_alpha = max(0.0, 1.0 - elapsed / 0.5) if elapsed < 0.5 else 0.0
        if mode_transition_alpha == 0.0:
            mode_transition_active = False

    if results.multi_hand_landmarks:
        if cross_progress < 0.1:
            lms = results.multi_hand_landmarks[0].landmark
            pts_3d, hand_rot = extract_hand_data_3d(lms, img_w, img_h, focal_length, cx, cy)

            if current_mode in [0, 1]:
                finger_local_pos, finger_active = handle_interaction(my_screen_3d, lms, pts_3d, hand_rot)
            elif current_mode == 2:
                kb_local = virtual_keyboard.get_local_finger_for_keyboard(pts_3d[8])
                idx_2d, mid_2d, w_2d, m_2d = lms[8], lms[12], lms[0], lms[9]
                palm_dist = math.hypot(w_2d.x - m_2d.x, w_2d.y - m_2d.y)
                is_pinched = (math.hypot(idx_2d.x - mid_2d.x, idx_2d.y - mid_2d.y) / max(palm_dist, 1e-6)) < 0.22

                virtual_keyboard.handle_touch(kb_local, is_pinched)
                my_screen_3d.is_dragging = False
                finger_active, finger_local_pos = True, kb_local

            global_sat_factor = my_screen_3d.buttons[my_screen_3d.current_saturation_mode]["sat"]

            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=C_WHITE, thickness=1, circle_radius=1),
                    mp_draw.DrawingSpec(color=C_CYAN, thickness=1, circle_radius=1)
                )

            if finger_active and finger_local_pos is not None:
                idx_2d = project_point(pts_3d[8], focal_length, cx, cy)
                T, rot = (virtual_keyboard.T, virtual_keyboard.rot) if current_mode == 2 else (my_screen_3d.T, my_screen_3d.rot)
                R = get_rotation_matrix(*rot)
                laser_hit_w = T + R[:, 0] * finger_local_pos[0] + R[:, 1] * finger_local_pos[1]
                laser_hit_2d = project_point(laser_hit_w, focal_length, cx, cy)

                if idx_2d and laser_hit_2d:
                    cv2.line(frame, idx_2d, laser_hit_2d, C_CYAN, 1, cv2.LINE_AA)
                    w = finger_local_pos[2]
                    dist_factor = min(max(abs(w) / 80.0, 0.0), 1.0)
                    ring_radius = int(5 + 20 * dist_factor)
                    ring_color = (0, int(255 * dist_factor), 255)
                    cv2.circle(frame, laser_hit_2d, ring_radius, ring_color, 2, cv2.LINE_AA)
                    cv2.circle(frame, laser_hit_2d, 3, C_WHITE, -1)
    else:
        my_screen_3d.is_dragging = False

    if current_mode == 1:
        t_pos, t_rot = face_tracker.get_screen_target_pos_and_rot(focal_length, cx, cy)
        if t_pos is not None:
            my_screen_3d.target_T = t_pos
            if not my_screen_3d.is_dragging:
                my_screen_3d.target_rot = t_rot

    if current_mode == 2:
        virtual_keyboard.update_pose()

    my_screen_3d.update_pose()
    frame = apply_saturation(frame, global_sat_factor)

    if current_mode == 2:
        canvas_kb = virtual_keyboard.draw_canvas()
        frame = render_quad_3d(frame, canvas_kb, virtual_keyboard.T, virtual_keyboard.rot, virtual_keyboard.w_3d, virtual_keyboard.h_3d, focal_length, cx, cy, C_MAGENTA, 'full', 0.8)
    else:
        canvas = my_screen_3d.draw_canvas(current_mode)
        color = C_YELLOW if my_screen_3d.is_dragging else [C_CYAN, C_ORANGE][current_mode]
        frame = render_quad_3d(frame, canvas, my_screen_3d.T, my_screen_3d.rot, my_screen_3d.w_3d, my_screen_3d.h_3d, focal_length, cx, cy, color, 'corners', 0.75, draw_axes=True)

    cv2.putText(frame, f"VIDEO SAT: {global_sat_factor:.1f}x", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_YELLOW, 2, cv2.LINE_AA)
    cv2.putText(frame, f"MODE: {['DEFAULT', 'FACE FOLLOW', 'KEYBOARD'][current_mode]}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [C_GREEN, C_ORANGE, C_MAGENTA][current_mode], 1, cv2.LINE_AA)
    
    status = "PANEL DRAGGING" if my_screen_3d.is_dragging else ("FACE TRACKING ACTIVE" if current_mode == 1 and face_tracker.detected else ("SEARCHING FACE..." if current_mode == 1 else ("KEYBOARD ACTIVE" if current_mode == 2 else "SYSTEM READY")))
    cv2.putText(frame, f"STATUS: {status}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_RED if "DRAGGING" in status or "SEARCHING" in status else C_GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, "X CROSS HANDS 2s -> SWITCH MODE", (20, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

    if cross_progress > 0.01:
        frame = draw_mode_switch_progress(frame, cross_progress, current_mode, (current_mode + 1) % 3)
        frame = draw_cross_indicator(frame, last_pose_landmarks, img_w, img_h)

    if mode_transition_active and mode_transition_alpha > 0:
        flash = np.full(frame.shape, [C_GREEN, C_ORANGE, C_MAGENTA][current_mode], dtype=np.uint8)
        frame = cv2.addWeighted(flash, mode_transition_alpha * 0.3, frame, 1.0 - mode_transition_alpha * 0.3, 0)

    cv2.imshow('Future Tech 3D XR HUD', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()