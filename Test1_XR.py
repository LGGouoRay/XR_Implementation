import cv2
import mediapipe as mp
import numpy as np
import math

# ==========================================
# 1. 初始化與科幻配色設定 (BGR 格式)
# ==========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,              # 追蹤一隻手以確保流暢度
    min_detection_confidence=0.8,  # 提高信任度以減少誤判
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

# 科幻 HUD 配色
CLR_CYAN = (255, 255, 0)
CLR_YELLOW = (0, 255, 255)
CLR_WHITE = (255, 255, 255)
CLR_RED = (0, 0, 255)
CLR_GREEN = (0, 255, 0)
CLR_BG_L = (10, 10, 10)  # 超暗背景色（做出玻璃擬態）

# 啟動鏡頭
cap = cv2.VideoCapture(0)

print("➔ 未來科技 XR HUD 已啟動！")
print("➔ 操控指南：")
print("  1. 移動螢幕：伸出【大拇指】與【食指】在方塊內捏合（兩指靠近），即可拖曳移動。")
print("  2. 切換飽和度：單獨用【食指尖】點擊方塊內的按鈕（B&W / NORM / VIVID）。")
print("➔ 按下 'q' 鍵可關閉程式。")


# ==========================================
# 2. 定義科幻螢幕類別
# ==========================================
class SciFiScreen:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.is_dragging = False
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.color_base = CLR_CYAN
        self.current_saturation_mode = 1  # 預設為 1 (Normal)

        # 定義三個按鈕在虛擬螢幕內的 (相對座標 X, Y, 寬 W, 高 H)
        btn_w = w // 3 - 20
        btn_h = 40
        btn_y = h - 60
        self.buttons = [
            {"label": "B&W",   "rect": (20, btn_y, btn_w, btn_h), "sat": 0.0},
            {"label": "NORM",  "rect": (btn_w + 40, btn_y, btn_w, btn_h), "sat": 1.0}, # 已修正語法錯誤
            {"label": "VIVID", "rect": (btn_w * 2 + 60, btn_y, btn_w, btn_h), "sat": 2.5}
        ]
        self.btn_pressed = [False, False, False]

    def draw(self, img):
        # 製作半透明主體暗色背板
        overlay = img.copy()
        cv2.rectangle(overlay, (self.x, self.y), (self.x + self.w, self.y + self.h), CLR_BG_L, cv2.FILLED)
        
        # 混合透明度 (拖曳時螢幕變顯眼)
        alpha = 0.6 if not self.is_dragging else 0.8
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # 繪製科幻風不連續邊框 (呈現 [ ] 樣式)
        t = 2        # 邊框線條厚度
        c_len = 30   # 邊角長度
        x, y, w, h = self.x, self.y, self.w, self.h
        color = CLR_YELLOW if self.is_dragging else self.color_base

        # 左上角
        cv2.line(img, (x, y), (x + c_len, y), color, t)
        cv2.line(img, (x, y), (x, y + c_len), color, t)
        # 右上角
        cv2.line(img, (x + w, y), (x + w - c_len, y), color, t)
        cv2.line(img, (x + w, y), (x + w, y + c_len), color, t)
        # 左下角
        cv2.line(img, (x, y + h), (x + c_len, y + h), color, t)
        cv2.line(img, (x, y + h), (x, y + h - c_len), color, t)
        # 右下角
        cv2.line(img, (x + w, y + h), (x + w - c_len, y + h), color, t)
        cv2.line(img, (x + w, y + h), (x + w, y + h - c_len), color, t)

        # 繪製 HUD 標題文字
        cv2.putText(img, "H.U.D. - SATURATION CTRL", (x + 15, y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.line(img, (x + 15, y + 35), (x + 180, y + 35), color, 1)

        # 繪製三個功能按鈕
        for i, btn in enumerate(self.buttons):
            bx, by, bw, bh = btn["rect"]
            abs_bx = x + bx
            abs_by = y + by
            
            # 判斷按鈕外觀狀態顏色
            b_color = CLR_WHITE
            if i == self.current_saturation_mode: b_color = CLR_YELLOW  # 目前生效中的模式
            if self.btn_pressed[i]: b_color = CLR_RED                   # 手指正在觸碰中

            # 畫按鈕外框
            cv2.rectangle(img, (abs_bx, abs_by), (abs_bx + bw, abs_by + bh), b_color, 2)
            # 填入按鈕標籤
            cv2.putText(img, btn["label"], (abs_bx + 12, abs_by + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, b_color, 1, cv2.LINE_AA)

        return img

    def check_interaction(self, hand_landmarks, img_w, img_h):
        # 抓取大拇指尖 (4號) 與食指尖 (8號) 節點
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        tx, ty = int(thumb_tip.x * img_w), int(thumb_tip.y * img_h)
        ix, iy = int(index_tip.x * img_w), int(index_tip.y * img_h)

        # 1. 移動偵測：計算兩指尖距離，小於 40 像素視為「捏起」
        dist = math.sqrt((tx - ix)**2 + (ty - iy)**2)
        pinch_threshold = 40 
        is_pinching = dist < pinch_threshold

        mid_x, mid_y = (tx + ix) // 2, (ty + iy) // 2
        in_screen = (self.x < mid_x < self.x + self.w) and (self.y < mid_y < self.y + self.h)

        if is_pinching and in_screen:
            if not self.is_dragging:
                self.is_dragging = True
                self.drag_offset_x = mid_x - self.x
                self.drag_offset_y = mid_y - self.y
            else:
                self.x = mid_x - self.drag_offset_x
                self.y = mid_y - self.drag_offset_y
        else:
            self.is_dragging = False 

        # 2. 點擊偵測：僅在「非拖曳狀態」下，偵測食指是否點擊按鈕
        for i in range(3): self.btn_pressed[i] = False
        
        if not self.is_dragging: 
            rel_ix = ix - self.x
            rel_iy = iy - self.y

            for i, btn in enumerate(self.buttons):
                bx, by, bw, bh = btn["rect"]
                if bx < rel_ix < bx + bw and by < rel_iy < by + bh:
                    self.btn_pressed[i] = True
                    self.current_saturation_mode = i 

        return self.buttons[self.current_saturation_mode]["sat"]


# ==========================================
# 3. 全局影像處理函數（HSV 飽和度調整）
# ==========================================
def apply_saturation(img, factor):
    if factor == 1.0: return img  # 1.0 倍為原始畫面，不處理以最佳化效能
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 調整 S (飽和度) 通道，轉為 float 運算避免溢位
    s_float = s.astype(np.float32) * factor
    s_final = np.clip(s_float, 0, 255).astype(np.uint8)
    
    hsv_final = cv2.merge([h, s_final, v])
    return cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)


# ==========================================
# 4. 主程式執行迴圈
# ==========================================
# 初始化虛擬螢幕物件（起始坐標 x=100, y=100, 寬=360, 高=220）
my_virtual_screen = SciFiScreen(100, 100, 360, 220)
global_sat_factor = 1.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 畫面水平鏡像翻轉
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # 將影像轉為 RGB 送入 MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # 核心互動邏輯
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 檢查手勢互動並獲取當前應採用的飽和度係數
            global_sat_factor = my_virtual_screen.check_interaction(hand_landmarks, w, h)
            
            # 繪製高科技感的手部骨架（白色關節，細青色骨幹）
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=CLR_WHITE, thickness=1, circle_radius=1),
                mp_draw.DrawingSpec(color=CLR_CYAN, thickness=1, circle_radius=1)
            )
    else:
        # 當畫面沒有偵測到手時，強制解除拖曳狀態，防止下次手進來時物件暴衝
        my_virtual_screen.is_dragging = False

    # 渲染視覺效果
    # 1. 先將主背景畫面套用飽和度濾鏡
    frame = apply_saturation(frame, global_sat_factor)

    # 2. 再將虛擬螢幕 UI 畫在最上層，確保 UI 本身不受濾鏡顏色影響
    frame = my_virtual_screen.draw(frame)

    # 3. 左上角系統 HUD 數據流狀態顯示
    status_color = CLR_GREEN if global_sat_factor == 1.0 else (CLR_RED if global_sat_factor < 1.0 else CLR_YELLOW)
    mode_names = ['Black & White', 'Normal', 'Vivid']
    sat_text = f"SYS_MODE: {mode_names[my_virtual_screen.current_saturation_mode]}"
    
    cv2.putText(frame, f"VIDEO SAT: {global_sat_factor:.1f}x", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
    cv2.putText(frame, sat_text, (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_WHITE, 1, cv2.LINE_AA)

    # 開啟視窗顯示
    cv2.imshow('Future Tech XR HUD', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()