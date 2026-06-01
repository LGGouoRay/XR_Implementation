# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        ██╗  ██╗██████╗       ██╗  ██╗██╗   ██╗██████╗            ║
║        ╚██╗██╔╝██╔══██╗      ██║  ██║██║   ██║██╔══██╗           ║
║         ╚███╔╝ ██████╔╝█████╗███████║██║   ██║██║  ██║           ║
║         ██╔██╗ ██╔══██╗╚════╝██╔══██║██║   ██║██║  ██║           ║
║        ██╔╝ ██╗██║  ██║      ██║  ██║╚██████╔╝██████╔╝           ║
║        ╚═╝  ╚═╝╚═╝  ╚═╝      ╚═╝  ╚═╝ ╚═════╝ ╚═════╝            ║
║                                                                  ║
║          FUTURE TECH 3D XR HUD SYSTEM v2.0                       ║
║                                                                  ║
║  多面板 XR 手勢控制系統                                            ║
║  • 3 個可獨立拖曳/旋轉的 3D 浮動面板                               ║
║  • 8 種進階手勢辨識                                                ║
║  • 粒子系統 + HUD 覆蓋層 + 8 種影像濾鏡                            ║
║  • 卡爾曼濾波深度估計 + 四元數消除萬向鎖                            ║
║  • 系統監控 (FPS / CPU / 記憶體)                                    ║
║  • 截圖系統 + 面板佈局記憶                                          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import math
import time
import json
import os
import sys

# ── 確保能從專案根目錄匯入所有模組 ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from core.camera import CameraManager
from core.hand_tracker import HandTracker3D
from core.math3d import get_rotation_matrix, project_point
from gestures.recognizer import GestureRecognizer, GestureType
from panels.base_panel import Panel3D
from panels.main_hud import MainHUDPanel
from panels.system_monitor import SystemMonitorPanel
from panels.media_control import MediaControlPanel
from effects.particles import ParticleSystem
from effects.hud_overlay import HUDOverlay
from effects.filters import FilterEngine


# ════════════════════════════════════════════════════════════════
#  XR HUD 系統主控制器
# ════════════════════════════════════════════════════════════════
class XRHUDSystem:
    """
    XR HUD 系統主控制器。
    
    整合所有子模組：
      - 攝影機管理 (CameraManager)
      - 手部追蹤 (HandTracker3D)
      - 手勢辨識 (GestureRecognizer)
      - 3 個 3D 面板 (MainHUD / SystemMonitor / MediaControl)
      - 粒子系統 (ParticleSystem)
      - HUD 覆蓋層 (HUDOverlay)
      - 影像濾鏡引擎 (FilterEngine)
      - 截圖系統
      - 面板佈局記憶
    """

    def __init__(self):
        """初始化 XR HUD 系統的所有子模組。"""
        
        print("╔══════════════════════════════════════════════════════════╗")
        print("║        FUTURE TECH 3D XR HUD SYSTEM v2.0               ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print()

        # ── 1. 攝影機 ──
        print("  [1/7] 初始化攝影機...")
        self.camera = CameraManager(CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT)
        img_w, img_h = self.camera.get_dimensions()
        print(f"         解析度: {img_w}×{img_h}")

        # ── 2. 手部追蹤器 ──
        print("  [2/7] 初始化手部追蹤引擎...")
        self.tracker = HandTracker3D()

        # ── 3. 手勢辨識器 ──
        print("  [3/7] 初始化手勢辨識系統...")
        self.gesture = GestureRecognizer()
        self._register_gesture_callbacks()

        # ── 4. 面板系統 ──
        print("  [4/7] 建立 3D 面板系統...")
        self.main_hud = MainHUDPanel()
        self.sys_monitor = SystemMonitorPanel()
        self.media_ctrl = MediaControlPanel()
        self.panels = [self.main_hud, self.sys_monitor, self.media_ctrl]
        print(f"         面板數量: {len(self.panels)}")

        # ── 5. 視覺效果 ──
        print("  [5/7] 初始化視覺效果引擎...")
        self.particles = ParticleSystem(img_w, img_h)
        self.hud_overlay = HUDOverlay(img_w, img_h)
        self.filter_engine = FilterEngine()

        # ── 6. 截圖系統 ──
        print("  [6/7] 初始化截圖系統...")
        self.screenshot_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            SCREENSHOT_DIR
        )
        os.makedirs(self.screenshot_dir, exist_ok=True)

        # ── 7. 面板佈局記憶 ──
        print("  [7/7] 載入面板佈局...")
        self.layout_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            LAYOUT_SAVE_FILE
        )
        self._load_layout()

        # ── 狀態變數 ──
        self.running = True
        self.frame_count = 0
        self.current_dragging_panel = None  # 當前被拖曳的面板引用
        self.was_pinching = False           # 上一幀是否正在捏合
        self.last_fps_time = time.time()
        self.fps = 0.0

        print()
        print("  ➔ 系統啟動完成！")
        print()
        print("  ╔════════════════════════════════════╗")
        print("  ║          操控指南                   ║")
        print("  ╠════════════════════════════════════╣")
        print("  ║  🤏 捏合   → 抓取/拖曳面板         ║")
        print("  ║  ☝️ 食指   → 懸停/點擊按鈕         ║")
        print("  ║  ✊ 握拳   → 隱藏最近面板           ║")
        print("  ║  🖐 張手   → 顯示所有面板           ║")
        print("  ║  👍 比讚   → 截圖保存               ║")
        print("  ║  ✌️ 比V    → 切換面板               ║")
        print("  ║  👋 揮手   → 切換濾鏡               ║")
        print("  ║  按 Q 鍵  → 結束程式               ║")
        print("  ╚════════════════════════════════════╝")
        print()

    # ────────────────────────────────────────────────────────────
    #  手勢回呼註冊
    # ────────────────────────────────────────────────────────────
    def _register_gesture_callbacks(self):
        """為各手勢類型註冊回呼函式。"""
        self.gesture.register_callback(GestureType.FIST, self._on_fist)
        self.gesture.register_callback(GestureType.OPEN_HAND, self._on_open_hand)
        self.gesture.register_callback(GestureType.THUMBS_UP, self._on_thumbs_up)
        self.gesture.register_callback(GestureType.PEACE, self._on_peace)
        self.gesture.register_callback(GestureType.SWIPE_LEFT, self._on_swipe_left)
        self.gesture.register_callback(GestureType.SWIPE_RIGHT, self._on_swipe_right)

    def _on_fist(self):
        """握拳 → 隱藏距離最近的可見面板"""
        # 找到最近的可見面板
        visible_panels = [p for p in self.panels if p.visible and p.target_show_anim > 0.5]
        if visible_panels:
            # 按深度排序，隱藏最近的
            closest = min(visible_panels, key=lambda p: p.get_depth())
            closest.hide()
            self.hud_overlay.add_status_message(
                f"PANEL [{closest.panel_id.upper()}] HIDDEN", CLR_YELLOW
            )

    def _on_open_hand(self):
        """張手 → 顯示所有隱藏的面板"""
        for panel in self.panels:
            if not panel.visible or panel.target_show_anim < 0.5:
                panel.show()
        self.hud_overlay.add_status_message("ALL PANELS RESTORED", CLR_GREEN)

    def _on_thumbs_up(self):
        """比讚 → 截圖保存"""
        self._take_screenshot()

    def _on_peace(self):
        """比V → 循環切換面板可見性"""
        # 依序切換：只顯示主面板 → 只顯示監控 → 只顯示媒體 → 全部顯示
        visible_ids = [p.panel_id for p in self.panels if p.visible and p.target_show_anim > 0.5]
        
        if len(visible_ids) == len(self.panels):
            # 全部顯示時，只保留主面板
            for p in self.panels:
                if p.panel_id != "main_hud":
                    p.hide()
            self.hud_overlay.add_status_message("MODE: MAIN HUD ONLY", CLR_CYAN)
        elif len(visible_ids) == 1 and "main_hud" in visible_ids:
            self.main_hud.hide()
            self.sys_monitor.show()
            self.hud_overlay.add_status_message("MODE: SYSTEM MONITOR", CLR_CYAN)
        elif len(visible_ids) == 1 and "sys_monitor" in visible_ids:
            self.sys_monitor.hide()
            self.media_ctrl.show()
            self.hud_overlay.add_status_message("MODE: MEDIA CONTROL", CLR_CYAN)
        else:
            # 其他情況：全部顯示
            for p in self.panels:
                p.show()
            self.hud_overlay.add_status_message("MODE: ALL PANELS", CLR_CYAN)

    def _on_swipe_left(self):
        """向左揮手 → 上一個濾鏡"""
        self.filter_engine.prev_mode()
        mode_name = self.filter_engine.get_mode_name()
        self.main_hud.current_filter_mode = self.filter_engine.get_mode_index()
        self.main_hud.mark_dirty()
        self.media_ctrl.set_filter(self.filter_engine.get_mode_index())
        # 更新主面板按鈕狀態
        for i, btn in enumerate(self.main_hud.buttons):
            btn.is_active = (i == self.filter_engine.get_mode_index())
        self.hud_overlay.add_status_message(f"FILTER: {mode_name}", CLR_YELLOW)

    def _on_swipe_right(self):
        """向右揮手 → 下一個濾鏡"""
        self.filter_engine.next_mode()
        mode_name = self.filter_engine.get_mode_name()
        self.main_hud.current_filter_mode = self.filter_engine.get_mode_index()
        self.main_hud.mark_dirty()
        self.media_ctrl.set_filter(self.filter_engine.get_mode_index())
        for i, btn in enumerate(self.main_hud.buttons):
            btn.is_active = (i == self.filter_engine.get_mode_index())
        self.hud_overlay.add_status_message(f"FILTER: {mode_name}", CLR_YELLOW)

    # ────────────────────────────────────────────────────────────
    #  截圖系統
    # ────────────────────────────────────────────────────────────
    def _take_screenshot(self):
        """擷取當前畫面並保存至 screenshots 資料夾。"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"xr_hud_{timestamp}.png"
        filepath = os.path.join(self.screenshot_dir, filename)

        if hasattr(self, '_last_rendered_frame') and self._last_rendered_frame is not None:
            cv2.imwrite(filepath, self._last_rendered_frame)
            self.hud_overlay.trigger_flash()
            self.hud_overlay.add_status_message(
                f"SCREENSHOT SAVED: {filename}", CLR_GREEN
            )
            # 產生粒子爆發效果 (畫面中央)
            img_w, img_h = self.camera.get_dimensions()
            self.particles.spawn_burst(img_w // 2, img_h // 2, count=40, color=CLR_WHITE)
            print(f"  📸 截圖已保存: {filepath}")

    # ────────────────────────────────────────────────────────────
    #  面板佈局保存/載入
    # ────────────────────────────────────────────────────────────
    def _save_layout(self):
        """保存所有面板的位置/旋轉狀態至 JSON 檔案。"""
        layout = {}
        for panel in self.panels:
            layout[panel.panel_id] = panel.to_dict()
        
        try:
            with open(self.layout_file, 'w', encoding='utf-8') as f:
                json.dump(layout, f, indent=2, ensure_ascii=False)
            print(f"  💾 面板佈局已保存: {self.layout_file}")
        except Exception as e:
            print(f"  ⚠️ 佈局保存失敗: {e}")

    def _load_layout(self):
        """從 JSON 檔案恢復面板佈局。"""
        if not os.path.exists(self.layout_file):
            return
        
        try:
            with open(self.layout_file, 'r', encoding='utf-8') as f:
                layout = json.load(f)
            
            for panel in self.panels:
                if panel.panel_id in layout:
                    panel.from_dict(layout[panel.panel_id])
            
            print(f"         佈局已恢復: {self.layout_file}")
        except Exception as e:
            print(f"         佈局載入失敗: {e}")

    # ════════════════════════════════════════════════════════════
    #  手部互動處理
    # ════════════════════════════════════════════════════════════
    def _handle_hand_interaction(self, hand_landmarks, pts_3d, hand_rot, f, cx, cy, frame):
        """
        處理單手的所有互動邏輯。
        
        流程:
          1. 手勢辨識 (確認連續幀穩定後觸發回呼)
          2. 捏合抓取/拖曳面板
          3. 食指觸碰/按鈕點擊
          4. 更新粒子軌跡
        """
        landmarks = hand_landmarks
        lm = landmarks.landmark
        
        # ── 1. 手勢辨識 ──
        gesture_result = self.gesture.update(landmarks, pts_3d)

        # ── 2. 捏合抓取邏輯 ──
        thumb_tip_3d = pts_3d[4]
        index_tip_3d = pts_3d[8]
        pinch_midpoint = (thumb_tip_3d + index_tip_3d) / 2.0

        # 使用 GestureRecognizer 的捏合數據
        (pinch_mid_2d, pinch_ratio) = self.gesture.get_pinch_data(landmarks)
        is_pinching = pinch_ratio < PINCH_THRESHOLD

        if is_pinching:
            if self.current_dragging_panel is None:
                # 尚未抓取任何面板 → 搜尋最近的可碰撞面板
                best_panel = None
                best_dist = float('inf')
                
                for panel in self.panels:
                    if not panel.visible or panel.show_anim < 0.3:
                        continue
                    dist = np.linalg.norm(pinch_midpoint - panel.T)
                    if dist < GRAB_RANGE and dist < best_dist:
                        best_dist = dist
                        best_panel = panel
                
                if best_panel is not None:
                    best_panel.start_drag(pinch_midpoint, hand_rot)
                    self.current_dragging_panel = best_panel
                    self.hud_overlay.add_status_message(
                        f"GRAB: {best_panel.panel_id.upper()}", CLR_YELLOW
                    )
            
            # 已抓取面板 → 更新拖曳
            if self.current_dragging_panel is not None:
                self.current_dragging_panel.update_drag(pinch_midpoint, hand_rot)
        else:
            # 鬆手 → 結束拖曳
            if self.current_dragging_panel is not None:
                self.current_dragging_panel.end_drag()
                self.current_dragging_panel = None

        self.was_pinching = is_pinching

        # ── 3. 食指觸碰按鈕邏輯 (非拖曳且非捏合狀態時) ──
        finger_local_pos = None
        finger_active = False

        if self.current_dragging_panel is None and not is_pinching:
            for panel in self.panels:
                if not panel.visible:
                    continue
                
                local_pos, is_click, hit_btn_id = panel.check_touch(index_tip_3d)
                
                if local_pos is not None:
                    finger_local_pos = local_pos
                    finger_active = True
                    
                    if is_click and hit_btn_id is not None:
                        # 觸發按鈕事件
                        self._handle_button_event(panel, hit_btn_id)
                        
                        # 在按鈕位置產生粒子爆發
                        idx_2d = project_point(index_tip_3d, f, cx, cy)
                        if idx_2d:
                            self.particles.spawn_burst(idx_2d[0], idx_2d[1], count=20)
                    
                    break  # 一次只觸碰一個面板

        # ── 4. 更新粒子軌跡 ──
        idx_2d = project_point(index_tip_3d, f, cx, cy)
        if idx_2d and finger_active:
            self.particles.update_trail(idx_2d[0], idx_2d[1], active=True)
        else:
            self.particles.update_trail(0, 0, active=False)

        # ── 5. 繪製雷射追蹤線 (食指懸停時) ──
        if finger_active and finger_local_pos is not None:
            idx_2d = project_point(pts_3d[8], f, cx, cy)
            
            # 找到被觸碰的面板，計算雷射著陸點
            for panel in self.panels:
                if not panel.visible:
                    continue
                R_screen = get_rotation_matrix(*panel.rot)
                laser_hit_world = panel.T + R_screen.dot(
                    np.array([finger_local_pos[0], finger_local_pos[1], 0.0])
                )
                laser_hit_2d = project_point(laser_hit_world, f, cx, cy)
                
                if idx_2d and laser_hit_2d:
                    cv2.line(frame, idx_2d, laser_hit_2d, CLR_CYAN, 1, cv2.LINE_AA)
                    cv2.circle(frame, laser_hit_2d, 5, CLR_YELLOW, -1)
                    cv2.circle(frame, laser_hit_2d, 10, CLR_CYAN, 1, cv2.LINE_AA)
                break

        return hand_rot

    # ────────────────────────────────────────────────────────────
    #  按鈕事件路由
    # ────────────────────────────────────────────────────────────
    def _handle_button_event(self, panel, button_id):
        """
        將按鈕事件路由至對應的面板處理器。
        
        Args:
            panel: 被觸碰的面板實例
            button_id: 按鈕的 action_id
        """
        if isinstance(panel, MainHUDPanel):
            panel.handle_button_press(button_id)
            # 同步濾鏡模式到濾鏡引擎和媒體面板
            self.filter_engine.set_mode(panel.get_current_filter_mode())
            self.media_ctrl.set_filter(panel.get_current_filter_mode())
            mode_name = self.filter_engine.get_mode_name()
            self.hud_overlay.add_status_message(f"FILTER: {mode_name}", CLR_YELLOW)
            
        elif isinstance(panel, MediaControlPanel):
            panel.handle_button_press(button_id)
            # 同步亮度/對比度到濾鏡引擎
            adj = panel.get_adjustments()
            self.filter_engine.set_brightness(adj["brightness"])
            self.filter_engine.set_contrast(adj["contrast"])
            
        elif isinstance(panel, SystemMonitorPanel):
            pass  # 系統監控面板沒有互動按鈕

    # ════════════════════════════════════════════════════════════
    #  主迴圈
    # ════════════════════════════════════════════════════════════
    def run(self):
        """執行 XR HUD 系統主迴圈。"""
        
        while self.running:
            # ── 讀取攝影機畫面 ──
            frame = self.camera.read_frame()
            if frame is None:
                print("  ⚠️ 攝影機讀取失敗，結束程式。")
                break
            
            img_h, img_w = frame.shape[:2]
            f, cx, cy = self.camera.get_intrinsics()
            
            # ── 計算 FPS ──
            now = time.time()
            dt = now - self.last_fps_time
            if dt > 0:
                self.fps = 1.0 / dt
            self.last_fps_time = now
            self.frame_count += 1

            # ── 手部追蹤 ──
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.tracker.process_frame(rgb_frame)

            tracking_quality = 0.0
            hand_rot = None
            hand_positions_for_radar = []

            if results.multi_hand_landmarks:
                tracking_quality = 1.0
                
                # 處理每隻手 (最多 2 隻)
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    lm = hand_landmarks.landmark
                    
                    # 3D 重建
                    pts_3d, h_rot = self.tracker.extract_hand_3d(
                        hand_landmarks, img_w, img_h, f, cx, cy, hand_index=hand_idx
                    )
                    
                    # 收集雷達顯示用的手部位置
                    wrist = lm[0]
                    hand_positions_for_radar.append(
                        (wrist.x, wrist.y, wrist.z)
                    )
                    
                    # 只讓第一隻手控制面板互動
                    if hand_idx == 0:
                        hand_rot = self._handle_hand_interaction(
                            hand_landmarks, pts_3d, h_rot, f, cx, cy, frame
                        )
                    
                    # 繪製手部骨架
                    self.tracker.draw_hand_landmarks(frame, hand_landmarks)
            else:
                # 沒有偵測到手部 → 結束所有拖曳
                if self.current_dragging_panel is not None:
                    self.current_dragging_panel.end_drag()
                    self.current_dragging_panel = None
                self.particles.update_trail(0, 0, active=False)

            # ── 更新主 HUD 遙測數據 ──
            if hand_rot is not None:
                self.main_hud.update_telemetry(
                    yaw=math.degrees(hand_rot[1]),
                    pitch=math.degrees(hand_rot[0]),
                    roll=math.degrees(hand_rot[2]),
                    depth=self.main_hud.T[2]
                )
                self.main_hud.mark_dirty()

            # ── 更新系統監控面板 ──
            self.sys_monitor.update_stats(self.fps, tracking_quality)

            # ── 同步媒體面板調整值到濾鏡引擎 ──
            adj = self.media_ctrl.get_adjustments()
            self.filter_engine.set_brightness(adj["brightness"])
            self.filter_engine.set_contrast(adj["contrast"])

            # ── 更新所有面板的姿態 (EMA 平滑) ──
            for panel in self.panels:
                panel.update_pose()

            # ── 套用影像濾鏡 (在面板渲染之前) ──
            frame = self.filter_engine.apply(frame)

            # ── 渲染 3D 面板 (按深度排序：遠的先畫) ──
            sorted_panels = sorted(
                [p for p in self.panels if p.visible],
                key=lambda p: p.get_depth(),
                reverse=True  # 深度大 (遠) 的先畫
            )
            for panel in sorted_panels:
                frame = panel.render_to_world(frame, f, cx, cy)

            # ── 更新並渲染粒子系統 ──
            self.particles.update()
            frame = self.particles.render(frame)

            # ── 更新並渲染 HUD 覆蓋層 ──
            self.hud_overlay.update(tracking_quality)
            frame = self.hud_overlay.render(
                frame,
                panels_info=None,
                hand_positions=hand_positions_for_radar
            )

            # ── 渲染頂層系統狀態 ──
            self._draw_top_status(frame)

            # ── 保存最後渲染的畫面 (截圖用) ──
            self._last_rendered_frame = frame.copy()

            # ── 顯示視窗 ──
            cv2.imshow('Future Tech 3D XR HUD v2.0', frame)

            # ── 鍵盤輸入 ──
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                self._take_screenshot()
            elif key == ord('r') or key == ord('R'):
                # 重設所有面板位置
                self.main_hud.T = MAIN_HUD_INITIAL_POS.copy().astype(float)
                self.main_hud.target_T = self.main_hud.T.copy()
                self.main_hud.rot = np.array([0.0, 0.0, 0.0])
                self.main_hud.target_rot = self.main_hud.rot.copy()
                
                self.sys_monitor.T = SYSMON_INITIAL_POS.copy().astype(float)
                self.sys_monitor.target_T = self.sys_monitor.T.copy()
                self.sys_monitor.rot = np.array([0.0, 0.0, 0.0])
                self.sys_monitor.target_rot = self.sys_monitor.rot.copy()
                
                self.media_ctrl.T = MEDIA_INITIAL_POS.copy().astype(float)
                self.media_ctrl.target_T = self.media_ctrl.T.copy()
                self.media_ctrl.rot = np.array([0.0, 0.0, 0.0])
                self.media_ctrl.target_rot = self.media_ctrl.rot.copy()
                
                for p in self.panels:
                    p.show()
                    p.mark_dirty()
                
                self.filter_engine.reset()
                self.media_ctrl.reset()
                self.hud_overlay.add_status_message("SYSTEM RESET", CLR_GREEN)
            elif key == ord('1'):
                self.main_hud.toggle_visibility()
            elif key == ord('2'):
                self.sys_monitor.toggle_visibility()
            elif key == ord('3'):
                self.media_ctrl.toggle_visibility()

        # ── 結束 ──
        self._cleanup()

    # ────────────────────────────────────────────────────────────
    #  頂層狀態文字
    # ────────────────────────────────────────────────────────────
    def _draw_top_status(self, frame):
        """
        在畫面上繪製頂層拖曳狀態指示。
        
        (注意：大部分 HUD 資訊已由 HUDOverlay 處理，
         這裡只顯示拖曳狀態)
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.current_dragging_panel is not None:
            panel_name = self.current_dragging_panel.panel_id.upper()
            cv2.putText(frame, f"DRAGGING: {panel_name}", (20, 120),
                        font, 0.45, CLR_RED, 1, cv2.LINE_AA)

    # ────────────────────────────────────────────────────────────
    #  清理與結束
    # ────────────────────────────────────────────────────────────
    def _cleanup(self):
        """釋放所有資源，保存佈局。"""
        print()
        print("  ➔ 正在關閉系統...")
        
        # 保存面板佈局
        self._save_layout()
        
        # 釋放資源
        self.tracker.release()
        self.camera.release()
        cv2.destroyAllWindows()
        
        print("  ➔ XR HUD 系統已安全關閉。")
        print()


# ════════════════════════════════════════════════════════════════
#  程式入口
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    system = XRHUDSystem()
    system.run()
