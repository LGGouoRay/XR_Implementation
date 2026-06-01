# -*- coding: utf-8 -*-
"""匯入驗證測試腳本"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

tests_passed = 0
tests_failed = 0

def test(name, func):
    global tests_passed, tests_failed
    try:
        func()
        print(f"  ✅ {name}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        tests_failed += 1

print("=" * 50)
print("  XR HUD 模組匯入驗證")
print("=" * 50)

# 1. Config
test("config", lambda: __import__("config"))

# 2. Core - math3d
def test_math3d():
    from core.math3d import get_rotation_matrix, project_point, quaternion_slerp
    from core.math3d import euler_to_quaternion, quaternion_to_euler
    import numpy as np
    R = get_rotation_matrix(0.1, 0.2, 0.3)
    assert R.shape == (3, 3)
    pt = project_point(np.array([10, 20, 100]), 500, 240, 135)
    assert pt is not None
    q = euler_to_quaternion(0.1, 0.2, 0.3)
    assert len(q) == 4
test("core.math3d", test_math3d)

# 3. Core - camera
test("core.camera", lambda: __import__("core.camera", fromlist=["CameraManager"]))

# 4. Core - hand_tracker
test("core.hand_tracker", lambda: __import__("core.hand_tracker", fromlist=["HandTracker3D"]))

# 5. Gestures
def test_gestures():
    from gestures.recognizer import GestureRecognizer, GestureType
    gr = GestureRecognizer()
    assert GestureType.PINCH.value == 1
test("gestures.recognizer", test_gestures)

# 6. Panels - base
test("panels.base_panel", lambda: __import__("panels.base_panel", fromlist=["Panel3D", "Button3D"]))

# 7. Panels - main_hud
test("panels.main_hud", lambda: __import__("panels.main_hud", fromlist=["MainHUDPanel"]))

# 8. Panels - system_monitor
test("panels.system_monitor", lambda: __import__("panels.system_monitor", fromlist=["SystemMonitorPanel"]))

# 9. Panels - media_control
test("panels.media_control", lambda: __import__("panels.media_control", fromlist=["MediaControlPanel"]))

# 10. Effects - particles
test("effects.particles", lambda: __import__("effects.particles", fromlist=["ParticleSystem"]))

# 11. Effects - hud_overlay
test("effects.hud_overlay", lambda: __import__("effects.hud_overlay", fromlist=["HUDOverlay"]))

# 12. Effects - filters
def test_filters():
    from effects.filters import FilterEngine
    fe = FilterEngine()
    assert fe.get_mode_name() == "NORMAL"
    fe.next_mode()
    assert fe.get_mode_name() == "VIVID"
test("effects.filters", test_filters)

# 13. Panel instantiation
def test_panels():
    from panels.main_hud import MainHUDPanel
    from panels.system_monitor import SystemMonitorPanel
    from panels.media_control import MediaControlPanel
    p1 = MainHUDPanel()
    p2 = SystemMonitorPanel()
    p3 = MediaControlPanel()
    assert len(p1.buttons) == 8
    assert p2.panel_id == "sys_monitor"
    assert p3.panel_id == "media_ctrl"
test("panel instantiation", test_panels)

# 14. Particle system
def test_particle_sys():
    from effects.particles import ParticleSystem
    ps = ParticleSystem(960, 540)
    assert ps.get_particle_count() > 0
    ps.spawn_burst(100, 100)
    ps.update()
test("particle system", test_particle_sys)

# 15. HUD overlay
def test_hud():
    from effects.hud_overlay import HUDOverlay
    hud = HUDOverlay(960, 540)
    hud.update(0.8)
    assert hud.get_frame_count() == 1
test("hud overlay", test_hud)

print("=" * 50)
print(f"  結果: {tests_passed} 通過, {tests_failed} 失敗")
print("=" * 50)

if tests_failed > 0:
    sys.exit(1)
else:
    print("  🎉 所有模組匯入測試通過！")
