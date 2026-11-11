"""
Unit tests for speed_estimator.py
"""
import cv2
import numpy as np
from speed_estimator import SpeedEstimator

def test_linear_speed():
    # Setup: 60 FPS, scale factor: 20 pixels = 1 meter
    # Car moving at 20 meters/sec = 72 km/h
    # In 60 FPS, car moves 20 meters in 60 frames = 1/3 meter per frame = 6.666 pixels/frame
    fps = 60.0
    scale = 20.0 # 20 px/meter
    estimator = SpeedEstimator(fps=fps, scale_factor=scale, window_size=20)
    
    car_id = 1
    speed = None
    
    # Simulate 30 frames of motion
    for frame in range(30):
        # Y position increments by 6.666 px per frame (down the road)
        y = 100.0 + frame * (scale * 20.0 / fps)
        bbox = [200, y - 50, 300, y]
        speed = estimator.update(frame, car_id, bbox)

    print(f"Velocidad calculada para trayectoria constante de 72 km/h: {speed:.2f} km/h")
    # Expected speed ~ 72 km/h (+- 2 km/h tolerance due to smoothing)
    assert 70.0 <= speed <= 74.0, f"Speed calculation out of bounds: {speed}"
    print("[OK] Test de velocidad lineal exitoso!")

def test_homography_speed():
    fps = 60.0
    estimator = SpeedEstimator(fps=fps, window_size=20)
    
    # 4 points defining a road trapezoid of 3.5m wide x 30m long
    src_pts = np.float32([[400, 200], [600, 200], [800, 800], [200, 800]])
    dst_pts = np.float32([[0.0, 0.0], [3.5, 0.0], [3.5, 30.0], [0.0, 30.0]])
    estimator.set_calibration_points(src_pts, real_width_meters=3.5, real_length_meters=30.0)
    
    # Inverse homography to map known constant physical motion (0 -> 30 meters at 108 km/h) to pixels
    inv_H = np.linalg.inv(estimator.homography_matrix)
    
    car_id = 2
    speed = None
    # 60 frames = 1 sec, moving 30 meters = 30 m/s = 108 km/h
    for frame in range(60):
        real_y = (frame / 59.0) * 30.0
        real_x = 1.75 # Center of 3.5m lane
        ground_pt = np.array([[[real_x, real_y]]], dtype=np.float32)
        pixel_pt = cv2.perspectiveTransform(ground_pt, inv_H)[0][0]
        
        px, py = pixel_pt[0], pixel_pt[1]
        bbox = [px - 50, py - 60, px + 50, py]
        speed = estimator.update(frame, car_id, bbox)

    print(f"Velocidad calculada con homografia de perspectiva: {speed:.2f} km/h")
    assert 104.0 <= speed <= 112.0, f"Homography speed out of expected range: {speed}"
    print("[OK] Test de velocidad con homografía exitoso!")

if __name__ == '__main__':
    test_linear_speed()
    test_homography_speed()
    print("=" * 60)
    print("TODAS LAS PRUEBAS DE ESTIMACION DE VELOCIDAD FUERON EXITOSAS!")
    print("=" * 60)
