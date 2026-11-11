"""
Speed Estimator Module for Vehicle Tracking.
Calculates real-world speed (km/h) using tracking IDs and pixel-to-meter transformation
(perspective homography or calibrated metric scaling), with temporal smoothing.
"""
from collections import defaultdict, deque
import cv2
import numpy as np


class SpeedEstimator:
    def __init__(self, fps=60.0, homography_matrix=None, scale_factor=None, window_size=15):
        """
        Args:
            fps (float): Frames per second of the video.
            homography_matrix (np.ndarray, optional): 3x3 matrix mapping image pixels to ground meters.
            scale_factor (float, optional): Pixels per meter (fallback when homography is not provided).
            window_size (int): Number of frames over which to calculate smoothed speed.
        """
        self.fps = float(fps)
        self.homography_matrix = homography_matrix
        # Default scale factor if neither is provided: ~20 pixels per meter (common for 1080p traffic cams)
        self.scale_factor = scale_factor if scale_factor is not None else 20.0
        self.window_size = window_size
        
        # History per vehicle: car_id -> deque of (frame_nmr, meter_x, meter_y)
        self.history = defaultdict(lambda: deque(maxlen=self.window_size))
        # Cache of latest calculated speeds: car_id -> speed_kmh
        self.current_speeds = {}

    def set_calibration_points(self, src_points, real_width_meters=3.5, real_length_meters=20.0):
        """
        Convenience method to compute homography from 4 road points (e.g. lane boundary trapezoid).
        src_points: array-like of 4 points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] (top-left, top-right, bottom-right, bottom-left)
        """
        src = np.float32(src_points)
        dst = np.float32([
            [0.0, 0.0],
            [real_width_meters, 0.0],
            [real_width_meters, real_length_meters],
            [0.0, real_length_meters]
        ])
        self.homography_matrix = cv2.getPerspectiveTransform(src, dst)

    def _transform_point(self, pixel_x, pixel_y):
        """
        Transforms image pixel coordinate to real-world ground meters (X, Y).
        """
        if self.homography_matrix is not None:
            pt = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self.homography_matrix)
            return float(transformed[0][0][0]), float(transformed[0][0][1])
        else:
            # Fallback linear approximation
            return float(pixel_x) / self.scale_factor, float(pixel_y) / self.scale_factor

    def update(self, frame_nmr, car_id, bbox):
        """
        Updates tracking position and computes speed in km/h.

        Args:
            frame_nmr (int): Current frame number.
            car_id (int or str): Unique tracker identifier for vehicle.
            bbox (list or tuple): [x1, y1, x2, y2] bounding box.

        Returns:
            float: Estimated speed in km/h, or None if insufficient history.
        """
        # Use bottom-center point of bounding box (where car tires touch asphalt)
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        bottom_y = y2

        meter_x, meter_y = self._transform_point(center_x, bottom_y)
        car_hist = self.history[car_id]
        car_hist.append((frame_nmr, meter_x, meter_y))

        # Need at least 5 frames of history to avoid instant noise
        if len(car_hist) < 5:
            return self.current_speeds.get(car_id, None)

        oldest_frame, old_x, old_y = car_hist[0]
        newest_frame, new_x, new_y = car_hist[-1]

        delta_frames = newest_frame - oldest_frame
        if delta_frames <= 0:
            return self.current_speeds.get(car_id, None)

        delta_time_sec = delta_frames / self.fps
        distance_meters = np.hypot(new_x - old_x, new_y - old_y)

        # Speed in meters/second converted to km/h (1 m/s = 3.6 km/h)
        instant_speed_kmh = (distance_meters / delta_time_sec) * 3.6

        # Exponential smoothing with prior estimate
        prev_speed = self.current_speeds.get(car_id, instant_speed_kmh)
        alpha = 0.3  # Smoothing factor
        smoothed_speed = alpha * instant_speed_kmh + (1 - alpha) * prev_speed

        self.current_speeds[car_id] = smoothed_speed
        return smoothed_speed
