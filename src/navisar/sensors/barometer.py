"""Barometer module. Provides barometer utilities for NAVISAR."""

import math
import time


class BarometerHeightEstimator:
    """Track height from MAVLink barometer/altitude messages."""
    def __init__(self, mavlink_interface, fallback_m=1.0):
        self.mavlink_interface = mavlink_interface
        self.fallback_m = float(fallback_m)
        self.current_m = None
        self.raw_press_hpa = None
        self.raw_temp_c = None
        self.raw_alt_m = None
        self.last_valid_m = self.fallback_m
        self.last_msg_time = None
        self._base_alt_m = None

    @staticmethod
    def _pressure_to_alt_m(press_hpa, temp_c):
        # Matches the standalone barometer script logic.
        if press_hpa is None:
            return None
        if press_hpa <= 0:
            return None
        if temp_c is None:
            # Fallback when temperature is not available.
            return 44330.0 * (1.0 - (press_hpa / 1013.25) ** 0.1903)
        t_k = temp_c + 273.15
        return (t_k / 0.0065) * (1.0 - (press_hpa / 1013.25) ** (1.0 / 5.255))

    def update(self):
        """Fetch the latest barometer message."""
        if self.mavlink_interface is None:
            return
        msg = self.mavlink_interface.recv_barometer()
        if msg is None:
            return
        self.last_msg_time = time.time()
        msg_type = msg.get_type()
        if msg_type.startswith("SCALED_PRESSURE"):
            press_raw = getattr(msg, "press_abs", None)
            temp_raw = getattr(msg, "temperature", None)
            press_hpa = float(press_raw) if press_raw is not None else None
            temp_c = float(temp_raw) / 100.0 if temp_raw is not None else None
        elif msg_type == "HIGHRES_IMU":
            press_raw = getattr(msg, "abs_pressure", None)
            temp_raw = getattr(msg, "temperature", None)
            press_hpa = float(press_raw) if press_raw is not None else None
            temp_c = float(temp_raw) if temp_raw is not None else None
        else:
            return
        alt_m = self._pressure_to_alt_m(press_hpa, temp_c)
        if press_hpa is not None:
            self.raw_press_hpa = press_hpa
        if temp_c is not None:
            self.raw_temp_c = temp_c
        if alt_m is None:
            return
        self.raw_alt_m = alt_m
        if self._base_alt_m is None:
            self._base_alt_m = alt_m
        height_m = alt_m - self._base_alt_m
        if math.isfinite(height_m):
            self.current_m = height_m
            self.last_valid_m = height_m

    def get_height_m(self):
        """Return the best-known height estimate."""
        if self.current_m is not None:
            return self.current_m
        return self.last_valid_m
