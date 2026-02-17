#!/usr/bin/env python3
"""Scan MAVLink ports/bauds and report barometer message availability."""

import argparse
import time
from collections import Counter

from pymavlink import mavutil

WATCH_MSGS = (
    "HEARTBEAT",
    "ATTITUDE",
    "SCALED_PRESSURE",
    "SCALED_PRESSURE2",
    "SCALED_PRESSURE3",
    "HIGHRES_IMU",
    "VFR_HUD",
    "GPS_RAW_INT",
    "GLOBAL_POSITION_INT",
    "BAD_DATA",
)

BARO_MSGS = {"SCALED_PRESSURE", "SCALED_PRESSURE2", "SCALED_PRESSURE3", "HIGHRES_IMU", "VFR_HUD"}


def parse_csv_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Debug MAVLink barometer stream availability")
    parser.add_argument(
        "--ports",
        default="/dev/ttyACM0,/dev/ttyAMA0,/dev/serial0,/dev/ttyUSB0",
        help="Comma-separated serial ports to test",
    )
    parser.add_argument(
        "--bauds",
        default="115200,57600,921600",
        help="Comma-separated bauds to test",
    )
    parser.add_argument("--heartbeat-timeout", type=float, default=4.0, help="Heartbeat wait timeout (s)")
    parser.add_argument("--sample-seconds", type=float, default=8.0, help="Sampling time per port/baud (s)")
    parser.add_argument("--request-rate", type=float, default=10.0, help="Requested barometer stream rate (Hz)")
    return parser.parse_args()


def request_interval(conn, msg_name, hz):
    msg_id = getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{msg_name}", None)
    if msg_id is None:
        return
    interval_us = int(1_000_000 / max(hz, 0.1))
    conn.mav.command_long_send(
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        msg_id,
        interval_us,
        0,
        0,
        0,
        0,
        0,
    )


def test_link(port, baud, hb_timeout, sample_seconds, request_rate):
    print(f"\n=== Testing {port} @ {baud} ===")
    try:
        conn = mavutil.mavlink_connection(port, baud=baud, source_system=245)
    except Exception as exc:
        print(f"OPEN_FAIL: {exc}")
        return None

    try:
        conn.wait_heartbeat(timeout=hb_timeout)
        print(f"HEARTBEAT_OK: target_system={conn.target_system} target_component={conn.target_component}")
    except Exception as exc:
        print(f"HEARTBEAT_FAIL: {exc}")
        return {
            "ok": False,
            "port": port,
            "baud": baud,
            "counts": Counter(),
            "baro_count": 0,
            "heartbeat": False,
        }

    for msg_name in ("SCALED_PRESSURE", "SCALED_PRESSURE2", "SCALED_PRESSURE3", "HIGHRES_IMU", "VFR_HUD", "ATTITUDE"):
        request_interval(conn, msg_name, request_rate)

    counts = Counter()
    start = time.time()
    while time.time() - start < sample_seconds:
        msg = conn.recv_match(blocking=True, timeout=1.0)
        if msg is None:
            continue
        msg_type = msg.get_type()
        counts[msg_type] += 1

    baro_count = sum(counts[k] for k in BARO_MSGS)
    print(
        "COUNTS:",
        " ".join(f"{k}={counts.get(k, 0)}" for k in WATCH_MSGS if counts.get(k, 0) > 0) or "none",
    )
    if baro_count > 0:
        print(f"BARO_OK: received {baro_count} barometer-related packets")
    else:
        print("BARO_MISSING: no SCALED_PRESSURE/HIGHRES_IMU/VFR_HUD seen")

    return {
        "ok": True,
        "port": port,
        "baud": baud,
        "counts": counts,
        "baro_count": baro_count,
        "heartbeat": True,
    }


def main():
    args = parse_args()
    ports = parse_csv_list(args.ports)
    bauds = [int(x) for x in parse_csv_list(args.bauds)]

    print("MAVLink barometer debug scan")
    print(f"Ports: {ports}")
    print(f"Bauds: {bauds}")

    results = []
    for port in ports:
        for baud in bauds:
            result = test_link(port, baud, args.heartbeat_timeout, args.sample_seconds, args.request_rate)
            if result is not None:
                results.append(result)

    print("\n=== Summary ===")
    best = None
    for r in results:
        hb = "yes" if r["heartbeat"] else "no"
        print(f"{r['port']} @ {r['baud']}: heartbeat={hb} baro_count={r['baro_count']}")
        if r["heartbeat"] and (best is None or r["baro_count"] > best["baro_count"]):
            best = r

    if best and best["baro_count"] > 0:
        print(
            "RECOMMEND: set config/pixhawk.yaml -> "
            f"device: {best['port']} and baud: {best['baud']}"
        )
    else:
        print(
            "RECOMMEND: no barometer stream found on tested links. "
            "Check Pixhawk SERIALx_PROTOCOL=2 (MAVLink), SERIALx_BAUD, and stream rates."
        )


if __name__ == "__main__":
    main()
