"""Mode for raw GPS serial passthrough with timestamped logging."""

import datetime as dt
import time
from pathlib import Path

import serial


class GpsPassthroughMode:
    """Forward raw GPS bytes from sensor input directly to Pixhawk GPS port."""

    def __init__(
        self,
        input_port,
        input_baud,
        output_port,
        output_baud,
        log_dir,
        print_enabled=True,
        warn_interval_s=2.0,
        max_chunk_bytes=4096,
    ):
        self.input_port = input_port
        self.input_baud = int(input_baud)
        self.output_port = output_port
        self.output_baud = int(output_baud)
        self.log_dir = Path(log_dir)
        self.print_enabled = bool(print_enabled)
        self.warn_interval_s = float(warn_interval_s)
        self.max_chunk_bytes = int(max_chunk_bytes)
        self._last_warn = 0.0
        self._last_print = 0.0
        self._in_ser = None
        self._out_ser = None
        self._log_file = None
        self._log_path = None
        self.last_payload = None

    def _warn(self, now, message):
        if now - self._last_warn >= self.warn_interval_s:
            print(message)
            self._last_warn = now

    def _start_session(self):
        if self._in_ser is not None and self._out_ser is not None and self._log_file is not None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"{stamp}.txt"
        idx = 1
        while log_path.exists():
            idx += 1
            log_path = self.log_dir / f"{stamp}_{idx}.txt"

        self._in_ser = serial.Serial(self.input_port, self.input_baud, timeout=0)
        self._out_ser = serial.Serial(self.output_port, self.output_baud, timeout=0)
        self._log_file = log_path.open("w", encoding="utf-8")
        self._log_path = log_path
        print(
            "GPS passthrough started: "
            f"{self.input_port}@{self.input_baud} -> {self.output_port}@{self.output_baud}"
        )
        print(f"GPS passthrough log: {self._log_path}")

    def _log_chunk(self, raw_bytes):
        if self._log_file is None:
            return
        ts = dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
        text = raw_bytes.decode("ascii", errors="replace")
        text = text.replace("\r", "\\r").replace("\n", "\\n")
        self._log_file.write(f"{ts} | {text}\n")
        self._log_file.flush()

    def handle(self, now, max_chunks=32):
        """Forward available bytes and append timestamped log entries."""
        try:
            self._start_session()
        except Exception as exc:
            self._warn(now, f"GPS passthrough unavailable ({exc})")
            return

        total_bytes = 0
        chunks = 0
        for _ in range(max_chunks):
            in_waiting = int(getattr(self._in_ser, "in_waiting", 0))
            if in_waiting <= 0:
                break
            raw = self._in_ser.read(min(in_waiting, self.max_chunk_bytes))
            if not raw:
                break
            self._out_ser.write(raw)
            self._out_ser.flush()
            self._log_chunk(raw)
            total_bytes += len(raw)
            chunks += 1

        if total_bytes <= 0:
            return
        self.last_payload = {
            "time_s": now,
            "bytes_forwarded": total_bytes,
            "chunks": chunks,
            "input_port": self.input_port,
            "output_port": self.output_port,
            "log_file": str(self._log_path) if self._log_path else None,
        }
        if self.print_enabled and now - self._last_print >= 1.0:
            print(
                "GPS passthrough: "
                f"forwarded={total_bytes} bytes chunks={chunks} "
                f"log={self._log_path}"
            )
            self._last_print = now

    def close(self):
        """Flush and close serial ports and log file."""
        if self._log_file is not None:
            try:
                self._log_file.flush()
            except Exception:
                pass
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
        if self._in_ser is not None:
            try:
                self._in_ser.close()
            except Exception:
                pass
            self._in_ser = None
        if self._out_ser is not None:
            try:
                self._out_ser.close()
            except Exception:
                pass
            self._out_ser = None
        if self._log_path is not None:
            print(f"GPS passthrough log saved: {self._log_path}")
