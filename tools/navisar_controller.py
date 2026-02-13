#!/usr/bin/env python3
"""Always-on controller for NAVISAR service start/stop."""

import json
import os
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HOST = os.getenv("NAVISAR_CONTROL_HOST", "0.0.0.0")
PORT = int(os.getenv("NAVISAR_CONTROL_PORT", "8770"))
SERVICE_NAME = os.getenv("NAVISAR_SERVICE_NAME", "navisar.service")
GUI_PORT = int(os.getenv("NAVISAR_DASHBOARD_PORT", "8765"))


HTML = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NAVISAR Control</title>
  <style>
    body {{ background:#000; color:#f5f5f5; font-family:system-ui,sans-serif; margin:0; padding:24px; }}
    .card {{ max-width:640px; margin:auto; border:1px solid #222; border-radius:12px; padding:16px; background:#050505; }}
    h1 {{ margin:0 0 12px; font-size:20px; }}
    .row {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; }}
    button {{ border:1px solid #444; border-radius:999px; padding:8px 14px; background:#111; color:#fff; cursor:pointer; }}
    .start {{ border-color:#2a7; color:#9fd; }}
    .stop {{ border-color:#a33; color:#fbb; }}
    .muted {{ color:#aaa; font-size:13px; }}
    code {{ color:#ffb347; }}
    #log {{ white-space:pre-wrap; background:#000; border:1px solid #222; border-radius:8px; padding:10px; font-family:monospace; min-height:72px; }}
    a {{ color:#7ec8ff; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>NAVISAR Service Control</h1>
    <div class="muted">Service: <code>{SERVICE_NAME}</code></div>
    <div class="muted">GUI link: <a id="gui-link" href="#" target="_blank" rel="noopener">Open GUI</a></div>
    <div class="row">
      <button id="refresh">Refresh Status</button>
      <button id="start" class="start">Start Script</button>
      <button id="stop" class="stop">Stop Script</button>
    </div>
    <div class="row"><div id="log">Loading...</div></div>
  </div>
  <script>
    const log = (m) => {{
      const el = document.getElementById("log");
      el.textContent = m;
    }};
    const guiBase = `${{window.location.protocol}}//${{window.location.hostname}}:{GUI_PORT}/`;
    const guiLink = document.getElementById("gui-link");
    guiLink.href = guiBase;
    guiLink.textContent = guiBase;

    async function status() {{
      try {{
        const res = await fetch("/service", {{ cache: "no-store" }});
        const data = await res.json();
        log(`state=${{data.state}}\\nservice=${{data.service}}`);
      }} catch (e) {{
        log("status failed");
      }}
    }}
    async function action(name) {{
      try {{
        const res = await fetch("/service", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ action: name }}),
        }});
        const data = await res.json();
        log(`action=${{name}} ok=${{data.ok}}\\nstate=${{data.state || ""}}\\nmsg=${{data.message || ""}}`);
        setTimeout(status, 1000);
      }} catch (e) {{
        log(`action=${{name}} failed`);
      }}
    }}
    document.getElementById("refresh").addEventListener("click", status);
    document.getElementById("start").addEventListener("click", () => action("start"));
    document.getElementById("stop").addEventListener("click", () => action("stop"));
    status();
    setInterval(status, 3000);
  </script>
</body>
</html>
"""


def run_cmd(action: str):
    cmd = ["sudo", "-n", "/usr/bin/systemctl", action, SERVICE_NAME]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    ok = proc.returncode == 0
    msg = out or err or ""
    if not ok and "password is required" in msg.lower():
        msg = "sudoers NOPASSWD rule missing for systemctl."
    return ok, msg


class Handler(BaseHTTPRequestHandler):
    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, code, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self._set_cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/service"):
            ok, state = run_cmd("is-active")
            self._send_json(200 if ok else 500, {"ok": ok, "state": state, "service": SERVICE_NAME})
            return
        data = HTML.encode("utf-8")
        self.send_response(200)
        self._set_cors()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors()
        self.end_headers()

    def do_POST(self):
        if not self.path.startswith("/service"):
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = {}
        action = str(payload.get("action", "")).strip().lower()
        if action not in {"start", "stop"}:
            self._send_json(400, {"ok": False, "error": "action must be start/stop"})
            return
        ok, msg = run_cmd(action)
        ok_state, state = run_cmd("is-active")
        self._send_json(200 if ok else 500, {
            "ok": ok,
            "action": action,
            "service": SERVICE_NAME,
            "message": msg,
            "state": state if ok_state else "unknown",
        })

    def log_message(self, *_args):
        return


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"NAVISAR controller running at http://{HOST}:{PORT}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
