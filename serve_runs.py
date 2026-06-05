"""Static server for runs/ with no-cache headers.

Reports are regenerated in place after every eval; without Cache-Control
browsers apply heuristic caching and serve stale copies (missing the run
picker, old numbers). no-store forces a fresh fetch every time.
"""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
PORT = 8899


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    # Loopback only — viewers reach this via SSH/editor port-forward, so there
    # is no reason to expose the run data on all interfaces.
    ThreadingHTTPServer(
        ("127.0.0.1", PORT), lambda *a, **kw: NoCacheHandler(*a, directory=str(RUNS), **kw)
    ).serve_forever()
