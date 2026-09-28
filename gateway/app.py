# =============================================================
# INTEGRATION GATEWAY — single public backend port
#
# This is NEW code, separate from both original projects. It does not
# contain any hate-speech-detection logic of its own; it only forwards
# HTTP requests byte-for-byte to whichever original backend owns the
# requested path, and forwards the response back unchanged.
#
#   /api/*   -> Sinhala backend  (AT final/backend/app.py),   127.0.0.1:5001
#   everything else -> Tamil backend (api/app.py),            127.0.0.1:5000
#
# The split matches the route prefixes each backend already uses on its
# own (Sinhala's routes are all under /api/, Tamil's are not), so no
# route in either backend needed to change.
# =============================================================

from flask import Flask, jsonify, request, Response
import requests

app = Flask(__name__)

TAMIL_BACKEND = "http://127.0.0.1:5000"
SINHALA_BACKEND = "http://127.0.0.1:5001"

# Headers that must not be forwarded as-is between hops (hop-by-hop /
# recalculated by the HTTP layer itself).
_EXCLUDED_HEADERS = {
    "content-length", "transfer-encoding", "connection",
    "host", "content-encoding",
}


def _proxy(target_base: str, path: str):
    target_url = f"{target_base}/{path}"

    forward_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _EXCLUDED_HEADERS
    }

    # If the target backend isn't running (crashed, not started yet, etc.),
    # requests raises here. Left uncaught, Flask's default error handler
    # returns an HTML page — which breaks any JSON-only caller (the
    # Streamlit shell, the Chrome extension) with a confusing
    # "Unexpected token '<'" parse error instead of a readable message.
    try:
        upstream = requests.request(
            method=request.method,
            url=target_url,
            headers=forward_headers,
            params=request.args,
            data=request.get_data(),
            timeout=1800,  # YouTube analysis routes can run long on both backends
            stream=True,
        )
    except requests.exceptions.RequestException as exc:
        response = jsonify({
            "error": f"Could not reach the backend at {target_base}. It may have crashed or not be running. ({exc.__class__.__name__})"
        })
        response.status_code = 502
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    response_headers = [
        (k, v) for k, v in upstream.headers.items()
        if k.lower() not in _EXCLUDED_HEADERS
    ]

    # Chrome's Private Network Access policy requires this header (set to
    # "true") on responses — including preflight OPTIONS — before it lets a
    # request from a public origin, including a browser extension's
    # popup/service worker, reach a loopback address like this gateway.
    # Tamil's own backend already sets it to "true". Sinhala's flask-cors
    # sets it to "false" by default on its own preflight responses, so a
    # simple add-if-missing isn't enough — the value must be overridden
    # outright, uniformly for both, without editing either original backend.
    response_headers = [
        (k, v) for k, v in response_headers
        if k.lower() != "access-control-allow-private-network"
    ]
    response_headers.append(("Access-Control-Allow-Private-Network", "true"))

    return Response(upstream.content, status=upstream.status_code, headers=response_headers)


@app.route(
    "/<path:path>",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
def proxy(path):
    if path.startswith("api/"):
        return _proxy(SINHALA_BACKEND, path)
    return _proxy(TAMIL_BACKEND, path)


@app.route("/", methods=["GET"])
def root():
    # Tamil's own root route ("/") describes its endpoints; forward there too.
    return _proxy(TAMIL_BACKEND, "")


if __name__ == "__main__":
    print("=" * 60)
    print("  Integration Gateway")
    print("  /api/*  -> Sinhala backend (127.0.0.1:5001)")
    print("  /*      -> Tamil backend   (127.0.0.1:5000)")
    print("=" * 60)
    app.run(host="0.0.0.0", port=8000, debug=False)
