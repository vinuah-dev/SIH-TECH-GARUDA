"""Camera sources that arrive from the dashboard, and checking one works.

The fleet file accepts more than the dashboard does - a video file path, for
replaying footage. The website does not: it sits on an API with no
authentication, and a field that took a path would let anybody who reaches
the port make the server open any file on the machine. From the website a
source is a webcam number, a camera's network address, or the demo feed, and
nothing else.

Camera addresses often carry a login (``rtsp://admin:pass@192.168.1.64``).
The fleet file has to keep it - the camera will not stream without it - but
nothing the API sends back ever shows the password.
"""

from __future__ import annotations

import base64
import re
import time
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

CAMERA_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}")
WEB_SCHEMES = ("rtsp", "rtsps", "http", "https")
MAX_WEBCAM = 9
# Query parameters some cameras take a login in (e.g. /video.cgi?user=admin&pwd=...).
SECRET_PARAMS = {"password", "pass", "passwd", "pwd", "token", "key", "auth"}


class ProbeFailed(Exception):
    """A source could not be opened, with a reason an operator can act on."""


def check_camera_id(value: Any) -> str:
    text = str(value or "").strip()
    if not CAMERA_ID.fullmatch(text):
        raise ValueError(
            "a camera name is 1-32 letters, digits, - or _, starting with a letter "
            "or digit - for example CAM-03"
        )
    return text


def webcam_index(source: str) -> int | None:
    """The device number of a webcam source, or None for anything else."""
    text = source.strip().lower()
    if text == "webcam":
        return 0
    if text.startswith("webcam:") and text[7:].isdigit():
        return int(text[7:])
    if text.isdigit():
        return int(text)
    return None


def normalize_source(value: Any) -> str:
    """A source the dashboard may add, in the form the fleet file stores."""
    text = str(value or "").strip()
    if not text:
        raise ValueError("no camera source given")
    if len(text) > 400 or any(ch.isspace() for ch in text):
        raise ValueError("a camera address cannot contain spaces or be that long")
    if text.lower() == "synthetic":
        return "synthetic"

    if re.fullmatch(r"(?:webcam:)?\d{1,3}", text, re.IGNORECASE):
        index = webcam_index(text)
        if index is None or index > MAX_WEBCAM:
            raise ValueError(f"a webcam number is 0 to {MAX_WEBCAM}")
        return f"webcam:{index}"

    parts = urlsplit(text)
    if parts.scheme.lower() not in WEB_SCHEMES:
        raise ValueError(
            "a camera source is a webcam number, an rtsp:// or http:// camera "
            "address, or the demo feed - files and other links are not accepted here"
        )
    try:
        host, _ = parts.hostname, parts.port
    except ValueError as exc:
        raise ValueError(f"that address has a bad port: {exc}") from exc
    if not host:
        raise ValueError("that address has no camera IP or hostname in it")
    return text


def redact_source(source: str) -> str:
    """The source with any password replaced, safe to show or log."""
    try:
        parts = urlsplit(source)
    except ValueError:
        return source
    if parts.scheme.lower() not in WEB_SCHEMES:
        return source
    netloc = parts.netloc
    if parts.password is not None:
        netloc = f"{parts.username or ''}:***@{netloc.rpartition('@')[2]}"
    query = parts.query
    if query:
        pairs = parse_qsl(query, keep_blank_values=True)
        if any(k.lower() in SECRET_PARAMS for k, _ in pairs):
            query = urlencode([(k, "***" if k.lower() in SECRET_PARAMS else v)
                               for k, v in pairs], safe="*")
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def describe_source(source: str) -> str:
    """A short label for a source: what kind of camera, and where."""
    text = source.strip()
    if text.lower() == "synthetic":
        return "Demo feed"
    index = webcam_index(text)
    if index is not None:
        return f"Webcam {index}"
    parts = urlsplit(text)
    if parts.scheme.lower() in WEB_SCHEMES:
        try:
            where = parts.hostname or ""
            if parts.port:
                where += f":{parts.port}"
        except ValueError:
            where = ""
        return f"{parts.scheme.upper()} · {where}" if where else parts.scheme.upper()
    return f"Video file · {text.replace(chr(92), '/').rpartition('/')[2]}"


# ------------------------------------------------------------------ probing


def _open_webcam(index: int):
    import cv2

    return cv2.VideoCapture(index)


def _read_one(capture, where: str, read_timeout: float):
    deadline = time.monotonic() + read_timeout
    while time.monotonic() < deadline:
        ok, image = capture.read()
        if ok and image is not None:
            return image
        time.sleep(0.05)
    raise ProbeFailed(f"{where} opened but sent no picture within {read_timeout:.0f}s")


def grab_frame(source: str, open_timeout: float = 4.0, read_timeout: float = 8.0,
               opener: Callable[[int], Any] | None = None):
    """Open a source, read one frame and close it again.

    Raises ProbeFailed saying which part failed - nothing answering, a
    login refused, or a stream that opens and sends nothing need different
    fixes, and "could not connect" alone helps with none of them.
    """
    if source == "synthetic":
        from .video.sources import SyntheticSource

        demo = SyntheticSource(frames_to_emit=1)
        try:
            return next(iter(demo.frames())).image
        finally:
            demo.release()

    index = webcam_index(source)
    if index is not None:
        capture = (opener or _open_webcam)(index)
        try:
            if not capture.isOpened():
                raise ProbeFailed(
                    f"no camera at webcam {index} - or another program is using it"
                )
            return _read_one(capture, f"webcam {index}", read_timeout)
        finally:
            capture.release()

    from .video.stream import DEFAULT_PORTS, open_network_capture, tcp_reachable

    parts = urlsplit(source)
    port = parts.port or DEFAULT_PORTS.get(parts.scheme.lower(), 554)
    where = f"{parts.hostname}:{port}"
    if not tcp_reachable(source, timeout=open_timeout):
        raise ProbeFailed(
            f"nothing answered at {where} - check the IP address and port, that the "
            f"camera is powered, and that this computer is on the same network"
        )
    capture = open_network_capture(source)
    try:
        if not capture.isOpened():
            raise ProbeFailed(
                f"{where} answered but would not send video - check the username, "
                f"password, channel and stream path"
            )
        return _read_one(capture, where, read_timeout)
    finally:
        capture.release()


def preview_data_url(image, width: int = 480) -> str:
    """A small JPEG of a frame, inline, for the dashboard to show."""
    import cv2

    height, current = image.shape[:2]
    if current > width:
        image = cv2.resize(image, (width, int(height * width / current)))
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def find_webcams(busy: dict[int, str], indices=range(4),
                 opener: Callable[[int], Any] | None = None) -> list[dict]:
    """Which webcam numbers have a camera behind them.

    A webcam a running camera already holds is reported as in use and never
    opened: on Windows a second open can fail, or steal the feed.
    """
    found = []
    for index in indices:
        if index in busy:
            found.append({"index": index, "in_use_by": busy[index]})
            continue
        capture = (opener or _open_webcam)(index)
        try:
            if not capture.isOpened():
                continue
            ok, image = capture.read()
            if ok and image is not None:
                height, width = image.shape[:2]
                found.append({"index": index, "width": width, "height": height})
        finally:
            capture.release()
    return found
