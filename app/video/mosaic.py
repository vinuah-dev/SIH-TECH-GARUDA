"""Tile a fleet's cameras into one window.

One window per camera does not survive contact with a real post. Four cameras
means four windows to find, drag apart and keep on top, and the operator spends
their attention on window management at exactly the moment they should be
watching a fence. A control room shows a wall, not a pile.

So above one camera the fleet draws a single mosaic. Three things about it are
deliberate:

**Tile order is fixed, by camera id.** Cameras arrive at different rates, and a
layout that reordered itself as frames landed would move CAM-03 under the
operator's cursor mid-glance. The grid is decided by which cameras exist, never
by which one spoke last.

**A camera with no frame yet still gets its tile.** Leaving it out would
renumber every camera after it the moment it connected. An empty tile that says
so is honest and keeps the wall still.

**Aspect ratio is preserved.** A stretched frame is a frame an operator has to
mentally un-stretch before judging whether somebody is running, so tiles are
letterboxed rather than squashed.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# Above this the window stops fitting on a laptop screen, and the whole point
# of the wall is that all of it is visible at once.
MAX_WIDTH = 1600
MAX_HEIGHT = 900

BACKGROUND = (18, 18, 18)
LABEL_BG = (32, 32, 32)
LABEL_FG = (235, 235, 235)
WAITING_FG = (120, 120, 120)


def grid_shape(count: int) -> tuple[int, int]:
    """Rows and columns for `count` tiles, kept as close to square as possible.

    Two cameras go side by side rather than stacked, because camera views are
    themselves wider than they are tall and a 1x2 row wastes less of the screen.
    """
    if count <= 1:
        return 1, 1
    if count == 2:
        return 1, 2
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    return rows, columns


def _fit(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale into the cell without distorting it, padding what is left over."""
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return np.full((height, width, 3), BACKGROUND, np.uint8)

    scale = min(width / w, height / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

    cell = np.full((height, width, 3), BACKGROUND, np.uint8)
    top, left = (height - new_h) // 2, (width - new_w) // 2
    cell[top:top + new_h, left:left + new_w] = resized
    return cell


def _label(cell: np.ndarray, text: str, live: bool) -> None:
    """Name the camera on its own tile, drawn in place."""
    height, width = cell.shape[:2]
    bar = max(18, height // 14)
    overlay = cell.copy()
    cv2.rectangle(overlay, (0, 0), (width, bar), LABEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, cell, 0.25, 0, cell)

    scale = max(0.35, min(0.6, width / 640))
    cv2.putText(cell, text, (6, int(bar * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, LABEL_FG if live else WAITING_FG, 1, cv2.LINE_AA)


def compose(
    frames: dict[str, np.ndarray],
    camera_ids: list[str] | None = None,
    max_width: int = MAX_WIDTH,
    max_height: int = MAX_HEIGHT,
) -> np.ndarray | None:
    """One image holding every camera in the fleet.

    `camera_ids` fixes the layout. Passing it means a camera that has not yet
    produced a frame still occupies its place, so tiles never renumber
    themselves underneath the operator.
    """
    order = sorted(camera_ids if camera_ids is not None else frames)
    if not order:
        return None

    if len(order) == 1:
        # One camera needs no wall. Show it as it is.
        return frames.get(order[0])

    rows, columns = grid_shape(len(order))
    cell_w = max(160, max_width // columns)
    cell_h = max(120, max_height // rows)

    canvas = np.full((cell_h * rows, cell_w * columns, 3), BACKGROUND, np.uint8)
    for index, camera_id in enumerate(order):
        frame = frames.get(camera_id)
        if frame is None:
            cell = np.full((cell_h, cell_w, 3), BACKGROUND, np.uint8)
            text = "waiting for frames"
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(cell, text,
                        ((cell_w - size[0]) // 2, cell_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WAITING_FG, 1, cv2.LINE_AA)
            _label(cell, camera_id, live=False)
        else:
            cell = _fit(frame, cell_w, cell_h)
            _label(cell, camera_id, live=True)

        row, column = divmod(index, columns)
        canvas[row * cell_h:(row + 1) * cell_h,
               column * cell_w:(column + 1) * cell_w] = cell

    return canvas
