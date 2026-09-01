"""IBVAP command-line entrypoint.

Examples
--------
    python main.py --source synthetic --detector sim
    python main.py --source data/samples/border.mp4 --view
    python main.py --source webcam:0 --camera CAM-02
    python main.py --source rtsp://user:pass@10.0.0.5:554/stream1 --camera CAM-03
    python main.py --source data/samples/person.jpg
"""

from __future__ import annotations

import argparse
import sys

from app import __version__, ui
from app.config import SurveillanceConfig
from app.pipeline import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ibvap",
        description="IBVAP - Intelligent Border Video Analytics Platform (MVP)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"IBVAP {__version__}")

    src = parser.add_argument_group("input")
    src.add_argument(
        "--source",
        default="synthetic",
        help="image path, video path, 'webcam[:index]', 'rtsp://...', or 'synthetic'",
    )
    src.add_argument("--camera", dest="camera_id", default="CAM-01", help="camera identifier")
    src.add_argument(
        "--cameras",
        metavar="PATH",
        default=None,
        help="run every camera in this file in one process, sharing one model",
    )
    src.add_argument(
        "--stride", type=int, default=1,
        help="analyse every Nth frame - covers less footage, does NOT raise fps",
    )
    src.add_argument("--limit", type=int, default=None, help="stop after N processed frames")
    src.add_argument("--loop", action="store_true", help="replay a video file continuously")
    src.add_argument(
        "--reconnect-attempts",
        type=int,
        default=0,
        help="give up on a dropped network stream after N retries (0 = never give up)",
    )
    src.add_argument("--width", type=int, default=960, help="downscale frames to this width")
    src.add_argument(
        "--synthetic-frames", type=int, default=60, help="frames for the synthetic source"
    )
    src.add_argument(
        "--tamper",
        action="store_true",
        help="simulated person walks up to the camera instead of across the scene",
    )
    src.add_argument(
        "--linger",
        action="store_true",
        help="simulated walker stops inside the zone instead of walking through",
    )

    det = parser.add_argument_group("detection")
    det.add_argument(
        "--detector",
        choices=["yolo", "sim"],
        default="yolo",
        help="'yolo' runs real inference; 'sim' replays a scripted walker",
    )
    det.add_argument("--weights", default="models/yolov8n.pt", help="YOLO weights file")
    det.add_argument("--confidence", type=float, default=0.40, help="minimum detection confidence")
    det.add_argument(
        "--detect",
        nargs="+",
        default=["person"],
        metavar="KIND",
        help="what to watch: person, vehicle, all, or a label such as truck",
    )
    det.add_argument(
        "--imgsz", type=int, default=512,
        help="model input size; smaller is faster but sees smaller people less far",
    )
    det.add_argument("--device", default="cpu", help="torch device, e.g. cpu or 0")
    det.add_argument(
        "--tracker",
        choices=["iou", "bytetrack"],
        default="iou",
        help="'iou' is built in; 'bytetrack' uses Ultralytics (pulls in lap)",
    )
    det.add_argument(
        "--faces",
        dest="faces_path",
        default=None,
        metavar="PATH",
        help="roster of people this post expects to see; recognising them lowers risk",
    )
    det.add_argument(
        "--registry",
        default=None,
        metavar="PATH",
        help="vehicle registration extract to verify read plates against",
    )
    det.add_argument(
        "--no-anpr", action="store_true", help="do not read number plates off vehicles"
    )
    det.add_argument(
        "--ocr",
        dest="ocr_backend",
        choices=["paddle-rec", "paddle", "easyocr"],
        default=None,
        help="which recogniser reads plate crops; the default picks the best "
             "that loads (paddle-rec: 73.2%% of plates exact, easyocr: 34.5%%)",
    )
    det.add_argument(
        "--no-plate-rescue",
        action="store_true",
        help="do not re-search a vehicle close-up when the frame-wide plate "
             "pass misses it (faster, but finds far fewer plates)",
    )
    det.add_argument(
        "--no-plate-log",
        action="store_true",
        help="do not log every vehicle that passes (only those that alert)",
    )
    det.add_argument(
        "--plate-log-window",
        type=float,
        default=10.0,
        help="seconds before the same vehicle counts as a second crossing",
    )
    det.add_argument(
        "--anpr-thread",
        action="store_true",
        help="read plates on a background thread (helps only when detection is on a GPU)",
    )
    det.add_argument(
        "--no-reid",
        action="store_true",
        help="do not recover a track that was occluded (appearance re-identification)",
    )

    zon = parser.add_argument_group("zones and alerting")
    zon.add_argument("--zones", dest="zones_path", default="config/zones.json")
    zon.add_argument(
        "--alert-on",
        nargs="+",
        default=["RESTRICTED"],
        metavar="KIND",
        help="zone kinds that raise alerts; others are observed only",
    )
    zon.add_argument(
        "--confirm-frames",
        type=int,
        default=3,
        help="consecutive in-zone frames required before alerting",
    )
    zon.add_argument(
        "--cooldown", type=float, default=30.0, help="seconds before the same track re-alerts"
    )

    ctx = parser.add_argument_group("context")
    ctx.add_argument("--night-start", type=int, default=18, help="hour the night window opens")
    ctx.add_argument("--night-end", type=int, default=6, help="hour the night window closes")
    night = ctx.add_mutually_exclusive_group()
    night.add_argument(
        "--force-night", action="store_true", help="treat the scene as night regardless of clock"
    )
    night.add_argument("--force-day", action="store_true", help="treat the scene as day")

    out = parser.add_argument_group("output")
    out.add_argument("--view", action="store_true", help="open an annotated live view window")
    out.add_argument("--verbose", action="store_true", help="log every detection every frame")
    out.add_argument("--no-evidence", action="store_true", help="do not save evidence snapshots")
    out.add_argument("--no-clips", action="store_true", help="do not save evidence video clips")
    out.add_argument(
        "--clip-seconds",
        type=float,
        default=4.0,
        help="seconds of footage kept either side of an alert",
    )
    out.add_argument("--no-color", action="store_true", help="disable ANSI colour output")
    out.add_argument(
        "--db",
        nargs="?",
        const="data/ibvap.db",
        default=None,
        metavar="PATH",
        help="also record alerts in a queryable SQLite store",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> SurveillanceConfig:
    force_night = True if args.force_night else (False if args.force_day else None)
    return SurveillanceConfig(
        camera_id=args.camera_id,
        source=args.source,
        stride=max(1, args.stride),
        limit=args.limit,
        loop=args.loop,
        max_width=args.width if args.width and args.width > 0 else None,
        synthetic_frames=args.synthetic_frames,
        reconnect_attempts=max(0, args.reconnect_attempts),
        sim_walk_fraction=0.45 if args.linger else 1.0,
        sim_approach_camera=0.90 if args.tamper else 0.0,
        detector=args.detector,
        weights=args.weights,
        confidence=args.confidence,
        detect=tuple(args.detect),
        anpr=not args.no_anpr,
        anpr_background=args.anpr_thread,
        ocr_backend=args.ocr_backend,
        plate_rescue=not args.no_plate_rescue,
        plate_log=not args.no_plate_log,
        plate_log_window=args.plate_log_window,
        registry_path=args.registry,
        faces_path=args.faces_path,
        imgsz=args.imgsz,
        device=args.device,
        tracker=args.tracker,
        reid=not args.no_reid,
        zones_path=args.zones_path,
        alert_kinds=frozenset(k.upper() for k in args.alert_on),
        confirm_frames=args.confirm_frames,
        cooldown_seconds=args.cooldown,
        night_start=args.night_start,
        night_end=args.night_end,
        force_night=force_night,
        view=args.view,
        verbose=args.verbose,
        save_evidence=not args.no_evidence,
        save_clips=not args.no_clips,
        clip_pre_seconds=args.clip_seconds,
        clip_post_seconds=args.clip_seconds,
        color=not args.no_color,
        db_path=args.db,
    )


def run_fleet(args: argparse.Namespace, config: SurveillanceConfig) -> int:
    """Multi-camera mode: one process, one model, many feeds."""
    from app.runner import MultiCameraRunner, load_camera_specs
    from app.store import EventStore

    specs = load_camera_specs(args.cameras)
    store = EventStore(config.db_path) if config.db_path else None
    try:
        workers = MultiCameraRunner(specs, config, store=store).run()
    finally:
        if store is not None:
            store.close()
    return 1 if any(w.error for w in workers) else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = config_from_args(args)
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    ui.init(use_color=not args.no_color)
    try:
        if args.cameras:
            return run_fleet(args, config)
        stats = run(config)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        ui.error(str(exc))
        return 1
    return 0 if stats.frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
