"""SENTINEL-X server entrypoint.

Runs the camera fleet and serves the dashboard and API:

    python serve.py
    python serve.py --cameras config/cameras.json --port 8000
    python serve.py --source synthetic --detector sim --force-night
"""

from __future__ import annotations

import argparse
import sys

from app import __version__, ui
from app.config import SurveillanceConfig
from app.runner import CameraSpec, load_camera_specs, load_site_map


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinelx-serve",
        description="SENTINEL-X - dashboard and API over the camera fleet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"SENTINEL-X {__version__}")

    net = parser.add_argument_group("server")
    net.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 to expose)")
    net.add_argument("--port", type=int, default=8000)
    net.add_argument("--db", default="data/sentinelx.db", help="event store to read and write")

    src = parser.add_argument_group("cameras")
    src.add_argument("--cameras", default="config/cameras.json", help="fleet configuration")
    src.add_argument(
        "--source",
        default=None,
        help="run a single source instead of the fleet file",
    )
    src.add_argument("--camera", dest="camera_id", default="CAM-01")
    src.add_argument("--zones", dest="zones_path", default="config/zones.json")
    src.add_argument("--detector", choices=["yolo", "sim"], default="yolo")
    src.add_argument("--confidence", type=float, default=0.40)
    src.add_argument("--synthetic-frames", type=int, default=600)

    ctx = parser.add_argument_group("context")
    night = ctx.add_mutually_exclusive_group()
    night.add_argument("--force-night", action="store_true")
    night.add_argument("--force-day", action="store_true")

    parser.add_argument(
        "--view",
        action="store_true",
        help="also open an annotated OpenCV window per camera",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="keep the per-detection console log"
    )
    parser.add_argument("--no-color", action="store_true")
    return parser


def exposed(host: str) -> bool:
    """Is this bind address reachable from outside this machine?

    Checked by exclusion rather than by listing the wildcards, because a LAN
    address like 192.168.1.7 exposes the API just as completely as 0.0.0.0 does
    and would slip past a check that only looked for the obvious two.
    """
    return host not in ("127.0.0.1", "localhost", "::1")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ui.init(use_color=not args.no_color)

    force_night = True if args.force_night else (False if args.force_day else None)
    config = SurveillanceConfig(
        camera_id=args.camera_id,
        source=args.source or "synthetic",
        zones_path=args.zones_path,
        detector=args.detector,
        confidence=args.confidence,
        synthetic_frames=args.synthetic_frames,
        force_night=force_night,
        # The console belongs to the server log; the dashboard is the operator
        # view. --verbose brings the per-detection log back.
        quiet=not args.verbose,
        view=args.view,
        print_summary=False,
        color=not args.no_color,
    )

    try:
        if args.source:
            specs = [CameraSpec(camera_id=args.camera_id, source=args.source)]
        else:
            specs = load_camera_specs(args.cameras)
    except (FileNotFoundError, ValueError) as exc:
        ui.error(str(exc))
        return 1

    from app.server import create_app

    # Geofences live in the same file as the cameras, so the map has areas to
    # draw without a second file to keep in step. A single --source run has no
    # fleet file and therefore no map, which is correct rather than an omission.
    fences = ()
    basemaps = None                       # the public defaults, unless the file says
    if not args.source:
        try:
            site = load_site_map(args.cameras)
            fences, basemaps = site.geofences, site.basemaps
        except (FileNotFoundError, ValueError) as exc:
            ui.warn(f"map areas not loaded: {exc}")

    # Cameras placed on the dashboard map are saved back into this same file.
    # A --source run has none, and the map says so instead of pretending to save.
    app = create_app(specs, config, db_path=args.db, geofences=fences,
                     basemaps=basemaps,
                     fleet_path=None if args.source else args.cameras)

    ui.banner()
    ui.info(f"Cameras: {', '.join(s.camera_id for s in specs)}")
    ui.ok(f"Dashboard: http://{args.host}:{args.port}")
    if exposed(args.host):
        # There is no authentication on this API - deliberately, and it is
        # documented as such. But the difference between "no auth on loopback"
        # and "no auth on the network" is the difference between a development
        # convenience and handing every alert, every event and every evidence
        # path to anyone who can reach the port. That is worth one loud line.
        ui.warn(
            f"Bound to {args.host}: this API has NO authentication, so anyone "
            f"who can reach port {args.port} can read every alert, every event "
            f"and the paths to the evidence files. Put it behind a VPN or a "
            f"reverse proxy that authenticates, or bind 127.0.0.1 instead."
        )
    ui.info("Press Ctrl+C to stop.")
    print()

    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    )

    if not args.view:
        server.run()
        return 0

    # OpenCV's GUI only works from the main thread, and uvicorn.run() would
    # occupy it. So the web server moves to a background thread and the main
    # thread is left free to pump the camera windows.
    import threading

    web = threading.Thread(target=server.run, daemon=True, name="sentinelx-web")
    web.start()
    try:
        service = app.state.service
        # Wait for the fleet to be *started*, not merely constructed.
        while not service.ready.wait(timeout=0.2):
            if not web.is_alive():
                return 1
        if service.runner is not None and service.runner.workers:
            service.runner.wait(poll=0.03)
        else:
            ui.warn("No camera window to show - serving the dashboard only.")
            web.join()
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        web.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
