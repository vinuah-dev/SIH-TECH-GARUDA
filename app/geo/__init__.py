"""Where the cameras physically are, so alerts can be shown on a map."""

from .location import CameraLocation, Geofence, InvalidLocation, SiteMap

__all__ = ["CameraLocation", "Geofence", "SiteMap", "InvalidLocation"]
