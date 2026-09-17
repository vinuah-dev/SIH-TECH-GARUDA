"""Where the cameras physically are, so alerts can be shown on a map."""

from .location import (DEFAULT_BASEMAPS, Basemap, CameraLocation, Geofence,
                       InvalidLocation, SiteMap)

__all__ = ["Basemap", "CameraLocation", "DEFAULT_BASEMAPS", "Geofence", "SiteMap",
           "InvalidLocation"]
