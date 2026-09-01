"""Vehicle registration lookup and plate-vehicle verification."""

from .vehicles import (
    FileRegistry,
    RegistryBackend,
    RegistryCheck,
    VehicleRecord,
    VehicleRegistry,
    canonical_class,
)

__all__ = [
    "VehicleRegistry",
    "FileRegistry",
    "RegistryBackend",
    "RegistryCheck",
    "VehicleRecord",
    "canonical_class",
]
