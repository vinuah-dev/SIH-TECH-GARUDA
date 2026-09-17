"""What SENTINEL-X is willing to detect, and how it groups it.

YOLO already recognises vehicles; the MVP simply threw them away. This registry
is the mapping back: COCO class ids to the labels an operator reads, grouped
into categories the rest of the system can reason about.

The grouping matters more than the label. A zone rule wants "no vehicles past
this line", not a list of six COCO ids. And several behaviours are meaningful
for one category and nonsense for another - a truck filling the frame is a
truck driving past the camera, not somebody reaching for the lens.
"""

from __future__ import annotations

from dataclasses import dataclass

PERSON = "PERSON"
VEHICLE = "VEHICLE"


@dataclass(frozen=True)
class ObjectClass:
    """One detectable class."""

    coco_id: int
    label: str
    category: str
    # Scales the zone's base risk. A truck through a border fence is a bigger
    # problem than a bicycle, and both differ from a person on foot.
    risk_weight: float = 1.0


REGISTRY: tuple[ObjectClass, ...] = (
    ObjectClass(0, "PERSON", PERSON, 1.00),
    ObjectClass(1, "BICYCLE", VEHICLE, 0.85),
    ObjectClass(3, "MOTORCYCLE", VEHICLE, 1.00),
    ObjectClass(2, "CAR", VEHICLE, 1.10),
    ObjectClass(7, "TRUCK", VEHICLE, 1.20),
    ObjectClass(5, "BUS", VEHICLE, 1.20),
)

BY_ID = {entry.coco_id: entry for entry in REGISTRY}
BY_LABEL = {entry.label: entry for entry in REGISTRY}

CATEGORIES = {
    PERSON: tuple(e.label for e in REGISTRY if e.category == PERSON),
    VEHICLE: tuple(e.label for e in REGISTRY if e.category == VEHICLE),
}

# What --detect accepts: a category name, or any single label.
SELECTORS = {**{name.lower(): labels for name, labels in CATEGORIES.items()},
             **{e.label.lower(): (e.label,) for e in REGISTRY},
             "all": tuple(e.label for e in REGISTRY)}


def resolve(selectors: list[str] | tuple[str, ...]) -> list[str]:
    """Turn CLI selectors such as ['person', 'vehicle'] into concrete labels."""
    labels: list[str] = []
    for selector in selectors:
        key = str(selector).strip().lower()
        if key not in SELECTORS:
            raise ValueError(
                f"unknown object selector {selector!r}; "
                f"expected one of {', '.join(sorted(SELECTORS))}"
            )
        for label in SELECTORS[key]:
            if label not in labels:
                labels.append(label)
    if not labels:
        raise ValueError("no object classes selected")
    return labels


def coco_ids(labels: list[str] | tuple[str, ...]) -> list[int]:
    return [BY_LABEL[label].coco_id for label in labels if label in BY_LABEL]


def category_of(label: str) -> str:
    entry = BY_LABEL.get(label)
    return entry.category if entry else PERSON


def risk_weight(label: str) -> float:
    entry = BY_LABEL.get(label)
    return entry.risk_weight if entry else 1.0


def is_vehicle(label: str) -> bool:
    return category_of(label) == VEHICLE


def is_person(label: str) -> bool:
    return category_of(label) == PERSON
