"""Map face boxes between detection and full-resolution coordinates."""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["map_detection_box_to_full_res"]


def map_detection_box_to_full_res(
    bbox_xywh: Sequence[float], detection_scale: float
) -> list[int]:
    """Map a downscaled-detection box back to full-resolution pixels.

    ``detection_scale`` is ``downscaled_width / original_width``. Each of
    ``x, y, w, h`` is divided by it and truncated with ``int(...)``, matching
    the upstream integer box convention rather than rounding. The same
    isotropic scale is applied to y/h because downscaling preserves aspect
    ratio up to sub-pixel rounding.

    When ``detection_scale == 1.0`` the division is a no-op but ``int(...)``
    truncation is still applied to every component.
    """

    if detection_scale <= 0:
        raise ValueError("detection_scale must be positive")
    return [int(value / detection_scale) for value in bbox_xywh]
