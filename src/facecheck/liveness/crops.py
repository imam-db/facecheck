"""Runtime port of upstream ``CropImage`` from Silent-Face-Anti-Spoofing.

Source: ``src/generate_patches.py`` in
https://github.com/minivision-ai/Silent-Face-Anti-Spoofing at commit
``b6d5f04ad78778917853b25c778acef6d5626d15``.

Copyright 2020 Minivision. Licensed under the Apache License 2.0. See
``third_party/Silent-Face-Anti-Spoofing.LICENSE`` and the repository-root
``NOTICE``.

Apache License 2.0 section 4(b) modification notice: Modified from upstream
b6d5f04 (Silent-Face ``generate_patches.py``) by adding type hints and the
``crop_face`` wrapper; crop arithmetic unchanged.

This is the runtime copy living under ``facecheck.liveness``. The copy in
``scripts/_crop_image.py`` remains the export/reference copy. The two are
asserted bit-identical in tests. Do not import from ``scripts/`` at runtime.

The arithmetic is kept identical to upstream: the scale is clamped to fit the
frame, the box is shifted inside the frame (never padded), the slice is
inclusive on both ends (``right_bottom + 1``), and ``cv2.resize`` uses its
default ``INTER_LINEAR`` interpolation.

``bbox`` is ``[x, y, w, h]`` in pixels; ``scale`` is the upstream expansion
factor (2.7 for MiniFASNetV2, 4.0 for MiniFASNetV1SE).
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["CropImage", "crop_face"]


class CropImage:
    @staticmethod
    def _get_new_box(
        src_w: int, src_h: int, bbox: list[int], scale: float
    ) -> tuple[int, int, int, int]:
        x = bbox[0]
        y = bbox[1]
        box_w = bbox[2]
        box_h = bbox[3]

        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w / 2 + x, box_h / 2 + y

        left_top_x = center_x - new_width / 2
        left_top_y = center_y - new_height / 2
        right_bottom_x = center_x + new_width / 2
        right_bottom_y = center_y + new_height / 2

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0

        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0

        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1

        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1

        return (
            int(left_top_x),
            int(left_top_y),
            int(right_bottom_x),
            int(right_bottom_y),
        )

    def crop(
        self,
        org_img: np.ndarray,
        bbox: list[int],
        scale: float,
        out_w: int,
        out_h: int,
        crop: bool = True,
    ) -> np.ndarray:
        if not crop:
            dst_img = cv2.resize(org_img, (out_w, out_h))
        else:
            src_h, src_w, _ = np.shape(org_img)
            left_top_x, left_top_y, right_bottom_x, right_bottom_y = self._get_new_box(
                src_w, src_h, bbox, scale
            )

            img = org_img[left_top_y : right_bottom_y + 1, left_top_x : right_bottom_x + 1]
            dst_img = cv2.resize(img, (out_w, out_h))
        return dst_img


def crop_face(
    org_img: np.ndarray,
    bbox_xywh: list[int],
    scale: float,
    out_w: int = 80,
    out_h: int = 80,
) -> np.ndarray:
    """Convenience wrapper around :class:`CropImage` for an 80x80 PAD crop."""

    return CropImage().crop(org_img, bbox_xywh, scale, out_w, out_h)
