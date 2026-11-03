"""Unit tests for the PAD evaluation harness.

The tests generate solid-colour PNGs and use fake embedder/liveness backends, so
they need no ONNX models and no real photographs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from eval.pad_eval import (
    ATTACK_CLASSES,
    BONA_FIDE,
    PRESENTATION_CLASSES,
    ManifestError,
    downscale_max_side,
    load_image_bgr,
    parse_manifest,
    run_evaluation,
)

from facecheck.embeddings import DEFAULT_MAX_PIXELS, EmbeddingResult, decode_image_bgr
from facecheck.liveness import LivenessResult
from facecheck.quality import NO_FACE, QualityError

FIXTURES = Path(__file__).parent / "fixtures"

BONA_OK = (10, 0, 0)
BONA_REJECT = (20, 0, 0)
BONA_DETECTION_FAIL = (30, 0, 0)
PRINT_ACCEPT = (40, 0, 0)
PRINT_REJECT = (50, 0, 0)
PHONE_REJECT = (60, 0, 0)
LAPTOP_ACCEPT = (70, 0, 0)
UNCONSENTED = (80, 0, 0)

HEADER = [
    "path",
    "subject_id",
    "session_id",
    "class",
    "attack_instrument",
    "capture_device",
    "condition",
    "consent",
    "notes",
]


def _top_left_color(image: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(value) for value in image[0, 0])


class ColorEmbedder:
    """Fake embedder whose result is derived from the image's top-left colour."""

    model_version = "fake_embedder_v1"
    embedding_dim = 3

    def __init__(self, fail_colors: set[tuple[int, int, int]] | None = None) -> None:
        self.fail_colors = fail_colors or set()

    def detect_and_embed(self, bgr_image: np.ndarray) -> EmbeddingResult:
        if _top_left_color(bgr_image) in self.fail_colors:
            raise QualityError(NO_FACE, "synthetic detection failure")
        height, width = bgr_image.shape[:2]
        return EmbeddingResult(
            embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            bbox=(0.0, 0.0, float(width), float(height)),
            landmarks=None,
            model_version=self.model_version,
        )


class ColorLivenessDetector:
    """Fake PAD detector that returns a scripted score per solid colour."""

    model_version = "fake_pad_v1"

    def __init__(self, scores: dict[tuple[int, int, int], float]) -> None:
        self.scores = scores

    def score(self, bgr_image: np.ndarray, bbox_xywh: list[int]) -> LivenessResult:
        value = float(self.scores[_top_left_color(bgr_image)])
        return LivenessResult(
            score=value,
            per_model={"fake_pad.onnx": value},
            model_version=self.model_version,
        )

    def score_face(
        self, full_res_image: np.ndarray, embedding_result: EmbeddingResult
    ) -> LivenessResult:
        return self.score(full_res_image, embedding_result.full_res_bbox_xywh)


def _write_png(
    path: Path, color: tuple[int, int, int], size: tuple[int, int] = (64, 48)
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    image = np.full((height, width, 3), color, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_manifest(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)


@pytest.fixture
def capture_setup(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "captures"
    rows = [
        [
            "presentations/subj_01/s01_bona_ok.png",
            "subj_01",
            "s01",
            "bona_fide",
            "",
            "phone_front",
            "daylight",
            "true",
            "",
        ],
        [
            "presentations/subj_01/s01_bona_reject.png",
            "subj_01",
            "s01",
            "bona_fide",
            "",
            "phone_front",
            "daylight",
            "true",
            "",
        ],
        [
            "presentations/subj_02/s02_bona_fail.png",
            "subj_02",
            "s02",
            "bona_fide",
            "",
            "phone_front",
            "dim",
            "true",
            "",
        ],
        [
            "presentations/subj_01/s01_print_accept.png",
            "subj_01",
            "s01",
            "print",
            "glossy_print",
            "phone_front",
            "daylight",
            "true",
            "",
        ],
        [
            "presentations/subj_02/s02_print_reject.png",
            "subj_02",
            "s02",
            "print",
            "glossy_print",
            "phone_front",
            "daylight",
            "true",
            "",
        ],
        [
            "presentations/subj_01/s01_phone_reject.png",
            "subj_01",
            "s01",
            "phone_screen",
            "phone_oled",
            "laptop_webcam",
            "dim",
            "true",
            "",
        ],
        [
            "presentations/subj_02/s02_laptop_accept.png",
            "subj_02",
            "s02",
            "laptop_screen",
            "laptop_lcd",
            "phone_front",
            "backlit",
            "true",
            "",
        ],
        [
            "presentations/subj_03/s03_missing.png",
            "subj_03",
            "s03",
            "print",
            "glossy_print",
            "phone_front",
            "daylight",
            "true",
            "intentionally missing file",
        ],
        [
            "presentations/subj_03/s03_unconsented.png",
            "subj_03",
            "s03",
            "bona_fide",
            "",
            "phone_front",
            "daylight",
            "false",
            "",
        ],
    ]
    _write_manifest(root / "manifest.csv", rows)

    for relative, color in (
        ("presentations/subj_01/s01_bona_ok.png", BONA_OK),
        ("presentations/subj_01/s01_bona_reject.png", BONA_REJECT),
        ("presentations/subj_02/s02_bona_fail.png", BONA_DETECTION_FAIL),
        ("presentations/subj_01/s01_print_accept.png", PRINT_ACCEPT),
        ("presentations/subj_02/s02_print_reject.png", PRINT_REJECT),
        ("presentations/subj_01/s01_phone_reject.png", PHONE_REJECT),
        ("presentations/subj_02/s02_laptop_accept.png", LAPTOP_ACCEPT),
        ("presentations/subj_03/s03_unconsented.png", UNCONSENTED),
    ):
        _write_png(root / relative, color)

    return root, root / "manifest.csv"


def _run(capture_setup: tuple[Path, Path], **overrides: object) -> dict:
    root, manifest = capture_setup
    production_scores = {
        BONA_OK: 0.95,
        BONA_REJECT: 0.40,
        PRINT_ACCEPT: 0.95,
        PRINT_REJECT: 0.20,
        PHONE_REJECT: 0.30,
        LAPTOP_ACCEPT: 0.95,
    }
    arguments: dict[str, object] = {
        "embedder": ColorEmbedder(fail_colors={BONA_DETECTION_FAIL}),
        "detector": ColorLivenessDetector(production_scores),
        "diagnostic_embedder": ColorEmbedder(),
        "diagnostic_detector": ColorLivenessDetector({BONA_DETECTION_FAIL: 0.60}),
        "production_threshold": 0.9,
        "diagnostic_threshold": 0.5,
        "bpcer_target": 0.05,
        "apcer_target": 0.10,
    }
    arguments.update(overrides)
    return run_evaluation(manifest, root, **arguments)


def test_parse_example_fixture_without_images() -> None:
    manifest = parse_manifest(FIXTURES / "pad_manifest_example.csv")
    assert manifest.total_rows == 4
    assert manifest.skipped_unconsented == 0
    assert {row.presentation_class for row in manifest.rows} == set(PRESENTATION_CLASSES)
    assert {row.subject_id for row in manifest.rows} == {"subj_01", "subj_02"}
    assert {row.attack_instrument for row in manifest.rows if row.attack_instrument} == {
        "glossy_print",
        "phone_oled",
        "laptop_lcd",
    }


def test_report_aggregates_consent_detection_and_rates(
    capture_setup: tuple[Path, Path],
) -> None:
    report = _run(capture_setup)

    assert report["schema_version"] == 1
    assert report["manifest"]["data_rows"] == 9
    assert report["manifest"]["included_presentations"] == 8
    assert report["manifest"]["skipped_unconsented"] == 1
    assert report["manifest"]["subjects"] == 3
    assert report["manifest"]["sessions"] == 3
    assert report["manifest"]["class_counts"] == {
        "bona_fide": 3,
        "print": 3,
        "phone_screen": 1,
        "laptop_screen": 1,
    }

    rates = report["rates"]
    assert rates["threshold"] == pytest.approx(0.9)
    assert rates["bona_presentations"] == 2
    assert rates["bona_rejected"] == 1
    assert rates["bpcer"] == pytest.approx(0.5)
    assert rates["attack_presentations"] == 4
    assert rates["attack_accepted"] == 2
    assert rates["apcer"] == pytest.approx(0.5)
    assert rates["apcer_per_class"]["print"]["apcer"] == pytest.approx(0.5)
    assert rates["apcer_per_class"]["phone_screen"]["apcer"] == pytest.approx(0.0)
    assert rates["apcer_per_class"]["laptop_screen"]["apcer"] == pytest.approx(1.0)
    assert rates["apcer_worst_class"] == "laptop_screen"
    assert rates["apcer_worst"] == pytest.approx(1.0)
    assert rates["iapmr"] is None
    assert rates["iapmr_status"] == "not_measured_without_matcher"

    summary = report["presentation_summary"]
    assert summary["bona_fide"]["presentations"] == 3
    assert summary["bona_fide"]["scored"] == 2
    assert summary["bona_fide"]["detection_failures"] == 1
    assert summary["bona_fide"]["detection_failure_rate"] == pytest.approx(1 / 3)
    assert summary["print"]["presentations"] == 3
    assert summary["print"]["scored"] == 2
    assert summary["print"]["load_failures"] == 1
    assert summary["print"]["detection_failures"] == 0


def test_diagnostic_scores_are_separate_from_production_rates(
    capture_setup: tuple[Path, Path],
) -> None:
    report = _run(capture_setup)
    diagnostic = report["diagnostic"]
    assert diagnostic["enabled"] is True
    assert diagnostic["production_detector_threshold"] == pytest.approx(0.9)
    assert diagnostic["diagnostic_detector_threshold"] == pytest.approx(0.5)

    bona_block = diagnostic["per_class"]["bona_fide"]
    assert bona_block["failed_production_detection"] == 1
    assert bona_block["recovered_with_diagnostic_threshold"] == 1
    assert bona_block["still_failed"] == 0
    assert bona_block["scores"]["n"] == 1
    assert bona_block["scores"]["mean"] == pytest.approx(0.60)

    # The recovered row is not a production score, so BPCER still sees only 2.
    assert report["rates"]["bona_presentations"] == 2


def test_operating_points_and_sweep(capture_setup: tuple[Path, Path]) -> None:
    report = _run(capture_setup)
    operating = report["operating_points"]

    assert operating["bpcer_target_status"] == "selected"
    assert operating["bpcer_target_threshold"] == pytest.approx(0.4)
    assert operating["bpcer_target_rates"]["bpcer"] == pytest.approx(0.0)
    assert "insufficient_attack_scores" in operating["apcer_target_status"]
    assert operating["apcer_target_rates"] is None

    assert report["threshold_sweep"]
    thresholds = [point["threshold"] for point in report["threshold_sweep"]]
    assert thresholds == sorted(thresholds)
    assert 0.9 in thresholds
    assert report["latency_ms"]["full_pipeline"]["n"] == 6


def test_reports_contain_only_aggregates(capture_setup: tuple[Path, Path]) -> None:
    report = _run(capture_setup)
    rendered = json.dumps(report)

    assert "subj_01" not in rendered
    assert "presentations/" not in rendered
    assert ".png" not in rendered
    assert str(capture_setup[0]) not in rendered
    assert "s01_bona_ok" not in rendered


def test_max_side_downscales_presentations(capture_setup: tuple[Path, Path]) -> None:
    report = _run(capture_setup, max_side=20)
    assert report["resolution"]["max_side_setting"] == 20
    assert report["resolution"]["loaded_presentations"] == 7
    assert report["resolution"]["downscaled_presentations"] == 7
    assert report["resolution"]["max_used_side_px"] == 20


def test_downscale_helpers() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    same, scale = downscale_max_side(image, 0)
    assert same is image
    assert scale == 1.0

    same, scale = downscale_max_side(image, 500)
    assert same is image
    assert scale == 1.0

    resized, scale = downscale_max_side(image, 50)
    assert resized.shape == (25, 50, 3)
    assert scale == pytest.approx(0.25)


def test_load_image_bgr_bypasses_the_api_pixel_limit(tmp_path: Path) -> None:
    width = height = 5001
    assert width * height > DEFAULT_MAX_PIXELS

    path = tmp_path / "large-phone-photo.png"
    image = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    del image

    with pytest.raises(QualityError, match="limit"):
        decode_image_bgr(path.read_bytes())

    loaded = load_image_bgr(path)
    assert loaded.shape == (height, width, 3)


def test_parse_manifest_requires_columns(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text("path,subject_id,session_id,class\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="missing column"):
        parse_manifest(path)


def test_parse_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    _write_manifest(
        path,
        [["../outside.png", "subj_01", "s01", "bona_fide", "", "", "", "true", ""]],
    )
    with pytest.raises(ManifestError, match="relative"):
        parse_manifest(path)


def test_parse_manifest_rejects_unknown_class(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    _write_manifest(
        path,
        [["a.png", "subj_01", "s01", "deepfake", "", "", "", "true", ""]],
    )
    with pytest.raises(ManifestError, match="unknown class"):
        parse_manifest(path)


def test_parse_manifest_rejects_duplicate_paths(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    _write_manifest(
        path,
        [
            ["a.png", "subj_01", "s01", "bona_fide", "", "", "", "true", ""],
            ["a.png", "subj_02", "s02", "print", "glossy_print", "", "", "true", ""],
        ],
    )
    with pytest.raises(ManifestError, match="duplicate path"):
        parse_manifest(path)


def test_non_true_consent_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    _write_manifest(
        path,
        [
            ["a.png", "subj_01", "s01", "bona_fide", "", "", "", "false", ""],
            ["b.png", "subj_01", "s01", "bona_fide", "", "", "", "no", ""],
            ["c.png", "subj_01", "s01", "bona_fide", "", "", "", "", ""],
            ["d.png", "subj_01", "s01", "print", "glossy_print", "", "", "TRUE", ""],
        ],
    )
    manifest = parse_manifest(path)
    assert manifest.total_rows == 4
    assert manifest.skipped_unconsented == 3
    assert len(manifest.rows) == 1
    assert manifest.rows[0].presentation_class == "print"


def test_run_evaluation_rejects_an_empty_usable_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    _write_manifest(
        path,
        [["a.png", "subj_01", "s01", "bona_fide", "", "", "", "false", ""]],
    )
    with pytest.raises(ManifestError, match="no usable presentations"):
        run_evaluation(
            path,
            tmp_path,
            embedder=ColorEmbedder(),
            detector=ColorLivenessDetector({}),
        )


def test_attack_classes_are_the_non_bona_classes() -> None:
    assert BONA_FIDE not in ATTACK_CLASSES
    assert set(PRESENTATION_CLASSES) == {BONA_FIDE, *ATTACK_CLASSES}
