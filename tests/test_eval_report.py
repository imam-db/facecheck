"""Tests for report hygiene and decode-inclusive latency helpers."""

from __future__ import annotations

from pathlib import Path

from eval.run_lfw import REPO_ROOT, _encode_jpeg, _timed_decode_embed, display_path

from tests.conftest import FakeEmbedder, image_with_marker


def test_display_path_is_relative_inside_the_repository() -> None:
    assert display_path(REPO_ROOT / "data" / "lfw_home") == "data/lfw_home"


def test_display_path_falls_back_to_the_name_outside_the_repository() -> None:
    assert display_path(Path("/tmp/somewhere/pairsDevTest.txt")) == "pairsDevTest.txt"


def test_encode_jpeg_produces_jpeg_bytes() -> None:
    data = _encode_jpeg(image_with_marker())
    assert data.startswith(b"\xff\xd8")


def test_timed_decode_embed_times_the_full_path() -> None:
    encoded = _encode_jpeg(image_with_marker())
    summary = _timed_decode_embed(FakeEmbedder(), encoded, repeats=2)
    assert summary is not None
    assert summary["n"] == 2
    assert summary["mean"] >= 0.0
