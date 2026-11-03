"""Tests for configuration helpers."""

from __future__ import annotations

import cv2

from facecheck.config import Settings, apply_opencv_thread_settings


def test_cv_threads_defaults_to_none() -> None:
    assert Settings(_env_file=None).cv_threads is None


def test_cv_threads_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("FACECHECK_CV_THREADS", "2")
    assert Settings(_env_file=None).cv_threads == 2


def test_apply_cv_threads_calls_set_num_threads(monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(cv2, "setNumThreads", calls.append)

    apply_opencv_thread_settings(Settings(_env_file=None, cv_threads=1))

    assert calls == [1]


def test_unset_cv_threads_leaves_opencv_alone(monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(cv2, "setNumThreads", calls.append)

    apply_opencv_thread_settings(Settings(_env_file=None))

    assert calls == []
