"""Runtime configuration for facecheck.

All settings can be overridden with ``FACECHECK_``-prefixed environment
variables, for example ``FACECHECK_DB_URL=sqlite:///./data/dev.db``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"
DEFAULT_RECOGNIZER_FILE = "face_recognition_sface_2021dec.onnx"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FACECHECK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_url: str = "sqlite:///./data/facecheck.db"
    models_dir: Path = Path("models")
    detector_file: str = DEFAULT_DETECTOR_FILE
    recognizer_file: str = DEFAULT_RECOGNIZER_FILE

    match_threshold: float = Field(default=0.363, gt=-1.0, le=1.0)
    enroll_threshold: float | None = Field(default=None, gt=-1.0, le=1.0)
    min_face_size: int = Field(default=80, ge=1)
    min_face_frac: float = Field(default=0.15, ge=0.0, le=1.0)
    detector_score_threshold: float = Field(default=0.9, gt=0.0, le=1.0)
    embedding_dim: int = Field(default=128, ge=1)

    detect_max_side: int = Field(default=960, ge=0)
    max_pixels: int = Field(default=25_000_000, ge=1)

    enroll_min_images: int = Field(default=3, ge=1)
    enroll_max_images: int = Field(default=5, ge=1)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)

    api_key: str | None = None
    template_key: str | None = None

    @property
    def detector_path(self) -> Path:
        return self.models_dir / self.detector_file

    @property
    def recognizer_path(self) -> Path:
        return self.models_dir / self.recognizer_file

    @property
    def effective_enroll_threshold(self) -> float:
        if self.enroll_threshold is not None:
            return self.enroll_threshold
        return self.match_threshold


@lru_cache
def get_settings() -> Settings:
    return Settings()
