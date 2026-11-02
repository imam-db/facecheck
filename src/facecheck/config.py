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

# Values committed to this repository for local experiments. They are public,
# so a production process must never start with them.
PLACEHOLDER_API_KEYS = frozenset({"dev-api-key"})
PLACEHOLDER_TEMPLATE_KEYS = frozenset({"ZGV2LW9ubHktZmFjZWNoZWNrLXRlbXBsYXRlLWtleSE="})


class ConfigurationError(RuntimeError):
    """Raised when the process is configured in a way that is unsafe to serve."""


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

    dev_mode: bool = False
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


def validate_startup_settings(settings: Settings) -> None:
    """Fail closed unless the process has usable authentication secrets.

    Keys must always be set. ``FACECHECK_DEV_MODE=true`` only relaxes the ban
    on the public placeholder values, so local runs can keep using them while a
    production process still refuses to serve.
    """

    api_key = (settings.api_key or "").strip()
    template_key = (settings.template_key or "").strip()
    problems: list[str] = []

    if not api_key:
        problems.append("FACECHECK_API_KEY is not set")
    elif not settings.dev_mode and api_key in PLACEHOLDER_API_KEYS:
        problems.append(
            "FACECHECK_API_KEY is a public development placeholder; set a strong key"
        )

    if not template_key:
        problems.append("FACECHECK_TEMPLATE_KEY is not set")
    elif not settings.dev_mode and template_key in PLACEHOLDER_TEMPLATE_KEYS:
        problems.append(
            "FACECHECK_TEMPLATE_KEY is a public development placeholder; generate a new key"
        )

    if problems:
        raise ConfigurationError(
            "refusing to start: "
            + "; ".join(problems)
            + ". Set both keys, or set FACECHECK_DEV_MODE=true for local use only"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
