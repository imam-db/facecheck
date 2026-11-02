"""Enrollment and verification use cases."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from facecheck.config import Settings
from facecheck.embeddings import FaceEmbedder
from facecheck.quality import QualityError
from facecheck.similarity import best_cosine, decide
from facecheck.storage import (
    DuplicateExternalIdError,
    Storage,
    bytes_to_embedding,
)


class ServiceError(Exception):
    """An expected, client-facing failure with an HTTP status."""

    status_code = 400

    def __init__(self, detail: str, *, reasons: list[dict[str, object]] | None = None) -> None:
        self.detail = detail
        self.reasons = reasons
        super().__init__(detail)


class PersonNotFoundError(ServiceError):
    status_code = 404


class DuplicatePersonError(ServiceError):
    status_code = 409


class EnrollmentError(ServiceError):
    status_code = 400


class VerificationError(ServiceError):
    status_code = 409


@dataclass(frozen=True)
class EnrollResult:
    enrolled: int
    model_version: str


@dataclass(frozen=True)
class VerifyResult:
    score: float
    decision: str
    threshold: float
    model_version: str


class FaceVerificationService:
    def __init__(self, *, settings: Settings, storage: Storage, embedder: FaceEmbedder) -> None:
        self.settings = settings
        self.storage = storage
        self.embedder = embedder

    def create_person(
        self,
        *,
        external_id: str | None = None,
        display_name: str | None = None,
    ):
        try:
            return self.storage.create_person(
                external_id=external_id, display_name=display_name
            )
        except DuplicateExternalIdError as exc:
            raise DuplicatePersonError(
                f"external_id {external_id!r} is already registered"
            ) from exc

    def enroll(self, person_id: str, images: Sequence[np.ndarray]) -> EnrollResult:
        if self.storage.get_person(person_id) is None:
            raise PersonNotFoundError(f"person {person_id} not found")

        minimum = self.settings.enroll_min_images
        maximum = self.settings.enroll_max_images
        if not minimum <= len(images) <= maximum:
            raise EnrollmentError(
                f"enroll requires between {minimum} and {maximum} images, got {len(images)}"
            )

        embeddings: list[np.ndarray] = []
        rejections: list[dict[str, object]] = []
        for index, image in enumerate(images):
            try:
                result = self.embedder.detect_and_embed(image)
            except QualityError as exc:
                rejections.append(
                    {"index": index, "reason": exc.reason, "message": str(exc)}
                )
                continue
            if result.embedding.shape[0] != self.settings.embedding_dim:
                raise EnrollmentError(
                    f"embedder returned {result.embedding.shape[0]} dimensions, "
                    f"expected {self.settings.embedding_dim}"
                )
            embeddings.append(result.embedding)

        if rejections:
            raise EnrollmentError(
                "one or more images failed the face quality gate",
                reasons=rejections,
            )

        self.storage.replace_templates(
            person_id,
            embeddings,
            model_version=self.embedder.model_version,
        )
        return EnrollResult(
            enrolled=len(embeddings), model_version=self.embedder.model_version
        )

    def verify(self, person_id: str, image: np.ndarray) -> VerifyResult:
        if self.storage.get_person(person_id) is None:
            raise PersonNotFoundError(f"person {person_id} not found")

        templates = self.storage.list_templates(person_id)
        if not templates:
            raise VerificationError(f"person {person_id} has no enrolled templates")

        compatible = [
            template
            for template in templates
            if template.model_version == self.embedder.model_version
        ]
        if not compatible:
            stored = sorted({template.model_version for template in templates})
            raise VerificationError(
                f"no templates for model {self.embedder.model_version!r}; "
                f"re-enroll (stored: {', '.join(stored)})"
            )

        result = self.embedder.detect_and_embed(image)
        score = best_cosine(
            result.embedding,
            [bytes_to_embedding(template.embedding, template.dim) for template in compatible],
        )
        threshold = self.settings.match_threshold
        return VerifyResult(
            score=score,
            decision=decide(score, threshold),
            threshold=threshold,
            model_version=self.embedder.model_version,
        )

    def delete_person(self, person_id: str) -> None:
        if not self.storage.delete_person(person_id):
            raise PersonNotFoundError(f"person {person_id} not found")
