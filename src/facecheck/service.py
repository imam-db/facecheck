"""Enrollment and verification use cases."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from facecheck.config import Settings
from facecheck.embeddings import FaceEmbedder
from facecheck.quality import QualityError
from facecheck.similarity import best_cosine, cosine_similarity, decide
from facecheck.storage import (
    DuplicateExternalIdError,
    Storage,
    bytes_to_embedding,
)


class ServiceError(Exception):
    """An expected, client-facing failure with an HTTP status and error code."""

    status_code = 400
    code = "service_error"

    def __init__(self, message: str, *, details: object | None = None) -> None:
        self.message = message
        self.details = details
        super().__init__(message)


class PersonNotFoundError(ServiceError):
    status_code = 404
    code = "person_not_found"


class DuplicatePersonError(ServiceError):
    status_code = 409
    code = "duplicate_person"


class EnrollmentError(ServiceError):
    status_code = 400
    code = "invalid_enrollment"


class InconsistentEnrollmentError(EnrollmentError):
    code = "inconsistent_enrollment"


class ConsentRequiredError(ServiceError):
    status_code = 409
    code = "consent_required"


class VerificationError(ServiceError):
    status_code = 409
    code = "verification_failed"


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
        consent_version: str | None = None,
        consent_at: datetime | None = None,
    ):
        try:
            return self.storage.create_person(
                external_id=external_id,
                consent_version=consent_version,
                consent_at=consent_at,
            )
        except DuplicateExternalIdError as exc:
            raise DuplicatePersonError(
                f"external_id {external_id!r} is already registered"
            ) from exc

    def enroll(self, person_id: str, images: Sequence[np.ndarray]) -> EnrollResult:
        person = self.storage.get_person(person_id)
        if person is None:
            raise PersonNotFoundError(f"person {person_id} not found")

        if not person.consent_version or person.consent_at is None:
            raise ConsentRequiredError(
                "person has no recorded consent; record consent_version and "
                "consent_at before enrolling biometric templates"
            )

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
                details={"reasons": rejections},
            )

        self._check_enrollment_consistency(embeddings)

        self.storage.replace_templates(
            person_id,
            embeddings,
            model_version=self.embedder.model_version,
        )
        return EnrollResult(
            enrolled=len(embeddings), model_version=self.embedder.model_version
        )

    def _check_enrollment_consistency(self, embeddings: list[np.ndarray]) -> None:
        threshold = self.settings.effective_enroll_threshold
        worst_similarity = 1.0
        worst_pair: tuple[int, int] | None = None
        for left in range(len(embeddings)):
            for right in range(left + 1, len(embeddings)):
                similarity = cosine_similarity(embeddings[left], embeddings[right])
                if similarity < worst_similarity:
                    worst_similarity = similarity
                    worst_pair = (left, right)

        if worst_pair is not None and worst_similarity < threshold:
            raise InconsistentEnrollmentError(
                "enrollment images do not look like the same person",
                details={
                    "reason": "inconsistent_enrollment",
                    "min_similarity": worst_similarity,
                    "threshold": threshold,
                    "pair": list(worst_pair),
                },
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
