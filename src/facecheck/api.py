"""FastAPI application exposing enroll / verify / delete / meta."""

from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from facecheck import __version__
from facecheck.config import Settings, get_settings
from facecheck.embeddings import (
    EmbedderError,
    FaceEmbedder,
    ModelNotFoundError,
    build_embedder,
    decode_image_bgr,
)
from facecheck.quality import INVALID_IMAGE, QualityError
from facecheck.service import (
    FaceVerificationService,
    ServiceError,
)
from facecheck.storage import Storage


class PersonCreate(BaseModel):
    external_id: str | None = Field(default=None, max_length=255)
    consent_version: str = Field(min_length=1, max_length=64)
    consent_at: datetime


class PersonCreated(BaseModel):
    id: str
    external_id: str | None
    consent_version: str
    consent_at: datetime
    created_at: datetime


class EnrollResponse(BaseModel):
    enrolled: int
    model_version: str


class VerifyResponse(BaseModel):
    score: float
    decision: str
    threshold: float
    model_version: str


class MetaResponse(BaseModel):
    model_version: str
    threshold: float
    enroll_threshold: float
    embedding_dim: int
    min_face_size: int
    min_face_frac: float
    detect_max_side: int


def _error(
    status_code: int,
    code: str,
    message: str,
    details: object | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"code": code, "message": message, "details": details},
    )


def create_app(
    settings: Settings | None = None,
    *,
    embedder: FaceEmbedder | None = None,
    storage: Storage | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = storage or Storage(settings.db_url, template_key=settings.template_key)
        embed = embedder or build_embedder(settings)
        app.state.service = FaceVerificationService(
            settings=settings, storage=store, embedder=embed
        )
        app.state.settings = settings
        try:
            yield
        finally:
            if storage is None:
                store.close()

    app = FastAPI(
        title="facecheck",
        version=__version__,
        description="Minimal 1:1 face verification for SFA attendance check-ins.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def _require_api_key(request: Request, call_next):
        if request.url.path == "/healthz" or not settings.api_key:
            return await call_next(request)

        provided = request.headers.get("x-api-key")
        if not provided:
            scheme, _, token = request.headers.get("authorization", "").partition(" ")
            if scheme.lower() == "bearer":
                provided = token

        if not provided or not hmac.compare_digest(provided, settings.api_key):
            return _error(
                401,
                "unauthorized",
                "missing or invalid API key; send X-API-Key or Authorization: Bearer",
            )
        return await call_next(request)

    @app.exception_handler(ServiceError)
    async def _service_error(_request: Request, exc: ServiceError) -> JSONResponse:
        return _error(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(QualityError)
    async def _quality_error(_request: Request, exc: QualityError) -> JSONResponse:
        return _error(400, "quality_error", str(exc), {"reason": exc.reason})

    @app.exception_handler(ModelNotFoundError)
    async def _model_error(_request: Request, exc: ModelNotFoundError) -> JSONResponse:
        return _error(503, "model_not_found", str(exc))

    @app.exception_handler(EmbedderError)
    async def _embedder_error(_request: Request, exc: EmbedderError) -> JSONResponse:
        return _error(422, "embedder_error", str(exc))

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/meta", response_model=MetaResponse)
    def meta(request: Request) -> MetaResponse:
        service: FaceVerificationService = request.app.state.service
        settings = service.settings
        return MetaResponse(
            model_version=service.embedder.model_version,
            threshold=settings.match_threshold,
            enroll_threshold=settings.effective_enroll_threshold,
            embedding_dim=service.embedder.embedding_dim,
            min_face_size=settings.min_face_size,
            min_face_frac=settings.min_face_frac,
            detect_max_side=settings.detect_max_side,
        )

    @app.post("/v1/persons", response_model=PersonCreated, status_code=201)
    def create_person(request: Request, payload: PersonCreate) -> PersonCreated:
        service: FaceVerificationService = request.app.state.service
        person = service.create_person(
            external_id=payload.external_id,
            consent_version=payload.consent_version,
            consent_at=payload.consent_at,
        )
        return PersonCreated(
            id=person.id,
            external_id=person.external_id,
            consent_version=person.consent_version or payload.consent_version,
            consent_at=person.consent_at or payload.consent_at,
            created_at=person.created_at,
        )

    @app.post("/v1/persons/{person_id}/enroll", response_model=EnrollResponse)
    def enroll(
        person_id: str,
        request: Request,
        files: Annotated[list[UploadFile], File(description="3 to 5 face images")],
    ) -> EnrollResponse:
        service: FaceVerificationService = request.app.state.service
        images = [_read_image(upload, service.settings) for upload in files]
        result = service.enroll(person_id, images)
        return EnrollResponse(enrolled=result.enrolled, model_version=result.model_version)

    @app.post("/v1/persons/{person_id}/verify", response_model=VerifyResponse)
    def verify(
        person_id: str,
        request: Request,
        file: Annotated[UploadFile, File(description="probe face image")],
    ) -> VerifyResponse:
        service: FaceVerificationService = request.app.state.service
        image = _read_image(file, service.settings)
        result = service.verify(person_id, image)
        return VerifyResponse(
            score=result.score,
            decision=result.decision,
            threshold=result.threshold,
            model_version=result.model_version,
        )

    @app.delete("/v1/persons/{person_id}", status_code=204)
    def delete_person(person_id: str, request: Request) -> Response:
        service: FaceVerificationService = request.app.state.service
        service.delete_person(person_id)
        return Response(status_code=204)

    return app


def _read_image(upload: UploadFile, settings: Settings) -> np.ndarray:
    data = upload.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise QualityError(
            INVALID_IMAGE,
            f"image exceeds the {settings.max_upload_bytes} byte upload limit",
        )
    return decode_image_bgr(data, max_pixels=settings.max_pixels)


app = create_app()
