"""FastAPI application exposing enroll / verify / delete / meta."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

from fastapi import FastAPI, File, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from facecheck import __version__
from facecheck.config import Settings, get_settings
from facecheck.embeddings import FaceEmbedder, ModelNotFoundError, build_embedder, decode_image_bgr
from facecheck.quality import INVALID_IMAGE, QualityError
from facecheck.service import (
    EnrollmentError,
    FaceVerificationService,
    ServiceError,
)
from facecheck.storage import Storage


class PersonCreate(BaseModel):
    external_id: str | None = Field(default=None, max_length=255)
    display_name: str | None = Field(default=None, max_length=255)


class PersonCreated(BaseModel):
    id: str
    external_id: str | None
    display_name: str | None
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
    embedding_dim: int
    min_face_size: int


def create_app(
    settings: Settings | None = None,
    *,
    embedder: FaceEmbedder | None = None,
    storage: Storage | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = storage or Storage(settings.db_url)
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
        description="Minimal 1:1 face verification for SFA attendance prototypes.",
        lifespan=lifespan,
    )

    @app.exception_handler(ServiceError)
    async def _service_error(_request: Request, exc: ServiceError) -> JSONResponse:
        body: dict[str, object] = {"detail": exc.detail}
        if exc.reasons is not None:
            body["reasons"] = exc.reasons
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(QualityError)
    async def _quality_error(_request: Request, exc: QualityError) -> JSONResponse:
        return JSONResponse(
            status_code=400, content={"detail": str(exc), "reason": exc.reason}
        )

    @app.exception_handler(ModelNotFoundError)
    async def _model_error(_request: Request, exc: ModelNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/meta", response_model=MetaResponse)
    def meta(request: Request) -> MetaResponse:
        service: FaceVerificationService = request.app.state.service
        return MetaResponse(
            model_version=service.embedder.model_version,
            threshold=service.settings.match_threshold,
            embedding_dim=service.embedder.embedding_dim,
            min_face_size=service.settings.min_face_size,
        )

    @app.post("/v1/persons", response_model=PersonCreated, status_code=201)
    def create_person(request: Request, payload: PersonCreate | None = None) -> PersonCreated:
        service: FaceVerificationService = request.app.state.service
        person = service.create_person(
            external_id=payload.external_id if payload else None,
            display_name=payload.display_name if payload else None,
        )
        return PersonCreated(
            id=person.id,
            external_id=person.external_id,
            display_name=person.display_name,
            created_at=person.created_at,
        )

    @app.post("/v1/persons/{person_id}/enroll", response_model=EnrollResponse)
    def enroll(
        person_id: str,
        request: Request,
        files: Annotated[list[UploadFile], File(description="3 to 5 face images")],
    ) -> EnrollResponse:
        service: FaceVerificationService = request.app.state.service
        settings = service.settings

        minimum = settings.enroll_min_images
        maximum = settings.enroll_max_images
        if not minimum <= len(files) <= maximum:
            raise EnrollmentError(
                f"enroll requires between {minimum} and {maximum} images, got {len(files)}"
            )

        images = [_read_image(upload, settings) for upload in files]
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


def _read_image(upload: UploadFile, settings: Settings):
    data = upload.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise QualityError(
            INVALID_IMAGE,
            f"image exceeds the {settings.max_upload_bytes} byte upload limit",
        )
    return decode_image_bgr(data)


app = create_app()
