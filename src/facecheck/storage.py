"""SQLAlchemy persistence for people and face templates.

Embeddings are encrypted with AES-GCM before they reach the database, so the
storage layer is the single choke point for template encryption. Each row
stores ``b"FCK1" + nonce + ciphertext``; the associated data binds the
ciphertext to ``person_id|template_id`` so rows cannot be swapped between
people. There is no vector extension: 1:1 verification loads a person's (at
most five) templates, decrypts them, and computes cosine similarity in Python.
"""

from __future__ import annotations

import base64
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy import ForeignKey, String, create_engine, event, select
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)
from sqlalchemy.types import DateTime, Integer, LargeBinary

ENVELOPE_PREFIX = b"FCK1"
NONCE_BYTES = 12
KEY_BYTES = 32


class DuplicateExternalIdError(ValueError):
    """Raised when an external_id is already used by another person."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    external_id: Mapped[str | None] = mapped_column(
        String(255), unique=True, index=True, nullable=True
    )
    consent_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    consent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    templates: Mapped[list[Template]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Template.created_at",
    )


class Template(Base):
    __tablename__ = "templates"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    person_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("persons.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    person: Mapped[Person] = relationship(back_populates="templates")


@dataclass(frozen=True)
class StoredTemplate:
    """A decrypted template returned by :class:`Storage`."""

    id: str
    person_id: str
    embedding: bytes
    dim: int
    model_version: str
    created_at: datetime | None = None


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=np.float32).reshape(-1).tobytes()


def bytes_to_embedding(payload: bytes, dim: int | None = None) -> np.ndarray:
    vector = np.frombuffer(payload, dtype=np.float32).copy()
    if dim is not None and vector.shape[0] != dim:
        raise ValueError(f"stored dim {dim} does not match {vector.shape[0]} floats in payload")
    return vector


def parse_template_key(value: str | bytes | None) -> bytes:
    """Parse a 32-byte AES key from urlsafe base64, standard base64, or hex."""

    if value is None or value == "":
        raise ValueError(
            "FACECHECK_TEMPLATE_KEY is required: provide a 32-byte key as "
            "urlsafe base64 or 64 hex characters"
        )

    if isinstance(value, bytes):
        if len(value) == KEY_BYTES:
            return value
        value = value.decode("ascii")

    text = value.strip()
    if len(text) == KEY_BYTES * 2:
        try:
            decoded = bytes.fromhex(text)
        except ValueError:
            decoded = b""
        if len(decoded) == KEY_BYTES:
            return decoded

    padded = text + "=" * (-len(text) % 4)
    for decoder in (base64.urlsafe_b64decode, base64.b64decode):
        try:
            decoded = decoder(padded.encode("ascii"))
        except (ValueError, UnicodeEncodeError):
            continue
        if len(decoded) == KEY_BYTES:
            return decoded

    raise ValueError(
        "FACECHECK_TEMPLATE_KEY must decode to exactly 32 bytes "
        "(urlsafe base64 or 64 hex characters)"
    )


def _aad(person_id: str, template_id: str) -> bytes:
    return f"{person_id}|{template_id}".encode()


def encrypt_embedding(key: bytes, person_id: str, template_id: str, payload: bytes) -> bytes:
    nonce = os.urandom(NONCE_BYTES)
    ciphertext = AESGCM(key).encrypt(nonce, payload, _aad(person_id, template_id))
    return ENVELOPE_PREFIX + nonce + ciphertext


def decrypt_embedding(key: bytes, person_id: str, template_id: str, payload: bytes) -> bytes:
    if not payload.startswith(ENVELOPE_PREFIX):
        raise ValueError(
            f"template {template_id} is not encrypted; re-enroll after upgrading"
        )
    nonce = payload[len(ENVELOPE_PREFIX) : len(ENVELOPE_PREFIX) + NONCE_BYTES]
    ciphertext = payload[len(ENVELOPE_PREFIX) + NONCE_BYTES :]
    return AESGCM(key).decrypt(nonce, ciphertext, _aad(person_id, template_id))


def _ensure_sqlite_parent(db_url: str) -> None:
    url = make_url(db_url)
    if not url.drivername.startswith("sqlite") or not url.database:
        return
    if url.database == ":memory:":
        return
    Path(url.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def build_engine(db_url: str) -> Engine:
    _ensure_sqlite_parent(db_url)
    kwargs: dict[str, object] = {"pool_pre_ping": True}
    if db_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(db_url, **kwargs)

    if db_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _configure_sqlite(dbapi_connection, _record):  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            # Overwrite deleted content instead of leaving it in free pages.
            cursor.execute("PRAGMA secure_delete=ON")
            cursor.close()

    return engine


class Storage:
    """Thin repository layer; one short-lived session per operation."""

    def __init__(self, db_url: str, *, template_key: str | bytes | None) -> None:
        self._key = parse_template_key(template_key)
        self.engine = build_engine(db_url)
        self._session_factory = sessionmaker(
            bind=self.engine, expire_on_commit=False, class_=Session
        )
        Base.metadata.create_all(self.engine)

    def close(self) -> None:
        self.engine.dispose()

    def create_person(
        self,
        *,
        external_id: str | None = None,
        consent_version: str | None = None,
        consent_at: datetime | None = None,
    ) -> Person:
        try:
            with self._session_factory() as session:
                person = Person(
                    external_id=external_id,
                    consent_version=consent_version,
                    consent_at=consent_at,
                )
                session.add(person)
                session.commit()
                return person
        except IntegrityError as exc:
            raise DuplicateExternalIdError(external_id or "") from exc

    def get_person(self, person_id: str) -> Person | None:
        with self._session_factory() as session:
            return session.get(Person, person_id)

    def delete_person(self, person_id: str) -> bool:
        with self._session_factory() as session:
            person = session.get(Person, person_id)
            if person is None:
                return False
            # Load and delete children explicitly so cascade works on SQLite
            # builds without foreign-key enforcement too.
            for template in list(person.templates):
                session.delete(template)
            session.delete(person)
            session.commit()
            return True

    def list_templates(self, person_id: str) -> list[StoredTemplate]:
        with self._session_factory() as session:
            stmt = (
                select(Template)
                .where(Template.person_id == person_id)
                .order_by(Template.created_at)
            )
            rows = list(session.scalars(stmt).all())
            return [
                StoredTemplate(
                    id=row.id,
                    person_id=row.person_id,
                    embedding=decrypt_embedding(
                        self._key, row.person_id, row.id, row.embedding
                    ),
                    dim=row.dim,
                    model_version=row.model_version,
                    created_at=row.created_at,
                )
                for row in rows
            ]

    def replace_templates(
        self,
        person_id: str,
        embeddings: list[np.ndarray],
        *,
        model_version: str,
    ) -> int:
        with self._session_factory() as session:
            person = session.get(Person, person_id)
            if person is None:
                raise KeyError(person_id)
            for template in list(person.templates):
                session.delete(template)
            for embedding in embeddings:
                vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
                template_id = _new_id()
                session.add(
                    Template(
                        id=template_id,
                        person_id=person_id,
                        embedding=encrypt_embedding(
                            self._key,
                            person_id,
                            template_id,
                            embedding_to_bytes(vector),
                        ),
                        dim=int(vector.shape[0]),
                        model_version=model_version,
                    )
                )
            session.commit()
            return len(embeddings)
