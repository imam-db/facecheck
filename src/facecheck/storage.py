"""SQLAlchemy persistence for people and face templates.

Embeddings are stored as raw little-endian float32 bytes in a BLOB/bytea
column. There is no vector extension: 1:1 verification loads a person's (at
most five) templates and computes cosine similarity in Python.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
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
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
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


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=np.float32).reshape(-1).tobytes()


def bytes_to_embedding(payload: bytes, dim: int | None = None) -> np.ndarray:
    vector = np.frombuffer(payload, dtype=np.float32).copy()
    if dim is not None and vector.shape[0] != dim:
        raise ValueError(f"stored dim {dim} does not match {vector.shape[0]} floats in payload")
    return vector


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
        def _enable_sqlite_foreign_keys(dbapi_connection, _record):  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


class Storage:
    """Thin repository layer; one short-lived session per operation."""

    def __init__(self, db_url: str) -> None:
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
        display_name: str | None = None,
    ) -> Person:
        try:
            with self._session_factory() as session:
                person = Person(external_id=external_id, display_name=display_name)
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

    def list_templates(self, person_id: str) -> list[Template]:
        with self._session_factory() as session:
            stmt = select(Template).where(Template.person_id == person_id)
            return list(session.scalars(stmt).all())

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
                session.add(
                    Template(
                        person_id=person_id,
                        embedding=embedding_to_bytes(vector),
                        dim=int(vector.shape[0]),
                        model_version=model_version,
                    )
                )
            session.commit()
            return len(embeddings)
