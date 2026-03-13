from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./ais_predictions.db"

# SQLite requires this flag for multithreaded FastAPI usage.
# For production at high concurrency, replace SQLite with PostgreSQL
# and use async SQLAlchemy (asyncpg driver).
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def init_db() -> None:
    """
    Create all tables. Called once at application startup.
    Safe to call on an already-initialised database — SQLAlchemy
    uses CREATE TABLE IF NOT EXISTS semantics.
    """
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    FastAPI dependency: yields a per-request DB session.
    Session is always closed on exit, even if an exception is raised.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
