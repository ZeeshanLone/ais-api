from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./ais_predictions.db"

# SQLite requires this flag for multithreaded FastAPI usage
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


def init_db():
    """
    Initialize database tables.
    Called once during application startup.
    """

    Base.metadata.create_all(bind=engine)


# Dependency used in FastAPI routes
# TODO: Current implementation for the get db return sync connection hence each request is blocking in nature
def get_db():
    """
    Provides a database session for request scope.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
