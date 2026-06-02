from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

# Determine if SQLite is used
is_sqlite = settings.DATABASE_URL.startswith("sqlite")

# Configure database engine
if is_sqlite:
    engine = create_engine(
        settings.DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get db session in FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def run_db_migrations():
    from sqlalchemy import text
    columns = [
        ("document_id", "VARCHAR"),
        ("upload_time", "DATETIME"),
        ("analysis_status", "VARCHAR"),
        ("risk_score", "FLOAT")
    ]
    with engine.begin() as conn:
        for col, col_type in columns:
            try:
                conn.execute(text(f"ALTER TABLE documents ADD COLUMN {col} {col_type}"))
            except Exception as e:
                # Column probably already exists or table doesn't exist
                pass
