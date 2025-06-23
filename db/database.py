from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.base import Base

# Load environment variables from .env
load_dotenv()

# Use env var or fallback to SQLite
DB_URL = os.getenv("DATABASE_URL", "sqlite:///db.sqlite3")

# SQLAlchemy engine + session
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    from db import models
    Base.metadata.create_all(bind=engine)
