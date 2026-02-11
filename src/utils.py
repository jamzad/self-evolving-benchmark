# src/utils.py
import hashlib
import uuid
from datetime import datetime, timezone

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def new_id() -> str:
    return str(uuid.uuid4())

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.strip().encode("utf-8")).hexdigest()
