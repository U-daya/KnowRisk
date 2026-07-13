"""Deploy entrypoint shim.

Some Render start commands reference `uvicorn serve:app`, while the real
FastAPI app lives in `backend/app.py`. This module re-exports it so BOTH
`serve:app` and `backend.app:app` resolve to the same application.
"""
from backend.app import app  # noqa: F401

__all__ = ["app"]
