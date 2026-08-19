import os
from dataclasses import dataclass


DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)
DEFAULT_MAX_UPLOAD_MB = 10


@dataclass(frozen=True)
class Settings:
    """Runtime values that may differ between local and deployed environments."""

    cors_origins: tuple[str, ...]
    max_upload_bytes: int

    @classmethod
    def from_env(cls) -> "Settings":
        origins_text = os.getenv(
            "AIGOAT_CORS_ORIGINS",
            ",".join(DEFAULT_CORS_ORIGINS),
        )
        cors_origins = tuple(
            origin.strip().rstrip("/")
            for origin in origins_text.split(",")
            if origin.strip()
        )
        if not cors_origins:
            raise ValueError("AIGOAT_CORS_ORIGINS must contain at least one origin.")

        max_upload_text = os.getenv(
            "AIGOAT_MAX_UPLOAD_MB",
            str(DEFAULT_MAX_UPLOAD_MB),
        )
        try:
            max_upload_mb = int(max_upload_text)
        except ValueError as exc:
            raise ValueError("AIGOAT_MAX_UPLOAD_MB must be an integer.") from exc

        if max_upload_mb <= 0:
            raise ValueError("AIGOAT_MAX_UPLOAD_MB must be greater than zero.")

        return cls(
            cors_origins=cors_origins,
            max_upload_bytes=max_upload_mb * 1024 * 1024,
        )
