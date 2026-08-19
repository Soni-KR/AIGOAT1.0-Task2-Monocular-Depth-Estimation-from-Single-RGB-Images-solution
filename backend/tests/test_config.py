import pytest

from backend.config import DEFAULT_CORS_ORIGINS, Settings


def test_settings_use_safe_local_defaults(monkeypatch):
    monkeypatch.delenv("AIGOAT_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("AIGOAT_MAX_UPLOAD_MB", raising=False)

    settings = Settings.from_env()

    assert settings.cors_origins == DEFAULT_CORS_ORIGINS
    assert settings.max_upload_bytes == 10 * 1024 * 1024


def test_settings_parse_deployment_values(monkeypatch):
    monkeypatch.setenv(
        "AIGOAT_CORS_ORIGINS",
        "https://depth.example.com/, https://admin.example.com",
    )
    monkeypatch.setenv("AIGOAT_MAX_UPLOAD_MB", "4")

    settings = Settings.from_env()

    assert settings.cors_origins == (
        "https://depth.example.com",
        "https://admin.example.com",
    )
    assert settings.max_upload_bytes == 4 * 1024 * 1024


@pytest.mark.parametrize("value", ["0", "-1", "large"])
def test_settings_reject_invalid_upload_limits(monkeypatch, value):
    monkeypatch.setenv("AIGOAT_MAX_UPLOAD_MB", value)

    with pytest.raises(ValueError, match="AIGOAT_MAX_UPLOAD_MB"):
        Settings.from_env()


def test_settings_reject_empty_cors_origins(monkeypatch):
    monkeypatch.setenv("AIGOAT_CORS_ORIGINS", " , ")

    with pytest.raises(ValueError, match="AIGOAT_CORS_ORIGINS"):
        Settings.from_env()
