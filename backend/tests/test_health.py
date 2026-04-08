from __future__ import annotations

from pathlib import Path

from rag_service.core.config import Settings, get_settings


def test_health_endpoint_returns_service_metadata(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "nexus-rag-platform"


def test_settings_can_be_loaded_from_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
app:
  name: test-rag
  version: 9.9.9
logging:
  level: DEBUG
""".strip(),
        encoding="utf-8",
    )

    settings = Settings.from_yaml(config_file)

    assert settings.app.name == "test-rag"
    assert settings.app.version == "9.9.9"
    assert settings.logging.level == "DEBUG"


def test_environment_overrides_yaml(monkeypatch) -> None:
    monkeypatch.setenv("RAG_APP__ENV", "test")
    monkeypatch.setenv("RAG_LOGGING__LEVEL", "WARNING")

    settings = get_settings()

    assert settings.app.env == "test"
    assert settings.logging.level == "WARNING"
