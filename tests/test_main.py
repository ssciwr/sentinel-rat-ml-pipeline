"""Tests for the ML Pipeline HTTP API, run with the tiny service models."""

import os
import shutil
from pathlib import Path

import httpx2
import pytest
import yaml
from fastapi.testclient import TestClient
from huggingface_hub.errors import LocalEntryNotFoundError

from ml_pipeline.config import settings
from ml_pipeline.main import app
from tests.service_support import (
    ANIMAL_BOX,
    CLASSIFIER_INFO,
    DETECTOR_INFO,
    HUMAN_BOX,
    RED_TAXON,
    config_values,
    write_two_colour_image,
)

CONFIGS_DIR = Path(__file__).parents[1] / "configs"


def _fresh_client(monkeypatch, detector_config: Path | None, classifier_config: Path | None) -> TestClient:
    """A test client of the app without a session, with the given env settings."""
    monkeypatch.setattr(settings, "DETECTOR_CONFIG", detector_config)
    monkeypatch.setattr(settings, "CLASSIFIER_CONFIG", classifier_config)
    monkeypatch.setattr(app.state, "session", None)
    return TestClient(app)


@pytest.fixture
def client(monkeypatch, service_configs) -> TestClient:
    """A test client of the app without a session, whose env settings name the tiny-model configs."""
    return _fresh_client(monkeypatch, *service_configs)


def test_health_endpoint(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ------------------------------------------------------------------------------- analyze


def test_analyze_endpoint(client, tmp_path) -> None:
    """The response has the README shape: one result per path in order, both models,
    labelled detections, classifications only for animals, and `error` null."""
    paths = [str(write_two_colour_image(tmp_path / name)) for name in ("a.png", "b.png")]

    response = client.post("/api/v1/analyze", json={"image_paths": paths})

    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["image_path"] for r in results] == paths
    result = results[0]
    assert set(result) == {"image_path", "detection_model", "classification_model", "detections", "error"}
    assert result["detection_model"] == DETECTOR_INFO
    assert result["classification_model"] == CLASSIFIER_INFO
    assert result["error"] is None
    animal, human = result["detections"]
    assert set(animal) == {"detected_class", "confidence", "bbox", "classifications"}
    assert animal["detected_class"] == "animal"
    assert animal["confidence"] == pytest.approx(0.9)
    assert animal["bbox"] == ANIMAL_BOX
    (classification,) = animal["classifications"]
    assert {key: classification[key] for key in RED_TAXON} == RED_TAXON
    assert set(classification) == {
        "kingdom",
        "phylum",
        "class_name",
        "order",
        "family",
        "genus",
        "species",
        "common_name",
        "confidence",
    }
    assert human["detected_class"] == "human"
    assert human["bbox"] == HUMAN_BOX
    assert human["classifications"] == []


def test_analyze_endpoint_reports_unreadable_image(client, tmp_path) -> None:
    """A missing image gets an error and no detections; the request still succeeds."""
    good = str(write_two_colour_image(tmp_path / "good.png"))
    missing = str(tmp_path / "missing.jpg")

    response = client.post("/api/v1/analyze", json={"image_paths": [missing, good]})

    assert response.status_code == 200
    failed, analysed = response.json()["results"]
    assert failed["image_path"] == missing
    assert isinstance(failed["error"], str) and failed["error"]
    assert failed["detections"] == []
    assert analysed["error"] is None
    assert len(analysed["detections"]) == 2


@pytest.mark.parametrize(
    "image_paths",
    [["image.jpg"], ["./images/image.jpg"], ["/data/a.jpg", "relative/b.jpg"], [""], ["/data/a\u0000b.jpg"]],
)
def test_analyze_endpoint_rejects_relative_paths(client, image_paths) -> None:
    """Every path must be absolute (and free of NUL bytes); otherwise the request is invalid (422)."""
    response = client.post("/api/v1/analyze", json={"image_paths": image_paths})
    assert response.status_code == 422


def test_analyze_endpoint_rejects_empty_list(client) -> None:
    response = client.post("/api/v1/analyze", json={"image_paths": []})
    assert response.status_code == 422


def test_analyze_endpoint_rejects_single_path(client) -> None:
    response = client.post("/api/v1/analyze", json={"image_path": "/fake/a.jpg"})
    assert response.status_code == 422


def test_analyze_uses_the_current_session(client, service_configs, tmp_path) -> None:
    """After a new session with a relabelling detector, /analyze runs the new models: the
    renamed `animal` boxes are no longer classified."""
    detector_config, _ = service_configs
    image = str(write_two_colour_image(tmp_path / "image.png"))
    uploaded = config_values(detector_config) | {"label_map": {"animal": "critter"}}

    assert client.post("/api/v1/session", json={"detector": uploaded}).status_code == 200
    (result,) = client.post("/api/v1/analyze", json={"image_paths": [image]}).json()["results"]

    assert [d["detected_class"] for d in result["detections"]] == ["critter", "human"]
    assert all(d["classifications"] == [] for d in result["detections"])


# ------------------------------------------------------------------------------- session


def test_get_session_builds_it_from_the_env_settings_once(client, service_configs) -> None:
    """The first GET builds the session from the env settings; the next GET shows the same one."""
    detector_config, classifier_config = service_configs

    first = client.get("/api/v1/session")
    second = client.get("/api/v1/session")

    assert first.status_code == 200
    body = first.json()
    assert set(body) == {"created_at", "models"}
    detector_card, classifier_card = body["models"]
    assert detector_card["role"] == "detector"
    assert detector_card["type"] == "TorchDetector"
    assert detector_card["info"] == DETECTOR_INFO
    assert detector_card["source"] == str(detector_config)
    assert detector_card["labels"] == ["animal", "human"]
    assert classifier_card["role"] == "classifier"
    assert classifier_card["type"] == "TorchClassifier"
    assert classifier_card["info"] == CLASSIFIER_INFO
    assert classifier_card["source"] == str(classifier_config)
    assert classifier_card["num_classes"] == 3
    assert "num_classes" not in detector_card and "labels" not in classifier_card
    assert second.json() == body


@pytest.fixture
def editable_configs(client, monkeypatch, service_configs, tmp_path) -> tuple[Path, Path]:
    """Copies of the tiny-model configs that the env settings point to, so tests can edit them.

    Depends on `client`, so it always runs after `client` has set the env settings it overrides.
    """
    copies = tuple(Path(shutil.copy(config, tmp_path / config.name)) for config in service_configs)
    monkeypatch.setattr(settings, "DETECTOR_CONFIG", copies[0])
    monkeypatch.setattr(settings, "CLASSIFIER_CONFIG", copies[1])
    return copies


def _rename_model(config: Path, name: str) -> None:
    """Change the `info.name` of a config file."""
    values = config_values(config)
    values["info"]["name"] = name
    config.write_text(yaml.safe_dump(values))


def test_post_session_without_body_rebuilds_both_from_the_env_config_files(client, editable_configs) -> None:
    """POST without a body re-reads both env config files: edits since the last session show up."""
    detector_config, classifier_config = editable_configs
    client.get("/api/v1/session")
    _rename_model(detector_config, "edited-detector")
    _rename_model(classifier_config, "edited-classifier")

    response = client.post("/api/v1/session")

    assert response.status_code == 200
    detector_card, classifier_card = response.json()["models"]
    assert detector_card["info"]["name"] == "edited-detector"
    assert classifier_card["info"]["name"] == "edited-classifier"
    assert client.get("/api/v1/session").json() == response.json()


def test_post_session_with_uploaded_config_switches_that_model(client, service_configs, editable_configs) -> None:
    """An uploaded detector config replaces the detector; the omitted classifier is rebuilt from
    its env config file."""
    detector_config, _ = service_configs
    _, env_classifier_config = editable_configs
    client.get("/api/v1/session")
    _rename_model(env_classifier_config, "edited-classifier")
    uploaded = config_values(detector_config) | {
        "info": {"name": "relabelled-detector", "version": "2"},
        "label_map": {"human": "person"},
    }

    response = client.post("/api/v1/session", json={"detector": uploaded})

    assert response.status_code == 200
    detector_card, classifier_card = response.json()["models"]
    assert detector_card["source"] == "request"
    assert detector_card["info"]["name"] == "relabelled-detector"
    assert detector_card["labels"] == ["animal", "person"]
    assert classifier_card["source"] == str(env_classifier_config)
    assert classifier_card["info"]["name"] == "edited-classifier"


def test_post_session_with_both_configs_needs_no_env(monkeypatch, service_configs) -> None:
    """With both configs uploaded, the env settings are not needed, even without a session yet."""
    client = _fresh_client(monkeypatch, None, None)
    detector_config, classifier_config = service_configs

    response = client.post(
        "/api/v1/session",
        json={"detector": config_values(detector_config), "classifier": config_values(classifier_config)},
    )

    assert response.status_code == 200
    assert [card["source"] for card in response.json()["models"]] == ["request", "request"]


def _hub_classifier_values(classifier_config: Path, **hub_args) -> tuple[dict, str]:
    """The classifier config with HF Hub weights (`hub_args`) instead of its local model_path,
    and that local path (what a stubbed hf_hub_download returns)."""
    values = config_values(classifier_config)
    local_weights = values.pop("model_path")
    return values | {"hf_repo": "org/rodent-models", "hf_filename": "cls.pt"} | hub_args, local_weights


def test_post_session_with_hub_config_downloads_the_weights(client, service_configs, monkeypatch) -> None:
    """An uploaded config with HF Hub arguments loads its weights through hf_hub_download, and
    the card names the Hub file."""
    _, classifier_config = service_configs
    uploaded, local_weights = _hub_classifier_values(classifier_config, hf_revision="v1")
    calls = []

    def fake_hf_hub_download(repo_id, filename, revision=None, **kwargs):
        calls.append((repo_id, filename, revision))
        return local_weights

    monkeypatch.setattr("ml_pipeline.models.model.hf_hub_download", fake_hf_hub_download)

    response = client.post("/api/v1/session", json={"classifier": uploaded})

    assert response.status_code == 200
    assert calls == [("org/rodent-models", "cls.pt", "v1")]
    classifier_card = response.json()["models"][1]
    assert classifier_card["source"] == "request"
    assert classifier_card["weights"] == "hf:org/rodent-models/cls.pt@v1"


def _without(values: dict, key: str) -> dict:
    """A copy of a config mapping without one key."""
    return {k: v for k, v in values.items() if k != key}


@pytest.mark.parametrize(
    ("role", "make_config"),
    [
        ("detector", lambda det, cls: det | {"type": "ResNet"}),
        ("detector", lambda det, cls: _without(det, "preprocessing")),
        ("detector", lambda det, cls: det | {"batch_size": 4}),
        ("detector", lambda det, cls: det | {"conf_limit": 2.0}),
        ("detector", lambda det, cls: det | {"model_path": "/nonexistent/detector.pt"}),
        ("detector", lambda det, cls: cls),
        ("classifier", lambda det, cls: det),
        (
            "detector",
            lambda det, cls: {
                "type": "SpeciesnetDetector",
                "info": det["info"],
                "preprocessing": {},
                "conf_limit": 0.1,
                "hf_repo": "org/speciesnet",
                "hf_filename": "model.pt",
            },
        ),
    ],
    ids=[
        "unknown_type",
        "missing_key",
        "unknown_key",
        "invalid_value",
        "missing_weights",
        "classifier_as_detector",
        "detector_as_classifier",
        "speciesnet_from_hf_repo",
    ],
)
def test_post_session_with_invalid_config_is_rejected_and_keeps_the_old_session(
    client, service_configs, role, make_config
) -> None:
    """An invalid uploaded config gives 422 with a message; the running session stays."""
    old = client.get("/api/v1/session").json()
    uploaded = make_config(*(config_values(path) for path in service_configs))

    response = client.post("/api/v1/session", json={role: uploaded})

    assert response.status_code == 422
    assert response.json()["detail"]
    assert client.get("/api/v1/session").json() == old


def test_post_session_with_unknown_role_is_rejected(client, service_configs) -> None:
    """A misspelled role is a 422, not a silent rebuild from env."""
    detector_config, _ = service_configs
    old = client.get("/api/v1/session").json()

    response = client.post("/api/v1/session", json={"detectr": config_values(detector_config)})

    assert response.status_code == 422
    assert client.get("/api/v1/session").json() == old


def test_post_session_with_unset_env_for_an_omitted_role_is_rejected(client, service_configs, monkeypatch) -> None:
    """A role without an uploaded config needs its env setting; otherwise 422 naming it."""
    detector_config, _ = service_configs
    old = client.get("/api/v1/session").json()
    monkeypatch.setattr(settings, "CLASSIFIER_CONFIG", None)

    response = client.post("/api/v1/session", json={"detector": config_values(detector_config)})

    assert response.status_code == 422
    assert "CLASSIFIER_CONFIG" in response.json()["detail"]
    assert client.get("/api/v1/session").json() == old


@pytest.mark.parametrize(
    "error",
    [LocalEntryNotFoundError("cannot reach the Hub and the file is not cached"), httpx2.ReadError("connection lost")],
    ids=["not_cached", "network"],
)
def test_post_session_with_failing_hub_download_is_rejected(client, service_configs, monkeypatch, error) -> None:
    """Weights that cannot be downloaded give 502; the running session stays."""
    _, classifier_config = service_configs
    old = client.get("/api/v1/session").json()

    def offline_hf_hub_download(repo_id, filename, revision=None, **kwargs):
        raise error

    monkeypatch.setattr("ml_pipeline.models.model.hf_hub_download", offline_hf_hub_download)
    uploaded, _ = _hub_classifier_values(classifier_config)

    response = client.post("/api/v1/session", json={"classifier": uploaded})

    assert response.status_code == 502
    assert response.json()["detail"]
    assert client.get("/api/v1/session").json() == old


@pytest.mark.parametrize("variable", ["DETECTOR_CONFIG", "CLASSIFIER_CONFIG"])
def test_get_session_without_env_setting_is_unavailable_until_set(client, monkeypatch, variable) -> None:
    """Without a session and an env setting, GET /session answers 503 naming the variable;
    once it is set, the next request builds the session."""
    configured = getattr(settings, variable)
    monkeypatch.setattr(settings, variable, None)

    response = client.get("/api/v1/session")
    assert response.status_code == 503
    assert variable in response.json()["detail"]

    monkeypatch.setattr(settings, variable, configured)
    assert client.get("/api/v1/session").status_code == 200


def test_analyze_without_env_setting_is_unavailable(client, monkeypatch, tmp_path) -> None:
    """/analyze needs the session, too: without a classifier setting it answers 503."""
    monkeypatch.setattr(settings, "CLASSIFIER_CONFIG", None)
    image = str(write_two_colour_image(tmp_path / "image.png"))

    response = client.post("/api/v1/analyze", json={"image_paths": [image]})

    assert response.status_code == 503
    assert "CLASSIFIER_CONFIG" in response.json()["detail"]


def test_lazy_session_with_failing_hub_download_is_unavailable_until_it_works(
    client, service_configs, monkeypatch, tmp_path
) -> None:
    """An env config whose Hub download fails makes the lazy build answer 503; once the
    download works, the next request builds the session."""
    _, classifier_config = service_configs
    values, local_weights = _hub_classifier_values(classifier_config)
    hub_config = tmp_path / "hub_classifier.yaml"
    hub_config.write_text(yaml.safe_dump(values))
    monkeypatch.setattr(settings, "CLASSIFIER_CONFIG", hub_config)
    hub_reachable = False

    def flaky_hf_hub_download(repo_id, filename, revision=None, **kwargs):
        if hub_reachable:
            return local_weights
        else:
            raise httpx2.ConnectError("no route to host")

    monkeypatch.setattr("ml_pipeline.models.model.hf_hub_download", flaky_hf_hub_download)

    response = client.get("/api/v1/session")
    assert response.status_code == 503
    assert "org/rodent-models" in response.json()["detail"]

    hub_reachable = True
    assert client.get("/api/v1/session").status_code == 200


def test_config_file_that_is_not_utf8_is_unavailable_or_rejected(client, service_configs, monkeypatch, tmp_path):
    """A config file in another encoding (here Latin-1 with an umlaut) cannot be read: the lazy
    build answers 503, and POST without a body (re-reading env) answers 422."""
    detector_config, _ = service_configs
    latin1_config = tmp_path / "detector_latin1.yaml"
    latin1_config.write_bytes(detector_config.read_text().replace("fixed", "füxed").encode("latin-1"))
    monkeypatch.setattr(settings, "DETECTOR_CONFIG", latin1_config)

    lazy = client.get("/api/v1/session")
    rebuilt = client.post("/api/v1/session")

    assert lazy.status_code == 503
    assert "UnicodeDecodeError" in lazy.json()["detail"]
    assert rebuilt.status_code == 422


def test_lazy_session_with_missing_config_file_is_unavailable(client, monkeypatch, tmp_path) -> None:
    """A configured config file that does not exist makes the lazy build fail with 503."""
    monkeypatch.setattr(settings, "DETECTOR_CONFIG", tmp_path / "missing.yaml")
    response = client.get("/api/v1/session")
    assert response.status_code == 503
    assert response.json()["detail"]


# ------------------------------------------------------------------- SpeciesNet (opt-in)


@pytest.mark.speciesnet_integration
def test_speciesnet_baseline_classifies_a_rodent(monkeypatch) -> None:
    """The shipped baseline configs (SpeciesNet v4.0.3b detector, v4.0.3a crop classifier)
    find the rodent and classify it as a rodent. Run with RUN_SPECIESNET_INTEGRATION=1."""
    if os.environ.get("RUN_SPECIESNET_INTEGRATION") != "1":
        pytest.skip("Set RUN_SPECIESNET_INTEGRATION=1 to run real SpeciesNet inference")
    client = _fresh_client(
        monkeypatch, CONFIGS_DIR / "speciesnet_detector.yaml", CONFIGS_DIR / "speciesnet_classifier.yaml"
    )
    rodent = str(Path(__file__).parent / "data" / "rodent.jpg")

    response = client.post("/api/v1/analyze", json={"image_paths": [rodent]})

    assert response.status_code == 200
    (result,) = response.json()["results"]
    animals = [d for d in result["detections"] if d["detected_class"] == "animal"]
    assert animals
    orders = {c["order"] for d in animals for c in d["classifications"]}
    assert "rodentia" in orders
