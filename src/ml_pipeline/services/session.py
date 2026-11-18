"""The session: the running detector and classifier, with the configs they were built from."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import yaml

from ml_pipeline.config import settings
from ml_pipeline.models.model import ClassifierBase, DetectorBase, build_model, model_class_of, read_config
from ml_pipeline.schemas.session import ClassifierCard, DetectorCard, SessionInfo


class SessionConfigError(Exception):
    """A session could not be built because a config is missing or invalid."""


@dataclass(frozen=True)
class ModelConfig:
    """A model config mapping and where it came from.

    Attributes:
        values: The config mapping (the YAML keys, including `type`).
        source: The config file path, or `"request"` for a config uploaded by the client.
    """

    values: dict[str, Any]
    source: str

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Read a config file.

        Args:
            path: The YAML config file.

        Returns:
            Its config, with the path as source.

        Raises:
            FileNotFoundError: The file does not exist.
            UnicodeDecodeError: The file is not UTF-8 text.
            yaml.YAMLError: The file is not valid YAML.
            TypeError: The file's top level is not a mapping.
        """
        return cls(values=read_config(path), source=str(path))

    @classmethod
    def from_request(cls, values: Mapping[str, Any]) -> Self:
        """Take a config uploaded by the client.

        Args:
            values: The uploaded config mapping.

        Returns:
            A copy of it, with `"request"` as source.
        """
        return cls(values=dict(values), source="request")


@dataclass(frozen=True)
class Session:
    """The running detector and classifier, the configs they were built from, and when.

    Built with `build`; a new session replaces the old one as a whole.

    Attributes:
        detector: The running detector.
        classifier: The running classifier.
        detector_config: The config the detector was built from.
        classifier_config: The config the classifier was built from.
        created_at: When the session was built (UTC).
    """

    detector: DetectorBase
    classifier: ClassifierBase
    detector_config: ModelConfig
    classifier_config: ModelConfig
    created_at: datetime

    @classmethod
    def build(cls, detector_config: ModelConfig, classifier_config: ModelConfig) -> Self:
        """Build both models.

        Both configs' model kinds are checked before either model is loaded, so a swapped config
        fails before a possibly large download.

        Args:
            detector_config: Config of the detector.
            classifier_config: Config of the classifier.

        Returns:
            The new session, created now (UTC).

        Raises:
            TypeError: A config names a model of the other kind, or has missing/unknown keys.
            ValueError, OSError, ImportError, RuntimeError: A config is invalid or its weights cannot
                be loaded (see `ml_pipeline.models.model.build_model`).
            WeightsDownloadError: Weights could not be downloaded from the Hugging Face Hub.
        """
        _check_kind(detector_config, DetectorBase, "detector")
        _check_kind(classifier_config, ClassifierBase, "classifier")
        return cls(
            detector=build_model(detector_config.values),
            classifier=build_model(classifier_config.values),
            detector_config=detector_config,
            classifier_config=classifier_config,
            created_at=datetime.now(UTC),
        )

    def info(self) -> SessionInfo:
        """Describe the session for the API.

        Returns:
            The session's creation time and the model cards of its detector and classifier.
        """
        detector_card = DetectorCard(
            role="detector",
            type=type(self.detector).__name__,
            info=self.detector.info,
            source=self.detector_config.source,
            conf_limit=self.detector.conf_limit,
            weights=self.detector.weights,
            labels=self.detector.output_labels,
        )
        classifier_card = ClassifierCard(
            role="classifier",
            type=type(self.classifier).__name__,
            info=self.classifier.info,
            source=self.classifier_config.source,
            conf_limit=self.classifier.conf_limit,
            weights=self.classifier.weights,
            num_classes=len(self.classifier.classes),
        )
        return SessionInfo(created_at=self.created_at, models=[detector_card, classifier_card])


def _check_kind(config: ModelConfig, kind: type, role: str) -> None:
    """Raise TypeError unless the config's `type` names a model of the given kind."""
    model_class = model_class_of(config.values)
    if not issubclass(model_class, kind):
        raise TypeError(f"the {role} config names a {model_class.__name__}, which is not a {role}")


def build_session(
    detector_values: Mapping[str, Any] | None = None, classifier_values: Mapping[str, Any] | None = None
) -> Session:
    """Build a session from uploaded configs; a role without one uses its env setting.

    Args:
        detector_values: Uploaded detector config, or None for `DETECTOR_CONFIG`.
        classifier_values: Uploaded classifier config, or None for `CLASSIFIER_CONFIG`.

    Returns:
        The new session.

    Raises:
        SessionConfigError: A config is missing (env setting unset) or invalid, or its local
            weights cannot be loaded; the message names the error.
        WeightsDownloadError: Weights could not be downloaded from the Hugging Face Hub.
    """
    detector_config = _uploaded_or_env_config(detector_values, "DETECTOR_CONFIG")
    classifier_config = _uploaded_or_env_config(classifier_values, "CLASSIFIER_CONFIG")
    try:
        return Session.build(detector_config, classifier_config)
    except (OSError, ValueError, TypeError, ImportError, RuntimeError) as error:
        # anticipated: every way a config can be invalid or point at unusable weights, including
        # download failures inside the SpeciesNet SDK (OSErrors), hf_repo for SpeciesNet
        # (NotImplementedError, a RuntimeError) and a pickled module whose package is missing
        # (ImportError); WeightsDownloadError is none of these and passes through for its own handling
        raise SessionConfigError(f"{type(error).__name__}: {error}") from error


def _uploaded_or_env_config(uploaded: Mapping[str, Any] | None, variable: str) -> ModelConfig:
    """The uploaded config of a role, or the config file named by the role's env setting.

    Args:
        uploaded: The uploaded config, or None.
        variable: Name of the role's setting (`DETECTOR_CONFIG` / `CLASSIFIER_CONFIG`).

    Raises:
        SessionConfigError: Nothing was uploaded and the env setting is unset, or its config file
            cannot be read.
    """
    if uploaded is not None:
        return ModelConfig.from_request(uploaded)
    path = getattr(settings, variable)
    if path is None:
        raise SessionConfigError(f"no model configured: set the {variable} environment variable to a model config file")
    try:
        return ModelConfig.from_file(path)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        # anticipated: the configured file is missing, unreadable, not UTF-8 text (UnicodeDecodeError
        # is a ValueError) or not a YAML mapping
        raise SessionConfigError(f"{variable}={path}: {type(error).__name__}: {error}") from error
