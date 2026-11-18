"""Schemas of the session endpoints: model cards of the running models and new-session requests."""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ml_pipeline.schemas.analysis import ModelInfo


class ModelCard(BaseModel):
    """Fields every model card has; `DetectorCard` and `ClassifierCard` add their role's fields."""

    # a card carries exactly its role's fields
    model_config = ConfigDict(extra="forbid")

    type: str  # model class, e.g. "SpeciesnetDetector"
    info: ModelInfo
    source: str  # the config file path, or "request" for an uploaded config
    conf_limit: float
    weights: str  # the model_path, or "hf:<repo>/<filename>[@<revision>]"


class DetectorCard(ModelCard):
    """Model card of the running detector."""

    role: Literal["detector"]
    labels: list[str]  # the labels the detector returns (label_map applied)


class ClassifierCard(ModelCard):
    """Model card of the running classifier."""

    role: Literal["classifier"]
    num_classes: int


class SessionInfo(BaseModel):
    """The running session: when it was built and the cards of its models."""

    created_at: datetime
    # detector first, then classifier; `role` decides the card type
    models: list[Annotated[DetectorCard | ClassifierCard, Field(discriminator="role")]]


class SessionRequest(BaseModel):
    """Request for a new session: complete model configs (the YAML keys, as JSON objects) per role.

    A role without a config is built from its env setting (`DETECTOR_CONFIG` / `CLASSIFIER_CONFIG`).
    """

    # a misspelled role must be an error, not a silent rebuild of that role from env
    model_config = ConfigDict(extra="forbid")

    detector: dict[str, Any] | None = None
    classifier: dict[str, Any] | None = None
