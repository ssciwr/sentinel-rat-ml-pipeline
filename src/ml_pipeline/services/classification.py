"""Species classification service."""

from ml_pipeline.schemas.analysis import Classification, Detection, ModelInfo

CLASSIFICATION_MODEL = ModelInfo(
    name="rodent-classifier",
    version="0.1",
    description="Simulated rodent species classifier",
)

_MURIDAE = {
    "kingdom": "Animalia",
    "phylum": "Chordata",
    "class_name": "Mammalia",
    "order": "Rodentia",
    "family": "Muridae",
}


def classify_species(image_path: str, detection: Detection) -> list[Classification]:
    """Return species candidates for one detected animal, most likely first."""
    # TODO: replace with real classification model
    return [
        Classification(
            **_MURIDAE,
            genus="Rattus",
            species="rattus",
            common_name="black rat",
            confidence=0.87,
        ),
        Classification(
            **_MURIDAE,
            genus="Rattus",
            species="norvegicus",
            common_name="brown rat",
            confidence=0.1,
        ),
    ]
