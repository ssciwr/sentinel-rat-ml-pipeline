"""Tiny torch modules used as pickled `.pt` fixtures in the model and service tests.

The classes live at module level of an importable module, because pickle stores a
class by its import path and `torch.load` must be able to re-import it.

Detector rows follow the torch detector contract: `(x1, y1, x2, y2, score, class_index)`.
"""

import torch


class TinyClassifier(torch.nn.Module):
    """Map an image batch to class logits that depend on the mean colour of each image.

    The dropout layer makes the output random in training mode, so identical repeated
    predictions show that the pipeline put the module into eval mode.

    Args:
        num_classes: Number of output classes C.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear = torch.nn.Linear(3, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits of shape [B, num_classes]."""
        mean_colour = images.mean(dim=(2, 3))
        return self.linear(self.dropout(mean_colour))


class ProbabilityClassifier(TinyClassifier):
    """Like `TinyClassifier`, but return softmax probabilities instead of logits.

    Args:
        num_classes: Number of output classes C.
    """

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return probabilities of shape [B, num_classes]."""
        return torch.softmax(super().forward(images), dim=1)


class ColourClassifier(torch.nn.Module):
    """Classify an image by its dominant colour: class 0 red, 1 green, 2 blue.

    Logits are `sharpness * mean RGB`, so a uniformly red input gives class 0 a softmax
    probability close to 1, while a half-red, half-blue input splits it between classes 0
    and 2. The service tests use this to check which image region reached the classifier.

    Args:
        sharpness: Factor applied to the mean colour.
    """

    def __init__(self, sharpness: float = 20.0):
        super().__init__()
        self.sharpness = sharpness

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits of shape [B, 3]."""
        return self.sharpness * images.mean(dim=(2, 3))


class ColourDetector(torch.nn.Module):
    """Return one box per image whose score and box depend on that image's colour.

    Score = 0.05 + 0.9 * mean red, box = (0.1, 0.1, 0.2 + 0.7 * mean green,
    0.2 + 0.7 * mean blue), class index 0, so images of different colours get
    distinguishable rows.
    """

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return detections of shape [B, 1, 6]."""
        red, green, blue = images.mean(dim=(2, 3)).unbind(dim=1)
        constant = torch.ones_like(red)
        rows = torch.stack(
            [
                0.1 * constant,
                0.1 * constant,
                0.2 + 0.7 * green,
                0.2 + 0.7 * blue,
                0.05 + 0.9 * red,
                0.0 * constant,
            ],
            dim=1,
        )
        return rows.unsqueeze(1)


class FixedRowsDetector(torch.nn.Module):
    """Return the same detection rows `[K, 6]` for every image of a batch.

    Args:
        rows: Detection rows `(x1, y1, x2, y2, score, class_index)`.
    """

    def __init__(self, rows: list[list[float]]):
        super().__init__()
        self.register_buffer("rows", torch.tensor(rows, dtype=torch.float32))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return detections of shape [B, K, 6]."""
        return self.rows.expand(images.shape[0], -1, -1).clone()


class VaryingRowsDetector(torch.nn.Module):
    """Return a different number of rows K on every call, cycling through `row_sets`.

    Args:
        row_sets: One list of detection rows per call, used in order and then repeated.
    """

    def __init__(self, row_sets: list[list[list[float]]]):
        super().__init__()
        self.row_sets = row_sets
        self.calls = 0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return detections of shape [B, K_call, 6]."""
        rows = torch.tensor(self.row_sets[self.calls % len(self.row_sets)], dtype=torch.float32)
        self.calls += 1
        return rows.expand(images.shape[0], -1, -1).clone()


class ContractBreaker(torch.nn.Module):
    """Return a deliberately malformed output, selected by `mode`, for contract tests.

    Args:
        mode: Classifier modes (C = `width`): "tuple", "rank3", "batch", "width",
            "not_normalised", "out_of_range", "nan". Detector modes: "det_tuple",
            "det_rank2", "det_five_columns", "det_batch", "det_score_range",
            "det_score_nan", "det_score_negative", "det_box_range", "det_box_inverted",
            "det_box_inverted_y", "det_class_fraction", "det_class_negative",
            "det_class_too_large".
        width: Number of classes C the classifier modes are built around.
    """

    def __init__(self, mode: str, width: int = 3):
        super().__init__()
        self.mode = mode
        self.width = width

    def forward(self, images: torch.Tensor):
        """Return the malformed output for the configured mode."""
        b, c = images.shape[0], self.width
        probabilities = torch.full((b, c), 1.0 / c)
        good_row = [0.1, 0.1, 0.5, 0.5, 0.9, 0.0]
        outputs = {
            "tuple": lambda: (probabilities, probabilities),
            "rank3": lambda: probabilities.unsqueeze(-1),
            "batch": lambda: torch.full((b + 1, c), 1.0 / c),
            "width": lambda: torch.full((b, c + 1), 1.0 / (c + 1)),
            "not_normalised": lambda: torch.full((b, c), 0.5),
            "out_of_range": lambda: torch.tensor([[1.5, -0.5] + [0.0] * (c - 2)]).expand(b, -1).clone(),
            "nan": lambda: torch.full((b, c), float("nan")),
            "det_tuple": lambda: (self._rows(b, [good_row]), self._rows(b, [good_row])),
            "det_rank2": lambda: torch.tensor([good_row]).expand(b, -1).clone(),
            "det_five_columns": lambda: self._rows(b, [good_row[:5]]),
            "det_batch": lambda: self._rows(b + 1, [good_row]),
            "det_score_range": lambda: self._rows(b, [[0.1, 0.1, 0.5, 0.5, 1.5, 0.0]]),
            "det_score_nan": lambda: self._rows(b, [[0.1, 0.1, 0.5, 0.5, float("nan"), 0.0]]),
            "det_score_negative": lambda: self._rows(b, [[0.1, 0.1, 0.5, 0.5, -0.5, 0.0]]),
            "det_box_range": lambda: self._rows(b, [[0.1, 0.1, 1.5, 0.5, 0.9, 0.0]]),
            "det_box_inverted": lambda: self._rows(b, [[0.6, 0.1, 0.5, 0.5, 0.9, 0.0]]),
            "det_box_inverted_y": lambda: self._rows(b, [[0.1, 0.6, 0.5, 0.5, 0.9, 0.0]]),
            "det_class_fraction": lambda: self._rows(b, [[0.1, 0.1, 0.5, 0.5, 0.9, 0.5]]),
            "det_class_negative": lambda: self._rows(b, [[0.1, 0.1, 0.5, 0.5, 0.9, -1.0]]),
            "det_class_too_large": lambda: self._rows(b, [[0.1, 0.1, 0.5, 0.5, 0.9, 99.0]]),
        }
        return outputs[self.mode]()

    @staticmethod
    def _rows(batch: int, rows: list[list[float]]) -> torch.Tensor:
        """Repeat detection rows for every image of the batch."""
        return torch.tensor(rows, dtype=torch.float32).expand(batch, -1, -1).clone()
