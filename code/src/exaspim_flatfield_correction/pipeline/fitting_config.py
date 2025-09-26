"""Configuration helpers for fitting-based flatfield correction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FittingConfig(BaseModel):
    """
    Fitting-specific knobs for illumination correction.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    med_factor_binned: float = Field(
        default=2,
        ge=0,
        description=(
            "Multiplier applied to the foreground median when clipping binned "
            "channels before profile extraction."
        ),
    )
    med_factor_unbinned: float = Field(
        default=2,
        ge=0,
        description=(
            "Multiplier applied to the foreground median when clipping "
            "unbinned channels before profile extraction."
        ),
    )
    percentile: float = Field(
        default=99,
        ge=0,
        le=100,
        description=(
            "Legacy percentile control kept for backwards compatibility. "
            "Expose it here so JSON overrides do not fail validation."
        ),
    )
    gaussian_sigma: float = Field(
        default=2,
        ge=0,
        description=(
            "Standard deviation of the Gaussian filter used to smooth masked "
            "profiles along each axis."
        ),
    )
    profile_percentile: float = Field(
        default=80,
        ge=0,
        le=100,
        description=(
            "Percentile of foreground voxels sampled when computing the "
            "masked intensity profiles."
        ),
    )
    profile_min_voxels: int = Field(
        default=0,
        ge=0,
        description=(
            "Minimum number of foreground voxels required in a slice for it to "
            "contribute to the masked profile computation."
        ),
    )
    spline_smoothing: float = Field(
        default=0,
        ge=0,
        description=(
            "Smoothing factor passed to the spline rescaling routine to "
            "regularize the fitted correction curves."
        ),
    )
    limits_xy: tuple[float, float] | None = Field(
        default=(0.25, 1.2),
        description=(
            "Lower and upper clipping bounds applied to the X and Y "
            "correction curves; set to null to disable clamping."
        ),
    )
    limits_z: tuple[float, float] | None = Field(
        default=(0.25, 1.2),
        description=(
            "Lower and upper clipping bounds applied to the Z correction "
            "curve; set to null to disable clamping."
        ),
    )
    global_factor_binned: float = Field(
        default=3200,
        ge=0,
        description=(
            "Target median intensity for binned channels after correction, "
            "used to derive the global scaling ratio."
        ),
    )
    global_factor_unbinned: float = Field(
        default=100,
        ge=0,
        description=(
            "Target median intensity for unbinned channels after correction, "
            "used to derive the global scaling ratio."
        ),
    )
    mask_probability_threshold: float = Field(
        default=0.9,
        ge=0,
        le=1,
        description=(
            "Probability cutoff applied to the GMM-derived soft tissue map "
            "to derive a binary mask."
        ),
    )
    mask_probability_min_size: int = Field(
        default=10000,
        ge=0,
        description=(
            "Minimum connected-component size retained from the probability "
            "mask projection."
        ),
    )
    gmm_n_components: int = Field(
        default=3,
        ge=1,
        description=(
            "Number of Gaussian mixture components used when fitting the "
            "foreground/background probability model."
        ),
    )
    gmm_max_samples: int = Field(
        default=2_000_000,
        ge=1,
        description=(
            "Maximum number of voxels sampled from each class when training "
            "the mixture model."
        ),
    )
    gmm_batch_size: int = Field(
        default=200_000,
        ge=1,
        description=(
            "Mini-batch size for the incremental GMM solver, trading memory "
            "usage for convergence speed."
        ),
    )
    erosion_radius: int = Field(
        default=10,
        ge=0,
        description=(
            "Binary erosion radius applied to the initial mask before "
            "probability fitting to focus on confident foreground."
        ),
    )
    gmm_random_state: int = Field(
        default=0,
        description="Random seed forwarded to scikit-learn's GMM implementation.",
    )

    @classmethod
    def from_file(cls, path: str | Path) -> "FittingConfig":
        """Load configuration values from a JSON file."""

        raw = Path(path).read_text()
        data: Any = json.loads(raw)
        return cls.model_validate(data)

    def to_file(self, path: str | Path, *, indent: int = 2) -> None:
        """Serialize the configuration to JSON on disk."""

        payload = json.dumps(self.model_dump(), indent=indent)
        Path(path).write_text(payload)


def load_fitting_config(path: str | Path | None = None) -> FittingConfig:
    """Return either a default or file-backed fitting configuration."""

    if path is None:
        return FittingConfig()
    return FittingConfig.from_file(path)
