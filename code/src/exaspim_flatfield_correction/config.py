"""Configuration helpers for fitting-based flatfield correction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import logging

from pydantic import BaseModel, ConfigDict, Field

_LOGGER = logging.getLogger(__name__)


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
    gaussian_sigma: float = Field(
        default=2,
        ge=0,
        description=(
            "Standard deviation of the Gaussian filter used to smooth masked "
            "profiles along each axis."
        ),
    )
    profile_percentile: float = Field(
        default=50,
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
    limits_x: tuple[float, float] | None = Field(
        default=(0.25, 1.2),
        description=(
            "Lower and upper clipping bounds applied to the X "
            "correction curve; set to null to disable clamping."
        ),
    )
    limits_y: tuple[float, float] | None = Field(
        default=(0.25, 1.2),
        description=(
            "Lower and upper clipping bounds applied to the Y "
            "correction curve; set to null to disable clamping."
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
    probability_bg_low_percentile: float = Field(
        default=50.0,
        ge=0,
        le=100,
        description=(
            "Lower percentile of background intensities anchoring the low end "
            "of the logistic weighting ramp."
        ),
    )
    probability_bg_high_percentile: float = Field(
        default=99.9,
        ge=0,
        le=100,
        description=(
            "Upper percentile of background intensities targeting weights near "
            "one in the logistic ramp."
        ),
    )
    probability_ramp_eps: float = Field(
        default=0.01,
        gt=0,
        lt=0.5,
        description=(
            "Logistic ramp value assigned at the low anchor percentile; also "
            "controls the sharpness of the transition."
        ),
    )
    probability_smooth_sigma: float = Field(
        default=1.0,
        ge=0,
        description=(
            "Gaussian sigma applied to softly smooth the percentile-based "
            "weight volume."
        ),
    )
    median_summary_path: Path | None = Field(
        default=None,
        description=(
            "Optional path to a JSON file with per-channel mean_of_medians "
            "values for overriding global normalization factors."
        ),
    )
    enable_gmm_refinement: bool = Field(
        default=False,
        description=(
            "When true, compute a probability volume via GMM and refine the "
            "initial mask; when false, skip the refinement and use the precomputed mask."
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
        default=2,
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

        payload = json.dumps(self.model_dump(mode="json"), indent=indent)
        Path(path).write_text(payload)


def load_fitting_config(path: str | Path | None = None) -> FittingConfig:
    """Return either a default or file-backed fitting configuration."""

    if path is None:
        return FittingConfig()
    return FittingConfig.from_file(path)


def read_median_intensity_summary(
    path: str | Path | None,
) -> dict[str, float]:
    """Load per-channel normalization targets from disk.

    Parameters
    ----------
    path : str or pathlib.Path or None
        Filesystem path pointing to ``median_intensity_summary.json``. When
        ``None``, the function returns an empty mapping.

    Returns
    -------
    dict of str to float
        Mapping from channel identifiers to their ``mean_of_medians`` value.

    Notes
    -----
    The function logs and skips channels that are missing or have
    non-numeric ``mean_of_medians`` entries.
    """

    if path is None:
        return {}

    summary_path = Path(path)
    if not summary_path.exists():
        _LOGGER.debug("Median intensity summary not found at %s", summary_path)
        return {}

    try:
        data = json.loads(summary_path.read_text())
    except Exception:  # noqa: BLE001 - want to log previously unseen errors
        _LOGGER.exception(
            "Failed to read median intensity summary at %s", summary_path
        )
        return {}

    overrides: dict[str, float] = {}
    for channel, payload in data.items():
        mean_value = None
        if isinstance(payload, dict):
            mean_value = payload.get("mean_of_medians")
        if mean_value is None:
            _LOGGER.warning(
                "Skipping channel %s in %s: missing mean_of_medians",
                channel,
                summary_path,
            )
            continue
        try:
            overrides[str(channel)] = float(mean_value)
        except (TypeError, ValueError):
            _LOGGER.warning(
                "Skipping channel %s in %s: mean_of_medians=%r is not numeric",
                channel,
                summary_path,
                mean_value,
            )
    return overrides


def apply_median_summary_override(
    fitting_config: FittingConfig,
    median_summary: dict[str, float],
    tile_name: str,
    *,
    is_binned_channel: bool,
    binned_channel: str | None,
) -> None:
    """Update global correction factors using the summary file when possible."""

    if not median_summary:
        return

    override_value: float | None = None
    binned_key = str(binned_channel) if binned_channel else None

    if is_binned_channel and binned_key:
        override_value = median_summary.get(binned_key)
    elif not is_binned_channel:
        tile_lower = tile_name.lower()
        for channel, value in median_summary.items():
            if channel == binned_key:
                continue
            if channel.lower() in tile_lower:
                override_value = value
                break
        if override_value is None:
            for channel, value in median_summary.items():
                if channel != binned_key:
                    override_value = value
                    break

    if override_value is None:
        return

    if is_binned_channel:
        if fitting_config.global_factor_binned != override_value:
            _LOGGER.info(
                "Overriding binned global factor with %.4f", override_value
            )
        fitting_config.global_factor_binned = override_value
    else:
        if fitting_config.global_factor_unbinned != override_value:
            _LOGGER.info(
                "Overriding unbinned global factor with %.4f", override_value
            )
        fitting_config.global_factor_unbinned = override_value
