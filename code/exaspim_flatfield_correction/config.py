"""Configuration helpers for flatfield correction."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from exaspim_flatfield_correction.utils.utils import (
    extract_channel_from_tile_name,
)

_LOGGER = logging.getLogger(__name__)


class PipelineConfig(BaseModel):
    """Top-level configuration for the flatfield correction pipeline."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    tile_paths: list[str] = Field(
        min_length=1,
        description="One or more input OME-Zarr tile paths.",
    )
    output: str = Field(
        min_length=1,
        description="Output OME-Zarr path for corrected data.",
    )
    save_outputs: bool = Field(
        default=False,
        description="Save intermediate output files.",
    )
    results_dir: str = Field(
        default_factory=lambda: os.path.join(os.getcwd(), "results"),
        description="Directory for results, diagnostics, and metadata.",
    )
    res: str = Field(
        default="0",
        min_length=1,
        description="Resolution level to process.",
    )
    method: Literal["basicpy", "reference", "fitting"] = Field(
        default="fitting",
        description="Flatfield correction method.",
    )
    overwrite: bool = Field(
        default=False,
        description="Overwrite existing outputs.",
    )
    skip_flat_field: bool = Field(
        default=False,
        description="Skip flatfield correction.",
    )
    skip_bkg_sub: bool = Field(
        default=False,
        description="Skip background subtraction.",
    )
    flatfield_path: str | None = Field(
        default=None,
        description="Reference flatfield image path.",
    )
    mask_dir: str | None = Field(
        default=None,
        description="Directory containing fitting masks.",
    )
    fitting_config: str | None = Field(
        default=None,
        description="Path to fitting configuration overrides.",
    )
    num_workers: int = Field(
        default=1,
        ge=1,
        description="Number of Dask workers to launch.",
    )
    worker_mode: Literal["processes", "threads"] = Field(
        default="processes",
        description="Dask worker execution mode.",
    )
    log_level: int | str = Field(
        default=logging.INFO,
        description="Python logging level for the pipeline logger.",
    )
    n_levels: int = Field(
        default=1,
        ge=1,
        description="Number of OME-Zarr pyramid levels to write.",
    )
    use_reference_bkg: bool = Field(
        default=False,
        description="Use a reference background image instead of estimating.",
    )
    background_smoothing_sigma: float = Field(
        default=1.0,
        ge=0,
        description="Gaussian sigma applied before background estimation.",
    )
    background_final_smoothing_sigma: float = Field(
        default=0.0,
        ge=0,
        description="Gaussian sigma applied to the final background image.",
    )
    binned_channel: str | None = Field(
        default=None,
        description="Substring identifying binned channel tiles.",
    )
    is_binned: bool = Field(
        default=False,
        description="Treat all configured tiles as binned channel data.",
    )
    median_summary_path: str | None = Field(
        default=None,
        description="Path to per-channel median intensity summary JSON.",
    )

    @field_validator("res", mode="before")
    @classmethod
    def _normalize_res(cls, value: Any) -> str:
        if value in (None, ""):
            raise ValueError("res cannot be empty")
        return str(value)

    @model_validator(mode="after")
    def _validate_method_requirements(self) -> "PipelineConfig":
        if self.method == "fitting" and self.mask_dir is None:
            raise ValueError("mask_dir is required when method='fitting'")
        if (
            self.method == "fitting"
            and not self.skip_flat_field
            and self.skip_bkg_sub
        ):
            raise ValueError(
                "active fitting correction requires background subtraction"
            )
        if (
            self.method == "reference"
            and not self.skip_flat_field
            and self.flatfield_path is None
        ):
            raise ValueError(
                "flatfield_path is required when active method='reference'"
            )

        if self.is_binned and self.binned_channel is None:
            tile_name = Path(self.tile_paths[0]).name
            inferred_channel = extract_channel_from_tile_name(tile_name)
            if inferred_channel is not None:
                self.binned_channel = inferred_channel
            else:
                _LOGGER.warning(
                    "Unable to infer channel number from tile %s despite "
                    "is_binned=true",
                    tile_name,
                )
        return self


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
    global_ratio_limits: tuple[float, float] | None = Field(
        default=None,
        description=(
            "Optional lower/upper clamp applied to the global correction "
            "ratio (global_factor / global_med); set to null to disable."
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
        default=0.0001,
        gt=0,
        lt=0.5,
        description=(
            "Logistic ramp value assigned at the low anchor percentile"
        ),
    )
    probability_ramp_start_frac: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description=(
            "Fraction along the percentile span where the logistic ramp "
            "begins; e.g. 0.2 means the ramp starts 20% of the way from the low to "
            "the high percentile."
        ),
    )
    probability_ramp_nu: float = Field(
        default=1.0,
        ge=1.0,
        description=(
            "Controls the skew of the logistic ramp; nu=1 matches a standard "
            "logistic, while larger values skew the ramp toward the high "
            "percentile."
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

    @model_validator(mode="after")
    def _validate_global_ratio_limits(self) -> "FittingConfig":
        limits = self.global_ratio_limits
        if limits is not None:
            lower, upper = limits
            if lower >= upper:
                raise ValueError(
                    "global_ratio_limits must be ordered as (min, max) with min < max"
                )
        return self


def load_fitting_config(path: str | Path | None = None) -> FittingConfig:
    """Return either a default or file-backed fitting configuration."""

    if path is None:
        return FittingConfig()
    return FittingConfig.from_file(path)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load and validate the top-level pipeline configuration."""

    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(f"Config file not found: {config_path}")
    raw = config_path.read_text()
    data: Any = json.loads(raw)
    return PipelineConfig.model_validate(data)


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
