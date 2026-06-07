# ExASPIM Flatfield Correction

ExASPIM Flatfield Correction provides background subtraction and flatfield correction for large (TBs) 3D images. The library targets automated, high-throughput processing and ships a command-line pipeline plus supporting utilities and notebooks.

## Key Capabilities
- Background estimation with slice rejection tailored for fluorescence microscopy.
- Reference-, BaSiC-, and spline-fitting-based illumination correction.
- Optional probabilistic foreground refinement to improve intensity profile estimation.
- Dask-powered chunked execution with configurable worker counts and memory limits.

## Installation
- Python >= 3.11 is required.
- Install the package and core dependencies with `pip install .` from the repository root.
- To enable the BaSiC workflow, install the optional extra: `pip install .[basicpy]`.

## Command Line Interface
The main entry point is exposed as `exaspim-flatfield-run`. You can also invoke the module directly via `python -m exaspim_flatfield_correction.pipeline`.

```sh
exaspim-flatfield-run --config /data/config.json
```

All pipeline options are supplied through the JSON config. Unknown keys are rejected, so use the canonical field names shown below.

```json
{
  "method": "fitting",
  "tile_paths": [
    "s3://aind-open-data/exaSPIM_826506_2026-05-19_14-00-17_processed_2026-06-04_14-18-27/denoised/SPIM.ome.zarr/tile_000001_ch_488.zarr",
    "s3://aind-open-data/exaSPIM_826506_2026-05-19_14-00-17_processed_2026-06-04_14-18-27/denoised/SPIM.ome.zarr/tile_000001_ch_561.zarr"
  ],
  "output": "s3://aind-open-data/exaSPIM_826506_2026-05-19_14-00-17_processed_2026-06-04_14-18-27/flatfield_correction/SPIM.ome.zarr",
  "binned_channel": "561",
  "res": 0,
  "skip_flat_field": false,
  "skip_bkg_sub": false,
  "mask_dir": "/data/masks",
  "fitting_config": "/data/fitting_config.json",
  "worker_mode": "processes",
  "background_smoothing_sigma": 3.0,
  "num_workers": 16,
  "n_levels": 7,
  "results_dir": "/results",
  "save_outputs": true,
  "overwrite": true
}
```

Frequently used config fields:
- `tile_paths`: one or more input tile OME-Zarr paths.
- `output`: destination OME-Zarr path for corrected volumes.
- `method`: `reference`, `basicpy`, or `fitting`. The fitting workflow requires `mask_dir`; the reference workflow expects `flatfield_path`.
- `res`: resolution key to process; numbers are normalized to strings.
- `num_workers`: number of Dask workers to launch.
- `results_dir`: directory for logs, diagnostics, and exported artifacts.
- `save_outputs`: persist intermediate TIFFs and plots inside `results_dir`.
- `n_levels`: number of multiscale pyramid levels to generate for outputs.
- `use_reference_bkg`: reuse precomputed background instead of estimating from the data.
- `background_smoothing_sigma`: Gaussian smoothing sigma applied before background estimation; set to `0` to disable smoothing.
- `fitting_config`: JSON file with overrides for mask refinement, percentile weights, spline smoothing, and global normalization.
- `median_summary_path`: per-channel normalization overrides produced by upstream statistics.
- `is_binned`: flag all configured tiles as binned; when `binned_channel` is omitted, it is inferred from the first tile name when possible.

The pipeline honours the `CO_MEMORY` environment variable (bytes) when sizing the Dask cluster and expects AWS credentials in the environment when reading or writing S3 paths.

## Outputs
Each tile run writes corrected data to the requested `output` location and emits auxiliary OME-Zarr artifacts alongside it:
- `mask/<tile>/0/…`: upsampled foreground mask used for fitting.
- `probability/<tile>/0/…`: percentile-based probability weights when GMM refinement is enabled in the fitting configuration.
- `results/`: pipeline metadata, diagnostics, and optional cached intermediates.

## Notebooks
Interactive examples live under `notebooks/`. `notebooks/probability_weight_demo.ipynb` demonstrates loading a tile, estimating probability weights, and visualising raw slices next to their weighting volume.

## Development
Install in editable mode with `pip install -e .[basicpy]`, set up your preferred environment, and run tests or linting as needed. Console scripts resolve relative to the `code/` package directory, and the `run` shell helper used in the Code Ocean capsule delegates to the same pipeline entry point.
