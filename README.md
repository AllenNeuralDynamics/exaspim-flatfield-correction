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
exaspim-flatfield-run \
  --zarr s3://bucket/tile_00001.ome.zarr \
  --output s3://bucket/corrected/tile_00001.ome.zarr \
  --method fitting \
  --mask-dir /path/to/masks \
  --num-workers 8 \
  --n-levels 4
```

Essential arguments:
- `--zarr`: input tile in OME-Zarr format (local path or S3 URI). When omitted, the pipeline searches for `../data/tile_*.json` metadata.
- `--output`: destination OME-Zarr path for the corrected volume.
- `--method`: `reference`, `basicpy`, or `fitting`. The fitting workflow requires `--mask-dir`; the reference workflow expects `--flatfield-path`.
- `--res`: resolution key to process (default `0`).

Frequently used options:
- `--num-workers`: number of Dask workers to launch (one thread per worker).
- `--results-dir`: directory for logs, diagnostics, and exported artifacts (defaults to `./results`).
- `--save-outputs`: persist intermediate TIFFs and plots inside `results/`.
- `--n-levels`: number of multiscale pyramid levels to generate for outputs.
- `--use-reference-bkg`: reuse precomputed background instead of estimating from the data.
- `--fitting-config`: JSON file with overrides for mask refinement, percentile weights, spline smoothing, and global normalization.
- `--median-summary-path`: per-channel normalization overrides produced by upstream statistics.
- `--is-binned`: flag tiles that correspond to a binned channel so the pipeline adjusts resolution handling.

The pipeline honours the `CO_MEMORY` environment variable (bytes) when sizing the Dask cluster and expects AWS credentials in the environment when reading or writing S3 paths.

## Outputs
Each tile run writes corrected data to the requested `--output` location and emits auxiliary OME-Zarr artifacts alongside it:
- `mask/<tile>/0/…`: upsampled foreground mask used for fitting.
- `probability/<tile>/0/…`: percentile-based probability weights when GMM refinement is enabled in the fitting configuration.
- `results/`: pipeline metadata, diagnostics, and optional cached intermediates.

## Notebooks
Interactive examples live under `notebooks/`. `notebooks/probability_weight_demo.ipynb` demonstrates loading a tile, estimating probability weights, and visualising raw slices next to their weighting volume.

## Development
Install in editable mode with `pip install -e .[basicpy]`, set up your preferred environment, and run tests or linting as needed. Console scripts resolve relative to the `code/` package directory, and the `run` shell helper used in the Code Ocean capsule delegates to the same pipeline entry point.
