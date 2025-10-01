# ExASPIM Flatfield Correction

ExASPIM Flatfield Correction is a Python library for robust flatfield and background correction of large-scale fluorescence microscopy datasets in Zarr format. It is designed to support high-throughput, automated correction workflows for multi-tile, multi-resolution volumetric imaging data, with a focus on reproducibility and scalability.

## Features
- Support for large (many Terabytes) 3D Zarr datasets (local or S3)
- Automated background estimation and illumination correction
- Dask-powered, chunked processing
- OME-NGFF metadata and multiscale pyramid output

## Supported Methods

The main pipeline supports the following correction methods:

- **reference**: Use provided reference flatfield and darkfield images (local or S3) to correct each tile.
- **basicpy**: Use the [BaSiC](https://github.com/CSBDeep/basicpy) algorithm to estimate flatfield and darkfield from the data itself.
- **fitting**: Axis-specific intesnity gradient correction via robust spline fitting followed by global inter-tile normalization

All methods support optional background subtraction and mask-based correction. The pipeline is highly configurable and can be run on local or cloud data.

## Installation

```sh
pip install .
```

## Usage

### Command Line

After installation, run the main pipeline with:

```sh
python -m exaspim_flatfield_correction.pipeline.run_pipeline --zarr <input_zarr> --output <output_zarr> --method <reference|basicpy|fitting> [options]
```

Key arguments:
- `--zarr`: Path to input Zarr (local or S3)
- `--output`: Output Zarr path
- `--method`: Correction method (`reference`, `basicpy`, or `fitting`)
- `--flatfield-path`: Path to reference flatfield (for `reference` method)
- `--mask-dir`: Directory with masks (for `fitting` method)
- `--skip-flat-field`: Skip flatfield correction
- `--skip-bkg-sub`: Skip background subtraction
- `--n-levels`: Number of Zarr pyramid levels (default: 7)

See `python -m exaspim_flatfield_correction.pipeline.run_pipeline --help` for all options.

You can also import and use individual correction functions for custom workflows.

## Example

```sh
python -m exaspim_flatfield_correction.pipeline.run_pipeline \
    --zarr s3://my-bucket/mydata.zarr \
    --output s3://my-bucket/corrected.zarr \
    --method fitting \
    --mask-dir ./masks \
    --n-levels 5
```

## Development
To install in editable mode:

```sh
pip install -e .
```
