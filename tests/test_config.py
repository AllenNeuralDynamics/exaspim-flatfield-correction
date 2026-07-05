from __future__ import annotations

import json

import pytest

from exaspim_flatfield_correction.config import (
    IOConcurrencyConfig,
    PipelineConfig,
    load_pipeline_config,
)


def _sample_pipeline_config() -> dict[str, object]:
    return {
        "method": "fitting",
        "tile_paths": [
            (
                "s3://aind-open-data/"
                "exaSPIM_826506_2026-05-19_14-00-17_processed_"
                "2026-06-04_14-18-27/denoised/SPIM.ome.zarr/"
                "tile_000001_ch_488.zarr"
            ),
            (
                "s3://aind-open-data/"
                "exaSPIM_826506_2026-05-19_14-00-17_processed_"
                "2026-06-04_14-18-27/denoised/SPIM.ome.zarr/"
                "tile_000001_ch_561.zarr"
            ),
        ],
        "output": (
            "s3://aind-open-data/"
            "exaSPIM_826506_2026-05-19_14-00-17_processed_"
            "2026-06-04_14-18-27/flatfield_correction/SPIM.ome.zarr"
        ),
        "binned_channel": "561",
        "res": 0,
        "skip_flat_field": False,
        "skip_bkg_sub": False,
        "mask_dir": "/data/masks",
        "fitting_config": "/data/fitting_config.json",
        "worker_mode": "processes",
        "background_smoothing_sigma": 3.0,
        "num_workers": 16,
        "n_levels": 7,
        "results_dir": "/results",
        "save_outputs": True,
        "overwrite": True,
    }


def test_pipeline_config_validates_sample_and_normalizes_res() -> None:
    config = PipelineConfig.model_validate(_sample_pipeline_config())

    assert config.res == "0"
    assert config.method == "fitting"
    assert config.tile_paths[0].endswith("tile_000001_ch_488.zarr")
    assert config.output.endswith("flatfield_correction/SPIM.ome.zarr")


def test_load_pipeline_config_from_file(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_sample_pipeline_config()))

    config = load_pipeline_config(config_path)

    assert config.res == "0"
    assert config.num_workers == 16


@pytest.mark.parametrize("extra_key", ["output_zarr", "resolution"])
def test_pipeline_config_rejects_legacy_schema_keys(extra_key: str) -> None:
    payload = _sample_pipeline_config()
    payload[extra_key] = "legacy-value"

    with pytest.raises(ValueError, match=extra_key):
        PipelineConfig.model_validate(payload)


def test_pipeline_config_rejects_negative_final_background_smoothing() -> None:
    payload = _sample_pipeline_config()
    payload["background_final_smoothing_sigma"] = -1

    with pytest.raises(
        ValueError,
        match="background_final_smoothing_sigma",
    ):
        PipelineConfig.model_validate(payload)


def test_pipeline_config_requires_mask_dir_for_fitting() -> None:
    payload = _sample_pipeline_config()
    payload.pop("mask_dir")

    with pytest.raises(ValueError, match="mask_dir"):
        PipelineConfig.model_validate(payload)


def test_pipeline_config_rejects_active_fitting_without_background() -> None:
    payload = _sample_pipeline_config()
    payload["skip_bkg_sub"] = True

    with pytest.raises(ValueError, match="background subtraction"):
        PipelineConfig.model_validate(payload)


def test_pipeline_config_requires_flatfield_for_active_reference() -> None:
    payload = _sample_pipeline_config()
    payload["method"] = "reference"
    payload.pop("mask_dir")

    with pytest.raises(ValueError, match="flatfield_path"):
        PipelineConfig.model_validate(payload)


def test_pipeline_config_infers_binned_channel_when_marked_binned() -> None:
    payload = _sample_pipeline_config()
    payload.pop("binned_channel")
    payload["is_binned"] = True

    config = PipelineConfig.model_validate(payload)

    assert config.binned_channel == "488"


def test_pipeline_config_defaults_for_zarr_v3_fields() -> None:
    config = PipelineConfig.model_validate(_sample_pipeline_config())

    assert config.io_backend == "tensorstore"
    assert config.output_zarr_format == "match"
    assert config.corrected_rank == -2


def test_pipeline_config_accepts_zarr_v3_field_overrides() -> None:
    payload = _sample_pipeline_config()
    payload["io_backend"] = "zarr"
    payload["output_zarr_format"] = "3"
    payload["corrected_rank"] = 0

    config = PipelineConfig.model_validate(payload)

    assert config.io_backend == "zarr"
    assert config.output_zarr_format == "3"
    assert config.corrected_rank == 0


@pytest.mark.parametrize(
    "field, value",
    [
        ("io_backend", "numpy"),
        ("output_zarr_format", "5"),
    ],
)
def test_pipeline_config_rejects_invalid_zarr_v3_fields(field, value) -> None:
    payload = _sample_pipeline_config()
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        PipelineConfig.model_validate(payload)


def test_io_concurrency_defaults_to_four_on_every_knob() -> None:
    config = PipelineConfig.model_validate(_sample_pipeline_config())

    io = config.io_concurrency
    assert io.zarr_async_concurrency == 8
    assert io.zarr_threading_max_workers == 8
    assert io.tensorstore_data_copy_concurrency == 8
    assert io.tensorstore_file_io_concurrency == 8
    assert io.tensorstore_s3_request_concurrency == 8
    assert io.tensorstore_http_request_concurrency == 8


def test_io_concurrency_converts_to_zarr_io_dataclass() -> None:
    converted = IOConcurrencyConfig().to_io_concurrency()

    # All six knobs populated -> both backend spec builders are fully configured.
    assert converted.zarr_config() == {
        "async.concurrency": 8,
        "threading.max_workers": 8,
    }
    assert converted.tensorstore_context_spec() == {
        "data_copy_concurrency": {"limit": 8},
        "file_io_concurrency": {"limit": 8},
        "s3_request_concurrency": {"limit": 8},
        "http_request_concurrency": {"limit": 8},
    }


def test_io_concurrency_accepts_overrides_and_null() -> None:
    payload = _sample_pipeline_config()
    payload["io_concurrency"] = {
        "tensorstore_s3_request_concurrency": 32,
        "zarr_async_concurrency": None,
    }

    config = PipelineConfig.model_validate(payload)

    assert config.io_concurrency.tensorstore_s3_request_concurrency == 32
    # null -> fall back to the backend library default for that single limit.
    assert config.io_concurrency.zarr_async_concurrency is None
    assert config.io_concurrency.to_io_concurrency().zarr_config() == {
        "threading.max_workers": 8,
    }


def test_io_concurrency_rejects_unknown_and_nonpositive_knobs() -> None:
    bad_key = _sample_pipeline_config()
    bad_key["io_concurrency"] = {"bogus_knob": 4}
    with pytest.raises(ValueError, match="bogus_knob"):
        PipelineConfig.model_validate(bad_key)

    bad_value = _sample_pipeline_config()
    bad_value["io_concurrency"] = {"tensorstore_file_io_concurrency": 0}
    with pytest.raises(ValueError, match="tensorstore_file_io_concurrency"):
        PipelineConfig.model_validate(bad_value)
