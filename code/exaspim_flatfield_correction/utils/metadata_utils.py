import json
from datetime import datetime
from pathlib import Path
from typing import Any

from aind_data_schema.core.processing import DataProcess
from aind_data_schema_models.process_names import ProcessName
from exaspim_flatfield_correction.utils.utils import get_parent_s3_path


def create_processing_metadata(
    args: Any,
    tile_path: str,
    output_path: str,
    start_date_time: datetime,
    res: str,
) -> DataProcess:
    """
    Create a DataProcess metadata object for flat field correction.

    Parameters
    ----------
    args : argparse.Namespace or similar
        Parsed command-line arguments or object with required attributes.
    tile_path : str
        Path to the input tile (zarr or image file).
    output_path : str
        Output path for the processed data.
    start_date_time : datetime
        Start time of the processing step.
    res : str
        Resolution string for the processing.

    Returns
    -------
    DataProcess
        A metadata object describing the flat field correction process.
    """
    processing_info = {
        "name": ProcessName.IMAGE_FLAT_FIELD_CORRECTION,
        "software_version": "0.0.1",
        "start_date_time": start_date_time,
        "end_date_time": datetime.now(),
        "input_location": get_parent_s3_path(tile_path),
        "output_location": output_path,
        "code_url": "https://codeocean.allenneuraldynamics.org/capsule/2321427/tree",
        "parameters": {
            "method": args.method,
            "skip_flat_field": args.skip_flat_field,
            "skip_bkg_sub": args.skip_bkg_sub,
            "resolution": res,
            "overwrite": args.overwrite,
            "flatfield_path": args.flatfield_path,
        },
        "outputs": {},
    }
    process = DataProcess(**processing_info)
    return process


def save_metadata(
    data_process: Any,
    out_path: str,
    tile_name: str,
    tile_path: str,
    results_dir: str,
) -> None:
    """Persist processing metadata alongside the corrected tile outputs.

    Parameters
    ----------
    data_process : Any
        Structured metadata object exposing ``model_dump_json``.
    out_path : str
        Target path of the corrected tile Zarr store.
    tile_name : str
        Identifier of the tile being processed.
    tile_path : str
        Source Zarr path for the tile.
    results_dir : str
        Directory in which metadata JSON artifacts are saved.
    """
    process_json = data_process.model_dump_json()
    process_json_path = str(
        Path(results_dir)
        / f"process_{Path(out_path).parent.name}_{tile_name}.json"
    )
    with open(process_json_path, "w") as f:
        f.write(process_json)

    input_metadata_path = get_parent_s3_path(get_parent_s3_path(tile_path))
    output_metadata_path = get_parent_s3_path(get_parent_s3_path(out_path))
    metadata_json_path = str(
        Path(results_dir)
        / f"metadata_paths_{Path(out_path).parent.name}_{tile_name}.json"
    )
    with open(metadata_json_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "input_metadata": input_metadata_path,
                    "output_metadata": output_metadata_path,
                }
            )
        )
