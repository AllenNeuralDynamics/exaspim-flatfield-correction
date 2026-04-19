import json
from datetime import datetime
from pathlib import Path
from typing import Any

from aind_data_schema.components.identifiers import Code, DataAsset
from aind_data_schema.core.processing import DataProcess, ProcessStage
from aind_data_schema_models.process_names import ProcessName
from exaspim_flatfield_correction import __version__
from exaspim_flatfield_correction.utils.utils import get_parent_s3_path

_CODE_URL = "https://github.com/AllenNeuralDynamics/exaspim-flatfield-correction"
_DEFAULT_EXPERIMENTER = "unknown"


def _get_input_data(tile_path: str) -> list[DataAsset]:
    """Build schema-compatible input data identifiers for a tile path."""
    if tile_path.startswith("s3://"):
        input_location = get_parent_s3_path(tile_path)
    else:
        input_location = str(Path(tile_path).parent)
    return [DataAsset(url=input_location)]


def _serialize_parameter_value(value: Any) -> Any:
    """Convert Namespace values into JSON-friendly metadata primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_serialize_parameter_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _serialize_parameter_value(item) for key, item in value.items()
        }
    return value


def _get_code_parameters(args: Any, output_path: str, res: str) -> dict[str, Any]:
    """Build metadata for the resolved pipeline invocation."""
    parameters = {
        key: _serialize_parameter_value(value)
        for key, value in vars(args).items()
    }
    if parameters.get("res") in (None, ""):
        parameters["res"] = res
    parameters["output_path"] = output_path
    return parameters


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
        "process_type": ProcessName.IMAGE_FLAT_FIELD_CORRECTION,
        "stage": ProcessStage.PROCESSING,
        "code": Code(
            url=_CODE_URL,
            version=__version__,
            input_data=_get_input_data(tile_path),
            parameters=_get_code_parameters(args, output_path, res),
        ),
        "experimenters": [_DEFAULT_EXPERIMENTER],
        "start_date_time": start_date_time,
        "end_date_time": datetime.now(),
    }
    process = DataProcess(**processing_info)
    return process


def save_metadata(
    data_process: Any,
    out_path: str,
    tile_name: str,
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
