from datetime import datetime

from aind_data_schema.core.processing import DataProcess
from aind_data_schema_models.process_names import ProcessName

from exaspim_flatfield_correction.utils.utils import get_parent_s3_path


def create_processing_metadata(
    args, tile_path: str, output_path: str, start_date_time: datetime, res: str
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
