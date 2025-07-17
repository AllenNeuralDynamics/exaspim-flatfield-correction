from pathlib import Path
from functools import partial
import logging
from typing import Callable

import zarr
from zarr.errors import ContainsGroupError
import numpy as np
import dask.array as da
import s3fs

from aind_data_transfer.transformations.ome_zarr import (
    downsample_and_store,
    store_array,
    write_ome_ngff_metadata,
)
from numcodecs import blosc
blosc.use_threads = False
from xarray_multiscale.reducers import windowed_rank

_LOGGER = logging.getLogger(__name__)


def store_ome_zarr(
    corrected: da.Array,
    output_zarr: str,
    n_levels: int = 1,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0),
    overwrite: bool = False,
    reducer: Callable = windowed_rank,
    aws_region: str = "us-west-2",
    s3_use_ssl: bool = False,
    # write_empty_chunks: bool = True
) -> None:
    """
    Store a Dask array as an OME-Zarr multiscale dataset, with optional S3
    support and metadata.

    Parameters
    ----------
    corrected : dask.array.Array
        The image data to store (will be expanded to 5D if needed).
    output_zarr : str
        Path or S3 URL to the output zarr group.
    n_levels : int, optional
        Number of pyramid levels to generate. Default is 1.
    voxel_size : tuple of float, optional
        Physical voxel size (ZYX) for metadata. Default is (1.0, 1.0, 1.0).
    origin : tuple of int, optional
        Origin for the OME-NGFF metadata. Default is (0, 0, 0, 0, 0).
    overwrite : bool, optional
        Whether to overwrite existing data. Default is False.
    reducer : Callable, optional
        Function to use for downsampling. Should accept (array, ...) and
        return downsampled array. Default is windowed_rank.
    aws_region : str, optional
        AWS region name for S3 access. Default is 'us-west-2'.
    s3_use_ssl : bool, optional
        Whether to use SSL for S3 access. Default is False.

    Returns
    -------
    None
    """
    if output_zarr.startswith("s3://"):
        s3 = s3fs.S3FileSystem(
            anon=False,
            client_kwargs={
                "region_name": aws_region,
            },
            config_kwargs={
                "s3": {
                    "multipart_threshold": 256
                    * 1024
                    * 1024,  # 256 MB, avoid multipart upload for small chunks
                },
                "retries": {
                    "total_max_attempts": 10,
                    "mode": "adaptive",
                },
            },
            use_ssl=s3_use_ssl,
            s3_additional_kwargs = {"batch_size": 64},
        )
        # Create a Zarr group on S3
        store = s3fs.S3Map(root=output_zarr, s3=s3, check=False)
    else:
        store = zarr.DirectoryStore(output_zarr)

    mode = "w" if overwrite else "w-"
    try:
        root_group = zarr.open_group(store=store, mode=mode)
    except ContainsGroupError:
        _LOGGER.info(f"Not overwriting tile {output_zarr}")
        return

    codec = blosc.Blosc(cname="zstd", clevel=1, shuffle=blosc.SHUFFLE)

    scale_factors = (1, 1, 2, 2, 2)
    # TODO: hardcoded block shape, should be configurable
    block_shape = (1, 1, 4096, 4096, 4096)

    while corrected.ndim < 5:
        corrected = corrected[np.newaxis, ...]

    _LOGGER.info("storing array")
    store_array(
        corrected, 
        root_group, 
        "0", 
        block_shape, 
        codec, 
        # write_empty_chunks=write_empty_chunks
    )
    _LOGGER.info("downsampling array")
    if reducer == windowed_rank:
        reducer = partial(reducer, rank=-2)
    downsample_and_store(
        corrected,
        root_group,
        n_levels,
        scale_factors,
        block_shape,
        codec,
        reducer,
        # write_empty_chunks=write_empty_chunks
    )
    _LOGGER.info("writing ome metadata")
    write_ome_ngff_metadata(
        root_group,
        corrected,
        Path(output_zarr).stem,
        n_levels,
        scale_factors[-3:],  # must be 3D
        voxel_size,  # must be 3D ZYX
        origin,
    )


def parse_ome_zarr_transformations(z: zarr.hierarchy.Group, res: str) -> dict:
    """
    Parse scale and translation transformations for a given resolution in
    an OME-Zarr dataset.

    Parameters
    ----------
    z : zarr.hierarchy.Group
        Opened zarr group for the dataset.
    res : str
        Resolution key to extract transformations for.

    Returns
    -------
    dict
        Dictionary containing 'scale' and 'translation' for the
        specified resolution.

    Raises
    ------
    ValueError
        If OME-Zarr metadata is not found.
    """

    # Read the metadata from .zattrs
    try:
        metadata = z.attrs.asdict()
    except KeyError:
        raise ValueError("OME-Zarr metadata not found.")

    res = str(res)

    # Extract transformations for the first dataset
    transformations = {}
    multiscales = metadata.get("multiscales", [])

    if multiscales:
        datasets = multiscales[0].get("datasets", [])
        for ds in datasets:
            if ds["path"] == res:
                coord_transforms = ds.get("coordinateTransformations", [])
                scale = next(
                    (
                        t["scale"]
                        for t in coord_transforms
                        if t["type"] == "scale"
                    ),
                    None,
                )
                translation = next(
                    (
                        t["translation"]
                        for t in coord_transforms
                        if t["type"] == "translation"
                    ),
                    None,
                )
                transformations = {"scale": scale, "translation": translation}
                break

    return transformations


def get_zarr_tiles(
    z, res: int = 5, chunks: tuple[int, ...] = None
) -> list[da.Array]:
    """
    Extract tiles from a zarr group at a given resolution as Dask arrays.

    Parameters
    ----------
    z : zarr.hierarchy.Group
        Zarr group containing tiles.
    res : int, optional
        Resolution level to extract. Default is 5.
    chunks : tuple of int, optional
        Chunk size for the Dask arrays.
        If None, use the dataset's default chunks.

    Returns
    -------
    list of dask.array.Array
        List of Dask arrays for each tile at the specified resolution.
    """
    tiles = []
    for tile in sorted(z.keys()):
        ds = z[tile][res]
        if chunks is None:
            chunks = ds.chunks
        data = da.from_array(ds, chunks=chunks).squeeze()
        tiles.append(data)
    return tiles
