import os
from typing import List, Optional

import h5py

allowed_params = {
    "catalog",
    "image",
    "psf_image",
    "sky_level",
    "pixel_scale",
    "g1",
    "g2",
    "seed",
}


def write_data_to_hdf5_file(
    filename: str,
    data: dict,
    overwrite: bool = False,
    compression: str = "gzip",
    compression_opts: int = 9,
):
    """Write data to an hdf5 file.
    Args:
        filename: Name of the file to write to.
        data: Dictionary of data to write to file.
        overwrite: Overwrite existing file?
        compression: Compression to use.
        compression_opts: Compression level.
    """

    assert set(data.keys()).issubset(allowed_params), "Invalid keys."
    if os.path.exists(filename) and not overwrite:
        raise ValueError(f"File {filename} exists and overwrite is False.")
    h5_kwargs = {
        "compression": compression,
        "compression_opts": compression_opts,
    }
    with h5py.File(filename, "w") as f:
        catalog_grp = f.create_group("catalog")
        for key in data["catalog"].keys():
            val = data["catalog"][key]
            catalog_grp.create_dataset(key, data=val, **h5_kwargs)
        for key in data.keys():
            if key != "catalog":
                f.create_dataset(key, data=data[key], **h5_kwargs)


def load_data_from_hdf5_file(
    filename: str,
    keys: Optional[List[str]] = None,
):
    """Load data from an hdf5 file.
    Args:
        filename: Name of the file to load from.
        keys: List of keys to load. If None, load all keys.
    Returns:
        Dictionary of data.
    """
    if keys is None:
        keys = allowed_params
    data = {}
    with h5py.File(filename, "r") as f:
        for key in keys:
            if key == "catalog":
                data[key] = {}
                for k in f[key].keys():
                    data[key][k] = f[key][k][:]
            else:
                data[key] = f[key][:]
    return data
