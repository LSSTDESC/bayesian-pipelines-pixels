import os
from typing import List, Optional

import h5py
from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5


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
    if os.path.exists(filename) and not overwrite:
        raise ValueError(f"File {filename} exists and overwrite is False.")
    with h5py.File(filename, "w") as f:
        for key, value in data.items():
            if key == "catalog":
                write_table_hdf5(
                    value,
                    f,
                    path=key,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            else:
                f.create_dataset(
                    key,
                    data=value,
                    compression=compression,
                    compression_opts=compression_opts,
                )


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
    with h5py.File(filename, "r") as f:
        if keys is None:
            keys = list(f.keys())
        data = {key: f[key][()] for key in keys}
    return data
