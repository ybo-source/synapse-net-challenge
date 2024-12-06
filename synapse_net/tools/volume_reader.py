import os
from typing import Callable, List, Optional, Sequence, Union

from elf.io import open_file, is_dataset
from napari.types import LayerData
from synapse_net.file_utils import read_mrc


PathLike = str
PathOrPaths = Union[PathLike, Sequence[PathLike]]
ReaderFunction = Callable[[PathOrPaths], List[LayerData]]


def get_reader(path: PathOrPaths) -> Optional[ReaderFunction]:
    # If we recognize the format, we return the actual reader function.
    if isinstance(path, str) and path.endswith((".mrc", ".rec", ".h5")):
        return read_image_volume
    # Otherwise we return None.
    return None


def _read_mrc(path, fname):
    data, voxel_size = read_mrc(path)
    metadata = {"file_path": path, "voxel_size": voxel_size}
    layer_attributes = {
        "name": fname,
        "colormap": "gray",
        "metadata": metadata
    }
    return [(data, layer_attributes)]


# For hdf5 files we read the full content.
def _read_hdf5(path):
    return_data = []

    def visitor(name, obj):
        if is_dataset(obj):
            data = obj[:]
            attributes = {"name": name}
            if str(data.dtype) in ("int32", "uint32", "int64", "uint64"):
                layer_type = "labels"
            else:
                layer_type = "image"
                attributes["colormap"] = "gray"

            return_data.append((data, attributes, layer_type))

    with open_file(path, mode="r") as f:
        f.visititems(visitor)

    return return_data


def read_image_volume(path: PathOrPaths) -> List[LayerData]:
    fname = os.path.basename(path)
    fname, ext = os.path.splitext(fname)

    try:
        if ext in (".mrc", ".rec"):
            return _read_mrc(path, fname)
        else:   # This is an hdf5 file
            return _read_hdf5(path)

    except Exception as e:
        print(f"Failed to read file: {e}")
        return
