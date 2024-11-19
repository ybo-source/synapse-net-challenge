import os

from typing import Callable, List, Optional, Sequence, Union
import mrcfile
from napari.types import LayerData

from elf.io import open_file, is_dataset

PathLike = str
PathOrPaths = Union[PathLike, Sequence[PathLike]]
ReaderFunction = Callable[[PathOrPaths], List[LayerData]]


def get_reader(path: PathOrPaths) -> Optional[ReaderFunction]:
    # If we recognize the format, we return the actual reader function.
    if isinstance(path, str) and path.endswith((".mrc", ".rec", ".h5")):
        return read_image_volume
    # Otherwise we return None.
    return None


# For mrcfiles we just read the data from it.
def _read_mrc(path, fname):
    with open_file(path, mode="r") as f:
        data = f["data"][:]
    
    metadata = {
        "file_path": path
    }
    read_voxel_size(path, metadata)
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


def read_voxel_size(input_path: str, layer_attributes: dict) -> None:
    with mrcfile.open(input_path) as mrc:
        try:
            voxel_size = mrc.voxel_size
            layer_attributes["voxel_size"] = {
                "x": voxel_size.x,
                "y": voxel_size.y,
                "z": voxel_size.z,
            }
        except Exception as e:
            print(f"Failed to read voxel size: {e}")
        # return mrc.voxel_size

# if __name__ == "__main__":
#     read_image_volume("/home/freckmann15/data/mitochondria/fidi_orig/20240722_WT/36859_J1_66K_TS_PS_01_rec_2kb1dawbp_crop_raw.mrc")