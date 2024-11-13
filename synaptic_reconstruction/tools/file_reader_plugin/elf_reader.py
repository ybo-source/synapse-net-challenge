from typing import Callable, List, Optional, Sequence, Union
from napari.types import LayerData
from elf.io import open_file

PathLike = str
PathOrPaths = Union[PathLike, Sequence[PathLike]]
ReaderFunction = Callable[[PathOrPaths], List[LayerData]]


def get_reader(path: PathOrPaths) -> Optional[ReaderFunction]:
    # If we recognize the format, we return the actual reader function
    if isinstance(path, str) and path.endswith(".mrc"):
        return elf_read_file
    # otherwise we return None.
    return None


def elf_read_file(path: PathOrPaths) -> List[LayerData]:
    try:
        with open_file(path, mode="r") as f:
            data = f["data"][:]
        layer_attributes = {
            "name": "Raw",
            "colormap": "gray",
            "blending": "additive"
            }
        return [(data, layer_attributes)]
    except Exception as e:
        print(f"Failed to read file: {e}")
        return
