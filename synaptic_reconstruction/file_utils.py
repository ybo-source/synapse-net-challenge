import os
from typing import List, Optional, Union


def get_data_path(folder: str, n_tomograms: Optional[int] = 1) -> Union[str, List[str]]:
    """Get the path to all tomograms stored as .rec or .mrc files in a folder.

    Args:
        folder: The folder with tomograms.
        n_tomograms: The expected number of tomograms.

    Returns:
        The filepath or list of filepaths of the tomograms in the folder.
    """
    file_names = os.listdir(folder)
    tomograms = []
    for fname in file_names:
        ext = os.path.splitext(fname)[1]
        if ext in (".rec", ".mrc"):
            tomograms.append(os.path.join(folder, fname))

    if n_tomograms is None:
        return tomograms
    assert len(tomograms) == n_tomograms, f"{folder}: {len(tomograms)}, {n_tomograms}"
    return tomograms[0] if n_tomograms == 1 else tomograms
