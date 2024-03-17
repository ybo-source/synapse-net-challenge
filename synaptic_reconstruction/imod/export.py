import shutil
import tempfile
from subprocess import run

import imageio.v3 as imageio
from elf.io import open_file


def export_segmentation(imod_path, mrc_path, object_id=None, output_path=None, require_object=True):
    cmd = "imodmop"
    cmd_path = shutil.which(cmd)
    assert cmd_path is not None, f"Could not find the {cmd} imod command."

    with tempfile.NamedTemporaryFile() as f:
        tmp_path = f.name

        if object_id is None:
            cmd = [cmd, "-ma", "1", imod_path, mrc_path, tmp_path]
        else:
            cmd = [cmd, "-ma", "1", "-o", str(object_id), imod_path, mrc_path, tmp_path]

        run(cmd)
        with open_file(tmp_path, ext=".mrc", mode="r") as f:
            data = f["data"][:]

    segmentation = data == 1
    if require_object and segmentation.sum() == 0:
        id_str = "" if object_id is None else f"for object {object_id}"
        raise RuntimeError(f"Segmentation extracted from {imod_path} {id_str} is empty.")

    if output_path is None:
        return segmentation

    imageio.imwrite(output_path, segmentation.astype("uint8"), compression="zlib")
