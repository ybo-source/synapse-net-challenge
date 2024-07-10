import os

ROOT = "/home/pape/Work/data/cooper/for_correction"


def get_root(version):
    return os.path.join(ROOT, f"v{version}")
