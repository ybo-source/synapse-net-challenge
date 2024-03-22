import os


def get_data_path(folder, n_tomograms=1):
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
