from torch_em.data.datasets.electron_microscopy.deepict import get_deepict_actin_data

root = "/mnt/lustre-grete/usr/u12086/data/deepict"
get_deepict_actin_data(root, download=False)
