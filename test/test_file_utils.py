import unittest

import numpy as np
from elf.io import open_file
from synapse_net.sample_data import get_sample_data


class TestFileUtils(unittest.TestCase):

    def test_read_mrc_2d(self):
        from synapse_net.file_utils import read_mrc

        file_path = get_sample_data("tem_2d")
        data, voxel_size = read_mrc(file_path)

        with open_file(file_path, "r") as f:
            data_exp = f["data"][:]

        self.assertTrue(data.shape, data_exp.shape)
        self.assertTrue(np.allclose(data, data_exp))

        resolution = 0.592
        self.assertTrue(np.isclose(voxel_size["x"], resolution))
        self.assertTrue(np.isclose(voxel_size["y"], resolution))
        self.assertTrue(np.isclose(voxel_size["z"], 0.0))

    def test_read_mrc_3d(self):
        from synapse_net.file_utils import read_mrc

        file_path = get_sample_data("tem_tomo")
        data, voxel_size = read_mrc(file_path)

        with open_file(file_path, "r") as f:
            data_exp = f["data"][:]

        self.assertTrue(data.shape, data_exp.shape)
        self.assertTrue(np.allclose(data, data_exp))

        resolution = 1.554
        self.assertTrue(np.isclose(voxel_size["x"], resolution))
        self.assertTrue(np.isclose(voxel_size["y"], resolution))
        self.assertTrue(np.isclose(voxel_size["z"], resolution))


if __name__ == "__main__":
    unittest.main()
