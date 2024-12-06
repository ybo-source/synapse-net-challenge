import unittest

import numpy as np


class TestDistanceMeasurement(unittest.TestCase):
    def test_measure_pairwise_object_distances(self):
        from synapse_net.distance_measurements import measure_pairwise_object_distances

        shape = (4, 64, 64)
        seg = np.zeros(shape, dtype="uint32")

        seg[1, 16, 0] = 1
        seg[1, 16, 16] = 2
        seg[1, 16, 32] = 3
        seg[1, 16, 48] = 4
        seg[1, 16, 63] = 5

        for resolution in (None, 2.3, 4.4):
            distances, _, _, seg_ids = measure_pairwise_object_distances(seg, resolution=resolution, n_threads=1)

            factor = 1 if resolution is None else resolution

            self.assertTrue(np.isclose(distances[0, 1], factor * 16))  # distance between object 1 and 2
            self.assertTrue(np.isclose(distances[0, 2], factor * 32))  # distance between object 1 and 3
            self.assertTrue(np.isclose(distances[0, 3], factor * 48))  # distance between object 1 and 4
            self.assertTrue(np.isclose(distances[0, 4], factor * 63))  # distance between object 1 and 5

            self.assertTrue(np.isclose(distances[1, 2], factor * 16))  # distance between object 2 and 3
            self.assertTrue(np.isclose(distances[1, 3], factor * 32))  # distance between object 2 and 4
            self.assertTrue(np.isclose(distances[1, 4], factor * 47))  # distance between object 2 and 5

            self.assertTrue(np.isclose(distances[2, 3], factor * 16))  # distance between object 3 and 4
            self.assertTrue(np.isclose(distances[2, 4], factor * 31))  # distance between object 3 and 5

            self.assertTrue(np.isclose(distances[3, 4], factor * 15))  # distance between object 3 and 5


if __name__ == "__main__":
    unittest.main()
