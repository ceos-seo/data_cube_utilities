import unittest

from datetime import datetime
import numpy as np
import xarray as xr

from data_cube_utilities.dc_mosaic import create_mosaic


class TestMosaic(unittest.TestCase):

    def setUp(self):
        self.times = [
            datetime(1999, 5, 6),
            datetime(2006, 1, 2),
            datetime(2006, 1, 16),
            datetime(2015, 12, 31),
            datetime(2016, 1, 1),
        ]

        self.latitudes = [1, 2]
        self.longitudes = [1, 2]

    def tearDown(self):
        pass

    def test_create_mosaic(self):
        data = np.array([[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]], [[4, 4], [4, 4]], [[5, 5], [5, 5]]])

        sample_clean_mask = np.array([[[True, True], [False, False]], [[True, False], [False, False]],
                                      [[False, False], [True, False]], [[False, False], [True, False]],
                                      [[True, True], [False, False]]])

        dataset = xr.Dataset(
            {
                'test_data': (('time', 'latitude', 'longitude'), data)
            },
            coords={'time': self.times,
                    'latitude': self.latitudes,
                    'longitude': self.longitudes})

        mosaic_dataset = create_mosaic(dataset, clean_mask=sample_clean_mask, no_data=-9999)
        mosaic_dataset_reversed = create_mosaic(dataset, clean_mask=sample_clean_mask, no_data=-9999, reverse_time=True)

        self.assertTrue((mosaic_dataset.test_data.values == np.array([[1, 1], [3, -9999]])).all())
        self.assertTrue((mosaic_dataset_reversed.test_data.values == np.array([[5, 5], [4, -9999]])).all())

    def test_create_mean_mosaic(self):
        pass

    def test_create_median_mosaic(self):
        pass

    def test_create_max_ndvi_mosaic(self):
        pass

    def test_create_min_ndvi_mosaic(self):
        pass
