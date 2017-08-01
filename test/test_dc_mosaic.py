import unittest

from datetime import datetime
import numpy as np
import xarray as xr

from data_cube_utilities.dc_mosaic import (create_mosaic, create_mean_mosaic, create_median_mosaic,
                                           create_max_ndvi_mosaic, create_min_ndvi_mosaic)


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

        # yapf: disable
        self.sample_clean_mask = np.array([[[True, True], [False, False]],
                                           [[True, False], [True, False]],
                                           [[False, False], [True, False]],
                                           [[False, True], [True, False]],
                                           [[True, True], [False, False]]])

        self.sample_data = np.array([[[1, 1], [1, 1]],
                                     [[2, 2], [2, 2]],
                                     [[3, 3], [3, 3]],
                                     [[4, 4], [4, 4]],
                                     [[5, 5], [5, 5]]])

        self.nir = np.array([[[0, 1], [0, 80]],
                             [[1, 4], [1, 60]],
                             [[0, 0], [2, 0]],
                             [[1, 5], [1, 20]],
                             [[2, 1], [1, 0]]])

        self.red = np.array([[[15, 1], [5, 1]],
                             [[1, 1], [1, 1]],
                             [[1, 5], [1, 1]],
                             [[1, 1], [1, 1]],
                             [[1, 1], [1, 4]]])
        # yapf: enable

    def tearDown(self):
        pass

    def test_create_mosaic(self):

        dataset = xr.Dataset(
            {
                'test_data': (('time', 'latitude', 'longitude'), self.sample_data)
            },
            coords={'time': self.times,
                    'latitude': self.latitudes,
                    'longitude': self.longitudes})

        mosaic_dataset = create_mosaic(dataset, clean_mask=self.sample_clean_mask, no_data=-9999)
        mosaic_dataset_reversed = create_mosaic(
            dataset, clean_mask=self.sample_clean_mask, no_data=-9999, reverse_time=True)

        self.assertTrue((mosaic_dataset.test_data.values == np.array([[1, 1], [2, -9999]])).all())
        self.assertTrue((mosaic_dataset_reversed.test_data.values == np.array([[5, 5], [4, -9999]])).all())

        self.assertTrue('time' not in mosaic_dataset)

        mosaic_dataset_iterated = create_mosaic(
            dataset,
            intermediate_product=mosaic_dataset,
            clean_mask=np.full(self.sample_clean_mask.shape, True),
            no_data=-9999)

        self.assertTrue((mosaic_dataset_iterated.test_data.values == np.array([[1, 1], [2, 1]])).all())

    def test_create_mean_mosaic(self):

        dataset = xr.Dataset(
            {
                'test_data': (('time', 'latitude', 'longitude'), self.sample_data)
            },
            coords={'time': self.times,
                    'latitude': self.latitudes,
                    'longitude': self.longitudes})

        mosaic_dataset = create_mean_mosaic(dataset, clean_mask=self.sample_clean_mask, no_data=-9999)

        self.assertTrue((mosaic_dataset.test_data.values == np.array([[2, 3], [3, -9999]])).all())

        self.assertTrue('time' not in mosaic_dataset)

    def test_create_median_mosaic(self):
        dataset = xr.Dataset(
            {
                'test_data': (('time', 'latitude', 'longitude'), self.sample_data)
            },
            coords={'time': self.times,
                    'latitude': self.latitudes,
                    'longitude': self.longitudes})

        mosaic_dataset = create_median_mosaic(dataset, clean_mask=self.sample_clean_mask, no_data=-9999)

        self.assertTrue((mosaic_dataset.test_data.values == np.array([[2, 4], [3, -9999]])).all())

        self.assertTrue('time' not in mosaic_dataset)

    def test_create_max_ndvi_mosaic(self):
        dataset = xr.Dataset(
            {
                'test_data': (('time', 'latitude', 'longitude'), self.sample_data),
                'red': (('time', 'latitude', 'longitude'), self.red),
                'nir': (('time', 'latitude', 'longitude'), self.nir)
            },
            coords={'time': self.times,
                    'latitude': self.latitudes,
                    'longitude': self.longitudes})

        mosaic_dataset = create_max_ndvi_mosaic(
            dataset, clean_mask=np.full(self.sample_clean_mask.shape, True), no_data=-9999)

        self.assertTrue((mosaic_dataset.test_data.values == np.array([[5, 4], [3, 1]])).all())
        self.assertTrue('time' not in mosaic_dataset)

        dataset_mins = dataset.copy(deep=True)
        dataset_mins.nir.values = np.array([[[0, 1], [0, 80]], [[1, 4], [1, 60]], [[100, 100], [100, 100]],
                                            [[1, 5], [1, 20]], [[2, 1], [1, 0]]])

        mosaic_dataset_iterated = create_max_ndvi_mosaic(
            dataset_mins,
            intermediate_product=mosaic_dataset,
            clean_mask=np.full(self.sample_clean_mask.shape, True),
            no_data=-9999)

        self.assertTrue((mosaic_dataset_iterated.test_data.values == np.array([[3, 3], [3, 3]])).all())

    def test_create_min_ndvi_mosaic(self):
        dataset = xr.Dataset(
            {
                'test_data': (('time', 'latitude', 'longitude'), self.sample_data),
                'red': (('time', 'latitude', 'longitude'), self.red),
                'nir': (('time', 'latitude', 'longitude'), self.nir)
            },
            coords={'time': self.times,
                    'latitude': self.latitudes,
                    'longitude': self.longitudes})

        mosaic_dataset = create_min_ndvi_mosaic(
            dataset, clean_mask=np.full(self.sample_clean_mask.shape, True), no_data=-9999)

        self.assertTrue((mosaic_dataset.test_data.values == np.array([[1, 3], [1, 3]])).all())
        self.assertTrue('time' not in mosaic_dataset)

        dataset_mins = dataset.copy(deep=True)
        dataset_mins.nir.values = np.array([[[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]], [[-100, -100], [-100, -100]],
                                            [[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]]])
        dataset_mins.red.values = np.array([[[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]], [[100, 100], [100, 100]],
                                            [[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]]])

        mosaic_dataset_iterated = create_min_ndvi_mosaic(
            dataset_mins,
            intermediate_product=mosaic_dataset,
            clean_mask=np.full(self.sample_clean_mask.shape, True),
            no_data=-9999)
        print(mosaic_dataset_iterated, mosaic_dataset)
        self.assertTrue((mosaic_dataset_iterated.test_data.values == np.array([[3, 3], [3, 3]])).all())
