# Copyright 2016 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. All Rights Reserved.
#
# Portion of this code is Copyright Geoscience Australia, Licensed under the
# Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License
# at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# The CEOS 2 platform is licensed under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
from datetime import datetime

import scipy.ndimage.filters as conv
import numpy as np


def coastline_classification(dataset, water_band='wofs'):
    kern = np.array([[1, 1, 1], [1, 0.001, 1], [1, 1, 1]])
    convolved = conv.convolve(dataset[water_band], kern, mode='constant') // 1

    ds = dataset.where(convolved > 0)
    ds = ds.where(convolved < 6)
    ds.wofs.values[~np.isnan(ds.wofs.values)] = 1
    ds.wofs.values[np.isnan(ds.wofs.values)] = 0
    ds.rename({"wofs": "coastline"}, inplace=True)

    return ds


def coastline_classification_2(dataset, water_band='wofs'):
    kern = np.array([[1, 1, 1], [1, 0.001, 1], [1, 1, 1]])
    convolved = conv.convolve(dataset[water_band], kern, mode='constant', cval=-999) // 1

    ds = dataset.copy(deep=True)
    ds.wofs.values[(~np.isnan(ds[water_band].values)) & (ds.wofs.values == 1)] = 1
    ds.wofs.values[convolved < 0] = 0
    ds.wofs.values[convolved > 6] = 0
    ds.rename({"wofs": "coastline"}, inplace=True)

    return ds
