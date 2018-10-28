from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
import numpy as np
from collections import OrderedDict
import xarray as xr
import matplotlib.pyplot as plt

def get_frequency_counts(classification):
    """
    Get the raw and fractional class frequency counts for an `xarray.Dataset`.
    Intended to be used with outputs from the `*_cluster_dataset()` functions.
    
    Parameters
    ----------
    classification: xarray.Dataset
        An `xarray.Dataset` with a `classification` data variable.
    
    Returns
    -------
    freqs: np.ndarray of np.float64
        A 2D NumPy array containing entries of the format 
        [class_number, count, frequency] ordered by class number.
    """
    classifications = classification.classification.values.flatten()
    class_nums, class_counts = np.unique(classifications, return_counts=True)
    num_classifications = len(classifications)
    fractional_freqs = [count/num_classifications for count in class_counts]
    freqs = np.array([(class_num, count, freq) for (class_num, count, freq) in 
                      zip(class_nums, class_counts, fractional_freqs)])
    return freqs

def clustering_pre_processing(dataset_in, bands):
    array_from = []
    for band in bands:
        array_from.append(dataset_in[band].values.flatten())
        
    features = np.array(array_from)
    features = np.swapaxes(features, 0, 1)
    np.set_printoptions(suppress=True)
    return features

def clustering_post_processing(classified, dataset_in, bands):
    classified_data = OrderedDict()
    shape = dataset_in[bands[0]].values.shape
    classification = classified.labels_.reshape(shape)
    coord_names = list(dataset_in.coords.keys())
    classified_data['classification'] = (coord_names, classification)
    dataset_out = xr.Dataset(classified_data,
                             dataset_in.coords)
    return dataset_out

def kmeans_cluster_dataset(dataset_in, bands, n_clusters=4):
    """
    Clusters a dataset with Kmeans clustering.
    
    Parameters
    ----------
    dataset_in: xarray.Dataset
        A Dataset containing the bands listed in `bands`.
    bands: list of str
        A list of names of the bands in `dataset_in` to cluster with.
        
    Returns
    -------
    clustered: xarray.Dataset
        A Dataset of the same shape as `dataset_in`, containing a single data variable 
        called "classification", which are the numeric class labels in range [0, n_clusters-1].
    """
    features = clustering_pre_processing(dataset_in, bands)
    """
    classified = AgglomerativeClustering(n_clusters=n_clusters).fit(np_array)
    classified = Birch(n_clusters=n_clusters).fit(np_array)
    classified = DBSCAN(eps=0.005, min_samples=5, n_jobs=-1).fit(np_array)
    """    
    classified = KMeans(n_clusters=n_clusters, n_jobs=-1).fit(features)
    return clustering_post_processing(classified, dataset_in, bands)

def birch_cluster_dataset(dataset_in, bands=['red', 'green', 'blue', 'swir1', 'swir2', 'nir'], n_clusters=4):
    array_from, np_array = clustering_pre_processing(dataset_in, bands)
    """
    classified = AgglomerativeClustering(n_clusters=n_clusters, n_jobs=-1).fit(np_array)
    classified = DBSCAN(eps=0.005, min_samples=5, n_jobs=-1).fit(np_array)
    classified = KMeans(n_clusters=n_clusters, n_jobs=-1).fit(np_array)
    """  
    classified = Birch(n_clusters=n_clusters, threshold=0.00001).fit(np_array)
    return clustering_post_processing(classified, dataset_in, bands)

def plot_kmeans_next_to_mosaic(da_a, da_b):  
    def mod_rgb(dataset,
        at_index = 0,
        bands = ['red', 'green', 'blue'],
        paint_on_mask = [],
        max_possible = 3500,
        width = 10
       ):    
        ### < Dataset to RGB Format, needs float values between 0-1 
        rgb = np.stack([dataset[bands[0]],
                        dataset[bands[1]],
                        dataset[bands[2]]], axis = -1).astype(np.int16)

        rgb[rgb<0] = 0    
        rgb[rgb > max_possible] = max_possible # Filter out saturation points at arbitrarily defined max_possible value

        rgb = rgb.astype(float)
        rgb *= 1 / np.max(rgb)
        ### > 

        ### < takes a T/F mask, apply a color to T areas  
        for mask, color in paint_on_mask:        
            rgb[mask] = np.array(color)/ 255.0
        ### > 

        if 'time' in dataset:
            plt.imshow((rgb[at_index]))
        else:
            plt.imshow(rgb)  

    fig = plt.figure(figsize =  (15,8))
    a=fig.add_subplot(1,2,1) 
    a.set_title('Kmeans')
    plt.imshow(da_a.values, cmap = "magma_r")

    b=fig.add_subplot(1,2,2)
    mod_rgb(da_b)
    b.set_title('RGB Composite')
