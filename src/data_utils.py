import os
import imageio
from time import time
from collections import Counter
from src.datasets import *
import utils
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import rasterio


def clip_and_scale_image(img, img_type='naip'):
    """
    Clips and scales bands to between [0, 1] for NAIP, RGB, and Landsat
    satellite images. Clipping applies for Landsat and EuroSAT only.
    """
    if img_type in ['naip', 'rgb']:  # eurosat images scaled from 0-2750 to 1-255
        return img / 255

    elif img_type == 'landsat':
        return np.clip(img, 0, 255) / (255)

    elif img_type =='eurosat':
        return (np.clip(img, 0, 2750) / 2750 * 255).astype(np.uint8)   # need range to be 0-255 for data augmentation

def load_tif_npy(img_fn, bands, bands_only=False):
    img = np.load(img_fn)
    #plt.imsave(paths.fig_dir_test_land + "test.jpg", img)
    if bands_only and bands > 1:  # if bands == 1, shape will be [:,:]
            img = img[:,:,:bands]
        #plt.imsave(paths.fig_dir_test_land + "test_bands.jpg", img)
    return img


def load_rgb(img_fn, bands, bands_only=False, is_npy=True):
    if is_npy:
        return load_tif_npy(img_fn, bands, bands_only)
    obj = gdal.Open(img_fn)
    img = obj.ReadAsArray().astype(np.uint8)
    del obj  # close GDAL dataset

    if bands_only and bands > 1:  # if bands == 1, shape will be [:,:]
        img = img[0:bands]  # only select the first X bands
        img = np.moveaxis(img, 0, -1)

    return img


def load_nlcd(img_fn):
    obj = gdal.Open(img_fn)
    img = obj.ReadAsArray().astype(np.uint8)
    del obj  # close GDAL dataset

    return img

def load_tif(img_fn, bands, bands_only=False, is_npy=True, normalize=False):
    """
    Loads tif image with gdal, returns image as array.
    Move bands (i.e. r,g,b,etc) to last dimension.
    """
    # working with rasterio: https://automating-gis-processes.github.io/CSC18/lessons/L6/reading-raster.html

    if is_npy:
        return load_tif_npy(img_fn, bands, bands_only)

    obj = rasterio.open(img_fn)
    img = obj.read()  # ndarray (bands, height, width)
    obj.close()

    if bands_only and bands > 1:  # if bands == 1, shape will be [:,:]
        img = img[0:bands]  # only select the first X bands
        img = np.moveaxis(img, 0, -1)

        # normalize the image to [0-256]
        # from OR_tif_B3214, scaling all bands with the same min/max values should be okay since the bands have
        # similar ranges. Could alternatively scale each band independently. Band min/max for OR_tif_B321:
        # B3 (Red): -4171/28275
        # B2 (Green): -3739/28290
        # B1: (Blue): -4190/28724
        # B4 (Near infrared): -1358/22582

        if normalize:  # use if working with unnormalized Landsat imagery
            img_min = -4190
            img_max = 28724
            img_scaled = (img - img_min) * (255 / (img_max - img_min))  # pixel range is now [0, 255]

            return img_scaled
    #return img
    img = clip_and_scale_image(img, 'eurosat')
    return img, img.min(), img.max()

def tif2npy(tif_dir, npy_dir, img_type, bands):
    count = 0
    min = 10000
    max = 0

    for img in os.listdir(tif_dir):
        img_name = img.split('.tif')[0]
        print(img_name)
        if img_type == 'nlcd':
            img = load_nlcd(tif_dir + img)
        elif img_type == 'rgb':
            img = load_rgb(tif_dir + img, bands, bands_only=True, is_npy=False)
        else:
            #img = load_tif(tif_dir + img, bands, bands_only=True, is_npy=False)
            img, img_min, img_max = load_tif(tif_dir + img, bands, bands_only=True, is_npy=False)
        if (img.shape[0] >= 48 and img.shape[1] >= 48):
            np.save(os.path.join(npy_dir, img_name + '.npy'), img)

        count += 1
        if img_min < min: min = img_min
        if img_max > max: max = img_max
    print(f'Saved  {str(count)} images to {npy_dir}')
    print(f'Overall min: {min}')
    print(f'Overall max: {max}')


# clip images, centered at middle of image
def clip_extent(npy, imgs_to_clip, extent):
    if npy:
        img_names = glob.glob(imgs_to_clip + "*.npy")
    else:
        img_names = glob.glob(imgs_to_clip + "*.tif")

    # Clip and save images
    clip_img(img_names, bands, patch_size=extent, patch_per_img=1,  # patch size 200 = 2km x 2km
             centered=True, save=True, verbose=True, npy=npy)

    print('Finished clipping images')


def calc_channel_means(img_type, split, paths):
    print(f'\n\nCalculating channel means for {split}')

    dataloader = satellite_dataloader(img_type, paths['img_dir'], paths['labels'], split=split, size=img_size,
                                      bands=bands, augment=False, batch_size=batch_size,
                                      shuffle=False, num_workers=num_workers, means=None)  # means MUST be None

    means, _ = get_channel_mean_stdDev(dataloader)

    print(f'Means: {means}')

    # save means
    img_path = os.path.normpath(paths.png_dir).split(os.path.sep)[-1]
    today = date.today()
    d = today.strftime('%b-%d-%Y')
    np.savetxt('channel_means' + img_path + '_' + d + '.txt', means)


def calc_channel_means_stdDevs(img_type, split, paths):
    print(f'\n\nCalculating channel means & standard deviations for {split}')

    dataloader = satellite_dataloader(img_type, paths['png_dir'], paths['labels'], split=split, size=img_size,
                                      bands=bands, augment=False, batch_size=1,
                                      shuffle=False, num_workers=num_workers, means=None)  # means MUST be None

    means, stdDevs = get_channel_mean_stdDev(dataloader)

    print(f'Means: {means}')
    print(f'Standard Deviations: {stdDevs}')

    # save means
    img_path = os.path.normpath(paths.png_dir).split(os.path.sep)[-1]
    today = date.today()
    d = today.strftime('%b-%d-%Y')
    # VALIDATE THIS (means+stdDevs)
    np.savetxt('channel_means' + img_path + '_' + d + '.txt', means + stdDevs)
