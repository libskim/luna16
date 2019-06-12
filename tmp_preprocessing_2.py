# Author: Seunghyun Kim
# Date: 18 Feb. 2019
# Last updated: 21 Feb. 2019

# * Attention:
# For 754,975 candidates, crop the 3D region around each candidate's
# coordinates and save it as npy. Please note that crop size affects
# the size of the output files and the total processing time.

import os
from glob import glob
from collections import namedtuple
import warnings

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import interpolation
from tqdm import tqdm


def _real_resize_factor(shape, spacing, adj_spacing=None):
    if adj_spacing is None:
        adj_spacing = list(np.ones_like(shape))
    new_shape = np.round(shape * (spacing / adj_spacing))
    real_resize_factor = new_shape / shape
    return real_resize_factor


def _resample(imgarr, spacing):
    real_resize_factor = _real_resize_factor(imgarr.shape, spacing)
    spacing = spacing / real_resize_factor
    imgarr = interpolation.zoom(imgarr, real_resize_factor, mode='nearest')
    return imgarr, spacing


def _normalize(imgarr, norm_min, norm_max):
    imgarr = np.where(imgarr < norm_min, norm_min, imgarr)
    imgarr = np.where(imgarr > norm_max, norm_max, imgarr)
    imgarr = (imgarr + abs(norm_min)) / (abs(norm_min) + abs(norm_max))
    return imgarr


def _crop(imgarr, pos, size, margin):
    # _extract(...) --rename--> _crop(...)
    shape = imgarr.shape
    half_size = np.rint(size / 2)
    vmin = (pos - half_size) - margin
    vmin = [np.max([0, int(i)]) for i in vmin]
    vmax = vmin + size + (margin * 2)
    vmax = [np.min([ax, int(i)]) for ax, i in zip(shape, vmax)]
    return imgarr[vmin[0]:vmax[0], vmin[1]:vmax[1], vmin[2]:vmax[2]]


def _wrap(cand_arr, size):
    shape = cand_arr.shape
    wrapped = np.ones(size) * np.min(cand_arr)
    vmin = np.rint((size - shape) / 2)
    vmin = np.array([int(i) for i in vmin])
    vmax = vmin + shape
    wrapped[vmin[0]:vmax[0], vmin[1]:vmax[1], vmin[2]:vmax[2]] = cand_arr
    return wrapped


def save_candidates(input_dir, output_dir,
                    norm_min=-1000., norm_max=400.,
                    crop_size=48, crop_margin=0):
    """ Save candidates.
    
    Acquire nodules in 3-dimensions and save it as npy file.
    This task requires a large storage space.
    (crop_size=48, crop_margin=0 --> 668.1 GB)
    
    params
        input_dir:
        output_dir:
        norm_min:
        norm_max:
        crop_size:
        crop_margin:
        
    returns
        none.
    """
    crop_size = np.array([crop_size] * 3)
    
    Subset = namedtuple('Subset', ['path', 'mhd_files'])
    subsets = []
    for subset in sorted(glob(os.path.join(input_dir, 'subset*'))):
        mhd_files = glob(os.path.join(subset, '*.mhd'))
        subsets.append(Subset(subset, mhd_files))
    
    def helper(uid):
        for subset in subsets:
            for mhd_file in subset.mhd_files:
                if uid in mhd_file: return mhd_file
    df = pd.read_csv(os.path.join(input_dir, 'candidates_V2.csv'))
    df['file'] = df['seriesuid'].apply(helper)
    df = df.dropna()
    df = df.drop(labels='seriesuid', axis='columns')
    df['x'], df['y'], df['z'] = [np.nan] * 3
    
    for subset in subsets:
        path = os.path.join(output_dir, os.path.basename(subset.path))
        for mhd_file in tqdm(subset.mhd_files):
            df_mhd = df[df['file'] == mhd_file]
            if df_mhd.shape[0] > 0:
                uid = os.path.basename(mhd_file)[:-4]
                os.makedirs(os.path.join(path, uid))
                
                img = sitk.ReadImage(mhd_file)
                imgarr = sitk.GetArrayFromImage(img)
                origin = np.array(img.GetOrigin())[::-1]
                spacing = np.array(img.GetSpacing())[::-1]
                imgarr, spacing = _resample(imgarr, spacing)
                imgarr = _normalize(imgarr, norm_min, norm_max)
                
                for i, r in df_mhd.iterrows():
                    center = np.array([r.coordZ, r.coordY, r.coordX])
                    vcenter = np.rint((center - origin) / spacing)
                    candarr = _crop(imgarr, vcenter, crop_size, crop_margin)
                    candarr = _wrap(candarr, crop_size)
                    cpath = os.path.join(path, uid, '%d_%d.npy'%(i, r['class']))
                    np.save(cpath, candarr)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', '.*output shape of zoom.*')

    input_dir = '/data/luna16/origin'
    output_dir = '/data/luna16/nodules'
    #norm_min = default
    #norm_max = default
    #crop_size = default
    #crop_margin = default

    save_candidates(input_dir=input_dir,
                    output_dir=output_dir)
