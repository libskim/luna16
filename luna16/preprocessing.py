# Author: Seunghyun Kim
# Date: 18 Feb 2019
# Last updated: 21 Feb 2019

import os
from collections import namedtuple
from glob import glob
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


def _resample(imgarr, spacing, only_csv):
    real_resize_factor = _real_resize_factor(imgarr.shape, spacing)
    spacing = spacing / real_resize_factor
    if only_csv:
        imgarr = None
    else:
        imgarr = interpolation.zoom(imgarr, real_resize_factor, mode='nearest')
    return imgarr, spacing


def _normalize(imgarr, norm_min, norm_max):
    imgarr = np.where(imgarr < norm_min, norm_min, imgarr)
    imgarr = np.where(imgarr > norm_max, norm_max, imgarr)
    imgarr = (imgarr + abs(norm_min)) / (abs(norm_min) + abs(norm_max))
    return imgarr


def _extract(imgarr, pos, size, margin):
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


def unzip_data(input_dir, output_dir):
    """
    Unzip all data.

    We assume that you have downloaded all the necessary data from
    the [LUNA16](https://luna16.grand-challenge.org) website. This
    task requires at least 120 Gb of free space and 7-zip package.
    If you see that the command is not found when you run the task,
    see the following URL: https://www.7-zip.org/
    """

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    zip_files = glob(os.path.join(input_dir, '*.zip'))
    zip_files.sort()
    for zip_file in tqdm(zip_files):
        os.system('7z x ' + zip_file + ' -o' + output_dir + ' -aos')


def resample_mhd_to_npy(input_dir, output_dir, csv_file, norm_min, norm_max,
                        only_csv=False):
    """
    Resample all mhd to npy.

    For preprocessing the data, resamples the coordinate system of
    raw files into a voxel-based coordinate system. This task takes
    a lot of time, so it is a good idea to save the results. But it
    requires at least 290 Gb of free space.

    Sometimes you only need a voxel CSV file. If you set the variable
    only_csv to True, resample_mhd_to_npy() save only the voxel CSV
    file. This takes a short time to calculate.
    """

    if not only_csv and os.path.isdir(output_dir):
        msg = 'The output directory already exists. ' + \
              'If you want to resample again, delete ' + \
              'the output directory and try again.'
        assert False, msg

    Subset = namedtuple('Subset', ['path', 'mhd_files'])

    subset_dirs = glob(os.path.join(input_dir, 'subset*'))
    subset_dirs.sort()

    subsets = []
    for subset_dir in subset_dirs:
        mhd_files = glob(os.path.join(subset_dir, '*.mhd'))
        subset = Subset(subset_dir, mhd_files)
        subsets.append(subset)

    def get_file_name(seriesuid):
        for subset in subsets:
            for mhd_file in subset.mhd_files:
                if seriesuid in mhd_file:
                    return mhd_file

    df = pd.read_csv(csv_file)
    df['file'] = df['seriesuid'].apply(get_file_name)
    df = df.dropna()
    df['vcoordX'], df['vcoordY'], df['vcoordZ'] = np.nan, np.nan, np.nan
    df['npy_file'] = None

    for subset in subsets:
        subset_name = os.path.basename(subset.path)
        new_subset_path = os.path.join(output_dir, subset_name)
        if not only_csv:
            os.makedirs(new_subset_path)

        for mhd_file in tqdm(subset.mhd_files):
            df_mhd = df[df['file'] == mhd_file]

            if df_mhd.shape[0] > 0:
                mhd_uid = str(os.path.basename(mhd_file)[:-4].split('.')[-1])
                npy_name = subset_name + '_' + mhd_uid + '.npy'
                npy_path = os.path.join(new_subset_path, npy_name)

                img = sitk.ReadImage(mhd_file)
                imgarr = sitk.GetArrayFromImage(img)
                origin = np.array(img.GetOrigin())[::-1]
                spacing = np.array(img.GetSpacing())[::-1]

                imgarr, spacing = _resample(imgarr, spacing, only_csv)
                if not only_csv:
                    imgarr = _normalize(imgarr, norm_min, norm_max)
                    np.save(npy_path, imgarr)

                for i, row in df_mhd.iterrows():
                    center = np.array([row['coordZ'], row['coordY'], row['coordX']])
                    vcenter = np.rint((center - origin) / spacing)
                    df.at[i, 'vcoordZ'] = vcenter[0]
                    df.at[i, 'vcoordY'] = vcenter[1]
                    df.at[i, 'vcoordX'] = vcenter[2]
                    df.at[i, 'npy_file'] = npy_name

    df = df.drop(['seriesuid', 'coordX', 'coordY', 'coordZ', 'file'], axis=1)
    new_csv_file = os.path.join(output_dir, 'voxel_' + os.path.basename(csv_file))
    df.to_csv(new_csv_file)


def extract_candidate(input_dir, output_dir, csv_file, size, margin=0, get_2d=True):
    """
    Extract all candidates from npy files.

    This task takes a lot of time, so it is a good idea to save
    the results. The required free space depends on the variables
    extract_size, margin, and get_2d.
    """

    if os.path.isdir(output_dir):
        msg = 'The output directory already exists. ' + \
              'If you want to resample again, delete ' + \
              'the output directory and try again.'
        assert False, msg

    if type(size) == type([]):
        size = np.array(size)

    Subset = namedtuple('Subset', ['path', 'npy_files'])

    subset_dirs = glob(os.path.join(input_dir, 'subset*'))
    subset_dirs.sort()

    subsets = []
    for subset_dir in subset_dirs:
        npy_files = glob(os.path.join(subset_dir, '*.npy'))
        subset = Subset(subset_dir, npy_files)
        subsets.append(subset)

    df = pd.read_csv(csv_file)
    fill = len(str(len(df)))

    for subset in subsets:
        subset_name = os.path.basename(subset.path)
        new_subset_path = os.path.join(output_dir, subset_name)
        os.makedirs(new_subset_path)

        for npy_file in tqdm(subset.npy_files):
            npy_name = os.path.basename(npy_file)
            df_npy = df[df['npy_file'] == npy_name]
            imgarr = np.load(npy_file)

            for i, row in df_npy.iterrows():
                vcenter = np.array([row['vcoordZ'], row['vcoordY'], row['vcoordX']])
                cand_arr = _extract(imgarr, vcenter, size, margin)
                cand_arr = _wrap(cand_arr, size)
                if get_2d:
                    cand_arr = cand_arr[int(cand_arr.shape[0]/2)]
                tag_r = 'row' + str(i).zfill(fill)
                tag_c = '_cls' + str(row['class'])
                cand_name = tag_r + tag_c + '.npy'
                cand_path = os.path.join(new_subset_path, cand_name)
                np.save(cand_path, cand_arr)


def main():
    warnings.filterwarnings('ignore', '.*output shape of zoom.*')

    # origin_dir:
    #   The directory path containing the compressed files downloaded from the website.
    #   The compression files we used are listed below:
    #   subset0.zip, ..., subset9.zip, candidates_V2.zip (total 11)
    #
    # unzip_dir:
    #   The directory path where the extracted files will be stored.
    #
    # resample_dir:
    #   The directory path where resampled data will be stored.
    #
    # extract_dir:
    #   The directory path where extracted candidates will be stored.
    origin_dir = '/data/datasets/luna16-origin'
    unzip_dir = '/data/datasets/luna16-unzip'
    resample_dir = '/data/datasets/luna16-resample'
    extract_dir = '/data/datasets/luna16-extracted'

    # Step 1. Unzip all data.
    #   We assume that you have downloaded all the necessary data from
    #   the [LUNA16](https://luna16.grand-challenge.org) website. This
    #   task requires at least 120 Gb of free space and 7-zip package.
    #   If you see that the command is not found when you run the task,
    #   see the following URL: https://www.7-zip.org/
    unzip_data(origin_dir, unzip_dir)

    # Step 2. Resample all mhd to npy.
    #   For preprocessing the data, resamples the coordinate system of
    #   raw files into a voxel-based coordinate system. This task takes
    #   a lot of time, so it is a good idea to save the results. But it
    #   requires at least 290 Gb of free space.
    #
    #   Sometimes you only need a voxel CSV file. If you set the variable
    #   only_csv to True, resample_mhd_to_npy() save only the voxel CSV
    #   file. This takes a short time to calculate.
    csv_file = os.path.join(unzip_dir, 'candidates_V2.csv')
    norm_min, norm_max = -1000, 400
    only_csv = False

    resample_mhd_to_npy(unzip_dir, resample_dir, csv_file, norm_min, norm_max, only_csv)

    # Step 3. Extract all candidates from npy files.
    #   This task takes a lot of time, so it is a good idea to save the
    #   results. The required free space depends on the variables
    #   extract_size, margin, and get_2d.
    voxel_csv_file = os.path.join(resample_dir, 'voxel_candidates_V2.csv')
    extract_size = [56, 56, 56]

    extract_candidate(resample_dir, extract_dir, voxel_csv_file, extract_size)


if __name__ == '__main__':
    main()
