import os
import os.path
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import h5py
from deepict import subtomos
from deepict import constants
from tqdm import tqdm

import mrcfile
import numpy as np


def write_dataset_hdf(output_path: str, tomo_data: np.array):
    with h5py.File(output_path, "w") as f:
        f[constants.HDF_INTERNAL_PATH] = tomo_data
    print("The hdf file has been writen in ", output_path)


def write_mrc_dataset(mrc_path: str, array: np.array, dtype="float32"):
    array = np.array(array, dtype=dtype)
    with mrcfile.new(mrc_path, overwrite=True) as mrc:
        mrc.set_data(array)
    print("Dataset saved in", mrc_path)
    return


def assemble_tomo_from_subtomos(
    output_path: str,
    partition_file_path: str,
    output_shape: tuple,
    subtomo_shape: tuple or list,
    subtomos_internal_path: str,
    class_number: int,
    overlap: int,
    final_activation: None or "sigmoid" = None,
    reconstruction_type: str = "prediction",
):
    print("Assembling data from", partition_file_path, ":")
    tomo_data = -10 * np.ones(output_shape)  # such that sigmoid(-10) ~ 0
    inner_subtomo_shape = tuple(
        [subtomo_dim - 2 * overlap for subtomo_dim in subtomo_shape]
    )
    with h5py.File(partition_file_path, "r") as f:
        subtomo_names = list(f[subtomos_internal_path])
        total_subtomos = len(subtomo_names)
        output_shape_overlap = tuple([sh + overlap for sh in output_shape])
        for index, subtomo_name in zip(tqdm(range(total_subtomos)), subtomo_names):
            subtomo_center = subtomos.get_coord_from_name(subtomo_name)
            start_corner, end_corner, lengths = subtomos.get_subtomo_corners(
                output_shape=output_shape_overlap,
                subtomo_shape=inner_subtomo_shape,
                subtomo_center=subtomo_center,
            )

            volume_slices = [slice(overlap, overlap + l) for l in lengths]
            if np.min(lengths) > 0:
                overlap_shift = overlap * np.array([1, 1, 1])
                start_corner -= overlap_shift
                end_corner -= overlap_shift
                subtomo_h5_internal_path = join(subtomos_internal_path, subtomo_name)
                if reconstruction_type == "prediction":
                    channels, *rest = f[subtomo_h5_internal_path][:].shape
                    assert class_number < channels
                    # noinspection PyTypeChecker
                    channel_slices = [class_number] + volume_slices
                    channel_slices = tuple(channel_slices)
                    subtomo_data = f[subtomo_h5_internal_path][:]
                    internal_subtomo_data = subtomo_data[channel_slices]
                else:
                    volume_slices = tuple(volume_slices)
                    internal_subtomo_data = f[subtomo_h5_internal_path][volume_slices]
                tomo_slices = tuple(
                    [slice(s, e) for s, e in zip(start_corner, end_corner)]
                )
                tomo_data[tomo_slices] = internal_subtomo_data
    if final_activation is not None:
        sigmoid = nn.Sigmoid()
        tomo_data = sigmoid(torch.from_numpy(tomo_data).float())
        tomo_data = tomo_data.float().numpy()
    ext = os.path.splitext(output_path)[-1].lower()
    if ext == ".mrc":
        write_mrc_dataset(mrc_path=output_path, array=tomo_data)
    elif ext == ".hdf":
        write_dataset_hdf(output_path, tomo_data)
    return
