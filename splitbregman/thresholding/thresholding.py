import os
import logging
from pathlib import Path

import numpy as np
import cupy

from ..utils.utils import (
    get_string_include_cucomplex, to_device_array, from_device_array
)

logger = logging.getLogger(__name__)
module_path = Path(os.path.abspath(__file__)).parent.absolute()

n_devices = cupy.cuda.runtime.getDeviceCount()
mempool = cupy.get_default_memory_pool()


def clean_all_gpu_mem():
    mempool.free_all_blocks()


include_cucomplex = get_string_include_cucomplex()


#
# compile cuda source
#

src_fn_soft_thresh_real = module_path.joinpath("cuda", "soft_thresh_real.cu")

with open(src_fn_soft_thresh_real, "r") as fp:
    contents = fp.readlines()
    src_soft_thresh_real = "".join(c for c in contents)

soft_thresh_real = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        soft_thresh_real[gpu_id] = cupy.RawKernel(
            src_soft_thresh_real, "soft_thresh_real"
        )

src_fn_soft_thresh_real_ratio = module_path.joinpath(
    "cuda", "soft_thresh_real_ratio.cu"
)

with open(src_fn_soft_thresh_real_ratio, "r") as fp:
    contents = fp.readlines()
    src_soft_thresh_real_ratio = "".join(c for c in contents)

soft_thresh_real_ratio = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        soft_thresh_real_ratio[gpu_id] = cupy.RawKernel(
            src_soft_thresh_real_ratio, "soft_thresh_real_ratio"
        )

src_fn_soft_thresh_complex = module_path.joinpath("cuda",
                                                  "soft_thresh_complex.cu")

with open(src_fn_soft_thresh_complex, "r") as fp:
    contents = fp.readlines()
    src_soft_thresh_complex = include_cucomplex
    src_soft_thresh_complex += "".join(c for c in contents)

soft_thresh_complex = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        soft_thresh_complex[gpu_id] = cupy.RawKernel(
            src_soft_thresh_complex, "soft_thresh_complex"
        )

#
# python code
#


def soft_thresholding(x, thresh, gpu_id=0, stream=None, n_threads=256):
    """Returns soft thresholded signal/image."""

    xp = cupy.get_array_module(x)

    if xp.iscomplexobj(x):
        kernel = soft_thresh_complex
    else:
        kernel = soft_thresh_real

    if x.ndim == 1:
        ravel = False
    else:
        ravel = True

    d_x = to_device_array(x, gpu_id=gpu_id, stream=stream, ravel=ravel)

    args = (d_x, np.float32(thresh), np.ulonglong(d_x.size))

    blk = (int(n_threads), 1, 1)
    grd = (int((d_x.size + n_threads - 1.0) / n_threads), 1, 1)

    if stream is None:
        with cupy.cuda.Device(gpu_id):
            stream = cupy.cuda.stream.Stream(ptds=True)

    with cupy.cuda.Device(gpu_id), stream as stream:
        kernel[gpu_id](
            grd,
            blk,
            args,
            stream=stream
        )

    if xp == np:    # return as numpy array
        x_thresh = from_device_array(
            d_x, gpu_id=gpu_id, stream=stream
        ).astype(x.dtype)
    else:
        x_thresh = d_x

    if ravel:
        return x_thresh.reshape(x.shape)
    else:
        return x_thresh


def soft_thresholding_ratio(x, thresh, gpu_id=0, stream=None, n_threads=256):
    """Returns max(s - thresh, 0) / s for Split Bregman TV recon."""

    if stream is None:
        with cupy.cuda.Device(gpu_id):
            stream = cupy.cuda.stream.Stream(ptds=True)

    xp = cupy.get_array_module(x)
    kernel = soft_thresh_real_ratio

    if x.ndim == 1:
        ravel = False
    else:
        ravel = True

    d_x = to_device_array(x, gpu_id=gpu_id, stream=stream, ravel=ravel)
    args = (d_x, np.float32(thresh), np.ulonglong(d_x.size))
    blk = (int(n_threads), 1, 1)
    grd = (int((d_x.size + n_threads - 1.0) / n_threads), 1, 1)

    with cupy.cuda.Device(gpu_id), stream as stream:
        kernel[gpu_id](
            grd,
            blk,
            args,
            stream=stream
        )

    if xp == np:    # return as numpy array
        x_thresh = from_device_array(
            d_x, gpu_id=gpu_id, stream=stream
        ).astype(x.dtype)
    else:
        x_thresh = d_x

    if ravel:
        return x_thresh.reshape(x.shape)
    else:
        return x_thresh
