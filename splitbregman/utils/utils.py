import logging
import os
import numpy as np
from pathlib import Path
from importlib import import_module

logger = logging.getLogger(__name__)

try:
    cupy = import_module('cupy')
except ModuleNotFoundError:
    cupy_not_found = True
else:
    globals()['cupy'] = cupy
    cupy_not_found = False


def to_device_array(x,
                    gpu_id,
                    stream,
                    dtype_real='float32',
                    dtype_complex='complex64',
                    shape=None,
                    ravel=False):

    xp = cupy.get_array_module(x)

    if xp.iscomplexobj(x):
        dtype = cupy.dtype(dtype_complex)
    else:
        dtype = cupy.dtype(dtype_real)

    d_x = None

    if xp == np:    # numpy -> cupy
        with cupy.cuda.Device(gpu_id), stream as stream:
            block_events = cupy.cuda.Event(block=True)
            if shape is None:
                if ravel:
                    d_x = cupy.empty((x.size,), dtype=dtype)
                    d_x.set(x.astype(dtype).ravel(), stream=stream)
                    block_events.record(stream)
                else:
                    d_x = cupy.empty(x.shape, dtype=dtype)
                    d_x.set(x.astype(dtype), stream=stream)
                    block_events.record(stream)
            else:
                temp_x = x.astype(dtype).reshape(shape)
                d_x = cupy.empty(temp_x.shape, dtype=dtype)
                d_x.set(temp_x, stream=stream)
                block_events.record(stream)

            block_events.synchronize()
            block_events = None

    else:    # already on device
        with cupy.cuda.Device(gpu_id), stream as stream:
            if shape is None:
                if ravel:
                    d_x = x.astype(dtype).ravel()
                else:
                    d_x = x.astype(dtype)
            else:
                d_x = x.astype(dtype).reshape(shape)

    return d_x


def from_device_array(d_x,
                      gpu_id,
                      stream):

    xp = cupy.get_array_module(d_x)

    if xp == np:
        x = d_x.copy()    # already on host

    else:
        with cupy.cuda.Device(gpu_id), stream as stream:
            block_events = cupy.cuda.Event(block=True)

            x = np.empty(d_x.shape, dtype=d_x.dtype)
            d_x.get(out=x, stream=stream)

            block_events.record(stream)
            block_events.synchronize()
            block_events = None

    return x


def get_string_include_cucomplex():
    """Return #include "cuComplex.h"."""
    # FIX-IT: os.environ.get('CUDA_PATH') is broken in ssh.exec_command().
    # For now, assume the conda environment.
    # CUDA_PATH = os.environ.get('CUDA_PATH')
    CUDA_PATH = Path(os.environ.get('_')).parent.parent

    include_cucomplex = ''.join(['#include "',
                                str(CUDA_PATH),
                                '/include/cuComplex.h"'
                                 ])

    return include_cucomplex
