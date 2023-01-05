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


def to_device_array(x, gpu_id, stream):

    if cupy_not_found:
        return x

    xp = cupy.get_array_module(x)

    if xp == np:    # numpy -> cupy
        with cupy.cuda.Device(gpu_id), stream as stream:
            block_events = cupy.cuda.Event(block=True)

            d_x = cupy.empty(x.shape, dtype=x.dtype)
            d_x.set(x, stream=stream)

            block_events.record(stream)
            block_events.synchronize()
            block_events = None

    else:    # already on device
        d_x = x

    return d_x


def from_device_array(d_x, gpu_id, stream):

    if cupy_not_found:
        return d_x

    xp = cupy.get_array_module(d_x)

    if xp == np:
        x = d_x

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
