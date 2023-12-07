import logging
from pathlib import Path
import os

import numpy as np
import cupy
from cupyx.scipy.sparse.linalg import LinearOperator

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

src_fn_d_x_real = module_path.joinpath("cuda", "d_x_real.cu")

with open(src_fn_d_x_real, "r") as fp:
    contents = fp.readlines()
    src_d_x_real = "".join(c for c in contents)

d_x_real = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        d_x_real[gpu_id] = cupy.RawKernel(
            src_d_x_real, "d_x_real"
        )

src_fn_d_y_real = module_path.joinpath("cuda", "d_y_real.cu")

with open(src_fn_d_y_real, "r") as fp:
    contents = fp.readlines()
    src_d_y_real = "".join(c for c in contents)

d_y_real = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        d_y_real[gpu_id] = cupy.RawKernel(
            src_d_y_real, "d_y_real"
        )

src_fn_d_z_real = module_path.joinpath("cuda", "d_z_real.cu")

with open(src_fn_d_z_real, "r") as fp:
    contents = fp.readlines()
    src_d_z_real = "".join(c for c in contents)

d_z_real = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        d_z_real[gpu_id] = cupy.RawKernel(
            src_d_z_real, "d_z_real"
        )

src_fn_dt_x_real = module_path.joinpath("cuda", "dt_x_real.cu")

with open(src_fn_dt_x_real, "r") as fp:
    contents = fp.readlines()
    src_dt_x_real = "".join(c for c in contents)

dt_x_real = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        dt_x_real[gpu_id] = cupy.RawKernel(
            src_dt_x_real, "dt_x_real"
        )

src_fn_dt_y_real = module_path.joinpath("cuda", "dt_y_real.cu")

with open(src_fn_dt_y_real, "r") as fp:
    contents = fp.readlines()
    src_dt_y_real = "".join(c for c in contents)

dt_y_real = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        dt_y_real[gpu_id] = cupy.RawKernel(
            src_dt_y_real, "dt_y_real"
        )

src_fn_dt_z_real = module_path.joinpath("cuda", "dt_z_real.cu")

with open(src_fn_dt_z_real, "r") as fp:
    contents = fp.readlines()
    src_dt_z_real = "".join(c for c in contents)

dt_z_real = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        dt_z_real[gpu_id] = cupy.RawKernel(
            src_dt_z_real, "dt_z_real"
        )

src_fn_dtd_x_real = module_path.joinpath("cuda", "dtd_x_real.cu")

with open(src_fn_dtd_x_real, "r") as fp:
    contents = fp.readlines()
    src_dtd_x_real = "".join(c for c in contents)

dtd_x_real = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        dtd_x_real[gpu_id] = cupy.RawKernel(
            src_dtd_x_real, "dtd_x_real"
        )

src_fn_dtd_y_real = module_path.joinpath("cuda", "dtd_y_real.cu")

with open(src_fn_dtd_y_real, "r") as fp:
    contents = fp.readlines()
    src_dtd_y_real = "".join(c for c in contents)

dtd_y_real = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        dtd_y_real[gpu_id] = cupy.RawKernel(
            src_dtd_y_real, "dtd_y_real"
        )

src_fn_dtd_z_real = module_path.joinpath("cuda", "dtd_z_real.cu")

with open(src_fn_dtd_z_real, "r") as fp:
    contents = fp.readlines()
    src_dtd_z_real = "".join(c for c in contents)

dtd_z_real = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        dtd_z_real[gpu_id] = cupy.RawKernel(
            src_dtd_z_real, "dtd_z_real"
        )

src_fn_d_x_complex = module_path.joinpath("cuda", "d_x_complex.cu")

with open(src_fn_d_x_complex, "r") as fp:
    contents = fp.readlines()
    src_d_x_complex = include_cucomplex
    src_d_x_complex += "".join(c for c in contents)

d_x_complex = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        d_x_complex[gpu_id] = cupy.RawKernel(
            src_d_x_complex, "d_x_complex"
        )

src_fn_d_y_complex = module_path.joinpath("cuda", "d_y_complex.cu")

with open(src_fn_d_y_complex, "r") as fp:
    contents = fp.readlines()
    src_d_y_complex = include_cucomplex
    src_d_y_complex += "".join(c for c in contents)

d_y_complex = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        d_y_complex[gpu_id] = cupy.RawKernel(
            src_d_y_complex, "d_y_complex"
        )

src_fn_d_z_complex = module_path.joinpath("cuda", "d_z_complex.cu")

with open(src_fn_d_z_complex, "r") as fp:
    contents = fp.readlines()
    src_d_z_complex = include_cucomplex
    src_d_z_complex += "".join(c for c in contents)

d_z_complex = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        d_z_complex[gpu_id] = cupy.RawKernel(
            src_d_z_complex, "d_z_complex"
        )

src_fn_dt_x_complex = module_path.joinpath("cuda", "dt_x_complex.cu")

with open(src_fn_dt_x_complex, "r") as fp:
    contents = fp.readlines()
    src_dt_x_complex = include_cucomplex
    src_dt_x_complex += "".join(c for c in contents)

dt_x_complex = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        dt_x_complex[gpu_id] = cupy.RawKernel(
            src_dt_x_complex, "dt_x_complex"
        )

src_fn_dt_y_complex = module_path.joinpath("cuda", "dt_y_complex.cu")

with open(src_fn_dt_y_complex, "r") as fp:
    contents = fp.readlines()
    src_dt_y_complex = include_cucomplex
    src_dt_y_complex += "".join(c for c in contents)

dt_y_complex = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        dt_y_complex[gpu_id] = cupy.RawKernel(
            src_dt_y_complex, "dt_y_complex"
        )

src_fn_dt_z_complex = module_path.joinpath("cuda", "dt_z_complex.cu")

with open(src_fn_dt_z_complex, "r") as fp:
    contents = fp.readlines()
    src_dt_z_complex = include_cucomplex
    src_dt_z_complex += "".join(c for c in contents)

dt_z_complex = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        dt_z_complex[gpu_id] = cupy.RawKernel(
            src_dt_z_complex, "dt_z_complex"
        )

src_fn_dtd_x_complex = module_path.joinpath("cuda", "dtd_x_complex.cu")

with open(src_fn_dtd_x_complex, "r") as fp:
    contents = fp.readlines()
    src_dtd_x_complex = include_cucomplex
    src_dtd_x_complex += "".join(c for c in contents)

dtd_x_complex = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        dtd_x_complex[gpu_id] = cupy.RawKernel(
            src_dtd_x_complex, "dtd_x_complex"
        )

src_fn_dtd_y_complex = module_path.joinpath("cuda", "dtd_y_complex.cu")

with open(src_fn_dtd_y_complex, "r") as fp:
    contents = fp.readlines()
    src_dtd_y_complex = include_cucomplex
    src_dtd_y_complex += "".join(c for c in contents)

dtd_y_complex = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        dtd_y_complex[gpu_id] = cupy.RawKernel(
            src_dtd_y_complex, "dtd_y_complex"
        )

src_fn_dtd_z_complex = module_path.joinpath("cuda", "dtd_z_complex.cu")

with open(src_fn_dtd_z_complex, "r") as fp:
    contents = fp.readlines()
    src_dtd_z_complex = include_cucomplex
    src_dtd_z_complex += "".join(c for c in contents)

dtd_z_complex = [None] * n_devices

for gpu_id in range(n_devices):
    with cupy.cuda.Device(gpu_id):
        dtd_z_complex[gpu_id] = cupy.RawKernel(
            src_dtd_z_complex, "dtd_z_complex"
        )

#
# python code
#


def get_kernels(is_complex, is_normal, direction):

    if is_complex:
        if is_normal:
            if direction == 'x':
                return dtd_x_complex, dtd_x_complex
            elif direction == 'y':
                return dtd_y_complex, dtd_y_complex
            elif direction == 'z':
                return dtd_z_complex, dtd_z_complex
            else:
                return None, None
        else:
            if direction == 'x':
                return d_x_complex, dt_x_complex
            elif direction == 'y':
                return d_y_complex, dt_y_complex
            elif direction == 'z':
                return d_z_complex, dt_z_complex
            else:
                return None, None
    else:
        if is_normal:
            if direction == 'x':
                return dtd_x_real, dtd_x_real
            elif direction == 'y':
                return dtd_y_real, dtd_y_real
            elif direction == 'z':
                return dtd_z_real, dtd_z_real
            else:
                return None, None
        else:
            if direction == 'x':
                return d_x_real, dt_x_real
            elif direction == 'y':
                return d_y_real, dt_y_real
            elif direction == 'z':
                return d_z_real, dt_z_real
            else:
                return None, None


class _FirstDerivative():

    def __init__(self,
                 direction,
                 sz_im,
                 gpu_id,
                 stream,
                 is_complex,
                 edge):

        self.direction = direction

        if is_complex:
            self.dtype = cupy.dtype('complex64')
        else:
            self.dtype = cupy.dtype('float32')

        self.n = np.int64(np.prod(sz_im))

        if len(sz_im) == 1:
            self.nz = 1
            self.ny = 1
            self.nx = sz_im[0]
        elif len(sz_im) == 2:
            self.nz = 1
            self.ny = sz_im[-2]
            self.nx = sz_im[-1]
        elif len(sz_im) == 3:
            self.nz = sz_im[-3]
            self.ny = sz_im[-2]
            self.nx = sz_im[-1]
        else:
            raise Exception("sz_im error: only support dim <= 3")

        self.edge = edge
        self.gpu_id = gpu_id
        self.stream = stream

        with cupy.cuda.Device(self.gpu_id), self.stream as _:
            self.block_events = cupy.cuda.Event(block=True)

        if ((self.ny == 1) and (self.nz == 1)):
            self.blk = (256, 1, 1)
        elif (self.nz == 1):
            self.blk = (16, 16, 1)
        else:
            self.blk = (8, 8, 8)

        self.grd = (
            int((self.nx + self.blk[0] - 1.0) / self.blk[0]),
            int((self.ny + self.blk[1] - 1.0) / self.blk[1]),
            int((self.nz + self.blk[2] - 1.0) / self.blk[2]),
        )

    def __del__(self):
        with cupy.cuda.Device(self.gpu_id), self.stream as stream:
            self.block_events.record(stream)
            self.block_events.synchronize()
            self.block_events = None

    def run(self, x, kernel):

        with cupy.cuda.Device(self.gpu_id), self.stream as _:
            d_y = cupy.empty((self.n), dtype=self.dtype)

        d_x = to_device_array(x,
                              gpu_id=self.gpu_id,
                              stream=self.stream,
                              dtype_real='float32',
                              dtype_complex='complex64')

        args = (d_y,
                d_x,
                np.uint32(self.nx),
                np.uint32(self.ny),
                np.uint32(self.nz),
                np.int32(self.edge))

        with cupy.cuda.Device(self.gpu_id):
            kernel[self.gpu_id](
                self.grd,
                self.blk,
                args,
                stream=self.stream
            )

        xp = cupy.get_array_module(x)

        if xp == np:    # return as numpy array
            y = from_device_array(
                d_y, gpu_id=self.gpu_id, stream=self.stream
            ).astype(x.dtype)
        else:
            y = d_y

        return y


class FirstDerivative(LinearOperator):

    def __init__(self,
                 direction,
                 sz_im,
                 gpu_id,
                 stream,
                 is_complex=True,
                 edge=False,
                 is_normal=False):

        self.kernel_forward, self.kernel_adjoint = get_kernels(
            is_complex, is_normal, direction
        )

        self.direction = direction
        self.edge = edge

        if is_complex:
            self.dtype = cupy.dtype('complex64')
        else:
            self.dtype = cupy.dtype('float32')

        self.d_op = _FirstDerivative(
            direction=direction,
            sz_im=sz_im,
            gpu_id=gpu_id,
            stream=stream,
            is_complex=is_complex,
            edge=edge,
        )

        self.shape = (self.d_op.n, self.d_op.n)

    def __del__(self):

        self.d_op = None

    def _matvec(self, x):

        return self.d_op.run(x, self.kernel_forward)

    def _rmatvec(self, x):

        return self.d_op.run(x, self.kernel_adjoint)
