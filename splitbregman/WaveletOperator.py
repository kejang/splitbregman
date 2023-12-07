import numpy as np
import cupy
from cupyx.scipy.sparse.linalg import LinearOperator

from pywt import (
    dwt_max_level, wavedecn, waverecn, fswavedecn, fswaverecn,
    ravel_coeffs, unravel_coeffs,
)

from .utils.utils import from_device_array, to_device_array

mempool = cupy.get_default_memory_pool()


def clean_all_gpu_mem():
    mempool.free_all_blocks()


class WaveletOperator(LinearOperator):

    def __init__(
            self,
            sz_im,
            wavelet,
            gpu_id,
            stream,
            full=False,
            mode='symmetric'):

        self.sz_im = sz_im
        self.wavelet = wavelet
        self.full = full
        self.mode = mode
        self.gpu_id = gpu_id
        self.stream = stream

        self.levels = [dwt_max_level(s, wavelet) for s in sz_im]
        self.min_level = min(self.levels)

        if full:
            self.fsresult = fswavedecn(
                np.zeros(sz_im, dtype=np.float32),
                self.wavelet,
                mode=self.mode,
                levels=self.levels
            )
            self.coeffs_shape = self.fsresult.coeffs.shape
            self.shape = (
                self.fsresult.coeffs.size,
                np.int64(np.prod(sz_im))
            )
        else:
            coeffs = wavedecn(
                np.zeros(sz_im, dtype=np.float32),
                self.wavelet,
                mode=self.mode,
                level=self.min_level
            )
            arr, self.coeff_slices, self.coeff_shapes = ravel_coeffs(coeffs)
            self.shape = (
                arr.size,
                np.int64(np.prod(sz_im))
            )

    def _matvec(self, x):

        xp = cupy.get_array_module(x)

        if xp == np:
            h_x = x.reshape(self.sz_im)
        else:
            h_x = from_device_array(
                x, self.gpu_id, self.stream
            ).reshape(self.sz_im)

        if self.full:
            h_y = fswavedecn(
                h_x,
                self.wavelet,
                mode=self.mode,
                levels=self.levels
            ).coeffs
        else:
            h_y = ravel_coeffs(
                wavedecn(
                    h_x,
                    self.wavelet,
                    mode=self.mode,
                    level=self.min_level
                )
            )[0]

        if xp == np:
            y = h_y.ravel()
        else:
            y = to_device_array(h_y, self.gpu_id, self.stream, ravel=True)

        return y

    def _rmatvec(self, x):

        xp = cupy.get_array_module(x)

        if xp == np:
            h_x = x
        else:
            h_x = from_device_array(
                x, self.gpu_id, self.stream
            )

        if self.full:
            self.fsresult.coeffs = h_x.reshape(self.coeffs_shape)
            h_y = fswaverecn(self.fsresult)
        else:
            h_y = waverecn(
                unravel_coeffs(
                    h_x.ravel(),
                    self.coeff_slices,
                    self.coeff_shapes,
                    output_format='wavedecn'
                ),
                wavelet=self.wavelet,
                mode=self.mode
            )

        if xp == np:
            y = h_y.ravel()
        else:
            y = to_device_array(h_y, self.gpu_id, self.stream, ravel=True)

        return y
