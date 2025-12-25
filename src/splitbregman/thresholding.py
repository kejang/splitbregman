import numpy as np
import cupy as cp

_soft_thresh_real_kernel = cp.ElementwiseKernel(
    "T x, float32 thresh",
    "T out",
    """
    T val = x;
    T abs_val = (val > 0) ? val : -val;
    T abs_val_thresh = abs_val - thresh;
    
    if (abs_val_thresh < 0) {
        out = 0;
    } else {
        out = (val > 0) ? abs_val_thresh : -abs_val_thresh;
    }
    """,
    "soft_thresh_real",
)

_soft_thresh_complex_kernel = cp.ElementwiseKernel(
    "T x, float32 thresh",
    "T out",
    """
    float abs_val = abs(x);
    float val_thresh = abs_val - thresh;
    
    if (val_thresh < 0) {
        out = 0;
    } else {
        float w = val_thresh / abs_val;
        out = x * w;
    }
    """,
    "soft_thresh_complex",
)

_soft_thresh_ratio_kernel = cp.ElementwiseKernel(
    "T x, float32 thresh",
    "T out",
    """
    T val_thresh = x - thresh;
    if (val_thresh < 0) {
        out = 0;
    } else {
        out = val_thresh / x;
    }
    """,
    "soft_thresh_real_ratio",
)


def soft_thresholding(x, thresh, gpu_id, stream):
    """Apply soft-thresholding to a real or complex array.

    For real inputs, the operation is::

        y = sign(x) * max(|x| - thresh, 0)

    For complex inputs, the operation is::

        y = x * max(|x| - thresh, 0) / |x|

    Parameters
    ----------
    x : numpy.ndarray or cupy.ndarray
        Input array. May be real or complex, and any shape.
    thresh : float
        Non-negative threshold value. Internally cast to `np.float32`.
    gpu_id : int
        CUDA device ID to run on.
    stream : cupy.cuda.Stream
        CUDA stream to use for device operations.

    Returns
    -------
    out : numpy.ndarray or cupy.ndarray
        Soft-thresholded array.
    """
    thresh = np.float32(thresh)
    xp = cp.get_array_module(x)
    is_numpy = xp is np

    with cp.cuda.Device(gpu_id), stream:
        is_complex = xp.iscomplexobj(x)
        dtype = cp.complex64 if is_complex else cp.float32

        if is_numpy:
            d_x = cp.array(x, dtype=dtype)
        else:
            d_x = cp.asarray(x).astype(dtype, copy=True)

        kernel = _soft_thresh_complex_kernel if is_complex else _soft_thresh_real_kernel
        kernel(d_x, thresh, d_x)

    with cp.cuda.Device(gpu_id), stream:
        if is_numpy:
            out = d_x.get(stream=stream)
            stream.synchronize()
            return out
        else:
            return d_x


def soft_thresholding_ratio(x, thresh, gpu_id, stream):
    """Compute a soft-thresholding ratio ``max(x - thresh, 0) / x`` (real-valued).

    This helper is commonly used in Split Bregman / TV-type reconstructions where
    the input `x` is expected to be non-negative (e.g., magnitudes).

    The operation is::

        y = 0                       if x - thresh < 0
        y = (x - thresh) / x        otherwise

    Parameters
    ----------
    x : numpy.ndarray or cupy.ndarray
        Input array (typically non-negative). Any shape.
    thresh : float
        Threshold value. Internally cast to `np.float32`.
    gpu_id : int
        CUDA device ID to run on.
    stream : cupy.cuda.Stream
        CUDA stream to use for device operations.

    Returns
    -------
    out : numpy.ndarray or cupy.ndarray
        Output array containing the ratio
    """
    thresh = np.float32(thresh)
    xp = cp.get_array_module(x)
    is_numpy = xp is np

    with cp.cuda.Device(gpu_id), stream:
        if is_numpy:
            d_x = cp.array(x, dtype=cp.float32)
        else:
            d_x = cp.asarray(x).astype(cp.float32, copy=True)

        _soft_thresh_ratio_kernel(d_x, thresh, d_x)

    with cp.cuda.Device(gpu_id), stream:
        if is_numpy:
            out = d_x.get(stream=stream)
            stream.synchronize()
            return out
        else:
            return d_x
