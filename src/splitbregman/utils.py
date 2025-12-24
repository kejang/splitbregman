import numpy as np
import cupy as cp


def to_device_array(
    x,
    gpu_id,
    stream,
    dtype_real="float32",
    dtype_complex="complex64",
    shape=None,
    ravel=False,
):

    xp = cp.get_array_module(x)

    if xp.iscomplexobj(x):
        dtype = cp.dtype(dtype_complex)
    else:
        dtype = cp.dtype(dtype_real)

    d_x = None

    if xp == np:  # numpy -> cp
        with cp.cuda.Device(gpu_id), stream as stream:
            block_events = cp.cuda.Event(block=True)
            if shape is None:
                if ravel:
                    d_x = cp.empty((x.size,), dtype=dtype)
                    d_x.set(x.astype(dtype).ravel(), stream=stream)
                    block_events.record(stream)
                else:
                    d_x = cp.empty(x.shape, dtype=dtype)
                    d_x.set(x.astype(dtype), stream=stream)
                    block_events.record(stream)
            else:
                temp_x = x.astype(dtype).reshape(shape)
                d_x = cp.empty(temp_x.shape, dtype=dtype)
                d_x.set(temp_x, stream=stream)
                block_events.record(stream)

            block_events.synchronize()
            block_events = None

    else:  # already on device
        with cp.cuda.Device(gpu_id), stream as stream:
            if shape is None:
                if ravel:
                    d_x = x.astype(dtype).ravel()
                else:
                    d_x = x.astype(dtype)
            else:
                d_x = x.astype(dtype).reshape(shape)

    return d_x


def from_device_array(d_x, gpu_id, stream):

    xp = cp.get_array_module(d_x)

    if xp == np:
        x = d_x.copy()  # already on host

    else:
        with cp.cuda.Device(gpu_id), stream as stream:
            block_events = cp.cuda.Event(block=True)

            x = np.empty(d_x.shape, dtype=d_x.dtype)
            d_x.get(out=x, stream=stream)

            block_events.record(stream)
            block_events.synchronize()
            block_events = None

    return x
