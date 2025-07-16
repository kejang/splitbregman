import logging
from typing import Optional, Callable, List, Set, Tuple

import numpy as np
import cupy as cp
from cupyx.scipy.sparse.linalg import LinearOperator, gmres, cg

from .derivative.FirstDerivative import FirstDerivative
from .thresholding.thresholding import soft_thresholding, soft_thresholding_ratio
from .utils.utils import to_device_array, from_device_array
from .WaveletOperator import WaveletOperator

logger = logging.getLogger(__name__)

n_devices = cp.cuda.runtime.getDeviceCount()
mempools = [[] for _ in range(n_devices)]
pinned_mempools = [[] for _ in range(n_devices)]
for gpu_id in range(n_devices):
    with cp.cuda.Device(gpu_id):
        mempools[gpu_id] = cp.cuda.MemoryPool()
        cp.cuda.set_allocator(mempools[gpu_id].malloc)
        pinned_mempools[gpu_id] = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(pinned_mempools[gpu_id].malloc)


def clean_up_gpu(gpu_id):

    with cp.cuda.Device(gpu_id):
        mempools[gpu_id].free_all_blocks()
        pinned_mempools[gpu_id].free_all_blocks()


class _GradModel(LinearOperator):
    def __init__(self, model, DtDs, mu, lams, gam, gpu_id, stream):
        self.shape = DtDs[0].shape
        self.dtype = DtDs[0].dtype

        self.model = model
        self.DtDs = DtDs
        self.mu = mu
        self.lams = lams
        self.gam = gam
        self.gpu_id = gpu_id
        self.stream = stream

    def run(self, x):
        with cp.cuda.Device(self.gpu_id), self.stream as _:
            y = 0.5 * self.mu * self.model.rmatvec(self.model.matvec(x))

            for lam, DtD in zip(self.lams, self.DtDs):
                y += 0.5 * lam * DtD.matvec(x)

            if self.gam > 0:
                y += 0.5 * self.gam * x

        return y

    def _matvec(self, x):
        return self.run(x)

    def _rmatvec(self, x):
        return self.run(x)


class SplitBregman:
    def __init__(self, model, sz_im, edge=False, economize_gpu=False):
        self.model = model
        self.sz_im = sz_im
        self.gpu_id = model.gpu_id
        self.stream = model.stream
        self.economize_gpu = economize_gpu

        is_complex = np.iscomplexobj(np.ones((1,), dtype=model.dtype))

        if len(sz_im) == 1:
            directions = ["x"]
            self.ndim = 1
        elif len(sz_im) == 2:
            directions = ["x", "y"]
            self.ndim = 2
        else:
            directions = ["x", "y", "z"]
            self.ndim = 3

        self.Ds = [
            FirstDerivative(
                direction=direction,
                sz_im=sz_im,
                gpu_id=self.gpu_id,
                stream=self.stream,
                is_complex=is_complex,
                edge=edge,
                is_normal=False,
            )
            for direction in directions
        ]

        self.DtDs = [
            FirstDerivative(
                direction=direction,
                sz_im=sz_im,
                gpu_id=self.gpu_id,
                stream=self.stream,
                is_complex=is_complex,
                edge=edge,
                is_normal=True,
            )
            for direction in directions
        ]

    def __del__(self):
        self.Ds = None
        self.DtDs = None

    def run(
        self,
        data,
        mu,
        lams,
        gam: float = 0,
        init=None,
        niter_inner: int = 2,
        niter_outer: int = 100,
        niter_solver: int = 4,
        wavelet_name: str = "",
        wavelet_full: bool = False,
        solver: str = "cg",
        save_every: Optional[int] = None,
        save_fraction: Optional[float] = None,
        on_checkpoint: Optional[Callable[[np.ndarray, int], None]] = None,
    ):
        if self.economize_gpu:
            return self.run_economize_gpu(
                data,
                mu,
                lams,
                gam,
                init,
                niter_inner,
                niter_outer,
                niter_solver,
                wavelet_name,
                wavelet_full,
                solver,
                save_every,
                save_fraction,
                on_checkpoint,
            )
        else:
            return self.run_full_gpu(
                data,
                mu,
                lams,
                gam,
                init,
                niter_inner,
                niter_outer,
                niter_solver,
                wavelet_name,
                wavelet_full,
                solver,
                save_every,
                save_fraction,
                on_checkpoint,
            )

    def run_full_gpu(
        self,
        data,
        mu,
        lams,
        gam,
        init,
        niter_inner,
        niter_outer,
        niter_solver,
        wavelet_name,
        wavelet_full,
        solver,
        save_every: Optional[int] = None,
        save_fraction: Optional[float] = None,
        on_checkpoint: Optional[Callable[[np.ndarray, int], None]] = None,
    ):

        checkpoint_iters: Set[int] = set()
        if save_every is not None and save_every > 0:
            checkpoint_iters = set(range(save_every, niter_outer + 1, save_every))
        elif save_fraction is not None and 0 < save_fraction < 1:
            num_saves = int(1 / save_fraction)
            checkpoint_iters = {
                int(niter_outer * k * save_fraction) for k in range(1, num_saves + 1)
            }
        intermediates: List[np.ndarray] = [] if on_checkpoint is None else []

        if len(lams) == 1:
            is_lam_scalar = True
        else:
            is_lam_scalar = all([np.isclose(lam, lams[0]) for lam in lams[1:]])

        f = to_device_array(
            data,
            ravel=True,
            gpu_id=self.gpu_id,
            stream=self.stream,
            dtype_real="float32",
            dtype_complex="complex64",
        )

        if init is None:
            u = self.model.rmatvec(f)
            with cp.cuda.Device(self.gpu_id), self.stream as _:
                grad_b_init = 0.5 * mu * u
        else:
            u = to_device_array(
                init,
                ravel=True,
                gpu_id=self.gpu_id,
                stream=self.stream,
                dtype_real="float32",
                dtype_complex="complex64",
            )
            grad_b_init = self.model.rmatvec(f)
            with cp.cuda.Device(self.gpu_id), self.stream as _:
                grad_b_init *= 0.5 * mu

        with cp.cuda.Device(self.gpu_id), self.stream as _:
            f = None

        with cp.cuda.Device(self.gpu_id), self.stream as _:
            ds = [cp.zeros(u.size, dtype=u.dtype) for _ in range(self.ndim)]
            bs = [cp.zeros(u.size, dtype=u.dtype) for _ in range(self.ndim)]
            if not is_lam_scalar:
                s_ts = [cp.zeros(u.size, dtype="float32") for _ in range(self.ndim)]

        if wavelet_name and (gam > 0):
            use_wavelet = True
            W = WaveletOperator(
                sz_im=self.sz_im,
                wavelet=wavelet_name,
                gpu_id=self.gpu_id,
                stream=self.stream,
                full=wavelet_full,
            )
            w = W.matvec(u)
            with cp.cuda.Device(self.gpu_id), self.stream as _:
                bw = cp.zeros((W.shape[0],), dtype=u.dtype)
        else:
            use_wavelet = False

        grad_model = _GradModel(
            self.model, self.DtDs, mu, lams, gam, self.gpu_id, self.stream
        )

        for _outer in range(niter_outer):
            for _inner in range(niter_inner):
                # Step 1 (update u)

                with cp.cuda.Device(self.gpu_id), self.stream as _:
                    grad_b = grad_b_init.copy()
                    for lam, d, b, D in zip(lams, ds, bs, self.Ds):
                        grad_b += 0.5 * lam * (D.rmatvec(d - b))

                    if use_wavelet:
                        w_minus_bw = w - bw
                        grad_b += 0.5 * gam * W.rmatvec(w_minus_bw)

                with cp.cuda.Device(self.gpu_id), self.stream as _:
                    if solver == "gmres":
                        u = gmres(grad_model, grad_b, x0=u, maxiter=niter_solver)[0]
                    else:
                        u = cg(grad_model, grad_b, x0=u, maxiter=niter_solver)[0]

                # Step 2 (update d)

                with cp.cuda.Device(self.gpu_id), self.stream as _:
                    Du_bs = [D.matvec(u) + b for D, b in zip(self.Ds, bs)]

                    s = cp.abs(Du_bs[0]) ** 2
                    for Du_b in Du_bs[1:]:
                        s += cp.abs(Du_b) ** 2
                    s = cp.sqrt(s)

                if is_lam_scalar:
                    s_ts = soft_thresholding_ratio(
                        s, 1.0 / lams[0], gpu_id=self.gpu_id, stream=self.stream
                    )
                else:
                    for i in range(self.ndim):
                        s_ts[i] = soft_thresholding_ratio(
                            s, 1.0 / lams[i], gpu_id=self.gpu_id, stream=self.stream
                        )

                if is_lam_scalar:
                    with cp.cuda.Device(self.gpu_id), self.stream as _:
                        ds = [Du_b * s_ts for Du_b in Du_bs]
                else:
                    with cp.cuda.Device(self.gpu_id), self.stream as _:
                        ds = [Du_b * s_t for Du_b, s_t in zip(Du_bs, s_ts)]

                if use_wavelet:
                    W_u = W.matvec(u)

                    with cp.cuda.Device(self.gpu_id), self.stream as stream:
                        w = soft_thresholding(
                            W_u + bw, 1.0 / gam, gpu_id=self.gpu_id, stream=stream
                        )

            # Bregman update

            with cp.cuda.Device(self.gpu_id), self.stream as _:
                bs = [Du_b - d for Du_b, d in zip(Du_bs, ds)]
                if use_wavelet:
                    bw += W_u - w

            # Checkpointing

            iter_idx = _outer + 1
            if iter_idx in checkpoint_iters:
                u_host = from_device_array(u, self.gpu_id, self.stream)
                if on_checkpoint:
                    on_checkpoint(u_host, _outer)
                else:
                    intermediates.append(u_host)

        if cp.get_array_module(data) == np:
            final_u = from_device_array(u, self.gpu_id, self.stream)
        else:
            final_u = u

        if on_checkpoint or (save_every is None and save_fraction is None):
            return final_u
        else:
            return final_u, intermediates

    def run_economize_gpu(
        self,
        data,
        mu,
        lams,
        gam,
        init,
        niter_inner,
        niter_outer,
        niter_solver,
        wavelet_name,
        wavelet_full,
        solver,
        save_every: Optional[int] = None,
        save_fraction: Optional[float] = None,
        on_checkpoint: Optional[Callable[[np.ndarray, int], None]] = None,
    ):

        checkpoint_iters: Set[int] = set()
        if save_every is not None and save_every > 0:
            checkpoint_iters = set(range(save_every, niter_outer + 1, save_every))
        elif save_fraction is not None and 0 < save_fraction < 1:
            num_saves = int(1 / save_fraction)
            checkpoint_iters = {
                int(niter_outer * k * save_fraction) for k in range(1, num_saves + 1)
            }
        intermediates: List[np.ndarray] = [] if on_checkpoint is None else []

        if len(lams) == 1:
            is_lam_scalar = True
        else:
            is_lam_scalar = all([np.isclose(lam, lams[0]) for lam in lams[1:]])

        f = to_device_array(
            data,
            ravel=True,
            gpu_id=self.gpu_id,
            stream=self.stream,
            dtype_real="float32",
            dtype_complex="complex64",
        )

        if init is None:
            u = self.model.rmatvec(f)
            with cp.cuda.Device(self.gpu_id), self.stream as _:
                grad_b_init = 0.5 * mu * u
        else:
            u = to_device_array(
                init,
                ravel=True,
                gpu_id=self.gpu_id,
                stream=self.stream,
                dtype_real="float32",
                dtype_complex="complex64",
            )
            grad_b_init = self.model.rmatvec(f)
            with cp.cuda.Device(self.gpu_id), self.stream as _:
                grad_b_init *= 0.5 * mu

        with cp.cuda.Device(self.gpu_id), self.stream as _:
            f = None

        if np.iscomplexobj(data):
            dtype = np.complex64
        else:
            dtype = np.float32

        ds = np.zeros((self.ndim, u.size), dtype=dtype)
        bs = np.zeros((self.ndim, u.size), dtype=dtype)
        Du_bs = np.zeros((self.ndim, u.size), dtype=dtype)

        if is_lam_scalar:
            s_ts = np.zeros(u.size, dtype="float32")
        else:
            s_ts = np.zeros((self.ndim, u.size), dtype="float32")

        if wavelet_name and (gam > 0):
            use_wavelet = True
            W = WaveletOperator(
                sz_im=self.sz_im,
                wavelet=wavelet_name,
                gpu_id=self.gpu_id,
                stream=self.stream,
                full=wavelet_full,
            )
            h_u = from_device_array(u, self.gpu_id, self.stream)
            w = W.matvec(h_u)
            bw = np.zeros((W.shape[0],), dtype=dtype)
        else:
            use_wavelet = False

        grad_model = _GradModel(
            self.model, self.DtDs, mu, lams, gam, self.gpu_id, self.stream
        )

        for _outer in range(niter_outer):
            for _inner in range(niter_inner):

                # Step 1 (update d_u)

                with cp.cuda.Device(self.gpu_id), self.stream as _:
                    grad_b = grad_b_init.copy()

                for lam, d, b, D in zip(lams, ds, bs, self.Ds):
                    d_minus_b = to_device_array(
                        d - b,
                        ravel=True,
                        gpu_id=self.gpu_id,
                        stream=self.stream,
                        dtype_real="float32",
                        dtype_complex="complex64",
                    )
                    with cp.cuda.Device(self.gpu_id), self.stream as _:
                        d_minus_b *= 0.5 * lam
                        grad_b += D.rmatvec(d_minus_b)

                if use_wavelet:
                    w_minus_bw = w - bw
                    with cp.cuda.Device(self.gpu_id), self.stream as stream:
                        grad_b += (
                            0.5
                            * gam
                            * to_device_array(
                                W.rmatvec(w_minus_bw),
                                ravel=True,
                                gpu_id=self.gpu_id,
                                stream=stream,
                                dtype_real="float32",
                                dtype_complex="complex64",
                            )
                        )

                with cp.cuda.Device(self.gpu_id), self.stream as _:
                    if solver == "gmres":
                        u = gmres(grad_model, grad_b, x0=u, maxiter=niter_solver)[0]
                    else:
                        u = cg(grad_model, grad_b, x0=u, maxiter=niter_solver)[0]

                # Step 2 (update d)

                for i in range(self.ndim):
                    Du_bs[i] = bs[i] + from_device_array(
                        self.Ds[i].matvec(u), self.gpu_id, self.stream
                    )

                # s = np.sqrt(np.sum(np.abs(Du_bs) ** 2, axis=0))
                s = np.linalg.norm(Du_bs, axis=0)

                if is_lam_scalar:
                    s_ts = soft_thresholding_ratio(
                        s, 1.0 / lams[0], gpu_id=self.gpu_id, stream=self.stream
                    )
                else:
                    for i in range(self.ndim):
                        s_ts[i] = soft_thresholding_ratio(
                            s, 1.0 / lams[i], gpu_id=self.gpu_id, stream=self.stream
                        )

                ds = Du_bs * s_ts

                if use_wavelet:
                    h_u = from_device_array(u, self.gpu_id, self.stream)
                    W_u = W.matvec(h_u)
                    w = soft_thresholding(
                        W_u + bw, 1.0 / gam, gpu_id=self.gpu_id, stream=self.stream
                    )

            # Bregman update

            bs = Du_bs - ds

            if use_wavelet:
                bw += W_u - w

            # Checkpointing

            iter_idx = _outer + 1
            if iter_idx in checkpoint_iters:
                u_host = from_device_array(u, self.gpu_id, self.stream)
                if on_checkpoint:
                    on_checkpoint(u_host, _outer)
                else:
                    intermediates.append(u_host)

        if cp.get_array_module(data) == np:
            final_u = from_device_array(u, self.gpu_id, self.stream)
        else:
            final_u = u

        if on_checkpoint or (save_every is None and save_fraction is None):
            return final_u
        else:
            return final_u, intermediates
