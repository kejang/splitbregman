import logging

import numpy as np
import cupy
from cupyx.scipy.sparse.linalg import LinearOperator
from cupyx.scipy.sparse.linalg import gmres, cg

from .derivative.FirstDerivative import FirstDerivative
from .thresholding.thresholding import soft_thresholding
from .utils.utils import to_device_array, from_device_array
from .WaveletOperator import WaveletOperator

logger = logging.getLogger(__name__)

mempool = cupy.get_default_memory_pool()


def clean_all_gpu_mem():
    mempool.free_all_blocks()


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

        with cupy.cuda.Device(self.gpu_id), self.stream as _:
            y = 0.5*self.mu*self.model.rmatvec(self.model.matvec(x))

            for lam, DtD in zip(self.lams, self.DtDs):
                y += 0.5*lam*DtD.matvec(x)

            if self.gam > 0:
                y += 0.5*self.gam*x

        return y

    def _matvec(self, x):

        return self.run(x)

    def _rmatvec(self, x):

        return self.run(x)


class SplitBregman():

    def __init__(
            self,
            model,
            sz_im,
            edge=False,
            economize_gpu=False):

        self.model = model
        self.sz_im = sz_im
        self.gpu_id = model.gpu_id
        self.stream = model.stream
        self.economize_gpu = economize_gpu

        is_complex = np.iscomplexobj(np.ones((1,), dtype=model.dtype))

        if len(sz_im) == 2:
            directions = ['x', 'y']
        else:
            directions = ['x', 'y', 'z']

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
            gam=0,
            init=None,
            niter_inner=2,
            niter_outer=100,
            niter_solver=4,
            wavelet_name='',
            wavelet_full=False,
            show=False,
            solver='cg'):

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
                show,
                solver)
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
                show,
                solver)

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
            show,
            solver):

        f = to_device_array(
            data,
            ravel=True,
            gpu_id=self.gpu_id,
            stream=self.stream,
            dtype_real='float32',
            dtype_complex='complex64'
        )

        if init is None:
            u = self.model.rmatvec(f)
        else:
            u = to_device_array(
                init,
                ravel=True,
                gpu_id=self.gpu_id,
                stream=self.stream,
                dtype_real='float32',
                dtype_complex='complex64'
            )

        n_dims = len(lams)

        with cupy.cuda.Device(self.gpu_id), self.stream as _:
            ds = [cupy.zeros(u.size, dtype=u.dtype) for _ in range(n_dims)]
            bs = [cupy.zeros(u.size, dtype=u.dtype) for _ in range(n_dims)]

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
            with cupy.cuda.Device(self.gpu_id), self.stream as _:
                bw = cupy.zeros((W.shape[0],), dtype=u.dtype)
        else:
            use_wavelet = False

        grad_model = _GradModel(
            self.model,
            self.DtDs,
            mu,
            lams,
            gam,
            self.gpu_id,
            self.stream
        )

        for _outer in range(niter_outer):
            if show:
                with cupy.cuda.Device(self.gpu_id), self.stream as _:
                    u_old = u.copy()

            for _inner in range(niter_inner):

                # Step 1 (update u)

                with cupy.cuda.Device(self.gpu_id), self.stream as _:
                    grad_b = 0.5*mu*self.model.rmatvec(f)
                    for lam, d, b, D in zip(lams, ds, bs, self.Ds):
                        grad_b += 0.5*lam*(D.rmatvec(d - b))

                    if use_wavelet:
                        w_minus_bw = w - bw
                        grad_b += 0.5*gam*W.rmatvec(w_minus_bw)

                with cupy.cuda.Device(self.gpu_id), self.stream as _:
                    if solver == 'gmres':
                        u = gmres(
                            grad_model, grad_b, x0=u, maxiter=niter_solver
                        )[0]
                    else:
                        u = cg(
                            grad_model, grad_b, x0=u, maxiter=niter_solver
                        )[0]

                # Step 2 (update d)

                with cupy.cuda.Device(self.gpu_id), self.stream as stream:
                    Du_bs = [D.matvec(u) + b for D, b in zip(self.Ds, bs)]

                    s = cupy.abs(Du_bs[0])**2
                    for Du_b in Du_bs[1:]:
                        s += cupy.abs(Du_b)**2

                    s = cupy.sqrt(s) + 1e-15

                s_ts = [
                    soft_thresholding(
                        s,
                        1.0/lam,
                        gpu_id=self.gpu_id,
                        stream=stream
                    )
                    for lam in lams
                ]

                with cupy.cuda.Device(self.gpu_id), self.stream as _:
                    ds = [Du_b*s_t/s for Du_b, s_t in zip(Du_bs, s_ts)]

                if use_wavelet:
                    W_u = W.matvec(u)

                    with cupy.cuda.Device(self.gpu_id), self.stream as _:
                        w = soft_thresholding(
                            W_u + bw,
                            1.0/gam,
                            gpu_id=self.gpu_id,
                            stream=stream
                        )

            # Bregman update

            with cupy.cuda.Device(self.gpu_id), stream as _:
                bs = [Du_b - d for Du_b, d in zip(Du_bs, ds)]

                if use_wavelet:
                    bw += W_u - w

            if show:
                with cupy.cuda.Device(self.gpu_id), self.stream as _:
                    print(cupy.linalg.norm(u - u_old))

        if cupy.get_array_module(data) == np:
            return from_device_array(u, self.gpu_id, self.stream)
        else:
            return u

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
            show,
            solver):

        f = from_device_array(data, self.gpu_id, self.stream).ravel()

        if np.iscomplexobj(f):
            f = f.astype('complex64')
        else:
            f = f.astype('float32')

        if init is None:
            h_u = self.model.rmatvec(f)
        else:
            h_u = from_device_array(init, self.gpu_id, self.stream).ravel()

        if np.iscomplexobj(h_u):
            h_u = h_u.astype('complex64')
        else:
            h_u = h_u.astype('float32')

        n_dims = len(lams)

        ds = [np.zeros(h_u.size, dtype=h_u.dtype) for _ in range(n_dims)]
        bs = [np.zeros(h_u.size, dtype=h_u.dtype) for _ in range(n_dims)]

        if wavelet_name and (gam > 0):
            use_wavelet = True
            W = WaveletOperator(
                sz_im=self.sz_im,
                wavelet=wavelet_name,
                gpu_id=self.gpu_id,
                stream=self.stream,
                full=wavelet_full,
            )
            w = W.matvec(h_u)
            bw = np.zeros((W.shape[0],), dtype=h_u.dtype)
        else:
            use_wavelet = False

        grad_model = _GradModel(
            self.model,
            self.DtDs,
            mu,
            lams,
            gam,
            self.gpu_id,
            self.stream
        )

        for _outer in range(niter_outer):
            if show:
                h_u_old = h_u.copy()

            for _inner in range(niter_inner):

                # Step 1 (update d_u)

                h_grad_b = 0.5*mu*self.model.rmatvec(f)
                for lam, d, b, D in zip(lams, ds, bs, self.Ds):
                    h_grad_b += 0.5*lam*(D.rmatvec(d - b))

                if use_wavelet:
                    w_minus_bw = w - bw
                    h_grad_b += 0.5*gam*W.rmatvec(w_minus_bw)

                with cupy.cuda.Device(self.gpu_id), self.stream as _:
                    d_u = to_device_array(
                        h_u,
                        gpu_id=self.gpu_id,
                        stream=self.stream,
                        dtype_real='float32',
                        dtype_complex='complex64'
                    )

                    d_grad_b = to_device_array(
                        h_grad_b,
                        gpu_id=self.gpu_id,
                        stream=self.stream,
                        dtype_real='float32',
                        dtype_complex='complex64'
                    )

                with cupy.cuda.Device(self.gpu_id), self.stream as _:
                    if solver == 'gmres':
                        d_u = gmres(
                            grad_model, d_grad_b, x0=d_u, maxiter=niter_solver
                        )[0]
                    else:
                        d_u = cg(
                            grad_model, d_grad_b, x0=d_u, maxiter=niter_solver
                        )[0]

                with cupy.cuda.Device(self.gpu_id), self.stream as _:
                    h_u = from_device_array(d_u, self.gpu_id, self.stream)

                # Step 2 (update d)

                Du_bs = [D.matvec(h_u) + b for D, b in zip(self.Ds, bs)]

                s = np.abs(Du_bs[0])**2
                for Du_b in Du_bs[1:]:
                    s += np.abs(Du_b)**2

                s = np.sqrt(s) + 1e-15

                s_ts = [
                    soft_thresholding(
                        s,
                        1.0/lam,
                        gpu_id=self.gpu_id,
                        stream=self.stream
                    )
                    for lam in lams
                ]

                ds = [Du_b*s_t/s for Du_b, s_t in zip(Du_bs, s_ts)]

                if use_wavelet:
                    W_u = W.matvec(h_u)
                    w = soft_thresholding(
                        W_u + bw,
                        1.0/gam,
                        gpu_id=self.gpu_id,
                        stream=self.stream
                    )

            # Bregman update

            bs = [Du_b - d for Du_b, d in zip(Du_bs, ds)]

            if use_wavelet:
                bw += W_u - w

            if show:
                print(np.linalg.norm(h_u - h_u_old))

        if cupy.get_array_module(data) == np:
            return h_u
        else:
            return to_device_array(h_u,
                                   gpu_id=self.gpu_id,
                                   stream=self.stream,
                                   dtype_real='float32',
                                   dtype_complex='complex64')
