import logging
import numpy as np

from scipy.sparse.linalg import LinearOperator as LinearOperatorScipy
from scipy.sparse.linalg import cg as cg_scipy
from scipy.sparse.linalg import gmres as gmres_scipy

import cupy
from cupyx.scipy.sparse.linalg import LinearOperator as LinearOperatorCupy
from cupyx.scipy.sparse.linalg import gmres as gmres_cupy
from cupyx.scipy.sparse.linalg import cg as cg_cupy

from waveletop.WaveletOperator import WaveletOperator
from waveletop.WaveletTimeOperator import WaveletTimeOperator

from derivative.FirstDerivativeCUDA import FirstDerivativeCUDA
from derivative.FirstDerivativeCPP import FirstDerivativeCPP
from derivative.FirstDerivativeTime import FirstDerivativeTime

from thresholding.soft_cpp import soft_thresholding as soft_thresholding_cpp
from thresholding.soft_cuda import soft_thresholding as soft_thresholding_cuda

from .utils.utils import to_device_array, from_device_array

logger = logging.getLogger(__name__)

mempool = cupy.get_default_memory_pool()


def clean_all_gpu_mem():
    mempool.free_all_blocks()


class HtHCupy(LinearOperatorCupy):

    def __init__(
        self,
        model,
        DtDs,
        mu,
        lam,
        gam,
        gam_t,
        gpu_id,
        stream
    ):
        self.shape = tuple([model.shape[1]]*2)
        self.dtype = model.dtype
        self.model = model
        self.DtDs = DtDs
        self.mu = mu
        self.lam = lam
        self.gam = gam
        self.gam_t = gam_t
        self.gpu_id = gpu_id
        self.stream = stream

    def run(self, x):

        with cupy.cuda.Device(self.gpu_id), self.stream as _:
            y = self.mu*self.model.rmatvec(self.model.matvec(x))

            if self.lam > 0:
                for DtD in self.DtDs:
                    y += self.lam*DtD._matvec(x)

            if self.gam > 0:
                y += self.gam*x

            if self.gam_t > 0:
                y += self.gam_t*x

        return y

    def _matvec(self, x):

        return self.run(x)

    def _rmatvec(self, x):

        return self.run(x)


class HtHStackedCupy(LinearOperatorCupy):

    def __init__(
        self,
        model_stack,
        DtDs,
        DtD_t,
        mu,
        lam,
        lam_t,
        gam,
        gam_t,
        gpu_id,
        stream
    ):

        nt = len(model_stack)
        self.shape = tuple([nt*model_stack[0].shape[1]]*2)
        self.dtype = model_stack[0].dtype
        self.model_stack = model_stack
        self.DtDs = DtDs
        self.DtD_t = DtD_t
        self.mu = mu
        self.lam = lam
        self.lam_t = lam_t
        self.gam = gam
        self.gam_t = gam_t
        self.gpu_id = gpu_id
        self.stream = stream

    def run(self, x):

        y_stack = []
        end_ind = 0
        for stack_ind in range(len(self.model_stack)):
            model_t = self.model_stack[stack_ind]
            nelem = model_t.shape[1]
            beg_ind = end_ind
            end_ind = beg_ind + nelem
            x_t = x[beg_ind:end_ind]

            with cupy.cuda.Device(self.gpu_id), self.stream as _:
                y_t = self.mu*model_t.rmatvec(model_t.matvec(x_t))

                if self.lam > 0:
                    for DtD in self.DtDs:
                        y_t += self.lam*DtD._matvec(x_t)

                if self.gam > 0:
                    y_t += self.gam*x_t

            y_stack.append(y_t)

        if cupy.get_array_module(y_t) == np:
            y = np.concatenate(y_stack).astype(self.dtype)
        else:
            y = cupy.concatenate(y_stack).astype(self.dtype)

        if self.lam_t > 0:
            y += self.lam_t*self.DtD_t._matvec(x)

        if self.gam_t > 0:
            y += self.gam_t*x

        return y

    def _matvec(self, x):

        return self.run(x)

    def _rmatvec(self, x):

        return self.run(x)


class HtHScipy(LinearOperatorScipy):

    def __init__(
        self,
        model,
        DtDs,
        mu,
        lam,
        gam,
        gam_t,
        gpu_id,
        stream
    ):
        self.shape = tuple([model.shape[1]]*2)
        self.dtype = model.dtype
        self.model = model
        self.DtDs = DtDs
        self.mu = mu
        self.lam = lam
        self.gam = gam
        self.gam_t = gam_t
        self.gpu_id = gpu_id
        self.stream = stream

    def run(self, x):

        with cupy.cuda.Device(self.gpu_id), self.stream as _:
            y = self.mu*self.model.rmatvec(self.model.matvec(x))

            if self.lam > 0:
                for DtD in self.DtDs:
                    y += self.lam*DtD._matvec(x)

            if self.gam > 0:
                y += self.gam*x

            if self.gam_t > 0:
                y += self.gam_t*x

        return y

    def _matvec(self, x):

        return self.run(x)

    def _rmatvec(self, x):

        return self.run(x)


class HtHStackedScipy(LinearOperatorScipy):

    def __init__(
        self,
        model_stack,
        DtDs,
        DtD_t,
        mu,
        lam,
        lam_t,
        gam,
        gam_t,
        gpu_id,
        stream
    ):

        nt = len(model_stack)
        self.shape = tuple([nt*model_stack[0].shape[1]]*2)
        self.dtype = model_stack[0].dtype
        self.model_stack = model_stack
        self.DtDs = DtDs
        self.DtD_t = DtD_t
        self.mu = mu
        self.lam = lam
        self.lam_t = lam_t
        self.gam = gam
        self.gam_t = gam_t
        self.gpu_id = gpu_id
        self.stream = stream

    def run(self, x):

        y_stack = []
        end_ind = 0
        for stack_ind in range(len(self.model_stack)):
            model_t = self.model_stack[stack_ind]
            nelem = model_t.shape[1]
            beg_ind = end_ind
            end_ind = beg_ind + nelem
            x_t = x[beg_ind:end_ind]

            with cupy.cuda.Device(self.gpu_id), self.stream as _:
                y_t = self.mu*model_t.rmatvec(model_t.matvec(x_t))

                if self.lam > 0:
                    for DtD in self.DtDs:
                        y_t += self.lam*DtD._matvec(x_t)

                if self.gam > 0:
                    y_t += self.gam*x_t

            y_stack.append(y_t)

        if cupy.get_array_module(y_t) == np:
            y = np.concatenate(y_stack).astype(self.dtype)
        else:
            y = cupy.concatenate(y_stack).astype(self.dtype)

        if self.lam_t > 0:
            y += self.lam_t*self.DtD_t._matvec(x)

        if self.gam_t > 0:
            y += self.gam_t*x

        return y

    def _matvec(self, x):

        return self.run(x)

    def _rmatvec(self, x):

        return self.run(x)


class SplitBregman():

    def __init__(self, model_stack, sz_im, gpu_id, stream):

        self.model_stack = model_stack
        self.nt = len(model_stack)
        self.sz_im = sz_im
        self.nu = np.int64(np.prod(sz_im))
        self.gpu_id = gpu_id
        self.stream = stream

        is_complex = np.iscomplexobj(
            np.ones((1,), dtype=model_stack[0].dtype)
        )
        if is_complex:
            self.dtype = 'complex64'
        else:
            self.dtype = 'float32'

        if len(sz_im) == 2:
            self.directions = ['x', 'y']
        else:
            self.directions = ['x', 'y', 'z']

    def run(
            self,
            data_stack,
            mu,
            lam,
            lam_t=0,
            gam=0,
            gam_t=0,
            init_stack=None,
            niter_inner=2,
            niter_outer=100,
            niter_solver=4,
            max_workers=1,
            wavelet_name='sym4',
            wavelet_full=False,
            solver='cg',
            isotropic_tv=True,
            gpu_derivative=True,
            gpu_thresholding=True,
            gpu_holding=True,
            gpu_solver=True):

        ndim = len(self.directions)

        if gpu_derivative:
            FirstDerivative = FirstDerivativeCUDA
        else:
            FirstDerivative = FirstDerivativeCPP

        if gpu_thresholding:
            soft_thresholding = soft_thresholding_cuda
        else:
            soft_thresholding = soft_thresholding_cpp

        if gpu_holding:
            f_stack = [
                to_device_array(
                    data_t.astype(self.dtype).ravel(),
                    gpu_id=self.gpu_id,
                    stream=self.stream
                )
                for data_t in data_stack
            ]
        else:
            f_stack = [
                from_device_array(
                    data_t.astype(self.dtype).ravel(),
                    self.gpu_id,
                    self.stream
                )
                for data_t in data_stack
            ]

        if init_stack is None:
            u_stack = []
            for stack_ind, f_t in enumerate(f_stack):
                model_t = self.model_stack[stack_ind]
                u_stack.append(model_t.rmatvec(f_t))
        else:
            if gpu_holding:
                u_stack = [
                    to_device_array(
                        init_t.astype(self.dtype).ravel(),
                        self.gpu_id,
                        self.stream
                    )
                    for init_t in init_stack
                ]
            else:
                u_stack = [
                    from_device_array(
                        init_t.astype(self.dtype).ravel(),
                        self.gpu_id,
                        self.stream
                    )
                    for init_t in init_stack
                ]

        if lam > 0:
            use_spatial_tv = True

            Ds = [
                FirstDerivative(
                    direction=direction,
                    sz_im=self.sz_im,
                    dtype=self.dtype,
                    gpu_id=self.gpu_id,
                    stream=self.stream,
                    is_normal=False,
                )
                for direction in self.directions
            ]

            DtDs = [
                FirstDerivative(
                    direction=direction,
                    sz_im=self.sz_im,
                    dtype=self.dtype,
                    gpu_id=self.gpu_id,
                    stream=self.stream,
                    is_normal=True,
                )
                for direction in self.directions
            ]

            ds_stack = []
            bs_stack = []
            if gpu_holding:
                with cupy.cuda.Device(self.gpu_id), self.stream as _:
                    for stack_ind in range(self.nt):
                        ds_stack.append(
                            [cupy.zeros(self.nu, dtype=self.dtype)
                             for i in range(ndim)]
                        )
                        bs_stack.append(
                            [cupy.zeros(self.nu, dtype=self.dtype)
                             for i in range(ndim)]
                        )
            else:
                for stack_ind in range(self.nt):
                    ds_stack.append(
                        [np.zeros(self.nu, dtype=self.dtype)
                         for _ in range(ndim)]
                    )
                    bs_stack.append(
                        [np.zeros(self.nu, dtype=self.dtype)
                         for _ in range(ndim)]
                    )

        else:
            use_spatial_tv = False

        if lam_t > 0:
            use_time_tv = True

            D_t = FirstDerivativeTime(
                nv=np.prod(self.sz_im).astype(np.int64),
                nt=self.nt,
                dtype=self.dtype,
                gpu_id=self.gpu_id,
                stream=self.stream,
                is_normal=False,
            )

            DtD_t = FirstDerivativeTime(
                nv=np.prod(self.sz_im).astype(np.int64),
                nt=self.nt,
                dtype=self.dtype,
                gpu_id=self.gpu_id,
                stream=self.stream,
                is_normal=True,
            )

            dt_stack = []
            bt_stack = []
            if gpu_holding:
                with cupy.cuda.Device(self.gpu_id), self.stream as _:
                    for stack_ind in range(self.nt):
                        dt_stack.append(cupy.zeros(self.nu, dtype=self.dtype))
                        bt_stack.append(cupy.zeros(self.nu, dtype=self.dtype))
            else:
                for _ in range(self.nt):
                    dt_stack.append(np.zeros(self.nu, dtype=self.dtype))
                    bt_stack.append(np.zeros(self.nu, dtype=self.dtype))

        else:
            use_time_tv = False

        if gam > 0:
            use_spatial_wavelet = True

            W = WaveletOperator(
                sz_im=self.sz_im,
                wavelet=wavelet_name,
                gpu_id=self.gpu_id,
                stream=self.stream,
                full=wavelet_full,
            )

            w_stack = [W._matvec(u_t) for u_t in u_stack]

            if gpu_holding:
                with cupy.cuda.Device(self.gpu_id), self.stream as _:
                    bw_stack = [cupy.zeros((W.shape[0],), dtype=self.dtype)
                                for i in range(self.nt)]
            else:
                bw_stack = [np.zeros((W.shape[0],), dtype=self.dtype)
                            for _ in range(self.nt)]

        else:
            use_spatial_wavelet = False

        if gam_t > 0:
            use_time_wavelet = True

            W_t = WaveletTimeOperator(
                nv=np.prod(self.sz_im).astype('int64'),
                nt=self.nt,
                max_workers=max_workers,
                wavelet=wavelet_name,
                gpu_id=self.gpu_id,
                stream=self.stream,
            )

            if gpu_holding:
                h_u_stack = [
                    from_device_array(u_t, self.gpu_id, self.stream)
                    for u_t in u_stack
                ]
            else:
                h_u_stack = u_stack

            wt_stack = W_t.forward(h_u_stack)
            h_u_stack = None

            bwt_stack = [np.zeros(wt_stack[0].size, dtype=self.dtype)
                         for _ in range(len(wt_stack))]

        else:
            use_time_wavelet = False

        if use_time_tv:
            if gpu_solver:
                HtHStacked = HtHStackedCupy(
                    self.model_stack,
                    DtDs,
                    DtD_t,
                    mu,
                    lam,
                    lam_t,
                    gam,
                    gam_t,
                    self.gpu_id,
                    self.stream
                )
            else:
                HtHStacked = HtHStackedScipy(
                    self.model_stack,
                    DtDs,
                    DtD_t,
                    mu,
                    lam,
                    lam_t,
                    gam,
                    gam_t,
                    self.gpu_id,
                    self.stream
                )
        else:
            HtHs = []
            for stack_ind in range(len(self.model_stack)):
                model_t = self.model_stack[stack_ind]
                if gpu_solver:
                    HtHs.append(
                        HtHCupy(
                            model_t,
                            DtDs,
                            mu,
                            lam,
                            gam,
                            gam_t,
                            self.gpu_id,
                            self.stream
                        )
                    )
                else:
                    HtHs.append(
                        HtHScipy(
                            model_t,
                            DtDs,
                            mu,
                            lam,
                            gam,
                            gam_t,
                            self.gpu_id,
                            self.stream
                        )
                    )

        for _outer in range(niter_outer):
            for _inner in range(niter_inner):
                # Step 1 (update u)
                # -----------------

                # Step 1-1: compute Hty (right-side of normal equation)

                Hty_stack = []

                for stack_ind in range(self.nt):
                    f_t = f_stack[stack_ind]
                    model_t = self.model_stack[stack_ind]
                    with cupy.cuda.Device(self.gpu_id), self.stream as _:
                        Hty_t = mu*model_t.rmatvec(f_t)

                    if use_spatial_tv:
                        ds_t = ds_stack[stack_ind]
                        bs_t = bs_stack[stack_ind]
                        with cupy.cuda.Device(self.gpu_id), self.stream as _:
                            for d_t, b_t, D in zip(ds_t, bs_t, Ds):
                                Hty_t += lam*(D._rmatvec(d_t - b_t))

                    if use_spatial_wavelet:
                        w_t = w_stack[stack_ind]
                        bw_t = bw_stack[stack_ind]
                        with cupy.cuda.Device(self.gpu_id), self.stream as _:
                            w_minus_bw = w_t - bw_t
                            Hty_t += gam*W._rmatvec(w_minus_bw)

                    Hty_stack.append(Hty_t)

                if use_time_tv:
                    d_minus_b = [
                        lam_t*(dt - bt)
                        for dt, bt in zip(dt_stack, bt_stack)
                    ]

                    Hty_t_stack = D_t._rmatvec(d_minus_b)

                    for i in range(self.nt):
                        Hty_stack[i] += Hty_t_stack[i]

                    Hty_t_stack = None
                    d_minus_b = None

                if use_time_wavelet:
                    w_minus_b = [(wt_t - bwt_t)
                                 for wt_t, bwt_t in zip(wt_stack, bwt_stack)]
                    Hty_wt_stack = W_t.adjoint(w_minus_b)
                    w_minus_b = None

                    for i in range(self.nt):
                        if gpu_holding:
                            Hty_stack[i] += (
                                gam_t
                                * to_device_array(
                                    Hty_wt_stack[i], self.gpu_id, self.stream
                                )
                            )
                        else:
                            Hty_stack[i] += gam_t*Hty_wt_stack[i]

                    Hty_wt_stack = None

                # Step 1-2: Send Hty to host or device

                if use_time_tv:
                    if gpu_holding:
                        Hty = cupy.concatenate(Hty_stack).astype(self.dtype)
                    else:
                        Hty = np.concatenate(Hty_stack).astype(self.dtype)

                    Hty_stack = None

                    if gpu_solver:
                        if gpu_holding:
                            d_Hty = Hty
                        else:
                            d_Hty = to_device_array(
                                Hty, gpu_id=self.gpu_id, stream=self.stream
                            )
                            Hty = None
                    else:
                        if gpu_holding:
                            h_Hty = from_device_array(
                                Hty, gpu_id=self.gpu_id, stream=self.stream
                            )
                            Hty = None
                        else:
                            h_Hty = Hty
                else:
                    if gpu_solver:
                        if gpu_holding:
                            d_Hty_stack = Hty_stack
                        else:
                            d_Hty_stack = [
                                to_device_array(
                                    Hty_t,
                                    gpu_id=self.gpu_id,
                                    stream=self.stream
                                )
                                for Hty_t in Hty_stack
                            ]
                            Hty_stack = None
                    else:
                        if gpu_holding:
                            h_Hty_stack = [
                                from_device_array(
                                    Hty_t,
                                    gpu_id=self.gpu_id,
                                    stream=self.stream
                                )
                                for Hty_t in Hty_stack
                            ]
                            Hty_stack = None
                        else:
                            h_Hty_stack = Hty_stack

                # Step 1-3: Send u to host or device

                if use_time_tv:
                    if gpu_solver:
                        if gpu_holding:
                            d_u = cupy.concatenate(u_stack).astype(self.dtype)
                        else:
                            d_u = to_device_array(
                                np.concatenate(u_stack).astype(self.dtype),
                                gpu_id=self.gpu_id,
                                stream=self.stream
                            )
                    else:
                        if gpu_holding:
                            h_u = from_device_array(
                                cupy.concatenate(u_stack).astype(self.dtype),
                                gpu_id=self.gpu_id,
                                stream=self.stream
                            )
                        else:
                            h_u = np.concatenate(u_stack).astype(self.dtype)
                else:
                    if gpu_solver:
                        if gpu_holding:
                            d_u_stack = u_stack
                        else:
                            d_u_stack = [
                                to_device_array(
                                    u_t, gpu_id=self.gpu_id, stream=self.stream
                                )
                                for u_t in u_stack
                            ]
                    else:
                        if gpu_holding:
                            h_u_stack = [
                                from_device_array(
                                    u_t, gpu_id=self.gpu_id, stream=self.stream
                                )
                                for u_t in u_stack
                            ]
                        else:
                            h_u_stack = u_stack

                # Step 1-4: Solve the normal equation

                if use_time_tv:
                    if gpu_solver:
                        with cupy.cuda.Device(self.gpu_id), self.stream as _:
                            if solver == 'gmres':
                                d_u = gmres_cupy(
                                    HtHStacked,
                                    d_Hty,
                                    x0=d_u,
                                    maxiter=niter_solver
                                )[0]
                            else:
                                d_u = cg_cupy(
                                    HtHStacked,
                                    d_Hty,
                                    x0=d_u,
                                    maxiter=niter_solver
                                )[0]
                            d_u_stack = cupy.array_split(d_u, self.nt)
                            d_u = None
                    else:
                        if solver == 'gmres':
                            h_u = gmres_scipy(
                                HtHStacked,
                                h_Hty,
                                x0=h_u,
                                maxiter=niter_solver
                            )[0]
                        else:
                            h_u = cg_scipy(
                                HtHStacked,
                                h_Hty,
                                x0=h_u,
                                maxiter=niter_solver
                            )[0]
                        h_u_stack = np.array_split(h_u, self.nt)
                        h_u = None

                    if gpu_holding:
                        if gpu_solver:
                            u_stack = d_u_stack
                        else:
                            u_stack = [
                                to_device_array(u_t, self.gpu_id, self.stream)
                                for u_t in h_u_stack
                            ]
                            h_u_stack = None
                    else:
                        if gpu_solver:
                            u_stack = [
                                from_device_array(
                                    u_t, self.gpu_id, self.stream
                                )
                                for u_t in d_u_stack
                            ]
                            d_u_stack = None
                        else:
                            u_stack = h_u_stack
                else:
                    new_u_stack = []
                    if gpu_solver:
                        for u_t, Hty_t, HtH_t in zip(
                            d_u_stack, d_Hty_stack, HtHs
                        ):
                            with cupy.cuda.Device(self.gpu_id),\
                                    self.stream as _:
                                if solver == 'gmres':
                                    new_u_stack.append(
                                        gmres_cupy(
                                            HtH_t,
                                            Hty_t,
                                            x0=u_t,
                                            maxiter=niter_solver
                                        )[0]
                                    )
                                else:
                                    new_u_stack.append(
                                        cg_cupy(
                                            HtH_t,
                                            Hty_t,
                                            x0=u_t,
                                            maxiter=niter_solver
                                        )[0]
                                    )
                    else:
                        for u_t, Hty_t, HtH_t in zip(
                            h_u_stack, h_Hty_stack, HtHs
                        ):
                            if solver == 'gmres':
                                new_u_stack.append(
                                    gmres_scipy(
                                        HtH_t,
                                        Hty_t,
                                        x0=u_t,
                                        maxiter=niter_solver
                                    )[0]
                                )
                            else:
                                new_u_stack.append(
                                    cg_scipy(
                                        HtH_t,
                                        Hty_t,
                                        x0=u_t,
                                        maxiter=niter_solver
                                    )[0]
                                )

                    if gpu_holding:
                        if gpu_solver:
                            u_stack = new_u_stack
                        else:
                            u_stack = [
                                to_device_array(u_t, self.gpu_id, self.stream)
                                for u_t in new_u_stack
                            ]
                            new_u_stack = None
                    else:
                        if gpu_solver:
                            u_stack = [
                                from_device_array(
                                    u_t, self.gpu_id, self.stream
                                )
                                for u_t in new_u_stack
                            ]
                            new_u_stack = None
                        else:
                            u_stack = new_u_stack

                # Step 2 (update d)
                # -----------------

                # Step 2-1: total variation, spatial domain

                if use_spatial_tv:
                    Du_bs_stack = []
                    ds_stack = []

                    for stack_ind in range(self.nt):
                        u_t = u_stack[stack_ind]
                        bs_t = bs_stack[stack_ind]
                        with cupy.cuda.Device(self.gpu_id), self.stream as _:
                            Du_bs_t = [
                                D._matvec(u_t) + b_t
                                for D, b_t in zip(Ds, bs_t)
                            ]
                            Du_bs_stack.append(Du_bs_t)

                        ds_t = []
                        if isotropic_tv:
                            if gpu_holding:
                                xp = cupy
                            else:
                                xp = np
                            with cupy.cuda.Device(self.gpu_id),\
                                    self.stream as stream:
                                s = xp.abs(Du_bs_t[0])**2
                                for Du_b_t in Du_bs_t[1:]:
                                    s += xp.abs(Du_b_t)**2
                                s = xp.sqrt(s)
                                s_thresh = soft_thresholding(
                                    s,
                                    1.0/lam,
                                    gpu_id=self.gpu_id,
                                    stream=stream
                                )
                                s_thresh_over_s = s_thresh/(s + 1e-15)

                                for Du_b_t in Du_bs_t:
                                    ds_t.append(Du_b_t*s_thresh_over_s)
                        else:
                            for Du_b_t in Du_bs_t:
                                ds_t.append(
                                    soft_thresholding(
                                        Du_b_t,
                                        1.0/lam,
                                        gpu_id=self.gpu_id,
                                        stream=self.stream
                                    )
                                )
                        ds_stack.append(ds_t)

                # Step 2-2: total variation along time axis

                if use_time_tv:
                    Du_bt_stack = D_t._matvec(u_stack)
                    for stack_ind in range(self.nt):
                        Du_bt_stack[stack_ind] += bt_stack[stack_ind]

                    dt_stack = [
                        soft_thresholding(
                            Du_bt,
                            1.0/lam_t,
                            gpu_id=self.gpu_id,
                            stream=self.stream
                        )
                        for Du_bt in Du_bt_stack
                    ]

                # Step 2-3: wavelet, spatial domain

                if use_spatial_wavelet:
                    Wu_stack = []
                    w_stack = []
                    for stack_ind in range(self.nt):
                        u_t = u_stack[stack_ind]
                        bw_t = bw_stack[stack_ind]
                        Wu_t = W._matvec(u_t)
                        Wu_stack.append(Wu_t)
                        w_stack.append(
                            soft_thresholding(
                                Wu_t + bw_t,
                                1.0/gam,
                                gpu_id=self.gpu_id,
                                stream=self.stream
                            )
                        )

                # Step 2-4: wavelet along time axis

                if use_time_wavelet:
                    if gpu_holding:
                        h_u_stack = [
                            from_device_array(u_t, self.gpu_id, self.stream)
                            for u_t in u_stack
                        ]
                    else:
                        h_u_stack = u_stack

                    W_t_u_stack = W_t.forward(h_u_stack)
                    h_u_stack = None

                    wt_stack = [
                        soft_thresholding(
                            (W_t_u_t + bwt_t),
                            1.0/gam_t,
                            gpu_id=self.gpu_id,
                            stream=self.stream
                        )
                        for W_t_u_t, bwt_t in zip(W_t_u_stack, bwt_stack)
                    ]

            # Step 3: Bregman update
            # ----------------------

            if use_spatial_tv:
                bs_stack = []
                for stack_ind in range(self.nt):
                    Du_bs_t = Du_bs_stack[stack_ind]
                    ds_t = ds_stack[stack_ind]
                    with cupy.cuda.Device(self.gpu_id), self.stream as _:
                        bs_stack.append(
                            [Du_b_t - d_t
                             for Du_b_t, d_t in zip(Du_bs_t, ds_t)]
                        )

            if use_time_tv:
                bt_stack = [
                    (Du_bt - dt)
                    for Du_bt, dt in zip(Du_bt_stack, dt_stack)
                ]

            if use_spatial_wavelet:
                for i in range(len(bw_stack)):
                    bw_stack[i] += (Wu_stack[i] - w_stack[i])

            if use_time_wavelet:
                for i in range(len(bwt_stack)):
                    bwt_stack[i] += (W_t_u_stack[i] - wt_stack[i])

        if cupy.get_array_module(data_stack[0]) == np:
            return [from_device_array(u, self.gpu_id, self.stream)
                    for u in u_stack]
        else:
            return [to_device_array(u, self.gpu_id, self.stream)
                    for u in u_stack]
