# Owner(s): ["module: linear algebra"]
# ruff: noqa: F841

import torch
import numpy as np

import unittest
import itertools
import warnings
import math
from math import inf, nan, isnan
import re
import random
from random import randrange
from itertools import product
from functools import reduce, partial
from typing import Union, Optional
from torch._prims_common import DimsType
from packaging import version

from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    TEST_SCIPY,
    IS_MACOS,
    IS_WINDOWS,
    slowTest,
    TEST_WITH_ROCM,
    IS_FBCODE,
    IS_REMOTE_GPU,
    iter_indices,
    make_fullrank_matrices_with_distinct_singular_values,
    freeze_rng_state,
    IS_ARM64,
    IS_SANDCASTLE,
    TEST_OPT_EINSUM,
    parametrize,
    skipIfTorchDynamo,
    setBlasBackendsToDefaultFinally,
    setLinalgBackendsToDefaultFinally,
    serialTest,
    runOnRocmArch,
    MI300_ARCH,
    TEST_CUDA,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    dtypes,
    has_cusolver,
    has_hipsolver,
    onlyCPU,
    skipCUDAIf,
    skipCUDAIfNoMagma,
    skipCPUIfNoLapack,
    precisionOverride,
    skipCUDAIfNoMagmaAndNoCusolver,
    skipCUDAIfRocm,
    onlyNativeDeviceTypes,
    dtypesIfCUDA,
    onlyCUDA,
    skipMeta,
    skipCUDAIfNoCusolver,
    skipCUDAIfNotRocm,
    skipCUDAIfRocmVersionLessThan,
    dtypesIfMPS,
    largeTensorTest,
)
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
    all_types,
    all_types_and_complex_and,
    floating_and_complex_types,
    integral_types,
    floating_and_complex_types_and,
    floating_types_and,
    complex_types,
)
from torch.testing._internal.common_cuda import (
    SM53OrLater,
    SM80OrLater,
    SM90OrLater,
    tf32_on_and_off,
    _get_magma_version,
    _get_torch_cuda_version,
    CDNA2OrLater,
    TEST_MULTIGPU,
)
from torch.testing._internal.common_quantization import (
    _group_quantize_tensor,
    _dynamically_quantize_per_channel,
    _group_quantize_tensor_symmetric,
)
from torch.testing._internal.common_mkldnn import reduced_f32_on_and_off
from torch.distributions.binomial import Binomial
import torch.backends.opt_einsum as opt_einsum
import operator
import contextlib

# Protects against includes accidentally setting the default dtype
assert torch.get_default_dtype() is torch.float32

if TEST_SCIPY:
    import scipy


def blaslt_supported_device():
    if torch.cuda.is_available():
        if torch.version.hip:
            ROCM_VERSION = tuple(int(v) for v in torch.version.hip.split(".")[:2])
            archs = ["gfx90a", "gfx94"]
            if ROCM_VERSION >= (6, 3):
                archs.extend(["gfx110", "gfx120"])
            if ROCM_VERSION >= (6, 5):
                archs.append("gfx95")
            for arch in archs:
                if arch in torch.cuda.get_device_properties(0).gcnArchName:
                    return True
        else:
            return True
    return False


def tunableop_matmul(device, dtype, result_filename=None, offline=False):
    # Helper function to test TunableOp in a subprocess
    # requires helper function since lambda function
    # not supported by multiprocessing module
    import os

    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"

    if offline:
        torch.cuda.tunable.tuning_enable(False)
        torch.cuda.tunable.record_untuned_enable(True)
    else:
        if result_filename is not None:
            torch.cuda.tunable.set_filename(result_filename)

    torch.cuda.tunable.set_max_tuning_duration(1)
    A = torch.randn((17, 17), device=device, dtype=dtype)
    B = torch.randn((17, 17), device=device, dtype=dtype)
    C = torch.matmul(A, B)
    del os.environ["PYTORCH_TUNABLEOP_ENABLED"]


def get_tunableop_validators():
    assert len(torch.cuda.tunable.get_validators()) > 0
    validators = dict(torch.cuda.tunable.get_validators())
    return validators


def find_tunableop_result(results, OpSig, ParamSig):
    assert isinstance(results, tuple)
    for inner_tuple in results:
        if OpSig in inner_tuple and ParamSig in inner_tuple:
            return inner_tuple
    return None


def get_tunableop_untuned_filename():
    import os

    ordinal = torch.cuda.current_device()
    untuned_filename_env = os.getenv("PYTORCH_TUNABLEOP_UNTUNED_FILENAME")
    untuned_filename_base, _, _ = untuned_filename_env.rpartition(".")
    untuned_filename = f"{untuned_filename_base}{ordinal}.csv"
    return untuned_filename


class TestLinalg(TestCase):
    @contextlib.contextmanager
    def _hip_allow_tf32(self):
        # for HIP/AMDGPU, tf32 is behind a flag because the TF32 support is new
        # and only for MI300+. Environment variable will be removed in the future.
        import os

        hip_allow_tf32 = os.environ.get("HIPBLASLT_ALLOW_TF32", None)
        os.environ["HIPBLASLT_ALLOW_TF32"] = "1"

        try:
            yield
        finally:
            if hip_allow_tf32 is not None:
                os.environ["HIPBLASLT_ALLOW_TF32"] = hip_allow_tf32
            else:
                del os.environ["HIPBLASLT_ALLOW_TF32"]

    def setUp(self):
        super().setUp()
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        super().tearDown()

    @contextlib.contextmanager
    def _tunableop_ctx(self):
        # Initialize and then tear down TunableOp
        import glob
        import os

        self._set_tunableop_defaults()
        torch.cuda.tunable.enable(True)

        try:
            yield
        finally:
            # disables TunableOp
            torch.cuda.tunable.enable(False)

            # clean up, remove any files that were generated
            results_filename = torch.cuda.tunable.get_filename()
            results_filename_pattern, _, _ = results_filename.rpartition(".")
            untuned_filename = get_tunableop_untuned_filename()
            untuned_filename_pattern, _, _ = untuned_filename.rpartition(".")
            patterns = [
                f"{results_filename_pattern[:-1]}*.csv",
                f"{untuned_filename_pattern[:-1]}*.csv",
            ]
            files = [f for pattern in patterns for f in glob.glob(pattern)]
            for file in files:
                try:
                    os.remove(file)
                # NB: The file is locked on Windows
                except (FileNotFoundError, PermissionError):
                    pass

            # undo all the environment variables set
            # loop through a list of potentially used
            # environment variables.
            env_list = [
                "PYTORCH_TUNABLEOP_BLAS_LOG",
                "PYTORCH_TUNABLEOP_NUMERICAL_CHECK",
                "PYTORCH_TUNABLEOP_UNTUNED_FILENAME",
            ]
            for env in env_list:
                try:
                    del os.environ[env]
                except KeyError:
                    pass

    def _set_tunableop_defaults(self):
        if not torch.cuda.is_available():
            # TunableOp not supported on CPU at this time.
            return

        # disable TunableOp and restore to default values
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.record_untuned_enable(False)
        torch.cuda.tunable.tuning_enable(True)
        torch.cuda.tunable.set_max_tuning_duration(30)
        torch.cuda.tunable.set_max_tuning_iterations(100)
        torch.cuda.tunable.set_rotating_buffer_size(-1)
        ordinal = torch.cuda.current_device()

        # Set filenames to be unique on a per test basis
        import os

        unique_id = self.id().split(".")[-1]
        torch.cuda.tunable.set_filename(f"tunableop_results_{unique_id}_{ordinal}.csv")
        # ordinal gets automatically appended
        os.environ["PYTORCH_TUNABLEOP_UNTUNED_FILENAME"] = (
            f"tunableop_untuned_{unique_id}_.csv"
        )

    def _compare_untuned_tuned_entries(
        self, untuned_filename=None, tuned_filename=None
    ):
        # Compare the entries of untuned and tuned Tunableop results
        # file. Verify that for each Op+Param Signature in the untuned file
        # there is a matching one in the tuned results file.
        import csv

        ok = False
        ordinal = torch.cuda.current_device()
        if untuned_filename is None:
            untuned_filename = get_tunableop_untuned_filename()
        if tuned_filename is None:
            tuned_filename = torch.cuda.tunable.get_filename()

        with open(untuned_filename) as file1:
            with open(tuned_filename) as file2:
                untuned_reader = csv.reader(file1)
                untuned_csv_entries = {(row[0], row[1]) for row in untuned_reader}

                tuned_reader = csv.reader(file2)
                for _ in range(5):  # Skip the first 5 lines for the validator
                    next(tuned_reader, None)

                result_csv_entries = {(row[0], row[1]) for row in tuned_reader}

                missing = untuned_csv_entries - result_csv_entries

                if missing:
                    ok = False
                else:
                    ok = True

        return ok

    exact_dtype = True

    @dtypes(torch.float, torch.cfloat)
    @precisionOverride({torch.float: 1e-06, torch.cfloat: 1e-06})
    @tf32_on_and_off(5e-3)
    @reduced_f32_on_and_off(5e-3)
    def test_inner(self, device, dtype):
        def check(a_sizes_, b_sizes_):
            for a_sizes, b_sizes in ((a_sizes_, b_sizes_), (b_sizes_, a_sizes_)):
                a = torch.randn(a_sizes, dtype=dtype, device=device)
                b = torch.randn(b_sizes, dtype=dtype, device=device)
                res = torch.inner(a, b)
                ref = np.inner(a.cpu().numpy(), b.cpu().numpy())
                self.assertEqual(res.cpu(), torch.from_numpy(np.array(ref)))
                out = torch.zeros_like(res)
                torch.inner(a, b, out=out)
                self.assertEqual(res, out)

        check([], [])  # scalar x scalar
        check([], [0])  # scalar x empty
        check([], [3])  # scalar x 1D
        check([], [2, 3, 4])  # scalar x 3D

        check([0], [0])  # empty x empty
        check([0], [2, 0])  # empty x 2D

        check([2], [2])  # 1D x 1D
        check([2], [3, 1, 2])  # 1D x 3D
        check([2], [3, 0, 2])  # 1D x 3D empty

        check([1, 2], [3, 2])  # 2D x 2D
        check([1, 2], [3, 4, 2])  # 2D x 3D
        check([2, 1, 3, 2], [1, 3, 2, 2])  # 4D x 4D

        # Test error message
        with self.assertRaisesRegex(
            RuntimeError,
            r"inner\(\) the last dimension must match on both "
            r"input tensors but got shapes \[2, 3\] and \[2, 2\]",
        ):
            torch.randn(2, 3, device=device, dtype=dtype).inner(
                torch.randn(2, 2, device=device, dtype=dtype)
            )

    # Tests torch.outer, and its alias, torch.ger, vs. NumPy
    @precisionOverride({torch.bfloat16: 1e-1})
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_outer(self, device, dtype):
        def run_test_case(a, b):
            if dtype == torch.bfloat16:
                a_np = a.to(torch.double).cpu().numpy()
                b_np = b.to(torch.double).cpu().numpy()
                exact_dtype = False
            else:
                a_np = a.cpu().numpy()
                b_np = b.cpu().numpy()
                exact_dtype = True
            expected = np.outer(a_np, b_np)

            self.assertEqual(torch.outer(a, b), expected, exact_dtype=False)
            self.assertEqual(torch.Tensor.outer(a, b), expected, exact_dtype=False)

            self.assertEqual(torch.ger(a, b), expected, exact_dtype=False)
            self.assertEqual(torch.Tensor.ger(a, b), expected, exact_dtype=False)

            # test out variant
            out = torch.empty(a.size(0), b.size(0), device=device, dtype=dtype)
            torch.outer(a, b, out=out)
            self.assertEqual(out, expected, exact_dtype=False)

            out = torch.empty(a.size(0), b.size(0), device=device, dtype=dtype)
            torch.ger(a, b, out=out)
            self.assertEqual(out, expected, exact_dtype=False)

        a = torch.randn(50).to(device=device, dtype=dtype)
        b = torch.randn(50).to(device=device, dtype=dtype)
        run_test_case(a, b)

        # test 0 strided tensor
        zero_strided = torch.randn(1).to(device=device, dtype=dtype).expand(50)
        run_test_case(zero_strided, b)
        run_test_case(a, zero_strided)

    def test_matrix_rank_removed_error(self, device):
        a = make_tensor(5, 5, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(
            RuntimeError,
            "This function was deprecated since version 1.9 and is now removed",
        ):
            torch.matrix_rank(a)

    def test_solve_removed_error(self, device):
        a = make_tensor(5, 5, device=device, dtype=torch.float32)
        b = make_tensor(5, 1, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(
            RuntimeError,
            "This function was deprecated since version 1.9 and is now removed",
        ):
            torch.solve(b, a)
        with self.assertRaisesRegex(
            RuntimeError,
            "This function was deprecated since version 1.9 and is now removed",
        ):
            b.solve(a)

    def test_eig_removed_error(self, device):
        a = make_tensor(5, 5, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(
            RuntimeError,
            "This function was deprecated since version 1.9 and is now removed",
        ):
            torch.eig(a)
        with self.assertRaisesRegex(
            RuntimeError,
            "This function was deprecated since version 1.9 and is now removed",
        ):
            a.eig()

    def test_symeig_removed_error(self, device):
        a = make_tensor(5, 5, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(
            RuntimeError,
            "This function was deprecated since version 1.9 and is now removed",
        ):
            torch.symeig(a)
        with self.assertRaisesRegex(
            RuntimeError,
            "This function was deprecated since version 1.9 and is now removed",
        ):
            a.symeig()

    def test_lstsq_removed_error(self, device):
        a = make_tensor(5, 5, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(
            RuntimeError,
            "This function was deprecated since version 1.9 and is now removed",
        ):
            torch.lstsq(a, a)
        with self.assertRaisesRegex(
            RuntimeError,
            "This function was deprecated since version 1.9 and is now removed",
        ):
            a.lstsq(a)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @skipIfTorchDynamo("flaky, needs investigation")
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_linalg_lstsq(self, device, dtype):
        from torch.testing._internal.common_utils import random_well_conditioned_matrix

        if self.device_type == "cpu":
            drivers = ("gels", "gelsy", "gelsd", "gelss", None)
        else:
            drivers = ("gels", None)

        def check_solution_correctness(a, b, sol):
            sol2 = a.pinverse() @ b
            self.assertEqual(sol, sol2, atol=1e-5, rtol=1e-5)

        def check_correctness_ref(a, b, res, ref, driver="default"):
            def apply_if_not_empty(t, f):
                if t.numel():
                    return f(t)
                else:
                    return t

            def select_if_not_empty(t, i):
                selected = apply_if_not_empty(t, lambda x: x.select(0, i))
                return selected

            m = a.size(-2)
            n = a.size(-1)
            nrhs = b.size(-1)
            batch_size = int(np.prod(a.shape[:-2]))
            if batch_size == 0:
                batch_size = 1
            a_3d = a.view(batch_size, m, n)
            b_3d = b.view(batch_size, m, nrhs)

            solution_3d = res.solution.view(batch_size, n, nrhs)
            residuals_2d = apply_if_not_empty(res.residuals, lambda t: t.view(-1, nrhs))
            rank_1d = apply_if_not_empty(res.rank, lambda t: t.view(-1))
            singular_values_2d = res.singular_values.view(
                batch_size, res.singular_values.shape[-1]
            )

            if a.numel() > 0:
                for i in range(batch_size):
                    sol, residuals, rank, singular_values = ref(
                        a_3d.select(0, i).numpy(), b_3d.select(0, i).numpy()
                    )
                    # Singular values are None when lapack_driver='gelsy' in SciPy
                    if singular_values is None:
                        singular_values = []
                    self.assertEqual(
                        sol, solution_3d.select(0, i), atol=1e-5, rtol=1e-5
                    )
                    self.assertEqual(
                        rank, select_if_not_empty(rank_1d, i), atol=1e-5, rtol=1e-5
                    )
                    self.assertEqual(
                        singular_values,
                        singular_values_2d.select(0, i),
                        atol=1e-5,
                        rtol=1e-5,
                    )

                    # SciPy and NumPy operate only on non-batched input and
                    # return an empty array with shape (0,) if rank(a) != n
                    # in PyTorch the batched inputs are supported and
                    # matrices in the batched input can have different ranks
                    # we compute residuals only if all matrices have rank == n
                    # see https://github.com/pytorch/pytorch/issues/56483
                    if m > n:
                        if torch.all(rank_1d == n):
                            self.assertEqual(
                                residuals,
                                select_if_not_empty(residuals_2d, i),
                                atol=1e-5,
                                rtol=1e-5,
                                exact_dtype=False,
                            )
                        else:
                            self.assertTrue(residuals_2d.numel() == 0)

            else:
                self.assertEqual(res.solution.shape, (*a.shape[:-2], n, nrhs))
                self.assertEqual(res.rank.shape, a.shape[:-2])

                # residuals are not always computed (and have non-zero shape)
                if m > n and driver != "gelsy":
                    self.assertEqual(res.residuals.shape, (*a.shape[:-2], 0))
                else:
                    self.assertEqual(res.residuals.shape, (0,))

                # singular_values are not always computed (and have non-zero shape)
                if driver == "default" or driver == "gelsd" or driver == "gelss":
                    self.assertEqual(
                        res.singular_values.shape, (*a.shape[:-2], min(m, n))
                    )
                else:
                    self.assertEqual(res.singular_values.shape, (0,))

        def check_correctness_scipy(a, b, res, driver, cond):
            # SciPy provides 3 driver options: gelsd, gelss, gelsy
            if TEST_SCIPY and driver in ("gelsd", "gelss", "gelsy"):
                import scipy.linalg

                def scipy_ref(a, b):
                    return scipy.linalg.lstsq(a, b, lapack_driver=driver, cond=cond)

                check_correctness_ref(a, b, res, scipy_ref, driver=driver)

        def check_correctness_numpy(a, b, res, driver, rcond):
            # NumPy uses only gelsd routine
            if driver == "gelsd":

                def numpy_ref(a, b):
                    return np.linalg.lstsq(a, b, rcond=rcond)

                check_correctness_ref(a, b, res, numpy_ref)

        ms = [2**i for i in range(5)]
        m_ge_n_sizes = [(m, m // 2) for m in ms] + [(m, m) for m in ms]
        # cases m < n are only supported on CPU and for cuSOLVER path on CUDA
        m_l_n_sizes = [(m // 2, m) for m in ms]
        include_m_l_n_case = has_cusolver() or device == "cpu"
        matrix_sizes = m_ge_n_sizes + (m_l_n_sizes if include_m_l_n_case else [])
        batches = [(), (2,), (2, 2), (2, 2, 2)]
        # we generate matrices with singular values sampled from a normal distribution,
        # that is why we use `cond=1.0`, the mean to cut roughly half of all
        # the singular values and compare whether torch.linalg.lstsq agrees with
        # SciPy and NumPy.
        # if rcond is True then set value for it based on the used algorithm
        # rcond == -1 or any other negative value forces LAPACK to use machine precision tolerance
        rconds = (None, True, -1)

        for batch, matrix_size, driver, rcond in itertools.product(
            batches, matrix_sizes, drivers, rconds
        ):
            # keep the rcond value if it is None or -1, set the driver specific value if it is True
            if rcond and rcond != -1:
                if driver in ("gelss", "gelsd"):
                    # SVD based algorithm; set to zero roughly half of all the singular values
                    rcond = 1.0
                else:
                    # driver == 'gelsy'
                    # QR based algorithm; setting the value too high might lead to non-unique solutions and flaky tests
                    # so we skip this case
                    continue

            # specifying rcond value has no effect for gels driver so no need to run the tests again
            if driver == "gels" and rcond is not None:
                continue

            shape = batch + matrix_size
            a = random_well_conditioned_matrix(*shape, dtype=dtype, device=device)
            b = torch.rand(*shape, dtype=dtype, device=device)

            m = a.size(-2)
            n = a.size(-1)
            res = torch.linalg.lstsq(a, b, rcond=rcond, driver=driver)
            sol = res.solution

            # Only checks gelsd, gelss, gelsy drivers
            check_correctness_scipy(a, b, res, driver, rcond)

            # Only checks gelsd driver
            check_correctness_numpy(a, b, res, driver, rcond)

            # gels driver is not checked by comparing to NumPy or SciPy implementation
            # because NumPy and SciPy do not implement this driver
            if driver == "gels" and rcond is None:
                check_solution_correctness(a, b, sol)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_linalg_lstsq_batch_broadcasting(self, device, dtype):
        from torch.testing._internal.common_utils import random_well_conditioned_matrix

        def check_correctness(a, b):
            sol = torch.linalg.lstsq(a, b).solution
            sol2 = a.pinverse() @ b
            self.assertEqual(sol, sol2, rtol=1e-5, atol=1e-5)

        ms = [2**i for i in range(5)]
        batches = [(), (0,), (2,), (2, 2), (2, 2, 2)]
        # the case when a single matrix is batch-broadcasted over the rhs
        for m, batch in itertools.product(ms, batches):
            a = random_well_conditioned_matrix(m, m, dtype=dtype, device=device).view(
                *([1] * len(batch)), m, m
            )
            b = torch.rand(*(batch + (m, m)), dtype=dtype, device=device)
            check_correctness(a, b)

        # cases with broadcastable shapes
        for m in ms:
            a = random_well_conditioned_matrix(
                1, 3, 1, 3, m, m, dtype=dtype, device=device
            )
            b = torch.rand(3, 1, 3, 1, m, m // 2, dtype=dtype, device=device)
            check_correctness(a, b)

            # rhs are vectors, not matrices in this test
            b = torch.rand(3, 1, 3, 1, m, dtype=dtype, device=device)
            # unsqueeze for b because `check_correctness` checks against
            # a.pinverse() @ b, which requires b to be a matrix
            check_correctness(a, b.unsqueeze(-1))

            a = random_well_conditioned_matrix(
                3, 1, 3, 1, m, m, dtype=dtype, device=device
            )
            b = torch.rand(1, 3, 1, 3, m, m // 2, dtype=dtype, device=device)
            check_correctness(a, b)

            # rhs are vectors, not matrices in this test
            b = torch.rand(1, 3, 1, 3, m, dtype=dtype, device=device)
            check_correctness(a, b.unsqueeze(-1))

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_linalg_lstsq_input_checks(self, device, dtype):
        # check empty inputs
        # empty batches
        a = torch.rand(0, 0, 3, 3, dtype=dtype, device=device)
        b = torch.rand(0, 0, 3, 2, dtype=dtype, device=device)
        self.assertEqual(
            torch.linalg.lstsq(a, b)[0],
            torch.zeros(0, 0, 3, 2, dtype=dtype, device=device),
        )
        # empty a and b
        a = torch.rand(2, 2, 0, 0, dtype=dtype, device=device)
        b = torch.rand(2, 2, 0, 0, dtype=dtype, device=device)
        self.assertEqual(
            torch.linalg.lstsq(a, b)[0],
            torch.zeros(2, 2, 0, 0, dtype=dtype, device=device),
        )
        # empty a and b
        a = torch.rand(2, 2, 3, 0, dtype=dtype, device=device)
        b = torch.rand(2, 2, 3, 0, dtype=dtype, device=device)
        self.assertEqual(
            torch.linalg.lstsq(a, b)[0],
            torch.zeros(2, 2, 0, 0, dtype=dtype, device=device),
        )
        # empty a but not b
        a = torch.rand(2, 2, 3, 0, dtype=dtype, device=device)
        b = torch.rand(2, 2, 3, 2, dtype=dtype, device=device)
        self.assertEqual(
            torch.linalg.lstsq(a, b)[0],
            torch.zeros(2, 2, 0, 2, dtype=dtype, device=device),
        )

        # empty a and b
        if torch.device(device).type == "cpu":
            # only CPU since CUDA does not support overdetermined systems
            a = torch.rand(2, 2, 0, 3, dtype=dtype, device=device)
            b = torch.rand(2, 2, 0, 3, dtype=dtype, device=device)
            self.assertEqual(
                torch.linalg.lstsq(a, b)[0],
                torch.zeros(2, 2, 3, 3, dtype=dtype, device=device),
            )

        a = torch.rand(2, 3, dtype=dtype, device=device)
        b = torch.rand(3, dtype=dtype, device=device)

        with self.assertRaisesRegex(
            RuntimeError, "input must have at least 2 dimensions"
        ):
            torch.linalg.lstsq(b, b)

        with self.assertRaisesRegex(
            RuntimeError, "other must have at least 1 dimension"
        ):
            torch.linalg.lstsq(a, torch.tensor(1, dtype=dtype, device=device))

        with self.assertRaisesRegex(
            RuntimeError, r"input.size\(-2\) should match other.size\(-1\)"
        ):
            torch.linalg.lstsq(a, b)

        with self.assertRaisesRegex(
            RuntimeError, r"input.size\(-2\) should match other.size\(-2\)"
        ):
            torch.linalg.lstsq(a, b.unsqueeze(-1))

        a = torch.randn(1, 1, 1, dtype=dtype, device=device)
        b = torch.randn(3, 1, dtype=dtype, device=device)

        with self.assertRaisesRegex(
            RuntimeError, r"input.size\(-2\) should match other.size\(-2\)"
        ):
            torch.linalg.lstsq(a, b)

        def complement_device(device):
            if device == "cpu" and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"

        a = torch.rand(2, 2, 2, 2, dtype=dtype, device=device)
        b = torch.rand(2, 2, 2, dtype=dtype, device=complement_device(device))
        if a.device != b.device:
            with self.assertRaisesRegex(RuntimeError, "be on the same device"):
                torch.linalg.lstsq(a, b)

        b = (torch.rand(2, 2, 2, dtype=dtype, device=device) * 100).long()
        with self.assertRaisesRegex(RuntimeError, "the same dtype"):
            torch.linalg.lstsq(a, b)

        a = torch.rand(2, 2, 2, 2, dtype=dtype, device=device)
        b = torch.rand(2, 2, 2, dtype=dtype, device=device)

        if device != "cpu":
            with self.assertRaisesRegex(
                RuntimeError, "`driver` other than `gels` is not supported on CUDA"
            ):
                torch.linalg.lstsq(a, b, driver="fictitious_driver")
        # if on cpu
        else:
            with self.assertRaisesRegex(
                RuntimeError,
                r"parameter `driver` should be one of \(gels, gelsy, gelsd, gelss\)",
            ):
                torch.linalg.lstsq(a, b, driver="fictitious_driver")

        # cuSOLVER path supports underdetermined systems
        version = torch.testing._internal.common_cuda._get_torch_cuda_version()
        cusolver_not_available = version < (10, 1)

        if device != "cpu" and cusolver_not_available:
            a = torch.rand(2, 3, dtype=dtype, device=device)
            b = torch.rand(2, 1, dtype=dtype, device=device)
            with self.assertRaisesRegex(RuntimeError, r"only overdetermined systems"):
                torch.linalg.lstsq(a, b)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_cholesky(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def run_test(shape, batch, contiguous):
            A = random_hermitian_pd_matrix(shape, *batch, dtype=dtype, device=device)
            if A.numel() > 0 and not contiguous:
                A = A.mT
                self.assertFalse(A.is_contiguous())
            expected_L = np.linalg.cholesky(A.cpu().numpy())
            actual_L = torch.linalg.cholesky(A)

            # For fp32 individual entries in matrices can differ between PyTorch and NumPy
            # Let's compare the norms of matrices instead
            if A.numel() > 0 and dtype in [torch.float32, torch.complex64]:
                # axis is specified to calculate matrix norm for batched input
                expected_norm = np.linalg.norm(expected_L, ord=1, axis=(-2, -1))
                actual_norm = torch.linalg.norm(actual_L, ord=1, axis=(-2, -1))
                # Compare the norms with standard tolerances
                self.assertEqual(actual_norm, expected_norm)
                # and individual values with a higher tolerance
                self.assertEqual(actual_L, expected_L, atol=1e-2, rtol=1e-5)
            else:
                self.assertEqual(actual_L, expected_L)

        shapes = (0, 3, 5)
        batches = ((), (3,), (2, 2))
        larger_input_case = [(100, (5,), True)]
        for shape, batch, contiguous in (
            list(itertools.product(shapes, batches, (True, False))) + larger_input_case
        ):
            run_test(shape, batch, contiguous)

        # check the out= variant
        A = random_hermitian_pd_matrix(3, 3, dtype=dtype, device=device)
        out = torch.empty_like(A)
        ans = torch.linalg.cholesky(A, out=out)
        self.assertEqual(ans, out)
        expected = torch.linalg.cholesky(A)
        self.assertEqual(expected, out)

        # check the upper= variant
        expected = torch.linalg.cholesky(A).mH
        actual = torch.linalg.cholesky(A, upper=True)
        self.assertEqual(expected, actual)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_cholesky_errors_and_warnings(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        # cholesky requires the input to be a square matrix or batch of square matrices
        A = torch.randn(2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(
            RuntimeError, r"must be batches of square matrices"
        ):
            torch.linalg.cholesky(A)
        A = torch.randn(2, 2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(
            RuntimeError, r"must be batches of square matrices"
        ):
            torch.linalg.cholesky(A)
        with self.assertRaisesRegex(
            np.linalg.LinAlgError, r"Last 2 dimensions of the array must be square"
        ):
            np.linalg.cholesky(A.cpu().numpy())

        # cholesky requires the input to be at least 2 dimensional tensor
        A = torch.randn(2, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, r"must have at least 2 dimensions"):
            torch.linalg.cholesky(A)
        with self.assertRaisesRegex(
            np.linalg.LinAlgError,
            r"1-dimensional array given\. Array must be at least two-dimensional",
        ):
            np.linalg.cholesky(A.cpu().numpy())

        # if the input matrix is not positive definite, an error should be raised
        A = torch.eye(3, 3, dtype=dtype, device=device)
        A[-1, -1] = 0  # Now A is not positive definite
        with self.assertRaisesRegex(
            torch.linalg.LinAlgError, r"minor of order 3 is not positive-definite"
        ):
            torch.linalg.cholesky(A)
        with self.assertRaisesRegex(
            np.linalg.LinAlgError, r"Matrix is not positive definite"
        ):
            np.linalg.cholesky(A.cpu().numpy())

        # if at least one matrix in the batch is singular, an error should be raised
        A = torch.eye(3, 3, dtype=dtype, device=device)
        A = A.reshape((1, 3, 3))
        A = A.repeat(5, 1, 1)
        A[4, -1, -1] = 0  # Now A[4] is not positive definite
        with self.assertRaisesRegex(
            torch.linalg.LinAlgError,
            r"\(Batch element 4\): The factorization could not be completed",
        ):
            torch.linalg.cholesky(A)

        # if out tensor with wrong shape is passed a warning is given
        A = random_hermitian_pd_matrix(3, dtype=dtype, device=device)
        out = torch.empty(2, 3, dtype=dtype, device=device)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.cholesky(A, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue(
                "An output with one or more elements was resized" in str(w[-1].message)
            )

        # dtypes should be safely castable
        out = torch.empty(*A.shape, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got int instead"):
            torch.linalg.cholesky(A, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = "cpu" if self.device_type != "cpu" else "cuda"
            out = torch.empty(0, device=wrong_device, dtype=dtype)
            with self.assertRaisesRegex(
                RuntimeError, "Expected all tensors to be on the same device"
            ):
                torch.linalg.cholesky(A, out=out)

    # NOTE: old_cholesky* tests were moved here from test_torch.py and test_autograd.py
    @slowTest
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_old_cholesky_batched_many_batches(self, device, dtype):
        from torch.testing._internal.common_utils import random_symmetric_pd_matrix

        def cholesky_test_helper(n, batchsize, device, upper):
            A = random_symmetric_pd_matrix(n, batchsize, dtype=dtype, device=device)
            chol_fact = torch.cholesky(A, upper=upper)
            if upper:
                # Correctness check
                self.assertEqual(A, chol_fact.mT.matmul(chol_fact))
                # Upper triangular check
                self.assertEqual(chol_fact, chol_fact.triu())
            else:
                # Correctness check
                self.assertEqual(A, chol_fact.matmul(chol_fact.mT))
                # Lower triangular check
                self.assertEqual(chol_fact, chol_fact.tril())

        for upper, batchsize in itertools.product([True, False], [262144, 524288]):
            cholesky_test_helper(2, batchsize, device, upper)

    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_old_cholesky_batched(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def cholesky_test_helper(n, batch_dims, upper):
            A = random_hermitian_pd_matrix(n, *batch_dims, dtype=dtype, device=device)
            cholesky_exp = torch.stack(
                [m.cholesky(upper=upper) for m in A.reshape(-1, n, n)]
            )
            cholesky_exp = cholesky_exp.reshape_as(A)
            self.assertEqual(cholesky_exp, torch.cholesky(A, upper=upper))

        for upper, batchsize in itertools.product(
            [True, False], [(3,), (3, 4), (2, 3, 4)]
        ):
            cholesky_test_helper(3, batchsize, upper)

    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @tf32_on_and_off(0.1 if TEST_WITH_ROCM else 0.01)
    @reduced_f32_on_and_off(0.01)
    def test_old_cholesky(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        A = random_hermitian_pd_matrix(10, dtype=dtype, device=device)

        # default Case
        C = torch.cholesky(A)
        B = torch.mm(C, C.t().conj())
        self.assertEqual(A, B, atol=1e-14, rtol=0)

        # test Upper Triangular
        U = torch.cholesky(A, True)
        B = torch.mm(U.t().conj(), U)
        self.assertEqual(
            A,
            B,
            atol=1e-14,
            rtol=0,
            msg="cholesky (upper) did not allow rebuilding the original matrix",
        )

        # test Lower Triangular
        L = torch.cholesky(A, False)
        B = torch.mm(L, L.t().conj())
        self.assertEqual(
            A,
            B,
            atol=1e-14,
            rtol=0,
            msg="cholesky (lower) did not allow rebuilding the original matrix",
        )

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_old_cholesky_empty(self, device, dtype):
        def run_test(upper):
            A = torch.empty(0, 0, dtype=dtype, device=device)
            chol = torch.cholesky(A, upper)
            chol_A = torch.matmul(chol, chol.t().conj())
            self.assertEqual(A, chol_A)

        for upper in [True, False]:
            run_test(upper)

    # Test for issue
    # https://github.com/pytorch/pytorch/issues/57032
    # torch.cholesky with upper=True for batched CUDA inputs was wrong
    # it was using the lower triangular part instead of the upper one
    @onlyCUDA
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    def test_old_cholesky_batched_upper(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        batchsize = 2
        A = random_hermitian_pd_matrix(3, batchsize, dtype=dtype, device=device)
        A_triu = A.triu()  # fill the lower triangular part with zero

        U = torch.cholesky(A_triu, upper=True)

        reconstruct_A = U.mH @ U
        self.assertEqual(A, reconstruct_A)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_cholesky_ex(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def run_test(n, batch):
            A = random_hermitian_pd_matrix(n, *batch, dtype=dtype, device=device)
            expected_L = np.linalg.cholesky(A.cpu().numpy())
            expected_info = torch.zeros(A.shape[:-2], dtype=torch.int32, device=device)
            actual_L, actual_info = torch.linalg.cholesky_ex(A)

            # For fp32 individual entries in matrices can differ between PyTorch and NumPy
            # Let's compare the norms of matrices instead
            if A.numel() > 0 and dtype in [torch.float32, torch.complex64]:
                # axis is specified to calculate matrix norm for batched input
                expected_norm = np.linalg.norm(expected_L, ord=1, axis=(-2, -1))
                actual_norm = torch.linalg.norm(actual_L, ord=1, axis=(-2, -1))
                # Compare the norms with standard tolerances
                self.assertEqual(actual_norm, expected_norm)
                # and individual values with a higher tolerance
                self.assertEqual(actual_L, expected_L, atol=1e-2, rtol=1e-5)
            else:
                self.assertEqual(actual_L, expected_L)
            self.assertEqual(actual_info, expected_info)

        ns = (0, 3, 5)
        batches = ((), (2,), (2, 1))
        for n, batch in itertools.product(ns, batches):
            run_test(n, batch)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_cholesky_ex_non_pd(self, device, dtype):
        # if the input matrix is not positive definite, info with positive integer is returned
        A = torch.eye(3, 3, dtype=dtype, device=device)
        A[-1, -1] = 0  # Now A is singular
        _, info = torch.linalg.cholesky_ex(A)
        self.assertEqual(info, 3)
        with self.assertRaisesRegex(
            torch.linalg.LinAlgError, r"minor of order 3 is not positive-definite"
        ):
            torch.linalg.cholesky_ex(A, check_errors=True)

        # if at least one matrix in the batch is not positive definite,
        # batched info with positive integer for the corresponding matrix is returned
        A = torch.eye(3, 3, dtype=dtype, device=device)
        A = A.reshape((1, 3, 3))
        A = A.repeat(5, 1, 1)
        A[3, -2, -2] = 0  # Now A[3] is singular
        _, info = torch.linalg.cholesky_ex(A)

        expected_info = torch.zeros(A.shape[:-2], dtype=torch.int32, device=device)
        expected_info[3] = 2
        self.assertEqual(info, expected_info)
        with self.assertRaisesRegex(
            torch.linalg.LinAlgError,
            r"\(Batch element 3\): The factorization could not be completed",
        ):
            torch.linalg.cholesky_ex(A, check_errors=True)

    def _test_addr_vs_numpy(self, device, dtype, beta=1, alpha=1):
        def check(m, a, b, beta, alpha):
            if dtype == torch.bfloat16:
                a_np = a.to(torch.double).cpu().numpy()
                b_np = b.to(torch.double).cpu().numpy()
                m_np = m.to(torch.double).cpu().numpy()
                exact_dtype = False
            else:
                a_np = a.cpu().numpy()
                b_np = b.cpu().numpy()
                m_np = m.cpu().numpy()
                exact_dtype = True
            if beta == 0:
                expected = alpha * np.outer(a_np, b_np)
            else:
                expected = beta * m_np + alpha * np.outer(a_np, b_np)

            res = torch.addr(m, a, b, beta=beta, alpha=alpha)
            self.assertEqual(res, expected, exact_dtype=exact_dtype)

            # Test out variant
            out = torch.empty_like(res)
            torch.addr(m, a, b, beta=beta, alpha=alpha, out=out)
            self.assertEqual(out, expected, exact_dtype=exact_dtype)

        m = make_tensor((50, 50), device=device, dtype=dtype, low=-2, high=2)
        a = make_tensor((50,), device=device, dtype=dtype, low=-2, high=2)
        b = make_tensor((50,), device=device, dtype=dtype, low=-2, high=2)

        check(m, a, b, beta, alpha)

        # test transpose
        m_transpose = torch.transpose(m, 0, 1)
        check(m_transpose, a, b, beta, alpha)

        # test 0 strided tensor
        zero_strided = make_tensor(
            (1,), device=device, dtype=dtype, low=-2, high=2
        ).expand(50)
        check(m, zero_strided, b, beta, alpha)

        # test scalar
        m_scalar = torch.tensor(1, device=device, dtype=dtype)
        check(m_scalar, a, b, beta, alpha)

        # test nans and infs are not propagated to the output when beta == 0
        float_and_complex_dtypes = floating_and_complex_types_and(
            torch.half, torch.bfloat16
        )
        if beta == 0 and dtype in float_and_complex_dtypes:
            m[0][10] = m[10][10] = m[20][20] = float("inf")
            m[1][10] = m[11][10] = m[21][20] = float("nan")
        check(m, a, b, 0, alpha)

    @dtypes(torch.bool)
    def test_addr_bool(self, device, dtype):
        self._test_addr_vs_numpy(device, dtype, beta=True, alpha=False)
        self._test_addr_vs_numpy(device, dtype, beta=False, alpha=True)
        self._test_addr_vs_numpy(device, dtype, beta=False, alpha=False)
        self._test_addr_vs_numpy(device, dtype, beta=True, alpha=True)

    @dtypes(*integral_types())
    def test_addr_integral(self, device, dtype):
        with self.assertRaisesRegex(
            RuntimeError, "argument beta must not be a floating point number."
        ):
            self._test_addr_vs_numpy(device, dtype, beta=2.0, alpha=1)
        with self.assertRaisesRegex(
            RuntimeError, "argument alpha must not be a floating point number."
        ):
            self._test_addr_vs_numpy(device, dtype, beta=2, alpha=1.0)
        with self.assertRaisesRegex(
            RuntimeError, "Boolean beta only supported for Boolean results."
        ):
            self._test_addr_vs_numpy(device, dtype, beta=True, alpha=1)
        with self.assertRaisesRegex(
            RuntimeError, "Boolean alpha only supported for Boolean results."
        ):
            self._test_addr_vs_numpy(device, dtype, beta=2, alpha=True)

        # when beta is zero
        self._test_addr_vs_numpy(device, dtype, beta=0, alpha=2)
        # when beta is not zero
        self._test_addr_vs_numpy(device, dtype, beta=2, alpha=2)

    @precisionOverride({torch.bfloat16: 1e-1})
    @dtypes(*floating_and_complex_types_and(torch.half, torch.bfloat16))
    def test_addr_float_and_complex(self, device, dtype):
        with self.assertRaisesRegex(
            RuntimeError, "Boolean beta only supported for Boolean results."
        ):
            self._test_addr_vs_numpy(device, dtype, beta=True, alpha=1)
        with self.assertRaisesRegex(
            RuntimeError, "Boolean alpha only supported for Boolean results."
        ):
            self._test_addr_vs_numpy(device, dtype, beta=2, alpha=True)

        # when beta is zero
        self._test_addr_vs_numpy(device, dtype, beta=0.0, alpha=2)
        # when beta is not zero
        self._test_addr_vs_numpy(device, dtype, beta=0.5, alpha=2)
        if dtype in complex_types():
            self._test_addr_vs_numpy(device, dtype, beta=(0 + 0.1j), alpha=(0.2 - 0.2j))

    @dtypes(
        *itertools.product(
            all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
            all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
        )
    )
    def test_outer_type_promotion(self, device, dtypes):
        a = torch.randn(5).to(device=device, dtype=dtypes[0])
        b = torch.randn(5).to(device=device, dtype=dtypes[1])
        for op in (torch.outer, torch.Tensor.outer, torch.ger, torch.Tensor.ger):
            result = op(a, b)
            self.assertEqual(result.dtype, torch.result_type(a, b))

    # don't use @dtypes decorator to avoid generating ~1700 tests per device
    def test_addr_type_promotion(self, device):
        for dtypes0, dtypes1, dtypes2 in product(
            all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool), repeat=3
        ):
            a = make_tensor((5,), device=device, dtype=dtypes0, low=-2, high=2)
            b = make_tensor((5,), device=device, dtype=dtypes1, low=-2, high=2)
            m = make_tensor((5, 5), device=device, dtype=dtypes2, low=-2, high=2)

            desired_dtype = torch.promote_types(
                torch.promote_types(dtypes0, dtypes1), dtypes2
            )
            for op in (torch.addr, torch.Tensor.addr):
                result = op(m, a, b)
                self.assertEqual(result.dtype, desired_dtype)

    # Tests migrated from test_torch.py
    # 1) test the shape of the result tensor when there is empty input tensor
    # 2) test the Runtime Exception when there is scalar input tensor
    def test_outer_ger_addr_legacy_tests(self, device):
        for size in ((0, 0), (0, 5), (5, 0)):
            a = torch.rand(size[0], device=device)
            b = torch.rand(size[1], device=device)

            self.assertEqual(torch.outer(a, b).shape, size)
            self.assertEqual(torch.ger(a, b).shape, size)

            m = torch.empty(size, device=device)
            self.assertEqual(torch.addr(m, a, b).shape, size)

        m = torch.randn(5, 6, device=device)
        a = torch.randn(5, device=device)
        b = torch.tensor(6, device=device)
        self.assertRaises(RuntimeError, lambda: torch.outer(a, b))
        self.assertRaises(RuntimeError, lambda: torch.outer(b, a))
        self.assertRaises(RuntimeError, lambda: torch.ger(a, b))
        self.assertRaises(RuntimeError, lambda: torch.ger(b, a))
        self.assertRaises(RuntimeError, lambda: torch.addr(m, a, b))
        self.assertRaises(RuntimeError, lambda: torch.addr(m, b, a))

    # Tests torch.det and its alias, torch.linalg.det, vs. NumPy
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double, torch.cdouble)
    def test_det(self, device, dtype):
        tensors = (
            torch.randn((2, 2), device=device, dtype=dtype),
            torch.randn((129, 129), device=device, dtype=dtype),
            torch.randn((3, 52, 52), device=device, dtype=dtype),
            torch.randn((4, 2, 26, 26), device=device, dtype=dtype),
        )

        ops = (torch.det, torch.Tensor.det, torch.linalg.det)
        for t in tensors:
            expected = np.linalg.det(t.cpu().numpy())
            for op in ops:
                actual = op(t)
                self.assertEqual(actual, expected)
                self.compare_with_numpy(op, np.linalg.det, t)

        # NOTE: det requires a 2D+ tensor
        t = torch.randn(1, device=device, dtype=dtype)
        with self.assertRaises(RuntimeError):
            op(t)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    def test_eigh(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_matrix

        def run_test(shape, batch, uplo):
            matrix = random_hermitian_matrix(shape, *batch, dtype=dtype, device=device)
            expected_w, expected_v = np.linalg.eigh(matrix.cpu().numpy(), UPLO=uplo)
            actual_w, actual_v = torch.linalg.eigh(matrix, UPLO=uplo)
            self.assertEqual(actual_w, expected_w)
            # sign of eigenvectors is not unique and therefore absolute values are compared
            self.assertEqual(abs(actual_v), abs(expected_v))
            # additionally we can multiply the eigenvector with a phase factor e^{i\phi} and then compare the values
            # let's choose the convention that the first element of the eigenvectors from torch and numpy be the same
            # for real inputs, this phase factor is plus or minus one
            if matrix.numel() > 0:
                phase = (
                    torch.from_numpy(expected_v[..., 0, :])
                    .to(device=device)
                    .div(actual_v[..., 0, :])
                )
                actual_v_rotated = actual_v * phase.unsqueeze(-2).expand_as(actual_v)
                self.assertEqual(actual_v_rotated, expected_v)

            # check the out= variant
            out_w = torch.empty_like(actual_w)
            out_v = torch.empty_like(actual_v)
            ans_w, ans_v = torch.linalg.eigh(matrix, UPLO=uplo, out=(out_w, out_v))
            self.assertEqual(ans_w, out_w)
            self.assertEqual(ans_v, out_v)
            self.assertEqual(ans_w, actual_w)
            self.assertEqual(abs(ans_v), abs(actual_v))

        shapes = (0, 3, 5)
        batches = ((), (3,), (2, 2))
        uplos = ["U", "L"]
        for shape, batch, uplo in itertools.product(shapes, batches, uplos):
            run_test(shape, batch, uplo)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    def test_eigh_lower_uplo(self, device, dtype):
        def run_test(shape, batch, uplo):
            # check lower case uplo
            # use non-symmetric input to check whether uplo argument is working as intended
            matrix = torch.randn(shape, shape, *batch, dtype=dtype, device=device)
            expected_w, expected_v = np.linalg.eigh(matrix.cpu().numpy(), UPLO=uplo)
            actual_w, actual_v = torch.linalg.eigh(matrix, UPLO=uplo)
            self.assertEqual(actual_w, expected_w)
            self.assertEqual(abs(actual_v), abs(expected_v))

        uplos = ["u", "l"]
        for uplo in uplos:
            run_test(3, (2, 2), uplo)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_eigh_errors_and_warnings(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_matrix

        # eigh requires a square matrix
        t = torch.randn(2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.linalg.eigh(t)

        # eigh requires 'uplo' parameter to be 'U' or 'L'
        t = torch.randn(3, 3, device=device, dtype=dtype)
        for uplo in ["a", "wrong"]:
            with self.assertRaisesRegex(RuntimeError, "be 'L' or 'U'"):
                torch.linalg.eigh(t, UPLO=uplo)
            with self.assertRaisesRegex(ValueError, "be 'L' or 'U'"):
                np.linalg.eigh(t.cpu().numpy(), UPLO=uplo)

        # if non-empty out tensor with wrong shape is passed a warning is given
        a = random_hermitian_matrix(3, dtype=dtype, device=device)
        real_dtype = a.real.dtype if dtype.is_complex else dtype
        out_w = torch.empty(7, 7, dtype=real_dtype, device=device)
        out_v = torch.empty(7, 7, dtype=dtype, device=device)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.eigh(a, out=(out_w, out_v))
            # Check warning occurs
            self.assertEqual(len(w), 2)
            self.assertTrue(
                "An output with one or more elements was resized" in str(w[-2].message)
            )
            self.assertTrue(
                "An output with one or more elements was resized" in str(w[-1].message)
            )

        # dtypes should be safely castable
        out_w = torch.empty(0, dtype=real_dtype, device=device)
        out_v = torch.empty(0, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got int instead"):
            torch.linalg.eigh(a, out=(out_w, out_v))

        out_w = torch.empty(0, dtype=torch.int, device=device)
        out_v = torch.empty(0, dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got int instead"):
            torch.linalg.eigh(a, out=(out_w, out_v))

        # device should match
        if torch.cuda.is_available():
            wrong_device = "cpu" if self.device_type != "cpu" else "cuda"
            out_w = torch.empty(0, device=wrong_device, dtype=dtype)
            out_v = torch.empty(0, device=device, dtype=dtype)
            with self.assertRaisesRegex(
                RuntimeError, "tensors to be on the same device"
            ):
                torch.linalg.eigh(a, out=(out_w, out_v))
            out_w = torch.empty(0, device=device, dtype=dtype)
            out_v = torch.empty(0, device=wrong_device, dtype=dtype)
            with self.assertRaisesRegex(
                RuntimeError, "tensors to be on the same device"
            ):
                torch.linalg.eigh(a, out=(out_w, out_v))

    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double)
    @unittest.skipIf(
        _get_torch_cuda_version() < (12, 1), "Test is fixed on cuda 12.1 update 1."
    )
    def test_eigh_svd_illcondition_matrix_input_should_not_crash(self, device, dtype):
        # See https://github.com/pytorch/pytorch/issues/94772, https://github.com/pytorch/pytorch/issues/105359
        # This test crashes with `cusolver error: CUSOLVER_STATUS_EXECUTION_FAILED` on cuda 11.8,
        # but passes on cuda 12.1 update 1 or later.
        a = torch.ones(512, 512, dtype=dtype, device=device)
        a[0, 0] = 1.0e-5
        a[-1, -1] = 1.0e5

        eigh_out = torch.linalg.eigh(a)
        svd_out = torch.linalg.svd(a)

        # Matrix input a is too ill-conditioned.
        # We'll just compare the first two singular values/eigenvalues. They are 1.0e5 and 511.0
        # The precision override with tolerance of 1.0 makes sense since ill-conditioned inputs are hard to converge
        # to exact values.
        self.assertEqual(
            eigh_out.eigenvalues.sort(descending=True).values[:2],
            [1.0e5, 511.0],
            atol=1.0,
            rtol=1.0e-2,
        )
        self.assertEqual(svd_out.S[:2], [1.0e5, 511.0], atol=1.0, rtol=1.0e-2)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    def test_eigvalsh(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_matrix

        def run_test(shape, batch, uplo):
            matrix = random_hermitian_matrix(shape, *batch, dtype=dtype, device=device)
            expected_w = np.linalg.eigvalsh(matrix.cpu().numpy(), UPLO=uplo)
            actual_w = torch.linalg.eigvalsh(matrix, UPLO=uplo)
            self.assertEqual(actual_w, expected_w)

            # check the out= variant
            out = torch.empty_like(actual_w)
            ans = torch.linalg.eigvalsh(matrix, UPLO=uplo, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, actual_w)

        shapes = (0, 3, 5)
        batches = ((), (3,), (2, 2))
        uplos = ["U", "L"]
        for shape, batch, uplo in itertools.product(shapes, batches, uplos):
            run_test(shape, batch, uplo)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_eigvalsh_errors_and_warnings(self, device, dtype):
        # eigvalsh requires a square matrix
        t = torch.randn(2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.linalg.eigvalsh(t)

        # eigvalsh requires 'uplo' parameter to be 'U' or 'L'
        t = torch.randn(3, 3, device=device, dtype=dtype)
        for uplo in ["a", "wrong"]:
            with self.assertRaisesRegex(RuntimeError, "be 'L' or 'U'"):
                torch.linalg.eigvalsh(t, UPLO=uplo)
            with self.assertRaisesRegex(ValueError, "be 'L' or 'U'"):
                np.linalg.eigvalsh(t.cpu().numpy(), UPLO=uplo)

        # if non-empty out tensor with wrong shape is passed a warning is given
        real_dtype = t.real.dtype if dtype.is_complex else dtype
        out = torch.empty_like(t).to(real_dtype)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.eigvalsh(t, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue(
                "An output with one or more elements was resized" in str(w[-1].message)
            )

        # dtypes should be safely castable
        out = torch.empty(0, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got int instead"):
            torch.linalg.eigvalsh(t, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = "cpu" if self.device_type != "cpu" else "cuda"
            out = torch.empty(0, device=wrong_device, dtype=dtype)
            with self.assertRaisesRegex(
                RuntimeError, "tensors to be on the same device"
            ):
                torch.linalg.eigvalsh(t, out=out)

    @onlyCPU
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_eigh_lwork_lapack(self, device, dtype):
        # test that the calculated lwork does not cause a crash, see https://github.com/pytorch/pytorch/issues/145801
        t = torch.rand(3000, 3000, device=device, dtype=dtype)
        y = torch.linalg.eigh(t)
        self.assertEqual(y.eigenvalues.shape, (3000,))

    @dtypes(*floating_and_complex_types())
    def test_kron(self, device, dtype):

        def run_test_case(a_shape, b_shape):
            a = torch.rand(a_shape, dtype=dtype, device=device)
            b = torch.rand(b_shape, dtype=dtype, device=device)

            expected = np.kron(a.cpu().numpy(), b.cpu().numpy())
            result = torch.kron(a, b)
            self.assertEqual(result, expected)

            # check the out= variant
            out = torch.empty_like(result)
            ans = torch.kron(a, b, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        shapes = [(4,), (2, 2), (1, 2, 3), (1, 2, 3, 3)]
        for a_shape, b_shape in itertools.product(shapes, reversed(shapes)):
            run_test_case(a_shape, b_shape)

    @dtypes(*floating_and_complex_types())
    def test_kron_empty(self, device, dtype):

        def run_test_case(empty_shape):
            a = torch.eye(3, dtype=dtype, device=device)
            b = torch.empty(empty_shape, dtype=dtype, device=device)
            result = torch.kron(a, b)
            expected = np.kron(a.cpu().numpy(), b.cpu().numpy())
            self.assertEqual(result, expected)

            # NumPy doesn't work if the first argument is empty
            result = torch.kron(b, a)
            self.assertEqual(result.shape, expected.shape)

        empty_shapes = [(0,), (2, 0), (1, 0, 3)]
        for empty_shape in empty_shapes:
            run_test_case(empty_shape)

    @dtypes(*floating_and_complex_types())
    def test_kron_errors_and_warnings(self, device, dtype):
        # if non-empty out tensor with wrong shape is passed a warning is given
        a = torch.eye(3, dtype=dtype, device=device)
        b = torch.ones((2, 2), dtype=dtype, device=device)
        out = torch.empty_like(a)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.kron(a, b, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue(
                "An output with one or more elements was resized" in str(w[-1].message)
            )

        # dtypes should match
        out = torch.empty_like(a).to(torch.int)
        with self.assertRaisesRegex(
            RuntimeError, "can't be cast to the desired output type"
        ):
            torch.kron(a, b, out=out)

    # This test confirms that torch.linalg.norm's dtype argument works
    # as expected, according to the function's documentation
    @dtypes(
        torch.float,
        torch.double,
        torch.cfloat,
        torch.cdouble,
        torch.bfloat16,
        torch.float16,
    )
    def test_norm_dtype(self, device, dtype):
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        def run_test_case(input_size, ord, keepdim, to_dtype):
            msg = (
                f"input_size={input_size}, ord={ord}, keepdim={keepdim}, "
                f"dtype={dtype}, to_dtype={to_dtype}"
            )
            input = make_arg(input_size)
            result = torch.linalg.norm(input, ord, keepdim=keepdim)
            self.assertEqual(result.dtype, input.real.dtype, msg=msg)

            result_out = torch.empty((0), dtype=result.dtype, device=device)
            torch.linalg.norm(input, ord, keepdim=keepdim, out=result_out)
            self.assertEqual(result, result_out, msg=msg)

            result = torch.linalg.norm(input.to(to_dtype), ord, keepdim=keepdim)
            result_with_dtype = torch.linalg.norm(
                input, ord, keepdim=keepdim, dtype=to_dtype
            )
            self.assertEqual(result, result_with_dtype, msg=msg)

            result_out_with_dtype = torch.empty_like(result_with_dtype)
            torch.linalg.norm(
                input, ord, keepdim=keepdim, dtype=to_dtype, out=result_out_with_dtype
            )
            self.assertEqual(result_with_dtype, result_out_with_dtype, msg=msg)

        ord_vector = [0, 1, -1, 2, -2, 3, -3, 4.5, -4.5, inf, -inf, None]

        # In these orders we are computing the 10-th power and 10-th root of numbers.
        # We avoid them for half-precision types as it makes the tests above too badly conditioned
        if dtype != torch.float16 and dtype != torch.bfloat16:
            ord_vector.extend([0.1, -0.1])
        ord_matrix = ["fro", "nuc", 1, -1, 2, -2, inf, -inf, None]
        S = 10

        if dtype == torch.cfloat:
            norm_dtypes = (torch.cfloat, torch.cdouble)
        elif dtype == torch.cdouble:
            norm_dtypes = (torch.cdouble,)
        elif dtype in (torch.float16, torch.bfloat16, torch.float):
            norm_dtypes = (torch.float, torch.double)
        elif dtype == torch.double:
            norm_dtypes = (torch.double,)
        else:
            raise RuntimeError("Unsupported dtype")

        for ord, keepdim, norm_dtype in product(ord_vector, (True, False), norm_dtypes):
            run_test_case((S,), ord, keepdim, norm_dtype)

        for ord, keepdim, norm_dtype in product(ord_matrix, (True, False), norm_dtypes):
            if ord in [2, -2, "nuc"]:
                # We need torch.svdvals
                if dtype == torch.float16 or dtype == torch.bfloat16:
                    continue

                # We need LAPACK or equivalent
                if (
                    torch.device(device).type == "cuda"
                    and not torch.cuda.has_magma
                    and not has_cusolver()
                ) or (torch.device(device).type == "cpu" and not torch._C.has_lapack):
                    continue
            run_test_case((S, S), ord, keepdim, norm_dtype)

    # This test confirms torch.linalg.norm bfloat16 and half get right result.
    @dtypes(torch.bfloat16, torch.float16)
    def test_norm_bfloat16_and_half(self, device, dtype):
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        def run_test_case(input_size, ord, keepdim):
            msg = (
                f"input_size={input_size}, ord={ord}, keepdim={keepdim}, "
                f"dtype={dtype}"
            )
            input = make_arg(input_size).fill_(1)
            result_ref = torch.linalg.norm(input.float(), ord, keepdim=keepdim).to(
                dtype=dtype
            )
            result = torch.linalg.norm(input, ord, keepdim=keepdim)
            self.assertEqual(result_ref, result, msg=msg)

        ord_vector = [0, 1, -1, 2, -2, 3, -3, 4.5, -4.5, inf, -inf, None]
        for S, ord, keepdim in product((10, 2049), ord_vector, (True, False)):
            run_test_case(
                (S,),
                ord,
                keepdim,
            )

    @dtypes(
        torch.float,
        torch.double,
        torch.cfloat,
        torch.cdouble,
        torch.bfloat16,
        torch.float16,
    )
    def test_vector_norm(self, device, dtype):
        if (
            IS_ARM64
            and device == "cpu"
            and dtype in [torch.float16, torch.bfloat16, torch.float32]
        ):
            raise unittest.SkipTest(
                "Fails on ARM, see https://github.com/pytorch/pytorch/issues/125438"
            )
        # have to use torch.randn(...).to(bfloat16) instead of
        # This test compares torch.linalg.vector_norm's output with
        # torch.linalg.norm given a flattened tensor
        ord_vector = [0, 0.9, 1, 2, 3, inf, -0.5, -1, -2, -3, -inf, 1 + 2j]
        input_sizes = [
            (1,),
            (10,),
            (4, 5),
            (3, 4, 5),
            (0,),
            (0, 10),
            (0, 0),
            (10, 0, 10),
        ]

        def vector_norm_reference(input, ord, dim=None, keepdim=False, dtype=None):
            if dim is None:
                input_maybe_flat = input.flatten(0, -1)
            else:
                input_maybe_flat = input

            result = torch.linalg.norm(
                input_maybe_flat, ord, dim=dim, keepdim=keepdim, dtype=dtype
            )
            if keepdim and dim is None:
                result = result.reshape([1] * input.dim())
            return result

        def run_test_case(input, ord, dim, keepdim, norm_dtype):
            if isinstance(ord, complex):
                error_msg = "Expected a non-complex scalar"
                with self.assertRaisesRegex(RuntimeError, error_msg):
                    torch.linalg.vector_norm(
                        input, ord, dim=dim, keepdim=keepdim, dtype=norm_dtype
                    )
            elif (
                input.numel() == 0
                and (ord < 0.0 or ord == inf)
                and (dim is None or input.shape[dim] == 0)
            ):
                # The operation does not have an identity.
                error_msg = "linalg.vector_norm cannot compute"
                with self.assertRaisesRegex(RuntimeError, error_msg):
                    torch.linalg.vector_norm(input, ord, dim=dim, keepdim=keepdim)
            else:
                msg = (
                    f"input.size()={input.size()}, ord={ord}, dim={dim}, "
                    f"keepdim={keepdim}, dtype={dtype}, norm_dtype={norm_dtype}"
                )
                result_dtype_reference = vector_norm_reference(
                    input, ord, dim=dim, keepdim=keepdim, dtype=norm_dtype
                )
                result_dtype = torch.linalg.vector_norm(
                    input, ord, dim=dim, keepdim=keepdim, dtype=norm_dtype
                )
                if dtype.is_complex:
                    result_dtype_reference = result_dtype_reference.real
                self.assertEqual(result_dtype, result_dtype_reference, msg=msg)

                if norm_dtype is not None:
                    ref = torch.linalg.vector_norm(
                        input.to(norm_dtype), ord, dim=dim, keepdim=keepdim
                    )
                    actual = torch.linalg.vector_norm(
                        input, ord, dim=dim, keepdim=keepdim, dtype=norm_dtype
                    )
                    self.assertEqual(ref, actual, msg=msg)

        if dtype == torch.cfloat:
            norm_dtypes = (None, torch.cfloat, torch.cdouble)
        elif dtype == torch.cdouble:
            norm_dtypes = (None, torch.cdouble)
        elif dtype in (torch.float16, torch.bfloat16, torch.float):
            norm_dtypes = (None, torch.float, torch.double)
        elif dtype == torch.double:
            norm_dtypes = (None, torch.double)
        else:
            raise RuntimeError("Unsupported dtype")

        for amp in [False, True]:
            with torch.autocast(device_type=device, enabled=amp):
                for input_size, ord, keepdim, norm_dtype in product(
                    input_sizes, ord_vector, [True, False], norm_dtypes
                ):
                    input = make_tensor(
                        input_size, dtype=dtype, device=device, low=-9, high=9
                    )
                    for dim in [None, random.randint(0, len(input_size) - 1)]:
                        run_test_case(input, ord, dim, keepdim, norm_dtype)

    def test_vector_norm_decom_unbacked_checks(self):
        from torch._refs.linalg import _check_vector_norm_args

        class Mod(torch.nn.Module):
            def __init__(self, ord, dim):
                super().__init__()
                self.ord = ord
                self.dim = dim

            def forward(self, a):
                x = a.item()
                tensor_unbacked_size = torch.ones(x, x + 1, x + 2)
                _check_vector_norm_args(tensor_unbacked_size, self.ord, self.dim)
                return tensor_unbacked_size

        def test(
            ord: Union[float, int],
            dim: Optional[DimsType],
            expect_numel_runtime_check: bool,
            expect_index_0_check: bool = False,
        ) -> None:
            m = Mod(ord, dim)
            exported_program: torch.export.ExportedProgram = torch.export.export(
                m, args=tuple(torch.tensor([1]))
            )
            self.assertEqual(
                "Runtime assertion failed for expression Ne(u0*(u0 + 1)*(u0 + 2), 0)"
                in exported_program.graph_module.code,
                expect_numel_runtime_check,
            )
            self.assertEqual(
                "Runtime assertion failed for expression Ne(u0, 0) | Ne(u0*(u0 + 1)*(u0 + 2), 0)"
                in exported_program.graph_module.code,
                expect_index_0_check,
            )

        # dim is int
        test(-1, 1, True)

        # dim is None
        test(-1, None, True)

        # len(dim) == 0
        test(-1, [], True)

        # shape[d] == 0
        test(-1, [0], False, True)

        # u0 + 1 == 0 is False we do not see a runtime assert in the generated graph.
        test(-1, [1], False, False)

        test(-1, [0, 1], False, True)
        test(-1, [0, 0], False, True)

    def test_vector_norm_dim_tuple_arg(self, device):
        test_cases = [
            # input size, dim, error, error message
            ((4,), (0,), None, None),
            ((4,), (1,), IndexError, r"Dimension out of range"),
            ((4,), (-2,), IndexError, r"Dimension out of range"),
            ((4, 3), (0, -1), None, None),
            (
                (4, 3),
                (0, 0),
                RuntimeError,
                r"dim 0 appears multiple times in the list of dims",
            ),
            (
                (4, 3),
                (0, -2),
                RuntimeError,
                r"dim 0 appears multiple times in the list of dims",
            ),
            ((4, 3), (0, 1.0), TypeError, r"argument 'dim' must be tuple of ints"),
            ((4, 3), (None,), TypeError, r"argument 'dim' must be tuple of ints"),
        ]
        for input_size, dim_tuple, error, error_msg in test_cases:
            input = torch.randn(input_size, device=device)
            # vector_norm should accept a tuple or a list for dim arg
            for dim in [dim_tuple, list(dim_tuple)]:
                if error is None:
                    torch.linalg.vector_norm(input, dim=dim)
                else:
                    with self.assertRaises(error):
                        torch.linalg.vector_norm(input, dim=dim)

    # This test compares torch.linalg.norm and numpy.linalg.norm to ensure that
    # their vector norm results match
    @dtypes(torch.float, torch.double)
    def test_norm_vector(self, device, dtype):
        def run_test_case(input, p, dim, keepdim):
            result = torch.linalg.norm(input, ord, dim, keepdim)
            input_numpy = input.cpu().numpy()
            result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)

            msg = f"input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}"
            self.assertEqual(result, result_numpy, msg=msg)

            result_out = torch.empty_like(result)
            torch.linalg.norm(input, ord, dim, keepdim, out=result_out)
            self.assertEqual(result, result_out, msg=msg)

        ord_vector = [0, 1, -1, 2, -2, 3, -3, 4.5, -4.5, inf, -inf]
        S = 10
        test_cases = [
            # input size, p settings, dim
            ((S,), ord_vector, None),
            ((S,), ord_vector, 0),
            ((S, S, S), ord_vector, 0),
            ((S, S, S), ord_vector, 1),
            ((S, S, S), ord_vector, 2),
            ((S, S, S), ord_vector, -1),
            ((S, S, S), ord_vector, -2),
        ]
        L = 1_000_000
        if dtype == torch.double:
            test_cases.append(((L,), ord_vector, None))
        for keepdim in [True, False]:
            for input_size, ord_settings, dim in test_cases:
                input = torch.randn(*input_size, dtype=dtype, device=device)
                for ord in ord_settings:
                    run_test_case(input, ord, dim, keepdim)

    # This test compares torch.linalg.norm, torch.linalg.matrix_norm and numpy.linalg.norm to
    # ensure that their matrix norm results match.
    @skipMeta  # https://github.com/pytorch/pytorch/issues/54082
    @skipCUDAIfNoMagma
    @dtypes(torch.float, torch.double)
    @precisionOverride({torch.float32: 2e-4})
    def test_norm_matrix(self, device, dtype):
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        def run_test_case(input, ord, dim, keepdim):
            msg = f"input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}"
            result = torch.linalg.norm(input, ord, dim, keepdim)
            input_numpy = input.cpu().numpy()
            result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)

            result = torch.linalg.norm(input, ord, dim, keepdim)
            self.assertEqual(result, result_numpy, msg=msg)
            if ord is not None and dim is not None:
                result = torch.linalg.matrix_norm(input, ord, dim, keepdim)
                self.assertEqual(result, result_numpy, msg=msg)

        ord_matrix = [1, -1, 2, -2, inf, -inf, "nuc", "fro"]
        S = 10
        test_cases = [
            # input size, dim
            ((S, S), None),
            ((S, S), (0, 1)),
            ((S, S), (1, 0)),
            ((S, S, S, S), (2, 0)),
            ((S, S, S, S), (-1, -2)),
            ((S, S, S, S), (-1, -3)),
            ((S, S, S, S), (-3, 2)),
        ]

        for (shape, dim), keepdim, ord in product(
            test_cases, [True, False], ord_matrix
        ):
            if ord in [2, -2, "nuc"]:
                # We need torch.svdvals
                if dtype == torch.float16 or dtype == torch.bfloat16:
                    continue
                # We need LAPACK or equivalent
                if (
                    torch.device(device).type == "cuda"
                    and not torch.cuda.has_magma
                    and not has_cusolver()
                ) or (torch.device(device).type == "cpu" and not torch._C.has_lapack):
                    continue
            run_test_case(make_arg(shape), ord, dim, keepdim)

    @onlyCUDA
    @dtypes(torch.bfloat16, torch.float16)
    def test_norm_fused_type_promotion(self, device, dtype):
        x = torch.randn(10, device=device, dtype=dtype)

        def profile_and_check(fn, x, kwargs):
            with torch.profiler.profile(
                activities=(torch.profiler.ProfilerActivity.CPU,)
            ) as p:
                fn(x, **kwargs, dtype=torch.float)
            # smoke check that profiler returned some events
            self.assertTrue("aten::linalg_vector_norm" in (e.name for e in p.events()))
            # test that there was no explicit copy
            self.assertFalse("aten::to" in (e.name for e in p.events()))

        for (
            f,
            kwargs,
        ) in zip((torch.linalg.vector_norm, torch.norm), ({}, {"p": 2})):
            profile_and_check(f, x, kwargs)

    @skipMeta  # https://github.com/pytorch/pytorch/issues/53739
    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-3})
    def test_cond(self, device, dtype):
        def run_test_case(input, p):
            result = torch.linalg.cond(input, p)
            result_numpy = np.linalg.cond(input.cpu().numpy(), p)
            self.assertEqual(
                result, result_numpy, rtol=1e-2, atol=self.precision, exact_dtype=False
            )
            self.assertEqual(result.shape, result_numpy.shape)

            # test out= variant
            out = torch.empty_like(result)
            ans = torch.linalg.cond(input, p, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        norm_types = [1, -1, 2, -2, inf, -inf, "fro", "nuc", None]
        input_sizes = [(32, 32), (2, 3, 3, 3)]
        for input_size in input_sizes:
            input = torch.randn(*input_size, dtype=dtype, device=device)
            for p in norm_types:
                run_test_case(input, p)

        # test empty batch sizes
        input_sizes = [(0, 3, 3), (0, 2, 5, 5)]
        for input_size in input_sizes:
            input = torch.randn(*input_size, dtype=dtype, device=device)
            for p in norm_types:
                run_test_case(input, p)

        # test non-square input
        input_sizes = [(16, 32), (32, 16), (2, 3, 5, 3), (2, 3, 3, 5)]
        for input_size in input_sizes:
            input = torch.randn(*input_size, dtype=dtype, device=device)
            for p in [2, -2, None]:
                run_test_case(input, p)

        # test for singular input
        a = torch.eye(3, dtype=dtype, device=device)
        a[-1, -1] = 0  # make 'a' singular
        for p in norm_types:
            try:
                run_test_case(a, p)
            except np.linalg.LinAlgError:
                # Numpy may fail to converge for some BLAS backends (although this is very rare)
                # See the discussion in https://github.com/pytorch/pytorch/issues/67675
                pass

        # test for 0x0 matrices. NumPy doesn't work for such input, we return 0
        input_sizes = [(0, 0), (2, 5, 0, 0)]
        for input_size in input_sizes:
            input = torch.randn(*input_size, dtype=dtype, device=device)
            for p in ["fro", 2]:
                expected_dtype = a.real.dtype if dtype.is_complex else dtype
                expected = torch.zeros(
                    input_size[:-2], dtype=expected_dtype, device=device
                )
                actual = torch.linalg.cond(input, p)
                self.assertEqual(actual, expected)

    @skipMeta  # https://github.com/pytorch/pytorch/issues/53739
    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-3})
    def test_cond_errors_and_warnings(self, device, dtype):
        norm_types = [1, -1, 2, -2, inf, -inf, "fro", "nuc", None]

        # cond expects the input to be at least 2-dimensional
        a = torch.ones(3, dtype=dtype, device=device)
        for p in norm_types:
            with self.assertRaisesRegex(RuntimeError, r"at least 2 dimensions"):
                torch.linalg.cond(a, p)

        # for some norm types cond expects the input to be square
        a = torch.ones(3, 2, dtype=dtype, device=device)
        norm_types = [1, -1, inf, -inf, "fro", "nuc"]
        for p in norm_types:
            with self.assertRaisesRegex(
                RuntimeError, r"must be batches of square matrices"
            ):
                torch.linalg.cond(a, p)

        # if non-empty out tensor with wrong shape is passed a warning is given
        a = torch.ones((2, 2), dtype=dtype, device=device)
        for p in ["fro", 2]:
            real_dtype = a.real.dtype if dtype.is_complex else dtype
            out = torch.empty(a.shape, dtype=real_dtype, device=device)
            with warnings.catch_warnings(record=True) as w:
                # Trigger warning
                torch.linalg.cond(a, p, out=out)
                # Check warning occurs
                self.assertEqual(len(w), 1)
                self.assertTrue(
                    "An output with one or more elements was resized"
                    in str(w[-1].message)
                )

        # dtypes should be safely castable
        out = torch.empty(0, dtype=torch.int, device=device)
        for p in ["fro", 2]:
            with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
                torch.linalg.cond(a, p, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = "cpu" if self.device_type != "cpu" else "cuda"
            out = torch.empty(0, dtype=dtype, device=wrong_device)
            for p in ["fro", 2]:
                with self.assertRaisesRegex(
                    RuntimeError, "tensors to be on the same device"
                ):
                    torch.linalg.cond(a, p, out=out)

        # for batched input if at least one matrix in the batch is not invertible,
        # we can't get the result for all other (possibly) invertible matrices in the batch without an explicit for loop.
        # this should change when at::inverse works with silent errors
        # NumPy works fine in this case because it's possible to silence the error and get the inverse matrix results
        # possibly filled with NANs
        batch_dim = 3
        a = torch.eye(3, 3, dtype=dtype, device=device)
        a = a.reshape((1, 3, 3))
        a = a.repeat(batch_dim, 1, 1)
        a[1, -1, -1] = 0  # now a[1] is singular
        for p in [1, -1, inf, -inf, "fro", "nuc"]:
            result = torch.linalg.cond(a, p)
            self.assertEqual(result[1], float("inf"))

        # check invalid norm type
        a = torch.ones(3, 3, dtype=dtype, device=device)
        for p in ["wrong_norm", 5]:
            with self.assertRaisesRegex(
                RuntimeError, f"linalg.cond got an invalid norm type: {p}"
            ):
                torch.linalg.cond(a, p)

    # This test calls torch.linalg.norm and numpy.linalg.norm with illegal arguments
    # to ensure that they both throw errors
    @dtypes(torch.float, torch.double)
    def test_norm_errors(self, device, dtype):
        def run_error_test_case(input, ord, dim, keepdim, error_type, error_regex):
            test_case_info = (
                f"test case input.size()={input.size()}, ord={ord}, dim={dim}, "
                f"keepdim={keepdim}, dtype={dtype}"
            )

            with self.assertRaisesRegex(error_type, error_regex, msg=test_case_info):
                torch.linalg.norm(input, ord, dim, keepdim)

            input_numpy = input.cpu().numpy()

            msg = f'numpy does not raise error but pytorch does, for case "{test_case_info}"'
            with self.assertRaises(Exception, msg=test_case_info):
                np.linalg.norm(input_numpy, ord, dim, keepdim)

        S = 10
        error_test_cases = [
            # input size, p settings, dim, error type, error regex
            (
                (S,),
                ["fro", "nuc"],
                None,
                RuntimeError,
                r"A must have at least 2 dimensions",
            ),
            (
                (S, S),
                [3.5],
                None,
                RuntimeError,
                r"matrix_norm: Order 3.5 not supported",
            ),
            ((S, S), [0], None, RuntimeError, r"matrix_norm: Order 0 not supported"),
            (
                (S, S),
                ["fail"],
                None,
                RuntimeError,
                r"matrix_norm: Order fail not supported",
            ),
            (
                (S, S),
                ["fro", "nuc"],
                0,
                RuntimeError,
                r"matrix_norm: dim must be a 2-tuple",
            ),
            (
                (S, S),
                ["fro", "nuc", 2],
                (0, 0),
                RuntimeError,
                r"dims must be different",
            ),
            (
                (S, S),
                ["fro", "nuc", 2],
                (-1, 1),
                RuntimeError,
                r"dims must be different",
            ),
            ((S, S), ["fro", "nuc", 2], (0, 4), IndexError, r"Dimension out of range"),
            ((S,), [0], (4,), IndexError, r"Dimension out of range"),
            ((S,), [None], (0, 0), RuntimeError, r"dim 0 appears multiple times"),
            (
                (S, S, S),
                [1],
                (0, 1, 2),
                RuntimeError,
                r"If dim is specified, it must be of length 1 or 2.",
            ),
            (
                (S, S, S),
                [1],
                None,
                RuntimeError,
                r"If dim is not specified but ord is, the input must be 1D or 2D",
            ),
        ]
        for keepdim in [True, False]:
            for (
                input_size,
                ord_settings,
                dim,
                error_type,
                error_regex,
            ) in error_test_cases:
                input = torch.randn(*input_size, dtype=dtype, device=device)
                for ord in ord_settings:
                    run_error_test_case(
                        input, ord, dim, keepdim, error_type, error_regex
                    )

    # Test complex number inputs for linalg.norm
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.cfloat, torch.cdouble)
    @precisionOverride({torch.cfloat: 5e-4})
    def test_norm_complex(self, device, dtype):
        def gen_error_message(input_size, ord, keepdim, dim=None):
            return f"complex norm failed for input size {input_size}, ord={ord}, keepdim={keepdim}, dim={dim}"

        vector_ords = [None, 0, 1, 2, 3, inf, -1, -2, -3, -inf]
        matrix_ords = [None, "fro", "nuc", 1, 2, inf, -1, -2, -inf]

        # Test supported ords
        for keepdim in [False, True]:
            # vector norm
            x = torch.randn(25, device=device, dtype=dtype)
            xn = x.cpu().numpy()
            for ord in vector_ords:
                res = torch.linalg.norm(x, ord, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, ord, keepdims=keepdim)
                msg = gen_error_message(x.size(), ord, keepdim)
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg, exact_dtype=False)

                res_out = torch.tensor([], device=device, dtype=res.dtype)
                torch.linalg.norm(x, ord, keepdim=keepdim, out=res_out)
                self.assertEqual(res_out.shape, expected.shape, msg=msg)
                self.assertEqual(res_out, expected, msg=msg)

            # matrix norm
            x = torch.randn(25, 25, device=device, dtype=dtype)
            xn = x.cpu().numpy()
            for ord in matrix_ords:
                res = torch.linalg.norm(x, ord, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, ord, keepdims=keepdim)
                msg = gen_error_message(x.size(), ord, keepdim)
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg, exact_dtype=False)

                res_out = torch.tensor([], device=device, dtype=res.dtype)
                torch.linalg.norm(x, ord, keepdim=keepdim, out=res_out)
                self.assertEqual(res_out.shape, expected.shape, msg=msg)
                self.assertEqual(res_out, expected, msg=msg)

    @onlyCPU
    def test_norm_complexhalf(self, device):
        def gen_error_message(input_size, ord, keepdim, dim=None):
            return f"complex norm failed for input size {input_size}, ord={ord}, keepdim={keepdim}, dim={dim}"

        vector_ords = [None, 0, 1, 2, 3, inf, -1, -2, -3, -inf]

        # Test supported ords
        for keepdim in [False, True]:
            # vector norm
            x = torch.randn(25, device=device, dtype=torch.chalf)
            x_cfloat = x.to(torch.cfloat)
            for ord in vector_ords:
                res = torch.linalg.norm(x, ord, keepdim=keepdim)
                res_float = torch.linalg.norm(x_cfloat, ord, keepdim=keepdim)
                msg = gen_error_message(x.size(), ord, keepdim)
                self.assertEqual(res.shape, res_float.shape, msg=msg)
                self.assertEqual(res.dtype, torch.half, msg=msg)
                self.assertEqual(res, res_float, msg=msg, exact_dtype=False)

                res_out = torch.tensor([], device=device, dtype=res.dtype)
                torch.linalg.norm(x, ord, keepdim=keepdim, out=res_out)
                self.assertEqual(res_out.shape, res_float.shape, msg=msg)
                self.assertEqual(res_out.dtype, torch.half, msg=msg)
                self.assertEqual(res_out, res_float, msg=msg, exact_dtype=False)

    # Test that linal.vector_norm gives the same result as numpy when inputs
    # contain extreme values (inf, -inf, nan)
    def test_vector_norm_extreme_values(self, device):
        vector_ords = [0, 1, 2, 3, inf, -1, -2, -3, -inf]
        vectors = []
        for pair in itertools.product([inf, -inf, 0.0, nan, 1.0], repeat=2):
            vectors.append(list(pair))
        for vector in vectors:
            x = torch.tensor(vector, device=device)
            x_n = x.cpu().numpy()
            for ord in vector_ords:
                msg = f"ord={ord}, vector={vector}"
                result = torch.linalg.vector_norm(x, ord=ord)
                result_n = np.linalg.norm(x_n, ord=ord)
                self.assertEqual(result, result_n, msg=msg)

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_vector_norm_reduce_over_1D_vector(self, device, dtype):
        input_sizes_and_dims = [
            ((6, 1), -1),
            ((3, 1, 2, 1), (1, 3)),
            ((1,), None),
        ]
        orders = [float("inf"), -float("inf"), 0, 1, -1, 2, -2]
        keepdims = [True, False]

        for input_size_and_dim, ord, keepdim in product(
            input_sizes_and_dims, orders, keepdims
        ):
            input_size = input_size_and_dim[0]
            dim = input_size_and_dim[1]
            if type(dim) is tuple and ord == 0:
                # skip because np.linalg.norm raises 'ValueError: Invalid norm order for matrices.'
                continue
            input = make_tensor(input_size, dtype=dtype, device=device, low=-9, high=9)
            result = torch.linalg.vector_norm(input, ord, dim, keepdim)
            result_numpy = np.linalg.norm(input.cpu().numpy(), ord, dim, keepdim)

            msg = f"input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}"
            self.assertEqual(result, result_numpy, msg=msg)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double)
    @precisionOverride({torch.float32: 2e-5})
    def test_matrix_norm(self, device, dtype):
        # Test only inputs for which torch.linalg.matrix_norm diverges from torch.linalg.norm
        A = make_tensor((2, 2, 2), dtype=dtype, device=device)

        with self.assertRaisesRegex(
            RuntimeError, r"linalg.matrix_norm:.*must have at least 2 dimensions.*"
        ):
            torch.linalg.matrix_norm(make_tensor((2,), dtype=dtype, device=device))
        with self.assertRaisesRegex(
            RuntimeError, r"linalg.matrix_norm:.*must be a 2-tuple.*"
        ):
            torch.linalg.matrix_norm(A, dim=(0,))
        with self.assertRaisesRegex(RuntimeError, r".*not supported.*"):
            torch.linalg.matrix_norm(A, ord=0)
        with self.assertRaisesRegex(RuntimeError, r".*not supported.*"):
            torch.linalg.matrix_norm(A, ord=3.0)
        with self.assertRaisesRegex(RuntimeError, "Expected a non-complex scalar"):
            torch.linalg.matrix_norm(A, ord=1 + 2j)

        # Test dim=None behavior
        ref = torch.linalg.norm(A, dim=(-2, -1))
        res = torch.linalg.matrix_norm(A)
        self.assertEqual(ref, res)

    # Test that linal.norm gives the same result as numpy when inputs
    # contain extreme values (inf, -inf, nan)
    @unittest.skipIf(IS_WINDOWS, "Skipped on Windows!")
    @unittest.skipIf(IS_MACOS, "Skipped on MacOS!")
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_norm_extreme_values(self, device):
        vector_ords = [0, 1, 2, 3, inf, -1, -2, -3, -inf]
        # matrix_ords 'nuc', 2, -2 are skipped currently
        # See issue https://github.com/pytorch/pytorch/issues/71911
        matrix_ords = ["fro", 1, inf, -1, -inf]
        vectors = []
        matrices = []
        for pair in itertools.product([inf, -inf, 0.0, nan, 1.0], repeat=2):
            vectors.append(list(pair))
            matrices.append([[pair[0], pair[1]]])
            matrices.append([[pair[0]], [pair[1]]])
        for vector in vectors:
            x = torch.tensor(vector).to(device)
            x_n = x.cpu().numpy()
            for ord in vector_ords:
                msg = f"ord={ord}, vector={vector}"
                result = torch.linalg.norm(x, ord=ord)
                result_n = np.linalg.norm(x_n, ord=ord)
                self.assertEqual(result, result_n, msg=msg)

        # TODO: Remove this function once the broken cases are fixed
        def is_broken_matrix_norm_case(ord, x):
            if self.device_type == "cuda":
                if x.size() == torch.Size([1, 2]):
                    if ord in ["nuc", 2, -2] and isnan(x[0][0]) and x[0][1] == 1:
                        # These cases are broken because of an issue with svd
                        # https://github.com/pytorch/pytorch/issues/43567
                        return True
                if ord in ["nuc", 2, -2]:
                    # These cases are broken because of another issue with svd
                    # https://github.com/pytorch/pytorch/issues/52633
                    return True
            return False

        for matrix in matrices:
            x = torch.tensor(matrix).to(device)
            x_n = x.cpu().numpy()
            for ord in matrix_ords:
                msg = f"ord={ord}, matrix={matrix}"
                if is_broken_matrix_norm_case(ord, x):
                    continue
                else:
                    result_n = np.linalg.norm(x_n, ord=ord)
                    result = torch.linalg.norm(x, ord=ord)
                    self.assertEqual(result, result_n, msg=msg)

    # Test degenerate shape results match numpy for linalg.norm vector norms
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_norm_vector_degenerate_shapes(self, device, dtype):
        def run_test_case(input, ord, dim, keepdim):
            msg = f"input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}"
            if (
                input.numel() == 0
                and (ord < 0.0 or ord == inf)
                and (dim is None or input.shape[dim] == 0)
            ):
                with self.assertRaises(RuntimeError):
                    torch.linalg.norm(input, ord, dim, keepdim)
            else:
                input_numpy = input.cpu().numpy()
                result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)
                result = torch.linalg.norm(input, ord, dim, keepdim)
                self.assertEqual(result, result_numpy, msg=msg)

        ord_vector = [0, 0.5, 1, 2, 3, inf, -0.5, -1, -2, -3, -inf]
        S = 10
        test_cases = [
            # input size, dim
            ((0,), None),
            ((0, S), 0),
            ((0, S), 1),
            ((S, 0), 0),
            ((S, 0), 1),
        ]
        for keepdim in [True, False]:
            for input_size, dim in test_cases:
                input = torch.randn(*input_size, dtype=dtype, device=device)
                for ord in ord_vector:
                    run_test_case(input, ord, dim, keepdim)

    # Test degenerate shape results match numpy for linalg.norm matrix norms
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_norm_matrix_degenerate_shapes(self, device, dtype):
        def run_test_case(input, ord, dim, keepdim, should_error):
            msg = f"input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}"
            input_numpy = input.cpu().numpy()
            ops = [torch.linalg.norm]

            if ord is not None and dim is not None:
                ops.append(torch.linalg.matrix_norm)

            if should_error:
                with self.assertRaises(ValueError):
                    np.linalg.norm(input_numpy, ord, dim, keepdim)
                for op in ops:
                    with self.assertRaises(IndexError):
                        op(input, ord, dim, keepdim)
            else:
                result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)
                for op in ops:
                    result = op(input, ord, dim, keepdim)
                    self.assertEqual(result, result_numpy, msg=msg)

        ord_matrix = ["fro", "nuc", 1, 2, inf, -1, -2, -inf, None]
        S = 10
        test_cases = [
            # input size, p settings that cause error, dim
            ((0, 0), [1, 2, inf, -1, -2, -inf], None),
            ((0, S), [2, inf, -2, -inf], None),
            ((S, 0), [1, 2, -1, -2], None),
            ((S, S, 0), [], (0, 1)),
            ((1, S, 0), [], (0, 1)),
            ((0, 0, S), [1, 2, inf, -1, -2, -inf], (0, 1)),
            ((0, 0, S), [1, 2, inf, -1, -2, -inf], (1, 0)),
        ]

        for keepdim in [True, False]:
            for input_size, error_ords, dim in test_cases:
                input = torch.randn(*input_size, dtype=dtype, device=device)
                for ord in ord_matrix:
                    run_test_case(input, ord, dim, keepdim, ord in error_ords)

    def test_norm_fastpaths(self, device):
        x = torch.randn(3, 5, device=device)

        # slow path
        result = torch.linalg.norm(x, 4.5, 1)
        expected = torch.pow(x.abs().pow(4.5).sum(1), 1.0 / 4.5)
        self.assertEqual(result, expected)

        # fast 0-norm
        result = torch.linalg.norm(x, 0, 1)
        expected = (x != 0).type_as(x).sum(1)
        self.assertEqual(result, expected)

        # fast 1-norm
        result = torch.linalg.norm(x, 1, 1)
        expected = x.abs().sum(1)
        self.assertEqual(result, expected)

        # fast 2-norm
        result = torch.linalg.norm(x, 2, 1)
        expected = torch.sqrt(x.pow(2).sum(1))
        self.assertEqual(result, expected)

        # fast 3-norm
        result = torch.linalg.norm(x, 3, 1)
        expected = torch.pow(x.pow(3).abs().sum(1), 1.0 / 3.0)
        self.assertEqual(result, expected)

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    # NumPy computes only in float64 and complex128 precisions
    # for float32 or complex64 results might be very different from float64 or complex128
    @dtypes(torch.float64, torch.complex128)
    def test_eig_numpy(self, device, dtype):
        def run_test(shape, *, symmetric=False):
            from torch.testing._internal.common_utils import random_symmetric_matrix

            if not dtype.is_complex and symmetric:
                # for symmetric real-valued inputs eigenvalues and eigenvectors have imaginary part equal to zero
                # unlike NumPy the result is not cast to float32 or float64 dtype in this case
                a = random_symmetric_matrix(
                    shape[-1], *shape[:-2], dtype=dtype, device=device
                )
            else:
                a = make_tensor(shape, dtype=dtype, device=device)

            actual = torch.linalg.eig(a)

            # compare with NumPy
            # the eigenvalues are not necessarily ordered
            # so order of NumPy and PyTorch can be different
            expected = np.linalg.eig(a.cpu().numpy())

            # sort NumPy output
            ind = np.argsort(expected[0], axis=-1)[::-1]
            expected = (
                np.take_along_axis(expected[0], ind, axis=-1),
                np.take_along_axis(expected[1], ind[:, None], axis=-1),
            )

            # sort PyTorch output
            # torch.argsort doesn't work with complex inputs, NumPy sorting on CPU is used instead
            # RuntimeError: _th_sort not supported on CUDAType for ComplexDouble
            # RuntimeError: "sorting_kernel_method_name" not implemented for 'ComplexDouble'
            ind = np.argsort(actual[0].cpu().numpy(), axis=-1)[::-1]
            actual_np = [x.cpu().numpy() for x in actual]
            sorted_actual = (
                np.take_along_axis(actual_np[0], ind, axis=-1),
                np.take_along_axis(actual_np[1], ind[:, None], axis=-1),
            )

            self.assertEqual(expected[0], sorted_actual[0], exact_dtype=False)
            self.assertEqual(abs(expected[1]), abs(sorted_actual[1]), exact_dtype=False)

        shapes = [
            (0, 0),  # Empty matrix
            (5, 5),  # Single matrix
            (0, 0, 0),
            (0, 5, 5),  # Zero batch dimension tensors
            (2, 5, 5),  # 3-dim tensors
            (2, 1, 5, 5),
        ]  # 4-dim tensors
        for shape in shapes:
            run_test(shape)
            run_test(shape, symmetric=True)

    @onlyCUDA
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    def test_eig_compare_backends(self, device, dtype):
        def run_test(shape, *, symmetric=False):
            from torch.testing._internal.common_utils import random_symmetric_matrix

            if not dtype.is_complex and symmetric:
                # for symmetric real-valued inputs eigenvalues and eigenvectors have imaginary part equal to zero
                a = random_symmetric_matrix(
                    shape[-1], *shape[:-2], dtype=dtype, device=device
                )
            else:
                a = make_tensor(shape, dtype=dtype, device=device)

            actual = torch.linalg.eig(a)

            complementary_device = "cpu"

            # compare with CPU
            expected = torch.linalg.eig(a.to(complementary_device))
            self.assertEqual(expected[0], actual[0])
            self.assertEqual(expected[1], actual[1])

        shapes = [
            (0, 0),  # Empty matrix
            (5, 5),  # Single matrix
            (0, 0, 0),
            (0, 5, 5),  # Zero batch dimension tensors
            (2, 5, 5),  # 3-dim tensors
            (2, 1, 5, 5),
        ]  # 4-dim tensors
        for shape in shapes:
            run_test(shape)
            run_test(shape, symmetric=True)

    @slowTest
    @onlyCUDA
    @skipCUDAIfNoMagma
    @dtypes(torch.float32)
    def test_eig_check_magma(self, device, dtype):
        # For CUDA inputs only matrices of size larger than 2048x2048 actually call MAGMA library
        shape = (2049, 2049)
        a = make_tensor(shape, dtype=dtype, device=device)
        w, v = torch.linalg.eig(a)
        # check correctness using eigendecomposition identity
        self.assertEqual(a.to(v.dtype) @ v, w * v, atol=1e-3, rtol=1e-3)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_eig_errors_and_warnings(self, device, dtype):
        # eig requires the input to be at least 2 dimensional tensor
        a = make_tensor(2, dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "must have at least 2 dimensions"):
            torch.linalg.eig(a)

        # eig requires a square matrix
        a = make_tensor((2, 3), dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.linalg.eig(a)

        # if out tensor with floating dtype is passed for complex output an error is thrown
        if not dtype.is_complex:
            # The characteristic equation is p(lambda) = lambda^2 - 2lambda + 5 = 0, with roots lambda = 1[+-]2i
            a = torch.tensor([[3.0, -2.0], [4.0, -1.0]], dtype=dtype, device=device)
            out0 = torch.empty(0, device=device, dtype=dtype)
            out1 = torch.empty(0, device=device, dtype=dtype)
            with self.assertRaisesRegex(
                RuntimeError, "Expected eigenvalues to be safely castable"
            ):
                torch.linalg.eig(a, out=(out0, out1))

            out0 = torch.empty(0, device=device, dtype=torch.complex128)
            with self.assertRaisesRegex(
                RuntimeError, "Expected eigenvectors to be safely castable"
            ):
                torch.linalg.eig(a, out=(out0, out1))

        # dtypes should be safely castable
        a = make_tensor((3, 3), dtype=dtype, device=device)
        out0 = torch.empty(0, dtype=torch.int, device=device)
        out1 = torch.empty(0, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got eigenvalues with dtype Int"):
            torch.linalg.eig(a, out=(out0, out1))

        out0 = torch.empty(0, dtype=torch.complex128, device=device)
        with self.assertRaisesRegex(
            RuntimeError, "but got eigenvectors with dtype Int"
        ):
            torch.linalg.eig(a, out=(out0, out1))

        # if non-empty out tensor with wrong shape is passed a warning is given
        a = make_tensor((3, 3), dtype=dtype, device=device)
        out0 = torch.empty(1, device=device, dtype=torch.complex128)
        out1 = torch.empty(1, device=device, dtype=torch.complex128)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.eig(a, out=(out0, out1))
            # Check warning occurs
            self.assertEqual(len(w), 2)
            self.assertTrue(
                "An output with one or more elements was resized" in str(w[-1].message)
            )
            self.assertTrue(
                "An output with one or more elements was resized" in str(w[-2].message)
            )

        # device should match
        if torch.cuda.is_available():
            wrong_device = "cpu" if self.device_type != "cpu" else "cuda"
            out_w = torch.empty(0, device=wrong_device, dtype=torch.complex128)
            out_v = torch.empty(0, device=device, dtype=torch.complex128)
            with self.assertRaisesRegex(
                RuntimeError, "tensors to be on the same device"
            ):
                torch.linalg.eig(a, out=(out_w, out_v))
            out_w = torch.empty(0, device=device, dtype=torch.complex128)
            out_v = torch.empty(0, device=wrong_device, dtype=torch.complex128)
            with self.assertRaisesRegex(
                RuntimeError, "tensors to be on the same device"
            ):
                torch.linalg.eig(a, out=(out_w, out_v))

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    def test_eig_with_nan(self, device, dtype):
        for val in [np.inf, np.nan]:
            for batch_dim in [(), (10,)]:
                a = make_tensor((*batch_dim, 5, 5), device=device, dtype=dtype)
                a[..., -1, -1] = val

                with self.assertRaisesRegex(
                    RuntimeError, "torch.linalg.eig: input tensor should not"
                ):
                    torch.linalg.eig(a)

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    # NumPy computes only in float64 and complex128 precisions
    # for float32 or complex64 results might be very different from float64 or complex128
    @dtypes(torch.float64, torch.complex128)
    def test_eigvals_numpy(self, device, dtype):
        def run_test(shape, *, symmetric=False):
            from torch.testing._internal.common_utils import random_symmetric_matrix

            if not dtype.is_complex and symmetric:
                # for symmetric real-valued inputs eigenvalues and eigenvectors have imaginary part equal to zero
                # unlike NumPy the result is not cast to float32 or float64 dtype in this case
                a = random_symmetric_matrix(
                    shape[-1], *shape[:-2], dtype=dtype, device=device
                )
            else:
                a = make_tensor(shape, dtype=dtype, device=device)

            actual = torch.linalg.eigvals(a)

            # compare with NumPy
            # the eigenvalues are not necessarily ordered
            # so order of NumPy and PyTorch can be different
            expected = np.linalg.eigvals(a.cpu().numpy())

            # sort NumPy output
            ind = np.argsort(expected, axis=-1)[::-1]
            expected = np.take_along_axis(expected, ind, axis=-1)

            # sort PyTorch output
            # torch.argsort doesn't work with complex inputs, NumPy sorting on CPU is used instead
            # RuntimeError: _th_sort not supported on CUDAType for ComplexDouble
            # RuntimeError: "sorting_kernel_method_name" not implemented for 'ComplexDouble'
            ind = np.argsort(actual.cpu().numpy(), axis=-1)[::-1]
            actual_np = actual.cpu().numpy()
            sorted_actual = np.take_along_axis(actual_np, ind, axis=-1)

            self.assertEqual(expected, sorted_actual, exact_dtype=False)

        shapes = [
            (0, 0),  # Empty matrix
            (5, 5),  # Single matrix
            (0, 0, 0),
            (0, 5, 5),  # Zero batch dimension tensors
            (2, 5, 5),  # 3-dim tensors
            (2, 1, 5, 5),
        ]  # 4-dim tensors
        for shape in shapes:
            run_test(shape)
            run_test(shape, symmetric=True)

    @onlyCUDA
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    def test_eigvals_compare_backends(self, device, dtype):
        def run_test(shape, *, symmetric=False):
            from torch.testing._internal.common_utils import random_symmetric_matrix

            if not dtype.is_complex and symmetric:
                # for symmetric real-valued inputs eigenvalues and eigenvectors have imaginary part equal to zero
                a = random_symmetric_matrix(
                    shape[-1], *shape[:-2], dtype=dtype, device=device
                )
            else:
                a = make_tensor(shape, dtype=dtype, device=device)

            actual = torch.linalg.eigvals(a)

            complementary_device = "cpu"

            # compare with CPU
            expected = torch.linalg.eigvals(a.to(complementary_device))
            self.assertEqual(expected, actual)

            # check out= variant
            complex_dtype = dtype
            if not dtype.is_complex:
                complex_dtype = (
                    torch.complex128 if dtype == torch.float64 else torch.complex64
                )
            out = torch.empty(0, dtype=complex_dtype, device=device)
            ans = torch.linalg.eigvals(a, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(expected.to(complex_dtype), out)

            # check non-contiguous out
            if a.numel() > 0:
                out = torch.empty(
                    2 * shape[0], *shape[1:-1], dtype=complex_dtype, device=device
                )[::2]
                self.assertFalse(out.is_contiguous())
                ans = torch.linalg.eigvals(a, out=out)
                self.assertEqual(ans, out)
                self.assertEqual(expected.to(complex_dtype), out)

        shapes = [
            (0, 0),  # Empty matrix
            (5, 5),  # Single matrix
            (0, 0, 0),
            (0, 5, 5),  # Zero batch dimension tensors
            (2, 5, 5),  # 3-dim tensors
            (2, 1, 5, 5),
        ]  # 4-dim tensors
        for shape in shapes:
            run_test(shape)
            run_test(shape, symmetric=True)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_eigvals_errors_and_warnings(self, device, dtype):
        # eig requires the input to be at least 2 dimensional tensor
        a = make_tensor(2, dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "must have at least 2 dimensions"):
            torch.linalg.eigvals(a)

        # eig requires a square matrix
        a = make_tensor((2, 3), dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.linalg.eigvals(a)

        # if out tensor with floating dtype is passed for complex output an error is thrown
        if not dtype.is_complex:
            # The characteristic equation is p(lambda) = lambda^2 - 2lambda + 5 = 0, with roots lambda = 1[+-]2i
            a = torch.tensor([[3.0, -2.0], [4.0, -1.0]], dtype=dtype, device=device)
            out = torch.empty(0, device=device, dtype=dtype)
            with self.assertRaisesRegex(
                RuntimeError, "Expected eigenvalues to be safely castable"
            ):
                torch.linalg.eigvals(a, out=out)

        # dtypes should be safely castable
        a = make_tensor((3, 3), dtype=dtype, device=device)
        out = torch.empty(0, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got eigenvalues with dtype Int"):
            torch.linalg.eigvals(a, out=out)

        # if non-empty out tensor with wrong shape is passed a warning is given
        out = torch.empty(1, device=device, dtype=torch.complex128)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.eigvals(a, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue(
                "An output with one or more elements was resized" in str(w[-1].message)
            )

        # device should match
        if torch.cuda.is_available():
            wrong_device = "cpu" if self.device_type != "cpu" else "cuda"
            out_w = torch.empty(0, device=wrong_device, dtype=torch.complex128)
            with self.assertRaisesRegex(
                RuntimeError, "tensors to be on the same device"
            ):
                torch.linalg.eigvals(a, out=out_w)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_norm_old(self, device):
        def gen_error_message(input_size, p, keepdim, dim=None):
            return f"norm failed for input size {input_size}, p={p}, keepdim={keepdim}, dim={dim}"

        # 'nuc' norm uses SVD, and thus its precision is much lower than other norms.
        # test_svd takes @precisionOverride({torch.float: 1e-4, torch.cfloat: 2e-4}),
        # and here we are doing the same thing for nuc norm.
        class PrecisionContext:
            def __init__(self, test, norm):
                self.norm = norm
                self.saved_overrides = getattr(test, "precision_overrides", None)
                self.target_test = test

            def __enter__(self):
                if "nuc" != self.norm:
                    return None
                self.target_test.precision_overrides = {
                    torch.float: 1e-4,
                    torch.cfloat: 2e-4,
                }
                return self.target_test.precision_overrides

            def __exit__(self, type, value, tb) -> bool:
                if "nuc" != self.norm:
                    return True
                if self.saved_overrides is None:
                    delattr(self.target_test, "precision_overrides")
                else:
                    self.target_test.precision_overrides = self.saved_overrides
                return True

        for keepdim in [False, True]:
            # full reduction
            x = torch.randn(25, device=device)
            xn = x.cpu().numpy()
            for p in [0, 1, 2, 3, 4, inf, -inf, -1, -2, -3, 1.5]:
                res = x.norm(p, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, p, keepdims=keepdim)
                self.assertEqual(
                    res,
                    expected,
                    atol=1e-5,
                    rtol=0,
                    msg=gen_error_message(x.size(), p, keepdim),
                )

            # one dimension
            x = torch.randn(25, 25, device=device)
            xn = x.cpu().numpy()
            for p in [0, 1, 2, 3, 4, inf, -inf, -1, -2, -3]:
                dim = 1
                res = x.norm(p, dim, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, p, dim, keepdims=keepdim)
                msg = gen_error_message(x.size(), p, keepdim, dim)
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg)

            # matrix norm
            for p in ["fro", "nuc"]:
                res = x.norm(p, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, p, keepdims=keepdim)
                msg = gen_error_message(x.size(), p, keepdim)
                with PrecisionContext(self, p):
                    self.assertEqual(res.shape, expected.shape, msg=msg)
                    self.assertEqual(res, expected, msg=msg)

            # zero dimensions
            x = torch.randn((), device=device)
            xn = x.cpu().numpy()
            res = x.norm(keepdim=keepdim).cpu()
            expected = np.linalg.norm(xn, keepdims=keepdim)
            msg = gen_error_message(x.size(), None, keepdim)
            self.assertEqual(res.shape, expected.shape, msg=msg)
            self.assertEqual(res, expected, msg=msg)

            # larger tensor sanity check
            self.assertEqual(
                2 * torch.norm(torch.ones(10000), keepdim=keepdim),
                torch.norm(torch.ones(40000), keepdim=keepdim),
            )

            # matrix norm with non-square >2-D tensors, all combinations of reduction dims
            x = torch.randn(5, 6, 7, 8, device=device)
            xn = x.cpu().numpy()
            for p in ["fro", "nuc"]:
                for dim in itertools.product(*[list(range(4))] * 2):
                    if dim[0] == dim[1]:
                        continue
                    res = x.norm(p=p, dim=dim, keepdim=keepdim).cpu()
                    expected = np.linalg.norm(xn, ord=p, axis=dim, keepdims=keepdim)
                    msg = gen_error_message(x.size(), p, keepdim, dim)
                    with PrecisionContext(self, p):
                        self.assertEqual(res.shape, expected.shape, msg=msg)
                        self.assertEqual(res, expected, msg=msg)

    # Test that torch.norm with p=+/-inf propagates NaN
    def test_norm_old_nan_propagation(self, device):
        ords = [inf, -inf]
        for pair in itertools.product([0.0, nan, 1.0], repeat=2):
            x = torch.tensor(list(pair), device=device)
            for ord in ords:
                result = torch.norm(x, p=ord)
                result_check = torch.linalg.norm(x, ord=ord)
                self.assertEqual(result, result_check)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_norm_complex_old(self, device):
        def gen_error_message(input_size, p, keepdim, dim=None):
            return f"complex norm failed for input size {input_size}, p={p}, keepdim={keepdim}, dim={dim}"

        for keepdim in [False, True]:
            # vector norm
            x = torch.randn(25, device=device) + 1j * torch.randn(25, device=device)
            xn = x.cpu().numpy()
            for p in [0, 1, 2, 3, inf, -1, -2, -3, -inf]:
                res = x.norm(p, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, p, keepdims=keepdim)
                msg = gen_error_message(x.size(), p, keepdim)
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg)

            # matrix norm
            x = torch.randn(25, 25, device=device) + 1j * torch.randn(
                25, 25, device=device
            )
            xn = x.cpu().numpy()
            for p in ["nuc", "fro"]:
                res = x.norm(p, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, p, keepdims=keepdim)
                msg = gen_error_message(x.size(), p, keepdim)
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg, rtol=4e-6, atol=6e-4)

    # Ensure torch.norm with p='fro' and p=2 give the same results for mutually supported input combinations
    @dtypes(torch.float)
    def test_norm_fro_2_equivalence_old(self, device, dtype):
        input_sizes = [
            (0,),
            (10,),
            (0, 0),
            (4, 30),
            (0, 45),
            (100, 0),
            (45, 10, 23),
            (0, 23, 59),
            (23, 0, 37),
            (34, 58, 0),
            (0, 0, 348),
            (0, 3434, 0),
            (0, 0, 0),
            (5, 3, 8, 1, 3, 5),
        ]

        for input_size in input_sizes:
            a = make_tensor(input_size, dtype=dtype, device=device, low=-9, high=9)

            # Try full reduction
            dim_settings = [None]

            # Try all possible 1-D reductions
            dim_settings += list(range(-a.dim(), a.dim()))

            def wrap_dim(dim, ndims):
                assert (dim < ndims) and (dim >= -ndims)
                if dim >= 0:
                    return dim
                else:
                    return dim + ndims

            # Try all possible 2-D reductions
            dim_settings += [
                (d0, d1)
                for d0, d1 in itertools.combinations(range(-a.dim(), a.dim()), 2)
                if wrap_dim(d0, a.dim()) != wrap_dim(d1, a.dim())
            ]

            for dim in dim_settings:
                for keepdim in [True, False]:
                    a_norm_2 = torch.norm(a, p=2, dim=dim, keepdim=keepdim)
                    a_norm_fro = torch.norm(a, p="fro", dim=dim, keepdim=keepdim)
                    self.assertEqual(a_norm_fro, a_norm_2)

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_nuclear_norm_axes_small_brute_force_old(self, device):
        def check_single_nuclear_norm(x, axes):
            if self.device_type != "cpu" and randrange(100) < 95:
                return  # too many cpu <==> device copies

            a = np.asarray(x.cpu())
            expected = np.linalg.norm(a, "nuc", axis=axes)

            ans = torch.norm(x, "nuc", dim=axes)
            self.assertTrue(ans.is_contiguous())
            self.assertEqual(ans.shape, expected.shape)
            self.assertEqual(
                ans.cpu(), expected, rtol=1e-02, atol=1e-03, equal_nan=True
            )

            out = torch.zeros(expected.shape, dtype=x.dtype, device=x.device)
            ans = torch.norm(x, "nuc", dim=axes, out=out)
            self.assertIs(ans, out)
            self.assertTrue(ans.is_contiguous())
            self.assertEqual(ans.shape, expected.shape)
            self.assertEqual(
                ans.cpu(), expected, rtol=1e-02, atol=1e-03, equal_nan=True
            )

        for n in range(1, 3):
            for m in range(1, 3):
                for axes in itertools.permutations([0, 1], 2):
                    # 2d, inner dimensions C
                    x = torch.randn(n, m, device=device)
                    check_single_nuclear_norm(x, axes)

                    # 2d, inner dimensions Fortran
                    x = torch.randn(m, n, device=device).mT
                    check_single_nuclear_norm(x, axes)

                    # 2d, inner dimensions non-contiguous
                    x = torch.randn(n, 2 * m, device=device)[:, ::2]
                    check_single_nuclear_norm(x, axes)

                    # 2d, all dimensions non-contiguous
                    x = torch.randn(7 * n, 2 * m, device=device)[::7, ::2]
                    check_single_nuclear_norm(x, axes)

                for o in range(1, 3):
                    for axes in itertools.permutations([0, 1, 2], 2):
                        # 3d, inner dimensions C
                        x = torch.randn(o, n, m, device=device)
                        check_single_nuclear_norm(x, axes)

                        # 3d, inner dimensions Fortran
                        x = torch.randn(o, m, n, device=device).mT
                        check_single_nuclear_norm(x, axes)

                        # 3d, inner dimensions non-contiguous
                        x = torch.randn(o, n, 2 * m, device=device)[:, :, ::2]
                        check_single_nuclear_norm(x, axes)

                        # 3d, all dimensions non-contiguous
                        x = torch.randn(7 * o, 5 * n, 2 * m, device=device)[
                            ::7, ::5, ::2
                        ]
                        check_single_nuclear_norm(x, axes)

                    for r in range(1, 3):
                        for axes in itertools.permutations([0, 1, 2, 3], 2):
                            # 4d, inner dimensions C
                            x = torch.randn(r, o, n, m, device=device)
                            check_single_nuclear_norm(x, axes)

                            # 4d, inner dimensions Fortran
                            x = torch.randn(r, o, n, m, device=device).mT
                            check_single_nuclear_norm(x, axes)

                            # 4d, inner dimensions non-contiguous
                            x = torch.randn(r, o, n, 2 * m, device=device)[:, :, :, ::2]
                            check_single_nuclear_norm(x, axes)

                            # 4d, all dimensions non-contiguous
                            x = torch.randn(7 * r, 5 * o, 11 * n, 2 * m, device=device)[
                                ::7, ::5, ::11, ::2
                            ]
                            check_single_nuclear_norm(x, axes)

    @skipCUDAIfNoMagma
    def test_nuclear_norm_exceptions_old(self, device):
        for lst in [], [1], [1, 2]:
            x = torch.tensor(lst, dtype=torch.double, device=device)
            for axes in (), (0,):
                self.assertRaises(RuntimeError, torch.norm, x, "nuc", axes)
            self.assertRaises(RuntimeError, torch.norm, x, "nuc", (0, 1))

        x = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.double, device=device)
        self.assertRaisesRegex(
            RuntimeError, "must be different", torch.norm, x, "nuc", (0, 0)
        )
        self.assertRaisesRegex(
            IndexError, "Dimension out of range", torch.norm, x, "nuc", (0, 2)
        )

    @skipCUDAIfNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.double, torch.cdouble)
    def test_svd_lowrank(self, device, dtype):
        from torch.testing._internal.common_utils import (
            random_lowrank_matrix,
            random_sparse_matrix,
        )

        def run_subtest(
            actual_rank, matrix_size, batches, device, svd_lowrank, **options
        ):
            density = options.pop("density", 1)
            if isinstance(matrix_size, int):
                rows = columns = matrix_size
            else:
                rows, columns = matrix_size
            if density == 1:
                a_input = random_lowrank_matrix(
                    actual_rank, rows, columns, *batches, device=device, dtype=dtype
                )
                a = a_input
            else:
                assert batches == ()
                a_input = random_sparse_matrix(
                    rows, columns, density, device=device, dtype=dtype
                )
                a = a_input.to_dense()

            q = min(*size)
            u, s, v = svd_lowrank(a_input, q=q, niter=3, **options)

            # check if u, s, v is a SVD
            u, s, v = u[..., :q], s[..., :q], v[..., :q]
            A = (u * s.unsqueeze(-2)).matmul(v.mH)
            self.assertEqual(A, a, rtol=1e-7, atol=2e-7)

            # check if svd_lowrank produces same singular values as linalg.svdvals
            U, S, Vh = torch.linalg.svd(a, full_matrices=False)
            V = Vh.mH
            self.assertEqual(s, S, rtol=5e-7, atol=1e-7)

            if density == 1:
                # actual_rank is known only for dense inputs
                #
                # check if pairs (u, U) and (v, V) span the same
                # subspaces, respectively
                u, v = u[..., :actual_rank], v[..., :actual_rank]
                U, V = U[..., :actual_rank], V[..., :actual_rank]
                expected_ones = u.mH.matmul(U).det().abs()
                self.assertEqual(expected_ones, torch.ones_like(expected_ones))
                self.assertEqual(
                    v.mH.matmul(V).det().abs(), torch.ones_like(expected_ones)
                )

        all_batches = [(), (1,), (3,), (2, 3)]
        for actual_rank, size, all_batches in [  # noqa: B020
            (2, (17, 4), all_batches),
            (4, (17, 4), all_batches),
            (4, (17, 17), all_batches),
            (10, (100, 40), all_batches),
            (7, (1000, 1000), [()]),
        ]:
            # dense input
            for batches in all_batches:
                run_subtest(actual_rank, size, batches, device, torch.svd_lowrank)
                if size != size[::-1]:
                    run_subtest(
                        actual_rank, size[::-1], batches, device, torch.svd_lowrank
                    )

        # sparse input
        for size in [(17, 4), (4, 17), (17, 17), (100, 40), (40, 100), (1000, 1000)]:
            for density in [0.005, 0.1]:
                run_subtest(None, size, (), device, torch.svd_lowrank, density=density)

        # jitting support
        jitted = torch.jit.script(torch.svd_lowrank)
        actual_rank, size, batches = 2, (17, 4), ()
        run_subtest(actual_rank, size, batches, device, jitted)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @precisionOverride({torch.float: 1e-4, torch.cfloat: 2e-4})
    @setLinalgBackendsToDefaultFinally
    @dtypes(*floating_and_complex_types())
    @serialTest()
    def test_svd(self, device, dtype):
        # tests linalg.svd, svd, linalg.svdvals
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        backends = ["default"]

        if torch.device(device).type == "cuda":
            if torch.cuda.has_magma:
                backends.append("magma")
            if has_cusolver() or has_hipsolver():
                backends.append("cusolver")

        ns = (12, 4, 2, 0)
        batches = ((), (0,), (1,), (2,), (2, 1), (0, 2))
        drivers = (None, "gesvd", "gesvdj", "gesvda")

        for backend in backends:
            torch.backends.cuda.preferred_linalg_library(backend)

            for batch, m, n, driver in product(batches, ns, ns, drivers):
                if not (backend == "cusolver" or driver is None):
                    # only test cases below and skip otherwise:
                    # - backend == 'cusolver' (driver can be anything)
                    # - backend != 'cusolver' (driver should only be None)
                    continue

                shape = batch + (m, n)
                k = min(m, n)
                A = make_arg(shape)
                U, S, Vh = torch.linalg.svd(A, full_matrices=False, driver=driver)
                self.assertEqual((U @ S.to(A.dtype).diag_embed()) @ Vh, A)

                U_f, S_f, Vh_f = torch.linalg.svd(A, full_matrices=True, driver=driver)
                self.assertEqual(S_f, S)
                self.assertEqual(
                    (U_f[..., :k] @ S_f.to(A.dtype).diag_embed()) @ Vh_f[..., :k, :], A
                )

                S_s = torch.linalg.svdvals(A, driver=driver)
                self.assertEqual(S_s, S)

                U, S, V = torch.svd(A, some=True)
                self.assertEqual((U @ S.to(A.dtype).diag_embed()) @ V.mH, A)

                U_f, S_f, V_f = torch.svd(A, some=False)
                self.assertEqual(S_f, S)
                self.assertEqual(
                    (U_f[..., :k] @ S_f.to(A.dtype).diag_embed()) @ V_f[..., :k].mH, A
                )

                S_s = torch.svd(A, compute_uv=False).S
                self.assertEqual(S_s, S)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.complex128)
    def test_invariance_error_spectral_decompositions(self, device, dtype):
        make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=True)
        A = make_arg((3, 3))
        with self.assertRaisesRegex(RuntimeError, "ill-defined"):
            U, _, Vh = torch.linalg.svd(A, full_matrices=False)
            (U + Vh).sum().abs().backward()

        A = make_arg((3, 3))
        with self.assertRaisesRegex(RuntimeError, "ill-defined"):
            V = torch.linalg.eig(A).eigenvectors
            V.sum().abs().backward()

        A = make_arg((3, 3))
        A = A + A.mH
        with self.assertRaisesRegex(RuntimeError, "ill-defined"):
            Q = torch.linalg.eigh(A).eigenvectors
            Q.sum().abs().backward()

    # I don't know how much memory this test uses but on complex64 it needs at least 4GB
    @largeTensorTest("4GB", device="cuda")
    @serialTest(TEST_CUDA)
    @skipCUDAIfNoCusolver  # MAGMA backend doesn't work in this case
    @precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4})
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_svd_memory_allocation(self, device, dtype):
        # test for https://github.com/pytorch/pytorch/issues/61949
        # the problem was that tensors of incorrect size were allocated and then narrowed
        m = 3
        n = 2**20
        a = make_tensor((m, n), dtype=dtype, device=device)
        # the following should run without errors
        S = torch.linalg.svdvals(a)
        result = torch.linalg.svd(a, full_matrices=False)
        self.assertEqual(result.S, S)

    def cholesky_solve_test_helper(self, A_dims, b_dims, upper, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        b = torch.randn(*b_dims, dtype=dtype, device=device)
        A = random_hermitian_pd_matrix(*A_dims, dtype=dtype, device=device)
        L = torch.cholesky(A, upper=upper)
        return b, A, L

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride(
        {
            torch.float32: 1e-3,
            torch.complex64: 1e-3,
            torch.float64: 1e-8,
            torch.complex128: 1e-8,
        }
    )
    def test_cholesky_solve(self, device, dtype):
        for (k, n), upper in itertools.product(
            zip([2, 3, 5], [3, 5, 7]), [True, False]
        ):
            b, A, L = self.cholesky_solve_test_helper(
                (n,), (n, k), upper, device, dtype
            )
            x = torch.cholesky_solve(b, L, upper=upper)
            self.assertEqual(b, np.matmul(A.cpu(), x.cpu()))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride(
        {
            torch.float32: 1e-3,
            torch.complex64: 1e-3,
            torch.float64: 1e-8,
            torch.complex128: 1e-8,
        }
    )
    def test_cholesky_solve_batched(self, device, dtype):
        def cholesky_solve_batch_helper(A_dims, b_dims, upper):
            b, A, L = self.cholesky_solve_test_helper(
                A_dims, b_dims, upper, device, dtype
            )
            x_exp_list = []
            for i in range(b_dims[0]):
                x_exp_list.append(torch.cholesky_solve(b[i], L[i], upper=upper))
            x_exp = torch.stack(x_exp_list)  # Stacked output
            x_act = torch.cholesky_solve(b, L, upper=upper)  # Actual output
            self.assertEqual(x_act, x_exp)  # Equality check
            Ax = np.matmul(A.cpu(), x_act.cpu())
            self.assertEqual(b, Ax)  # Correctness check

        for upper, batchsize in itertools.product([True, False], [1, 3, 4]):
            cholesky_solve_batch_helper((5, batchsize), (batchsize, 5, 10), upper)

    @slowTest
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride(
        {
            torch.float32: 1e-3,
            torch.complex64: 1e-3,
            torch.float64: 1e-8,
            torch.complex128: 1e-8,
        }
    )
    def test_cholesky_solve_batched_many_batches(self, device, dtype):
        for A_dims, b_dims in zip([(5, 256, 256), (5,)], [(5, 10), (512, 512, 5, 10)]):
            for upper in [True, False]:
                b, A, L = self.cholesky_solve_test_helper(
                    A_dims, b_dims, upper, device, dtype
                )
                x = torch.cholesky_solve(b, L, upper)
                Ax = torch.matmul(A, x)
                self.assertEqual(Ax, b.expand_as(Ax))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride(
        {
            torch.float32: 1e-3,
            torch.complex64: 1e-3,
            torch.float64: 1e-8,
            torch.complex128: 1e-8,
        }
    )
    def test_cholesky_solve_batched_broadcasting(self, device, dtype):
        from numpy.linalg import solve
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def run_test(A_dims, b_dims, upper):
            A_matrix_size = A_dims[-1]
            A_batch_dims = A_dims[:-2]
            A = random_hermitian_pd_matrix(
                A_matrix_size, *A_batch_dims, dtype=dtype, device="cpu"
            )
            b = torch.randn(*b_dims, dtype=dtype, device="cpu")
            x_exp = torch.tensor(
                solve(A.numpy(), b.numpy()), dtype=dtype, device=device
            )
            A, b = A.to(dtype=dtype, device=device), b.to(dtype=dtype, device=device)
            L = torch.linalg.cholesky(A, upper=upper)
            x = torch.cholesky_solve(b, L, upper=upper)
            self.assertEqual(x, x_exp)
            # https://github.com/pytorch/pytorch/issues/42695
            x = torch.cholesky_solve(b, L, upper=upper, out=x)
            self.assertEqual(x, x_exp)

        # test against numpy.linalg.solve
        for upper in [True, False]:
            run_test((2, 1, 3, 4, 4), (2, 1, 3, 4, 6), upper)  # no broadcasting
            run_test((2, 1, 3, 4, 4), (4, 6), upper)  # broadcasting b
            run_test((4, 4), (2, 1, 3, 4, 2), upper)  # broadcasting A
            run_test((1, 3, 1, 4, 4), (2, 1, 3, 4, 5), upper)  # broadcasting A & b

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_cholesky_solve_out_errors_and_warnings(self, device, dtype):
        # dtypes should be safely castable
        a = torch.eye(2, dtype=dtype, device=device)
        b = torch.randn(2, 1, dtype=dtype, device=device)
        out = torch.empty(0, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
            torch.cholesky_solve(b, a, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = "cpu" if self.device_type != "cpu" else "cuda"
            out = torch.empty(0, dtype=dtype, device=wrong_device)
            with self.assertRaisesRegex(
                RuntimeError, "tensors to be on the same device"
            ):
                torch.cholesky_solve(b, a, out=out)

        # if out tensor with wrong shape is passed a warning is given
        with warnings.catch_warnings(record=True) as w:
            out = torch.empty(1, dtype=dtype, device=device)
            # Trigger warning
            torch.cholesky_solve(b, a, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue(
                "An output with one or more elements was resized" in str(w[-1].message)
            )

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_cholesky_solve_backward(self, device, dtype):
        b_dims = (5, 2)
        L_dims = (5, 5)

        for test_L_grad in (False, True):
            b = torch.randn(*b_dims, dtype=dtype, device=device, requires_grad=True)
            L = torch.randn(
                *L_dims, dtype=dtype, device=device, requires_grad=test_L_grad
            )
            if test_L_grad:
                torch.autograd.gradcheck(
                    lambda b, L: torch.cholesky_solve(b, torch.tril(L), upper=False),
                    (b, L),
                )
            else:
                torch.autograd.gradcheck(
                    lambda b: torch.cholesky_solve(b, L, upper=False), (b,)
                )

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride(
        {
            torch.float32: 2e-3,
            torch.complex64: 2e-3,
            torch.float64: 1e-8,
            torch.complex128: 1e-8,
        }
    )
    def test_inverse(self, device, dtype):
        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_arg = partial(make_fullrank, device=device, dtype=dtype)

        def run_test(torch_inverse, matrix, batches, n):
            matrix_inverse = torch_inverse(matrix)

            # Compare against NumPy output
            # NumPy uses 'gesv' LAPACK routine solving the equation A A_inv = I
            # But in PyTorch 'gertf' + 'getrs' is used. As such, there may be some element-wise differences
            expected = np.linalg.inv(matrix.cpu().numpy())
            self.assertEqual(
                matrix_inverse, expected, atol=self.precision, rtol=self.precision
            )

            # Additional correctness tests, check matrix*matrix_inverse == identity
            identity = torch.eye(n, dtype=dtype, device=device)
            self.assertEqual(
                identity.expand_as(matrix),
                np.matmul(matrix.cpu(), matrix_inverse.cpu()),
            )
            self.assertEqual(
                identity.expand_as(matrix),
                np.matmul(matrix_inverse.cpu(), matrix.cpu()),
            )

            # check the out= variant
            # prepare the expected out tensor
            matrix_inverse_out = torch.empty(*batches, n, n, dtype=dtype, device=device)
            matrix_inverse_out_t = matrix_inverse_out.mT.clone(
                memory_format=torch.contiguous_format
            )
            matrix_inverse_out = matrix_inverse_out_t.mT
            ans = torch_inverse(matrix, out=matrix_inverse_out)
            self.assertEqual(matrix_inverse_out, ans, atol=0, rtol=0)
            self.assertEqual(matrix_inverse_out, matrix_inverse, atol=0, rtol=0)

            # batched matrices: 3+ dimensional tensors, check matrix_inverse same as single-inverse for each matrix
            if matrix.ndim > 2 and batches[0] != 0:
                expected_inv_list = []
                p = int(
                    np.prod(batches)
                )  # use `p` instead of -1, so that the test works for empty input as well
                for mat in matrix.contiguous().view(p, n, n):
                    expected_inv_list.append(torch_inverse(mat))
                expected_inv = torch.stack(expected_inv_list).view(*batches, n, n)
                if self.device_type == "cuda" and dtype in [
                    torch.float32,
                    torch.complex64,
                ]:
                    # single-inverse is done using cuSOLVER, while batched inverse is done using MAGMA
                    # individual values can be significantly different for fp32, hence rather high rtol is used
                    # the important thing is that torch_inverse passes above checks with identity
                    self.assertEqual(matrix_inverse, expected_inv, atol=1e-1, rtol=1e-2)
                else:
                    self.assertEqual(matrix_inverse, expected_inv)

        # helper function for testing torch.linalg.inv_ex
        def test_inv_ex(input, out=None):
            if out is not None:
                info = torch.empty(0, dtype=torch.int32, device=device)
                return torch.linalg.inv_ex(input, out=(out, info)).inverse
            return torch.linalg.inv_ex(input).inverse

        for torch_inverse in [torch.inverse, torch.linalg.inv, test_inv_ex]:
            for batches, n in itertools.product([[], [0], [2], [2, 1]], [0, 5]):
                matrices = make_arg(*batches, n, n)
                run_test(torch_inverse, matrices, batches, n)

                # test non-contiguous input
                run_test(torch_inverse, matrices.mT, batches, n)
                if n > 0:
                    run_test(
                        torch_inverse,
                        make_arg(*batches, 2 * n, 2 * n)
                        .view(-1, n * 2, n * 2)[:, ::2, ::2]
                        .view(*batches, n, n),
                        batches,
                        n,
                    )

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_inv_ex_info_device(self, device, dtype):
        A = torch.eye(3, 3, dtype=dtype, device=device)
        info = torch.linalg.inv_ex(A).info
        self.assertTrue(info.device == A.device)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_inv_ex_singular(self, device, dtype):
        # if the input matrix is not invertible, info with positive integer is returned
        A = torch.eye(3, 3, dtype=dtype, device=device)
        A[-1, -1] = 0  # Now A is singular
        info = torch.linalg.inv_ex(A).info
        self.assertEqual(info, 3)
        with self.assertRaisesRegex(
            torch.linalg.LinAlgError,
            r"diagonal element 3 is zero, the inversion could not be completed",
        ):
            torch.linalg.inv_ex(A, check_errors=True)

        # if at least one matrix in the batch is not positive definite,
        # batched info with positive integer for the corresponding matrix is returned
        A = torch.eye(3, 3, dtype=dtype, device=device)
        A = A.reshape((1, 3, 3))
        A = A.repeat(5, 1, 1)
        A[3, -2, -2] = 0  # Now A[3] is singular
        info = torch.linalg.inv_ex(A).info

        expected_info = torch.zeros(A.shape[:-2], dtype=torch.int32, device=device)
        expected_info[3] = 2
        self.assertEqual(info, expected_info)
        with self.assertRaisesRegex(
            torch.linalg.LinAlgError,
            r"\(Batch element 3\): The diagonal element 2 is zero",
        ):
            torch.linalg.inv_ex(A, check_errors=True)

    @slowTest
    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride(
        {
            torch.float32: 2e-3,
            torch.complex64: 2e-3,
            torch.float64: 1e-5,
            torch.complex128: 1e-5,
        }
    )
    def test_inverse_many_batches(self, device, dtype):
        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_arg = partial(make_fullrank, device=device, dtype=dtype)

        def test_inverse_many_batches_helper(torch_inverse, b, n):
            matrices = make_arg(b, n, n)
            matrices_inverse = torch_inverse(matrices)

            # Compare against NumPy output
            expected = np.linalg.inv(matrices.cpu().numpy())
            self.assertEqual(matrices_inverse, expected, atol=self.precision, rtol=1e-3)

        for torch_inverse in [torch.inverse, torch.linalg.inv]:
            test_inverse_many_batches_helper(torch_inverse, 5, 256)
            test_inverse_many_batches_helper(torch_inverse, 3, 512)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @onlyNativeDeviceTypes  # TODO: XLA doesn't raise exception
    @dtypes(*floating_and_complex_types())
    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/129882")
    def test_inverse_errors(self, device, dtype):
        # inverse expects batches of square matrices as input
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.inverse(torch.randn(2, 3, 4, 3))

        # if input is not invertible, RuntimeError is raised mentioning the first non-invertible batch
        def run_test_singular_input(batch_dim, n):
            x = (
                torch.eye(3, 3, dtype=dtype, device=device)
                .reshape((1, 3, 3))
                .repeat(batch_dim, 1, 1)
            )
            x[n, -1, -1] = 0
            with self.assertRaisesRegex(
                torch.linalg.LinAlgError,
                rf"\(Batch element {n}\): The diagonal element 3 is zero",
            ):
                torch.inverse(x)

        for params in [(1, 0), (2, 0), (2, 1), (4, 0), (4, 2), (10, 2)]:
            run_test_singular_input(*params)

    @unittest.skipIf(
        IS_FBCODE or IS_SANDCASTLE,
        "Test fails for float64 on GPU (P100, V100) on Meta infra",
    )
    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @onlyNativeDeviceTypes  # TODO: XLA doesn't raise exception
    @dtypes(*floating_and_complex_types())
    def test_inverse_errors_large(self, device, dtype):
        # Test batched inverse of singular matrices reports errors without crashing (gh-51930)
        x = torch.empty((8, 10, 616, 616), dtype=dtype, device=device)
        x[:] = torch.eye(616, dtype=dtype, device=device)
        x[..., 10, 10] = 0
        with self.assertRaisesRegex(
            torch.linalg.LinAlgError,
            r"\(Batch element 0\): The diagonal element 11 is zero",
        ):
            torch.inverse(x)

    @precisionOverride(
        {
            torch.float32: 1e-3,
            torch.complex64: 1e-3,
            torch.float64: 1e-7,
            torch.complex128: 1e-7,
        }
    )
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_pinv(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def run_test_main(A, hermitian):
            # Testing against definition for pseudo-inverses
            A_pinv = torch.linalg.pinv(A, hermitian=hermitian)
            np_A = A.cpu().numpy()
            np_A_pinv = A_pinv.cpu().numpy()
            if A.numel() > 0:
                self.assertEqual(
                    A, np_A @ np_A_pinv @ np_A, atol=self.precision, rtol=self.precision
                )
                self.assertEqual(
                    A_pinv,
                    np_A_pinv @ np_A @ np_A_pinv,
                    atol=self.precision,
                    rtol=self.precision,
                )
                self.assertEqual(
                    np_A @ np_A_pinv, (np_A @ np_A_pinv).conj().swapaxes(-2, -1)
                )
                self.assertEqual(
                    np_A_pinv @ np_A, (np_A_pinv @ np_A).conj().swapaxes(-2, -1)
                )
            else:
                self.assertEqual(
                    A.shape, A_pinv.shape[:-2] + (A_pinv.shape[-1], A_pinv.shape[-2])
                )

            # Check out= variant
            out = torch.empty_like(A_pinv)
            ans = torch.linalg.pinv(A, hermitian=hermitian, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, A_pinv)

        def run_test_numpy(A, hermitian):
            # Check against NumPy output
            # Test float rcond, and specific value for each matrix
            rconds = [
                float(torch.rand(1)),
            ]
            # Test different types of rcond tensor
            for rcond_type in all_types():
                rconds.append(
                    torch.rand(A.shape[:-2], dtype=torch.double, device=device).to(
                        rcond_type
                    )
                )
            # Test broadcasting of rcond
            if A.ndim > 2:
                rconds.append(torch.rand(A.shape[-3], device=device))
            for rcond in rconds:
                actual = torch.linalg.pinv(A, rcond=rcond, hermitian=hermitian)
                torch_rtol = torch.linalg.pinv(A, rtol=rcond, hermitian=hermitian)
                self.assertEqual(actual, torch_rtol)
                numpy_rcond = rcond if isinstance(rcond, float) else rcond.cpu().numpy()
                expected = np.linalg.pinv(
                    A.cpu().numpy(), rcond=numpy_rcond, hermitian=hermitian
                )
                self.assertEqual(actual, expected, atol=self.precision, rtol=1e-5)

        for sizes in [
            (5, 5),
            (3, 5, 5),
            (3, 2, 5, 5),  # square matrices
            (3, 2),
            (5, 3, 2),
            (2, 5, 3, 2),  # fat matrices
            (2, 3),
            (5, 2, 3),
            (2, 5, 2, 3),  # thin matrices
            (0, 0),
            (0, 2),
            (2, 0),
            (3, 0, 0),
            (0, 3, 0),
            (0, 0, 3),
        ]:  # zero numel matrices
            A = torch.randn(*sizes, dtype=dtype, device=device)
            hermitian = False
            run_test_main(A, hermitian)
            run_test_numpy(A, hermitian)

        # Check hermitian = True
        for sizes in [
            (5, 5),
            (3, 5, 5),
            (3, 2, 5, 5),  # square matrices
            (0, 0),
            (3, 0, 0),
        ]:  # zero numel square matrices
            A = random_hermitian_pd_matrix(
                sizes[-1], *sizes[:-2], dtype=dtype, device=device
            )
            hermitian = True
            run_test_main(A, hermitian)
            run_test_numpy(A, hermitian)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_pinv_errors_and_warnings(self, device, dtype):
        # pinv requires at least 2D tensor
        a = torch.randn(1, device=device, dtype=dtype)
        with self.assertRaisesRegex(
            RuntimeError, "expected a tensor with 2 or more dimensions"
        ):
            torch.linalg.pinv(a)

        # if non-empty out tensor with wrong shape is passed a warning is given
        a = torch.randn(3, 3, dtype=dtype, device=device)
        out = torch.empty(7, 7, dtype=dtype, device=device)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.pinv(a, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue(
                "An output with one or more elements was resized" in str(w[-1].message)
            )

        # dtypes of out and input should be safely castable
        out = torch.empty_like(a).to(torch.int)
        with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
            torch.linalg.pinv(a, out=out)

        if torch.cuda.is_available():
            # device of out and input should match
            wrong_device = "cpu" if self.device_type != "cpu" else "cuda"
            out = torch.empty_like(a).to(wrong_device)
            with self.assertRaisesRegex(
                RuntimeError,
                "Expected result and input tensors to be on the same device",
            ):
                torch.linalg.pinv(a, out=out)

            # device of rcond and input should match
            wrong_device = "cpu" if self.device_type != "cpu" else "cuda"
            rcond = torch.full((), 1e-2, device=wrong_device)
            with self.assertRaisesRegex(
                RuntimeError, "Expected all tensors to be on the same device"
            ):
                torch.linalg.pinv(a, rcond=rcond)

        # rcond can't be complex
        rcond = torch.full((), 1j, device=device)
        with self.assertRaisesRegex(
            RuntimeError, "rcond tensor of complex type is not supported"
        ):
            torch.linalg.pinv(a, rcond=rcond)

        # atol can't be complex
        atol = torch.full((), 1j, device=device)
        with self.assertRaisesRegex(
            RuntimeError, "atol tensor of complex type is not supported"
        ):
            torch.linalg.pinv(a, atol=atol)

        # rtol can't be complex
        rtol = torch.full((), 1j, device=device)
        with self.assertRaisesRegex(
            RuntimeError, "rtol tensor of complex type is not supported"
        ):
            torch.linalg.pinv(a, rtol=rtol)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/129882")
    def test_inv_errors_and_warnings(self, device, dtype):
        # inv expects batches of square matrices as input
        a = torch.randn(2, 3, 4, 3, dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.linalg.inv(a)

        # inv requires the input to be at least 2 dimensional tensor
        a = torch.randn(2, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "must have at least 2 dimensions"):
            torch.linalg.inv(a)

        # if input is not invertible, RuntimeError is raised mentioning the first non-invertible batch
        def run_test_singular_input(batch_dim, n):
            a = (
                torch.eye(3, 3, dtype=dtype, device=device)
                .reshape((1, 3, 3))
                .repeat(batch_dim, 1, 1)
            )
            a[n, -1, -1] = 0
            with self.assertRaisesRegex(
                torch.linalg.LinAlgError,
                rf"\(Batch element {n}\): The diagonal element 3 is zero",
            ):
                torch.linalg.inv(a)

        for params in [(1, 0), (2, 0), (2, 1), (4, 0), (4, 2), (10, 2)]:
            run_test_singular_input(*params)

        # dtypes should match
        a = torch.eye(2, dtype=dtype, device=device)
        out = torch.empty(0, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got int instead"):
            torch.linalg.inv(a, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = "cpu" if self.device_type != "cpu" else "cuda"
            out = torch.empty(0, device=wrong_device, dtype=dtype)
            with self.assertRaisesRegex(
                RuntimeError, "tensors to be on the same device"
            ):
                torch.linalg.inv(a, out=out)

        # if out tensor with wrong shape is passed a warning is given
        with warnings.catch_warnings(record=True) as w:
            a = torch.eye(2, dtype=dtype, device=device)
            out = torch.empty(1, dtype=dtype, device=device)
            # Trigger warning
            torch.linalg.inv(a, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue(
                "An output with one or more elements was resized" in str(w[-1].message)
            )

        # if out tensor in batched column major format but with wrong a warning is given
        with warnings.catch_warnings(record=True) as w:
            a = torch.eye(2, dtype=dtype, device=device)
            out = torch.empty(3, 3, dtype=dtype, device=device)
            out = out.mT.clone(memory_format=torch.contiguous_format)
            out = out.mT
            self.assertTrue(out.mT.is_contiguous())
            # Trigger warning
            torch.linalg.inv(a, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue(
                "An output with one or more elements was resized" in str(w[-1].message)
            )

    def solve_test_helper(self, A_dims, b_dims, device, dtype):
        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_A = partial(make_fullrank, device=device, dtype=dtype)

        b = torch.randn(*b_dims, dtype=dtype, device=device)
        A = make_A(*A_dims)
        return b, A

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3})
    def test_solve(self, device, dtype):
        def run_test(n, batch, rhs):
            A_dims = (*batch, n, n)
            b_dims = (*batch, n, *rhs)
            b, A = self.solve_test_helper(A_dims, b_dims, device, dtype)

            # Correctness test
            x = torch.linalg.solve(A, b)
            if rhs == ():
                Ax = np.matmul(A.cpu(), x.unsqueeze(-1).cpu())
                Ax.squeeze_(-1)
            else:
                Ax = np.matmul(A.cpu(), x.cpu())
            self.assertEqual(b.expand_as(Ax), Ax)

            # Check against NumPy
            if rhs == ():
                # In NumPy 2, "b" can no longer be a vector (i.e. rhs == ()) if has batch dimensions.
                # So, reshape it to a matrix and back. Related documentation:
                # https://numpy.org/doc/1.26/reference/generated/numpy.linalg.solve.html
                # https://numpy.org/doc/2.0/reference/generated/numpy.linalg.solve.html
                expected = np.linalg.solve(
                    A.cpu().numpy(), b.cpu().numpy().reshape(*b.shape, 1)
                ).reshape(b.shape)
            else:
                expected = np.linalg.solve(A.cpu().numpy(), b.cpu().numpy())
            self.assertEqual(x, expected)

        batches = [(), (0,), (3,), (2, 3)]
        ns = [0, 5, 32]
        nrhs = [(), (1,), (5,)]
        for n, batch, rhs in itertools.product(ns, batches, nrhs):
            run_test(n, batch, rhs)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_solve_batched_broadcasting(self, device, dtype):
        from numpy.linalg import solve

        def run_test(A_dims, B_dims):
            A_matrix_size = A_dims[-1]
            A_batch_dims = A_dims[:-2]
            B, A = self.solve_test_helper(
                A_batch_dims + (A_matrix_size, A_matrix_size), B_dims, device, dtype
            )
            actual = torch.linalg.solve(A, B)
            expected = solve(A.cpu().numpy(), B.cpu().numpy())
            self.assertEqual(actual, expected)

        # test against numpy.linalg.solve
        run_test((5, 5), (2, 0, 5, 3))  # broadcasting with 0 batch dim
        run_test((2, 0, 5, 5), (5, 3))  # broadcasting with 0 batch dim
        run_test((2, 1, 3, 4, 4), (4, 6))  # broadcasting B
        run_test((4, 4), (2, 1, 3, 4, 2))  # broadcasting A
        run_test((1, 3, 1, 4, 4), (2, 1, 3, 4, 5))  # broadcasting A & B

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    @precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4})
    def test_tensorsolve(self, device, dtype):
        def run_test(a_shape, dims):
            a = torch.randn(a_shape, dtype=dtype, device=device)
            b = torch.randn(a_shape[:2], dtype=dtype, device=device)
            result = torch.linalg.tensorsolve(a, b, dims=dims)
            expected = np.linalg.tensorsolve(
                a.cpu().numpy(), b.cpu().numpy(), axes=dims
            )
            self.assertEqual(result, expected)

            # check the out= variant
            out = torch.empty_like(result)
            ans = torch.linalg.tensorsolve(a, b, dims=dims, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        a_shapes = [(2, 3, 6), (3, 4, 4, 3)]
        dims = [None, (0, 2)]
        for a_shape, d in itertools.product(a_shapes, dims):
            run_test(a_shape, d)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_tensorsolve_empty(self, device, dtype):
        # Check for empty inputs. NumPy does not work for these cases.
        a = torch.empty(0, 0, 1, 2, 3, 0, dtype=dtype, device=device)
        b = torch.empty(a.shape[:2], dtype=dtype, device=device)
        x = torch.linalg.tensorsolve(a, b)
        self.assertEqual(torch.tensordot(a, x, dims=len(x.shape)), b)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32)
    def test_tensorsolve_errors_and_warnings(self, device, dtype):
        # tensorsolve expects the input that can be reshaped to a square matrix
        a = torch.eye(2 * 3 * 4, dtype=dtype, device=device).reshape(
            (2 * 3, 4, 2, 3, 4)
        )
        b = torch.randn(8, 4, dtype=dtype, device=device)
        self.assertTrue(np.prod(a.shape[2:]) != np.prod(b.shape))
        with self.assertRaisesRegex(
            RuntimeError, r"Expected self to satisfy the requirement"
        ):
            torch.linalg.tensorsolve(a, b)

        # if non-empty out tensor with wrong shape is passed a warning is given
        out = torch.empty_like(a)
        b = torch.randn(6, 4, dtype=dtype, device=device)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.tensorsolve(a, b, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue(
                "An output with one or more elements was resized" in str(w[-1].message)
            )

        # dtypes should be safely castable
        out = torch.empty_like(a).to(torch.int)
        with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
            torch.linalg.tensorsolve(a, b, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = "cpu" if self.device_type != "cpu" else "cuda"
            out = torch.empty(0, dtype=dtype, device=wrong_device)
            with self.assertRaisesRegex(
                RuntimeError, "tensors to be on the same device"
            ):
                torch.linalg.tensorsolve(a, b, out=out)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float: 1e-3, torch.cfloat: 1e-3})
    def test_tensorinv(self, device, dtype):

        def run_test(a_shape, ind):
            a = torch.randn(a_shape, dtype=dtype, device=device)
            a_numpy = a.cpu().numpy()
            result = torch.linalg.tensorinv(a, ind=ind)
            expected = np.linalg.tensorinv(a_numpy, ind=ind)
            self.assertEqual(result, expected)

            # check the out= variant
            out = torch.empty_like(result)
            ans = torch.linalg.tensorinv(a, ind=ind, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        # compare to NumPy output
        run_test((12, 3, 4), ind=1)
        run_test((3, 8, 24), ind=2)
        run_test((18, 3, 3, 2), ind=1)
        run_test((1, 4, 2, 2), ind=2)
        run_test((2, 3, 5, 30), ind=3)
        run_test((24, 2, 2, 3, 2), ind=1)
        run_test((3, 4, 2, 3, 2), ind=2)
        run_test((1, 2, 3, 2, 3), ind=3)
        run_test((3, 2, 1, 2, 12), ind=4)

    @skipMeta  # See https://github.com/pytorch/pytorch/issues/53739
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_tensorinv_empty(self, device, dtype):
        for ind in range(1, 4):
            # Check for empty inputs. NumPy does not work for these cases.
            a = torch.empty(0, 0, 1, 2, 3, 0, dtype=dtype, device=device)
            a_inv = torch.linalg.tensorinv(a, ind=ind)
            self.assertEqual(a_inv.shape, a.shape[ind:] + a.shape[:ind])

    @skipMeta  # See https://github.com/pytorch/pytorch/issues/53739
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_tensorinv_errors_and_warnings(self, device, dtype):

        def check_shape(a_shape, ind):
            # tensorinv requires the input to satisfy
            # prod(a.shape[ind:]) == prod(a.shape[:ind])
            a = torch.randn(a_shape, dtype=dtype, device=device)
            with self.assertRaisesRegex(
                RuntimeError, "Expected self to satisfy the requirement"
            ):
                torch.linalg.tensorinv(a, ind=ind)

        def check_ind(a_shape, ind):
            a = torch.randn(a_shape, dtype=dtype, device=device)
            with self.assertRaisesRegex(
                RuntimeError, "Expected a strictly positive integer"
            ):
                torch.linalg.tensorinv(a, ind=ind)

        def check_out(a_shape, ind):
            # if non-empty out tensor with wrong shape is passed a warning is given
            a = torch.randn(a_shape, dtype=dtype, device=device)
            out = torch.empty_like(a)
            with warnings.catch_warnings(record=True) as w:
                # Trigger warning
                torch.linalg.tensorinv(a, ind=ind, out=out)
                # Check warning occurs
                self.assertEqual(len(w), 1)
                self.assertTrue(
                    "An output with one or more elements was resized"
                    in str(w[-1].message)
                )

            # dtypes should be safely castable
            out = torch.empty(0, dtype=torch.int, device=device)
            with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
                torch.linalg.tensorinv(a, ind=ind, out=out)

            # device should match
            if torch.cuda.is_available():
                wrong_device = "cpu" if self.device_type != "cpu" else "cuda"
                out = torch.empty(0, dtype=dtype, device=wrong_device)
                with self.assertRaisesRegex(
                    RuntimeError, "tensors to be on the same device"
                ):
                    torch.linalg.tensorinv(a, ind=ind, out=out)

        # test for invalid shape
        check_shape((2, 3, 4), ind=1)
        check_shape((1, 2, 3, 4), ind=3)

        # test for invalid ind
        check_ind((12, 3, 4), ind=-1)
        check_ind((18, 3, 3, 2), ind=0)

        # test for invalid out tensor
        check_out((12, 3, 4), ind=1)
        check_out((3, 8, 24), ind=2)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_tensorinv_singular_input(self, device, dtype):

        def check_singular_input(a_shape, ind):
            prod_ind_end = np.prod(a_shape[ind:])
            a = torch.eye(prod_ind_end, dtype=dtype, device=device)
            a[-1, -1] = 0  # Now `a` is singular
            a = a.reshape(a_shape)
            with self.assertRaisesRegex(
                torch.linalg.LinAlgError, "The diagonal element"
            ):
                torch.linalg.tensorinv(a, ind=ind)

        # test for non-invertible input
        check_singular_input((12, 3, 4), ind=1)
        check_singular_input((3, 6, 18), ind=2)

    def _test_dot_vdot_vs_numpy(self, device, dtype, torch_fn, np_fn):
        def check(x, y):
            # Compare with numpy
            res = torch_fn(x, y)
            if x.dtype == torch.bfloat16:
                ref = torch.from_numpy(
                    np.array(np_fn(x.cpu().float().numpy(), y.cpu().float().numpy()))
                )
            else:
                ref = torch.from_numpy(
                    np.array(np_fn(x.cpu().numpy(), y.cpu().numpy()))
                )
            if res.dtype == torch.bfloat16:
                self.assertEqual(res.cpu(), ref.bfloat16())
            else:
                self.assertEqual(res.cpu(), ref)

            # Test out variant
            out = torch.empty_like(res)
            torch_fn(x, y, out=out)
            self.assertEqual(out, res)

        # Empty
        x = torch.tensor([], dtype=dtype, device=device)
        y = torch.tensor([], dtype=dtype, device=device)
        check(x, y)

        # Contiguous
        x = 0.1 * torch.randn(5000, dtype=dtype, device=device)
        y = 0.1 * torch.randn(5000, dtype=dtype, device=device)
        check(x, y)

        # 0 strided
        y = 0.1 * torch.randn(1, dtype=dtype, device=device).expand(5000)
        check(x, y)

        # 2 strided
        check(x[::2], y[::2])

    @dtypes(torch.float, torch.cfloat, torch.bfloat16, torch.float16)
    @dtypesIfCUDA(torch.float, torch.cfloat)
    @precisionOverride({torch.cfloat: 1e-4, torch.float32: 5e-5, torch.bfloat16: 1e-0})
    def test_dot_vs_numpy(self, device, dtype):
        self._test_dot_vdot_vs_numpy(device, dtype, torch.dot, np.dot)

    @dtypes(torch.float, torch.cfloat)
    @precisionOverride({torch.cfloat: 1e-4, torch.float32: 5e-5})
    def test_vdot_vs_numpy(self, device, dtype):
        self._test_dot_vdot_vs_numpy(device, dtype, torch.vdot, np.vdot)

    def _test_dot_vdot_invalid_args(self, device, torch_fn, complex_dtypes=False):
        def check(x, y, regex):
            with self.assertRaisesRegex(RuntimeError, regex):
                torch_fn(x, y)

        if complex_dtypes:
            x = torch.randn(1, dtype=torch.cfloat, device=device)
            y = torch.randn(3, dtype=torch.cdouble, device=device)
        else:
            x = torch.randn(1, dtype=torch.float, device=device)
            y = torch.randn(3, dtype=torch.double, device=device)

        check(x, y, "dot : expected both vectors to have same dtype")
        check(x.reshape(1, 1), y, "1D tensors expected")
        check(x.expand(9), y.to(x.dtype), "inconsistent tensor size")

        if self.device_type != "cpu":
            x_cpu = x.expand(3).cpu()
            check(x_cpu, y.to(x.dtype), "Expected all tensors to be on the same device")

    @onlyNativeDeviceTypes
    def test_vdot_invalid_args(self, device):
        self._test_dot_vdot_invalid_args(device, torch.vdot)
        self._test_dot_vdot_invalid_args(device, torch.vdot, complex_dtypes=True)

    @onlyNativeDeviceTypes
    def test_dot_invalid_args(self, device):
        self._test_dot_vdot_invalid_args(device, torch.dot)
        self._test_dot_vdot_invalid_args(device, torch.dot, complex_dtypes=True)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_matrix_rank(self, device, dtype):
        matrix_rank = torch.linalg.matrix_rank

        def run_test(shape0, shape1, batch):
            a = torch.randn(*batch, shape0, shape1, dtype=dtype, device=device)
            rank_a = matrix_rank(a)

            self.assertEqual(rank_a, matrix_rank(a.mH))
            aaH = torch.matmul(a, a.mH)
            rank_aaH = matrix_rank(aaH)
            rank_aaH_hermitian = matrix_rank(aaH, hermitian=True)
            self.assertEqual(rank_aaH, rank_aaH_hermitian)
            aHa = torch.matmul(a.mH, a)
            self.assertEqual(matrix_rank(aHa), matrix_rank(aHa, hermitian=True))

            # check against NumPy
            self.assertEqual(rank_a, np.linalg.matrix_rank(a.cpu().numpy()))
            self.assertEqual(
                matrix_rank(a, 0.01), np.linalg.matrix_rank(a.cpu().numpy(), 0.01)
            )

            self.assertEqual(rank_aaH, np.linalg.matrix_rank(aaH.cpu().numpy()))
            self.assertEqual(
                matrix_rank(aaH, 0.01), np.linalg.matrix_rank(aaH.cpu().numpy(), 0.01)
            )

            # hermitian flag for NumPy was added in 1.14.0
            if np.lib.NumpyVersion(np.__version__) >= "1.14.0":
                self.assertEqual(
                    rank_aaH_hermitian,
                    np.linalg.matrix_rank(aaH.cpu().numpy(), hermitian=True),
                )
                self.assertEqual(
                    matrix_rank(aaH, 0.01, True),
                    np.linalg.matrix_rank(aaH.cpu().numpy(), 0.01, True),
                )

            # check out= variant
            out = torch.empty(a.shape[:-2], dtype=torch.int64, device=device)
            ans = matrix_rank(a, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, rank_a)

        shapes = (3, 13)
        batches = (
            (),
            (0,),
            (4,),
            (
                3,
                5,
            ),
        )
        for (shape0, shape1), batch in zip(
            itertools.product(shapes, reversed(shapes)), batches
        ):
            run_test(shape0, shape1, batch)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_matrix_rank_atol(self, device, dtype):

        def run_test_atol(shape0, shape1, batch):
            a = make_tensor((*batch, shape0, shape1), dtype=dtype, device=device)
            # Check against NumPy output
            # Test float tol, and specific value for each matrix
            tolerances = [
                float(torch.rand(1)),
            ]
            # Test different types of tol tensor
            for tol_type in all_types():
                tolerances.append(
                    make_tensor(a.shape[:-2], dtype=tol_type, device=device, low=0)
                )
            # Test broadcasting of tol
            if a.ndim > 2:
                tolerances.append(
                    make_tensor(a.shape[-3], dtype=torch.float32, device=device, low=0)
                )
            for tol in tolerances:
                actual = torch.linalg.matrix_rank(a, atol=tol)
                actual_tol = torch.linalg.matrix_rank(a, tol=tol)
                self.assertEqual(actual, actual_tol)
                numpy_tol = tol if isinstance(tol, float) else tol.cpu().numpy()
                expected = np.linalg.matrix_rank(a.cpu().numpy(), tol=numpy_tol)
                self.assertEqual(actual, expected)

        shapes = (3, 13)
        batches = (
            (),
            (0,),
            (4,),
            (
                3,
                5,
            ),
        )
        for (shape0, shape1), batch in zip(
            itertools.product(shapes, reversed(shapes)), batches
        ):
            run_test_atol(shape0, shape1, batch)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float64)
    def test_matrix_rank_atol_rtol(self, device, dtype):
        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_arg = partial(make_fullrank, device=device, dtype=dtype)

        # creates a matrix with singular values rank=n and singular values in range [2/3, 3/2]
        # the singular values are 1 + 1/2, 1 - 1/3, 1 + 1/4, 1 - 1/5, ...
        n = 9
        a = make_arg(n, n)

        # test float and tensor variants
        for tol_value in [0.81, torch.tensor(0.81, device=device)]:
            # using rtol (relative tolerance) takes into account the largest singular value (1.5 in this case)
            result = torch.linalg.matrix_rank(a, rtol=tol_value)
            self.assertEqual(
                result, 2
            )  # there are 2 singular values above 1.5*0.81 = 1.215

            # atol is used directly to compare with singular values
            result = torch.linalg.matrix_rank(a, atol=tol_value)
            self.assertEqual(result, 7)  # there are 7 singular values above 0.81

            # when both are specified the maximum tolerance is used
            result = torch.linalg.matrix_rank(a, atol=tol_value, rtol=tol_value)
            self.assertEqual(
                result, 2
            )  # there are 2 singular values above max(0.81, 1.5*0.81)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_matrix_rank_empty(self, device, dtype):
        matrix_rank = torch.linalg.matrix_rank

        # NumPy doesn't work for input with no elements
        def run_test(shape0, shape1, batch):
            a = torch.randn(*batch, shape0, shape1, dtype=dtype, device=device)
            rank_a = matrix_rank(a)
            expected = torch.zeros(batch, dtype=torch.int64, device=device)

            self.assertEqual(rank_a, matrix_rank(a.mH))

            aaH = torch.matmul(a, a.mH)
            rank_aaH = matrix_rank(aaH)
            rank_aaH_hermitian = matrix_rank(aaH, hermitian=True)
            self.assertEqual(rank_aaH, rank_aaH_hermitian)

            aHa = torch.matmul(a.mH, a)
            self.assertEqual(matrix_rank(aHa), matrix_rank(aHa, hermitian=True))

            self.assertEqual(rank_a, expected)
            self.assertEqual(matrix_rank(a, 0.01), expected)

            self.assertEqual(rank_aaH, expected)
            self.assertEqual(matrix_rank(aaH, 0.01), expected)

            self.assertEqual(rank_aaH_hermitian, expected)
            self.assertEqual(matrix_rank(aaH, 0.01, True), expected)

        batches = (
            (),
            (4,),
            (
                3,
                5,
            ),
        )
        for batch in batches:
            run_test(0, 0, batch)
            run_test(0, 3, batch)
            run_test(3, 0, batch)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_matrix_rank_out_errors_and_warnings(self, device, dtype):
        # dtypes should be safely castable
        a = torch.eye(2, dtype=dtype, device=device)
        out = torch.empty(0, dtype=torch.bool, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got result with dtype Bool"):
            torch.linalg.matrix_rank(a, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = "cpu" if self.device_type != "cpu" else "cuda"
            out = torch.empty(0, dtype=dtype, device=wrong_device)
            with self.assertRaisesRegex(
                RuntimeError, "tensors to be on the same device"
            ):
                torch.linalg.matrix_rank(a, out=out)

        # if out tensor with wrong shape is passed a warning is given
        with warnings.catch_warnings(record=True) as w:
            out = torch.empty(3, dtype=dtype, device=device)
            # Trigger warning
            torch.linalg.matrix_rank(a, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue(
                "An output with one or more elements was resized" in str(w[-1].message)
            )

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_matrix_rank_basic(self, device, dtype):
        matrix_rank = torch.linalg.matrix_rank

        a = torch.eye(10, dtype=dtype, device=device)
        self.assertEqual(matrix_rank(a).item(), 10)
        self.assertEqual(matrix_rank(a, hermitian=True).item(), 10)

        a[5, 5] = 0
        self.assertEqual(matrix_rank(a).item(), 9)
        self.assertEqual(matrix_rank(a, hermitian=True).item(), 9)

    @onlyNativeDeviceTypes
    @dtypes(torch.double)
    # This tests only the cases where torch.chain_matmul differs from torch.linalg.multi_dot which this is an "alias" for.
    def test_chain_matmul(self, device, dtype):
        # chain_matmul accepts a single input tensor while multi_dot does not
        t = make_tensor((2, 2), dtype=dtype, device=device)
        self.assertEqual(t, torch.chain_matmul(t))
        with self.assertRaisesRegex(
            RuntimeError, r"chain_matmul\(\): Expected one or more matrices"
        ):
            torch.chain_matmul()

        # chain_matmul expects all tensors to be 2D whereas multi_dot allows the first and last tensors to
        # be either 1D or 2D
        with self.assertRaisesRegex(
            RuntimeError, r"Tensor dimension is 1, expected 2 instead"
        ):
            torch.chain_matmul(
                make_tensor(1, dtype=dtype, device=device),
                make_tensor(1, dtype=dtype, device=device),
            )

    @onlyNativeDeviceTypes
    @dtypes(torch.double, torch.cdouble)
    def test_multi_dot(self, device, dtype):
        def check(*shapes):
            tensors = [
                make_tensor(shape, dtype=dtype, device=device) for shape in shapes
            ]
            np_arrays = [tensor.cpu().numpy() for tensor in tensors]
            res = torch.linalg.multi_dot(tensors).cpu()
            ref = torch.from_numpy(np.array(np.linalg.multi_dot(np_arrays)))
            self.assertEqual(res, ref)

        # test for inputs with empty dimensions
        check([0], [0])
        check([2], [2, 0])
        check([1, 0], [0])
        check([0, 2], [2, 1])
        check([2, 2], [2, 0])
        check([2, 0], [0, 3])
        check([0, 0], [0, 1])
        check([4, 2], [2, 0], [0, 3], [3, 2])

        # test variable output shapes
        check([2], [2])
        check([1, 2], [2])
        check([2], [2, 1])
        check([1, 2], [2, 1])
        check([3, 2], [2, 4])

        # test multiple input tensors
        check([3], [3, 4], [4, 2], [2, 5], [5])
        check([1, 2], [2, 2], [2, 3], [3, 1])

        # test large tensors
        check([10, 100], [100, 5], [5, 50])
        check([10, 20], [20, 30], [30, 5])

    @onlyNativeDeviceTypes
    @dtypes(torch.float)
    def test_multi_dot_errors(self, device, dtype):
        def check(tensors, out, msg):
            with self.assertRaisesRegex(RuntimeError, msg):
                torch.linalg.multi_dot(tensors, out=out)

        a = make_tensor(2, dtype=dtype, device=device)

        check([], None, "expected at least 2 tensors")
        check([a], None, "expected at least 2 tensors")

        check(
            [torch.tensor(1, device=device, dtype=dtype), a],
            None,
            "the first tensor must be 1D or 2D",
        )
        check(
            [a, torch.tensor(1, device=device, dtype=dtype)],
            None,
            "the last tensor must be 1D or 2D",
        )

        check([a, a, a], None, "tensor 1 must be 2D")
        check(
            [a, make_tensor((2, 2, 2), dtype=dtype, device=device), a],
            None,
            "tensor 1 must be 2D",
        )

        check(
            [a, make_tensor(2, dtype=torch.double, device=device)],
            None,
            "all tensors must have be the same dtype",
        )
        check(
            [a, a],
            torch.empty(0, device=device, dtype=torch.double),
            "expected out tensor to have dtype",
        )

        if self.device_type == "cuda":
            check(
                [a, make_tensor(2, dtype=dtype, device="cpu")],
                None,
                "all tensors must be on the same device",
            )
            check(
                [a, a],
                torch.empty(0, dtype=dtype),
                "expected out tensor to be on device",
            )

        check(
            [a, make_tensor(3, dtype=dtype, device=device)],
            None,
            "cannot be multiplied",
        )
        check(
            [a, make_tensor((3, 2), dtype=dtype, device=device), a],
            None,
            "cannot be multiplied",
        )

    @precisionOverride({torch.float32: 5e-6, torch.complex64: 5e-6})
    @skipCUDAIfNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_qr(self, device, dtype):
        def run_test(tensor_dims, some):
            A = torch.randn(*tensor_dims, dtype=dtype, device=device)
            Q, R = torch.qr(A, some=some)

            # Check0: Q[-2:] = (m, n_columns), R[-2:] = (n_columns, n)
            m, n = tensor_dims[-2:]
            n_columns = m if (not some) and m > n else min(m, n)
            self.assertEqual(Q.size(-2), m)
            self.assertEqual(R.size(-1), n)
            self.assertEqual(Q.size(-1), n_columns)

            A_ = A.cpu().numpy()
            Q_ = Q.cpu().numpy()
            R_ = R.cpu().numpy()

            # Check1: A = QR
            self.assertEqual(A_, np.matmul(Q_, R_))

            # Check2: A = QR (with out)
            Q_out, R_out = torch.full_like(Q, math.nan), torch.full_like(R, math.nan)
            torch.qr(A, some=some, out=(Q_out, R_out))
            Q_out_ = Q_out.cpu().numpy()
            R_out_ = R_out.cpu().numpy()
            self.assertEqual(A_, np.matmul(Q_out_, R_out_))

            # Check3: Q == Q_out, R == R_out
            self.assertEqual(Q_, Q_out_)
            self.assertEqual(R_, R_out_)

            # Check4: Q^{T}Q = I, triu(R) = R
            eye = (
                torch.eye(n_columns, device=device, dtype=dtype)
                .expand(Q.shape[:-2] + (n_columns, n_columns))
                .cpu()
                .numpy()
            )
            self.assertEqual(np.matmul(Q_.swapaxes(-1, -2).conj(), Q_), eye)
            self.assertEqual(R.triu(), R)

        tensor_dims_list = [
            (0, 5),
            (0, 0),
            (5, 0),  # Empty Tensors
            (2, 1, 0, 5),
            (2, 1, 0, 0),
            (2, 1, 5, 0),
            (2, 0, 5, 5),  # Batched empty Tensors
            (3, 5),
            (5, 5),
            (5, 3),  # Single matrix
            (7, 3, 5),
            (7, 5, 5),
            (7, 5, 3),  # 3-dim Tensors
            (7, 5, 3, 5),
            (7, 5, 5, 5),
            (7, 5, 5, 3),
        ]  # 4-dim Tensors
        for tensor_dims, some in itertools.product(tensor_dims_list, [True, False]):
            run_test(tensor_dims, some)

    @skipCUDAIfNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_qr_vs_numpy(self, device, dtype):
        """
        test torch.linalg.qr vs numpy.linalg.qr
        """
        sizes_to_test = [
            (7, 5),
            (5, 7),
            (5, 0),  # empty
            (0, 5),  # empty
        ]
        for size in sizes_to_test:
            t = torch.randn(size, device=device, dtype=dtype)
            np_t = t.cpu().numpy()
            for mode in ["reduced", "complete"]:
                exp_q, exp_r = np.linalg.qr(np_t, mode=mode)
                q, r = torch.linalg.qr(t, mode=mode)
                self.assertEqual(q, exp_q)
                self.assertEqual(r, exp_r)
            #
            # for mode='r' we need a special logic because numpy returns only r
            exp_r = np.linalg.qr(np_t, mode="r")
            q, r = torch.linalg.qr(t, mode="r")
            # check that q is empty
            self.assertEqual(q.shape, (0,))
            self.assertEqual(q.dtype, t.dtype)
            self.assertEqual(q.device, t.device)
            # check r
            self.assertEqual(r, exp_r)

    @skipCUDAIfNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.float)
    def test_linalg_qr_autograd(self, device, dtype):
        # Check differentiability for modes as specified in the docs.
        # Differentiability in all cases is only guaranteed if first k = min(m, n) columns are linearly independent.
        # Mode 'reduced' is always differentiable.
        # Mode 'r' is never differentiable.
        # Mode 'complete' is differentiable for m <= n.
        for mode in "complete", "reduced", "r":
            for m, n in [(5, 7), (7, 5)]:
                # Random matrix inputs will effectively satisfy rank requirement of k = min(m, n) columns linearly
                # independent.
                inp = torch.randn(
                    (m, n), device=device, dtype=dtype, requires_grad=True
                )
                q, r = torch.linalg.qr(inp, mode=mode)
                b = torch.sum(r)
                if mode == "complete" and m > n:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "The QR decomposition is not differentiable when mode='complete' and "
                        "nrows > ncols",
                    ):
                        b.backward()
                elif mode == "r":
                    # torch.linalg.qr(mode='r') returns only 'r' and discards 'q', but
                    # without 'q' you cannot compute the backward pass. Check that
                    # linalg_qr_backward complains cleanly in that case.
                    self.assertEqual(q.shape, (0,))  # empty tensor
                    with self.assertRaisesRegex(
                        RuntimeError, "The derivative of linalg.qr depends on Q"
                    ):
                        b.backward()
                else:
                    b.backward()

    @skipCUDAIfNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_qr_batched(self, device, dtype):
        """
        test torch.linalg.qr vs numpy.linalg.qr. We need some special logic
        because numpy does not support batched qr
        """

        def np_qr_batched(a, mode):
            """poor's man batched version of np.linalg.qr"""
            all_q = []
            all_r = []
            for matrix in a:
                result = np.linalg.qr(matrix, mode=mode)
                if mode == "r":
                    all_r.append(result)
                else:
                    q, r = result
                    all_q.append(q)
                    all_r.append(r)
            if mode == "r":
                return np.array(all_r)
            else:
                return np.array(all_q), np.array(all_r)

        t = torch.randn((3, 7, 5), device=device, dtype=dtype)
        np_t = t.cpu().numpy()
        for mode in ["reduced", "complete"]:
            exp_q, exp_r = np_qr_batched(np_t, mode=mode)
            q, r = torch.linalg.qr(t, mode=mode)
            self.assertEqual(q, exp_q)
            self.assertEqual(r, exp_r)
        # for mode='r' we need a special logic because numpy returns only r
        exp_r = np_qr_batched(np_t, mode="r")
        q, r = torch.linalg.qr(t, mode="r")
        # check that q is empty
        self.assertEqual(q.shape, (0,))
        self.assertEqual(q.dtype, t.dtype)
        self.assertEqual(q.device, t.device)
        # check r
        self.assertEqual(r, exp_r)

    @skipCUDAIfNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.float)
    def test_qr_error_cases(self, device, dtype):
        t1 = torch.randn(5, device=device, dtype=dtype)
        with self.assertRaisesRegex(
            RuntimeError,
            "linalg.qr: The input tensor A must have at least 2 dimensions.",
        ):
            torch.linalg.qr(t1)
        t2 = torch.randn((5, 7), device=device, dtype=dtype)
        with self.assertRaisesRegex(
            RuntimeError, "qr received unrecognized mode 'hello'"
        ):
            torch.linalg.qr(t2, mode="hello")

    def _check_einsum(self, *args, np_args=None):
        if np_args is None:
            np_args = [
                arg.cpu().numpy() if isinstance(arg, torch.Tensor) else arg
                for arg in args
            ]
        ref = np.einsum(*np_args)
        res = torch.einsum(*args)
        self.assertEqual(ref, res)

        # Check that the other variations for opt_einsum work too
        if TEST_OPT_EINSUM:
            with opt_einsum.flags(enabled=False):
                res = torch.einsum(*args)
                self.assertEqual(ref, res)

            with opt_einsum.flags(enabled=True, strategy="greedy"):
                res = torch.einsum(*args)
                self.assertEqual(ref, res)

            with opt_einsum.flags(enabled=True, strategy="optimal"):
                res = torch.einsum(*args)
                self.assertEqual(ref, res)

    @dtypes(torch.double, torch.cdouble)
    def test_einsum(self, device, dtype):
        # Test cases from https://gist.github.com/rockt/15ee013889d65342088e9260a377dc8f
        x = make_tensor((5,), dtype=dtype, device=device)
        y = make_tensor((7,), dtype=dtype, device=device)
        A = make_tensor((3, 5), dtype=dtype, device=device)
        B = make_tensor((2, 5), dtype=dtype, device=device)
        C = make_tensor((2, 3, 5), dtype=dtype, device=device)
        D = make_tensor((2, 5, 7), dtype=dtype, device=device)
        E = make_tensor((7, 9), dtype=dtype, device=device)
        F = make_tensor((2, 3, 3, 5), dtype=dtype, device=device)
        G = make_tensor((5, 4, 6), dtype=dtype, device=device)
        H = make_tensor((4, 4), dtype=dtype, device=device)
        I = make_tensor((2, 3, 2), dtype=dtype, device=device)

        # Vector operations
        self._check_einsum("i->", x)  # sum
        self._check_einsum("i,i->", x, x)  # dot
        self._check_einsum("i,i->i", x, x)  # vector element-wisem mul
        self._check_einsum("i,j->ij", x, y)  # outer

        # Matrix operations
        self._check_einsum("ij->ji", A)  # transpose
        self._check_einsum("ij->j", A)  # row sum
        self._check_einsum("ij->i", A)  # col sum
        self._check_einsum("ij,ij->ij", A, A)  # matrix element-wise mul
        self._check_einsum("ij,j->i", A, x)  # matrix vector multiplication
        self._check_einsum("ij,kj->ik", A, B)  # matmul
        self._check_einsum("ij,ab->ijab", A, E)  # matrix outer product

        # Tensor operations
        self._check_einsum("Aij,Ajk->Aik", C, D)  # batch matmul
        self._check_einsum("ijk,jk->i", C, A)  # tensor matrix contraction
        self._check_einsum("aij,jk->aik", D, E)  # tensor matrix contraction
        self._check_einsum("abCd,dFg->abCFg", F, G)  # tensor tensor contraction
        self._check_einsum(
            "ijk,jk->ik", C, A
        )  # tensor matrix contraction with double indices
        self._check_einsum(
            "ijk,jk->ij", C, A
        )  # tensor matrix contraction with double indices
        self._check_einsum("ijk,ik->j", C, B)  # non contiguous
        self._check_einsum("ijk,ik->jk", C, B)  # non contiguous with double indices

        # Test diagonals
        self._check_einsum("ii", H)  # trace
        self._check_einsum("ii->i", H)  # diagonal
        self._check_einsum("iji->j", I)  # non-contiguous trace
        self._check_einsum(
            "ngrg...->nrg...", make_tensor((2, 1, 3, 1, 4), dtype=dtype, device=device)
        )

        # Test ellipsis
        self._check_einsum("i...->...", H)
        self._check_einsum("ki,...k->i...", A.t(), B)
        self._check_einsum("k...,jk->...", A.t(), B)
        self._check_einsum("...ik, ...j -> ...ij", C, x)
        self._check_einsum(
            "Bik,k...j->i...j", C, make_tensor((5, 3), dtype=dtype, device=device)
        )
        self._check_einsum(
            "i...j, ij... -> ...ij",
            C,
            make_tensor((2, 5, 2, 3), dtype=dtype, device=device),
        )

        # torch.bilinear with noncontiguous tensors
        l = make_tensor((5, 10), dtype=dtype, device=device, noncontiguous=True)
        r = make_tensor((5, 20), dtype=dtype, device=device, noncontiguous=True)
        w = make_tensor((15, 10, 20), dtype=dtype, device=device)
        self._check_einsum("bn,anm,bm->ba", l, w, r)

        # with strided tensors
        self._check_einsum("bn,Anm,bm->bA", l[:, ::2], w[:, ::2, ::2], r[:, ::2])

        # test multiple inputs
        self._check_einsum("...,be,b...,beg,gi,bc...->bi...", A, B, C, D, E, F)

    @dtypes(torch.double, torch.cdouble)
    def test_einsum_sublist_format(self, device, dtype):
        x = make_tensor((5,), dtype=dtype, device=device)
        y = make_tensor((7,), dtype=dtype, device=device)
        A = make_tensor((3, 5), dtype=dtype, device=device)
        B = make_tensor((2, 5), dtype=dtype, device=device)
        C = make_tensor((2, 1, 3, 1, 4), dtype=dtype, device=device)

        self._check_einsum(x, [0])
        self._check_einsum(x, [0], [])
        self._check_einsum(x, [0], y, [1], [0, 1])
        self._check_einsum(A, [0, 1], [1, 0])
        self._check_einsum(A, [0, 1], x, [1], [0])
        self._check_einsum(A, [0, 1], B, [2, 1])
        self._check_einsum(A, [0, 1], B, [2, 1], [0, 2])
        self._check_einsum(C, [0, 1, 2, 1, Ellipsis], [0, 2, 1, Ellipsis])
        self._check_einsum(A.t(), [0, 1], B, [Ellipsis, 0])
        self._check_einsum(A.t(), [0, 1], B, [Ellipsis, 0], [1, Ellipsis])
        self._check_einsum(A.t(), [0, Ellipsis], B, [1, 0], [Ellipsis])

        # torch.bilinear with noncontiguous tensors
        l = make_tensor((5, 10), dtype=dtype, device=device, noncontiguous=True)
        r = make_tensor((5, 20), dtype=dtype, device=device, noncontiguous=True)
        w = make_tensor((15, 10, 20), dtype=dtype, device=device)
        self._check_einsum(l, [40, 41], w, [2, 41, 50], r, [40, 50], [40, 2])

    @dtypes(torch.double, torch.cdouble)
    def test_einsum_random(self, device, dtype):
        def convert_label(label):
            if label == ...:
                return "..."
            elif label < 26:
                return chr(ord("A") + label)
            else:
                return chr(ord("a") + label - 26)

        def convert_sublist(sublist):
            return "".join(convert_label(label) for label in sublist)

        def test(
            n=10,  # how many tests to generate
            n_labels=5,  # how many labels available
            min_ops=1,
            max_ops=4,  # min and max number of operands per test
            min_dims=1,
            max_dims=3,  # min and max number of dimensions per operand
            min_size=1,
            max_size=8,  # min and max size of each dimension
            max_out_dim=3,  # max number of dimensions for the output
            enable_diagonals=True,  # controls if labels can be repeated for diagonals
            ellipsis_prob=0.5,  # probability of including ellipsis in operand
            broadcasting_prob=0.1,
        ):  # probability of turning some dim sizes 1 for broadcasting

            all_labels = torch.arange(52)

            assert 0 <= n
            assert 0 <= n_labels < len(all_labels)
            assert 0 < min_ops <= max_ops
            assert 0 <= min_dims <= max_dims
            assert 0 <= min_size <= max_size
            assert 0 <= max_out_dim
            assert enable_diagonals or max_dims <= n_labels

            for _ in range(n):

                # Select a subset of labels for this test and give them random sizes
                possible_labels = all_labels[torch.randperm(len(all_labels))[:n_labels]]
                labels_size = torch.randint_like(all_labels, min_size, max_size + 1)
                ellipsis_shape = torch.randint(
                    min_size, max_size + 1, (max_dims - min_dims,)
                )

                operands = []
                sublists = []

                ell_size = 0
                valid_labels = set()

                # create random input operands
                for _ in range(random.randint(min_ops, max_ops)):
                    n_dim = random.randint(min_dims, max_dims)
                    labels_idx = torch.ones(len(possible_labels)).multinomial(
                        n_dim, enable_diagonals
                    )
                    labels = possible_labels[labels_idx]
                    valid_labels.update(labels.tolist())
                    shape = labels_size[labels]

                    # turn some dimensions to size 1 for testing broadcasting
                    mask = Binomial(probs=broadcasting_prob).sample((n_dim,))
                    broadcast_labels = torch.unique(labels[mask == 1])
                    shape[(labels[..., None] == broadcast_labels).any(-1)] = 1

                    labels = labels.tolist()
                    shape = shape.tolist()

                    # include ellipsis if not all dimensions were assigned a label already
                    if n_dim < max_dims and torch.rand(1) < ellipsis_prob:
                        ell_num_dim = random.randint(1, max_dims - n_dim)
                        ell_size = max(ell_size, ell_num_dim)
                        ell_shape = ellipsis_shape[-ell_num_dim:]
                        # again, turn some dimensions to size 1 for broadcasting
                        mask = Binomial(probs=broadcasting_prob).sample((ell_num_dim,))
                        ell_shape[mask == 1] = 1
                        ell_index = random.randint(0, n_dim)
                        shape[ell_index:ell_index] = ell_shape
                        labels.insert(ell_index, ...)

                    operands.append(make_tensor(shape, dtype=dtype, device=device))
                    sublists.append(labels)

                # NumPy has a bug with the sublist format so for now we compare PyTorch sublist
                # implementation against the equation format implementation of NumPy
                # see https://github.com/numpy/numpy/issues/10926
                np_operands = [op.cpu().numpy() for op in operands]

                # test equation format
                equation = ",".join(convert_sublist(l) for l in sublists)
                self._check_einsum(
                    equation, *operands, np_args=(equation, *np_operands)
                )

                # test sublist format
                args = list(itertools.chain.from_iterable(zip(operands, sublists)))
                self._check_einsum(*args, np_args=(equation, *np_operands))

                # generate an explicit output
                out_sublist = []
                num_out_labels = max(
                    0, random.randint(0, min(max_out_dim, len(valid_labels))) - ell_size
                )
                if num_out_labels > 0:
                    out_labels_idx = torch.ones(len(valid_labels)).multinomial(
                        num_out_labels
                    )
                    out_sublist = torch.tensor(list(valid_labels))[
                        out_labels_idx
                    ].tolist()
                out_sublist.insert(random.randint(0, num_out_labels), ...)

                # test equation format with explicit output
                equation += "->" + convert_sublist(out_sublist)
                self._check_einsum(
                    equation, *operands, np_args=(equation, *np_operands)
                )

                # test sublist format with explicit output
                args.append(out_sublist)
                self._check_einsum(*args, np_args=(equation, *np_operands))

        test(500)

    def test_einsum_corner_cases(self, device):
        def check(equation, *operands, expected_output):
            tensors = [
                (
                    torch.tensor(operand, device=device, dtype=torch.float32)
                    if not isinstance(operand, tuple)
                    else make_tensor(operand, dtype=torch.float32, device=device)
                )
                for operand in operands
            ]
            output = torch.einsum(equation, tensors)
            self.assertEqual(
                output,
                torch.tensor(expected_output, dtype=torch.float32, device=device),
            )

        # Test equation variations
        check(" ", 1, expected_output=1)
        check(" -> ", 1, expected_output=1)
        check(" , ", 2, 2, expected_output=4)
        check(" , , ", 2, 2, 2, expected_output=8)
        check(" , -> ", 2, 2, expected_output=4)
        check(" i ", [1], expected_output=[1])
        check(" i -> ", [1], expected_output=1)
        check(" i -> i ", [1], expected_output=[1])
        check(" i , i ", [2], [2], expected_output=4)
        check(" i , i -> i ", [2], [2], expected_output=[4])

        # Test tensors with 0 size dimensions
        check("i", [], expected_output=[])
        check(" i j -> j", [[], []], expected_output=[])
        check("ij->i", [[], []], expected_output=[0.0, 0.0])
        check(" i j k  ,  k  -> i j ", (3, 0, 6), (6,), expected_output=[[], [], []])

        # Test broadcasting
        check("i,j", [2], [1, 2], expected_output=[[2, 4]])
        check(
            "i,ij->ij",
            [1, 2],
            [[1, 2, 3], [2, 3, 4]],
            expected_output=[[1, 2, 3], [4, 6, 8]],
        )

        # Test ellipsis broadcasting
        check("...", 1, expected_output=1)
        check("...->", 1, expected_output=1)
        check("...->...", 1, expected_output=1)
        check("...", [1], expected_output=[1])
        check("...->", [1], expected_output=1)
        check("z...->z", [1], expected_output=[1])
        check("Z...->...Z", [1], expected_output=[1])
        check("...a->", [[2], [4]], expected_output=6)
        check("a...b->ab", [[[1], [2]], [[3], [4]]], expected_output=[[3], [7]])

    def test_einsum_error_cases(self, device):
        def check(*args, regex, exception=RuntimeError):
            with self.assertRaisesRegex(exception, r"einsum\(\):.*" + regex):
                torch.einsum(*args)

        x = make_tensor((2,), dtype=torch.float32, device=device)
        y = make_tensor((2, 3), dtype=torch.float32, device=device)

        check("", [], regex=r"at least one operand", exception=ValueError)
        check(
            ". ..",
            [x],
            regex=r"found \'.\' for operand 0 that is not part of any ellipsis",
        )
        check(
            "... ...",
            [x],
            regex=r"found \'.\' for operand 0 for which an ellipsis was already found",
        )
        check("1", [x], regex=r"invalid subscript given at index 0")
        check(
            ",",
            [x],
            regex=r"fewer operands were provided than specified in the equation",
        )
        check(
            "",
            [x, x],
            regex=r"more operands were provided than specified in the equation",
        )
        check(
            "",
            [x],
            regex=r"the number of subscripts in the equation \(0\) does not match the number "
            r"of dimensions \(1\) for operand 0 and no ellipsis was given",
        )
        check(
            "ai",
            [x],
            regex=r"the number of subscripts in the equation \(2\) does not match the number "
            r"of dimensions \(1\) for operand 0 and no ellipsis was given",
        )
        check(
            "ai...",
            [x],
            regex=r"the number of subscripts in the equation \(2\) is more than the number "
            r"of dimensions \(1\) for operand 0",
        )
        check(
            "a->... .",
            [x],
            regex=r"found \'.\' for output but an ellipsis \(...\) was already found",
        )
        check(
            "a->..",
            [x],
            regex=r"found \'.\' for output that is not part of any ellipsis \(...\)",
        )
        check("a->1", [x], regex=r"invalid subscript given at index 3")
        check(
            "a->aa",
            [x],
            regex=r"output subscript a appears more than once in the output",
        )
        check(
            "a->i",
            [x],
            regex=r"output subscript i does not appear in the equation for any input operand",
        )
        check(
            "aa",
            [y],
            regex=r"subscript a is repeated for operand 0 but the sizes don\'t match, 3 != 2",
        )
        check("...,...", [x, y], regex=r"does not broadcast")
        check(
            "a,a",
            [x, make_tensor((3,), dtype=torch.float32, device=device)],
            regex=r"does not broadcast",
        )
        check(
            "a, ba",
            [x, y],
            regex=r"subscript a has size 3 for operand 1 which does not broadcast with previously"
            r" seen size 2",
        )

        check(
            x, [-1], regex=r"not within the valid range \[0, 52\)", exception=ValueError
        )
        check(
            x, [52], regex=r"not within the valid range \[0, 52\)", exception=ValueError
        )

    def _gen_shape_inputs_linalg_triangular_solve(
        self, shape, dtype, device, well_conditioned=False
    ):
        make_arg = partial(make_tensor, dtype=dtype, device=device)
        make_fullrank = partial(
            make_fullrank_matrices_with_distinct_singular_values,
            dtype=dtype,
            device=device,
        )
        b, n, k = shape
        for left, uni, expand_a, tr_a, conj_a, expand_b, tr_b, conj_b in product(
            (True, False), repeat=8
        ):
            # expand means that we generate a batch of matrices with a stride of zero in the batch dimension
            if (conj_a or conj_b) and not dtype.is_complex:
                continue
            # We just expand on the batch size
            if (expand_a or expand_b) and b == 1:
                continue

            size_a = (b, n, n) if left else (b, k, k)
            size_b = (b, n, k) if not tr_b else (b, k, n)

            # If expand_a or expand_b, we'll expand them to the correct size later
            if b == 1 or expand_a:
                size_a = size_a[1:]
            if b == 1 or expand_b:
                size_b = size_b[1:]

            if well_conditioned:
                PLU = torch.linalg.lu(make_fullrank(*size_a))
                if uni:
                    # A = L from PLU
                    A = PLU[1].transpose(-2, -1).contiguous()
                else:
                    # A = U from PLU
                    A = PLU[2].contiguous()
            else:
                A = make_arg(size_a)
                A.triu_()

            diag = A.diagonal(0, -2, -1)
            if uni:
                diag.fill_(1.0)
            else:
                diag[diag.abs() < 1e-6] = 1.0

            B = make_arg(size_b)

            if tr_a:
                A.transpose_(-2, -1)
            if tr_b:
                B.transpose_(-2, -1)
            if conj_a:
                A = A.conj()
            if conj_b:
                B = B.conj()
            if expand_a:
                A = A.expand(b, *size_a)
            if expand_b:
                B = B.expand(b, n, k)
            yield A, B, left, not tr_a, uni

    def _test_linalg_solve_triangular(self, A, B, upper, left, uni):
        X = torch.linalg.solve_triangular(
            A, B, upper=upper, left=left, unitriangular=uni
        )
        if left:
            self.assertEqual(A @ X, B)
        else:
            self.assertEqual(X @ A, B)
        out = B
        # B may be expanded
        if not B.is_contiguous() and not B.transpose(-2, -1).is_contiguous():
            out = B.clone()
        torch.linalg.solve_triangular(
            A, B, upper=upper, left=left, unitriangular=uni, out=out
        )
        self.assertEqual(X, out)

    # Tolerances dictated by widest acceptable range on CPU before failure
    @dtypes(*floating_and_complex_types())
    @precisionOverride(
        {
            torch.float32: 1e-3 if TEST_WITH_ROCM else 1e-1,
            torch.float64: 1e-8,
            torch.complex64: 1e-1,
            torch.complex128: 1e-8,
        }
    )
    def test_linalg_solve_triangular(self, device, dtype):
        # This exercises the API + BLAS CPU + batched cuBLAS
        ks = (3, 1, 0)
        ns = (5, 0)
        bs = (1, 2, 0)

        gen_inputs = self._gen_shape_inputs_linalg_triangular_solve
        for b, n, k in product(bs, ns, ks):
            for A, B, left, upper, uni in gen_inputs(
                (b, n, k), dtype, device, well_conditioned=True
            ):
                self._test_linalg_solve_triangular(A, B, upper, left, uni)

    @slowTest
    @unittest.skipIf(
        IS_FBCODE or IS_SANDCASTLE,
        "Test fails for float64 on GPU (P100, V100) on Meta infra",
    )
    @onlyCUDA
    @skipCUDAIfNoMagma  # Magma needed for the PLU decomposition
    @dtypes(*floating_and_complex_types())
    @precisionOverride(
        {
            torch.float32: 1e-2,
            torch.complex64: 1e-2,
            torch.float64: 1e-8,
            torch.complex128: 1e-8,
        }
    )
    def test_linalg_solve_triangular_large(self, device, dtype):
        # Exercises magma and cublas
        magma = (9, 513, 1)
        iterative_cublas = (2, 64, 1)

        gen_inputs = self._gen_shape_inputs_linalg_triangular_solve
        for shape in (magma, iterative_cublas):
            for A, B, left, upper, uni in gen_inputs(
                shape, dtype, device, well_conditioned=True
            ):
                self._test_linalg_solve_triangular(A, B, upper, left, uni)

    @dtypes(*floating_and_complex_types())
    @precisionOverride(
        {
            torch.float32: 1e-2,
            torch.complex64: 1e-2,
            torch.float64: 1e-8,
            torch.complex128: 1e-8,
        }
    )
    def test_linalg_solve_triangular_broadcasting(self, device, dtype):
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        sizes = (
            ((2, 1, 3, 4, 4), (2, 1, 3, 4, 6)),
            ((2, 1, 3, 4, 4), (4, 6)),
            ((4, 4), (2, 1, 3, 4, 2)),
            ((1, 3, 1, 4, 4), (2, 1, 3, 4, 5)),
        )
        for size_A, size_B in sizes:
            for left, upper, uni in itertools.product([True, False], repeat=3):
                A = make_arg(size_A)
                if upper:
                    A.triu_()
                else:
                    A.tril_()
                diag = A.diagonal(0, -2, -1)
                if uni:
                    diag.fill_(1.0)
                else:
                    diag[diag.abs() < 1e-6] = 1.0
                B = make_arg(size_B)
                if not left:
                    B.transpose_(-2, -1)

                X = torch.linalg.solve_triangular(
                    A, B, upper=upper, left=left, unitriangular=uni
                )
                if left:
                    B_other = A @ X
                else:
                    B_other = X @ A

                self.assertEqual(*torch.broadcast_tensors(B, B_other))

    def triangular_solve_test_helper(
        self, A_dims, b_dims, upper, unitriangular, device, dtype
    ):
        triangle_function = torch.triu if upper else torch.tril
        b = torch.randn(*b_dims, dtype=dtype, device=device)
        A = torch.randn(*A_dims, dtype=dtype, device=device)
        # create positive definite matrix
        A = torch.matmul(A, A.mT)
        A_triangular = triangle_function(A)
        if unitriangular:
            A_triangular.diagonal(dim1=-2, dim2=-1).fill_(1.0)
        return b, A_triangular

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @skipIfTorchDynamo("flaky, needs investigation")
    @dtypes(*floating_and_complex_types())
    @precisionOverride(
        {
            torch.float32: 1e-3,
            torch.complex64: 1e-3,
            torch.float64: 1e-8,
            torch.complex128: 1e-8,
        }
    )
    def test_triangular_solve(self, device, dtype):
        ks = [0, 1, 3]
        ns = [0, 5]
        for k, n, (upper, unitriangular, transpose) in itertools.product(
            ks, ns, itertools.product([True, False], repeat=3)
        ):
            b, A = self.triangular_solve_test_helper(
                (n, n), (n, k), upper, unitriangular, device, dtype
            )
            x = torch.triangular_solve(
                b, A, upper=upper, unitriangular=unitriangular, transpose=transpose
            )[0]
            if transpose:
                self.assertEqual(b, np.matmul(A.t().cpu(), x.cpu()))
            else:
                self.assertEqual(b, np.matmul(A.cpu(), x.cpu()))

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    @precisionOverride(
        {
            torch.float32: 1e-3,
            torch.complex64: 1e-3,
            torch.float64: 1e-8,
            torch.complex128: 1e-8,
        }
    )
    def test_triangular_solve_batched(self, device, dtype):
        def triangular_solve_batch_helper(
            A_dims, b_dims, upper, unitriangular, transpose
        ):
            b, A = self.triangular_solve_test_helper(
                A_dims, b_dims, upper, unitriangular, device, dtype
            )
            x_exp_list = []
            for i in range(b_dims[0]):
                x_exp_list.append(
                    torch.triangular_solve(
                        b[i],
                        A[i],
                        upper=upper,
                        unitriangular=unitriangular,
                        transpose=transpose,
                    )[0]
                )
            x_exp = torch.stack(x_exp_list)  # Stacked output
            x_act = torch.triangular_solve(
                b, A, upper=upper, unitriangular=unitriangular, transpose=transpose
            )[
                0
            ]  # Actual output
            self.assertEqual(x_act, x_exp)  # Equality check
            if transpose:
                A = A.mT

            Ax = np.matmul(A.cpu(), x_act.cpu())
            self.assertEqual(b, Ax)

        def triangular_solve_zero_batch_helper(
            A_dims, b_dims, upper, unitriangular, transpose
        ):
            b, A = self.triangular_solve_test_helper(
                A_dims, b_dims, upper, unitriangular, device, dtype
            )
            x = torch.triangular_solve(
                b, A, upper=upper, unitriangular=unitriangular, transpose=transpose
            )[0]
            self.assertTrue(x.shape == b.shape)

        for upper, unitriangular, transpose in itertools.product(
            [True, False], repeat=3
        ):
            batchsize = 3
            triangular_solve_batch_helper(
                (batchsize, 5, 5), (batchsize, 5, 10), upper, unitriangular, transpose
            )

            # test empty input
            triangular_solve_batch_helper(
                (batchsize, 0, 0), (batchsize, 0, 10), upper, unitriangular, transpose
            )
            triangular_solve_batch_helper(
                (batchsize, 0, 0), (batchsize, 0, 0), upper, unitriangular, transpose
            )

            # test zero batch case
            batchsize = 0
            triangular_solve_zero_batch_helper(
                (batchsize, 5, 5), (batchsize, 5, 10), upper, unitriangular, transpose
            )

    @slowTest
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride(
        {
            torch.float32: 1e-3,
            torch.complex64: 1e-3,
            torch.float64: 1e-8,
            torch.complex128: 1e-8,
        }
    )
    def test_triangular_solve_batched_many_batches(self, device, dtype):
        for upper, transpose, unitriangular in itertools.product(
            [True, False], repeat=3
        ):
            # test batched A case
            b, A = self.triangular_solve_test_helper(
                (256, 256, 5, 5), (5, 1), upper, unitriangular, device, dtype
            )
            x, _ = torch.triangular_solve(
                b, A, upper=upper, transpose=transpose, unitriangular=unitriangular
            )
            if transpose:
                A = A.mT

            Ax = torch.matmul(A, x)

            rtol = 1e-2 if dtype in [torch.float32, torch.complex64] else self.precision
            self.assertEqual(Ax, b.expand_as(Ax), atol=self.precision, rtol=rtol)

            # test batched b case
            b, A = self.triangular_solve_test_helper(
                (3, 3), (512, 512, 3, 1), upper, unitriangular, device, dtype
            )
            x, _ = torch.triangular_solve(
                b, A, upper=upper, transpose=transpose, unitriangular=unitriangular
            )
            if transpose:
                A = A.mT

            self.assertEqual(torch.matmul(A, x), b)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @skipIfTorchDynamo("flaky, needs investigation")
    @dtypes(*floating_and_complex_types())
    def test_triangular_solve_batched_broadcasting(self, device, dtype):
        from scipy.linalg import solve_triangular as tri_solve

        def scipy_tri_solve_batched(A, B, upper, trans, diag):
            batch_dims_A, batch_dims_B = A.shape[:-2], B.shape[:-2]
            single_dim_A, single_dim_B = A.shape[-2:], B.shape[-2:]
            expand_dims = tuple(
                torch._C._infer_size(torch.Size(batch_dims_A), torch.Size(batch_dims_B))
            )
            expand_A = np.broadcast_to(A, expand_dims + single_dim_A)
            expand_B = np.broadcast_to(B, expand_dims + single_dim_B)
            flat_A = expand_A.reshape((-1,) + single_dim_A)
            flat_B = expand_B.reshape((-1,) + single_dim_B)
            flat_X = np.vstack(
                [
                    tri_solve(
                        a, b, lower=(not upper), trans=int(trans), unit_diagonal=diag
                    )
                    for a, b in zip(flat_A, flat_B)
                ]
            )
            return flat_X.reshape(expand_B.shape)

        def run_test(A_dims, b_dims, device, upper, transpose, unitriangular):
            b, A = self.triangular_solve_test_helper(
                A_dims, b_dims, upper, unitriangular, device, dtype
            )
            x_exp = torch.as_tensor(
                scipy_tri_solve_batched(
                    A.cpu().numpy(), b.cpu().numpy(), upper, transpose, unitriangular
                )
            )
            x = torch.triangular_solve(
                b, A, upper=upper, transpose=transpose, unitriangular=unitriangular
            )[0]

            self.assertEqual(x, x_exp.to(device))

        for upper, transpose, unitriangular in itertools.product(
            [True, False], repeat=3
        ):
            # test against scipy.linalg.solve_triangular
            run_test(
                (2, 1, 3, 4, 4),
                (2, 1, 3, 4, 6),
                device,
                upper,
                transpose,
                unitriangular,
            )  # no broadcasting
            run_test(
                (2, 1, 3, 4, 4), (4, 6), device, upper, transpose, unitriangular
            )  # broadcasting b
            run_test(
                (4, 4), (2, 1, 3, 4, 2), device, upper, transpose, unitriangular
            )  # broadcasting A
            run_test(
                (1, 3, 1, 4, 4),
                (2, 1, 3, 4, 5),
                device,
                upper,
                transpose,
                unitriangular,
            )  # broadcasting A & b

    @onlyCUDA
    @dtypes(torch.float)
    def test_triangular_solve_large(self, device, dtype):
        # Repro for https://github.com/pytorch/pytorch/issues/79191
        A = torch.randn(1, 2, 2, device=device, dtype=dtype).tril_()
        B = torch.randn(1, 2, 524281, device=device, dtype=dtype)
        X = torch.linalg.solve_triangular(A, B, upper=False)
        self.assertEqual(A @ X, B)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_triangular_solve_out_errors_and_warnings(self, device, dtype):
        # dtypes should be safely castable
        a = torch.eye(2, dtype=dtype, device=device)
        b = torch.randn(2, 1, dtype=dtype, device=device)
        out = torch.empty_like(b).to(torch.int)
        clone_a = torch.empty_like(a)
        with self.assertRaisesRegex(RuntimeError, "Expected out tensor to have dtype"):
            torch.triangular_solve(b, a, out=(out, clone_a))

        out = torch.empty_like(b)
        clone_a = clone_a.to(torch.int)
        with self.assertRaisesRegex(RuntimeError, "Expected out tensor to have dtype"):
            torch.triangular_solve(b, a, out=(out, clone_a))

        # device should match
        if torch.cuda.is_available():
            wrong_device = "cpu" if self.device_type != "cpu" else "cuda"
            out = torch.empty(0, dtype=dtype, device=wrong_device)
            clone_a = torch.empty_like(a)
            with self.assertRaisesRegex(
                RuntimeError, "tensors to be on the same device"
            ):
                torch.triangular_solve(b, a, out=(out, clone_a))
            out = torch.empty(0, dtype=dtype, device=device)
            clone_a = torch.empty_like(a).to(wrong_device)
            with self.assertRaisesRegex(
                RuntimeError, "tensors to be on the same device"
            ):
                torch.triangular_solve(b, a, out=(out, clone_a))

        # Trigger the WARN_ONCE deprecation error
        torch.triangular_solve(b, a)

        # if out tensor with wrong shape is passed a warning is given
        with warnings.catch_warnings(record=True) as w:
            out = torch.empty(1, dtype=dtype, device=device)
            clone_a = torch.empty(1, dtype=dtype, device=device)
            # Trigger warning
            torch.triangular_solve(b, a, out=(out, clone_a))
            # Check warning occurs
            self.assertEqual(len(w), 2)
            self.assertTrue(
                "An output with one or more elements was resized" in str(w[0].message)
            )
            self.assertTrue(
                "An output with one or more elements was resized" in str(w[1].message)
            )

    def check_single_matmul(self, x, y):

        def assertEqual(answer, expected):
            if x.dtype.is_floating_point or x.dtype.is_complex:
                k = max(x.shape[-1], 1)  # Scale the atol with the size of the matrix
                self.assertEqual(
                    answer,
                    expected,
                    msg=f"{x.shape} x {y.shape} = {answer.shape}",
                    atol=k * 5e-5,
                    rtol=1e-4,
                )
            else:
                self.assertEqual(
                    answer, expected, msg=f"{x.shape} x {y.shape} = {answer.shape}"
                )

        # test x @ y
        expected = np.matmul(x.cpu(), y.cpu())
        ans = torch.matmul(x, y)
        self.assertTrue(ans.is_contiguous())
        assertEqual(ans, expected)

        # test out
        out = torch.empty_like(ans)
        ans = torch.matmul(x, y, out=out)
        self.assertIs(ans, out)
        self.assertTrue(ans.is_contiguous())
        assertEqual(ans, expected)

    def gen_sizes_matmul(self, x_dim, y_dim=4, matrix_size=4, batch_size=3):
        """
        Generates sequences of tuples (x, y) of with size(x) = x_dim and
        size(y) <= y_dim that are compatible wrt. matmul
        """
        assert x_dim >= 1
        assert y_dim >= 2
        x = x_dim
        for y in range(1, y_dim + 1):
            for batch, mn in product(
                product(range(batch_size), repeat=max(x - 2, y - 2, 0)),
                product(range(matrix_size), repeat=min(y, 2)),
            ):
                if x == 1:
                    size_x = mn[:1]
                    size_y = batch + mn
                    yield size_x, size_y
                else:
                    for k in range(matrix_size):
                        size_x = (k,) + mn[:1]
                        if x > 2:
                            size_x = batch[-(x - 2) :] + size_x
                        size_y = mn
                        if y > 2:
                            size_y = batch[-(y - 2) :] + size_y
                        yield size_x, size_y

    @dtypesIfCUDA(torch.float, torch.complex64)  # Integer matmul just supported on CPU
    @dtypes(torch.int64, torch.float, torch.complex64)
    @setBlasBackendsToDefaultFinally
    def test_matmul_small_brute_force_1d_Nd(self, device, dtype):
        for backend in ["cublas", "cublaslt"]:
            if torch.device(device).type == "cuda":
                torch.backends.cuda.preferred_blas_library(backend)

            make_arg = partial(make_tensor, device=device, dtype=dtype)

            for (size_x, size_y), nctg_x, nctg_y in product(
                self.gen_sizes_matmul(1), (True, False), (True, False)
            ):
                x = make_arg(size_x, noncontiguous=nctg_x)
                y = make_arg(size_y, noncontiguous=nctg_y)
                self.check_single_matmul(x, y)

    @dtypesIfCUDA(torch.float, torch.complex64)  # Integer matmul just supported on CPU
    @dtypes(torch.int64, torch.float, torch.complex64)
    @setBlasBackendsToDefaultFinally
    def test_matmul_small_brute_force_2d_Nd(self, device, dtype):
        for backend in ["cublas", "cublaslt"]:
            if torch.device(device).type == "cuda":
                torch.backends.cuda.preferred_blas_library(backend)

            make_arg = partial(make_tensor, device=device, dtype=dtype)

            for (size_x, size_y), nctg_x, nctg_y in product(
                self.gen_sizes_matmul(2), (True, False), (True, False)
            ):
                x = make_arg(size_x, noncontiguous=nctg_x)
                y = make_arg(size_y, noncontiguous=nctg_y)
                self.check_single_matmul(x, y)

    @dtypesIfCUDA(torch.float, torch.complex64)  # Integer matmul just supported on CPU
    @dtypes(torch.int64, torch.float, torch.complex64)
    @setBlasBackendsToDefaultFinally
    def test_matmul_small_brute_force_3d_Nd(self, device, dtype):
        for backend in ["cublas", "cublaslt"]:
            if torch.device(device).type == "cuda":
                torch.backends.cuda.preferred_blas_library(backend)

            make_arg = partial(make_tensor, device=device, dtype=dtype)

            for (size_x, size_y), nctg_x, nctg_y in product(
                self.gen_sizes_matmul(3), (True, False), (True, False)
            ):
                x = make_arg(size_x, noncontiguous=nctg_x)
                y = make_arg(size_y, noncontiguous=nctg_y)
                self.check_single_matmul(x, y)

    @onlyCUDA
    @skipCUDAIfNotRocm  # Skipping due to SM89 OOM in CI, UT doesn't do much on NV anyways
    @dtypes(*floating_types_and(torch.half))
    @precisionOverride(
        {torch.float16: 1e-1}
    )  # TunableOp may occasionally find less precise solution
    def test_matmul_small_brute_force_tunableop(self, device, dtype):
        # disable tunableop buffer rotation for all tests everywhere, it can be slow
        # We set the TunableOp numerical check environment variable here because it is
        # possible to hit some invalid numerical solutions due to the small matrix sizes.

        with self._tunableop_ctx():
            torch.cuda.tunable.set_rotating_buffer_size(0)
            # Numerical check adds significant overhead, unsure if this is needed
            # or if there was a transient problem at the time.
            # if dtype is torch.half:
            #     os.environ["PYTORCH_TUNABLEOP_NUMERICAL_CHECK"] = "1"
            ordinal = torch.cuda.current_device()

            # set these to single iterations to keep it short but still exercise the code
            torch.cuda.tunable.set_max_tuning_duration(1)
            torch.cuda.tunable.set_max_tuning_iterations(1)

            make_arg = partial(make_tensor, device=device, dtype=dtype)
            # Using gen_sizes_matmul(2) to ensure we cover
            # 'NN', 'TN', 'TT', and 'NN' cases.
            for (size_x, size_y), nctg_x, nctg_y in product(
                self.gen_sizes_matmul(2, y_dim=3), (True, False), (True, False)
            ):
                x = make_arg(size_x, noncontiguous=nctg_x)
                y = make_arg(size_y, noncontiguous=nctg_y)
                self.check_single_matmul(x, y)

            filename1 = torch.cuda.tunable.get_filename()
            unique_id = self.id().split(".")[-1]
            filename2 = f"{filename1}_tmp1.csv"
            filename3 = f"{filename1}_tmp2.csv"
            ordinal = torch.cuda.current_device()
            assert filename1 == f"tunableop_results_{unique_id}_{ordinal}.csv"
            assert len(torch.cuda.tunable.get_results()) > 0

            assert torch.cuda.tunable.write_file()  # use default filename
            assert torch.cuda.tunable.write_file(
                filename2
            )  # use custom, one-time filename
            torch.cuda.tunable.set_filename(filename3)
            assert torch.cuda.tunable.write_file()  # use previously set filename
            assert (
                torch.cuda.tunable.read_file()
            )  # use previously set filename, will ignore duplicates and return True

            with open(filename1) as file1:
                file1_contents = file1.read()
            with open(filename2) as file2:
                file2_contents = file2.read()
            with open(filename3) as file3:
                file3_contents = file3.read()
            assert file1_contents == file2_contents
            assert file1_contents == file3_contents

            # We need to reset the filename to the default value so we can properly
            # clean up intermediate files
            self._set_tunableop_defaults()

    @onlyCUDA
    @skipCUDAIfNotRocm
    @dtypes(torch.half)
    def test_matmul_offline_tunableop(self, device, dtype):
        # Main offline tunableop test
        # NOTE: The offline tuning does not support certain tensor
        # shapes as noted below. Submatrics / matrix slices are
        # not supported at all.

        def has_any_dim_size_one(tensor: torch.Tensor):
            """Check if any dimension of a PyTorch tensor has size 1."""
            return any(dim == 1 for dim in tensor.shape)

        def is_mm_compatible(A, B):
            """Check if two matrices A and B are compatible for torch.mm."""
            return A.dim() == 2 and B.dim() == 2 and A.shape[1] == B.shape[0]

        def is_bmm_compatible(A, B):
            """Check if two 3D tensors are compatible for torch.bmm."""
            return (
                A.dim() == 3
                and B.dim() == 3
                and A.shape[0] == B.shape[0]  # Batch size must match
                and A.shape[2] == B.shape[1]  # Inner dimensions must align
            )

        with self._tunableop_ctx():
            torch.cuda.tunable.set_rotating_buffer_size(0)

            ordinal = torch.cuda.current_device()

            # record GEMM
            torch.cuda.tunable.tuning_enable(False)
            torch.cuda.tunable.record_untuned_enable(True)
            self.assertTrue(torch.cuda.tunable.record_untuned_is_enabled())

            make_arg = partial(make_tensor, device=device, dtype=dtype)
            # offline tuning only handles matmuls on two dimensionsal tensors
            # matmul that require broadcasting are
            # not supported either.
            # Below we check the different transA and transB combinations.
            for size_x, size_y in self.gen_sizes_matmul(
                x_dim=2, y_dim=2, matrix_size=4
            ):
                x = make_arg(size_x, noncontiguous=False)
                y = make_arg(size_y, noncontiguous=False)

                if is_mm_compatible(x, y):
                    self.check_single_matmul(x, y)
                else:
                    continue

                if is_mm_compatible(x.t(), y):
                    self.check_single_matmul(x.t(), y)
                else:
                    continue

                if is_mm_compatible(x, y.t()):
                    self.check_single_matmul(x, y.t())
                else:
                    continue

                if is_mm_compatible(x.t(), y.t()):
                    self.check_single_matmul(x.t(), y.t())
                else:
                    continue

            # offline tuning only handles batched matmuls on
            # three dimensionsal tensors
            # matmul that require broadcasting are
            # not supported either.
            # Below we check the different transA and transB combinations.
            for size_x, size_y in self.gen_sizes_matmul(
                x_dim=3, y_dim=3, matrix_size=4
            ):
                x = make_arg(size_x, noncontiguous=False)
                y = make_arg(size_y, noncontiguous=False)

                if has_any_dim_size_one(x) or has_any_dim_size_one(y):
                    continue

                if is_bmm_compatible(x, y):
                    self.check_single_matmul(x, y)
                else:
                    continue

                if is_bmm_compatible(x.transpose(1, 2), y):
                    self.check_single_matmul(x.transpose(1, 2), y)
                else:
                    continue

                if is_bmm_compatible(x, y.transpose(1, 2)):
                    self.check_single_matmul(x, y.transpose(1, 2))
                else:
                    continue

                if is_bmm_compatible(x.transpose(1, 2), y.transpose(1, 2)):
                    self.check_single_matmul(x.transpose(1, 2), y.transpose(1, 2))
                else:
                    continue

            self.assertTrue(torch.cuda.tunable.is_enabled())
            self.assertTrue(torch.cuda.tunable.tuning_is_enabled() is False)

            untuned_filename = get_tunableop_untuned_filename()

            # tuning the untuned GEMMs in file
            torch.cuda.tunable.tuning_enable(True)
            torch.cuda.tunable.record_untuned_enable(False)

            # set these to single iterations to keep it short but still exercise the code
            torch.cuda.tunable.set_max_tuning_duration(1)
            torch.cuda.tunable.set_max_tuning_iterations(1)

            ref_results = len(torch.cuda.tunable.get_results())
            torch.cuda.tunable.tune_gemm_in_file(untuned_filename)
            new_results = len(torch.cuda.tunable.get_results())

            self.assertGreater(new_results - ref_results, 0)
            self.assertTrue(torch.cuda.tunable.write_file())

            # Compare Param Signature of untuned and tuned results
            ok = self._compare_untuned_tuned_entries()
            self.assertTrue(ok)

    @onlyCUDA
    @skipCUDAIfNotRocm
    @runOnRocmArch(MI300_ARCH)
    @dtypes(torch.torch.float8_e4m3fnuz, torch.float8_e5m2fnuz)
    def test_scaled_gemm_offline_tunableop(self, device, dtype):
        # This test is the offline version of test_scaled_gemm_tunableop

        with self._tunableop_ctx():
            ordinal = torch.cuda.current_device()
            torch.cuda.tunable.set_rotating_buffer_size(0)

            # record GEMM
            torch.cuda.tunable.tuning_enable(False)
            torch.cuda.tunable.record_untuned_enable(True)
            self.assertTrue(torch.cuda.tunable.record_untuned_is_enabled())

            # Scaled GEMM parameters
            fillA = 0.25
            fillB = 0.75
            n = 16
            m = 32
            k = 64
            scaleA = torch.tensor(0.8, device=device)
            scaleB = torch.tensor(0.9, device=device)

            dtypeA = dtypeB = dtype
            matA = torch.full((m, k), fillA, dtype=dtypeA, device=device)
            matB = torch.full((n, k), fillB, dtype=dtypeB, device=device).t()

            # Summary of bias types that are supported:
            # - bias vector not supported when out_dtype = fp32
            # - bias_dtype allowed in PyTorch are Half or BFloat16
            # - bias_dtype in hipBLASLt restrictions can be found here:
            #   https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/api-reference.html
            fillbias = 0.10
            biasf16 = torch.full((n,), fillbias, dtype=torch.half, device=device)
            biasbf16 = torch.full((n,), fillbias, dtype=torch.bfloat16, device=device)

            # out_dtype = dtype
            torch._scaled_mm(
                matA, matB, scale_a=scaleA, scale_b=scaleB, out_dtype=dtype
            )
            # out_dtype = dtype with bias vector
            torch._scaled_mm(
                matA,
                matB,
                scale_a=scaleA,
                scale_b=scaleB,
                out_dtype=dtype,
                bias=biasf16,
            )
            # out_dtype = float32
            torch._scaled_mm(
                matA, matB, scale_a=scaleA, scale_b=scaleB, out_dtype=torch.float32
            )
            # out_dtype = bfloat16
            torch._scaled_mm(
                matA, matB, scale_a=scaleA, scale_b=scaleB, out_dtype=torch.bfloat16
            )
            # out_dtype = bfloat16 with bias vector
            torch._scaled_mm(
                matA,
                matB,
                scale_a=scaleA,
                scale_b=scaleB,
                out_dtype=torch.bfloat16,
                bias=biasbf16,
            )
            # out_dtype = float16
            torch._scaled_mm(
                matA, matB, scale_a=scaleA, scale_b=scaleB, out_dtype=torch.half
            )

            # rowwise scaling, only supported for this dtype combination
            if dtype is torch.torch.float8_e4m3fnuz:
                scaleA = torch.ones((matA.shape[0], 1), device=device)
                scaleB = torch.ones((1, matB.shape[1]), device=device)
                torch._scaled_mm(
                    matA, matB, scale_a=scaleA, scale_b=scaleB, out_dtype=torch.bfloat16
                )

            self.assertTrue(torch.cuda.tunable.is_enabled())
            self.assertTrue(torch.cuda.tunable.tuning_is_enabled() is False)

            untuned_filename = get_tunableop_untuned_filename()

            # tuning the untuned GEMMs in file
            torch.cuda.tunable.tuning_enable(True)
            torch.cuda.tunable.record_untuned_enable(False)

            # set these to single iterations to keep it short but still exercise the code
            torch.cuda.tunable.set_max_tuning_duration(1)
            torch.cuda.tunable.set_max_tuning_iterations(1)

            ref_results = len(torch.cuda.tunable.get_results())
            torch.cuda.tunable.tune_gemm_in_file(untuned_filename)
            new_results = len(torch.cuda.tunable.get_results())

            # This stores total number of cumulative results
            total_num_results = new_results - ref_results

            # Rowwise case will have an extra solution
            if dtype is torch.torch.float8_e4m3fnuz:  # rowwise
                count = 7
            else:
                count = 6
            self.assertEqual(total_num_results, count)

            self.assertTrue(torch.cuda.tunable.write_file())

            # Compare Param Signature of untuned and tuned results
            ok = self._compare_untuned_tuned_entries()
            self.assertTrue(ok)

    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    @onlyCUDA
    @skipCUDAIfNotRocm
    @dtypes(torch.float)
    def test_matmul_offline_mgpu_tunableop(self, device, dtype):
        # Offline tuning with multiple GPUs.
        # Case where you record GEMMs on one GPU, but then tune
        # on multiple GPUs
        import os

        with self._tunableop_ctx():
            # Use all available GPUs for this test
            total_gpus = torch.cuda.device_count()

            ordinal = torch.cuda.current_device()

            # Untuned filename has unique id, but results file
            # does not because it is executed in a subprocess
            untuned_filename = get_tunableop_untuned_filename()
            torch.cuda.tunable.set_filename(f"tunableop_results{ordinal}.csv")

            #  turn on untuned GEMM recording and turn off tuning
            torch.cuda.tunable.tuning_enable(False)
            torch.cuda.tunable.record_untuned_enable(True)

            # Choose matrix sizes that have not been used before
            m = n = k = 23

            # Create at least one GEMM per GPU, so when the GEMMs
            # are distributed to the GPUs there is at least one
            # GEMM per GPU.
            for g in range(1, total_gpus + 1):
                A = torch.rand(m * g, k * g, device=device, dtype=dtype)
                B = torch.rand(k * g, n * g, device=device, dtype=dtype)
                C = torch.matmul(A, B)

            # check the untuned file was written and make sure that it is not zero
            self.assertTrue(os.path.exists(untuned_filename))
            self.assertGreater(os.path.getsize(untuned_filename), 0)

            # Perform multi-GPU tuning
            torch.cuda.tunable.mgpu_tune_gemm_in_file(untuned_filename, total_gpus)

            # check the results files where written, one per gpu
            # Check that the results file is not empty and store
            # that in a local variable for the next loop.
            for i in range(total_gpus):
                result_filename = f"tunableop_results{i}.csv"
                self.assertTrue(os.path.exists(result_filename))
                self.assertGreater(os.path.getsize(result_filename), 0)
                if i == 0:  # Store for next loop
                    result_size = os.path.getsize(result_filename)

            # Check the full results files was written, one per gpu
            # check that the size of the full results file for
            # GPU 0 is greater than that of the individual results
            # for GPU 0.
            # Lastly, check that all tunableop_results_full{i} have
            # the same size as tunableop_results_full0.
            for i in range(total_gpus):
                result_full_filename = f"tunableop_results_full{i}.csv"
                self.assertTrue(os.path.exists(result_full_filename))
                if i == 0:  # Store for next subsequent iterations
                    result_full_size = os.path.getsize(result_full_filename)
                    self.assertGreater(result_full_size, result_size)
                self.assertEqual(
                    os.path.getsize(result_full_filename), result_full_size
                )

    @onlyCUDA
    @dtypes(torch.float)
    def test_rotating_buffer_tunableop(self, device, dtype):
        # Test the TunableOp rotating buffer API
        # Test the default value, will return the l2_cache_size
        self._set_tunableop_defaults()
        l2_cache_size = torch.cuda.tunable.get_rotating_buffer_size()
        self.assertGreater(l2_cache_size, 0)
        # Test zero
        torch.cuda.tunable.set_rotating_buffer_size(0)
        self.assertEqual(torch.cuda.tunable.get_rotating_buffer_size(), 0)
        # Test one MB
        torch.cuda.tunable.set_rotating_buffer_size(1)
        self.assertEqual(torch.cuda.tunable.get_rotating_buffer_size(), 1024 * 1024)
        # Test negative value, which will return the l2 cache size
        torch.cuda.tunable.set_rotating_buffer_size(-1)
        self.assertEqual(torch.cuda.tunable.get_rotating_buffer_size(), l2_cache_size)

    @onlyCUDA
    @skipCUDAIfNotRocm
    @dtypes(torch.float)
    def test_bmm_tunableop_rocm(self, device, dtype):
        # buffer rotation (on by default) with strided batched gemm tunableop was causing a mem fault
        with self._tunableop_ctx():
            torch.cuda.tunable.set_max_tuning_iterations(10)
            # Make sure the rotating buffer is not zero, otherwise this test does nothing useful.
            rotating_buffer = torch.cuda.tunable.get_rotating_buffer_size()
            self.assertGreater(rotating_buffer, 0)
            # the following 3 cases cover all previous failure cases and are here to catch regressions
            B = 16
            N = M = K = 256
            dtype = torch.bfloat16
            device = torch.device("cuda:0")
            # case 1
            i1 = torch.randn((B, N, M), device=device, dtype=dtype)
            i2 = torch.randn((B, M, K), device=device, dtype=dtype)
            out = torch.bmm(i1, i2)
            # case 2
            i1 = torch.randn((B, N, M), device=device, dtype=dtype)
            i1 = torch.permute(i1, (1, 2, 0))
            i2 = torch.randn((B, M, K), device=device, dtype=dtype)
            i2 = torch.permute(i2, (1, 0, 2))
            out = torch.bmm(i1, i2)
            # case 3
            i1 = torch.randn((N, B, M), device=device, dtype=dtype)
            i1 = torch.permute(i1, (1, 0, 2))
            i2 = torch.randn((M, B, K), device=device, dtype=dtype)
            i2 = torch.permute(i2, (1, 2, 0))
            out = torch.bmm(i1, i2)
            # case 4
            input_tensor = torch.rand((1920, 1, 100), device=device, dtype=dtype)
            input_tensor = torch.as_strided(
                input_tensor, size=(1920, 1, 100), stride=(100, 100, 1)
            )
            batch1_tensor = torch.rand((1920, 256, 512), device=device, dtype=dtype)
            batch1_tensor = torch.as_strided(
                batch1_tensor, size=(1920, 256, 512), stride=(512, 983040, 1)
            )
            batch2_tensor = torch.rand((1920, 512, 100), device=device, dtype=dtype)
            batch2_tensor = torch.as_strided(
                batch2_tensor, size=(1920, 512, 100), stride=(51200, 100, 1)
            )
            out = torch.baddbmm(input_tensor, batch1_tensor, batch2_tensor)
            # case 5
            q = torch.randn([16, 16, 1024, 64], device=device, dtype=dtype)
            k = torch.randn([16, 16, 1024, 64], device=device, dtype=dtype)
            q_chunks = q.split(512, dim=-2)
            k_chunks = k.split(64, dim=-2)
            C = torch.matmul(q_chunks[0], k_chunks[0])

    @onlyCUDA
    @skipCUDAIfNotRocm
    @dtypes(torch.bfloat16)
    def test_numeric_check_leak_tunableop_rocm(self, device, dtype):
        import os
        from torch.testing._internal.common_utils import CudaMemoryLeakCheck

        # run operator first without tuning to ensure all rocm libs are loaded,
        # otherwise false positive mem leak
        B = 5
        N = M = K = 29
        device = torch.device("cuda:0")
        i1 = torch.randn((B, N, M), device=device, dtype=dtype)
        i2 = torch.randn((B, M, K), device=device, dtype=dtype)
        out = torch.bmm(i1, i2)

        with self._tunableop_ctx():
            torch.cuda.tunable.set_rotating_buffer_size(0)
            # enable tunableop numeric check via env variable.
            os.environ["PYTORCH_TUNABLEOP_NUMERICAL_CHECK"] = "1"

            ordinal = torch.cuda.current_device()

            iterations = torch.cuda.tunable.get_max_tuning_iterations()
            torch.cuda.tunable.set_max_tuning_iterations(10)
            with CudaMemoryLeakCheck(self):
                out = torch.bmm(i1, i2)
                torch.cuda.tunable.set_max_tuning_iterations(iterations)
                torch.cuda.tunable.enable(False)

    @onlyCUDA
    @skipCUDAIfNotRocm
    @dtypes(torch.float)
    def test_validator_tunableop_rocm(self, device, dtype):
        # Test that the validator on ROCM has exactly 5 lines
        # Format of the Validator is as follows:
        # Validator,PT_VERSION,X.Y.Z.
        # Validator,ROCBLAS_VERSION,X.Y,Z
        # Validator,HIPBLASLT_VERSION,X,Y.Z
        # Validator,ROCM_Version,X,Y.Z
        # Validator,GCN_ARCH_NAME,<architecture name>
        validator_num_lines = 5

        with self._tunableop_ctx():
            # set these to single iterations to keep it short but still exercise the code
            torch.cuda.tunable.set_max_tuning_iterations(1)

            N = M = K = 4
            A = torch.randn(N, K, device=device, dtype=dtype)
            B = torch.randn(K, M, device=device, dtype=dtype)
            C = torch.matmul(A, B)
            self.assertEqual(
                len(torch.cuda.tunable.get_validators()), validator_num_lines
            )

            validators = get_tunableop_validators()
            # Check for rocBLAS and hipBLASLt
            self.assertTrue("ROCBLAS_VERSION" in validators)
            # format: [major].[minor].[patch].[tweak].[commit id]
            self.assertTrue(re.match(r"^\d+[a-z0-9.]+$", validators["ROCBLAS_VERSION"]))
            self.assertTrue("HIPBLASLT_VERSION" in validators)
            self.assertTrue(
                re.match(r"^\d+-[a-z0-9]+$", validators["HIPBLASLT_VERSION"])
            )

    @onlyCUDA
    @dtypes(torch.half)
    def test_minimum_tuning_iteration_tunableop(self, device, dtype):
        # Make sure that there is at least one tuning iteration occurs
        # when the max tuning duration and max tuning iteration are set
        # to zero.
        with self._tunableop_ctx():
            # Tune a single GEMM and verify that we get a new tuning result
            torch.cuda.tunable.set_max_tuning_duration(0)
            torch.cuda.tunable.set_max_tuning_iterations(0)

            # Reference number of results
            ref_num_results = len(torch.cuda.tunable.get_results())

            N = M = K = 8
            A = torch.randn(N, K, device=device, dtype=dtype)
            B = torch.randn(K, M, device=device, dtype=dtype)
            C = torch.matmul(A, B)

            # This stores total number of cumulative results
            total_num_results = len(torch.cuda.tunable.get_results())

            # There must be a new tuning result
            self.assertEqual((total_num_results - ref_num_results), 1)

    @onlyCUDA
    @dtypes(torch.half)
    def test_matmul_check_entries_tunableop(self, device, dtype):
        # Tune a couple of matrix multiplies
        # Verify we get the correct number of results
        with self._tunableop_ctx():
            # set these to single iterations to keep it short but still exercise the code
            torch.cuda.tunable.set_max_tuning_iterations(1)

            # Reference number of results
            ref_num_results = len(torch.cuda.tunable.get_results())

            # Execute matrix multiplies. We intentionally throw in M list the same index
            # twice. The CSV file should only get unique GEMMs
            count_matmul = 4
            K = 64
            for M in [32, 64, 32]:
                for N in [32, 64]:
                    A = torch.randn(N, K, device=device, dtype=dtype)
                    B = torch.randn(K, M, device=device, dtype=dtype)
                    C = torch.matmul(A, B)

            # This stores total number of cumulative results
            total_num_results = len(torch.cuda.tunable.get_results())

            # Take the difference to calculate the number of results from
            # the this test and verify that it agrees with the number of
            # GEMMs.
            self.assertEqual((total_num_results - ref_num_results), count_matmul)

    @onlyCUDA
    @dtypes(torch.float)
    def test_disable_tuning_tunableop(self, device, dtype):
        # Test that the Python API for disabling tuning stops
        # additional tunings even when TunableOp is enabled.
        # In other words, test that:
        # PYTORCH_TUNABLEOP_ENABLED=1
        # PYTORCH_TUNABLEOP_TUNING=0
        # is no longer tuning GEMMs.
        with self._tunableop_ctx():
            # set these to single iterations to keep it short but still exercise the code
            torch.cuda.tunable.set_max_tuning_iterations(1)

            # Reference number of results
            ref_num_results = len(torch.cuda.tunable.get_results())

            # Tune one GEMMs to make sure TunableOp is enabled
            M = 11
            N = 13
            K = 17
            A = torch.randn(N, K, device=device, dtype=dtype)
            B = torch.randn(K, M, device=device, dtype=dtype)
            C = torch.matmul(A, B)

            # This stores total number of cumulative results
            total_num_results = len(torch.cuda.tunable.get_results())

            # Take the difference to calculate the number of results from
            # this test. There should be one additional tuned GEMM
            self.assertEqual((total_num_results - ref_num_results), 1)

            # New total number of results becomes new reference result
            ref_num_results = total_num_results

            # Now disable further tuning, while keeping TunableOp Enabled
            torch.cuda.tunable.tuning_enable(False)

            # Try to tune one more GEMM
            M = 11
            N = 13
            K = 18
            A = torch.randn(N, K, device=device, dtype=dtype)
            B = torch.randn(K, M, device=device, dtype=dtype)
            C = torch.matmul(A, B)

            # Take the difference to calculate the number of results from
            # this test. There should be no change in the number of results
            # since tuning is disable.
            self.assertEqual((total_num_results - ref_num_results), 0)

    @onlyCUDA
    @dtypes(torch.float)
    def test_dump_results_on_exit_tunableop(self, device, dtype):
        # Test that the TunableOp results file is created
        # and is NOT empty.
        # To test this we create a subprocess and then
        # execute a matmul from within the subprocess
        import os
        import multiprocessing as mp

        with self._tunableop_ctx():
            filename = torch.cuda.tunable.get_filename()

            # force=True needed according to:
            # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.set_start_method
            # This is because a different test in this process could have
            # already set the start method
            mp.set_start_method("spawn", force=True)

            p = mp.Process(
                target=tunableop_matmul, args=(device, dtype, filename, False)
            )
            p.start()
            p.join()

            # Make sure the results file exists and that it is not zero.
            self.assertTrue(os.path.exists(filename))
            self.assertTrue(os.path.getsize(filename) > 0)

    @onlyCUDA
    @dtypes(torch.bfloat16)
    def test_gemm_bias_tunableop(self, device, dtype):
        # Test GEMM and bias tuning
        with self._tunableop_ctx():
            # set these to single iterations to keep it short but still exercise the code
            torch.cuda.tunable.set_max_tuning_iterations(1)

            # Reference number of results
            ref_num_results = len(torch.cuda.tunable.get_results())

            m = 3
            n = 5
            k = 7
            # 'TN' case
            X = torch.rand(m, k, dtype=dtype, device=device)
            matA = torch.rand(n, k, dtype=dtype, device=device)
            bias = torch.rand(n, dtype=dtype, device=device)

            torch.nn.functional.linear(X, matA, bias)

            # 'NT' case
            X = torch.rand(k, m, dtype=dtype, device=device).t()
            matA = torch.rand(k, n, dtype=dtype, device=device).t()
            bias = torch.rand(n, dtype=dtype, device=device)

            torch.nn.functional.linear(X, matA, bias)

            # This stores total number of cumulative results
            total_num_results = len(torch.cuda.tunable.get_results())

            # There must be a new tuning result
            self.assertEqual((total_num_results - ref_num_results), 2)

    @onlyCUDA
    @skipCUDAIfNotRocm
    @dtypes(torch.bfloat16)
    def test_gemm_bias_offline_tunableop(self, device, dtype):
        # This test is the offline version of test_gemm_bias_tunableop
        ordinal = torch.cuda.current_device()

        with self._tunableop_ctx():
            torch.cuda.tunable.set_rotating_buffer_size(0)

            # record GEMM
            torch.cuda.tunable.tuning_enable(False)
            torch.cuda.tunable.record_untuned_enable(True)
            self.assertTrue(torch.cuda.tunable.record_untuned_is_enabled())

            m = 5
            n = 7
            k = 9
            # 'TN' case
            X = torch.rand(m, k, dtype=dtype, device=device)
            matA = torch.rand(n, k, dtype=dtype, device=device)
            bias = torch.rand(n, dtype=dtype, device=device)

            torch.nn.functional.linear(X, matA, bias)

            # 'NT' case
            X = torch.rand(k, m, dtype=dtype, device=device).t()
            matA = torch.rand(k, n, dtype=dtype, device=device).t()
            bias = torch.rand(n, dtype=dtype, device=device)

            torch.nn.functional.linear(X, matA, bias)
            self.assertTrue(torch.cuda.tunable.is_enabled())
            self.assertTrue(torch.cuda.tunable.tuning_is_enabled() is False)

            untuned_filename = get_tunableop_untuned_filename()

            # tuning the untuned GEMMs in file
            torch.cuda.tunable.tuning_enable(True)
            torch.cuda.tunable.record_untuned_enable(False)

            # set these to single iterations to keep it short but still exercise the code
            torch.cuda.tunable.set_max_tuning_duration(1)
            torch.cuda.tunable.set_max_tuning_iterations(1)

            ref_results = len(torch.cuda.tunable.get_results())
            torch.cuda.tunable.tune_gemm_in_file(untuned_filename)
            new_results = len(torch.cuda.tunable.get_results())

            # This stores total number of cumulative results
            total_num_results = new_results - ref_results

            # There must be a new tuning results
            self.assertEqual(total_num_results, 2)

            self.assertTrue(torch.cuda.tunable.write_file())

            # Compare Param Signature of untuned and tuned results
            ok = self._compare_untuned_tuned_entries()
            self.assertTrue(ok)

    @onlyCUDA
    @skipCUDAIfNotRocm
    @runOnRocmArch(MI300_ARCH)
    @dtypes(torch.torch.float8_e4m3fnuz, torch.float8_e5m2fnuz)
    def test_scaled_gemm_tunableop(self, device, dtype):
        # Test Scaled GEMM tuning.
        # We do not test the full set of scaled GEMM parameters, since
        # hipBLASLt does not support all combinations.
        # Here is a short list of extra parameters that are not tested
        # - amax
        # - use_fast_accum
        # - bias dtype that are different than torch.half
        #
        # Refer to test/test_matmul_cuda for support combinations that are
        # tested by PyTorch
        with self._tunableop_ctx():
            # set these to single iterations to keep it short but still exercise the code
            torch.cuda.tunable.set_rotating_buffer_size(0)
            torch.cuda.tunable.set_max_tuning_iterations(1)

            # Reference number of results
            ref_num_results = len(torch.cuda.tunable.get_results())

            # Scaled GEMM parameters
            fillA = 0.25
            fillB = 0.75
            n = 64
            m = 16
            k = 32
            scaleA = torch.tensor(0.8, device=device)
            scaleB = torch.tensor(0.9, device=device)

            dtypeA = dtypeB = dtype
            matA = torch.full((m, k), fillA, dtype=dtypeA, device=device)
            matB = torch.full((n, k), fillB, dtype=dtypeB, device=device).t()

            # Summary of bias types that are supported:
            # - bias vector not supported when out_dtype = fp32
            # - bias_dtype allowed in PyTorch are Half or BFloat16
            # - bias_dtype in hipBLASLt restrictions can be found here:
            #   https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/api-reference.html
            fillbias = 0.10
            biasf16 = torch.full((n,), fillbias, dtype=torch.half, device=device)
            biasbf16 = torch.full((n,), fillbias, dtype=torch.bfloat16, device=device)

            # out_dtype = dtype
            torch._scaled_mm(
                matA, matB, scale_a=scaleA, scale_b=scaleB, out_dtype=dtype
            )
            # out_dtype = dtype with bias vector
            torch._scaled_mm(
                matA,
                matB,
                scale_a=scaleA,
                scale_b=scaleB,
                out_dtype=dtype,
                bias=biasf16,
            )
            # out_dtype = float32
            torch._scaled_mm(
                matA, matB, scale_a=scaleA, scale_b=scaleB, out_dtype=torch.float32
            )
            # out_dtype = bfloat16
            torch._scaled_mm(
                matA, matB, scale_a=scaleA, scale_b=scaleB, out_dtype=torch.bfloat16
            )
            # out_dtype = bfloat16 with bias vector
            torch._scaled_mm(
                matA,
                matB,
                scale_a=scaleA,
                scale_b=scaleB,
                out_dtype=torch.bfloat16,
                bias=biasbf16,
            )
            # out_dtype = float16
            torch._scaled_mm(
                matA, matB, scale_a=scaleA, scale_b=scaleB, out_dtype=torch.half
            )

            # rowwise scaling, only supported for this dtype combination
            if dtype is torch.torch.float8_e4m3fnuz:
                scaleA = torch.ones((matA.shape[0], 1), device=device)
                scaleB = torch.ones((1, matB.shape[1]), device=device)
                torch._scaled_mm(
                    matA, matB, scale_a=scaleA, scale_b=scaleB, out_dtype=torch.bfloat16
                )

            # This stores total number of cumulative results
            total_num_results = len(torch.cuda.tunable.get_results())

            # Rowwise case will have an extra solution
            if dtype is torch.torch.float8_e4m3fnuz:  # rowwise
                count = 7
            else:
                count = 6
            self.assertEqual((total_num_results - ref_num_results), count)

    @onlyCUDA
    @skipCUDAIfNotRocm
    @runOnRocmArch(MI300_ARCH)
    @dtypes(torch.float)
    def test_tf32_tunableop(self, device, dtype):
        # Test TunableOp with TF32. Supported by hipblasLT on MI300+.
        # for HIP/AMDGPU, tf32 is behind a flag because the TF32 support is new
        # and only for MI300+. Eventually this flag will go away.
        tf32_ctx = self._hip_allow_tf32 if torch.version.hip else contextlib.nullcontext

        try:
            with self._tunableop_ctx(), tf32_ctx():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.cuda.tunable.set_rotating_buffer_size(0)

                # Reference number of results
                ref_num_results = len(torch.cuda.tunable.get_results())

                N = M = K = 37
                A = torch.randn(N, K, device=device, dtype=dtype)
                B = torch.randn(K, M, device=device, dtype=dtype)
                C = torch.matmul(A, B)

                # This stores total number of cumulative results
                total_num_results = len(torch.cuda.tunable.get_results())

                # There must be a new tuning result
                self.assertEqual((total_num_results - ref_num_results), 1)

                # The results must NOT be from rocBLAS
                # result can be either Default or Hipblaslt
                # Additionally, the Op Signature must be tf32
                last_result = torch.cuda.tunable.get_results()
                found_result = find_tunableop_result(
                    last_result, "GemmTunableOp_tf32_NN", "nn_37_37_37_ld_37_37_37"
                )
                self.assertTrue(found_result is not None)
                self.assertTrue("Rocblas" not in found_result)

                # Now disable TF32
                torch.backends.cuda.matmul.allow_tf32 = False

                # Update the number of reference results
                ref_num_results = total_num_results

                # Tune the same GEMM again
                C = torch.matmul(A, B)

                # This stores total number of cumulative results
                total_num_results = len(torch.cuda.tunable.get_results())

                # There must be a new tuning result
                self.assertEqual((total_num_results - ref_num_results), 1)

                # The new tuning result must be of type float
                last_result = torch.cuda.tunable.get_results()
                found_result = find_tunableop_result(
                    last_result, "GemmTunableOp_float_NN", "nn_37_37_37_ld_37_37_37"
                )
                self.assertTrue(found_result is not None)

        finally:
            # Disable TF32
            torch.backends.cuda.matmul.allow_tf32 = False

    @onlyCUDA
    @skipCUDAIfNotRocm
    @runOnRocmArch(MI300_ARCH)
    @dtypes(torch.float)
    def test_tf32_offline_tunableop(self, device, dtype):
        # This test is the offline version of test_tf32_tunableop
        import os

        # Test TunableOp with TF32. Supported by hipblasLT on MI300+.
        # for HIP/AMDGPU, tf32 is behind a flag because the TF32 support is new
