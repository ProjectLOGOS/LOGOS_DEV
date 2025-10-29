# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
# Owner(s): ["module: tests"]

import torch
import torch.utils.data
import numpy as np

import contextlib
import gc
import io
import inspect
import itertools
import math
import random
import re
import copy
import os
import tempfile
import unittest
import warnings
import types
import pickle
import textwrap
import subprocess
import weakref
import sys
import copyreg
from torch import inf, nan
from itertools import product, combinations, permutations, chain
from functools import partial
from torch import multiprocessing as mp
from torch.testing import make_tensor
from torch.testing._internal.common_optimizers import (
    optim_db, optims, _get_optim_inputs_including_global_cliquey_kwargs)

from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
    MI300_ARCH, TEST_WITH_TORCHINDUCTOR, TEST_WITH_ROCM, run_tests, IS_JETSON,
    IS_FILESYSTEM_UTF8_ENCODING,
    IS_SANDCASTLE, IS_FBCODE, IS_REMOTE_GPU, skipIfRocmArch, skipIfTorchInductor, load_tests, slowTest, slowTestIf,
    skipIfCrossRef, TEST_WITH_CROSSREF, skipIfTorchDynamo, skipRocmIfTorchInductor, set_default_dtype,
    skipCUDAMemoryLeakCheckIf, BytesIOContext,
    skipIfRocm, skipIfNoSciPy, TemporaryFileName, TemporaryDirectoryName,
    wrapDeterministicFlagAPITest, DeterministicGuard, CudaSyncGuard,
    bytes_to_scalar, parametrize, skipIfMPS, noncontiguous_like,
    AlwaysWarnTypedStorageRemoval, TEST_WITH_TORCHDYNAMO, xfailIfTorchDynamo,
    xfailIfS390X, set_warn_always_context)
from multiprocessing.reduction import ForkingPickler
from torch.testing._internal.common_device_type import (
    expectedFailureMeta,
    expectedFailureXLA,
    instantiate_device_type_tests,
    onlyCUDA, onlyCPU,
    dtypes, dtypesIfCUDA, dtypesIfCPU, deviceCountAtLeast,
    skipMeta, PYTORCH_CUDA_MEMCHECK, largeTensorTest, onlyNativeDeviceTypes, skipCUDAIfNotRocm,
    get_all_device_types, skipXLA)
import torch.backends.quantized
import torch.testing._internal.data
from torch.testing._internal.common_cuda import (
    tf32_on_and_off, TEST_CUDNN, TEST_MULTIGPU,
    _create_scaling_case, _create_scaling_models_optimizers)
from torch.testing._internal.common_mkldnn import reduced_f32_on_and_off
from torch.testing._internal.common_dtype import (
    floating_types_and, get_all_math_dtypes, all_types_and_complex_and, complex_types,
    all_types_and, floating_types, floating_and_complex_types, integral_types_and,
    get_all_qint_dtypes, all_types_complex_float8_and,
)
from torch.testing._internal.two_tensor import TwoTensor

if TEST_WITH_TORCHINDUCTOR:
    from torch._inductor.test_case import TestCase
else:
    from torch.testing._internal.common_utils import TestCase  # type: ignore[assignment]


# Protects against includes accidentally setting the default dtype
assert torch.get_default_dtype() is torch.float32

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

AMPERE_OR_ROCM = TEST_WITH_ROCM or torch.cuda.is_tf32_supported()

@contextlib.contextmanager
def torch_vital_set(value):
    stash = None
    if 'TORCH_VITAL' in os.environ:
        stash = os.environ['TORCH_VITAL']
    os.environ['TORCH_VITAL'] = value
    try:
        yield
    finally:
        if stash:
            os.environ['TORCH_VITAL'] = stash
        else:
            del os.environ['TORCH_VITAL']

# Tests Vital Signs for Torch
# FIXME: document or deprecate whatever this is
class TestBasicVitalSigns(TestCase):
    def test_basic_vitals(self):
        with torch_vital_set(''):
            self.assertFalse(torch.vitals_enabled())
        with torch_vital_set('ON'):
            self.assertTrue(torch.vitals_enabled())

    def test_basic_vitals_read_write(self):
        with torch_vital_set('ON'):
            self.assertTrue(torch.vitals_enabled())
            # This tests the code path of setting a vital
            self.assertTrue(torch.set_vital('Dataloader', 'basic_unit_test', 'TEST_VALUE_STRING'))
            self.assertIn('TEST_VALUE_STRING', torch.read_vitals())
            self.assertIn('CUDA.used', torch.read_vitals())

    def test_dataloader_vitals(self):
        with torch_vital_set('ON'):
            inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
            tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
            dataset = torch.utils.data.TensorDataset(inps, tgts)
            torch.utils.data.DataLoader(dataset, batch_size=2)
            self.assertIn('Dataloader.enabled\t\t True', torch.read_vitals())

# FIXME: document or deprecate whatever this is
class TestVitalSignsCuda(TestCase):
    @onlyCUDA
    def test_cuda_vitals_gpu_only(self, device):
        with torch_vital_set('ON'):
            self.assertIn('CUDA.used\t\t true', torch.read_vitals())


is_cuda_sm86 = torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (8, 6)

class TestTorchDeviceType(TestCase):
    exact_dtype = True

    # TODO: move all tensor creation to common ops
    def _rand_shape(self, dim, min_size, max_size):
        shape = []
        for i in range(dim):
            shape.append(random.randint(min_size, max_size))
        return tuple(shape)

    # Validates that mathematical constants are defined properly, as required by
    # the Python Array API (https://data-apis.org/array-api/latest/API_specification/constants.html)
    @onlyCPU
    def test_constants(self, device):
        self.assertIsInstance(torch.e, float)
        self.assertEqual(torch.e, math.e, atol=0, rtol=0)

        self.assertIsInstance(torch.pi, float)
        self.assertEqual(torch.pi, math.pi, atol=0, rtol=0)

        self.assertIsInstance(torch.nan, float)
        self.assertEqual(torch.nan, math.nan, equal_nan=True)

        self.assertIsInstance(torch.inf, float)
        self.assertEqual(torch.inf, math.inf)

    @onlyNativeDeviceTypes
    @dtypes(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64,
            torch.bool, torch.float32, torch.complex64, torch.float64,
            torch.complex128, torch.uint16, torch.uint32, torch.uint64)
    def test_bytes_to_scalar(self, device, dtype):
        def rand_byte():
            if dtype == torch.bool:
                return torch.randint(0, 2, ()).item()
            else:
                return torch.randint(0, 256, ()).item()

        element_size = torch._utils._element_size(dtype)

        for i in range(10):
            bytes_list = [rand_byte() for _ in range(element_size)]
            scalar = bytes_to_scalar(bytes_list, dtype, device)
            self.assertEqual(scalar.storage().untyped().tolist(), bytes_list)

    # For testing in64 support in upsample_nearest3d
    @onlyCUDA
    @largeTensorTest('56GB', device='cuda')
    @dtypes(torch.bfloat16)
    @unittest.skipIf(IS_JETSON, "Large tensor tests are too large for Jetson.")
    def test_int64_upsample3d(self, device, dtype):
        x = torch.ones((1, 256, 16, 720, 1280), dtype=dtype, device=device)
        try:
            torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    @dtypes(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64,
            torch.bool, torch.float32, torch.complex64, torch.float64,
            torch.complex128, torch.uint16, torch.uint32, torch.uint64)
    def test_storage(self, device, dtype):
        v = make_tensor((3, 5), dtype=dtype, device=device, low=-9, high=9)
        self.assertEqual(v.storage()[0], v[0][0])
        self.assertEqual(v.storage()[14], v[2][4])
        v_s = v.storage()

        for el_num in range(v.numel()):
            dim0 = el_num // v.size(1)
            dim1 = el_num % v.size(1)
            self.assertEqual(
                v_s[el_num],
                v[dim0][dim1])

        v_s_byte = v.storage().untyped()
        el_size = v.element_size()

        for el_num in range(v.numel()):
            start = el_num * el_size
            end = start + el_size
            dim0 = el_num // v.size(1)
            dim1 = el_num % v.size(1)
            self.assertEqual(
                bytes_to_scalar(v_s_byte[start:end], dtype, device),
                v[dim0][dim1])

    @onlyNativeDeviceTypes
    @dtypes(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64,
            torch.bool, torch.float32, torch.complex64, torch.float64,
            torch.complex128, torch.quint8, torch.qint8, torch.qint32,
            torch.quint4x2)
    def test_storage_setitem(self, device, dtype):
        # Skip quantized dtypes for CUDA, since they're not supported
        if torch.device(device).type == 'cuda':
            if dtype in [torch.quint8, torch.qint8, torch.qint32, torch.quint4x2]:
                return

        storage_type_name = torch.storage._dtype_to_storage_type_map()[dtype]
        if torch.device(device).type == 'cuda':
            storage_type = eval('torch.cuda.' + storage_type_name)
        else:
            storage_type = eval('torch.' + storage_type_name)

        N = 10

        s = storage_type(N)
        s[:] = 0
        l = [0] * N
        self.assertEqual(s, storage_type(l))

        for i in range(N):
            s[i] = i
            l[i] = i

        self.assertEqual(s, storage_type(l))

        l[2:7] = [1] * 5
        s[2:7] = 1
        self.assertEqual(s, storage_type(l))

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    @onlyNativeDeviceTypes
    def test_storage_use_count(self, device):
        a = torch.randn(10, device=device)
        prev_cf = torch._C._storage_Use_Count(a.untyped_storage()._cdata)
        self.assertEqual(prev_cf, 1)
        b = a.view(2, 5)
        self.assertEqual(torch._C._storage_Use_Count(b.untyped_storage()._cdata), prev_cf + 1)

    @xfailIfTorchDynamo
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_tensor_storage_type(self, device, dtype):
        a = make_tensor((10,), dtype=dtype, device=device, low=-9, high=9)

        module = torch.cuda if (torch.device(device).type == 'cuda') else torch
        expected_storage_type = getattr(module, torch.storage._dtype_to_storage_type_map()[dtype])

        self.assertEqual(a.storage_type(), expected_storage_type)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16, torch.uint16, torch.uint32, torch.uint64))
    def test_tensor_from_storage(self, device, dtype):
        a = make_tensor((4, 5, 3), dtype=dtype, device=device, low=-9, high=9)
        a_s = a.storage()
        b = torch.tensor(a_s, device=device, dtype=dtype).reshape(a.size())
        self.assertEqual(a, b)
        c = torch.tensor(a_s.untyped(), device=device, dtype=dtype).reshape(a.size())
        self.assertEqual(a, c)

        for error_dtype in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16):
            if error_dtype == dtype:
                continue
            with self.assertRaisesRegex(RuntimeError, r'Expected a Storage of type'):
                error_storage = a.to(error_dtype).storage()
                torch.tensor(error_storage, device=device, dtype=dtype)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_set_storage(self, device, dtype):
        a = make_tensor((4, 5, 3), dtype=dtype, device=device, low=-9, high=9)
        a_s = a.storage()
        b = torch.tensor([], device=device, dtype=dtype).set_(a_s).reshape(a.size())
        self.assertEqual(a, b)
        c = torch.tensor([], device=device, dtype=dtype).set_(a_s.untyped()).reshape(a.size())
        self.assertEqual(a, c)

        for error_dtype in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16):
            if error_dtype == dtype:
                continue
            with self.assertRaisesRegex(RuntimeError, r'Expected a Storage of type'):
                error_storage = a.to(error_dtype).storage()
                b = torch.tensor([], device=device, dtype=dtype).set_(error_storage)

    def _check_storage_meta(self, s, s_check):
        self.assertTrue(
            isinstance(s, (torch.UntypedStorage, torch.TypedStorage)) and
            isinstance(s_check, type(s)),
            (
                's and s_check must both be one of UntypedStorage or '
                'TypedStorage, but got'
                f' {type(s).__name__} and {type(s_check).__name__}'))

        self.assertEqual(s.device.type, 'meta')
        self.assertEqual(s.nbytes(), s_check.nbytes())
        self.assertEqual(s.size(), s_check.size())
        self.assertEqual(s.data_ptr(), 0)

        with self.assertRaisesRegex(NotImplementedError, r'Not available'):
            s[0]

        if isinstance(s, torch.TypedStorage):
            self.assertEqual(s.dtype, s_check.dtype)
            self._check_storage_meta(s.untyped(), s_check.untyped())

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_typed_storage_meta(self, device, dtype):
        args_list = [
            [],
            [0],
            [100],
            [[1, 2, 3, 4, 5, 6]],
        ]
        for args in args_list:
            s_check = torch.TypedStorage(*args, dtype=dtype, device=device)
            s = torch.TypedStorage(*args, dtype=dtype, device='meta')
            self._check_storage_meta(s, s_check)

    @onlyNativeDeviceTypes
    def test_untyped_storage_meta(self, device):
        args_list = [
            [],
            [0],
            [100],
            [[1, 2, 3, 4, 5, 6]],
        ]
        for args in args_list:
            s_check = torch.UntypedStorage(*args, device=device)
            s = torch.UntypedStorage(*args, device='meta')
            self._check_storage_meta(s, s_check)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_storage_meta_from_tensor(self, device, dtype):
        t_check = make_tensor((4, 5, 3), dtype=dtype, device=device, low=-9, high=9)
        t = t_check.to('meta')

        s_check = t_check.storage()
        s = t.storage()
        self._check_storage_meta(s, s_check)

    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_storage_meta_errors(self, device, dtype):
        s0 = torch.TypedStorage([1, 2, 3, 4], device='meta', dtype=dtype)

        with self.assertRaisesRegex(NotImplementedError, r'Cannot copy out'):
            s0.cpu()

        with self.assertRaisesRegex(RuntimeError, r'only available on CPU'):
            s0._share_fd_cpu_()

        with self.assertRaisesRegex(RuntimeError, r'only available on CPU'):
            s0._share_filename_cpu_()

        if torch.cuda.is_available():
            with self.assertRaisesRegex(NotImplementedError, r'Cannot copy out'):
                s0.cuda()

            with self.assertRaisesRegex(RuntimeError, r'only available on CUDA'):
                s0._share_cuda_()

            with self.assertRaisesRegex(TypeError, r"cannot pin 'torch.storage.UntypedStorage' only CPU memory can be pinned"):
                s0.pin_memory()

        with self.assertRaisesRegex(RuntimeError, r'only available on CPU'):
            s0.share_memory_()

        with self.assertRaisesRegex(NotImplementedError, r'Not available'):
            s0.tolist()

        with tempfile.NamedTemporaryFile() as f:
            with self.assertRaisesRegex(NotImplementedError, r'Cannot copy out'):
                s0._write_file(f, True, True, s0.element_size())

        for device in ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']:
            s1 = torch.TypedStorage([1, 2, 3, 4], device=device, dtype=dtype)

            with self.assertRaisesRegex(NotImplementedError, r'Cannot copy out'):
                s1.copy_(s0)

    @onlyCPU
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_storage_meta_ok(self, device, dtype):
        s0 = torch.TypedStorage([1, 2, 3, 4], device='meta', dtype=dtype)

        # This is OK, it changes the meta storage size without allocating
        s0.resize_(10)

    @onlyCUDA
    def test_module_share_memory(self):
        # Test fix for issue #80733
        # See https://github.com/pytorch/pytorch/issues/80733
        model = torch.nn.Linear(3, 1)
        _model_cuda = model.to('cuda')
        model.share_memory()

    @dtypes(torch.float32, torch.complex64)
    def test_deepcopy(self, device, dtype):
        from copy import deepcopy
        a = torch.randn(5, 5, dtype=dtype, device=device)
        b = torch.randn(5, 5, dtype=dtype, device=device)
        c = a.view(25)
        q = [a, [a.storage(), b.storage()], b, c]
        w = deepcopy(q)
        self.assertEqual(w[0], q[0], atol=0, rtol=0)
        self.assertEqual(w[1][0], q[1][0], atol=0, rtol=0)
        self.assertEqual(w[1][1], q[1][1], atol=0, rtol=0)
        self.assertEqual(w[1], q[1], atol=0, rtol=0)
        self.assertEqual(w[2], q[2], atol=0, rtol=0)

        # Check that deepcopy preserves sharing
        w[0].add_(1)
        for i in range(a.numel()):
            self.assertEqual(w[1][0][i], q[1][0][i] + 1)
        self.assertEqual(w[3], c + 1)
        w[2].sub_(1)
        for i in range(a.numel()):
            self.assertEqual(w[1][1][i], q[1][1][i] - 1)

        # Check that deepcopy preserves attributes
        a.foo = 3
        self.assertEqual(deepcopy(a).foo, 3)

    @dtypes(torch.float32, torch.complex64)
    def test_deepcopy_scalar(self, device, dtype):
        from copy import deepcopy
        a = torch.tensor(5, dtype=dtype, device=device)
        self.assertEqual(a.size(), deepcopy(a).size())
        self.assertEqual(a, deepcopy(a))

    def check_internal_mem_overlap(self, inplace_op, num_inputs,
                                   dtype, device,
                                   expected_failure=False):
        if isinstance(inplace_op, str):
            inplace_op = getattr(torch.Tensor, inplace_op)
        input = torch.randn(1, dtype=dtype, device=device).expand(3, 3)
        inputs = [input] + [torch.randn_like(input)
                            for i in range(num_inputs - 1)]
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, 'single memory location'):
                inplace_op(*inputs)
        else:
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, 'single memory location'):
                    inplace_op(*inputs)

    def unary_check_input_output_mem_overlap(self, data, sz, op,
                                             expected_failure=False):

        def _test(op, output, input):
            output_exp = torch.empty_like(output)
            op(input, out=output_exp)
            self.assertEqual(op(input, out=output), output_exp, msg=op.__name__)

        # output is identical to input:
        _test(op, output=data[0:sz], input=data[0:sz])
        # output and input are independent:
        _test(op, output=data[0:sz], input=data[sz:2 * sz])
        # output partially overlaps with input:
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                _test(op, data[0:sz], data[1:sz + 1])
        else:
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                    _test(op, data[0:sz], data[1:sz + 1])
        # output is transpose of input:
        length = int(math.sqrt(sz))
        input = data[:length**2].view([length, length])
        out = input.t()
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                _test(op, out, input)
        else:
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                    _test(op, out, input)

    def ternary_check_input_output_mem_overlap(self, op, device,
                                               expected_failure=False):
        sz = 9
        data = torch.randn(2 * sz, device=device)
        other1 = torch.randn(sz, device=device)
        other2 = torch.randn(sz, device=device)

        self.unary_check_input_output_mem_overlap(
            data, sz, lambda input, out:
                op(input, other1.view(input.shape), other2.view(input.shape), out=out),
            expected_failure=expected_failure)

        self.unary_check_input_output_mem_overlap(
            data, sz, lambda input, out:
                op(other1.view(input.shape), input, other2.view(input.shape), out=out),
            expected_failure=expected_failure)

        self.unary_check_input_output_mem_overlap(
            data, sz, lambda input, out:
                op(other1.view(input.shape), other2.view(input.shape), input, out=out),
            expected_failure=expected_failure)

    def _select_broadcastable_dims(self, dims_full=None):
        # select full dimensionality
        if dims_full is None:
            dims_full = []
            ndims = random.randint(1, 4)
            dims_full = [random.randint(1, 8) for _ in range(ndims)]
        else:
            ndims = len(dims_full)

        # select actual dimensions for ops:
        # larger: full ndims, individual sizes may be reduced
        # smaller: possibly reduced ndims, sizes may be reduced
        smaller_ndims = random.randint(1, ndims)
        dims_small = []
        dims_large = []
        for i in range(ndims - 1, -1, -1):
            j = random.randint(1, 3)
            if j == 1:  # no reduced singleton dimension
                ds = dims_full[i]
                dl = dims_full[i]
            elif j == 2:  # larger may have reduced singleton dimension
                ds = dims_full[i]
                dl = 1 if len(dims_small) < smaller_ndims else dims_full[i]
            elif j == 3:  # smaller may have reduced singleton dimension
                ds = 1
                dl = dims_full[i]
            dims_large = [dl] + dims_large
            if len(dims_small) < smaller_ndims:
                dims_small = [ds] + dims_small
        return (dims_small, dims_large, dims_full)

    # collected tests of ops that used scalar_check in Declarations.cwrap for
    # correctness
    def test_scalar_check(self, device):
        zero_d = torch.randn((), device=device)
        one_d = torch.randn((1,), device=device)

        # remainder
        self.assertEqual((), torch.remainder(zero_d, zero_d).shape)
        self.assertEqual((), torch.remainder(zero_d, 2).shape)
        self.assertEqual((1,), torch.remainder(zero_d, one_d).shape)
        self.assertEqual((1,), torch.remainder(one_d, zero_d).shape)

        # fmod
        self.assertEqual((), torch.fmod(zero_d, zero_d).shape)
        self.assertEqual((), torch.fmod(zero_d, 2).shape)
        self.assertEqual((1,), torch.fmod(zero_d, one_d).shape)
        self.assertEqual((1,), torch.fmod(one_d, zero_d).shape)

        # exp, cos, cosh, tan, atan, tanh, erf, erfc, reciprocal
        self.assertEqual((), torch.exp(zero_d).shape)
        self.assertEqual((), torch.cos(zero_d).shape)
        self.assertEqual((), torch.cosh(zero_d).shape)
        self.assertEqual((), torch.tan(zero_d).shape)
        self.assertEqual((), torch.atan(zero_d).shape)
        self.assertEqual((), torch.acosh(zero_d).shape)
        self.assertEqual((), torch.asinh(zero_d).shape)
        self.assertEqual((), torch.atanh(zero_d).shape)
        self.assertEqual((), torch.tanh(zero_d).shape)
        self.assertEqual((), torch.erf(zero_d).shape)
        self.assertEqual((), torch.erfc(zero_d).shape)
        self.assertEqual((), torch.reciprocal(zero_d).shape)
        self.assertEqual((1,), torch.exp(one_d).shape)
        self.assertEqual((1,), torch.cos(one_d).shape)
        self.assertEqual((1,), torch.cosh(one_d).shape)
        self.assertEqual((1,), torch.tan(one_d).shape)
        self.assertEqual((1,), torch.atan(one_d).shape)
        self.assertEqual((1,), torch.acosh(one_d).shape)
        self.assertEqual((1,), torch.asinh(one_d).shape)
        self.assertEqual((1,), torch.atanh(one_d).shape)
        self.assertEqual((1,), torch.tanh(one_d).shape)
        self.assertEqual((1,), torch.erf(one_d).shape)
        self.assertEqual((1,), torch.erfc(one_d).shape)
        self.assertEqual((1,), torch.reciprocal(one_d).shape)

        # clamp
        self.assertEqual((), torch.clamp(zero_d, min=0, max=1).shape)
        self.assertEqual((), torch.clamp(zero_d, min=0).shape)
        self.assertEqual((), torch.clamp(zero_d, max=1).shape)
        self.assertEqual((1,), torch.clamp(one_d, min=0, max=1).shape)
        self.assertEqual((1,), torch.clamp(one_d, min=0).shape)
        self.assertEqual((1,), torch.clamp(one_d, max=1).shape)

        # cumsum, cumprod, cummax, cummin
        self.assertEqual((), torch.logcumsumexp(zero_d, 0).shape)
        self.assertEqual((), torch.cumsum(zero_d, 0).shape)
        self.assertEqual((), torch.cumprod(zero_d, 0).shape)
        self.assertEqual((), torch.cummax(zero_d, 0)[0].shape)
        self.assertEqual((), torch.cummin(zero_d, 0)[0].shape)

        # sort, topk
        self.assertEqual([(), ()], [x.shape for x in torch.sort(zero_d, 0, False)])
        self.assertEqual([(), ()], [x.shape for x in torch.sort(zero_d, 0, True)])
        self.assertEqual([(), ()], [x.shape for x in torch.topk(zero_d, 1, 0, False)])
        self.assertEqual([(), ()], [x.shape for x in torch.topk(zero_d, 1, 0, True)])

        # max, min
        self.assertEqual((), torch.max(zero_d, zero_d).shape)
        self.assertEqual((1,), torch.max(one_d, zero_d).shape)
        self.assertEqual((1,), torch.max(zero_d, one_d).shape)
        self.assertEqual((), torch.min(zero_d, zero_d).shape)
        self.assertEqual((1,), torch.min(one_d, zero_d).shape)
        self.assertEqual((1,), torch.min(zero_d, one_d).shape)

        zero_d_int = torch.tensor(1, device=device)
        one_d_int = torch.tensor([1], device=device)

        # lshift, rshift
        self.assertEqual((), (zero_d_int >> zero_d_int).shape)
        self.assertEqual((), (zero_d_int >> 1).shape)
        self.assertEqual((1,), (one_d_int >> zero_d_int).shape)
        self.assertEqual((1,), (zero_d_int >> one_d_int).shape)
        self.assertEqual((1,), (one_d_int >> 1).shape)

        self.assertEqual((), (zero_d_int << zero_d_int).shape)
        self.assertEqual((), (zero_d_int << 1).shape)
        self.assertEqual((1,), (one_d_int << zero_d_int).shape)
        self.assertEqual((1,), (zero_d_int << one_d_int).shape)
        self.assertEqual((1,), (one_d_int << 1).shape)

        # or
        self.assertEqual((), (zero_d_int | zero_d_int).shape)
        self.assertEqual((), (zero_d_int | 1).shape)
        self.assertEqual((1,), (one_d_int | zero_d_int).shape)
        self.assertEqual((1,), (zero_d_int | one_d_int).shape)
        self.assertEqual((1,), (one_d_int | 1).shape)

        # and
        self.assertEqual((), (zero_d_int & zero_d_int).shape)
        self.assertEqual((), (zero_d_int & 1).shape)
        self.assertEqual((1,), (one_d_int & zero_d_int).shape)
        self.assertEqual((1,), (zero_d_int & one_d_int).shape)
        self.assertEqual((1,), (one_d_int & 1).shape)

        # clone
        self.assertEqual((), zero_d.clone().shape)

        zero_d_bool = torch.tensor(True, device=device)
        one_d_bool = torch.tensor([True], device=device)

        # masked_select
        self.assertEqual((1,), torch.masked_select(zero_d_bool, zero_d_bool).shape)
        self.assertEqual((1,), torch.masked_select(zero_d_bool, one_d_bool).shape)
        self.assertEqual((1,), torch.masked_select(one_d_bool, zero_d_bool).shape)

        torch.tensor(1, dtype=torch.uint8, device=device)
        torch.tensor([1], dtype=torch.uint8, device=device)

        # mode
        self.assertEqual([(), ()], [x.shape for x in torch.mode(zero_d, dim=0, keepdim=True)])
        self.assertEqual([(), ()], [x.shape for x in torch.mode(zero_d, dim=0, keepdim=False)])
        self.assertEqual([(1,), (1,)], [x.shape for x in torch.mode(one_d, dim=0, keepdim=True)])
        self.assertEqual([(), ()], [x.shape for x in torch.mode(one_d, dim=0, keepdim=False)])

        # max
        self.assertEqual([(), ()], [x.shape for x in torch.max(zero_d, dim=0, keepdim=True)])
        self.assertEqual([(), ()], [x.shape for x in torch.max(zero_d, dim=0, keepdim=False)])
        self.assertEqual([(1,), (1,)], [x.shape for x in torch.max(one_d, dim=0, keepdim=True)])
        self.assertEqual([(), ()], [x.shape for x in torch.max(one_d, dim=0, keepdim=False)])

        # amax
        self.assertEqual((), torch.amax(zero_d, dim=0, keepdim=True).shape)
        self.assertEqual((), torch.amax(zero_d, dim=0, keepdim=False).shape)
        self.assertEqual((1,), torch.amax(one_d, dim=0, keepdim=True).shape)
        self.assertEqual((), torch.amax(one_d, dim=0, keepdim=False).shape)

        # min
        self.assertEqual([(), ()], [x.shape for x in torch.min(zero_d, dim=0, keepdim=True)])
        self.assertEqual([(), ()], [x.shape for x in torch.min(zero_d, dim=0, keepdim=False)])
        self.assertEqual([(1,), (1,)], [x.shape for x in torch.min(one_d, dim=0, keepdim=True)])
        self.assertEqual([(), ()], [x.shape for x in torch.min(one_d, dim=0, keepdim=False)])

        # amin
        self.assertEqual((), torch.amin(zero_d, dim=0, keepdim=True).shape)
        self.assertEqual((), torch.amin(zero_d, dim=0, keepdim=False).shape)
        self.assertEqual((1,), torch.amin(one_d, dim=0, keepdim=True).shape)
        self.assertEqual((), torch.amin(one_d, dim=0, keepdim=False).shape)

        # set_
        zero_d_clone = zero_d.clone()
        one_d_clone = one_d.clone()
        self.assertEqual((), zero_d_clone.set_(one_d.storage(), 0, (), ()).shape)
        self.assertEqual((1,), zero_d_clone.set_(one_d.storage(), 0, (1,), (1,)).shape)
        self.assertEqual((), one_d_clone.set_(one_d.storage(), 0, (), ()).shape)
        self.assertEqual((1,), one_d_clone.set_(one_d.storage(), 0, (1,), (1,)).shape)

        self.assertEqual((), zero_d.clone().set_(zero_d).shape)
        self.assertEqual((), one_d.clone().set_(zero_d).shape)
        self.assertEqual((1,), zero_d.clone().set_(one_d).shape)
        self.assertEqual((1,), one_d.clone().set_(one_d).shape)

        # take
        self.assertEqual((), torch.randn((2, 3), device=device).take(zero_d_int).shape)
        self.assertEqual((1,), torch.randn((2, 3), device=device).take(one_d_int).shape)

        # gather
        self.assertEqual((), torch.gather(zero_d, 0, torch.zeros((), dtype=torch.int64, device=device)).shape)
        self.assertEqual((1,), torch.gather(zero_d, 0, torch.zeros((1,), dtype=torch.int64, device=device)).shape)
        self.assertEqual((), torch.gather(one_d, 0, torch.zeros((), dtype=torch.int64, device=device)).shape)
        self.assertEqual((1,), torch.gather(one_d, 0, torch.zeros((1,), dtype=torch.int64, device=device)).shape)

        # normal
        # std must be >= 0
        zero_d_ge_0 = torch.rand((), device=device)
        # documentation says out shape matches shape of mean
        self.assertEqual((), torch.normal(zero_d, zero_d_ge_0).shape)
        self.assertEqual((1,), torch.normal(one_d, zero_d_ge_0).shape)
        self.assertEqual((), torch.normal(1, zero_d_ge_0).shape)
        self.assertEqual((), torch.normal(zero_d, 1).shape)
        self.assertEqual((1,), torch.normal(one_d, 1).shape)
        # TODO: this behavior differs on CPU and GPU, see https://github.com/pytorch/pytorch/issues/30480.
        # self.assertEqual((), torch.normal(zero_d, one_d).shape)
        # self.assertEqual((), torch.normal(1, one_d).shape)

        # convolutions.  Yes, we are testing nn.functional here; seems justified
        # given its similar to the other tests
        w = torch.randn(2, 1, 3, 3, device=device).div_(2).requires_grad_()
        self.assertRaises(RuntimeError, lambda: torch.nn.functional.conv2d(zero_d, w, groups=1))
        self.assertRaises(RuntimeError, lambda: torch.nn.functional.conv2d(zero_d, w, groups=2))

        # nll_loss -- verify input can't be 0-dimensional.
        self.assertRaises(ValueError, lambda: torch.nn.functional.nll_loss(zero_d, zero_d, reduction='none'))
        self.assertRaises(ValueError, lambda: torch.nn.functional.nll_loss(zero_d, one_d, reduction='none'))
        # verify output is 0-dimensional when reduction != 'none'
        for (input, target) in ((torch.randn(1, 1, device=device), torch.tensor([0], device=device)),
                                (torch.randn(1, 1, 1, 1, device=device), torch.tensor([[[0]]], device=device))):
            self.assertEqual((), torch.nn.functional.nll_loss(input, target, reduction='mean').shape)
            self.assertEqual((), torch.nn.functional.nll_loss(input, target, reduction='sum').shape)

    # Test that `torch._check_tensor_all` raises errors in the correct cases
    def test_check_tensor_all(self, device):
        default_message = 'Expected cond to be True'
        check_fn = torch._check_tensor_all
        expected_error = RuntimeError

        # cond must be a tensor
        with self.assertRaisesRegex(TypeError, 'cond must be a tensor'):
            check_fn(True)

        # cond tensor must be boolean
        with self.assertRaisesRegex(TypeError, 'cond tensor must have dtype torch.bool'):
            check_fn(torch.ones(1, device=device))

        test_sizes = [
            (),
            (1,),
            (10,),
            (1, 1),
            (1, 10),
            (10, 1),
            (10, 10),
            (1, 1, 1),
            (10, 1, 1),
            (1, 10, 1),
            (10, 10, 10),
        ]
        for size in test_sizes:
            t_all_true = torch.ones(size, dtype=torch.bool, device=device)
            t_all_false = torch.zeros(size, dtype=torch.bool, device=device)

            # Should not raise error
            check_fn(t_all_true)

            with self.assertRaisesRegex(expected_error, default_message):
                check_fn(t_all_false)

            if t_all_true.numel() > 1:
                t_all_true_but_one = t_all_true.clone()
                # Choose a random element to set to false
                idx = (random.choice(range(dim_size)) for dim_size in size)
                t_all_true_but_one[(..., *idx)] = False

                with self.assertRaisesRegex(expected_error, default_message):
                    check_fn(t_all_true_but_one)

            # Test a simple failure message
            message = 'message'
            with self.assertRaisesRegex(expected_error, message):
                check_fn(t_all_false, lambda: message)

            # Test message with tensor
            def message():
                return torch.arange(4)

            with self.assertRaisesRegex(expected_error, re.escape(str(message()))):
                check_fn(t_all_false, message)

            # Test format string message
            def message():
                return f"{'test'} {[1, 2, 'a', True]} {True} {100} {torch.arange(4)}"

            with self.assertRaisesRegex(expected_error, re.escape(str(message()))):
                check_fn(t_all_false, message)

    # Test that `TORCH_CHECK_TENSOR_ALL` raises errors that propagate from C++ to Python
    def test_check_tensor_internal(self, device):
        test_sizes = [
            (),
            (1,),
            (10,),
            (1, 1),
            (1, 10),
            (10, 1),
            (10, 10),
            (1, 1, 1),
            (10, 1, 1),
            (1, 10, 1),
            (10, 10, 10),
        ]
        for size in test_sizes:
            t_all_true = torch.ones(size, dtype=torch.bool, device=device)
            t_all_false = torch.zeros(size, dtype=torch.bool, device=device)

            # Should not raise error
            torch._test_check_tensor(t_all_true)

            with self.assertRaisesRegex(RuntimeError, "Test message for TORCH_CHECK_TENSOR_ALL"):
                torch._test_check_tensor(t_all_false)

            if t_all_true.numel() > 1:
                t_all_true_but_one = t_all_true.clone()
                # Choose a random element to set to false
                idx = (random.choice(range(dim_size)) for dim_size in size)
                t_all_true_but_one[(..., *idx)] = False

                with self.assertRaisesRegex(RuntimeError, "Test message for TORCH_CHECK_TENSOR_ALL"):
                    torch._test_check_tensor(t_all_true_but_one)

    # Uses mismatched arange out size to trigger a warning
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    @unittest.skipIf(TEST_WITH_CROSSREF, "crossref perturbs line numbering")
    def test_cpp_warnings_have_python_context(self, device):
        # Creates long string in advance to avoid a too-long Python line
        s = ".+Triggered internally at.+RangeFactories.+"
        # nvfuser deprecation warning filter
        warnings.filterwarnings("ignore", "torch::jit::fuser::cuda", UserWarning)

        def cpp_warn_fn():
            out = torch.empty((5,))
            torch.arange(0, 3, out=out)
            return out

        # Checks eager-mode cpp warning
        with warnings.catch_warnings(record=True) as w:
            cpp_warn_fn()
            frameinfo = inspect.getframeinfo(inspect.currentframe())
            warning = w[0]

            # Checks for cpp context in the warning message
            escaped_warning_message = str(warning.message).encode('unicode_escape')
            self.assertTrue(re.search(s, repr(escaped_warning_message), re.IGNORECASE) is not None)

            # Checks the Python features of the warning
            # Note: the eager mode warning refers to the line in the function
            # that throws the warning.
            self.assertEqual(frameinfo.lineno - 6, warning.lineno)
            self.assertEqual(len(w), 1)

        # Checks jitted cpp warning
        with warnings.catch_warnings(record=True) as w:
            scripted_cpp_warn_fn = torch.jit.script(cpp_warn_fn)
            scripted_cpp_warn_fn()
            warning = w[0]

            # Checks for cpp context in the warning message
            escaped_warning_message = str(warning.message).encode('unicode_escape')
            self.assertTrue(re.search(s, repr(escaped_warning_message), re.IGNORECASE) is not None)

            # Checks the Python features of the warning
            # Note: the jitted warning's lineno refers to the call to the jitted
            # function, which in our test suite has a layer of indirection
            # that makes checking the Python lineno fragile
            self.assertEqual(len(w), 1)

        # Checks jitted Python warning
        def warn_fn():
            warnings.warn("Warning!")

        # The jit mimics an eager-mode Python warning in this case
        with warnings.catch_warnings(record=True) as w:
            scripted_warn_fn = torch.jit.script(warn_fn)
            scripted_warn_fn()
            frameinfo = inspect.getframeinfo(inspect.currentframe())
            warning = w[0]

            self.assertTrue(re.search('Warning!', str(warning.message)) is not None)

            # Checks the Python features of the warning
            self.assertEqual(frameinfo.lineno - 6, warning.lineno)
            self.assertEqual(len(w), 1)

    # FIXME: move to test_testing
    @onlyCPU
    def test_warn_always_caught(self, device):
        # Check that we can catch a TORCH_WARN_ONCE warning twice
        # since assertWarnsOnceRegex uses set_warn_always(True) which changes
        # TORCH_WARN_ONCE to TORCH_WARN
        a = np.arange(10)
        a.flags.writeable = False
        with self.assertWarnsOnceRegex(UserWarning, '.*non-writable.*'):
            torch.from_numpy(a)

        # OK, got it once, now try again
        with self.assertWarnsOnceRegex(UserWarning, '.*non-writable.*'):
            torch.from_numpy(a)

        # Make sure emitting two warnings will pass the assertWarnsOnceRegex
        # context manager
        with self.assertWarnsOnceRegex(UserWarning, '.*non-writable.*'):
            torch.from_numpy(a)
            torch.from_numpy(a)

    @onlyNativeDeviceTypes
    def test_complex_half_experimental_warning(self, device):
        msg = 'ComplexHalf support is experimental'
        with self.assertWarnsOnceRegex(UserWarning, msg):
            t = torch.randn(3, dtype=torch.chalf, device=device)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.rand(3, dtype=torch.chalf, device=device)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.empty(3, dtype=torch.chalf, device=device)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.ones(3, dtype=torch.chalf, device=device)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.zeros(3, dtype=torch.chalf, device=device)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.randn_like(t)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.rand_like(t)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.empty_like(t)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.ones_like(t)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.zeros_like(t)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            # t + 1 allocates a new tensor for result using empty
            t + 1

    @onlyCUDA
    def test_dtypetensor_warnings(self, device):
        msg = 'The torch.cuda.*DtypeTensor constructors are no longer recommended'
        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.cuda.FloatTensor([0])

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.cuda.DoubleTensor([0])

    def test_set_default_tensor_type_warnings(self, device):
        msg = '.*is deprecated as of PyTorch 2.1, please use torch.set_default_dtype().*'
        default_type = torch.tensor([]).type()
        try:
            with self.assertWarnsOnceRegex(UserWarning, msg):
                torch.set_default_tensor_type(torch.FloatTensor)

            if torch.cuda.is_available():
                with self.assertWarnsOnceRegex(UserWarning, msg):
                    torch.set_default_tensor_type(torch.cuda.FloatTensor)
        finally:
            torch.set_default_tensor_type(default_type)

    # TODO: this test should be in test_nn.py
    def test_conv_transposed_backward_agnostic_to_memory_format(self, device):
        in_channels = 64
        out_channels = 128
        scale_factor = 8
        batch_size = 8
        length = 16

        conv = torch.nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=scale_factor * 2, stride=scale_factor).to(device)
        layer_norm = torch.nn.LayerNorm(out_channels).to(device)

        input_ = torch.randn(batch_size, in_channels, length).to(device).contiguous()
        input_ = conv(input_).contiguous()
        input_ = layer_norm(input_.transpose(1, 2).contiguous()).contiguous()
        input_.sum().backward()

        # 3d
        conv = torch.nn.ConvTranspose3d(3, 3, kernel_size=3).to(device)
        input = torch.randn(batch_size, 3, length, length, length, device=device)
        out = conv(input)
        out.backward(torch.ones_like(out).transpose(-2, -1))

    # TODO: this test should be in test_nn.py
    @onlyCUDA
    @largeTensorTest('12GB')
    def test_conv_transposed_large(self, device):
        # ConvTranspose3d works for large input tensors (gh-32866)
        in_channels = 64
        out_channels = 128
        kernel_size = 5

        conv = torch.nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=2, padding=2, output_padding=1).to(device)

        x = torch.rand([1, 64, 8, 128, 172]).to(device)
        conv(x)

    def test_is_set_to(self, device):
        t1 = torch.empty(3, 4, 9, 10, device=device)
        t2 = torch.empty(3, 4, 9, 10, device=device)
        t3 = torch.tensor([], device=device).set_(t1)
        t4 = t3.clone().resize_(12, 90)
        self.assertFalse(t1.is_set_to(t2))
        self.assertTrue(t1.is_set_to(t3))
        self.assertTrue(t3.is_set_to(t1), "is_set_to should be symmetric")
        self.assertFalse(t1.is_set_to(t4))
        self.assertFalse(torch.tensor([]).is_set_to(torch.tensor([])),
                         "Tensors with no storages should not appear to be set "
                         "to each other")

        t1 = torch.tensor([True, True], dtype=torch.bool, device=device)
        t2 = torch.tensor([0], dtype=torch.bool, device=device).set_(t1)
        self.assertTrue(t1.is_set_to(t2))

        # test that sizes must match
        t1 = torch.empty([2, 3, 4], device=device)
        t2 = t1.view(4, 3, 2)
        self.assertFalse(t1.is_set_to(t2))
        self.assertFalse(t2.is_set_to(t1))

        # test that legacy empty size behavior used to be respected (i.e. all
        # empty tensors were logically collapsed to size [0]).
        t1 = torch.empty([2, 5, 0], device=device)
        t2 = t1.view([0])
        self.assertFalse(t1.is_set_to(t2))
        self.assertFalse(t2.is_set_to(t1))

    # See https://github.com/pytorch/pytorch/issues/72650
    @skipIfMPS
    @skipMeta
    @parametrize(
        "fn",
        [
            "dist", "atan2", "pow", "lerp", "add", "sub", "mul", "div", "fmod", "remainder", "eq", "ge", "gt", "le",
            "lt", "max", "min", "ne", "addcdiv", "addcmul", "masked_scatter", "masked_select", "masked_fill", "map",
            "map2", "copy",
        ],
    )
    def test_broadcast(self, fn, device):
        # functions with three tensor arguments
        fns_3_args = {"map2"}
        fns_value_kwarg = {"addcdiv", "addcmul"}

        (dims_small, dims_large, dims_full) = self._select_broadcastable_dims()
        full1d = torch.randn(*dims_full, device=device).flatten().float()
        small = torch.randn(*dims_small, device=device).float()
        large = torch.randn(*dims_large, device=device).float()
        small_expanded = small.expand(*dims_full)
        large_expanded = large.expand(*dims_full)
        small2 = None
        small2_expanded = None
        if fn in fns_3_args or fn in fns_value_kwarg:
            # create another smaller tensor
            (dims_small2, _, _) = self._select_broadcastable_dims(dims_full)
            small2 = torch.randn(*dims_small2, device=device).float()
            small2_expanded = small2.expand(*dims_full)

        if small.is_cuda and fn in ['map', 'map2']:
            # map and map2 are not implemented on CUDA tensors
            return

        if hasattr(large_expanded, fn):
            # run through tensor versions of functions
            # and verify fully expanded inputs give same results
            expanded = {large: large_expanded, small: small_expanded, small2: small2_expanded}

            def tensorfn(myfn, t1, t2):
                if fn == "lerp":
                    return myfn(t1, 0.5)
                elif fn == "masked_select":
                    return myfn(t1 < 0)
                elif fn == "masked_scatter":
                    return myfn(t1 < 0.5, full1d)
                elif fn == "masked_fill":
                    return myfn(t1 < 0.5, 1.0)
                elif fn in fns_3_args:
                    return myfn(1, t1, t2)
                elif fn in fns_value_kwarg:
                    return myfn(t1, t2, value=1)
                else:
                    return myfn(t1)

            # test various orders
            for first, second, third in [(large, small, small2), (small, large, small2),
                                         (small2, small, large), (small2, large, small)]:
                if first is None:
                    break  # ignore last iter when small2 is None
                method_expanded = getattr(expanded[first], fn)
                method = getattr(first, fn)
                r1 = tensorfn(method_expanded, expanded[second], expanded[third])
                r2 = tensorfn(method, second, third)
                self.assertEqual(r1, r2)

        # now for torch. versions of functions
        if hasattr(torch, fn):
            fntorch = getattr(torch, fn)
            expanded = {large: large_expanded, small: small_expanded, small2: small2_expanded}

            def torchfn(t1, t2, t3):
                if fn == "lerp":
                    return fntorch(t1, t2, 0.5)
                elif fn == "masked_select":
                    return fntorch(t1, t2 < 0)
                elif fn == "masked_scatter":
                    return fntorch(t1, t2 < 0.5, full1d)
                elif fn == "masked_fill":
                    return fntorch(t1, t2 < 0.5, 1.0)
                elif fn in fns_3_args:
                    return fntorch(t1, 1.0, t2, t3)
                elif fn in fns_value_kwarg:
                    return fntorch(t1, t2, t3, value=1.0)
                else:
                    return fntorch(t1, t2)

            # test various orders
            for first, second, third in [(large, small, small2), (small, large, small2),
                                         (small2, small, large), (small2, large, small)]:
                if first is None:
                    break  # ignore last iter when small2 is None
                r1 = torchfn(expanded[first], expanded[second], expanded[third])
                r2 = torchfn(first, second, third)
                self.assertEqual(r1, r2)

        # now for in place functions
        # in-place tensor is not broadcastable; test only guaranteed
        # to work by broadcasting other argument(s)
        if not hasattr(large_expanded, fn + "_"):
            return

        # need to clone largeExpanded so we can reuse, since functions are in-place
        large_expanded_clone = large_expanded.clone()

        def tensorfn_inplace(t0, t1, t2=None):
            t0_fn = getattr(t0, fn + "_")
            if fn == "lerp":
                return t0_fn(t1, 0.5)
            elif fn == "masked_scatter":
                return t0_fn(t1 < 0.5, full1d)
            elif fn == "masked_fill":
                return t0_fn(t1 < 0.5, 1.0)
            elif fn == "map":
                return t0_fn(t1, lambda x, y: x + y)
            elif fn == "map2":
                return t0_fn(t1, t2, lambda x, y, z: x + y + z)
            elif fn in fns_3_args:
                return t0_fn(1.0, t1, t2)
            elif fn in fns_value_kwarg:
                return t0_fn(t1, t2, value=1.0)
            else:
                return t0_fn(t1)
        # in-place pointwise operations don't actually work if the in-place
        # tensor is 0-strided (numpy has the same issue)
        if (0 not in large_expanded.stride() and 0 not in large_expanded_clone.stride()):
            r1 = tensorfn_inplace(large_expanded, small_expanded, small2_expanded)
            r2 = tensorfn_inplace(large_expanded_clone, small, small2)
            self.assertEqual(r1, r2)

        def broadcastable(t0, t1, t2=None):
            try:
                t1.expand_as(t0)
                if t2 is not None:
                    t2.expand_as(t0)
            except RuntimeError:
                return False
            return True

        def _test_in_place_broadcastable(t0, t1, t2=None):
            if not broadcastable(t0, t1, t2):
                same_size = t0.numel() == t1.numel() and (t0.numel() == t2.numel() if t2 is not None else True)
                if not same_size:
                    # Functionalization converts the inplace to an out-of-place, which causes us to error.
                    # We should fix this, but "error probably on bad inputs" isn't a hi-pri PT2 item.
                    if not TEST_WITH_TORCHINDUCTOR:
                        self.assertRaises(RuntimeError, lambda: tensorfn_inplace(t0, t1, t2))
            else:
                tensorfn_inplace(t0, t1, t2)

        if fn not in fns_3_args and fn not in fns_value_kwarg:
            _test_in_place_broadcastable(small, large_expanded)
            _test_in_place_broadcastable(small, large)
        else:
            _test_in_place_broadcastable(small2, small_expanded, large_expanded)
            _test_in_place_broadcastable(small2, small, large)

    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "cublas runtime error")
    @onlyCUDA
    @wrapDeterministicFlagAPITest
    def test_cublas_config_nondeterministic_alert(self, device):
        test_cases = [
            # (function, (tensor sizes))
            ('mm', ((2, 2), (2, 2),)),
            ('mv', ((2, 2), (2,),)),
            ('bmm', ((1, 2, 2), (1, 2, 2),))]

        test_configs = [
            # (CuBLAS workspace config, is deterministic)
            ('garbage', False),
            (None, False),
            (':4096:8', True),
            (':16:8', True)]

        cublas_var_name = 'CUBLAS_WORKSPACE_CONFIG'
        is_cuda10_2_or_higher = (torch.version.cuda is not None)

        def test_case_info(fn_name, config):
            return f'function "{fn_name}" with config "{"" if config is None else config}"'

        # Create processes to test each combination of test cases and config settings
        for fn_name, arg_sizes in test_cases:
            for config, is_config_deterministic in test_configs:
                env = os.environ.copy()
                if config is None:
                    if env.get(cublas_var_name) is not None:
                        del env[cublas_var_name]
                else:
                    env[cublas_var_name] = config
                should_throw_error = is_cuda10_2_or_higher and not is_config_deterministic
                script = f"""
import torch
torch.use_deterministic_algorithms(True)
fn = torch.{fn_name}
arg_sizes = {arg_sizes}
device = '{device}'
should_throw_error = {should_throw_error}
args = []
for arg_size in arg_sizes:
    args.append(torch.randn(*arg_size, device=device))
try:
    fn(*args)
except RuntimeError as e:
    if not should_throw_error:
        raise RuntimeError('Did not expect any error to be raised')
    elif 'Deterministic behavior was enabled with either' not in str(e):
        raise RuntimeError('Expected a CuBLAS nondeterministic error, but got a different error')
else:
    if should_throw_error:
        raise RuntimeError('Expected a CuBLAS nondeterministic error, but it was not raised')

"""
                try:
                    subprocess.check_output(
                        [sys.executable, '-c', script],
                        stderr=subprocess.STDOUT,
                        # On Windows, opening the subprocess with the default CWD makes `import torch`
                        # fail, so just set CWD to this script's directory
                        cwd=os.path.dirname(os.path.realpath(__file__)),
                        env=env)
                except subprocess.CalledProcessError as e:
                    self.fail(msg=(
                        f'Subprocess exception while attempting to run {test_case_info(fn_name, config)}:\n'
                        + e.output.decode("utf-8")))

    @onlyCPU
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    @dtypes(*get_all_qint_dtypes())
    def test_nondeterministic_resize_quantized(self, device, dtype):
        a = torch.tensor([-1, 0, 1, 2, 3], dtype=torch.float, device=device)
        b = torch.quantize_per_tensor(a, 0.1, 10, dtype)
        self.check_nondeterministic_alert(
            lambda: b.resize_((10,)),
            'quantized_resize_cpu_')

    @skipXLA
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16, torch.uint16, torch.uint32, torch.uint64))
    def test_deterministic_resize(self, device, dtype):
        test_cases = [
            # size, stride, resize_size
            ((10,), (1,), (5,)),
            ((10,), (0,), (10,)),
            ((10,), (1,), (20,)),
            ((2, 3, 4), None, (2, 3, 4)),
            ((2, 3, 4), None, (6, 3, 4)),
            ((2, 3, 4), None, (2, 5, 4)),
            ((2, 3, 4), None, (2, 3, 6)),
            ((2, 3, 4), None, (3, 4, 5)),
            ((2, 3, 4), (1, 4, 12), (2, 3, 4)),
            ((2, 3, 4), (1, 4, 12), (4, 3, 4)),
            ((2, 3, 4), (1, 4, 12), (2, 4, 4)),
            ((2, 3, 4), (1, 4, 12), (2, 3, 5)),
            ((2, 3, 4), (1, 4, 12), (3, 4, 5)),
            ((2, 3, 4), (1, 0, 1), (2, 4, 5)),
        ]

        for size, stride, resize_size in test_cases:
            if stride is None:
                a = torch.zeros(size, dtype=dtype, device=device)
            else:
                a = torch.empty_strided(size, stride, dtype=dtype, device=device).fill_(0)
            old_storage = a.untyped_storage().clone()
            with DeterministicGuard(True, fill_uninitialized_memory=True):
                a.resize_(resize_size)

            new_storage = a.untyped_storage()

            # If storage size was increased, check that the new section is
            # filled with NaN/MAX_INT. Otherwise, check that the storages are
            # equal.
            old_tensor = torch.tensor(old_storage, dtype=dtype)
            old_numel = old_tensor.numel()
            new_tensor = torch.tensor(new_storage, dtype=dtype)
            new_numel = new_tensor.numel()

            if new_numel > old_numel:
                self.assertEqual(new_tensor[:old_numel], old_tensor)
                fill_section = new_tensor[old_numel:]

                if dtype.is_floating_point or dtype.is_complex:
                    self.assertTrue(fill_section.isnan().all())
                else:
                    if dtype == torch.bool:
                        max_val = True
                    else:
                        max_val = torch.iinfo(dtype).max
                    self.assertTrue(fill_section.eq(max_val).all())
            else:
                self.assertEqual(old_tensor, new_tensor)

    # When deterministic algorithms are enabled, `torch.empty` should fill floating
    # point tensors with NaN and integer tensors with MAX_INT
    @skipXLA
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    @dtypes(*all_types_and_complex_and(
        torch.half, torch.bool, torch.bfloat16, torch.uint16, torch.uint32, torch.uint64, torch.complex32))
    def test_deterministic_empty(self, device, dtype):
        gen_fns = [
            lambda: torch.empty(10, 9, device=device, dtype=dtype),
            lambda: torch.empty(10, 9, out=torch.zeros(1, device=device, dtype=dtype)),
            lambda: torch.empty_like(torch.zeros(10, 9, device=device, dtype=dtype)),
            lambda: torch.empty_like(torch.zeros(10, 9, device=device, dtype=dtype), memory_format=torch.contiguous_format),
            lambda: torch.empty_strided((10, 9), (1, 5), device=device, dtype=dtype),
            lambda: torch.empty_permuted((2, 3, 5), (1, 0, 2), device=device, dtype=dtype),
        ]

        for gen_fn in gen_fns:
            with DeterministicGuard(True, fill_uninitialized_memory=True):
                res = gen_fn()

            if dtype.is_floating_point or dtype.is_complex:
                self.assertTrue(res.isnan().all())
            else:
                if dtype == torch.bool:
                    max_val = True
                else:
                    max_val = torch.iinfo(dtype).max
                self.assertTrue(res.eq(max_val).all())

    # FIXME: update OpInfos to support "nondeterministic samples" and port these tests
    #   to that architecture
    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_AvgPool3d(self, device):
        module = torch.nn.AvgPool3d(3)
        input = torch.randn(2, 3, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'avg_pool3d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_AdaptiveAvgPool2d(self, device):
        module = torch.nn.AdaptiveAvgPool2d(3)
        input = torch.randn(2, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'adaptive_avg_pool2d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_AdaptiveAvgPool3d(self, device):
        module = torch.nn.AdaptiveAvgPool3d(3)
        input = torch.randn(2, 3, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'adaptive_avg_pool3d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_MaxPool3d(self, device):
        module = torch.nn.MaxPool3d(3)
        input = torch.randn(2, 3, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'max_pool3d_with_indices_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_AdaptiveMaxPool2d(self, device):
        module = torch.nn.AdaptiveMaxPool2d(3)
        input = torch.randn(2, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'adaptive_max_pool2d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_FractionalMaxPool2d(self, device):
        module = torch.nn.FractionalMaxPool2d(2, output_ratio=0.5)
        input = torch.randn(2, 3, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'fractional_max_pool2d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_FractionalMaxPool3d(self, device):
        module = torch.nn.FractionalMaxPool3d(2, output_ratio=0.5)
        input = torch.randn(2, 3, 3, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'fractional_max_pool3d_backward_cuda',
            torch.device(device).type == 'cuda')

    @dtypes(*floating_types_and(torch.half))
    @onlyNativeDeviceTypes
    def test_nondeterministic_alert_MaxUnpool1d(self, device, dtype):
        if dtype == torch.half and torch.device(device).type == 'cpu':
            self.skipTest('float16 not implemented on CPU')

        module = torch.nn.MaxUnpool1d(3, 1)
        input = torch.randn(1, 1, 7, dtype=dtype, device=device)
        indices = torch.zeros_like(input, dtype=torch.long, device=device)

        self.check_nondeterministic_alert(
            lambda: module(input, indices),
            'max_unpooling2d_forward_out')

    @dtypes(*floating_types_and(torch.half))
    @onlyNativeDeviceTypes
    def test_nondeterministic_alert_MaxUnpool2d(self, device, dtype):
        if dtype == torch.half and torch.device(device).type == 'cpu':
            self.skipTest('float16 not implemented on CPU')

        module = torch.nn.MaxUnpool2d(3, 1)
        input = torch.randn(1, 1, 7, 7, dtype=dtype, device=device)
        indices = torch.zeros_like(input, dtype=torch.long, device=device)

        self.check_nondeterministic_alert(
            lambda: module(input, indices),
            'max_unpooling2d_forward_out')

    @dtypes(*floating_types_and(torch.half))
    @onlyNativeDeviceTypes
    def test_nondeterministic_alert_MaxUnpool3d(self, device, dtype):
        if dtype == torch.half and torch.device(device).type == 'cpu':
            self.skipTest('float16 not implemented on CPU')

        module = torch.nn.MaxUnpool3d(3, 1)
        input = torch.randn(1, 1, 7, 7, 7, dtype=dtype, device=device)
        indices = torch.zeros_like(input, dtype=torch.long, device=device)

        self.check_nondeterministic_alert(
            lambda: module(input, indices),
            'max_unpooling3d_forward_out')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_interpolate_linear(self, device):
        input = torch.randn(1, 2, 4, device=device, requires_grad=True)
        res = torch.nn.functional.interpolate(
            input,
            size=12,
            mode='linear',
            align_corners=False)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad),
            'upsample_linear1d_backward_out_cuda',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_interpolate_bilinear(self, device):
        input = torch.randn(1, 2, 4, 4, device=device, requires_grad=True)
        res = torch.nn.functional.interpolate(
            input,
            size=12,
            mode='bilinear',
            align_corners=False)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad),
            'upsample_bilinear2d_backward_out_cuda',
            torch.device(device).type == 'cuda')

    def test_no_nondeterministic_alert_interpolate_bilinear(self, device):
        input = torch.randn(1, 2, 4, 4, device=device, requires_grad=True)

        def fn():
            res = torch.nn.functional.interpolate(
                input,
                size=12,
                mode='bilinear',
                align_corners=False)
            grad = torch.ones_like(res)
            return res.backward(grad)

        self.check_nondeterministic_alert(
            fn,
            'upsample_bilinear2d_backward_out_cuda',
            False)

    def test_no_nondeterministic_alert_interpolate_trilinear(self, device):
        input = torch.randn(1, 2, 4, 4, 4, device=device, requires_grad=True)

        def fn():
            res = torch.nn.functional.interpolate(
                input,
                size=12,
                mode='trilinear',
                align_corners=False)
            grad = torch.ones_like(res)
            return res.backward(grad)

        self.check_nondeterministic_alert(
            fn,
            'upsample_trilinear3d_backward_out_cuda',
            False)

    @skipIfTorchInductor("aot-autograd issue")
    def test_deterministic_replication_pad2d(self, device):
        test_cases = [
            # size, padding
            [(1, 2, 4, 4), (0, 0, 0, 0)],
            [(1, 2, 4, 4), (3, 4, 5, 6)],
            [(3, 8, 7), (0, 0, 0, 0)],
            [(3, 8, 7), (4, 3, 2, 7)],
        ]

        if torch.device(device).type != 'xla':
            test_cases += [
                [(4, 3, 5, 10), (-9, 4, 5, 6)],
                [(3, 8, 7), (-4, -2, -2, -3)],
            ]

        for size, padding in test_cases:
            input = torch.randn(*size, device=device, requires_grad=True)
            grad = None
            with DeterministicGuard(True):
                res = torch.nn.functional.pad(
                    input,
                    padding,
                    mode='replicate')
                res.backward(torch.ones_like(res))
                if grad is None:
                    grad = input.grad
                else:
                    self.assertEqual(grad, input.grad, atol=0, rtol=0)
                input.grad = None

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_deterministic_interpolate_bilinear(self, device):
        input = torch.randn(1, 2, 4, 4, device=device, requires_grad=True)
        grad = None
        with DeterministicGuard(True):
            for _ in range(5):
                res = torch.nn.functional.interpolate(
                    input,
                    size=12,
                    mode='bilinear',
                    align_corners=False)
                res.backward(torch.ones_like(res))
                if grad is None:
                    grad = input.grad
                else:
                    self.assertEqual(grad, input.grad, atol=0, rtol=0)
                input.grad = None

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_interpolate_bicubic(self, device):
        input = torch.randn(1, 2, 4, 4, device=device, requires_grad=True)
        res = torch.nn.functional.interpolate(
            input,
            size=12,
            mode='bicubic',
            align_corners=False)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad),
            'upsample_bicubic2d_backward_out_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_interpolate_trilinear(self, device):
        input = torch.randn(1, 2, 4, 4, 4, device=device, requires_grad=True)
        res = torch.nn.functional.interpolate(
            input,
            size=12,
            mode='trilinear',
            align_corners=False)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad),
            'upsample_trilinear3d_backward_out_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_ReflectionPad1d(self, device):
        module = torch.nn.ReflectionPad1d((1, 2))
        input = torch.randn(2, 3, 8, device=device, requires_grad=True)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'reflection_pad1d_backward_out_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_ReflectionPad3d(self, device):
        module = torch.nn.ReflectionPad3d((1, 2, 3, 4, 5, 6))
        input = torch.randn(2, 3, 8, 8, 8, device=device, requires_grad=True)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'reflection_pad3d_backward_out_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_ReplicationPad1d(self, device):
        module = torch.nn.ReplicationPad1d((1, 2))
        input = torch.randn(2, 3, 4, device=device, requires_grad=True)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'replication_pad1d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_ReplicationPad2d(self, device):
        module = torch.nn.ReplicationPad2d((1, 2, 3, 4))
        input = torch.randn(2, 3, 4, 4, device=device, requires_grad=True)
        res = module(input)
        grad = torch.ones_like(res)

        # Nondeterministic alert should only be raised if the forward call was
        # nondeterministic
        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'replication_pad2d_backward_cuda',
            torch.device(device).type == 'cuda')

        with DeterministicGuard(True):
            res = module(input)

        grad = torch.ones_like(res)

        # If the forward call was deterministic, nondeterministic alert should
        # not be raised
        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'replication_pad2d_backward_cuda',
            False)

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_ReplicationPad3d(self, device):
        module = torch.nn.ReplicationPad3d((1, 2, 3, 4, 5, 6))
        input = torch.randn(2, 3, 4, 4, 4, device=device, requires_grad=True)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'replication_pad3d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfTorchDynamo("Warning is not raised.")
    def test_nondeterministic_alert_NLLLoss(self, device):
        module = torch.nn.NLLLoss()
        input = torch.randn(2, 3, 5, 5, device=device)
        target = torch.rand(2, 5, 5, device=device).mul(3).floor().long()


        self.check_nondeterministic_alert(
            lambda: module(input, target),
            'nll_loss2d_forward_out_cuda_template',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_CTCLoss(self, device):
        module = torch.nn.CTCLoss()
        input = torch.randn(50, 3, 15, device=device, requires_grad=True)
        target = torch.randint(0, 14, (3, 30), device=device)
        input_lengths = [50, 50, 50]
        target_lengths = [30, 25, 20]
        res = module(input, target, input_lengths, target_lengths)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'ctc_loss_backward_gpu',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_EmbeddingBag_max(self, device):
        module = torch.nn.EmbeddingBag(
            4, 3, None, 2., False, 'max',
            _weight=torch.randn(4, 3, device=device, requires_grad=True))
        input = torch.randint(0, 3, (4, 3), device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'embedding_bag_backward_cuda_max',
            torch.device(device).type == 'cuda')

    @skipIfRocmArch(MI300_ARCH)
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    @onlyCUDA
    def test_deterministic_cumsum(self, device):
        test_cases = [
            # size, dim
            [(1025,), 0],
            [(8193,), 0],
            [(8191,), 0],
            [(128256,), 0],
            [(1282560,), 0],
            [(12825600,), 0],
        ]
        for size, dim in test_cases:
            input = 100 * torch.rand(*size, device=device)
            with DeterministicGuard(True):
                res0 = input.cumsum(dim)
                for _ in range(3):
                    res1 = input.cumsum(dim)
                    self.assertEqual(res0, res1, atol=0, rtol=0)

            res_cpu = input.cpu().cumsum(dim)
            self.assertEqual(res0, res_cpu, atol=1e-3, rtol=1e-2)
            # test double complex that has fewer threads than CTAs
            # that's a problem for large GPUS (H100 with 132 sms)
            # test is very tailored for that and will provide 0 signal
            # on smaller GPUs
            num_sms = 132
            elems_per_cta = 256 * 16
            N = num_sms * elems_per_cta
            input = torch.rand(N, dtype=torch.complex128, device=device)
            with DeterministicGuard(True):
                res0 = input.cumsum(dim)
                for _ in range(3):
                    res1 = input.cumsum(dim)
                    self.assertEqual(res0, res1, atol=0, rtol=0)

            res_cpu = input.cpu().cumsum(dim)
            self.assertEqual(res0, res_cpu, atol=1e-3, rtol=1e-2)

    @onlyCUDA
    @largeTensorTest('49GB')
    def test_cumsum_64bit_indexing(self, device):
        b = torch.ones(2 * 4096 * 8, 100000, dtype=torch.float, device='cuda')
        b /= 100000
        d = b.cumsum(dim=-1)
        chunk = 2**30 // b.shape[-1]
        for i in range(0, b.shape[0], chunk):
            end = min(i + chunk, b.shape[0])
            b[i:end, :].cumsum_(dim=-1)
        # cheat a bit to avoid OOM
        self.assertEqual(b[0, :], d[0, :], atol=3e-5, rtol=3e-5)
        self.assertEqual(b[-1, :], d[-1, :], atol=3e-5, rtol=3e-5)

    @expectedFailureMeta  # expected a non-determinitic error, but it was not raised
    @onlyNativeDeviceTypes
    def test_nondeterministic_alert_put(self, device):
        a = torch.randn(10, device=device)
        indices = torch.tensor([0, 0], device=device)
        values = torch.tensor([0., 1.], device=device)

        for op_call in [torch.Tensor.put, torch.Tensor.put_]:
            self.check_nondeterministic_alert(
                lambda: op_call(a, indices, values, accumulate=False),
                'put_')

    # warn_only=False correctly raises RuntimeError: put_ does not have a deterministic implementation
    # warn_only=True logs warning from the FallbackKernel: torch.ops.aten.put_.default, instead of as UserWarning:
    # [W Context.cpp:%(lineno)] Warning: put_ does not have a deterministic implementation
    @skipIfTorchInductor("warning is logged from the FallbackKernel: torch.ops.aten.put_.default when warn_only=True")
    def test_nondeterministic_alert_put_accumulate(self, device):
        a = torch.randn(10, device=device)
        indices = torch.tensor([0, 0], device=device)
        values = torch.tensor([0., 1.], device=device)

        for op_call in [torch.Tensor.put, torch.Tensor.put_]:
            self.check_nondeterministic_alert(
                lambda: op_call(a, indices, values, accumulate=True),
                'put_',
                torch.device(device).type == 'cuda')

    @dtypes(torch.float32)
    @dtypesIfCUDA(torch.float32, torch.int32)
    @skipIfMPS
    def test_nondeterministic_alert_histc(self, device, dtype):
        a = torch.tensor([], device=device, dtype=dtype)
        for op_call in [torch.histc, torch.Tensor.histc]:
            self.check_nondeterministic_alert(
                lambda: op_call(a, min=0, max=3),
                '_histc_cuda with floating point input',
                torch.device(device).type == 'cuda' and dtype.is_floating_point)

    @skipIfMPS
    def test_nondeterministic_alert_bincount(self, device):
        a = torch.tensor([], device=device, dtype=torch.long)
        weights = torch.tensor([], device=device)

        for op_call in [torch.bincount, torch.Tensor.bincount]:
            # Error should only be raised when device is CUDA and weights are
            # given
            self.check_nondeterministic_alert(
                lambda: op_call(a, weights),
                '_bincount_cuda',
                torch.device(device).type == 'cuda')

            self.check_nondeterministic_alert(
                lambda: op_call(a),
                '_bincount_cuda',
                False)

    # Ensures that kthvalue throws nondeterministic alerts in the correct cases
    @dtypes(torch.double)
    def test_nondeterministic_alert_kthvalue(self, device, dtype):
        def test_func(call_type):
            S = 10
            k = 5
            a = torch.randn(S, device=device)
            if call_type == 'function':
                torch.kthvalue(a, k)
            elif call_type == 'method':
                a.kthvalue(k)
            elif call_type == 'out':
                values = torch.empty_like(a)
                indices = torch.empty((), device=device, dtype=torch.long)
                torch.kthvalue(a, k, out=(values, indices))
            else:
                self.fail(f"'{call_type}' is not a valid call type")

        for call_type in ['function', 'method', 'out']:
            self.check_nondeterministic_alert(
                lambda: test_func('function'),
                'kthvalue CUDA',
                torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_grid_sample_2d(self, device):
        input = torch.empty(1, 1, 2, 2, device=device, requires_grad=True)
        grid = torch.empty(1, 1, 1, 2, device=device)
        res = torch.nn.functional.grid_sample(input, grid, align_corners=False)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'grid_sampler_2d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_grid_sample_3d(self, device):
        input = torch.empty(1, 1, 2, 2, 2, device=device, requires_grad=True)
        grid = torch.empty(1, 1, 1, 2, 3, device=device)
        res = torch.nn.functional.grid_sample(input, grid, align_corners=False)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'grid_sampler_3d_backward_cuda',
            torch.device(device).type == 'cuda')

    def test_invalid_shapes_grid_sampler(self, device):
        make_arg = partial(
            make_tensor, device=device, dtype=torch.float64, requires_grad=True)

        inputs = (
            # input, grid
            ((5, 5, 5, 5, 5,), (1, 1, 1, 4, 4,)),  # 3d
            ((5, 5, 5, 5,), (1, 1, 4, 4,)),  # 2d
        )

        interpolation_mode = 0
        padding_mode = 0
        align_corners = True

        err = "expected grid and input to have same batch size"

        for input, grid in inputs:
            input = make_arg(input)
            grid = make_arg(grid, low=-1, high=1)

            # Wrapper for the 2d, 3d, and cuDNN functions listed below.
            with self.assertRaisesRegex(RuntimeError, err):
                torch.grid_sampler(
                    input, grid, interpolation_mode, padding_mode,
                    align_corners)

            # Expects 2d input.
            with self.assertRaisesRegex(RuntimeError, err):
                torch.grid_sampler_2d(
                    input, grid, interpolation_mode, padding_mode,
                    align_corners)

            # Expects 3d input.
            with self.assertRaisesRegex(RuntimeError, err):
                torch.grid_sampler_3d(
                    input, grid, interpolation_mode, padding_mode,
                    align_corners)

            # Expects 2d input.
            with self.assertRaisesRegex(RuntimeError, err):
                torch._grid_sampler_2d_cpu_fallback(
                    input, grid, interpolation_mode, padding_mode,
                    align_corners)

            # Expects 2d input, on CUDA.
            # Doesn't work on CPU and ROCm.
            if device != 'cpu' and TEST_CUDNN and not TEST_WITH_ROCM:
                with self.assertRaisesRegex(RuntimeError, err):
                    torch.cudnn_grid_sampler(input, grid)

    def test_dist(self, device):
        def run_test(x, y):
            for p in [0, 1, 2, 3, 4, inf, -inf]:
                dist_xy = torch.dist(x, y, p)
                dist_xy_norm = torch.norm(x - y, p)
                self.assertEqual(dist_xy, dist_xy_norm)

        run_test(torch.randn(5, device=device), torch.randn(5, device=device))

        x = torch.zeros(3, device=device)
        y = torch.zeros(3, device=device)
        y[1] = 1.
        run_test(x, y)

    # Ensures that median throws nondeterministic alerts in the correct cases
    @dtypes(torch.double)
    def test_nondeterministic_alert_median(self, device, dtype):
        def test_func(call_type):
            S = 10
            a = torch.randn(S, device=device)
            if call_type == 'function':
                torch.median(a)
            elif call_type == 'function with indices':
                torch.median(a, 0)
            elif call_type == 'method':
                a.median()
            elif call_type == 'method with indices':
                a.median(0)
            elif call_type == 'out with indices':
                result = torch.empty_like(a)
                indices = torch.empty((), dtype=torch.long, device=device)
                torch.median(a, 0, out=(result, indices))
            else:
                self.fail(f"'{call_type}' is not a valid call type")

        def test_func_expect_error(call_type, should_error):
            self.check_nondeterministic_alert(
                lambda: test_func(call_type),
                'median CUDA with indices output',
                should_error)

        is_cuda = torch.device(device).type == 'cuda'

        test_func_expect_error('function', False)
        test_func_expect_error('function with indices', is_cuda)
        test_func_expect_error('method', False)
        test_func_expect_error('method with indices', is_cuda)
        test_func_expect_error('out with indices', is_cuda)

    # FIXME: move to test_scatter_gather_ops
    def _test_gather_backward_one_dim(self, device, deterministic: bool = False) -> None:
        with DeterministicGuard(deterministic):
            m = random.randint(2000, 3000)
            elems = random.randint(10 * m, 20 * m)
            dim = 0
            src = torch.randn(m, device=device, requires_grad=True)
            idx = torch.randint(m, (elems,), device=device)
            res = torch.gather(src, dim, idx)
            weight = torch.rand_like(res, device=device) * 10 ** 6
            res.backward(weight)
            assert src.grad is not None
            grad = src.grad.detach().clone()

            if torch.device(device).type == 'cuda':
                for _ in range(2):
                    src.grad.data.zero_()
                    res = torch.gather(src, dim, idx)
                    res.backward(weight)
                    self.assertEqual(src.grad, grad, atol=0, rtol=0)
            else:
                expected = torch.zeros_like(src, device=device)
                for i in range(elems):
                    expected[idx[i]] += weight[i]
                self.assertEqual(grad, expected, atol=0, rtol=0)

    # FIXME: move to test_scatter_gather_ops
    @onlyNativeDeviceTypes
    def test_gather_backward_deterministic_path(self, device) -> None:
        self._test_gather_backward_one_dim(device, True)

    # FIXME: move to test_scatter_gather_ops
    @onlyCPU
    def test_gather_backward_one_dim(self, device) -> None:
        self._test_gather_backward_one_dim(device, False)

    # FIXME: move to test_scatter_gather_ops
    @onlyNativeDeviceTypes
    def test_scatter_add_one_dim_deterministic(self, device) -> None:
        with DeterministicGuard(True):
            m = random.randint(20, 30)
            elems = random.randint(2000 * m, 3000 * m)
            dim = 0
            src = torch.randn(elems, device=device)
            idx = torch.randint(m, (elems,), device=device)

            x = torch.zeros(m, device=device)
            res = x.scatter_add(dim, idx, src)

            # Checking if scatter_add is deterministic
            for i in range(5):
                res_next = x.scatter_add(dim, idx, src)
                self.assertEqual(res, res_next, atol=0, rtol=0)
                res = res_next

            expected = torch.zeros(m, device=device)
            for i in range(elems):
                expected[idx[i]] += src[i]

            self.assertEqual(res, expected, atol=1e-4, rtol=1e-5)

    # FIXME: move to test_scatter_gather_ops
    @onlyNativeDeviceTypes
    def test_scatter_zero_size_index(self, device) -> None:
        null_index = torch.zeros((0, 4), dtype=torch.int64)
        null_arr = torch.zeros((0, 4))
        original = torch.arange(4, dtype=torch.float32)
        result = original.scatter(0, null_index, null_arr)
        self.assertEqual(result, original, atol=0, rtol=0)

    @onlyCUDA
    @skipIfTorchInductor("FIXME")
    def test_sync_warning(self, device):

        def _sync_raises_helper(f, level):
            with CudaSyncGuard(level):
                if level == 1:
                    with self.assertWarnsRegex(UserWarning, "called a synchronizing "):
                        f()
                elif level == 2:
                    with self.assertRaisesRegex(RuntimeError, "called a synchronizing "):
                        f()

        def _no_sync_helper(f, level):
            with CudaSyncGuard(level):
                f()

        def _ind_put_fn(x, ind, val):
            x[ind] = val
            return x

        def _ind_get_fn(x, ind):
            return x[ind]

        def _cond_fn(x):
            if x:  # taking boolean value of a tensor synchronizes
                return x
            else:
                return 2 * x

        # prepare inputs for subsequent ops
        size = 4
        x = torch.rand(size, device=device)
        y = torch.rand((), device=device)
        ind = torch.randint(size, (3,), device=device)
        ind_cpu = ind.cpu()
        repeats = torch.full((1,), 2, device=device)
        mask = torch.randint(2, (size,), device=device, dtype=bool)
        mask_cpu = mask.cpu()
        expect_no_sync = (lambda: _ind_put_fn(x, mask, 1.),
                          lambda: _ind_put_fn(x, mask_cpu, y),
                          lambda: _ind_put_fn(x, ind, y),
                          lambda: _ind_get_fn(x, mask_cpu),
                          lambda: _ind_get_fn(x, ind),
                          lambda: torch.nn.functional.one_hot(ind, num_classes=size),
                          lambda: torch.randperm(20000, device=device),
                          lambda: torch.repeat_interleave(x, 2, output_size=2 * size),
                          lambda: torch.repeat_interleave(x, repeats, output_size=2 * size),
                          lambda: torch.any(y))
        expect_sync = (lambda: _ind_put_fn(x, mask, y),
                       lambda: _ind_put_fn(x, ind_cpu, y),
                       lambda: _ind_get_fn(x, mask),
                       lambda: _ind_get_fn(x, ind_cpu),
                       lambda: x.nonzero(),
                       lambda: _cond_fn(y),
                       lambda: torch.nn.functional.one_hot(ind),
                       lambda: torch.repeat_interleave(x, repeats))
        for f, level in product(expect_no_sync, (1, 2)):
            _no_sync_helper(f, level)
        for f, level in product(expect_sync, (1, 2)):
            _sync_raises_helper(f, level)


    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    @skipIfMPS
    def test_log_normal(self, device, dtype):
        a = torch.tensor([10], dtype=dtype, device=device).log_normal_()
        self.assertEqual(a.dtype, dtype)
        self.assertEqual(a.size(), torch.Size([1]))

    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    @skipIfMPS
    def test_geometric(self, device, dtype):
        a = torch.tensor([10], dtype=dtype, device=device).geometric_(0.5)
        self.assertEqual(a.dtype, dtype)
        self.assertEqual(a.size(), torch.Size([1]))

    @skipIfMPS
    def test_repeat_interleave(self, device):
        y = torch.tensor([[1, 2], [3, 4]], device=device)
        # exercise single argument function signature
        temp = y.repeat_interleave(2)
        self.assertEqual(torch.Size([8]), temp.size())

        for dtype in [torch.int, torch.long]:
            lengths = torch.tensor([1, 2], dtype=dtype, device=device)
            output_size = torch.sum(lengths)
            a = torch.repeat_interleave(
                y,
                lengths,
                dim=0,
            )
            self.assertEqual(a.dtype, y.dtype)
            self.assertEqual(a.size(), torch.Size([3, 2]))

            a_with_output = torch.repeat_interleave(
                y,
                lengths,
                dim=0,
                output_size=output_size,
            )
            self.assertEqual(a_with_output.dtype, y.dtype)
            self.assertEqual(a_with_output.size(), torch.Size([3, 2]))

    @dtypes(*floating_types())
    @dtypesIfCPU(*floating_types_and(torch.bfloat16, torch.half))
    @dtypesIfCUDA(*floating_types_and(torch.half))
    def test_bernoulli_p(self, device, dtype):
        for trivial_p in ([0, 1], [1, 0, 1, 1, 0, 1]):
            x = torch.tensor(trivial_p, dtype=dtype, device=device)
            self.assertEqual(x.bernoulli().tolist(), trivial_p)

        def isBinary(t):
            return torch.ne(t, 0).mul_(torch.ne(t, 1)).sum().item() == 0

        p = torch.rand(5, 5, dtype=dtype, device=device)
        self.assertTrue(isBinary(p.bernoulli()))

        p = torch.rand(5, dtype=dtype, device=device).expand(5, 5)
        self.assertTrue(isBinary(p.bernoulli()))

        p = torch.rand(5, 5, dtype=dtype, device=device)
        torch.bernoulli(torch.rand_like(p), out=p)
        self.assertTrue(isBinary(p))

    # RngUniform not implemented for Integral type in XLA test
    @dtypes(*floating_types())
    @dtypesIfCPU(*all_types_and(torch.bool, torch.half))
    @dtypesIfCUDA(*all_types_and(torch.bool, torch.half))
    def test_bernoulli_self(self, device, dtype):

        def isBinary(t):
            return torch.ne(t, 0).mul_(torch.ne(t, 1)).sum().item() == 0

        t = torch.empty(10, 10, dtype=dtype, device=device)

        t.fill_(2)
        t.bernoulli_(0.5)
        self.assertTrue(isBinary(t))

        for p_dtype in floating_types_and(*[torch.half] if device.startswith('cuda') else []):
            p = torch.rand(10, dtype=p_dtype, device=device).expand(10, 10)
            t.fill_(2)
            t.bernoulli_(p)
            self.assertTrue(isBinary(t))

            t.fill_(2)
            torch.bernoulli(torch.rand_like(t, dtype=p_dtype), out=t)
            self.assertTrue(isBinary(t))

            t.fill_(2)
            t.bernoulli_(torch.rand_like(t, dtype=p_dtype))
            self.assertTrue(isBinary(t))

    @slowTest
    @dtypes(*floating_types_and(torch.half))
    @dtypesIfCUDA(*floating_types_and(torch.half))
    def test_bernoulli_edge_cases(self, device, dtype):
        # Need to draw a lot of samples to cover every random floating point number.
        a = torch.zeros(10000, 10000, dtype=dtype, device=device)  # probability of drawing "1" is 0
        num_ones = (torch.bernoulli(a) == 1).sum()
        self.assertEqual(num_ones, 0)

        b = torch.ones(10000, 10000, dtype=dtype, device=device)  # probability of drawing "1" is 1
        num_zeros = (torch.bernoulli(b) == 0).sum()
        self.assertEqual(num_zeros, 0)

    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    @skipIfMPS
    def test_exponential(self, device, dtype):
        a = torch.tensor([10], dtype=dtype, device=device).exponential_(0.5)
        self.assertEqual(a.dtype, dtype)
        self.assertEqual(a.size(), torch.Size([1]))

        # Tests extremal behavior
        t = torch.empty((1,), device=device, dtype=dtype).exponential_(float('inf'))
        self.assertTrue(t.item() == 0)

        # Tests that negative lambda fails
        with self.assertRaises(RuntimeError):
            torch.empty((1,), device=device, dtype=dtype).exponential_(-0.5)

    @onlyCUDA
    @dtypes(torch.half, torch.float)
    def test_exponential_no_zero(self, device, dtype):
        # naively, 0 in exponential can be generated with probability 2^-24
        # so we need more samples to check if it's not generated
        # instead of doing one
        # don't test CPU, that would be a long test
        x = torch.empty(50000000, device=device, dtype=dtype).exponential_()
        self.assertTrue(x.min() > 0)

    def _generate_correlation_tensors(self, device, dtype):
        yield make_tensor((0, 0), dtype=dtype, device=device)
        yield make_tensor((1, 0), dtype=dtype, device=device)
        yield make_tensor((0, 1), dtype=dtype, device=device)
        yield make_tensor((2,), dtype=dtype, device=device)
        yield make_tensor((2, 1), dtype=dtype, device=device)
        yield make_tensor((2, 2), dtype=dtype, device=device)
        yield make_tensor((2, 3), dtype=dtype, device=device)
        yield make_tensor((5, 10), dtype=dtype, device=device)
        yield make_tensor((5, 10), dtype=dtype, device=device, noncontiguous=True)
        if dtype != torch.int:
            yield torch.tensor([0, -2, nan, 10.2, inf], dtype=dtype, device=device)

    @tf32_on_and_off(0.005)
    @onlyNativeDeviceTypes
    @dtypes(torch.int, torch.float, torch.cfloat)
    def test_corrcoef(self, device, dtype):
        for x in self._generate_correlation_tensors(device, dtype):
            res = torch.corrcoef(x)
            ref = np.corrcoef(x.cpu().numpy())
            self.assertEqual(res, ref, atol=1e-04, rtol=1e-03, exact_dtype=False)

    @skipRocmIfTorchInductor
    @dtypes(torch.int, torch.float, torch.cfloat)
    def test_cov(self, device, dtype):
        def check(t, correction=1, fweights=None, aweights=None):
            res = torch.cov(t, correction=correction, fweights=fweights, aweights=aweights)
            t = t.cpu().numpy()
            fweights = fweights.cpu().numpy() if fweights is not None else None
            aweights = aweights.cpu().numpy() if aweights is not None else None
            ref = np.cov(t, ddof=correction, fweights=fweights, aweights=aweights)
            self.assertEqual(res, ref, atol=1e-05, rtol=1e-05, exact_dtype=False)

        for x in self._generate_correlation_tensors(device, dtype):
            check(x)
            num_observations = x.numel() if x.ndim < 2 else x.size(1)
            if num_observations > 0:
                fweights = torch.randint(1, 10, (num_observations,), device=device)
                aweights = make_tensor((num_observations,), dtype=torch.float, device=device, low=1)
                for correction, fw, aw in product([0, 1, 2], [None, fweights], [None, aweights]):
                    check(x, correction, fweights, aweights)

    @skipIfNoSciPy
    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_uniform_kstest(self, device, dtype):
        from scipy import stats
        size = 1000
        for from_ in [-42, 0, 4.2]:
            for to_ in [-4.2, 0, 42]:
                if to_ > from_:
                    t = torch.empty(size, dtype=dtype, device=device).uniform_(from_, to_)
                    res = stats.kstest(t.cpu().to(torch.double), 'uniform', args=(from_, (to_ - from_)))
                    self.assertTrue(res.statistic < 0.1)

    @skipIfNoSciPy
    @dtypes(*floating_types_and(torch.half))
    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    def test_normal_kstest(self, device, dtype):
        from scipy import stats
        size = 1000
        for mean in [-10, 0, 50]:
            for std in [1, 5, 10]:
                t = torch.empty(size, dtype=dtype, device=device).normal_(mean=mean, std=std)
                res = stats.kstest(t.cpu().to(torch.double), 'norm', args=(mean, std))
                self.assertTrue(res.statistic < 0.1)

    @skipIfMPS
    @skipIfNoSciPy
    @skipRocmIfTorchInductor
    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_lognormal_kstest(self, device, dtype):
        from scipy import stats
        size = 1000
        for mean in [-3, 0, 7]:
            for std in [1, 5, 7]:
                t = torch.empty(size, dtype=dtype, device=device).log_normal_(mean=mean, std=std)
                res = stats.kstest(t.cpu().to(torch.double), 'lognorm', args=(std, 0, math.exp(mean)))
                if dtype == torch.half:
                    self.assertTrue(res.statistic < 0.3)
                else:
                    self.assertTrue(res.statistic < 0.1)

    @skipIfMPS
    @skipIfNoSciPy
    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_exponential_kstest(self, device, dtype):
        from scipy import stats
        size = 1000
        for lambd in [0.5, 1.0, 5.0]:
            t = torch.empty(size, dtype=dtype, device=device).exponential_(lambd=lambd)
            res = stats.kstest(t.cpu().to(torch.double), 'expon', args=(0, 1 / lambd,))
            self.assertTrue(res.statistic < 0.1)

    @skipIfMPS
    @skipIfNoSciPy
    @skipRocmIfTorchInductor
    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_cauchy_kstest(self, device, dtype):
        from scipy import stats
        size = 1000
        for median in [-10, 0, 50]:
            for sigma in [0.5, 1.0, 10.0]:
                t = torch.empty(size, dtype=dtype, device=device).cauchy_(median=median, sigma=sigma)
                res = stats.kstest(t.cpu().to(torch.double), 'cauchy', args=(median, sigma))
                self.assertTrue(res.statistic < 0.1)

    @slowTest
    @onlyCUDA
    @dtypes(torch.bfloat16, torch.float32)
    def test_cauchy_no_inf(self, device, dtype):
        # torch.float16 will have `inf` because of its smaller range.
        for _ in range((2**16) * 2):
            x = torch.empty((2**16), dtype=dtype, device=device)
            x.cauchy_()
            self.assertFalse(x.isinf().sum())

    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_cauchy(self, device, dtype):
        a = torch.tensor([10], dtype=dtype, device=device).cauchy_(0.0, 0.5)
        self.assertEqual(a.dtype, dtype)
        self.assertEqual(a.size(), torch.Size([1]))

        # Tests extremal behavior
        t = torch.empty((1,), device=device, dtype=dtype).cauchy_(float('inf'), 0.5)
        self.assertTrue(t.item() == float('inf'))

        # Tests non-positive rate fails
        with self.assertRaises(RuntimeError):
            torch.empty((1,), device=device, dtype=dtype).cauchy_(0.0, 0.0)

    @skipIfMPS
    @skipIfNoSciPy
    @skipRocmIfTorchInductor
    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    def test_geometric_kstest(self, device, dtype):
        from scipy import stats
        size = 1000
        for p in [0.2, 0.5, 0.8]:
            t = torch.empty(size, dtype=dtype, device=device).geometric_(p=p)
            actual = np.histogram(t.cpu().to(torch.double), np.arange(1, 100))[0]
            expected = stats.geom(p).pmf(np.arange(1, 99)) * size
            res = stats.chisquare(actual, expected)
            self.assertEqual(res.pvalue, 1.0, atol=0.1, rtol=0)

    # FIXME: find test suite for pdist and cdist
    def test_pairwise_distance_empty(self, device):
        shape = (2, 0)
        x = torch.randn(shape, device=device)
        y = torch.randn(shape, device=device)

        self.assertEqual(torch.zeros(2, device=device), torch.pairwise_distance(x, y))
        self.assertEqual(torch.zeros((2, 1), device=device), torch.pairwise_distance(x, y, keepdim=True))

        shape = (0, 2)
        x = torch.randn(shape, device=device)
        y = torch.randn(shape, device=device)
        self.assertEqual(torch.zeros(0, device=device), torch.pairwise_distance(x, y))
        self.assertEqual(torch.zeros((0, 1), device=device), torch.pairwise_distance(x, y, keepdim=True))

    def test_pdist_empty(self, device):
        shape = (0, 2)
        x = torch.randn(shape, device=device)
        self.assertEqual(torch.empty(0, device=device), torch.pdist(x))

        shape = (1, 2)
        x = torch.randn(shape, device=device)
        self.assertEqual(torch.empty(0, device=device), torch.pdist(x))

        shape = (3, 0)
        x = torch.randn(shape, device=device)
        self.assertEqual(torch.zeros(3, device=device), torch.pdist(x))

    def test_cdist_empty(self, device):
        x = torch.randn((0, 5), device=device)
        y = torch.randn((4, 5), device=device)
        self.assertEqual(torch.empty(0, 4, device=device), torch.cdist(x, y))

        x = torch.randn((2, 5), device=device)
        y = torch.randn((0, 5), device=device)
        self.assertEqual(torch.empty(2, 0, device=device), torch.cdist(x, y))

        x = torch.randn((2, 0), device=device)
        y = torch.randn((3, 0), device=device)
        self.assertEqual(torch.zeros(2, 3, device=device), torch.cdist(x, y))

        x = torch.randn((2, 0), device=device)
        y = torch.randn((0, 0), device=device)
        self.assertEqual(torch.empty(2, 0, device=device), torch.cdist(x, y))

    def _brute_cdist(self, x, y, p=2):
        r1 = x.shape[-2]
        r2 = y.shape[-2]
        if r1 == 0 or r2 == 0:
            return torch.empty(r1, r2, device=x.device)
        return torch.norm(x[..., None, :] - y[..., None, :, :], p=p, dim=-1)

    @skipIfMPS
    def test_cdist_norm(self, device):
        for r1 in [3, 4, 5, 6]:
            for m in [2, 3, 4, 10]:
                for r2 in [4, 6, 7, 8]:
                    for p in [0, 1, 2, 3, 1.5, 2.5, float('inf')]:
                        x = torch.randn(r1, m, device=device)
                        y = torch.randn(r2, m, device=device)
                        if p == 2:
                            for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
                                actual = torch.cdist(x, y, p=2, compute_mode=cm)
                                expected = self._brute_cdist(x, y, p=2)
                                self.assertEqual(expected, actual, rtol=0, atol=0.02)
                        else:
                            actual = torch.cdist(x, y, p=p)
                            expected = self._brute_cdist(x, y, p=p)
                            self.assertEqual(expected, actual)

    @skipIfMPS
    def test_cdist_norm_batch(self, device):
        for r1 in [3, 4, 5, 6]:
            for m in [2, 3, 4, 10]:
                for r2 in [4, 6, 7, 8]:
                    for p in [0, 1, 2, 3, 1.5, 2.5, float('inf')]:
                        x = torch.randn(2, 3, 6, r1, m, device=device)
                        y = torch.randn(2, 3, 6, r2, m, device=device)
                        if p == 2:
                            for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
                                actual = torch.cdist(x, y, p=2, compute_mode=cm)
                                expected = self._brute_cdist(x, y, p=2)
                                self.assertEqual(expected, actual, rtol=0, atol=0.02)
                        else:
                            actual = torch.cdist(x, y, p=p)
                            expected = self._brute_cdist(x, y, p=p)
                            self.assertEqual(expected, actual)

    @onlyCUDA
    def test_cdist_cuda_backward(self, device):
        for l1 in [1, 511, 513]:
            for l2 in [1, 511, 513]:
                for p in [0, 1, 2, 3, 1.5, 2.5, float('inf')]:
                    x1 = torch.randn(4, l1, 32, device=device, requires_grad=True)
                    x2 = x1.clone().detach_().requires_grad_()
                    y1 = torch.randn(4, l2, 32, device=device, requires_grad=True)
                    y2 = y1.clone().detach_().requires_grad_()
                    if p == 2:
                        for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
                            z1 = torch.cdist(x1, y1, p=2, compute_mode=cm).mean()
                            z2 = self._brute_cdist(x2, y2, p=2).mean()
                            z1.backward()
                            z2.backward()
                            self.assertEqual(x1.grad, x2.grad, rtol=0, atol=0.001)
                            self.assertEqual(y1.grad, y2.grad, rtol=0, atol=0.001)
                    else:
                        z1 = torch.cdist(x1, y1, p=p).mean()
                        z2 = self._brute_cdist(x2, y2, p=p).mean()
                        self.assertEqual(x1.grad, x2.grad, rtol=0, atol=0.001)
                        self.assertEqual(y1.grad, y2.grad, rtol=0, atol=0.001)

    @tf32_on_and_off(0.05 if TEST_WITH_ROCM else 0.005)
    @reduced_f32_on_and_off(0.08)
    def test_cdist_large(self, device):
        for cm in ['use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            x = torch.randn(1000, 10, device=device)
            y = torch.randn(1000, 10, device=device)
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertEqual(expected, actual)

    @slowTest
    @tf32_on_and_off(0.01)
    @reduced_f32_on_and_off(0.08)
    def test_cdist_large_batch(self, device):
        for cm in ['use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            x = torch.randn(4, 3, 1000, 10, device=device)
            y = torch.randn(4, 3, 1000, 10, device=device)
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertEqual(expected, actual)

    @tf32_on_and_off(0.005)
    @reduced_f32_on_and_off(0.04)
    def test_cdist_non_contiguous(self, device):
        for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            x = torch.randn(5, 7, device=device).mT
            y = torch.randn(5, 3, device=device).mT
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertFalse(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            self.assertEqual(expected, actual)

            x = torch.randn(7, 5, device=device)
            y = torch.randn(5, 3, device=device).t()
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertTrue(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            self.assertEqual(expected, actual)

            x = torch.randn(5, 7, device=device).t()
            y = torch.randn(3, 5, device=device)
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertFalse(x.is_contiguous())
            self.assertTrue(y.is_contiguous())
            self.assertEqual(expected, actual)

    @tf32_on_and_off(0.005)
    @reduced_f32_on_and_off(0.04)
    def test_cdist_non_contiguous_batch(self, device):
        for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            x = torch.randn(4, 3, 2, 5, 7, device=device).mT
            y = torch.randn(4, 3, 2, 5, 3, device=device).mT
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertFalse(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            self.assertEqual(expected, actual)

            x = torch.randn(7, 2, 7, 5, device=device)
            y = torch.randn(7, 2, 5, 3, device=device).mT
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertTrue(x.is_contiguous())
            self.assertFalse(y.is_contiguous())
            self.assertEqual(expected, actual)

            x = torch.randn(4, 5, 7, device=device).mT
            y = torch.randn(4, 3, 5, device=device)
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertFalse(x.is_contiguous())
            self.assertTrue(y.is_contiguous())
            self.assertEqual(expected, actual)

    # Maybe merge into OpInfo?
    def test_cdist_euclidean_large(self, device):
        def _test_euclidean_large_cdist(sizex, sizey=None):
            if sizey is None:
                sizey = sizex
            x = torch.randn(sizex, device=device, dtype=torch.float)
            y = torch.randn(sizey, device=device, dtype=torch.float)
            eps = 1e-6
            # to avoid extremum
            x = x - (((x - y) < eps).float() * 2 * eps)
            x.requires_grad = True
            y.requires_grad = True
            dist = torch.cdist(x, y, p=2)
            # Do a backward pass to check that it is valid for large
            # matrices
            loss = dist.sum()
            loss.backward()

        _test_euclidean_large_cdist((2000, 5))

    # Ensure that cdist backward with p<1 does not produce NaNs
    @skipIfMPS
    def test_cdist_grad_p_lt_1_no_nan(self, device):
        for p in [0.99, 0.7, 0.5, 0.1, 0.01]:
            x = torch.randn(1, 2, device=device)
            y = x.detach().clone() + torch.tensor([[1., 0.]], device=device)
            x.requires_grad = True
            y.requires_grad = True
            result = torch.cdist(x, y, p=p)
            result.backward(torch.ones_like(result))
            self.assertFalse(torch.isnan(x.grad).any())
            self.assertFalse(torch.isnan(y.grad).any())

    def test_cdist_same_inputs(self, device):
        # Test to detect issues in cdist gradient calculation
        # When the distances are 0
        sizex = (1, 27, 32)
        for p in [0, 1, 2, 3, 1.5, 2.5, float('inf')]:
            x = torch.randn(sizex, device=device, dtype=torch.float)
            dist_grad = torch.randn((1, 27, 27), device=device, dtype=torch.float)
            y = x.clone()
            x.requires_grad = True
            d = torch.cdist(x, y)
            d.backward(dist_grad)
            # Check that the backward pass does not contain invalid
            # values such as nan or inf
            assert torch.isfinite(x.grad).all()

    @skipIfMPS
    def test_cumsum(self, device):
        x = torch.rand(100, 100, device=device)
        res1 = torch.cumsum(x, 1)
        res2 = torch.tensor([]).to(device)
        torch.cumsum(x, 1, out=res2)
        self.assertEqual(res1, res2)
        x.cumsum_(1)
        self.assertEqual(res1, x)

        a = torch.tensor([[True, False, True],
                          [False, False, False],
                          [True, True, True]], device=device)
        b = a.byte()
        aRes = torch.cumsum(a, 0)
        bRes = torch.cumsum(b, 0)
        self.assertEqual(aRes, bRes)
        self.assertEqual(aRes, torch.tensor([[1, 0, 1],
                                             [1, 0, 1],
                                             [2, 1, 2]]))

        aRes = torch.cumsum(a, 1)
        bRes = torch.cumsum(b, 1)
        self.assertEqual(aRes, bRes)
        self.assertEqual(aRes, torch.tensor([[1, 1, 2],
                                             [0, 0, 0],
                                             [1, 2, 3]]))

        # Check that cumulative sum over a zero length dimension doesn't crash on backprop.
        # Also check that cumsum over other dimensions in a tensor with a zero-length
        # dimensiuon also works
        # Also include a basic suite of similar tests for other bases cases.
        shapes = [[2, 0], [2, 1, 4], [0, 2, 3], [1], [5]]
        for shape in shapes:
            for dim in range(len(shape)):
                raw_tensor = torch.zeros(*shape, requires_grad=True)
                integrated = raw_tensor.cumsum(dim=dim)
                # Check that backward does not crash
                integrated.sum().backward()
                # Check that output maintained correct shape
                self.assertEqual(raw_tensor.shape, raw_tensor.grad.shape)

        # Check a scalar example
        raw_tensor = torch.tensor(3., requires_grad=True)
        integrated = raw_tensor.cumsum(dim=-1)
        self.assertEqual(raw_tensor, integrated)
        # Check that backward does not crash
        integrated.sum().backward()
        # Check that output maintained correct shape
        self.assertEqual(raw_tensor.shape, raw_tensor.grad.shape)

    @skipIfMPS
    def test_cumprod(self, device):
        x = torch.rand(100, 100, device=device)
        res1 = torch.cumprod(x, 1)
        res2 = torch.tensor([]).to(device)
        if not TEST_WITH_TORCHINDUCTOR:
            torch.cumprod(x, 1, out=res2)
            self.assertEqual(res1, res2)
        x.cumprod_(1)
        self.assertEqual(res1, x)

        a = torch.tensor([[True, False, True],
                          [False, False, False],
                          [True, True, True]], dtype=torch.bool, device=device)
        b = a.byte()
        aRes = torch.cumprod(a, 0)
        bRes = torch.cumprod(b, 0)
        self.assertEqual(aRes, bRes)
        self.assertEqual(aRes, torch.tensor([[1, 0, 1],
                                             [0, 0, 0],
                                             [0, 0, 0]]))

        aRes = torch.cumprod(a, 1)
        bRes = torch.cumprod(b, 1)
        self.assertEqual(aRes, bRes)
        self.assertEqual(aRes, torch.tensor([[1, 0, 0],
                                             [0, 0, 0],
                                             [1, 1, 1]]))

        # Check that cumulative prod over a zero length dimension doesn't crash on backprop.
        # Also check that cumprod over other dimensions in a tensor with a zero-length
        # dimensiuon also works
        # Also include a basic suite of similar tests for other bases cases.
        shapes = [[2, 0], [2, 1, 4], [0, 2, 3], [1], [5]]
        for shape in shapes:
            for dim in range(len(shape)):
                raw_tensor = torch.zeros(*shape, requires_grad=True)
                integrated = raw_tensor.cumprod(dim=dim)
                # Check that backward does not crash
                integrated.sum().backward()
                # Check that output maintained correct shape
                self.assertEqual(raw_tensor.shape, raw_tensor.grad.shape)

        # Check a scalar example
        raw_tensor = torch.tensor(3., requires_grad=True)
        integrated = raw_tensor.cumprod(dim=-1)
        self.assertEqual(raw_tensor, integrated)
        # Check that backward does not crash
        integrated.sum().backward()
        # Check that output maintained correct shape
        self.assertEqual(raw_tensor.shape, raw_tensor.grad.shape)

    @skipIfMPS
    def test_cummax_cummin(self, device):
        def test_ops(op, string_of_function_name, expected_output1, expected_output2):
            x = torch.rand(100, 100, device=device)
            out1 = op(x, 1)
            res2 = torch.empty(0, device=device)
            indices2 = torch.empty(0, dtype=torch.int64, device=device)
            op(x, 1, out=(res2, indices2))
            self.assertEqual(out1[0], res2)
            self.assertEqual(out1[1], indices2)

            a = torch.tensor([[True, False, True],
                              [False, False, False],
                              [True, True, True]], dtype=torch.bool, device=device)
            b = a.byte()
            aRes = op(a, 0)
            bRes = op(b, 0)
            self.assertEqual(aRes[0], bRes[0].bool())
            self.assertEqual(aRes[0], expected_output1.bool())

            # test inf and nan input
            x = torch.tensor([4, inf, 1.5, -inf, 0, nan, 1])
            xRes = op(x, 0)[0]
            self.assertEqual(xRes, expected_output2)

            # op shouldn't support values, indices with a dtype, device type or layout
            # different from that of input tensor
            t = torch.randn(10)
            values = torch.empty(0, dtype=torch.int16)
            indices = torch.empty(0, dtype=torch.int64)
            with self.assertRaisesRegex(
                    RuntimeError,
                    'expected scalar_type Float but found Short'):
                op(t, 0, out=(values, indices))

            # Range-check 0-d tensors
            x = torch.rand([])
            dim = 100
            with self.assertRaisesRegex(
                    IndexError,
                    'Expected reduction dim -1 or 0 for scalar but got 100'):
                op(x, dim)

            # Check that op over a zero length dimension doesn't crash on backprop.
            # Also check that op over other dimensions in a tensor with a zero-length
            # dimension also works
            # Also include a basic suite of similar tests for other bases cases.
            shapes = [[2, 0], [2, 1, 4], [0, 2, 3], [1], [5]]
            for shape in shapes:
                for dim in range(len(shape)):
                    raw_tensor = torch.zeros(*shape, requires_grad=True)
                    integrated = getattr(raw_tensor, string_of_function_name)(dim=dim)
                    # Check that backward does not crash
                    integrated[0].sum().backward()
                    # Check that output maintained correct shape
                    self.assertEqual(raw_tensor.shape, raw_tensor.grad.shape)

            # Check a scalar example
            raw_tensor = torch.tensor(3., requires_grad=True)
            integrated = getattr(raw_tensor, string_of_function_name)(dim=-1)
            # Check that backward does not crash
            integrated[0].sum().backward()
            # Check that output maintained correct shape
            self.assertEqual(raw_tensor.shape, raw_tensor.grad.shape)

        expected_out = torch.tensor([4, inf, inf, inf, inf, nan, nan])
        test_ops(torch.cummax, "cummax", torch.tensor([[1, 0, 1],
                                                       [1, 0, 1],
                                                       [1, 1, 1]]), expected_out)

        expected_out = torch.tensor([4, 4, 1.5, -inf, -inf, nan, nan])
        test_ops(torch.cummin, "cummin", torch.tensor([[1, 0, 1],
                                                       [0, 0, 0],
                                                       [0, 0, 0]]), expected_out)

    @skipIfMPS
    def test_logcumsumexp(self, device):
        def logcumsumexp(a, axis):
            return torch.cumsum(a.exp(), axis=axis).log_()

        axis = -1
        a = torch.randn(100, 100, device=device)

        actual = a.logcumsumexp(axis)
        expected = logcumsumexp(a, axis)
        self.assertEqual(a.dtype, actual.dtype)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected, actual)

        # check -inf and nan handling
        x = torch.tensor([-float('inf'), -float('inf'), 1.0, 1.0, float('inf'),
                         float('inf'), float('nan'), 1.0, 1.0], device=device)
        x2d = x.unsqueeze(0).expand(2, -1)

        for inp in (x, x2d):
            actual = inp.logcumsumexp(axis)
            expected = logcumsumexp(inp, axis)
            self.assertEqual(expected, actual)

        # Check that out is actually inplace
        b = torch.randn(5, 2, device=device)
        inplace_out = torch.zeros(5, 2, device=device)

        expected = logcumsumexp(b, axis)
        torch.logcumsumexp(b, axis=axis, out=inplace_out)

        self.assertEqual(inplace_out, expected)

        # Check input and inplace_output type mismatch
        b = torch.randn(5, 2, device=device, dtype=torch.float64)
        inplace_out = torch.zeros(5, 2, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(
                RuntimeError,
                'expected scalar_type Double but found Float'):
            torch.logcumsumexp(b, axis, out=inplace_out)

    def _test_diff_numpy(self, t, dims=None):
        # Helper for test_diff to compare with NumPy reference implementation
        def to_np(t):
            if t.dtype == torch.bfloat16:
                return t.to(dtype=torch.float, device="cpu").numpy()
            else:
                return t.cpu().numpy()

        for dim in dims if dims else range(t.dim()):
            prepend = t.narrow(dim, 0, 1)
            append = t.narrow(dim, 0, 1)
            np_t = to_np(t)

            # test when no prepend and append
            for n in range(t.size(dim)):
                actual = torch.diff(t, dim=dim, n=n)
                expected = torch.from_numpy(np.diff(np_t, axis=dim, n=n))
                self.assertEqual(actual, expected.to(t.dtype))

            # test when prepend and append's size along dim is 1
            for n in range(1, t.size(dim) + 4):
                actual = torch.diff(t, dim=dim, n=n, prepend=prepend, append=append)
                expected = torch.from_numpy(np.diff(np_t, axis=dim, n=n, prepend=to_np(prepend), append=to_np(append)))
                self.assertEqual(actual, expected.to(t.dtype))

            # test when prepend and append's size along dim != 1
            for n in range(1, t.size(dim) * 3):
                actual = torch.diff(t, dim=dim, n=n, prepend=t, append=t)
                expected = torch.from_numpy(np.diff(np_t, axis=dim, n=n, prepend=np_t, append=np_t))
                self.assertEqual(actual, expected.to(t.dtype))

    # All tensors appear contiguous on XLA
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool))
    def test_diff_noncontig(self, device, dtype):
        shapes = (
            (1,),
            (1, 5),
            (3, 5),
            (1, 5, 1),
            (2, 3, 5))

        for shape in shapes:
            contig = make_tensor(shape, dtype=dtype, device=device, low=-9, high=9)

            non_contig = torch.empty(shape + (2, 2), device=device, dtype=dtype)[..., 0]
            non_contig = non_contig.select(-1, -1)
            non_contig.copy_(contig)
            self.assertTrue(not non_contig.is_contiguous() or shape == (1,))

            self._test_diff_numpy(non_contig)

    # RngNormal not implemented for type f16 for XLA
    @dtypes(*all_types_and_complex_and(torch.bool))
    @dtypesIfCPU(*all_types_and_complex_and(torch.half, torch.bool))
    @dtypesIfCUDA(*all_types_and_complex_and(torch.half, torch.bool))
    def test_diff(self, device, dtype):
        shapes = (
            (1,),
            (1, 5),
            (3, 5),
            (1, 5, 1),
            (2, 3, 5))

        for shape in shapes:
            contig = make_tensor(shape, dtype=dtype, device=device, low=-9, high=9)
            self._test_diff_numpy(contig)

        t = torch.ones(2, 3)

        with self.assertRaisesRegex(
                RuntimeError, 'diff expects prepend or append to be the same dimension as input'):
            invalid_prepend = torch.tensor([1, 2, 3], device=device, dtype=dtype)
            t.diff(dim=0, prepend=invalid_prepend)

        with self.assertRaisesRegex(
                RuntimeError, 'diff expects the shape of tensor to prepend or append to match that of input'):
            invalid_prepend = torch.tensor([[0, 1]], device=device, dtype=dtype)
            t.diff(dim=0, prepend=invalid_prepend)

        with self.assertRaisesRegex(
                RuntimeError, 'diff expects input to be at least one-dimensional'):
            scalar = torch.tensor(2, device=device, dtype=dtype)
            torch.diff(scalar)

    # if the given input arg is not a list, it returns a list of single element: [arg]
    def _wrap_to_list(self, input_array):
        return list(input_array) if isinstance(input_array, (list, tuple)) else [input_array]

    # To ensure inf, -inf, and nan values do not cause divergence between Numpy and PyTorch.
    # There are two types of possible divergence:
    # 1. When we compute a,b both real numbers and has very small absolute values (i.e. very near to 0.0)
    # then, result of a/b be inf, -inf and nan, and this cause divergence.
    # 2. When we are dividing complex numbers by zero. For example, when a = torch.tensor(3+5j) we have
    # a/0 to be equal to nan + nan*j in PyTorch and inf + inf*j in Numpy.
    def _inf_nan_preprocess(self, actual, expected):
        for i in range(len(expected)):
            expected[i] = np.nan_to_num(expected[i], nan=nan, posinf=nan, neginf=nan)
            # nan_to_num is not defined for complex tensors in PyTorch.
            if actual[i].dtype == torch.complex64 :
                actual[i].real = torch.nan_to_num(actual[i].real, nan=nan, posinf=nan, neginf=nan)
                actual[i].imag = torch.nan_to_num(actual[i].imag, nan=nan, posinf=nan, neginf=nan)
            else:
                actual[i] = torch.nan_to_num(actual[i], nan=nan, posinf=nan, neginf=nan)

        return actual, expected

    @onlyNativeDeviceTypes
    @dtypes(torch.long, torch.float32, torch.complex64)
    def test_gradient_all(self, device, dtype):
        def create_scalar(shape):
            return make_tensor((1,), device='cpu', dtype=dtype, low=1.).item()

        def create_list(shape):
            return make_tensor((len(shape),), device='cpu', dtype=dtype, low=1.).tolist()

        def create_coordinate_tensors(shape):
            tensor_list = []
            for i in range(len(shape)):
                tensor_list.append(make_tensor((shape[i],), device=device, dtype=dtype))
            return tensor_list

        def filter_shape(shape, dim):
            filtered_shape = []
            for i in range(len(dim)):
                filtered_shape.append(shape[dim[i]])
            return filtered_shape

        # shape, dims format
        test_cases = (
            ((5,), (0,)),
            ((4, 4), (0, 1)),
            ((3, 3, 3), (-1, 0)),
            ((4, 4, 4), (2,)),
            ((4, 4, 4), (0, 1)),
            ((4, 4, 4, 3), (0, 2, 3)),
            ((4, 5, 3, 4, 3), (1, 2)),
            ((4, 3, 6, 5, 3), (2, 4)),
            ((4, 3, 3, 5, 3), (0, 1, 2, 3, 4)),
            ((1, 3, 3), (1, 2)),
            ((1, 5), (1,)),
        )

        for case, contig, edge_order, space_fn in product(test_cases, [True, False], [1, 2],
                                                          (create_scalar, create_list, create_coordinate_tensors)):
            shape, dims = case
            # filter shape by dims before passing filtered shape to create_* functions
            filtered_shape = filter_shape(shape, dims)

            spacing = space_fn(filtered_shape)
            t = make_tensor(shape, device=device, dtype=dtype, noncontiguous=not contig)
            t_np = t.cpu().numpy()

            actual = torch.gradient(t, spacing=spacing, dim=dims, edge_order=edge_order)
            if space_fn == create_coordinate_tensors and spacing[0].device != 'cpu':
                spacing = [space.cpu().detach().numpy() for space in spacing]
            expected = np.gradient(t_np, *self._wrap_to_list(spacing), axis=dims, edge_order=edge_order)
            actual, expected = self._inf_nan_preprocess(list(actual), self._wrap_to_list(expected))
            self.assertEqual(actual, expected, equal_nan=True, atol=1e-4, rtol=0, exact_dtype=False)

    @onlyNativeDeviceTypes
    @slowTestIf(TEST_WITH_TORCHINDUCTOR)
    @dtypes(torch.long, torch.float32, torch.complex64)
    def test_gradient_extreme_cases(self, device, dtype):
        # Test behaviour for inf and nan values
        actual = torch.gradient(torch.tensor([2, -2, inf, inf, -inf, -inf, inf, 3, -inf, 2, nan, nan, 3, inf, nan]))
        expected = np.gradient(np.array([2, -2, inf, inf, -inf, -inf, inf, 3, -inf, 2, nan, nan, 3, inf, nan]))
        self.assertEqual(actual, self._wrap_to_list(expected), exact_dtype=False)

        # Test behaviour in very big tensors
        large_size = 100000
        t = make_tensor((large_size,), dtype=dtype, device=device)
        t_np = t.cpu().numpy()
        coordinates_np = np.random.randn(large_size)
        coordinates = [torch.tensor(coordinates_np, device=device)]
        actual = torch.gradient(t, spacing=coordinates, dim=0, edge_order=1)
        expected = [np.gradient(t_np, coordinates_np, axis=0, edge_order=1)]
        self.assertEqual(actual, expected, exact_dtype=False)

        actual = torch.gradient(t, spacing=coordinates, dim=0, edge_order=2)
        expected = [np.gradient(t_np, coordinates_np, axis=0, edge_order=2)]
        self.assertEqual(actual, expected, exact_dtype=False)

    @onlyNativeDeviceTypes
    def test_gradient_type_promotion(self, device):
        inputs = (
            make_tensor((4, 4), device=device, dtype=torch.float32),
            make_tensor((4, 4), device=device, dtype=torch.complex64),
            make_tensor((4, 4), device=device, dtype=torch.int64),
        )

        spacing = (
            make_tensor((1,), device='cpu', dtype=torch.float32).item(),
            make_tensor((1,), device='cpu', dtype=torch.int64).item(),
            make_tensor((1,), device='cpu', dtype=torch.complex64).item(),
            make_tensor((2,), device='cpu', dtype=torch.float32, low=0.1).tolist(),
            make_tensor((2,), device='cpu', dtype=torch.int64, low=1).tolist(),
            make_tensor((2,), device='cpu', dtype=torch.complex64).tolist(),
            [make_tensor((4,), device=device, dtype=torch.float32),
             make_tensor((4,), device=device, dtype=torch.float32)],
            [make_tensor((4,), device=device, dtype=torch.int64),
             make_tensor((4,), device=device, dtype=torch.int64)],
            [make_tensor((4,), device=device, dtype=torch.complex64),
             make_tensor((4,), device=device, dtype=torch.complex64)],
        )

        for input, spacing_or_coord, edge_order in product(inputs, spacing, [1, 2]):
            input_np = input.cpu().numpy()
            input_np = input.cpu().numpy()
            actual = torch.gradient(input, spacing=spacing_or_coord, dim=(0, 1), edge_order=edge_order)
            spacing_or_coord_wrapped = self._wrap_to_list(spacing_or_coord)
            spacing_or_coord_np = []
            if torch.is_tensor(spacing_or_coord_wrapped[0]) and torch.device(spacing_or_coord_wrapped[0].device).type != 'cpu':
                for i in range(len(spacing_or_coord_wrapped)):
                    spacing_or_coord_np.append(spacing_or_coord_wrapped[i].detach().clone().cpu().numpy())
            else:
                spacing_or_coord_np = spacing_or_coord_wrapped
            expected = np.gradient(input_np, *spacing_or_coord_np, axis=(0, 1), edge_order=edge_order)
            if actual[0].dtype == torch.complex64 and input.dtype != torch.complex64:
                for i in range(len(actual)):
                    self.assertEqual(actual[i].real, expected[i].real, exact_dtype=False)
                    # Type promotion fails on Numpy when spacing is given as complex number and input is given as real.
                    # Result is given just as real number and all the imaginary parts to be equal to zero.
                    self.assertEqual(expected[i].imag, torch.zeros(actual[i].shape), exact_dtype=False)
            else:
                actual, expected = self._inf_nan_preprocess(list(actual), list(expected))
                self.assertEqual(actual, expected, equal_nan=True, exact_dtype=False)

    @onlyNativeDeviceTypes
    @dtypes(torch.long, torch.float32, torch.complex64)
    def test_gradient_spacing_list_length_error(self, device, dtype):
        t = make_tensor((2, 2), device=device, dtype=dtype)

        spacing = (make_tensor((2,), device=device, dtype=dtype),)
        with self.assertRaisesRegex(RuntimeError, r'expected spacing to be'):
            torch.gradient(t, spacing=spacing)

        spacing = (make_tensor((2,), device=device, dtype=dtype),) * 2
        torch.gradient(t, spacing=spacing)

        spacing = (make_tensor((2,), device=device, dtype=dtype),) * 3
        with self.assertRaisesRegex(RuntimeError, r'expected spacing to be'):
            torch.gradient(t, spacing=spacing)

        spacing = (2,)
        with self.assertRaisesRegex(RuntimeError, r'expected spacing to be'):
            torch.gradient(t, spacing=spacing)

        spacing = (2, 2)
        torch.gradient(t, spacing=spacing)

        spacing = (2, 2, 2)
        with self.assertRaisesRegex(RuntimeError, r'expected spacing to be'):
            torch.gradient(t, spacing=spacing)

    def _test_large_cum_fn_helper(self, x, fn):
        expected = fn(x.cpu().float())
        actual = fn(x).cpu().float()
        # Avoid self.assertEqual to save memory.
        torch.testing.assert_close(expected, actual)

    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "sandcastle OOM with current tpx gpu/re configuration")
    @unittest.skipIf(IS_JETSON, "psutil issue for largeTensorTest. Too large for Jetson.")
    @onlyCUDA
    @dtypes(torch.half)  # only small dtype not to get oom
    @largeTensorTest('25GB', device='cpu')
    @largeTensorTest('4GB', device='cuda')
    def test_large_cumsum(self, device, dtype):
        # initialization to avoid overflow and half caveats
        x = torch.empty(2**30 + 200, device=device, dtype=dtype)
        x[::3] = -3
        x[1::3] = 2
        x[2::3] = 1
        self._test_large_cum_fn_helper(x, lambda x: torch.cumsum(x, 0))

    @onlyCUDA
    @dtypes(torch.half)  # only small dtype not to get oom
    @largeTensorTest('25GB', device='cpu')
    @largeTensorTest('4GB', device='cuda')
    @unittest.skipIf(IS_JETSON, "psutil issue for largeTensorTest. Too large for Jetson.")
    def test_large_cumprod(self, device, dtype):
        # initialization to avoid overflow and half caveats
        x = torch.empty(2**30 + 200, device=device, dtype=dtype)
        x[::3] = 8
        x[1::3] = .25
        x[2::3] = .5
        self._test_large_cum_fn_helper(x, lambda x: torch.cumprod(x, 0))

    @skipIfTorchDynamo("Torchdynamo fails with unknown reason")
    @skipIfMPS
    def test_discontiguous_out_cumsum(self, device):
        x = torch.randn(4, 8, device=device)
        y = torch.empty(4, 16, device=device)[:, ::2]
        out = torch.cumsum(x, 0)
        torch.cumsum(x, 0, out=y)
        self.assertFalse(y.is_contiguous())
        self.assertEqual(out, y, atol=0., rtol=0.)

    def _test_cumminmax_helper(self, x, fn, expected_val, expected_ind):
        val, ind = fn(x, -1)
        self.assertEqual(val, expected_val, atol=0, rtol=0)
        self.assertEqual(ind, expected_ind, atol=0, rtol=0)
        out_val = torch.empty_like(val).t().contiguous().t()
        out_ind = torch.empty_like(ind).t().contiguous().t()
        fn(x, -1, out=(out_val, out_ind))
        # TODO: Fix this. It reproduces with aot_eager too, and looks like a functionalization bug.
        # (the problematic case seems rare, as we're calling an out= op directly from user code,
        # where the passed-in out tensors are non-contiguous).
        if not TEST_WITH_TORCHINDUCTOR:
            self.assertFalse(out_val.is_contiguous())
            self.assertFalse(out_ind.is_contiguous())
        self.assertEqual(out_val, expected_val, atol=0, rtol=0)
        self.assertEqual(out_ind, expected_ind, atol=0, rtol=0)

    @skipIfMPS
    def test_cummax_discontiguous(self, device):
        x = torch.tensor([[0, 1, 2, 3, 2, 1], [4, 5, 6, 5, 6, 7]], device=device, dtype=torch.float).t().contiguous().t()
        expected_val = torch.tensor([[0, 1, 2, 3, 3, 3], [4, 5, 6, 6, 6, 7]], device=device, dtype=torch.float)
        expected_ind = torch.tensor([[0, 1, 2, 3, 3, 3], [0, 1, 2, 2, 4, 5]], device=device, dtype=torch.long)
        self._test_cumminmax_helper(x, torch.cummax, expected_val, expected_ind)

    @skipIfMPS
    def test_cummin_discontiguous(self, device):
        x = torch.tensor([[3, 2, 1, 0, 1, 2], [7, 6, 5, 4, 5, 2]], device=device, dtype=torch.float).t().contiguous().t()
        expected_val = torch.tensor([[3, 2, 1, 0, 0, 0], [7, 6, 5, 4, 4, 2]], device=device, dtype=torch.float)
        expected_ind = torch.tensor([[0, 1, 2, 3, 3, 3], [0, 1, 2, 3, 3, 5]], device=device, dtype=torch.long)
        self._test_cumminmax_helper(x, torch.cummin, expected_val, expected_ind)

    def test_bool_tensor_value_change(self, device):
        x = torch.tensor([True, False], dtype=torch.bool, device=device)
        x[0] = False
        x[1] = True
        self.assertEqual(x, torch.tensor([False, True], dtype=torch.bool, device=device))

    # FIXME: move to data movement test suite
    def test_copy_all_dtypes_and_devices(self, device):
        from copy import copy
        for dt in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16):
            x = torch.tensor([1, 2, 3, 4], dtype=dt, device=device)
            _x_clone = x.clone()
            y = copy(x)
            y.fill_(1)
            # copy is a shallow copy, only copies the tensor view,
            # not the data
            self.assertEqual(x, y)

    @onlyCPU
    def test_bfloat16_neg_abs(self, device):
        src = torch.randn(256)
        src[0] = torch.nan
        src[1] = -torch.nan
        src[2] = torch.inf
        src[3] = -torch.inf
        src_bf16 = src.bfloat16()
        self.assertEqual(src.neg().bfloat16(), src_bf16.neg())
        self.assertEqual(src.abs().bfloat16(), src_bf16.abs())

    @onlyCPU
    @dtypes(torch.bfloat16, torch.half)
    def test_reduced_type_float_copy(self, device, dtype):
        for shape in [(20, 7), (249, 137), (1029, 917), (1, 7, 19, 17), (3, 77, 1091)]:
            input = torch.randn(shape, dtype=torch.float, device=device)
            out1 = input.to(dtype=dtype)
            self.assertEqual(input, out1, atol=None, rtol=None, exact_dtype=False)
            out2 = out1.to(torch.float)
            self.assertEqual(out2, out1, atol=0, rtol=0, exact_dtype=False)

            input_s = input[..., ::2, :]
            out1 = input_s.to(dtype=dtype)
            self.assertEqual(input_s, out1, atol=None, rtol=None, exact_dtype=False)
            out2 = out1.to(torch.float)
            self.assertEqual(out2, out1, atol=0, rtol=0, exact_dtype=False)

    # FIXME: move to data movement test suite
    @onlyNativeDeviceTypes
    def test_copy_math_view(self, device):
        for dst_dtype, src_dtype in [
                (torch.float32, torch.float32),
                (torch.float64, torch.float32),
                (torch.int64, torch.int32),
                (torch.complex128, torch.complex64),
        ]:
            src = make_tensor((100,), dtype=src_dtype, device=device)
            dst = torch.empty(100, dtype=dst_dtype, device=device)

            dst.copy_(src)
            self.assertEqual(dst, src, exact_dtype=False)

            dst.copy_(src._neg_view())
            self.assertEqual(dst, src.neg(), exact_dtype=False)

            dst._neg_view().copy_(torch._neg_view(src))
            self.assertEqual(dst, src, exact_dtype=False)

            dst._neg_view().copy_(src)
            self.assertEqual(dst, src.neg(), exact_dtype=False)

            # issue: https://github.com/pytorch/pytorch/issues/106051
            dst._neg_view().copy_(dst)
            self.assertEqual(dst, src, exact_dtype=False)

        for dst_dtype, src_dtype in [
                (torch.complex64, torch.complex64),
                (torch.complex128, torch.complex64),
        ]:
            src = make_tensor((100,), dtype=src_dtype, device=device)
            dst = torch.empty(100, dtype=dst_dtype, device=device)

            dst.conj().copy_(src)
            self.assertEqual(dst, src.conj_physical(), exact_dtype=False)

            dst.conj().copy_(src._neg_view())
            self.assertEqual(dst, src.neg().conj_physical(), exact_dtype=False)

    # FIXME: move to data movement test suite
    @onlyNativeDeviceTypes
    @dtypes(torch.int64, torch.float32, torch.complex64)
    def test_copy_transpose_math_view(self, device, dtype):
        src = make_tensor((100, 100), dtype=dtype, device=device).transpose(0, 1)
        dst = torch.empty((100, 100), dtype=dtype, device=device)

        dst._neg_view().copy_(src)
        self.assertEqual(dst, -src)
        dst._neg_view().copy_(src._neg_view())
        self.assertEqual(dst, src)
        dst.copy_(src._neg_view())
        self.assertEqual(dst, -src)

        if dtype.is_complex:
            dst.conj().copy_(src)
            self.assertEqual(dst, src.conj_physical())
            dst.conj().copy_(src.conj())
            self.assertEqual(dst, src)
            dst.copy_(src.conj())
            self.assertEqual(dst, src.conj_physical())

    def test_clone_all_dtypes_and_devices(self, device):
        for dt in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16):
            x = torch.tensor((1, 1), dtype=dt, device=device)
            y = x.clone()
            self.assertEqual(x, y)

    def test_clone_zero_stride_dim(self, device):
        # stride zero, size 1 axis, not contiguous
        x = torch.randn(10)
        y = x.as_strided([2, 1, 5], [1, 0, 2])
        self.assertEqual(y, y.clone())

    def test_clone_not_memory_dense(self):
        # github issue: https://github.com/pytorch/pytorch/issues/64176
        x = torch.randn(10, 8).t()[::2, ::2]
        y = x.clone()
        # should retain permutation after densification
        self.assertTrue(y.stride() == (1, 4))

    # FIXME: move to elementwise ternary test suite
    @parametrize("use_cpu_scalar", [True, False])
    @dtypesIfCUDA(*set(get_all_math_dtypes('cuda')))
    @dtypes(*set(get_all_math_dtypes('cpu')))
    def test_addcmul(self, device, dtype, use_cpu_scalar):
        # Returns floating or integral scalar corresponding to dtype
        def _number(floating, integer, dtype):
            if dtype in [torch.half, torch.float, torch.double, torch.bfloat16]:
                return floating
            elif dtype in [torch.cfloat, torch.cdouble]:
                return floating * (1 + 1j)
            else:
                return integer

        def rand_tensor(size, dtype, device):
            if dtype.is_floating_point or dtype.is_complex:
                return torch.rand(size=size, dtype=dtype, device=device)
            if dtype == torch.uint8:
                return torch.randint(1, 5, size=size, dtype=dtype, device=device)
            else:
                return torch.randint(-5, 5, size=size, dtype=dtype, device=device)

        a = rand_tensor((2, 2), dtype=dtype, device=device)
        b = rand_tensor((2, 2), dtype=dtype, device=device)
        if use_cpu_scalar:
            c = rand_tensor([], device="cpu", dtype=dtype)
        else:
            c = rand_tensor((2, 2), dtype=dtype, device=device)

        alpha = _number(0.5, 3, dtype)

        actual = torch.addcmul(a, b, c, value=alpha)
        expected = a + alpha * b * c

        self.assertEqual(expected, actual)

        with self.assertWarnsOnceRegex(
                UserWarning, "This overload of addcmul is deprecated"):
            self.assertEqual(actual, torch.addcmul(a, alpha, b, c))

        if self.device_type == 'cuda' and dtype == torch.half:
            a = torch.tensor([60000.0], device=device, dtype=dtype)
            b = torch.tensor([60000.0], device=device, dtype=dtype)
            c = torch.tensor([2.0], device=device, dtype=dtype)
            out = torch.addcmul(a, b, c, value=-1)
            self.assertTrue(not (out.isnan() or out.isinf()))

    @onlyCUDA
    def test_addcmul_cuda_errors_with_cpu_scalars(self, device):
        # Logic is dtype agnostic, so dtype isn't tested
        alpha = 0.5

        a = torch.rand((2, 2), device=device)
        b = torch.rand((2, 2), device=device)
        c = torch.rand((2, 2), device=device)
        scalar = torch.rand([], device="cpu")

        with self.assertRaisesRegex(RuntimeError, r'CPU Scalar support for tensor1 argument'):
            torch.addcmul(a, scalar, c, value=alpha)
        with self.assertRaisesRegex(RuntimeError, r'CPU Scalar support for self argument'):
            torch.addcmul(scalar, b, c, value=alpha)

    # FIXME: move to shape ops test suite
    def test_narrow_empty(self, device):
        x = torch.randn(2, 3, 4, device=device)
        for d in range(x.dim()):
            y = x.narrow(d, x.size(d), 0)
            sz = list(x.size())
            sz[d] = 0
            self.assertEqual(sz, y.size())

    def test_narrow_copy_non_contiguous(self, device):
        # see https://github.com/pytorch/pytorch/issues/91690.
        inp = torch.randn(10, 2, device=device).movedim(-1, 0)
        expected = torch.narrow_copy(inp.contiguous(), 1, 0, 10)
        actual = torch.narrow_copy(inp, 1, 0, 10)
        self.assertEqual(expected, actual)

    # FIXME: move to indexing test suite
    @parametrize("reduce", ['prod', 'amin', 'amax', 'mean'])
    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    def test_index_reduce(self, device, dtype, reduce):
        size = (3, 4, 5)
        index_dtypes = [torch.int, torch.long]
        include_selfs = [True, False]
        amin_init = float('inf') if dtype.is_floating_point else torch.iinfo(dtype).max
        amax_init = -float('inf') if dtype.is_floating_point else torch.iinfo(dtype).min
        reduction_init = {'prod': 1, 'mean': 0, 'amin': amin_init, 'amax': amax_init}

        for dest_noncontig, src_noncontig, index_noncontig in product([True, False], repeat=3):
            for idx_dtype, include_self in product(index_dtypes, include_selfs):
                for dim in range(len(size)):
                    num_src = np.random.randint(10)
                    num_dest = size[dim]
                    dest = make_tensor(size, device=device, dtype=dtype, noncontiguous=dest_noncontig)
                    src_size = size[:dim] + (num_src,) + size[dim + 1:]
                    src = make_tensor(src_size, device=device, dtype=dtype, noncontiguous=src_noncontig)
                    idx = torch.testing.make_tensor(
                        num_src, low=0, high=num_dest, dtype=idx_dtype, device=device, noncontiguous=index_noncontig
                    )
                    expected = dest.clone()
                    dest.index_reduce_(dim, idx, src, reduce, include_self=include_self)
                    # fill rows in idx with reduction inits if include_self=False
                    if (not include_self):
                        expected.index_fill_(dim, idx.long(), reduction_init[reduce])
                    expected = expected.transpose(0, dim)
                    src = src.transpose(0, dim)
                    for i in range(num_src):
                        if reduce == 'prod':
                            expected[idx[i]] *= src[i]
                        elif reduce == 'amin':
                            torch.minimum(expected[idx[i]], src[i], out=expected[idx[i]])
                        elif reduce == 'amax':
                            torch.maximum(expected[idx[i]], src[i], out=expected[idx[i]])
                        else:
                            expected[idx[i]] += src[i]
                    if reduce == 'mean':
                        counts = torch.ones_like(expected) if include_self else torch.zeros_like(expected)
                        counts.index_add_(0, idx, torch.ones_like(src))
                        counts.masked_fill_(counts == 0, 1)
                        if (dtype.is_floating_point):
                            expected.div_(counts)
                        else:
                            expected.div_(counts, rounding_mode="floor")
                    expected = expected.transpose(0, dim)

                    self.assertEqual(dest, expected)

    # FIXME: move to test indexing
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_index_copy(self, device, dtype):
        # We just test for num_copy <= num_dest, as otherwise there are repeated indices
        # and the behavior is undefined
        num_copy, num_dest = 3, 5

        def make_arg(batch_sizes, n, dim, contig):
            size_arg = batch_sizes[:dim] + (n,) + batch_sizes[dim:]
            return make_tensor(size_arg, dtype=dtype, device=device, low=None, high=None, noncontiguous=not contig)

        def ref_index_copy(tgt, dim, idx, src):
            for i in range(idx.size(0)):
                idx_dest = dim * (slice(None),) + (idx[i],)
                idx_src = dim * (slice(None),) + (i,)
                tgt[idx_dest] = src[idx_src]

        # More thorough testing as in index_add
        for dest_contig, src_contig, index_contig in product([True, False], repeat=3):
            for other_sizes in ((), (4, 5)):
                for dim in range(len(other_sizes)):
                    dest = make_arg(other_sizes, num_dest, dim, dest_contig)
                    src = make_arg(other_sizes, num_copy, dim, src_contig)
                    idx = torch.randperm(num_dest, dtype=torch.int64, device=device)[:num_copy]
                    if not index_contig:
                        idx = torch.repeat_interleave(idx, 2, dim=-1)
                        idx = idx[..., ::2]
                    dest2 = dest.clone()
                    dest.index_copy_(dim, idx, src)
                    ref_index_copy(dest2, dim, idx, src)
                    self.assertEqual(dest, dest2)

    # FIXME: move to test indexing
    # onlyNativeDeviceTypes due to an XLA error:
    # https://github.com/pytorch/pytorch/issues/53256
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_index_copy_scalars(self, device, dtype):
        # Create the 8 possible combinations of scalar sizes for target / index / source
        scalars = ((make_tensor(size_t, dtype=dtype, device=device, low=None, high=None),
                    make_tensor(size_i, dtype=torch.int64, device=device, low=0, high=1),
                    make_tensor(size_s, dtype=dtype, device=device, low=None, high=None))
                   for size_t, size_i, size_s in product([(), (1,)], repeat=3))
        for target, idx, source in scalars:
            target.index_copy_(0, idx, source)
            self.assertEqual(target.item(), source.item())

    # FIXME: move to test indexing
    @onlyCPU
    def test_errors_index_copy(self, device):
        # We do not test the GPU as the CUDA_ASSERT would break the CUDA context
        idx_dim = 8
        tgt_dim = 5
        batch_dim = 3

        # Too large of an index
        a = torch.randn(batch_dim, tgt_dim, device=device)
        idx = torch.full((idx_dim,), tgt_dim, device=device)
        c = torch.zeros(batch_dim, idx_dim, device=device)
        with self.assertRaises(IndexError):
            a.index_copy_(1, idx, c)

        # Too small (negative indices)
        idx = torch.full((idx_dim,), -1, device=device)
        with self.assertRaises(IndexError):
            a.index_copy_(1, idx, c)

        # Too small (very negative indices) - they should be unsupported even
        # when support for negative indices is implemented for index_copy_
        idx = torch.full((idx_dim,), -tgt_dim - 1, device=device)
        with self.assertRaises(IndexError):
            a.index_copy_(1, idx, c)

    def _prepare_data_for_index_copy_and_add_deterministic(
        self, dim: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (dim >= 0 and dim < 3)
        a = [5, 4, 3]
        a[dim] = 2000
        x = torch.zeros(a, device=device)
        b = a.copy()
        elems = a[dim] * 20
        b[dim] = elems
        src = torch.rand(b, device=device)
        index = torch.randint(a[dim], (elems,), device=device)
        return (x, index, src)

    # FIXME: move to test indexing
    @onlyNativeDeviceTypes
    def test_index_copy_deterministic(self, device: torch.device) -> None:
        for dim in range(3):
            x, index, src = self._prepare_data_for_index_copy_and_add_deterministic(dim, device)
            with DeterministicGuard(True):
                y0 = torch.index_copy(x, dim, index, src)

            x0 = x.detach().clone()
            index_list = index.tolist()
            for i in range(len(index_list)):
                if dim == 0:
                    x0[index_list[i], :, :] = src[i, :, :]
                elif dim == 1:
                    x0[:, index_list[i], :] = src[:, i, :]
                elif dim == 2:
                    x0[:, :, index_list[i]] = src[:, :, i]

            self.assertEqual(x0, y0, atol=0, rtol=0)

    # FIXME: move to test indexing
    @onlyNativeDeviceTypes
    def test_index_add_deterministic(self, device: torch.device) -> None:
        for dim in range(3):
            x, index, src = self._prepare_data_for_index_copy_and_add_deterministic(dim, device)
            alpha = random.random() + 1
            # on CPU it should be deterministic regardless of the deterministic mode
            with DeterministicGuard(True):
                y0 = torch.index_add(x, dim, index, src, alpha=alpha)
                for _ in range(3):
                    y = torch.index_add(x, dim, index, src, alpha=alpha)
                    self.assertEqual(y, y0, atol=0, rtol=0)

            with DeterministicGuard(False):
                for _ in range(3):
                    y_nd = torch.index_add(x, dim, index, src, alpha=alpha)
                    self.assertEqual(y_nd, y0, atol=1e-3, rtol=1e-5)

    # FIXME: find a test suite for the put operator
    @onlyNativeDeviceTypes
    def test_index_put_non_accumulate_deterministic(self, device) -> None:
        with DeterministicGuard(True):
            for i in range(3):
                m = random.randint(10, 20)
                elems = random.randint(20000, 30000)
                values = torch.rand(elems, device=device)
                indices = torch.randint(m, (elems,), device=device)
                input = torch.rand(m, device=device)
                output = input.index_put((indices,), values, accumulate=False)

                input_list = input.tolist()
                indices_list = indices.tolist()
                values_list = values.tolist()
                for i, v in zip(indices_list, values_list):
                    input_list[i] = v

                self.assertEqual(output, input_list)

    # FIXME: move to test indexing
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @skipIfMPS
    def test_index_fill(self, device, dtype):
        x = torch.tensor([[1, 2], [4, 5]], dtype=dtype, device=device)
        index = torch.tensor([0], device=device)
        x.index_fill_(1, index, 0)
        self.assertEqual(x, torch.tensor([[0, 2], [0, 5]], dtype=dtype, device=device))
        if not x.is_complex() and not device == "meta":
            with self.assertRaisesRegex(RuntimeError, r"Scalar"):
                x.index_fill_(1, index, 1 + 1j)
        # Make sure that the result stays 0-dim while applied to
        # a 0-dim input
        x = torch.tensor(1, dtype=dtype, device=device)
        self.assertEqual(0, x.index_fill(0, index, -1).dim())
        self.assertEqual(0, x.index_fill_(0, index, -1).dim())

    # FIXME: move to test indexing
    # The test fails for zero-dimensional tensors on XLA
    @onlyNativeDeviceTypes
    @dtypes(*all_types_complex_float8_and(torch.half, torch.bool, torch.bfloat16))
    def test_index_select(self, device, dtype):
        num_src, num_out = 3, 5

        def make_arg(batch_sizes, n, dim, contig):
            size_arg = batch_sizes[:dim] + (n,) + batch_sizes[dim:]
            return make_tensor(size_arg, dtype=dtype, device=device, low=None, high=None, noncontiguous=not contig)

        def ref_index_select(src, dim, idx):
            # some types not supported on numpy
            not_np_dtypes = (torch.bfloat16, torch.float8_e5m2, torch.float8_e5m2fnuz, torch.float8_e4m3fn, torch.float8_e4m3fnuz)
            if dtype in not_np_dtypes:
                src = src.float()
            out = torch.from_numpy(np.take(src.cpu().numpy(), idx.cpu().numpy(), axis=dim))
            if dtype in not_np_dtypes:
                out = out.to(device=device, dtype=dtype)
            return out

        for src_contig, idx_contig in product([True, False], repeat=2):
            for other_sizes in ((), (4, 5)):
                for dim in range(len(other_sizes)):
                    src = make_arg(other_sizes, num_src, dim, src_contig)
                    idx = make_tensor(
                        (num_out,), dtype=torch.int64, device=device, low=0, high=num_src, noncontiguous=not idx_contig
                    )
                    out = torch.index_select(src, dim, idx)
                    out2 = ref_index_select(src, dim, idx)
                    self.assertEqual(out, out2)

        for idx_type in (torch.int32, torch.int64):
            other_sizes = (3, 2)
            dim = 1
            src = make_arg(other_sizes, num_src, dim, True)
            idx = make_tensor((num_out,), dtype=idx_type, device=device, low=0, high=num_src, noncontiguous=False)
            out = torch.index_select(src, dim, idx)
            out2 = ref_index_select(src, dim, idx)
            self.assertEqual(out, out2)

        # Create the 4 possible combinations of scalar sizes for index / source
        scalars = ((make_tensor(size_s, dtype=dtype, device=device),
                    torch.zeros(size_i, dtype=torch.int64, device=device))
                   for size_s, size_i in product([(), (1,)], repeat=2))
        for source, idx in scalars:
            out = source.index_select(0, idx)
            self.assertEqual(out.item(), source.item())

    # FIXME: find a test suite for the take operator
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_take(self, device, dtype):
        idx_size = (4,)

        make_arg = partial(make_tensor, device=device, dtype=dtype)
        make_idx = partial(make_tensor, low=0, device=device, dtype=torch.int64)

        def ref_take(src, idx):
            if dtype == torch.bfloat16:
                src = src.half()
            src = src.cpu().numpy()
            idx = idx.cpu().numpy()
            out = torch.from_numpy(np.take(src, idx)).to(device=device, dtype=dtype)
            return out

        for src_contig, idx_contig, idx_reshape in product([True, False], repeat=3):
            for src_size in ((5,), (4, 5)):
                src = make_arg(src_size, noncontiguous=not src_contig)
                idx = make_idx(idx_size, high=src.numel(), noncontiguous=not idx_contig)
                if idx_reshape:
                    idx = idx.reshape(2, 2)
                out = torch.take(src, idx)
                out2 = ref_take(src, idx)
                self.assertEqual(out, out2)

        # Create the 4 possible combinations of scalar sizes for source / index
        for size_s, size_i in product([(), (1,)], repeat=2):
            source = make_arg(size_s)
            idx = make_idx(size_i, high=1)
            out = source.take(idx)
            self.assertEqual(out.item(), source.item())

    # FIXME: find a test suite for the put operator
    # The bool instance does not work on GPU. See
    # https://github.com/pytorch/pytorch/issues/54317
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_put(self, device, dtype):
        src_size = (4,)

        make_arg = partial(make_tensor, device=device, dtype=dtype)
        make_idx = partial(make_tensor, low=0, device=device, dtype=torch.int64)

        def ref_put(dst, idx, src, accumulate):
            new_dst = dst.clone(memory_format=torch.contiguous_format).view(-1)
            new_idx = idx.contiguous().view(-1)
            new_src = src.contiguous().view(-1)
            method = new_dst.index_add_ if accumulate else new_dst.index_copy_
            return method(0, new_idx, new_src).view_as(dst)

        for dst_contig, src_contig, idx_contig, idx_reshape, accumulate in product([True, False], repeat=5):
            for dst_size in ((5,), (4, 5)):
                dst = make_arg(dst_size, noncontiguous=not dst_contig)
                src = make_arg(src_size, noncontiguous=not src_contig)

                # If accumulate=True, `put_` should be deterministic regardless of the inputs on CPU
                # On CUDA it may not be, but the test has enough tolerance to account for this
                if accumulate:
                    idx = make_idx(src_size, high=dst.numel())
                else:
                    idx = torch.randperm(dst.numel(), dtype=torch.int64, device=device)[:src_size[0]]
                if not idx_contig:
                    idx = torch.repeat_interleave(idx, 2, dim=-1)[..., ::2]
                if idx_reshape:
                    idx = idx.reshape(2, 2)
                out = torch.put(dst, idx, src, accumulate)
                # out-place
                reference = ref_put(dst, idx, src, accumulate)
                self.assertEqual(out, reference)

                # in-place
                dst.put_(idx, src, accumulate)
                self.assertEqual(dst, reference)


        # Create the 8 possible combinations of scalar sizes for target / index / source
        scalars = ((make_arg(size_t),
                    make_idx(size_i, high=1),
                    make_arg(size_s))
                   for size_t, size_i, size_s in product([(), (1,)], repeat=3))
        for (dest, idx, source), accumulate in product(scalars, [True, False]):
            dest_init = dest.clone()
            # out-place
            out = torch.put(dest, idx, source, accumulate=accumulate)
            # in-place
            dest1 = dest.clone()
            dest1.put_(idx, source, accumulate=accumulate)
            for d in [out, dest1]:
                if accumulate:
                    self.assertEqual(d.item(), (dest_init + source).item())
                else:
                    self.assertEqual(d.item(), source.item())

        # Empty case
        dest = make_arg((3, 2))
        reference = dest.clone()
        idx = make_idx((0,), high=1)
        source = make_arg((0,))
        for accumulate in [True, False]:
            out = torch.put(dest, idx, source, accumulate=accumulate)
            self.assertEqual(out, reference)
            dest.put_(idx, source, accumulate=accumulate)
            self.assertEqual(dest, reference)

    # FIXME: find a test suite for the put operator
    # The bool instance does not work on GPU. See
    # https://github.com/pytorch/pytorch/issues/54317
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_put_accumulate(self, device, dtype):
        # Test for parallel adds with accumulate == True
        low_precision = dtype == torch.half or dtype == torch.bfloat16
        # Less numbers to avoid overflow with low_precision
        # Grainsize is 3000 for the for_loop to be parallelized on CPU
        sizes = ((100,)) if low_precision else ((200,), (3002,))
        # Bfloat16 has a particularly bad performance here
        # This operation is nondeterministic on GPU, so we are generous with the rtol
        rtol, atol = (1e-1, 1e-2) if low_precision else (1e-3, 1e-4)

        make_arg = partial(make_tensor, low=-2, high=3, device=device, dtype=dtype)
        # Dump everything into the 0-th position
        make_idx = partial(torch.zeros, device=device, dtype=torch.int64)
        args = ((make_idx(size), make_arg(size)) for size in sizes)

        for idx, source in args:
            orig = make_arg((1,))
            out = orig.put(idx, source, accumulate=True)
            self.assertEqual(out, orig + source.sum(), rtol=rtol, atol=atol)

    # FIXME: find a test suite for the take operator
    @skipIfMPS
    def test_take_empty(self, device):
        for input_shape in [(0,), (0, 1, 2, 0), (1, 2, 3)]:
            for indices_shape in [(0,), (0, 1, 2, 0)]:
                input = torch.empty(input_shape, device=device)
                indices = torch.empty(indices_shape, dtype=torch.int64, device=device)
                self.assertEqual(indices, torch.take(input, indices), exact_dtype=False)

    # FIXME: find a test suite for the put operator
    def test_put_empty(self, device):
        for dst_shape in [(0,), (0, 1, 2, 0), (1, 2, 3)]:
            for indices_shape in [(0,), (0, 1, 2, 0)]:
                for accumulate in [False, True]:
                    dst = torch.randn(dst_shape, device=device)
                    indices = torch.empty(indices_shape, dtype=torch.int64, device=device)
                    src = torch.randn(indices_shape, device=device)
                    self.assertEqual(dst, dst.put_(indices, src, accumulate=accumulate))

    # FIXME: port to test_scatter_gather_ops.py
    def scatter_allow_reduce(self, device, dtype, reduceop):
        device_type = torch.device(device).type
        return device_type != 'cuda' or (reduceop == 'multiply' and dtype.is_floating_point)

    @dtypes(*floating_and_complex_types())
    @dtypesIfCPU(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @dtypesIfCUDA(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_scatter_reduce_operations_to_large_input(self, device, dtype):
        index = torch.tensor([[1], [2]], device=device, dtype=torch.long)
        test_data = [
            (torch.zeros(4, 4, device=device, dtype=dtype),
             torch.ones(2, 2, device=device, dtype=dtype),
             torch.tensor([[0, 0, 0, 0],
                           [1, 0, 0, 0],
                           [1, 0, 0, 0],
                           [0, 0, 0, 0]],
                          device=device, dtype=dtype), "add"),
            (torch.tensor([2], device=device, dtype=dtype).repeat(4, 4),
             torch.tensor([6], device=device, dtype=dtype).repeat(2, 2),
             torch.tensor([[2, 2, 2, 2],
                           [12, 2, 2, 2],
                           [12, 2, 2, 2],
                           [2, 2, 2, 2]], device=device, dtype=dtype), "multiply"),
        ]

        for input, src, result, operation in test_data:
            if not self.scatter_allow_reduce(device, dtype, operation):
                continue
            input.scatter_(0, index, src, reduce=operation)
            self.assertEqual(input, result)

    @dtypes(*floating_and_complex_types())
    @dtypesIfCPU(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @dtypesIfCUDA(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_scatter_reduce_scalar(self, device, dtype):
        index = torch.tensor([[1], [2]], device=device, dtype=torch.long)
        test_data = [
            (torch.zeros(4, 4, device=device, dtype=dtype), 1,
             torch.tensor([[0, 0, 0, 0],
                           [1, 0, 0, 0],
                           [1, 0, 0, 0],
                           [0, 0, 0, 0]],
                          device=device, dtype=dtype), "add"),
            (torch.tensor([2], device=device, dtype=dtype).repeat(4, 4), 2,
             torch.tensor([[2, 2, 2, 2],
                           [4, 2, 2, 2],
                           [4, 2, 2, 2],
                           [2, 2, 2, 2]], device=device, dtype=dtype), "multiply"),
        ]

        for input, src, result, operation in test_data:
            if not self.scatter_allow_reduce(device, dtype, operation):
                continue
            input.scatter_(0, index, src, reduce=operation)
            self.assertEqual(input, result)

    # FIXME: port to test_scatter_gather_ops.py
    # TODO: remove this after scatter_add_ is deprecated.
    def test_scatter_add_non_unique_index(self, device):
        height = 2
        width = 65536
        input = torch.ones(height, width, device=device)
        index = torch.zeros(height, width, dtype=torch.long, device=device)
        src = torch.ones(height, width, device=device)
        input.scatter_add_(0, index, src)

        self.assertEqual(input,
                         torch.tensor([[3], [1]], device=device,
                                      dtype=torch.float32).repeat(1, width))

    @dtypes(*floating_and_complex_types())
    @dtypesIfCPU(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @dtypesIfCUDA(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_scatter_reduce_non_unique_index(self, device, dtype):
        height = 2
        width = 2
        index = torch.zeros(height, width, dtype=torch.long, device=device)
        test_data = [
            (torch.ones(height, width, device=device, dtype=dtype),
             torch.ones(height, width, device=device, dtype=dtype),
             torch.tensor([[3], [1]], device=device, dtype=dtype).repeat(1, width), "add"),
            (torch.tensor([2], device=device, dtype=dtype).repeat(height, width),
             torch.tensor([2], device=device, dtype=dtype).repeat(height, width),
             torch.tensor([[8], [2]], device=device,
                          dtype=dtype).repeat(1, width), "multiply"),
        ]

        for input, src, result, operation in test_data:
            if not self.scatter_allow_reduce(device, dtype, operation):
                continue
            input.scatter_(0, index, src, reduce=operation)
            self.assertEqual(input, result, msg=f"result: {result} input: {input} method: {str(operation)}")

    @onlyCUDA
    @dtypes(*complex_types())
    def test_scatter_reduce_multiply_unsupported_dtypes(self, device, dtype):
        height = 2
        width = 2
        index = torch.zeros(height, width, dtype=torch.long, device=device)
        input = torch.ones(height, width, device=device, dtype=dtype)
        src = torch.ones(height, width, device=device, dtype=dtype)
        with self.assertRaises(RuntimeError):
            input.scatter_(0, index, src, reduce="multiply")

    # FIXME: port to test_scatter_gather_ops.py
    def test_scatter_to_large_input(self, device):
        input = torch.zeros(4, 4, device=device)
        src = torch.ones(2, 2, device=device)
        index = torch.tensor([[1], [2]], device=device, dtype=torch.long)
        input.scatter_(0, index, src)
        self.assertEqual(input, torch.tensor([[0, 0, 0, 0],
                                              [1, 0, 0, 0],
                                              [1, 0, 0, 0],
                                              [0, 0, 0, 0]], device=device, dtype=torch.float32))

    # FIXME: port to test_scatter_gather_ops.py
    def test_scatter_add_to_large_input(self, device):
        input = torch.zeros(4, 4, device=device)
        src = torch.ones(2, 2, device=device)
        index = torch.tensor([[1], [2]], device=device, dtype=torch.long)
        input.scatter_add_(0, index, src)
        self.assertEqual(input, torch.tensor([[0, 0, 0, 0],
                                              [1, 0, 0, 0],
                                              [1, 0, 0, 0],
                                              [0, 0, 0, 0]], device=device, dtype=torch.float32))

    # FIXME: port to test_scatter_gather_ops.py
    def test_scatter_bool(self, device):
        x = torch.tensor([[True, True, True], [True, True, True]], device=device)
        res = torch.zeros(3, 3, dtype=torch.bool, device=device)
        res = res.scatter_(0, torch.tensor([[0, 1, 2], [0, 1, 2]], device=device), x)
        self.assertEqual(res, torch.tensor([[True, False, False],
                                            [False, True, False],
                                            [False, False, True]], device=device))

    # FIXME: port to test_scatter_gather_ops.py
    def test_scatter_add_bool(self, device):
        x = torch.tensor([[True, True, True, True, True], [True, True, True, True, True]], device=device)
        res = torch.zeros(3, 5, dtype=torch.bool, device=device)
        res = res.scatter_add_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], device=device), x)
        self.assertEqual(res, torch.tensor([[True, True, True, True, True],
                                            [False, True, False, True, False],
                                            [True, False, True, False, True]], device=device))

    # FIXME: find a test suite for the masked scatter operator
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_masked_scatter(self, device, dtype):
        dt = dtype
        num_copy, num_dest = 3, 10
        dest = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dt, device=device)
        dest2 = dest.clone()
        dest_ones = dest.clone()
        dest_ones_expected = dest.clone()
        src = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dt, device=device)
        src_ones = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=dt, device=device)
        mask = torch.tensor((0, 0, 0, 0, 1, 0, 1, 0, 1, 0), dtype=torch.bool, device=device)

        dest.masked_scatter_(mask, src)
        j = 0
        for i in range(num_dest):
            if mask[i]:
                dest2[i] = src[j]
                dest_ones_expected[i] = src_ones[j]
                j += 1
        self.assertEqual(dest, dest2, atol=0, rtol=0)

        dest_ones.masked_scatter_(mask, src_ones)
        self.assertEqual(dest_ones, dest_ones_expected, atol=0, rtol=0)

        # Bound checking in CUDA is done inside a kernel
        # in order to avoid synchronization, but this means
        # we can not clear the failures. So there is no way
        # to test it then recover.
        if self.device_type != 'cuda':
            # make src smaller. this should fail
            src = torch.zeros(num_copy - 1, dtype=dt, device=device)
            with self.assertRaises(RuntimeError):
                dest.masked_scatter_(mask, src)

        # empty tensor
        dest = torch.empty((5, 0, 5), dtype=dt, device=device)
        mask = torch.ones_like(dest, dtype=torch.bool, device=device)
        src = torch.empty((0,), dtype=dt, device=device)
        dest.masked_scatter_(mask, src)

        dest = torch.empty((5, 0, 5), dtype=dt, device=device)
        mask = torch.ones((5, 1, 5), dtype=torch.bool, device=device)
        src = torch.empty((0,), dtype=dt, device=device)
        dest.masked_scatter_(mask, src)

    # FIXME: find a test suite for the masked scatter operator
    @skipIfMPS
    def test_masked_scatter_bool_tensor(self, device):
        src = torch.tensor([True, True, True], device=device)
        dst = torch.tensor([False, False, False], device=device)
        mask = torch.tensor([False, True, False], device=device)

        dst.masked_scatter_(mask, src)
        self.assertEqual(dst, torch.tensor([False, True, False], device=device))

        mask = torch.tensor([True, False, True], device=device)
        dst = dst.masked_scatter(mask, src)
        self.assertEqual(dst, torch.tensor([True, True, True], device=device))

    # FIXME: find a test suite for the masked scatter operator
    #   test_scatter_gather_ops or test_masked_ops?
    @onlyCUDA
    @largeTensorTest('30GB')
    def test_masked_scatter_large_tensor(self, device):
        t_cpu = torch.empty(2**31 + 1, dtype=torch.bool).random_()
        t = t_cpu.to(device)
        result_cpu = t_cpu.masked_scatter(t_cpu, t_cpu)
        result = t.masked_scatter(t, t)
        self.assertEqual(result, result_cpu)

    # FIXME: find a test suite for the masked select operator
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_masked_select(self, device, dtype):
        for maskType in integral_types_and(torch.bool):
            num_src = 10
            src = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dtype, device=device)
            mask = torch.randint(2, (num_src,), device=device, dtype=maskType)

            if maskType is not torch.bool:
                with self.assertRaisesRegex(RuntimeError, r'expected BoolTensor for mask'):
                    dst = src.masked_select(mask)
                continue
            else:
                dst = src.masked_select(mask)
            dst2 = []
            for i in range(num_src):
                if mask[i]:
                    dst2 += [src[i]]
            self.assertEqual(dst, torch.tensor(dst2), atol=0, rtol=0)

            dst3 = torch.empty(0, device=device, dtype=dtype)
            torch.masked_select(src, mask, out=dst3)
            self.assertEqual(dst3, torch.tensor(dst2, dtype=dst3.dtype), atol=0, rtol=0)

        # Since half on CPU is not supported, need to skip the remaining test cases
        if dtype == torch.half and torch.device(device).type == 'cpu':
            return

        # Ensure that masks are expanded to match tensor properly
        a = torch.rand(100, 100, device=device).mul(100).to(dtype)
        mask_first_el_each_row = torch.zeros(100, device=device, dtype=torch.bool)
        mask_first_el_each_row[0] = True
        a_masked = a.masked_select(mask_first_el_each_row)
        self.assertEqual(a_masked, a[:, 0])

        mask_first_row = torch.zeros(100, 1, device=device, dtype=torch.bool)
        mask_first_row[0][0] = True
        a_masked = a.masked_select(mask_first_row)
        self.assertEqual(a_masked, a[0, :])

        # Ensure that tensor is expanded to match mask properly
        a = torch.rand(100, device=device).mul(100).to(dtype)
        mask_copy_3_times = torch.tensor([[True], [True], [False], [True]], device=device)
        a_masked = a.masked_select(mask_copy_3_times)
        self.assertEqual(a_masked, a.unsqueeze(0).expand(3, 100).flatten())

    # FIXME: find a test suite for the masked select operator
    def test_masked_select_discontiguous(self, device):
        for size in (10, 200):
            vals = torch.rand(size, size, device=device)
            mask = torch.full((size, size), False, dtype=torch.bool, device=device)
            mask[:, ::2] = True
            vals_list = (vals, vals.t())
            mask_list = (mask, mask.t())
            out_dc = torch.empty(size * size, device=device)[::2]
            for v, m in product(vals_list, mask_list):
                if m.is_contiguous():
                    expected = v[:, ::2].clone().reshape((-1, ))
                else:
                    expected = v[::2].clone().reshape((-1, ))
                out = torch.masked_select(v, m)
                self.assertEqual(out, expected, atol=0, rtol=0)
                torch.masked_select(v, m, out=out_dc)
                self.assertEqual(out_dc, expected, atol=0, rtol=0)

    # FIXME: find a test suite for the masked fill operator
    @dtypes(*product(all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16), (torch.uint8, torch.bool)))
    def test_masked_fill(self, device, dtypes):
        dtype = dtypes[0]
        mask_dtype = dtypes[1]

        num_dest = 10
        dst = torch.zeros(num_dest, dtype=dtype)
        mask = torch.randint(2, (num_dest,), dtype=mask_dtype)
        val = random.random()
        dst2 = dst.clone()

        if mask_dtype is not torch.bool:
            with self.assertRaisesRegex(RuntimeError, 'only supports boolean masks'):
                dst.masked_fill_(mask, val)
            return

        dst.masked_fill_(mask, val)
        for i in range(num_dest):
            if mask[i]:
                dst2[i] = val
        self.assertEqual(dst, dst2, atol=0, rtol=0)

        # test non-contiguous case
        dst = ((torch.randn(num_dest, num_dest, num_dest) * 10).to(dtype)).permute((2, 0, 1))
        dst2 = dst.contiguous()
        if dtype.is_complex:
            mask = dst.abs() > 0
        else:
            mask = dst > 0
        self.assertTrue(not dst.is_contiguous())
        self.assertTrue(dst2.is_contiguous())
        dst.masked_fill_(mask.to(mask_dtype), val)
        dst2.masked_fill_(mask.to(mask_dtype), val)
        self.assertEqual(dst, dst2, atol=0, rtol=0)

    # FIXME: find a test suite for the masked fill operator
    def test_masked_fill_bool_tensor(self, device):
        dst = torch.tensor([True, False, True], device=device)
        mask = torch.tensor([False, True, False], device=device)

        dst.masked_fill_(mask, True)
        self.assertEqual(dst, torch.tensor([True, True, True], device=device))

        dst = dst.masked_fill(mask, False)
        self.assertEqual(dst, torch.tensor([True, False, True], device=device))

    def test_tensor_shape_empty(self, device):
        x = torch.randn((0, 1, 3, 0), device=device)
        # flatten
        self.assertEqual((0,), torch.flatten(x, 0, 3).shape)
        self.assertEqual((0, 0), torch.flatten(x, 0, 2).shape)
        self.assertEqual((0, 3, 0), torch.flatten(x, 1, 2).shape)

        # squeeze, unsqueeze
        self.assertEqual((0, 1, 1, 3, 0), torch.unsqueeze(x, 1).shape)
        self.assertEqual((0, 3, 0), torch.squeeze(x, 1).shape)
        self.assertEqual((0, 3, 0), torch.squeeze(x).shape)

        # transpose, t
        self.assertEqual((0, 0, 3, 1), torch.transpose(x, 1, 3).shape)
        y = torch.randn((5, 0), device=device)
        self.assertEqual((0, 5), y.t().shape)

        # select
        self.assertEqual((0, 1, 0), torch.select(x, 2, 2).shape)

        # repeat, permute
        self.assertEqual((9, 0, 5, 6, 0), x.repeat(9, 7, 5, 2, 3).shape)
        self.assertEqual((3, 0, 0, 1), x.permute(2, 3, 0, 1).shape)

        # diagonal, diagflat
        self.assertEqual((0,), torch.diagonal(torch.randn((5, 0), device=device)).shape)
        self.assertEqual((0,), torch.diagonal(torch.randn((0, 5), device=device)).shape)
        # off the end offsets are valid
        self.assertEqual((0,), torch.diagonal(torch.randn((5, 0), device=device), offset=1).shape)
        self.assertEqual((0,), torch.diagonal(torch.randn((0, 5), device=device), offset=1).shape)
        # check non-zero sized offsets off the end
        self.assertEqual((5, 6, 0), torch.diagonal(torch.randn((3, 4, 5, 6), device=device), offset=45252).shape)
        self.assertEqual((5, 6, 0), torch.diagonal(torch.randn((3, 4, 5, 6), device=device), offset=-45252).shape)

        self.assertEqual((0, 0), torch.diagflat(torch.tensor([], device=device)).shape)
        self.assertEqual(torch.zeros(1, 1), torch.diagflat(torch.tensor([], device=device), offset=1))
        self.assertEqual((0, 0), torch.diagflat(torch.tensor([[]], device=device)).shape)
        self.assertEqual(torch.zeros(1, 1), torch.diagflat(torch.tensor([[]], device=device), offset=1))

        # stack, split, chunk
        self.assertEqual((4, 0, 1, 3, 0), torch.stack((x, x, x, x)).shape)
        self.assertEqual([(0, 1, 3, 0)],
                         [z.shape for z in torch.chunk(x, 1, dim=0)])

        self.assertEqual([(0, 1, 3, 0), ] * 3, [z.shape for z in torch.chunk(x, 3, dim=0)])
        self.assertEqual([(0, 1, 1, 0), ] * 3, [z.shape for z in torch.chunk(x, 3, dim=2)])

        # NOTE: split_with_sizes behaves differently than NumPy in that it
        # takes sizes rather than offsets
        self.assertEqual([(0, 1, 0, 0), (0, 1, 1, 0), (0, 1, 2, 0)],
                         [z.shape for z in torch.split(x, (0, 1, 2), dim=2)])

        self.assertRaises(RuntimeError, lambda: torch.split(x, 0, dim=1))
        # This is strange because the split size is larger than the dim size, but consistent with
        # how split handles that case generally (when no 0s are involved).
        self.assertEqual([(0, 1, 3, 0)], [z.shape for z in torch.split(x, 1, dim=0)])
        self.assertEqual([(0, 1, 3, 0)], [z.shape for z in torch.split(x, 0, dim=0)])

    # functions that operate over a dimension but don't reduce.
    def test_dim_function_empty(self, device):
        shape = (0, 1, 2, 0)
        x = torch.randn(shape, device=device)

        # size stride
        self.assertEqual(0, x.size(3))
        self.assertEqual(2, x.size(2))
        self.assertEqual(2, x.stride(0))
        self.assertEqual(1, x.stride(2))

        self.assertEqual(x, torch.nn.functional.glu(x, 0))
        self.assertEqual((0, 1, 1, 0), torch.nn.functional.glu(x, 2).shape)

        # softmax, logsoftmax
        self.assertEqual(x, torch.nn.functional.softmax(x, 0))
        self.assertEqual(x, torch.nn.functional.softmax(x, 2))
        self.assertEqual(x, torch.nn.functional.softmax(x, 3))

        self.assertEqual(x, torch.nn.functional.log_softmax(x, 0))
        self.assertEqual(x, torch.nn.functional.log_softmax(x, 2))
        self.assertEqual(x, torch.nn.functional.log_softmax(x, 3))

        # cumsum, cumprod, cummax, cummin
        self.assertEqual(shape, torch.cumsum(x, 0).shape)
        self.assertEqual(shape, torch.cumsum(x, 2).shape)
        self.assertEqual(shape, torch.cumprod(x, 0).shape)
        self.assertEqual(shape, torch.cumprod(x, 2).shape)
        self.assertEqual(shape, torch.cummax(x, 0)[0].shape)
        self.assertEqual(shape, torch.cummax(x, 2)[0].shape)
        self.assertEqual(shape, torch.cummin(x, 0)[0].shape)
        self.assertEqual(shape, torch.cummin(x, 2)[0].shape)
        self.assertEqual(shape, torch.logcumsumexp(x, 0).shape)
        self.assertEqual(shape, torch.logcumsumexp(x, 2).shape)

        # flip
        self.assertEqual(x, x.flip(0))
        self.assertEqual(x, x.flip(2))

        # roll
        self.assertEqual(x, x.roll(0, 1).roll(0, -1))
        self.assertEqual(x, x.roll(1, x.size(1)))
        self.assertEqual(x, x.roll(1))
        self.assertEqual(x, x.roll((1, 1), (3, 1)))

        # unbind
        self.assertEqual((), x.unbind(0))
        self.assertEqual((torch.empty((0, 1, 0), device=device), torch.empty((0, 1, 0), device=device)),
                         x.unbind(2))

        # cross
        y = torch.randn((0, 1, 3, 0), device=device)
        self.assertEqual(y.shape, torch.cross(y, y).shape)

        # renorm
        self.assertEqual(shape, torch.renorm(x, 1, 0, 5).shape)
        self.assertEqual(shape, torch.renorm(x, 1, 2, 5).shape)

        # sort
        self.assertEqual([shape, shape], [z.shape for z in torch.sort(x, dim=0)])
        self.assertEqual([shape, shape], [z.shape for z in torch.sort(x, dim=2)])

        # topk
        self.assertEqual([shape, shape], [z.shape for z in torch.topk(x, 0, dim=0)])
        self.assertEqual([(0, 1, 1, 0), (0, 1, 1, 0)], [z.shape for z in torch.topk(x, 1, dim=2)])

        y = torch.randn((2, 3, 4), device=device)
        self.assertEqual([(2, 3, 0), (2, 3, 0)], [z.shape for z in torch.topk(y, 0)])

        # gather
        self.assertEqual(shape, torch.gather(x, 0, torch.empty(shape, dtype=torch.int64, device=device)).shape)
        self.assertEqual(shape, torch.gather(x, 2, torch.empty(shape, dtype=torch.int64, device=device)).shape)
        larger_shape = torch.empty((0, 1, 3, 0), dtype=torch.int64, device=device)
        self.assertEqual(larger_shape.shape, torch.gather(x, 2, larger_shape).shape)
        smaller_shape = torch.empty((0, 1, 0, 0), dtype=torch.int64, device=device)
        self.assertEqual(smaller_shape.shape, torch.gather(x, 2, smaller_shape).shape)
        y = torch.randn((2, 3, 4), device=device)
        self.assertEqual((0, 3, 4),
                         torch.gather(y, 0, torch.empty((0, 3, 4), dtype=torch.int64, device=device)).shape)

        # scatter, scatter_add
        for dim in [0, 2]:
            y = torch.randn(shape, device=device)
            y_src = torch.randn(shape, device=device)
            ind = torch.empty(shape, dtype=torch.int64, device=device)
            self.assertEqual(shape, y.scatter_(dim, ind, y_src).shape)
            self.assertEqual(shape, y.scatter_add_(dim, ind, y_src).shape)

        z = torch.randn((2, 3, 4), device=device)
        z_src = torch.randn((2, 3, 4), device=device)
        self.assertEqual(z, z.scatter_(2, torch.empty((2, 3, 0), dtype=torch.int64, device=device), z_src))
        self.assertEqual(z, z.scatter_add_(2, torch.empty((2, 3, 0), dtype=torch.int64, device=device), z_src))

        # index_fill, index_copy, index_add
        c = x.clone()
        c_clone = c.clone()
        ind_empty = torch.tensor([], dtype=torch.int64, device=device)
        ind_01 = torch.tensor([0, 1], dtype=torch.int64, device=device)
        self.assertEqual(c_clone, c.index_fill_(0, ind_empty, -1))
        self.assertEqual(c_clone, c.index_fill_(2, ind_empty, -1))
        self.assertEqual(c_clone, c.index_fill_(2, ind_01, -1))
        self.assertEqual(c_clone, c.index_copy_(0, ind_empty, torch.empty((0, 1, 2, 0), device=device)))
        self.assertEqual(c_clone, c.index_copy_(2, ind_empty, torch.empty((0, 1, 0, 0), device=device)))
        self.assertEqual(c_clone, c.index_copy_(2, ind_01, torch.empty((0, 1, 2, 0), device=device)))
        self.assertEqual(c_clone, c.index_add_(0, ind_empty, torch.empty((0, 1, 2, 0), device=device)))
        self.assertEqual(c_clone, c.index_add_(2, ind_empty, torch.empty((0, 1, 0, 0), device=device)))
        self.assertEqual(c_clone, c.index_add_(2, ind_01, torch.empty((0, 1, 2, 0), device=device)))

        c = torch.randn((0, 1, 2), device=device)
        c_clone = c.clone()
        self.assertEqual(c_clone, c.index_fill_(0, ind_empty, -1))
        self.assertEqual(c_clone, c.index_copy_(0, ind_empty, torch.empty((0, 1, 2), device=device)))
        self.assertEqual(c_clone, c.index_add_(0, ind_empty, torch.empty((0, 1, 2), device=device)))
        self.assertEqual(c_clone, c.index_fill_(0, ind_empty, -1))
        self.assertEqual(c_clone, c.index_copy_(0, ind_empty, torch.empty((0, 1, 2), device=device)))
        self.assertEqual(c_clone, c.index_add_(0, ind_empty, torch.empty((0, 1, 2), device=device)))

        # index fill/copy/add non-empty
        z = torch.randn((2, 3, 4), device=device)
        self.assertEqual(z, z.index_fill_(0, ind_empty, -1))
        z = torch.randn((2, 3, 4), device=device)
        self.assertEqual(z, z.index_copy_(0, ind_empty, torch.empty((0, 3, 4), device=device)))
        z = torch.randn((2, 3, 4), device=device)
        self.assertEqual(z, z.index_add_(0, ind_empty, torch.empty((0, 3, 4), device=device)))

        # index_select
        self.assertEqual(x, x.index_select(0, ind_empty))
        self.assertEqual((0, 1, 0, 0), x.index_select(2, ind_empty).shape)
        self.assertEqual(x, x.index_select(2, ind_01))
        z = torch.randn((2, 3, 4), device=device)  # non-empty
        self.assertEqual((0, 3, 4), z.index_select(0, ind_empty).shape)
        c = torch.randn((0, 1, 2), device=device)
        self.assertEqual(c, c.index_select(0, ind_empty))
        c = torch.randn((0, 1, 2), device=device)
        self.assertEqual(c, c.index_select(0, ind_empty))
        w = torch.randn((0, 3), device=device)
        self.assertEqual((0, 2), w.index_select(1, ind_01).shape)
        w = torch.randn((3, 0), device=device)
        self.assertEqual((2, 0), w.index_select(0, ind_01).shape)
        ind_01_int32 = torch.tensor([0, 1], dtype=torch.int32, device=device)
        self.assertEqual((2, 0), w.index_select(0, ind_01_int32).shape)
        s = torch.randn([], device=device)
        ind_0 = torch.tensor([0], dtype=torch.int32, device=device)
        self.assertEqual([], s.index_select(0, ind_0).shape)
        if device == 'cpu':
            w = torch.randn((0, 3), device=device)
            with self.assertRaisesRegex(RuntimeError, "self indexing axis dim should be positive"):
                torch.index_select(w, 0, ind_01)
            ind_05 = torch.tensor([0, 5], dtype=torch.int64, device=device)
            with self.assertRaisesRegex(RuntimeError, "INDICES element is out of DATA bounds"):
                torch.index_select(w, 1, ind_05)
            with self.assertRaisesRegex(RuntimeError, "Index to scalar can have only 1 value"):
                torch.index_select(s, 0, ind_empty)
        with self.assertRaisesRegex(RuntimeError, "Index to scalar can have only 1 value"):
            torch.ones([]).index_select(0, torch.Tensor([0, 0]).int())

    # FIXME: find a test suite for the pdist operator
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "sandcastle OOM with current tpx gpu/re configuration")
    @skipIfRocm
    @onlyCUDA
    @largeTensorTest('32GB', device='cpu')
    @largeTensorTest('5GB', device='cuda')
    def test_pdist_norm_large(self, device):
        # use dim0>=46342 for forward, see:
        # https://github.com/pytorch/pytorch/issues/30583
        # Compare output using GPU with the CPU implementation
        x = torch.randn(50000, 1, dtype=torch.float32)      # 50k * 4 bytes = 200 KB
        # Will require 1249975000 float32s
        expected_cpu = torch.pdist(x, p=2)                  # ~1250M * 4 bytes = 5 GB on CPU
        actual_cpu = torch.pdist(x.to(device), p=2).cpu()         # 5 GB on GPU + 5GB on CPU
        # Workaround for large memory overhead of self.assertTrue (see #84944)
        self.assertTrue(torch.allclose(expected_cpu, actual_cpu))  # ~20GB in allclose

    # FIXME: move to elementwise ternary test suite
    @onlyNativeDeviceTypes
    @dtypesIfCUDA(*set(get_all_math_dtypes('cuda')))
    @dtypes(*set(get_all_math_dtypes('cpu')))
    def test_addcdiv(self, device, dtype):
        # Returns floating or integral scalar corresponding to dtype
        def _number(floating, integer, dtype):
            if dtype in [torch.half, torch.float, torch.double, torch.bfloat16]:
                return floating
            elif dtype in [torch.cfloat, torch.cdouble]:
                return floating * (1 + 1j)
            else:
                return integer

        def non_zero_rand(size, dtype, device):
            if dtype.is_floating_point or dtype.is_complex:
                a = torch.rand(size=size, dtype=dtype, device=device)
            elif dtype == torch.uint8:
                a = torch.randint(1, 5, size=size, dtype=dtype, device=device)
            else:
                a = torch.randint(-5, 5, size=size, dtype=dtype, device=device)
            return a + (a == 0).to(dtype)

        def _test_addcdiv():
            a = non_zero_rand((2, 2), dtype=dtype, device=device)
            b = non_zero_rand((2, 2), dtype=dtype, device=device)
            c = non_zero_rand((2, 2), dtype=dtype, device=device)
            alpha = _number(0.5, 3, dtype)

            expected = a + (alpha * b) / c
            actual = torch.addcdiv(a, b, c, value=alpha)
            self.assertEqual(expected, actual)

            with self.assertWarnsOnceRegex(
                    UserWarning, "This overload of addcdiv is deprecated"):
                self.assertEqual(actual, torch.addcdiv(a, alpha, b, c))

        if not (dtype.is_floating_point or dtype.is_complex):
            # Integer division with addcdiv is prohibited
            with self.assertRaises(RuntimeError):
                _test_addcdiv()
        else:
            _test_addcdiv()

        if self.device_type == 'cuda' and dtype == torch.half:
            a = torch.tensor([60000.0], device=device, dtype=dtype)
            b = torch.tensor([60000.0], device=device, dtype=dtype)
            c = torch.tensor([1.0], device=device, dtype=dtype)
            out = torch.addcmul(a, b, c, value=-2)
            self.assertTrue(not (out.isnan() or out.isinf()))

    def test_nullary_op_mem_overlap(self, device):
        ops = (
            ("random_", ()),
            ("uniform_", ()),
            ("cauchy_", ()),
            ("log_normal_", ()),
            ("exponential_", ()),
            ("geometric_", (0.5,)),
            ("normal_", ()),
        )

        x = torch.rand((1, 3)).expand((3, 3))
        for op, args in ops:
            with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                getattr(x, op)(*args)

    # FIXME: move to an elementwise ternary test suite and make this an OpInfo test
    # https://github.com/pytorch/pytorch/issues/126474
    @xfailIfTorchDynamo
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/126474")
    @dtypes(torch.double)
    def test_ternary_op_mem_overlap(self, device, dtype):
        if device == "cpu" and TEST_WITH_TORCHINDUCTOR:
            self.skipTest("Failing on cpu")

        ops = [
            ("addcmul", True, True, 'cpu'),
            ("addcmul", True, True, 'cuda'),
            ("addcdiv", True, True, 'cpu'),
            ("addcdiv", True, True, 'cuda'),
            ("lerp", True, True, 'cpu'),
            ("lerp", True, True, 'cuda')
        ]

        for (fn, has_input_output_mem_overlap_check,
             has_internal_mem_overlap_check, dev) in ops:
            if dev != device:
                continue
            out_op = getattr(torch, fn)
            inplace_op = getattr(torch.Tensor, fn + '_')
            self.check_internal_mem_overlap(
                inplace_op, 3, dtype, device,
                expected_failure=not has_internal_mem_overlap_check)
            self.ternary_check_input_output_mem_overlap(out_op, dev,
                                                        expected_failure=not has_input_output_mem_overlap_check)

    @expectedFailureMeta  # RuntimeError not raised
    @dtypes(torch.double)
    @onlyNativeDeviceTypes
    def test_copy_mem_overlap(self, device, dtype):
        self.check_internal_mem_overlap(
            torch.Tensor.copy_, num_inputs=2, dtype=dtype, device=device)
        sz = 9
        doubles = torch.randn(2 * sz, dtype=dtype, device=device)
        self.unary_check_input_output_mem_overlap(
            doubles, sz, lambda input, out: out.copy_(input))

    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @onlyNativeDeviceTypes
    def test_index_add_mem_overlap(self, device):
        x = torch.rand((1,), device=device).expand((6,))
        y = torch.rand((6,), device=device)
        ind = torch.tensor([2, 1, 0], device=device)
        value = torch.rand((3,), device=device)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x.index_add_(0, ind, value)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            y.index_add_(0, ind, y[:3])
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.index_add_(0, ind, ind.clone())
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.index_add_(0, ind.clone(), ind)

    @onlyCUDA
    @skipCUDAIfNotRocm  # This UT throws an OOM error on CUDA
    def test_index_add_large_inputs(self, device):
        D = 6144
        x = torch.zeros([16384, D], device=device, dtype=torch.bfloat16)
        index = torch.randint(0, 16384, (1, 32, 16384), device=device, dtype=torch.int64)
        output = torch.ones([1, 32, 16384, D], device=device, dtype=torch.bfloat16)  # Use random values for test

        x_before = x.clone()
        # Manually update x_before to generate expected values
        for batch in range(output.shape[1]):  # Loop over batch size (32 in this case)
            for idx in range(output.shape[2]):  # Loop over index
                idx_val = index[0, batch, idx].item()
                x_before[idx_val] += output[0, batch, idx]

        # Run index_add to get actual values
        x.index_add_(0, index.view(-1), output.view(-1, D))

        self.assertEqual(x_before, x)

    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @onlyNativeDeviceTypes
    def test_index_copy_mem_overlap(self, device):
        x = torch.rand((1,), device=device).expand((6,))
        y = torch.rand((6,), device=device)
        ind = torch.tensor([2, 1, 0], device=device)
        value = torch.rand((3,), device=device)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x.index_copy_(0, ind, value)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            y.index_copy_(0, ind, y[:3])
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.index_copy_(0, ind, ind.clone())
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.index_copy_(0, ind.clone(), ind)

    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @expectedFailureMeta  # Warning not triggered
    @onlyNativeDeviceTypes
    def test_index_fill_mem_overlap(self, device):
        x = torch.rand((1,), device=device).expand((6,))
        ind = torch.tensor([2, 1, 0], device=device)

        with self.assertWarnsRegex(UserWarning, "index_fill_ on expanded tensors"):
            x.index_fill_(0, ind, 1.0)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.index_fill_(0, ind, 0)

    # FIXME: convert to ErrorInputs
    @expectedFailureMeta  # RuntimeError not raised
    @onlyNativeDeviceTypes
    def test_shift_mem_overlap(self, device):
        x = torch.rand(3, device=device)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x[:-1] <<= x[1:]
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x[:-1] >>= x[1:]

    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors)
    @expectedFailureMeta  # RuntimeError not raised
    @onlyNativeDeviceTypes
    def test_bernoulli_mem_overlap(self, device):
        x = torch.rand((1,), device=device).expand((6,))

        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x.bernoulli_()
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x.bernoulli_(p=0.1)
        p = torch.rand(6, device=device)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x.bernoulli_(p=p)

    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @expectedFailureMeta  # RuntimeError not raised
    @onlyNativeDeviceTypes
    def test_put_mem_overlap(self, device):
        x = torch.rand((1,), device=device).expand((6,))
        y = torch.rand((6,), device=device)
        ind = torch.tensor([2, 1, 0], device=device)
        value = torch.rand((3,), device=device)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x.put_(ind, value)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            y.put_(ind[0], y[0])
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.put_(ind, ind)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            y.put_(ind, y[:3])
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.put_(ind, ind.clone())
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.put_(ind.clone(), ind)

    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @expectedFailureMeta  # UserWarning not triggered
    @onlyNativeDeviceTypes
    def test_index_put_mem_overlap(self, device):
        x = torch.rand((1,), device=device).expand((6,))
        y = torch.rand((6,), device=device)
        ind = torch.tensor([2, 1, 0], device=device)
        value = torch.rand((3,), device=device)
        with self.assertWarnsRegex(UserWarning, 'expanded tensors'):
            x.index_put_((ind,), value)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            y.index_put_((ind,), y[0])
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.index_put_((ind,), ind)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            y.index_put_((ind,), y[:3])
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.index_put_((ind,), ind.clone())
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.index_put_((ind.clone(),), ind)

    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @expectedFailureMeta  # UserWarning not triggered
    @onlyNativeDeviceTypes
    def test_masked_fill_mem_overlap(self, device):
        x = torch.rand((1,), device=device).expand((6,))
        mask = torch.tensor([True, False, True, True, False, False], device=device)
        with self.assertWarnsRegex(UserWarning, 'expanded tensors'):
            x.masked_fill_(mask, 0.)

        fill_val = torch.tensor(0., device=device)
        with self.assertWarnsRegex(UserWarning, 'expanded tensors'):
            x.masked_fill_(mask, fill_val)

        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            mask[1:].masked_fill_(mask[:-1], False)

    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @expectedFailureMeta  # RuntimeError not raised
    @onlyNativeDeviceTypes
    def test_masked_scatter_mem_overlap(self, device):
        x = torch.rand((1,), device=device).expand((6,))
        src = torch.rand((3,), device=device)
        mask = torch.tensor([True, False, True, True, False, False], device=device)

        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x.masked_scatter_(mask, src)

    # FIXME: convert to ErrorInputs
    # (but have to extend ErrorInputs to handle inplace-only errors!)
    @onlyNativeDeviceTypes
    def test_scatter_mem_overlap(self, device):
        x = torch.rand((1,), device=device).expand((6,))
        src = torch.rand((3,), device=device)
        ind = torch.tensor([2, 1, 0], device=device, dtype=torch.int64)

        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            x.scatter_(0, ind, src)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            src.scatter_(0, ind, src)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            ind.scatter_(0, ind, ind.clone())

    # FIXME: move to test distributions
    @onlyCUDA
    def test_multinomial_device_constrain(self, device):
        x = torch.empty(3, device="cpu")
        y = torch.empty(3, device=device)
        self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device",
            lambda: torch.multinomial(x, 2, out=y))

    # FIXME: move to test distributions
    @deviceCountAtLeast(2)
    @onlyCUDA
    @skipIfTorchInductor("FIXME: error not thrown")
    def test_multinomial_gpu_device_constrain(self, devices):
        x = torch.empty(3, device=devices[0])
        y = torch.empty(3, device=devices[1], dtype=torch.long)
        self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device",
            lambda: torch.multinomial(x, 2, out=y))

    # FIXME: convert this to an automated OpInfo test
    @deviceCountAtLeast(2)
    @onlyCUDA
    def test_device_guard(self, devices):
        # verify that all operators with `device_guard: False` behave properly with multiple devices.
        # TODO: if we had operator introspection we could figure out this set of operators automatically...
        x = torch.randn((1, 2, 3), device=devices[1])
        y = torch.zeros((1, 3, 2), device=devices[1])
        scalar = torch.tensor(5, device=devices[1])

        # property ops
        torch.cudnn_is_acceptable(x)
        x.is_distributed()
        x.is_floating_point()
        x.is_complex()
        x.is_same_size(y)
        x.is_signed()
        x.size(0)
        x.stride(0)
        x.numel()
        x.is_set_to(y)
        x.data_ptr()
        scalar.is_nonzero()

        # sparse property ops
        y[0][1] = 5
        y_sparse = y.to_sparse()
        y_sparse.sparse_dim()
        y_sparse._dimI()
        y_sparse.dense_dim()
        y_sparse._dimV()
        y_sparse._nnz()
        y_sparse.is_coalesced()
        y_sparse._indices()
        y_sparse._values()
        y_sparse.indices()
        y_sparse.values()

        # in-place ops
        def inplace():
            return torch.randn((1, 2, 3), device=devices[1])
        inplace().as_strided_(y.size(), y.stride())
        inplace().resize_(y.size())
        inplace().squeeze_()
        inplace().squeeze_(0)
        inplace().unsqueeze_(2)
        inplace().transpose_(1, 2)
        inplace().squeeze_().t_()
        inplace().set_(x.storage())
        inplace().set_(x.storage(), x.storage_offset(), x.size(), x.stride())
        inplace().set_(x)
        inplace().set_()
        y_sparse._coalesced_(True)

        # shape modification
        x.as_strided(y.size(), y.stride())
        x.expand((5, 2, 3))
        x.expand_as(x)
        x.sum_to_size((1,))
        torch.broadcast_tensors(x , x)
        x.reshape((1, 3, 2))
        x.reshape_as(y)
        x.squeeze()
        x.squeeze(0)
        x.squeeze().t()
        x.transpose(1, 2)
        x.unsqueeze(2)
        x.view((1, 3, 2))
        x.view_as(y)

        # chunk, split, etc.
        x.chunk(2, dim=1)
        x.split(1, dim=2)
        x.split_with_sizes([1, 2], dim=2)
        x.unfold(dimension=2, size=1, step=1)

        x.narrow(1, 1, 1)
        x.select(1, 1)
        torch.isnan(x)

        torch.empty((1, 3, 2), out=y)
        torch.empty_like(x)
        torch.empty_like(x, dtype=torch.int64)

        # to
        x.to(x)
        x.to(y)
        x.to(x, copy=True)

    def test_is_signed(self, device):
        self.assertEqual(torch.IntTensor(5).to(device).is_signed(), True)
        self.assertEqual(torch.ByteTensor(5).to(device).is_signed(), False)
        self.assertEqual(torch.CharTensor(5).to(device).is_signed(), True)
        self.assertEqual(torch.FloatTensor(5).to(device).is_signed(), True)
        self.assertEqual(torch.HalfTensor(10).to(device).is_signed(), True)

    def test_tensor_type(self):
        for t in torch._tensor_classes:
            if 'cuda' in t.__module__:
                self.assertEqual(t.is_cuda, True)
            else:
                self.assertEqual(t.is_cuda, False)
            if 'xpu' in t.__module__:
                self.assertEqual(t.is_xpu, True)
            else:
                self.assertEqual(t.is_xpu, False)

    # Note - reports a leak of 512 bytes on CUDA device 1
    @deviceCountAtLeast(2)
    @skipCUDAMemoryLeakCheckIf(True)
    @onlyCUDA
    def test_tensor_set_errors_multigpu(self, devices):
        f_cuda0 = torch.randn((2, 3), dtype=torch.float32, device=devices[0])
        f_cuda1 = torch.randn((2, 3), dtype=torch.float32, device=devices[1])

        self.assertRaises(RuntimeError, lambda: f_cuda0.set_(f_cuda1.storage()))
        self.assertRaises(RuntimeError,
                          lambda: f_cuda0.set_(f_cuda1.storage(), 0, f_cuda1.size(), f_cuda1.stride()))
        self.assertRaises(RuntimeError, lambda: f_cuda0.set_(f_cuda1))

    # FIXME: move to test_serialization
    @onlyCUDA
    @deviceCountAtLeast(1)  # Note: Tests works with one but prefers more devices
    def test_serialization(self, devices):
        def _test_serialization(filecontext_lambda):
            t0 = torch.cuda.FloatTensor(5).fill_(1)
            with torch.cuda.device(devices[-1]):
                tn = torch.cuda.FloatTensor(3).fill_(2)
            torch.cuda.set_device(devices[0])
            b = (t0, tn)
            with filecontext_lambda() as f:
                torch.save(b, f)
                f.seek(0)
                c = torch.load(f)
                self.assertEqual(b, c, atol=0, rtol=0)
                u0, un = c
                self.assertEqual(str(u0.device), devices[0])
                self.assertEqual(str(un.device), devices[-1])

        _test_serialization(tempfile.NamedTemporaryFile)
        _test_serialization(BytesIOContext)

    # FIXME: move memory format tests to their own test class/suite
    def test_memory_format_preserved_after_permute(self, device):
        x = torch.randn(4, 3, 8, 8, device=device)
        nhwc = x.contiguous(memory_format=torch.channels_last)
        y = nhwc.permute(0, 1, 3, 2).permute(0, 1, 3, 2)
        self.assertTrue(y.is_contiguous(memory_format=torch.channels_last))

        x = torch.randn(4, 3, 8, 8, 8, device=device)
        ndhwc = x.contiguous(memory_format=torch.channels_last_3d)
        y = ndhwc.permute(0, 1, 4, 3, 2).permute(0, 1, 4, 3, 2)
        self.assertTrue(y.is_contiguous(memory_format=torch.channels_last_3d))

    def test_memory_format_propagation_rules(self, device):

        contiguous = torch.rand(10, 3, 5, 5, device=device)
        cl = torch.rand(10, 3, 5, 5, device=device).contiguous(memory_format=torch.channels_last)
        ambiguous = torch.rand(10, 3, 1, 1, device=device).contiguous(memory_format=torch.channels_last)
        self.assertTrue(ambiguous.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ambiguous.is_contiguous(memory_format=torch.contiguous_format))
        bias = torch.rand(1, 1, 1, 1, device=device).contiguous(memory_format=torch.channels_last)

        def _test_propagation_rules(self, contiguous, cl, ambiguous, bias):
            options = ((ambiguous, contiguous, torch.contiguous_format),
                       (ambiguous, cl, torch.channels_last),
                       (contiguous, ambiguous, torch.contiguous_format),
                       (contiguous, cl, torch.contiguous_format),
                       (cl, ambiguous, torch.channels_last),
                       (cl, contiguous, torch.channels_last),
                       (bias, cl, torch.channels_last),
                       (cl, bias, torch.channels_last),)

            for a, b, mf in options:
                result = a + b
                self.assertTrue(result.is_contiguous(memory_format=mf))

        _test_propagation_rules(self, contiguous, cl, ambiguous, bias)

        cl = cl.to(memory_format=torch.channels_last)
        ambiguous = ambiguous.to(memory_format=torch.channels_last)
        bias = bias.to(memory_format=torch.channels_last)

        _test_propagation_rules(self, contiguous, cl, ambiguous, bias)

        # test cases when strides matter in ambiguous tensors
        for mf in (torch.channels_last, torch.contiguous_format):
            ambiguous = torch.rand(10, 3, 1, 1, device=device).to(memory_format=mf)
            bias = torch.rand(3, 1, 1, device=device)
            result = ambiguous + bias
            self.assertEqual(ambiguous.stride(), result.stride())
            result = bias + ambiguous
            self.assertEqual(ambiguous.stride(), result.stride())
            result = ambiguous * 5
            self.assertEqual(ambiguous.stride(), result.stride())

    @skipIfMPS
    def test_memory_format_empty_like(self, device):
        def test_helper(x, memory_format):
            xc = x.contiguous(memory_format=memory_format)

            like = torch.empty_like(xc, memory_format=torch.preserve_format)
            self.assertFalse(like.is_contiguous())
            self.assertTrue(like.is_contiguous(memory_format=memory_format))

            like_x = torch.empty_like(x, memory_format=torch.preserve_format)
            self.assertTrue(like_x.is_contiguous())
            self.assertFalse(like_x.is_contiguous(memory_format=memory_format))

            like = torch.empty_like(x, memory_format=memory_format)
            self.assertFalse(like.is_contiguous())
            self.assertTrue(like.is_contiguous(memory_format=memory_format))

            like = torch.empty_like(xc, memory_format=torch.contiguous_format)
            self.assertTrue(like.is_contiguous())
            self.assertFalse(like.is_contiguous(memory_format=memory_format))

            like = torch.empty_like(xc)
            self.assertFalse(like.is_contiguous())
            self.assertTrue(like.is_contiguous(memory_format=memory_format))

            sparse = x.to_sparse()
            with self.assertRaises(RuntimeError):
                torch.empty_like(sparse, memory_format=torch.preserve_format)

        test_helper(torch.randn(4, 3, 8, 8, device=device), torch.channels_last)
        test_helper(torch.randn(4, 3, 8, 8, 8, device=device), torch.channels_last_3d)

    def test_memory_format_consistency(self, device):
        x = torch.randn(10, 3, 1, 1, device=device)
        x_rep = x.as_strided(x.size(), x.stride())
        self.assertEqual(x.size(), x_rep.size())
        self.assertEqual(x.stride(), x_rep.stride())
        self.assertEqual(x.is_contiguous(), x_rep.is_contiguous())
        self.assertEqual(x.is_contiguous(memory_format=torch.channels_last), x_rep.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(
            x.is_contiguous(memory_format=torch.channels_last_3d), x_rep.is_contiguous(memory_format=torch.channels_last_3d))

    # FIXME: make this a elementwise unary and elementwise binary OpInfo test
    def test_memory_format_operators(self, device):
        def _chunk_op(x, y):
            x1, x2 = x.chunk(2, dim=1)
            return x1 + x2

        def _unsqueeze_op_add(x, y):
            return x[0].unsqueeze(0) + 3

        def _unsqueeze_op_clone(x, y):
            return x[0].unsqueeze(0).clone()

        def _test_helper(x, y, bias, memory_format):
            return_contig_fns = [
                lambda x, y: y + x,
                lambda x, y: y * x,
                lambda x, y: y.addcdiv(x, y, value=2),
                lambda x, y: y.addcmul(x, y, value=2),
            ]
            bias_fns = [
                lambda x, b: x + b,
                lambda x, b: b + x,
            ]
            fns = [
                lambda x, y: x.clone(),
                lambda x, y: x + 3,
                lambda x, y: 3 * x,
                lambda x, y: x + y,
                lambda x, y: x * y,
                lambda x, y: abs(x),
                lambda x, y: x.abs(),
                lambda x, y: x.abs_(),
                lambda x, y: x.acos(),
                lambda x, y: x.acos_(),
                lambda x, y: x.add(y, alpha=3),
                lambda x, y: x.add_(y, alpha=3),
                lambda x, y: x.addcdiv(y, y, value=2),
                lambda x, y: x.addcdiv_(y, y, value=2),
                lambda x, y: x.addcmul(y, y, value=2),
                lambda x, y: x.addcmul_(y, y, value=2),
                lambda x, y: x.acosh(),
                lambda x, y: x.acosh_(),
                lambda x, y: x.asinh(),
                lambda x, y: x.asinh_(),
                lambda x, y: x.atanh(),
                lambda x, y: x.atanh_(),
                lambda x, y: x.asin(),
                lambda x, y: x.asin_(),
                lambda x, y: x.atan(),
                lambda x, y: x.atan2(y),
                lambda x, y: x.atan2_(y),
                lambda x, y: x.ceil(),
                lambda x, y: x.ceil_(),
                lambda x, y: x.clamp(-1, 1),
                lambda x, y: x.cos(),
                lambda x, y: x.cosh(),
                lambda x, y: x.div(0.5),
                lambda x, y: x.div_(0.5),
                lambda x, y: x.div(y),
                lambda x, y: x.div_(y),
                lambda x, y: x.digamma(),
                lambda x, y: x.digamma_(),
                lambda x, y: x.erf(),
                lambda x, y: x.erfc(),
                lambda x, y: x.erfinv(),
                lambda x, y: x.erfinv_(),
                lambda x, y: x.exp(),
                lambda x, y: x.expm1(),
                lambda x, y: x.expm1_(),
                lambda x, y: x.floor(),
                lambda x, y: x.floor_(),
                lambda x, y: x.fmod(2),
                lambda x, y: x.frac(),
                lambda x, y: x.hypot(y),
                lambda x, y: x.hypot_(y),
                lambda x, y: x.i0(),
                lambda x, y: x.i0_(),
                lambda x, y: x.lerp(y, 0.5),
                lambda x, y: x.log(),
                lambda x, y: x.log_(),
                lambda x, y: x.log10(),
                lambda x, y: x.log10_(),
                lambda x, y: x.log1p(),
                lambda x, y: x.log1p_(),
                lambda x, y: x.log2(),
                lambda x, y: x.log2_(),
                lambda x, y: x.mul(3),
                lambda x, y: x.mul_(3),
                lambda x, y: x.neg(),
                lambda x, y: x.neg_(),
                lambda x, y: x.pow(3),
                lambda x, y: x.pow_(3),
                lambda x, y: x.pow(0.0),
                lambda x, y: x.pow(1.0),
                lambda x, y: x.reciprocal(),
                lambda x, y: x.remainder(2),
                lambda x, y: x.round(),
                lambda x, y: x.round_(),
                lambda x, y: x.rsqrt(),
                lambda x, y: x.rsqrt_(),
                lambda x, y: x.sigmoid(),
                lambda x, y: x.sigmoid_(),
                lambda x, y: x.logit(),
                lambda x, y: x.logit_(),
                lambda x, y: x.logit(1e-6),
                lambda x, y: x.logit_(1e-6),
                lambda x, y: x.sign(),
                lambda x, y: x.sign_(),
                lambda x, y: x.sgn(),
                lambda x, y: x.sgn_(),
                lambda x, y: x.sin(),
                lambda x, y: x.sin_(),
                lambda x, y: x.sinh(),
                lambda x, y: x.sinh_(),
                lambda x, y: x.sqrt(),
                lambda x, y: x.sqrt_(),
                lambda x, y: x.tan(),
                lambda x, y: x.tanh(),
                lambda x, y: x.trunc(),
                lambda x, y: x.trunc_(),
                _chunk_op,
                _unsqueeze_op_add,
                _unsqueeze_op_clone,
            ]
            x_c = x.contiguous()
            y_c = y.contiguous()
            b_c = bias.contiguous()
            for fn in fns:
                is_inplace = '_(' in inspect.getsource(fn)
                x_clone = x.clone() if is_inplace else x
                x_c_clone = x_c.clone() if is_inplace else x_c
                result_c = fn(x_c_clone, y_c)
                result = fn(x_clone, y)
                self.assertEqual(result, result_c, f"Failed for '{inspect.getsource(fn).strip()}'")
                self.assertTrue(
                    result.is_contiguous(memory_format=memory_format),
                    f"result of the '{inspect.getsource(fn).strip()}' is not in '{memory_format}' format")

            for fn in bias_fns:
                result_c = fn(x_c, b_c)
                result = fn(x, bias)
                self.assertEqual(result, result_c, f"Failed for '{inspect.getsource(fn).strip()}'")
                self.assertTrue(
                    result.is_contiguous(memory_format=memory_format),
                    f"result of the '{inspect.getsource(fn).strip()}' is not in '{memory_format}' format")

            for fn in return_contig_fns:
                result_c = fn(x_c, y_c)
                result = fn(x, y)
                self.assertEqual(result, result_c, f"Failed for '{inspect.getsource(fn).strip()}'")
                self.assertTrue(
                    result.is_contiguous(memory_format=torch.contiguous_format),
                    f"result of the '{inspect.getsource(fn).strip()}' is not in '{torch.contiguous_format}' format")

        _test_helper(
            torch.randn((4, 3, 8, 8), device=device).contiguous(memory_format=torch.channels_last),
            abs(torch.randn((4, 3, 8, 8), device=device)) + 1,
            torch.randn((1, 3, 1, 1), device=device).contiguous(memory_format=torch.channels_last),
            torch.channels_last)
        _test_helper(
            torch.randn((4, 3, 8, 8, 8), device=device).contiguous(memory_format=torch.channels_last_3d),
            abs(torch.randn((4, 3, 8, 8, 8), device=device)) + 1,
            torch.randn((1, 3, 1, 1, 1), device=device).contiguous(memory_format=torch.channels_last_3d),
            torch.channels_last_3d)

    # FIXME: make this a elementwise unary and elementwise binary OpInfo test
    def test_strides_propagation(self, device):
        def _test_helper(x, op, unary=False):
            def compare_strides(s1, s2, div):
                sdiv = [s // div for s in s1]
                self.assertEqual(sdiv, s2)

            dim = x.dim()
            # we produce memory dense outputs, so when input is strided on the last dimension
            # we need to divide by that dimension stride to compare input and result strides
            div = x.stride(-1)
            for p in permutations(range(dim)):
                xp = x.permute(p)
                if not unary:
                    y = torch.randn(xp.size(-1), device=x.device, dtype=x.dtype)
                    for inputs in ((xp, xp), (xp, y), (y, xp)):
                        res = op(*inputs)
                        compare_strides(xp.stride(), res.stride(), div)
                        self.assertEqual(xp.size(), res.size())
                        out = torch.empty(0, device=xp.device, dtype=res.dtype)
                        res = op(*inputs, out=out)
                        compare_strides(xp.stride(), res.stride(), div)
                        self.assertEqual(xp.size(), res.size())
                else:
                    res = op(xp)
                    compare_strides(xp.stride(), res.stride(), div)
                    self.assertEqual(xp.size(), res.size())
                    out = torch.empty(0, device=xp.device, dtype=res.dtype)
                    res = op(xp, out=out)
                    compare_strides(xp.stride(), res.stride(), div)
                    self.assertEqual(xp.size(), res.size())

        # torch.eq by default calls TensorIterator with defined output, torch.add with undefined
        binary_ops = (torch.eq, torch.add)
        unary_ops = (torch.exp,)
        # memory dense, sliced and ambiguous sliced (ambiguous dense loses permutation information)
        xs = (torch.randn(2, 3, 4, device=device), torch.randn(2, 3, 8, device=device)[:, :, ::2],
              torch.randn(1, 1, 4, 12, device=device)[:, :, :, ::2])
        for op in binary_ops:
            for x in xs:
                _test_helper(x, op)
        for op in unary_ops:
            for x in xs:
                _test_helper(x, op, unary=True)

    @onlyCUDA
    @unittest.skipIf(PYTORCH_CUDA_MEMCHECK, "is_pinned uses failure to detect pointer property")
    @skipIfTorchDynamo("NotImplementedError: PrimTorch does not support pinned memory")
    def test_pin_memory_from_constructor(self, device):
        def _get_like(t, **kwargs):
            return [
                torch.rand_like(t, **kwargs),
                torch.randn_like(t, **kwargs),
                torch.empty_like(t, **kwargs),
                torch.full_like(t, 4, **kwargs),
                torch.zeros_like(t, **kwargs),
                torch.ones_like(t, **kwargs),
            ]

        def _get_tensors(**kwargs):
            return [
                torch.tensor([10, 11], **kwargs),
                torch.randn(3, 5, **kwargs),
                torch.rand(3, **kwargs),
                # torch.randint(3, 5, **kwargs), // unsupported
                torch.zeros(3, **kwargs),
                torch.randperm(3, **kwargs),
                torch.empty(6, **kwargs),
                torch.ones(6, **kwargs),
                torch.eye(6, **kwargs),
                torch.arange(3, 5, **kwargs)]

        pinned_tensors = _get_tensors(pin_memory=True) + _get_like(torch.empty(5, dtype=torch.float64), pin_memory=True)
        for x in pinned_tensors:
            self.assertTrue(x.is_pinned())

        tensors = _get_tensors() + _get_like(torch.empty(5, dtype=torch.float64, pin_memory=True))
        for x in tensors:
            self.assertFalse(x.is_pinned())

    @deviceCountAtLeast(1)
    @onlyCUDA
    @parametrize("non_blocking", (True, False))
    def test_storage_all_devices(self, devices, non_blocking):
        for device in devices:
            t = torch.randn(6, device=device)
            self.assertEqual(t.dtype, t.storage().dtype)
            s = t.untyped_storage()
            s_cpu = s.to(device='cpu', non_blocking=non_blocking)
            if non_blocking:
                torch.cuda.synchronize()
                self.assertTrue(s_cpu.is_pinned())
            else:
                self.assertFalse(s_cpu.is_pinned())
            t_cpu = torch.empty(()).set_(s_cpu)
            self.assertEqual(t.cpu(), t_cpu)

    # Note [lazy_clone_ tests with inductor enabled]
    # These `lazy_clone_` tests are written in a way that makes them pass in
    # both eager mode and compiled mode (`PYTORCH_TEST_WITH_INDUCTOR=1`). There
    # are cases where COW tensors can materialize at different times and in
    # different ways in compiled mode versus eager mode, and those cases need to
    # be avoided. There are two main wrinkles the be aware of.
    #
    # The first wrinkle is that these tests have to check the internal
    # properties of tensors to make sure they materialize in the expected way,
    # and those checks cause dynamo graph breaks. Depending on the situation, a
    # graph break in-between two compiled graphs that operate on the same COW
    # tensor can make the tensor materialize when it would not materialize in
    # eager mode, causing the checks to fail. The strategy for avoiding this is
    # to make all the operations on COW tensors get compiled into the same
    # graph, by not doing any checks between the operations, and just do all the
    # checks at the end of the test. If we really do want to perform checks
    # between two operations, `op1` and `op2`, the solution is to create two
    # different tests. One test performs just `op1` and then checks. The other
    # test performs `op1` followed immediately by `op2` and then checks.
    #
    # The second wrinkle is that in eager mode, if we perform writes on two COW
    # tensors where one is a lazy clone of the other, the first tensor to be
    # written will be materialized with a new data pointer, and the second
    # tensor will just reuse the original data pointer when it is materialized.
    # But in compiled mode, if these writes happen in the same graph, the order
    # in which the tensors materialize can be different than in eager mode. So
    # in this case the strategy is to purposefully cause a graph break to happen
    # in-between the two write operations, by adding checks between them, so
    # that they have to materialize in the expected order.
    @skipXLA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_lazy_clone(self, device, dtype):
        t = torch.tensor([[0, 1], [2, 3]], device=device, dtype=dtype)
        t_orig_storage_addr = torch._C._storage_address(t)
        orig_data_ptr = torch._C._data_address(t)
        clone = t._lazy_clone()

        # Lazy cloning a tensor should cause both it and its clone to become COW
        # tensors. They should have different storages, but the same data
        # pointer.

        self.assertTrue(torch._C._is_cow_tensor(clone))
        self.assertTrue(torch._C._is_cow_tensor(t))

        self.assertTrue(torch._C._storage_address(t) == t_orig_storage_addr)
        self.assertTrue(torch._C._storage_address(clone) != t_orig_storage_addr)

        self.assertTrue(torch._C._data_address(t) == orig_data_ptr)
        self.assertTrue(torch._C._data_address(clone) == orig_data_ptr)

    # See Note [lazy_clone_ tests with inductor enabled]
    @skipXLA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_lazy_clone_view(self, device, dtype):
        t = torch.tensor([[0, 1], [2, 3]], device=device, dtype=dtype)
        t_orig_storage_addr = torch._C._storage_address(t)
        orig_data_ptr = torch._C._data_address(t)
        clone = t._lazy_clone()
        view = t.view([4])

        # Viewing `t` should not cause a copy (materialize) to happen. All the
        # tensors should still be COW and have the same data pointer. `view` and
        # `t` should have the same storage, and `clone` should have a different
        # storage.

        self.assertTrue(torch._C._is_cow_tensor(t))
        self.assertTrue(torch._C._is_cow_tensor(view))
        self.assertTrue(torch._C._is_cow_tensor(clone))

        self.assertTrue(torch._C._storage_address(t) == t_orig_storage_addr)
        self.assertTrue(torch._C._storage_address(view) == t_orig_storage_addr)
        self.assertTrue(torch._C._storage_address(clone) != t_orig_storage_addr)

        self.assertTrue(torch._C._data_address(t) == orig_data_ptr)
        self.assertTrue(torch._C._data_address(clone) == orig_data_ptr)
        self.assertTrue(torch._C._data_address(view) == orig_data_ptr)

    # See Note [lazy_clone_ tests with inductor enabled]
    @skipXLA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_lazy_clone_view_materialize(self, device, dtype):
        t = torch.tensor([[0, 1], [2, 3]], device=device, dtype=dtype)
        t_orig_storage_addr = torch._C._storage_address(t)
        orig_data_ptr = torch._C._data_address(t)
        clone = t._lazy_clone()
        view = t.view([4])
        view += torch.ones(1, device=device, dtype=dtype)

        # Writing to `t` should cause the storage under `t` and `view` to be
        # copied (materialized), but should not affect `clone`.

        self.assertFalse(torch._C._is_cow_tensor(t))
        self.assertFalse(torch._C._is_cow_tensor(view))
        self.assertTrue(torch._C._is_cow_tensor(clone))

        self.assertTrue(torch._C._storage_address(t) == t_orig_storage_addr)
        self.assertTrue(torch._C._storage_address(view) == t_orig_storage_addr)
        self.assertTrue(torch._C._storage_address(clone) != t_orig_storage_addr)

        t_new_data_addr = torch._C._data_address(t)
        self.assertTrue(t_new_data_addr != orig_data_ptr)
        self.assertTrue(torch._C._data_address(view) == t_new_data_addr)
        self.assertTrue(torch._C._data_address(clone) == orig_data_ptr)

        clone += torch.ones(1, device=device, dtype=dtype)

        # Writing to `clone` should materialize it, so it should no longer
        # be COW. However, since `clone`'s storage is the only COW storage
        # left that holds a reference to the original data pointer, this
        # materialization should not actually cause a copy--it should
        # just reuse the original data pointer.

        self.assertFalse(torch._C._is_cow_tensor(t))
        self.assertFalse(torch._C._is_cow_tensor(view))
        self.assertFalse(torch._C._is_cow_tensor(clone))

        self.assertTrue(torch._C._storage_address(t) == t_orig_storage_addr)
        self.assertTrue(torch._C._storage_address(view) == t_orig_storage_addr)
        self.assertTrue(torch._C._storage_address(clone) != t_orig_storage_addr)

        self.assertTrue(torch._C._data_address(t) == t_new_data_addr)
        self.assertTrue(torch._C._data_address(view) == t_new_data_addr)
        self.assertTrue(torch._C._data_address(clone) == orig_data_ptr)

    @skipXLA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_lazy_clone_binary_op_no_materialize(self, device, dtype):
        t = torch.tensor([[0, 1], [2, 3]], device=device, dtype=dtype)
        clone = t._lazy_clone()
        t + clone
        self.assertTrue(torch._C._is_cow_tensor(t))
        self.assertTrue(torch._C._is_cow_tensor(clone))

    # This tests that if a COW materialization is attempted inside an
    # `at::parallel_for` loop function, then an error is raised. This test is
    # implemented in Python rather than C++ because the C++ tests are built
    # without multithreading support in `at::parallel_for`.
    @skipXLA
    @skipIfTorchDynamo("Torchdynamo fails and we do not need to test it here anyway")
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_parallel_cow_materialize_error(self, device, dtype):

        def run(num_threads, num_parallel, skip_first, should_error):
            orig_num_threads = torch.get_num_threads()

            try:
                torch.set_num_threads(num_threads)

                a = torch.tensor([[0, 1], [2, 3]], device=device, dtype=dtype)._lazy_clone()

                if should_error:
                    with self.assertRaisesRegex(RuntimeError, r'Materializing a storage'):
                        torch._test_parallel_materialize(
                            a, num_parallel, skip_first)
                else:
                    torch._test_parallel_materialize(a, num_parallel, skip_first)

                # Error should not raise in any case if the tensor is not COW
                b = torch.tensor([[0, 1], [2, 3]], device=device, dtype=dtype)
                torch._test_parallel_materialize(b, num_parallel, skip_first)

            finally:
                torch.set_num_threads(orig_num_threads)

        run(1, 1, False, True)
        run(1, 1, True, False)
        run(1, 10, False, True)
        run(1, 10, True, True)
        run(10, 1, False, True)
        run(10, 1, True, False)
        run(10, 10, False, True)
        run(10, 10, True, True)
        run(10, 2, False, True)
        run(10, 2, True, True)

    # FIXME: move to test distributions
    @skipIfMPS
    @dtypesIfCUDA(torch.float, torch.double, torch.half)
    @dtypes(torch.float, torch.double, torch.half)
    def test_multinomial(self, device, dtype):
        def make_prob_dist(shape, is_contiguous):
            if is_contiguous:
                if dtype == torch.half:
                    return torch.zeros(shape, device=device).uniform_().to(dtype=torch.half)
                return torch.zeros(shape, device=device, dtype=dtype).uniform_()
            elif len(shape) == 1:
                if dtype == torch.half:
                    return torch.zeros((shape + [5]), device=device).uniform_().to(dtype=torch.half)[:, 2]
                return torch.zeros((shape + [5]), device=device, dtype=dtype).uniform_()[:, 2]
            else:
                # num dim = 2
                new_shape = [2, shape[1], 7, 1, shape[0], 1, 10]
                if dtype == torch.half:
                    prob_dist = torch.zeros(new_shape, device=device).uniform_().to(dtype=torch.half)
                else:
                    prob_dist = torch.zeros(new_shape, device=device, dtype=dtype).uniform_()
                prob_dist = prob_dist.transpose(1, 4)
                prob_dist = prob_dist[1, :, 5, 0, :, 0, 4]
                assert not prob_dist.is_contiguous()  # sanity check
                return prob_dist

        for is_contiguous in (True, False):
            # with replacement
            n_row = 3
            for n_col in range(4, 5 + 1):
                prob_dist = make_prob_dist([n_row, n_col], is_contiguous)
                # indices that shouldn't be sampled (<0 means none)
                zero_prob_indices = torch.LongTensor(n_row).random_(-2, n_col).tolist()
                for i, j in enumerate(zero_prob_indices):
                    if j >= 0:
                        prob_dist[i, j] = 0
                n_sample = n_col * 3
                sample_indices = torch.multinomial(prob_dist, n_sample, True)
                self.assertEqual(prob_dist.dim(), 2)
                self.assertEqual(sample_indices.size(1), n_sample)
                for i in range(n_row):
                    zero_prob_idx = zero_prob_indices[i]
                    if zero_prob_idx < 0:
                        continue
                    for j in range(n_sample):
                        self.assertNotEqual(sample_indices[i, j], zero_prob_idx,
                                            msg="sampled an index with zero probability")

            # without replacement
            n_row = 3
            for n_col in range(2, 10 + 1, 2):
                prob_dist = make_prob_dist([n_row, n_col], is_contiguous)
                # indices that shouldn't be sampled (<0 means none)
                zero_prob_indices = torch.LongTensor(n_row).random_(-1, n_col).tolist()
                for i, j in enumerate(zero_prob_indices):
                    if j >= 0:
                        prob_dist[i, j] = 0
                n_sample = max(1, n_col - 2)
                sample_indices = torch.multinomial(prob_dist, n_sample, False)
                self.assertEqual(prob_dist.dim(), 2)
                self.assertEqual(sample_indices.size(1), n_sample)
                for i in range(n_row):
                    row_samples = {}
                    zero_prob_idx = zero_prob_indices[i]
                    for j in range(n_sample):
                        sample_idx = sample_indices[i, j]
                        if zero_prob_idx >= 0:
                            self.assertNotEqual(sample_idx, zero_prob_idx,
                                                msg="sampled an index with zero probability")
                        self.assertNotIn(sample_idx, row_samples, "sampled an index twice")
                        row_samples[sample_idx] = True

            # vector
            n_col = 4
            prob_dist = make_prob_dist([n_col], is_contiguous).fill_(1)
            zero_prob_idx = 1  # index that shouldn't be sampled
            prob_dist[zero_prob_idx] = 0
            n_sample = 20
            sample_indices = torch.multinomial(prob_dist, n_sample, True)
            for sample_index in sample_indices:
                self.assertNotEqual(sample_index, zero_prob_idx, msg="sampled an index with zero probability")
            self.assertEqual(sample_indices.dim(), 1, msg="wrong number of dimensions")
            self.assertEqual(prob_dist.dim(), 1, msg="wrong number of prob_dist dimensions")
            self.assertEqual(sample_indices.size(0), n_sample, msg="wrong number of samples")

        # CUDA misalignment issue (#46702)
        n_row, n_col = 2, 3
        prob_dist = make_prob_dist([n_row, n_col], True)
        n_sample = 1
        sample_indices = torch.multinomial(prob_dist, n_sample, True)
        self.assertEqual(sample_indices.dim(), 2, msg="wrong number of dimensions")
        self.assertEqual(sample_indices.size(1), n_sample, msg="wrong number of samples")

    # FIXME: move to test distributions
    @onlyCUDA
    @dtypes(torch.float, torch.double, torch.half)
    def test_multinomial_deterministic(self, device, dtype):
        gen = torch.Generator(device=device)

        trials = 5
        seed = 0
        prob_dist = torch.rand(10000, 1000, device=device, dtype=dtype)
        n_sample = 1

        for i in range(trials):
            gen.manual_seed(seed)
            samples_1 = torch.multinomial(prob_dist, n_sample, True, generator=gen)

            gen.manual_seed(seed)
            samples_2 = torch.multinomial(prob_dist, n_sample, True, generator=gen)

            self.assertEqual(samples_1, samples_2)
            self.assertEqual(samples_1.dim(), 2, msg="wrong number of dimensions")
            self.assertEqual(samples_1.size(1), n_sample, msg="wrong number of samples")

    # FIXME: move to test distributions
    @slowTest
    @dtypes(torch.float)
    def test_multinomial_rng_state_advance(self, device, dtype):
        corpus_size = 100000
        freqs = torch.ones(corpus_size, dtype=torch.float, device=device)
        n_sample = 100
        samples1 = torch.multinomial(freqs, n_sample, replacement=True)
        samples2 = torch.multinomial(freqs, n_sample, replacement=True)
        samples = torch.cat([samples1, samples2])
        # expect no more than 1 repeating elements generated in 2 attempts
        # the probability of at least element being repeated is surprisingly large, 18%
        self.assertLessEqual(2 * n_sample - samples.unique().size(0), 2)
        samples1 = torch.multinomial(freqs, n_sample, replacement=False)
        samples2 = torch.multinomial(freqs, n_sample, replacement=False)
        samples = torch.cat([samples1, samples2])
        # expect no more than 1 repeating elements generated in 2 attempts
        self.assertLessEqual(2 * n_sample - samples.unique().size(0), 1)

    def _test_memory_format_transformations(self, device, input_generator_fn, transformation_fn,
                                            memory_format, compare_data=True, default_is_preserve=False):

        assert memory_format == torch.channels_last or memory_format == torch.channels_last_3d

        # xc is a channels last tensor
        xc = input_generator_fn(device)
        # xc is not memory dense, but looks like channels last
        # We don't preserve non-dense striding
        if not TEST_WITH_TORCHINDUCTOR:
            if memory_format == torch.channels_last:
                xc = xc[..., ::2, ::2]
            else:
                xc = xc[..., ::2, ::2, ::2]

        clone = transformation_fn(xc, memory_format=torch.preserve_format)


        self.assertFalse(clone.is_contiguous())
        self.assertTrue(clone.is_contiguous(memory_format=memory_format))
        if not TEST_WITH_TORCHINDUCTOR:
            self.assertFalse(xc.is_contiguous())
            self.assertFalse(xc.is_contiguous(memory_format=memory_format))
        if compare_data:
            self.assertEqual(xc, clone.to(xc))

        xc = input_generator_fn(device)
        clone = transformation_fn(xc, memory_format=torch.contiguous_format)
        self.assertTrue(clone.is_contiguous())
        self.assertFalse(clone.is_contiguous(memory_format=memory_format))
        if compare_data:
            self.assertEqual(xc, clone.to(xc))

        xc = input_generator_fn(device)
        clone = transformation_fn(xc)

        if default_is_preserve:
            self.assertFalse(clone.is_contiguous())
            self.assertTrue(clone.is_contiguous(memory_format=memory_format))
        else:
            self.assertTrue(clone.is_contiguous())
            self.assertFalse(clone.is_contiguous(memory_format=memory_format))
        if compare_data:
            self.assertEqual(xc, clone.to(xc))

        # TODO copy _like constructors to stride permutation instead of just layout
        if not TEST_WITH_TORCHINDUCTOR:
            x = torch.randn((3, 4, 5, 6, 7, 8, 9), device=device)
            for i in range(10):
                permutation = list(range(len(x.shape)))
                random.shuffle(permutation)
                x = x.permute(permutation)
                self.assertEqual(x.stride(), transformation_fn(x, memory_format=torch.preserve_format).stride())

    def test_memory_format_to(self, device):
        def get_generator(memory_format, shape):
            def input_generator_fn(device):
                return torch.randn(shape, device=device, dtype=torch.float32).contiguous(memory_format=memory_format)
            return input_generator_fn

        def transformation_fn(tensor, **kwargs):
            return tensor.to(dtype=torch.float64, **kwargs)

        formats_shapes = (
            (torch.channels_last, (4, 3, 8, 8)),
            (torch.channels_last_3d, (4, 3, 8, 8, 8)))

        for mf, shape in formats_shapes:
            self._test_memory_format_transformations(
                device, get_generator(mf, shape), transformation_fn, mf, default_is_preserve=True)

    def test_memory_format_type(self, device):
        def get_generator(memory_format, shape):
            def input_generator_fn(device):
                return torch.randn(shape, device=device, dtype=torch.float32).contiguous(memory_format=memory_format)
            return input_generator_fn

        def transformation_fn(tensor, **kwargs):
            return tensor.to(torch.float64, **kwargs)

        formats_shapes = (
            (torch.channels_last, (4, 3, 8, 8)),
            (torch.channels_last_3d, (4, 3, 8, 8, 8)))

        for mf, shape in formats_shapes:
            self._test_memory_format_transformations(
                device, get_generator(mf, shape), transformation_fn, mf, default_is_preserve=True)

    def test_memory_format_clone(self, device):
        def get_generator(memory_format, shape):
            def input_generator_fn(device):
                return torch.randn(shape, device=device, dtype=torch.float32).contiguous(memory_format=memory_format)
            return input_generator_fn

        def transformation_fn(tensor, **kwargs):
            return tensor.clone(**kwargs)

        formats_shapes = (
            (torch.channels_last, (4, 3, 8, 8)),
            (torch.channels_last_3d, (4, 3, 8, 8, 8)))

        for mf, shape in formats_shapes:
            self._test_memory_format_transformations(
                device, get_generator(mf, shape), transformation_fn, mf, True, default_is_preserve=True)

    def test_memory_format_factory_like_functions_preserve(self, device):
        def get_generator(memory_format, shape):
            def input_generator_fn(device):
                return torch.randn(shape, device=device, dtype=torch.float32).contiguous(memory_format=memory_format)
            return input_generator_fn

        transformation_fns = [
            lambda t, **kwargs: torch.zeros_like(t, **kwargs),
            lambda t, **kwargs: torch.ones_like(t, **kwargs),
            lambda t, **kwargs: torch.randint_like(t, 10, 100, **kwargs),
            lambda t, **kwargs: torch.randint_like(t, 100, **kwargs),
            lambda t, **kwargs: torch.randn_like(t, **kwargs),
            lambda t, **kwargs: torch.rand_like(t, **kwargs),
            lambda t, **kwargs: torch.full_like(t, 7, **kwargs),
            lambda t, **kwargs: torch.empty_like(t, **kwargs)]

        formats_shapes = (
            (torch.channels_last, (4, 3, 8, 8)),
            (torch.channels_last_3d, (4, 3, 8, 8, 8)))

        for mf, shape, in formats_shapes:
            for transformation_fn in transformation_fns:
                self._test_memory_format_transformations(
                    device, get_generator(mf, shape), transformation_fn, mf, compare_data=False, default_is_preserve=True)

    def test_memory_format_type_shortcuts(self, device):
        def get_generator(memory_format, shape, dtype):
            def input_generator_fn(device):
                return torch.randn(shape, device=device, dtype=dtype).clamp(0, 1) \
                    .round().contiguous(memory_format=memory_format)
            return input_generator_fn


        def get_fn(fn_name):
            def transformation_fn(tensor, **kwargs):
                fn = getattr(tensor, fn_name)
                return fn(**kwargs)
            return transformation_fn

        shortcuts = ['byte', 'char', 'double', 'bool', 'half', 'int', 'long', 'short']
        if device == 'cpu':
            shortcuts += ['bfloat16']

        formats_shapes = (
            (torch.channels_last, (4, 3, 8, 8)),
            (torch.channels_last_3d, (4, 3, 8, 8, 8)))

        for mf, shape in formats_shapes:
            for fn_name in shortcuts:
                self._test_memory_format_transformations(
                    device, get_generator(mf, shape, torch.float32), get_fn(fn_name), mf, default_is_preserve=True)

        # Test 'float' separately to avoid float->float no-op.
        for mf, shape in formats_shapes:
            self._test_memory_format_transformations(
                device, get_generator(mf, shape, torch.float64), get_fn('float'), mf, default_is_preserve=True)

    @onlyCUDA
    def test_memory_format_cpu_and_cuda_ops(self, device):
        def get_generator(memory_format, shape):
            def input_generator_fn(device):
                return torch.randn(shape, device=device, dtype=torch.float32).contiguous(memory_format=memory_format)
            return input_generator_fn

        def transformation_cpu_fn(tensor, **kwargs):
            return tensor.cpu(**kwargs)

        def transformation_cuda_fn(tensor, **kwargs):
            return tensor.cuda(**kwargs)

        formats_shapes = (
            (torch.channels_last, (4, 3, 8, 8)),
            (torch.channels_last_3d, (4, 3, 8, 8, 8)))

        for mf, shape in formats_shapes:
            self._test_memory_format_transformations(
                'cuda', get_generator(mf, shape), transformation_cpu_fn, mf, default_is_preserve=True)
            self._test_memory_format_transformations(
                'cpu', get_generator(mf, shape), transformation_cuda_fn, mf, default_is_preserve=True)

    # FIXME: move to test_serialization
    @onlyNativeDeviceTypes
    def test_pickle_gradscaler(self, device):
        # This test should pass in 3 cases for cuda:
        #  1. cuda is not available.
        #  2. cuda is available but device is not cuda.
        #  3. cuda is available and device is cuda.
        # In case 1, a and b disable themselves on construction and shouldn't try to pickle workhorse attributes.
        # In case 2, a and b are enabled.  Workhorse attributes participate in pickling, but none are lazy-inited
        # to cuda Tensors, because I don't want to do cuda things if device is not cuda.
        # In case 3, a and b are enabled and we may also try lazy-initing _scale to a cuda tensor.
        device = torch.device(device)
        try_lazy_inits = (True, False)
        GradScaler = partial(torch.GradScaler, device=device.type)
        for lazy_init_scale in try_lazy_inits:
            a = GradScaler(init_scale=3., growth_factor=4., backoff_factor=.5, growth_interval=2)
            if device.type == "cuda":
                self.assertTrue(not a.is_enabled() if torch.cuda.amp.common.amp_definitely_not_available() else a.is_enabled())
            else:
                self.assertTrue(a.is_enabled())
            if lazy_init_scale:
                # Dummy a.scale() call lazy-inits a._scale Tensor.
                a.scale(torch.tensor([4.0], dtype=torch.float32, device=device))
                self.assertTrue(a._scale.device.type == device.type)
            # The following three lines should work whether or not cuda is available.
            serialized = pickle.dumps(a)
            b = pickle.loads(serialized)
            self.assertEqual(b.is_enabled(), a.is_enabled())
            if a.is_enabled():
                self.assertEqual(b.get_scale(), 3.)
                self.assertEqual(b.get_growth_factor(), 4.)
                self.assertEqual(b.get_backoff_factor(), .5)
                self.assertEqual(b.get_growth_interval(), 2)
                self.assertEqual(b._init_growth_tracker, 0)
                # supplies a dummy key to test the defaultdict's default_factory
                self.assertEqual(b._per_optimizer_states["fdsa"],
                                 torch.amp.grad_scaler._refresh_per_optimizer_state())
                if lazy_init_scale:
                    self.assertEqual(b.scale(torch.tensor([4.0], dtype=torch.float32, device=device)), 12.0)

    # FIXME: move to test distributions
    def _test_multinomial_empty(self, device, replacement, num_samples):
        probs = torch.ones(0, 3, device=device)
        expected = torch.empty(0, num_samples, dtype=torch.int64)
        out = torch.multinomial(probs, num_samples=num_samples, replacement=replacement)
        self.assertEqual(out, expected)

    # FIXME: move to test distributions
    def test_multinomial_empty_w_replacement(self, device):
        self._test_multinomial_empty(device, True, 1)
        self._test_multinomial_empty(device, True, 2)

    # FIXME: move to test distributions
    def test_multinomial_empty_wo_replacement(self, device):
        self._test_multinomial_empty(device, False, 1)
        self._test_multinomial_empty(device, False, 2)

    @onlyNativeDeviceTypes
    @dtypes(torch.float, torch.double)
    def test_grad_scaling_unscale(self, device, dtype):
        device = torch.device(device)
        device0 = "cuda:0" if device.type == "cuda" else "cpu"
        inv_scale = torch.full((1,), 0.25, dtype=torch.float, device=device0)
        found_inf = torch.full((1,), 0.0, dtype=torch.float, device=device0)

        size = 20
        g = torch.full((size, size), 4.0, dtype=dtype, device=device0)
        ginf = g.clone()
        ginf[2, 2] = float('inf')
        gnan = g.clone()
        gnan[2, 2] = float('nan')

        # Tries selected combinations of
        #  - contiguous grads
        #  - g.clone().t() which is not contiguous but still non overlapping and dense
        #  - variants of g.clone()[:, :5] which are not non overlapping and dense
        # Non overlapping and dense grads route into a multi tensor apply kernel,
        # others use a fallback per-tensor kernel, so we should try both.
        cases = (
            ([g.clone(), g.clone()], False),
            ([g.clone(), g.clone().t()], False),
            ([g.clone(), g.clone()[:, :5]], False),
            ([g.clone()[:, :5], g.clone()[:, :5]], False),
            ([g.clone(), ginf.clone()], True),
            ([g.clone(), gnan.clone()], True),
            ([g.clone(), ginf.clone()[:, :5]], True),
            ([g.clone(), gnan.clone()[:, :5]], True),
            ([ginf.clone(), g.clone()[:, :5]], True),
            ([ginf.clone()[:, :5], g.clone()[:, :5]], True),
        )

        for grads, has_inf in cases:
            found_inf.zero_()
            torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale)
            if has_inf:
                self.assertEqual(found_inf, 1.0)
            else:
                self.assertEqual(found_inf, 0.0)
                for grad in grads:
                    self.assertEqual(grad, torch.ones_like(grad), rtol=1e-5, atol=1e-7)

        # When passing lists with mismatched dtypes to a raw
        # _amp_foreach_non_finite_check_and_unscale_ call on CUDA,
        # it's expected to fall back to single-tensor TensorIterator kernel.
        grads = [g.clone(), g.to(dtype=torch.float16)]
        torch._amp_foreach_non_finite_check_and_unscale_(grads, found_inf, inv_scale)
        for grad in grads:
            self.assertEqual(grad, torch.on