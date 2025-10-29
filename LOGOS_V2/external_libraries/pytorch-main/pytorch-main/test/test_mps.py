# Owner(s): ["module: mps"]
# ruff: noqa: F841
import io
import sys
import math
import random
import unittest
import warnings
import shutil
import subprocess
import tempfile
import os
import copy
import gc
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from collections import defaultdict
from torch import inf
from torch.nn import Buffer, Parameter
from torch.testing._internal import opinfo
from torch.testing._internal.common_utils import \
    (gradcheck, gradgradcheck, parametrize, run_tests, TestCase, download_file, MACOS_VERSION, IS_CI,
     NoTest, skipIfSlowGradcheckEnv, suppress_warnings, serialTest, instantiate_parametrized_tests, xfailIf)
from torch.testing._internal.common_mps import mps_ops_modifier, mps_ops_grad_modifier, mps_ops_error_inputs_modifier
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import get_all_dtypes, integral_types
import torch.backends.mps
from torch.distributions import Uniform, Exponential
from functools import partial

from torch.testing._internal.common_methods_invocations import (
    op_db,
    UnaryUfuncInfo,
    ReductionOpInfo,
    SpectralFuncInfo,
    BinaryUfuncInfo,
)
from torch.testing._internal.common_device_type import ops, dtypes, instantiate_device_type_tests, OpDTypes
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_quantization import _group_quantize_tensor, _dynamically_quantize_per_channel
import numpy as np
import torch
import torch.utils._pytree as pytree
from itertools import product
import operator

test_consistency_op_db = copy.deepcopy(op_db)
test_error_inputs_op_db = copy.deepcopy(op_db)

# Add bicubic2d_aa to test_consistency_op_db
for op in op_db:
    if op.name != "_upsample_bilinear2d_aa":
        continue
    op = copy.deepcopy(op)
    op.name = "_upsample_bicubic2d_aa"
    op.op = torch.ops.aten._upsample_bicubic2d_aa
    test_consistency_op_db.append(op)
    break

# Copied from `test_ops.py` for the purposes of duplicating `test_numpy_ref`
_ref_test_ops = tuple(
    filter(
        lambda op: not isinstance(
            op, (UnaryUfuncInfo, ReductionOpInfo, SpectralFuncInfo, BinaryUfuncInfo)
        )
        and op.ref is not None,
        op_db,
    )
)

# Same logic as test_cuda.py
if not torch.backends.mps.is_available():
    print('MPS not available, skipping tests', file=sys.stderr)
    TestCase = NoTest  # noqa: F811
    NNTestCase = NoTest  # noqa: F811

total_memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))

# Determine whether to enable MPS memory leak check (uses same code as CUDA).
TEST_MPS_MEM_LEAK_CHECK = os.getenv('PYTORCH_TEST_MPS_MEM_LEAK_CHECK', '0') == '1'

def skipMPSMemoryLeakCheckIf(condition):
    def dec(fn):
        if getattr(fn, '_do_mps_memory_leak_check', True):
            fn._do_mps_memory_leak_check = not condition
        return fn
    return dec

class MpsMemoryLeakCheck:
    def __init__(self, testcase, name=None):
        self.name = testcase.id() if name is None else name
        self.testcase = testcase

    def __enter__(self):
        # Performs a gc if required (required if any memory is held)
        caching_allocator_mem_allocated = torch.mps.current_allocated_memory()
        if caching_allocator_mem_allocated > 0:
            gc.collect()
            torch.mps.empty_cache()

        # Acquires caching allocator and driver statistics before the test is run
        self.caching_allocator_before = torch.mps.current_allocated_memory()
        self.driver_before = torch.mps.driver_allocated_memory()

    def __exit__(self, exc_type, exc_value, traceback):
        # Don't check for leaks if an exception was thrown
        if exc_type is not None:
            return
        # Compares caching allocator before/after statistics
        # An increase in allocated memory is a discrepancy indicating a possible memory leak
        discrepancy_detected = False
        caching_allocator_mem_allocated = torch.mps.current_allocated_memory()
        if caching_allocator_mem_allocated > self.caching_allocator_before:
            discrepancy_detected = True

        # Short-circuits if no discrepancy detected
        if not discrepancy_detected:
            return
        # Validates the discrepancy persists after garbage collection and
        # is confirmed by the driver API
        gc.collect()
        torch.mps.empty_cache()

        discrepancy_detected = True
        # Query memory multiple items to ensure leak was not transient
        for n in range(3):
            caching_allocator_mem_allocated = torch.mps.current_allocated_memory()
            driver_mem_allocated = torch.mps.driver_allocated_memory()

            caching_allocator_discrepancy = False
            driver_discrepancy = False

            if caching_allocator_mem_allocated > self.caching_allocator_before:
                caching_allocator_discrepancy = True

            if driver_mem_allocated > self.driver_before:
                driver_discrepancy = True

            if not (caching_allocator_discrepancy or driver_discrepancy):
                # Leak was false positive, exit loop
                discrepancy_detected = False
                break

        if caching_allocator_discrepancy and not driver_discrepancy:
            # Just raises a warning if the leak is not validated by the driver API
            msg = ("MPS caching allocator reports a memory leak not "
                   f"verified by the driver API in {self.name}! "
                   f"Caching allocator allocated memory was {self.caching_allocator_before} "
                   f"and is now reported as {caching_allocator_mem_allocated}. "
                   f"MPS driver allocated memory was {self.driver_before} and is now {driver_mem_allocated}.")
            warnings.warn(msg)
        elif caching_allocator_discrepancy and driver_discrepancy:
            # A caching allocator discrepancy validated by the driver API is a failure
            msg = (f"MPS driver API confirmed a leak in {self.name}! "
                   f"Caching allocator allocated memory was {self.caching_allocator_before} "
                   f"and is now reported as {caching_allocator_mem_allocated}. "
                   f"MPS driver allocated memory was {self.driver_before} and is now {driver_mem_allocated}.")

            raise RuntimeError(msg)

class TestAutocastMPS(TestCase):

    def test_matmul_autocast(self):
        autocast_tensor_A = torch.rand((8, 8), device="mps")
        autocast_tensor_B = torch.rand((8, 8), device="mps")
        tensor_A = autocast_tensor_A.detach().clone()
        tensor_B = autocast_tensor_B.detach().clone()
        autocast_output_tensor = torch.empty(8, 8)
        output_tensor = autocast_output_tensor.detach().clone()

        with torch.autocast(device_type="mps"):
            autocast_output_tensor = torch.mm(autocast_tensor_A, autocast_tensor_B)
            autocast_output_tensor = torch.mm(autocast_tensor_A, autocast_output_tensor)

        output_tensor = torch.mm(tensor_A, tensor_B)
        output_tensor = torch.mm(tensor_A, output_tensor)

        self.assertEqual(autocast_output_tensor.dtype, torch.float16, "Autocast output tensor was not expected type float16")
        self.assertEqual(autocast_output_tensor,
                         output_tensor.to(torch.float16),
                         f"Autocast & non-autocast tensors did not match, \
                         got:\n{autocast_output_tensor} \n{output_tensor.to(torch.float16)}")

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_scaled_dot_product_attention_autocast(self, dtype):
        # Regression test for https://github.com/pytorch/pytorch/issues/141774
        if dtype == torch.bfloat16 and MACOS_VERSION < 14.0:
            raise unittest.SkipTest("bfloat16 needs MacOS14+")

        query = torch.rand(4, 1, 16, 8, dtype=torch.float32, device="mps")
        key = torch.rand(4, 1, 16, 8, dtype=torch.float32, device="mps")
        value = torch.rand(4, 1, 16, 8, dtype=dtype, device="mps")

        with torch.amp.autocast(device_type="mps"):
            y_autocast = F.scaled_dot_product_attention(query, key, value)

        y = F.scaled_dot_product_attention(query, key, value.to(torch.float32))
        self.assertEqual(y.to(y_autocast.dtype), y_autocast)

    def test_gradscaler_mps(self):
        # big model to force chunking/depth in the gradscaler dispatch
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 2048)
                self.fc2 = nn.Linear(2048, 2048)
                self.fc3 = nn.Linear(2048, 2048)
                self.fc4 = nn.Linear(2048, 2048)
                self.fc5 = nn.Linear(2048, 5)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.relu(self.fc3(x))
                x = self.relu(self.fc4(x))
                return self.fc5(x)
        torch.manual_seed(42)

        def helper(model_cpu, model_mps, dtype, iterations, batch_size, atol=3e-4, rtol=1e-5):
            if dtype == torch.bfloat16 and MACOS_VERSION < 14.0:
                raise unittest.SkipTest("bfloat16 needs MacOS14+")
            optimizer_cpu = torch.optim.SGD(model_cpu.parameters(), lr=0.01)
            optimizer_mps = torch.optim.SGD(model_mps.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()

            input_cpu = torch.randn(batch_size, 10)
            target_cpu = torch.randn(batch_size, 5)
            input_mps = input_cpu.to('mps')
            target_mps = target_cpu.to('mps')

            scaler_cpu = torch.amp.GradScaler(device="cpu")
            scaler_mps = torch.amp.GradScaler(device="mps")
            for _ in range(iterations):
                optimizer_cpu.zero_grad()
                optimizer_mps.zero_grad()

                with torch.amp.autocast(device_type="cpu", dtype=dtype):
                    output_cpu = model_cpu(input_cpu)
                    loss_cpu = loss_fn(output_cpu, target_cpu)
                scaler_cpu.scale(loss_cpu).backward()
                scaler_cpu.step(optimizer_cpu)
                scaler_cpu.update()

                with torch.autocast(device_type="mps", dtype=dtype):
                    output_mps = model_mps(input_mps)
                    loss_mps = loss_fn(output_mps, target_mps)
                scaler_mps.scale(loss_mps).backward()
                scaler_mps.step(optimizer_mps)
                scaler_mps.update()

            for p_cpu, p_mps in zip(model_cpu.parameters(), model_mps.parameters()):
                self.assertEqual(p_mps.cpu(), p_cpu, rtol=rtol, atol=atol)

        model_cpu = Model().to('cpu')
        model_mps = Model().to('mps')
        model_mps.load_state_dict(model_cpu.state_dict())

        helper(model_cpu, model_mps, torch.float16, iterations=5, batch_size=4)
        helper(model_cpu, model_mps, torch.bfloat16, iterations=5, batch_size=4)

    def test_non_fast_path_amp_unscale(self):
        torch.manual_seed(42)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 10)
                self.linear2 = nn.Linear(10, 10)

            def forward(self, x):
                x = self.linear1(x)
                x = F.relu(x)
                x = self.linear2(x)
                x = x.mean(dim=1)
                return x

        cpu_model = Model().to("cpu")
        mps_model = copy.deepcopy(cpu_model).to("mps")

        cpu_optimizer = torch.optim.SGD(cpu_model.parameters(), lr=0.01)
        mps_optimizer = torch.optim.SGD(mps_model.parameters(), lr=0.01)
        cpu_scaler = torch.amp.GradScaler(device="cpu")
        mps_scaler = torch.amp.GradScaler(device="mps")

        def helper(model, optimizer, scaler, device, input, target, apply_grad_transform=False):
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                output = model(input)
                loss = nn.MSELoss()(output, target)
            scaler.scale(loss).backward()

            if apply_grad_transform:
                for p in model.parameters():
                    if p.grad is not None and p.grad.dim() >= 2:
                        p.grad = p.grad.as_strided(p.grad.size(), (1,) * p.grad.dim())

            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

        # CPU forward/backward pass
        input_cpu = torch.randn(32, 10, device="cpu")
        target_cpu = torch.randn(32, device="cpu")
        helper(cpu_model, cpu_optimizer, cpu_scaler, "cpu", input_cpu, target_cpu)

        # MPS forward/backward pass
        input_mps = input_cpu.to("mps")
        target_mps = target_cpu.to("mps")
        helper(mps_model, mps_optimizer, mps_scaler, "mps", input_mps, target_mps, apply_grad_transform=True)

        updated_linear1_weight_cpu = cpu_model.linear1.weight.detach()
        updated_linear2_weight_cpu = cpu_model.linear2.weight.detach()
        updated_linear1_weight_mps = mps_model.linear1.weight.detach().cpu()
        updated_linear2_weight_mps = mps_model.linear2.weight.detach().cpu()

        self.assertEqual(updated_linear1_weight_cpu, updated_linear1_weight_mps, atol=6e-4, rtol=1e-6)
        self.assertEqual(updated_linear2_weight_cpu, updated_linear2_weight_mps, atol=6e-4, rtol=1e-6)

# Expand TestCase class with Memory Leak Detection on MPS device
class TestCaseMPS(TestCase):
    _do_mps_memory_leak_check = True

    def __init__(self, method_name='runTest'):
        super().__init__(method_name)
        test_method = getattr(self, method_name, None)
        if test_method is not None:
            # Wraps the tested method if we should do MPS memory check.
            if TEST_MPS_MEM_LEAK_CHECK:
                if self._do_mps_memory_leak_check:
                    self.wrap_with_mps_policy(method_name, self.assertLeaksNoMpsTensors)

    def assertLeaksNoMpsTensors(self, name=None):
        name = self.id() if name is None else name
        return MpsMemoryLeakCheck(self, name)

    def wrap_with_mps_policy(self, method_name, policy):
        test_method = getattr(self, method_name)
        setattr(self, method_name, super().wrap_method_with_policy(test_method, policy))

    # checks for leaks even if TEST_MPS_MEM_LEAK_CHECK is 0
    def wrap_with_mps_memory_check(self, method):
        return super().wrap_method_with_policy(method, self.assertLeaksNoMpsTensors)

class TestMemoryLeak(TestCaseMPS):
    def test_mps_memory_leak_detection(self):
        l = []

        @self.wrap_with_mps_memory_check
        def no_leak():
            pass

        # Trigger an intentional memory leak
        @self.wrap_with_mps_memory_check
        def leak_gpu0():
            # increasing to 8MB to force acquiring a new block and overcome blocksize differences across platforms
            l.append(torch.randn(1024 * 1024 * 8, device=torch.device("mps")))

        no_leak()

        # check if a runtime error for memory leak was emitted which would
        # confirm whether memory leak detection worked successfully or not.
        with self.assertRaisesRegex(RuntimeError, r"MPS driver API confirmed .+"):
            leak_gpu0()

    def test_copy_cast_no_leak(self):

        def step(x):
            x = x.to(device='cpu', dtype=torch.float32)
            x = x.to(device='mps', dtype=torch.float16)

        a = torch.randn(128, 128, device='mps', dtype=torch.float16)
        # Warm up / prebuild MPS shaders (otherwise check fails on 13.2)
        step(a)
        torch.mps.empty_cache()
        driver_before = torch.mps.driver_allocated_memory()
        step(a)
        torch.mps.empty_cache()
        driver_after = torch.mps.driver_allocated_memory()
        self.assertEqual(driver_before, driver_after, f"Detected {driver_after - driver_before} bytes leak of GPU memory")


class TestPixelShuffle(TestCaseMPS):
    def test_pixel_shuffle_unshuffle(self):
        def _test_pixel_shuffle_unshuffle_helper(num_input_dims, valid_channels_dim=True,
                                                 upscale_factor=None, is_contiguous=True):

            def generate_input():
                # If valid_channels_dim=False, add 1 to make channels dim indivisible by upscale_factor ** 2.
                channels = random.randint(1, 4) * upscale_factor ** 2 + (0 if valid_channels_dim else 1)
                height = random.randint(5, 10)
                width = random.randint(5, 10)

                if num_input_dims == 1:
                    input = torch.rand(channels, requires_grad=True, device='mps')
                    assert is_contiguous
                elif num_input_dims == 2:
                    input = torch.rand(width, height, requires_grad=True, device='mps').T
                    if is_contiguous:
                        input = input.contiguous()
                else:
                    batch_sizes = [random.randint(1, 3) for _ in range(num_input_dims - 3)]
                    input = torch.rand(*batch_sizes, channels, width, height, requires_grad=True, device='mps')
                    input = input.transpose(-1, -2)
                    if is_contiguous:
                        input = input.contiguous()

                if not is_contiguous and len(input.reshape(-1)) > 0:
                    assert not input.is_contiguous()

                input = input.detach().clone()
                input.requires_grad = True
                return input

            # Function to imperatively ensure pixels are shuffled to the correct locations.
            # Used to validate the batch operations in pixel_shuffle.
            def _verify_pixel_shuffle(input, output, upscale_factor):
                for c in range(output.size(-3)):
                    for h in range(output.size(-2)):
                        for w in range(output.size(-1)):
                            height_idx = h // upscale_factor
                            weight_idx = w // upscale_factor
                            channel_idx = (upscale_factor * (h % upscale_factor)) + (w % upscale_factor) + \
                                          (c * upscale_factor ** 2)
                            self.assertEqual(output[..., c, h, w], input[..., channel_idx, height_idx, weight_idx])

            upscale_factor = random.randint(2, 5) if upscale_factor is None else upscale_factor
            input = generate_input()

            ps = nn.PixelShuffle(upscale_factor)
            pus = nn.PixelUnshuffle(downscale_factor=upscale_factor)

            if num_input_dims >= 3 and valid_channels_dim and upscale_factor > 0:
                output = ps(input)
                _verify_pixel_shuffle(input, output, upscale_factor)
                output.backward(output.data)
                self.assertEqual(input.data, input.grad.data)

                # Ensure unshuffle properly inverts shuffle.
                unshuffle_output = pus(output)
                self.assertEqual(input, unshuffle_output)
            else:
                self.assertRaises(RuntimeError, lambda: ps(input))

        def _test_pixel_unshuffle_error_case_helper(num_input_dims, valid_height_dim=True, valid_width_dim=True,
                                                    downscale_factor=None):
            downscale_factor = random.randint(2, 5) if downscale_factor is None else downscale_factor
            channels = random.randint(1, 4)
            # If valid_height_dim=False, add 1 to make height dim indivisible by downscale_factor.
            height = random.randint(3, 5) * abs(downscale_factor) + (0 if valid_height_dim else 1)
            # If valid_width_dim=False, add 1 to make width dim indivisible by downscale_factor.
            width = random.randint(3, 5) * abs(downscale_factor) + (0 if valid_width_dim else 1)

            if num_input_dims == 1:
                input = torch.rand(channels, requires_grad=True, device='mps')
            elif num_input_dims == 2:
                input = torch.rand(height, width, requires_grad=True, device='mps')
            else:
                batch_sizes = [random.randint(1, 3) for _ in range(num_input_dims - 3)]
                input = torch.rand(*batch_sizes, channels, height, width, requires_grad=True, device='mps')

            pus = nn.PixelUnshuffle(downscale_factor)
            self.assertRaises(RuntimeError, lambda: pus(input))

        def _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims):
            # For 1D - 2D, this is an error case.
            # For 3D - 5D, this is a success case for pixel_shuffle + pixel_unshuffle.
            is_contiguous_check = [True, False] if num_input_dims > 1 else [True]
            for is_contiguous in is_contiguous_check:
                _test_pixel_shuffle_unshuffle_helper(
                    num_input_dims=num_input_dims, is_contiguous=is_contiguous
                )
                _test_pixel_shuffle_unshuffle_helper(
                    num_input_dims=num_input_dims, valid_channels_dim=False, is_contiguous=is_contiguous
                )
                _test_pixel_shuffle_unshuffle_helper(
                    num_input_dims=num_input_dims, upscale_factor=0, is_contiguous=is_contiguous
                )
                _test_pixel_shuffle_unshuffle_helper(
                    num_input_dims=num_input_dims, upscale_factor=-2, is_contiguous=is_contiguous
                )

                # Error cases for pixel_unshuffle.
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, valid_height_dim=False)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, valid_width_dim=False)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, downscale_factor=0)
            _test_pixel_unshuffle_error_case_helper(num_input_dims=num_input_dims, downscale_factor=-2)

        def test_pixel_shuffle_unshuffle_1D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=1)

        def test_pixel_shuffle_unshuffle_2D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=2)

        def test_pixel_shuffle_unshuffle_3D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=3)

        def test_pixel_shuffle_unshuffle_4D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=4)

        def test_pixel_shuffle_unshuffle_5D():
            _test_pixel_shuffle_unshuffle_for_input_dims(num_input_dims=5)

        test_pixel_shuffle_unshuffle_1D()
        test_pixel_shuffle_unshuffle_2D()
        test_pixel_shuffle_unshuffle_3D()
        test_pixel_shuffle_unshuffle_4D()
        test_pixel_shuffle_unshuffle_5D()

class MPSReluTest(TestCaseMPS):
    def _npRelu(self, np_features):
        return np.maximum(np_features, np.zeros(np_features.shape)).astype(np_features.dtype)

    def testNpRelu(self):
        torch.testing.assert_close(
            np.array([[0., 0.7, 0.0, 0.3, 0.0], [0.1, 0.0, 0.5, 0.0, 0.9]]),
            self._npRelu(
                np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                         0.9]])))

    def _testRelu(self, np_features, device):
        np_relu = self._npRelu(np_features)
        # Convert the numpy array to a PyTorch Tensor,
        # and move the Tensor to the CPU/GPU based on the "device" parameter
        py_tensor = torch.from_numpy(np_features).to(device)
        py_relu = torch.nn.ReLU(inplace=False)(py_tensor)
        py_relu_cpu = py_relu.to("cpu")

        self.assertEqual(np_relu, py_relu_cpu)

    def _testReluInPlace(self, np_features, device):
        np_relu = self._npRelu(np_features)
        # Convert the numpy array to a PyTorch Tensor,
        # and move the Tensor to the CPU/GPU based on the "device" parameter
        py_tensor = torch.from_numpy(np_features).to(device)
        py_relu = torch.nn.ReLU(inplace=True)(py_tensor)
        py_relu_cpu = py_relu.to("cpu")

        self.assertEqual(np_relu, py_relu_cpu)
        # Inplace Relu modifies the initial input and it should match the output of Relu
        self.assertEqual(np_relu, py_tensor.to("cpu"))

    def testNumbersCPU(self):
        for t in [np.int32]:
            # Force execution on CPU even if a GPU kernel is available for the type.
            self._testRelu(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                device="cpu")
            self._testReluInPlace(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                device="cpu")

    def testNumbersGPU(self):
        for t in [np.float16, np.float32]:
            self._testRelu(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                device="mps")
            self._testReluInPlace(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                device="mps")
            self._testRelu(np.array([]).astype(t), device="mps")
            self._testReluInPlace(np.array([]).astype(t), device="mps")

class MatmulTest(TestCaseMPS):
    def _helper(self, shape_tensor_1, shape_tensor_2, expand_tensor_1_shape=None, expand_tensor_2_shape=None):
        if expand_tensor_1_shape:
            tensor1_mps = torch.randn(shape_tensor_1, device="mps").expand(expand_tensor_1_shape)
        else:
            tensor1_mps = torch.randn(shape_tensor_1, device="mps")

        if expand_tensor_2_shape:
            tensor2_mps = torch.randn(shape_tensor_2, device="mps").expand(expand_tensor_2_shape)
        else:
            tensor2_mps = torch.randn(shape_tensor_2, device="mps")

        tensor1_cpu = tensor1_mps.to("cpu")
        tensor2_cpu = tensor2_mps.to("cpu")

        matmul_cpu = torch.matmul(tensor1_cpu, tensor2_cpu)
        matmul_mps = torch.matmul(tensor1_mps, tensor2_mps)

        self.assertEqual(matmul_cpu, matmul_mps.to("cpu"))

    def test_vector_x_vector(self):
        # uses `dot`
        self._helper(3, 3)

    def test_matrix_x_vector(self):
        # uses `addmv`
        self._helper((3, 4), 4)

    def test_batched_matrix_x_broadcasted_vector(self):
        self._helper((10, 3, 4), 4)

    def test_batched_matrix_x_batched_matrix(self):
        # uses `bmm.out`
        self._helper((10, 3, 4), (10, 4, 5))

    def test_batched_matrix_x_broadcasted_matrix(self):
        self._helper((10, 3, 4), (4, 5))

    def test_large_matmul(self):
        # Issue: #141909
        tensor1_mps = torch.randn(1, 1, 72250, dtype=torch.half)
        tensor2_mps = torch.randn(1, 72250, 1, dtype=torch.half)
        matmul_mps = torch.matmul(tensor1_mps, tensor2_mps)

        tensor1_cpu = tensor1_mps.to("cpu")
        tensor2_cpu = tensor2_mps.to("cpu")
        matmul_cpu = torch.matmul(tensor1_cpu, tensor2_cpu)

        self.assertEqual(matmul_cpu, matmul_mps.to("cpu"))

class MPSLeakyReluTest(TestCaseMPS):
    def _npLeakyRelu(self, np_features, negative_slope=0.1):
        return np.maximum(np_features, negative_slope * np_features).astype(np_features.dtype)

    def testNpLeakyRelu(self):
        torch.testing.assert_close(
            np.array([[-0.09, 0.7, -0.05, 0.3, -0.01],
                      [0.1, -0.03, 0.5, -0.07, 0.9]]),
            self._npLeakyRelu(
                np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                         0.9]]),
                negative_slope=0.1))

    def _testLeakyRelu(self, shape, dtype, negative_slope, contiguous):
        cpu_x = torch.randn(shape, device='cpu', dtype=dtype)
        mps_x = cpu_x.detach().clone().to('mps')

        if not contiguous and not (0 in shape or len(shape) < 2):
            # Transposing will make the tensor non-contiguous
            cpu_x = cpu_x.transpose(0, 1)
            mps_x = mps_x.transpose(0, 1)
            assert not mps_x.is_contiguous()

        cpu_x.requires_grad_()
        mps_x.requires_grad_()

        relu_op = torch.nn.LeakyReLU(negative_slope)

        cpu_leaky_relu = relu_op(cpu_x)
        mps_leaky_relu = relu_op(mps_x)
        torch.testing.assert_close(cpu_leaky_relu, mps_leaky_relu.to('cpu'))

        # test backward pass

        cpu_grad = torch.ones_like(cpu_leaky_relu)
        mps_grad = cpu_grad.to('mps')

        mps_leaky_relu.backward(gradient=mps_grad)
        cpu_leaky_relu.backward(gradient=cpu_grad)

        assert cpu_x.grad is not None  # Check that the grad is well-populated
        self.assertEqual(cpu_x.grad, mps_x.grad)

    def testNumbersCPU(self):
        for t in [torch.float, torch.half]:
            for shape in [[], (0,), (0, 3), (4,), (4, 3), (5, 4, 3)]:
                for contiguous in [True, False]:
                    self._testLeakyRelu(shape,
                                        dtype=t,
                                        negative_slope=0.2,
                                        contiguous=contiguous)

class TestAvgPool(TestCaseMPS):
    def _sum_pool2d(self, x, kernel_size):
        windows = torch.nn.functional.unfold(x, kernel_size=kernel_size, stride=kernel_size)
        return torch.sum(windows, dim=1)

    def _sum_pool3d(self, x, kernel_size):
        # Because unfold does not support 3D sliding window we will split tensor to multiple tensors and calculate sum
        h = kernel_size[0]
        splited_x = [t.sum(0) for t in x.split(h) if t.size(0) == h]
        # sum_pool2d assumes tensor in (1, 1, n, m) view, so unsqueeze two times
        splited_x = [self._sum_pool2d(t.unsqueeze(0).unsqueeze(0), kernel_size[1:]) for t in splited_x]
        joined_x = torch.cat(splited_x)
        return joined_x.view(1, joined_x.numel())

    def _avg_pool2d(self, x, kernel_size):
        size = reduce(operator.mul, kernel_size)  # noqa: F821
        return self._sum_pool2d(x, kernel_size) / size

    def _avg_pool3d(self, x, kernel_size):
        size = reduce(operator.mul, kernel_size)  # noqa: F821
        return self._sum_pool3d(x, kernel_size) / size

    def test_avg_pool2d_with_zero_divisor(self):
        self.assertRaisesRegex(RuntimeError, "divisor must be not zero",
                               lambda: F.avg_pool2d(torch.zeros(3, 3, 3), (2, 2), divisor_override=0))

    def test_doubletensor_avg_pool2d_with_divisor(self):
        n, m = 3, 3
        input = torch.rand(1, 1, n, m)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                for divisor in [1, 7, i * j]:
                    actual = F.avg_pool2d(input[0], (i, j), divisor_override=divisor)
                    actual = actual.view(1, actual.numel())
                    expected = self._sum_pool2d(input, (i, j)) / divisor
                    self.assertEqual(actual, expected, rtol=0, atol=1e-5)

    def test_avg_pool2d_ceil_mode(self):
        # Regression test for gh-36977
        x = 10 * torch.randn((1, 16, 4, 4))
        y = torch.nn.functional.avg_pool2d(
            x, ceil_mode=True, count_include_pad=True, kernel_size=(1, 2),
            padding=(0, 1), stride=2)
        self.assertFalse(torch.isnan(y).any())
        y = torch.nn.functional.avg_pool2d(
            x.to('mps'), ceil_mode=True, count_include_pad=True, kernel_size=(1, 2),
            padding=(0, 1), stride=2)
        self.assertFalse(torch.isnan(y).any())


class TestMPS(TestCaseMPS):
    def test_exp(self, device="mps", dtype=torch.float):
        for v in (2, -2) + ((1j, 1 + 1j) if dtype.is_complex else ()):
            b = torch.arange(18, dtype=dtype, device=device) / 3 * math.pi
            a = torch.tensor(v, dtype=dtype, device="mps") * b
            self.compare_with_numpy(torch.exp, np.exp, a)

    @xfailIf(MACOS_VERSION > 15.0)
    def test_conv_raises_error(self, device='mps', dtype=torch.float):
        conv = nn.Conv1d(1, 65537, 3, padding=1).to('mps')

        x = torch.ones([1, 1, 3])
        with self.assertRaises(NotImplementedError):
            y = conv(x.to("mps"))

    @xfailIf(MACOS_VERSION < 15.1)
    def test_conv_high_channel_size(self):
        out_channels = 65537
        weight = torch.randn(out_channels, 1, 1)
        x = torch.ones([1, 1, 1])
        y_cpu = F.conv1d(x.to("cpu"), weight.to("cpu"))
        y_mps = F.conv1d(x.to("mps"), weight.to("mps"))
        self.assertEqual(y_cpu, y_mps)

    def test_triu_inf(self, device="mps", dtype=torch.float):
        for diag in [-1, 0, 1]:
            mask = torch.full((3, 6, 6), float("-inf"))
            mask_mps = mask.detach().clone().to('mps')
            cpu_ref = torch.triu(mask, diagonal=diag)
            mps_out = torch.triu(mask_mps, diagonal=diag)
            self.assertEqual(cpu_ref, mps_out)

    def test_exp1(self, device="mps", dtype=torch.float):
        input = torch.tensor([-0.1, 1.0, -0.9, 0.1], device=device, dtype=dtype)
        output = torch.exp(input)
        output_cpu = torch.exp(input.cpu())
        # If exponentWithTensor: MPS call is used on M1 running 14.5 test will fail with
        # Mismatched elements: 3 / 4 (75.0%)
        # Greatest absolute difference: 1.1920928955078125e-07 at index (3,) (up to 1e-08 allowed)
        # Greatest relative difference: 1.0786502002702036e-07 at index (3,) (up to 1e-08 allowed)
        self.assertEqual(output, output_cpu, atol=1e-8, rtol=1e-8)

    def test_exp_strided_output(self):
        x = torch.rand((256, 10), device='mps')
        x_cpu = x.to("cpu")

        x = x.permute(1, 0)
        x_cpu = x_cpu.permute(1, 0)

        res = x.exp()
        res_cpu = x_cpu.exp()
        self.assertEqual(res, res_cpu)

    def _testLeakyRelu(self, np_features, negative_slope, device):
        cpu_x = torch.from_numpy(np_features).requires_grad_()
        mps_x = torch.from_numpy(np_features).to('mps').requires_grad_()
        relu_op = torch.nn.LeakyReLU(negative_slope)

        cpu_leaky_relu = relu_op(cpu_x)
        mps_leaky_relu = relu_op(mps_x)
        torch.testing.assert_close(cpu_leaky_relu, mps_leaky_relu.to('cpu'))

        # test backward pass
        cpu_grad = torch.ones_like(cpu_leaky_relu)
        mps_grad = cpu_grad.to('mps')
        cpu_leaky_relu.backward(gradient=cpu_grad)
        mps_leaky_relu.backward(gradient=mps_grad)
        torch.testing.assert_close(cpu_x.grad, mps_x.grad.to('cpu'))

    def testNumbersGPU(self):
        for t in [np.float32]:
            self._testLeakyRelu(
                np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
                negative_slope=0.1,
                device="mps")

    def test_fill(self):

        def helper(val, shape, dtype):
            tensor = torch.zeros(shape, device='mps', dtype=dtype)
            tensor_mps = tensor.fill_(val)

            tensor_0 = torch.zeros(shape, device='cpu', dtype=dtype)
            tensor_cpu = tensor_0.fill_(val)

            self.assertEqual(tensor_mps, tensor_cpu)

        helper(0, [1024], torch.float32)
        helper(0.2, [2, 3], torch.float32)
        helper(0.2 + 0.5j, [2, 3], torch.complex64)

    def test_fill_storage_offset(self):
        shape = [2, 10]
        val = 0.2
        tensor = torch.ones(shape, device="mps")
        tensor_mps = tensor[:][1].fill_(val)
        tensor_0 = torch.ones(shape, device="cpu")
        tensor_cpu = tensor_0[:][1].fill_(val)

        self.assertEqual(tensor_mps, tensor_cpu)
        self.assertEqual(tensor, tensor_0)

        shape = [1, 10]
        val = 0.0
        tensor = torch.ones(shape, device="mps")
        val_tensor_mps = torch.tensor(val, device="mps")
        tensor_mps = tensor[:, 9].fill_(val_tensor_mps)
        # Regression test for https://github.com/pytorch/pytorch/issues/114692
        tensor[:, 5].fill_(val_tensor_mps)
        tensor_0 = torch.ones(shape, device="cpu")
        val_tensor_cpu = torch.tensor(val, device="cpu")
        tensor_cpu = tensor_0[:, 9].fill_(val_tensor_cpu)
        tensor_0[:, 5].fill_(val_tensor_cpu)

        self.assertEqual(tensor_mps.to(device="cpu"), tensor_cpu)
        self.assertEqual(tensor.to(device="cpu"), tensor_0)

    def test_cdist_large(self, device="mps"):
        for cm in ['use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            x = torch.randn(100, 10, device=device)
            y = torch.randn(100, 10, device=device)
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertEqual(expected, actual)

    def test_cdist_large_batch(self, device="mps"):
        for cm in ['use_mm_for_euclid_dist_if_necessary', 'use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
            x = torch.randn(4, 3, 100, 10, device=device)
            y = torch.randn(4, 3, 100, 10, device=device)
            actual = torch.cdist(x, y, p=2, compute_mode=cm)
            expected = self._brute_cdist(x, y, p=2)
            self.assertEqual(expected, actual)

    def test_cdist_non_contiguous(self, device="mps"):
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

    def test_cdist_non_contiguous_batch(self, device="mps"):
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

    def test_cdist_euclidean_large(self, device="mps"):
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

    def test_cdist_same_inputs(self, device="mps"):
        # Test to detect issues in cdist gradient calculation
        # When the distances are 0
        sizex = (1, 27, 32)
        for p in [0, 1, 2, 3, 1.5, 2.5, float('inf')]:
            x = torch.randn(sizex, device=device, dtype=torch.float)
            dist_grad = torch.randn((1, 27, 27), device=device, dtype=torch.float)
            y = x.clone()
            eps = 1e-6
            x.requires_grad = True
            d = torch.cdist(x, y)
            d.backward(dist_grad)
            # Check that the backward pass does not contain invalid
            # values such as nan or inf
            assert torch.isfinite(x.grad).all()


    def _brute_cdist(self, x, y, p=2):
        r1 = x.shape[-2]
        r2 = y.shape[-2]
        if r1 == 0 or r2 == 0:
            return torch.empty(r1, r2, device=x.device)
        return torch.norm(x[..., None, :] - y[..., None, :, :], p=p, dim=-1)

    def test_cdist_norm(self, device="mps"):
        for r1 in [3, 4]:
            for m in [2, 3]:
                for r2 in [4, 6]:
                    for p in [0, 1, 1.5, 2.5, float('inf')]:
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

    def test_cdist_norm_batch(self, device="mps"):
        for r1 in [3, 4]:
            for m in [2, 3]:
                for r2 in [4, 6]:
                    for p in [0, 3, 1.5, 2.5, float('inf')]:
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

    def test_mm(self):
        B = torch.ones(5, 6).to("mps")
        C = torch.ones(6, 5).to("mps")
        D = torch.mm(B, C).cpu()
        torch.testing.assert_close(D, torch.full((5, 5), 6.0))

    def test_linalg_cross(self):
        def helper(dtype):
            device = "mps"
            if dtype is torch.int32 or dtype is torch.int64:
                x = torch.randint(0, 99999, (100, 3, 100), dtype=dtype, device=device)
                y = torch.randint(0, 99999, (100, 3, 100), dtype=dtype, device=device)
            else:
                x = torch.rand(100, 3, 100, dtype=dtype, device=device)
                y = torch.rand(100, 3, 100, dtype=dtype, device=device)
            x_cpu = x.to("cpu")
            y_cpu = y.to("cpu")
            res1 = torch.linalg.cross(x, y, dim=1)
            res2 = torch.tensor((), dtype=dtype, device=device)
            res1_cpu = torch.linalg.cross(x_cpu, y_cpu, dim=1)
            res2_cpu = torch.tensor((), dtype=dtype, device="cpu")
            torch.linalg.cross(x, y, dim=1, out=res2)
            torch.linalg.cross(x_cpu, y_cpu, dim=1, out=res2_cpu)
            self.assertEqual(res1, res2)
            self.assertEqual(res1, res1_cpu)
            self.assertEqual(res2, res2_cpu)

            # test for broadcastable inputs
            if dtype is torch.int32 or dtype is torch.int64:
                x = torch.randint(0, 99999, (1, 3, 2), dtype=dtype, device=device)
                y = torch.randint(0, 99999, (4, 3, 1), dtype=dtype, device=device)
            else:
                x = torch.rand(1, 3, 2, dtype=dtype, device=device)
                y = torch.rand(4, 3, 1, dtype=dtype, device=device)
            x_cpu = x.to("cpu")
            y_cpu = y.to("cpu")
            res1 = torch.linalg.cross(x, y, dim=1)
            res2 = torch.tensor((), dtype=dtype, device=device)
            res1_cpu = torch.linalg.cross(x_cpu, y_cpu, dim=1)
            res2_cpu = torch.tensor((), dtype=dtype, device="cpu")
            torch.linalg.cross(x, y, dim=1, out=res2)
            torch.linalg.cross(x_cpu, y_cpu, dim=1, out=res2_cpu)
            self.assertEqual(res1, res2)
            self.assertEqual(res1, res1_cpu)
            self.assertEqual(res2, res2_cpu)
        [helper(dtype) for dtype in [torch.int32, torch.int64, torch.float32]]

    def test_cross(self):
        a = torch.randn(4, 3, device="mps")
        b = torch.randn(4, 3, device="mps")
        a_cpu = a.to("cpu")
        b_cpu = b.to("cpu")
        res = torch.cross(a, b, dim=1)
        res_cpu = torch.cross(a_cpu, b_cpu, dim=1)
        self.assertEqual(res, res_cpu)

    def test_addmm(self):
        A = torch.ones(5, 5).to("mps")
        B = torch.ones(5, 6).to("mps")
        C = torch.ones(6, 5).to("mps")
        D = torch.addmm(A, B, C).to("cpu")
        torch.testing.assert_close(D, torch.full((5, 5), 7.0))

    def test_bmm(self):
        batch1_cpu = torch.randn(10, 3, 4)
        batch2_cpu = torch.randn(10, 4, 5)

        batch1_mps = batch1_cpu.detach().clone().to("mps")
        batch2_mps = batch2_cpu.detach().clone().to("mps")

        output_cpu = torch.bmm(batch1_cpu, batch2_cpu)
        output_mps = torch.bmm(batch1_mps, batch2_mps)

        self.assertEqual(output_cpu, output_mps)
        self.assertEqual(output_cpu.size(), output_mps.size())

    @xfailIf(MACOS_VERSION < 15.0)
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_large_bmm(self, dtype):
        B, M, N = 11, 20064, 128
        batch1 = torch.randn(B, M, N, dtype=dtype, device='mps')
        batch2 = torch.randn(B, N, M, dtype=dtype, device='mps')
        output_mps = torch.bmm(batch1, batch2)

        # For performance reasons, check only one(non-first) batch for correctness
        # TODO: Check two when https://github.com/pytorch/pytorch/issues/153560 is fixed
        batch_idx = torch.randint(1, B, size=()).item()
        output_cpu = torch.mm(batch1[batch_idx].cpu(), batch2[batch_idx].cpu())
        # Using the low precision comparison for FP16
        tol = 1e-2 if dtype == torch.float16 else None
        self.assertEqual(output_cpu, output_mps[batch_idx], atol=tol, rtol=tol)

    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_take_along_dim(self, dtype):
        if dtype == torch.bfloat16 and MACOS_VERSION < 14.0:
            raise unittest.SkipTest("bfloat16 needs MacOS14+")

        x = torch.tensor([[-5.], [0.], [5.]], dtype=dtype)
        inds = torch.tensor([[0], [1], [2]])
        ref = torch.take_along_dim(x, inds, 0)
        x_mps = x.detach().clone().to('mps')
        inds_mps = inds.detach().clone().to('mps')
        res = torch.take_along_dim(x_mps, inds_mps, 0)
        self.assertEqual(res, ref)

    def test_addr(self):
        A = torch.ones(5, 10).to("mps")
        B = torch.ones(5).to("mps")
        C = torch.ones(10).to("mps")
        D = torch.addr(A, B, C).to("cpu")
        torch.testing.assert_close(D, torch.full((5, 10), 2.0))

    def test_trace(self):
        M_cpu = torch.randn(3, 3)
        M_mps = M_cpu.detach().clone().to("mps")

        output_cpu = torch.trace(M_cpu)
        output_mps = torch.trace(M_mps)

        self.assertEqual(output_cpu, output_mps)
        self.assertEqual(output_cpu.size(), output_mps.size())

    def test_addbmm(self):
        M_cpu = torch.randn(3, 5)
        batch1_cpu = torch.randn(10, 3, 4)
        batch2_cpu = torch.randn(10, 4, 5)

        M_mps = M_cpu.detach().clone().to("mps")
        batch1_mps = batch1_cpu.detach().clone().to("mps")
        batch2_mps = batch2_cpu.detach().clone().to("mps")

        output_cpu = torch.addbmm(M_cpu, batch1_cpu, batch2_cpu)
        output_mps = torch.addbmm(M_mps, batch1_mps, batch2_mps)

        self.assertEqual(output_cpu, output_mps)
        self.assertEqual(output_cpu.size(), output_mps.size())

    def test_baddbmm(self):
        def helper(input_shape, batch1_shape, batch2_shape):
            M_cpu = torch.randn(input_shape)
            batch1_cpu = torch.randn(batch1_shape)
            batch2_cpu = torch.randn(batch2_shape)
            alpha = 1.2
            beta = 0.8

            M_mps = M_cpu.detach().clone().to("mps")
            batch1_mps = batch1_cpu.detach().clone().to("mps")
            batch2_mps = batch2_cpu.detach().clone().to("mps")

            output_cpu = torch.baddbmm(M_cpu, batch1_cpu, batch2_cpu, beta=beta, alpha=alpha)
            output_mps = torch.baddbmm(M_mps, batch1_mps, batch2_mps, beta=beta, alpha=alpha)

            self.assertEqual(output_cpu, output_mps)
            self.assertEqual(output_cpu.size(), output_mps.size())

        helper(input_shape=(3, 5), batch1_shape=(10, 3, 4), batch2_shape=(10, 4, 5))
        helper(input_shape=(10, 3, 5), batch1_shape=(10, 3, 4), batch2_shape=(10, 4, 5))
        helper(input_shape=(1, 77, 77), batch1_shape=(8, 77, 64), batch2_shape=(8, 64, 77))

    def test_local_scalar_dense_mps(self):
        x_cpu = torch.randn(1)
        y_mps = x_cpu.to("mps")
        torch.testing.assert_close(x_cpu.item(), y_mps.item())

    def test_linear_1d_weight(self):
        device = 'cpu'
        projected = torch.rand([8]).to(device)
        x = torch.rand([1, 2, 2, 8]).to(device)
        x_mps = x.to('mps')
        projected_mps = projected.to('mps')
        linear = F.linear(x, projected)
        linear_mps = F.linear(x_mps, projected_mps)

        self.assertEqual(linear, linear_mps)

        projected = torch.rand([1, 8]).to(device)
        x = torch.rand([1, 2, 2, 8]).to(device)
        x_mps = x.to('mps')
        projected_mps = projected.to('mps')
        linear = F.linear(x, projected)
        linear_mps = F.linear(x_mps, projected_mps)

        self.assertEqual(linear, linear_mps)

    def test_linear_bias(self):
        def helper(bias_shape):
            device = "cpu"
            x = torch.randn(2, 2, 2, 64, device=device)
            linear = torch.nn.Linear(64, 4, device=device)
            linear.bias = torch.nn.Parameter(torch.randn(bias_shape, dtype=torch.float32, device=device))
            y = linear(x)
            device = "mps"
            x_mps = x.to(device)
            linear.to(device)
            y_mps = linear(x_mps)
            self.assertEqual(y, y_mps)

        helper(())
        helper((2, 4))

    def test_linear_errors(self):
        # Mixed CPU<->MPS tensors
        size = (3, 3)

        # Unsupported dtypes
        with self.assertRaisesRegex(RuntimeError, "does not support linear for non-float weights"):
            torch.nn.functional.linear(torch.rand(size, device='mps'),
                                       torch.randint(-10, 10, size, dtype=torch.int8, device='mps'))

        # Weights on wrong device
        with self.assertRaisesRegex(RuntimeError, "argument weight is on cpu but expected on mps"):
            torch.nn.functional.linear(torch.rand(size, device='mps'),
                                       torch.rand(size, device='cpu'))

        # Input on wrong device
        with self.assertRaisesRegex(RuntimeError, "argument input is on cpu but expected on mps"):
            torch.nn.functional.linear(torch.rand(size, device='cpu'),
                                       torch.rand(size, device='mps'))

    def _linear_helper(self, in_features, out_features, shape, bias=True, backward_pass=False):
        cpu_linear = torch.nn.Linear(in_features=in_features, out_features=out_features, device="cpu", bias=bias)
        mps_linear = torch.nn.Linear(in_features=in_features, out_features=out_features, device="mps", bias=bias)

        # Use the same weights and bias as the ones from the cpu
        mps_linear.weight.data = cpu_linear.weight.data.detach().clone().to("mps")

        if bias:
            mps_linear.bias.data = cpu_linear.bias.data.detach().clone().to("mps")

        linear_mps_input = torch.randn(shape).to('mps')
        linear_cpu_input = linear_mps_input.detach().clone().to('cpu')

        if backward_pass:
            linear_mps_input = linear_mps_input.requires_grad_()
            linear_cpu_input = linear_cpu_input.requires_grad_()

        linear_cpu_output = cpu_linear(linear_cpu_input)
        linear_mps_output = mps_linear(linear_mps_input)

        self.assertEqual(linear_cpu_output, linear_mps_output.to('cpu'))
        self.assertEqual(linear_cpu_output.size(), linear_mps_output.size())

        if backward_pass:
            cpu_grad = torch.rand_like(linear_cpu_output, requires_grad=True)
            grad = cpu_grad.detach().to('mps').requires_grad_()

            linear_cpu_output.backward(gradient=cpu_grad, create_graph=True)
            linear_mps_output.backward(gradient=grad, create_graph=True)

            self.assertEqual(linear_cpu_input.grad.size(), linear_mps_input.grad.size())
            self.assertEqual(linear_cpu_input.grad, linear_mps_input.grad.to("cpu"), atol=8e-04, rtol=10.4e-05)

            self.assertEqual(cpu_linear.weight.grad.size(), mps_linear.weight.grad.size())
            self.assertEqual(cpu_linear.weight.grad, mps_linear.weight.grad.to("cpu"), atol=8e-04, rtol=10.4e-05)
            if bias:
                self.assertEqual(cpu_linear.bias.grad.size(), mps_linear.bias.grad.size())
                self.assertEqual(cpu_linear.bias.grad, mps_linear.bias.grad.to("cpu"), atol=8e-04, rtol=10.4e-05)

            # test gradgrad
            x_grad_out = torch.rand_like(linear_cpu_input)
            x_grad_out_mps = x_grad_out.to("mps")
            w_grad_out = torch.rand_like(cpu_linear.weight)
            w_grad_out_mps = w_grad_out.to("mps")

            linear_cpu_input.grad.detach().zero_()
            linear_mps_input.grad.detach().zero_()
            cpu_linear.weight.grad.detach().zero_()
            mps_linear.weight.grad.detach().zero_()
            if bias:
                b_grad_out = torch.rand_like(cpu_linear.bias)
                b_grad_out_mps = b_grad_out.to("mps")
                cpu_linear.bias.grad.detach().zero_()
                mps_linear.bias.grad.detach().zero_()

            linear_cpu_input.grad.backward(x_grad_out, retain_graph=True)
            linear_mps_input.grad.backward(x_grad_out_mps, retain_graph=True)
            cpu_linear.weight.grad.backward(w_grad_out, retain_graph=True)
            mps_linear.weight.grad.backward(w_grad_out_mps, retain_graph=True)
            if bias:
                cpu_linear.bias.grad.backward(b_grad_out, retain_graph=True)
                mps_linear.bias.grad.backward(b_grad_out_mps, retain_graph=True)

            self.assertEqual(cpu_grad.grad, grad.grad)
            self.assertEqual(linear_cpu_input.grad, linear_mps_input.grad)
            self.assertEqual(cpu_linear.weight.grad, mps_linear.weight.grad)
            if bias:
                self.assertEqual(cpu_linear.bias.grad, mps_linear.bias.grad)

    def test_linear1D(self):
        self._linear_helper(in_features=2, out_features=3, shape=([2]), bias=True, backward_pass=False)

    def test_linear1D_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=([2]), bias=True, backward_pass=True)

    def test_linear2D(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 2)), bias=True, backward_pass=False)

    def test_linear2D_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 2)), bias=True, backward_pass=True)

    def test_linear2D_no_bias(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 2)), bias=False, backward_pass=False)

    def test_linear2D_no_bias_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 2)), bias=False, backward_pass=True)

    def test_linear3D(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 5, 2)), bias=True, backward_pass=False)

    def test_linear3D_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 5, 2)), bias=True, backward_pass=True)

    def test_linear3D_no_bias(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 5, 2)), bias=True, backward_pass=False)

    def test_linear3D_no_bias_backward(self):
        self._linear_helper(in_features=2, out_features=3, shape=((4, 5, 2)), bias=True, backward_pass=True)

    @xfailIf(MACOS_VERSION < 14.0)
    def test_linear_large(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/122045
        x_cpu = torch.randn(9, 1024, 1, device='cpu')
        w_cpu = torch.randn(50304, 1, device='cpu')
        x_mps = x_cpu.detach().clone().to('mps')
        w_mps = w_cpu.detach().clone().to('mps')

        out_cpu = F.linear(x_cpu, w_cpu, None)
        out_mps = F.linear(x_mps, w_mps, None)

        self.assertEqual(out_cpu, out_mps)

    def test_uniform(self):
        low = torch.zeros(5, 5, requires_grad=True)
        high = (torch.ones(5, 5) * 3).requires_grad_()
        low_1d = torch.zeros(1, requires_grad=True)
        high_1d = (torch.ones(1) * 3).requires_grad_()
        self.assertEqual(Uniform(low, high).sample().size(), (5, 5))
        self.assertEqual(Uniform(low, high).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Uniform(low_1d, high_1d).sample().size(), (1,))
        self.assertEqual(Uniform(low_1d, high_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Uniform(0.0, 1.0).sample((1,)).size(), (1,))

        # Check log_prob computation when value outside range
        uniform = Uniform(low_1d, high_1d, validate_args=False)
        above_high = torch.tensor([4.0])
        below_low = torch.tensor([-1.0])
        self.assertEqual(uniform.log_prob(above_high).item(), -inf)
        self.assertEqual(uniform.log_prob(below_low).item(), -inf)

        # check cdf computation when value outside range
        self.assertEqual(uniform.cdf(below_low).item(), 0)
        self.assertEqual(uniform.cdf(above_high).item(), 1)

        state = torch.get_rng_state()
        rand = low.new(low.size()).uniform_()
        torch.set_rng_state(state)
        u = Uniform(low, high).rsample()
        u.backward(torch.ones_like(u))
        self.assertEqual(low.grad, 1 - rand)
        self.assertEqual(high.grad, rand)
        low.grad.zero_()
        high.grad.zero_()

    def test_randperm(self, device="mps"):
        rng_device = None
        for n in (5, 100, 50000, 100000):
            for dtype in (torch.long, torch.half, torch.float):
                if n > 2049 and dtype == torch.half:  # Large n for torch.half will raise an exception, do not test here.
                    continue
                if n > 256 and dtype == torch.bfloat16:
                    continue
                with torch.random.fork_rng(devices=rng_device):
                    res1 = torch.randperm(n, dtype=dtype, device=device)
                res2 = torch.empty(0, dtype=dtype, device=device)
                torch.randperm(n, out=res2, dtype=dtype, device=device)
                self.assertEqual(res1.cpu().sort().values.long(), torch.arange(n, device=device))

        # Default type is long
        for n in (100, 10000):
            self.assertEqual(torch.randperm(n, device=device).dtype, torch.long)

        # randperm of 0 elements is an empty tensor
        res1 = torch.randperm(0)
        res2 = torch.tensor(5, dtype=dtype, device=device)
        torch.randperm(0, out=res2)
        self.assertEqual(res1.numel(), 0)
        self.assertEqual(res2.numel(), 0)

        # Test non-contiguous tensors
        for n in (4, 5, 6, 10, 20):
            non_contiguous_tensor = torch.zeros((2, 3), dtype=torch.long, device=device).t()
            self.assertFalse(non_contiguous_tensor.is_contiguous())
            with torch.random.fork_rng(devices=rng_device):
                res = torch.randperm(n, dtype=torch.long, device=device)
            torch.randperm(n, out=non_contiguous_tensor)
            self.assertEqual(res.cpu().sort().values.long(), torch.arange(n, device=device))

    # Test forward maxpool2d
    def test_max_pool2d(self):
        def helper(shape, ks, padding=0, dilation=1, ceil_mode=False, return_indices=False, test_ties=False):

            cpu_x = None
            if (test_ties):
                cpu_x = torch.ones(shape, device='cpu', dtype=torch.float, requires_grad=True)
            else:
                cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            pool = torch.nn.MaxPool2d(kernel_size=ks, padding=padding, dilation=dilation,
                                      ceil_mode=ceil_mode, return_indices=return_indices)

            if (return_indices is False):
                y = pool(x)
                ref_y = pool(cpu_x)

                cpu_grad = torch.ones_like(ref_y)
                grad = cpu_grad.to('mps')

                y.backward(gradient=grad)
                ref_y.backward(gradient=cpu_grad)

                self.assertEqual(y, ref_y)
                self.assertEqual(x.grad, cpu_x.grad)
            else:
                y, idx = pool(x)
                ref_y, ref_idx = pool(cpu_x)

                cpu_grad = torch.ones_like(ref_y)
                grad = cpu_grad.to('mps')

                y.backward(gradient=grad)
                ref_y.backward(gradient=cpu_grad)

                self.assertEqual(y, ref_y)
                self.assertEqual(idx, ref_idx)
                self.assertEqual(x.grad, cpu_x.grad)

        # Test with no batch dimension
        helper((8, 4, 4), ks=2)
        helper((2, 8, 4, 4), ks=2)
        helper((1, 1000, 32, 32), ks=4)
        helper((1, 1000, 1, 4), ks=(1, 4))  # test for max_pool1d
        # Test padding
        helper((1, 1000, 32, 32), ks=4, padding=1)
        helper((1, 1000, 1, 4), ks=(1, 4), padding=(0, 1))  # test for max_pool1d
        # Test dilation
        helper((1, 1000, 32, 32), ks=4, dilation=2)
        helper((1, 1000, 1, 4), ks=(1, 4), padding=(0, 2))  # test for max_pool1d
        # Test ceil mode
        helper((1, 1000, 32, 32), ks=4, ceil_mode=True)
        helper((1, 1000, 1, 4), ks=(1, 4), ceil_mode=True)  # test for max_pool1d

        # Test return indices
        for test_ties in [False, True]:
            # Test with no batch dimension
            helper((8, 4, 4), ks=2, return_indices=True, test_ties=test_ties)
            helper((2, 8, 4, 4), ks=2, return_indices=True, test_ties=test_ties)
            helper((1, 1000, 32, 32), ks=4, return_indices=True, test_ties=test_ties)
            helper((1, 1000, 1, 4), ks=(1, 4), return_indices=True, test_ties=test_ties)  # test for max_pool1d
            # Test padding
            helper((1, 1000, 32, 32), ks=4, padding=1, return_indices=True, test_ties=test_ties)
            helper((1, 1000, 1, 4), ks=(1, 4), padding=(0, 1),
                   return_indices=True, test_ties=test_ties)  # test for max_pool1d
            # Test dilation
            helper((1, 1000, 32, 32), ks=4, dilation=2, return_indices=True, test_ties=test_ties)
            helper((1, 1000, 1, 4), ks=(1, 4), padding=(0, 2),
                   return_indices=True, test_ties=test_ties)  # test for max_pool1d
            # Test ceil mode
            helper((1, 1000, 32, 32), ks=4, ceil_mode=True, return_indices=True, test_ties=test_ties)
            helper((1, 1000, 1, 4), ks=(1, 4), ceil_mode=True,
                   return_indices=True, test_ties=test_ties)  # test for max_pool1d

    def test_adaptive_avg_pool2d_output_size_one(self):
        def helper(size, memory_format):
            x = torch.randint(1, 10, size, dtype=torch.float, device='mps', requires_grad=True)
            if memory_format == 'non_contiguous':
                x = x[::2, ::2, ::2, ::2]
            else:
                x = x.to(memory_format=memory_format)

            net = torch.nn.AdaptiveAvgPool2d((1, 1))
            out = net(x)
            ref_out = x.contiguous().mean((-1, -2)).view((x.size(0), x.size(1), 1, 1))

            out.sum().backward()    # make sure it doesn't crash

            self.assertEqual(out, ref_out)
            if memory_format == torch.channels_last:
                self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
                c = out.size(1)
                self.assertEqual(out.stride(), [c, 1, c, c])
            else:
                self.assertTrue(out.is_contiguous())
                c = out.size(1)
                self.assertEqual(out.stride(), [c, 1, 1, 1])

        helper((2, 3, 6, 6), torch.contiguous_format)

    def test_masked_scatter(self):
        def helper(shape):
            x_mps = torch.randn(shape, device="mps")
            x_cpu = x_mps.detach().clone().cpu()

            mask_mps = torch.rand(shape, device="mps") < 0.6
            mask_cpu = mask_mps.detach().clone().cpu()

            y_mps = torch.randn(shape, device="mps")
            y_cpu = y_mps.detach().clone().cpu()

            y_mps.masked_scatter_(mask_mps, x_mps)
            y_cpu.masked_scatter_(mask_cpu, x_cpu)

            self.assertEqual(y_mps, y_cpu)
        helper([2, 5])
        helper([10, 10])
        helper([5, 10, 3])
        helper([10, 5, 10, 3])
        helper([10, 5, 10, 3, 20])

    def test_masked_fill(self):
        device = "mps"
        dtype = torch.float32
        mask_dtype = torch.bool
        num_dest = 10

        dst = torch.zeros(num_dest, dtype=dtype, device=device)
        mask = torch.randint(2, (num_dest,), dtype=mask_dtype, device=device)
        val = random.random()
        dst2 = torch.zeros(num_dest, dtype=dtype)
        mask_cpu = mask.to("cpu")

        dst.masked_fill_(mask, val)
        for i in range(num_dest):
            if mask_cpu[i]:
                dst2[i] = val
        self.assertEqual(dst.to("cpu"), dst2, atol=0, rtol=0)

        if MACOS_VERSION >= 14.0:
            # Regression test for https://github.com/pytorch/pytorch/issues/143477
            # Allocating 48x25x1024x1024 tensor crashes on MacOS-13
            mask_bool = torch.triu(torch.ones(1024, 1024, device=device), diagonal=1).bool()
            attn_scores = torch.rand(48, 25, 1024, 1024, device=device)
            attn_scores.masked_fill_(mask_bool, 0)

    def test_masked_fill__non_contiguous(self):
        shape = (3, 5)

        x_mps = torch.randn(shape, device="mps")
        x_cpu = x_mps.detach().clone().cpu()
        mask_mps = torch.zeros(shape, device="mps", dtype=torch.bool)
        mask_cpu = mask_mps.detach().clone().cpu()

        x_mps_strided = x_mps.T
        x_cpu_strided = x_cpu.T

        x_mps_strided.masked_fill_(mask_mps.T, float("-inf"))
        x_cpu_strided.masked_fill_(mask_cpu.T, float("-inf"))

        self.assertEqual(x_mps_strided, x_cpu_strided)
        self.assertFalse((x_mps_strided == float("-inf")).any())

    def test_nhwc_operation(self):
        def helper(shape, channels_last=False):
            import numpy as np
            np.random.seed(332)
            arr = (256 - 128) * np.random.random_sample(size=shape) + 128
            cpu_x = torch.tensor(arr, device='cpu', dtype=torch.float, requires_grad=True)
            if (channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            # This passes
            self.assertEqual(x, cpu_x)

        helper((2, 2, 2, 2), True)

    # Test forward batch norm
    def test_batch_norm(self):
        def helper(shape, eps=1, momentum=0.1, wts=False, training=False, channels_last=False,
                   track_running_stats=True, test_module=False):

            import numpy as np
            np.random.seed(332)
            arr = (256 - 128) * np.random.random_sample(size=shape) + 128
            cpu_x = torch.tensor(arr, device='cpu', dtype=torch.float, requires_grad=True)
            if (channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            mean_shape = [shape[1]]
            cpu_running_mean = None
            cpu_running_var = None
            running_mean = None
            running_var = None
            if (track_running_stats):
                mean_arr = (240 - 140) * np.random.random_sample(size=mean_shape) + 140
                cpu_running_mean = torch.tensor(mean_arr, device='cpu', dtype=torch.float)
                var_arr = 32 * np.random.random_sample(size=mean_shape)
                cpu_running_var = torch.tensor(var_arr, device='cpu', dtype=torch.float)
                running_mean = cpu_running_mean.detach().clone().to('mps')
                running_var = cpu_running_var.detach().clone().to('mps')

            weight = None
            cpu_weight = None
            bias = None
            cpu_bias = None
            if (wts):
                cpu_weight = torch.randn(mean_shape, device='cpu', dtype=torch.float, requires_grad=True)
                weight = cpu_weight.detach().clone().to('mps').requires_grad_()
                cpu_bias = torch.randn(mean_shape, device='cpu', dtype=torch.float, requires_grad=True)
                bias = cpu_bias.detach().clone().to('mps').requires_grad_()

            y = None
            ref_y = None

            if (not test_module):
                y = torch.nn.functional.batch_norm(x, running_mean, running_var,
                                                   weight=weight,
                                                   bias=bias,
                                                   training=training,
                                                   momentum=momentum, eps=eps)
                ref_y = torch.nn.functional.batch_norm(cpu_x, cpu_running_mean, cpu_running_var,
                                                       weight=cpu_weight,
                                                       bias=cpu_bias,
                                                       training=training,
                                                       momentum=momentum, eps=eps)

            else:

                batchnorm_op = None
                mps_batchnorm_op = None

                if (len(shape) == 3):
                    batchnorm_op = torch.nn.BatchNorm1d(shape[1],
                                                        eps=eps,
                                                        momentum=momentum,
                                                        affine=wts,
                                                        track_running_stats=track_running_stats,
                                                        device='cpu')
                    mps_batchnorm_op = torch.nn.BatchNorm1d(shape[1],
                                                            eps=eps,
                                                            momentum=momentum,
                                                            affine=wts,
                                                            track_running_stats=track_running_stats,
                                                            device='mps')
                elif (len(shape) == 4):
                    batchnorm_op = torch.nn.BatchNorm2d(shape[1],
                                                        eps=eps,
                                                        momentum=momentum,
                                                        affine=wts,
                                                        track_running_stats=track_running_stats,
                                                        device='cpu')
                    mps_batchnorm_op = torch.nn.BatchNorm2d(shape[1],
                                                            eps=eps,
                                                            momentum=momentum,
                                                            affine=wts,
                                                            track_running_stats=track_running_stats,
                                                            device='mps')
                elif (len(shape) == 5):
                    batchnorm_op = torch.nn.BatchNorm3d(shape[1],
                                                        eps=eps,
                                                        momentum=momentum,
                                                        affine=wts,
                                                        track_running_stats=track_running_stats,
                                                        device='cpu')
                    mps_batchnorm_op = torch.nn.BatchNorm3d(shape[1],
                                                            eps=eps,
                                                            momentum=momentum,
                                                            affine=wts,
                                                            track_running_stats=track_running_stats,
                                                            device='mps')

                if (track_running_stats):
                    batchnorm_op.running_mean = cpu_running_mean
                    batchnorm_op.running_var = cpu_running_var
                    mps_batchnorm_op.running_mean = running_mean
                    mps_batchnorm_op.running_var = running_var
                if (wts):
                    batchnorm_op.weight = torch.nn.Parameter(cpu_weight)
                    batchnorm_op.bias = torch.nn.Parameter(cpu_bias)
                    mps_batchnorm_op.weight = torch.nn.Parameter(weight)
                    mps_batchnorm_op.bias = torch.nn.Parameter(bias)

                ref_y = batchnorm_op(cpu_x)
                y = mps_batchnorm_op(x)

            self.assertEqual(y, ref_y)
            if (not test_module):
                self.assertEqual(running_mean, cpu_running_mean)
                self.assertEqual(running_var, cpu_running_var)
            else:
                self.assertEqual(mps_batchnorm_op.running_mean, batchnorm_op.running_mean)
                self.assertEqual(mps_batchnorm_op.running_var, batchnorm_op.running_var)

            cpu_grad = torch.randn(ref_y.shape)
            grad = cpu_grad.to('mps')
            ref_y.backward(gradient=cpu_grad)
            y.backward(gradient=grad)

            self.assertEqual(x.grad, cpu_x.grad)
            if (wts):
                if (not test_module):
                    self.assertEqual(weight.grad, cpu_weight.grad)
                    self.assertEqual(bias.grad, cpu_bias.grad)
                else:
                    self.assertEqual(mps_batchnorm_op.weight.grad, batchnorm_op.weight.grad)
                    self.assertEqual(mps_batchnorm_op.bias.grad, batchnorm_op.bias.grad)

        for shape in [(2, 3, 2, 2), (2, 3, 2, 2, 2), (2, 3, 2)]:
            for test_module in [False, True]:
                for track_running_stats in [True, False]:
                    for channels_last in [False]:
                        if (channels_last and len(shape) != 4):
                            continue
                        # Running stats must be tracked in eval mode
                        if (track_running_stats):
                            helper(shape, eps=0, momentum=1, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=1e-05, momentum=0.1, wts=False, training=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=0, momentum=1.0, wts=False, training=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=1, momentum=1, wts=True, training=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=3, momentum=0.67, wts=True, training=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=1e-05, momentum=0.1, wts=False, training=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=0, momentum=1.0, wts=False, training=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=1, momentum=1, wts=True, training=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=3, momentum=0.67, wts=True, training=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)

    def test_batch_norm_backward(self):
        inputs = torch.rand(1, 8, 4, 4, device="mps", requires_grad=True)
        x = torch.nn.BatchNorm2d(8).to("mps")
        y = torch.nn.BatchNorm2d(8).to("mps")
        y.weight.requires_grad = False
        y.bias.requires_grad = False
        outputs = y(x(inputs))
        # This used to crash, see https://github.com/pytorch/pytorch/issues/98602
        outputs.sum().backward()

    def test_batch_norm_slices(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/133520
        bn_cpu = nn.BatchNorm2d(100, affine=False, device='cpu')
        bn_mps = nn.BatchNorm2d(100, affine=False, device='mps')

        x_cpu = torch.randn(100, 100, 35, 45).to('cpu')
        x_mps = x_cpu.to('mps')

        res_cpu = bn_cpu(x_cpu[5:])
        res_mps = bn_mps(x_mps[5:])

        self.assertEqual(res_cpu, res_mps)

    def test_batch_norm_backward_weight_bias_gradients(self):
        # See issue: https://github.com/pytorch/pytorch/issues/156555
        N, C, L = 4, 3, 5
        x = torch.randn(N, C, L)
        y = torch.randn(N, C, L)
        bn_cpu = nn.BatchNorm1d(C, affine=True).cpu().train()
        bn_mps = nn.BatchNorm1d(C, affine=True).to('mps').train()
        bn_mps.load_state_dict(bn_cpu.state_dict())

        out_cpu = bn_cpu(x)
        out_mps = bn_mps(x.to('mps'))

        loss_cpu = ((out_cpu - y) ** 2).mean()
        loss_mps = ((out_mps - y.to('mps')) ** 2).mean()
        loss_cpu.backward()
        loss_mps.backward()

        self.assertEqual(bn_cpu.weight.grad, bn_mps.weight.grad, atol=1e-5, rtol=1e-5)
        self.assertEqual(bn_cpu.bias.grad, bn_mps.bias.grad, atol=1e-5, rtol=1e-5)

    def test_layer_norm_backward(self):
        inputs = torch.rand(4, 4, device="mps", requires_grad=True)
        x = torch.nn.LayerNorm(4).to("mps")
        y = torch.nn.LayerNorm(4).to("mps")
        y.weight.requires_grad = False
        y.bias.requires_grad = False
        outputs = y(x(inputs))
        # This used to crash, see https://github.com/pytorch/pytorch/issues/98602
        outputs.sum().backward()

    def test_norm(self):
        a = torch.arange(9, dtype=torch.float, device="mps") - 4
        b = a.reshape((3, 3))

        a_cpu = torch.arange(9, dtype=torch.float, device="cpu") - 4
        b_cpu = a_cpu.reshape((3, 3))

        res = torch.norm(a)
        res_cpu = torch.norm(a_cpu)
        self.assertEqual(res, res_cpu)

        res = torch.norm(b)
        res_cpu = torch.norm(b_cpu)
        self.assertEqual(res, res_cpu)

        res = torch.norm(a, float('inf'))
        res_cpu = torch.norm(a_cpu, float('inf'))
        self.assertEqual(res, res_cpu)

        res = torch.norm(b, float('inf'))
        res_cpu = torch.norm(b_cpu, float('inf'))
        self.assertEqual(res, res_cpu)

        c = torch.tensor([[1, 2, 3], [-1, 1, 4]], dtype=torch.float, device="mps")
        c_cpu = torch.tensor([[1, 2, 3], [-1, 1, 4]] , dtype=torch.float, device="cpu")

        res = torch.norm(c, dim=0)
        res_cpu = torch.norm(c_cpu, dim=0)
        self.assertEqual(res, res_cpu)

        res = torch.norm(c, dim=1)
        res_cpu = torch.norm(c_cpu, dim=1)
        self.assertEqual(res, res_cpu)

        res = torch.norm(c, p=1, dim=1)
        res_cpu = torch.norm(c_cpu, p=1, dim=1)
        self.assertEqual(res, res_cpu)

        d = torch.arange(8, dtype=torch.float, device="mps").reshape(2, 2, 2)
        d_cpu = torch.arange(8, dtype=torch.float, device="cpu").reshape(2, 2, 2)

        res = torch.norm(d, dim=(1, 2))
        res_cpu = torch.norm(d_cpu, dim=(1, 2))
        self.assertEqual(res, res_cpu)

        res = torch.norm(d[0, :, :]), torch.norm(d[1, :, :])
        res_cpu = torch.norm(d_cpu[0, :, :]), torch.norm(d_cpu[1, :, :])
        self.assertEqual(res, res_cpu)

    def test_linalg_vector_norm(self):
        x_mps = torch.tensor([0, 0, 0, 2, 3], dtype=torch.float, device="mps")
        x_cpu = x_mps.detach().clone().cpu()

        res_mps = torch.linalg.vector_norm(x_mps, ord=0)
        res_cpu = torch.linalg.vector_norm(x_cpu, ord=0)
        self.assertEqual(res_mps, res_cpu)

        a_mps = torch.arange(27, dtype=torch.float, device="mps") - 4
        a_cpu = torch.arange(27, dtype=torch.float, device="cpu") - 4

        B_mps = a_mps.reshape(3, 3, 3)
        B_cpu = a_cpu.reshape(3, 3, 3)

        res_mps = torch.linalg.vector_norm(a_mps, ord=3.5)
        res_cpu = torch.linalg.vector_norm(a_cpu, ord=3.5)
        self.assertEqual(res_mps, res_cpu)

        res_mps = torch.linalg.vector_norm(B_mps, ord=3.5)
        res_cpu = torch.linalg.vector_norm(B_cpu, ord=3.5)
        self.assertEqual(res_mps, res_cpu)

        for dim in range(0, B_mps.dim()):
            res_mps = torch.linalg.vector_norm(B_mps, ord=3.5, dim=dim)
            res_cpu = torch.linalg.vector_norm(B_cpu, ord=3.5, dim=dim)
            self.assertEqual(res_mps, res_cpu)

    def test_linalg_lu_factor_ex(self):
        from torch.testing._internal.common_utils import make_fullrank_matrices_with_distinct_singular_values

        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_arg = partial(make_fullrank, device="cpu", dtype=torch.float32)

        def run_lu_factor_ex_test(size, *batch_dims, check_errors, atol=1e-5, rtol=1e-6):
            input_cpu = make_arg(*batch_dims, size, size)
            input_mps = input_cpu.to('mps')
            out_cpu = torch.linalg.lu_factor_ex(input_cpu, check_errors=check_errors)
            out_mps = torch.linalg.lu_factor_ex(input_mps, check_errors=check_errors)
            self.assertEqual(out_cpu, out_mps, atol=atol, rtol=rtol)

            out_cpu = torch.linalg.lu_factor_ex(input_cpu.mT, check_errors=check_errors)
            out_mps = torch.linalg.lu_factor_ex(input_mps.mT, check_errors=check_errors)
            self.assertEqual(out_cpu, out_mps, atol=atol, rtol=rtol)

        # test with different even/odd matrix sizes
        matrix_sizes = [1, 2, 3, 4]
        # even/odd batch sizes
        batch_sizes = [1, 2, 4]

        for check_errors in [True, False]:
            for size in matrix_sizes:
                for batch_size in batch_sizes:
                    run_lu_factor_ex_test(size, batch_size, check_errors=check_errors)
        # test >3D matrices
        run_lu_factor_ex_test(32, 10, 10, check_errors=False)
        run_lu_factor_ex_test(32, 2, 2, 10, 10, check_errors=True)
        # big matrix check with batch size > 1
        run_lu_factor_ex_test(256, 2, check_errors=False, atol=3e-5, rtol=5e-6)

    def test_linalg_solve(self):
        from torch.testing._internal.common_utils import make_fullrank_matrices_with_distinct_singular_values

        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_arg = partial(make_fullrank, device="cpu", dtype=torch.float32)

        def run_linalg_solve_test(size, *batch_dims):
            A_cpu = make_arg(*batch_dims, size, size)
            A_mps = A_cpu.to('mps')

            for left in [True, False]:
                if left:
                    b_cpu = torch.randn(*batch_dims, size, 3, device='cpu', dtype=torch.float32)
                else:
                    b_cpu = torch.randn(*batch_dims, 3, size, device='cpu', dtype=torch.float32)

                b_mps = b_cpu.to('mps')

                # Solve the system
                X_cpu = torch.linalg.solve(A_cpu, b_cpu, left=left)
                X_mps = torch.linalg.solve(A_mps, b_mps, left=left)
                self.assertEqual(X_cpu, X_mps)

                # Test with transposed matrices
                X_cpu_t = torch.linalg.solve(A_cpu.mT, b_cpu, left=left)
                X_mps_t = torch.linalg.solve(A_mps.mT, b_mps, left=left)
                self.assertEqual(X_cpu_t, X_mps_t)

        # test with different even/odd matrix sizes
        matrix_sizes = [1, 2, 3, 4]
        # even/odd batch sizes
        batch_sizes = [1, 2, 4]

        for size in matrix_sizes:
            for batch_size in batch_sizes:
                run_linalg_solve_test(size, batch_size)

        # test >3D matrices
        run_linalg_solve_test(32, 10, 10)
        run_linalg_solve_test(32, 2, 2, 2, 2, 10, 10)

    def test_linalg_solve_with_broadcasting(self):
        from functools import partial
        import torch
        from torch.testing._internal.common_utils import (
            make_fullrank_matrices_with_distinct_singular_values
        )

        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_arg = partial(make_fullrank, device="cpu", dtype=torch.float32)

        batch_size = 4
        size = 3

        A_cpu = make_arg(batch_size, size, size)
        A_mps = A_cpu.to('mps')

        for left in [True, False]:
            b_cpu = torch.randn(batch_size, size, device='cpu', dtype=torch.float32)
            b_mps = b_cpu.to('mps')

            if left:
                b_cpu = b_cpu.unsqueeze(-1)
                b_mps = b_mps.unsqueeze(-1)
            else:
                b_cpu = b_cpu.view(batch_size, 1, size)
                b_mps = b_mps.view(batch_size, 1, size)

            X_cpu = torch.linalg.solve(A_cpu, b_cpu, left=left)
            X_mps = torch.linalg.solve(A_mps, b_mps, left=left)
            self.assertEqual(X_cpu, X_mps)

            X_cpu_t = torch.linalg.solve(A_cpu.mT, b_cpu, left=left)
            X_mps_t = torch.linalg.solve(A_mps.mT, b_mps, left=left)
            self.assertEqual(X_cpu_t, X_mps_t)

    def test_linalg_det(self):
        from torch.testing._internal.common_utils import make_fullrank_matrices_with_distinct_singular_values

        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_arg = partial(make_fullrank, device="cpu", dtype=torch.float32)

        def run_det_test(size, *batch_dims):
            input_cpu = make_arg(*batch_dims, size, size)
            input_mps = input_cpu.to('mps')
            out_cpu = torch.linalg.det(input_cpu)
            out_mps = torch.linalg.det(input_mps)
            self.assertEqual(out_cpu, out_mps)

            # non-contiguous matrices
            input_cpu_T = input_cpu.mT
            input_mps_T = input_mps.mT
            out_cpu_T = torch.linalg.det(input_cpu_T)
            out_mps_T = torch.linalg.det(input_mps_T)
            self.assertEqual(out_cpu_T, out_mps_T)

        # test with different even/odd matrix sizes
        matrix_sizes = [2, 3, 4]
        # even/odd batch sizes
        batch_sizes = [1, 2, 4]

        for size in matrix_sizes:
            for batch_size in batch_sizes:
                run_det_test(size, batch_size)

        # test >3D matrices
        run_det_test(32, 10, 10)
        run_det_test(32, 2, 2, 10, 10)

    def test_layer_norm(self):
        def helper(input_shape, normalized_shape, eps=1e-05, elementwise_affine=True, dtype=torch.float32, non_contiguous=False):
            cpu_x = torch.randn(input_shape, device='cpu', dtype=dtype, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()
            if non_contiguous:
                x = x.mT
                cpu_x = cpu_x.mT
                normalized_shape[-1], normalized_shape[-2] = normalized_shape[-2], normalized_shape[-1]

            cpu_op = torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device='cpu', dtype=dtype)
            mps_op = torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device='mps', dtype=dtype)
            cpu_wt = torch.randn(normalized_shape, device='cpu', dtype=dtype, requires_grad=True)
            wt = cpu_wt.detach().clone().to('mps').requires_grad_()
            cpu_bias = torch.randn(normalized_shape, device='cpu', dtype=dtype, requires_grad=True)
            bias = cpu_bias.detach().clone().to('mps').requires_grad_()

            if (elementwise_affine):
                cpu_op.weight = torch.nn.Parameter(cpu_wt)
                mps_op.weight = torch.nn.Parameter(wt)
                cpu_op.bias = torch.nn.Parameter(cpu_bias)
                mps_op.bias = torch.nn.Parameter(bias)

            cpu_result = cpu_op(cpu_x)
            result = mps_op(x)

            cpu_grad = torch.randn(cpu_result.shape)
            grad = cpu_grad.to('mps')

            cpu_result.backward(cpu_grad)
            result.backward(grad)

            self.assertEqual(result, cpu_result)
            self.assertEqual(x.grad, cpu_x.grad)
            if (elementwise_affine):
                self.assertEqual(mps_op.weight.grad, cpu_op.weight.grad)
                self.assertEqual(mps_op.bias.grad, cpu_op.bias.grad)

        for (elementwise_affine, non_contiguous) in itertools.product([True, False], [True, False]):
            helper((2, 2, 2, 2), [2, 2], elementwise_affine=elementwise_affine, non_contiguous=non_contiguous)
            helper((2, 3, 4, 5), [4, 5], elementwise_affine=elementwise_affine, non_contiguous=non_contiguous)
            helper((2, 3, 4, 5, 6), [4, 5, 6], elementwise_affine=elementwise_affine, non_contiguous=non_contiguous)

        # Regression test for https://github.com/pytorch/pytorch/issues/96113
        torch.nn.LayerNorm((16,), elementwise_affine=True).to("mps")(torch.randn(1, 2, 16).to("mps", dtype=torch.float16))

    @xfailIf(MACOS_VERSION < 14.0)
    def test_ifft(self):
        # See: https://github.com/pytorch/pytorch/issues/124096
        device = torch.device("mps")

        N = 64
        signal = torch.rand(N, device=device)
        fft_result = torch.fft.rfft(signal)
        ifft_result = torch.fft.irfft(fft_result, n=signal.shape[0])

        # Expecting the inverted to yield the original signal
        self.assertEqual(ifft_result, signal)

    def test_fftfreq(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/135223
        freq_cpu = torch.fft.fftfreq(10**4, device='cpu')
        freq_mps = torch.fft.fftfreq(10**4, device='mps')
        self.assertEqual(freq_cpu, freq_mps)

    def test_instance_norm(self):
        def helper(shape, eps=1, momentum=0.1, wts=False, channels_last=False, track_running_stats=True, test_module=False):

            import numpy as np
            np.random.seed(332)
            arr = (256 - 128) * np.random.random_sample(size=shape) + 128
            cpu_x = torch.tensor(arr, device='cpu', dtype=torch.float, requires_grad=True)
            if (channels_last):
                cpu_x = cpu_x.to(memory_format=torch.channels_last)
                cpu_x.retain_grad()
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            mean_shape = [shape[1]]
            cpu_running_mean = None
            cpu_running_var = None
            running_mean = None
            running_var = None
            if (track_running_stats):
                mean_arr = (240 - 140) * np.random.random_sample(size=mean_shape) + 140
                cpu_running_mean = torch.tensor(mean_arr, device='cpu', dtype=torch.float)
                var_arr = 32 * np.random.random_sample(size=mean_shape)
                cpu_running_var = torch.tensor(var_arr, device='cpu', dtype=torch.float)
                running_mean = cpu_running_mean.detach().clone().to('mps')
                running_var = cpu_running_var.detach().clone().to('mps')

            weight = None
            cpu_weight = None
            bias = None
            cpu_bias = None
            if (wts):
                cpu_weight = torch.randn(mean_shape, device='cpu', dtype=torch.float, requires_grad=True)
                weight = cpu_weight.detach().clone().to('mps').requires_grad_()
                cpu_bias = torch.randn(mean_shape, device='cpu', dtype=torch.float, requires_grad=True)
                bias = cpu_bias.detach().clone().to('mps').requires_grad_()

            y = None
            ref_y = None

            if (not test_module):
                ref_y = torch.nn.functional.instance_norm(cpu_x, cpu_running_mean, cpu_running_var,
                                                          weight=cpu_weight,
                                                          bias=cpu_bias,
                                                          momentum=momentum, eps=eps)
                y = torch.nn.functional.instance_norm(x, running_mean, running_var,
                                                      weight=weight,
                                                      bias=bias,
                                                      momentum=momentum, eps=eps)

            else:

                instancenorm_op = None
                mps_instancenorm_op = None

                if (len(shape) == 3):
                    instancenorm_op = torch.nn.InstanceNorm1d(shape[1],
                                                              eps=eps,
                                                              momentum=momentum,
                                                              affine=wts,
                                                              track_running_stats=track_running_stats,
                                                              device='cpu')
                    mps_instancenorm_op = torch.nn.InstanceNorm1d(shape[1],
                                                                  eps=eps,
                                                                  momentum=momentum,
                                                                  affine=wts,
                                                                  track_running_stats=track_running_stats,
                                                                  device='mps')
                elif (len(shape) == 4):
                    instancenorm_op = torch.nn.InstanceNorm2d(shape[1],
                                                              eps=eps,
                                                              momentum=momentum,
                                                              affine=wts,
                                                              track_running_stats=track_running_stats,
                                                              device='cpu')
                    mps_instancenorm_op = torch.nn.InstanceNorm2d(shape[1],
                                                                  eps=eps,
                                                                  momentum=momentum,
                                                                  affine=wts,
                                                                  track_running_stats=track_running_stats,
                                                                  device='mps')
                elif (len(shape) == 5):
                    instancenorm_op = torch.nn.InstanceNorm3d(shape[1],
                                                              eps=eps,
                                                              momentum=momentum,
                                                              affine=wts,
                                                              track_running_stats=track_running_stats,
                                                              device='cpu')
                    mps_instancenorm_op = torch.nn.InstanceNorm3d(shape[1],
                                                                  eps=eps,
                                                                  momentum=momentum,
                                                                  affine=wts,
                                                                  track_running_stats=track_running_stats,
                                                                  device='mps')

                if (track_running_stats):
                    instancenorm_op.running_mean = cpu_running_mean
                    instancenorm_op.running_var = cpu_running_var
                    mps_instancenorm_op.running_mean = running_mean
                    mps_instancenorm_op.running_var = running_var
                if (wts):
                    instancenorm_op.weight = torch.nn.Parameter(cpu_weight)
                    instancenorm_op.bias = torch.nn.Parameter(cpu_bias)
                    mps_instancenorm_op.weight = torch.nn.Parameter(weight)
                    mps_instancenorm_op.bias = torch.nn.Parameter(bias)

                ref_y = instancenorm_op(cpu_x)
                y = mps_instancenorm_op(x)

            self.assertEqual(y, ref_y)
            if (not test_module):
                self.assertEqual(running_mean, cpu_running_mean)
                self.assertEqual(running_var, cpu_running_var)
            else:
                self.assertEqual(mps_instancenorm_op.running_mean, instancenorm_op.running_mean)
                self.assertEqual(mps_instancenorm_op.running_var, instancenorm_op.running_var)

            cpu_grad = torch.randn(ref_y.shape)
            grad = cpu_grad.to('mps')
            ref_y.backward(gradient=cpu_grad)
            y.backward(gradient=grad)

            self.assertEqual(x.grad, cpu_x.grad)
            if (wts):
                if (not test_module):
                    self.assertEqual(weight.grad, cpu_weight.grad)
                    self.assertEqual(bias.grad, cpu_bias.grad)
                else:
                    self.assertEqual(mps_instancenorm_op.weight.grad, instancenorm_op.weight.grad)
                    self.assertEqual(mps_instancenorm_op.bias.grad, instancenorm_op.bias.grad)

        for shape in [(2, 3, 2, 2), (2, 3, 2, 2, 2), (2, 3, 2)]:
            for test_module in [False, True]:
                for track_running_stats in [True, False]:
                    for channels_last in [False]:
                        if (channels_last and len(shape) != 4):
                            continue
                        # Running stats must be tracked in eval mode
                        if (track_running_stats):
                            helper(shape, eps=0, momentum=1, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=1e-05, momentum=0.1, wts=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=0, momentum=1.0, wts=False, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=1, momentum=1, wts=True, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                            helper(shape, eps=3, momentum=0.67, wts=True, channels_last=channels_last,
                                   track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=1e-05, momentum=0.1, wts=False, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=0, momentum=1.0, wts=False, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=1, momentum=1, wts=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)
                        helper(shape, eps=3, momentum=0.67, wts=True, channels_last=channels_last,
                               track_running_stats=track_running_stats, test_module=test_module)

    def test_weight_norm(self):
        def validate_weight_norm_equality(model, cpu_model, x, cpu_x, dim):
            cpu_norm = torch.nn.utils.parametrizations.weight_norm(cpu_model, dim=dim)
            norm = torch.nn.utils.parametrizations.weight_norm(model, dim=dim)

            cpu_out = cpu_norm(cpu_x)
            out = norm(x)

            self.assertEqual(cpu_out, out)

            cpu_grad = torch.randn(cpu_out.shape)
            grad = cpu_grad.to('mps')
            cpu_out.backward(gradient=cpu_grad)
            out.backward(gradient=grad)

            self.assertEqual(cpu_model.parametrizations.weight.original0.grad, model.parametrizations.weight.original0.grad)
            self.assertEqual(cpu_model.parametrizations.weight.original1.grad, model.parametrizations.weight.original1.grad)

            self.assertEqual(x.grad, cpu_x.grad)

        def helper(dim, layer='linear', dtype=torch.float32):
            # linear layer
            if layer == 'linear':
                cpu_x = torch.randn((2, 5), device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()

                cpu_weight = torch.randn(10, 5, device='cpu', dtype=dtype, requires_grad=True)
                weight = cpu_weight.detach().clone().to('mps').requires_grad_()

                cpu_bias = torch.randn(10, device='cpu', dtype=dtype, requires_grad=True)
                bias = cpu_bias.detach().clone().to('mps').requires_grad_()

                cpu_linear = torch.nn.Linear(5, 10, device='cpu')
                linear = torch.nn.Linear(5, 10, device='mps')

                with torch.no_grad():
                    cpu_linear.weight.copy_(cpu_weight)
                    cpu_linear.bias.copy_(cpu_bias)
                    linear.weight.copy_(weight)
                    linear.bias.copy_(bias)
                validate_weight_norm_equality(linear, cpu_linear, x, cpu_x, dim)

            # conv layer
            if layer == 'conv':
                cpu_x = torch.randn((3, 5, 5), device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()

                cpu_conv = torch.nn.Conv2d(3, 3, 3, device='cpu')
                conv = torch.nn.Conv2d(3, 3, 3, device='mps')

                with torch.no_grad():
                    conv.weight.copy_(cpu_conv.weight)
                    conv.bias.copy_(cpu_conv.bias)

                validate_weight_norm_equality(conv, cpu_conv, x, cpu_x, dim)

            # conv3d layer
            if layer == 'conv3d':
                cpu_x = torch.randn((3, 5, 5, 4), device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()

                cpu_conv = torch.nn.Conv3d(3, 3, 3, device='cpu')
                conv = torch.nn.Conv3d(3, 3, 3, device='mps')

                with torch.no_grad():
                    conv.weight.copy_(cpu_conv.weight)
                    conv.bias.copy_(cpu_conv.bias)

                validate_weight_norm_equality(conv, cpu_conv, x, cpu_x, dim)

        helper(0, layer='linear')
        helper(1, layer='linear')
        helper(-1, layer='linear')

        helper(0, layer='conv')
        helper(1, layer='conv')
        helper(2, layer='conv')
        helper(3, layer='conv')
        helper(-1, layer='conv')

        if MACOS_VERSION >= 13.2:
            # Conv3d is only available from MacOS 13 onwards
            helper(0, layer='conv3d')
            helper(1, layer='conv3d')
            helper(2, layer='conv3d')
            helper(3, layer='conv3d')
            helper(4, layer='conv3d')
            helper(-1, layer='conv3d')

    # Test conv2d
    def test_conv2d_unit(self):
        def helper(input_shape, wt_shape,
                   stride=1, padding=0,
                   dilation=1, groups=1,
                   bias_shape=None):

            cpu_x = torch.randn(input_shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            cpu_wt = torch.randn(wt_shape, device='cpu', dtype=torch.float, requires_grad=True)
            wt = cpu_wt.detach().clone().to('mps').requires_grad_()

            cpu_bias = None
            bias = None

            if (bias_shape is not None):
                cpu_bias = torch.randn(bias_shape, device='cpu', dtype=torch.float, requires_grad=True)
                bias = cpu_bias.detach().clone().to('mps').requires_grad_()

            y = torch.nn.functional.conv2d(x, wt, bias=bias, stride=stride,
                                           padding=padding, dilation=dilation, groups=groups)
            ref_y = torch.nn.functional.conv2d(cpu_x, cpu_wt, bias=cpu_bias, stride=stride,
                                               padding=padding, dilation=dilation, groups=groups)

            cpu_grad = torch.ones_like(ref_y)
            grad = cpu_grad.to('mps')

            y.backward(gradient=grad)
            ref_y.backward(gradient=cpu_grad)

            self.assertEqual(y, ref_y, rtol=2.6e-05, atol=2e-04)
            self.assertEqual(x.grad, cpu_x.grad, rtol=2.6e-06, atol=2e-05)
            self.assertEqual(wt.grad, cpu_wt.grad, atol=8e-04, rtol=10.4e-05)
            if (bias_shape is not None):
                self.assertEqual(bias.grad, cpu_bias.grad, atol=8e-04, rtol=10.4e-05)

        N = 1
        C_in = 3
        C_out = 64
        H = 64
        W = 64
        kH = 4
        kW = 4
        stride = 2
        padding = 1

        helper((N, C_in, H, W), (C_out, C_in, kH, kW), stride=stride, padding=padding)

        N = 4
        C_in = 16
        H = 32
        W = 32

        C_out = 8
        kH = 3
        kW = 3

        for groups in [1, 2, 4]:
            helper((N, C_in, H, W), (C_out, C_in // groups, kH, kW), groups=groups)
            helper((N, C_in, H, W), (C_out, C_in // groups, kH, kW), groups=groups)

            helper((N, C_in, H, W), (C_out, C_in // groups, kH, kW), bias_shape=(C_out), groups=groups)
            helper((N, C_in, H, W), (C_out, C_in // groups, kH, kW), bias_shape=(C_out), groups=groups)

            helper((N, C_in * 2, H * 2, W * 2), (C_out * 2, (C_in * 2) // groups, kH + 2, kW + 2), groups=groups)
            helper((N, C_in * 2, H * 2, W * 2), (C_out * 2, (C_in * 2) // groups, kH + 2, kW + 2), groups=groups)

            helper((N, C_in * 2, H * 2, W * 2), (C_out * 2, (C_in * 2) // groups,
                   kH + 2, kW + 2), bias_shape=(C_out * 2), groups=groups)
            helper((N, C_in * 2, H * 2, W * 2), (C_out * 2, (C_in * 2) // groups,
                   kH + 2, kW + 2), bias_shape=(C_out * 2), groups=groups)

    # Test conv transpose 2d
    def test_conv_transpose2d(self):
        def helper(input_shape, wt_shape,
                   stride=1, padding=0,
                   output_padding=0,
                   dilation=1, groups=1,
                   bias_shape=None):

            cpu_x = torch.randn(input_shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            cpu_wt = torch.randn(wt_shape, device='cpu', dtype=torch.float, requires_grad=True)
            wt = cpu_wt.detach().clone().to('mps').requires_grad_()

            cpu_bias = None
            bias = None

            if (bias_shape is not None):
                cpu_bias = torch.randn(bias_shape, device='cpu', dtype=torch.float, requires_grad=True)
                bias = cpu_bias.detach().clone().to('mps').requires_grad_()

            y = torch.nn.functional.conv_transpose2d(
                x, wt, bias=bias, stride=stride, padding=padding, output_padding=output_padding, groups=groups, dilation=dilation)
            ref_y = torch.nn.functional.conv_transpose2d(
                cpu_x, cpu_wt, bias=cpu_bias, stride=stride, padding=padding,
                output_padding=output_padding, groups=groups, dilation=dilation)

            cpu_grad = torch.randn(ref_y.shape)
            grad = cpu_grad.to('mps')

            y.backward(gradient=grad)
            ref_y.backward(gradient=cpu_grad)

            self.assertEqual(y, ref_y, rtol=2.6e-05, atol=2e-04)
            self.assertEqual(x.grad, cpu_x.grad, rtol=2.6e-06, atol=2e-05)
            self.assertEqual(wt.grad, cpu_wt.grad, atol=8e-04, rtol=10.4e-05)

            # if (bias_shape is not None):
            #  print(cpu_bias.grad)
            #  print(bias.grad.to('cpu'))
            #  self.assertEqual(bias.grad, cpu_bias.grad)

        N = 4
        C_in = 2
        H = 32
        W = 32

        C_out = 8
        groups = 1
        kH = 3
        kW = 3

        for stride in [1, 2, 3]:
            for padding in [0, 1, 2]:
                for output_padding in [0, 1, 2]:
                    for dilation in [1, 2]:
                        if (output_padding >= stride or output_padding >= dilation):
                            continue
                        helper((N, C_out, H, W), (C_out, C_in, kH, kW), stride=stride,
                               padding=padding, output_padding=output_padding, dilation=dilation)
                        helper((N, C_out, H, W), (C_out, C_in, kH, kW), stride=stride,
                               padding=padding, output_padding=output_padding, dilation=dilation)

                        helper((N, C_out, H, W), (C_out, C_in, kH, kW), bias_shape=(C_in), stride=stride,
                               padding=padding, output_padding=output_padding, dilation=dilation)
                        helper((N, C_out, H, W), (C_out, C_in, kH, kW), bias_shape=(C_in), stride=stride,
                               padding=padding, output_padding=output_padding, dilation=dilation)

    # Test sigmoid
    def test_sigmoid(self):
        def helper(shape):

            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            sigmoid_op = torch.nn.Sigmoid()

            y = sigmoid_op(x)
            ref_y = sigmoid_op(cpu_x)

            cpu_grad = torch.ones_like(ref_y)
            grad = cpu_grad.to('mps')

            y.backward(gradient=grad)
            ref_y.backward(gradient=cpu_grad)

            self.assertEqual(y, ref_y)
            self.assertEqual(x.grad, cpu_x.grad)

        helper((2, 3, 4, 5))
        helper((2, 3, 4))
        helper((2, 8, 4, 5))

    # Test tanh
    def test_tanh(self):
        def helper(shape):

            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            tanh_op = torch.nn.Tanh()

            y = tanh_op(x)
            ref_y = tanh_op(cpu_x)

            cpu_grad = torch.ones_like(ref_y)
            grad = cpu_grad.to('mps')

            y.backward(gradient=grad)
            ref_y.backward(gradient=cpu_grad)

            self.assertEqual(y, ref_y)
            self.assertEqual(x.grad, cpu_x.grad)

        helper((2, 3, 4, 5))
        helper((2, 3, 4))
        helper((2, 8, 4, 5))

    def test_threshold(self):
        def helper(threshold, value, num_elems, inplace=False, requires_grad=True):
            m = nn.Threshold(threshold=threshold, value=value, inplace=inplace)

            input_cpu = torch.randn(num_elems, requires_grad=requires_grad, dtype=torch.float)
            input_mps = input_cpu.detach().clone().to('mps').requires_grad_(requires_grad)

            output_cpu = m(input_cpu)
            output_mps = m(input_mps)

            cpu_grad = torch.ones_like(output_cpu)
            mps_grad = cpu_grad.to('mps')

            self.assertEqual(output_cpu, output_mps)

            if requires_grad:
                output_cpu.backward(gradient=cpu_grad)
                output_mps.backward(gradient=mps_grad)

                self.assertEqual(input_cpu.grad, input_mps.grad)

        helper(threshold=0.1, value=20, num_elems=2)
        helper(threshold=-0.1, value=10, num_elems=10)
        helper(threshold=0.5, value=-15, num_elems=100)
        helper(threshold=1, value=10, num_elems=100, inplace=True, requires_grad=False)

    # Test pow
    def test_pow(self):
        def helper(shape):
            # aten::pow.Tensor_Tensor
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')
            z = torch.pow(x, y)
            ref_z = torch.pow(cpu_x, cpu_y)

            self.assertEqual(z, ref_z)

            # aten::pow.Tensor_Scalar
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')
            exp = random.random()
            z = torch.pow(x, exp)
            ref_z = torch.pow(cpu_x, exp)

            self.assertEqual(z, ref_z)

            # aten::pow.Scalar
            x = random.random()
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')
            z = torch.pow(x, y)
            ref_z = torch.pow(x, cpu_y)

            self.assertEqual(z, ref_z)

        helper((2, 8, 4, 5))

    # Test addcmul
    def test_addcmul(self):
        def helper(shape, value, xtype=torch.float32, ytype=None, ztype=None):
            def rand_helper(dtype):
                if dtype.is_floating_point:
                    return torch.randn(shape, device='cpu', dtype=dtype, requires_grad=False)
                return torch.randint(10, shape, dtype=dtype, device='cpu', requires_grad=False)

            cpu_x = rand_helper(xtype)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = rand_helper(ytype if ytype is not None else xtype)
            y = cpu_y.detach().clone().to('mps')

            cpu_z = rand_helper(ztype if ztype is not None else xtype)
            z = cpu_z.detach().clone().to('mps')

            y = torch.addcmul(x, y, z, value=value)
            ref_y = torch.addcmul(cpu_x, cpu_y, cpu_z, value=value)

            self.assertEqual(y, ref_y)

        helper((2, 3, 4, 5), 0.1)
        helper((2, 8, 4, 5), 0.1)
        helper((2, 3, 4, 5), 0.2)
        helper((2, 8, 4, 5), 0.2)
        # Integral types
        helper((2, 2), 1.0, xtype=torch.int32)
        helper((2, 2), 2.0, xtype=torch.int16)

        # Mixed types
        helper((2, 2), 1.0, xtype=torch.float16, ytype=torch.float32)
        helper((3, 2), 1.0, ytype=torch.float16)
        helper((2, 3), 1.0, ztype=torch.float16)
        helper((2, 2), 1.0, xtype=torch.int32, ytype=torch.int16, ztype=torch.uint8)
        helper((2, 2), 1.0, ytype=torch.int16, ztype=torch.uint8)

    # Test addcdiv
    def test_addcdiv(self):
        def helper(shape, value):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            # clamp to avoid division by 0
            cpu_z = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False).clamp_min_(0.1)
            cpu_out = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)

            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            mps_z = cpu_z.detach().clone().to('mps')
            mps_out = cpu_out.detach().clone().to('mps')

            result_div_mps = torch.addcdiv(mps_x, mps_y, mps_z, value=value)
            result_div_cpu = torch.addcdiv(cpu_x, cpu_y, cpu_z, value=value)
            self.assertEqual(result_div_mps, result_div_cpu)
            # test .out variant
            self.assertEqual(torch.addcdiv(mps_x, mps_y, mps_z, out=mps_out, value=value), result_div_cpu)

        helper((2, 3, 4, 5), 0.1)
        helper((2, 8, 4, 5), 0.2)
        helper((2, 3, 4, 5), 1.0)  # value of 1 should be ignored internally

    def test_addcdiv_transpose(self):
        # Regression test for issue https://github.com/pytorch/pytorch/issues/118115
        # Testing continuity of all input tensors

        def helper(shape, value):
            shape_t = shape[::-1]
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        x = torch.rand(shape, device="cpu") if i == 0 else torch.rand(shape_t, device="cpu").t()
                        y = torch.rand(shape, device="cpu") if j == 0 else torch.rand(shape_t, device="cpu").t()
                        z = torch.rand(shape, device="cpu") if k == 0 else torch.rand(shape_t, device="cpu").t()

                        x_mps = x.detach().clone().to(device="mps")
                        y_mps = y.detach().clone().to(device="mps")
                        z_mps = z.detach().clone().to(device="mps")

                        result_cpu = x.addcdiv_(y, z, value=value)
                        result_mps = x_mps.addcdiv(y_mps, z_mps, value=value)
                        result_mps_out = result_cpu.detach().clone().to('mps')
                        torch.addcdiv(x_mps, y_mps, z_mps, out=result_mps_out, value=value)

                        self.assertEqual(result_cpu, result_mps)
                        self.assertEqual(result_cpu, result_mps_out)

        helper((2, 3), 1.0)
        helper((2, 3), 0.2)
        helper((100, 300), 1.0)
        helper((100, 300), 0.2)

    def test_buffer_size_match(self):
        # this test shouldn't cause any crash
        size = 16
        cpu_A = torch.rand(size, device='cpu')
        cpu_F = torch.rand(size, size, size, device='cpu')

        mps_A = cpu_A.to('mps')
        mps_F = cpu_F.to('mps')
        self.assertEqual(cpu_A @ cpu_F, mps_A @ mps_F)

    def test_transpose_inplace(self):
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        cpu_x = torch.tensor(values, device='cpu')
        mps_x = torch.tensor(values, device='mps')

        cpu_x.transpose_(0, 1)
        mps_x.transpose_(0, 1)
        self.assertEqual(cpu_x, mps_x.to('cpu'))

    def test_expand_cpu_to_mps_copy(self):
        # https://github.com/pytorch/pytorch/issues/78642

        x = torch.tensor(1).expand([10]).to("mps")
        x_cpu = torch.tensor(1).expand([10])

        self.assertEqual(x_cpu, x.cpu())

    def test_cpu_to_strided_mps_copy(self):
        # https://github.com/pytorch/pytorch/issues/86975

        a1 = torch.Tensor([[1, 2], [3, 4], [5, 6]]).to(torch.device("mps"))
        b1 = torch.Tensor([-1, -1])
        a1[1:, 1] = b1

        a2 = torch.Tensor([[1, 2], [3, 4], [5, 6]]).to(torch.device("mps"))
        b2 = torch.Tensor([-1, -1]).to(torch.device("mps"))
        a2[1:, 1] = b2

        self.assertEqual(a1, a2)

    def test_view_slice_reshape(self):
        x = torch.randn([1, 4, 4], device="mps")
        y = x[0, :1, 1:]

        x_cpu = x.to("cpu")
        y_cpu = x_cpu[0, :1, 1:]

        r = y + 1
        r_cpu = y_cpu + 1
        self.assertEqual(r, r_cpu)

    def test_slice_reshape(self):
        x = torch.randn([1, 6, 4, 2], dtype=torch.float, device="mps")
        x_cpu = x.detach().clone().to("cpu")

        x = x[:, 3:].view(2, 3, 4, 1)
        x_cpu = x_cpu[:, 3:].view(2, 3, 4, 1)
        self.assertEqual(x, x_cpu)

        x = x + 2
        x_cpu = x_cpu + 2
        self.assertEqual(x, x_cpu)

        # Regression test for https://github.com/pytorch/pytorch/issues/143140
        def slice_and_reshape(t):
            return t[:, :, :, :3, :3].reshape(18, 1, 3)

        x = torch.rand(1, 1, 1, 4, 5, 6, dtype=torch.cfloat, device="mps")
        x_cpu = x.detach().clone().cpu()
        self.assertEqual(slice_and_reshape(x_cpu), slice_and_reshape(x).cpu())

    def test_reshape_storage_offset(self):
        # https://github.com/pytorch/pytorch/issues/95883
        B = 4
        T = 1

        lin_cpu = nn.Linear(10, 256)
        lin_mps = nn.Linear(10, 256, device="mps")

        # Use the same weights and bias as the ones from the cpu
        lin_mps.weight.data = lin_cpu.weight.data.detach().clone().to("mps").requires_grad_()
        lin_mps.bias.data = lin_cpu.bias.data.detach().clone().to("mps").requires_grad_()

        x_mps = torch.rand([B, T, 10], device="mps", requires_grad=True)
        x_cpu = x_mps.detach().clone().cpu().requires_grad_()
        x_mps = lin_mps(x_mps)
        x_cpu = lin_cpu(x_cpu)

        self.assertEqual(x_mps.shape, (B, T, 256))
        self.assertEqual(x_cpu.shape, (B, T, 256))

        cls_token_mps = torch.rand([1, 256], device="mps", requires_grad=True).repeat(B, 1, 1)
        cls_token_cpu = cls_token_mps.detach().clone().cpu()
        x_mps = torch.cat([cls_token_mps, x_mps], dim=1)
        x_cpu = torch.cat([cls_token_cpu, x_cpu], dim=1)

        x_mps = x_mps.transpose(0, 1)
        x_cpu = x_cpu.transpose(0, 1)

        target_mps = torch.rand_like(x_mps)
        target_cpu = target_mps.detach().clone().cpu()
        loss_mps = F.mse_loss(x_mps, target_mps)
        loss_cpu = F.mse_loss(x_cpu, target_cpu)
        self.assertEqual(loss_mps, loss_cpu)

        loss_mps.backward()
        loss_cpu.backward()
        self.assertEqual(x_mps.grad, x_cpu.grad)

    def test_stack_storage_offset(self):
        # https://github.com/pytorch/pytorch/issues/87856
        x_cpu = torch.tensor([[1, 2]])
        x_mps = x_cpu.detach().clone().to("mps")

        y_cpu = torch.stack((x_cpu[:, :1], x_cpu[:, -1:]), dim=-1)
        y_mps = torch.stack((x_mps[:, :1], x_mps[:, -1:]), dim=-1)

        self.assertEqual(y_cpu, y_mps)

        t_mps = torch.tensor([1, 2, 3, 4], device="mps")
        t_cpu = t_mps.detach().cpu().detach()

        x_mps = t_mps[2:]
        y_mps = t_mps[:2]

        x_cpu = t_cpu[2:]
        y_cpu = t_cpu[:2]

        res_mps = torch.stack((y_mps, x_mps), dim=-1)
        res_cpu = torch.stack((y_cpu, x_cpu), dim=-1)

        self.assertEqual(res_mps, res_cpu)

    def test_unsafe_chunk(self):
        # https://github.com/pytorch/pytorch/issues/91065
        a = torch.rand(5, dtype=torch.float32, device="cpu")
        ret = a.unsafe_chunk(4, 0)
        y = ret[0] * ret[2]
        a_mps = a.to("mps")
        ret_mps = a_mps.unsafe_chunk(4, 0)
        y_mps = ret_mps[0] * ret_mps[2]
        self.assertEqual(y, y_mps)

    def test_slice_casting(self):
        # generate random binary numbers
        cpu_in = torch.bernoulli(torch.empty(1, 1, 128, 128).uniform_(0, 1)).to(torch.uint8)
        mps_in = cpu_in.detach().clone().to("mps")
        # check copy_cast(unit8 -> bool) on tensors with storage offset
        cpu_out = cpu_in[:, :, 11 : 12, :12].to(torch.bool)
        mps_out = mps_in[:, :, 11 : 12, :12].to(torch.bool)
        self.assertEqual(cpu_out, mps_out)

    def test_slice_reshape_contg_view(self):
        import torch

        x_mps = torch.randn(1, 4800, 2, device="mps")
        x_cpu = x_mps.detach().clone().cpu()

        r_mps = x_mps + 2
        r_cpu = x_cpu + 2

        self.assertEqual(r_mps, r_cpu)

    def test_contiguous_slice_2d(self):
        def helper(shape):
            for i in range(0, shape[0]):
                for j in range(0, shape[1]):
                    t_mps = torch.randn(shape, device="mps")
                    t_cpu = t_mps.detach().clone().cpu()

                    y_mps = t_mps[i:, :j]
                    y_cpu = t_cpu[i:, :j]
                    self.assertEqual(y_mps + 1, y_cpu + 1)

                    y_mps = t_mps[i:, j]
                    y_cpu = t_cpu[i:, j]
                    self.assertEqual(y_mps + 1, y_cpu + 1)

                    y_mps = t_mps[i, :j]
                    y_cpu = t_cpu[i, :j]
                    self.assertEqual(y_mps + 1, y_cpu + 1)

                    y_mps = t_mps[:i, :j]
                    y_cpu = t_cpu[:i, :j]
                    self.assertEqual(y_mps + 1, y_cpu + 1)

                    y_mps = t_mps[:i, j]
                    y_cpu = t_cpu[:i, j]
                    self.assertEqual(y_mps + 1, y_cpu + 1)

                    y_mps = t_mps[:i, j:]
                    y_cpu = t_cpu[:i, j:]
                    self.assertEqual(y_mps + 1, y_cpu + 1)

        l = []
        for N in range(1, 3):
            l.append(N)
            for C in range(1, 3):
                l.append(C)
                helper(l)
                for D in range(1, 3):
                    l.append(D)
                    helper(l)
                    for H in range(1, 3):
                        l.append(H)
                        helper(l)
                        for W in range(1, 3):
                            l.append(W)
                            helper(l)
                            l.pop()
                        l.pop()
                    l.pop()
                l.pop()
            l.pop()

        helper([9, 15, 4])
        helper([9, 3, 2])
        helper([3, 4, 18, 22])
        helper([3, 4, 18, 22, 150])

    def test_contiguous_slice_3d(self):
        x = torch.randn(2, 3, 3, device="mps")
        x_cpu = x.detach().clone().cpu()
        x = x[:1]
        x_cpu = x_cpu[:1]
        out = x[:, 0:1, 0:1] * x[:, 1:2, 1:2]
        out_cpu = x_cpu[:, 0:1, 0:1] * x_cpu[:, 1:2, 1:2]
        self.assertEqual(out, out_cpu)

    def test_view_slice(self):
        # https://github.com/pytorch/pytorch/issues/83995
        NUM_SAMPLES = 60
        s = (0, 1)

        X = torch.rand(8000, 3, dtype=torch.float32, device='cpu')
        X_mps = X.detach().clone().to("cpu")

        idx = torch.randint(0, X.shape[0], (1,)).repeat(len(s))
        pts = torch.randint(0, X.shape[0], (NUM_SAMPLES, X.shape[1]))
        idx_mps = idx.to("mps")
        pts_mps = pts.to("mps")
        pts[:, s] = idx
        pts_mps[:, s] = idx_mps

        actual_pts = torch.zeros(NUM_SAMPLES, X.shape[1], dtype=torch.float)
        actual_pts_mps = torch.zeros(NUM_SAMPLES, X.shape[1], dtype=torch.float, device="mps")

        for i in range(NUM_SAMPLES):
            for j in range(X.shape[1]):
                actual_pts_mps[i, j] = X_mps[pts_mps[i, j], j]
                actual_pts[i, j] = X[pts[i, j], j]
                self.assertEqual(actual_pts[i, j], actual_pts_mps[i, j])

    def test_slice_scatter(self):
        shape = (4, 4)
        tensor = torch.randint(10, shape, device="mps")
        tensor_before = tensor.clone()
        torch.empty(shape[0], shape[1] * 2, device="mps")[:, ::2].copy_(tensor)
        torch.testing.assert_close(tensor, tensor_before)

    def test_slice(self):
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        cpu_x = torch.tensor(values, device='cpu')
        mps_x = (torch.tensor(values, device='mps', dtype=torch.float))

        cpu_slice1 = cpu_x[:2, :]
        mps_slice1 = mps_x[:2, :]
        self.assertEqual(cpu_slice1, mps_slice1)

        cpu_slice2 = cpu_x[:, :1]
        mps_slice2 = mps_x[:, :1]
        self.assertEqual(cpu_slice2, mps_slice2)

        cpu_slice3 = cpu_x[1:2, :]
        mps_slice3 = mps_x[1:2, :]
        self.assertEqual(cpu_slice3, mps_slice3.to('cpu'))

        cpu_slice4 = cpu_x[1, :]
        mps_slice4 = mps_x[1, :].to('cpu')
        self.assertEqual(cpu_slice4, mps_slice4)

    @parametrize("torch_type", arg_values=[torch.float16, torch.float32, torch.bfloat16])
    def test_slice_view_api(self, torch_type: torch.dtype):

        def helper(x_tensor, y_func, z_func, r_func=None):
            x_mps = x_tensor.detach().clone().to("mps")

            y = y_func(x_tensor)
            y_mps = y_func(x_mps)
            self.assertEqual(y, y_mps)

            z = z_func(y)
            z_mps = z_func(y_mps)
            self.assertEqual(z, z_mps)
            self.assertEqual(z.storage_offset(), z_mps.storage_offset())

            if r_func:
                r = r_func(z)
                r_mps = r_func(z_mps)
                self.assertEqual(r, r_mps)

        # Skip bfloat16 before MacOS15
        if not (MACOS_VERSION < 15.0 and torch_type == torch.bfloat16):
            # Tests for previously encountered MPS bugs
            helper(
                torch.randn(4, 4, dtype=torch_type),
                lambda x: x[1],
                lambda y: y.reshape(2, 2),
                lambda z: z + 1
            )
            helper(
                torch.randn(2, 4, dtype=torch_type),
                lambda x: x[1],
                lambda y: y + torch.ones(4, device=y.device)
            )
            helper(
                torch.randn(4, 6, dtype=torch_type),
                lambda x: x[1],
                lambda y: y.reshape(3, 2).t(),
                lambda z: z + 1
            )
            helper(
                torch.arange(4, dtype=torch_type).resize(1, 2, 2),
                lambda x: x.permute(2, 0, 1),
                lambda y: y + 1
            )
            helper(
                torch.randn(4, 8, dtype=torch_type),
                lambda x: x.transpose(0, 1).reshape(-1),
                lambda y: y[:2],
                lambda z: z + 1
            )
            helper(
                torch.randn(1, dtype=torch_type),
                lambda x: x.expand(2, 3),
                lambda y: y + torch.ones(2, 3, device=y.device)
            )

    def test_slice_reshape_contiguous(self):
        x = torch.randn(4, 4)
        x_mps = x.detach().clone().to("mps")

        y = x[1]
        y_mps = x_mps[1]
        self.assertEqual(y, y_mps)

        z = y.reshape(2, 2)
        z_mps = y_mps.reshape(2, 2)
        self.assertEqual(z, z_mps)
        self.assertEqual(z.storage_offset(), z_mps.storage_offset())

    def test_scalar_from_slice_unary(self):
        # https://github.com/pytorch/pytorch/issues/82543
        tensor_list = torch.tensor([1.0, 1.2], device="mps")

        for scalar in tensor_list:
            r_mps = torch.ceil(scalar)
            r_cpu = torch.ceil(scalar.to("cpu"))
            self.assertEqual(r_mps.cpu(), r_cpu)

    def test_scalar_from_slice_binary(self):
        # https://github.com/pytorch/pytorch/issues/82543
        def helper(binary_op):
            tensor_list = torch.tensor([1.0, 1.2, 2.5, 1.0], device="mps")

            for scalar in tensor_list:
                r_mps = binary_op(scalar, 1.0)
                r_cpu = binary_op(scalar.cpu(), 1.0)
                self.assertEqual(r_mps.cpu(), r_cpu)
        helper(torch.sub)
        helper(torch.add)
        helper(torch.not_equal)
        helper(torch.eq)

    def test_slice_contiguous_view(self):
        # https://github.com/pytorch/pytorch/issues/77750

        def helper(operator):
            t_mps = torch.tensor([1, 2, 3, 4], device="mps")
            t_cpu = torch.tensor([1, 2, 3, 4], device="cpu")

            # contiguous view
            x_mps = t_mps[2:]  # 3, 4
            y_mps = t_mps[:2]  # 1, 2

            x_cpu = t_cpu[2:]
            y_cpu = t_cpu[:2]

            res_mps = res_cpu = None
            if operator == "<=":
                res_mps = x_mps <= y_mps
                res_cpu = x_cpu <= y_cpu
            elif operator == "<":
                res_mps = x_mps < y_mps
                res_cpu = x_cpu < y_cpu
            elif operator == ">=":
                res_mps = x_mps >= y_mps
                res_cpu = x_cpu >= y_cpu
            elif operator == ">":
                res_mps = x_mps >= y_mps
                res_cpu = x_cpu >= y_cpu
            elif operator == "==":
                res_mps = x_mps == y_mps
                res_cpu = x_cpu == y_cpu
            elif operator == "!=":
                res_mps = x_mps != y_mps
                res_cpu = x_cpu != y_cpu
            elif operator == "stack":
                res_mps = torch.stack((y_mps, x_mps), dim=-1)
                res_cpu = torch.stack((y_cpu, x_cpu), dim=-1)

            self.assertEqual(res_mps, res_cpu)

        for op in ["<=", "<", ">=", ">", "==", "!=", "stack"]:
            helper(op)

    def test_slice_of_slice(self):
        x = torch.tensor([0.5, 0.5], device="cpu")
        x_mps = torch.tensor([0.5, 0.5], device="mps")

        tensor = x[1][None]
        tensor_mps = x_mps[1][None]

        res = tensor.ne(0)
        res_mps = tensor_mps.ne(0)

        self.assertEqual(res, res_mps)

    def test_index_storage_offset(self):
        # https://github.com/pytorch/pytorch/issues/78107

        a = torch.tensor([8.2670e-01, -1.0293e+00])
        b_cpu = a[0]
        c_cpu = a[1]

        # both 'b' and 'c' are views of 'a'
        # 'b' has a storage offset of 0, while 'c' has a storage offset of 1
        # when copying from 'cpu' to 'mps', c will have a storage_offset of 1 which needs to be taking into account,
        # otherwise it ends with same value as 'b'
        b = b_cpu.to('mps')
        c = c_cpu.to('mps')

        res_mps = b > c
        res_cpu = b_cpu > c_cpu
        self.assertEqual(res_mps, res_cpu)

        res_mps = c > b
        res_cpu = c_cpu > b_cpu
        self.assertEqual(res_mps, res_cpu)

    def test_flatten(self):
        values = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        cpu_x = torch.tensor(values, device='cpu')
        mps_x = torch.tensor(values, device='mps')

        cpu_flatten1 = cpu_x.flatten()
        mps_flatten1 = mps_x.flatten().to('cpu')
        self.assertEqual(cpu_flatten1, mps_flatten1)

        cpu_flatten2 = cpu_x.flatten(start_dim=1)
        mps_flatten2 = mps_x.flatten(start_dim=1).to('cpu')
        self.assertEqual(cpu_flatten2, mps_flatten2)

        cpu_flatten3 = cpu_x.flatten(end_dim=1)
        mps_flatten3 = mps_x.flatten(end_dim=1).to('cpu')
        self.assertEqual(cpu_flatten3, mps_flatten3)

    # Test repeat
    def test_repeat(self):
        def helper(shape, repeats):

            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            y = x.repeat(repeats)
            ref_y = cpu_x.repeat(repeats)

            cpu_grad = torch.randn(ref_y.shape)
            grad = cpu_grad.to('mps')

            y.backward(gradient=grad)
            ref_y.backward(gradient=cpu_grad)

            self.assertEqual(y, ref_y)
            self.assertEqual(x.grad, cpu_x.grad)

        helper((2, 3, 4, 5), (2, 3, 4, 5))
        helper((2, 3, 4), (4, 3, 2, 5, 7, 2))
        helper((3, 4, 5), (2, 3, 4, 5))
        helper((3, 4, 5), (2, 2, 2))

    def test_torch_repeat_interleave(self, device="mps"):
        y = torch.tensor([[1, 2], [3, 4]], device=device)
        # exercise single argument function signature
        temp = y.repeat_interleave(2)
        self.assertEqual(torch.Size([8]), temp.size())

        for dtype in [torch.int, torch.long]:
            lengths = torch.tensor([1, 2], dtype=dtype, device="mps")
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

    def test_repeat_interleave(self, device="mps"):
        x = torch.tensor([0, 1, 2, 3], device=device)
        expected = torch.tensor([1, 2, 2, 3, 3, 3], device=device)
        # Prior to macos 13.3, input of dtype=torch.int64 returns dtype=torch.int32
        self.assertEqual(torch.repeat_interleave(x), expected, exact_dtype=MACOS_VERSION >= 13.3)

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.arange(4, device=device).reshape(2, 2))

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.arange(4.0, device=device))

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.tensor([1, 2, -1, 3, 4], device=device))

        y = torch.tensor([[1, 2], [3, 4]], device=device)

        y1_v1 = torch.repeat_interleave(y, 2)
        y1_v2 = torch.repeat_interleave(y, torch.tensor(2, device=device))
        y1_v3 = torch.repeat_interleave(y, torch.tensor([2], device=device))
        y1_expect = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4], device=device)
        self.assertEqual(y1_v1, y1_expect)
        self.assertEqual(y1_v2, y1_expect)
        self.assertEqual(y1_v3, y1_expect)

        y2 = torch.repeat_interleave(y, 3, dim=1)
        y2_expect = torch.tensor([[1, 1, 1, 2, 2, 2],
                                  [3, 3, 3, 4, 4, 4]], device=device)
        self.assertEqual(y2, y2_expect)

        y3 = torch.repeat_interleave(y, torch.tensor([1, 2], device=device), dim=0)
        y3_expect = torch.tensor([[1, 2],
                                  [3, 4],
                                  [3, 4]], device=device)
        self.assertEqual(y3, y3_expect)

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(y, torch.tensor([1, 2, 3], device=device), dim=0)

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(y, torch.arange(9, device=device).reshape(3, 3), dim=0)

        # test zero sized dimension
        x = torch.zeros((5, 0), device=device)
        y = torch.repeat_interleave(x, repeats=3, dim=1)
        self.assertEqual(y, x.new_zeros(5, 0, device=device))

        x = torch.tensor([], dtype=torch.int64, device=device)
        y = torch.repeat_interleave(x, x)
        self.assertEqual(y, x)

    def test_repeat_interleave_simple(self):
        def helper(shape, dtype=torch.float32, num_repeats=torch.Tensor(), dim=None):
            x = torch.randn(shape, dtype=dtype, device="mps")
            x_cpu = x.detach().clone().cpu()

            num_repeats_cpu = num_repeats.detach().clone().cpu()

            repeats = torch.repeat_interleave(x, num_repeats, dim)
            repeats_cpu = torch.repeat_interleave(x_cpu, num_repeats_cpu, dim)

            self.assertEqual(repeats, repeats_cpu)
        helper(shape=3, num_repeats=torch.tensor([100], device="mps"))
        helper(shape=(2, 2), num_repeats=torch.tensor([3, 3], device="mps"), dim=0)
        helper(shape=(10, 15, 8), num_repeats=torch.arange(10, device="mps"), dim=0)
        helper(shape=(10, 15, 8), num_repeats=torch.randint(0, 100, (15, ), device="mps"), dim=1)
        helper(shape=(10, 15, 30), num_repeats=torch.randint(0, 100, (30, ), device="mps"), dim=2)

    def test_count_nonzero(self):
        def helper(dtype):
            n = [
                [[1, 0, 2], [3, 0, 2], [7, 9, -4]],
                [[0, 2, 3], [3, 2, 1], [2, 0, 0]],
            ]
            cpu_x = torch.tensor(n, dtype=dtype)
            mps_x = torch.tensor(n, dtype=dtype).to('mps')

            # All non-zeros
            self.assertEqual(
                torch.count_nonzero(cpu_x),
                torch.count_nonzero(mps_x)
            )

            # dim=1
            self.assertEqual(
                torch.count_nonzero(cpu_x, dim=1),
                torch.count_nonzero(mps_x, dim=1)
            )

            # dim=(0, 1)
            self.assertEqual(
                torch.count_nonzero(cpu_x, dim=(0, 1)),
                torch.count_nonzero(mps_x, dim=(0, 1))
            )
        helper(torch.int32)
        helper(torch.int64)
        helper(torch.float16)
        helper(torch.float32)

    def _test_module_empty_input(self, module, inp, check_size=True):
        inp.requires_grad_(True)
        out = module(inp)
        gO = torch.rand_like(out)
        out.backward(gO)
        if check_size:
            self.assertEqual(out.size(), inp.size())
        for p in module.parameters():
            if p.requires_grad:
                self.assertEqual(p.grad, torch.zeros_like(p.grad))
        self.assertEqual(inp.grad, torch.zeros_like(inp))

    # Test dtype casting, with and without simultaneous device change
    def test_to(self):
        values = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        cpu_x = torch.tensor(values, device='cpu')
        mps_x = torch.tensor(values, device='mps')

        self.assertEqual(cpu_x.int(), mps_x.int().cpu())
        self.assertEqual(cpu_x.bool(), mps_x.bool().cpu())
        self.assertEqual(cpu_x.float(), mps_x.float().cpu())

        self.assertEqual(torch.tensor(1.3, device='mps').int().cpu(),
                         torch.tensor(1, dtype=torch.int32))
        self.assertEqual(torch.tensor(0.0, device='mps').bool().cpu(), torch.tensor(False))
        self.assertEqual(torch.tensor(0.1, device='mps').bool().cpu(), torch.tensor(True))
        self.assertEqual(torch.tensor(0.1, device='mps').bool().int().cpu(),
                         torch.tensor(1, dtype=torch.int32))
        self.assertEqual(torch.tensor(0.1, device='mps').bool().int().float().cpu(),
                         torch.tensor(1.0))
        self.assertEqual(torch.tensor(4.25, device='mps').to('cpu', torch.int),
                         torch.tensor(4, dtype=torch.int32))
        self.assertEqual(torch.tensor(4.25, device='cpu').to('mps', torch.int).cpu(),
                         torch.tensor(4, dtype=torch.int32))
        self.assertEqual(torch.tensor(-8.34, device='cpu').to('mps', torch.int),
                         torch.tensor(-8.34, device='cpu').to('mps').to(torch.int))
        # Cast int8 and uint8 to float and compare results
        # See https://github.com/pytorch/pytorch/issues/80009 for more details
        cpu_byte = torch.tensor([60, 160, 20, 220], dtype=torch.uint8)
        cpu_char = torch.tensor([60, -60, 20, -120], dtype=torch.uint8)
        for x_cpu in [cpu_byte, cpu_char]:
            x_mps = x_cpu.to('mps')
            self.assertEqual(x_mps.to(torch.float32), x_cpu.to(torch.float32))


    def test_setitem_scalar(self) -> None:
        device = 'mps'
        for dtype in [torch.int32, torch.float32, torch.int64]:
            for i in range(3, 6):
                for j in range(3, 6):
                    t = torch.zeros(i, j, dtype=dtype, device=device)
                    self.assertEqual(t.sum(), 0)
                    t[1, 1] = 1
                    t[2, 1] = j
                    t[1, 2] = i
                    self.assertEqual(t[1, 1], 1)
                    self.assertEqual(t[1, 2], i)
                    self.assertEqual(t[2, 1], j)
                    self.assertEqual(t.sum(), 1 + i + j)

    def test_stride_of_strides(self) -> None:
        x = torch.rand(32, 1, device='mps')
        y = x.as_strided(size=(32, 2), stride=(1, 0))
        # Casting stride of strided tensor to CPU use to crash with "buffer is not large enough." assert
        # See https://github.com/pytorch/pytorch/issues/79181#issuecomment-1154683435
        z = y.as_strided(size=(32, 3), stride=(1, 0)).to("cpu")
        self.assertEqual(x.to("cpu").as_strided(size=(32, 3), stride=(1, 0)), z)

    def test_type_casting(self):
        # https://github.com/pytorch/pytorch/issues/81567
        def helper(data, to_dtype):
            a_cpu = torch.tensor(data)
            a_mps = a_cpu.to(torch.device('mps'))

            res_cpu = a_cpu.type(to_dtype)
            res_mps = a_mps.type(to_dtype)
            self.assertEqual(res_cpu, res_mps)

        helper([9.0, 3.0, 5.0, 4.0], torch.LongTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.FloatTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.IntTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.ShortTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.HalfTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.CharTensor)
        helper([9.0, 3.0, 5.0, 4.0], torch.ByteTensor)

    def test_to_casting(self):
        # https://github.com/pytorch/pytorch/issues/81567
        def helper(data, to_dtype):
            a_cpu = torch.tensor(data)
            a_mps = a_cpu.to(torch.device('mps'))

            res_cpu = a_cpu.to(to_dtype)
            res_mps = a_mps.to(to_dtype)
            self.assertEqual(res_cpu, res_mps)

        helper([9.0, 3.0, 5.0, 4.0], torch.int64)
        helper([9.0, 3.0, 5.0, 4.0], torch.float)
        helper([9.0, 3.0, 5.0, 4.0], torch.int32)
        helper([9.0, 3.0, 5.0, 4.0], torch.short)
        helper([9.0, 3.0, 5.0, 4.0], torch.half)
        helper([9.0, 3.0, 5.0, 4.0], torch.int8)
        helper([9.0, 3.0, 5.0, 4.0], torch.uint8)

    def test_storage_offset_greater_than_src_nbytes(self):
        # https://github.com/pytorch/pytorch/issues/80844
        n_tensors = 100
        n_tensor_elems = 784
        elems = torch.arange(n_tensors * n_tensor_elems, dtype=torch.float32)

        tensor_list = []
        for i in range(0, n_tensors - 1):
            # create a list of contiguous view tensors (view tensor created by the slice op)
            t = elems[n_tensor_elems * i : n_tensor_elems * (i + 1)]
            tensor_list.append(t)

        for i in range(0, n_tensors - 1):
            t = tensor_list[i].view(1, n_tensor_elems)
            t_mps = t.to("mps")
            self.assertEqual(t, t_mps.cpu(), f"i={i}")

    # See https://github.com/pytorch/pytorch/issues/82427
    # and https://github.com/pytorch/pytorch/issues/83692
    def test_full_bugs(self):
        # Test should not crash
        x = torch.full((3, 3), True, device='mps')
        # torch.full should work for uint8
        y_mps = torch.full((2, 2), 247, device='mps', dtype=torch.uint8)
        y_cpu = torch.full((2, 2), 247, device='cpu', dtype=torch.uint8)
        self.assertEqual(y_mps, y_cpu)

    def test_div_bugs(self):
        for (dtype, mode) in itertools.product(integral_types(), ['trunc', 'floor']):
            x = torch.tensor(list(range(1, 11)), device='mps', dtype=dtype)
            y = torch.div(x, 101, rounding_mode=mode)
            self.assertEqual(y.sum(), 0)

    # See https://github.com/pytorch/pytorch/issues/82663
    def test_bool_expand(self):
        x = torch.tensor([[1], [0]], dtype=torch.bool, device='mps')
        y = torch.tensor([0, 1], dtype=torch.bool, device='mps')
        self.assertFalse(torch.equal(x.expand(2, 2), y.expand(2, 2)))

    def test_int_expand(self):
        x = torch.tensor([[1], [0]], dtype=torch.int8, device='mps')
        y = torch.tensor([0, 1], dtype=torch.int8, device='mps')
        self.assertFalse(torch.equal(x.expand(2, 2), y.expand(2, 2)))

    # Empty unary op should return tensor of the same size
    def test_empty_neg(self):
        x = torch.tensor([[]], device='mps')
        y = -x
        self.assertEqual(x, y)

    def _test_unique_scalar_empty(self, dtype, device, f):
        # test scalar
        x = torch.tensor(0, dtype=dtype, device=device)
        unique, inverse, counts = f(x, return_inverse=True, return_counts=True)
        expected_unique = torch.tensor([0], dtype=dtype, device=device)
        expected_inverse = torch.tensor(0, device=device)
        expected_counts = torch.tensor([1], device=device)
        self.assertEqual(unique, expected_unique)
        self.assertEqual(inverse, expected_inverse)
        self.assertEqual(counts, expected_counts)

        # test zero sized tensor
        x = torch.zeros((0, 0, 3), dtype=dtype, device=device)
        unique, inverse, counts = f(x, return_inverse=True, return_counts=True)
        expected_unique = torch.tensor([], dtype=dtype, device=device)
        expected_inverse = torch.empty((0, 0, 3), dtype=torch.long, device=device)
        expected_counts = torch.tensor([], dtype=torch.long, device=device)
        self.assertEqual(unique, expected_unique)
        self.assertEqual(inverse, expected_inverse)
        self.assertEqual(counts, expected_counts)

    def _test_unique_with_expects(self, device, dtype, f, x, expected_unique, expected_inverse, expected_counts, additional_shape):
        def ensure_tuple(x):
            if isinstance(x, torch.Tensor):
                return (x,)
            return x

        for return_inverse in [True, False]:
            for return_counts in [True, False]:
                # test with expected
                ret = ensure_tuple(f(x, return_inverse=return_inverse, return_counts=return_counts))
                self.assertEqual(len(ret), 1 + int(return_inverse) + int(return_counts))
                self.assertEqual(expected_unique, ret[0])
                if return_inverse:
                    self.assertEqual(expected_inverse, ret[1])
                if return_counts:
                    count_index = 1 + int(return_inverse)
                    self.assertEqual(expected_counts, ret[count_index])

                # tests per-element unique on a higher rank tensor.
                y = x.view(additional_shape)
                y_unique, y_inverse, y_counts = f(y, return_inverse=True, return_counts=True)
                self.assertEqual(expected_unique, y_unique)
                self.assertEqual(expected_inverse.view(additional_shape), y_inverse)
                self.assertEqual(expected_counts, y_counts)

    def test_unique_all_dtypes(self, device="mps"):
        def helper(dtype):
            def ensure_tuple(x):
                if isinstance(x, torch.Tensor):
                    return (x,)
                return x

            if dtype is torch.bool:
                x = torch.tensor([True, False, False, False, True, False, True, False], dtype=torch.bool, device=device)
                expected_unique = torch.tensor([False, True], dtype=torch.bool, device=device)
                expected_inverse = torch.tensor([1, 0, 0, 0, 1, 0, 1, 0], dtype=torch.long, device=device)
                expected_counts = torch.tensor([5, 3], dtype=torch.long, device=device)
            else:
                x = torch.tensor([1, 2, 3, 2, 8, 5, 2, 3], dtype=dtype, device=device)
                expected_unique = torch.tensor([1, 2, 3, 5, 8], dtype=dtype, device=device)
                expected_inverse = torch.tensor([0, 1, 2, 1, 4, 3, 1, 2], device=device)
                expected_counts = torch.tensor([1, 3, 2, 1, 1], device=device)

            # test sorted unique
            fs = (
                lambda x, **kwargs: torch.unique(x, sorted=True, **kwargs),
                lambda x, **kwargs: x.unique(sorted=True, **kwargs),
            )
            x_sliced = torch.empty(x.size(0) * 2, dtype=dtype, device=device)[::2].copy_(x)
            xs = (x, x_sliced)
            for f, x in product(fs, xs):
                self._test_unique_with_expects(device, dtype, f, x, expected_unique, expected_inverse, expected_counts, (2, 2, 2))
                self._test_unique_scalar_empty(dtype, device, f)

            # test unsorted unique
            fs = (
                lambda x, **kwargs: torch.unique(x, sorted=False, **kwargs),
                lambda x, **kwargs: x.unique(sorted=False, **kwargs)
            )
            for f, x in product(fs, xs):
                self._test_unique_scalar_empty(dtype, device, f)
                for return_inverse, return_counts in product((True, False), repeat=2):
                    ret = ensure_tuple(f(x, return_inverse=return_inverse, return_counts=return_counts))
                    self.assertEqual(len(ret), 1 + int(return_inverse) + int(return_counts))
                    x_list = x.tolist()
                    x_unique_list = ret[0].tolist()
                    self.assertEqual(expected_unique.tolist(), sorted(x_unique_list))
                    if return_inverse:
                        x_inverse_list = ret[1].tolist()
                        for i, j in enumerate(x_inverse_list):
                            self.assertEqual(x_list[i], x_unique_list[j])
                    if return_counts:
                        count_index = 1 + int(return_inverse)
                        x_counts_list = ret[count_index].tolist()
                        for i, j in zip(x_unique_list, x_counts_list):
                            count = 0
                            for k in x_list:
                                if k == i:
                                    count += 1
                            self.assertEqual(j, count)
        [helper(dtype) for dtype in [torch.float32, torch.int64, torch.int32, torch.int16, torch.uint8]]

    def test_unique(self):
        def helper(x, return_inverse, return_counts):
            cpu_x = x
            x = cpu_x.detach().clone().to('mps')

            result = torch.unique(x, return_inverse=return_inverse, return_counts=return_counts)
            result_cpu = torch.unique(cpu_x, return_inverse=return_inverse, return_counts=return_counts)

            self.assertEqual(result, result_cpu)
        helper(torch.tensor([1, 2, 4, 2, 1]), False, False)
        helper(torch.randint(3, (10, )), False, False)
        helper(torch.randint(3, (10, )), True, False)
        helper(torch.randint(3, (10, )), False, True)
        helper(torch.randint(3, (10, )), True, True)
        helper(torch.randint(3, (1, )), True, True)
        helper(torch.randint(3, (0, )), True, True)
        # Regression test for https://github.com/pytorch/pytorch/issues/104879
        x = torch.arange(2, device="mps")
        self.assertEqual(x.reshape(1, 1, 2).unique(), x)

    def test_unique_consecutive(self):
        def helper(x, dim, return_inverse, return_counts):
            cpu_x = x
            x = cpu_x.detach().clone().to('mps')

            result = torch.unique_consecutive(x, dim=dim, return_inverse=return_inverse, return_counts=return_counts)
            result_cpu = torch.unique_consecutive(cpu_x, dim=dim, return_inverse=return_inverse, return_counts=return_counts)

            self.assertEqual(result, result_cpu)
        helper(torch.tensor([1, 2, 4, 2, 1]), 0, False, False)
        helper(torch.randint(3, (10, )), 0, False, False)
        helper(torch.randint(3, (10, )), 0, True, False)
        helper(torch.randint(3, (10, )), 0, False, True)
        helper(torch.randint(3, (10, )), 0, True, True)
        helper(torch.randint(3, (10, )), 0, True, True)
        helper(torch.randint(3, (1, )), 0, True, True)
        helper(torch.randint(3, (0, )), 0, True, True)

        helper(torch.tensor([[1, 1, 2, 3, 3, 2], [1, 1, 1, 2, 2, 1]]), 0, False, False)
        helper(torch.tensor([[1, 1, 2, 3, 3, 2], [1, 1, 1, 2, 2, 1]]), 0, True, True)
        helper(torch.randint(2, (20, 2)), 0, True, True)
        helper(torch.randint(2, (1, 2)), 0, True, True)
        helper(torch.randint(2, (0, 2)), 0, True, True)

        helper(torch.tensor([[1, 1, 2, 3, 3, 2], [1, 1, 1, 2, 2, 1]]), 1, False, False)
        helper(torch.tensor([[1, 1, 2, 3, 3, 2], [1, 1, 1, 2, 2, 1]]), 1, True, True)
        helper(torch.randint(2, (2, 20)), 1, True, True)
        helper(torch.randint(2, (2, 1)), 1, True, True)
        helper(torch.randint(2, (2, 0)), 1, True, True)

    # See https://github.com/pytorch/pytorch/issues/85675
    def test_cat_non_contiguous(self):
        def rotate_subset(data, dim):
            x1 = data[:, :, :2, :]
            x2 = data[:, :, 2:, :]
            self.assertFalse(x1.is_contiguous())
            self.assertFalse(x2.is_contiguous())
            return torch.concat((x1, x2), dim=dim)
        for dtype in MPS_DTYPES:
            if dtype == torch.bool or (dtype.is_complex and MACOS_VERSION < 14.0):
                continue
            data = torch.arange(48).to(dtype=dtype).reshape(1, 2, 4, 6)
            data = data.to(memory_format=torch.channels_last)
            mps_data = data.to("mps")
            self.assertEqual(data, mps_data)
            for dim in range(data.dim()):
                cpu_result = rotate_subset(data, dim)
                mps_result = rotate_subset(mps_data, dim)
                self.assertEqual(cpu_result, mps_result.to("cpu"))
                # TODO: enable memory format test
                # self.assertEqual(cpu_result.is_contiguous(), mps_result.is_contiguous())

    # See https://github.com/pytorch/pytorch/issues/152701
    def test_jacfwd_cat(self):
        def fn(x, y):
            return torch.cat((x, y))

        x = torch.rand(2, device="mps")
        y = torch.rand(3, device="mps")
        rc = torch.func.jacfwd(fn)(x, y)
        self.assertEqual(rc.shape, (5, 2))

    # See https://github.com/pytorch/pytorch/issues/85967
    def test_from_numpy_non_contiguous(self):
        a = np.arange(9).reshape(3, 3)[:, :2]
        t_cpu = torch.tensor(a, device="cpu")
        t_mps = torch.tensor(a, device="mps")
        self.assertEqual(t_cpu, t_mps.to("cpu"))

    # See https://github.com/pytorch/pytorch/issues/86954
    def test_copy_non_contiguous(self):
        x = torch.arange(27).reshape(3, 3, 3).permute(2, 0, 1)
        self.assertFalse(x.is_contiguous())
        y = x.to('mps')
        self.assertFalse(y.is_contiguous())
        self.assertEqual(x, y.to('cpu'))

        x = torch.arange(4**3).reshape(4, 4, 4).permute((2, 0, 1))[1:, ::2]
        y = x.to('mps')
        self.assertEqual(x, y.to('cpu'))

        x = torch.full((4, 4, 4, 4), 13, device="cpu")
        y = torch.full((4, 4, 4, 4), 13, device="mps")
        z = torch.arange(4**4).reshape(4, 4, 4, 4).permute(3, 2, 0, 1)[1::, ::2]
        x.permute(3, 2, 1, 0)[1::, ::2] = z
        # As y is on MPS and z on CPU, this dispatches to a copy operator
        y.permute(3, 2, 1, 0)[1::, ::2] = z
        self.assertEqual(x, y.to('cpu'))

    # See https://github.com/pytorch/pytorch/issues/95417
    def test_copy_storage_offset(self):
        x_cpu = torch.zeros(5, device="cpu", dtype=torch.float32)
        x_mps = torch.zeros(5, device="mps", dtype=torch.float32)
        update_cpu = torch.tensor([1, 1], device="cpu", dtype=torch.int64)
        update_mps = torch.tensor([1, 1], device="mps", dtype=torch.int64)
        x_cpu[2:4] = update_cpu
        x_mps[2:4] = update_mps  # implicit type casting and copy
        self.assertEqual(x_cpu, x_mps)

        x_cpu[2:4] = update_mps  # implicit device moving and copy
        self.assertEqual(x_cpu, x_mps)

    def test_copy_broadcasting(self):
        def helper(src_shape, dst_shape, src_dtype, dst_dtype):
            cpu_src = torch.randint(0, 127, src_shape).to(src_dtype)
            cpu_dst = torch.randint(0, 127, dst_shape).to(dst_dtype)
            cpu_result = cpu_dst.copy_(cpu_src)
            mps_src = cpu_src.to("mps")
            mps_dst = cpu_dst.to("mps")
            mps_result = mps_dst.copy_(mps_src)
            self.assertEqual(cpu_result, mps_result)

        test_dtypes = [torch.float32, torch.int32, torch.int16, torch.int8]

        for (src_dtype, dst_dtype) in itertools.product(test_dtypes, test_dtypes):
            helper((2, 1), (2, 3), src_dtype, dst_dtype)
            helper((2, 1), (2, 2), src_dtype, dst_dtype)
            helper((3, 1, 4, 1), (3, 4, 4, 5), src_dtype, dst_dtype)
            helper((3,), (2, 3), src_dtype, dst_dtype)
            helper((2,), (2, 2), src_dtype, dst_dtype)
            helper((4, 1, 5), (3, 4, 4, 5), src_dtype, dst_dtype)
            helper((4, 1, 5), (4, 0, 5), src_dtype, dst_dtype)
            helper((1, 5), (4, 0, 5), src_dtype, dst_dtype)
            helper((3, 1, 0), (3, 5, 0), src_dtype, dst_dtype)
            helper((0, 1, 0), (0, 5, 0), src_dtype, dst_dtype)
        # Regression test for https://github.com/pytorch/pytorch/issues/107867
        self.assertEqual(torch.tensor([[1]], device='mps').item(), 1.0)

    # See https://github.com/pytorch/pytorch/pull/84742
    # and https://github.com/pytorch/pytorch/pull/78319
    @parametrize("binop", ['add', 'sub', 'mul', 'div'])
    def test_binops_dtype_precedence(self, binop):
        # Test dtype precedence (casting order) in binary operations by comparing to CPU result
        # Example values for all dtypes supported on the MPS backend
        sample_vals = {
            torch.bool: [False, True],
            torch.int16: [-15, 0, 1, 10],
            torch.int32: [-376, 0, 1, 13],
            torch.int64: [-8, 0, 1, 77],
            torch.float16: [-234.5, 0.0, 1.0, 2.0],
            torch.float32: [-1.0, 0.0, 0.1, 111.99],
        }
        # Test all combinations of dtypes, operations, dimensionality
        for dtype1, dtype2 in itertools.product(sample_vals, repeat=2):
            # bool minus bool is generally unsupported, so skip
            if binop == 'sub' and (dtype1 == torch.bool or dtype2 == torch.bool):
                continue
            full_shape = (10,)
            for val1, val2 in itertools.product(sample_vals[dtype1], sample_vals[dtype2]):
                # print(f'{dtype1},{dtype2}: ({val1}).{binop}({val2})')
                # print(getattr(torch.tensor(val1, dtype=dtype1, device='mps'), binop)
                #            (torch.tensor(val2, dtype=dtype2, device='mps')))
                # print(getattr(torch.tensor(val1, dtype=dtype1, device='cpu'), binop)
                #            (torch.tensor(val2, dtype=dtype2, device='cpu')))
                self.assertEqual(
                    getattr(torch.tensor(val1, dtype=dtype1, device='mps'), binop)
                           (torch.tensor(val2, dtype=dtype2, device='mps')),
                    getattr(torch.tensor(val1, dtype=dtype1, device='cpu'), binop)
                           (torch.tensor(val2, dtype=dtype2, device='cpu')))
                self.assertEqual(
                    getattr(torch.tensor([val1], dtype=dtype1, device='mps'), binop)
                           (torch.tensor([val2], dtype=dtype2, device='mps')),
                    getattr(torch.tensor([val1], dtype=dtype1, device='cpu'), binop)
                           (torch.tensor([val2], dtype=dtype2, device='cpu')))
                self.assertEqual(
                    getattr(torch.tensor(val1, dtype=dtype1, device='mps'), binop)
                           (torch.tensor([val2], dtype=dtype2, device='mps')),
                    getattr(torch.tensor(val1, dtype=dtype1, device='cpu'), binop)
                           (torch.tensor([val2], dtype=dtype2, device='cpu')))
                self.assertEqual(
                    getattr(torch.tensor([val1], dtype=dtype1, device='mps'), binop)
                           (torch.tensor(val2, dtype=dtype2, device='mps')),
                    getattr(torch.tensor([val1], dtype=dtype1, device='cpu'), binop)
                           (torch.tensor(val2, dtype=dtype2, device='cpu')))
                # Test tensors created with torch.full
                x1 = torch.full(full_shape, val1, dtype=dtype1, device='mps')
                y1 = torch.tensor(val2, dtype=dtype2, device='mps')
                x2 = torch.full(full_shape, val1, dtype=dtype1, device='cpu')
                y2 = torch.tensor(val2, dtype=dtype2, device='cpu')
                self.assertEqual(getattr(x1, binop)(y1), getattr(x2, binop)(y2))
                x3 = torch.tensor(val1, dtype=dtype1, device='mps')
                y3 = torch.full(full_shape, val2, dtype=dtype2, device='mps')
                x4 = torch.tensor(val1, dtype=dtype1, device='cpu')
                y4 = torch.full(full_shape, val2, dtype=dtype2, device='cpu')
                self.assertEqual(getattr(x3, binop)(y3), getattr(x4, binop)(y4))
                self.assertEqual(
                    getattr(torch.tensor(val1, dtype=dtype1, device='mps'), binop)
                           (torch.full(full_shape, val2, dtype=dtype2, device='mps')),
                    getattr(torch.tensor(val1, dtype=dtype1, device='cpu'), binop)
                           (torch.full(full_shape, val2, dtype=dtype2, device='cpu')))

    def test_xor_non_contigous(self):
        # See https://github.com/pytorch/pytorch/issues/145203
        x_mps = torch.randint(-16000, 16000, (10, 2), dtype=torch.int16, device="mps")
        x_cpu = x_mps.detach().cpu()

        x_mps[:, 0] ^= 3
        x_cpu[:, 0] ^= 3

        self.assertEqual(x_mps.cpu(), x_cpu)

    def test_nansum(self):
        def helper(dtype, noncontiguous, dim):
            zero_cpu = torch.zeros((), dtype=dtype)

            # Randomly scale the values
            scale = random.randint(10, 100)
            x_cpu: torch.Tensor = make_tensor(
                (5, 5), dtype=dtype, device='cpu',
                low=-scale, high=scale, noncontiguous=noncontiguous)

            if dtype.is_floating_point:
                nan_mask_cpu = x_cpu < (0.2 * scale)
                x_no_nan_cpu = torch.where(nan_mask_cpu, zero_cpu, x_cpu)
                x_cpu[nan_mask_cpu] = np.nan
            else:
                x_no_nan_cpu = x_cpu

            x_mps = x_cpu.to('mps')
            actual_out_mps = torch.empty(0, dtype=dtype, device='mps')
            expect_out_cpu = torch.empty(0, dtype=dtype)
            dim_kwargs = {"dim": dim} if dim is not None else {}
            expect = torch.sum(x_no_nan_cpu, **dim_kwargs)

            actual_cpu = torch.nansum(x_cpu, **dim_kwargs)
            # Sanity check on CPU
            self.assertEqual(expect, actual_cpu)

            # Test MPS
            actual_mps = torch.nansum(x_mps, **dim_kwargs)
            # Test out= variant
            torch.nansum(x_mps, out=actual_out_mps, **dim_kwargs)
            torch.nansum(x_cpu, out=expect_out_cpu, **dim_kwargs)
            self.assertEqual(expect, actual_mps)
            self.assertEqual(expect_out_cpu, actual_out_mps)

        args = itertools.product(
            (torch.float16, torch.float32, torch.int32, torch.int64),   # dtype
            (True, False),                                              # noncontiguous
            (0, 1, None),                                               # dim
        )

        for dtype, noncontiguous, dim in args:
            with self.subTest(dtype=dtype, noncontiguous=noncontiguous, dim=dim):
                helper(dtype, noncontiguous, dim)

    def test_cumsum_all_dtypes(self):
        def helper(dtype):
            t = torch.tensor([1, 1, 1, 1], device="mps", dtype=dtype)
            t_cpu = torch.tensor([1, 1, 1, 1], device="cpu")

            a = t.cumsum(0, dtype=dtype)
            a_cpu = t_cpu.cumsum(0, dtype=dtype)

            self.assertEqual(a.cpu(), a_cpu)
        [helper(dtype) for dtype in [torch.int8, torch.int16, torch.int32, torch.float32]]

        try:
            helper(torch.int64)
        except Exception as e:
            e_string = str(e)
            self.assertEqual(e_string, "MPS does not support cumsum_out_mps op with int64 input." +
                             " Support has been added in macOS 13.3")

    def test_cumsum_bool(self):
        a = torch.ones(2**16, dtype=torch.bool)
        t_cpu = a.cumsum(0)
        t_mps = a.to("mps").cumsum(0)

        self.assertEqual(t_cpu, t_mps)

    def test_cumsum_minus_one_axis(self):
        def helper(dtype):
            # Test with axis -1
            cpu_x = None
            if dtype == torch.float32:
                cpu_x = torch.randn(10, 3, device='cpu', dtype=torch.float32)
            else:
                cpu_x = torch.randint(0, 20, (10, 3), device='cpu', dtype=torch.float32)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = cpu_x.cumsum(-1)
            y = x.cumsum(-1)

            self.assertEqual(y, cpu_y)

        [helper(dtype) for dtype in [torch.float32, torch.int16, torch.int32, torch.uint8]]

    def test_cumprod_all_dtypes(self):
        def helper(dtype):
            t = torch.tensor([1, 1, 1, 1], device="mps", dtype=dtype)
            t_cpu = torch.tensor([1, 1, 1, 1], device="cpu")

            a = t.cumprod(0, dtype=dtype)
            a_cpu = t_cpu.cumprod(0, dtype=dtype)

            self.assertEqual(a.cpu(), a_cpu)
        [helper(dtype) for dtype in [torch.int8, torch.int16, torch.int32, torch.float32]]

        try:
            helper(torch.int64)
        except Exception as e:
            e_string = str(e)
            self.assertEqual(e_string, "MPS does not support cumprod_out_mps op with int64 input."
                             + " Support has been added in macOS 13.3")

    def test_cumprod_minus_one_axis(self):
        def helper(dtype):
            # Test with axis -1
            cpu_x = None
            if dtype == torch.float32:
                cpu_x = torch.randn(10, 3, device='cpu', dtype=torch.float32)
            else:
                cpu_x = torch.randint(0, 20, (10, 3), device='cpu', dtype=torch.float32)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = cpu_x.cumprod(-1)
            y = x.cumprod(-1)

            self.assertEqual(y, cpu_y)

        [helper(dtype) for dtype in [torch.float32, torch.int16, torch.int32, torch.uint8]]

    def test_median_int16(self):
        def helper(shape, dtype):
            cpu_x = torch.randint(-9999, 9999, shape, device='cpu', dtype=dtype)
            x = cpu_x.detach().clone().to('mps')

            median_result = torch.median(x)
            median_result_cpu = torch.median(cpu_x)
            self.assertEqual(median_result, median_result_cpu)

        helper((2, 8, 4, 5), torch.int16)

    def test_activation_checkpoint_does_not_error(self):
        from torch.utils.checkpoint import checkpoint

        for use_reentrant in (True, False):
            a = torch.tensor(1., device="mps", requires_grad=True)

            def fn(x):
                return x.sin().cos().exp()

            out = checkpoint(fn, a, use_reentrant=use_reentrant)
            out.backward()

    def test_as_strided(self):
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        values_1 = [[1.0, 1.0], [1.0, 1.0]]
        cpu_x = torch.tensor(values, device='cpu')
        ones1 = torch.tensor(values_1, device='mps')
        x = cpu_x.detach().clone().to('mps').requires_grad_()
        strided_cpu = torch.as_strided(cpu_x, (2, 2), (1, 2))
        strided_mps = torch.as_strided(x, (2, 2), (1, 2))
        self.assertEqual(strided_mps, strided_cpu)
        strided_cpu_out = strided_cpu + ones1.to('cpu')
        strided_mps_out = strided_mps + ones1
        self.assertEqual(strided_cpu_out, strided_mps_out)

        # test with storage offsets
        cpu_x = torch.rand(3, 3, device='cpu')
        mps_x = cpu_x.to('mps')
        strided_cpu1 = torch.as_strided(cpu_x, (2, 2), (1, 2), 0)
        strided_mps1 = torch.as_strided(mps_x, (2, 2), (1, 2), 0)
        strided_cpu2 = torch.as_strided(cpu_x, (2, 2), (1, 2), 1)
        strided_mps2 = torch.as_strided(mps_x, (2, 2), (1, 2), 1)
        strided_cpu_out = strided_cpu1 - strided_cpu2
        strided_mps_out = strided_mps1 - strided_mps2
        self.assertEqual(strided_cpu_out, strided_mps_out)

    def test_unfold(self):
        x = torch.arange(1., 8)
        x_mps = torch.arange(1., 8, device="mps")

        y = x.unfold(0, 2, 1)
        y_mps = x_mps.unfold(0, 2, 1)

        self.assertEqual(y, y_mps)

    def test_unfold_all_devices_and_dtypes(self):
        supported_dtypes = [torch.float32, torch.float16, torch.int64, torch.int32, torch.int16, torch.uint8]
        for dt in supported_dtypes:
            x = torch.empty((0, 1, 3, 0), dtype=dt, device="mps")
            self.assertEqual((0, 1, 1, 0, 3), x.unfold(2, 3, 2).shape)

    def test_unfold_scalars(self):
        x = torch.tensor(0.5, device="mps")
        # unfold on a 0-dimensional tensor should always return a 1-d dimensional
        # tensor of shape [size] (i.e., the second parameter to unfold)

        self.assertEqual(torch.empty(0, device="mps"), x.unfold(0, 0, 1))
        self.assertEqual(torch.empty(0, device="mps"), x.unfold(0, 0, 2))
        self.assertEqual(torch.tensor([0.5], device="mps"), x.unfold(0, 1, 1))

    def test_bincount_simple(self):
        input = torch.randint(0, 8, (5,), dtype=torch.int32, device="mps")
        input_cpu = input.to("cpu")
        weights = torch.linspace(0, 1, steps=5, device="mps", dtype=torch.float32)
        weights_cpu = weights.to("cpu")

        x = torch.bincount(input)
        x_cpu = torch.bincount(input_cpu)
        self.assertEqual(x, x_cpu)

        y = input.bincount(weights)
        y_cpu = input_cpu.bincount(weights_cpu)
        self.assertEqual(y, y_cpu)

    def test_bincount_reduction(self):
        device = "mps"
        # negative input throws
        with self.assertRaisesRegex(RuntimeError, '1-d non-negative integral'):
            torch.bincount(torch.tensor([1, -1], device=device, dtype=torch.int32))
        # n-d input, with n > 1 throws
        with self.assertRaisesRegex(RuntimeError, '1-d non-negative integral'):
            torch.bincount(torch.tensor([[1, 2], [3, 4]], device=device))
        # minlength < 0 throws
        with self.assertRaisesRegex(RuntimeError, 'minlength should be >= 0'):
            torch.bincount(torch.tensor([1, 3], device=device),
                           torch.tensor([.2, .2], device=device),
                           minlength=-1)
        # n-d weights, with n > 1 throws
        with self.assertRaisesRegex(RuntimeError, '1-d'):
            torch.bincount(torch.tensor([1, 0], device=device, dtype=torch.int32),
                           torch.tensor([[1., 0.3], [1., 0.3]], device=device, dtype=torch.float))
        # input and weights dim mismatch
        with self.assertRaisesRegex(RuntimeError, 'same length'):
            torch.bincount(torch.tensor([1, 0], device=device, dtype=torch.int32),
                           torch.tensor([1., 0.3, 0.5], device=device, dtype=torch.float))
        # 1-d input with no elements and default minlength
        self.assertEqual(torch.bincount(torch.tensor([], device=device, dtype=torch.long)),
                         torch.zeros(0, dtype=torch.long, device=device))
        # 1-d input with no elements and specified minlength
        self.assertEqual(torch.bincount(torch.tensor([], device=device, dtype=torch.long), minlength=10),
                         torch.zeros(10, dtype=torch.long, device=device))

        # test tensor method without weights
        long_counts = torch.tensor(
            [0, 3, 2, 1, 3], dtype=torch.uint8, device=device).bincount()
        self.assertEqual(
            torch.tensor([1, 1, 1, 2], dtype=torch.int64, device=device),
            long_counts)
        # test avoiding overflow for uint8 (#76979)
        count_uint8 = torch.tensor([0, 1, 2, 3, 255], dtype=torch.uint8, device=device).bincount()
        count_int16 = torch.tensor([0, 1, 2, 3, 255], dtype=torch.int16, device=device).bincount()
        self.assertEqual(count_uint8, count_int16)
        # test minlength functionality
        int_counts = torch.bincount(
            torch.tensor([1, 1, 1, 1], device=device, dtype=torch.int32), minlength=5)
        self.assertEqual(
            torch.tensor([0, 4, 0, 0, 0], dtype=torch.int64, device=device),
            int_counts)
        # test weights
        byte_counts = torch.bincount(
            torch.tensor([0, 1, 1, 1, 4], device=device, dtype=torch.int32),
            torch.tensor([.1, .2, .3, .4, .5], device=device))
        self.assertEqual(
            torch.tensor([0.1, 0.9, 0, 0, 0.5], device=device), byte_counts)
        byte_counts = torch.bincount(
            torch.tensor([0, 1, 1, 1, 4], device=device, dtype=torch.int32),
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.int8, device=device))
        self.assertEqual(
            torch.tensor([1, 9, 0, 0, 5], device=device, dtype=torch.int32), byte_counts)
        # test non-contiguous inputs and weights
        inputs = torch.tensor([[0, 0], [3, 1], [2, 1], [1, 1], [3, 4]], device=device, dtype=torch.int32)
        weights = torch.tensor([[.1, 1], [.2, 2], [.3, 3], [.4, 4], [.5, 5]], device=device)
        for i in [0, 1]:
            assert not inputs[:, i].is_contiguous(), "Inputs are supposed to be non-contiguous"
            assert not weights[:, i].is_contiguous(), "Weights are supposed to be non-contiguous"
        # inputs are non-contiguous but weights are contiguous
        self.assertEqual(inputs[:, 0].bincount(), torch.tensor([1, 1, 1, 2]))
        # inputs and weights are non-contiguous
        self.assertEqual(
            inputs[:, 1].bincount(weights[:, 1]),
            torch.tensor([1, 9, 0, 0, 5], dtype=torch.float32))
        # weights are non-contiguous but inputs are contiguous
        self.assertEqual(inputs[:, 1].contiguous().bincount(weights[:, 1]),
                         torch.tensor([1, 9, 0, 0, 5], dtype=torch.float32))

        # test bincount on non-contiguous slices
        all0s = torch.zeros((32, 2), dtype=torch.int32, device=device)
        self.assertEqual(all0s[:, 0].bincount(), torch.tensor([32]))

        all1s = torch.ones((32, 2), dtype=torch.int32, device=device)
        self.assertEqual(all1s[:, 0].bincount(), torch.tensor([0, 32]))

        # test large number of bins - global memory use
        big_exp = torch.zeros(100, device=device)
        big_exp[-1] = 50.0
        big_w = torch.tensor([.5] * 100, device=device)
        big_out = torch.tensor([99] * 100, device=device, dtype=torch.int32).bincount(big_w)
        self.assertEqual(big_exp, big_out)
        # test large input size
        big_exp = torch.zeros(2, device=device, dtype=torch.int64)
        big_exp[1] = 10
        big_out = torch.ones(10, dtype=torch.int8, device=device).bincount()
        self.assertEqual(big_exp, big_out)

    def test_bincount(self):
        device = "mps"
        input_size = (5000,)
        w = torch.randn(input_size, dtype=torch.float, device=device)
        w_cpu = w.cpu()

        t = torch.randint(50, input_size, dtype=torch.int8, device=device)
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))

        t = torch.randint(500, input_size, dtype=torch.int32, device=device)
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))

        t = torch.randint(2000, input_size, dtype=torch.int32, device=device)
        self.assertEqual(t.cpu().bincount(), t.bincount())
        self.assertEqual(t.cpu().bincount(w_cpu), t.bincount(w))

        t = torch.zeros([10], dtype=torch.int32, device=device)
        t[0] = 35488
        counted = t.bincount(minlength=65536)
        self.assertEqual(torch.sum(counted), 10)

    def test_sum_backward(self):
        def helper(n, c):
            values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
            cpu_x = torch.tensor(values, device='cpu', requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            all_sum = torch.sum(x)
            all_sum_cpu = torch.sum(cpu_x)

            all_sum.backward()
            all_sum_cpu.backward()
            self.assertEqual(all_sum, all_sum_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        helper(3, 3)

    # L1 loss
    def test_l1_loss(self):
        def helper(shape, reduction):
            # create the criterion
            loss = torch.nn.L1Loss(reduction=reduction)

            inputCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            targetCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()
            targetMPS = targetCPU.detach().clone().to('mps')

            # forward pass
            outputCPU = loss(inputCPU, targetCPU)
            outputMPS = loss(inputMPS, targetMPS)
            self.assertEqual(outputCPU, outputMPS)

            # backward pass
            if reduction != 'none':
                # chose 2 just to make the grad_output > 1 in backward pass
                outputCPU.backward(gradient=torch.full_like(outputCPU, 2))
                outputMPS.backward(gradient=torch.full_like(outputMPS, 2))
                self.assertEqual(inputCPU.grad, inputMPS.grad)

        helper([8, 5, 4], 'none')
        helper([7, 5, 2, 4], 'sum')
        # verify if changes in shape would cause cached graph lookup problems
        helper([7, 5, 2, 4, 6], 'sum')
        helper([8, 4, 5, 7, 6], 'mean')

    # Mean Squared Error
    def test_mse_loss(self):
        def helper(shape, reduction):
            # create the criterion
            loss = torch.nn.MSELoss(reduction=reduction)

            inputCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            targetCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()
            targetMPS = targetCPU.detach().clone().to('mps')

            # forward pass
            outputCPU = loss(inputCPU, targetCPU)
            outputMPS = loss(inputMPS, targetMPS)
            self.assertEqual(outputCPU, outputMPS)

            # backward pass
            if reduction != 'none':
                # chose 2 just to make the grad_output > 1 in backward pass
                outputCPU.backward(gradient=torch.full_like(outputCPU, 2))
                outputMPS.backward(gradient=torch.full_like(outputMPS, 2))
                self.assertEqual(inputCPU.grad, inputMPS.grad)

        helper([8, 5, 4], 'none')
        helper([7, 5, 2, 4], 'sum')
        # verify if changes in shape would cause cached graph lookup problems
        helper([7, 5, 2, 4, 6], 'sum')
        helper([8, 4, 5, 7, 6], 'mean')
        helper((3, 3, 0), 'sum')
        helper((3, 3, 0), 'mean')
        helper((3, 3, 0), 'none')

    def test_mse_loss_strided_output(self):
        # https://github.com/pytorch/pytorch/issues/124621
        lf = nn.MSELoss(reduction='none')
        model_cpu = nn.Sequential(
            nn.Conv1d(3, 3, 1),
        )
        model_mps = copy.deepcopy(model_cpu).to("mps")

        x = torch.randn(128, 10, 3)
        x = x.permute(0, 2, 1)

        x_mps = x.detach().clone().to("mps").permute(0, 2, 1)
        x_mps = x_mps.permute(0, 2, 1)

        y = model_cpu(x)
        y_mps = model_mps(x_mps)

        y = y.permute(0, 2, 1)[:, :5, :]
        y_mps = y_mps.permute(0, 2, 1)[:, :5, :]

        y_hat = torch.randn(128, 5, 3)
        y_hat_mps = y_hat.detach().clone().to("mps")

        loss = lf(y, y_hat)
        loss_mps = lf(y_mps, y_hat_mps)
        self.assertEqual(loss, loss_mps)

    def test_mse_loss_unsupported_types(self):
        loss = nn.MSELoss()
        for dtype in MPS_DTYPES:
            a_mps = torch.tensor([0, 1, 2], dtype=dtype, device='mps')
            a_cpu = torch.tensor([0, 1, 2], dtype=dtype, device='cpu')
            if dtype.is_floating_point:
                self.assertEqual(loss(a_mps, a_mps), loss(a_cpu, a_cpu))
                continue
            self.assertRaises(RuntimeError, lambda: loss(a_mps, a_mps))
            self.assertRaises(RuntimeError, lambda: loss(a_cpu, a_cpu))

    # Binary Cross Enropy
    def test_bce_loss_simple(self):
        def helper(shape, reduction):
            # create the criterion
            loss = torch.nn.BCELoss(reduction=reduction)

            # input and target must be within [0..1]
            input_t = np.random.random_sample(size=shape).astype(np.float32)
            target_t = np.random.random_sample(size=shape).astype(np.float32)
            inputCPU = torch.tensor(input_t, device='cpu', dtype=torch.float, requires_grad=True)
            targetCPU = torch.tensor(target_t, device='cpu', dtype=torch.float, requires_grad=False)
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()
            targetMPS = targetCPU.detach().clone().to('mps')

            # forward pass
            outputCPU = loss(inputCPU, targetCPU)
            outputMPS = loss(inputMPS, targetMPS)
            self.assertEqual(outputCPU, outputMPS)

            # backward pass
            if reduction != 'none':
                # chose 0.6 just to have the grad_output != 1
                outputCPU.backward(gradient=torch.full_like(outputCPU, 0.6))
                outputMPS.backward(gradient=torch.full_like(outputMPS, 0.6))
                self.assertEqual(inputCPU.grad, inputMPS.grad)

        helper([8, 5, 4], 'none')
        helper([7, 5, 2, 4], 'sum')
        # verify if changes in shape would cause cached graph lookup problems
        helper([7, 5, 2, 4, 6], 'sum')
        helper([8, 4, 5, 7, 6], 'mean')
        helper([1, 1, 32, 32], 'mean')

    def test_bce_loss_always_nonnegative(self):
        target = torch.ones(5, device='mps')
        input = torch.ones(5, device='mps')
        self.assertEqual((nn.BCELoss()(input, target) < 0).sum(), 0)

        target = torch.zeros(5, device='mps')
        input = torch.zeros(5, device='mps')
        self.assertEqual((nn.BCELoss()(input, target) < 0).sum(), 0)

    def test_bce_loss_size_mismatch(self):
        bceloss = nn.BCELoss()
        a = torch.rand(25, device='mps')
        b = torch.rand(25, 1, device='mps')
        with self.assertRaisesRegex(ValueError, r'Using a target size \('):
            bceloss(a, b)

    def test_bce_with_logits_gives_same_result_as_sigmoid_and_bce_loss_large_tensors_with_grad(self):
        x_size = 1024
        y_size = 256
        target = torch.rand(x_size, y_size, device='mps')

        for reduction in ['none', 'mean', 'sum']:
            output_sig = torch.rand(x_size, y_size, device='mps') - 0.5
            output_logits = output_sig.detach().clone()

            output_sig.requires_grad = True
            output_logits.requires_grad = True
            weight = torch.rand(y_size, device='mps')

            loss_sig = nn.BCELoss(weight, reduction=reduction)(
                torch.sigmoid(output_sig), target
            )
            loss_logits = nn.BCEWithLogitsLoss(weight, reduction=reduction)(
                output_logits, target
            )

            self.assertEqual(loss_logits, loss_sig)

            if reduction == 'none':
                grad = torch.rand(x_size, y_size, device='mps')
                loss_sig.backward(grad)
                loss_logits.backward(grad)
            else:
                loss_sig.backward()
                loss_logits.backward()

            self.assertEqual(output_sig.grad, output_logits.grad)

    def test_bce_with_logits_has_correct_grad_at_zero(self):
        output = torch.zeros(3, 1, requires_grad=True, device='mps')
        target = torch.zeros(3, 1, device='mps')
        nn.BCEWithLogitsLoss(reduction='sum')(output, target).backward()
        expected_grad = torch.empty(3, 1, device='mps').fill_(0.5)
        self.assertEqual(output.grad, expected_grad)

    def test_bce_with_logits_broadcasts_weights(self):
        target = torch.rand(16, 4, device='mps')
        output = torch.rand(16, 4, device='mps') - 0.5

        weight = torch.rand(4, device='mps')
        out1 = nn.BCEWithLogitsLoss(weight)(output, target)

        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCEWithLogitsLoss(weight)(output, target)

        self.assertEqual(out1, out2)

        weight = torch.rand(16, 1, device='mps')
        out1 = nn.BCEWithLogitsLoss(weight)(output, target)

        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCEWithLogitsLoss(weight)(output, target)

        self.assertEqual(out1, out2)

    def test_bce_with_logits_ones_in_pos_weights_are_the_same_as_none(self):
        target = torch.rand(64, 4, device='mps')
        output = torch.rand(64, 4, device='mps') - 0.5
        pos_weight = torch.ones(64, 4, device='mps')

        self.assertEqual(nn.BCEWithLogitsLoss()(output, target),
                         nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target))

    def test_bce_with_logits_broadcasts_pos_weights(self):
        target = torch.rand(64, 4, device='mps')
        output = torch.rand(64, 4, device='mps') - 0.5
        pos_weight = torch.rand(4, device='mps')
        out1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target)

        pos_weight1 = pos_weight.expand(1, 4)
        out2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight1)(output, target)

        pos_weight2 = pos_weight.expand(64, 4)
        out3 = nn.BCEWithLogitsLoss(pos_weight=pos_weight2)(output, target)

        self.assertEqual(out1, out2)
        self.assertEqual(out1, out3)

    def test_bce_with_logits_with_pos_weight_has_correct_grad_at_zero(self):
        output = torch.zeros(3, 1, requires_grad=True, device='mps')
        target = torch.zeros(3, 1, device='mps')
        pos_weight = torch.ones(3, 1, device='mps')
        nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum')(output, target).backward()
        expected_grad = torch.empty(3, 1, device='mps').fill_(0.5)
        grad = output.grad
        self.assertEqual(grad, expected_grad)

    def test_bce_with_logits_stability(self):
        output = torch.tensor([0., -120.], device='mps')
        target = torch.tensor([0., 1.], device='mps')
        pos_weight = torch.tensor([1., 1.], device='mps')

        out1 = nn.BCEWithLogitsLoss()(output, target)
        self.assertTrue(torch.isfinite(out1).all().item())

        out2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target)
        self.assertTrue(torch.isfinite(out2).all().item())

    def test_bce_loss_broadcasts_weights(self):
        sigmoid = nn.Sigmoid()
        target = torch.rand(16, 4, device='mps')
        output = torch.rand(16, 4, device='mps') - 0.5

        weight = torch.rand(4, device='mps')
        out1 = nn.BCELoss(weight)(sigmoid(output), target)

        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCELoss(weight)(sigmoid(output), target)

        self.assertEqual(out1, out2)

        weight = torch.rand(16, 1, device='mps')
        out1 = nn.BCELoss(weight)(sigmoid(output), target)

        weight = weight.expand(16, 4).contiguous()
        out2 = nn.BCELoss(weight)(sigmoid(output), target)

        self.assertEqual(out1, out2)

    def test_cross_entropy_loss(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/116095
        loss = nn.CrossEntropyLoss()
        pred = torch.randn(3, 5, requires_grad=True, dtype=torch.float16, device='mps')
        target = torch.ones(3, dtype=torch.long, device='mps')
        output = loss(pred, target)
        output.backward()

    def test_log_softmax(self):
        values = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        cpu_x = torch.tensor(values, device='cpu', requires_grad=True)
        mps_x = torch.tensor(values, device='mps', requires_grad=True)

        cpu_log_softmax = F.log_softmax(cpu_x, dim=0)
        mps_log_softmax = F.log_softmax(mps_x, dim=0)
        self.assertEqual(cpu_log_softmax, mps_log_softmax.to('cpu'))

        cpu_grad = torch.ones_like(cpu_log_softmax)
        mps_grad = torch.ones_like(cpu_log_softmax).to('mps')

        cpu_log_softmax.backward(gradient=cpu_grad)
        mps_log_softmax.backward(gradient=mps_grad)

        self.assertEqual(cpu_x.grad, mps_x.grad.to('cpu'))

    def test_log_softmax_large_numbers(self):
        values = [
            [10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0],
            [-10.0, -100.0, -1000.0, -10000.0, -100000.0, -1000000.0]
        ]
        cpu_x = torch.tensor(values, device='cpu', requires_grad=True)
        mps_x = torch.tensor(values, device='mps', requires_grad=True)

        cpu_log_softmax = F.log_softmax(cpu_x, dim=-1)
        mps_log_softmax = F.log_softmax(mps_x, dim=-1)
        self.assertEqual(cpu_log_softmax, mps_log_softmax.to('cpu'))

        cpu_grad = torch.ones_like(cpu_log_softmax)
        mps_grad = torch.ones_like(cpu_log_softmax).to('mps')

        cpu_log_softmax.backward(gradient=cpu_grad)
        mps_log_softmax.backward(gradient=mps_grad)

        self.assertEqual(cpu_x.grad, mps_x.grad.to('cpu'))

    def test_eq(self):
        values1 = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        values2 = [[[1.0, 2.0, 15.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [0.0, 11.0, 12.0]]]
        mps_x = torch.tensor(values1, device='mps')
        mps_y = torch.tensor(values2, device='mps')
        cpu_x = torch.tensor(values1, device='cpu')
        cpu_y = torch.tensor(values2, device='cpu')
        result_mps = torch.eq(mps_x, mps_y)
        result_cpu = torch.eq(cpu_x, cpu_y)

        self.assertEqual(result_cpu, result_mps.to('cpu'))

    def test_signed_vs_unsigned_comparison(self):
        cpu_x = torch.tensor((-1, 2, 3), device='cpu', dtype=torch.uint8)
        mps_x = torch.tensor((-1, 2, 3), device='mps', dtype=torch.uint8)
        # in the comparison of signed vs. unsigned we should always cast to unsigned
        self.assertEqual(cpu_x == -1, mps_x == -1)
        self.assertEqual(cpu_x > -1, mps_x > -1)
        self.assertEqual(cpu_x < -1, mps_x < -1)

    def test_eq_int64(self):
        values1 = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        values2 = [[[1, 2, 15], [4, 5, 6]], [[7, 8, 9], [0, 11, 12]]]
        mps_x = torch.tensor(values1, device='mps')
        mps_y = torch.tensor(values2, device='mps')
        cpu_x = torch.tensor(values1, device='cpu')
        cpu_y = torch.tensor(values2, device='cpu')
        result_mps = torch.eq(mps_x, mps_y)
        result_cpu = torch.eq(cpu_x, cpu_y)

        self.assertEqual(result_cpu, result_mps.to('cpu'))

    def test_ne(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            result_mps = torch.ne(mps_x, mps_y)
            result_cpu = torch.ne(cpu_x, cpu_y)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_ne_scalar(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            result_mps = torch.ne(mps_x, 0.0)
            result_cpu = torch.ne(cpu_x, 0.0)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_lt(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            result_mps = torch.lt(mps_x, mps_y)
            result_cpu = torch.lt(cpu_x, cpu_y)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_lt_scalar(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            result_mps = torch.lt(mps_x, 0.0)
            result_cpu = torch.lt(cpu_x, 0.0)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_le(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            result_mps = torch.le(mps_x, mps_y)
            result_cpu = torch.le(cpu_x, cpu_y)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_le_scalar(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            result_mps = torch.le(mps_x, 0.0)
            result_cpu = torch.le(cpu_x, 0.0)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_ge(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            result_mps = torch.ge(mps_x, mps_y)
            result_cpu = torch.ge(cpu_x, cpu_y)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_ge_scalar(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            result_mps = torch.ge(mps_x, 0.0)
            result_cpu = torch.ge(cpu_x, 0.0)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_gt(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            cpu_y = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')
            result_mps = torch.gt(mps_x, mps_y)
            result_cpu = torch.gt(cpu_x, cpu_y)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_gt_scalar(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float)
            mps_x = cpu_x.detach().clone().to('mps')
            result_mps = torch.gt(mps_x, 0.0)
            result_cpu = torch.gt(cpu_x, 0.0)

            self.assertEqual(result_cpu, result_mps.to('cpu'))

        helper((2, 3, 4, 5))

    def test_argmax(self):
        # https://github.com/pytorch/pytorch/issues/98191
        cpu_tensor = torch.tensor([[0, 1], [2, 1], [1, 0]])
        res_cpu = torch.argmax(cpu_tensor, dim=1)

        mps_tensor = cpu_tensor.to(torch.device('mps'))
        res_mps = torch.argmax(mps_tensor, dim=1)
        self.assertEqual(res_cpu, res_mps)

        # https://github.com/pytorch/pytorch/issues/92311
        mps_tensor = torch.randn(10, 2, device='mps', dtype=torch.float32)
        cpu_tensor = mps_tensor.detach().clone().cpu()

        res_mps = torch.argmax(mps_tensor, dim=1)
        res_cpu = torch.argmax(cpu_tensor, dim=1)
        self.assertEqual(res_cpu, res_mps)

    # Test forward argmin argmax
    def test_argmin_argmax(self):
        def helper(n, c, h, w, reduction_type, dtype=torch.float32):
            if reduction_type == "max":
                arg_reduction_fn = torch.argmax
            else:
                arg_reduction_fn = torch.argmin

            cpu_x = None
            x = None
            if (dtype not in [torch.float32, torch.bool]):
                cpu_x = torch.randint(50, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            elif (dtype == torch.bool):
                cpu_x = torch.randint(2, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            else:
                cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()

            y = arg_reduction_fn(x)
            ref_y = arg_reduction_fn(cpu_x)
            self.assertEqual(y, ref_y)

            y_0 = arg_reduction_fn(x, dim=0)
            refy_0 = arg_reduction_fn(cpu_x, dim=0)
            self.assertEqual(y_0, refy_0)

            y_0dim = arg_reduction_fn(x, dim=0, keepdim=True)
            refy_0dim = arg_reduction_fn(cpu_x, dim=0, keepdim=True)
            self.assertEqual(y_0dim, refy_0dim)

            y_1 = arg_reduction_fn(x, dim=1)
            refy_1 = arg_reduction_fn(cpu_x, dim=1)
            self.assertEqual(y_1, refy_1)

            y_1dim = arg_reduction_fn(x, dim=1, keepdim=True)
            refy_1dim = arg_reduction_fn(cpu_x, dim=1, keepdim=True)
            self.assertEqual(y_1dim, refy_1dim)

            y_2 = arg_reduction_fn(x, dim=2)
            refy_2 = arg_reduction_fn(cpu_x, dim=2)
            self.assertEqual(y_2, refy_2)

            y_2dim = arg_reduction_fn(x, dim=2, keepdim=True)
            refy_2dim = arg_reduction_fn(cpu_x, dim=2, keepdim=True)
            self.assertEqual(y_2dim, refy_2dim)

            y_3 = arg_reduction_fn(x, dim=3)
            refy_3 = arg_reduction_fn(cpu_x, dim=3)
            self.assertEqual(y_3, refy_3)

            y_3dim = arg_reduction_fn(x, dim=3, keepdim=True)
            refy_3dim = arg_reduction_fn(cpu_x, dim=3, keepdim=True)
            self.assertEqual(y_3dim, refy_3dim)

        helper(2, 8, 4, 4, "max", torch.float32)
        helper(2, 8, 4, 4, "max", torch.int32)
        helper(2, 8, 4, 4, "max", torch.float16)
        helper(2, 8, 4, 4, "max", torch.int64)
        helper(2, 8, 4, 4, "min", torch.float32)
        helper(2, 8, 4, 4, "min", torch.int32)
        helper(2, 8, 4, 4, "min", torch.float16)
        helper(2, 8, 4, 4, "min", torch.int64)

    @unittest.skipIf(MACOS_VERSION < 13.3, "Long data type supported from macOS 13.3 and above")
    def test_reduction_sum_max_long_val(self):
        x_mps = torch.tensor([sys.maxsize, sys.maxsize - 10, sys.maxsize - 5, sys.maxsize - 18], device="mps")
        x_cpu = x_mps.detach().clone().cpu()

        res_mps = torch.sum(x_mps)
        res_cpu = torch.sum(x_cpu)
        self.assertEqual(res_mps, res_cpu)

    # Test forward max
    # Note - don't test grad now
    def test_max_el(self):
        def helper(n, c, h, w, dtype=torch.float32):

            if (dtype not in [torch.float32, torch.bool]):
                cpu_x = torch.randint(50, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            elif (dtype == torch.bool):
                cpu_x = torch.randint(2, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            else:
                cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps')

            ref_y = torch.max(cpu_x)
            y = torch.max(x)
            self.assertEqual(y, ref_y)

            for dim in [0, 1, 2, 3]:
                for keepdim in [True, False]:
                    y, idx = torch.max(x, dim=dim, keepdim=keepdim)
                    refy, refidx = torch.max(cpu_x, dim=dim, keepdim=keepdim)
                    self.assertEqual(y, refy)
                    self.assertEqual(idx, refidx)

            y_0 = torch.ones(c, h, w, device='mps', dtype=dtype)
            idx_0 = torch.ones(c, h, w, device='mps', dtype=torch.int64)
            torch.max(x, dim=0, out=(y_0, idx_0))
            refy_0, refidx_0 = torch.max(cpu_x, dim=0)
            self.assertEqual(y_0, refy_0)
            self.assertEqual(idx_0, refidx_0)

            y_0dim = torch.ones(1, c, h, w, device='mps', dtype=dtype)
            idx_0dim = torch.ones(1, c, h, w, device='mps', dtype=torch.int64)
            torch.max(x, dim=0, keepdim=True, out=(y_0dim, idx_0dim))
            refy_0dim, refidx_0dim = torch.max(cpu_x, dim=0, keepdim=True)
            self.assertEqual(y_0dim, refy_0dim)
            self.assertEqual(idx_0dim, refidx_0dim)

            y_1 = torch.ones(n, h, w, device='mps', dtype=dtype)
            idx_1 = torch.ones(n, h, w, device='mps', dtype=torch.int64)
            torch.max(x, dim=1, out=(y_1, idx_1))
            refy_1, refidx_1 = torch.max(cpu_x, dim=1)
            self.assertEqual(y_1, refy_1)
            self.assertEqual(idx_1, refidx_1)

            y_1dim = torch.ones(n, 1, h, w, device='mps', dtype=dtype)
            idx_1dim = torch.ones(n, 1, h, w, device='mps', dtype=torch.int64)
            torch.max(x, dim=1, keepdim=True, out=(y_1dim, idx_1dim))
            refy_1dim, refidx_1dim = torch.max(cpu_x, keepdim=True, dim=1)
            self.assertEqual(y_1dim, refy_1dim)
            self.assertEqual(idx_1dim, refidx_1dim)

            y_2 = torch.ones(n, c, w, device='mps', dtype=dtype)
            idx_2 = torch.ones(n, c, w, device='mps', dtype=torch.int64)
            torch.max(x, dim=2, out=(y_2, idx_2))
            refy_2, refidx_2 = torch.max(cpu_x, dim=2)
            self.assertEqual(y_2, refy_2)
            self.assertEqual(idx_2, refidx_2)

            y_2dim = torch.ones(n, c, 1, w, device='mps', dtype=dtype)
            idx_2dim = torch.ones(n, c, 1, w, device='mps', dtype=torch.int64)
            torch.max(x, dim=2, keepdim=True, out=(y_2dim, idx_2dim))
            refy_2dim, refidx_2dim = torch.max(cpu_x, dim=2, keepdim=True,)
            self.assertEqual(y_2dim, refy_2dim)
            self.assertEqual(idx_2dim, refidx_2dim)

            y_3 = torch.ones(n, c, h, device='mps', dtype=dtype)
            idx_3 = torch.ones(n, c, h, device='mps', dtype=torch.int64)
            torch.max(x, dim=3, out=(y_3, idx_3))
            refy_3, refidx_3 = torch.max(cpu_x, dim=3)
            self.assertEqual(y_3, refy_3)
            self.assertEqual(idx_3, refidx_3)

            y_3dim = torch.ones(n, c, h, 1, device='mps', dtype=dtype)
            idx_3dim = torch.ones(n, c, h, 1, device='mps', dtype=torch.int64)
            torch.max(x, dim=3, keepdim=True, out=(y_3dim, idx_3dim))
            refy_3dim, refidx_3dim = torch.max(cpu_x, dim=3, keepdim=True,)
            self.assertEqual(y_3dim, refy_3dim)
            self.assertEqual(idx_3dim, refidx_3dim)

        helper(2, 8, 4, 5, torch.float32)
        helper(2, 8, 4, 5, torch.int32)
        # helper(2, 8, 4, 5, torch.int64)

    def test_median(self):
        def helper_dtype_int32(n1, n2, n3):
            cpu_x = torch.randint(50, (n1, n2, n3), device='cpu', dtype=torch.int32)
            mps_x = cpu_x.detach().clone().to('mps')

            result_cpu = torch.median(cpu_x)
            result_mps = torch.median(mps_x)

            self.assertEqual(result_cpu, result_mps)

            for dim in [0, 1, 2]:
                for keepdim in [True, False]:
                    y, idx = torch.median(cpu_x, dim=dim, keepdim=keepdim)
                    refy, refidx = torch.median(mps_x, dim=dim, keepdim=keepdim)
                    self.assertEqual(y, refy)
                    self.assertEqual(idx, refidx)

        def helper_dtype_float32(n1, n2, n3):
            cpu_x = torch.randn(n1, n2, n3, device='cpu', dtype=torch.float32)
            mps_x = cpu_x.detach().clone().to('mps')

            result_cpu = torch.median(cpu_x)
            result_mps = torch.median(mps_x)

            self.assertEqual(result_cpu, result_mps)

            for dim in [0, 1, 2]:
                for keepdim in [True, False]:
                    y, idx = torch.median(cpu_x, dim=dim, keepdim=keepdim)
                    refy, refidx = torch.median(mps_x, dim=dim, keepdim=keepdim)
                    self.assertEqual(y, refy)
                    self.assertEqual(idx, refidx)

        helper_dtype_int32(10, 10, 10)  # median at even place
        helper_dtype_int32(3, 3, 3)  # median at odd place
        helper_dtype_int32(1, 1, 1)
        helper_dtype_int32(1, 2, 3)
        helper_dtype_float32(10, 10, 10)
        helper_dtype_float32(3, 3, 3)
        helper_dtype_float32(1, 1, 1)

    def test_any(self):
        def helper(shape):
            input_xs = []
            prod = 1

            for i in range(len(shape)):
                prod *= shape[i]
            input_xs.append(torch.randn(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.ones(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.zeros(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.ones(prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.zeros(prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.int).reshape(shape).bool())
            input_xs.append(torch.ones(prod, dtype=torch.int).reshape(shape).bool())
            input_xs.append(torch.zeros(prod, dtype=torch.int).reshape(shape).bool())

            for i, cpu_x in enumerate(input_xs):
                x = cpu_x.detach().clone().to('mps')
                y = torch.any(x)
                ref_y = torch.any(cpu_x)
                self.assertEqual(y, ref_y)

                y_0 = torch.any(x, dim=0)
                refy_0 = torch.any(cpu_x, dim=0)
                self.assertEqual(y_0, refy_0)

                y_0dim = torch.any(x, dim=0, keepdim=True)
                refy_0dim = torch.any(cpu_x, dim=0, keepdim=True)
                self.assertEqual(y_0dim, refy_0dim)

                y_0dim = torch.any(x, dim=0, keepdim=True)
                refy_0dim = torch.any(cpu_x, dim=0, keepdim=True)
                self.assertEqual(y_0dim, refy_0dim)

                y_1 = torch.any(x, dim=1)
                refy_1 = torch.any(cpu_x, dim=1)
                self.assertEqual(y_1, refy_1)

                y_1dim = torch.any(x, dim=1, keepdim=True)
                refy_1dim = torch.any(cpu_x, dim=1, keepdim=True)
                self.assertEqual(y_1dim, refy_1dim)

                if (len(shape) > 2):
                    y_2 = torch.any(x, dim=2)
                    refy_2 = torch.any(cpu_x, dim=2)
                    self.assertEqual(y_2, refy_2)

                    y_2dim = torch.any(x, dim=2, keepdim=True)
                    refy_2dim = torch.any(cpu_x, dim=2, keepdim=True)
                    self.assertEqual(y_2dim, refy_2dim)

                    y_3 = torch.any(x, dim=3)
                    refy_3 = torch.any(cpu_x, dim=3)
                    self.assertEqual(y_3, refy_3)

                    y_3dim = torch.any(x, dim=3, keepdim=True)
                    refy_3dim = torch.any(cpu_x, dim=3, keepdim=True)
                    self.assertEqual(y_3dim, refy_3dim)
        helper((1, 1, 1, 1))
        helper((1, 1, 3, 3))
        helper((7, 13))
        helper((2, 8, 4, 5))

    def test_reduction_ops_5D(self):
        def helper(fn, dim):
            shape = (1, 1, 2, 1, 1)
            x_cpu = fn(torch.zeros(shape), dim=dim)
            x_mps = fn(torch.zeros(shape, device="mps"), dim=dim)
            self.assertEqual(x_cpu, x_mps.cpu())
        for fn in [torch.any, torch.all]:
            for dim in range(0, 4):
                helper(fn, dim)

        # 6D tensor reductions
        # Regression test for https://github.com/pytorch/pytorch/issues/95538
        x = (torch.rand(2, 3, 4, 3, 4, 2, device="mps") - .5).relu()
        self.assertEqual(x.all(), x.cpu().all())
        for i in range(-5, 6):
            self.assertEqual(x.all(dim=i), x.cpu().all(dim=i))

    def test_all(self):
        def helper(shape):
            input_xs = []
            prod = 1

            for i in range(len(shape)):
                prod *= shape[i]
            input_xs.append(torch.randn(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.ones(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.zeros(prod, dtype=torch.float).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.ones(prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.zeros(prod, dtype=torch.int).reshape(shape))
            input_xs.append(torch.arange(0, prod, dtype=torch.int).reshape(shape).bool())
            input_xs.append(torch.ones(prod, dtype=torch.int).reshape(shape).bool())
            input_xs.append(torch.zeros(prod, dtype=torch.int).reshape(shape).bool())

            for i, cpu_x in enumerate(input_xs):
                x = cpu_x.detach().clone().to('mps')
                y = torch.all(x)
                ref_y = torch.all(cpu_x)
                self.assertEqual(y, ref_y)

                y_0 = torch.all(x, dim=0)
                refy_0 = torch.all(cpu_x, dim=0)
                self.assertEqual(y_0, refy_0)

                y_0dim = torch.all(x, dim=0, keepdim=True)
                refy_0dim = torch.all(cpu_x, dim=0, keepdim=True)
                self.assertEqual(y_0dim, refy_0dim)

                y_0dim = torch.all(x, dim=0, keepdim=True)
                refy_0dim = torch.all(cpu_x, dim=0, keepdim=True)
                self.assertEqual(y_0dim, refy_0dim)

                y_1 = torch.all(x, dim=1)
                refy_1 = torch.all(cpu_x, dim=1)
                self.assertEqual(y_1, refy_1)

                y_1dim = torch.all(x, dim=1, keepdim=True)
                refy_1dim = torch.all(cpu_x, dim=1, keepdim=True)
                self.assertEqual(y_1dim, refy_1dim)
                if (len(shape) > 2):
                    y_2 = torch.all(x, dim=2)
                    refy_2 = torch.all(cpu_x, dim=2)
                    self.assertEqual(y_2, refy_2)

                    y_2dim = torch.all(x, dim=2, keepdim=True)
                    refy_2dim = torch.all(cpu_x, dim=2, keepdim=True)
                    self.assertEqual(y_2dim, refy_2dim)

                    y_3 = torch.all(x, dim=3)
                    refy_3 = torch.all(cpu_x, dim=3)
                    self.assertEqual(y_3, refy_3)

                    y_3dim = torch.all(x, dim=3, keepdim=True)
                    refy_3dim = torch.all(cpu_x, dim=3, keepdim=True)
                    self.assertEqual(y_3dim, refy_3dim)

        helper((1, 1, 1, 1))
        helper((1, 1, 3, 3))
        helper((7, 13))
        helper((2, 8, 4, 5))
        # Empty tensor
        x_cpu = torch.tensor([], dtype=torch.bool)
        x_mps = x_cpu.to("mps")
        self.assertEqual(x_cpu.all(), x_mps.all().cpu())

    # Test forward min
    def test_min_el(self):
        def helper(n, c, h, w):
            cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            y = torch.min(x)
            ref_y = torch.min(cpu_x)
            self.assertEqual(y, ref_y)

            y_0, idx_0 = torch.min(x, dim=0)
            refy_0, refidx_0 = torch.min(cpu_x, dim=0)
            self.assertEqual(y_0, refy_0)
            self.assertEqual(idx_0, refidx_0)

            y_0 = torch.ones(c, h, w, device='mps', dtype=torch.float)
            idx_0 = torch.ones(c, h, w, device='mps', dtype=torch.int64)
            torch.min(x, dim=0, out=(y_0, idx_0))
            refy_0, refidx_0 = torch.min(cpu_x, dim=0)
            self.assertEqual(y_0, refy_0)
            self.assertEqual(idx_0, refidx_0)

            y_0dim, idx_0dim = torch.min(x, dim=0, keepdim=True)
            refy_0dim, refidx_0dim = torch.min(cpu_x, dim=0, keepdim=True)
            self.assertEqual(y_0dim, refy_0dim)
            self.assertEqual(idx_0dim, refidx_0dim)

            y_0dim = torch.ones(1, c, h, w, device='mps', dtype=torch.float)
            idx_0dim = torch.ones(1, c, h, w, device='mps', dtype=torch.int64)
            torch.min(x, dim=0, keepdim=True, out=(y_0dim, idx_0dim))
            refy_0dim, refidx_0dim = torch.min(cpu_x, dim=0, keepdim=True)
            self.assertEqual(y_0dim, refy_0dim)
            self.assertEqual(idx_0dim, refidx_0dim)

            y_1, idx_1 = torch.min(x, dim=1)
            refy_1, refidx_1 = torch.min(cpu_x, dim=1)
            self.assertEqual(y_1, refy_1)
            self.assertEqual(idx_1, refidx_1)

            y_1 = torch.ones(n, h, w, device='mps', dtype=torch.float)
            idx_1 = torch.ones(n, h, w, device='mps', dtype=torch.int64)
            torch.min(x, dim=1, out=(y_1, idx_1))
            refy_1, refidx_1 = torch.min(cpu_x, dim=1)
            self.assertEqual(y_1, refy_1)
            self.assertEqual(idx_1, refidx_1)

            y_1dim, idx_1dim = torch.min(x, dim=1, keepdim=True)
            refy_1dim, refidx_1dim = torch.min(cpu_x, dim=1, keepdim=True)
            self.assertEqual(y_1dim, refy_1dim)
            self.assertEqual(idx_1dim, refidx_1dim)

            y_1dim = torch.ones(n, 1, h, w, device='mps', dtype=torch.float)
            idx_1dim = torch.ones(n, 1, h, w, device='mps', dtype=torch.int64)
            torch.min(x, dim=1, keepdim=True, out=(y_1dim, idx_1dim))
            refy_1dim, refidx_1dim = torch.min(cpu_x, keepdim=True, dim=1)
            self.assertEqual(y_1dim, refy_1dim)
            self.assertEqual(idx_1dim, refidx_1dim)

            y_2, idx_2 = torch.min(x, dim=2)
            refy_2, refidx_2 = torch.min(cpu_x, dim=2)
            self.assertEqual(y_2, refy_2)
            self.assertEqual(idx_2, refidx_2)

            y_2 = torch.ones(n, c, w, device='mps', dtype=torch.float)
            idx_2 = torch.ones(n, c, w, device='mps', dtype=torch.int64)
            torch.min(x, dim=2, out=(y_2, idx_2))
            refy_2, refidx_2 = torch.min(cpu_x, dim=2)
            self.assertEqual(y_2, refy_2)
            self.assertEqual(idx_2, refidx_2)

            y_2dim, idx_2dim = torch.min(x, dim=2, keepdim=True)
            refy_2dim, refidx_2dim = torch.min(cpu_x, dim=2, keepdim=True)
            self.assertEqual(y_2dim, refy_2dim)
            self.assertEqual(idx_2dim, refidx_2dim)

            y_2dim = torch.ones(n, c, 1, w, device='mps', dtype=torch.float)
            idx_2dim = torch.ones(n, c, 1, w, device='mps', dtype=torch.int64)
            torch.min(x, dim=2, keepdim=True, out=(y_2dim, idx_2dim))
            refy_2dim, refidx_2dim = torch.min(cpu_x, dim=2, keepdim=True,)
            self.assertEqual(y_2dim, refy_2dim)
            self.assertEqual(idx_2dim, refidx_2dim)

            y_3, idx_3 = torch.min(x, dim=3)
            refy_3, refidx_3 = torch.min(cpu_x, dim=3)
            self.assertEqual(y_3, refy_3)
            self.assertEqual(idx_3, refidx_3)

            y_3 = torch.ones(n, c, h, device='mps', dtype=torch.float)
            idx_3 = torch.ones(n, c, h, device='mps', dtype=torch.int64)
            torch.min(x, dim=3, out=(y_3, idx_3))
            refy_3, refidx_3 = torch.min(cpu_x, dim=3)
            self.assertEqual(y_3, refy_3)
            self.assertEqual(idx_3, refidx_3)

            y_3dim, idx_3dim = torch.min(x, dim=3, keepdim=True)
            refy_3dim, refidx_3dim = torch.min(cpu_x, dim=3, keepdim=True)
            self.assertEqual(y_3dim, refy_3dim)
            self.assertEqual(idx_3dim, refidx_3dim)

            y_3dim = torch.ones(n, c, h, 1, device='mps', dtype=torch.float)
            idx_3dim = torch.ones(n, c, h, 1, device='mps', dtype=torch.int64)
            torch.min(x, dim=3, keepdim=True, out=(y_3dim, idx_3dim))
            refy_3dim, refidx_3dim = torch.min(cpu_x, dim=3, keepdim=True,)
            self.assertEqual(y_3dim, refy_3dim)
            self.assertEqual(idx_3dim, refidx_3dim)

        helper(2, 8, 4, 5)

    def test_fmin(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/143933
        scalar = torch.tensor(.5)
        x_mps = torch.rand(32, device="mps")
        x_cpu = x_mps.detach().cpu()
        self.assertEqual(torch.fmin(x_mps, scalar), torch.fmin(x_cpu, scalar))

    # Test forward sum
    def test_sum(self):
        def helper(n, c, h, w, dtype=torch.float32):
            cpu_x = None
            x = None
            if (dtype not in [torch.float32, torch.bool]):
                cpu_x = torch.randint(50, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            elif (dtype == torch.bool):
                cpu_x = torch.randint(2, (n, c, h, w), device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            else:
                cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()

            all_sum = torch.sum(x)
            all_sum_cpu = torch.sum(cpu_x)

            self.assertEqual(all_sum, all_sum_cpu)

            nil_dim_sum = torch.sum(x, dim=[])
            nil_dim_sum_cpu = torch.sum(cpu_x, dim=[])

            self.assertEqual(nil_dim_sum, nil_dim_sum_cpu)

            nil_dim_sum_keepdim = torch.sum(x, dim=[], keepdim=True)
            nil_dim_sum_cpu_keepdim = torch.sum(cpu_x, dim=[], keepdim=True)

            self.assertEqual(nil_dim_sum_keepdim, nil_dim_sum_cpu_keepdim)

            zero_dim_sum = torch.sum(x, dim=[0])
            zero_dim_sum_cpu = torch.sum(cpu_x, dim=[0])

            self.assertEqual(zero_dim_sum, zero_dim_sum_cpu)

            zero_dim_sum_keepdim = torch.sum(x, dim=[0], keepdim=True)
            zero_dim_sum_cpu_keepdim = torch.sum(cpu_x, dim=[0], keepdim=True)

            self.assertEqual(zero_dim_sum_keepdim, zero_dim_sum_cpu_keepdim)

            zero_one_dim_sum = torch.sum(x, dim=[0, 1])
            zero_one_dim_sum_cpu = torch.sum(cpu_x, dim=[0, 1])

            self.assertEqual(zero_one_dim_sum, zero_one_dim_sum_cpu)

            zero_one_dim_sum_keepdim = torch.sum(x, dim=[0, 1], keepdim=True)
            zero_one_dim_sum_cpu_keepdim = torch.sum(cpu_x, dim=[0, 1], keepdim=True)

            self.assertEqual(zero_one_dim_sum_keepdim, zero_one_dim_sum_cpu_keepdim)

            two_three_dim_sum = torch.sum(x, dim=[2, 3])
            two_three_dim_sum_cpu = torch.sum(cpu_x, dim=[2, 3])

            self.assertEqual(two_three_dim_sum, two_three_dim_sum_cpu)

            two_three_keepdim_sum = torch.sum(x, dim=[2, 3], keepdim=True)
            two_three_dim_keepsum_cpu = torch.sum(cpu_x, dim=[2, 3], keepdim=True)

            self.assertEqual(two_three_keepdim_sum, two_three_dim_keepsum_cpu)

        helper(2, 8, 4, 5)
        helper(2, 8, 4, 5, dtype=torch.int32)
        helper(2, 8, 4, 5, dtype=torch.int64)
        helper(2, 8, 4, 5, dtype=torch.bool)
        # Regression test for https://github.com/pytorch/pytorch/issues/136132
        x = torch.ones(2, 4, 1, 30, 1, device='mps').sum(dim=-2)
        self.assertEqual(x.numel(), 8)
        self.assertEqual(x.max().item(), 30.0)

    # Test forward prod
    def test_prod(self):
        def helper(shape, dtype=torch.float32):
            cpu_x = None
            x = None
            if (dtype not in [torch.float32, torch.bool]):
                cpu_x = torch.randint(1, 6, shape, device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            elif (dtype == torch.bool):
                cpu_x = torch.randint(2, shape, device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
            else:
                cpu_x = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()

            all_prod = torch.prod(x)
            all_prod_cpu = torch.prod(cpu_x)

            self.assertEqual(all_prod, all_prod_cpu)

            for dim in range(len(shape)):
                dim_prod = torch.prod(x, dim=dim)
                dim_prod_cpu = torch.prod(cpu_x, dim=dim)

                self.assertEqual(dim_prod, dim_prod_cpu)

                dim_prod_keepdim = torch.prod(x, dim=dim, keepdim=True)
                dim_prod_cpu_keepdim = torch.prod(cpu_x, dim=dim, keepdim=True)

                self.assertEqual(dim_prod_keepdim, dim_prod_cpu_keepdim)

        for dtype in [torch.float32, torch.int32, torch.int64, torch.bool]:
            helper((2, 3), dtype)

    # Test forward mean
    def test_mean(self):
        def helper(n, c, h, w):
            cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            all_mean = torch.mean(x)
            all_mean_cpu = torch.mean(cpu_x)

            self.assertEqual(all_mean, all_mean_cpu)

            nil_dim_mean = torch.mean(x, dim=[])
            nil_dim_mean_cpu = torch.mean(cpu_x, dim=[])

            self.assertEqual(nil_dim_mean, nil_dim_mean_cpu)

            nil_dim_mean_keepdim = torch.mean(x, dim=[], keepdim=True)
            nil_dim_mean_cpu_keepdim = torch.mean(cpu_x, dim=[], keepdim=True)

            self.assertEqual(nil_dim_mean_keepdim, nil_dim_mean_cpu_keepdim)

            zero_dim_mean = torch.mean(x, dim=[0])
            zero_dim_mean_cpu = torch.mean(cpu_x, dim=[0])

            self.assertEqual(zero_dim_mean, zero_dim_mean_cpu)

            zero_dim_mean_keepdim = torch.mean(x, dim=[0], keepdim=True)
            zero_dim_mean_cpu_keepdim = torch.mean(cpu_x, dim=[0], keepdim=True)

            self.assertEqual(zero_dim_mean_keepdim, zero_dim_mean_cpu_keepdim)

            zero_one_dim_mean = torch.mean(x, dim=[0, 1])
            zero_one_dim_mean_cpu = torch.mean(cpu_x, dim=[0, 1])

            self.assertEqual(zero_one_dim_mean, zero_one_dim_mean_cpu)

            zero_one_dim_mean_keepdim = torch.mean(x, dim=[0, 1], keepdim=True)
            zero_one_dim_mean_cpu_keepdim = torch.mean(cpu_x, dim=[0, 1], keepdim=True)

            self.assertEqual(zero_one_dim_mean_keepdim, zero_one_dim_mean_cpu_keepdim)

            two_three_dim_mean = torch.mean(x, dim=[2, 3])
            two_three_dim_mean_cpu = torch.mean(cpu_x, dim=[2, 3])

            self.assertEqual(two_three_dim_mean, two_three_dim_mean_cpu)

            two_three_keepdim_mean = torch.mean(x, dim=[2, 3], keepdim=True)
            two_three_dim_keepmean_cpu = torch.mean(cpu_x, dim=[2, 3], keepdim=True)

            self.assertEqual(two_three_keepdim_mean, two_three_dim_keepmean_cpu)

        helper(2, 8, 4, 5)

    # Test std
    def test_std(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            all_std = torch.std(x, unbiased=False)
            all_std_cpu = torch.std(cpu_x, unbiased=False)

            self.assertEqual(all_std, all_std_cpu)

            nil_dim_std = torch.std(x, dim=[], unbiased=False)
            nil_dim_std_cpu = torch.std(cpu_x, dim=[], unbiased=False)

            self.assertEqual(nil_dim_std, nil_dim_std_cpu)

            nil_dim_std_keepdim = torch.std(x, dim=[], keepdim=True, unbiased=False)
            nil_dim_std_cpu_keepdim = torch.std(cpu_x, dim=[], keepdim=True, unbiased=False)

            self.assertEqual(nil_dim_std_keepdim, nil_dim_std_cpu_keepdim)

            zero_dim_std = torch.std(x, dim=[0], unbiased=False)
            zero_dim_std_cpu = torch.std(cpu_x, dim=[0], unbiased=False)

            self.assertEqual(zero_dim_std, zero_dim_std_cpu)

            zero_dim_std_keepdim = torch.std(x, dim=[0], keepdim=True, unbiased=False)
            zero_dim_std_cpu_keepdim = torch.std(cpu_x, dim=[0], keepdim=True, unbiased=False)

            self.assertEqual(zero_dim_std_keepdim, zero_dim_std_cpu_keepdim)

            zero_one_dim_std = torch.std(x, dim=[0, 1], unbiased=False)
            zero_one_dim_std_cpu = torch.std(cpu_x, dim=[0, 1], unbiased=False)

            self.assertEqual(zero_one_dim_std, zero_one_dim_std_cpu)

            zero_one_dim_std_keepdim = torch.std(x, dim=[0, 1], keepdim=True, unbiased=False)
            zero_one_dim_std_cpu_keepdim = torch.std(cpu_x, dim=[0, 1], keepdim=True, unbiased=False)

            self.assertEqual(zero_one_dim_std_keepdim, zero_one_dim_std_cpu_keepdim)

            two_three_dim_std = torch.std(x, dim=[2, 3], unbiased=False)
            two_three_dim_std_cpu = torch.std(cpu_x, dim=[2, 3], unbiased=False)

            self.assertEqual(two_three_dim_std, two_three_dim_std_cpu)

            two_three_keepdim_std = torch.std(x, dim=[2, 3], keepdim=True, unbiased=False)
            two_three_dim_keepstd_cpu = torch.std(cpu_x, dim=[2, 3], keepdim=True, unbiased=False)

            self.assertEqual(two_three_keepdim_std, two_three_dim_keepstd_cpu)

            all_std = torch.std(x, unbiased=True)
            all_std_cpu = torch.std(cpu_x, unbiased=True)

            self.assertEqual(all_std, all_std_cpu)

            nil_dim_std = torch.std(x, dim=[], unbiased=True)
            nil_dim_std_cpu = torch.std(cpu_x, dim=[], unbiased=True)

            self.assertEqual(nil_dim_std, nil_dim_std_cpu)

            nil_dim_std_keepdim = torch.std(x, dim=[], keepdim=True, unbiased=True)
            nil_dim_std_cpu_keepdim = torch.std(cpu_x, dim=[], keepdim=True, unbiased=True)

            self.assertEqual(nil_dim_std_keepdim, nil_dim_std_cpu_keepdim)

            zero_dim_std = torch.std(x, dim=[0], unbiased=True)
            zero_dim_std_cpu = torch.std(cpu_x, dim=[0], unbiased=True)

            self.assertEqual(zero_dim_std, zero_dim_std_cpu)

            zero_dim_std_keepdim = torch.std(x, dim=[0], keepdim=True, unbiased=True)
            zero_dim_std_cpu_keepdim = torch.std(cpu_x, dim=[0], keepdim=True, unbiased=True)

            self.assertEqual(zero_dim_std_keepdim, zero_dim_std_cpu_keepdim)

            zero_one_dim_std = torch.std(x, dim=[0, 1], unbiased=True)
            zero_one_dim_std_cpu = torch.std(cpu_x, dim=[0, 1], unbiased=True)

            self.assertEqual(zero_one_dim_std, zero_one_dim_std_cpu)

            zero_one_dim_std_keepdim = torch.std(x, dim=[0, 1], keepdim=True, unbiased=True)
            zero_one_dim_std_cpu_keepdim = torch.std(cpu_x, dim=[0, 1], keepdim=True, unbiased=True)

            self.assertEqual(zero_one_dim_std_keepdim, zero_one_dim_std_cpu_keepdim)

            two_three_dim_std = torch.std(x, dim=[2, 3], unbiased=True)
            two_three_dim_std_cpu = torch.std(cpu_x, dim=[2, 3], unbiased=True)

            self.assertEqual(two_three_dim_std, two_three_dim_std_cpu)

            two_three_keepdim_std = torch.std(x, dim=[2, 3], keepdim=True, unbiased=True)
            two_three_dim_keepstd_cpu = torch.std(cpu_x, dim=[2, 3], keepdim=True, unbiased=True)

            self.assertEqual(two_three_keepdim_std, two_three_dim_keepstd_cpu)

        helper((4, 5, 6, 7))
        # verify if a change in shape of input would cause problems with graph caching
        helper((9, 5, 6, 7))

    # Test var
    def test_var_simple(self):
        def helper():

            shape = [2, 3, 4, 5]

            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            for unbiased in [False, True]:
                for keepdim in [False, True]:

                    zero_dim_var = x.var(-1, keepdim=keepdim, unbiased=unbiased)
                    zero_dim_var_cpu = cpu_x.var(-1, keepdim=keepdim, unbiased=unbiased)

                    self.assertEqual(zero_dim_var, zero_dim_var_cpu)

                    all_var = torch.var(x, unbiased=unbiased)
                    all_var_cpu = torch.var(cpu_x, unbiased=unbiased)

                    self.assertEqual(all_var, all_var_cpu)

                    nil_dim_var = torch.var(x, dim=[], keepdim=keepdim, unbiased=unbiased)
                    nil_dim_var_cpu = torch.var(cpu_x, dim=[], keepdim=keepdim, unbiased=unbiased)

                    self.assertEqual(nil_dim_var, nil_dim_var_cpu)

                    zero_dim_var = torch.var(x, dim=[0], keepdim=keepdim, unbiased=unbiased)
                    zero_dim_var_cpu = torch.var(cpu_x, dim=[0], keepdim=keepdim, unbiased=unbiased)

                    self.assertEqual(zero_dim_var, zero_dim_var_cpu)

                    zero_one_dim_var = torch.var(x, dim=[0, -1], keepdim=keepdim, unbiased=unbiased)
                    zero_one_dim_var_cpu = torch.var(cpu_x, dim=[0, -1], keepdim=keepdim, unbiased=unbiased)

                    self.assertEqual(zero_one_dim_var, zero_one_dim_var_cpu)

                    two_three_dim_var = torch.var(x, dim=[2, 3], keepdim=keepdim, unbiased=unbiased)
                    two_three_dim_var_cpu = torch.var(cpu_x, dim=[2, 3], keepdim=keepdim, unbiased=unbiased)

                    self.assertEqual(two_three_dim_var, two_three_dim_var_cpu)

        helper()

    # Test forward amax
    def test_amax(self):
        def helper(shape, dim, keepdim):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            result = torch.amax(x, dim=dim, keepdim=keepdim)
            result_cpu = torch.amax(cpu_x, dim=dim, keepdim=keepdim)

            cpu_grad = torch.randn(result_cpu.shape)
            grad = cpu_grad.to('mps')

            result_cpu.backward(gradient=cpu_grad)
            result.backward(gradient=grad)

            self.assertEqual(result, result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        for dim in ([], [0], [0, 1], [2, 3]):
            for keepdim in [False, True]:
                helper((2, 8, 4, 5), dim, keepdim)

    # Test forward amin
    def test_amin(self):
        def helper(shape, dim, keepdim):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            result = torch.amin(x, dim=dim, keepdim=keepdim)
            result_cpu = torch.amin(cpu_x, dim=dim, keepdim=keepdim)

            cpu_grad = torch.randn(result_cpu.shape)
            grad = cpu_grad.to('mps')

            result_cpu.backward(gradient=cpu_grad)
            result.backward(gradient=grad)

            self.assertEqual(result, result_cpu)
            self.assertEqual(x.grad, cpu_x.grad)

        for dim in ([], [0], [0, 1], [2, 3]):
            for keepdim in [False, True]:
                helper((2, 8, 4, 5), dim, keepdim)

    # Test minimum and maximum
    def test_minimum_maximum(self):
        def helper(n, c, h, w):
            cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            cpu_y = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            mps_x = cpu_x.detach().clone().to('mps')
            mps_y = cpu_y.detach().clone().to('mps')

            minimum_result_cpu = torch.minimum(cpu_x, cpu_y)
            minimum_result_mps = torch.minimum(mps_x, mps_y)
            self.assertEqual(minimum_result_cpu, minimum_result_mps)

            maximum_result_cpu = torch.maximum(cpu_x, cpu_y)
            maximum_result_mps = torch.maximum(mps_x, mps_y)
            self.assertEqual(maximum_result_cpu, maximum_result_mps)

        helper(1, 1, 4, 5)

    def test_minimum_maximum_nan_propagation(self):
        x = torch.rand(32, device="mps")
        y = torch.rand(32, device="mps")
        x[3] = torch.nan
        y[5] = torch.nan
        self.assertTrue(torch.minimum(x, y).isnan().any().item())
        self.assertTrue(torch.maximum(x, y).isnan().any().item())

    def test_clamp_fp16_fp32(self):
        cpu_x = torch.randn(10, device='cpu', dtype=torch.float, requires_grad=False)
        x = cpu_x.detach().clone().to('mps')

        dtype = torch.float16

        clamp_min_vals_mps = torch.ones(10, device="mps").to(torch.float16)
        clamp_max_vals_mps = torch.ones(10, device="mps").to(torch.float16) * 10
        clamp_result_mps = torch.clamp(x, clamp_min_vals_mps, clamp_max_vals_mps)

        clamp_min_vals_cpu = torch.ones(10, device="cpu").to(torch.float16)
        clamp_max_vals_cpu = torch.ones(10, device="cpu").to(torch.float16) * 10
        clamp_result_cpu = torch.clamp(cpu_x, clamp_min_vals_cpu, clamp_max_vals_cpu)

        self.assertEqual(clamp_result_mps, clamp_result_cpu)

    def test_clamp_nan(self):
        t_mps = torch.tensor([torch.nan, 1, 2], device="mps")
        t_cpu = torch.tensor([torch.nan, 1, 2], device="cpu")

        clamp_min_max_mps = torch.clamp(t_mps, min=-100, max=100)
        clamp_min_max_cpu = torch.clamp(t_cpu, min=-100, max=100)

        self.assertEqual(clamp_min_max_mps, clamp_min_max_cpu)

        clamp_min_mps = torch.clamp(t_mps, min=-100)
        clamp_min_cpu = torch.clamp(t_cpu, min=-100)

        self.assertEqual(clamp_min_mps, clamp_min_cpu)

        clamp_max_mps = torch.clamp(t_mps, max=100)
        clamp_max_cpu = torch.clamp(t_cpu, max=100)

        self.assertEqual(clamp_max_mps, clamp_max_cpu)

    # Test clamp_min
    def test_clamp_min(self):
        def helper(n, c, h, w):
            cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_min_t = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            min_t = cpu_min_t.detach().clone().to('mps')

            clamp_min_result = torch.clamp_min(x, min=5.0)
            clamp_min_result_cpu = torch.clamp_min(cpu_x, min=5.0)

            self.assertEqual(clamp_min_result, clamp_min_result_cpu)

            clamp_min_t_result = torch.clamp_min(x, min=min_t)
            clamp_min_t_result_cpu = torch.clamp_min(cpu_x, min=cpu_min_t)

            self.assertEqual(clamp_min_t_result, clamp_min_t_result_cpu)

        helper(2, 8, 4, 5)

    # Test clamp_max

    def test_clamp_max(self):
        def helper(n, c, h, w):
            cpu_x = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_max_t = torch.randn(n, c, h, w, device='cpu', dtype=torch.float, requires_grad=False)
            max_t = cpu_max_t.detach().clone().to('mps')

            clamp_max_result = torch.clamp_max(x, max=100.0)
            clamp_max_result_cpu = torch.clamp_max(cpu_x, max=100.0)

            self.assertEqual(clamp_max_result, clamp_max_result_cpu)

            clamp_max_t_result = torch.clamp_max(x, max=max_t)
            clamp_max_t_result_cpu = torch.clamp_max(cpu_x, max=cpu_max_t)

            self.assertEqual(clamp_max_t_result, clamp_max_t_result_cpu)

        helper(2, 8, 4, 5)

    # Test clamp
    def test_clamp(self):
        def helper(n, c, h, w):
            import numpy as np
            upper_bound = 1000
            half_upper_bound = upper_bound / 2

            # x=[0..1000)
            x_arr = upper_bound * np.random.random_sample(size=(n, c, h, w)).astype(np.float32)
            cpu_x = torch.tensor(x_arr, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            # x=[0..500)
            min_arr = half_upper_bound * np.random.random_sample(size=(n, c, h, w)).astype(np.float32)
            cpu_min_t = torch.tensor(min_arr, device='cpu', dtype=torch.float, requires_grad=False)
            min_t = cpu_min_t.detach().clone().to('mps')

            # x=[500..1000), to ensure max's are greater than mins
            max_arr = (half_upper_bound * np.random.random_sample(size=(n, c, h, w)).astype(np.float32)) + half_upper_bound
            cpu_max_t = torch.tensor(max_arr, device='cpu', dtype=torch.float, requires_grad=False)
            max_t = cpu_max_t.detach().clone().to('mps')

            # [200..600]: just an arbitrary range between [0..1000]
            clamp_result = torch.clamp(x, min=200.0, max=600.0)
            clamp_result_cpu = torch.clamp(cpu_x, min=200.0, max=600.0)
            self.assertEqual(clamp_result, clamp_result_cpu)

            # test optional scalar refs and cached graph keys by passing only max
            clamp_opt_result = torch.clamp(x, max=600.0)
            clamp_opt_result_cpu = torch.clamp(cpu_x, max=600.0)
            self.assertEqual(clamp_opt_result, clamp_opt_result_cpu)

            clamp_t_result = torch.clamp(x, min=min_t, max=max_t)
            clamp_t_result_cpu = torch.clamp(cpu_x, min=cpu_min_t, max=cpu_max_t)
            self.assertEqual(clamp_t_result, clamp_t_result_cpu)

            # test optional tensor refs and cached graph keys by passing only max
            clamp_topt_result = torch.clamp(x, max=max_t)
            clamp_topt_result_cpu = torch.clamp(cpu_x, max=cpu_max_t)
            self.assertEqual(clamp_topt_result, clamp_topt_result_cpu)

            # test strided x
            clamp_result = torch.clamp(x.movedim(0, -1), min=200.0, max=600.0)
            clamp_result_cpu = torch.clamp(cpu_x.movedim(0, -1), min=200.0, max=600.0)
            self.assertEqual(clamp_result, clamp_result_cpu)

            # test strided x, min_t, max_t
            clamp_result = torch.clamp(x.movedim(0, -1), min=min_t.movedim(0, -1), max=max_t.movedim(0, -1))
            clamp_result_cpu = torch.clamp(cpu_x.movedim(0, -1), min=cpu_min_t.movedim(0, -1), max=cpu_max_t.movedim(0, -1))
            self.assertEqual(clamp_result, clamp_result_cpu)

            # test strided min_t, max_t
            clamp_result = torch.clamp(
                x.movedim(0, -1).clone(memory_format=torch.contiguous_format),
                min=min_t.movedim(0, -1),
                max=max_t.movedim(0, -1)
            )
            clamp_result_cpu = torch.clamp(
                cpu_x.movedim(0, -1).clone(memory_format=torch.contiguous_format),
                min=cpu_min_t.movedim(0, -1),
                max=cpu_max_t.movedim(0, -1)
            )
            self.assertEqual(clamp_result, clamp_result_cpu)

            # test inplace clamping
            x.clamp_(min=200.0, max=600.0)
            cpu_x.clamp_(min=200.0, max=600.0)
            self.assertEqual(cpu_x, x)

        helper(2, 8, 4, 5)

    def test_divmode(self):
        def helper(shape, rounding_mode):
            for dtype in [torch.float32, torch.float16, torch.int32, torch.int64]:
                if ((rounding_mode is not None and "floor" in rounding_mode and dtype == torch.int64) or
                        (rounding_mode is not None and "trunc" in rounding_mode and dtype == torch.float16)) is False:
                    cpu_x = None
                    cpu_y = None
                    if (dtype in [torch.float32, torch.float16]):
                        cpu_x = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=False)
                        cpu_y = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=False)
                    else:
                        cpu_x = torch.randint(-10, 0, shape, device='cpu', dtype=dtype, requires_grad=False)
                        cpu_y = torch.randint(-10, 0, shape, device='cpu', dtype=dtype, requires_grad=False)

                    mps_x = cpu_x.detach().clone().to('mps')
                    # clamp to avoid division by 0
                    mps_y = cpu_y.detach().clone().to('mps')

                    if (rounding_mode == "floor_divide"):
                        result_div_cpu = torch.floor_divide(cpu_x, cpu_y)
                        result_div_mps = torch.floor_divide(mps_x, mps_y)
                        self.assertEqual(result_div_mps, result_div_cpu)
                    else:
                        result_div_cpu = torch.div(cpu_x, cpu_y, rounding_mode=rounding_mode)
                        result_div_mps = torch.div(mps_x, mps_y, rounding_mode=rounding_mode)
                        self.assertEqual(result_div_mps, result_div_cpu)

        helper((2, 8, 4, 5), None)
        helper((2, 8, 4, 5), "floor")
        helper((2, 8, 4, 5), "trunc")
        helper((2, 8, 4, 5), "floor_divide")

    def test_rounding(self):
        def helper(shape):
            cpu_x = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=False)
            mps_x = cpu_x.detach().clone().to('mps')

            result_floor_cpu = torch.floor(cpu_x)
            result_floor_mps = torch.floor(mps_x)
            self.assertEqual(result_floor_mps, result_floor_cpu)

            result_ceil_cpu = torch.ceil(cpu_x)
            result_ceil_mps = torch.ceil(mps_x)
            self.assertEqual(result_ceil_mps, result_ceil_cpu)

            result_trunc_cpu = torch.trunc(cpu_x)
            result_trunc_mps = torch.trunc(mps_x)
            self.assertEqual(result_trunc_mps, result_trunc_cpu)

            result_round_cpu = torch.round(cpu_x)
            result_round_mps = torch.round(mps_x)
            self.assertEqual(result_round_mps, result_round_cpu)

        helper((2, 6, 3, 5))
        helper((2, 8, 4, 5))

    def test_remainder(self):
        res_cpu = torch.remainder(
            torch.tensor([-3, -2, -1, 1, 2, 3], dtype=torch.int32, device="cpu"), torch.tensor(2, device="cpu", dtype=torch.int32))
        res_mps = torch.remainder(
            torch.tensor([-3, -2, -1, 1, 2, 3], dtype=torch.int32, device="mps"), torch.tensor(2, device="mps", dtype=torch.int32))
        self.assertEqual(res_cpu, res_mps)

        res_cpu = torch.remainder(
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device="cpu"), -1.5)
        res_mps = torch.remainder(
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device="mps"), -1.5)
        self.assertEqual(res_cpu, res_mps)

        # Regression test for https://github.com/pytorch/pytorch/issues/154171
        # Essentially remained over integral types should rely on integers ops
        self.assertEqual(torch.tensor(42309891, device='mps') % torch.tensor(31, device='mps'), torch.tensor(6, device='mps'))

    def test_expand(self):
        def helper(n, c):
            values = [[1.0], [4.0], [7.0]]
            cpu_x = torch.tensor(values, device='cpu')
            x = cpu_x.detach().clone().to('mps')

            strided_cpu = torch.as_strided(cpu_x, (3, 4), (1, 0))
            strided_mps = torch.as_strided(x, (3, 4), (1, 0))

            self.assertEqual(strided_mps, strided_cpu)

        helper(3, 1)

    def test_im2col(self):
        def helper(x):
            return torch.nn.functional.unfold(x, kernel_size=(10, 15), dilation=2, padding=5, stride=3)
        x_cpu = torch.rand(1, 1, 200, 100)
        x = x_cpu.detach().clone().to('mps')
        self.assertEqual(helper(x_cpu), helper(x))

    def test_col2im(self):
        def helper(shapes, output_size, kernel_size, padding, stride, contiguous, dtype=torch.float32, test_bool=False):
            atol = 1e-5 if dtype == torch.float else 1e-2
            rtol = 1e-3 if dtype == torch.float else 1e-2
            x_cpu = torch.rand(*shapes, dtype=dtype)
            if test_bool:
                x_cpu = x_cpu > 0.5
            x_mps = x_cpu.clone().to('mps')
            if not contiguous:
                x_cpu = x_cpu.mT
                x_mps = x_mps.mT
            out_cpu = torch.nn.functional.fold(
                x_cpu,
                output_size=output_size,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride
            )
            out_mps = torch.nn.functional.fold(
                x_mps,
                output_size=output_size,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride
            )
            self.assertEqual(out_cpu, out_mps, atol=atol, rtol=rtol)

        helper((4, 27, 1600), (40, 40), 3, 1, 1, True)
        helper((1, 27, 1600), (40, 40), 3, 1, 1, True)
        helper((27, 1600), (40, 40), 3, 1, 1, True)
        helper((27, 320), (80, 4), 3, 1, 1, True)
        helper((27, 320), (4, 80), 3, 1, 1, True)
        helper((320, 27), (4, 80), 3, 1, 1, False)
        helper((4, 75, 1600), (40, 40), 5, 2, 1, True)
        helper((4, 75, 441), (41, 41), 5, 2, 2, True)
        helper((4, 12, 100), (20, 20), 2, 0, 2, True)
        helper((4, 48, 225), (30, 30), 4, 1, 2, True)
        helper((100, 75), (20, 20), 5, 2, 2, False)
        helper((4, 15, 1600), (40, 40), (3, 5), (1, 2), (1, 1), True)
        helper((4, 45, 187), (35, 33), (3, 5), (0, 1), (2, 3), True)
        helper((1600, 15), (40, 40), (3, 5), (1, 2), (1, 1), False)
        if MACOS_VERSION >= 14.0:
            helper((20, 15), (2, 10), (3, 5), (1, 2), (1, 1), False, torch.bfloat16)
        helper((20, 15), (2, 10), (3, 5), (1, 2), (1, 1), False, torch.float16)
        helper((20, 15), (2, 10), (3, 5), (1, 2), (1, 1), False, test_bool=True)

    def test_select(self):
        def helper(n, c):
            cpu_x = torch.randn(n, c, device='cpu', dtype=torch.float, requires_grad=True)
            x = cpu_x.detach().clone().to('mps').requires_grad_()

            strided_cpu = torch.as_strided(cpu_x, (3, 1), (3, 1))
            strided_mps = torch.as_strided(x, (3, 1), (3, 1))
            self.assertEqual(strided_mps, strided_cpu)

            strided_cpu = torch.as_strided(cpu_x, (1, 3), (3, 1))
            strided_mps = torch.as_strided(x, (1, 3), (3, 1))
            self.assertEqual(strided_mps, strided_cpu)

            strided_cpu = torch.as_strided(cpu_x, (3, 1), (3, 1), storage_offset=1)
            strided_mps = torch.as_strided(x, (3, 1), (3, 1), storage_offset=1)

            self.assertEqual(strided_mps, strided_cpu)

        helper(3, 3)

    def test_sort(self):
        for SIZE in (4, 2049):
            device = 'mps'
            x = torch.rand(4, SIZE, device=device)
            res1val, res1ind = torch.sort(x)

            res2val = torch.tensor((), device=device)
            res2ind = torch.tensor((), device=device, dtype=torch.long)
            torch.sort(x, out=(res2val, res2ind))
            self.assertEqual(res1val, res2val, atol=0, rtol=0)
            self.assertEqual(res1ind, res2ind, atol=0, rtol=0)
            self.assertEqual(torch.argsort(x), res1ind)
            self.assertEqual(x.argsort(), res1ind)

            self.assertEqual(
                torch.sort(torch.tensor((50, 40, 30, 20, 10), device=device))[0],
                torch.tensor((10, 20, 30, 40, 50), device=device),
                atol=0, rtol=0
            )

    def test_linalg_cholesky(self):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def run_cholesky_test(size, *batch_dims, upper=False, check_errors=False):
            if check_errors:
                # expect failure for non-positive definite matrix
                input_mps = torch.eye(size, dtype=torch.float32, device="mps")
                input_mps[0, 0] = -1
                error_msg = r'The factorization could not be completed because the input is not positive-definite'
                with self.assertRaisesRegex(RuntimeError, error_msg):
                    torch.linalg.cholesky_ex(input_mps, upper=upper, check_errors=check_errors)
                return
            # output checks for positive definite matrix
            input_cpu = random_hermitian_pd_matrix(size, *batch_dims, dtype=torch.float32, device="cpu")
            input_mps = input_cpu.to('mps')
            output_cpu = torch.linalg.cholesky_ex(input_cpu, upper=upper)
            output_mps = torch.linalg.cholesky_ex(input_mps, upper=upper)
            self.assertEqual(output_cpu, output_mps, atol=2e-5, rtol=1e-6)

        # test with different even/odd matrix sizes
        matrix_sizes = [1, 2, 3, 4, 8, 17, 64, 128, 154]
        # even/odd batch sizes
        batch_sizes = [1, 2, 4, 8, 16, 17]

        for upper in [True, False]:
            for size in matrix_sizes:
                for batch_size in batch_sizes:
                    run_cholesky_test(size, batch_size, upper=upper)

        # test >3D matrices
        run_cholesky_test(128, 10, 10, upper=False)
        run_cholesky_test(128, 2, 2, 2, 2, 10, 10, upper=True)
        run_cholesky_test(32, 2, upper=False, check_errors=True)
        run_cholesky_test(32, 2, upper=True, check_errors=True)

    def test_linalg_cholesky_info(self):
        # non psd matrix with leading minor of order 2 being not positive definite
        A = torch.tensor([
            [4.0, 1.0, 0.0],
            [1.0, -2.0, 1.0],
            [0.0, 1.0, 3.0]
        ], device="mps")
        with self.assertRaisesRegex(RuntimeError, r'leading minor of order 2 is not positive-definite'):
            torch.linalg.cholesky_ex(A, check_errors=True)

    def test_upsample_nearest2d(self):
        def helper(N, C, H, W, memory_format):
            inputCPU = torch.arange(N * C * H * W, device='cpu', dtype=torch.float,
                                    requires_grad=True).reshape(N, C, H, W).to(memory_format=memory_format)
            inputCPU.retain_grad()
            inputMPS = inputCPU.detach().to('mps').requires_grad_()

            values = [1, 2, 5, 10, 40]

            for i in values:
                for j in values:
                    upsample_nearest2d = nn.UpsamplingNearest2d(scale_factor=(i, j))

                    outputCPU = upsample_nearest2d(inputCPU)
                    outputMPS = upsample_nearest2d(inputMPS)

                    self.assertEqual(outputCPU, outputMPS)
                    upsample_nearest2d = nn.UpsamplingNearest2d((i * H, j * W))

                    outputCPU = upsample_nearest2d(inputCPU)
                    outputMPS = upsample_nearest2d(inputMPS)

                    self.assertEqual(outputCPU, outputMPS)

                    outputCPU.backward(gradient=torch.full_like(outputCPU, 0.3))
                    outputMPS.backward(gradient=torch.full_like(outputMPS, 0.3))

                    self.assertEqual(inputCPU.grad, inputMPS.grad)

        for memory_format in [torch.channels_last, torch.contiguous_format]:
            helper(1, 1, 4, 4, memory_format=memory_format)
            helper(7, 5, 3, 2, memory_format=memory_format)

    def test_upsample_bilinear2d(self):
        def helper(N, C, H, W):
            inputCPU = torch.arange(N * C * H * W, device='cpu', dtype=torch.float,
                                    requires_grad=True).reshape(N, C, H, W)
            inputCPU.retain_grad()
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()

            values = [1, 2, 5, 10, 40]

            for i in values:
                for j in values:
                    upsample_bilinear2d = nn.UpsamplingBilinear2d(scale_factor=(i, j))

                    outputCPU = upsample_bilinear2d(inputCPU)
                    outputMPS = upsample_bilinear2d(inputMPS)

                    self.assertEqual(outputCPU, outputMPS)

                    upsample_bilinear2d = nn.UpsamplingBilinear2d((i * H, j * W))

                    outputCPU = upsample_bilinear2d(inputCPU)
                    outputMPS = upsample_bilinear2d(inputMPS)

                    self.assertEqual(outputCPU, outputMPS)

                    outputCPU.backward(gradient=torch.full_like(outputCPU, 0.3))
                    outputMPS.backward(gradient=torch.full_like(outputMPS, 0.3))

                    self.assertEqual(inputCPU.grad, inputMPS.grad)

        helper(1, 1, 4, 4)
        helper(7, 5, 3, 2)

    def test_interpolate(self):
        def helper(shape, output_size, scales, mode, align_corners=False):
            inputCPU = torch.randn(shape, device='cpu', dtype=torch.float, requires_grad=True)
            inputCPU.retain_grad()
            inputMPS = inputCPU.detach().clone().to('mps').requires_grad_()

            # align_corners is used for 2D interpolation only
            if (align_corners is True and len(shape) > 3 and mode == 'bilinear'):
                if scales is not None:
                    outputCPU = nn.functional.interpolate(inputCPU, scale_factor=scales, mode=mode, align_corners=align_corners)
                    outputMPS = nn.functional.interpolate(inputMPS, scale_factor=scales, mode=mode, align_corners=align_corners)
                else:
                    outputCPU = nn.functional.interpolate(inputCPU, size=output_size, mode=mode, align_corners=align_corners)
                    outputMPS = nn.functional.interpolate(inputMPS, size=output_size, mode=mode, align_corners=align_corners)
            elif scales is not None:
                outputCPU = nn.functional.interpolate(inputCPU, scale_factor=scales, mode=mode)
                outputMPS = nn.functional.interpolate(inputMPS, scale_factor=scales, mode=mode)
            else:
                outputCPU = nn.functional.interpolate(inputCPU, size=output_size, mode=mode)
                outputMPS = nn.functional.interpolate(inputMPS, size=output_size, mode=mode)

            self.assertEqual(outputCPU, outputMPS)

            # backward pass (chose 0.6 just to have the grad_output != 1)
            outputCPU.backward(gradient=torch.full_like(outputCPU, 0.6))
            outputMPS.backward(gradient=torch.full_like(outputMPS, 0.6))
            self.assertEqual(inputCPU.grad, inputMPS.grad)

        # 1D interpolation
        for mode in ['nearest', 'nearest-exact']:
            helper([2, 3, 4], [3], None, mode)  # downsample with size
            helper([2, 3, 4], [6], None, mode)  # upsample with size
            helper([2, 3, 4], None, [0.6], mode)  # downsample with scale factor
            helper([2, 3, 4], None, [1.7], mode)  # upsample with scale factor
        # 2D interpolation
        for mode in ['nearest', 'nearest-exact', 'bilinear']:
            helper([2, 3, 4, 5], [3, 4], None, mode)  # downsample_nearest with size
            helper([2, 3, 4, 5], [6, 7], None, mode)  # upsample_nearest with size
            helper([2, 3, 4, 5], None, [0.6, 0.7], mode)  # downsample_nearest with scale factor
            helper([2, 3, 4, 5], None, [1.4, 1.7], mode)  # upsample_nearest with scale factor
        # align_corners=True
        helper([2, 3, 4, 5], [3, 4], None, 'bilinear', True)
        helper([2, 3, 4, 5], None, [1.4, 1.7], 'bilinear', True)
        # Regression test for https://github.com/pytorch/pytorch/issues/144245
        inp = torch.tensor([[[1.]], [[2]], [[4]]], device='mps')
        for align_corners in [True, False]:
            def interp(x):
                return F.interpolate(x, 3, mode='linear', align_corners=align_corners)
            self.assertEqual(interp(inp).cpu(), interp(inp.cpu()))

    # Test concat forward
    def test_cat1(self):
        def helper(shape_x, shape_y, shape_z):
            cpu_x = torch.randn(shape_x, device='cpu', dtype=torch.float, requires_grad=False)
            x = cpu_x.detach().clone().to('mps')

            cpu_y = torch.randn(shape_y, device='cpu', dtype=torch.float, requires_grad=False)
            y = cpu_y.detach().clone().to('mps')

            cpu_z = torch.randn(shape_z, device='cpu', dtype=torch.float, requires_grad=False)
            z = cpu_z.detach().clone().to('mps')

            cat = torch.cat([x, y, z], dim=1)
            cat_cpu = torch.cat([cpu_x, cpu_y, cpu_z], dim=1)

            self.assertEqual(cat, cat_cpu)

        helper([2, 2, 4, 5], [2, 3, 4, 5], [2, 5, 4, 5])
        helper([2, 2, 6, 5], [2, 3, 6, 5], [2, 5, 6, 5])
        helper([0, 2, 4, 5], [0, 3, 4, 5], [0, 5, 4, 5])
        helper([2, 2, 6, 5], [0], [2, 5, 6, 5])
        helper([0], [2, 3, 6, 5], [2, 5, 6, 5])
        helper([2, 3, 4, 5], [2, 5, 4, 5], [0])
        helper([2, 2, 6, 5], [2, 0, 6, 5], [2, 5, 6, 5])
        helper([2, 0, 6, 5], [2, 3, 6, 5], [2, 5, 6, 5])
        helper([2, 0, 6, 5], [2, 3, 6, 5], [2, 0, 6, 5])

    # Test stack forward
    def test_stack(self):
        # All shapes must be same
        def helper(shape, dtype=torch.float32):

            x, cpu_x = None, None
            y, cpu_y = None, None
            z, cpu_z = None, None

            if (dtype not in [torch.float32, torch.bool]):
                cpu_x = torch.randint(50, shape, device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
                cpu_y = torch.randint(50, shape, device='cpu', dtype=dtype, requires_grad=False)
                y = cpu_y.detach().clone().to('mps')
                cpu_z = torch.randint(50, shape, device='cpu', dtype=dtype, requires_grad=False)
                z = cpu_z.detach().clone().to('mps')
            elif (dtype == torch.bool):
                cpu_x = torch.randint(2, shape, device='cpu', dtype=dtype, requires_grad=False)
                x = cpu_x.detach().clone().to('mps')
                cpu_y = torch.randint(2, shape, device='cpu', dtype=dtype, requires_grad=False)
                y = cpu_y.detach().clone().to('mps')
                cpu_z = torch.randint(2, shape, device='cpu', dtype=dtype, requires_grad=False)
                z = cpu_z.detach().clone().to('mps')
            else:
                cpu_x = torch.randn(shape, device='cpu', dtype=dtype, requires_grad=True)
                x = cpu_x.detach().clone().to('mps').requires_grad_()
                cpu_y = torch.randn(shape, device='cpu