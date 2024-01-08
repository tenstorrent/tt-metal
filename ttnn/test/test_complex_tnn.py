# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand_complex
from models.utility_functions import ttl_complex_2_torch_complex


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        # (torch.Size([1, 1, 32, 32])),
        # (torch.Size([1, 1, 320, 384])),
        # (torch.Size([1, 3, 320, 384])),
    ),
)
class TestBinaryComplex:
    def test_ttnn_complex_add(self, input_shapes):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = gen_rand_complex(size=input_shapes, low=-100, high=100)
        torch_b = gen_rand_complex(size=input_shapes, low=-100, high=100)

        tempx = torch.cat([torch_a.real, torch_a.imag], -1).to(torch.bfloat16)
        tempy = torch.cat([torch_b.real, torch_b.imag], -1).to(torch.bfloat16)

        a = ttnn.from_torch(tempx)
        b = ttnn.from_torch(tempy)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output = ttnn.complex_add(a, b)

        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"shape: {output.shape}")
        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")

        ttnn.close(device)

    def test_ttnn_complex_sub(self, input_shapes):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = gen_rand_complex(size=input_shapes, low=-100, high=100)
        torch_b = gen_rand_complex(size=input_shapes, low=-100, high=100)

        tempx = torch.cat([torch_a.real, torch_a.imag], -1).to(torch.bfloat16)
        tempy = torch.cat([torch_b.real, torch_b.imag], -1).to(torch.bfloat16)

        a = ttnn.from_torch(tempx)
        b = ttnn.from_torch(tempy)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output = ttnn.complex_sub(a, b)

        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"shape: {output.shape}")
        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")

        ttnn.close(device)

    def test_ttnn_complex_mul(self, input_shapes):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = gen_rand_complex(size=input_shapes, low=-100, high=100)
        torch_b = gen_rand_complex(size=input_shapes, low=-100, high=100)

        tempx = torch.cat([torch_a.real, torch_a.imag], -1).to(torch.bfloat16)
        tempy = torch.cat([torch_b.real, torch_b.imag], -1).to(torch.bfloat16)

        a = ttnn.from_torch(tempx)
        b = ttnn.from_torch(tempy)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output = ttnn.complex_mul(a, b)

        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"shape: {output.shape}")
        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")

        ttnn.close(device)

    def test_ttnn_complex_div(self, input_shapes):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = gen_rand_complex(size=input_shapes, low=-100, high=100)
        torch_b = gen_rand_complex(size=input_shapes, low=-100, high=100)

        tempx = torch.cat([torch_a.real, torch_a.imag], -1).to(torch.bfloat16)
        tempy = torch.cat([torch_b.real, torch_b.imag], -1).to(torch.bfloat16)

        a = ttnn.from_torch(tempx)
        b = ttnn.from_torch(tempy)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)

        output = ttnn.complex_div(a, b)

        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"shape: {output.shape}")
        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")

        ttnn.close(device)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 64])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
class TestUnaryComplex:
    def test_ttnn_complex_abs(self, input_shapes):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = gen_rand_complex(size=input_shapes, low=-100, high=100)

        tempx = torch.cat([torch_a.real, torch_a.imag], -1).to(torch.bfloat16)

        a = ttnn.from_torch(tempx)

        a = ttnn.to_device(a, device)

        output = ttnn.complex_abs(a)

        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"shape: {output.shape}")
        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")

        ttnn.close(device)
