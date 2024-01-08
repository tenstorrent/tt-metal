# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
class TestEltwiseTernary:
    def test_ttnn_where(self, input_shapes):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = torch.rand(input_shapes, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shapes, dtype=torch.bfloat16)
        torch_c = torch.rand(input_shapes, dtype=torch.bfloat16)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)
        c = ttnn.from_torch(torch_c)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)
        c = ttnn.to_device(c, device)

        output = ttnn.where(a, b, c)
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"shape: {output.shape}")
        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")

        ttnn.close(device)

    def test_ttnn_mac(self, input_shapes):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = torch.rand(input_shapes, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shapes, dtype=torch.bfloat16)
        torch_c = torch.rand(input_shapes, dtype=torch.bfloat16)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)
        c = ttnn.from_torch(torch_c)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)
        c = ttnn.to_device(c, device)

        output = ttnn.mac(a, b, c)
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"shape: {output.shape}")
        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")

        ttnn.close(device)

    @pytest.mark.parametrize("value", [1.0])
    def test_ttnn_addcdiv(self, input_shapes, value):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = torch.rand(input_shapes, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shapes, dtype=torch.bfloat16)
        torch_c = torch.rand(input_shapes, dtype=torch.bfloat16)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)
        c = ttnn.from_torch(torch_c)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)
        c = ttnn.to_device(c, device)

        output = ttnn.addcdiv(a, b, c, value)
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"shape: {output.shape}")
        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")

        ttnn.close(device)

    @pytest.mark.parametrize("value", [1.0])
    def test_ttnn_addcmul(self, input_shapes, value):
        device_id = 0
        device = ttnn.open(device_id)

        torch.manual_seed(0)

        torch_a = torch.rand(input_shapes, dtype=torch.bfloat16)
        torch_b = torch.rand(input_shapes, dtype=torch.bfloat16)
        torch_c = torch.rand(input_shapes, dtype=torch.bfloat16)

        a = ttnn.from_torch(torch_a)
        b = ttnn.from_torch(torch_b)
        c = ttnn.from_torch(torch_c)

        a = ttnn.to_device(a, device)
        b = ttnn.to_device(b, device)
        c = ttnn.to_device(c, device)

        output = ttnn.addcmul(a, b, c, value)
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

        print(f"shape: {output.shape}")
        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")

        ttnn.close(device)
