# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn

device_id = 0
device = ttnn.open_device(device_id=device_id)

torch_input_tensor_a = torch.rand(2048, 2048, dtype=torch.float32)
torch_input_tensor_b = torch.rand(2048, 2048, dtype=torch.float32)
input_tensor_a = ttnn.from_torch(
    torch_input_tensor_a,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
input_tensor_b = ttnn.from_torch(
    torch_input_tensor_b,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)

output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.L1_MEMORY_CONFIG)
ttnn.deallocate(input_tensor_a)
ttnn.deallocate(input_tensor_b)

torch_output_tensor = ttnn.to_torch(output_tensor)
ttnn.deallocate(output_tensor)

ttnn.close_device(device)
