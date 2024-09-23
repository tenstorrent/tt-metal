# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

device_id = 0
device = ttnn.open_device(device_id=device_id)

torch_input_tensor_a = torch.rand(4, 7, dtype=torch.float32)
input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
output_tensor = ttnn.exp(input_tensor_a)
torch_output_tensor = ttnn.to_torch(output_tensor)

torch_input_tensor_b = torch.rand(7, 1, dtype=torch.float32)
input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
matmul_output_tensor = input_tensor_a @ input_tensor_b
torch_matmul_output_tensor = ttnn.to_torch(matmul_output_tensor)

print(torch_matmul_output_tensor)

ttnn.close_device(device)
