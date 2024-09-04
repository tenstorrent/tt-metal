# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn


device_id = 0
device = ttnn.open_device(device_id=device_id)

torch_input_tensor = torch.rand(1, 1, 2, 4, dtype=torch.float32)
input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
output_tensor = ttnn.exp(input_tensor)  # exp migrated to ttnn
torch_output_tensor = ttnn.to_torch(output_tensor)

ttnn.close_device(device)
