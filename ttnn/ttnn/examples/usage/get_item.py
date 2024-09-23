# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


# Note that this not a view, unlike torch tensor

import torch
import ttnn

device_id = 0
device = ttnn.open_device(device_id=device_id)

torch_input_tensor = torch.rand(3, 96, 128, dtype=torch.float32)
input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
output_tensor = input_tensor[:1, 32:64, 32:64]  # this particular slice will run on the device
torch_output_tensor = ttnn.to_torch(output_tensor)

ttnn.close_device(device)
