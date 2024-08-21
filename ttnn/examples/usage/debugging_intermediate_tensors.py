# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn

device_id = 0
device = ttnn.open_device(device_id=device_id)

torch_input_tensor = torch.rand(32, 32, dtype=torch.float32)
input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
with ttnn.manage_config("enable_comparison_mode", True):
    with ttnn.manage_config(
        "comparison_mode_pcc", 0.9998
    ):  # This is optional in case default value of 0.9999 is too high
        output_tensor = ttnn.exp(input_tensor)
torch_output_tensor = ttnn.to_torch(output_tensor)

ttnn.close_device(device)
