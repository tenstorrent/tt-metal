# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
import time

device_id = 0
device = ttnn.open_device(device_id=device_id)

ttnn.enable_program_cache(device)

torch_input_tensor = torch.rand(2, 4, dtype=torch.float32)
input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# Running the first time will compile the program and cache it
start_time = time.time()
output_tensor = ttnn.exp(input_tensor)
torch_output_tensor = ttnn.to_torch(output_tensor)
end_time = time.time()
duration = end_time - start_time
print(f"duration of the first run: {duration}")
# stdout: duration of the first run: 0.6391518115997314

# Running the subsequent time will use the cached program
start_time = time.time()
output_tensor = ttnn.exp(input_tensor)
torch_output_tensor = ttnn.to_torch(output_tensor)
end_time = time.time()
duration = end_time - start_time
print(f"duration of the second run: {duration}")
# stdout: duration of the subsequent run: 0.0007393360137939453

ttnn.close_device(device)
