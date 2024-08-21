# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn

torch_input_tensor = torch.zeros(2, 4, dtype=torch.float32)
tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16)
torch_output_tensor = ttnn.to_torch(tensor)
