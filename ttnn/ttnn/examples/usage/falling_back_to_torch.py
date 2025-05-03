# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn

torch_input_tensor = torch.zeros(2, 4, dtype=torch.float32)

input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16)

# Recommended approach
tensor = ttnn.to_torch(input_tensor)
tensor = torch.nn.functional.silu(tensor)
output_tensor = ttnn.from_torch(tensor, dtype=ttnn.bfloat16)

# Alternative approach that only works with some operations
output_tensor = ttnn.get_fallback_function(ttnn.silu)(input_tensor)
