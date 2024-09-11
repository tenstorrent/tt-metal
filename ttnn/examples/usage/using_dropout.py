# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import torch
import numpy as np

with ttnn.manage_device(device_id=0) as device:
    device.enable_program_cache()
    device.enable_async(True)
    t = torch.ones(
        (
            32,
            1,
            32,
            32,
        )
    )
    t_tt = ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_ratios = []
    s = 124
    prob = 0.2
    for _ in range(100):
        output = ttnn.dropout(t_tt, seed=s, probability=prob, scale=1.0 / (1.0 - prob))
        output_torch = ttnn.to_torch(output)
        r = 1.0 - (torch.count_nonzero(output_torch) / torch.count_nonzero(t)).item()
        tt_ratios.append(r)

    mean = np.mean(tt_ratios)
    std = np.std(tt_ratios)
    print("TTNN:")
    print("Expected probability:", prob)
    print("Mean:", mean)
    print("Standard Deviation:", std)
    print("MinMax output:", output_torch.min().item(), output_torch.max().item())
    torch_ratios = []
    dropout = torch.nn.Dropout(p=prob)
    dropout.train()
    for _ in range(100):
        output_torch = dropout(t)
        r = 1.0 - (torch.count_nonzero(output_torch) / torch.count_nonzero(t)).item()
        torch_ratios.append(r)

    mean = np.mean(torch_ratios)
    std = np.std(torch_ratios)
    print("PYTORCH:")
    print("Expected probability:", prob)
    print("Mean:", mean)
    print("Standard Deviation:", std)
    print("MinMax output:", output_torch.min().item(), output_torch.max().item())
