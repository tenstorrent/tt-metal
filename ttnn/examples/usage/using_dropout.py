# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import torch
import numpy as np


def get_tensor_ratio(t, out):
    t_sum = t.sum().item()
    out_sum = out.sum().item()
    return 1.0 - out_sum / t_sum


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

    ratios = []
    s = 124
    prob = 0.2
    for _ in range(100):
        output = ttnn.dropout(t_tt, seed=s, probability=prob, scale=1.0)
        output1 = ttnn.to_torch(output)
        r = get_tensor_ratio(t, output1)
        ratios.append(r)

    mean = np.mean(ratios)
    std = np.std(ratios)
    print("TTNN:")
    print("Expected probability:", prob)
    print("Mean:", mean)
    print("Standard Deviation:", std)

    torch_ratios = []
    dropout = torch.nn.Dropout(p=prob)
    dropout.train()
    for _ in range(100):
        output = dropout(t) * (1 - prob)
        r = get_tensor_ratio(t, output)
        torch_ratios.append(r)

    mean = np.mean(torch_ratios)
    std = np.std(torch_ratios)
    print("PYTORCH:")
    print("Expected probability:", prob)
    print("Mean:", mean)
    print("Standard Deviation:", std)
