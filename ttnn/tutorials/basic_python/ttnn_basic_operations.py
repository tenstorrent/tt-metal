# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import torch
import numpy as np
import ttnn
from loguru import logger


def main():
    # Open the Device
    device = ttnn.open_device(device_id=0)

    logger.info("\n--- TT-NN Tensor Creation with Tiles (32x32) ---")
    host_rand = torch.rand((32, 32), dtype=torch.float32)

    # Helper to create a TT-NN tensor from torch with TILE_LAYOUT and bfloat16
    def to_tt_tile(torch_tensor):
        return ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_t1 = to_tt_tile(host_rand)

    tt_t2 = ttnn.full(
        shape=(32, 32),
        fill_value=1.0,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_t3 = ttnn.zeros(
        shape=(32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_t4 = ttnn.ones(
        shape=(32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    t5 = np.array([[5, 6], [7, 8]], dtype=np.float32).repeat(16, axis=0).repeat(16, axis=1)
    tt_t5 = ttnn.Tensor(t5, device=device, layout=ttnn.TILE_LAYOUT)

    logger.info("\n--- TT-NN Tensor Operations on (32x32) Tiles ---")
    add_result = ttnn.add(tt_t1, tt_t4)
    logger.info(f"Addition:\n{add_result}")

    mul_result = ttnn.mul(tt_t1, tt_t5)
    logger.info(f"Element-wise Multiplication:\n{mul_result}")

    matmul_result = ttnn.matmul(tt_t4, tt_t1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    logger.info(f"Matrix Multiplication:\n{matmul_result}")

    logger.info("\n--- Simulated Broadcasting (32x32 + Broadcasted Row Vector) ---")
    broadcast_vector = torch.tensor(np.arange(0, 32), dtype=torch.float32).repeat(32, 1)
    logger.info(f"Broadcast Row Vector:\n{broadcast_vector}")

    broadcast_tt = to_tt_tile(broadcast_vector)
    broadcast_add_result = ttnn.add(tt_t1, broadcast_tt)
    logger.info(f"Broadcast Add Result (TT-NN):\n{ttnn.to_torch(broadcast_add_result)}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
