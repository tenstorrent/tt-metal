# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import ttnn


def main():
    # Open Tenstorrent device
    device = ttnn.open_device(device_id=0)

    try:
        # Helper to create a TT-NN tensor from torch with TILE_LAYOUT and bfloat16
        def to_tt_tile(torch_tensor):
            return ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Helper to create (32, 32) torch tensor from scalar or numpy
        def create_host_tensor(fill_value):
            if isinstance(fill_value, (int, float)):
                return torch.full((32, 32), fill_value, dtype=torch.float32)
            elif isinstance(fill_value, np.ndarray):
                return torch.from_numpy(fill_value.astype(np.float32))
            else:
                raise ValueError("Unsupported type for fill_value")

        print("\n--- TT-NN Tensor Creation with Tiles (32x32) ---")
        host_t1 = create_host_tensor(1)
        host_t2 = torch.zeros((32, 32), dtype=torch.float32)
        host_t3 = torch.ones((32, 32), dtype=torch.float32)
        host_t4 = torch.rand((32, 32), dtype=torch.float32)
        host_np_array = np.array([[5, 6], [7, 8]]).repeat(16, axis=0).repeat(16, axis=1)
        host_t5 = create_host_tensor(host_np_array)

        tt_t1 = to_tt_tile(host_t1)
        tt_t2 = to_tt_tile(host_t2)
        tt_t3 = to_tt_tile(host_t3)
        tt_t4 = to_tt_tile(host_t4)
        tt_t5 = to_tt_tile(host_t5)

        print("Tensor from fill value 1:\n", ttnn.to_torch(tt_t1))
        print("Zeros:\n", ttnn.to_torch(tt_t2))
        print("Ones:\n", ttnn.to_torch(tt_t3))
        print("Random:\n", ttnn.to_torch(tt_t4))
        print("From expanded NumPy (TT-NN):\n", ttnn.to_torch(tt_t5))

        print("\n--- TT-NN Tensor Operations on (32x32) Tiles ---")
        add_result = ttnn.add(tt_t3, tt_t4)
        mul_result = ttnn.mul(tt_t4, tt_t5)
        matmul_result = ttnn.matmul(tt_t3, tt_t4, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        ttnn_add = ttnn.to_torch(add_result)
        print("Addition:\n", ttnn_add)

        ttnn_mul = ttnn.to_torch(mul_result)
        print("Element-wise Multiplication:\n", ttnn_mul)

        ttnn_matmul = ttnn.to_torch(matmul_result)
        print("Matrix Multiplication:\n", ttnn_matmul)

        print("\n--- Simulated Broadcasting (32x32 + Broadcasted Row Vector) ---")
        broadcast_vector = torch.tensor([[1.0] * 32], dtype=torch.float32).repeat(32, 1)
        broadcast_tt = to_tt_tile(broadcast_vector)
        broadcast_add_result = ttnn.add(tt_t4, broadcast_tt)
        print("Broadcast Add Result (TT-NN):\n", ttnn.to_torch(broadcast_add_result))

        print("\nAll TT-NN Part 1 tensor basics with tiles tests completed.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
