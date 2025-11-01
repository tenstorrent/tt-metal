# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import ttnn


def main():
    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    m = 1024  # Number of rows in matrix A and result
    k = 1024  # Number of columns in A / rows in B (must match for valid matmul)
    n = 1024  # Number of columns in matrix B and result

    a = ttnn.rand((m, k), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    b = ttnn.rand((k, n), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    output = a @ b

    output = a @ b

    print(output.layout)

    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)

    print("Printing ttnn tensor")
    print(f"shape: {output.shape}")
    print(f"chunk of a tensor:\n{output[:1, :32]}")

    a = ttnn.rand(
        (m, k), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    b = ttnn.rand(
        (k, n), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    output = ttnn.matmul(a, b, memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=ttnn.CoreGrid(y=8, x=8))

    output = ttnn.matmul(a, b, memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=ttnn.CoreGrid(y=8, x=8))

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
