# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import ttnn
from loguru import logger


def main():
    # Open Tenstorrent device
    device = ttnn.open_device(device_id=0)

    # Create two TT-NN tensors with TILE_LAYOUT
    tt_tensor1 = ttnn.full(
        shape=(32, 32),
        fill_value=1.0,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_tensor2 = ttnn.full(
        shape=(32, 32),
        fill_value=2.0,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    logger.info("Input tensors:")
    logger.info(tt_tensor1)
    logger.info(tt_tensor2)

    # Perform eltwise addition on the device
    tt_result = ttnn.add(tt_tensor1, tt_tensor2)

    # Log output tensor
    logger.info("Output tensor:")
    logger.info(tt_result)

    ttnn.close_device(device)
    logger.info("Device closed successfully.")


if __name__ == "__main__":
    main()
