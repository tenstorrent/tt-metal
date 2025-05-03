# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Optional

from loguru import logger
import ttnn


def Linear(in_features: int, out_features: int, weight: List[Union[int, float]], bias, device):
    """
    Returns a function that performs a Linear operation with optional bias.

    ``weight`` must be the weight as a tilized list of values.
    """
    logger.warning("Linear will be deprecated soon, please use linear in models/helper_funcs")
    assert weight.get_layout() == ttnn.TILE_LAYOUT
    weight = weight.to(device)

    if bias is None:
        bias = None
    else:
        assert bias.get_layout() == ttnn.TILE_LAYOUT
        bias = bias.to(device)

    def linear_(activation):
        weight_T = ttnn.transpose(weight, -2, -1)
        output = ttnn.matmul(activation, weight_T)

        if bias is not None:
            output_plus_bias = ttnn.add(output, bias)
            return output_plus_bias

        return output

    return linear_
