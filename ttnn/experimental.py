# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from .core import Tensor

import tt_lib as ttl


def exp(input_tensor: Tensor) -> Tensor:
    input_tensor = input_tensor._tensor
    output_tensor = ttl.tensor.exp(input_tensor)
    return Tensor(output_tensor)
