# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def _golden_function(input_tensor, *args, **kwargs):
    import torch

    return torch.exp(input_tensor)


ttnn.experimental.tensor.exp.golden_function = _golden_function
