# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def _golden_function(input_tensor: ttnn.Tensor, weight: ttnn.Tensor, **_):
    import torch

    # torch.nn.functional.embedding requires integer indices, but ttnn indices may be
    # UINT32/BFloat16; cast to int64 to avoid a dtype error during comparison mode.
    output_tensor = torch.nn.functional.embedding(input_tensor.to(torch.int64), weight)
    return output_tensor


ttnn.attach_golden_function(ttnn.embedding, golden_function=_golden_function)

EmbeddingsType = ttnn._ttnn.operations.embedding.EmbeddingsType

__all__ = []
