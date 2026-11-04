# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from loguru import logger
from tt_py_test_utils_common.comparison_funcs import comp_equal


def run_embeddings_tests(
    batch_size, num_embeddings, embedding_dim, num_rows, dtype, in0_mem_config, out_mem_config, device, tilized=False
):
    torch.manual_seed(1234)

    dev = device

    input_rows_shape = [batch_size, 1, 1, num_rows]

    input_rows_torch = torch.randint(0, num_embeddings - 1, tuple(input_rows_shape))

    weights_shape = [1, 1, num_embeddings, embedding_dim]
    weights_torch = torch.randn(weights_shape).bfloat16()

    input_tensor = ttnn.Tensor(input_rows_torch, ttnn.uint32).to(dev, in0_mem_config)
    weights_tensor = ttnn.Tensor(weights_torch, dtype).to(dev, in0_mem_config)

    out_layout = ttnn.TILE_LAYOUT if tilized else ttnn.ROW_MAJOR_LAYOUT
    ttz = ttnn.embedding(input_tensor, weights_tensor, layout=out_layout, memory_config=out_mem_config)
    ttz = ttnn.reshape(ttz, [ttz.shape[0], 1, ttz.shape[1], ttz.shape[2]])

    if tilized:
        tt_data = ttz.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    else:
        tt_data = ttz.cpu().to_torch()

    tt_got_back = torch.Tensor(tt_data).reshape((batch_size, 1, num_rows, embedding_dim))
    t_ref = torch.nn.functional.embedding(
        input_rows_torch.reshape((batch_size, num_rows)),
        weights_torch.reshape((num_embeddings, embedding_dim)),
    ).reshape((batch_size, 1, num_rows, embedding_dim))

    passing_pcc, output_pcc = comp_equal(t_ref, tt_got_back)
    logger.debug(f"Out passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")
    assert passing_pcc
