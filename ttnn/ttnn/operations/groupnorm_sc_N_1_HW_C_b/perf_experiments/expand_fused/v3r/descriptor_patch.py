# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""expand_fused CB layout on top of the op's descriptor (kernel CT/RT args unchanged):
  - cb_gamma_rows (13) holds the gamma map + beta map (2W bf16 pages; gamma only: W); cb_beta_rows (14) is gone
    (same L1: W + W pages before);
  - cb_mr_e (25, Float32, 2 pages) is a view on alias group B (x^2 / output ring), idle between pass 1 and pass 2.
"""
import ttnn

CB_X_SQUARED, CB_GAMMA_ROWS, CB_BETA_ROWS, CB_MR_E = 4, 13, 14, 25
F32_TILE_BYTES, BF16_TILE_BYTES = 4096, 2048


def patch(desc, plan, input_tensor, output_tensor, **kw):
    out = []
    for cb in list(desc.cbs):
        idxs = [f.buffer_index for f in cb.format_descriptors]
        if CB_BETA_ROWS in idxs:
            continue
        if CB_GAMMA_ROWS in idxs:
            cb.total_size = (2 if plan.has_beta else 1) * plan.W * BF16_TILE_BYTES
        if CB_X_SQUARED in idxs:
            fds = list(cb.format_descriptors)
            fds.append(
                ttnn.CBFormatDescriptor(buffer_index=CB_MR_E, data_format=ttnn.float32, page_size=F32_TILE_BYTES)
            )
            cb.format_descriptors = fds
            cb.total_size = max(cb.total_size, 2 * F32_TILE_BYTES)
        out.append(cb)
    desc.cbs = out
    return desc
