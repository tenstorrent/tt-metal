// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_padding_config {

// Builds the per-device MoE padding config — [local_real_tokens, pad_side] — ON DEVICE, from this
// chunk's `actual_start` / `actual_end` held in two 1-element uint32 tensors.
//
// This is the traceable counterpart of the host builder (TtMoEGatePrefill.build_padding_config), which
// ends in a ttnn.from_torch and so cannot run inside a trace capture. The consumers
// (moe_grouped_topk's gate and the dispatch op) already read padding_config as a device tensor, so
// moving only its producer on-device makes the whole padding-aware path traceable with no downstream
// change: one captured program recomputes the correct config on every replay.
//
// Counts are derived for the KV-pad-aware ROTATED block-cyclic layout, mirroring
// update_padded_kv_cache's writer (same boundary_slab / boundary_chip / boundary_offset), and reduce
// exactly to the sequential formula when `actual_start` is slab-aligned (including 0) — so one op
// covers both the rotated and non-rotated cases.
//
// NOT covered: the `is_balanced=True` zigzag placement, whose per-device count is not expressible in
// this form. Rotated chunked prefill implies is_balanced=False (ttMLA._chunked_attn asserts it), so
// that combination is unreachable here; balanced callers must keep using the host builder.
//
// In-place: returns a handle to `config`. `config` is caller-owned and must be persistent (a stable
// address) for a captured trace to keep writing the same buffer across replays.
//
// Args:
//   config:          per-device [.., 2] UINT32 ROW_MAJOR DRAM tensor (the global tensor is
//                    [sp_factor, 2] sharded along `cluster_axis`, so each chip holds one row).
//   actual_start:    1-element uint32 DRAM tensor ([1,1,1,1], ROW_MAJOR, replicated) — absolute KV
//                    position of this chunk's first real token.
//   actual_end:      1-element uint32 DRAM tensor (same layout) — one past its last real token.
//   tokens_per_chip: tokens this chip carries per chunk (the MoE's sp_dim).
//   pad_side:        0 = right, 1 = left.
//   cluster_axis:    mesh axis the config is sharded along (the SP axis).
ttnn::Tensor moe_padding_config(
    const ttnn::Tensor& config,
    const ttnn::Tensor& actual_start,
    const ttnn::Tensor& actual_end,
    uint32_t tokens_per_chip,
    uint32_t pad_side,
    uint32_t cluster_axis);

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_padding_config

namespace ttnn {
using operations::experimental::deepseek_prefill::moe_padding_config::moe_padding_config;
}  // namespace ttnn
