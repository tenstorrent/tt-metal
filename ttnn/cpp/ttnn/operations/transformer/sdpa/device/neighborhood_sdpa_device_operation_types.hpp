// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa/device/neighborhood_plan.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// 3D neighborhood attention. Tensors arrive in BRICKED site order (see neighborhood_plan.hpp
// and models/tt_dit/layers/neighborhood_permute.py): 32 consecutive sites are one compact 3D
// box, so one tile row is one brick of video rather than a pencil along width.
struct NeighborhoodSDPAParams {
    // Volume, context window, stride and brick. Every geometric decision derives from this;
    // the kernels never see it, only the offsets it produces.
    transformer::neighborhood::NeighborhoodConfig config;

    // Heads cannot be read off the shape: Q arrives as [batch, 1, bricked_sites, heads * head_dim]
    // so that a TILE row is 32 SITES. ttnn tiles the last two dimensions, so any shape putting
    // heads there would cut tiles across heads instead of across the volume.
    uint32_t head_count = 1;

    // LTX passes 1.0 because Q arrives pre-scaled. Compile-time: folded into the program hash.
    float scale = 1.0f;

    // Optional Cauchy–Schwarz softmax offset: sqrt(head_dim) * max|k_norm_weight|.
    // When set, the kernel subtracts ||q|| * this from every score instead of an online row
    // max, which deletes the max pass and the per-chunk rescale. Legal only when K was
    // RMS-normed with that weight (RoPE preserves the bound). The VALUE is a runtime arg so
    // the eight stage-5 blocks share one program; presence is hashed.
    std::optional<float> k_norm_bound;

    // KV tiles per flash chunk. Should divide the gather so the last chunk is not ragged;
    // derive it from the plan rather than fixing it.
    uint32_t tiles_per_kv_chunk = 8;

    // Diagnostic ablation. 0 is the shipped kernel. Any other value produces WRONG output on
    // purpose: it skips one stage of the fused op (keeping the circular-buffer handshake) so a
    // host timer can attribute wall time. Hashed, because each value is a different program.
    //
    //   0 full          shipped path
    //   1 skip_kv       reader issues no K/V DRAM reads
    //   2 mask_memset   reader fills every mask tile with a constant
    //   3 drain         compute waits and pops Q/K/V/mask, copies Q to output
    //   4 qk            drain + QK^T (no softmax, no PV)
    //   5 qk_softmax    QK + softmax (no PV)
    //   6 qk_pv         QK + PV, skip softmax
    uint32_t probe = 0;

    // 0 auto, 1 interior-only kernel, 2 edge-only kernel. Hashed. See reader_arg::path_mode.
    // path_mode 0 at the C++ entry auto-splits stride-1 + relative table into 1 then 2.
    uint32_t path_mode = 0;

    DeviceComputeKernelConfig compute_kernel_config;
    tt::tt_metal::MemoryConfig output_memory_config;
};

struct NeighborhoodSDPAInputs {
    // [batch, head_count, brick_count * SITES_PER_BRICK, head_dim], TILE layout, bricked order.
    Tensor query_tensor;
    Tensor key_tensor;
    Tensor value_tensor;

    // [1, 1, brick_count, 4] uint32 ROW_MAJOR: the (time, height, width) origin of each
    // chunk's gather, from NeighborhoodPlan::gather_origin_by_chunk. Padded to 4 for
    // alignment; the fourth column is unused.
    //
    // Uploaded rather than computed in the kernel on purpose: the reader and the compute
    // kernel then cannot disagree about geometry, which is the failure mode
    // sliding_window_geometry.hpp warns about in its own header.
    Tensor gather_origin_table;

    // [1, 1, 32, gather_brick_count * 32] additive {0, -inf}, TILE. One tile per gather slot,
    // valid for any query brick clear of the volume boundary. Optional: without it the reader
    // evaluates every mask itself.
    std::optional<Tensor> interior_mask;

    std::optional<Tensor> output_tensor;
};

}  // namespace ttnn::prim
