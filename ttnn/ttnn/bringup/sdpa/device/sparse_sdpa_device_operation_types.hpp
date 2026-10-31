// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../sparse_sdpa.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "block_cyclic_layout.hpp"  // ttnn::prim::bringup::BlockCyclicLayout (shared)
#include <optional>

namespace ttnn::prim::bringup {

// Sparse MLA prefill (DeepSeek DSA).
struct SparseSDPAParams {
    float scale = 1.0f;  // compile-time (folded into the program hash)
    uint32_t v_dim;      // width of V (= leading v_dim cols of the K_DIM-wide KV cache); the output width
    transformer::bringup::SparseKVFormat kv_format;
    uint32_t k_chunk_size = 128;
    DeviceComputeKernelConfig compute_kernel_config;
    // Indexed KV cache: when set, kv is a [B,1,T,K_DIM] shared cache and this selects the batch slot to
    // attend to (the gather page ids are offset by cache_batch_idx * T). It is a DYNAMIC runtime arg
    // (excluded from the program hash, re-applied every dispatch), so changing it does NOT recompile.
    std::optional<uint32_t> cache_batch_idx = std::nullopt;
    bool has_indexed_kv_cache() const { return cache_batch_idx.has_value(); }
    // The remap configuration is compile-time; T is part of the program hash for this path.
    std::optional<BlockCyclicLayout> block_cyclic = std::nullopt;
    bool has_block_cyclic() const { return block_cyclic.has_value(); }
    // Fork option: Float32 output / row-sum running-state CBs and exact softmax exp (compile-time; hashed).
    // Default = the source program.
    bool high_precision = false;
    // Fork option (needs high_precision): the output in Float32 (default: q's dtype). The normalized output CB is
    // Float32 and unpacks straight to the fp32 DEST for the untilize (exact). Compile-time; hashed.
    bool fp32_output = false;
    // Fork option (needs high_precision and fp32_output): scores / probabilities in a Float32 CB (default bf16), every
    // Float32 value the FPU reads later rounded to TF32 (nearest) in DEST before its pack, and a FLOAT32 q accepted
    // (Float32 Q CBs). Compile-time (define SPARSE_SDPA_FP32_SCORES); hashed.
    bool fp32_scores = false;
    // Fork option (needs fp32_scores and a bf16 q): a Float32 running max, and the probabilities rounded to bf16
    // (nearest) in DEST after the row-sum, before P@V (DeepSeek's sparse_attn: fp32 scores / max / sum, bf16 P);
    // fp32_output may then be false (bf16 output). Compile-time (define SPARSE_SDPA_BF16_PROBS); hashed.
    bool bf16_probs = false;
    // Fork option (deepseek_v41_flash_d_p FIX.15, agent rule 18: at most 110 Tensix cores an op on a Galaxy): the
    // compute grid is cut to the rectangle (0, 0) .. (max_grid.x - 1, max_grid.y - 1) of the device's grid. Default
    // (none) = the device's whole compute grid (the source program). Compile-time; hashed.
    std::optional<tt::tt_metal::CoreCoord> max_grid = std::nullopt;
    bool has_scaled_kv() const { return kv_format == transformer::bringup::SparseKVFormat::SCALED_FP8; }
    // Fork option: bfloat4_b row records + a bf16 window tensor (define SPARSE_SDPA_ROW_BFP4; hashed via kv_format).
    bool has_row_bfp4() const { return kv_format == transformer::bringup::SparseKVFormat::ROW_BFP4; }
    // Fork option: bfloat8_b row records, the window only (the ROW_BFP4 kernels with the bfloat8_b window; kv is the
    // window).
    bool has_row_bfp8() const { return kv_format == transformer::bringup::SparseKVFormat::ROW_BFP8; }
};

struct SparseSDPAInputs {
    Tensor q;        // [1, H, S, K_DIM] bf16/fp8_e4m3 ROW_MAJOR  (K_DIM = head dim, e.g. 576)
    Tensor kv;       // Plain [B,1,T,K_DIM] or packed scaled-FP8 rows; format is explicit in SparseSDPAParams
    Tensor indices;  // [1, 1, S, TOPK] uint32 ROW_MAJOR  (0xFFFFFFFF = masked)
    // [1, 1, 1, H] ROW_MAJOR BF16; scaled like QK.
    std::optional<Tensor> attention_sink;
    // ROW_BFP4 only: [1, 1, Tw, K_DIM] bf16 ROW_MAJOR (or [1, 1, Tw, 576] uint8 bfloat8_b row records), the rows the
    // first k_chunk of the index list point at.
    std::optional<Tensor> kv_window;
};

}  // namespace ttnn::prim::bringup
