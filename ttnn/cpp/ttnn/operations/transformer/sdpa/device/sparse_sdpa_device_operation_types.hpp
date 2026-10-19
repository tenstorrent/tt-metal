// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/transformer/sdpa/sparse_sdpa.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "block_cyclic_layout.hpp"  // ttnn::prim::BlockCyclicLayout (shared)
#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/global_semaphore.hpp"
#include <optional>

namespace ttnn::prim {

// PROTOTYPE: a sparse KV push (high_bw_all_gather push_part 1, the next layer's prefix rows) fused into this op. Its
// reader/writer/mux kernels run on push_core_grid (disjoint from worker_core_grid); every attention core tells the
// push senders to stop when it finishes, and the senders record how far they got for the next layer's resume.
struct SparseSDPAFusedPush {
    uint32_t stripe_chunk = 0;
    uint32_t max_rows = 0;
    uint32_t num_links = 2;
    uint32_t input_batch_index = 0;  // cache slot pushed; a runtime arg (not hashed)
    std::optional<GlobalSemaphore> ready_semaphore;
    std::optional<GlobalSemaphore> data_valid_semaphore;
    tt::tt_metal::CoreRangeSet push_core_grid;
};

// Sparse MLA prefill (DeepSeek DSA).
struct SparseSDPAParams {
    float scale = 1.0f;  // compile-time (folded into the program hash)
    uint32_t v_dim;      // width of V (= leading v_dim cols of the K_DIM-wide KV cache); the output width
    transformer::SparseKVFormat kv_format;
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
    bool has_scaled_kv() const { return kv_format == transformer::SparseKVFormat::SCALED_FP8; }
    // Worker cores, resolved from subdevice_id / sub_core_grids (default: all cores of the first subdevice).
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt;
    tt::tt_metal::CoreRangeSet worker_core_grid;
    std::optional<SparseSDPAFusedPush> fused_push = std::nullopt;
};

struct SparseSDPAInputs {
    Tensor q;        // [1, H, S, K_DIM] bf16/fp8_e4m3 ROW_MAJOR  (K_DIM = head dim, e.g. 576)
    Tensor kv;       // Plain [B,1,T,K_DIM] or packed scaled-FP8 rows; format is explicit in SparseSDPAParams
    Tensor indices;  // [1, 1, S, TOPK] uint32 ROW_MAJOR  (0xFFFFFFFF = masked)
    // [1, 1, 1, H] ROW_MAJOR BF16; scaled like QK.
    std::optional<Tensor> attention_sink;
    // Fused push (see SparseSDPAFusedPush): KVPE cache pushed from, all-gathered plan tables, the requester's compact
    // buffer, and the per-sender progress tensor ([1, 1, workers, 16] uint32).
    std::optional<Tensor> push_input;
    std::optional<Tensor> push_table;
    std::optional<Tensor> push_output;
    std::optional<Tensor> push_progress;
};

}  // namespace ttnn::prim
