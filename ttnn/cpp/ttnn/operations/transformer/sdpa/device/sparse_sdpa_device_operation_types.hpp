// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/transformer/sdpa/sparse_sdpa.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "block_cyclic_layout.hpp"  // ttnn::prim::BlockCyclicLayout (shared)
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <optional>

namespace ttnn::prim {

// Pushes the next layer's prefix rows (sparse_kv_push part 1) on push_core_grid, disjoint from worker_core_grid.
struct SparseSDPAFusedPush {
    uint32_t stripe_chunk = 0;
    uint32_t max_rows = 0;
    uint32_t num_links = 2;
    // Not hashed, like the semaphores. With push_batch_index_tensor the slot is
    // tensor[0] * batch_slot_num_layers + batch_slot_layer_idx, read on device.
    uint32_t input_batch_index = 0;
    uint32_t batch_slot_num_layers = 1;
    uint32_t batch_slot_layer_idx = 0;
    std::optional<GlobalSemaphore> ready_semaphore;
    std::optional<GlobalSemaphore> data_valid_semaphore;
    tt::tt_metal::CoreRangeSet push_core_grid;
    tt::tt_fabric::FabricConfig fabric_config = tt::tt_fabric::FabricConfig::DISABLED;
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
    // Worker cores: sub_core_grids, or all cores of the first subdevice.
    tt::tt_metal::CoreRangeSet worker_core_grid;
    std::optional<SparseSDPAFusedPush> fused_push = std::nullopt;
};

struct SparseSDPAInputs {
    Tensor q;        // [1, H, S, K_DIM] bf16/fp8_e4m3 ROW_MAJOR  (K_DIM = head dim, e.g. 576)
    Tensor kv;       // Plain [B,1,T,K_DIM] or packed scaled-FP8 rows; format is explicit in SparseSDPAParams
    Tensor indices;  // [1, 1, S, TOPK] uint32 ROW_MAJOR  (0xFFFFFFFF = masked)
    // [1, 1, 1, H] ROW_MAJOR BF16; scaled like QK.
    std::optional<Tensor> attention_sink;
    // Fused push only.
    std::optional<Tensor> push_input;
    std::optional<Tensor> push_table;
    std::optional<Tensor> push_output;
    std::optional<Tensor> push_batch_index_tensor;
};

}  // namespace ttnn::prim
