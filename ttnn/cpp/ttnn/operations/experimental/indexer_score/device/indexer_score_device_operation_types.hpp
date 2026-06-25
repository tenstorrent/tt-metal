// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/base_types.hpp>

namespace ttnn::operations::experimental::indexer_score {

// Work-unit knobs (elements, tile-aligned; SDPAProgramConfig analogue).
// One unit = q_chunk rows x k_chunk keys; heads stream in head_group blocks.
struct IndexerScoreProgramConfig {
    std::size_t q_chunk_size = 32;
    std::size_t k_chunk_size = 32;
    std::size_t head_group_size = 1;  // heads resident at once; 1 always fits L1, raise for perf (0 = all)
};

// Resolve head_group_size to a concrete head count (0 = all Hi). Single-sourced so validate and the
// factory can't drift on the "0 = all" contract.
inline uint32_t resolve_head_group(const IndexerScoreProgramConfig& cfg, uint32_t Hi) {
    return cfg.head_group_size == 0 ? Hi : static_cast<uint32_t>(cfg.head_group_size);
}

struct operation_attributes_t {
    // Absolute chunk_start of rank 0 (lowest cluster_axis coord; the only device on a single chip). Rank r
    // uses chunk_start_idx + r*Sq (Sq = per-device q seq len; r = linearized index along cluster_axis), so
    // the per-device value is derived host-side and passed to compute as a RUNTIME arg, hash-excluded --
    // distinct values reuse one program (see compute_program_hash).
    uint32_t chunk_start_idx{0};             // absolute chunk_start of rank 0 (elements, tile-aligned)
    std::optional<uint32_t> cluster_axis{};  // mesh axis that is the SP ring; unset = linear device order
    IndexerScoreProgramConfig program_config{};
    // Resolved (not optional) so it is part of the reflected program-cache key; the public callable
    // fills it from the user's optional config, defaulting math_fidelity to the dtype-derived choice.
    DeviceComputeKernelConfig compute_kernel_config{};
    // Indexed KV cache: selects the batch slot of a shared [B,1,T,D] k (page ids offset by
    // cache_batch_idx * Tt * Dt). k may then also be ND-sharded across DRAM banks. Value is NOT hashed and
    // is re-applied in override_runtime_arguments, so switching slots does NOT recompile.
    std::optional<uint32_t> cache_batch_idx{std::nullopt};
    bool has_indexed_kv_cache() const { return cache_batch_idx.has_value(); }
    // Runtime KV length: the valid prefix this dispatch of a k allocated at its full T; the rest is
    // masked out. Value is NOT hashed (re-applied per dispatch), so growing kv_len <= T reuses ONE program.
    // grid/work-split/output width stay keyed on the hashed T. nullopt == T.
    std::optional<uint32_t> kv_len{std::nullopt};
    bool has_runtime_kv_len() const { return kv_len.has_value(); }
};

struct tensor_args_t {
    const Tensor& q;
    const Tensor& k;
    const Tensor& weights;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::indexer_score
