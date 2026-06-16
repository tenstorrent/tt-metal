// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

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
    uint32_t chunk_start_idx{0};
    IndexerScoreProgramConfig program_config{};
    // Resolved (not optional) so it is part of the reflected program-cache key; the public callable
    // fills it from the user's optional config, defaulting math_fidelity to the dtype-derived choice.
    DeviceComputeKernelConfig compute_kernel_config{};
};

struct tensor_args_t {
    const Tensor& q;
    const Tensor& k;
    const Tensor& weights;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::indexer_score
