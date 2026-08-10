// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct AttnResMergeParams {
    uint32_t site;
    // Above zero, live_scores carries the statistics the live score is derived
    // from rather than the score, stacked one pair per tensor-parallel rank, and
    // the derivation happens here. inv_hidden_size and eps are that derivation's
    // constants and are unread at zero.
    uint32_t num_partials;
    float inv_hidden_size;
    float eps;
    tt::tt_metal::MemoryConfig output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;

    // `site` shapes no kernel — it resolves to page offsets the reader takes as
    // common runtime args — so it is kept out of the hash and re-applied per
    // dispatch via get_dynamic_runtime_args. Without that, a caller batching R
    // read sites compiles and caches R copies of one program.
    static constexpr auto attribute_names =
        std::forward_as_tuple("num_partials", "inv_hidden_size", "eps", "output_mem_config", "compute_kernel_config");
    auto attribute_values() const {
        return std::forward_as_tuple(num_partials, inv_hidden_size, eps, output_mem_config, compute_kernel_config);
    }
};

// partial - [R, 1, H, W], TILE layout. prefix_sum - [1, 1, H, W]: the live
//   stream is one plane behind every site, so only the partial batches.
// shift, mass - [R, 1, H, 1], TILE layout. One scalar per row, physically a tile
//   whose column 0 holds it; that is already the layout BroadcastType::COL reads,
//   so no pre-pass builds it.
// live_scores - [R, 1, H, 1] at num_partials 0, the same one-scalar-per-row
//   layout. Above zero it is instead [1, 2 * num_partials, H, 1]: each rank's
//   sum of squares then its dots, stacked rank-major the way a gathering
//   collective leaves them.
//
// An operand's dim 0 is a read-site axis, and `site` picks the plane. At R == 1
// the operand is shared by every site and `site` does not apply to it, which is
// what lets one call mix a batched partial, shift and mass with a live_scores
// that was computed for this site alone. The output is one plane either way.
struct AttnResMergeInputs {
    ttnn::Tensor partial;
    ttnn::Tensor prefix_sum;
    ttnn::Tensor shift;
    ttnn::Tensor mass;
    ttnn::Tensor live_scores;
};

}  // namespace ttnn::experimental::prim
