// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// Fused sequence-parallel gather-then-multiply: all_gather(input, dim=2, cluster_axis) -> gathered [B,1,S,K], then
// gathered @ weight -> mm [B,1,S,N], with the matmul consuming each gathered slice as soon as it lands
// (MatmulFusedOpSignaler SP_ALL_GATHER, see sp_matmul_fusion_common).
struct AllGatherMatmulSpAsyncParams {
    /* All gather (dim = 2, cluster_axis set, num_workers_per_link resolved) */
    AllGatherAsyncParams all_gather;
    AllGatherAsyncInputs all_gather_tensor_args;
    /* Matmul (2D-mcast program config on the [B*T,1,S/T,K] view, fuse_batch=false) */
    ttnn::prim::MatmulParams matmul;
    /* Core grid split: the all-gather workers take the bottom `ccl_core_rows` rows, the matmul the rows above */
    uint32_t ccl_core_rows = 1;
    CoreCoord all_gather_core_grid_offset;
    CoreCoord matmul_grid;
    // Debug (env TT_SP_AG_MM_SERIALIZE=1): matmul waits for the whole all-gather before its first slice, i.e. no
    // overlap. For perf decomposition only.
    bool debug_serialize_ag = false;
    // The all-gather reader signals a forwarded slice when it has landed (true, default) rather than after it has
    // been forwarded one hop further (the historical fused-AG timing, one slice-time later; env
    // TT_SP_AG_SIGNAL_LATE=1 selects it for perf decomposition).
    bool ag_signal_on_receive = true;
    // Keep each core's in1 (weight) slab resident in L1 across the sub-batches when it fits (MatmulFusedOpSignaler::
    // sp_in1_resident); env TT_SP_IN1_STREAM=1 disables it for perf decomposition.
    bool in1_resident = true;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "matmul",
        "ccl_core_rows",
        "all_gather_core_grid_offset",
        "matmul_grid",
        "debug_serialize_ag",
        "ag_signal_on_receive",
        "in1_resident");
    auto attribute_values() const {
        return std::forward_as_tuple(
            this->matmul,
            this->ccl_core_rows,
            this->all_gather_core_grid_offset,
            this->matmul_grid,
            this->debug_serialize_ag,
            this->ag_signal_on_receive,
            this->in1_resident);
    }
};

struct AllGatherMatmulSpAsyncInputs {
    Tensor input;   // [B,1,S/T,K], this rank's sequence shard
    Tensor weight;  // [1,1,K,N], or [1,1,N,K] with transpose_b
    std::optional<const Tensor> bias;
};

// {gathered [B,1,S,K], mm [B,1,S,N]}
using AllGatherMatmulSpAsyncResult = std::vector<Tensor>;
using AllGatherMatmulSpAsyncResultSpec = std::vector<tt::tt_metal::TensorSpec>;

}  // namespace ttnn::experimental::prim
