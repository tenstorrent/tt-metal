// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::experimental::prim {

// Fused sequence-parallel "multiply-then-scatter" on dim 2:
//   rs [B,1,S/T,N] = reduce_scatter(input [B,1,S,K] @ weight, dim=2, cluster_axis)
// The matmul runs on the sub-batched view [B*T,1,S/T,K] -> [B*T,1,S/T,N] (fuse_batch=false) so that one matmul batch
// iteration produces exactly one (batch b, sequence slice s) RS slice, and the RS readers wait per (b, s) via the
// sp_slice_schedule ordinal table instead of once per batch.
struct MatmulReduceScatterSpAsyncParams {
    // dim == 2, ring_size == T (mesh extent along cluster_axis), cluster_axis set, using_persistent_buffers == false,
    // num_workers_per_link resolved (never nullopt: the RS builders' data-size heuristic would pick 2-8 workers and
    // blow the reserved core budget).
    ReduceScatterMinimalAsyncParams reduce_scatter_params;
    // bcast_batch == true; program_config == nullopt means "derive per sub-batch" (sp_matmul_program_config).
    ttnn::prim::MatmulParams matmul_params;
    // Bottom rows of the compute grid reserved for the CCL workers; the matmul gets grid.x x (grid.y - ccl_core_rows)
    // cores at (0,0).
    uint32_t ccl_core_rows = 2;
    // Measurement knob: make every reduce-scatter wait resolve only after the last matmul sub-batch (no overlap).
    bool debug_serialize_reduce_scatter = false;
    // Keep each core's in1 (weight) slab resident in L1 across the sub-batches when it fits (MatmulFusedOpSignaler::
    // sp_in1_resident); env TT_SP_IN1_STREAM=1 disables it for perf decomposition.
    bool in1_resident = true;

    // Reflection (logging / graph reports). The nested ReduceScatterMinimalAsyncParams is deliberately not listed:
    // it is both an aggregate and carries its own attribute_names, which makes tt_stl's to_json ambiguous.
    static constexpr auto attribute_names = std::forward_as_tuple(
        "dim",
        "num_links",
        "ring_size",
        "topology",
        "cluster_axis",
        "num_workers_per_link",
        "matmul_params",
        "ccl_core_rows",
        "debug_serialize_reduce_scatter",
        "in1_resident");
    auto attribute_values() const {
        return std::forward_as_tuple(
            this->reduce_scatter_params.dim,
            this->reduce_scatter_params.num_links,
            this->reduce_scatter_params.ring_size,
            this->reduce_scatter_params.topology,
            this->reduce_scatter_params.cluster_axis,
            this->reduce_scatter_params.num_workers_per_link,
            this->matmul_params,
            this->ccl_core_rows,
            this->debug_serialize_reduce_scatter,
            this->in1_resident);
    }
};

struct MatmulReduceScatterSpAsyncInputs {
    Tensor input;   // [B,1,S,K] activations (K = this rank's K shard)
    Tensor weight;  // [1,1,K,N], or [1,1,N,K] with transpose_b
};

// Layout of the op's output vector (every tensor is allocated by the op; nothing is caller-owned):
//   [kMmPartialIdx]      matmul partial [B,1,S,N] (the RS input)
//   [kRsIntermediateIdx] RS intermediate (Ring: chunk-paged staging; Linear: [2B,1,S,N] tiled)
//   [kRsOutputIdx]       RS output [B,1,S/T,N] -- the user-visible result
//   [kRsPenultIdx]       Ring contiguous-staging path only: the penult intermediate
inline constexpr size_t kMmPartialIdx = 0;
inline constexpr size_t kRsIntermediateIdx = 1;
inline constexpr size_t kRsOutputIdx = 2;
inline constexpr size_t kRsPenultIdx = 3;

}  // namespace ttnn::experimental::prim
