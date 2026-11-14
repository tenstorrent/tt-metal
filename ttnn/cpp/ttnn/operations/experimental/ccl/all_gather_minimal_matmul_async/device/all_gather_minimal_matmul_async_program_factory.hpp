// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

#include <tt-metalium/program.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include <optional>
#include <tuple>
#include <vector>

namespace ttnn::experimental::prim {

// Whether every core computes at most one M block. With one N block per core that is the core's only output block,
// so no write is deferred behind a later block and the output buffer only has to hold the row being written. The
// program hash keys on this (the row count itself is not in the hash).
inline bool agmm_single_m_block(uint32_t M_tiles, uint32_t M_block_tiles, uint32_t in0_parallel_axis_cores) {
    return M_block_tiles > 0 && in0_parallel_axis_cores > 0 &&
           (M_tiles + in0_parallel_axis_cores - 1) / in0_parallel_axis_cores <= M_block_tiles;
}

// Block and subblock sizes the factory picks when no config is given: (M, K, N block tiles, subblock_h, subblock_w).
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> default_block_sizes(
    uint32_t M, uint32_t K, uint32_t N, bool fp32_dest_acc_en);

struct AllGatherMinimalMatmulAsyncProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const AllGatherMinimalMatmulAsyncParams& operation_attributes,
        const AllGatherMinimalMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const AllGatherMinimalMatmulAsyncParams& operation_attributes,
        const AllGatherMinimalMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& mesh_coordinate = std::nullopt);
};

}  // namespace ttnn::experimental::prim
