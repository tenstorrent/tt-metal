// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

#include <tt-metalium/program.hpp>
#include <tt-metalium/runtime_args_data.hpp>

#include <functional>
#include <optional>
#include <tuple>
#include <vector>

namespace ttnn::experimental::prim {

// Whether every core computes at most one M block, so the output buffer holds only one row. Keyed in the program hash.
inline bool agmm_single_m_block(uint32_t M_tiles, uint32_t M_block_tiles, uint32_t in0_parallel_axis_cores) {
    return M_block_tiles > 0 && in0_parallel_axis_cores > 0 &&
           (M_tiles + in0_parallel_axis_cores - 1) / in0_parallel_axis_cores <= M_block_tiles;
}

// Block and subblock sizes the factory picks when no config is given: (M, K, N block tiles, subblock_h, subblock_w).
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> default_block_sizes(
    uint32_t M, uint32_t K, uint32_t N, bool fp32_dest_acc_en);

// The mesh workload: one program per coordinate range, with per-device runtime args (a device's rank and routes), or
// one per coordinate where the devices' programs differ in more. A cache hit computes the call's addresses, semaphores
// and row count once and writes them into each program's common args (it owns every slot that can change between
// calls).
struct AllGatherMinimalMatmulAsyncProgramFactory {
    struct shared_variables_t {
        // The argument objects, not their data(): dispatch may relocate the payload.
        std::reference_wrapper<tt::tt_metal::RuntimeArgsData> in0_sender;
        std::reference_wrapper<tt::tt_metal::RuntimeArgsData> in0_receiver_no_fabric;
        std::reference_wrapper<tt::tt_metal::RuntimeArgsData> in0_receiver_fabric;
        std::reference_wrapper<tt::tt_metal::RuntimeArgsData> in1_sender;
        std::reference_wrapper<tt::tt_metal::RuntimeArgsData> in1_receiver;
        std::reference_wrapper<tt::tt_metal::RuntimeArgsData> compute;
        // gather_rows: the row movers' common args (in the mux kernel's slot).
        std::optional<std::reference_wrapper<tt::tt_metal::RuntimeArgsData>> mover;
        std::optional<std::reference_wrapper<tt::tt_metal::RuntimeArgsData>> mover_reader;
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const AllGatherMinimalMatmulAsyncParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const AllGatherMinimalMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AllGatherMinimalMatmulAsyncParams& operation_attributes,
        const AllGatherMinimalMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
