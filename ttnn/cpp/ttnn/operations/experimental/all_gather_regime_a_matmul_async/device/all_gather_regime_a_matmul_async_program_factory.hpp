// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "all_gather_regime_a_matmul_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct AllGatherRegimeAMatmulAsyncProgramFactory {
    struct shared_variables_t {
        uint32_t num_cores{};
        std::vector<tt::tt_metal::CoreCoord> cores;  // logical worker coords, index i = bank*preaders + slice
        std::vector<uint32_t> core_noc;              // per-core NoC group (0 => A/g0, 1 => B/g1)
        // Split-NOC kernel handles. readerA/writerA run on the noc==0 group, readerB/writerB on noc==1.
        tt::tt_metal::KernelHandle readerA{};
        tt::tt_metal::KernelHandle readerB{};
        tt::tt_metal::KernelHandle writerA{};
        tt::tt_metal::KernelHandle writerB{};
        tt::tt_metal::KernelHandle compute{};
        // Fused-epilogue / output-split layout (so override_runtime_arguments can locate the appended writer
        // args on a program-cache replay with fresh buffers). Writer fused args begin at index 17.
        bool has_bias{false};
        bool has_ternary{false};
        uint32_t n_chunks{1};
        // Fused-gather block location and shape, so a program-cache replay can refresh the staging buffer,
        // the local-shard base and the (ping-ponged) global semaphore address. Without this the fused path
        // silently reads whatever buffer the FIRST invocation happened to allocate.
        bool fused_gather{false};
        uint32_t fused_rt_base{};  // index of the first fused-gather writer arg
        uint32_t preaders{1};      // ring groups; core i is a fabric client iff (i % preaders) == 0
    };
    // MESH WORKLOAD, not a single broadcast program. The fused fabric gather needs PER-DEVICE runtime
    // args -- each rank has its own index in the TP group and its own forward/backward neighbour
    // FabricNodeIds -- which a single program broadcast to every device cannot express. The single-chip
    // path (tp == 1) simply builds an identical program at every coordinate, so it is unaffected.
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const AllGatherRegimeAMatmulAsyncParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const AllGatherRegimeAMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const AllGatherRegimeAMatmulAsyncParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const AllGatherRegimeAMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const AllGatherRegimeAMatmulAsyncParams& operation_attributes,
        const AllGatherRegimeAMatmulAsyncInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
