// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operation.hpp"
#include "dit_fused_distributed_groupnorm_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct DitFusedDistributedGroupnormSharedVariables {
    // reader_kernel_ids = {sender (mcast-group master), receiver}. Multi-core mcast layout:
    // sender cores run the master reader (intra-device combine + fabric + mcast-back), receiver
    // cores run the receiver reader (signal + wait). Compute + writer run on all cores.
    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> compute_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> forwarder_kernel_ids;
    std::vector<tt::tt_metal::CoreCoord> forwarder_cores;
    std::vector<tt::tt_metal::CoreCoord> cores;           // all reduction cores (writer/compute)
    std::vector<tt::tt_metal::CoreCoord> sender_cores;    // mcast-group masters (sender reader)
    std::vector<tt::tt_metal::CoreCoord> receiver_cores;  // non-master cores (receiver reader)
    // Index of the stats DRAM scratch address inside each master reader's runtime-args vector,
    // parallel to sender_cores. The index varies per core because the preceding mcast sub-group
    // blocks are optional. Empty on the is_local (ring_size==1) path.
    std::vector<size_t> stats_dram_addr_reader_arg_idx;
    // Virtual (group column, tile-row slot) coordinates, parallel to cores, sender_cores and
    // receiver_cores. The size-dependent runtime args are rewritten from them on every call.
    std::vector<tt::tt_metal::CoreCoord> virtual_cores;
    std::vector<tt::tt_metal::CoreCoord> sender_virtual_cores;
    std::vector<tt::tt_metal::CoreCoord> receiver_virtual_cores;
    uint32_t num_virtual_rows = 1;
    uint32_t per_core_Nt = 1;
};

// The reduction grid: num_virtual_cols group columns of num_virtual_rows cores each, laid out as
// (num_virtual_cols * col_replication) x num_actual_rows cores. It depends on the input only through
// the channel count and, for inputs with fewer tile rows than the grid has rows, the tile row count.
struct DitFusedDistributedGroupnormGrid {
    uint32_t num_virtual_cols = 1;
    uint32_t num_virtual_rows = 1;
    uint32_t num_actual_rows = 1;
    uint32_t col_replication = 1;
};

DitFusedDistributedGroupnormGrid gn_choose_grid(const DitFusedDistributedGroupnormParams& args, const Tensor& input);

struct DitFusedDistributedGroupnormMeshWorkloadFactory {
    using shared_variables_t = DitFusedDistributedGroupnormSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const DitFusedDistributedGroupnormParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const DitFusedDistributedGroupnormInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const DitFusedDistributedGroupnormParams& operation_attributes,
        const DitFusedDistributedGroupnormInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create_at(
        const DitFusedDistributedGroupnormParams& args,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const DitFusedDistributedGroupnormInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
