// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>

#include "gate_up_matmul_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/mesh_coord.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

// Per-quadrant group of cores sharing the same compile-time m_blocks_local and
// n_blocks_local.  The 2-D M×N grid produces at most four such groups.
//
// BRISC kernels are split by column role:
//   sender_x_id   — reader_x_mcast_sender.cpp on x=0 cores (reads DRAM, multicasts).
//   receiver_x_id — reader_x_mcast_receiver.cpp on x>0 cores (receives multicast).
//   When n_n_cores==1 there are no receivers; receiver_x_id stays 0 and
//   sender_x_id uses num_receivers=0 (plain DRAM read, no multicast).
//
// NCRISC (reader_weights_id) and TRISC (compute_id) apply to all cores in the
// quadrant regardless of their x position.
struct KernelGroupInfo {
    tt::tt_metal::KernelHandle sender_x_id = 0;    // BRISC sender   (x=0 column)
    tt::tt_metal::KernelHandle receiver_x_id = 0;  // BRISC receiver (x>0 columns; 0 if none)
    tt::tt_metal::KernelHandle reader_weights_id = 0;
    tt::tt_metal::KernelHandle compute_id = 0;

    std::vector<CoreCoord> sender_cores;    // ordered x=0 cores for runtime arg assignment
    std::vector<CoreCoord> receiver_cores;  // ordered x>0 cores
    std::vector<CoreCoord> all_cores;       // sender_cores ++ receiver_cores (for NCRISC/TRISC)
};

struct GateUpMatmulSharedVariables {
    std::vector<KernelGroupInfo> groups;
};

struct GateUpMatmulProgramFactory {
    using shared_variables_t = GateUpMatmulSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;
    using tensor_return_value_t = Tensor;

    static cached_mesh_workload_t create_mesh_workload(
        const GateUpMatmulParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const GateUpMatmulInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const GateUpMatmulParams& operation_attributes,
        const GateUpMatmulInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn
