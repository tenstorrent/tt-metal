// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/program/program.hpp"

namespace ttnn {
namespace experimental {
namespace ccl {

struct CoreSemPair{
    CoreCoord core;
    uint32_t sem_id;

    CoreSemPair() {}
    CoreSemPair(CoreCoord core, uint32_t sem_id) : core(core), sem_id(sem_id) {}
};

struct AllGatherFusedOpSignaler {
    uint32_t num_fused_op_cores_to_signal;
    std::vector<CoreCoord> fused_op_receiver_cores_noc;
    std::vector<uint32_t> fused_op_receiver_signal_semaphores;

    /* All Gather specific */
    std::vector<CoreCoord> all_gather_worker_cores_noc;
    uint32_t all_gather_worker_sync_semaphore;

    bool initialized_fused_op = false;
    bool initialized_all_gather = false;

    AllGatherFusedOpSignaler() {}

    void init_fused_op(
        const std::vector<CoreCoord>& fused_op_receiver_cores_noc,
        const std::vector<uint32_t>& fused_op_receiver_signal_semaphores
    );

    void init_all_gather(
        Program& program,
        Device const* device,

        CoreRangeSet const& all_gather_workers,
        std::vector<CoreCoord>& all_gather_worker_cores
    );

    void emit_all_gather_fused_op_rt_args(
        std::vector<uint32_t>& rt_args,

        uint32_t num_workers_to_sync,
        uint32_t curr_worker_index,
        uint32_t all_gather_direction,
        std::optional<CoreSemPair> start_signal_core_sem_pair = {}
    );

};

// Used to propagate semaphore information from matmul to all_gather in all_gather_matmul op
struct MatmulFusedOpSignaler {
    uint32_t num_fused_op_cores_to_signal;
    std::vector<CoreCoord> fused_op_receiver_cores_noc;
    std::vector<uint32_t> fused_op_receiver_signal_semaphores; // [dir0, dir1]

    /* All Gather specs */
    uint32_t num_transfers;
    uint32_t ring_size;
    uint32_t start_ring_index;
    uint32_t tensor_slice_shape_width;
    uint32_t output_page_offset;
    uint32_t last_output_page_offset;
    bool is_clockwise_dir;

    uint32_t weight_output_page_offset;

    bool initialized_all_gather = false;
    bool initialized_fused_op = false;

    MatmulFusedOpSignaler() {}

    void init_all_gather(
        uint32_t num_transfers,
        uint32_t ring_size,
        uint32_t start_ring_index,
        uint32_t tensor_slice_shape_width,
        uint32_t output_page_offset,
        bool is_clockwise_direction,

        uint32_t weight_tensor_width
    );

    void init_fused_op(
        Program& program,
        Device const* device,
        const std::variant<CoreRange, CoreRangeSet>& core_range_to_signal
    );

    void emit_matmul_fused_op_rt_args(
        std::vector<uint32_t>& rt_args,
        bool use_in1_offset
    );
};


}  // namespace ccl
}  // namespace experimental
}  // namespace ttnn
