// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/program/program.hpp"

namespace ttnn {
namespace experimental {
namespace ccl {

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
    ) {
        this->fused_op_receiver_cores_noc = fused_op_receiver_cores_noc;
        this->fused_op_receiver_signal_semaphores = fused_op_receiver_signal_semaphores;
        this->num_fused_op_cores_to_signal = fused_op_receiver_cores_noc.size();

        initialized_fused_op = true;
    }

    void init_all_gather(
        Program& program,
        Device const* device,

        CoreRangeSet const& all_gather_workers,
        std::vector<CoreCoord>& all_gather_worker_cores
    ) {
        // Create the sync semaphore for the all gather workers
        this->all_gather_worker_sync_semaphore = CreateSemaphore(program, all_gather_workers, 0);

        // Get the noc coords for the all gather workers
        this->all_gather_worker_cores_noc.clear();
        for (const auto& core : all_gather_worker_cores) {
            this->all_gather_worker_cores_noc.push_back(device->worker_core_from_logical_core(core));
        }
        initialized_all_gather = true;
    }

    void emit_all_gather_fused_op_ct_args(
        std::vector<uint32_t>& ct_args,

        uint32_t num_workers_to_sync,
        uint32_t curr_worker_index
    ) {
        TT_ASSERT(initialized_fused_op && initialized_all_gather, "AllGatherFusedOpSignaler not initialized fully.");

        ct_args.push_back(static_cast<uint32_t>(num_workers_to_sync));
        ct_args.push_back(static_cast<uint32_t>(curr_worker_index));
        ct_args.push_back(static_cast<uint32_t>(this->all_gather_worker_sync_semaphore));

    }


    void emit_all_gather_fused_op_rt_args(
        std::vector<uint32_t>& rt_args,

        bool all_gather_direction
    ) {
        TT_ASSERT(initialized_fused_op && initialized_all_gather, "AllGatherFusedOpSignaler not initialized fully.");

        // Push the worker core noc coords
        for (const auto& core : this->all_gather_worker_cores_noc) {
            rt_args.push_back(static_cast<uint32_t>(core.x));
            rt_args.push_back(static_cast<uint32_t>(core.y));
        }

        // Push the number of fused op cores to signal
        rt_args.push_back(static_cast<uint32_t>(this->num_fused_op_cores_to_signal));

        // Push the fused op receiver core noc coords
        for (const auto& core : this->fused_op_receiver_cores_noc) {
            rt_args.push_back(static_cast<uint32_t>(core.x));
            rt_args.push_back(static_cast<uint32_t>(core.y));
        }

        // Push the fused op signal semaphore addrs
        // Direction 0: clockwise
        // Direction 1: counter-clockwise
        rt_args.push_back(
            static_cast<uint32_t>(this->fused_op_receiver_signal_semaphores[all_gather_direction])
        );

    }

    static uint32_t get_num_ct_args() {
        return 3;
    }

};

// Used to propagate semaphore information from matmul to all_gather in all_gather_matmul op
struct MatmulFusedOpSignaler {
    uint32_t num_fused_op_cores_to_signal;
    std::vector<CoreCoord> fused_op_receiver_cores_noc;
    uint32_t fused_op_receiver_signal_semaphores[2]; // [dir0, dir1]

    /* All Gather specs */
    uint32_t num_transfers;
    uint32_t ring_size;
    uint32_t start_ring_index;
    uint32_t tensor_slice_shape_width;
    uint32_t output_page_offset;
    uint32_t last_output_page_offset;
    bool is_clockwise_dir;

    uint32_t weight_tensor_width;

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
    ) {
        this->num_transfers = num_transfers;
        this->ring_size = ring_size;
        this->start_ring_index = start_ring_index;
        this->tensor_slice_shape_width = tensor_slice_shape_width;
        this->output_page_offset = output_page_offset;
        this->last_output_page_offset = (ring_size - 1) * output_page_offset;
        this->is_clockwise_dir = is_clockwise_direction;

        this->weight_tensor_width = weight_tensor_width;

        initialized_all_gather = true;
    }

    void init_fused_op(
        Program& program,
        Device const* device,
        const CoreRange& core_range_to_signal
    ) {

        // Get the noc coords for the fused op receiver cores
        this->fused_op_receiver_cores_noc.clear();
        const auto& cores = grid_to_cores(core_range_to_signal.start_coord, core_range_to_signal.end_coord, true);
        for (auto& core : cores) {
            this->fused_op_receiver_cores_noc.push_back(device->worker_core_from_logical_core(core));
        }

        // Create the semaphores
        this->fused_op_receiver_signal_semaphores[0] = CreateSemaphore(program, core_range_to_signal, 0);
        this->fused_op_receiver_signal_semaphores[1] = CreateSemaphore(program, core_range_to_signal, 0);

        // Set the number of fused op cores to signal
        this->num_fused_op_cores_to_signal = this->fused_op_receiver_cores_noc.size();

        initialized_fused_op = true;
    }

    void emit_matmul_fused_op_ct_args(
        std::vector<uint32_t>& ct_args
    ) {
        TT_ASSERT(initialized_all_gather && initialized_fused_op, "MatmulFusedOpSignaler not initialized fully.");

        ct_args.push_back(static_cast<uint32_t>(this->num_transfers));
        ct_args.push_back(static_cast<uint32_t>(this->ring_size));
        ct_args.push_back(static_cast<uint32_t>(this->start_ring_index));
        ct_args.push_back(static_cast<uint32_t>(this->tensor_slice_shape_width));
        ct_args.push_back(static_cast<uint32_t>(this->output_page_offset));
        ct_args.push_back(static_cast<uint32_t>(this->last_output_page_offset));
        ct_args.push_back(static_cast<uint32_t>(this->is_clockwise_dir));
        ct_args.push_back(static_cast<uint32_t>(this->fused_op_receiver_signal_semaphores[0]));
        ct_args.push_back(static_cast<uint32_t>(this->fused_op_receiver_signal_semaphores[1]));
    }

};


}  // namespace ccl
}  // namespace experimental
}  // namespace ttnn
