// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/program/program.hpp"
#include "ttnn/operations/experimental/ccl/ccl_op_fusion.hpp"

namespace ttnn {
namespace experimental {
namespace ccl {

void AllGatherFusedOpSignaler::init_fused_op(
    const std::vector<CoreCoord>& fused_op_receiver_cores_noc,
    const std::vector<uint32_t>& fused_op_receiver_signal_semaphores
) {
    this->fused_op_receiver_cores_noc = fused_op_receiver_cores_noc;
    this->fused_op_receiver_signal_semaphores = fused_op_receiver_signal_semaphores;
    this->num_fused_op_cores_to_signal = fused_op_receiver_cores_noc.size();

    initialized_fused_op = true;
}

void AllGatherFusedOpSignaler::init_all_gather(
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

void AllGatherFusedOpSignaler::emit_all_gather_fused_op_ct_args(
    std::vector<uint32_t>& ct_args,

    uint32_t num_workers_to_sync,
    uint32_t curr_worker_index
) {
    TT_ASSERT(initialized_fused_op && initialized_all_gather, "AllGatherFusedOpSignaler not initialized fully.");

    ct_args.push_back(static_cast<uint32_t>(num_workers_to_sync));
    ct_args.push_back(static_cast<uint32_t>(curr_worker_index));
    ct_args.push_back(static_cast<uint32_t>(this->all_gather_worker_sync_semaphore));

}


void AllGatherFusedOpSignaler::emit_all_gather_fused_op_rt_args(
    std::vector<uint32_t>& rt_args,

    uint32_t all_gather_direction,
    std::optional<CoreSemPair> start_signal_core_sem_pair
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

    // Push the fused op signal semaphore addrs. Direction 0: clockwise, Direction 1: counter-clockwise
    rt_args.push_back(
        static_cast<uint32_t>(this->fused_op_receiver_signal_semaphores[all_gather_direction])
    );

    // Push the params for the start signal
    bool wait_for_start_signal = !start_signal_core_sem_pair.has_value() && all_gather_direction == 0;
    bool send_start_signal = start_signal_core_sem_pair.has_value() && all_gather_direction == 0;

    rt_args.push_back(static_cast<uint32_t>(wait_for_start_signal));
    rt_args.push_back(static_cast<uint32_t>(send_start_signal));

    if (send_start_signal) {
        rt_args.push_back(static_cast<uint32_t>(start_signal_core_sem_pair->core.x));
        rt_args.push_back(static_cast<uint32_t>(start_signal_core_sem_pair->core.y));
        rt_args.push_back(static_cast<uint32_t>(start_signal_core_sem_pair->sem_id));
    }

}


// Used to propagate semaphore information from matmul to all_gather in all_gather_matmul op
void MatmulFusedOpSignaler::init_all_gather(
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
    this->is_clockwise_dir = is_clockwise_direction;

    this->weight_tensor_width = weight_tensor_width;

    initialized_all_gather = true;
}

void MatmulFusedOpSignaler::init_fused_op(
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
    this->fused_op_receiver_signal_semaphores.push_back(CreateSemaphore(program, core_range_to_signal, 0));
    this->fused_op_receiver_signal_semaphores.push_back(CreateSemaphore(program, core_range_to_signal, 0));

    // Set the number of fused op cores to signal
    this->num_fused_op_cores_to_signal = this->fused_op_receiver_cores_noc.size();

    initialized_fused_op = true;
}

void MatmulFusedOpSignaler::emit_matmul_fused_op_ct_args(
    std::vector<uint32_t>& ct_args
) {
    TT_ASSERT(initialized_all_gather && initialized_fused_op, "MatmulFusedOpSignaler not initialized fully.");

    ct_args.push_back(static_cast<bool>(true));
    ct_args.push_back(static_cast<uint32_t>(this->num_transfers));
    ct_args.push_back(static_cast<uint32_t>(this->ring_size));
    ct_args.push_back(static_cast<uint32_t>(this->start_ring_index));
    ct_args.push_back(static_cast<uint32_t>(this->tensor_slice_shape_width));
    ct_args.push_back(static_cast<uint32_t>(this->output_page_offset));
    ct_args.push_back(static_cast<uint32_t>((this->ring_size - 1) * this->output_page_offset));
    ct_args.push_back(static_cast<uint32_t>(this->is_clockwise_dir));
    ct_args.push_back(static_cast<uint32_t>(this->fused_op_receiver_signal_semaphores[0]));
    ct_args.push_back(static_cast<uint32_t>(this->fused_op_receiver_signal_semaphores[1]));
}



}  // namespace ccl
}  // namespace experimental
}  // namespace ttnn
