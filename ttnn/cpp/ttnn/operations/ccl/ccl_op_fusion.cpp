// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

using namespace tt::tt_metal;

namespace ttnn {
namespace experimental {
namespace ccl {

void AllGatherFusedOpSignaler::init_fused_op(
    const std::vector<CoreCoord>& fused_op_receiver_cores_noc,
    const std::vector<uint32_t>& fused_op_receiver_signal_semaphores,
    FusedOpSignalerMode fused_op_signaler_mode) {
    this->fused_op_receiver_cores_noc = fused_op_receiver_cores_noc;
    this->fused_op_receiver_signal_semaphores = fused_op_receiver_signal_semaphores;
    this->num_fused_op_cores_to_signal = fused_op_receiver_cores_noc.size();
    this->fused_op_signaler_mode = fused_op_signaler_mode;

    initialized_fused_op = true;
}

void AllGatherFusedOpSignaler::init_all_gather(
    Program& program,
    IDevice const* device,

    CoreRangeSet const& all_gather_workers,
    std::vector<CoreCoord>& all_gather_worker_cores) {
    // Create the sync semaphore for the all gather workers
    if (all_gather_worker_cores.size() > 1) {
        this->all_gather_worker_sync_semaphore = CreateSemaphore(program, all_gather_workers, 0);
    }

    // Get the noc coords for the all gather workers
    this->all_gather_worker_cores_noc.clear();
    for (const auto& core : all_gather_worker_cores) {
        this->all_gather_worker_cores_noc.push_back(device->worker_core_from_logical_core(core));
    }
    initialized_all_gather = true;
}

void AllGatherFusedOpSignaler::push_all_gather_fused_op_rt_args(
    std::vector<uint32_t>& out_rt_args,

    uint32_t num_workers_to_sync,
    uint32_t curr_worker_index,
    uint32_t all_gather_direction) {
    TT_FATAL(initialized_fused_op && initialized_all_gather, "AllGatherFusedOpSignaler not initialized fully.");

    out_rt_args.push_back(static_cast<uint32_t>(num_workers_to_sync));
    out_rt_args.push_back(static_cast<uint32_t>(curr_worker_index));
    out_rt_args.push_back(static_cast<uint32_t>(this->all_gather_worker_sync_semaphore));

    // Push the worker core noc coords
    for (const auto& core : this->all_gather_worker_cores_noc) {
        out_rt_args.push_back(static_cast<uint32_t>(core.x));
        out_rt_args.push_back(static_cast<uint32_t>(core.y));
    }

    // Push the number of fused op cores to signal
    out_rt_args.push_back(static_cast<uint32_t>(this->num_fused_op_cores_to_signal));

    // Push the fused op receiver core noc coords
    for (const auto& core : this->fused_op_receiver_cores_noc) {
        out_rt_args.push_back(static_cast<uint32_t>(core.x));
        out_rt_args.push_back(static_cast<uint32_t>(core.y));
    }

    // Push the fused op signal semaphore addrs. Direction 0: clockwise, Direction 1: counter-clockwise
    out_rt_args.push_back(static_cast<uint32_t>(this->fused_op_receiver_signal_semaphores[all_gather_direction]));
    out_rt_args.push_back(static_cast<uint32_t>(this->fused_op_signaler_mode == FusedOpSignalerMode::SINGLE ? 0 : 1));
}

void ReduceScatterFusedOpSignaler::init_reduce_scatter(
    Program& program, const IDevice* device, const std::variant<CoreRange, CoreRangeSet>& core_range_to_signal) {
    // Clear the existing receiver cores
    this->fused_op_receiver_cores_noc.clear();

    // Visit the variant to handle CoreRange and CoreRangeSet differently
    std::visit(
        [&](auto& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, CoreRange>) {
                // Handle CoreRange
                const auto& cores = grid_to_cores(arg.start_coord, arg.end_coord, true);

                for (auto& core : cores) {
                    this->fused_op_receiver_cores_noc.push_back(device->worker_core_from_logical_core(core));
                }
            } else if constexpr (std::is_same_v<T, CoreRangeSet>) {
                // Handle CoreRangeSet
                for (const auto& range : arg.ranges()) {
                    const auto& cores = grid_to_cores(range.start_coord, range.end_coord, true);
                    for (auto& core : cores) {
                        this->fused_op_receiver_cores_noc.push_back(device->worker_core_from_logical_core(core));
                    }
                }
            }
        },
        core_range_to_signal);
    // Create the semaphores
    this->fused_op_receiver_signal_semaphores.push_back(CreateSemaphore(program, core_range_to_signal, 0));

    // Set the number of fused op cores to signal
    this->num_fused_op_cores_to_signal = this->fused_op_receiver_cores_noc.size();

    if (this->num_fused_op_cores_to_signal > 1) {
        this->fused_op_signaler_mode = FusedOpSignalerMode::MULTI;
    } else {
        this->fused_op_signaler_mode = FusedOpSignalerMode::SINGLE;
    }

    initialized_reduce_scatter = true;
}

void ReduceScatterFusedOpSignaler::init_fused_op() { initialized_fused_op = true; }

void ReduceScatterFusedOpSignaler::push_reduce_scatter_fused_op_rt_args(std::vector<uint32_t>& out_rt_args) {
    TT_FATAL(initialized_reduce_scatter && initialized_fused_op, "ReduceScatterFusedOpSignaler not initialized fully.");
    out_rt_args.push_back(static_cast<uint32_t>(this->fused_op_receiver_signal_semaphores[0]));
}

// Used to propagate semaphore information from matmul to all_gather in all_gather_matmul op
void MatmulFusedOpSignaler::init_all_gather(
    uint32_t num_transfers,
    uint32_t ring_size,
    uint32_t start_ring_index,
    uint32_t tensor_slice_shape_width,
    uint32_t output_page_offset,
    bool is_clockwise_direction,

    uint32_t weight_output_page_offset) {
    this->num_transfers = num_transfers;
    this->ring_size = ring_size;
    this->start_ring_index = start_ring_index;
    this->tensor_slice_shape_width = tensor_slice_shape_width;
    this->output_page_offset = output_page_offset;
    this->is_clockwise_dir = is_clockwise_direction;

    this->weight_output_page_offset = weight_output_page_offset;

    initialized_all_gather = true;
}

// Used to propagate semaphore information from matmul to reduce_scatter in matmul_reduce_scatter op
void MatmulFusedOpSignaler::init_reduce_scatter(
    const std::vector<CoreCoord>& fused_op_receiver_cores_noc,
    const std::vector<uint32_t>& fused_op_receiver_signal_semaphores,
    FusedOpSignalerMode fused_op_signaler_mode) {
    this->fused_op_receiver_cores_noc = fused_op_receiver_cores_noc;
    this->fused_op_receiver_signal_semaphores = fused_op_receiver_signal_semaphores;
    this->num_fused_op_cores_to_signal = fused_op_receiver_cores_noc.size();
    this->fused_op_signaler_mode = fused_op_signaler_mode;

    initialized_reduce_scatter = true;
}

void MatmulFusedOpSignaler::init_fused_op(
    Program& program,
    const IDevice* device,
    const CoreRange& matmul_workers,
    const std::vector<CoreCoord>& matmul_worker_cores) {
    // Create the sync semaphore for the matmul workers
    if (matmul_worker_cores.size() > 1) {
        this->matmul_worker_sync_semaphore = CreateSemaphore(program, matmul_workers, 0);
    }

    // Get the noc coords for the matmul workers
    this->matmul_worker_cores_noc.clear();
    this->matmul_worker_cores.clear();
    for (const auto& core : matmul_worker_cores) {
        this->matmul_worker_cores_noc.push_back(device->worker_core_from_logical_core(core));
        this->matmul_worker_cores.push_back(core);
    }
    initialized_fused_op = true;
}

void MatmulFusedOpSignaler::init_fused_op(
    Program& program,
    const IDevice* device,
    const std::variant<CoreRange, CoreRangeSet>& core_range_to_signal,
    FusedOpSignalerMode fused_op_signaler_mode) {
    this->fused_op_signaler_mode = fused_op_signaler_mode;

    // Clear the existing receiver cores
    this->fused_op_receiver_cores_noc.clear();

    // Visit the variant to handle CoreRange and CoreRangeSet differently
    std::visit(
        [&](auto& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, CoreRange>) {
                // Handle CoreRange
                const auto& cores = grid_to_cores(arg.start_coord, arg.end_coord, true);

                for (auto& core : cores) {
                    this->fused_op_receiver_cores_noc.push_back(device->worker_core_from_logical_core(core));
                }
            } else if constexpr (std::is_same_v<T, CoreRangeSet>) {
                // Handle CoreRangeSet
                for (const auto& range : arg.ranges()) {
                    const auto& cores = grid_to_cores(range.start_coord, range.end_coord, true);
                    for (auto& core : cores) {
                        this->fused_op_receiver_cores_noc.push_back(device->worker_core_from_logical_core(core));
                    }
                }
            }
        },
        core_range_to_signal);
    // Create the semaphores
    this->fused_op_receiver_signal_semaphores.push_back(CreateSemaphore(program, core_range_to_signal, 0));
    this->fused_op_receiver_signal_semaphores.push_back(CreateSemaphore(program, core_range_to_signal, 0));

    // Set the number of fused op cores to signal
    this->num_fused_op_cores_to_signal = this->fused_op_receiver_cores_noc.size();

    initialized_fused_op = true;
}

void MatmulFusedOpSignaler::push_matmul_fused_op_rt_args(
    std::vector<uint32_t>& out_rt_args, uint32_t curr_worker_in0_idx, uint32_t curr_worker_in1_idx) {
    TT_FATAL(initialized_fused_op && initialized_reduce_scatter, "MatmulFusedOpSignaler not initialized fully.");

    out_rt_args.push_back(static_cast<uint32_t>(this->matmul_worker_cores_noc.size()));
    uint32_t curr_worker_index = 0;
    bool core_found = false;
    for (uint32_t i = 0; i < this->matmul_worker_cores.size(); i++) {
        if ((curr_worker_in0_idx == matmul_worker_cores.at(i).y) &&
            (curr_worker_in1_idx == matmul_worker_cores.at(i).x)) {
            curr_worker_index = i;
            core_found = true;
            break;
        }
    }
    TT_FATAL(core_found, "MatmulFusedOpSignaler did not find curr_worker_index.");
    out_rt_args.push_back(static_cast<uint32_t>(curr_worker_index));
    out_rt_args.push_back(static_cast<uint32_t>(this->matmul_worker_sync_semaphore));

    // Push the worker core noc coords
    for (const auto& core : this->matmul_worker_cores_noc) {
        out_rt_args.push_back(static_cast<uint32_t>(core.x));
        out_rt_args.push_back(static_cast<uint32_t>(core.y));
    }

    // Push the number of fused op cores to signal
    out_rt_args.push_back(static_cast<uint32_t>(this->num_fused_op_cores_to_signal));

    // Push the fused op receiver core noc coords
    for (const auto& core : this->fused_op_receiver_cores_noc) {
        out_rt_args.push_back(static_cast<uint32_t>(core.x));
        out_rt_args.push_back(static_cast<uint32_t>(core.y));
    }

    // Push the fused op signal semaphore addrs.
    out_rt_args.push_back(static_cast<uint32_t>(this->fused_op_receiver_signal_semaphores[0]));
    out_rt_args.push_back(static_cast<uint32_t>(this->fused_op_signaler_mode == FusedOpSignalerMode::SINGLE ? 0 : 1));
}

void MatmulFusedOpSignaler::push_matmul_fused_op_rt_args(std::vector<uint32_t>& out_rt_args, bool use_in1_offset) {
    TT_FATAL(initialized_all_gather && initialized_fused_op, "MatmulFusedOpSignaler not initialized fully.");

    out_rt_args.push_back(static_cast<uint32_t>(this->num_transfers));
    out_rt_args.push_back(static_cast<uint32_t>(this->ring_size));
    out_rt_args.push_back(static_cast<uint32_t>(this->start_ring_index));
    out_rt_args.push_back(static_cast<uint32_t>(this->tensor_slice_shape_width));
    if (use_in1_offset) {
        out_rt_args.push_back(static_cast<uint32_t>(this->weight_output_page_offset));
        out_rt_args.push_back(static_cast<uint32_t>((this->ring_size - 1) * this->weight_output_page_offset));
    } else {
        out_rt_args.push_back(static_cast<uint32_t>(this->output_page_offset));
        out_rt_args.push_back(static_cast<uint32_t>((this->ring_size - 1) * this->output_page_offset));
    }
    out_rt_args.push_back(static_cast<uint32_t>(this->is_clockwise_dir));
    out_rt_args.push_back(static_cast<uint32_t>(this->fused_op_receiver_signal_semaphores[0]));
    out_rt_args.push_back(static_cast<uint32_t>(this->fused_op_receiver_signal_semaphores[1]));
}

bool MatmulFusedOpSignaler::is_all_gather() { return fused_op_type == MatmulFusedOpSignalerType::ALL_GATHER; }

bool MatmulFusedOpSignaler::is_reduce_scatter() { return fused_op_type == MatmulFusedOpSignalerType::REDUCE_SCATTER; }

}  // namespace ccl
}  // namespace experimental
}  // namespace ttnn
