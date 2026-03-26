// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "exp_ring_fusion.hpp"

using namespace tt::tt_metal;
namespace ttnn::prim {

void ExpRingSDPAFusedOpSignaler::init_all_gather(uint32_t ring_size, uint32_t ring_index) {
    this->ring_size = ring_size;
    this->ring_index = ring_index;

    this->initialized_all_gather = true;
}

void ExpRingSDPAFusedOpSignaler::init_fused_op(
    Program& program,
    const IDevice* device,
    const std::variant<CoreRange, CoreRangeSet>& core_range_to_signal,
    ttnn::experimental::ccl::FusedOpSignalerMode fused_op_signaler_mode) {
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

    this->initialized_fused_op = true;
}

void ExpRingSDPAFusedOpSignaler::push_ring_sdpa_fused_op_rt_args(
    std::vector<uint32_t>& out_rt_args, uint32_t direction) {
    TT_ASSERT(
        this->initialized_all_gather && this->initialized_fused_op,
        "ExpRingSDPAFusedOpSignaler not initialized fully.");

    out_rt_args.push_back(static_cast<uint32_t>(this->ring_size));
    out_rt_args.push_back(static_cast<uint32_t>(this->ring_index));
    out_rt_args.push_back(static_cast<uint32_t>(direction));
    out_rt_args.push_back(static_cast<uint32_t>(this->fused_op_receiver_signal_semaphores[direction]));
}
}  // namespace ttnn::prim
