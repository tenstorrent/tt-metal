// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_fusion.hpp"

using namespace tt::tt_metal;
namespace ttnn::prim {

void RingSDPAFusedOpSignaler::init_all_gather(
    uint32_t ring_size, uint32_t ring_index, uint32_t forward_writes_expected, uint32_t backward_writes_expected) {
    this->ring_size = ring_size;
    this->ring_index = ring_index;
    this->forward_writes_expected = forward_writes_expected;
    this->backward_writes_expected = backward_writes_expected;

    this->initialized_all_gather = true;
}

void RingSDPAFusedOpSignaler::init_fused_op(
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

void RingSDPAFusedOpSignaler::push_ring_sdpa_fused_op_rt_args(std::vector<uint32_t>& out_rt_args) {
    TT_ASSERT(
        this->initialized_all_gather && this->initialized_fused_op, "RingSDPAFusedOpSignaler not initialized fully.");

    out_rt_args.push_back(static_cast<uint32_t>(this->ring_size));
    out_rt_args.push_back(static_cast<uint32_t>(this->ring_index));
    out_rt_args.push_back(static_cast<uint32_t>(this->forward_writes_expected));
    out_rt_args.push_back(static_cast<uint32_t>(this->backward_writes_expected));
    out_rt_args.push_back(static_cast<uint32_t>(this->fused_op_receiver_signal_semaphores[0]));
    out_rt_args.push_back(static_cast<uint32_t>(this->fused_op_receiver_signal_semaphores[1]));
}
}  // namespace ttnn::prim
