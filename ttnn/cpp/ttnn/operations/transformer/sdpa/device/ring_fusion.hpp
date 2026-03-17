// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::prim {

struct RingSDPAFusedOpSignaler {
    uint32_t num_fused_op_cores_to_signal = 0;
    std::vector<CoreCoord> fused_op_receiver_cores_noc;
    std::vector<uint32_t> fused_op_receiver_signal_semaphores;  // [dir0, dir1] — indexed by direction
    ttnn::experimental::ccl::FusedOpSignalerMode fused_op_signaler_mode =
        ttnn::experimental::ccl::FusedOpSignalerMode::MULTI;

    /* All Gather specs */
    uint32_t ring_size = 0;
    uint32_t ring_index = 0;

    bool initialized_all_gather = false;
    bool initialized_fused_op = false;

    void init_all_gather(uint32_t ring_size, uint32_t ring_index);

    void init_fused_op(
        tt::tt_metal::Program& program,
        const tt::tt_metal::IDevice* device,
        const std::variant<CoreRange, CoreRangeSet>& core_range_to_signal,
        ttnn::experimental::ccl::FusedOpSignalerMode fused_op_signaler_mode =
            ttnn::experimental::ccl::FusedOpSignalerMode::MULTI);

    // direction: 0 = backward, 1 = forward. Selects the appropriate semaphore from
    // fused_op_receiver_signal_semaphores and encodes direction for the kernel sequencer.
    void push_ring_sdpa_fused_op_rt_args(std::vector<uint32_t>& out_rt_args, uint32_t direction);
};

}  // namespace ttnn::prim
