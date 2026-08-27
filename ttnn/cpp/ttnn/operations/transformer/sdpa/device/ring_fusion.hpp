// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
    std::vector<tt::tt_metal::CoreCoord> fused_op_receiver_cores_noc;
    std::vector<uint32_t> fused_op_receiver_signal_semaphores;  // [dir0, dir1]
    ttnn::experimental::ccl::FusedOpSignalerMode fused_op_signaler_mode =
        ttnn::experimental::ccl::FusedOpSignalerMode::MULTI;

    /* All Gather specs */
    uint32_t ring_size = 0;
    uint32_t ring_index = 0;
    uint32_t forward_writes_expected = 0;
    uint32_t backward_writes_expected = 0;

    // Set by the program factory when the all-gather relays the diametric slice split across both links
    bool split_forwarding_enabled = false;

    bool initialized_all_gather = false;
    bool initialized_fused_op = false;

    void init_all_gather(
        uint32_t ring_size, uint32_t ring_index, uint32_t forward_writes_expected, uint32_t backward_writes_expected);

    void init_fused_op(
        tt::tt_metal::Program& program,
        const tt::tt_metal::IDevice* device,
        const std::variant<CoreRange, CoreRangeSet>& core_range_to_signal,
        ttnn::experimental::ccl::FusedOpSignalerMode fused_op_signaler_mode =
            ttnn::experimental::ccl::FusedOpSignalerMode::MULTI);

    // Runtime args push_ring_sdpa_fused_op_rt_args() appends:
    // {ring_size, ring_index, fwd_writes, bwd_writes, sem0, sem1, split_fwd, split_shard_id,
    // split_second_half_wait}. A kernel whose receiver is built with wait_for_op_signal=false
    // consumes only the first four and must step over the rest to read its own args (see
    // ring_joint_writer.cpp); the ring-joint factory TT_FATALs on this count so adding a word here
    // fails the build instead of silently shifting those args.
    static constexpr uint32_t kRtArgCount = 9;

    void push_ring_sdpa_fused_op_rt_args(std::vector<uint32_t>& out_rt_args);
};

}  // namespace ttnn::prim
