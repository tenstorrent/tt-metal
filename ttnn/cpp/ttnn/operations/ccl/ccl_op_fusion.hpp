// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program.hpp>

namespace ttnn {
namespace experimental {
namespace ccl {

struct CoreSemPair {
    CoreCoord core = {0, 0};
    uint32_t sem_id = 0;

    CoreSemPair() {}
    CoreSemPair(CoreCoord core, uint32_t sem_id) : core(core), sem_id(sem_id) {}
};

enum class FusedOpSignalerMode {
    // When signaling the fused op, only one core is signaled
    // when a tensor slice is ready to be processed by the fused op
    SINGLE,
    // When signaling the fused op, all the cores of the fused op are signaled
    // when a tensor slice is ready to be processed by the fused op
    MULTI
};

struct AllGatherFusedOpSignaler {
    uint32_t num_fused_op_cores_to_signal = 0;
    std::vector<CoreCoord> fused_op_receiver_cores_noc = {};
    std::vector<uint32_t> fused_op_receiver_signal_semaphores = {};
    FusedOpSignalerMode fused_op_signaler_mode = FusedOpSignalerMode::MULTI;

    /* All Gather specific */
    std::vector<CoreCoord> all_gather_worker_cores_noc = {};
    uint32_t all_gather_worker_sync_semaphore = 0;

    bool initialized_fused_op = false;
    bool initialized_all_gather = false;

    AllGatherFusedOpSignaler() {}

    void init_fused_op(
        const std::vector<CoreCoord>& fused_op_receiver_cores_noc,
        const std::vector<uint32_t>& fused_op_receiver_signal_semaphores,
        FusedOpSignalerMode fused_op_signaler_mode = FusedOpSignalerMode::MULTI);

    void init_all_gather(
        tt::tt_metal::Program& program,
        tt::tt_metal::IDevice const* device,

        CoreRangeSet const& all_gather_workers,
        std::vector<CoreCoord>& all_gather_worker_cores);

    void push_all_gather_fused_op_rt_args(
        std::vector<uint32_t>& out_rt_args,

        uint32_t num_workers_to_sync,
        uint32_t curr_worker_index,
        uint32_t all_gather_direction);
};

// Used to propagate semaphore information from matmul to reduce scatter in matmul_reduce_scatter op
struct ReduceScatterFusedOpSignaler {
    uint32_t num_fused_op_cores_to_signal = 1;
    std::vector<CoreCoord> fused_op_receiver_cores_noc = {};
    std::vector<uint32_t> fused_op_receiver_signal_semaphores = {};
    FusedOpSignalerMode fused_op_signaler_mode = FusedOpSignalerMode::SINGLE;

    bool initialized_reduce_scatter = false;
    bool initialized_fused_op = false;

    void init_reduce_scatter(
        tt::tt_metal::Program& program,
        const tt::tt_metal::IDevice* device,
        const std::variant<CoreRange, CoreRangeSet>& core_range_to_signal);

    void init_fused_op();

    void push_reduce_scatter_fused_op_rt_args(std::vector<uint32_t>& out_rt_args);
};

enum class MatmulFusedOpSignalerType { ALL_GATHER, REDUCE_SCATTER, EMPTY };

// Used to propagate semaphore information from matmul to all_gather or reduce_scatter
struct MatmulFusedOpSignaler {
    MatmulFusedOpSignalerType fused_op_type = MatmulFusedOpSignalerType::EMPTY;

    /* Matmul info for All Gather */
    uint32_t num_fused_op_cores_to_signal = 0;
    std::vector<CoreCoord> fused_op_receiver_cores_noc = {};
    std::vector<uint32_t> fused_op_receiver_signal_semaphores = {};  // [dir0, dir1]
    FusedOpSignalerMode fused_op_signaler_mode = FusedOpSignalerMode::MULTI;

    /* All Gather specs */
    uint32_t num_transfers = 0;
    uint32_t ring_size = 0;
    uint32_t start_ring_index = 0;
    uint32_t tensor_slice_shape_width = 0;
    uint32_t output_page_offset = 0;
    uint32_t last_output_page_offset = 0;
    bool is_clockwise_dir = true;
    uint32_t weight_output_page_offset = 0;

    /* Matmul info for Reduce Scatter */
    std::vector<CoreCoord> matmul_worker_cores_noc = {};
    std::vector<CoreCoord> matmul_worker_cores = {};
    uint32_t matmul_worker_sync_semaphore = 0;

    bool initialized_all_gather = false;
    bool initialized_reduce_scatter = false;
    bool initialized_fused_op = false;

    MatmulFusedOpSignaler(MatmulFusedOpSignalerType signaler_type) { fused_op_type = signaler_type; }

    void init_all_gather(
        uint32_t num_transfers,
        uint32_t ring_size,
        uint32_t start_ring_index,
        uint32_t tensor_slice_shape_width,
        uint32_t output_page_offset,
        bool is_clockwise_direction,

        uint32_t weight_tensor_width);

    void init_reduce_scatter(
        const std::vector<CoreCoord>& fused_op_receiver_cores_noc,
        const std::vector<uint32_t>& fused_op_receiver_signal_semaphores,
        FusedOpSignalerMode fused_op_signaler_mode);

    void init_fused_op(
        tt::tt_metal::Program& program,
        const tt::tt_metal::IDevice* device,
        const CoreRange& matmul_workers,
        const std::vector<CoreCoord>& matmul_worker_cores);

    void init_fused_op(
        tt::tt_metal::Program& program,
        const tt::tt_metal::IDevice* device,
        const std::variant<CoreRange, CoreRangeSet>& core_range_to_signal,
        FusedOpSignalerMode fused_op_signaler_mode = FusedOpSignalerMode::MULTI);

    bool is_all_gather();
    bool is_reduce_scatter();

    void push_matmul_fused_op_rt_args(
        std::vector<uint32_t>& out_rt_args, uint32_t curr_worker_in0_idx, uint32_t curr_worker_in1_idx);
    void push_matmul_fused_op_rt_args(std::vector<uint32_t>& out_rt_args, bool use_in1_offset);
};

}  // namespace ccl
}  // namespace experimental
}  // namespace ttnn
