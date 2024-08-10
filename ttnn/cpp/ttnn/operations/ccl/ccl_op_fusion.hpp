// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/program/program.hpp"

namespace ttnn {
namespace ccl {

struct AllGatherFusedOpSignaler {

    std::vector<CoreCoord> fused_op_receiver_cores;
    std::vector<CoreCoord> fused_op_receiver_cores_noc;
    std::vector<uint32_t> fused_op_receiver_signal_semaphores;

    /* All Gather specific */
    std::vector<CoreCoord> all_gather_worker_cores_noc;
    uint32_t all_gather_worker_sync_semaphore;

    AllGatherFusedOpSignaler(
        std::vector<CoreCoord> fused_op_receiver_cores,
        std::vector<uint32_t> fused_op_receiver_signal_semaphores)
        : fused_op_receiver_cores(fused_op_receiver_cores),
            fused_op_receiver_signal_semaphores(fused_op_receiver_signal_semaphores) {

        }

    void set_fused_op_args(
        Device const* device
    ) {
        // Get the noc coords for the fused op receiver cores
        this->fused_op_receiver_cores_noc.clear();
        for (const auto& core : this->fused_op_receiver_cores) {
            this->fused_op_receiver_cores_noc.push_back(device->worker_core_from_logical_core(core));
        }
    }

    void set_all_gather_args(
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
    }

    void emit_all_gather_fused_op_ct_args(
        std::vector<uint32_t>& ct_args,

        uint32_t num_workers_to_sync,
        uint32_t curr_worker_index
    ) {

        ct_args.push_back(static_cast<uint32_t>(num_workers_to_sync));
        ct_args.push_back(static_cast<uint32_t>(curr_worker_index));
        ct_args.push_back(static_cast<uint32_t>(this->all_gather_worker_sync_semaphore));

    }


    void emit_all_gather_fused_op_rt_args(
        std::vector<uint32_t>& rt_args,

        bool all_gather_direction
    ) {

        // Push the worker core noc coords
        for (const auto& core : this->all_gather_worker_cores_noc) {
            rt_args.push_back(static_cast<uint32_t>(core.x));
            rt_args.push_back(static_cast<uint32_t>(core.y));
        }

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

};


}  // namespace ccl
}  // namespace ttnn
