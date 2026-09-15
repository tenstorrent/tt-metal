// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"

inline void llama_matmul_signal_reduce_scatter(const Noc& noc, uint32_t& rt_args_idx) {
    const uint32_t privileged_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t privileged_y = get_arg_val<uint32_t>(rt_args_idx++);
    Semaphore<> matmul_done(get_arg_val<uint32_t>(rt_args_idx++));
    const bool is_privileged = get_arg_val<uint32_t>(rt_args_idx++) != 0;

    if (is_privileged) {
        const uint32_t target = get_arg_val<uint32_t>(rt_args_idx++);
        const uint32_t num_receivers = get_arg_val<uint32_t>(rt_args_idx++);
        Semaphore<> reduce_scatter_ready(get_arg_val<uint32_t>(rt_args_idx++));
        matmul_done.wait(target);
        // Each receiver starts at INVALID (0) and receives exactly one signal.
        for (uint32_t i = 0; i < num_receivers; ++i) {
            const uint32_t x = get_arg_val<uint32_t>(rt_args_idx++);
            const uint32_t y = get_arg_val<uint32_t>(rt_args_idx++);
            reduce_scatter_ready.up(noc, x, y, 1);
        }
    } else {
        matmul_done.up(noc, privileged_x, privileged_y, 1);
    }
    noc.async_atomic_barrier();
}
