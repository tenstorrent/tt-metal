// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"

// Copy this Ca-core's Cb partner shards (win_kv, win_gate, position_bias) from the
// partner's L1 into local CBs. The partner already holds valid WIDTH_SHARDED data
// before the op launches, so no handshake is required.
void kernel_main() {
    const uint32_t partner_noc_x = get_arg_val<uint32_t>(0);
    const uint32_t partner_noc_y = get_arg_val<uint32_t>(1);
    const uint32_t win_kv_addr = get_arg_val<uint32_t>(2);
    const uint32_t win_gate_addr = get_arg_val<uint32_t>(3);
    const uint32_t bias_addr = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_win_kv = get_compile_time_arg_val(0);
    constexpr uint32_t cb_win_gate = get_compile_time_arg_val(1);
    constexpr uint32_t cb_bias_cb = get_compile_time_arg_val(2);
    constexpr uint32_t win_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t bias_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(5);

    const uint32_t win_bytes = win_tiles * face_bytes;
    const uint32_t bias_bytes = bias_tiles * face_bytes;

    Noc noc;
    CircularBuffer win_kv_cb(cb_win_kv);
    CircularBuffer win_gate_cb(cb_win_gate);
    CircularBuffer bias_cb(cb_bias_cb);
    UnicastEndpoint partner;

    win_kv_cb.reserve_back(win_tiles);
    win_gate_cb.reserve_back(win_tiles);
    bias_cb.reserve_back(bias_tiles);

    noc.async_read(
        partner,
        CoreLocalMem<uint32_t>(win_kv_cb.get_write_ptr()),
        win_bytes,
        {.noc_x = partner_noc_x, .noc_y = partner_noc_y, .addr = win_kv_addr},
        {});
    noc.async_read(
        partner,
        CoreLocalMem<uint32_t>(win_gate_cb.get_write_ptr()),
        win_bytes,
        {.noc_x = partner_noc_x, .noc_y = partner_noc_y, .addr = win_gate_addr},
        {});
    noc.async_read(
        partner,
        CoreLocalMem<uint32_t>(bias_cb.get_write_ptr()),
        bias_bytes,
        {.noc_x = partner_noc_x, .noc_y = partner_noc_y, .addr = bias_addr},
        {});
    noc.async_read_barrier();

    win_kv_cb.push_back(win_tiles);
    win_gate_cb.push_back(win_tiles);
    bias_cb.push_back(bias_tiles);
}
