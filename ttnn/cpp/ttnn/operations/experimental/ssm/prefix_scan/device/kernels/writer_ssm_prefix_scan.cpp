// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t NUM_TILES_IN_TILIZED_CHUNK = 32;
constexpr uint32_t NUM_BYTES_IN_BFLOAT16 = 2;
constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t NUM_BYTES_IN_TILIZED_CHUNK = NUM_TILES_IN_TILIZED_CHUNK * TILE_WIDTH * NUM_BYTES_IN_BFLOAT16;

void kernel_main() {
    Noc noc;

    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(0);
    const uint32_t hidden_state_len = get_arg_val<uint32_t>(1);
    const uint32_t h_shard_l1_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_h_acc = get_compile_time_arg_val(1);

    CircularBuffer cb_out_obj(cb_out);
    CircularBuffer cb_h_acc_obj(cb_h_acc);

    const uint32_t hidden_state_len_bytes = hidden_state_len * NUM_BYTES_IN_BFLOAT16;

    cb_out_obj.wait_front(num_tiles_per_core);

    // Write accumulated hidden state back to the h_prev shard (in-place update).
    // The shard lives in this core's own L1, so the write is a NOC unicast to the local
    // core's coordinates (mirrors the reader's UnicastEndpoint read of the same shard).
    const uint32_t local_noc_x = my_x[noc.get_noc_id()];
    const uint32_t local_noc_y = my_y[noc.get_noc_id()];
    CoreLocalMem<uint32_t> h_acc_src(cb_h_acc_obj.get_read_ptr());
    UnicastEndpoint h_shard_dst;
    noc.async_write(
        h_acc_src,
        h_shard_dst,
        hidden_state_len_bytes,
        {},
        {.noc_x = local_noc_x, .noc_y = local_noc_y, .addr = h_shard_l1_addr});
    noc.async_write_barrier();

    // cb_out is waited only as an output-ready handshake (the write above sources from cb_h_acc);
    // pop it so the CB is left balanced.
    cb_out_obj.pop_front(num_tiles_per_core);
}
