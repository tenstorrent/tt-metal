// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    uint32_t head_size = get_arg(args::head_size);
    uint32_t num_q_heads = get_arg(args::num_q_heads);
    uint32_t num_q_heads_per_core = get_arg(args::num_q_heads_per_core);
    uint32_t remote_q_head_start_idx = get_arg(args::remote_q_head_start_idx);
    uint32_t start_q_x = get_arg(args::start_q_x);
    uint32_t start_q_y = get_arg(args::start_q_y);
    // Bare Q shard base (the same L1 address on every source core); the head start is added below.
    uint32_t q_base_addr = TensorAccessor(tensor::input_q).get_bank_base_address();
    uint32_t q_offset = get_arg(args::q_offset);

    uint32_t num_x = get_arg(args::num_x);
    // The NoC coordinate tables of the source grid ride the positional vararg block: num_x x-coordinates,
    // then the y-coordinates.
    auto in0_mcast_noc_x = [](uint32_t x) { return get_vararg(x); };
    auto in0_mcast_noc_y = [num_x](uint32_t y) { return get_vararg(num_x + y); };

    DataflowBuffer dfb_q_out(dfb::q_out);
    UnicastEndpoint src_ep;

    uint32_t q_x = start_q_x;
    uint32_t q_y = start_q_y;
    uint32_t remote_q_head_idx = remote_q_head_start_idx;
    // An instance with no Q heads to read is handed the coordinates the next instance starts from,
    // which on the last core lie one row past the coordinate tables; look them up only when used.
    uint32_t q_src_noc_x = 0;
    uint32_t q_src_noc_y = 0;
    if (num_q_heads > 0) {
        q_src_noc_x = in0_mcast_noc_x(q_x);
        q_src_noc_y = in0_mcast_noc_y(q_y);
    }
    // The host passes only the shard base; the first head this core reads starts
    // remote_q_head_start_idx heads into the source shard.
    uint32_t q_src_addr = q_base_addr + remote_q_head_start_idx * head_size;
    uint32_t q_write_addr = dfb_q_out.get_write_ptr() + q_offset;

    for (uint32_t q = 0; q < num_q_heads; ++q) {
        // Q
        noc.async_read(
            src_ep,
            CoreLocalMem<uint32_t>(q_write_addr),
            head_size,
            {.noc_x = q_src_noc_x, .noc_y = q_src_noc_y, .addr = q_src_addr},
            {});
        q_src_addr += head_size;
        q_write_addr += head_size;
        remote_q_head_idx++;
        // Advance to the next source core only while heads remain: after the last head the row wrap
        // would index the coordinate tables one row past their end.
        if (remote_q_head_idx == num_q_heads_per_core && q + 1 < num_q_heads) {
            remote_q_head_idx = 0;
            q_x++;
            if (q_x == num_x) {
                q_x = 0;
                q_y++;
                q_src_noc_x = in0_mcast_noc_x(q_x);
                q_src_noc_y = in0_mcast_noc_y(q_y);
                q_src_addr = q_base_addr;
            }
        }
        noc.async_read_barrier();
    }

#ifdef READ_KV_HEADS
    // K/V heads are read only on the cores that also hold a K/V output shard (the leading cores of the Q
    // grid, in row-major order).  The remaining Q cores are built without this block, and without the K/V
    // output buffer it fills.
    {
        uint32_t num_kv_heads = get_arg(args::num_kv_heads);
        uint32_t num_kv_heads_per_core = get_arg(args::num_kv_heads_per_core);
        uint32_t remote_kv_head_start_idx = get_arg(args::remote_kv_head_start_idx);
        uint32_t start_kv_x = get_arg(args::start_kv_x);
        uint32_t start_kv_y = get_arg(args::start_kv_y);
#ifdef READ_FROM_INPUT_TENSOR_KV
        uint32_t kv_buffer_addr = TensorAccessor(tensor::input_kv).get_bank_base_address();  // bare KV shard base
#else
        uint32_t kv_buffer_addr = q_base_addr;  // fused QKV: the K/V sections live in the Q shard
#endif
        uint32_t kv_section_offset = get_arg(args::kv_section_offset);  // byte offset of the K/V section in that shard
        uint32_t num_kv_tiles = get_arg(args::num_kv_tiles);

        DataflowBuffer dfb_kv_out(dfb::kv_out);

        uint32_t kv_x = start_kv_x;
        uint32_t kv_y = start_kv_y;
        uint32_t remote_kv_head_idx = remote_kv_head_start_idx;
        uint32_t kv_src_noc_x = in0_mcast_noc_x(kv_x);
        uint32_t kv_src_noc_y = in0_mcast_noc_y(kv_y);
        // Section base (K for the reader instance, V for the writer instance) and the first head
        // this core reads, both derived on device from the bare shard base.
        uint32_t kv_base_addr = kv_buffer_addr + kv_section_offset;
        uint32_t kv_src_addr = kv_base_addr + remote_kv_head_start_idx * head_size;
        dfb_kv_out.reserve_back(num_kv_tiles);
        uint32_t kv_write_addr = dfb_kv_out.get_write_ptr();

        // K or V
        for (uint32_t kv = 0; kv < num_kv_heads; ++kv) {
            noc.async_read(
                src_ep,
                CoreLocalMem<uint32_t>(kv_write_addr),
                head_size,
                {.noc_x = kv_src_noc_x, .noc_y = kv_src_noc_y, .addr = kv_src_addr},
                {});
            kv_src_addr += head_size;
            kv_write_addr += head_size;
            remote_kv_head_idx++;
            // As above: no source-core advance after the last head.
            if (remote_kv_head_idx == num_kv_heads_per_core && kv + 1 < num_kv_heads) {
                remote_kv_head_idx = 0;
                kv_x++;
                if (kv_x == num_x) {
                    kv_x = 0;
                    kv_y++;
                }
                kv_src_noc_x = in0_mcast_noc_x(kv_x);
                kv_src_noc_y = in0_mcast_noc_y(kv_y);
                kv_src_addr = kv_base_addr;
            }
            noc.async_read_barrier();
        }
        dfb_kv_out.push_back(num_kv_tiles);
    }
#endif
}
