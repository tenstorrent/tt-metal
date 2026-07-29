// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

// Named binding tokens (this source is instantiated twice — reader-config and writer-config):
//   dfb::q_out  = Q output CB (legacy c_16, borrowed from the Q output buffer), written by both instances
//                 at disjoint offsets (1P+1C dual-instance work-split).
//   dfb::kv_out = K output CB (c_17) for the reader-config instance / V output CB (c_18) for the writer-config
//                 instance (borrowed from the K resp. V output buffer; self-loop).
//   tensor::in_q  = clean Q input buffer base (Case 2 raw-pointer binding).
//   tensor::in_kv = clean K/V input buffer base (the KV tensor if present, else the fused Q tensor).
// The per-remote-core NoC coordinate arrays arrive as runtime varargs: [noc_x[0..num_x), noc_y[0..num_y)].

void kernel_main() {
    Noc noc;

    uint32_t head_size = get_arg(args::head_size);
    uint32_t num_q_heads = get_arg(args::num_q_heads);
    uint32_t num_q_heads_per_core = get_arg(args::num_q_heads_per_core);
    uint32_t remote_q_head_start_idx = get_arg(args::remote_q_head_start_idx);
    uint32_t start_q_x = get_arg(args::start_q_x);
    uint32_t start_q_y = get_arg(args::start_q_y);
    uint32_t q_region_offset = get_arg(args::q_region_offset);  // byte offset of Q within the input shard (0)
    uint32_t q_offset = get_arg(args::q_offset);                // L1 write offset (destination CB)

    // Clean Q input buffer base via the Case 2 binding (was a host-folded base RTA); the raw walk below is
    // unchanged — rebuild the source address from base + separately-passed region/head offsets.
    TensorAccessor in_q(tensor::in_q);
    uint32_t q_base_addr = in_q.get_bank_base_address();

    bool read_kv_heads = get_arg(args::read_kv_heads);

    uint32_t num_x = get_arg(args::num_x);

    DataflowBuffer cb_q_out(dfb::q_out);
    UnicastEndpoint src_ep;

    uint32_t q_x = start_q_x;
    uint32_t q_y = start_q_y;
    uint32_t remote_q_head_idx = remote_q_head_start_idx;
    uint32_t q_src_noc_x = get_vararg(q_x);
    uint32_t q_src_noc_y = get_vararg(num_x + q_y);
    uint32_t q_region_base = q_base_addr + q_region_offset;
    uint32_t q_src_addr = q_region_base + remote_q_head_start_idx * head_size;
    uint32_t q_write_addr = cb_q_out.get_write_ptr() + q_offset;

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
        if (remote_q_head_idx == num_q_heads_per_core) {
            remote_q_head_idx = 0;
            q_x++;
            if (q_x == num_x) {
                q_x = 0;
                q_y++;
                q_src_noc_x = get_vararg(q_x);
                q_src_noc_y = get_vararg(num_x + q_y);
                q_src_addr = q_region_base;
            }
        }
        noc.async_read_barrier();
    }

    if (read_kv_heads) {
        uint32_t num_kv_heads = get_arg(args::num_kv_heads);
        uint32_t num_kv_heads_per_core = get_arg(args::num_kv_heads_per_core);
        uint32_t remote_kv_head_start_idx = get_arg(args::remote_kv_head_start_idx);
        uint32_t start_kv_x = get_arg(args::start_kv_x);
        uint32_t start_kv_y = get_arg(args::start_kv_y);
        uint32_t kv_region_offset = get_arg(args::kv_region_offset);  // byte offset of K/V within the input shard
        uint32_t num_kv_tiles = get_arg(args::num_kv_tiles);

        // Clean K/V input buffer base via the Case 2 binding.
        TensorAccessor in_kv(tensor::in_kv);
        uint32_t kv_base_addr = in_kv.get_bank_base_address();

        DataflowBuffer cb_kv_out(dfb::kv_out);

        uint32_t kv_x = start_kv_x;
        uint32_t kv_y = start_kv_y;
        uint32_t remote_kv_head_idx = remote_kv_head_start_idx;
        uint32_t kv_src_noc_x = get_vararg(kv_x);
        uint32_t kv_src_noc_y = get_vararg(num_x + kv_y);
        uint32_t kv_region_base = kv_base_addr + kv_region_offset;
        uint32_t kv_src_addr = kv_region_base + remote_kv_head_start_idx * head_size;
        cb_kv_out.reserve_back(num_kv_tiles);
        uint32_t kv_write_addr = cb_kv_out.get_write_ptr();

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
            if (remote_kv_head_idx == num_kv_heads_per_core) {
                remote_kv_head_idx = 0;
                kv_x++;
                if (kv_x == num_x) {
                    kv_x = 0;
                    kv_y++;
                }
                kv_src_noc_x = get_vararg(kv_x);
                kv_src_noc_y = get_vararg(num_x + kv_y);
                kv_src_addr = kv_region_base;
            }
            noc.async_read_barrier();
        }
        cb_kv_out.push_back(num_kv_tiles);
    }
}
