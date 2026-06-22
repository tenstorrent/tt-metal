// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    uint32_t head_size = get_arg(args::head_size);
    uint32_t num_q_heads = get_arg(args::num_q_heads);
    uint32_t num_q_heads_per_core = get_arg(args::num_q_heads_per_core);
    uint32_t remote_q_head_start_idx = get_arg(args::remote_q_head_start_idx);
    uint32_t start_q_x = get_arg(args::start_q_x);
    uint32_t start_q_y = get_arg(args::start_q_y);
    // Source-shard base addresses are recovered from the typed tensor binding(s) (Case 2 bridge):
    // the legacy raw-uint32_t q_base_addr / kv_base_addr RTAs become host-computed *offsets* added
    // to the accessor base on the kernel side. q_offset is the within-shard write offset (RISC1).
    uint32_t q_start_offset = get_arg(args::q_start_offset);
    uint32_t q_offset = get_arg(args::q_offset);

    // NoC coordinates of the input shard cores arrive as common runtime varargs, laid out
    // [x0..x_{num_x-1}, y0..y_{num_y-1}]: get_common_vararg(x) for x-coords,
    // get_common_vararg(num_x + y) for y-coords.
    uint32_t num_x = get_arg(args::num_x);

    // Q input shard base (Case 2 bridge): get_bank_base_address() returns the per-shard local
    // address, identical across shard cores, which is exactly the legacy q_base_addr.
    const auto input_q = TensorAccessor(ta::input_q);
    const uint32_t q_base_addr = input_q.get_bank_base_address();

    DataflowBuffer cb_q_out(dfb::q_out);
    UnicastEndpoint src_ep;

    bool read_kv_heads = get_arg(args::read_kv_heads);

    uint32_t q_x = start_q_x;
    uint32_t q_y = start_q_y;
    uint32_t remote_q_head_idx = remote_q_head_start_idx;
    uint32_t q_src_noc_x = get_common_vararg(q_x);
    uint32_t q_src_noc_y = get_common_vararg(num_x + q_y);
    uint32_t q_src_addr = q_base_addr + q_start_offset;
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
                q_src_noc_x = get_common_vararg(q_x);
                q_src_noc_y = get_common_vararg(num_x + q_y);
                q_src_addr = q_base_addr;
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
        // KV-source base offsets (Case 2 bridge). When a separate KV input tensor is present its
        // base comes from its own accessor; otherwise the K/V source is the Q input shard.
        uint32_t kv_base_offset = get_arg(args::kv_base_offset);
        uint32_t kv_start_offset = get_arg(args::kv_start_offset);
        uint32_t num_kv_tiles = get_arg(args::num_kv_tiles);

#ifdef READ_FROM_INPUT_TENSOR_KV
        const auto input_kv = TensorAccessor(ta::input_kv);
        const uint32_t kv_source_base = input_kv.get_bank_base_address();
#else
        const uint32_t kv_source_base = q_base_addr;
#endif
        uint32_t kv_base_addr = kv_source_base + kv_base_offset;

        DataflowBuffer cb_kv_out(dfb::kv_out);

        uint32_t kv_x = start_kv_x;
        uint32_t kv_y = start_kv_y;
        uint32_t remote_kv_head_idx = remote_kv_head_start_idx;
        uint32_t kv_src_noc_x = get_common_vararg(kv_x);
        uint32_t kv_src_noc_y = get_common_vararg(num_x + kv_y);
        uint32_t kv_src_addr = kv_source_base + kv_start_offset;
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
                kv_src_noc_x = get_common_vararg(kv_x);
                kv_src_noc_y = get_common_vararg(num_x + kv_y);
                kv_src_addr = kv_base_addr;
            }
            noc.async_read_barrier();
        }
        cb_kv_out.push_back(num_kv_tiles);
    }
}
