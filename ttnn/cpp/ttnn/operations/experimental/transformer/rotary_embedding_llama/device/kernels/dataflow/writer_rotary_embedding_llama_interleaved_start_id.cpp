// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    auto batch_start = get_arg(args::batch_start);
    auto batch_end = get_arg(args::batch_end);
    auto seq_t_start = get_arg(args::seq_t_start);
    auto seq_t_end = get_arg(args::seq_t_end);

    constexpr auto n_heads = get_arg(args::n_heads);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto Ht = get_arg(args::Ht);
    constexpr auto rotary_Ht = get_arg(args::rotary_Ht);

    const auto s = TensorAccessor(tensor::output);

    DataflowBuffer dfb_out(dfb::out);
    // Zero-fill staging region (Wt tiles). Formerly a self-looped DFB — this kernel bound both
    // PRODUCER and CONSUMER, a shape Gen2 rejects for DM kernels — converted to a Scratchpad per
    // dm_self_loop_dfbs.md. The fake-FIFO calls translated away entirely: both FIFO pointers were
    // only ever read at their initial (base) position, so no index or wrap survives.
    Scratchpad<volatile uint32_t> zero_stage(scratch::zero);

    const uint32_t tile_bytes = dfb_out.get_entry_size();
    // Per-tile stride/size within the zero region. Its entries share `out`'s entry size: every
    // binding factory sets both to output_single_tile_size.
    const uint32_t zero_tile_bytes = tile_bytes;

    for (uint32_t i = 0; i < zero_stage.size(); ++i) {
        zero_stage[i] = 0;
    }

    for (uint32_t batch_id = batch_start; batch_id < batch_end; ++batch_id) {
        for (uint32_t head_num = 0; head_num < n_heads; ++head_num) {
            for (uint32_t seq_tile = seq_t_start; seq_tile < seq_t_end; ++seq_tile) {
                uint32_t output_curr_idx = batch_id * n_heads * Ht * Wt + head_num * Ht * Wt + seq_tile * Wt;
                const bool write_rotary_output = seq_tile < rotary_Ht;
                if (write_rotary_output) {
                    dfb_out.wait_front(Wt);
                }

                uint32_t l1_read_addr = write_rotary_output ? dfb_out.get_read_ptr() : zero_stage.get_base_address();
                const uint32_t l1_read_stride = write_rotary_output ? tile_bytes : zero_tile_bytes;
                const uint32_t write_bytes = write_rotary_output ? tile_bytes : zero_tile_bytes;
                for (uint32_t j = 0; j < Wt; j++) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(l1_read_addr), s, write_bytes, {}, {.page_id = output_curr_idx});
                    l1_read_addr += l1_read_stride;
                    output_curr_idx++;
                }
                noc.async_write_barrier();

                if (write_rotary_output) {
                    dfb_out.pop_front(Wt);
                }
            }
        }
    }
}
