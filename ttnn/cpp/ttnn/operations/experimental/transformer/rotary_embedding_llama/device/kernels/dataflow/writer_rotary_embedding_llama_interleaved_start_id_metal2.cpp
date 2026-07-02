// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of writer_rotary_embedding_llama_interleaved_start_id.cpp. The legacy writer is
// still bound by the PrefillSharded factory on the ProgramDescriptor path, so the Metal 2.0
// MultiCore (interleaved) factory binds this forked copy with named args (args::), DFB handles
// (dfb::), and a typed output tensor binding (TensorAccessor(tensor::output)). Behavior is
// identical to the legacy kernel — only the CB/arg/accessor idioms differ.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    uint32_t batch_start = get_arg(args::batch_start);
    uint32_t batch_end = get_arg(args::batch_end);
    uint32_t seq_t_start = get_arg(args::seq_t_start);
    uint32_t seq_t_end = get_arg(args::seq_t_end);

    constexpr auto cb_id_out = dfb::out;
    constexpr auto cb_id_zero = dfb::zero;
    constexpr uint32_t n_heads = get_arg(args::n_heads);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t rotary_Ht = get_arg(args::rotary_Ht);

    DataflowBuffer cb_out(cb_id_out);
    DataflowBuffer cb_zero(cb_id_zero);

    const uint32_t tile_bytes = cb_out.get_entry_size();
    const uint32_t zero_tile_bytes = cb_zero.get_entry_size();
    const auto s = TensorAccessor(tensor::output);

    // The reader fills the Wt zero tiles (DM kernels cannot self-loop a DFB on Gen1); the writer
    // consumes them once and reuses the same Wt tiles for every zero-fill tail tile.
    cb_zero.wait_front(Wt);

    for (uint32_t batch_id = batch_start; batch_id < batch_end; ++batch_id) {
        for (uint32_t head_num = 0; head_num < n_heads; ++head_num) {
            for (uint32_t seq_tile = seq_t_start; seq_tile < seq_t_end; ++seq_tile) {
                uint32_t output_curr_idx = batch_id * n_heads * Ht * Wt + head_num * Ht * Wt + seq_tile * Wt;
                const bool write_rotary_output = seq_tile < rotary_Ht;
                if (write_rotary_output) {
                    cb_out.wait_front(Wt);
                }

                uint32_t l1_read_addr = write_rotary_output ? cb_out.get_read_ptr() : cb_zero.get_read_ptr();
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
                    cb_out.pop_front(Wt);
                }
            }
        }
    }

    cb_zero.pop_front(Wt);
}
