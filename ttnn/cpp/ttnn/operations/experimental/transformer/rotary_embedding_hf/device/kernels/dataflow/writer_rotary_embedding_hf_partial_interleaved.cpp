// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Partial-rotary writer (see reader_rotary_embedding_hf_partial_interleaved.cpp): per row, the Wt rotated tiles from
// the output CB, then the ROT_WT_IN - Wt pass-through tiles from PASS_CB, to consecutive output tiles.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_rows = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t output_cb_id = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    constexpr uint32_t Wt_in = ROT_WT_IN;
    constexpr uint32_t Wt = ROT_WT;
    constexpr uint32_t Pt = Wt_in - Wt;
    constexpr uint32_t pass_cb_id = PASS_CB;

    const uint32_t output_tile_bytes = get_tile_size(output_cb_id);
    const auto s = TensorAccessor(dst_args, dst_addr, output_tile_bytes);
    CircularBuffer cb_output(output_cb_id);
    CircularBuffer cb_pass(pass_cb_id);

    uint32_t out_id = start_id;
    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {
            cb_output.wait_front(1);
            noc.async_write(
                CoreLocalMem<uint32_t>(cb_output.get_read_ptr()), s, output_tile_bytes, {}, {.page_id = out_id++});
            noc.async_write_barrier();
            cb_output.pop_front(1);
        }
        cb_pass.wait_front(Pt);
        uint32_t src = cb_pass.get_read_ptr();
        for (uint32_t p = 0; p < Pt; ++p) {
            noc.async_write(CoreLocalMem<uint32_t>(src), s, output_tile_bytes, {}, {.page_id = out_id++});
            src += output_tile_bytes;
        }
        noc.async_write_barrier();
        cb_pass.pop_front(Pt);
    }
}
