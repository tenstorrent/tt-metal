// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// pad writer.
//   dfb_pad_row   explicit sync (listed in this kernel's disable_dfb_implicit_sync_for); template writes are
//                 tagged with the user id trid 0, barrier on the same id.
//   dfb_out_rows  implicit sync: one tagged write per entry from the ring's cursor, the DM0 interrupt handler
//                 acks the entry; no wait / pop.
//
// Block operations (op_design.md, Block schedule):
//   store_pad_rows    iff the core has pad rows: wait the template, walk the output rows from
//                     pad_start_out_row (skipping whole data runs) and write every pad row of this core from
//                     the template, one write barrier per pad_rows_per_barrier writes; pop the template. Done
//                     first: pad rows depend only on the template, so they fill the stall before the first
//                     data block lands.
//   store_data_block  one implicit write per data row, to its output page.
//   release_trid      the hart's tag register is handed back (last statement): the implicit writes left a
//                     ring's id in it, and the next program's untagged writes would be counted for a ring.
//
// Thread w of W writes this core's pad rows w, w+W, ... (in the core's pad order) and the data rows the STRIDED
// ring deals it (pad_reader.cpp: reader p pushes its rows p, p+R, ... in ops of block_rows):
//   W >= R (W = m*R)  reader p = w mod R rotates its ops over writers p, p+R, ...: this thread takes that
//                     reader's ops k = w/R, w/R + m, ... whole;
//   R > W             this thread rotates over readers w, w+W, ... one entry at a time: its i-th entry is
//                     core-local row w + i*W.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t W = get_arg(args::writer_threads);

    const uint32_t data_start = get_arg(args::data_start);
    const uint32_t data_count = get_arg(args::data_count);
    const uint32_t pad_count = get_arg(args::pad_count);
    const uint32_t pad_start_out_row = get_arg(args::pad_start_out_row);
    const uint32_t pad_rows_per_barrier = get_arg(args::pad_rows_per_barrier);
    const uint32_t block_rows = get_arg(args::block_rows);
    const uint32_t out_row_bytes = get_arg(args::out_row_bytes);
    // Input row space (N, C, H), output row space (N_o, C_o, H_o), `before` offsets of the outer dims.
    const uint32_t N = get_arg(args::N);
    const uint32_t C = get_arg(args::C);
    const uint32_t H = get_arg(args::H);
    const uint32_t C_o = get_arg(args::C_o);
    const uint32_t H_o = get_arg(args::H_o);
    const uint32_t bn = get_arg(args::bn);
    const uint32_t bc = get_arg(args::bc);
    const uint32_t bh = get_arg(args::bh);

    const uint32_t w = get_my_thread_id();
    constexpr NocOptVals user_txn = {.trid = 0};

    DataflowBuffer dfb_out_rows(dfb::out_rows);
    DataflowBuffer dfb_pad_row(dfb::pad_row);
    Noc noc;
    const auto acc_out = TensorAccessor(tensor::output);

    // store_pad_rows
    if (pad_count > 0) {
        dfb_pad_row.wait_front(1);
        uint32_t r = pad_start_out_row;
        uint32_t h_o = r % H_o;
        uint32_t c_o = (r / H_o) % C_o;
        uint32_t n_o = r / (H_o * C_o);
        uint32_t in_flight = 0;
        for (uint32_t q = 0; q < pad_count;) {
            const bool in_box = n_o >= bn && n_o < bn + N && c_o >= bc && c_o < bc + C;
            uint32_t step = 1;
            if (in_box && h_o >= bh && h_o < bh + H) {
                step = bh + H - h_o;  // a data run: skip it whole (it ends at or before the end of the H range)
            } else {
                if (q % W == w) {
                    noc.async_write<NocOptions::TXN_ID>(
                        dfb_pad_row, acc_out, out_row_bytes, {.offset_bytes = 0}, {.page_id = r}, user_txn);
                    if (++in_flight == pad_rows_per_barrier) {
                        noc.async_write_barrier<NocOptions::TXN_ID>(user_txn);
                        in_flight = 0;
                    }
                }
                ++q;
            }
            r += step;
            h_o += step;
            if (h_o == H_o) {
                h_o = 0;
                if (++c_o == C_o) {
                    c_o = 0;
                    ++n_o;
                }
            }
        }
        noc.async_write_barrier<NocOptions::TXN_ID>(user_txn);
        dfb_pad_row.pop_front(1);
    }

    // store_data_block: data row i of the input lands on output row ((n+bn)*C_o + (c+bc))*H_o + (h+bh).
    // Counters advance by the dealing stride; one division per op.
    auto write_rows = [&](uint32_t first_row, uint32_t count, uint32_t step) {
        const uint32_t first = data_start + first_row;
        uint32_t h = first % H;
        uint32_t c = (first / H) % C;
        uint32_t n = first / (H * C);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t out_row = ((n + bn) * C_o + (c + bc)) * H_o + (h + bh);
            noc.async_write<NocOptions::TXN_ID>(dfb_out_rows, acc_out, {}, {.page_id = out_row});
            h += step;
            while (h >= H) {
                h -= H;
                if (++c == C) {
                    c = 0;
                    ++n;
                }
            }
        }
    };
    if constexpr (W >= R) {
        constexpr uint32_t m = W / R;
        const uint32_t p = w % R;
        const uint32_t reader_rows = data_count / R + (p < data_count % R ? 1u : 0u);
        for (uint32_t k = w / R; k * block_rows < reader_rows; k += m) {
            const uint32_t j0 = k * block_rows;
            const uint32_t rows = reader_rows - j0 < block_rows ? reader_rows - j0 : block_rows;
            write_rows(p + j0 * R, rows, R);
        }
    } else {
        write_rows(w, data_count / W + (w < data_count % W ? 1u : 0u), W);
    }

    // release_trid: firmware waits for the implicit writes to land when the kernel returns.
    noc_async_write_set_trid(0, noc.get_noc_id());
}
