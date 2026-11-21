// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// expand reader (regime R-H): load_source_block + replicate_rows_block.
//
// This cluster owns the run [run_start, run_start + run_tiles) of the source-major output order
// L = s * fan_out_tiles + r. The run touches the source tiles s_first..s_last; they are taken in blocks of
// block_src_tiles. Per block:
//   load_source_block     reserve block_src_tiles entries; one whole-page DRAM read per source of the block
//                         into its entry; one read barrier.
//   replicate_rows_block  build each entry in place into the tile whose 32 rows all equal its row 0: doubling
//                         self-aimed copies inside faces 0 and 1 (rows [0,1)->[1,2), [0,2)->[2,4), [0,4)->[4,8),
//                         [0,8)->[8,16)), then faces 0-1 -> faces 2-3. Each step is issued for every tile of
//                         the block before one barrier (5 barriers per block). Then one push of the nominal
//                         block (a short last block pushes unwritten entries too, so every push and pop has the
//                         same quantum and no block straddles the ring's wrap).
// The input tile's rows 1..31 are padding (R4); they are overwritten from row 0 before the push.
//
// No helper covers these steps: the DRAM reads are raw Noc calls through the TensorAccessor (helpers in
// kernel_lib are compute-side or L1->L1 only); the replication uses kernel_lib's local_addr for the self-aimed
// source. Its stateful read pair is not used: it bakes one size into the state, and the steps have five sizes.
//
// Explicit sync: the host opts the ring out of implicit sync, and every NoC transfer carries the user
// transaction id 0 (an untagged transfer would reuse the hart's sticky tag, see the target page).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

void kernel_main() {
    const uint32_t run_start = get_arg(args::run_start);
    const uint32_t run_tiles = get_arg(args::run_tiles);
    constexpr uint32_t block_src_tiles = get_arg(args::block_src_tiles);
    constexpr uint32_t fan_out_tiles = get_arg(args::fan_out_tiles);
    constexpr uint32_t R = get_arg(args::reader_threads);
    static_assert(R == 1, "regime R-H deals blocks to one reader thread");
    constexpr NocOptVals user_txn = {.trid = 0};

    DataflowBuffer ring(dfb::cb_row_bcast_tiles);
    Noc noc;
    UnicastEndpoint self;
    const auto acc_in = TensorAccessor(tensor::input);
    const uint32_t page_size = ring.get_tile_size();
    // A tile is four 16x16 faces, row-major inside each face: face 0 holds rows 0-15 of columns 0-15, face 1
    // rows 0-15 of columns 16-31, faces 2 and 3 rows 16-31.
    const uint32_t face_bytes = page_size / 4;
    const uint32_t face_row_bytes = face_bytes / 16;

    if (run_tiles == 0) {
        return;
    }
    const uint32_t s_first = run_start / fan_out_tiles;
    const uint32_t s_end = (run_start + run_tiles - 1) / fan_out_tiles + 1;

    for (uint32_t s0 = s_first; s0 < s_end; s0 += block_src_tiles) {
        const uint32_t n = (s_end - s0 < block_src_tiles) ? (s_end - s0) : block_src_tiles;

        // load_source_block. The input has one tile row per plane, so source s is input page s.
        ring.reserve_back(block_src_tiles);
        const uint32_t base = ring.get_write_ptr();
        for (uint32_t k = 0; k < n; ++k) {
            noc.async_read<NocOptions::TXN_ID>(
                acc_in, ring, page_size, {.page_id = s0 + k}, {.offset_bytes = k * page_size}, user_txn);
        }
        noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);

        // replicate_rows_block: rows [0, rows) -> [rows, 2*rows) in faces 0 and 1.
        for (uint32_t rows = 1; rows < 16; rows *= 2) {
            const uint32_t bytes = rows * face_row_bytes;
            for (uint32_t k = 0; k < n; ++k) {
                for (uint32_t face = 0; face < 2; ++face) {
                    const uint32_t face_off = k * page_size + face * face_bytes;
                    noc.async_read<NocOptions::TXN_ID>(
                        self,
                        ring,
                        bytes,
                        dataflow_kernel_lib::local_addr(base + face_off, noc.get_noc_id()),
                        {.offset_bytes = face_off + bytes},
                        user_txn);
                }
            }
            noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
        }
        // Faces 0-1 (rows 0-15, every row now row 0) -> faces 2-3 (rows 16-31).
        for (uint32_t k = 0; k < n; ++k) {
            noc.async_read<NocOptions::TXN_ID>(
                self,
                ring,
                2 * face_bytes,
                dataflow_kernel_lib::local_addr(base + k * page_size, noc.get_noc_id()),
                {.offset_bytes = k * page_size + 2 * face_bytes},
                user_txn);
        }
        noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
        ring.push_back(block_src_tiles);
    }
    // Hand the hart back with the user id in its sticky tag register.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
