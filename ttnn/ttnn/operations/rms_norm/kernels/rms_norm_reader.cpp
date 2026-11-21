// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm reader: hart p of R feeds NEOs p, p+R, ... (C/R of them, its `group`) through three STRIDED rings,
// one entry (input) or one block (gamma) each, in strict rotation.
//
// Unit of work: `rows_per_block` tile-rows of Wt tiles. NEO t's unit in round k is unit k*C + t of this
// cluster when it exists, else NEO 0's unit of that round (every NEO takes the same entries; the writer
// stores the recomputed unit again, same bytes, same pages). A row of a unit past the tensor's last tile-row
// (a ragged last unit) maps to the unit's first row, the same way.
//
//   input_tiles  implicit sync: one tagged read per tile, page row*Wt + w; the unit stays resident in the
//                ring across both compute passes (capacity per NEO = exactly one unit).
//   gamma_tiles  explicit sync (opted out on the host): an entry is assembled from two whole-page reads of
//                the weight's row w (64 B, gamma of tile column w) -> entry + 0 (face 0 row 0 = cols 0..15)
//                and entry + 480 (its bytes 32..63 land at +512 = face 1 row 0 = cols 16..31); everything
//                else in the entry is don't-care for a row broadcast. Two NoC reads instead of a RISC copy:
//                the DM cores' write-back caches would otherwise need a flush. Entry i of a reserved block
//                sits at i * stride (a STRIDED ring interleaves the NEOs' entries).
//   scaler       explicit sync: one bf16 SUM scaler tile (1.0) per NEO, once, via the reduce-scaler helper.
//
// Overlap: after pushing gamma block j of unit k, the hart issues unit k+1's input reads of the same tiles;
// their slots are freed by pass 2 of unit k as it consumes that gamma block.
//
// Every transfer is tagged: the input reads with the ring's id, the gamma reads with the user id 0 (barrier
// on the same id). The last statement hands the tag register back (target page, ring protocols).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    const uint32_t num_units = get_arg(args::num_units);
    const uint32_t unit_start = get_arg(args::unit_start);
    const uint32_t total_rows = get_arg(args::total_rows);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t rows_per_block = get_arg(args::rows_per_block);
    constexpr uint32_t gamma_block = get_arg(args::gamma_block_tiles);
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t group = C / R;
    constexpr uint32_t gamma_blocks_per_row = Wt / gamma_block;
    static_assert(Wt % gamma_block == 0, "gamma_block_tiles must divide Wt");
    static_assert(C % R == 0, "R must divide C");

    // One weight page is the 32 gammas of a tile column; it lands as rows 0..1 of face 0 of the entry, and its
    // second half goes again to row 0 of face 1 (offset one face). Both sizes come from the host.
    constexpr uint32_t gamma_page_bytes = get_arg(args::gamma_page_bytes);
    constexpr uint32_t face1_row0_offset = get_arg(args::gamma_face_bytes);
    constexpr uint32_t face_row_bytes = face1_row0_offset / 16;
    static_assert(
        gamma_page_bytes == 2 * face_row_bytes,
        "a weight page must be two face rows of the gamma entry's format (32 gammas, one per column)");
    constexpr uint32_t second_read_offset = face1_row0_offset - gamma_page_bytes / 2;
    constexpr NocOptVals user_txn = {.trid = 0};

    const uint32_t p = get_my_thread_id();
    const uint32_t rounds = (num_units + C - 1) / C;

    Noc noc;
    DataflowBuffer input_tiles(dfb::input_tiles);
    DataflowBuffer gamma_tiles(dfb::gamma_tiles);
    DataflowBuffer scaler(dfb::scaler);
    const auto acc_input = TensorAccessor(tensor::input);
    const auto acc_weight = TensorAccessor(tensor::weight);
    const uint32_t gamma_stride = gamma_tiles.get_stride_size();

    // Tile-row r of NEO `neo`'s unit in round k.
    auto row_of = [&](uint32_t neo, uint32_t k, uint32_t r) __attribute__((always_inline)) {
        const uint32_t local = k * C + neo;
        const uint32_t first_row = (unit_start + (local < num_units ? local : k * C)) * rows_per_block;
        const uint32_t row = first_row + r;
        return row < total_rows ? row : first_row;
    };

    // Input reads of round k, tiles [w_begin, w_end) of tile-row r, for every NEO of the group in rotation.
    auto load_input = [&](uint32_t k, uint32_t r, uint32_t w_begin, uint32_t w_end) __attribute__((always_inline)) {
        for (uint32_t w = w_begin; w < w_end; ++w) {
            for (uint32_t j = 0; j < group; ++j) {
                noc.async_read<NocOptions::TXN_ID>(
                    acc_input, input_tiles, {.page_id = row_of(p + j * R, k, r) * Wt + w}, {});
            }
        }
    };

    // prepare_scaler: one tile per NEO of the group (each call reserves and pushes one entry: rotation).
    for (uint32_t j = 0; j < group; ++j) {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            dfb::scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW>(scaler);
    }

    if (rounds > 0) {
        for (uint32_t r = 0; r < rows_per_block; ++r) {
            load_input(0, r, 0, Wt);  // load_row_block of round 0
        }
    }
    for (uint32_t k = 0; k < rounds; ++k) {
        const bool prefetch_next = (k + 1) < rounds;
        for (uint32_t r = 0; r < rows_per_block; ++r) {
            for (uint32_t b = 0; b < gamma_blocks_per_row; ++b) {
                const uint32_t w0 = b * gamma_block;
                // load_gamma_block: the same gammas for every NEO (gamma does not vary along tile-rows).
                for (uint32_t j = 0; j < group; ++j) {
                    gamma_tiles.reserve_back(gamma_block);
                    for (uint32_t i = 0; i < gamma_block; ++i) {
                        const uint32_t entry = i * gamma_stride;
                        noc.async_read<NocOptions::TXN_ID>(
                            acc_weight, gamma_tiles, gamma_page_bytes, {.page_id = w0 + i}, {.offset_bytes = entry},
                            user_txn);
                        noc.async_read<NocOptions::TXN_ID>(
                            acc_weight,
                            gamma_tiles,
                            gamma_page_bytes,
                            {.page_id = w0 + i},
                            {.offset_bytes = entry + second_read_offset},
                            user_txn);
                    }
                    noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
                    gamma_tiles.push_back(gamma_block);
                }
                if (prefetch_next) {
                    load_input(k + 1, r, w0, w0 + gamma_block);  // load_row_block of round k+1, interleaved
                }
            }
        }
    }
    // Hand the hart back with the user id in its tag register: a program that runs next with untagged
    // transfers would otherwise inherit a ring's id.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
