// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// softmax reader (dim = -2, regime column_resident): load_block and build_h_mask.
//
// Reader hart p of R feeds NEOs p, p+R, ... (C/R of them) through the STRIDED rings, one entry each in strict
// rotation. NEO t's column in round r is column r*C + t of this cluster when it exists, else NEO 0's column of
// the round (every NEO takes the same number of entries; the writer re-stores the identical result).
// Per NEO the ring holds its columns column-major: column k's tile ht lands at entry k*Ht + ht. The reader does
// not need the block boundaries: the ring's capacity (in_depth whole blocks per NEO) throttles it.
//
// Two protocols in one kernel (references/quasar_op_examples/toy_thread_plan/kernels/reader_mixed.cpp):
//   cb_input_tiles  implicit sync: one tagged read per entry, the DM0 interrupt handler posts the credits.
//   cb_h_mask       explicit sync (listed in disable_dfb_implicit_sync_for): the hart writes the entry itself
//                   with local L1 stores, no NoC transfer. Built once per kernel per NEO. A full-tile store loop
//                   is used here on purpose: it runs once per program (not per block or per tile), and the NoC
//                   zero-fill would put an untagged transfer into a kernel that owns an explicit ring.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

namespace {

// Additive mask for the last tile-row of a column: rows < h_valid_last hold 0, the padding rows hold NEG_BIG,
// across all 32 columns. Tile layout: four 16x16 faces (f0 rows 0-15 cols 0-15, f1 rows 0-15 cols 16-31,
// f2 rows 16-31 cols 0-15, f3 rows 16-31 cols 16-31); a face row is 16 bf16 = 32 contiguous bytes.
template <uint32_t h_valid_last, uint32_t neg_big>
FORCE_INLINE void build_h_mask(uint32_t addr) {
    constexpr uint32_t face_row_bytes = 16 * sizeof(uint16_t);
    constexpr uint32_t face_bytes = 16 * face_row_bytes;
    for (uint32_t r = 0; r < 32; ++r) {
        const uint32_t value = r < h_valid_last ? 0u : neg_big;
        const uint32_t face_pair_base = addr + (r < 16 ? 0u : 2 * face_bytes) + (r % 16) * face_row_bytes;
        dataflow_kernel_lib::fill_l1_range<sizeof(uint16_t)>(face_pair_base, face_row_bytes, value);
        dataflow_kernel_lib::fill_l1_range<sizeof(uint16_t)>(face_pair_base + face_bytes, face_row_bytes, value);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t rounds = get_arg(args::rounds);
    const uint32_t num_cols = get_arg(args::num_cols);
    const uint32_t col_start = get_arg(args::col_start);
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t C = get_arg(args::compute_threads);
    constexpr uint32_t has_mask = get_arg(args::has_mask);
    constexpr uint32_t h_valid_last = get_arg(args::h_valid_last);
    constexpr uint32_t neg_big = get_arg(args::neg_big);
    constexpr uint32_t group = C / R;

    const uint32_t p = get_my_thread_id();

    DataflowBuffer in(dfb::cb_input_tiles);
    DataflowBuffer mask(dfb::cb_h_mask);
    Noc noc;
    const auto acc_in = TensorAccessor(tensor::input);

    if constexpr (has_mask) {
        for (uint32_t j = 0; j < group; ++j) {
            mask.reserve_back(1);
            const uint32_t addr = mask.get_write_ptr();
            build_h_mask<h_valid_last, neg_big>(addr);
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
            flush_l2_cache_range(addr, mask.get_tile_size());
#endif
            mask.push_back(1);
        }
    }

    // First page of column-unit u = nc*Wt + wt is (nc*Ht)*Wt + wt; tile-row ht adds ht*Wt.
    auto first_page_of = [&](uint32_t neo, uint32_t round) __attribute__((always_inline)) {
        const uint32_t local = round * C + neo;
        const uint32_t u = col_start + (local < num_cols ? local : round * C);
        const uint32_t nc = u / Wt;
        return nc * Ht * Wt + (u - nc * Wt);
    };

    for (uint32_t round = 0; round < rounds; ++round) {
        uint32_t page[group];
        for (uint32_t j = 0; j < group; ++j) {
            page[j] = first_page_of(p + j * R, round);
        }
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            for (uint32_t j = 0; j < group; ++j) {
                noc.async_read<NocOptions::TXN_ID>(acc_in, in, {.page_id = page[j] + ht * Wt}, {});
            }
        }
    }
    // Hand the hart back with the user id in its tag register (target page, ring protocols).
    noc_async_read_set_trid(0, noc.get_noc_id());
}
