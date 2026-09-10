// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// writer_starvation bench — CANDIDATE compute: sub-tile-row output push.
//
// IDEA (a). `compute_kernel_lib::tilize` pushes `block_width_tiles` output
// pages ONCE per tile-row, after the whole row has been packed. The writer's
// `cb_wait_front` therefore cannot release until tile `bw-1` is in L1, so the
// first NoC write is issued after the LAST tile is packed. This kernel pushes
// every `push_tiles` tiles instead, so the writer can issue write #0 while the
// packer is still working on the row's trailing tiles.
//
// ─────────────────────────────────────────────────────────────────────────────
// RAW-LLK JUSTIFICATION — CAPABILITY gap, at two levels. Both are concrete and
// both are in a public signature, so a helper author can act on either.
//
//   1. `compute_kernel_lib::tilize` (tilize_helpers.inl:236-241) hardcodes the
//      handshake around the LLK call as
//          out_dfb.reserve_back(block_width_tiles);
//          fast_tilize_block(input_dfb, block_width_tiles, output_dfb);
//          out_dfb.push_back(block_width_tiles);
//      There is no push-granularity parameter anywhere in its template or
//      runtime signature. MISSING: a `push_tiles` (or `OutputPushMode`)
//      parameter that splits that reserve/push pair inside the row.
//
//   2. Even one level down, `ckernel::fast_tilize_block(icb, block, ocb,
//      in_idx, out_idx)` (api/compute/tilize.h) opens with
//          uint32_t full_dim = block;
//      — the ROW WIDTH the unpacker strides by is *derived from* the number of
//      tiles the call is asked to produce. So "tilize tiles [0,4) OF AN 8-WIDE
//      ROW" is not expressible through it at any argument combination: passing
//      block=4 also reprograms the source row stride to 4 tiles and reads the
//      wrong bytes. MISSING: `full_dim` as a parameter independent of `block`.
//
//      The LLK underneath already has exactly that separation —
//      `llk_unpack_fast_tilize_block(icb, tile_index, unit_dim, num_units,
//      full_dim)` takes `full_dim` and a column `tile_index` as distinct
//      arguments (llk_unpack_tilize_api.h:294; the address math is
//      `base + tile_index*TILE_C_DIM` with the Y-stride programmed from
//      `full_dim`, llk_unpack_tilize.h:759). This kernel is that call written
//      out, plus the matching math/pack pair and the finer CB push. It is a
//      strict re-decomposition of the SAME work at the SAME precision: the
//      init/uninit pair, `unit_dim`, the formats and the reconfig are all the
//      helper's own.
//
//      `push_tiles == block_width_tiles` reproduces the helper's decomposition
//      exactly for every width the focus regime uses; the bench runs that as
//      the `raw_full` CONTROL so any measured delta can be attributed to the
//      push granularity rather than to "raw LLK vs helper".
// ─────────────────────────────────────────────────────────────────────────────
//
// SECOND-ORDER EFFECT (expected, and why this is not only a writer story).
// A `bw <= 8` tile-row is ONE dest section in the helper's decomposition, so
// MATH fills all `bw` tiles and only then does PACK drain them — the two
// compute threads are serialized. Splitting the row into two dest sections
// lets PACK drain section 0 while MATH fills section 1 (half-sync dest has two
// banks). Whatever this kernel wins on that axis is real but is NOT the
// writer-starvation effect, which is why the bench also reports the
// `writer_wait_out` / BRISC-span shift and not just the wall.
//
// PRECISION IS UNTOUCHED: same `fast_tilize` path, same `unit_dim`, same
// formats, same `DST_ACCUM_MODE`, same reconfig. Output is bit-identical to the
// baseline and the bench gates on `torch.equal`.
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "bench_zone.hpp"

namespace {

constexpr uint32_t cb_input_rows = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
constexpr uint32_t block_width_tiles = get_compile_time_arg_val(2);
constexpr uint32_t tensor_row_blocks = get_compile_time_arg_val(3);
constexpr uint32_t num_row_groups = get_compile_time_arg_val(4);
constexpr uint32_t num_w_chunks = get_compile_time_arg_val(5);
constexpr uint32_t push_tiles = get_compile_time_arg_val(6);

// The helper's own `unit_dim` choice for Wormhole (api/compute/tilize.h:
// `unit_dim = full_dim == 1 ? 1 : 2`), reproduced so init and block agree.
constexpr uint32_t unit_dim = (block_width_tiles == 1) ? 1u : 2u;
constexpr uint32_t num_units = push_tiles / unit_dim;
// Half-sync dest holds 8 bf16 tiles (4 with fp32 dest accumulate) — the same
// `dest_size` the helper computes.
constexpr uint32_t dest_capacity_tiles = DST_ACCUM_MODE ? 4u : 8u;

static_assert(push_tiles >= 1 && push_tiles <= block_width_tiles, "push_tiles must be in [1, block_width_tiles]");
static_assert(block_width_tiles % push_tiles == 0, "push_tiles must divide block_width_tiles");
static_assert(push_tiles % unit_dim == 0, "push_tiles must be a multiple of unit_dim");
static_assert(push_tiles <= dest_capacity_tiles, "push_tiles must fit one dest section");
// This kernel writes out the FAST-tilize decomposition only. If the helper
// would not have taken that path for this (width, format, sync-mode) triple,
// the comparison would not be like-for-like, so fail the build instead.
static_assert(
    compute_kernel_lib::can_use_fast_tilize<block_width_tiles, cb_input_rows, cb_output_tiles>(),
    "writer_starvation fine-push variant reconstructs the fast_tilize path only");

}  // namespace

void kernel_main() {
    const uint32_t start_block_id = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(1);
    const uint32_t block_stride = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup(cb_input_rows, cb_output_tiles);

    // The helper's `UnpackAndPackReconfigure` prologue, verbatim
    // (tilize_helpers.inl:163-180): srcA + srcB (WH fast tilize uses both) and
    // the pack format. Held identical so the candidate pays what the baseline
    // pays.
    reconfig_data_format_srca(cb_input_rows);
    reconfig_data_format_srcb(cb_input_rows);
    pack_reconfig_data_format(cb_output_tiles);

    fast_tilize_init(cb_input_rows, block_width_tiles, cb_output_tiles);

    for (uint32_t b = 0; b < num_blocks_this_core; ++b) {
        const uint32_t block_id = start_block_id + b * block_stride;
        const uint32_t row_group = block_id / num_w_chunks;
        const uint32_t row_start = (row_group * tensor_row_blocks) / num_row_groups;
        const uint32_t row_end = ((row_group + 1) * tensor_row_blocks) / num_row_groups;
        const uint32_t block_row_extent = row_end - row_start;

        BenchZone("compute_tilize_block");
        for (uint32_t r = 0; r < block_row_extent; ++r) {
            // The INPUT quantum is unchanged — a whole tile-row, because a
            // tilize of fewer than 32 source rows is not a thing. Only the
            // OUTPUT push moves.
            cb_wait_front(cb_input_rows, block_width_tiles);

            for (uint32_t col = 0; col < block_width_tiles; col += push_tiles) {
                cb_reserve_back(cb_output_tiles, push_tiles);

                MATH((llk_math_wait_for_dest_available()));
                PACK((llk_packer_wait_for_math_done()));

                // `col` is the tile-COLUMN offset inside the row and
                // `block_width_tiles` stays the row stride — the separation the
                // public `fast_tilize_block` collapses.
                UNPACK((llk_unpack_fast_tilize_block(cb_input_rows, col, unit_dim, num_units, block_width_tiles)));
                MATH((llk_math_fast_tilize_block_(0, cb_input_rows, unit_dim, num_units)));
                // Output tile index 0: the reserve above has already advanced
                // `fifo_wr_ptr` past everything pushed, and the packer resolves
                // its L1 address from that pointer live
                // (llk_pack_common_api.h:70).
                PACK((llk_pack_fast_tilize_block(0, cb_output_tiles, 0, unit_dim, num_units)));

                MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
                PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));

                cb_push_back(cb_output_tiles, push_tiles);
            }

            cb_pop_front(cb_input_rows, block_width_tiles);
        }
    }

    fast_tilize_uninit(cb_input_rows, cb_output_tiles, block_width_tiles);
}
