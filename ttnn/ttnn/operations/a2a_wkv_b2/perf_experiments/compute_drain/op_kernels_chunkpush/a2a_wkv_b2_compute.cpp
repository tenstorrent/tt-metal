// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
//
// a2a_wkv_b2 worker compute — compute_drain candidate "chunkpush" (Perf 2 tournament).
//
// Per head, per tile row (mt_block == 1):
//   tilize_attn_block : cb_attn_rm (one Kt-wide RM tile row) -> cb_attn_tiled, pushed in TCH-tile column chunks
//   project_block     : [1 x Kt] @ W[h] [Kt x Nt] -> cb_out (Nt bf8_b tiles), K in DEST in one block; the
//                       K loop waits for cb_attn_tiled chunk by chunk, so the matmul starts as soon as the
//                       first tilized chunk is packed instead of after the whole row
// After the head's last block: release_weight_block (W retained across the head's blocks).
//
// Raw LLK instead of compute_kernel_lib::tilize (capability): the helper pushes the output once per block
// (the whole Kt-wide row), so the matmul waits for the LAST tilized tile. Here the BH fast-tilize sequence the
// helper itself runs (fast_tilize_block body: unpack/math/pack per <=4-tile unit) is issued one 4-tile column
// chunk at a time (unpack column offset c*4 of the full-width row, one fast_tilize_init per row with
// full_dim = Kt) and each chunk is pushed as soon as it is packed.
// Raw LLK instead of compute_kernel_lib::matmul_block (capability): the helper waits for the whole in0
// K-block before the first K step. Here the K loop waits per tilize chunk; same LLK calls, same DEST
// accumulation order => bit-identical output. Measured (single-core bench, rows landing every 1.9 us like the
// critical cores): helpers 2.41 us per tile row, this 2.26 us; last block written 0.63 us earlier.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace {
constexpr uint32_t cb_attn_rm = get_compile_time_arg_val(0);
constexpr uint32_t cb_attn_tiled = get_compile_time_arg_val(1);
constexpr uint32_t cb_weight = get_compile_time_arg_val(2);
constexpr uint32_t cb_out = get_compile_time_arg_val(3);
constexpr uint32_t Kt = get_compile_time_arg_val(4);
constexpr uint32_t Nt = get_compile_time_arg_val(5);
constexpr uint32_t Mt = get_compile_time_arg_val(6);
constexpr uint32_t mt_block = get_compile_time_arg_val(7);
constexpr uint32_t TCH = 4;  // BH fast-tilize unit (tiles per chunk)
static_assert(mt_block == 1, "chunkpush compute handles one tile row per block");
static_assert(Kt % TCH == 0, "Kt must be a multiple of the fast-tilize unit");

ALWI void tilize_row_chunked() {
    reconfig_data_format_srca(cb_attn_rm);
    pack_reconfig_data_format(cb_attn_tiled);
    fast_tilize_init(cb_attn_rm, Kt, cb_attn_tiled);
    cb_wait_front(cb_attn_rm, Kt);
    for (uint32_t c = 0; c < Kt / TCH; ++c) {
        cb_reserve_back(cb_attn_tiled, TCH);
        PACK((llk_pack_fast_tilize_row_begin(cb_attn_tiled, 0)));
        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));
        UNPACK((llk_unpack_fast_tilize_block(cb_attn_rm, 0, TCH, c * TCH)));
        MATH((llk_math_fast_tilize_block_<DST_ACCUM_MODE>(0, cb_attn_rm, 4)));
        PACK((llk_pack_fast_tilize_row_chunk(0, cb_attn_tiled, TCH)));
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_pack_fast_tilize_row_end()));
        cb_push_back(cb_attn_tiled, TCH);
    }
    cb_pop_front(cb_attn_rm, Kt);
    fast_tilize_uninit(cb_attn_rm, cb_attn_tiled, Kt);
}

ALWI void project_row_chunk_wait() {
    reconfig_data_format(cb_weight, cb_attn_tiled);  // srcA <- W (in1), srcB <- attn (in0)
    pack_reconfig_data_format(cb_out);
    ckernel::matmul_block_init(cb_attn_tiled, cb_weight, false, Nt, 1, Kt);
    tile_regs_acquire();
    for (uint32_t k = 0; k < Kt; ++k) {
        if (k % TCH == 0) {
            cb_wait_front(cb_attn_tiled, k + TCH);
        }
        ckernel::matmul_block(cb_attn_tiled, cb_weight, k, k * Nt, 0, false, Nt, 1, Kt);
    }
    tile_regs_commit();
    cb_reserve_back(cb_out, Nt);
    tile_regs_wait();
    pack_block(0, cb_out, Nt);
    tile_regs_release();
    cb_push_back(cb_out, Nt);
    cb_pop_front(cb_attn_tiled, Kt);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t num_mt_blocks = Mt;

    // Common runtime args only (launch gap, Perf 1): [grid_x, per logical core index: head_start | num_heads << 8]
    const uint32_t num_heads =
        (get_common_arg_val<uint32_t>(
             1 + get_relative_logical_y() * get_common_arg_val<uint32_t>(0) + get_relative_logical_x()) >>
         8) &
        0xFF;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_attn_tiled, cb_weight, cb_out);

    for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
        for (uint32_t mb = 0; mb < num_mt_blocks; ++mb) {
#ifdef A2A_ABL_NO_COMPUTE
            {
                MaybeDeviceZoneScope("compute_tilize");
                cb_wait_front(cb_attn_rm, Kt);
                cb_pop_front(cb_attn_rm, Kt);
            }
            {
                MaybeDeviceZoneScope("compute_matmul");
                if (mb == 0) {
                    cb_wait_front(cb_weight, Kt * Nt);
                }
                cb_reserve_back(cb_out, Nt);
                cb_push_back(cb_out, Nt);
            }
#else
            {
                // unpack: waits on the reader (cb_attn_rm); pack: on cb_attn_tiled space
                MaybeDeviceZoneScope("compute_tilize");
                tilize_row_chunked();
            }
            {
                // unpack: waits on W (first block of a head) + each tiled chunk; pack: cb_out space
                MaybeDeviceZoneScope("compute_matmul");
                if (mb == 0) {
                    cb_wait_front(cb_weight, Kt * Nt);
                }
                project_row_chunk_wait();
            }
#endif
        }
        // release_weight_block
        cb_pop_front(cb_weight, Kt * Nt);
    }
}
