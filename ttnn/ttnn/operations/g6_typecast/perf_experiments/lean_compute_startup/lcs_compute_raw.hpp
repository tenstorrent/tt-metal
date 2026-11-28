// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// lean_compute_startup — raw-LLK g6_typecast compute (bake-off candidate).
//
// Helper bypassed: compute_kernel_lib::eltwise_chain (CopyTile -> Typecast -> PackTile).
// Mechanism exploited (selected by the including kernel's macros):
//   LCS_SKIP_NOOP_SFPU : arms whose typecast_tile emits no SFPU work (bf16->fp32, bf16->bf8b, fp32->bf8b,
//                        int32<->uint32, ...) skip typecast_tile_init too (the SFPU config/addr-mod init
//                        is dead code for them) — the chain always emits the element's init().
//   LCS_PACK_BLOCK     : one pack_block(0, out, n) instead of n pack_tile calls.
//   (always)           : no data-format reconfig after compute_kernel_hw_startup (the hw_startup already
//                        programmed unpack/math/pack for exactly these two CBs; the chain's first call
//                        re-emits reconfig_data_format_srca + pack_reconfig_data_format).
// Precision contract untouched: same compute config, same LLK conversion calls (typecast_tile<in,out>),
// same int32->bf16 two-step composition (Int32->Float32 then Float32->Float16_b, both RNE).
#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "g6_typecast_zones.hpp"

namespace lcs {
constexpr uint32_t df_bf16 = static_cast<uint32_t>(DataFormat::Float16_b);
constexpr uint32_t df_fp32 = static_cast<uint32_t>(DataFormat::Float32);
constexpr uint32_t df_bf8b = static_cast<uint32_t>(DataFormat::Bfp8_b);
constexpr uint32_t df_int32 = static_cast<uint32_t>(DataFormat::Int32);
constexpr uint32_t df_uint32 = static_cast<uint32_t>(DataFormat::UInt32);

// Pairs for which typecast_tile<in,out> emits no SFPU instruction (conversion is done by unpacker/packer).
constexpr bool sfpu_noop(uint32_t i, uint32_t o) {
    return (i == df_bf16 && o == df_fp32) || (i == df_bf16 && o == df_bf8b) || (i == df_fp32 && o == df_bf8b) ||
           (i == df_bf8b && (o == df_bf16 || o == df_fp32)) || (i == df_int32 && o == df_uint32) ||
           (i == df_uint32 && o == df_int32) || i == o;
}
// L1 format of a CB as every TRISC sees it (pack thread only has pack_dst_format).
template <uint32_t cb>
constexpr uint32_t l1_format() {
#if defined(UCK_CHLKC_PACK)
    return pack_dst_format[cb];
#else
    return unpack_src_format[cb];
#endif
}
}  // namespace lcs

#ifndef LCS_SKIP_NOOP_SFPU
#define LCS_SKIP_NOOP_SFPU 0
#endif
#ifndef LCS_PACK_BLOCK
#define LCS_PACK_BLOCK 0
#endif

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);
    const uint32_t core_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t in_df = lcs::l1_format<cb_in>();
    constexpr uint32_t out_df = lcs::l1_format<cb_out>();
    constexpr bool two_step = in_df == lcs::df_int32 && out_df == lcs::df_bf16;
    constexpr bool skip_sfpu = LCS_SKIP_NOOP_SFPU && lcs::sfpu_noop(in_df, out_df);
    // DEST half capacity (SyncHalf): 4 tiles with 32-bit DEST, 8 with 16-bit.
    constexpr uint32_t dest_cap = DST_ACCUM_MODE ? 4 : 8;

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);
    if constexpr (!two_step && !skip_sfpu) {
        typecast_tile_init<in_df, out_df>();
    }

    G6_ZONE("CONVERT");
    for (uint32_t done = 0; done < core_tiles; done += block_tiles) {
        const uint32_t n = (core_tiles - done < block_tiles) ? (core_tiles - done) : block_tiles;
        cb_wait_front(cb_in, n);
        cb_reserve_back(cb_out, n);
        for (uint32_t s = 0; s < n; s += dest_cap) {
            const uint32_t m = (n - s < dest_cap) ? (n - s) : dest_cap;
            tile_regs_acquire();
            for (uint32_t i = 0; i < m; ++i) {
                copy_tile(cb_in, s + i, i);
                if constexpr (two_step) {
                    typecast_tile_init<lcs::df_int32, lcs::df_fp32>();
                    typecast_tile<lcs::df_int32, lcs::df_fp32>(i);
                    typecast_tile_init<lcs::df_fp32, lcs::df_bf16>();
                    typecast_tile<lcs::df_fp32, lcs::df_bf16>(i);
                } else if constexpr (!skip_sfpu) {
                    typecast_tile<in_df, out_df>(i);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            if constexpr (LCS_PACK_BLOCK) {
                pack_block(0, cb_out, m);
            } else {
                for (uint32_t i = 0; i < m; ++i) {
                    pack_tile(i, cb_out);
                }
            }
            tile_regs_release();
        }
        cb_pop_front(cb_in, n);
        cb_push_back(cb_out, n);
    }
}
