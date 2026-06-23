// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_fill.hpp"         // FillInt
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // MulIntBinary, AddIntBinary

// addcmul (integer): out = in0 + scalar * (in1 * in2), all int32 (ADDCMUL_DATA_FORMAT).
//   copy in0/in1/in2 -> D0/D1/D2 ; D3 = scalar ; D3 = D3*in1 ; D2 = D3*in2 ;
//   D0 = in0 + D2 ; pack D0.
// All LLK already have chain elements — no new helper additions. Each CopyTile uses
// CopyTileReconfig::None, which emits copy_tile_init(Cb) — identical to the original's
// per-input copy_tile_init(cb_inX). Lifecycle: 3 Streaming inputs (wait1/pop1) + 1
// Streaming output (reserve1/push1) per iter — matches the original counts exactly.
namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input_a
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input_b
    constexpr auto cb_in2 = tt::CBIndex::c_2;  // input_c
    constexpr auto cb_out = tt::CBIndex::c_3;

    unary_op_init_common(cb_in0, cb_out);  // caller-owned BIG init

    // 2D: process num_tiles_per_cycle tiles per DEST window (Chunked + Block index)
    // instead of one tile at a time. Host keeps num_tiles_per_cycle <= input CB depth, so the
    // Chunked wait_front(block) can't hang; the chain also clamps block to chain_max_block_v
    // (DEST/lane) as a DEST-overflow safety net.
    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(num_tiles, num_tiles_per_cycle),
        ckl::CopyTile<
            cb_in0,
            ckl::Dst::D0,
            ckl::InputLifecycle::Chunked,
            ckl::CopyTileReconfig::None,
            ckl::OperandKind::Block>{},
        ckl::CopyTile<
            cb_in1,
            ckl::Dst::D1,
            ckl::InputLifecycle::Chunked,
            ckl::CopyTileReconfig::None,
            ckl::OperandKind::Block>{},
        ckl::CopyTile<
            cb_in2,
            ckl::Dst::D2,
            ckl::InputLifecycle::Chunked,
            ckl::CopyTileReconfig::None,
            ckl::OperandKind::Block>{},
        ckl::FillInt<ADDCMUL_DATA_FORMAT, ckl::Dst::D3>{scalar_arg},
        ckl::MulIntBinary<ADDCMUL_DATA_FORMAT, ckl::Dst::D3, ckl::Dst::D1, ckl::Dst::D3>{},  // D3 = scalar*in1
        ckl::MulIntBinary<ADDCMUL_DATA_FORMAT, ckl::Dst::D3, ckl::Dst::D2, ckl::Dst::D2>{},  // D2 = D3*in2
        ckl::AddIntBinary<ADDCMUL_DATA_FORMAT, ckl::Dst::D0, ckl::Dst::D2, ckl::Dst::D0>{},  // D0 = in0 + D2
        ckl::PackTile<cb_out, ckl::OutputLifecycle::Chunked, ckl::PackTileReconfig::None>{});
}
