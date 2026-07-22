// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// Dest-format-aware LLK wrappers and binary tile->CB orchestrators. The *_with_dt thin wrappers
// prepend reconfig_data_format* / pack_reconfig_data_format when FP32_DEST_ACC_EN is set, otherwise
// no-op. The *_to_cb wrappers chain a binary tile op through a single DST register and push the
// result to ocb.
//
// CB operations migrated to Device 2.0 CircularBuffer method form. LLK primitives (mul_tiles,
// add_tiles, sub_tiles, pack_tile, tile_regs_*, *_init, *reconfig*) remain raw — they are LLK
// surface, not D1.x dataflow.
//
// Namespace placement is split intentionally so callers do not need to add any qualification when
// swapping the include source.

// File-scope wrappers.

ALWI void pack_tile_with_dt(uint32_t ifrom_dst, uint32_t icb) {
#if defined FP32_DEST_ACC_EN
    pack_reconfig_data_format(icb);
#endif
    pack_tile(ifrom_dst, icb);
}

ALWI void copy_tile_init_with_dt(uint32_t icb, uint32_t transpose = 0) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format_srca(icb);
#endif
    copy_tile_to_dst_init_short(icb, transpose);
}

ALWI void add_tiles_init_with_dt(uint32_t icb0 = 0, uint32_t icb1 = 1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    add_init(icb0, icb1);
}

ALWI void sub_tiles_init_with_dt(uint32_t icb0 = 0, uint32_t icb1 = 1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    sub_init(icb0, icb1);
}

ALWI void mul_tiles_init_with_dt(uint32_t icb0 = 0, uint32_t icb1 = 1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);
#endif
    mul_init(icb0, icb1);
}

// Binary tile->CB orchestrators (in namespace ckernel).
namespace ckernel {

ALWI void mul_tiles_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;
    CircularBuffer cb_in0(icb0);
    CircularBuffer cb_in1(icb1);
    CircularBuffer cb_out(ocb);

    cb_out.reserve_back(onetile);
    cb_in0.wait_front(itile0 + 1);
    cb_in1.wait_front(itile1 + 1);

    tile_regs_acquire();
    mul_tiles_init_with_dt(icb0, icb1);
    mul_tiles(icb0, icb1, itile0, itile1, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop0) {
        cb_in0.pop_front(pop0);
    }
    if (pop1) {
        cb_in1.pop_front(pop1);
    }

    cb_out.push_back(onetile);
}

ALWI void add_tiles_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;
    CircularBuffer cb_in0(icb0);
    CircularBuffer cb_in1(icb1);
    CircularBuffer cb_out(ocb);

    cb_out.reserve_back(onetile);
    cb_in0.wait_front(itile0 + 1);
    cb_in1.wait_front(itile1 + 1);

    tile_regs_acquire();
    add_tiles_init_with_dt(icb0, icb1);
    add_tiles(icb0, icb1, itile0, itile1, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop0) {
        cb_in0.pop_front(pop0);
    }
    if (pop1) {
        cb_in1.pop_front(pop1);
    }

    cb_out.push_back(onetile);
}

ALWI void sub_tiles_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;
    CircularBuffer cb_in0(icb0);
    CircularBuffer cb_in1(icb1);
    CircularBuffer cb_out(ocb);

    cb_out.reserve_back(onetile);
    cb_in0.wait_front(itile0 + 1);
    cb_in1.wait_front(itile1 + 1);

    tile_regs_acquire();
    sub_tiles_init_with_dt(icb0, icb1);
    sub_tiles(icb0, icb1, itile0, itile1, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop0) {
        cb_in0.pop_front(pop0);
    }
    if (pop1) {
        cb_in1.pop_front(pop1);
    }

    cb_out.push_back(onetile);
}

}  // namespace ckernel
