// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation file for accumulate_helpers_compute.hpp
// Do not include directly - include accumulate_helpers_compute.hpp instead

#pragma once

namespace compute_kernel_lib {

ALWI BlockAccumulate BlockAccumulate::arm(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out, uint32_t granularity) {
    // A run is ONE acquire/commit/pack pass, so the CB granularity must fit the DEST register.
    // dest_helpers.hpp derives the real capacity from the JIT sync + accum mode, which is the number
    // the host is trying to reproduce when it writes `fp32_dest_acc_en ? 4 : 8`. Callers whose block
    // is genuinely larger want run_chunked() instead.
    ASSERT(granularity <= DEST_AUTO_LIMIT);
    // Op-level init only, programmed once — hoisting this out of the caller's loop is the point, since
    // most of these kernels re-issue it per chunk. Hardware startup stays with the kernel (see the
    // header's ownership note): compute_kernel_hw_startup and binary_op_init_common are NOT
    // interchangeable, so arm() must not pick one on the caller's behalf.
    add_tiles_init(cb_a, cb_b, false);
    return BlockAccumulate(cb_a, cb_b, cb_out, granularity);
}

ALWI void BlockAccumulate::rearm() {
    // Restoring the op init alone is NOT enough. add_tiles_init issues state_configure (the
    // ComputeKernelSentinel tracker), the math binary init and llk_unpack_AB_init — but NOT
    // reconfig_data_format. An interleaved op that touched the unpack/pack data-format registers
    // therefore leaves them pointing at ITS operands, so the formats must be re-established explicitly
    // or the next add_tiles unpacks this accumulator's CBs through the wrong format.
    reconfig_data_format(cb_a_, cb_b_);
    pack_reconfig_data_format(cb_out_);
    add_tiles_init(cb_a_, cb_b_, false);
    programmed_seeded_ = false;
}

ALWI void BlockAccumulate::ensure_mode(bool seeded) {
    if (programmed_seeded_ != seeded) {
        // acc_to_dest distinguishes "DST = a + b" from "DST += a + b"; only re-program on a real change.
        add_tiles_init(cb_a_, cb_b_, seeded);
        programmed_seeded_ = seeded;
    }
}

ALWI void BlockAccumulate::run(uint32_t num_tiles) {
    cb_wait_front(cb_a_, granularity_);
    cb_wait_front(cb_b_, granularity_);

    ensure_mode(false);

    tile_regs_acquire();
    for (uint32_t i = 0; i < num_tiles; ++i) {
        add_tiles(cb_a_, cb_b_, i, i, i);
    }
    tile_regs_commit();

    // Pop before reserving out, so the reader's slots free as early as possible. This ordering is
    // what the shipped reduction collectives were verified with; see run_chunked()'s note.
    cb_pop_front(cb_a_, granularity_);
    cb_pop_front(cb_b_, granularity_);

    cb_reserve_back(cb_out_, granularity_);
    tile_regs_wait();
    for (uint32_t i = 0; i < num_tiles; ++i) {
        pack_tile(i, cb_out_, i);
    }
    tile_regs_release();
    cb_push_back(cb_out_, granularity_);
}

ALWI void BlockAccumulate::run_seeded(uint32_t cb_seed, uint32_t num_tiles) {
    cb_wait_front(cb_seed, granularity_);
    cb_wait_front(cb_a_, granularity_);
    cb_wait_front(cb_b_, granularity_);

    tile_regs_acquire();
    // Seed DST from cb_seed FIRST, then accumulate onto it. Note this does not rely on DST being
    // zero at acquire — tile_regs_acquire() does not zero it (see the header's warning).
    copy_tile_init(cb_seed);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        copy_tile(cb_seed, i, i);
    }
    // copy_tile_init reprogrammed the unpacker, so the add init must follow it unconditionally
    // rather than going through ensure_mode(); record the resulting mode for the next run().
    add_tiles_init(cb_a_, cb_b_, true);  // DST += a + b
    programmed_seeded_ = true;
    for (uint32_t i = 0; i < num_tiles; ++i) {
        add_tiles(cb_a_, cb_b_, i, i, i);
    }
    tile_regs_commit();

    cb_pop_front(cb_seed, granularity_);
    cb_pop_front(cb_a_, granularity_);
    cb_pop_front(cb_b_, granularity_);

    cb_reserve_back(cb_out_, granularity_);
    tile_regs_wait();
    for (uint32_t i = 0; i < num_tiles; ++i) {
        pack_tile(i, cb_out_, i);
    }
    tile_regs_release();
    cb_push_back(cb_out_, granularity_);
}

ALWI void BlockAccumulate::run_chunked(uint32_t num_tiles, uint32_t out_capacity) {
    cb_wait_front(cb_a_, granularity_);
    cb_wait_front(cb_b_, granularity_);

    ensure_mode(false);

    // Reserve before the first pack, and pop only after the last pass has read a and b — the reason
    // this cannot reuse run()'s ordering, and therefore why it is a separate method.
    cb_reserve_back(cb_out_, out_capacity);
    for (uint32_t base = 0; base < num_tiles; base += DEST_AUTO_LIMIT) {
        const uint32_t n = (num_tiles - base) < DEST_AUTO_LIMIT ? (num_tiles - base) : DEST_AUTO_LIMIT;
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            add_tiles(cb_a_, cb_b_, base + i, base + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            // In the default in-order pack mode the output tile index is IGNORED: pack_tile writes to
            // the next slot and its internal pointer is reset only by cb_push_back, which happens once
            // after the whole loop. So successive passes append naturally and passing `base + i` here
            // would be dead arithmetic implying a placement guarantee this mode does not offer.
            pack_tile(i, cb_out_);
        }
        tile_regs_release();
    }
    cb_pop_front(cb_a_, granularity_);
    cb_pop_front(cb_b_, granularity_);
    cb_push_back(cb_out_, out_capacity);
}

}  // namespace compute_kernel_lib
