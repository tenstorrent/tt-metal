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
    // Seed DST from cb_seed FIRST, then accumulate onto it — the seed is data plumbing (the third
    // addend lives in a CB), not zero-safety; see the header's DST-zero-invariant note.
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

ALWI void sum_blocks(uint32_t cb_in, uint32_t cb_out, uint32_t num_blocks, uint32_t block_num_tiles, bool pop_input) {
    cb_wait_front(cb_in, num_blocks * block_num_tiles);
    cb_reserve_back(cb_out, block_num_tiles);

    // Odd block count: block 0 seeds DST via copy_tile and the PAIR loop starts at block 1, so
    // every add_tiles below has a real partner block. Even count: pairs start at block 0 and
    // accumulate onto DST's zero start (see the banner's DST-zero invariant).
    const bool seed_first_block = (num_blocks % 2) != 0;
    const uint32_t first_pair_block = seed_first_block ? 1 : 0;

    if (!seed_first_block) {
        add_tiles_init(cb_in, cb_in, true);
    }

    uint32_t tiles_done = 0;
    while (tiles_done < block_num_tiles) {
        const uint32_t n =
            (block_num_tiles - tiles_done) < DEST_AUTO_LIMIT ? (block_num_tiles - tiles_done) : DEST_AUTO_LIMIT;
        tile_regs_acquire();
        if (seed_first_block) {
            copy_tile_init(cb_in);
            for (uint32_t i = 0; i < n; ++i) {
                copy_tile(cb_in, tiles_done + i, i);
            }
            // copy_tile_init reprogrammed the unpacker, so re-establish the add per chunk.
            add_tiles_init(cb_in, cb_in, true);
        }
        for (uint32_t block = first_pair_block; block < num_blocks; block += 2) {
            for (uint32_t i = 0; i < n; ++i) {
                add_tiles(
                    cb_in,
                    cb_in,
                    block * block_num_tiles + tiles_done + i,
                    (block + 1) * block_num_tiles + tiles_done + i,
                    i);
            }
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            // In-order pack mode: the output index would be ignored (see run_chunked's note).
            pack_tile(i, cb_out);
        }
        tile_regs_release();
        tiles_done += n;
    }
    if (pop_input) {
        cb_pop_front(cb_in, num_blocks * block_num_tiles);
    }
    cb_push_back(cb_out, block_num_tiles);
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
