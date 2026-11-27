// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_expand reader (NCRISC, NoC0). One source, two compile-time roles on one grid column:
//   injector (top core): fetch row 0 of each x page of the block, replicate it into all 32 rows
//                        in place, multicast the block down the column, publish it.
//   receiver (below):    reserve the block, take the multicast, publish it.
// Every core of a column runs the same block sequence with nominal reserve/push counts, so the
// multicast always lands on the slot each receiver has reserved at the injector's address.
//
// fetch_row0_block uses raw Noc + TensorAccessor: kernel_lib has no DRAM page reader
// (local_copy_helpers_dataflow.hpp only builds self-aimed sources). The replicate and the
// multicast use kernel_lib helpers (set_read_state/read_with_state, McastArgs).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

namespace {

// bf16 32x32 tile: faces 0..3 of 16x16, 512 B each; one face row is 32 B. Row 0 of the tile is
// face 0 row 0 (left half, page offset 0) and face 1 row 0 (right half, page offset 512).
constexpr uint32_t FACE_ROWS = 16;
constexpr uint32_t FACE_ROW_BYTES = 32;
constexpr uint32_t FACE_BYTES = FACE_ROWS * FACE_ROW_BYTES;
constexpr uint32_t LOWER_FACES_OFFSET = 2 * FACE_BYTES;
constexpr uint32_t ROW0_HALF_OFFSETS[2] = {0, FACE_BYTES};

constexpr uint32_t cb_pattern_tiles_id = get_compile_time_arg_val(0);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
constexpr uint32_t block_pattern_tiles = get_compile_time_arg_val(2);
constexpr bool is_injector = get_compile_time_arg_val(3) != 0;
constexpr auto x_args = TensorAccessorArgs<4>();

static_assert(tile_bytes == 4 * FACE_BYTES, "g6_expand reader assumes a bfloat16 32x32 tile");

// Only bytes [0, 32) and [512, 544) of each x page cross DRAM: rows 1..31 and the tile padding of
// x never leave it. Each segment lands at its own in-page offset, which keeps the DRAM source and
// the L1 destination congruent modulo the 64-B Blackhole DRAM read alignment.
template <typename Accessor>
FORCE_INLINE void fetch_row0_block(
    const Noc& noc, const Accessor& x, const CircularBuffer& cb, uint32_t first_pattern, uint32_t num_patterns) {
    for (uint32_t j = 0; j < num_patterns; ++j) {
        for (uint32_t half : ROW0_HALF_OFFSETS) {
            noc.async_read<NocOptions::DEFAULT, FACE_ROW_BYTES>(
                x,
                cb,
                FACE_ROW_BYTES,
                {.page_id = first_pattern + j, .offset_bytes = half},
                {.offset_bytes = j * tile_bytes + half});
        }
    }
    noc.async_read_barrier();
}

// Rows 1..15 of the upper face and rows 0..15 of the lower face of each half are copies of that
// half's row 0, made in the page itself with self-aimed NoC reads: a byte copy, so every bit pattern
// (NaN payloads, -0, denormals) survives.
FORCE_INLINE void replicate_block(Noc& noc, uint32_t block_base, uint32_t num_patterns) {
    for (uint32_t j = 0; j < num_patterns; ++j) {
        const uint32_t page = block_base + j * tile_bytes;
        for (uint32_t half : ROW0_HALF_OFFSETS) {
            const uint32_t row0 = page + half;
            dataflow_kernel_lib::set_read_state<FACE_ROW_BYTES>(noc, row0);
            for (uint32_t r = 1; r < FACE_ROWS; ++r) {
                dataflow_kernel_lib::read_with_state<FACE_ROW_BYTES>(noc, row0 + r * FACE_ROW_BYTES, row0);
            }
            const uint32_t lower = row0 + LOWER_FACES_OFFSET;
            for (uint32_t r = 0; r < FACE_ROWS; ++r) {
                dataflow_kernel_lib::read_with_state<FACE_ROW_BYTES>(noc, lower + r * FACE_ROW_BYTES, row0);
            }
        }
    }
    noc.async_read_barrier();
}

template <typename Mcast, typename Accessor>
FORCE_INLINE void run_injector(
    const Mcast& pattern_mcast,
    Noc& noc,
    CircularBuffer& cb,
    const Accessor& x,
    uint32_t pattern_start,
    uint32_t core_pattern_tiles,
    uint32_t num_blocks) {
    auto mcast_block = [&](auto& sender, uint32_t block_base, uint32_t bytes) {
        if constexpr (Mcast::active) {
            DeviceZoneScopedN("G6_MCAST_SEND");
            sender.send(block_base, block_base, bytes);
        }
    };
    auto run = [&](auto& sender) {
        for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            const uint32_t block_offset = block_idx * block_pattern_tiles;
            const uint32_t remaining = core_pattern_tiles - block_offset;
            const uint32_t this_block_pattern_tiles = remaining < block_pattern_tiles ? remaining : block_pattern_tiles;
            {
                DeviceZoneScopedN("G6_CB_RESERVE");
                cb.reserve_back(block_pattern_tiles);
            }
            const uint32_t block_base = cb.get_write_ptr();
            {
                DeviceZoneScopedN("G6_FETCH");
                fetch_row0_block(noc, x, cb, pattern_start + block_offset, this_block_pattern_tiles);
            }
            {
                DeviceZoneScopedN("G6_REPLICATE");
                replicate_block(noc, block_base, this_block_pattern_tiles);
            }
            mcast_block(sender, block_base, this_block_pattern_tiles * tile_bytes);
            cb.push_back(block_pattern_tiles);
        }
    };
    if constexpr (Mcast::active) {
        auto sender = pattern_mcast.sender(noc);
        run(sender);
    } else {
        int no_sender = 0;
        run(no_sender);
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}

template <typename Mcast>
FORCE_INLINE void run_receiver(const Mcast& pattern_mcast, Noc& noc, CircularBuffer& cb, uint32_t num_blocks) {
    auto receiver = pattern_mcast.receiver(noc);
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        {
            DeviceZoneScopedN("G6_CB_RESERVE");
            cb.reserve_back(block_pattern_tiles);
        }
        {
            DeviceZoneScopedN("G6_MCAST_WAIT");
            receiver.receive(block_idx);
        }
        cb.push_back(block_pattern_tiles);
    }
    // The ready signals are remote atomic increments on the injector; none may outlive the kernel.
    noc.async_atomic_barrier();
}

template <bool INJECTOR, typename Mcast>
FORCE_INLINE void run_role(const Mcast& pattern_mcast) {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t pattern_start = get_arg_val<uint32_t>(0);
    const uint32_t core_pattern_tiles = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = (core_pattern_tiles + block_pattern_tiles - 1) / block_pattern_tiles;

    Noc noc;
    CircularBuffer cb(cb_pattern_tiles_id);
    if constexpr (INJECTOR) {
        const auto x = TensorAccessor(x_args, x_addr);
        run_injector(pattern_mcast, noc, cb, x, pattern_start, core_pattern_tiles, num_blocks);
    } else {
        static_assert(Mcast::active, "a receiver core exists only on a column with a multicast channel");
        run_receiver(pattern_mcast, noc, cb, num_blocks);
    }
}

}  // namespace

void kernel_main() {
    constexpr auto pattern_mcast = dataflow_kernel_lib::McastArgs<
        get_named_compile_time_arg_val("pattern_mcast_ct_offset"),
        get_named_compile_time_arg_val("pattern_mcast_rt_offset")>();
    run_role<is_injector>(pattern_mcast);
}
