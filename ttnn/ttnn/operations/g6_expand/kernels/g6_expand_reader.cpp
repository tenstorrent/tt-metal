// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_expand reader, compiled once per role (CT arg 0).
//
// Sender (x = 0 of each column group): load_source_block -> expand_rows_block -> publish_block.
// Receiver (x > 0): receive_block. Both hand each block to the writer through cb_expanded_tiles,
// whose slot address is identical on every core of the rectangle (the multicast landing address).
//
// No kernel_lib helper reads tile pages into a CB or replicates an existing tile row:
// load_source_block and expand_rows_block use the Noc API directly. expand_rows_block follows
// generate_index_tile's seed -> fence -> fan-out -> lower-face construction
// (kernel_lib/index_tile_dataflow.hpp) but seeds by copying row 0 instead of generating values.
// The broadcast goes through the kernel_lib Mcast pipe.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/debug/kernel_profiler.hpp"
#include "ckernel.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

namespace {

constexpr uint32_t is_sender = get_compile_time_arg_val(0);
constexpr uint32_t cb_expanded_tiles = get_compile_time_arg_val(1);
constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
constexpr uint32_t block_src_col_tiles = get_compile_time_arg_val(3);
constexpr uint32_t seed_rows = get_compile_time_arg_val(4);
constexpr uint32_t face_row_bytes = get_compile_time_arg_val(5);
constexpr uint32_t face_rows = get_compile_time_arg_val(6);
constexpr uint32_t faces_per_band = get_compile_time_arg_val(7);  // faces side by side in a tile
constexpr uint32_t bands_per_tile = get_compile_time_arg_val(8);  // faces stacked in a tile
constexpr auto x_args = TensorAccessorArgs<9>();

constexpr uint32_t face_bytes = face_rows * face_row_bytes;
constexpr uint32_t band_bytes = faces_per_band * face_bytes;
constexpr uint32_t face_row_words = face_row_bytes / sizeof(uint32_t);
constexpr uint32_t seed_bytes = seed_rows * face_row_bytes;

static_assert(face_row_bytes % 16 == 0, "self-read offsets must stay L1-aligned");
static_assert(seed_rows >= 1 && face_rows % seed_rows == 0, "seed rows must tile the face height");
static_assert(bands_per_tile * band_bytes == page_bytes, "face geometry must cover the page");

// Rewrites `extent` raw x pages at `slot` in place so every row of each page equals its row 0.
// Reads only row 0 of the top band (x's rows 1..31 are padding and never feed a logical element).
FORCE_INLINE void expand_rows_block(const Noc& noc, uint32_t slot, uint32_t extent) {
    const uint8_t noc_id = noc.get_noc_id();
    const uint32_t self_x = my_x[noc_id];
    const uint32_t self_y = my_y[noc_id];
    UnicastEndpoint local_l1;

    if constexpr (seed_rows > 1) {
        volatile tt_l1_ptr uint32_t* last_store = nullptr;
        for (uint32_t p = 0; p < extent; ++p) {
            for (uint32_t f = 0; f < faces_per_band; ++f) {
                volatile tt_l1_ptr uint32_t* face =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + p * page_bytes + f * face_bytes);
                for (uint32_t r = 1; r < seed_rows; ++r) {
                    for (uint32_t w = 0; w < face_row_words; ++w) {
                        face[r * face_row_words + w] = face[w];
                    }
                }
                last_store = face + seed_rows * face_row_words - 1;
            }
        }
        // A RISC-V store can retire before it lands in L1 and the NoC is a separate L1 client:
        // read the last stored word back so every seed is in L1 before the fan-out reads it.
        (void)ckernel::load_blocking(last_store);
    }

    for (uint32_t p = 0; p < extent; ++p) {
        for (uint32_t f = 0; f < faces_per_band; ++f) {
            const uint32_t face_addr = slot + p * page_bytes + f * face_bytes;
            for (uint32_t r = seed_rows; r < face_rows; r += seed_rows) {
                noc.async_read(
                    local_l1,
                    CoreLocalMem<uint32_t>(face_addr + r * face_row_bytes),
                    seed_bytes,
                    {.noc_x = self_x, .noc_y = self_y, .addr = face_addr},
                    {});
            }
        }
    }
    noc.async_read_barrier();

    for (uint32_t p = 0; p < extent; ++p) {
        const uint32_t page_addr = slot + p * page_bytes;
        for (uint32_t b = 1; b < bands_per_tile; ++b) {
            noc.async_read(
                local_l1,
                CoreLocalMem<uint32_t>(page_addr + b * band_bytes),
                band_bytes,
                {.noc_x = self_x, .noc_y = self_y, .addr = page_addr},
                {});
        }
    }
    noc.async_read_barrier();
}

template <typename Broadcast>
FORCE_INLINE void run_sender(
    const Noc& noc, CircularBuffer& cb, uint32_t src_start, uint32_t src_count, Broadcast&& broadcast) {
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), page_bytes);
    uint32_t block_first_src = src_start;
    for (uint32_t done = 0; done < src_count; done += block_src_col_tiles) {
        const uint32_t extent = src_count - done < block_src_col_tiles ? src_count - done : block_src_col_tiles;
        // Every push is the nominal block, also on a ragged tail, so all cores of the group keep
        // their CB write pointers (the multicast landing address) in lockstep.
        cb.reserve_back(block_src_col_tiles);
        const uint32_t slot = cb.get_write_ptr();
        {
            DeviceZoneScopedN("load_source_block");
            for (uint32_t i = 0; i < extent; ++i) {
                noc.async_read(x, cb, page_bytes, {.page_id = block_first_src + i}, {.offset_bytes = i * page_bytes});
            }
            noc.async_read_barrier();
        }
        {
            DeviceZoneScopedN("expand_rows_block");
            expand_rows_block(noc, slot, extent);
        }
        {
            DeviceZoneScopedN("publish_block");
            // Push before the broadcast: the local writer and the multicast both only read E.
            // send() returns only once the multicast has read E, so the next reserve cannot
            // overwrite it.
            cb.push_back(block_src_col_tiles);
            broadcast(slot, extent * page_bytes);
        }
        block_first_src += block_src_col_tiles;
    }
}

template <bool IS_SENDER, typename Channel>
FORCE_INLINE void run(const Channel& channel) {
    const uint32_t src_start = get_arg_val<uint32_t>(0);
    const uint32_t src_count = get_arg_val<uint32_t>(1);

    Noc noc;
    CircularBuffer cb(cb_expanded_tiles);

    if constexpr (IS_SENDER) {
        if constexpr (Channel::active) {
            auto pipe = channel.sender(noc);
            run_sender(
                noc, cb, src_start, src_count, [&](uint32_t slot, uint32_t bytes) { pipe.send(slot, slot, bytes); });
        } else {
            run_sender(noc, cb, src_start, src_count, [](uint32_t, uint32_t) {});
        }
    } else {
        static_assert(Channel::active, "a receiver core requires a present multicast channel");
        auto pipe = channel.receiver(noc);
        uint32_t round = 0;
        for (uint32_t done = 0; done < src_count; done += block_src_col_tiles) {
            cb.reserve_back(block_src_col_tiles);
            {
                DeviceZoneScopedN("receive_block");
                // The ready-ack is sent only after the slot is reserved: the sender may land the
                // payload as soon as every receiver has acked.
                pipe.receive(round);
            }
            cb.push_back(block_src_col_tiles);
            ++round;
        }
    }

    // BB-03: multicast payloads and flags (sender) and ready increments (receiver) must have
    // completed before the next program starts.
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}

}  // namespace

void kernel_main() {
    constexpr auto channel = dataflow_kernel_lib::McastArgs<
        get_named_compile_time_arg_val("expand_mcast_ct_offset"),
        get_named_compile_time_arg_val("expand_mcast_rt_offset")>();
    run<is_sender != 0>(channel);
}
