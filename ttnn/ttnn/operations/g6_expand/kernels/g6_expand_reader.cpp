// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_expand reader, CT-specialized per role (NCRISC, NoC0).
//
// Sender (core (0, y) of each column group), per block:
//   load_source_block  - read the block's x pages into consecutive cb_expanded_tiles slots
//   expand_rows_block  - replicate row 0 of every page into all rows, in place (E tiles)
//   publish_block      - push the block to this core's writer, then multicast it to the group
// Receiver (cores (1.., y)), per block:
//   receive_block      - reserve the same slot, ack the sender, wait for the flag, push to the writer
//
// Raw Noc API for the x read and the expansion: no kernel_lib helper reads tile pages into a CB, and
// generate_index_tile writes generated values instead of copying an existing row; its
// seed -> fence -> fan-out -> lower-face construction is reused here.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/kernel_profiler.hpp"
#include "ckernel.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

namespace {

constexpr uint32_t cb_expanded_tiles = get_compile_time_arg_val(0);
constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
constexpr uint32_t block_src_col_tiles = get_compile_time_arg_val(2);
constexpr bool is_sender = get_compile_time_arg_val(3) != 0;
constexpr uint32_t seed_rows = get_compile_time_arg_val(4);
constexpr uint32_t face_rows = get_compile_time_arg_val(5);
constexpr uint32_t face_row_bytes = get_compile_time_arg_val(6);
constexpr uint32_t face_bytes = get_compile_time_arg_val(7);
constexpr uint32_t faces_per_tile_row = get_compile_time_arg_val(8);
constexpr uint32_t face_rows_per_tile = get_compile_time_arg_val(9);
constexpr auto x_args = TensorAccessorArgs<10>();

constexpr uint32_t face_row_words = face_row_bytes / sizeof(uint32_t);
constexpr uint32_t seed_bytes = seed_rows * face_row_bytes;
// The top face row of a tile (faces 0 .. faces_per_tile_row - 1); every lower face row is a copy of it.
constexpr uint32_t face_band_bytes = faces_per_tile_row * face_bytes;

static_assert(face_rows % seed_rows == 0, "seed_rows must divide the face height");
static_assert(face_row_bytes % 16 == 0, "NoC self-reads need 16 B aligned face rows");
static_assert(face_band_bytes * face_rows_per_tile == page_bytes, "face geometry must cover the page");

// Only row 0 of faces 0 .. faces_per_tile_row - 1 is read; every other byte of each page is overwritten,
// so the x padding rows never reach out.
FORCE_INLINE void expand_rows_block(const Noc& noc, uint32_t slot_base, uint32_t extent) {
    const uint8_t noc_id = noc.get_noc_id();
    const uint32_t self_x = my_x[noc_id];
    const uint32_t self_y = my_y[noc_id];
    UnicastEndpoint local_l1;

    if constexpr (seed_rows > 1) {
        // Row 0 landed through the NoC; drop any cached copy of these lines before the RISC-V reads it.
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* last_store = nullptr;
        for (uint32_t i = 0; i < extent; ++i) {
            const uint32_t tile_addr = slot_base + i * page_bytes;
            for (uint32_t j = 0; j < faces_per_tile_row; ++j) {
                volatile tt_l1_ptr uint32_t* face =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr + j * face_bytes);
                for (uint32_t r = 1; r < seed_rows; ++r) {
                    for (uint32_t w = 0; w < face_row_words; ++w) {
                        face[r * face_row_words + w] = face[w];
                    }
                }
                last_store = face + seed_rows * face_row_words - 1;
            }
        }
        // A store can retire before it lands in L1, and the NoC is a separate L1 client: read the last
        // seed word back so every seed row is in L1 before the self-reads copy it.
        (void)ckernel::load_blocking(last_store);
    }

    for (uint32_t i = 0; i < extent; ++i) {
        const uint32_t tile_addr = slot_base + i * page_bytes;
        for (uint32_t j = 0; j < faces_per_tile_row; ++j) {
            const uint32_t face_addr = tile_addr + j * face_bytes;
            for (uint32_t k = seed_rows; k < face_rows; k += seed_rows) {
                noc.async_read(
                    local_l1,
                    CoreLocalMem<uint32_t>(face_addr + k * face_row_bytes),
                    seed_bytes,
                    {.noc_x = self_x, .noc_y = self_y, .addr = face_addr},
                    {});
            }
        }
    }
    noc.async_read_barrier();

    for (uint32_t i = 0; i < extent; ++i) {
        const uint32_t tile_addr = slot_base + i * page_bytes;
        for (uint32_t b = 1; b < face_rows_per_tile; ++b) {
            noc.async_read(
                local_l1,
                CoreLocalMem<uint32_t>(tile_addr + b * face_band_bytes),
                face_band_bytes,
                {.noc_x = self_x, .noc_y = self_y, .addr = tile_addr},
                {});
        }
    }
    noc.async_read_barrier();
}

template <typename Broadcast>
FORCE_INLINE void sender_blocks(
    const Noc& noc, CircularBuffer& cb, uint32_t x_addr, uint32_t src_start, uint32_t src_count, Broadcast broadcast) {
    const auto x = TensorAccessor(x_args, x_addr, page_bytes);
    for (uint32_t first = 0; first < src_count; first += block_src_col_tiles) {
        const uint32_t extent = src_count - first < block_src_col_tiles ? src_count - first : block_src_col_tiles;
        uint32_t slot_base;
        {
            DeviceZoneScopedN("load_source_block");
            // Nominal quantum on a ragged block too: sender and receivers keep their write pointers in
            // lockstep, so the sender's slot address is every receiver's landing address.
            cb.reserve_back(block_src_col_tiles);
            slot_base = cb.get_write_ptr();
            for (uint32_t i = 0; i < extent; ++i) {
                noc.async_read(x, cb, page_bytes, {.page_id = src_start + first + i}, {.offset_bytes = i * page_bytes});
            }
            noc.async_read_barrier();
        }
        {
            DeviceZoneScopedN("expand_rows_block");
            expand_rows_block(noc, slot_base, extent);
        }
        {
            DeviceZoneScopedN("publish_block");
            // Push first: the local writer and the multicast only read E, so they overlap. The next
            // reserve_back cannot reuse this slot before send() has read it out.
            cb.push_back(block_src_col_tiles);
            broadcast(slot_base, extent * page_bytes);
        }
    }
}

template <typename Channel>
FORCE_INLINE void run_sender(const Channel& group_mcast, const Noc& noc, CircularBuffer& cb) {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t src_start = get_arg_val<uint32_t>(0);
    const uint32_t src_count = get_arg_val<uint32_t>(1);
    if constexpr (Channel::active) {
        auto pipe = group_mcast.sender(noc);
        sender_blocks(noc, cb, x_addr, src_start, src_count, [&](uint32_t slot_base, uint32_t bytes) {
            pipe.send(slot_base, slot_base, bytes);
        });
    } else {
        sender_blocks(noc, cb, x_addr, src_start, src_count, [](uint32_t, uint32_t) {});
    }
}

template <typename Channel>
FORCE_INLINE void run_receiver(const Channel& group_mcast, const Noc& noc, CircularBuffer& cb) {
    static_assert(Channel::active, "a receiver core exists only when the group multicast is present");
    const uint32_t src_count = get_arg_val<uint32_t>(1);
    auto pipe = group_mcast.receiver(noc);
    uint32_t round = 0;
    for (uint32_t first = 0; first < src_count; first += block_src_col_tiles, ++round) {
        DeviceZoneScopedN("receive_block");
        // The ack inside receive() is sent only after the slot is reserved, so the payload never lands
        // in a slot the writer is still reading.
        cb.reserve_back(block_src_col_tiles);
        pipe.receive(round);
        cb.push_back(block_src_col_tiles);
    }
}

template <bool IS_SENDER, typename Channel>
FORCE_INLINE void run_reader(const Channel& group_mcast) {
    Noc noc;
    CircularBuffer cb(cb_expanded_tiles);
    if constexpr (IS_SENDER) {
        run_sender(group_mcast, noc, cb);
    } else {
        run_receiver(group_mcast, noc, cb);
    }
    {
        DeviceZoneScopedN("reader_drain");
        // Multicast payload/flag writes and the receivers' ready increments must complete before the
        // next program starts (BB-03).
        noc.async_write_barrier();
        noc.async_atomic_barrier();
    }
}

}  // namespace

void kernel_main() {
    constexpr auto group_mcast = McastArgs<
        get_named_compile_time_arg_val("group_mcast_ct_offset"),
        get_named_compile_time_arg_val("group_mcast_rt_offset")>();
    run_reader<is_sender>(group_mcast);
}
