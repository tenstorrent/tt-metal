// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// g6_expand reader (NCRISC, NoC0), one source specialized per role by CT arg 0.
//
// Sender (0, y): per block, load_source_block -> expand_rows_block -> publish_block. The x pages of
// the block land in cb_expanded_tiles, row 0 of every tile is replicated into all rows in place, then
// the block is pushed to this core's writer and multicast into the same slot of every receiver of
// grid row y.
// Receiver (x > 0, y): per block, receive_block: reserve the slot, ack + wait for the multicast,
// push to the writer.
//
// Raw NoC API, no kernel_lib helper: no helper reads tile pages into a CB, and no helper replicates
// an existing tile row (generate_index_tile writes generated values; its seed -> fence -> fan-out ->
// lower-face construction is the one realized here). The multicast itself uses the kernel_lib pipe.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/kernel_profiler.hpp"
#include "hostdevcommon/common_values.hpp"
#include "ckernel.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

namespace {

// Turns `extent` consecutive x pages at `block_addr` into expanded tiles E in place: every row of
// the tile becomes a copy of row 0. Reads only row 0 of the upper faces; the padding rows of x are
// overwritten, never read. Fixed costs are paid once per block, not per tile.
template <
    uint32_t page_bytes,
    uint32_t face_row_bytes,
    uint32_t face_bytes,
    uint32_t faces_per_band,
    uint32_t face_height,
    uint32_t num_bands,
    uint32_t seed_rows>
FORCE_INLINE void expand_rows_block(const Noc& noc, uint32_t block_addr, uint32_t extent) {
    constexpr uint32_t row_words = face_row_bytes / sizeof(uint32_t);
    constexpr uint32_t seed_bytes = seed_rows * face_row_bytes;
    constexpr uint32_t band_bytes = faces_per_band * face_bytes;
    static_assert(face_row_bytes % sizeof(uint32_t) == 0);
    static_assert(face_height % seed_rows == 0, "seed rows must tile the face height");

    const uint8_t noc_id = noc.get_noc_id();
    const uint32_t self_x = my_x[noc_id];
    const uint32_t self_y = my_y[noc_id];
    UnicastEndpoint local_l1;

    if constexpr (seed_rows > 1) {
        // Row 0 was landed by the NoC; drop any stale cache lines this RISC holds over it.
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* last_store = nullptr;
        for (uint32_t t = 0; t < extent; ++t) {
            for (uint32_t f = 0; f < faces_per_band; ++f) {
                volatile tt_l1_ptr uint32_t* face =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(block_addr + t * page_bytes + f * face_bytes);
                for (uint32_t r = 1; r < seed_rows; ++r) {
                    for (uint32_t w = 0; w < row_words; ++w) {
                        face[r * row_words + w] = face[w];
                    }
                }
                last_store = face + seed_rows * row_words - 1;
            }
        }
        // A RISC-V store may retire before it lands in L1, and the NoC is a separate L1 client: read
        // the last stored word back so every seed row is in L1 before the fan-out reads it.
        (void)ckernel::load_blocking(last_store);
    }

    for (uint32_t t = 0; t < extent; ++t) {
        for (uint32_t f = 0; f < faces_per_band; ++f) {
            const uint32_t face_addr = block_addr + t * page_bytes + f * face_bytes;
            for (uint32_t r = seed_rows; r < face_height; r += seed_rows) {
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

    if constexpr (num_bands > 1) {
        for (uint32_t t = 0; t < extent; ++t) {
            const uint32_t page_addr = block_addr + t * page_bytes;
            for (uint32_t b = 1; b < num_bands; ++b) {
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
}

}  // namespace

void kernel_main() {
    constexpr bool is_sender = get_compile_time_arg_val(0) != 0;
    constexpr uint32_t cb_expanded_tiles_id = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t block_src_col_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t face_row_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t faces_per_band = get_compile_time_arg_val(6);
    constexpr uint32_t face_height = get_compile_time_arg_val(7);
    constexpr uint32_t num_bands = get_compile_time_arg_val(8);
    constexpr uint32_t seed_rows = get_compile_time_arg_val(9);
    constexpr auto x_args = TensorAccessorArgs<10>();
    constexpr auto group = McastArgs<
        get_named_compile_time_arg_val("group_ct_offset"),
        get_named_compile_time_arg_val("group_rt_offset")>();

    const uint32_t src_start = get_arg_val<uint32_t>(0);
    const uint32_t src_count = get_arg_val<uint32_t>(1);
    const uint32_t num_src_blocks = (src_count + block_src_col_tiles - 1) / block_src_col_tiles;

    Noc noc;
    CircularBuffer cb_expanded_tiles(cb_expanded_tiles_id);

    if constexpr (is_sender) {
        const auto x_accessor = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
        auto sender_pipe = group.optional_sender(noc);

        for (uint32_t block_idx = 0; block_idx < num_src_blocks; ++block_idx) {
            const uint32_t first = block_idx * block_src_col_tiles;
            const uint32_t extent = src_count - first < block_src_col_tiles ? src_count - first : block_src_col_tiles;

            // Every push/pop is the nominal block (also on a ragged last block), so sender and
            // receivers advance their write pointers in lockstep: the sender's slot is the landing
            // address on every receiver.
            uint32_t block_addr;
            {
                DeviceZoneScopedN("load_source_block");
                cb_expanded_tiles.reserve_back(block_src_col_tiles);
                block_addr = cb_expanded_tiles.get_write_ptr();
                for (uint32_t t = 0; t < extent; ++t) {
                    noc.async_read(
                        x_accessor,
                        cb_expanded_tiles,
                        page_bytes,
                        {.page_id = src_start + first + t},
                        {.offset_bytes = t * page_bytes});
                }
                noc.async_read_barrier();
            }
            {
                DeviceZoneScopedN("expand_rows_block");
                expand_rows_block<
                    page_bytes,
                    face_row_bytes,
                    face_bytes,
                    faces_per_band,
                    face_height,
                    num_bands,
                    seed_rows>(noc, block_addr, extent);
            }
            {
                DeviceZoneScopedN("publish_block");
                // Push first: this core's writer and the multicast only read E, so the writes start
                // while the broadcast is in flight. send() returns only once E has been read, so the
                // next reserve cannot overwrite it under the multicast.
                cb_expanded_tiles.push_back(block_src_col_tiles);
                if (sender_pipe) {
                    sender_pipe->send(block_addr, block_addr, extent * page_bytes);
                }
            }
        }
    } else {
        auto receiver_pipe = group.optional_receiver(noc);
        ASSERT(receiver_pipe.has_value());
        for (uint32_t block_idx = 0; block_idx < num_src_blocks; ++block_idx) {
            DeviceZoneScopedN("receive_block");
            // The ready-ack inside receive() must follow the reserve: it tells the sender this slot
            // is free to land in.
            cb_expanded_tiles.reserve_back(block_src_col_tiles);
            receiver_pipe->receive(block_idx);
            cb_expanded_tiles.push_back(block_src_col_tiles);
        }
    }

    // BB-03: the next program starts right away; no multicast write or ready increment may remain
    // in flight.
    {
        DeviceZoneScopedN("exit_barrier");
        noc.async_write_barrier();
        noc.async_atomic_barrier();
    }
}
