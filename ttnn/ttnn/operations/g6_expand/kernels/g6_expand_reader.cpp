// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_expand reader, CT-specialized per role.
//   sender   (x = 0 of each column group): load_source_block -> expand_rows_block -> publish_block
//   receiver (x > 0):                       receive_block
// Every block op is a byte copy (NoC or RISC-V word store): no value passes through unpack/DEST/pack,
// so NaN payloads, denormals and signed zeros survive bit-exact.
//
// Raw Noc API for load/expand: no kernel_lib helper reads tile pages into a CB, and
// generate_index_tile (index_tile_dataflow.hpp) writes generated indices instead of copying an
// existing row. Its seed -> fence -> fan-out -> lower-face construction is reused here.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ckernel.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

namespace {

constexpr uint32_t cb_expanded_tiles_id = get_compile_time_arg_val(0);
constexpr bool is_sender = get_compile_time_arg_val(1) != 0;
constexpr uint32_t block_src_col_tiles = get_compile_time_arg_val(2);
constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
constexpr uint32_t face_row_bytes = get_compile_time_arg_val(4);
constexpr uint32_t face_bytes = get_compile_time_arg_val(5);
constexpr uint32_t face_rows = get_compile_time_arg_val(6);
constexpr uint32_t faces_per_tile_row = get_compile_time_arg_val(7);
constexpr uint32_t face_rows_per_tile = get_compile_time_arg_val(8);
constexpr uint32_t seed_rows = get_compile_time_arg_val(9);
constexpr auto x_args = TensorAccessorArgs<10>();

constexpr uint32_t block_bytes = block_src_col_tiles * page_bytes;

static_assert(face_rows % seed_rows == 0, "seed_rows must divide the face height");
static_assert(face_row_bytes % 16 == 0, "self-read offsets must stay 16-B aligned");

FORCE_INLINE uint32_t num_blocks(uint32_t src_count) {
    return (src_count + block_src_col_tiles - 1) / block_src_col_tiles;
}

// x pages [first_page, first_page + extent) into consecutive slots of the reserved block.
template <typename Accessor>
FORCE_INLINE void load_source_block(
    const Noc& noc, CircularBuffer& cb, const Accessor& x_acc, uint32_t first_page, uint32_t extent) {
    for (uint32_t i = 0; i < extent; ++i) {
        noc.async_read(x_acc, cb, page_bytes, {.page_id = first_page + i}, {.offset_bytes = i * page_bytes});
    }
    noc.async_read_barrier();
}

// In place: row 0 of each page (face row 0 of the upper faces) is copied into every row of the tile.
// Only row 0 is ever read, so x's padding rows and their contents never reach the output.
FORCE_INLINE void expand_rows_block(const Noc& noc, uint32_t slot, uint32_t extent) {
    constexpr uint32_t face_row_words = face_row_bytes / sizeof(uint32_t);
    constexpr uint32_t seed_bytes = seed_rows * face_row_bytes;
    constexpr uint32_t upper_faces_bytes = faces_per_tile_row * face_bytes;
    const uint8_t noc_id = noc.get_noc_id();
    const uint32_t self_x = my_x[noc_id];
    const uint32_t self_y = my_y[noc_id];
    UnicastEndpoint local_l1;

    if constexpr (seed_rows > 1) {
        // The RISC-V data cache may hold lines of this slot from an earlier block; the NoC just rewrote them.
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* last_store = nullptr;
        for (uint32_t p = 0; p < extent; ++p) {
            for (uint32_t j = 0; j < faces_per_tile_row; ++j) {
                volatile tt_l1_ptr uint32_t* face =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + p * page_bytes + j * face_bytes);
                for (uint32_t r = 1; r < seed_rows; ++r) {
                    for (uint32_t w = 0; w < face_row_words; ++w) {
                        face[r * face_row_words + w] = face[w];
                    }
                }
                last_store = face + seed_rows * face_row_words - 1;
            }
        }
        // A RISC-V store can retire before it lands in L1, and the NoC is a different L1 client: read
        // the last stored word back so every seed row is in L1 before the self-reads copy it.
        (void)ckernel::load_blocking(last_store);
    }

    for (uint32_t p = 0; p < extent; ++p) {
        for (uint32_t j = 0; j < faces_per_tile_row; ++j) {
            const uint32_t face_addr = slot + p * page_bytes + j * face_bytes;
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

    for (uint32_t p = 0; p < extent; ++p) {
        const uint32_t page_addr = slot + p * page_bytes;
        for (uint32_t fr = 1; fr < face_rows_per_tile; ++fr) {
            noc.async_read(
                local_l1,
                CoreLocalMem<uint32_t>(page_addr + fr * upper_faces_bytes),
                upper_faces_bytes,
                {.noc_x = self_x, .noc_y = self_y, .addr = page_addr},
                {});
        }
    }
    noc.async_read_barrier();
}

template <typename McastT>
FORCE_INLINE void run_sender(const McastT& mcast, const Noc& noc, CircularBuffer& cb) {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t src_start = get_arg_val<uint32_t>(0);
    const uint32_t src_count = get_arg_val<uint32_t>(1);
    const auto x_acc = TensorAccessor(x_args, x_addr, page_bytes);
    auto sender = mcast.optional_sender(noc);

    const uint32_t blocks = num_blocks(src_count);
    for (uint32_t block_idx = 0; block_idx < blocks; ++block_idx) {
        const uint32_t first = block_idx * block_src_col_tiles;
        const uint32_t extent = (src_count - first < block_src_col_tiles) ? src_count - first : block_src_col_tiles;
        // Nominal reserve/push on a ragged block keeps every core's write pointer in lockstep: the
        // sender's slot address is the receivers' landing address.
        cb.reserve_back(block_src_col_tiles);
        const uint32_t slot = cb.get_write_ptr();
        {
            DeviceZoneScopedN("load_source_block");
            load_source_block(noc, cb, x_acc, src_start + first, extent);
        }
        {
            DeviceZoneScopedN("expand_rows_block");
            expand_rows_block(noc, slot, extent);
        }
        {
            DeviceZoneScopedN("publish_block");
            // Push first: the local writer and the multicast only read E, so they overlap. send()
            // returns only after the source is read, so the next reserve cannot overwrite it.
            cb.push_back(block_src_col_tiles);
            if constexpr (McastT::active) {
                sender->send(slot, slot, block_bytes);
            }
        }
    }
}

template <typename McastT>
FORCE_INLINE void run_receiver(const McastT& mcast, const Noc& noc, CircularBuffer& cb) {
    const uint32_t src_count = get_arg_val<uint32_t>(1);
    auto receiver = mcast.receiver(noc);

    const uint32_t blocks = num_blocks(src_count);
    for (uint32_t block_idx = 0; block_idx < blocks; ++block_idx) {
        DeviceZoneScopedN("receive_block");
        // Reserve before the ready-ack: the ack tells the sender this slot is free to land in.
        cb.reserve_back(block_src_col_tiles);
        receiver.receive(block_idx);
        cb.push_back(block_src_col_tiles);
    }
}

template <bool IS_SENDER, typename McastT>
FORCE_INLINE void run(const McastT& mcast) {
    Noc noc;
    CircularBuffer cb(cb_expanded_tiles_id);
    if constexpr (IS_SENDER) {
        run_sender(mcast, noc, cb);
    } else {
        run_receiver(mcast, noc, cb);
    }
    // BB-03: multicast payload/flag writes and the receivers' ready increments must complete before exit.
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}

}  // namespace

void kernel_main() {
    constexpr auto mcast = McastArgs<
        get_named_compile_time_arg_val("x_mcast_ct_offset"),
        get_named_compile_time_arg_val("x_mcast_rt_offset")>();
    run<is_sender>(mcast);
}
