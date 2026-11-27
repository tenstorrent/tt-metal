// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_expand reader (NCRISC, NoC0).
//
// Injector (top core of a strip): load_block -> replicate_rows_block -> publish_block.
//   The group's x pages land in the CB slot, row 0 of every page is copied into all 32 rows in
//   place, the block goes to the local writer and is multicast down the strip into the same slot.
// Receiver (rest of the strip): receive_block -> hand the multicast slot to the local writer.
//
// Every core of a strip reserves and pushes the nominal block_pages on every block, including a
// ragged last one, so all write pointers stay in lockstep: the multicast lands at the injector's
// slot address, which must be the slot each receiver reserved.
//
// replicate_rows_block has no covering kernel_lib helper: tile_column_dataflow moves a column,
// fill_l1_range replicates a <= 4 B scalar, and a compute Row-broadcast copy passes values through
// the unpacker/FPU/packer (no bit identity for denormals / signalling NaNs). It is built here from
// RISC-V word stores and the self-aimed copy helpers of local_copy_helpers_dataflow.hpp.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/tile_constants.h"
#include "ckernel.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

namespace {

constexpr uint32_t cb_row_bcast_tiles = get_compile_time_arg_val(0);
constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
constexpr uint32_t block_pages = get_compile_time_arg_val(2);
constexpr bool is_injector = get_compile_time_arg_val(3) != 0;
constexpr uint32_t seed_rows = get_compile_time_arg_val(4);
constexpr auto x_args = TensorAccessorArgs<5>();

// A 32x32 tile is four 16x16 faces stored face-major, rows row-major inside a face:
// faces 0/1 hold tile rows 0-15 (left/right half), faces 2/3 hold rows 16-31.
using tt::constants::FACE_HEIGHT;
constexpr uint32_t tile_faces = tt::constants::TILE_HW / tt::constants::FACE_HW;
constexpr uint32_t face_bytes = page_bytes / tile_faces;
constexpr uint32_t row_bytes = face_bytes / FACE_HEIGHT;
constexpr uint32_t row_words = row_bytes / sizeof(uint32_t);
constexpr uint32_t seed_bytes = seed_rows * row_bytes;
constexpr uint32_t upper_faces_bytes = 2 * face_bytes;
static_assert(seed_rows >= 1 && FACE_HEIGHT % seed_rows == 0, "seed_rows must divide the face height");

// Row 0 of faces 0/1 is the only source: tile rows >= 1 of x are padding, and every one of them is
// overwritten here, so no out element depends on x padding.
//   1. RISC-V stores copy row 0 of faces 0/1 into rows [1, seed_rows) of the same face.
//   2. Self-aimed NoC reads copy those seed rows down the rest of faces 0/1.
//   3. One self-aimed NoC read per page copies faces 0/1 onto faces 2/3.
// Every page of the block goes through each step together: one fence per step per block.
FORCE_INLINE void replicate_rows_block(const Noc& noc, uint32_t slot_addr, uint32_t pages) {
    if constexpr (seed_rows > 1) {
        // Row 0 arrived over the NoC: drop any stale data-cache line before the RISC-V loads it.
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* last_store = nullptr;
        for (uint32_t k = 0; k < pages; ++k) {
            for (uint32_t f = 0; f < 2; ++f) {
                volatile tt_l1_ptr uint32_t* face =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_addr + k * page_bytes + f * face_bytes);
                uint32_t row0[row_words];
                for (uint32_t w = 0; w < row_words; ++w) {
                    row0[w] = face[w];
                }
                for (uint32_t r = 1; r < seed_rows; ++r) {
                    for (uint32_t w = 0; w < row_words; ++w) {
                        face[r * row_words + w] = row0[w];
                    }
                }
                last_store = face + seed_rows * row_words - 1;
            }
        }
        // A baby-RISC-V store can retire before it lands in L1, and the NoC is a separate L1 client
        // with no program-order guarantee: read the last store back so all seeds are in L1 before
        // a NoC read (step 2/3, the multicast, or the writer) sees them.
        (void)ckernel::load_blocking(last_store);
    }

    if constexpr (seed_rows < FACE_HEIGHT) {
        for (uint32_t k = 0; k < pages; ++k) {
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t face_addr = slot_addr + k * page_bytes + f * face_bytes;
                set_read_state<seed_bytes>(noc, face_addr);
                for (uint32_t dst = face_addr + seed_bytes; dst < face_addr + face_bytes; dst += seed_bytes) {
                    read_with_state<seed_bytes>(noc, dst, face_addr);
                }
            }
        }
        noc.async_read_barrier();
    }

    for (uint32_t k = 0; k < pages; ++k) {
        const uint32_t page_addr = slot_addr + k * page_bytes;
        set_read_state<upper_faces_bytes>(noc, page_addr);
        read_with_state<upper_faces_bytes>(noc, page_addr + upper_faces_bytes, page_addr);
    }
    noc.async_read_barrier();
}

template <typename Publish>
FORCE_INLINE void injector_blocks(
    const Noc& noc, CircularBuffer& cb, uint32_t group_q_start, uint32_t group_pages, Publish&& publish) {
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), page_bytes);
    uint32_t q = group_q_start;
    for (uint32_t remaining = group_pages; remaining > 0;) {
        const uint32_t pages = remaining < block_pages ? remaining : block_pages;
        uint32_t slot_addr;
        {
            DeviceZoneScopedN("G6E-load");
            cb.reserve_back(block_pages);
            slot_addr = cb.get_write_ptr();
            for (uint32_t k = 0; k < pages; ++k) {
                noc.async_read(x, cb, page_bytes, {.page_id = q + k}, {.offset_bytes = k * page_bytes});
            }
            noc.async_read_barrier();
        }
        {
            DeviceZoneScopedN("G6E-replicate");
            replicate_rows_block(noc, slot_addr, pages);
        }
        {
            DeviceZoneScopedN("G6E-publish");
            cb.push_back(block_pages);
            publish(slot_addr, pages * page_bytes);
        }
        q += pages;
        remaining -= pages;
    }
}

template <typename Bcast>
FORCE_INLINE void run_injector(const Bcast& bcast, const Noc& noc, CircularBuffer& cb, uint32_t q0, uint32_t pages) {
    if constexpr (Bcast::active) {
        // send() waits for every receiver's ready-ack, multicasts the slot to the same address on
        // the strip, then the data-ready flag; with the default source guard it returns only once
        // the NoC has read the slot, so the next reserve_back may reuse it.
        auto sender = bcast.sender(noc);
        injector_blocks(
            noc, cb, q0, pages, [&](uint32_t slot_addr, uint32_t bytes) { sender.send(slot_addr, slot_addr, bytes); });
    } else {
        injector_blocks(noc, cb, q0, pages, [](uint32_t, uint32_t) {});
    }
    DeviceZoneScopedN("G6E-drain");
    noc.async_write_barrier();
}

template <typename Bcast>
FORCE_INLINE void run_receiver(const Bcast& bcast, const Noc& noc, CircularBuffer& cb, uint32_t group_pages) {
    auto receiver = bcast.receiver(noc);
    const uint32_t num_blocks = (group_pages + block_pages - 1) / block_pages;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        DeviceZoneScopedN("G6E-receive-wait");
        cb.reserve_back(block_pages);
        receiver.receive(block);
        cb.push_back(block_pages);
    }
    // The ready-acks are remote atomic increments that receive() does not drain.
    DeviceZoneScopedN("G6E-drain");
    noc.async_atomic_barrier();
}

template <bool INJECTOR, typename Bcast>
FORCE_INLINE void run(const Bcast& bcast) {
    const uint32_t group_q_start = get_arg_val<uint32_t>(0);
    const uint32_t group_pages = get_arg_val<uint32_t>(1);
    Noc noc;
    CircularBuffer cb(cb_row_bcast_tiles);
    if constexpr (INJECTOR) {
        run_injector(bcast, noc, cb, group_q_start, group_pages);
    } else {
        run_receiver(bcast, noc, cb, group_pages);
    }
}

}  // namespace

void kernel_main() {
    constexpr auto bcast = McastArgs<
        get_named_compile_time_arg_val("bcast_ct_offset"),
        get_named_compile_time_arg_val("bcast_rt_offset")>();
    run<is_injector>(bcast);
}
