// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// width_split reduce-case writer (BRISC / NoC1): the cross-core combine + the output writes.
//
// Combine (k > 1 only), one round per tile-row, symmetric all-gather of the partials:
//   * every member q of a k-member group writes its partial into slot q of EVERY member's
//     cb_gather (its own included), then bumps each member's semaphore by 1;
//   * every member waits until its semaphore has counted k arrivals for this row, then hands the
//     k slots to compute, which sums them.
// There is no root and no second (broadcast-back) round: each member ends up with all k
// partials after ONE exchange. For k = 2 this is a symmetric swap. Cost per member per row:
// k writes of the partial + k semaphore increments.
//
// Partial size: a REDUCE_ROW result lives in column 0 of the tile, i.e. in faces 0 and 2 (rows
// 0-15 and 16-31). Only those two 16x16 faces are sent (2 x face_bytes = 1 KB of the 2 KB bf16
// tile). Faces 1 and 3 of a gather slot are never written; compute adds them as garbage but the
// column broadcast only ever reads column 0 of faces 0 and 2, so they never reach the output.
//
// Flow control: cb_gather holds 2 rows of k slots (2k tiles) and is pushed k tiles per row, so
// row n's slots sit at the CB base + (n % 2) * k tiles on every member (CB addresses are the same
// on every Tensix core of the program). A member can be at most one row ahead of a peer (it needs
// that peer's partial for row n before it can finish row n), so the parity slots never collide.
// The semaphore counts cumulatively (row n is ready at (n + 1) * k): no reset round trip.
//
// The data writes and the semaphore increment go to the same destination on the same NoC and the
// same unicast-write VC, which delivers them in order, so a flush (writes have left this core),
// not a full barrier (writes acknowledged), is enough before signalling.
//
// WS_ROW_MAJOR (Layout::ROW_MAJOR output): the combine is unchanged; the output of a tile-row is
// the untilized slice (S tile-sized pages = 32 rows x slice_bytes) written back as 32 stick pieces
// of slice_bytes at byte offset slice_offset into each output stick, one NoC barrier per tile-row.
// The partials are tiles either way, so face_bytes is a tile face (the CB page is a tile).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_send = 2;
    constexpr uint32_t cb_gather = 3;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(1);
    constexpr uint32_t sem_id = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();
#ifdef WS_ROW_MAJOR
    constexpr uint32_t tile_bytes = 32 * 32 * 2;  // CB pages are bf16 tiles; page_bytes is a stick
#else
    constexpr uint32_t tile_bytes = page_bytes;
#endif
    constexpr uint32_t face_bytes = tile_bytes / 4;  // bf16 32x32 tile = 4 faces of 16x16

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t wt = get_arg_val<uint32_t>(1);
    const uint32_t slice = get_arg_val<uint32_t>(2);
    const uint32_t slice_start = get_arg_val<uint32_t>(3);
    const uint32_t row_start = get_arg_val<uint32_t>(4);
    const uint32_t num_rows = get_arg_val<uint32_t>(5);
    const uint32_t blk = get_arg_val<uint32_t>(6);
    const uint32_t k = get_arg_val<uint32_t>(7);
    const uint32_t member = get_arg_val<uint32_t>(8);
    // args 9 .. 9 + 2k - 1: NoC (x, y) of the group's members, in member order.
#ifdef WS_ROW_MAJOR
    // args 9 + 2k, 10 + 2k: bytes of each stick this core owns, and their byte offset in the stick.
    const uint32_t slice_bytes = get_arg_val<uint32_t>(9 + 2 * k);
    const uint32_t slice_offset = get_arg_val<uint32_t>(10 + 2 * k);
#endif

    const auto out_acc = TensorAccessor(out_args, dst_addr, page_bytes);
    Noc noc;
    Semaphore<> arrived(sem_id);
    const uint32_t gather_base = get_write_ptr(cb_gather);

    uint32_t n = 0;  // tile-rows combined so far (selects the parity half of cb_gather)
    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t r = 0; r < num_rows; ++r) {
            if (k > 1) {
                cb_wait_front(cb_send, 1);
                const uint32_t src = get_read_ptr(cb_send);
                const uint32_t slot = gather_base + ((n & 1) * k + member) * tile_bytes;
                for (uint32_t q = 0; q < k; ++q) {
                    const uint32_t x = get_arg_val<uint32_t>(9 + 2 * q);
                    const uint32_t y = get_arg_val<uint32_t>(10 + 2 * q);
                    noc_async_write(src, get_noc_addr(x, y, slot), face_bytes);  // face 0
                    noc_async_write(
                        src + 2 * face_bytes, get_noc_addr(x, y, slot + 2 * face_bytes), face_bytes);  // face 2
                }
                noc_async_writes_flushed();
                for (uint32_t q = 0; q < k; ++q) {
                    arrived.up(noc, get_arg_val<uint32_t>(9 + 2 * q), get_arg_val<uint32_t>(10 + 2 * q), 1);
                }
                cb_pop_front(cb_send, 1);

                cb_reserve_back(cb_gather, k);
                arrived.wait_min((n + 1) * k);
                cb_push_back(cb_gather, k);
                ++n;
            }

#ifdef WS_ROW_MAJOR
            const uint32_t first_stick = (row_start + r) * 32;
            cb_wait_front(cb_out, slice);
            const uint32_t l1_read_addr = get_read_ptr(cb_out);
            for (uint32_t s = 0; s < 32; ++s) {
                noc_async_write(
                    l1_read_addr + s * slice_bytes, out_acc.get_noc_addr(first_stick + s, slice_offset), slice_bytes);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, slice);
#else
            const uint32_t first_page = (row_start + r) * wt + slice_start;
            for (uint32_t p = 0; p < slice; p += blk) {
                cb_wait_front(cb_out, blk);
                const uint32_t l1_read_addr = get_read_ptr(cb_out);
                for (uint32_t i = 0; i < blk; ++i) {
                    noc_async_write(
                        l1_read_addr + i * page_bytes, out_acc.get_noc_addr(first_page + p + i), page_bytes);
                }
                noc_async_write_barrier();
                cb_pop_front(cb_out, blk);
            }
#endif
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
