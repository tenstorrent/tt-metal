// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// permute staged_gather: one data-movement kernel, gather_threads worker harts, no rings.
//
// The host cuts the canonical rank-4 input into boxes with extents e[0..3] (choose_gather_block()), numbered in
// output-axis order, and deals them to clusters (split_work_to_cores); worker thread t of a cluster takes the
// cluster's boxes t, t+G, ... For each box it runs three block operations (op_design.md "Block schedule"):
//
//   gather_block   read the box's whole input footprint into this thread's staging region (scratchpad), every
//                  read tagged trid 0; ONE read barrier and ONE L2 invalidate of the staging per box.
//   shuffle_block  assemble each output page of the box in an output slot, element by element (RISC-V moves).
//   store_block    flush the slot to L1 and write it to its output page (+ offset), out_slots writes in flight;
//                  one writes-flushed wait per out_slots pages.
//
// The staging offset of an input element is a sum of one term per input axis, off(j, r_j), r_j its offset in
// the box: RM rows are e[3]-element slots (64 B aligned), so off(j, r) = r * stg[j]; TILE staging is whole tiles,
// and the in-tile face layout is separable too: row y -> (y>>4)*512 + (y&15)*16, column x -> (x>>4)*256 + (x&15).
// So the shuffle is a base per output page plus one table lookup per output element.
//
// Explicit sync without rings: every NoC transfer of this kernel carries user transaction id 0 (target page,
// "Ring protocols"), which also leaves the hart's sticky tag at 0 for the next program.
//
// DM caches (write-back L1 D$ + shared L2): staging lines are invalidated after the read barrier (the NoC wrote
// them behind the cache); each slot is flushed before its NoC write. Thread regions are 64-B aligned and
// disjoint, so no cache line is shared between threads.

#include <cstdint>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/kernel_thread_globals.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/quasar/cache.h"

namespace {

constexpr uint32_t TILE = 32;
constexpr uint32_t INVALID_OFF = 0xFFFFFFFFu;

// In-tile element offset of row y / column x (face-major 32x32 tile of four 16x16 faces).
inline uint32_t tile_row_off(uint32_t y) { return ((y >> 4) << 9) + ((y & 15u) << 4); }
inline uint32_t tile_col_off(uint32_t x) { return ((x >> 4) << 8) + (x & 15u); }

template <bool tile_layout>
inline uint32_t axis_off(uint32_t j, uint32_t r, const uint32_t* stg) {
    if constexpr (tile_layout) {
        if (j == 2) {
            return (r >> 5) * stg[2] + tile_row_off(r & 31u);
        }
        if (j == 3) {
            return (r >> 5) * stg[3] + tile_col_off(r & 31u);
        }
    }
    return r * stg[j];
}

inline uint32_t align_up(uint32_t x, uint32_t a) { return (x + a - 1) / a * a; }

}  // namespace

void kernel_main() {
    constexpr uint32_t G = get_arg(args::gather_threads);
    constexpr uint32_t OUT_SLOTS = get_arg(args::out_slots);
    constexpr uint32_t EB = get_arg(args::element_bytes);
    constexpr bool TILE_LAYOUT = get_arg(args::tile_layout) != 0;
    static_assert(EB == 2 || EB == 4, "element moves are 16- or 32-bit");
    using T = std::conditional_t<EB == 2, uint16_t, uint32_t>;
    constexpr uint32_t TILE_BYTES = TILE * TILE * EB;

    const uint32_t num_blocks = get_arg(args::num_blocks);
    const uint32_t start_block = get_arg(args::start_block);

    const uint32_t s[4] = {get_arg(args::s0), get_arg(args::s1), get_arg(args::s2), get_arg(args::s3)};
    const uint32_t e[4] = {get_arg(args::e0), get_arg(args::e1), get_arg(args::e2), get_arg(args::e3)};
    const uint32_t nb[4] = {get_arg(args::nb0), get_arg(args::nb1), get_arg(args::nb2), get_arg(args::nb3)};
    const uint32_t d[4] = {get_arg(args::d0), get_arg(args::d1), get_arg(args::d2), get_arg(args::d3)};
    const uint32_t ps[4] = {get_arg(args::ps0), get_arg(args::ps1), get_arg(args::ps2), get_arg(args::ps3)};
    const uint32_t stg[4] = {get_arg(args::stg0), get_arg(args::stg1), get_arg(args::stg2), get_arg(args::stg3)};
    const uint32_t og1 = get_arg(args::og1);
    const uint32_t og2 = get_arg(args::og2);
    const uint32_t og3 = get_arg(args::og3);
    const uint32_t per_thread_bytes = get_arg(args::per_thread_bytes);
    const uint32_t staging_bytes = get_arg(args::staging_bytes);
    const uint32_t slot_stride = get_arg(args::slot_stride);
    const uint32_t row_stride_bytes = get_arg(args::row_stride_bytes);
    const uint32_t xfer_align = get_arg(args::xfer_align);
    const uint32_t a = d[2];
    const uint32_t b = d[3];

    const uint32_t tid = get_my_thread_id();
    Scratchpad<uint8_t> pad(scratch::gather);
    const uint32_t region = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&pad[0])) +
                            tid * per_thread_bytes;
    const uint32_t staging = region;
    const uint32_t slots = region + staging_bytes;
    const T* stage = reinterpret_cast<const T*>(static_cast<uintptr_t>(staging));

    Noc noc;
    const auto acc_in = TensorAccessor(tensor::input);
    const auto acc_out = TensorAccessor(tensor::output);
    constexpr NocOptVals user_txn = {.trid = 0};

    uint32_t slot_idx = 0;
    uint32_t col_off[TILE];

    for (uint32_t lb = tid; lb < num_blocks; lb += G) {
        // Box coordinates: block index is mixed radix over nb[], output axis d[0] slowest ... d[3] fastest.
        uint32_t blk = start_block + lb;
        uint32_t start[4], act[4];
        for (int k = 3; k >= 0; --k) {
            const uint32_t j = d[k];
            start[j] = (blk % nb[j]) * e[j];
            blk /= nb[j];
        }
        for (uint32_t j = 0; j < 4; ++j) {
            act[j] = s[j] - start[j] < e[j] ? s[j] - start[j] : e[j];
        }

        // ---- gather_block: the box's input footprint into staging, one barrier + one invalidate ----
        if constexpr (TILE_LAYOUT) {
            const uint32_t nh = (e[2] + TILE - 1) / TILE, nw = (e[3] + TILE - 1) / TILE;
            const uint32_t nh_act = (act[2] + TILE - 1) / TILE, nw_act = (act[3] + TILE - 1) / TILE;
            const uint32_t h0 = start[2] / TILE, w0 = start[3] / TILE;
            for (uint32_t r0 = 0; r0 < act[0]; ++r0) {
                for (uint32_t r1 = 0; r1 < act[1]; ++r1) {
                    const uint32_t lead_page = (start[0] + r0) * ps[0] + (start[1] + r1) * ps[1];
                    const uint32_t lead_slot = (r0 * e[1] + r1) * nh;
                    for (uint32_t rh = 0; rh < nh_act; ++rh) {
                        for (uint32_t rw = 0; rw < nw_act; ++rw) {
                            const uint32_t page = lead_page + (h0 + rh) * ps[2] + (w0 + rw);
                            const uint32_t dst = staging + ((lead_slot + rh) * nw + rw) * TILE_BYTES;
                            noc.async_read<NocOptions::TXN_ID>(
                                acc_in, CoreLocalMem<uint8_t>(dst), TILE_BYTES, {.page_id = page, .offset_bytes = 0},
                                {.offset_bytes = 0}, user_txn);
                        }
                    }
                }
            }
        } else {
            const uint32_t rd_bytes = align_up(act[3] * EB, xfer_align);
            const uint32_t rd_offset = start[3] * EB;
            for (uint32_t r0 = 0; r0 < act[0]; ++r0) {
                for (uint32_t r1 = 0; r1 < act[1]; ++r1) {
                    for (uint32_t r2 = 0; r2 < act[2]; ++r2) {
                        const uint32_t page = (start[0] + r0) * ps[0] + (start[1] + r1) * ps[1] + (start[2] + r2);
                        const uint32_t dst = staging + ((r0 * e[1] + r1) * e[2] + r2) * row_stride_bytes;
                        noc.async_read<NocOptions::TXN_ID>(
                            acc_in, CoreLocalMem<uint8_t>(dst), rd_bytes, {.page_id = page, .offset_bytes = rd_offset},
                            {.offset_bytes = 0}, user_txn);
                    }
                }
            }
        }
        noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
        invalidate_l2_cache_range(staging, staging_bytes);

        // ---- shuffle_block + store_block: one output page per slot ----
        const uint32_t d0 = d[0], d1 = d[1];
        if constexpr (TILE_LAYOUT) {
            const uint32_t n_rows = (act[a] + TILE - 1) / TILE, n_cols = (act[b] + TILE - 1) / TILE;
            for (uint32_t k0 = 0; k0 < act[d0]; ++k0) {
                for (uint32_t k1 = 0; k1 < act[d1]; ++k1) {
                    const uint32_t base = axis_off<true>(d0, k0, stg) + axis_off<true>(d1, k1, stg);
                    const uint32_t lead_page = ((start[d0] + k0) * og1 + (start[d1] + k1)) * og2;
                    for (uint32_t rh = 0; rh < n_rows; ++rh) {
                        for (uint32_t rw = 0; rw < n_cols; ++rw) {
                            for (uint32_t x = 0; x < TILE; ++x) {
                                const uint32_t rb = rw * TILE + x;
                                col_off[x] = rb < act[b] ? axis_off<true>(b, rb, stg) : INVALID_OFF;
                            }
                            if (slot_idx == OUT_SLOTS) {
                                noc.async_writes_flushed<NocOptions::TXN_ID>(user_txn);
                                slot_idx = 0;
                            }
                            const uint32_t slot_addr = slots + slot_idx * slot_stride;
                            T* slot = reinterpret_cast<T*>(static_cast<uintptr_t>(slot_addr));
                            for (uint32_t y = 0; y < TILE; ++y) {
                                const uint32_t ra = rh * TILE + y;
                                T* out_row = slot + tile_row_off(y);
                                if (ra < act[a]) {
                                    const T* in_row = stage + base + axis_off<true>(a, ra, stg);
                                    for (uint32_t x = 0; x < TILE; ++x) {
                                        const uint32_t off = col_off[x];
                                        out_row[tile_col_off(x)] = off == INVALID_OFF ? T(0) : in_row[off];
                                    }
                                } else {
                                    for (uint32_t x = 0; x < TILE; ++x) {
                                        out_row[tile_col_off(x)] = T(0);
                                    }
                                }
                            }
                            flush_l2_cache_range(slot_addr, TILE_BYTES);
                            const uint32_t page = (lead_page + start[a] / TILE + rh) * og3 + start[b] / TILE + rw;
                            noc.async_write<NocOptions::TXN_ID>(
                                CoreLocalMem<uint8_t>(slot_addr), acc_out, TILE_BYTES, {.offset_bytes = 0},
                                {.page_id = page, .offset_bytes = 0}, user_txn);
                            ++slot_idx;
                        }
                    }
                }
            }
        } else {
            const uint32_t d2 = d[2];
            const uint32_t wr_bytes = align_up(act[b] * EB, xfer_align);
            const uint32_t wr_offset = start[b] * EB;
            const uint32_t stg_b = stg[b];
            for (uint32_t k0 = 0; k0 < act[d0]; ++k0) {
                for (uint32_t k1 = 0; k1 < act[d1]; ++k1) {
                    const uint32_t base01 = axis_off<false>(d0, k0, stg) + axis_off<false>(d1, k1, stg);
                    const uint32_t lead_page = ((start[d0] + k0) * og1 + (start[d1] + k1)) * og2 + start[d2];
                    for (uint32_t k2 = 0; k2 < act[d2]; ++k2) {
                        if (slot_idx == OUT_SLOTS) {
                            noc.async_writes_flushed<NocOptions::TXN_ID>(user_txn);
                            slot_idx = 0;
                        }
                        const uint32_t slot_addr = slots + slot_idx * slot_stride;
                        T* slot = reinterpret_cast<T*>(static_cast<uintptr_t>(slot_addr));
                        const T* in = stage + base01 + axis_off<false>(d2, k2, stg);
                        for (uint32_t x = 0; x < act[b]; ++x) {
                            slot[x] = *in;
                            in += stg_b;
                        }
                        flush_l2_cache_range(slot_addr, wr_bytes);
                        noc.async_write<NocOptions::TXN_ID>(
                            CoreLocalMem<uint8_t>(slot_addr), acc_out, wr_bytes, {.offset_bytes = 0},
                            {.page_id = lead_page + k2, .offset_bytes = wr_offset}, user_txn);
                        ++slot_idx;
                    }
                }
            }
        }
    }
    // Writes from the slots must land before the program ends (firmware waits only for implicit-sync writes).
    noc.async_write_barrier<NocOptions::TXN_ID>(user_txn);
}
