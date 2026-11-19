// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm member reader (NCRISC / NoC0) — acquire_x_block.
//
// cb_x holds this member's whole block (core_row_tiles x pitch tiles), written ONCE per call: it is the fabric
// landing slot for the remote slice(s) this member owns (the neighbour's port sender writes at this core's cb_x
// address — identical on every core of every chip) and is filled from the input for the own slice.
//   own part   : TensorAccessor reads (input tile t * sw + c0 + j -> row t, column off + j), one barrier per
//                tile-row; slice-last tile tails (columns >= v) zero-filled; then one local increment of each
//                sending port's sem_own_loaded (the ports read the block straight out of this L1).
//   remote part: wait sem_landed >= expected_landings (the port readers notify per landed entry), zero the slice
//                tails of the landed copy (padding only — a concurrent relay read of it is harmless: every
//                receiver zero-fills again).
// One cb_reserve_back / cb_push_back of the whole block. The 1.0 reduce scaler (and, LN with W_local % 32 != 0, the
// 0/1 row-0 mask of a slice-last tile) are built while the remote slices are in flight.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "all_gather_rms_norm_common.hpp"

using namespace agrms_cb;

constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
constexpr uint32_t pitch = get_compile_time_arg_val(1);
constexpr uint32_t sw = get_compile_time_arg_val(2);
constexpr uint32_t v = get_compile_time_arg_val(3);  // valid columns of a slice-last tile
constexpr uint32_t elem_bytes = get_compile_time_arg_val(4);
constexpr uint32_t sem_landed_id = get_compile_time_arg_val(5);
constexpr uint32_t sem_own_loaded_id = get_compile_time_arg_val(6);
constexpr auto x_args = TensorAccessorArgs<7>();

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;

// Zero columns [v, 32) of the tile at `tile_addr` (v in 1..31). Whole faces through the NoC zero engine; partial
// face rows (< 16 datums each) by RISC-V stores (only the padding positions, never the whole tile).
FORCE_INLINE void zero_tail(const Noc& noc, const CircularBuffer& cb, uint32_t tile_addr) {
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * elem_bytes;
    constexpr uint32_t face_row_bytes = FACE_DIM * elem_bytes;
    const uint32_t rel = tile_addr - cb.get_write_ptr();
    if constexpr (v <= FACE_DIM) {
        noc.async_write_zeros(cb, face_bytes, {.offset_bytes = rel + face_bytes});
        noc.async_write_zeros(cb, face_bytes, {.offset_bytes = rel + 3 * face_bytes});
        if constexpr (v < FACE_DIM) {
            for (uint32_t f = 0; f < 4; f += 2) {
                for (uint32_t row = 0; row < FACE_DIM; ++row) {
                    dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                        tile_addr + f * face_bytes + row * face_row_bytes + v * elem_bytes,
                        (FACE_DIM - v) * elem_bytes,
                        0);
                }
            }
        }
        noc.write_zeros_l1_barrier();
    } else {
        for (uint32_t f = 1; f < 4; f += 2) {
            for (uint32_t row = 0; row < FACE_DIM; ++row) {
                dataflow_kernel_lib::fill_l1_range<elem_bytes>(
                    tile_addr + f * face_bytes + row * face_row_bytes + (v - FACE_DIM) * elem_bytes,
                    (TILE_DIM - v) * elem_bytes,
                    0);
            }
        }
    }
}

FORCE_INLINE uint32_t arg(uint32_t i) { return get_arg_val<uint32_t>(i); }

void kernel_main() {
    const uint32_t x_addr = arg(0);
    const uint32_t row_start = arg(1);
    const uint32_t rows = arg(2);
    const uint32_t own_n = arg(3);
    const uint32_t own_c0 = arg(4);
    const uint32_t own_off = arg(5);
    const uint32_t expected_landings = arg(6);
    const uint32_t n_sig = arg(7);
    const uint32_t sig_idx = 8;
    const uint32_t n_tails = arg(sig_idx + n_sig);
    const uint32_t tails_idx = sig_idx + n_sig + 1;

    Noc noc;
    CircularBuffer cb(cb_x);
    cb_reserve_back(cb_x, rows * pitch);
    const uint32_t base = get_write_ptr(cb_x);

    auto zero_tails = [&](bool own_only) {
        if constexpr (v < TILE_DIM) {
            for (uint32_t i = 0; i < n_tails; ++i) {
                const uint32_t col = arg(tails_idx + i);
                if (own_only && !(own_n > 0 && col == own_off + own_n - 1 && own_c0 + own_n == sw)) {
                    continue;
                }
                for (uint32_t r = 0; r < rows; ++r) {
                    zero_tail(noc, cb, base + (r * pitch + col) * tile_bytes);
                }
            }
        }
    };

    if (own_n > 0) {
        AGZ("rd_own");
        const auto x_acc = TensorAccessor(x_args, x_addr, tile_bytes);
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t first = (row_start + r) * sw + own_c0;
            const uint32_t dst = base + (r * pitch + own_off) * tile_bytes;
            for (uint32_t j = 0; j < own_n; ++j) {
                noc_async_read(x_acc.get_noc_addr(first + j), dst + j * tile_bytes, tile_bytes);
            }
        }
        noc_async_read_barrier();
        zero_tails(true);
        const uint32_t own_sem = get_semaphore(sem_own_loaded_id);
        for (uint32_t i = 0; i < n_sig; ++i) {
            const uint32_t xy = arg(sig_idx + i);
            noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, own_sem), 1);
        }
    }

    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_ones, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
#ifdef AGRMS_MASK
    dataflow_kernel_lib::prepare_reduce_mask<cb_mask, ckernel::ReduceDim::REDUCE_ROW>(v);
#endif

#ifdef AGRMS_DENSE
    // 0/1 shift matrices of the dense assembly: S[e][e + delta] = 1 for e in [e0, e1) (output dtype). Zeroed through
    // the NoC zero engine, then only the <= 32 ones are stored.
    {
        const uint32_t d_idx = tails_idx + n_tails;
        const uint32_t n_mats = arg(d_idx);
        CircularBuffer cbm(cb_shift_mats);
        cb_reserve_back(cb_shift_mats, n_mats);
        const uint32_t mbase = get_write_ptr(cb_shift_mats);
        for (uint32_t i = 0; i < n_mats; ++i) {
            noc.async_write_zeros(cbm, tile_bytes, {.offset_bytes = i * tile_bytes});
        }
        noc.write_zeros_l1_barrier();
        for (uint32_t i = 0; i < n_mats; ++i) {
            const uint32_t w = arg(d_idx + 1 + i);
            const uint32_t e0 = w & 0xFF, e1 = (w >> 8) & 0xFF;
            const int32_t delta = static_cast<int32_t>(w >> 16) - 64;
            for (uint32_t e = e0; e < e1; ++e) {
                const uint32_t c = static_cast<uint32_t>(static_cast<int32_t>(e) + delta);
                const uint32_t idx = ((e / FACE_DIM) * 2 + c / FACE_DIM) * (FACE_DIM * FACE_DIM) +
                                     (e % FACE_DIM) * FACE_DIM + (c % FACE_DIM);
                const uint32_t addr = mbase + i * tile_bytes + idx * elem_bytes;
                if constexpr (elem_bytes == 4) {
                    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = 0x3F800000u;
                } else {
                    *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr) = 0x3F80u;
                }
            }
        }
        cb_push_back(cb_shift_mats, n_mats);
    }
#endif

#if defined(AGRMS_OWNER) && defined(AGRMS_LN)
    // owner combine weights: n_m at column 0 of row m % 32 of tile m / 32 (accumulator format), zero elsewhere —
    // zeroed through the NoC zero engine, then only the k weights are stored.
    {
        uint32_t w_idx = tails_idx + n_tails;
#ifdef AGRMS_DENSE
        w_idx += 1 + arg(w_idx);
#endif
        const uint32_t k = arg(w_idx);
        const uint32_t gp = (k + agrms_owner::ROW_GATHER - 1) / agrms_owner::ROW_GATHER;
        const uint32_t acc_tile = get_tile_size(cb_wcol);
        CircularBuffer cbw(cb_wcol);
        cb_reserve_back(cb_wcol, gp);
        const uint32_t wbase = get_write_ptr(cb_wcol);
        for (uint32_t g = 0; g < gp; ++g) {
            noc.async_write_zeros(cbw, acc_tile, {.offset_bytes = g * acc_tile});
        }
        noc.write_zeros_l1_barrier();
        for (uint32_t m = 0; m < k; ++m) {
            const uint32_t rho = m % agrms_owner::ROW_GATHER;
            const uint32_t idx = (rho < FACE_DIM ? 0 : 2 * FACE_DIM * FACE_DIM) + (rho % FACE_DIM) * FACE_DIM;
            const uint32_t bits = arg(w_idx + 1 + m);
            const uint32_t tile = wbase + (m / agrms_owner::ROW_GATHER) * acc_tile;
            if (acc_tile == 4096) {
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile + idx * 4) = bits;
            } else {  // bf16 accumulator: round to nearest even
                const uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
                *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile + idx * 2) = static_cast<uint16_t>(rounded >> 16);
            }
        }
        cb_push_back(cb_wcol, gp);
    }
#endif

    if (expected_landings > 0) {
        AGZ("rd_land");
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_landed_id)), expected_landings);
        zero_tails(false);
    }
    cb_push_back(cb_x, rows * pitch);
    noc_async_atomic_barrier();
}
