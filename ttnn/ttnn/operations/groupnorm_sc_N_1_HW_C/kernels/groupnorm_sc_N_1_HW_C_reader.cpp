// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// groupnorm_sc_N_1_HW_C — reader (data movement).
//
// Per image owned by this core:
//   pass 1: for each column group cg: for each row chunk rc: load the chunk_rows x cols tile block of x
//           (TILE input: DRAM tiles -> cb_x_pass1 [+ cb_x_pass2 credit when resident];
//            RM input:   sticks -> cb_x_rm, compute tilizes), then generate E^T for cg -> cb_membership.
//   pass 2: for each column group cg: generate E for cg -> cb_membership, gamma/beta row-0 tiles for cg,
//           then (streaming regime only) re-stream the x chunks of cg into cb_x_pass2 / cb_x_rm.
//
// Constant tiles: the reduce scaler (once per kernel). Membership / affine-row pages are zeroed over the NoC
// (DM engine) before the few real lanes are written by the RISC.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

namespace {

constexpr uint32_t ONE_F32_BITS = 0x3F800000u;

// Byte offset of element (r, c) inside a 32x32 tile made of four 16x16 faces of `elem_bytes` elements.
constexpr uint32_t tile_elem_offset(uint32_t r, uint32_t c, uint32_t elem_bytes) {
    return (((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15)) * elem_bytes;
}

}  // namespace

void kernel_main() {
    // ---------------- compile-time args ----------------
    constexpr uint32_t cb_x_pass1 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x_pass2 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x_rm = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t cb_membership = get_compile_time_arg_val(4);
    constexpr uint32_t cb_gamma_row = get_compile_time_arg_val(5);
    constexpr uint32_t cb_beta_row = get_compile_time_arg_val(6);
    constexpr bool is_rm = get_compile_time_arg_val(7) != 0;
    constexpr bool resident = get_compile_time_arg_val(8) != 0;
    constexpr bool has_gamma = get_compile_time_arg_val(9) != 0;
    constexpr bool has_beta = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t chunk_rows = get_compile_time_arg_val(11);
    constexpr uint32_t cols = get_compile_time_arg_val(12);
    constexpr uint32_t Kg = get_compile_time_arg_val(13);
    constexpr uint32_t num_col_groups = get_compile_time_arg_val(14);
    constexpr uint32_t num_row_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t x_tile_bytes = get_compile_time_arg_val(16);
    constexpr uint32_t x_elem_size = get_compile_time_arg_val(17);
    constexpr uint32_t x_page_bytes = get_compile_time_arg_val(18);
    constexpr uint32_t g_elem_size = get_compile_time_arg_val(19);
    constexpr bool g_is_tile = get_compile_time_arg_val(20) != 0;
    constexpr bool g_is_bfp = get_compile_time_arg_val(21) != 0;  // bfloat8_b affine: whole-page reads
    [[maybe_unused]] constexpr uint32_t g_page_bytes = get_compile_time_arg_val(22);
    constexpr uint32_t C = get_compile_time_arg_val(23);
    constexpr uint32_t G = get_compile_time_arg_val(24);
    constexpr uint32_t HW = get_compile_time_arg_val(25);
    constexpr uint32_t Ht = get_compile_time_arg_val(26);
    constexpr uint32_t Ct = get_compile_time_arg_val(27);
    constexpr uint32_t TA_BASE = 28;

    constexpr auto x_args = TensorAccessorArgs<TA_BASE>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    // ---------------- runtime args ----------------
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    [[maybe_unused]] const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    [[maybe_unused]] const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t image_begin = get_arg_val<uint32_t>(3);
    const uint32_t image_count = get_arg_val<uint32_t>(4);
    const uint32_t image_stride = get_arg_val<uint32_t>(5);
    const uint32_t row_begin = get_arg_val<uint32_t>(6);
    const uint32_t col_begin = get_arg_val<uint32_t>(7);
    const uint32_t active = get_arg_val<uint32_t>(8);

    if (active == 0) {
        return;  // idle core inside an image rectangle: no reads, no records
    }

    constexpr uint32_t Cg = C / G;
    constexpr uint32_t chunk = chunk_rows * cols;
    constexpr uint32_t membership_tiles = cols * Kg;
    constexpr uint32_t f32_tile_bytes = get_tile_size(cb_membership);

    Noc noc;
    CircularBuffer membership_cb(cb_membership);

    // ---------------- constants ----------------
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>();

    const auto x_acc = TensorAccessor(x_args, x_addr, x_page_bytes);

    // E^T (pass 1, transposed = true) or E (pass 2) for column group cg: cols x Kg Float32 0/1 tiles,
    // tile index T_local * Kg + kg. E_T[g', c] = 1 iff channel 32T + c belongs to group 32kg + g'.
    auto fill_membership = [&](uint32_t cg, bool transposed) {
        cb_reserve_back(cb_membership, membership_tiles);
        noc.async_write_zeros(membership_cb, membership_tiles * f32_tile_bytes);
        noc.write_zeros_l1_barrier();
        const uint32_t base = get_write_ptr(cb_membership);
        for (uint32_t tl = 0; tl < cols; ++tl) {
            const uint32_t T = col_begin + cg * cols + tl;
            for (uint32_t c = 0; c < 32; ++c) {
                const uint32_t ch = T * 32 + c;
                if (ch >= C) {
                    break;  // padded lanes of a ragged last channel tile stay zero
                }
                const uint32_t g = ch / Cg;
                const uint32_t kg = g >> 5;
                const uint32_t gl = g & 31;
                const uint32_t r = transposed ? c : gl;
                const uint32_t cc = transposed ? gl : c;
                const uint32_t addr = base + (tl * Kg + kg) * f32_tile_bytes + tile_elem_offset(r, cc, 4);
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = ONE_F32_BITS;
            }
        }
        cb_push_back(cb_membership, membership_tiles);
    };

    // TILE input: chunk_rows x cols tiles of image n, block (cg, rc), row-major within the chunk.
    auto load_x_chunk_tiles = [&](uint32_t n, uint32_t cg, uint32_t rc, bool pass2) {
        const uint32_t cb_id = pass2 ? cb_x_pass2 : cb_x_pass1;
        cb_reserve_back(cb_id, chunk);
        if constexpr (resident) {
            cb_reserve_back(cb_x_pass2, chunk);  // aliased region: pass-2 credits move in lockstep
        }
        uint32_t l1 = get_write_ptr(cb_id);
        for (uint32_t i = 0; i < chunk_rows; ++i) {
            const uint32_t r = row_begin + rc * chunk_rows + i;
            for (uint32_t j = 0; j < cols; ++j) {
                const uint32_t t = col_begin + cg * cols + j;
                const uint32_t page = n * Ht * Ct + r * Ct + t;
                noc_async_read(x_acc.get_noc_addr(page), l1, x_tile_bytes);
                l1 += x_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_id, chunk);
        if constexpr (resident) {
            cb_push_back(cb_x_pass2, chunk);
        }
    };

    // RM input: 32*chunk_rows sticks, cols*32 elements wide, into cb_x_rm (cols tile pages per tile-row).
    auto load_x_chunk_sticks = [&](uint32_t n, uint32_t cg, uint32_t rc) {
        if constexpr (is_rm) {
            dataflow_kernel_lib::read_sticks_for_tilize<cb_x_rm>(
                x_acc,
                32 * chunk_rows,
                cols * 32 * x_elem_size,
                n * HW + 32 * (row_begin + rc * chunk_rows),
                (col_begin + cg * cols) * 32 * x_elem_size);
        }
    };

    auto load_x_chunk = [&](uint32_t n, uint32_t cg, uint32_t rc, bool pass2) {
        if constexpr (is_rm) {
            load_x_chunk_sticks(n, cg, rc);
        } else {
            load_x_chunk_tiles(n, cg, rc, pass2);
        }
    };

    // gamma / beta: row-0-valid tiles for the cols channel tiles of column group cg. Rows 1..31 zero.
    // The 32 values of channel tile T sit contiguously in the RM stick (64 B for bf16) but must land as
    // lanes 0..15 -> face 0 row 0 and lanes 16..31 -> face 1 row 0 (+face_bytes). A NoC read must keep the
    // source and destination residues modulo the DRAM alignment (64 B on Blackhole) equal, so the second
    // half cannot be read straight into face 1 (+32 B source vs +face_bytes destination). Read the whole
    // aligned run into face 0 rows 0..1, then move row 1 to face 1 row 0 with a few word stores and re-zero it.
    auto fill_affine_rows = [&](uint32_t cb_row, const auto& acc, uint32_t cg) {
        constexpr uint32_t half_row_bytes = 16 * g_elem_size;
        constexpr uint32_t face_bytes = 256 * g_elem_size;
        const uint32_t g_tile_bytes = get_tile_size(cb_row);
        CircularBuffer row_cb(cb_row);
        cb_reserve_back(cb_row, cols);
        noc.async_write_zeros(row_cb, cols * g_tile_bytes);
        noc.write_zeros_l1_barrier();
        const uint32_t l1_base = get_write_ptr(cb_row);
        uint32_t l1 = l1_base;
        for (uint32_t tl = 0; tl < cols; ++tl) {
            const uint32_t T = col_begin + cg * cols + tl;
            if constexpr (g_is_tile && g_is_bfp) {
                // bfloat8_b TILE affine: a block format has no addressable row 0 (shared-exponent header +
                // packed mantissas), so fetch the whole tile page; rows 1..31 are the tensor's own zero padding.
                noc_async_read(acc.get_noc_addr(T, 0), l1, g_tile_bytes);
            } else if constexpr (g_is_tile) {
                // TILE affine: row 0 of faces 0 and 1 of page T are already at the right offsets.
                noc_async_read(acc.get_noc_addr(T, 0), l1, half_row_bytes);
                noc_async_read(acc.get_noc_addr(T, face_bytes), l1 + face_bytes, half_row_bytes);
            } else {
                // RM affine: one aligned read of all 32 values into face 0 rows 0..1 (fixed up below).
                noc_async_read(acc.get_noc_addr(0, T * 32 * g_elem_size), l1, 2 * half_row_bytes);
            }
            l1 += g_tile_bytes;
        }
        noc_async_read_barrier();
        if constexpr (!g_is_tile) {
            l1 = l1_base;
            for (uint32_t tl = 0; tl < cols; ++tl) {
                volatile tt_l1_ptr uint32_t* row1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + half_row_bytes);
                volatile tt_l1_ptr uint32_t* face1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + face_bytes);
                for (uint32_t w = 0; w < half_row_bytes / 4; ++w) {
                    face1[w] = row1[w];
                    row1[w] = 0u;
                }
                l1 += g_tile_bytes;
            }
        }
        // c_non_aligned: the last channel tile's lanes >= C carry whatever followed the stick in DRAM (RM
        // affine reads a full 32-lane run). They must be ZERO, not merely masked: pass 2 writes
        // y = 0 * a_T + b_T = beta into those output lanes, and a bfloat8_b output tile shares one exponent per
        // 16-lane block, so garbage there destroys the valid neighbouring channels (seen as PCC 0.96 on
        // (1,1,64,50) bf8b). TILE affine pages are zero-padded by the tensor itself; bf8b pages are read whole.
        if constexpr (!g_is_bfp) {
            constexpr uint32_t c_tail = C % 32;
            if (c_tail != 0) {
                const uint32_t T_last = Ct - 1;
                const uint32_t T_first = col_begin + cg * cols;
                if (T_last >= T_first && T_last < T_first + cols) {
                    const uint32_t tile_l1 = l1_base + (T_last - T_first) * g_tile_bytes;
                    for (uint32_t lane = c_tail; lane < 32; ++lane) {
                        const uint32_t off = (lane < 16) ? lane * g_elem_size : face_bytes + (lane - 16) * g_elem_size;
                        volatile tt_l1_ptr uint8_t* p = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(tile_l1 + off);
                        for (uint32_t b = 0; b < g_elem_size; ++b) {
                            p[b] = 0u;
                        }
                    }
                }
            }
        }
        cb_push_back(cb_row, cols);
    };

    for (uint32_t img = 0; img < image_count; ++img) {
        const uint32_t n = image_begin + img * image_stride;

        // ---------------- pass 1: statistics ----------------
        for (uint32_t cg = 0; cg < num_col_groups; ++cg) {
            for (uint32_t rc = 0; rc < num_row_chunks; ++rc) {
                load_x_chunk(n, cg, rc, /*pass2=*/false);
            }
            fill_membership(cg, /*transposed=*/true);
        }

        // ---------------- pass 2: apply ----------------
        for (uint32_t cg = 0; cg < num_col_groups; ++cg) {
            fill_membership(cg, /*transposed=*/false);
            if constexpr (has_gamma) {
                const auto gamma_acc = TensorAccessor(gamma_args, gamma_addr, g_page_bytes);
                fill_affine_rows(cb_gamma_row, gamma_acc, cg);
            }
            if constexpr (has_beta) {
                const auto beta_acc = TensorAccessor(beta_args, beta_addr, g_page_bytes);
                fill_affine_rows(cb_beta_row, beta_acc, cg);
            }
            if constexpr (!resident) {
                for (uint32_t rc = 0; rc < num_row_chunks; ++rc) {
                    load_x_chunk(n, cg, rc, /*pass2=*/true);
                }
            }
        }
    }
}
