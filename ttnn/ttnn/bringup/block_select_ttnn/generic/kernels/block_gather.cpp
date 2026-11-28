// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.block_gather (block_select.py): one data-movement kernel on both RISCs of every core. Per row r:
//   out[r, j B + o] = scores[r, ids[r, j] B + o]   for j < K, o < B   (ids[r, j] >= NBW, e.g. 0xFFFFFFFF: -inf)
// scores [rows, W] bf16 ROW_MAJOR (a page a row), ids [rows, K] uint32 ROW_MAJOR, out [rows, K B] bf16 ROW_MAJOR.
// Work unit = (row, part): KP = K / PARTS blocks of one row; units split evenly over the 2 NCORES workers (worker
// 2 c + HALF) at run time from common args (no per-core args). A block is B (a power of two <= 32) adjacent bf16
// values: 2 B bytes inside one 64-byte aligned window of the row (Blackhole's DRAM read alignment), read whole: into
// the output row when 2 B == 64, else into its own 64-byte L1 slot and copied.
// compile-time args: cb, B, K, NBW, GX, PARTS, HALF, NCORES, scores / ids / out accessors; common args: scores, ids,
// out addresses, rows.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t NBW = get_compile_time_arg_val(3);
    constexpr uint32_t GX = get_compile_time_arg_val(4);
    constexpr uint32_t PARTS = get_compile_time_arg_val(5);
    constexpr uint32_t HALF = get_compile_time_arg_val(6);
    constexpr uint32_t NCORES = get_compile_time_arg_val(7);
    constexpr auto s_args = TensorAccessorArgs<8>();
    constexpr auto i_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr uint32_t BLK_BYTES = 2 * B;
    constexpr uint32_t KP = K / PARTS;  // blocks a unit
    constexpr uint32_t IDS_BYTES = 4 * KP;
    constexpr uint32_t OUT_BYTES = BLK_BYTES * KP;
    constexpr bool DIRECT = BLK_BYTES == 64;  // (a block is its own aligned window: read into the output row)
    static_assert(B >= 1 && B <= 32 && (B & (B - 1)) == 0, "B: a power of two <= 32");
    static_assert(K % PARTS == 0 && (PARTS == 1 || IDS_BYTES % 64 == 0), "parts at 64-byte aligned id offsets");
    const auto scores = TensorAccessor(s_args, get_common_arg_val<uint32_t>(0));
    const auto ids = TensorAccessor(i_args, get_common_arg_val<uint32_t>(1));
    const auto out = TensorAccessor(o_args, get_common_arg_val<uint32_t>(2));
    const uint32_t rows = get_common_arg_val<uint32_t>(3);
    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t units = rows * PARTS;
    const uint32_t nw = 2 * NCORES, w = 2 * c + HALF;
    const uint32_t per = units / nw, extra = units % nw;
    const uint32_t count = per + (w < extra ? 1u : 0u);
    const uint32_t first = w * per + (w < extra ? w : extra);
    if (count == 0) {
        return;
    }

    // this RISC's scratch: [ids | out | slots], 64-byte aligned
    constexpr uint32_t SPAN = ((IDS_BYTES + 63u) & ~63u) + ((OUT_BYTES + 63u) & ~63u) + (DIRECT ? 0u : 64u * KP);
    cb_reserve_back(cb, 1);
    const uint32_t ids_l1 = ((get_write_ptr(cb) + 63u) & ~63u) + HALF * SPAN;
    const uint32_t out_l1 = ids_l1 + ((IDS_BYTES + 63u) & ~63u);
    const uint32_t slots = out_l1 + ((OUT_BYTES + 63u) & ~63u);
    const uint32_t* idv = reinterpret_cast<const uint32_t*>(ids_l1);
    uint16_t* ov = reinterpret_cast<uint16_t*>(out_l1);

    for (uint32_t u = first; u < first + count; ++u) {
        const uint32_t r = u / PARTS, part = u % PARTS;
        noc_async_read(ids.get_noc_addr(r, part * IDS_BYTES), ids_l1, IDS_BYTES);
        const uint64_t row = scores.get_noc_addr(r);  // (a page is one bank's contiguous run: offsets add)
        noc_async_read_barrier();
        for (uint32_t j = 0; j < KP; ++j) {
            const uint32_t id = idv[j];
            if (id < NBW) {
                noc_async_read(row + ((id * BLK_BYTES) & ~63u), DIRECT ? out_l1 + j * 64u : slots + j * 64u, 64);
            }
        }
        noc_async_read_barrier();
        for (uint32_t j = 0; j < KP; ++j) {
            const uint32_t id = idv[j];
            if (id < NBW) {
                if constexpr (!DIRECT) {
                    const uint32_t src = slots + j * 64u + ((id * BLK_BYTES) & 63u);
                    if constexpr (B >= 2) {  // (4-byte aligned: whole words)
                        const uint32_t* sw = reinterpret_cast<const uint32_t*>(src);
                        uint32_t* dw = reinterpret_cast<uint32_t*>(out_l1 + j * BLK_BYTES);
                        for (uint32_t o = 0; o < B / 2; ++o) {
                            dw[o] = sw[o];
                        }
                    } else {
                        ov[j] = *reinterpret_cast<const uint16_t*>(src);
                    }
                }
            } else {
                for (uint32_t o = 0; o < B; ++o) {
                    ov[j * B + o] = 0xFF80u;  // -inf
                }
            }
        }
        noc_async_write(out_l1, out.get_noc_addr(r, part * OUT_BYTES), OUT_BYTES);
        noc_async_writes_flushed();  // (the next unit rewrites out_l1)
    }
    noc_async_write_barrier();
}
