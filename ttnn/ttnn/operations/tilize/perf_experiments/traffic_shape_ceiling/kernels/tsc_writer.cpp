// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// traffic_shape_ceiling WRITER (BRISC / NoC1) — bare traffic, no tilize.
//
// RECONSTRUCTION of the tilize writer's `store_block` on the focus shape
// `[1,1,32,16384]` (tilize_writer.cpp:164-244) with everything that is not a
// NoC transfer deleted. On that shape `block_row_extent == 1`, so the op's
// batching loop runs exactly once and its whole write stream is:
//
//   * `num_writes` (= block_width_tiles = 8) transfers of `write_bytes`
//     (= out_tile_bytes = 2048) bytes, ALL behind ONE
//     `noc_async_write_barrier()`;
//   * to interleaved destination pages `page_base + i` where
//     `page_base = 8 * block_id` (the op's `(row_start + rows_done + r) *
//     tensor_col_tiles + col_base` collapses to `w_chunk * 8` at R == 1);
//   * from L1 at `src + i * write_bytes` — addressed by INDEX;
//   * in the op's COLUMN-ROTATED issue order (`i = cs + col_rot`, wrap by
//     subtraction), with `col_rot = block_id % 8`. The op's ROW rotation is
//     inert here (`rows_this_batch == 1`).
//   * through `noc_async_write<write_bytes>` — the ONE-PACKET issue path the
//     op deliberately takes (dataflow_api.h:838), so the RISC-V issue cost per
//     transfer matches too.
//
// DEPENDENCY SWITCH (`use_cb`) — the whole experiment:
//   use_cb = 1  cb_wait_front / get_read_ptr / cb_pop_front on cb_in. The
//               writer cannot issue a single byte until the reader has landed
//               all of its reads and pushed. This is the op's chain.
//   use_cb = 0  the write stream sources `cb_scratch`, a DIFFERENT L1 buffer
//               that nothing produces into. The same transfers, the same sizes,
//               the same destination pages, the same order — with no ordering
//               relation to the read stream at all, so reads and writes coexist
//               from kernel start. The bytes written are garbage BY DESIGN.
//
// TRAFFIC VERIFICATION for the dependency-free rungs (`verify_fill`). Since the
// payload is garbage, the destination cannot be value-checked against an input.
// Instead the kernel stamps a per-(core, transfer) MARKER into the first and the
// LAST 4 bytes of each 2048-B scratch page before issuing. The host then asserts
// that destination page `8*block_id + i` carries exactly marker (block_id, i) at
// both ends — which pins down, per transfer: the destination page id, that the
// transfer really was `write_bytes` long (the tail marker sits at
// `write_bytes - 4`), and that the L1 source stride was `write_bytes`. 16
// 32-bit stores per core, issued before the write loop; ~tens of ns against a
// ~12 us wall, and identical across every scratch-sourced rung.
//
// `write_enabled` is a host-side switch (the reads-only rungs simply do not
// instantiate this kernel), so there is no ablated branch in here.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t use_cb = get_compile_time_arg_val(2);
    constexpr uint32_t cb_pages = get_compile_time_arg_val(3);
    constexpr uint32_t num_writes = get_compile_time_arg_val(4);
    constexpr uint32_t write_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t verify_fill = get_compile_time_arg_val(6);
    constexpr auto dst_args = TensorAccessorArgs<7>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t page_base = get_arg_val<uint32_t>(1);
    const uint32_t col_rot = get_arg_val<uint32_t>(2);
    const uint32_t block_id = get_arg_val<uint32_t>(3);

    const auto dst_acc = TensorAccessor(dst_args, dst_addr);

    uint32_t src;
    if constexpr (use_cb) {
        cb_wait_front(cb_in, cb_pages);
        src = get_read_ptr(cb_in);
    } else {
        src = get_write_ptr(cb_scratch);
        if constexpr (verify_fill) {
            // Two bf16-representable u16 words per marker (see the header note):
            // head = {0x4000 | block_id, 0x4100 | i}, tail = {0x4200 | block_id,
            // 0x4300 | i}, little-endian inside the u32.
            for (uint32_t i = 0; i < num_writes; ++i) {
                const uint32_t p = src + i * write_bytes;
                *reinterpret_cast<volatile uint32_t*>(p) = ((0x4100u | i) << 16) | (0x4000u | block_id);
                *reinterpret_cast<volatile uint32_t*>(p + write_bytes - 4) =
                    ((0x4300u | i) << 16) | (0x4200u | block_id);
            }
        }
    }

    // The op's column-rotated issue order, wrap by subtraction.
    for (uint32_t cs = 0; cs < num_writes; ++cs) {
        uint32_t i = cs + col_rot;
        if (i >= num_writes) {
            i -= num_writes;
        }
        noc_async_write<write_bytes>(src + i * write_bytes, dst_acc.get_noc_addr(page_base + i), write_bytes);
    }
    noc_async_write_barrier();

    if constexpr (use_cb) {
        cb_pop_front(cb_in, cb_pages);
    }
}
