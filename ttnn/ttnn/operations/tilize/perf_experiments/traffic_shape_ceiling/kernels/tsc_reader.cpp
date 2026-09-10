// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// traffic_shape_ceiling READER (NCRISC / NoC0) — bare traffic, no tilize.
//
// This is a RECONSTRUCTION of the tilize reader's NoC traffic on the focus
// shape `[1,1,32,16384]` with everything that is not a NoC transfer deleted:
// no compute, no accessor branch selection, no padding/retile/native legs.
// The transfer stream it issues is transfer-for-transfer identical to
// `tilize_kernel::read_sticks_rotated`'s (tilize_stick_read.hpp:72-113) on that
// shape:
//
//   * `num_reads` transfers of `read_bytes` bytes each, ALL behind ONE
//     `noc_async_read_barrier()`;
//   * issued as TWO STRAIGHT RUNS from a core-dependent rotation `rot`
//     (`[rot, n)` then `[0, rot)`) — the same form, so the per-transfer RISC-V
//     issue cost (plain increments, no modulo, no wrap branch inside the loop)
//     is the same too;
//   * transfer k of index `idx` reads source page `page_base + idx` at byte
//     offset `byte_offset` into L1 at `l1_base + idx * read_bytes` — addressed
//     by INDEX, not by issue slot, which is why the rotation cannot move a byte.
//
// Two read shapes, selected by the host through `num_reads` / `read_bytes`
// alone (the loop is the same code):
//   32 x 512 B  from a `[1,1,32,16384]` bf16 ROW_MAJOR tensor (32 pages of
//               32768 B), `page_base = 0`, `byte_offset = 512 * block_id`
//               — EXACTLY the op's focus-shape read.
//    8 x 2048 B from a 2048-B-paged tensor, `page_base = 8 * block_id`,
//               `byte_offset = 0` — the same 16 KB per core in 4x-wider
//               transactions. This is the H1 (transaction shape) rung.
//
// DEPENDENCY SWITCH (`do_push`). This is the whole experiment:
//   do_push = 1  reserve the block in cb_in, read, barrier, push. The writer
//                waits on that push, so the read stream GATES the write stream
//                exactly as the op's pipeline does (minus the compute hop).
//   do_push = 0  no CB handshake at all: read into the same L1 the CB owns and
//                stop. The writer sources a different, unrelated L1 buffer, so
//                the two streams have no ordering relation and coexist for the
//                whole kernel. That is the traffic shape's CEILING.
// The read loop itself is byte-identical between the two — only the reserve /
// push calls compile in or out — so any measured delta is the dependency and
// nothing else.
//
// `read_enabled = 0` compiles the whole read away (the writes-only calibration
// rung); nothing else changes, so NCRISC simply has no payload.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_pages = get_compile_time_arg_val(1);  // pages reserved/pushed per block
    constexpr uint32_t read_enabled = get_compile_time_arg_val(2);
    constexpr uint32_t do_push = get_compile_time_arg_val(3);
    constexpr uint32_t num_reads = get_compile_time_arg_val(4);
    constexpr uint32_t read_bytes = get_compile_time_arg_val(5);
    constexpr auto src_args = TensorAccessorArgs<6>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t page_base = get_arg_val<uint32_t>(1);
    const uint32_t byte_offset = get_arg_val<uint32_t>(2);
    const uint32_t rot = get_arg_val<uint32_t>(3);  // core-dependent issue rotation

    const auto src_acc = TensorAccessor(src_args, src_addr);

    if constexpr (do_push) {
        cb_reserve_back(cb_in, cb_pages);
    }
    const uint32_t l1_base = get_write_ptr(cb_in);

    if constexpr (read_enabled) {
        // Two straight runs — the rotated order of tilize_stick_read.hpp.
        uint32_t page = page_base + rot;
        uint32_t l1 = l1_base + rot * read_bytes;
        for (uint32_t idx = rot; idx < num_reads; ++idx) {
            noc_async_read(src_acc.get_noc_addr(page, byte_offset), l1, read_bytes);
            ++page;
            l1 += read_bytes;
        }
        page = page_base;
        l1 = l1_base;
        for (uint32_t idx = 0; idx < rot; ++idx) {
            noc_async_read(src_acc.get_noc_addr(page, byte_offset), l1, read_bytes);
            ++page;
            l1 += read_bytes;
        }
        noc_async_read_barrier();
    }

    if constexpr (do_push) {
        cb_push_back(cb_in, cb_pages);
    }
}
