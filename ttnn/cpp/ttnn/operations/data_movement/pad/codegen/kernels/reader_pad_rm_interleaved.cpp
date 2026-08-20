// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pad reader: RM interleaved, front+back padding on all dims.
// NCRISC. For each output stick in [N_out, C_out, H_out, W_out]:
//   - If the stick is a data stick (within input bounds after front offsets),
//     read from DRAM with optional front/back W-pad from pad constant buffer.
//   - If the stick is a pad-only stick (H/C/N padding), fill entirely
//     from the pad constant buffer (L1 self-read, no DRAM traffic).
//
// Front-padding on N/C/H: data sticks start at front_h/front_c/front_n
// in the output coordinate space.
// Front-padding on W: each data stick has [front_w_pad | data | back_w_pad].
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

template <uint32_t num_bytes>
inline __attribute__((always_inline)) void fill_with_val(
    uint32_t dst, uint32_t val) {
    static_assert(num_bytes % sizeof(uint16_t) == 0, "RM pad values are 2B or 4B scalars");
    CoreLocalMem<volatile uint32_t> ptr(dst);
    for (uint32_t i = 0; i < num_bytes / sizeof(uint32_t); ++i) {
        ptr[i] = val;
    }
    if constexpr (num_bytes % sizeof(uint32_t) != 0) {
        CoreLocalMem<volatile uint16_t> tail(
            dst + (num_bytes / sizeof(uint32_t)) * sizeof(uint32_t));
        *tail = static_cast<uint16_t>(val);
    }
}

void kernel_main() {
    // Runtime args
    uint32_t src_addr            = get_arg_val<uint32_t>(0);
    uint32_t num_sticks          = get_arg_val<uint32_t>(1);
    uint32_t start_src_stick     = get_arg_val<uint32_t>(2);
    uint32_t start_h             = get_arg_val<uint32_t>(3);
    uint32_t start_c             = get_arg_val<uint32_t>(4);
    uint32_t start_n             = get_arg_val<uint32_t>(5);

    // Compile-time args
    constexpr uint32_t H              = get_compile_time_arg_val(0);  // input H
    constexpr uint32_t C              = get_compile_time_arg_val(1);  // input C
    constexpr uint32_t N              = get_compile_time_arg_val(2);  // input N
    constexpr uint32_t H_padded       = get_compile_time_arg_val(3);
    constexpr uint32_t C_padded       = get_compile_time_arg_val(4);
    constexpr uint32_t N_padded       = get_compile_time_arg_val(5);
    constexpr uint32_t stick_size     = get_compile_time_arg_val(6);  // input W * elem_size
    constexpr uint32_t stick_size_out = get_compile_time_arg_val(7);  // W_padded * elem_size
    constexpr uint32_t stick_size_out_aligned = get_compile_time_arg_val(8);
    constexpr uint32_t back_pad_w_bytes  = get_compile_time_arg_val(9);  // back W pad bytes
    constexpr uint32_t packed_pad_val = get_compile_time_arg_val(10);
    constexpr uint32_t BATCH          = get_compile_time_arg_val(11);
    constexpr uint32_t cb_out         = get_compile_time_arg_val(12);
    constexpr uint32_t cb_pad         = get_compile_time_arg_val(13);
    constexpr uint32_t front_pad_w_bytes = get_compile_time_arg_val(14); // front W pad bytes
    constexpr uint32_t front_h        = get_compile_time_arg_val(15);
    constexpr uint32_t front_c        = get_compile_time_arg_val(16);
    constexpr uint32_t front_n        = get_compile_time_arg_val(17);
    constexpr uint32_t cb_stage       = get_compile_time_arg_val(18);
    // Accessor args start at index 19 (byte-identical ABI to the original); the
    // fast path reads exactly the same args at the same indices as before.
    constexpr auto src_args = TensorAccessorArgs<19>();
    // in_read_size and dram_alignment live AFTER the accessor args (repeat/expand
    // convention), so they never shift anything the fast path relies on.
    constexpr uint32_t in_read_size =
        get_compile_time_arg_val(src_args.next_compile_time_args_offset());  // DRAM-aligned input page bytes
    // HW DRAM alignment (64 on Blackhole, 32 on Wormhole). This is the NOC-read
    // *size* granularity that empirically governs pad correctness (see below).
    constexpr uint32_t dram_alignment =
        get_compile_time_arg_val(src_args.next_compile_time_args_offset() + 1);

    // FAST-PATH SAFETY PREDICATE.
    //
    // The fast path issues pure-NOC reads and NO RISC memmove: it is the only path
    // that hits pad's perf target (~0.7-0.9x vs ttnn on the common tile-aligned
    // pad). It is SAFE only when every NOC read it issues has a *size* that is a
    // multiple of the HW DRAM alignment (dram_alignment: 64B on Blackhole, 32B on
    // Wormhole). This is the empirically-measured NOC-read granularity on silicon:
    //   - Aligned W=512 + 64B back-pad (both 64B multiples): pcc = 1.0 (correct).
    //   - Aligned W=256 + 32B back-pad (32B is 16B-aligned but NOT 64B): pcc~0.87
    //     -> a sub-dram_alignment back-pad NOC read corrupts.
    //   - int32 W=100 -> 400B stick (16B-aligned, NOT 64B): pcc~0.01 -> a
    //     sub-dram_alignment data read corrupts.
    // 16B (L1) alignment is therefore INSUFFICIENT; the true granularity is
    // dram_alignment for BOTH the DRAM data read and the L1<->L1 pad reads.
    //
    // On the fast path the three NOC reads it can issue are:
    //   - data read: moves stick_size            -> require stick_size % A == 0
    //   - back-pad read: moves back_pad_w_bytes   -> require back_pad_w_bytes % A == 0
    //   - pad-only full-stick fill: moves stick_size_out = stick_size + back_pad_w_bytes
    //     (front_pad_w_bytes == 0 on the fast path) -> automatically % A == 0 when
    //     both parts are.
    // With A = dram_alignment, requiring the data read and back-pad read sizes to be
    // A-multiples makes EVERY fast-path NOC transfer (size AND destination address,
    // since l1_addr + stick_size is then A-aligned too) an A-multiple by construction
    // -- provably safe on any arch, because A is that arch's native NOC granularity.
    //
    // Anything else routes through the staging path (aligned DRAM read of
    // in_read_size + full-stick pad pre-fill + RISC memmove of exactly stick_size),
    // which is byte-exact for any width/offset:
    //   - front W-pad (data would land at a non-A-aligned offset),
    //   - non-A-aligned input stick (data read size not an A-multiple),
    //   - any back W-pad whose size is not an A-multiple (e.g. 20 cols bf16 -> 40B).
    constexpr bool NEEDS_STAGE = (front_pad_w_bytes > 0)
                                 || (stick_size % dram_alignment != 0)
                                 || (back_pad_w_bytes % dram_alignment != 0);

    // BACK-PAD RISC-FILL sub-path (strict SUBSET of NEEDS_STAGE): the ONLY reason
    // this class stages is a sub-A back-pad. The input data read is itself
    // A-aligned (stick_size % A == 0) and there is no front pad, so we can take the
    // fast aligned DRAM read straight into l1_addr and fill JUST the back-pad tail
    // with a RISC store — RISC stores have no NOC-granularity constraint, so the
    // sub-A back-pad that corrupts a NOC self-read (pcc 0.87, see above) is safe
    // here. This replaces the full-stick prefill + aligned staging read + full-
    // stick RISC memmove with (1 aligned data read + a ~62-124B RISC fill), i.e.
    // fast-path cost. Being a subset of NEEDS_STAGE, it steals ONLY these cells;
    // the true-staging (front pad / unaligned stick) and pure-fast (A-aligned or
    // no back-pad) cells are byte-for-byte unchanged.
    constexpr bool BACK_PAD_RISC_FILL = (front_pad_w_bytes == 0)
                                        && (stick_size % dram_alignment == 0)
                                        && (back_pad_w_bytes > 0)
                                        && (back_pad_w_bytes % dram_alignment != 0);

    // The argument above reasons entirely about NOC read SIZES. That is not the
    // binding constraint. A DRAM->L1 read also requires the DESTINATION address
    // to be congruent to the source modulo dram_alignment, and the destination
    // here is l1_addr = cb_out_base + k*stick_size_out_aligned (stride at :239).
    // The fast path satisfies that only as a by-product of its own predicate
    // (stick % A == 0 AND back % A == 0 => stick_size_out % A == 0); this
    // sub-path drops the second term, so it does NOT inherit the guarantee.
    // Between 2026-07-25 and 2026-08-06 the host rounded the pitch to 16 and
    // three of every four sticks were undefined behaviour, silently.
    // Host fix: ops/pad/spec.py, _pitch_align = max(_L1_ALIGN, dram_alignment).
    static_assert((NEEDS_STAGE && !BACK_PAD_RISC_FILL)
                      || (stick_size_out_aligned % dram_alignment == 0),
                  "cb_out page pitch must be a dram_alignment multiple whenever a "
                  "data stick is DRAM-read directly into the CB slot: a DRAM->L1 "
                  "NOC read requires l1_addr % A == dram_addr % A.");

    // No explicit page-size override: the 2-arg TensorAccessor derives the
    // tensor's real bank-page pitch from its spec (align(stick, buffer align);
    // DRAM 64 on BH / 32 on WH, L1 16). Passing a hand-computed pitch mis-
    // addresses every page >= 1 whenever it disagrees with that real pitch
    // (row 0 exact, rest garbage/NaN) — e.g. a non-64-aligned bf16 width. Same
    // robust pattern as the repeat / fold RM readers. Reads move only
    // stick_size logical bytes via get_noc_addr.
    const auto s = TensorAccessor(src_args, src_addr);

    Noc noc;
    CircularBuffer out_cb(cb_out);
    CircularBuffer pad_cb(cb_pad);
    CircularBuffer stage_cb(cb_stage);

    // Fill the pad constant buffer once with packed_pad_val.
    // We use this as a source for L1 self-reads to fill pad regions.
    // Size it to cover the largest possible fill: full output stick.
    fill_with_val<stick_size_out>(pad_cb.get_write_ptr(), packed_pad_val);
    const uint32_t pad_l1_addr = pad_cb.get_read_ptr();
    UnicastEndpoint self_ep;
    const uint32_t my_noc_x = my_x[noc.get_noc_id()];
    const uint32_t my_noc_y = my_y[noc.get_noc_id()];

    uint32_t curr_h = start_h;
    uint32_t curr_c = start_c;
    uint32_t curr_n = start_n;
    uint32_t src_stick = start_src_stick;
    uint32_t sticks_left = num_sticks;

    while (sticks_left > 0) {
        uint32_t batch = (sticks_left < BATCH) ? sticks_left : BATCH;
        out_cb.reserve_back(batch);
        uint32_t l1_addr = out_cb.get_write_ptr();

        for (uint32_t t = 0; t < batch; t++) {
            bool is_data = (curr_h >= front_h) && (curr_h < front_h + H) &&
                           (curr_c >= front_c) && (curr_c < front_c + C) &&
                           (curr_n >= front_n) && (curr_n < front_n + N);

            if (is_data) {
                if constexpr (BACK_PAD_RISC_FILL) {
                    // Aligned fast data read straight into place, then RISC-fill
                    // the sub-A back-pad tail. The async data read writes only
                    // [l1_addr, l1_addr+stick_size); the RISC fill writes the
                    // DISJOINT [l1_addr+stick_size, +back_pad_w_bytes) — no alias.
                    // The batch barrier completes the read before push_back.
                    CoreLocalMem<uint32_t> out_mem(l1_addr);
                    noc.async_read(
                        s, out_mem, stick_size,
                        {.page_id = src_stick, .offset_bytes = 0},
                        {.offset_bytes = 0});
                    fill_with_val<back_pad_w_bytes>(
                        l1_addr + stick_size, packed_pad_val);
                } else if constexpr (NEEDS_STAGE) {
                    // Robust path for front W-pad and/or non-16B-aligned input
                    // widths. If there is any W-pad, pre-fill the whole output
                    // stick with the pad value (covers front AND back W-pad). Then
                    // DRAM-read the ALIGNED input page (in_read_size bytes) into the
                    // aligned staging CB and RISC-memmove only the stick_size real
                    // bytes into place at l1_addr + front_pad_w_bytes. Every NOC
                    // transfer here uses an aligned source address and the placement
                    // copy is byte-granular, so this is correct for any width/offset.
                    // Uses cb_stage (not cb_pad) so the pad value buffer stays
                    // intact. Mirrors the repeat/expand aligned-DRAM-read + local-
                    // copy convention. (front-pad-only case is byte-identical to the
                    // original front-pad path.)
                    if constexpr (front_pad_w_bytes > 0 || back_pad_w_bytes > 0) {
                        CoreLocalMem<uint32_t> out_mem(l1_addr);
                        noc.async_read(
                            self_ep, out_mem, stick_size_out,
                            {.noc_x = my_noc_x, .noc_y = my_noc_y,
                             .addr = pad_l1_addr},
                            {.offset_bytes = 0});
                        noc.async_read_barrier();
                    }
                    noc.async_read(
                        s, stage_cb, in_read_size,
                        {.page_id = src_stick, .offset_bytes = 0},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                    memmove(
                        reinterpret_cast<void*>(l1_addr + front_pad_w_bytes),
                        reinterpret_cast<void*>(stage_cb.get_read_ptr()),
                        static_cast<size_t>(stick_size));
                } else {
                    // Fast path: 16B-aligned input width, back-only (or no) W-pad.
                    // Data starts at l1_addr (aligned); the back-W-pad L1 read
                    // lands at l1_addr + stick_size (16B-aligned) and the DRAM read
                    // moves a 16B-aligned size. Byte-identical to the original.
                    CoreLocalMem<uint32_t> out_mem(l1_addr);
                    noc.async_read(
                        s, out_mem, stick_size,
                        {.page_id = src_stick, .offset_bytes = 0},
                        {.offset_bytes = 0});
                    if constexpr (back_pad_w_bytes > 0) {
                        noc.async_read(
                            self_ep, out_mem, back_pad_w_bytes,
                            {.noc_x = my_noc_x, .noc_y = my_noc_y,
                             .addr = pad_l1_addr},
                            {.offset_bytes = stick_size});
                    }
                }
                src_stick++;
            } else {
                // Full pad stick: fill from pad constant buffer
                CoreLocalMem<uint32_t> out_mem(l1_addr);
                noc.async_read(
                    self_ep, out_mem, stick_size_out,
                    {.noc_x = my_noc_x, .noc_y = my_noc_y,
                     .addr = pad_l1_addr},
                    {.offset_bytes = 0});
            }

            l1_addr += stick_size_out_aligned;

            // Advance output coordinates (innermost H, then C, then N)
            curr_h++;
            if (curr_h == H_padded) {
                curr_h = 0;
                curr_c++;
                if (curr_c == C_padded) {
                    curr_c = 0;
                    curr_n++;
                }
            }
        }
        noc.async_read_barrier();  // ONE barrier for entire batch
        out_cb.push_back(batch);
        sticks_left -= batch;
    }
}
