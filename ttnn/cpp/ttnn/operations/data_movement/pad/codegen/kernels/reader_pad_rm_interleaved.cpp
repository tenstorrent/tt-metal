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

template <uint32_t num_bytes>
inline __attribute__((always_inline)) void fill_with_val(uint32_t dst, uint32_t val) {
    static_assert(num_bytes % sizeof(uint16_t) == 0, "RM pad values are 2B or 4B scalars");
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t i = 0; i < num_bytes / sizeof(uint32_t); ++i) {
        ptr[i] = val;
    }
    if constexpr (num_bytes % sizeof(uint32_t) != 0) {
        *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst + (num_bytes / sizeof(uint32_t)) * sizeof(uint32_t)) =
            static_cast<uint16_t>(val);
    }
}

void kernel_main() {
    // Runtime args
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks = get_arg_val<uint32_t>(1);
    uint32_t start_src_stick = get_arg_val<uint32_t>(2);
    uint32_t start_h = get_arg_val<uint32_t>(3);
    uint32_t start_c = get_arg_val<uint32_t>(4);
    uint32_t start_n = get_arg_val<uint32_t>(5);

    // Compile-time args
    constexpr uint32_t H = get_compile_time_arg_val(0);  // input H
    constexpr uint32_t C = get_compile_time_arg_val(1);  // input C
    constexpr uint32_t N = get_compile_time_arg_val(2);  // input N
    constexpr uint32_t H_padded = get_compile_time_arg_val(3);
    constexpr uint32_t C_padded = get_compile_time_arg_val(4);
    constexpr uint32_t N_padded = get_compile_time_arg_val(5);
    constexpr uint32_t stick_size = get_compile_time_arg_val(6);      // input W * elem_size
    constexpr uint32_t stick_size_out = get_compile_time_arg_val(7);  // W_padded * elem_size
    constexpr uint32_t stick_size_out_aligned = get_compile_time_arg_val(8);
    constexpr uint32_t back_pad_w_bytes = get_compile_time_arg_val(9);  // back W pad bytes
    constexpr uint32_t packed_pad_val = get_compile_time_arg_val(10);
    constexpr uint32_t BATCH = get_compile_time_arg_val(11);
    constexpr uint32_t cb_out = get_compile_time_arg_val(12);
    constexpr uint32_t cb_pad = get_compile_time_arg_val(13);
    constexpr uint32_t front_pad_w_bytes = get_compile_time_arg_val(14);  // front W pad bytes
    constexpr uint32_t front_h = get_compile_time_arg_val(15);
    constexpr uint32_t front_c = get_compile_time_arg_val(16);
    constexpr uint32_t front_n = get_compile_time_arg_val(17);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(18);
    // Accessor args start at index 19 (byte-identical ABI to the original); the
    // fast path reads exactly the same args at the same indices as before.
    constexpr auto src_args = TensorAccessorArgs<19>();
    // in_read_size and dram_alignment live AFTER the accessor args (repeat/expand
    // convention), so they never shift anything the fast path relies on.
    constexpr uint32_t in_read_size =
        get_compile_time_arg_val(src_args.next_compile_time_args_offset());  // DRAM-aligned input page bytes
    // HW DRAM alignment (64 on Blackhole, 32 on Wormhole). This is the NOC-read
    // *size* granularity that empirically governs pad correctness (see below).
    constexpr uint32_t dram_alignment = get_compile_time_arg_val(src_args.next_compile_time_args_offset() + 1);

    // NOC TRANSFER GRANULARITY.
    //
    // Every NOC read issued below must move a *size* that is a multiple of the HW DRAM
    // alignment (dram_alignment: 64B on Blackhole, 32B on Wormhole). This is the
    // empirically-measured granularity on silicon, and 16B L1 alignment is NOT enough:
    //   - Aligned W=512 + 64B back-pad (both 64B multiples): pcc = 1.0 (correct).
    //   - Aligned W=256 + 32B back-pad (32B is 16B-aligned but NOT 64B): pcc~0.87
    //     -> a sub-dram_alignment back-pad NOC read corrupts.
    //   - int32 W=100 -> 400B stick (16B-aligned, NOT 64B): pcc~0.01 -> a
    //     sub-dram_alignment data read corrupts.
    // It binds the L1<->L1 pad reads as well as the DRAM data read, and it binds destination
    // addresses as well as sizes -- hence the dram_alignment page pitch the factory hands us,
    // which is what keeps every l1_addr below on a granule boundary.
    //
    // Only the DATA placement can force staging. A front W-pad lands data at a
    // non-A-aligned offset, and a non-A-aligned input stick makes the DRAM read size
    // itself illegal; neither is expressible as NOC reads, so both fall back to an
    // aligned read into scratch plus a RISC memmove. A ragged *pad* region does not
    // force it: pad bytes have no source to honour, so the A-multiple part rides the
    // NOC and the sub-A remainder is written by RISC, which has no granularity rule.
    // That distinction is worth 1.25-1.57x of native on Blackhole where staging gives 0.51-0.76x.
    constexpr bool NEEDS_STAGE = (front_pad_w_bytes > 0) || (stick_size % dram_alignment != 0);

    // Split points for the two pad fills: NOC the whole A-multiples, RISC the remainder. Each
    // remainder sits at an A-multiple offset from an A-aligned page, so the stores are word-aligned
    // and land clear of the granule any concurrent NOC read on this page can settle into.
    constexpr uint32_t back_pad_noc_bytes = (back_pad_w_bytes / dram_alignment) * dram_alignment;
    constexpr uint32_t back_pad_tail_bytes = back_pad_w_bytes - back_pad_noc_bytes;
    constexpr uint32_t pad_stick_noc_bytes = (stick_size_out / dram_alignment) * dram_alignment;
    constexpr uint32_t pad_stick_tail_bytes = stick_size_out - pad_stick_noc_bytes;

    // No explicit page-size override: the 2-arg TensorAccessor derives the
    // tensor's real bank-page pitch from its spec (align(stick, buffer align);
    // DRAM 64 on BH / 32 on WH, L1 16). Passing a hand-computed pitch mis-
    // addresses every page >= 1 whenever it disagrees with that real pitch
    // (row 0 exact, rest garbage/NaN) — e.g. a non-64-aligned bf16 width. Same
    // robust pattern as the repeat / fold RM readers. Reads move only
    // stick_size logical bytes via get_noc_addr.
    const auto s = TensorAccessor(src_args, src_addr);

    // Fill the pad constant buffer once with packed_pad_val.
    // We use this as a source for L1 self-reads to fill pad regions.
    // Size it to cover the largest possible fill: full output stick.
    fill_with_val<stick_size_out>(get_write_ptr(cb_pad), packed_pad_val);
    uint64_t pad_noc_addr = get_noc_addr(get_read_ptr(cb_pad));

    uint32_t curr_h = start_h;
    uint32_t curr_c = start_c;
    uint32_t curr_n = start_n;
    uint32_t src_stick = start_src_stick;
    uint32_t sticks_left = num_sticks;

    while (sticks_left > 0) {
        uint32_t batch = (sticks_left < BATCH) ? sticks_left : BATCH;
        cb_reserve_back(cb_out, batch);
        uint32_t l1_addr = get_write_ptr(cb_out);

        for (uint32_t t = 0; t < batch; t++) {
            bool is_data = (curr_h >= front_h) && (curr_h < front_h + H) && (curr_c >= front_c) &&
                           (curr_c < front_c + C) && (curr_n >= front_n) && (curr_n < front_n + N);

            if (is_data) {
                if constexpr (NEEDS_STAGE) {
                    // Data placement the NOC cannot express: pre-fill the whole output stick with
                    // the pad value, read the A-aligned input page into scratch, then RISC-memmove
                    // the stick_size real bytes into place. Byte-granular, so correct at any width
                    // or offset. Scratch is cb_stage, not cb_pad, so the pad constant survives.
                    //
                    // The pre-fill and the DRAM read address disjoint buffers, so they share one
                    // barrier; only the memmove depends on both.
                    if constexpr (front_pad_w_bytes > 0 || back_pad_w_bytes > 0) {
                        if constexpr (pad_stick_noc_bytes > 0) {
                            noc_async_read(pad_noc_addr, l1_addr, pad_stick_noc_bytes);
                        }
                        if constexpr (pad_stick_tail_bytes > 0) {
                            fill_with_val<pad_stick_tail_bytes>(l1_addr + pad_stick_noc_bytes, packed_pad_val);
                        }
                    }
                    uint32_t stage_addr = get_write_ptr(cb_stage);
                    noc_async_read(s.get_noc_addr(src_stick), stage_addr, in_read_size);
                    noc_async_read_barrier();
                    memmove(
                        reinterpret_cast<void*>(l1_addr + front_pad_w_bytes),
                        reinterpret_cast<void*>(get_read_ptr(cb_stage)),
                        static_cast<size_t>(stick_size));
                } else {
                    noc_async_read(s.get_noc_addr(src_stick), l1_addr, stick_size);
                    if constexpr (back_pad_noc_bytes > 0) {
                        noc_async_read(pad_noc_addr, l1_addr + stick_size, back_pad_noc_bytes);
                    }
                    if constexpr (back_pad_tail_bytes > 0) {
                        fill_with_val<back_pad_tail_bytes>(l1_addr + stick_size + back_pad_noc_bytes, packed_pad_val);
                    }
                }
                src_stick++;
            } else {
                // Full pad stick: fill from pad constant buffer
                if constexpr (pad_stick_noc_bytes > 0) {
                    noc_async_read(pad_noc_addr, l1_addr, pad_stick_noc_bytes);
                }
                if constexpr (pad_stick_tail_bytes > 0) {
                    fill_with_val<pad_stick_tail_bytes>(l1_addr + pad_stick_noc_bytes, packed_pad_val);
                }
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
        noc_async_read_barrier();  // ONE barrier for entire batch
        cb_push_back(cb_out, batch);
        sticks_left -= batch;
    }
}
