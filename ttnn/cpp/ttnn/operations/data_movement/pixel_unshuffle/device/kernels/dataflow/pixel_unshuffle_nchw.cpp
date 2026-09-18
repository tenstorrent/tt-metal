// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// pixel_unshuffle on NCHW ROW_MAJOR interleaved tensors.
//
// This one source runs on BOTH dataflow RISCs; each gets a disjoint range of input rows
// and does the whole read -> deinterleave -> write pipeline for it. The split matters
// because the cost of this op is the deinterleave, which is a RISC-V L1 copy: profiling
// a reader/writer pair showed the reader idle after ~24 DRAM reads while the writer
// carried every element. Running the copy on both RISCs doubles the throughput that
// actually counts. Each RISC owns private L1 buffers, so there is no CB handshake.
//
// Work unit is one INPUT ROW, which feeds exactly r output sticks (one per rw). Rows are
// read in linear order, so the input page index is simply the row index. Decoding row p:
//   n = p / (C*H);  c_in = (p % (C*H)) / H;  h_in = p % H;  h_out = h_in / r;  rh = h_in % r
//   CHANNEL_MAJOR: c_out(rw) = c_in*r^2 + rh*r + rw         -> output page stride over rw = Ho
//   SPATIAL_MAJOR: c_out(rw) = rh*(r*C) + rw*C + c_in       -> output page stride over rw = C*Ho
//   page(rw) = n*C_out*Ho + c_out(rw)*Ho + h_out
//
// Rows are processed in batches of `batch`: one read barrier and one write barrier per
// batch instead of per stick.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_nbytes_in = get_compile_time_arg_val(0);           // W * datum_size
    constexpr uint32_t aligned_stick_nbytes_in = get_compile_time_arg_val(1);   // aligned input stick
    constexpr uint32_t stick_nbytes_out = get_compile_time_arg_val(2);          // Wo * datum_size
    constexpr uint32_t aligned_stick_nbytes_out = get_compile_time_arg_val(3);  // aligned output stick
    constexpr uint32_t cb_id_in = get_compile_time_arg_val(4);                  // private input buffer
    constexpr uint32_t cb_id_scratch = get_compile_time_arg_val(5);             // private scratch buffer
    constexpr uint32_t r = get_compile_time_arg_val(6);                         // downscale factor
    constexpr uint32_t W = get_compile_time_arg_val(7);                         // input width
    constexpr uint32_t C = get_compile_time_arg_val(8);                         // input channels
    constexpr uint32_t H = get_compile_time_arg_val(9);                         // input height
    constexpr uint32_t Ho = get_compile_time_arg_val(10);                       // H / r
    constexpr uint32_t channel_order = get_compile_time_arg_val(11);            // 0=CH_MAJOR, 1=SP_MAJOR
    constexpr uint32_t depth = get_compile_time_arg_val(12);                    // pipeline slots
    constexpr auto src_args = TensorAccessorArgs<13, 0>();
    constexpr auto dst_args =
        TensorAccessorArgs<src_args.next_compile_time_args_offset(), src_args.num_common_runtime_args()>();

    constexpr uint32_t SPATIAL_MAJOR = 1;
    constexpr uint32_t datum_nbytes = stick_nbytes_in / W;
    constexpr uint32_t r2 = r * r;
    constexpr uint32_t Wo = W / r;
    constexpr uint32_t C_out = C * r2;
    constexpr uint32_t CH = C * H;
    // Distance in output pages between consecutive rw for the same input row.
    constexpr uint32_t rw_page_stride = (channel_order == SPATIAL_MAJOR) ? (C * Ho) : Ho;

    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_row = get_arg_val<uint32_t>(2);
    const uint32_t num_rows = get_arg_val<uint32_t>(3);
    // Rows of this RISC are start_row, start_row + row_stride, start_row + 2*row_stride...
    // Strided rather than contiguous so that concurrent reads from every RISC land on
    // different DRAM banks; see the note in the program factory.
    const uint32_t row_stride = get_arg_val<uint32_t>(4);

    if (num_rows == 0) {
        return;
    }

    const auto s_in = TensorAccessor(src_args, src_addr);
    const auto s_out = TensorAccessor(dst_args, dst_addr);
    Noc noc;
    experimental::CB cb_in(cb_id_in);
    experimental::CB cb_scratch(cb_id_scratch);

    // Both CBs are used purely as private L1 allocations: nothing is pushed or popped, so
    // the pointers stay at the buffer base and slots are addressed by explicit offset.
    const uint32_t in_base = cb_in.get_write_ptr();
    const uint32_t scratch_base = cb_scratch.get_write_ptr();

    // Software pipeline over rows, `depth` slots deep, with a NOC transaction id per slot.
    //
    // The deinterleave below - not the NOC - is the critical path: with the copy compiled
    // out, the two block-A calls together drop from ~195us to ~55us, so the data movement
    // itself is only about a quarter of the runtime. The pipeline exists to hide that
    // quarter behind the copy, and it stays shallow on purpose: `depth` 2 measured best
    // (195us) against 4 (198us), 6 (197us) and 8 (202us).
    //
    // A plain (untagged) barrier per row would leave only ONE read in flight, and batching
    // rows behind a shared barrier queues more reads but stops reads overlapping writes -
    // measured worse at every size (B=12: 271us, B=8: 260us, B=4: 250us, B=1: 236us).
    // A trid per slot gets both: `depth` reads in flight AND a barrier that waits for one
    // row only.
    uint32_t row = start_row;

    // Prefetch up to `depth` rows before entering the loop, each tagged with its slot's
    // transaction id, so `depth` reads are in flight rather than one.
    {
        const uint32_t pf = (num_rows < depth) ? num_rows : depth;
        uint32_t pr = start_row;
        for (uint32_t s = 0; s < pf; s++) {
            noc.async_read<NocOptions::TXN_ID>(
                s_in,
                cb_in,
                stick_nbytes_in,
                {.page_id = pr},
                {.offset_bytes = s * aligned_stick_nbytes_in},
                {.trid = s + 1});
            pr += row_stride;
        }
    }

    for (uint32_t i = 0; i < num_rows; i++) {
        const uint32_t slot = i % depth;
        const uint32_t rtrid = slot + 1;          // read  trids: 1 .. depth
        const uint32_t wtrid = depth + slot + 1;  // write trids: depth+1 .. 2*depth

        // Wait only for THIS row's read; the other depth-1 reads stay in flight.
        noc.async_read_barrier<NocOptions::TXN_ID>({.trid = rtrid});

        // Row i-depth shares this scratch group, so its writes must land before the copy
        // below overwrites it. Only that row's writes are waited on.
        if (i >= depth) {
            noc.async_write_barrier<NocOptions::TXN_ID>({.trid = wtrid});
        }

        // ---- decode row -> base output page ----
        const uint32_t p = row;
        const uint32_t n = p / CH;
        const uint32_t rem = p - n * CH;
        const uint32_t c_in = rem / H;
        const uint32_t h_in = rem - c_in * H;
        const uint32_t h_out = h_in / r;
        const uint32_t rh = h_in - h_out * r;

        uint32_t c_out0;  // output channel for rw = 0
        if constexpr (channel_order == SPATIAL_MAJOR) {
            c_out0 = rh * (r * C) + c_in;  // + rw*C
        } else {
            c_out0 = c_in * r2 + rh * r;  // + rw
        }
        const uint32_t page0 = n * (C_out * Ho) + c_out0 * Ho + h_out;

        const uint32_t src_l1 = in_base + slot * aligned_stick_nbytes_in;
        const uint32_t sbase = slot * r;  // scratch group for this row

        // Loads must not be hoisted above the read barrier.
        asm volatile("" ::: "memory");

        // Pointers are deliberately NOT volatile: volatile forces exactly one in-order
        // access per element, which on this in-order RISC-V serialises an L1 round trip
        // per element (~18 cycles/element measured); non-volatile lets the compiler
        // unroll and keep several loads in flight.
        if constexpr (datum_nbytes == 2 && (Wo % 2) == 0) {
            // 32-bit aligned gather/pack. A group of 2r consecutive source elements
            // occupies exactly r consecutive 32-bit words and contributes exactly two
            // elements - one adjacent pair - to each of the r output sticks. So r aligned
            // contiguous loads produce r aligned stores: 1 memory op per element, versus
            // 2 for a naive element copy. Every access is 32-bit and naturally aligned.
            //
            // For group m, w[t] = src32[r*m + t] holds elements 2r*m+2t (low half) and
            // 2r*m+2t+1 (high half). Element e belongs to stick e%r at position e/r, so
            // stick j takes q = j (position 2m) and q = r+j (position 2m+1):
            //   low  half of d32[j][m] <- half (j & 1)     of w[j >> 1]
            //   high half of d32[j][m] <- half ((r+j) & 1) of w[(r+j) >> 1]
            tt_l1_ptr uint32_t* src32 = (tt_l1_ptr uint32_t*)src_l1;
            tt_l1_ptr uint32_t* d32[r];
            for (uint32_t j = 0; j < r; j++) {
                d32[j] = (tt_l1_ptr uint32_t*)(scratch_base + (sbase + j) * aligned_stick_nbytes_out);
            }
            // Groups handled per iteration. Every group's r loads are issued before any
            // store, so this many loads are in flight at once; L1 accesses on this core do
            // pipeline, and keeping ~12 outstanding is what makes the copy fast. Dividing
            // by r keeps that number constant as r changes. Measured combined runtime over
            // the two block-A calls: 4 loads 236us, 8 -> 209us, 12 -> 198us, 16 -> 200us,
            // 24 -> 199us; 12 is the optimum and the curve is flat either side of it.
            constexpr uint32_t loads_in_flight = 12;
            constexpr uint32_t gpi = (loads_in_flight / r) ? (loads_in_flight / r) : 1u;
            uint32_t wbase = 0;
            if constexpr ((Wo % (2 * gpi)) == 0) {
                for (uint32_t mm = 0; mm < Wo / (2 * gpi); mm++) {
                    uint32_t w[gpi][r];
#pragma GCC unroll 16
                    for (uint32_t g = 0; g < gpi; g++) {
#pragma GCC unroll 16
                        for (uint32_t t = 0; t < r; t++) {
                            w[g][t] = src32[wbase + g * r + t];
                        }
                    }
#pragma GCC unroll 16
                    for (uint32_t j = 0; j < r; j++) {
#pragma GCC unroll 16
                        for (uint32_t g = 0; g < gpi; g++) {
                            const uint32_t wl = w[g][j >> 1];
                            const uint32_t lo = (j & 1u) ? (wl >> 16) : (wl & 0xffffu);
                            const uint32_t wh = w[g][(r + j) >> 1];
                            const uint32_t hi = ((r + j) & 1u) ? (wh & 0xffff0000u) : (wh << 16);
                            d32[j][gpi * mm + g] = lo | hi;
                        }
                    }
                    wbase += gpi * r;
                }
            } else {
                for (uint32_t m = 0; m < Wo / 2; m++) {
                    uint32_t w[r];
#pragma GCC unroll 16
                    for (uint32_t t = 0; t < r; t++) {
                        w[t] = src32[wbase + t];
                    }
#pragma GCC unroll 16
                    for (uint32_t j = 0; j < r; j++) {
                        const uint32_t wl = w[j >> 1];
                        const uint32_t lo = (j & 1u) ? (wl >> 16) : (wl & 0xffffu);
                        const uint32_t wh = w[(r + j) >> 1];
                        const uint32_t hi = ((r + j) & 1u) ? (wh & 0xffff0000u) : (wh << 16);
                        d32[j][m] = lo | hi;
                    }
                    wbase += r;
                }
            }
        } else if constexpr (datum_nbytes == 2) {
            // Odd Wo: positions do not pair up, fall back to element copies.
            tt_l1_ptr uint16_t* src16 = (tt_l1_ptr uint16_t*)src_l1;
            for (uint32_t j = 0; j < r; j++) {
                tt_l1_ptr uint16_t* sj = src16 + j;
                tt_l1_ptr uint16_t* d16 = (tt_l1_ptr uint16_t*)(scratch_base + (sbase + j) * aligned_stick_nbytes_out);
                uint32_t si = 0;
                for (uint32_t k = 0; k < Wo; k++) {
                    d16[k] = sj[si];
                    si += r;
                }
            }
        } else {
            // 4-byte elements (float32, uint32, int32): each element is already a full
            // 32-bit access, so there is nothing to pack; unroll the strided reads.
            tt_l1_ptr uint32_t* src32 = (tt_l1_ptr uint32_t*)src_l1;
            for (uint32_t j = 0; j < r; j++) {
                tt_l1_ptr uint32_t* sj = src32 + j;
                tt_l1_ptr uint32_t* d32 = (tt_l1_ptr uint32_t*)(scratch_base + (sbase + j) * aligned_stick_nbytes_out);
                uint32_t si = 0;
                if constexpr ((Wo % 4) == 0) {
                    for (uint32_t k = 0; k < Wo; k += 4) {
                        const uint32_t a = sj[si];
                        const uint32_t b = sj[si + r];
                        const uint32_t c = sj[si + 2 * r];
                        const uint32_t e = sj[si + 3 * r];
                        d32[k] = a;
                        d32[k + 1] = b;
                        d32[k + 2] = c;
                        d32[k + 3] = e;
                        si += 4 * r;
                    }
                } else {
                    for (uint32_t k = 0; k < Wo; k++) {
                        d32[k] = sj[si];
                        si += r;
                    }
                }
            }
        }

        // The copy has drained this input slot, so it can be refilled now; this read
        // overlaps the writes issued just below and the next rows' copies.
        if (i + depth < num_rows) {
            noc.async_read<NocOptions::TXN_ID>(
                s_in,
                cb_in,
                stick_nbytes_in,
                {.page_id = row + depth * row_stride},
                {.offset_bytes = slot * aligned_stick_nbytes_in},
                {.trid = rtrid});
        }

        for (uint32_t j = 0; j < r; j++) {
            noc.async_write<NocOptions::TXN_ID>(
                use<experimental::CB::AddrSelector::WRITE_PTR>(cb_scratch),
                s_out,
                stick_nbytes_out,
                {.offset_bytes = (sbase + j) * aligned_stick_nbytes_out},
                {.page_id = page0 + j * rw_page_stride},
                {.trid = wtrid});
        }

        row += row_stride;
    }

    noc.async_write_barrier();  // drain every slot's writes
}
