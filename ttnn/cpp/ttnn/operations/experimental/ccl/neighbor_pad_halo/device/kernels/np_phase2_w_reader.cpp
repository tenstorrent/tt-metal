// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 2 W-fabric reader for neighbor_pad_halo.
//
// Reads W-boundary sticks from the input tensor (interior rows) and the compact halo
// buffer (H-padded rows) into a CB that the paired writer ships over the W fabric. H must
// finish first: the reader takes the H->W barrier once (upfront, or at the interior->corner
// transition on the interior-first path) before touching any H-halo corner row.

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_halo/device/kernels/np_reorder.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_halo/device/kernels/np_zero_pad.hpp"
#include <cstdint>

using address_t = uint32_t;

// Compile-time args (uniform across all W fabric reader cores)
constexpr uint32_t cb_output_id = get_compile_time_arg_val(0);
constexpr bool is_padding_zeros = get_compile_time_arg_val(1);
constexpr uint32_t stick_size = get_compile_time_arg_val(2);
// Halo buffer TensorAccessorArgs start at index 3 (variable length)
constexpr auto dst_args = TensorAccessorArgs<3>();
constexpr uint32_t ct_after_dst = dst_args.next_compile_time_args_offset();
// Input tensor TensorAccessorArgs follow the halo buffer args
constexpr auto src_args = TensorAccessorArgs<ct_after_dst>();
constexpr uint32_t ct_after_src = src_args.next_compile_time_args_offset();
// W-send bank-major coalesce factor (0 = per-stick). When > 0 (halo-only, pw==1, 8-aligned bases), a
// middle device gathers same-dst-bank sticks (rel, rel+8, ...) into the CB so the writer ships N of them
// as one N*page fabric packet. BH has 8 interleaved DRAM banks.
constexpr uint32_t W_COALESCE = get_compile_time_arg_val(ct_after_src);
// Uniform-mux mode: all W devices (incl. edges) use the coalesce path so the recv-sem targeting is
// consistent across the whole W chain. Edge devices skip the send-gather for their no-neighbor direction.
constexpr uint32_t W_MUX_MODE = get_compile_time_arg_val(ct_after_src + 1);
// Padded-output border mode: after this core has observed its compact rows (W-recv + H->W barrier),
// it writes the padded BORDER for those rows directly — visibility-safe, since it only reads compact
// sections it waited on. dir==0 cores write W-left + H-top/H-bot (pad rows); dir==1 write W-right. The
// interior is written by the free-core scatter. The CRTA [3]/[4+] slots carry the border args:
// [3]=padded_addr, [4]=wleft_base, [5]=wright_base, [6]=pad2_right.
constexpr uint32_t SCATTER_BORDER = get_compile_time_arg_val(ct_after_src + 2);
constexpr uint32_t SCATTER_SCRATCH_CB = get_compile_time_arg_val(ct_after_src + 3);  // private L1 scratch
constexpr auto padded_args = TensorAccessorArgs<ct_after_src + 4>();
constexpr uint32_t NP_NUM_DRAM_BANKS = 8;

void kernel_main() {
    // Common runtime args (uniform across all cores, updated between dispatches)
    const address_t output_tensor_address = get_common_arg_val<address_t>(0);
    const uint32_t barrier_sem_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t w_neighbor_sem_addr = get_common_arg_val<uint32_t>(2);
    // CRTA [3]/[4+] carry the padded-output border args when SCATTER_BORDER is set — read there, not here.

    // Per-core runtime args
    uint32_t arg_idx = 0;
    // outer_dim_size: rows this core processes = slice_frames * h_total (T-frames x padded-H rows) — NOT
    // B*T. outer_dim_start: this core's first such row (link-local slice offset).
    const uint32_t outer_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t outer_dim_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding = get_arg_val<uint32_t>(arg_idx++);        // W-halo cols per side this core emits (pad2)
    const uint32_t barrier_count = get_arg_val<uint32_t>(arg_idx++);  // H->W barrier signals to await (H producers)
    const uint32_t output_row_width = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pad2_left = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_interior_sticks = get_arg_val<uint32_t>(arg_idx++);  // interior W width (aka W_dev / Wd)
    const bool is_first_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool is_last_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    const address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t input_H_dev = get_arg_val<uint32_t>(arg_idx++);  // interior H per device (aka H_dev / Hd)
    const uint32_t padding_h = get_arg_val<uint32_t>(arg_idx++);
    // Compact-buffer page offset to the H-bottom section; H-top section base is 0.
    const uint32_t h_halo_hbot_base = get_arg_val<uint32_t>(arg_idx++);
    // Padded-input mode (0 = contiguous). When >0 the input is [.,H+2*input_pad_h,W+2*input_pad_w,C] and
    // the W-edge input reads target its INTERIOR (row stride = padded W, frame stride = padded H*W).
    const uint32_t input_pad_h = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_pad_w = get_arg_val<uint32_t>(arg_idx++);
    // Input page for interior (t, h_in, w_col); honors padded-input strides (identity when pad==0).
    const uint32_t in_Wp = num_interior_sticks + 2 * input_pad_w;
    auto in_page = [&](uint32_t t, uint32_t h_in, uint32_t w_col) -> uint32_t {
        return t * (input_H_dev + 2 * input_pad_h) * in_Wp + (h_in + input_pad_h) * in_Wp + (w_col + input_pad_w);
    };
    const uint32_t h_total = input_H_dev + 2 * padding_h;  // padded H rows per frame (aka Hp)

    const auto input_accessor = TensorAccessor(src_args, input_tensor_address, stick_size);
    const auto dst_accessor = TensorAccessor(dst_args, output_tensor_address, stick_size);

    // output_row_width is unused here; it stays in the RTA layout only to keep it stable across call sites.
    (void)output_row_width;

    volatile tt_l1_ptr uint32_t* w_neighbor_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_neighbor_sem_addr);

    // Fused padded-output border scatter. Called once per W-reader core at its completion path (this core
    // has waited its W-recv + the H->W barrier, so the compact sections it reads are visible to IT — no
    // cross-core race). dir==0 writes W-left + H-top/H-bot interior (pad rows); dir==1 writes W-right.
    // Compact page layout (t-major): H-top base 0, H-bot base h_halo_hbot_base, W-left base wleft_base,
    // W-right base wright_base; W section stride = h_total * pad, H section stride = padding_h * Wd.
    auto scatter_border_rows = [&]() {
        if constexpr (SCATTER_BORDER) {
            const uint32_t padded_addr = get_common_arg_val<uint32_t>(3);
            const uint32_t wleft_base = get_common_arg_val<uint32_t>(4);
            const uint32_t wright_base = get_common_arg_val<uint32_t>(5);
            const uint32_t pad2_right = get_common_arg_val<uint32_t>(6);
            const uint32_t logical_h = get_common_arg_val<uint32_t>(7);         // 0 = no H masking
            const uint32_t device_h_offset = get_common_arg_val<uint32_t>(8);   // global H of this shard row 0
            const uint32_t logical_w = get_common_arg_val<uint32_t>(9);         // 0 = no W masking
            const uint32_t device_w_offset = get_common_arg_val<uint32_t>(10);  // global W of this shard col 0
            const auto padded = TensorAccessor(padded_args, padded_addr, stick_size);
            const uint32_t Wd = num_interior_sticks;  // interior W (== num_sticks_per_halo_dim)
            const uint32_t pH = padding_h;
            const uint32_t Hd = input_H_dev;
            const uint32_t Hp = h_total;                      // padded H = Hd + 2*pH
            const uint32_t Wp = Wd + pad2_left + pad2_right;  // padded W
            const uint32_t h_sec = h_halo_hbot_base;          // H-bot compact base (H-top base = 0)
            const uint64_t zeros_noc = get_noc_addr(MEM_ZEROS_BASE);
            // Batch compact->L1 reads and L1->padded writes to amortize the NOC barriers: per-stick
            // read+write barriers dominate otherwise (thousands of border sticks, serial on this core).
            constexpr uint32_t BATCH = 16;
            const uint32_t l1_base = get_write_ptr(SCATTER_SCRATCH_CB);
            uint32_t dpg[BATCH];
            uint32_t n = 0;
            auto flush = [&]() {
                if (n == 0) {
                    return;
                }
                noc_async_read_barrier();
                for (uint32_t i = 0; i < n; i++) {
                    noc_async_write(l1_base + i * stick_size, get_noc_addr(dpg[i], padded), stick_size);
                }
                noc_async_write_barrier();
                n = 0;
            };
            // masked reads pull zeros from MEM_ZEROS instead of the compact buffer (logical padding region).
            auto copy = [&](uint32_t src, uint32_t dpage, bool masked) {
                if (masked) {
                    noc_async_read(zeros_noc, l1_base + n * stick_size, stick_size);
                } else {
                    noc_async_read(get_noc_addr(src, dst_accessor), l1_base + n * stick_size, stick_size);
                }
                dpg[n++] = dpage;
                if (n == BATCH) {
                    flush();
                }
            };
            for (uint32_t rel = 0; rel < outer_dim_size; rel++) {
                const uint32_t gi = outer_dim_start + rel;
                const uint32_t t = gi / Hp;
                const uint32_t hp = gi - t * Hp;
                const uint32_t dframe = t * Hp * Wp + hp * Wp;
                // A content row (pH <= hp < pH+Hd) is masked when its global H index reaches logical_h; the
                // whole row's border (W-edge + interior) is then zeroed.
                const bool h_row_masked =
                    logical_h > 0 && hp >= pH && hp < pH + Hd && device_h_offset + (hp - pH) >= logical_h;
                if (direction == 0) {
                    for (uint32_t wc = 0; wc < pad2_left; wc++) {
                        copy(wleft_base + t * Hp * pad2_left + hp * pad2_left + wc, dframe + wc, h_row_masked);
                    }
                    if (hp < pH || hp >= pH + Hd) {
                        const uint32_t pr = (hp < pH) ? hp : (hp - pH - Hd);
                        const uint32_t hbase = (hp < pH) ? 0u : h_sec;
                        for (uint32_t w = 0; w < Wd; w++) {
                            const bool w_masked = logical_w > 0 && device_w_offset + w >= logical_w;
                            copy(hbase + t * pH * Wd + pr * Wd + w, dframe + pad2_left + w, w_masked);
                        }
                    }
                } else {
                    for (uint32_t wc = 0; wc < pad2_right; wc++) {
                        copy(
                            wright_base + t * Hp * pad2_right + hp * pad2_right + wc,
                            dframe + pad2_left + Wd + wc,
                            h_row_masked);
                    }
                }
            }
            flush();
        }
    };

    // Frames in this core's slice (link-local; the partial last batch lives here when T % N != 0).
    const uint32_t slice_frames = (h_total > 0) ? (outer_dim_size / h_total) : 0;
    // Interior-first layout: rows [0, interior_rows) are H-independent (read from INPUT), then corner
    // rows [interior_rows, ...) (read from the halo H-section). Barrier taken at the transition.
    const uint32_t interior_rows = slice_frames * input_H_dev;

    // H->W ordering:
    //   Fast path (interior_first_p0): frames align to this core's slice, so np_reorder_batch below emits
    //   ALL interior rows first (overlapping the H exchange), then ALL corners. The H->W barrier is then
    //   taken ONCE at the interior->corner transition (outer_dim == interior_rows) below.
    //   Fallback (partial last frame): reorder can't cover a partial frame safely, so read linearly and
    //   take the H->W barrier upfront here.
    // Coalescing (middle device) uses the upfront barrier (bank-major mixes interior+corner), so it
    // opts out of interior-first.
    // Coalesce for middle devices always; in uniform-mux mode, edge devices coalesce too.
    const bool w_coalesce_active = (W_COALESCE > 0) && (W_MUX_MODE || (!is_first_chip && !is_last_chip));
    // Send-side neighbor for THIS direction (writer sends only when it exists). Per-core args are
    // direction-swapped by the factory, so the send condition is uniformly !is_last_chip. Edge devices
    // skip the gather for their no-neighbor direction (paired mux writer also skips) so the CB isn't left full.
    const bool has_send_neighbor = !is_last_chip;
    const bool interior_first_p0 =
        (barrier_count > 0) && (outer_dim_size == slice_frames * h_total) && !w_coalesce_active;
    // Fallback path (not interior-first): take the whole H->W barrier upfront before any corner read.
    if (!interior_first_p0 && barrier_count > 0) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr), barrier_count);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr), 0);
    }

    // Coalesced bank-major gather (middle device): read the W-edge stick for rows in dst-bank order
    // (rel, rel+8, ...) into the CB in groups of up to W_COALESCE, so the paired writer ships each group
    // as one contiguous fabric packet. Corners (H-padded rows) are safe here — the upfront barrier above
    // guaranteed the H exchange is complete.
    if constexpr (W_COALESCE > 0) {
        if (w_coalesce_active) {
            const uint32_t w_col = direction ? 0u : (num_interior_sticks - 1u);
            for (uint32_t j = 0; has_send_neighbor && j < NP_NUM_DRAM_BANKS; j++) {
                uint32_t r = j;
                while (r < outer_dim_size) {
                    uint32_t g = 0;
                    for (uint32_t rr = r; g < W_COALESCE && rr < outer_dim_size; rr += NP_NUM_DRAM_BANKS) {
                        g++;
                    }
                    cb_reserve_back(cb_output_id, g);
                    const uint32_t base_l1 = get_write_ptr(cb_output_id);
                    for (uint32_t m = 0; m < g; m++) {
                        const uint32_t rel = r + m * NP_NUM_DRAM_BANKS;
                        const uint32_t global_idx = outer_dim_start + rel;
                        const uint32_t t_idx = global_idx / h_total;
                        const uint32_t hp = global_idx % h_total;
                        const uint32_t l1_addr = base_l1 + m * stick_size;
                        // hp is the padded-H row: interior content lives in the input tensor, the top/bottom
                        // pad rows in the compact H-sections. Same page math recurs in the main loop below.
                        if (hp >= padding_h && hp < padding_h + input_H_dev) {
                            const uint32_t page = in_page(t_idx, hp - padding_h, w_col);
                            noc_async_read(get_noc_addr(page, input_accessor), l1_addr, stick_size);
                        } else {
                            // Compact H-section is [t][pad_row][w], row-major with stride padding_h*W_dev per
                            // frame. Top pad (hp<padding_h) starts at section base 0; bottom pad at hbot_base.
                            uint32_t halo_page;
                            if (hp < padding_h) {
                                halo_page = t_idx * padding_h * num_interior_sticks + hp * num_interior_sticks + w_col;
                            } else {
                                const uint32_t pad_row = hp - padding_h - input_H_dev;
                                halo_page = h_halo_hbot_base + t_idx * padding_h * num_interior_sticks +
                                            pad_row * num_interior_sticks + w_col;
                            }
                            noc_async_read(get_noc_addr(halo_page, dst_accessor), l1_addr, stick_size);
                        }
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_output_id, g);
                    r += g * NP_NUM_DRAM_BANKS;  // next group start on this bank
                }
            }
            // Receive-completion: wait for all incoming W-halo rows to land before the op returns.
            if (!is_first_chip) {
                noc_semaphore_wait_min(w_neighbor_sem_ptr, outer_dim_size);
                noc_semaphore_set(w_neighbor_sem_ptr, 0);
            }
            scatter_border_rows();
            return;
        }
    }

    // Main loop: read W-boundary sticks → CB for the paired writer.
    for (uint32_t outer_dim = 0; outer_dim < outer_dim_size; outer_dim++) {
        // outer_dim maps to (t, h_padded). Per-batch two-pass reorder (interior rows of a batch before
        // its corners) for FULL batches; linear for the partial last batch (and when not batching).
        uint32_t t_idx;
        uint32_t h_padded;
        if (interior_first_p0) {
            // Interior rows first (H-independent), corners last. W does interior work while H produces.
            uint32_t frame_in_slice;
            np_reorder_batch(outer_dim, slice_frames, input_H_dev, padding_h, frame_in_slice, h_padded);
            t_idx = (outer_dim_start / h_total) + frame_in_slice;
        } else {
            const uint32_t global_idx = outer_dim_start + outer_dim;
            t_idx = global_idx / h_total;
            h_padded = global_idx % h_total;
        }
        const bool h_interior = (h_padded >= padding_h && h_padded < padding_h + input_H_dev);

        // Interior-first path: take the H->W barrier ONCE at the first corner row (all interior done).
        if (interior_first_p0 && outer_dim == interior_rows) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr), barrier_count);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr), 0);
        }

        if (is_first_chip) {
            if (!is_padding_zeros) {
                cb_reserve_back(cb_output_id, 1);
                uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                uint32_t w_col = direction ? (num_interior_sticks - 1) : 0;
                if (h_interior) {
                    uint32_t h_in = h_padded - padding_h;
                    uint32_t page = in_page(t_idx, h_in, w_col);
                    noc_async_read(get_noc_addr(page, input_accessor), dst_l1_addr, stick_size);
                } else {
                    uint32_t halo_page;
                    if (h_padded < padding_h) {
                        uint32_t pad_row = h_padded;
                        halo_page = t_idx * padding_h * num_interior_sticks + pad_row * num_interior_sticks + w_col;
                    } else {
                        uint32_t pad_row = h_padded - padding_h - input_H_dev;
                        halo_page = h_halo_hbot_base + t_idx * padding_h * num_interior_sticks +
                                    pad_row * num_interior_sticks + w_col;
                    }
                    noc_async_read(get_noc_addr(halo_page, dst_accessor), dst_l1_addr, stick_size);
                }
                noc_async_read_barrier();
                cb_push_back(cb_output_id, 1);
            } else {
                cb_reserve_back(cb_output_id, 1);
                zeroPadCb<stick_size>(cb_output_id);
                noc_async_read_barrier();
                cb_push_back(cb_output_id, 1);
            }
        }

        if (!is_last_chip) {
            for (uint32_t pad_id = padding; pad_id > 0; pad_id--) {
                cb_reserve_back(cb_output_id, 1);
                uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                uint32_t w_col;
                if (direction) {
                    w_col = padding - pad_id;
                } else {
                    w_col = num_interior_sticks - pad_id;
                }
                if (h_interior) {
                    uint32_t h_in = h_padded - padding_h;
                    uint32_t page = in_page(t_idx, h_in, w_col);
                    noc_async_read(get_noc_addr(page, input_accessor), dst_l1_addr, stick_size);
                } else {
                    uint32_t halo_page;
                    if (h_padded < padding_h) {
                        uint32_t pad_row = h_padded;
                        halo_page = t_idx * padding_h * num_interior_sticks + pad_row * num_interior_sticks + w_col;
                    } else {
                        uint32_t pad_row = h_padded - padding_h - input_H_dev;
                        halo_page = h_halo_hbot_base + t_idx * padding_h * num_interior_sticks +
                                    pad_row * num_interior_sticks + w_col;
                    }
                    noc_async_read(get_noc_addr(halo_page, dst_accessor), dst_l1_addr, stick_size);
                }
                noc_async_read_barrier();
                cb_push_back(cb_output_id, 1);
            }
        }
    }

    // The op's output IS the halo buffer, so the receiver must wait for ALL incoming W-halo rows to land
    // in DRAM before the kernel exits — otherwise the host reads an in-flight buffer.
    if (!is_first_chip) {
        noc_semaphore_wait_min(w_neighbor_sem_ptr, outer_dim_size);
        noc_semaphore_set(w_neighbor_sem_ptr, 0);  // monotonic across the loop; single end-of-kernel reset
    }
    // Non-coalesce completion: scatter the padded border (the coalesce path already returned above).
    scatter_border_rows();
}
