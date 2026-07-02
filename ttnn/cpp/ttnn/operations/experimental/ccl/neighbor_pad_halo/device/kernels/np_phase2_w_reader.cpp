// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 2 W fabric reader for the fused neighbor_pad + conv3d op.
//
// Reads W-boundary sticks from the input tensor (interior rows) and the compact halo
// buffer (H-padded rows) into a CB that the paired writer ships over the W fabric.
// Per-T-batch, the receiver-side reader increments a progress semaphore on every
// conv3d reader core so conv3d compute can start processing early T-blocks while
// later ones are still in flight.

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
// Granularity (in T-input frames) for per-batch progress-sem signals. Must match the
// conv3d reader's progress_t_batch_size compile-time arg.
constexpr uint32_t progress_t_batch_size = get_compile_time_arg_val(ct_after_src);

// Global two-pass gate, set per-shape by the program factory (lockstep with np_writer via the same
// factory value). ON: defer all corners past H's finish (NP-bound win, regresses conv-bound).
// Follows progress_t_batch_size (ct_after_src) in the W-reader arg layout.
constexpr bool W_TWO_PASS = get_compile_time_arg_val(ct_after_src + 1);

// W-send bank-major coalesce factor (0 = per-stick). When > 0 (halo-only, pw==1, 8-aligned bases), a
// middle device gathers same-dst-bank sticks (rel, rel+8, ...) into the CB so the writer ships N of them
// as one N*page fabric packet. BH has 8 interleaved DRAM banks.
constexpr uint32_t W_COALESCE = get_compile_time_arg_val(ct_after_src + 2);
// Uniform-mux mode: all W devices (incl. edges) use the coalesce path so the recv-sem targeting is
// consistent across the whole W chain. Edge devices skip the send-gather for their no-neighbor direction.
constexpr uint32_t W_MUX_MODE = get_compile_time_arg_val(ct_after_src + 3);
constexpr uint32_t NP_NUM_DRAM_BANKS = 8;

void kernel_main() {
    // Common runtime args (uniform across all cores, updated between dispatches)
    const address_t output_tensor_address = get_common_arg_val<address_t>(0);
    const uint32_t barrier_sem_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t w_neighbor_sem_addr = get_common_arg_val<uint32_t>(2);
    // [3] num_reader_cores: number of conv3d reader cores to signal.
    // [4+]: NOC coords of conv3d reader cores.
    const uint32_t num_reader_cores = get_common_arg_val<uint32_t>(3);

    // Per-core runtime args
    uint32_t arg_idx = 0;
    const uint32_t outer_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t outer_dim_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t barrier_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_row_width = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pad2_left = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_interior_sticks = get_arg_val<uint32_t>(arg_idx++);
    const bool is_first_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool is_last_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    const address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t input_H_dev = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding_h = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t h_halo_hbot_base = get_arg_val<uint32_t>(arg_idx++);
    // Per-batch region progress sem for this (W-direction, link). W-edge/corner conv3d tiles wait on
    // just this link's count, race-free across links (one producer per (region,link) -> monotonic).
    const uint32_t w_region_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t h_total = input_H_dev + 2 * padding_h;

    // Per-batch corner H-gate: each corner W-stick read waits only the H-batch it needs (no upfront
    // H->W barrier). H-region sems for all H-links live in CRTA after the reader coords: HT[0..3],
    // HB[0..3], h_batches_per_link, num_h_links, h_total_batches.
    uint32_t htop_sem[4] = {0, 0, 0, 0};
    uint32_t hbot_sem[4] = {0, 0, 0, 0};
    uint32_t h_batches_per_link = 0, num_h_links = 0, h_total_batches = 0;
    if constexpr (progress_t_batch_size > 0) {
        const uint32_t h_base = 4 + 2 * num_reader_cores;
        for (uint32_t l = 0; l < 4; l++) {
            htop_sem[l] = get_common_arg_val<uint32_t>(h_base + l);
            hbot_sem[l] = get_common_arg_val<uint32_t>(h_base + 4 + l);
        }
        h_batches_per_link = get_common_arg_val<uint32_t>(h_base + 8);
        num_h_links = get_common_arg_val<uint32_t>(h_base + 9);
        h_total_batches = get_common_arg_val<uint32_t>(h_base + 10);
    }

    const auto input_accessor = TensorAccessor(src_args, input_tensor_address, stick_size);
    const auto dst_accessor = TensorAccessor(dst_args, output_tensor_address, stick_size);

    // output_row_width and pad2_left are unused in the fabric-only path; keep the RTA layout stable
    // so the factory can share code with the standalone NP op.
    (void)output_row_width;
    (void)pad2_left;

    if constexpr (progress_t_batch_size > 0) {
        (void)barrier_sem_addr;
        (void)barrier_count;
    }

    volatile tt_l1_ptr uint32_t* w_neighbor_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_neighbor_sem_addr);

    // outer_dim runs over T_in * h_total; one progress-sem batch = progress_t_batch_size * h_total sticks.
    const uint32_t sticks_per_batch = progress_t_batch_size * h_total;
    // Frames in this core's slice (link-local; the partial last batch lives here when T % N != 0).
    const uint32_t slice_frames = (h_total > 0) ? (outer_dim_size / h_total) : 0;
    // Interior-first layout: rows [0, interior_rows) are H-independent (read from INPUT), then corner
    // rows [interior_rows, ...) (read from the halo H-section). Signal / barrier at the transition.
    const uint32_t interior_rows = slice_frames * input_H_dev;
    const uint32_t corner_rows_per_batch = progress_t_batch_size * 2 * padding_h;

    // H->W ordering (progress==0, halo-only, no conv):
    //   Fast path (interior_first_p0): frames align to this core's slice, so np_reorder_batch below emits
    //   ALL interior rows first (overlapping the H exchange), then ALL corners. The H->W barrier is then
    //   taken ONCE at the interior->corner transition (outer_dim == interior_rows) below.
    //   Fallback (partial last frame): reorder can't cover a partial frame safely, so read linearly and
    //   take the H->W barrier upfront here.
    // (progress>0 fused path gates corners per-batch on HT/HB and ignores this barrier.)
    // Coalescing (middle device) uses the upfront barrier (bank-major mixes interior+corner), so it
    // opts out of interior-first.
    // Coalesce for middle devices always; in uniform-mux mode, edge devices coalesce too.
    const bool w_coalesce_active = (W_COALESCE > 0) && (W_MUX_MODE || (!is_first_chip && !is_last_chip));
    // Send-side neighbor for THIS direction (writer sends only when it exists). Per-core args are
    // direction-swapped by the factory, so the send condition is uniformly !is_last_chip. Edge devices
    // skip the gather for their no-neighbor direction (paired mux writer also skips) so the CB isn't left full.
    const bool has_send_neighbor = !is_last_chip;
    const bool interior_first_p0 = (progress_t_batch_size == 0) && (barrier_count > 0) &&
                                   (outer_dim_size == slice_frames * h_total) && !w_coalesce_active;
    if constexpr (progress_t_batch_size == 0) {
        if (!interior_first_p0 && barrier_count > 0) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr), barrier_count);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr), 0);
        }
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
                        if (hp >= padding_h && hp < padding_h + input_H_dev) {
                            const uint32_t page = t_idx * input_H_dev * num_interior_sticks +
                                                  (hp - padding_h) * num_interior_sticks + w_col;
                            noc_async_read(get_noc_addr(page, input_accessor), l1_addr, stick_size);
                        } else {
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
            return;
        }
    }

    // Main loop: read W-boundary sticks → CB for the paired writer.
    for (uint32_t outer_dim = 0; outer_dim < outer_dim_size; outer_dim++) {
        // outer_dim maps to (t, h_padded). Per-batch two-pass reorder (interior rows of a batch before
        // its corners) for FULL batches; linear for the partial last batch (and when not batching).
        uint32_t t_idx;
        uint32_t h_padded;
        bool use_reorder;
        if constexpr (progress_t_batch_size > 0) {
            use_reorder = W_TWO_PASS;
        } else {
            use_reorder = interior_first_p0;
        }
        if (use_reorder) {
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

        // progress==0 fast path: take the H->W barrier ONCE at the first corner row (all interior done).
        if constexpr (progress_t_batch_size == 0) {
            if (interior_first_p0 && outer_dim == interior_rows) {
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr), barrier_count);
                noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr), 0);
            }
        }

        // A non-interior row is a corner stick built from H-halo. Before reading it, wait this
        // frame's H batch committed across ALL H-links (the W-reader's frames span the H partition).
        // Top-pad → HT, bot-pad → HB. addr 0 = no H neighbor on that side → skip (zero-pad).
        if constexpr (progress_t_batch_size > 0) {
            if (!h_interior && h_batches_per_link > 0) {
                uint32_t need = t_idx / progress_t_batch_size + 1;
                if (need > h_total_batches) {
                    need = h_total_batches;
                }
                uint32_t hlink = (need - 1) / h_batches_per_link;
                if (hlink >= num_h_links) {
                    hlink = num_h_links - 1;
                }
                const uint32_t* hsem = (h_padded < padding_h) ? htop_sem : hbot_sem;
                for (uint32_t l = 0; l <= hlink; l++) {
                    if (hsem[l] == 0) {
                        continue;
                    }
                    const uint32_t end_b = (l + 1) * h_batches_per_link;
                    const uint32_t thr = (need < end_b ? need : end_b) - l * h_batches_per_link;
                    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hsem[l]), thr);
                }
            }
        }

        if (is_first_chip) {
            if (!is_padding_zeros) {
                cb_reserve_back(cb_output_id, 1);
                uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                uint32_t w_col = direction ? (num_interior_sticks - 1) : 0;
                if (h_interior) {
                    uint32_t h_in = h_padded - padding_h;
                    uint32_t page = t_idx * input_H_dev * num_interior_sticks + h_in * num_interior_sticks + w_col;
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
                    uint32_t page = t_idx * input_H_dev * num_interior_sticks + h_in * num_interior_sticks + w_col;
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

        // Per-batch conv-signalling — fused (progress>0) path only. Receiver signals a conv W-edge tile's
        // region sem after wait_min(w_neighbor_sem, outer_dim+1) confirms this batch's remote fabric write
        // landed. The halo-only op (progress==0) has no conv consumer, so it skips all of this and does a
        // single receive-completion wait at the end (see tail below) — the "no signalling" standalone path.
        if constexpr (progress_t_batch_size > 0) {
            bool do_signal;
            if constexpr (W_TWO_PASS) {
                do_signal = (outer_dim >= interior_rows) && (corner_rows_per_batch > 0) &&
                            ((outer_dim + 1 - interior_rows) % corner_rows_per_batch == 0);
            } else {
                do_signal = ((outer_dim + 1) % sticks_per_batch == 0);
            }
            if (!is_first_chip && do_signal) {
                noc_semaphore_wait_min(w_neighbor_sem_ptr, outer_dim + 1);
                noc_async_write_barrier();
                for (uint32_t i = 0; i < num_reader_cores; i++) {
                    const uint32_t rx = get_common_arg_val<uint32_t>(4 + i * 2);
                    const uint32_t ry = get_common_arg_val<uint32_t>(4 + i * 2 + 1);
                    noc_semaphore_inc(get_noc_addr(rx, ry, w_region_sem_addr), 1);
                }
                noc_async_atomic_barrier();
            }
        }
    }

    if constexpr (progress_t_batch_size > 0) {
        // Fused path: tail conv-signal for the partial last batch (receiver side only).
        bool tail_signal;
        if constexpr (W_TWO_PASS) {
            tail_signal =
                (corner_rows_per_batch > 0) && ((outer_dim_size - interior_rows) % corner_rows_per_batch != 0);
        } else {
            tail_signal = (outer_dim_size % sticks_per_batch != 0);
        }
        if (!is_first_chip && tail_signal) {
            noc_semaphore_wait_min(w_neighbor_sem_ptr, outer_dim_size);
            noc_async_write_barrier();
            for (uint32_t i = 0; i < num_reader_cores; i++) {
                const uint32_t rx = get_common_arg_val<uint32_t>(4 + i * 2);
                const uint32_t ry = get_common_arg_val<uint32_t>(4 + i * 2 + 1);
                noc_semaphore_inc(get_noc_addr(rx, ry, w_region_sem_addr), 1);
            }
            noc_async_atomic_barrier();
        }
    } else {
        // Halo-only op: no conv consumer. The op's output IS the halo buffer, so the receiver must wait
        // for ALL incoming W-halo rows to land in DRAM before the kernel exits — otherwise the host reads
        // an in-flight buffer. One receive-completion wait replaces the per-batch conv signalling.
        if (!is_first_chip) {
            noc_semaphore_wait_min(w_neighbor_sem_ptr, outer_dim_size);
        }
    }
    // Single end-of-kernel reset; sem was monotonically increasing across the loop.
    if (!is_first_chip) {
        noc_semaphore_set(w_neighbor_sem_ptr, 0);
    }
    // Trace-safe self-reset of the H-region sems this core consumed in the corner gate above: the H
    // producer increments HT/HB on the W-reader cores too, and the per-batch corner waits guarantee
    // those increments have landed, so zero them for the next dispatch without a host-side reset.
    if constexpr (progress_t_batch_size > 0) {
        for (uint32_t l = 0; l < num_h_links && l < 4; l++) {
            if (htop_sem[l] != 0) {
                noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(htop_sem[l]), 0);
            }
            if (hbot_sem[l] != 0) {
                noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hbot_sem[l]), 0);
            }
        }
    }
}
