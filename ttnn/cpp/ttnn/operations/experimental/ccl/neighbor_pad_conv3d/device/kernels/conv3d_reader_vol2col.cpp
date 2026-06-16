// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_conv3d/device/kernels/conv3d_vol2col_lib.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_conv3d/device/kernels/np_halo_block.hpp"

// Halo-aware gather dispatch: in-bounds blocks take the optimized interior gather
// (coalesced / trid-ring / DRAM-staged), boundary blocks read the NP halo buffer.
// Lives at file scope so the enclosing-scope constexpr template params are passed explicitly.
template <
    uint32_t C_in_block_bytes,
    bool is_padding_zeros,
    uint32_t H_shard_max_W_shard_max,
    uint32_t W_shard_max,
    uint32_t T_in,
    uint32_t H_in,
    uint32_t W_in,
    uint32_t H_in_W_in,
    uint32_t in_row_size_bytes,
    uint32_t gather_trids,
    bool enable_coalesced_shard_reads,
    bool enable_dram_read_staging,
    uint32_t dram_read_alignment,
    typename Reader>
FORCE_INLINE void gather_rows_selected_or_halo(
    Noc noc,
    const Reader& in_reader,
    const Reader& halo_reader,
    const experimental::CB& shard_cb,
    const experimental::CB& dram_read_scratch_cb,
    uint32_t shard_l1_base,
    uint32_t coalesced_scratch_offset,
    bool all_in_bounds,
    bool use_coalesced,
    uint32_t batch_page_base,
    uint32_t batch_idx,
    uint32_t c_in_offset_bytes,
    int32_t t_shard_start,
    uint32_t T_shard_cur,
    int32_t h_shard_start,
    uint32_t h_start,
    uint32_t h_end,
    int32_t w_shard_start,
    uint32_t w_col_start,
    uint32_t w_count,
    uint32_t coalesced_scratch_rows,
    uint32_t h_halo_outer_dim_size,
    uint32_t h_halo_H,
    uint32_t h_halo_W,
    uint32_t h_halo_padding_h,
    uint32_t h_halo_padding_w,
    uint32_t h_halo_hbot_base,
    uint32_t h_halo_wleft_base,
    uint32_t h_halo_wright_base) {
    if (all_in_bounds) {
        gather_rows_to_shard_selected<
            C_in_block_bytes,
            is_padding_zeros,
            H_shard_max_W_shard_max,
            W_shard_max,
            T_in,
            H_in,
            W_in,
            H_in_W_in,
            in_row_size_bytes,
            gather_trids,
            enable_coalesced_shard_reads,
            enable_dram_read_staging,
            dram_read_alignment>(
            noc,
            in_reader,
            shard_cb,
            dram_read_scratch_cb,
            shard_l1_base,
            coalesced_scratch_offset,
            /*all_in_bounds=*/true,
            use_coalesced,
            batch_page_base,
            c_in_offset_bytes,
            t_shard_start,
            T_shard_cur,
            h_shard_start,
            h_start,
            h_end,
            w_shard_start,
            w_col_start,
            w_count,
            coalesced_scratch_rows);
    } else {
        gather_rows_halo<
            C_in_block_bytes,
            H_shard_max_W_shard_max,
            W_shard_max,
            T_in,
            H_in,
            W_in,
            H_in_W_in,
            in_row_size_bytes>(
            noc,
            in_reader,
            halo_reader,
            shard_cb,
            batch_page_base,
            batch_idx,
            c_in_offset_bytes,
            t_shard_start,
            T_shard_cur,
            h_shard_start,
            h_start,
            h_end,
            w_shard_start,
            w_col_start,
            w_count,
            h_halo_outer_dim_size,
            h_halo_H,
            h_halo_W,
            h_halo_padding_h,
            h_halo_padding_w,
            h_halo_hbot_base,
            h_halo_wleft_base,
            h_halo_wright_base);
    }
}

void kernel_main() {
    constexpr uint32_t cb_vol2col = get_compile_time_arg_val(0);
    constexpr uint32_t N = get_compile_time_arg_val(1);
    constexpr uint32_t T_in = get_compile_time_arg_val(2);
    constexpr uint32_t H_in = get_compile_time_arg_val(3);
    constexpr uint32_t W_in = get_compile_time_arg_val(4);
    constexpr uint32_t C_in = get_compile_time_arg_val(5);
    constexpr uint32_t T_out = get_compile_time_arg_val(6);
    constexpr uint32_t H_out = get_compile_time_arg_val(7);
    constexpr uint32_t W_out = get_compile_time_arg_val(8);
    constexpr uint32_t C_out = get_compile_time_arg_val(9);
    constexpr uint32_t padding_t = get_compile_time_arg_val(10);
    constexpr uint32_t padding_h = get_compile_time_arg_val(11);
    constexpr uint32_t padding_w = get_compile_time_arg_val(12);
    constexpr uint32_t kT = get_compile_time_arg_val(13);
    constexpr uint32_t kH = get_compile_time_arg_val(14);
    constexpr uint32_t kW = get_compile_time_arg_val(15);
    constexpr uint32_t T_block_size = get_compile_time_arg_val(16);
    constexpr uint32_t H_block_size = get_compile_time_arg_val(17);
    constexpr uint32_t W_block_size = get_compile_time_arg_val(18);
    constexpr uint32_t C_out_num_blocks = get_compile_time_arg_val(19);
    constexpr uint32_t in_row_size_bytes = get_compile_time_arg_val(20);
    constexpr uint32_t C_in_block_bytes = get_compile_time_arg_val(21);
    constexpr uint32_t out_row_size_bytes = get_compile_time_arg_val(22);
    constexpr bool is_padding_zeros = get_compile_time_arg_val(23) == 1;
    constexpr uint32_t semaphore_id = get_compile_time_arg_val(24);
    constexpr uint32_t stride_t = get_compile_time_arg_val(25);
    constexpr uint32_t stride_h = get_compile_time_arg_val(26);
    constexpr uint32_t stride_w = get_compile_time_arg_val(27);
    constexpr uint32_t dilation_t = get_compile_time_arg_val(28);
    constexpr uint32_t dilation_h = get_compile_time_arg_val(29);
    constexpr uint32_t dilation_w = get_compile_time_arg_val(30);
    // L1 prefetch buffer parameters
    constexpr uint32_t cb_input_shard = get_compile_time_arg_val(31);
    constexpr uint32_t T_shard_max = get_compile_time_arg_val(32);
    constexpr uint32_t H_shard_max = get_compile_time_arg_val(33);
    constexpr uint32_t W_shard_max = get_compile_time_arg_val(34);

    // Padding bytes to append after each patch row to reach tile-aligned CB page width
    constexpr uint32_t patch_pad_bytes = get_compile_time_arg_val(35);

    // Upstream gather tuning (re-based from conv3d #43541/#44418): trid-ring depth, coalesced
    // bank-major reads, and DRAM-read staging.  Host classifier (program factory) selects these.
    constexpr uint32_t gather_trids = get_compile_time_arg_val(36);
    constexpr bool enable_coalesced_shard_reads = get_compile_time_arg_val(37) == 1;
    constexpr uint32_t coalesced_scratch_rows = get_compile_time_arg_val(38);
    constexpr uint32_t cb_dram_read_scratch = get_compile_time_arg_val(39);
    constexpr bool enable_dram_read_staging = get_compile_time_arg_val(40) == 1;
    constexpr uint32_t dram_read_alignment = get_compile_time_arg_val(41);

    // Pipelining CT arg: per-(region,link) progress sems are signaled every `progress_t_batch_size`
    // input T-frames; the per-T-block halo wait below ramps on that batch count.
    constexpr uint32_t progress_t_batch_size = get_compile_time_arg_val(42);
    // halo_last: bulk-core two-phase. The conv processes its FULL output range in two passes —
    // Phase 0 = spatial-interior blocks (receptive field stays on-device, no NP wait), Phase 1 =
    // boundary blocks (h/w touch the device edge → need the halo) AFTER waiting NP. The interior
    // overlaps NP fully; boundary runs once NP has landed (no per-T-block stall). Identical
    // is_edge classification in compute/writer keeps the three kernels in lock-step.
    constexpr bool halo_last = get_compile_time_arg_val(43) == 1;
    constexpr uint32_t padded_page_bytes = kT * kH * kW * C_in_block_bytes + patch_pad_bytes;

    // Load input/output addresses and range parameters
    uint32_t argidx = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t c_in_block_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t c_in_block_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t c_out_block_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t c_out_block_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t t_out_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t t_out_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_out_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_out_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t w_out_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t w_out_end = get_arg_val<uint32_t>(argidx++);
    // Nonzero when this dispatch has W-halo signaling, used as the "halo waits apply" guard below.
    const uint32_t input_progress_signal_count = get_arg_val<uint32_t>(argidx++);

    const uint32_t h_halo_buffer_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_halo_outer_dim_size = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_halo_H = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_halo_W = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_halo_padding_h = get_arg_val<uint32_t>(argidx++);
    const uint32_t h_halo_padding_w = get_arg_val<uint32_t>(argidx++);
    // Per-(W-direction, link) progress sems: a W-edge tile maps the batch it needs to the owning
    // link (batch-aligned partition) and polls only that link's sem — race-free across links.
    const uint32_t w_pad2_num_links = get_arg_val<uint32_t>(argidx++);    // [18]
    const uint32_t w_batches_per_link = get_arg_val<uint32_t>(argidx++);  // [19]
    uint32_t wleft_sem_addr[4];                                           // [20..23]
    uint32_t wright_sem_addr[4];                                          // [24..27]
    for (uint32_t l = 0; l < 4; l++) {
        wleft_sem_addr[l] = get_arg_val<uint32_t>(argidx++);
    }
    for (uint32_t l = 0; l < 4; l++) {
        wright_sem_addr[l] = get_arg_val<uint32_t>(argidx++);
    }
    const uint32_t w_total_batches = get_arg_val<uint32_t>(argidx++);  // [28] cap for the W threshold
    // CRTA [29..39]: per-(H-region, link) sems + params. H-edge tiles wait these (no barrier), same
    // per-link scheme as W. addr 0 = no H neighbor on that side (zero-pad) → skip.
    uint32_t htop_sem_addr[4];  // [29..32]
    uint32_t hbot_sem_addr[4];  // [33..36]
    for (uint32_t l = 0; l < 4; l++) {
        htop_sem_addr[l] = get_arg_val<uint32_t>(argidx++);
    }
    for (uint32_t l = 0; l < 4; l++) {
        hbot_sem_addr[l] = get_arg_val<uint32_t>(argidx++);
    }
    const uint32_t h_batches_per_link = get_arg_val<uint32_t>(argidx++);  // [37]
    const uint32_t h_num_links = get_arg_val<uint32_t>(argidx++);         // [38]
    const uint32_t h_total_batches = get_arg_val<uint32_t>(argidx++);     // [39]
    // Compact buffer layout: [H-top | H-bot | W-left | W-right]
    const uint32_t h_halo_hbot_base = h_halo_outer_dim_size * h_halo_padding_h * h_halo_W;
    const uint32_t h_halo_wleft_base = 2u * h_halo_outer_dim_size * h_halo_padding_h * h_halo_W;
    const uint32_t h_halo_wright_base = h_halo_wleft_base + h_halo_outer_dim_size * h_halo_padding_w * h_halo_H;

    // Tensor accessor for input tensor and halo buffer (halo reuses in_args: both are
    // DRAM interleaved with the same page layout).
    constexpr auto in_args = TensorAccessorArgs<44>();
    const auto in_reader = TensorAccessor(in_args, in_addr);
    const auto halo_reader = TensorAccessor(in_args, h_halo_buffer_addr);

    Noc noc;

    constexpr uint32_t num_patches = T_block_size * H_block_size * W_block_size;
    constexpr uint32_t H_in_W_in = H_in * W_in;
    constexpr uint32_t T_in_H_in_W_in = T_in * H_in * W_in;

    // L1 prefetch: enabled when the host allocated a shard buffer (T_shard_max > 0).
    // The host decides based on kernel size, dilation, and L1 budget.
    constexpr bool use_l1_prefetch = (T_shard_max > 0);
    constexpr uint32_t H_shard_max_W_shard_max = H_shard_max * W_shard_max;

    // Reserve shard buffer once (used as scratch space, not streaming CB)
    experimental::CB shard_cb(cb_input_shard);
    experimental::CB dram_read_scratch_cb(cb_dram_read_scratch);
    if constexpr (enable_dram_read_staging) {
        dram_read_scratch_cb.reserve_back(1);
    }
    uint32_t shard_l1_base = 0;
    if constexpr (use_l1_prefetch) {
        constexpr uint32_t shard_total = T_shard_max * H_shard_max_W_shard_max;
        constexpr uint32_t coalesced_scratch_pages =
            enable_coalesced_shard_reads ? coalesced_scratch_rows * W_shard_max : 0;
        shard_cb.reserve_back(shard_total + coalesced_scratch_pages);
        shard_l1_base = shard_cb.get_write_ptr();
    }

    // A core needs the halo only when its output range touches a padded boundary; interior-only cores
    // never wait. The per-T-block halo wait below is gated on core_needs_halo, so interior tiles can
    // overlap the NP exchange while boundary tiles wait for the halo to land.
    const bool needs_h_top = padding_h > 0 && (h_out_start * stride_h < padding_h);
    const bool needs_h_bot = padding_h > 0 && ((h_out_end - 1) * stride_h + (kH - 1) * dilation_h >= H_in + padding_h);
    const bool needs_w_left = padding_w > 0 && (w_out_start * stride_w < padding_w);
    const bool needs_w_right =
        padding_w > 0 && ((w_out_end - 1) * stride_w + (kW - 1) * dilation_w >= W_in + padding_w);
    const bool core_needs_halo = needs_h_top || needs_h_bot || needs_w_left || needs_w_right;

    // Process each batch element
    for (uint32_t batch_idx = 0; batch_idx < N; batch_idx++) {
        const uint32_t batch_page_base = batch_idx * T_in_H_in_W_in;
        for (uint32_t c_in_block = c_in_block_start; c_in_block < c_in_block_end; c_in_block++) {
            const uint32_t c_in_offset_bytes = c_in_block * C_in_block_bytes;
            // Iterate only over assigned C_out blocks
            for (uint32_t c_out_block = c_out_block_start; c_out_block < c_out_block_end; c_out_block++) {
                // halo_last: pass 0 = interior blocks (overlap NP); pass 1 = boundary blocks after NP.
                for (uint32_t phase = 0; phase < (halo_last ? 2u : 1u); phase++) {
                    // 3D blocking loops over assigned ranges:
                    uint32_t t_iter = 0;
                    for (uint32_t t_block = t_out_start; t_block < t_out_end; t_block += T_block_size, ++t_iter) {
                        // Phase 0 (interior) needs no halo; phase 1 keeps the per-T wait, already satisfied
                        // since NP completes during phase 0. Legacy (!halo_last) always gates.
                        if ((!halo_last || phase == 1u) && input_progress_signal_count > 0 && core_needs_halo) {
                            // Batches (progress_t_batch_size units) that must be complete for this T-block
                            // to read its halo: the last T-input frame it touches is
                            // (t_block + T_block_size - 1) * stride_t + kT - 1 - padding_t; ceil-divide +1.
                            const uint32_t last_t_in = (t_block + T_block_size - 1) * stride_t + kT - 1 - padding_t;
                            const uint32_t desired_batches =
                                (last_t_in + progress_t_batch_size) / progress_t_batch_size;
                            // H-edge tiles wait the per-(H-side, link) sems (no barrier), same
                            // wait-all-links scheme as W. needs_h_top → HT, needs_h_bot → HB.
                            if ((needs_h_top || needs_h_bot) && h_batches_per_link > 0) {
                                uint32_t need = desired_batches;
                                if (need > h_total_batches) {
                                    need = h_total_batches;
                                }
                                uint32_t hlink = (need - 1) / h_batches_per_link;
                                if (hlink >= h_num_links) {
                                    hlink = h_num_links - 1;
                                }
                                for (uint32_t l = 0; l <= hlink; l++) {
                                    const uint32_t end_b = (l + 1) * h_batches_per_link;
                                    const uint32_t thr = (need < end_b ? need : end_b) - l * h_batches_per_link;
                                    if (needs_h_top && htop_sem_addr[l] != 0) {
                                        noc_semaphore_wait_min(
                                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(htop_sem_addr[l]), thr);
                                    }
                                    if (needs_h_bot && hbot_sem_addr[l] != 0) {
                                        noc_semaphore_wait_min(
                                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hbot_sem_addr[l]), thr);
                                    }
                                }
                            }
                            // W-edge/corner: poll only the (W-side, link) sem that owns the last needed
                            // batch. Each (side,link) has one producer → its count is monotonic.
                            if ((needs_w_left || needs_w_right) && w_batches_per_link > 0) {
                                uint32_t need = desired_batches;
                                if (need > w_total_batches) {
                                    need = w_total_batches;
                                }
                                uint32_t wlink = (need - 1) / w_batches_per_link;
                                if (wlink >= w_pad2_num_links) {
                                    wlink = w_pad2_num_links - 1;
                                }
                                for (uint32_t l = 0; l <= wlink; l++) {
                                    const uint32_t end_b = (l + 1) * w_batches_per_link;
                                    const uint32_t thr = (need < end_b ? need : end_b) - l * w_batches_per_link;
                                    if (needs_w_left && wleft_sem_addr[l] != 0) {
                                        noc_semaphore_wait_min(
                                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wleft_sem_addr[l]), thr);
                                    }
                                    if (needs_w_right && wright_sem_addr[l] != 0) {
                                        noc_semaphore_wait_min(
                                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wright_sem_addr[l]), thr);
                                    }
                                }
                            }
                        }
                        const uint32_t t_block_end = std::min(t_block + T_block_size, t_out_end);

                        for (uint32_t h_block = h_out_start; h_block < h_out_end; h_block += H_block_size) {
                            const uint32_t h_block_end = std::min(h_block + H_block_size, h_out_end);

                            // H rows persist across w_blocks for sliding window W reuse.
                            uint32_t h_rows_gathered = 0;
                            // halo_last: the first PROCESSED w_block in this h_block acts as is_first_w, so
                            // the W sliding window is rebuilt fresh across any phase-skipped blocks.
                            bool win_fresh = true;
                            const int32_t t_shard_start =
                                static_cast<int32_t>(t_block * stride_t) - static_cast<int32_t>(padding_t);
                            const int32_t h_shard_start =
                                static_cast<int32_t>(h_block * stride_h) - static_cast<int32_t>(padding_h);
                            const uint32_t T_shard_cur = (t_block_end - 1 - t_block) * stride_t + kT;
                            const uint32_t H_shard_cur = (h_block_end - 1 - h_block) * stride_h + kH;
                            constexpr uint32_t kW_bytes = kW * C_in_block_bytes;
                            static_assert(kW_bytes <= NOC_MAX_BURST_SIZE, "kW_bytes exceeds NOC_MAX_BURST_SIZE");

                            // Precompute T/H bounds for shard_all_in_bounds (W is per-w_block).
                            const bool th_in_bounds =
                                t_shard_start >= 0 &&
                                (t_shard_start + static_cast<int32_t>(T_shard_cur) - 1) < static_cast<int32_t>(T_in) &&
                                h_shard_start >= 0 &&
                                (h_shard_start + static_cast<int32_t>(H_shard_cur) - 1) < static_cast<int32_t>(H_in);

                            for (uint32_t w_block = w_out_start; w_block < w_out_end; w_block += W_block_size) {
                                const uint32_t w_block_end = std::min(w_block + W_block_size, w_out_end);
                                if constexpr (halo_last) {
                                    // Boundary blocks (touch a device spatial edge) need the cross-device
                                    // halo and run in phase 1; interior blocks run in phase 0. The writer
                                    // uses the same predicate to stay in lock-step.
                                    const bool is_edge =
                                        np_is_boundary_block(h_block, h_block_end, w_block, w_block_end, H_out, W_out);
                                    if ((phase == 0u) == is_edge) {
                                        win_fresh = true;
                                        continue;
                                    }
                                }
                                if constexpr (use_l1_prefetch) {
                                    const int32_t w_shard_start =
                                        static_cast<int32_t>(w_block * stride_w) - static_cast<int32_t>(padding_w);
                                    const uint32_t W_shard_cur = (w_block_end - 1 - w_block) * stride_w + kW;
                                    const bool shard_all_in_bounds =
                                        th_in_bounds && w_shard_start >= 0 &&
                                        (w_shard_start + static_cast<int32_t>(W_shard_cur) - 1) <
                                            static_cast<int32_t>(W_in);
                                    const bool coalesce_this_block = enable_coalesced_shard_reads &&
                                                                     shard_all_in_bounds &&
                                                                     W_shard_cur > NUM_DRAM_BANKS;
                                    constexpr uint32_t coalesced_scratch_offset =
                                        T_shard_max * H_shard_max_W_shard_max * C_in_block_bytes;

                                    // --- SLIDING WINDOW W + H-ROW INTERLEAVED GATHER ---
                                    const bool is_first_w = halo_last ? win_fresh : (w_block == w_out_start);
                                    constexpr uint32_t overlap_w = kW > stride_w ? kW - stride_w : 0;

                                    if (is_first_w || overlap_w == 0) {
                                        h_rows_gathered = 0;
                                    }

                                    if (!is_first_w && overlap_w > 0 && h_rows_gathered > 0) {
                                        shift_retained_w_columns<
                                            C_in_block_bytes,
                                            H_shard_max_W_shard_max,
                                            W_shard_max,
                                            kW,
                                            stride_w>(noc, shard_cb, T_shard_cur, h_rows_gathered);

                                        const uint32_t new_w_cols = W_shard_cur - overlap_w;
                                        gather_rows_selected_or_halo<
                                            C_in_block_bytes,
                                            is_padding_zeros,
                                            H_shard_max_W_shard_max,
                                            W_shard_max,
                                            T_in,
                                            H_in,
                                            W_in,
                                            H_in_W_in,
                                            in_row_size_bytes,
                                            gather_trids,
                                            enable_coalesced_shard_reads,
                                            enable_dram_read_staging,
                                            dram_read_alignment>(
                                            noc,
                                            in_reader,
                                            halo_reader,
                                            shard_cb,
                                            dram_read_scratch_cb,
                                            shard_l1_base,
                                            coalesced_scratch_offset,
                                            shard_all_in_bounds,
                                            shard_all_in_bounds && new_w_cols > NUM_DRAM_BANKS,
                                            batch_page_base,
                                            batch_idx,
                                            c_in_offset_bytes,
                                            t_shard_start,
                                            T_shard_cur,
                                            h_shard_start,
                                            0u,
                                            h_rows_gathered,
                                            w_shard_start,
                                            overlap_w,
                                            new_w_cols,
                                            coalesced_scratch_rows,
                                            h_halo_outer_dim_size,
                                            h_halo_H,
                                            h_halo_W,
                                            h_halo_padding_h,
                                            h_halo_padding_w,
                                            h_halo_hbot_base,
                                            h_halo_wleft_base,
                                            h_halo_wright_base);
                                    }

                                    ChunkWriter<cb_vol2col, padded_page_bytes, patch_pad_bytes> chunk(noc);
                                    chunk.init(num_patches);

                                    for (uint32_t t = t_block; t < t_block_end; t++) {
                                        const uint32_t t_base = (t - t_block) * stride_t;
                                        for (uint32_t h = h_block; h < h_block_end; h++) {
                                            const uint32_t h_base = (h - h_block) * stride_h;

                                            const uint32_t h_needed = h_base + kH;
                                            if (h_needed > h_rows_gathered) {
                                                gather_rows_selected_or_halo<
                                                    C_in_block_bytes,
                                                    is_padding_zeros,
                                                    H_shard_max_W_shard_max,
                                                    W_shard_max,
                                                    T_in,
                                                    H_in,
                                                    W_in,
                                                    H_in_W_in,
                                                    in_row_size_bytes,
                                                    gather_trids,
                                                    enable_coalesced_shard_reads,
                                                    enable_dram_read_staging,
                                                    dram_read_alignment>(
                                                    noc,
                                                    in_reader,
                                                    halo_reader,
                                                    shard_cb,
                                                    dram_read_scratch_cb,
                                                    shard_l1_base,
                                                    coalesced_scratch_offset,
                                                    shard_all_in_bounds,
                                                    coalesce_this_block,
                                                    batch_page_base,
                                                    batch_idx,
                                                    c_in_offset_bytes,
                                                    t_shard_start,
                                                    T_shard_cur,
                                                    h_shard_start,
                                                    h_rows_gathered,
                                                    h_needed,
                                                    w_shard_start,
                                                    0u,
                                                    W_shard_cur,
                                                    coalesced_scratch_rows,
                                                    h_halo_outer_dim_size,
                                                    h_halo_H,
                                                    h_halo_W,
                                                    h_halo_padding_h,
                                                    h_halo_padding_w,
                                                    h_halo_hbot_base,
                                                    h_halo_wleft_base,
                                                    h_halo_wright_base);
                                                h_rows_gathered = h_needed;
                                            }

                                            // Coalesced gather reorders through scratch into the same natural
                                            // shard layout, so vol2col always keeps the contiguous kW-row fast path.
                                            vol2col_shard_to_cb<
                                                kT,
                                                kH,
                                                kW,
                                                C_in_block_bytes,
                                                H_shard_max_W_shard_max,
                                                W_shard_max,
                                                stride_w,
                                                cb_vol2col,
                                                padded_page_bytes,
                                                patch_pad_bytes>(
                                                noc, shard_l1_base, t_base, h_base, w_block, w_block_end, chunk);
                                        }
                                    }
                                    chunk.flush();

                                } else {
                                    // ============================================================
                                    // DIRECT READER (for 1x1x1 or dilated kernels, no spatial reuse)
                                    // ============================================================
                                    const uint32_t t_block_s_start = t_block * stride_t;
                                    const uint32_t t_block_s_end = t_block_end * stride_t;
                                    const uint32_t h_block_s_start = h_block * stride_h;
                                    const uint32_t h_block_s_end = h_block_end * stride_h;
                                    const uint32_t w_block_s_start = w_block * stride_w;
                                    const uint32_t w_block_s_end = w_block_end * stride_w;

                                    ChunkWriter<cb_vol2col, padded_page_bytes, patch_pad_bytes> chunk(noc);
                                    chunk.init(num_patches);

                                    for (uint32_t t = t_block_s_start; t < t_block_s_end; t += stride_t) {
                                        for (uint32_t h = h_block_s_start; h < h_block_s_end; h += stride_h) {
                                            for (uint32_t w = w_block_s_start; w < w_block_s_end; w += stride_w) {
                                                for (uint32_t kt = 0; kt < kT; kt++) {
                                                    int32_t t_idx = static_cast<int32_t>(t + kt * dilation_t) -
                                                                    static_cast<int32_t>(padding_t);
                                                    const bool outside_t =
                                                        (t_idx < 0 || t_idx >= static_cast<int32_t>(T_in));
                                                    t_idx = clampIndex(t_idx, 0, static_cast<int32_t>(T_in) - 1);

                                                    for (uint32_t kh = 0; kh < kH; kh++) {
                                                        int32_t h_idx = static_cast<int32_t>(h + kh * dilation_h) -
                                                                        static_cast<int32_t>(padding_h);
                                                        const bool outside_h =
                                                            (h_idx < 0 || h_idx >= static_cast<int32_t>(H_in));
                                                        h_idx = clampIndex(h_idx, 0, static_cast<int32_t>(H_in) - 1);

                                                        for (uint32_t kw = 0; kw < kW; kw++) {
                                                            int32_t w_idx = static_cast<int32_t>(w + kw * dilation_w) -
                                                                            static_cast<int32_t>(padding_w);
                                                            const bool outside_w =
                                                                (w_idx < 0 || w_idx >= static_cast<int32_t>(W_in));
                                                            const bool in_padding = outside_t || outside_h || outside_w;
                                                            w_idx =
                                                                clampIndex(w_idx, 0, static_cast<int32_t>(W_in) - 1);

                                                            if constexpr (is_padding_zeros) {
                                                                if (in_padding) {
                                                                    zeroPad<C_in_block_bytes>(
                                                                        noc, chunk.cb, chunk.write_offset);
                                                                    chunk.write_offset += C_in_block_bytes;
                                                                    continue;
                                                                }
                                                            }

                                                            const uint32_t page_idx =
                                                                batch_page_base +
                                                                static_cast<uint32_t>(t_idx) * H_in_W_in +
                                                                static_cast<uint32_t>(h_idx) * W_in +
                                                                static_cast<uint32_t>(w_idx);
                                                            read_input_row_maybe_staged<
                                                                enable_dram_read_staging,
                                                                dram_read_alignment>(
                                                                noc,
                                                                in_reader,
                                                                page_idx,
                                                                c_in_offset_bytes,
                                                                in_row_size_bytes,
                                                                chunk.cb,
                                                                chunk.write_offset,
                                                                C_in_block_bytes,
                                                                dram_read_scratch_cb);
                                                            chunk.write_offset += C_in_block_bytes;
                                                        }
                                                    }
                                                }

                                                chunk.advance();
                                            }
                                        }
                                    }
                                    chunk.flush();
                                }
                                win_fresh = false;  // processed: the W window is now valid for the next block
                                // End of w_block
                            }
                            // End of h_block
                        }
                        // End of t_block
                    }
                }  // End of phase pass
            }
        }
    }
}
