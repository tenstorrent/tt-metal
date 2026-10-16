// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#if defined(ARCH_BLACKHOLE)
#include "ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/device/kernels/dataflow/moe_grouped_topk_gather.hpp"
#endif

void kernel_main() {
    constexpr uint32_t cb_in_scores_id = get_named_compile_time_arg_val("cb_in_scores");
    constexpr uint32_t cb_in_bias_id = get_named_compile_time_arg_val("cb_in_bias");
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr uint32_t scores_page_size = get_named_compile_time_arg_val("scores_page_size");
    constexpr uint32_t bias_page_size = get_named_compile_time_arg_val("bias_page_size");

    // Get scores and bias tensor accessors
    constexpr auto scores_args = TensorAccessorArgs<0>();
    constexpr auto bias_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();

    const uint32_t scores_addr = get_arg_val<uint32_t>(0);
    const uint32_t bias_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_height_tile = get_arg_val<uint32_t>(2);
    const uint32_t end_height_tile = get_arg_val<uint32_t>(3);

    const auto scores_accessor = TensorAccessor(scores_args, scores_addr, scores_page_size);
    const auto bias_accessor = TensorAccessor(bias_args, bias_addr, bias_page_size);

    Noc noc;
    CircularBuffer cb_in_scores(cb_in_scores_id);
    CircularBuffer cb_in_bias(cb_in_bias_id);
    const uint32_t scores_tile_bytes = cb_in_scores.get_tile_size();
    const uint32_t bias_tile_bytes = cb_in_bias.get_tile_size();

#if defined(ARCH_BLACKHOLE)
    constexpr uint32_t cb_helper_job_id = get_named_compile_time_arg_val("cb_helper_job");
    constexpr uint32_t cb_helper_done_id = get_named_compile_time_arg_val("cb_helper_done");
    constexpr uint32_t index_u16_mask = get_named_compile_time_arg_val("index_u16_mask");
    constexpr uint32_t n_groups = get_named_compile_time_arg_val("n_groups");
    constexpr uint32_t topk_groups = get_named_compile_time_arg_val("topk_groups");
    constexpr uint32_t n_activated_experts = get_named_compile_time_arg_val("n_activated_experts");
    // The writer's choice: slot positions in the grouped path when no sort compares index words.
    constexpr bool slot_indices = n_groups != 1 && (index_u16_mask & 0x4) != 0;
    // Transaction ids 1 to 15: a score tile waits only for itself while the first 8 are in flight; the bias goes
    // after the activation, so its tiles share ids.
    constexpr uint32_t score_trids = 8;
    constexpr uint32_t bias_trids = 7;

    // All of a row's reads are issued at once, scores first; each tile is pushed when its transaction id lands, so the
    // DRAM latency is paid once a row and compute starts on the first tile.
    auto read_row = [&](uint32_t height_tile) {
        const uint32_t base_page = height_tile * width_tiles;
        cb_in_scores.reserve_back(width_tiles);
        cb_in_bias.reserve_back(width_tiles);
        for (uint32_t width_tile = 0; width_tile < width_tiles; width_tile++) {
            noc_async_read_set_trid(1 + width_tile % score_trids);
            noc.async_read(
                scores_accessor,
                cb_in_scores,
                scores_tile_bytes,
                {.page_id = base_page + width_tile},
                {.offset_bytes = width_tile * scores_tile_bytes});
        }
        for (uint32_t width_tile = 0; width_tile < width_tiles; width_tile++) {
            noc_async_read_barrier_with_trid(1 + width_tile % score_trids);
            cb_in_scores.push_back(1);
        }
        // The bias is needed only after the activation, so its reads do not share DRAM with the logits'.
        for (uint32_t width_tile = 0; width_tile < width_tiles; width_tile++) {
            noc_async_read_set_trid(1 + score_trids + width_tile % bias_trids);
            noc.async_read(
                bias_accessor,
                cb_in_bias,
                bias_tile_bytes,
                {.page_id = base_page + width_tile},
                {.offset_bytes = width_tile * bias_tile_bytes});
        }
        noc_async_read_set_trid(0);
        for (uint32_t width_tile = 0; width_tile < width_tiles; width_tile++) {
            noc_async_read_barrier_with_trid(1 + score_trids + width_tile % bias_trids);
            cb_in_bias.push_back(1);
        }
    };
    {
        // The reader also takes half of the writer's gathers, from jobs the writer posts.
        using gate_gather::tile_height_max;
        CircularBuffer job_cb(cb_helper_job_id);
        CircularBuffer done_cb(cb_helper_done_id);
        constexpr bool by_id = get_named_compile_time_arg_val("winning_groups_by_id") != 0;
        // Each row's groups in each slot, kept from the slot job for the gather job (two rows in flight).
        alignas(4) uint16_t slot_groups[2][slot_indices ? topk_groups * tile_height_max : 2];
        auto take_job = [&]() -> volatile tt_l1_ptr uint32_t* {
            job_cb.wait_front(1);
            return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(job_cb.get_read_ptr());
        };
        // The writer's odd winning-group slots.
        auto serve_slots = [&](uint16_t* groups) {
            volatile tt_l1_ptr uint32_t* job = take_job();
            const uint32_t tokens = job[gate_gather::TOKENS];
            const uint32_t src_base = job[gate_gather::SRC];
            const uint32_t dst_base = job[gate_gather::DST];
            const uint32_t order = job[gate_gather::ORDER];
            job_cb.pop_front(1);
            asm volatile("" ::: "memory");
            gate_gather::slot_groups_from_order<topk_groups, by_id>(
                reinterpret_cast<const uint16_t*>(order), tokens, groups);
            noc_async_read_one_packet_set_state(get_noc_addr(src_base), score_tile::face_line_bytes);
            for (uint32_t k = 1; k < topk_groups; k += 2) {
                gate_gather::read_slot_rows(
                    src_base, dst_base + k * score_tile::tile_size_bytes, groups + k * tile_height_max, tokens);
                noc_async_read_barrier();
                done_cb.reserve_back(1);
                done_cb.push_back(1);
            }
        };
        // The second half of the tokens of the selected experts' unbiased scores.
        auto serve_gather = [&](const uint16_t* groups) {
            volatile tt_l1_ptr uint32_t* job = take_job();
            const uint32_t tokens = job[gate_gather::TOKENS];
            uint32_t* idx = reinterpret_cast<uint32_t*>(job[gate_gather::SRC]);
            const uint32_t* sig = reinterpret_cast<const uint32_t*>(job[gate_gather::DST]);
            uint32_t* out = reinterpret_cast<uint32_t*>(job[gate_gather::OUT]);
            const uint32_t first = job[gate_gather::FIRST_TOKEN];
            job_cb.pop_front(1);
            asm volatile("" ::: "memory");
            gate_gather::gather_rows<n_activated_experts, slot_indices>(first, tokens, idx, sig, out, groups);
            done_cb.reserve_back(1);
            done_cb.push_back(1);
        };
        // Rows go in the compute kernel's pairs. A row's reads wait for the compute to take the row two before, so
        // each goes out just before the job that the compute's next step waits on.
        for (uint32_t height_tile = start_height_tile;
             height_tile < end_height_tile && height_tile < start_height_tile + 2;
             height_tile++) {
            read_row(height_tile);
        }
        for (uint32_t height_tile = start_height_tile; height_tile < end_height_tile; height_tile += 2) {
            const uint32_t rows = (height_tile + 1 < end_height_tile) ? 2 : 1;
            for (uint32_t row = 0; row < rows; row++) {
                if (height_tile + 2 + row < end_height_tile) {
                    read_row(height_tile + 2 + row);
                }
                if constexpr (slot_indices) {
                    serve_slots(slot_groups[row]);
                }
            }
            for (uint32_t row = 0; row < rows; row++) {
                serve_gather(slot_groups[row]);
            }
        }
    }
#else
    for (uint32_t height_tile = start_height_tile; height_tile < end_height_tile; height_tile++) {
        uint32_t base_page = height_tile * width_tiles;
        for (uint32_t width_tile = 0; width_tile < width_tiles; width_tile++) {
            uint32_t page = base_page + width_tile;
            cb_in_scores.reserve_back(1);
            cb_in_bias.reserve_back(1);
            noc.async_read(scores_accessor, cb_in_scores, scores_tile_bytes, {.page_id = page}, {.offset_bytes = 0});
            noc.async_read(bias_accessor, cb_in_bias, bias_tile_bytes, {.page_id = page}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_in_scores.push_back(1);
            cb_in_bias.push_back(1);
        }
    }
#endif
}
