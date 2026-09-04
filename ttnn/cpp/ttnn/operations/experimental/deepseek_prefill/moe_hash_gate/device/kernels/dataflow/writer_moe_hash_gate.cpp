// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Hash-gate writer: gather the unbiased activated scores at the (reader-produced) hash indices,
// sentinel-patch padded token rows, and write the weights + indices out. Shares gather()/scalar gen/
// sentinel helpers with moe_grouped_topk via the common header.
#include "ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/device/kernels/dataflow/moe_gate_common_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_out_weights = get_named_compile_time_arg_val("cb_out_weights");
    constexpr uint32_t cb_out_indices = get_named_compile_time_arg_val("cb_out_indices");
    constexpr uint32_t cb_sigmoid_scores = get_named_compile_time_arg_val("cb_sigmoid_scores");
    constexpr uint32_t cb_gathered_sigmoid = get_named_compile_time_arg_val("cb_gathered_sigmoid");
    constexpr uint32_t cb_reduce_ones_scalar = get_named_compile_time_arg_val("cb_reduce_ones_scalar");
    constexpr uint32_t cb_epsilon_scalar = get_named_compile_time_arg_val("cb_epsilon_scalar");
    constexpr uint32_t cb_route_scale_scalar = get_named_compile_time_arg_val("cb_route_scale_scalar");
    constexpr uint32_t cb_padding_config = get_named_compile_time_arg_val("cb_padding_config");

    constexpr uint32_t scores_page_size = get_named_compile_time_arg_val("scores_page_size");
    constexpr uint32_t weights_page_size = get_named_compile_time_arg_val("weights_page_size");
    constexpr uint32_t indices_page_size = get_named_compile_time_arg_val("indices_page_size");
    constexpr uint32_t experts = get_named_compile_time_arg_val("experts");
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr uint32_t tile_height = get_named_compile_time_arg_val("tile_height");
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t n_activated_experts = get_named_compile_time_arg_val("n_activated_experts");
    constexpr uint32_t n_activated_expert_tiles = get_named_compile_time_arg_val("n_activated_expert_tiles");
    constexpr uint32_t packed_one_scalar = get_named_compile_time_arg_val("packed_one_scalar");
    constexpr uint32_t packed_epsilon = get_named_compile_time_arg_val("packed_epsilon");
    constexpr uint32_t packed_route_scale = get_named_compile_time_arg_val("packed_route_scale");
    constexpr uint32_t seq_len_tiles = get_named_compile_time_arg_val("seq_len_tiles");
    constexpr uint32_t remainder_tokens_per_tile = get_named_compile_time_arg_val("remainder_tokens_per_tile");

    const uint32_t weights_addr = get_arg_val<uint32_t>(0);
    const uint32_t indices_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_height_tile = get_arg_val<uint32_t>(2);
    const uint32_t end_height_tile = get_arg_val<uint32_t>(3);
    const uint32_t padding_config_addr = get_arg_val<uint32_t>(4);

    constexpr auto weights_args = TensorAccessorArgs<0>();
    constexpr auto indices_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    constexpr auto padding_config_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();

    const auto weights_accessor = TensorAccessor(weights_args, weights_addr, weights_page_size);
    const auto indices_accessor = TensorAccessor(indices_args, indices_addr, indices_page_size);
    const auto padding_config_accessor = TensorAccessor(padding_config_args, padding_config_addr);

    // Optional padding awareness: when a per-device [num_real_tokens, pad_side] row is supplied
    // (addr != 0), padded token rows have their output expert indices overwritten with the sentinel
    // (= experts), so downstream masked_bincount / dispatch / combine skip them.
    uint32_t num_real_tokens = 0xFFFFFFFF;  // default: no padding -> every row is real
    uint32_t pad_side = 0;
    Noc noc;
    if (padding_config_addr != 0) {
        CircularBuffer padding_config_cb(cb_padding_config);
        padding_config_cb.reserve_back(1);
        const uint32_t padding_config_l1_addr = padding_config_cb.get_write_ptr();
        noc.async_read(
            padding_config_accessor,
            padding_config_cb,
            padding_config_cb.get_tile_size(),
            {.page_id = 0},
            {.offset_bytes = 0});
        noc.async_read_barrier();

        volatile tt_l1_ptr uint32_t* padding_config_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(padding_config_l1_addr);
        num_real_tokens = padding_config_ptr[0];
        pad_side = padding_config_ptr[1];
    }

    CircularBuffer out_indices_cb(cb_out_indices);
    CircularBuffer out_weights_cb(cb_out_weights);

    // Scalars consumed by normalize_scores / scale on the compute kernel.
    generate_reduce_scalar(cb_reduce_ones_scalar, packed_one_scalar, n_activated_experts);
    write_single_scalar(cb_epsilon_scalar, packed_epsilon);
    write_single_scalar(cb_route_scale_scalar, packed_route_scale);

    for (uint32_t height_tile = start_height_tile; height_tile < end_height_tile; height_tile++) {
        // Use remainder_tokens_per_tile only for the LAST tile of the sequence, otherwise full tile_height.
        uint32_t tokens_per_tile = ((height_tile + 1) % seq_len_tiles == 0) ? remainder_tokens_per_tile : tile_height;

        out_indices_cb.wait_front(1);

        // Gather the unbiased activated scores at the hash indices produced by the reader.
        gather<
            cb_out_indices,
            cb_sigmoid_scores,
            cb_gathered_sigmoid,
            width_tiles,
            n_activated_experts,
            n_activated_expert_tiles>(tokens_per_tile);

        // Overwrite indices for padded token rows with SENTINEL = experts (num_routed_experts).
        if (num_real_tokens < tokens) {
            constexpr uint16_t sentinel = static_cast<uint16_t>(experts);
            uint32_t tile_start_row = height_tile * tile_height;
            volatile tt_l1_ptr uint16_t* idx_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out_indices_cb.get_read_ptr());

            for (uint32_t row = 0; row < tokens_per_tile; row++) {
                uint32_t global_row = tile_start_row + row;
                bool is_padded = (pad_side == 0) ? (global_row >= num_real_tokens)           // right-pad
                                                 : (global_row < tokens - num_real_tokens);  // left-pad
                if (is_padded) {
                    overwrite_index_row_with_sentinel(idx_ptr, row, n_activated_experts, sentinel);
                }
            }
        }

        noc.async_write(out_indices_cb, indices_accessor, indices_page_size, {}, {.page_id = height_tile});
        out_weights_cb.wait_front(1);
        noc.async_write(out_weights_cb, weights_accessor, weights_page_size, {}, {.page_id = height_tile});
        noc.async_writes_flushed();
        out_indices_cb.pop_front(1);
        out_weights_cb.pop_front(1);
    }
    noc.async_write_barrier();
}
