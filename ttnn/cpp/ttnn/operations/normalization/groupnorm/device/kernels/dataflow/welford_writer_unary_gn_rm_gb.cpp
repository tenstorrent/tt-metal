// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"
#include "experimental/tensor.h"

void kernel_main() {
    constexpr bool is_mcast_sender = get_named_compile_time_arg_val("is_mcast_sender") == 1;
    constexpr bool fuse_gamma = get_named_compile_time_arg_val("fuse_gamma") == 1;
    constexpr bool fuse_beta = get_named_compile_time_arg_val("fuse_beta") == 1;

    constexpr uint32_t num_cols_tile_gamma_beta = get_named_compile_time_arg_val("num_cols_tile_gamma_beta");

    constexpr uint32_t per_core_M = get_named_compile_time_arg_val("per_core_M");
    constexpr uint32_t per_core_N = get_named_compile_time_arg_val("per_core_N");
    constexpr uint32_t per_core_N_bytes = get_named_compile_time_arg_val("per_core_N_bytes");
    constexpr uint32_t per_core_N_bytes_with_stride = get_named_compile_time_arg_val("per_core_N_bytes_with_stride");

    constexpr uint32_t num_groups_per_core = get_named_compile_time_arg_val("num_groups_per_core");
    constexpr uint32_t num_batches_per_core = get_named_compile_time_arg_val("num_batches_per_core");

    constexpr uint32_t num_cols_per_group = get_named_compile_time_arg_val("num_cols_per_group");
    constexpr uint32_t num_tiles_per_batch = get_named_compile_time_arg_val("num_tiles_per_batch");

    constexpr uint32_t block_w_last = get_named_compile_time_arg_val("block_w_last");
    constexpr uint32_t GROUP_SIZE_IS_POWER_OF_2 = get_named_compile_time_arg_val("GROUP_SIZE_IS_POWER_OF_2");
    constexpr uint32_t GROUP_SIZE_SMALLER_THAN_TILE_W =
        get_named_compile_time_arg_val("GROUP_SIZE_SMALLER_THAN_TILE_W");
    constexpr uint32_t group_row_offset = get_named_compile_time_arg_val("group_row_offset");
    constexpr uint32_t num_out_blocks = get_named_compile_time_arg_val("num_out_blocks");

    constexpr uint32_t block_h = get_named_compile_time_arg_val("block_h");
    constexpr uint32_t block_w = get_named_compile_time_arg_val("block_w");
    constexpr uint32_t block_hw = get_named_compile_time_arg_val("block_hw");

    constexpr uint32_t use_welford = get_named_compile_time_arg_val("groupnorm_mode") > 0;
    constexpr uint32_t page_size = get_named_compile_time_arg_val("page_size");

    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto gamma_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    constexpr auto input_mask_args = TensorAccessorArgs<beta_args.next_compile_time_args_offset()>();

    constexpr uint32_t tile_width = get_named_compile_time_arg_val("TILE_WIDTH");

    constexpr uint32_t block_w_minus_one = block_w - 1;
    constexpr uint32_t block_w_minus_two = block_w - 2;
    constexpr uint32_t tile_w_minux_group_size = tile_width - num_cols_per_group;

    const uint32_t eps_val = get_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_arg_val<uint32_t>(3);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(4);
    const uint32_t beta_addr = get_arg_val<uint32_t>(5);
    const uint32_t input_mask_addr = get_arg_val<uint32_t>(6);
    const uint32_t out_start_id = get_arg_val<uint32_t>(7);
    const uint32_t gamma_tile_start_id = get_arg_val<uint32_t>(8);
    const uint32_t beta_tile_start_id = get_arg_val<uint32_t>(9);
    const uint32_t input_mask_tile_start_id = get_arg_val<uint32_t>(10);
    const uint32_t num_channels_tiles = get_arg_val<uint32_t>(11);

    constexpr uint32_t cb_eps_id = tt::CBIndex::c_3;
    constexpr uint32_t cb_gamma_id = tt::CBIndex::c_5;
    constexpr uint32_t cb_beta_id = tt::CBIndex::c_6;
    constexpr uint32_t cb_input_mask_id = tt::CBIndex::c_28;
    constexpr uint32_t cb_in_id = tt::CBIndex::c_29;

    constexpr uint32_t cb_reread_write_out_id = tt::CBIndex::c_22;
    constexpr uint32_t cb_out0_id = tt::CBIndex::c_16;
#ifdef UNTILIZE_OUT
    constexpr uint32_t cb_out_id = tt::CBIndex::c_30;
#else
    constexpr uint32_t cb_out_id = (fuse_gamma or fuse_beta) ? cb_out0_id : cb_reread_write_out_id;
#endif

    experimental::Noc noc;
    experimental::CircularBuffer cb_input_mask(cb_input_mask_id);
    experimental::CircularBuffer cb_gamma(cb_gamma_id);
    experimental::CircularBuffer cb_beta(cb_beta_id);
    experimental::CircularBuffer cb_out(cb_out_id);

    constexpr uint32_t single_tile_size_bytes = get_tile_size(cb_out_id);
    constexpr uint32_t input_mask_single_tile_size_bytes = get_tile_size(cb_input_mask_id);

    const auto mask = TensorAccessor(input_mask_args, input_mask_addr, input_mask_single_tile_size_bytes);

    constexpr uint32_t out_block_h_normal = block_h / num_out_blocks;
    constexpr uint32_t out_block_hw_normal = out_block_h_normal * block_w;
    uint32_t num_out_blocks_padded = num_out_blocks;
    uint32_t extra_out_block = false;
    uint32_t out_block_h_last = out_block_h_normal;
    uint32_t out_block_hw_last = out_block_hw_normal;
    if constexpr (block_h % num_out_blocks != 0) {
        extra_out_block = true;
        uint32_t residual = block_h - (num_out_blocks * out_block_h_normal);
        num_out_blocks_padded += (residual / out_block_h_normal + 1);
        out_block_h_last = residual % out_block_h_normal;
        out_block_hw_last = out_block_h_last * block_w;
    }

    // Send eps, mask, gamma, and beta to the compute kernel
    generate_bcast_col_scalar(cb_eps_id, eps_val);

    cb_input_mask.reserve_back(block_w * num_groups_per_core);
    uint32_t l1_write_addr_input_mask = cb_input_mask.get_write_ptr();
    uint32_t input_mask_tile_id = input_mask_tile_start_id;
    for (uint32_t i = 0; i < num_groups_per_core; ++i) {
        for (uint32_t j = 0; j < block_w; ++j) {
            noc.async_read(
                mask,
                experimental::CoreLocalMem<uint32_t>(l1_write_addr_input_mask),
                input_mask_single_tile_size_bytes,
                {.page_id = input_mask_tile_id},
                {});
            input_mask_tile_id += 1;
            noc.async_read_barrier();
            l1_write_addr_input_mask += input_mask_single_tile_size_bytes;
        }
    }
    cb_input_mask.push_back(block_w * num_groups_per_core);

    if constexpr (fuse_gamma) {
        constexpr uint32_t gamma_tile_bytes = get_tile_size(cb_gamma_id);
        const auto gamma = TensorAccessor(gamma_args, gamma_addr, page_size);

        cb_gamma.reserve_back(num_cols_tile_gamma_beta);

        const uint32_t base_l1_write_addr_gamma = cb_gamma.get_write_ptr();
        uint32_t l1_write_addr_gamma = base_l1_write_addr_gamma;

        // We want this data to appear as the first row of the tile.
        // This is 32B at the start of the first face, 32B at the start of the second face
        // However we must read at a 64 byte granularity for Blackhole NOC compatibility on DRAM reads
        // So instead of two 32B reads to the correct addresses, we read 64 bytes into the first face here
        // Then later, copy the second set of 32 bytes into the start of the second face
        // L1-L1 NOC transactions only need 16 byte alignment on BH, so this is legal after data is loaded
        // to L1

        // Read the first 64 bytes of the tile into the first face
        for (uint32_t w = 0; w < num_cols_tile_gamma_beta; w++) {
            uint32_t tile_id = gamma_tile_start_id + w;
            noc.async_read(
                gamma, experimental::CoreLocalMem<uint32_t>(l1_write_addr_gamma), 64, {.page_id = tile_id}, {});
            l1_write_addr_gamma += gamma_tile_bytes;
        }
        noc.async_read_barrier();

        // Copy the second set of 32 bytes into the second face
        l1_write_addr_gamma = base_l1_write_addr_gamma;

        experimental::UnicastEndpoint self_ep_gamma;
        for (uint32_t w = 0; w < num_cols_tile_gamma_beta; w++) {
            noc.async_read(
                self_ep_gamma,
                experimental::CoreLocalMem<uint32_t>(l1_write_addr_gamma + 512),
                32,
                {.noc_x = my_x[0], .noc_y = my_y[0], .addr = l1_write_addr_gamma + 32},
                {});
            l1_write_addr_gamma += gamma_tile_bytes;
        }

        noc.async_read_barrier();
        cb_gamma.push_back(num_cols_tile_gamma_beta);
    }

    if constexpr (fuse_beta) {
        // Just like gamma, we read at a 64 byte granularity for Blackhole NOC compatibility
        // Then copy the second set of 32 bytes into the second face
        constexpr uint32_t beta_tile_bytes = get_tile_size(cb_beta_id);
        const auto beta = TensorAccessor(beta_args, beta_addr, page_size);

        cb_beta.reserve_back(num_cols_tile_gamma_beta);

        const uint32_t base_l1_write_addr_beta = cb_beta.get_write_ptr();
        uint32_t l1_write_addr_beta = base_l1_write_addr_beta;

        // Read the first 64 bytes of the tile into the first face
        for (uint32_t w = 0; w < num_cols_tile_gamma_beta; w++) {
            uint32_t tile_id = beta_tile_start_id + w;
            noc.async_read(
                beta, experimental::CoreLocalMem<uint32_t>(l1_write_addr_beta), 64, {.page_id = tile_id}, {});
            l1_write_addr_beta += beta_tile_bytes;
        }
        noc.async_read_barrier();

        // Copy the second set of 32 bytes into the second face
        l1_write_addr_beta = base_l1_write_addr_beta;

        experimental::UnicastEndpoint self_ep_beta;
        for (uint32_t w = 0; w < num_cols_tile_gamma_beta; w++) {
            noc.async_read(
                self_ep_beta,
                experimental::CoreLocalMem<uint32_t>(l1_write_addr_beta + 512),
                32,
                {.noc_x = my_x[0], .noc_y = my_y[0], .addr = l1_write_addr_beta + 32},
                {});
            l1_write_addr_beta += beta_tile_bytes;
        }

        noc.async_read_barrier();
        cb_beta.push_back(num_cols_tile_gamma_beta);
    }

    const auto dst_a = TensorAccessor(out_args, out_addr, single_tile_size_bytes);

    uint32_t index_b_offset = 0;
    for (uint32_t b = 0; b < num_batches_per_core; ++b) {
        uint32_t out_block_index_offset = 0;
        for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
            uint32_t out_block_h_actual, out_block_hw_actual;
            if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                out_block_h_actual = out_block_h_last;
                out_block_hw_actual = out_block_hw_last;
            } else {
                out_block_h_actual = out_block_h_normal;
                out_block_hw_actual = out_block_hw_normal;
            }

            uint32_t mt_offset = 0;
            for (uint32_t mt = 0; mt < out_block_h_actual; mt++) {
                for (uint32_t nt = 0; nt < per_core_N; ++nt) {
                    cb_out.wait_front(1);
                    const uint32_t l1_read_addr = cb_out.get_read_ptr();
                    noc.async_write(
                        experimental::CoreLocalMem<uint32_t>(l1_read_addr),
                        dst_a,
                        single_tile_size_bytes,
                        {},
                        {.page_id = out_start_id + index_b_offset + out_block_index_offset + mt_offset + nt});
                    noc.async_write_barrier();
                    cb_out.pop_front(1);
                }
                mt_offset += num_channels_tiles;
            }
            out_block_index_offset += out_block_h_actual * num_channels_tiles;
        }
        index_b_offset += num_tiles_per_batch;
    }
}
