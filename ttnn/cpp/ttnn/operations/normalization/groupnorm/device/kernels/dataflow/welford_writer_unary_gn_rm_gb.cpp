// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "tt-metalium/constants.hpp"
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"

// Queue the DRAM read of a gamma/beta row (TILE_WIDTH datums) into face 0; byte offsets scale with
// datum size (2B bf16 / 4B fp32). Full row goes to face 0 (Blackhole DRAM reads need 64B granularity).
template <typename AccessorType>
void async_read_row_face0(
    const Noc& noc, const AccessorType& accessor, uint32_t page_id, uint32_t l1_dst_addr, uint32_t element_bytes) {
    const uint32_t row_bytes = tt::constants::TILE_WIDTH * element_bytes;
    noc.async_read(accessor, CoreLocalMem<uint32_t>(l1_dst_addr), row_bytes, {.page_id = page_id}, {});
}

// L1->L1-copy the row's second half into face 1. Only legal after the face-0 read barrier.
void copy_row_half_to_face1(const Noc& noc, uint32_t l1_dst_addr, uint32_t element_bytes) {
    const uint32_t face_bytes = tt::constants::FACE_HW * element_bytes;
    const uint32_t half_row_bytes = tt::constants::FACE_WIDTH * element_bytes;
    UnicastEndpoint self;
    noc.async_read(
        self,
        CoreLocalMem<uint32_t>(l1_dst_addr + face_bytes),
        half_row_bytes,
        {.noc_x = my_x[0], .noc_y = my_y[0], .addr = l1_dst_addr + half_row_bytes},
        {});
}

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

    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto gamma_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    constexpr auto input_mask_args = TensorAccessorArgs<beta_args.next_compile_time_args_offset()>();

    constexpr uint32_t tile_width = get_named_compile_time_arg_val("TILE_WIDTH");

    constexpr uint32_t block_w_minus_one = block_w - 1;
    constexpr uint32_t block_w_minus_two = block_w - 2;
    constexpr uint32_t tile_w_minux_group_size = tile_width - num_cols_per_group;

    const uint32_t eps_val = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(2);
    const uint32_t beta_addr = get_arg_val<uint32_t>(3);
    const uint32_t input_mask_addr = get_arg_val<uint32_t>(4);
    const uint32_t out_start_id = get_arg_val<uint32_t>(5);
    const uint32_t gamma_tile_start_id = get_arg_val<uint32_t>(6);
    const uint32_t beta_tile_start_id = get_arg_val<uint32_t>(7);
    const uint32_t input_mask_tile_start_id = get_arg_val<uint32_t>(8);
    const uint32_t num_channels_tiles = get_arg_val<uint32_t>(9);

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

    Noc noc;
    CircularBuffer cb_input_mask(cb_input_mask_id);
    CircularBuffer cb_gamma(cb_gamma_id);
    CircularBuffer cb_beta(cb_beta_id);
    CircularBuffer cb_out(cb_out_id);

    constexpr uint32_t single_tile_size_bytes = get_tile_size(cb_out_id);
    constexpr uint32_t input_mask_single_tile_size_bytes = get_tile_size(cb_input_mask_id);

    const auto mask = TensorAccessor(input_mask_args, input_mask_addr);

    constexpr uint32_t out_block_h_normal = block_h / num_out_blocks;
    uint32_t num_out_blocks_padded = num_out_blocks;
    uint32_t extra_out_block = false;
    uint32_t out_block_h_last = out_block_h_normal;
    if constexpr (block_h % num_out_blocks != 0) {
        extra_out_block = true;
        uint32_t residual = block_h - (num_out_blocks * out_block_h_normal);
        num_out_blocks_padded += (residual / out_block_h_normal + 1);
        out_block_h_last = residual % out_block_h_normal;
    }

    // Send eps, mask, gamma, and beta to the compute kernel
    generate_bcast_col_scalar(CircularBuffer(cb_eps_id), eps_val);

    cb_input_mask.reserve_back(block_w * num_groups_per_core);
    uint32_t l1_write_addr_input_mask = cb_input_mask.get_write_ptr();
    uint32_t input_mask_tile_id = input_mask_tile_start_id;
    for (uint32_t i = 0; i < num_groups_per_core; ++i) {
        for (uint32_t j = 0; j < block_w; ++j) {
            noc.async_read(
                mask,
                CoreLocalMem<uint32_t>(l1_write_addr_input_mask),
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
        constexpr uint32_t gamma_element_bytes = gamma_tile_bytes / tt::constants::TILE_HW;
        const auto gamma = TensorAccessor(gamma_args, gamma_addr);

        cb_gamma.reserve_back(num_cols_tile_gamma_beta);
        const uint32_t base_l1_write_addr_gamma = cb_gamma.get_write_ptr();

        uint32_t l1_write_addr_gamma = base_l1_write_addr_gamma;
        for (uint32_t w = 0; w < num_cols_tile_gamma_beta; w++) {
            async_read_row_face0(noc, gamma, gamma_tile_start_id + w, l1_write_addr_gamma, gamma_element_bytes);
            l1_write_addr_gamma += gamma_tile_bytes;
        }
        noc.async_read_barrier();

        l1_write_addr_gamma = base_l1_write_addr_gamma;
        for (uint32_t w = 0; w < num_cols_tile_gamma_beta; w++) {
            copy_row_half_to_face1(noc, l1_write_addr_gamma, gamma_element_bytes);
            l1_write_addr_gamma += gamma_tile_bytes;
        }
        noc.async_read_barrier();
        cb_gamma.push_back(num_cols_tile_gamma_beta);
    }

    if constexpr (fuse_beta) {
        constexpr uint32_t beta_tile_bytes = get_tile_size(cb_beta_id);
        constexpr uint32_t beta_element_bytes = beta_tile_bytes / tt::constants::TILE_HW;
        const auto beta = TensorAccessor(beta_args, beta_addr);

        cb_beta.reserve_back(num_cols_tile_gamma_beta);
        const uint32_t base_l1_write_addr_beta = cb_beta.get_write_ptr();

        uint32_t l1_write_addr_beta = base_l1_write_addr_beta;
        for (uint32_t w = 0; w < num_cols_tile_gamma_beta; w++) {
            async_read_row_face0(noc, beta, beta_tile_start_id + w, l1_write_addr_beta, beta_element_bytes);
            l1_write_addr_beta += beta_tile_bytes;
        }
        noc.async_read_barrier();

        l1_write_addr_beta = base_l1_write_addr_beta;
        for (uint32_t w = 0; w < num_cols_tile_gamma_beta; w++) {
            copy_row_half_to_face1(noc, l1_write_addr_beta, beta_element_bytes);
            l1_write_addr_beta += beta_tile_bytes;
        }
        noc.async_read_barrier();
        cb_beta.push_back(num_cols_tile_gamma_beta);
    }

    const auto dst_a = TensorAccessor(out_args, out_addr);

    uint32_t index_b_offset = 0;
    for (uint32_t b = 0; b < num_batches_per_core; ++b) {
        uint32_t out_block_index_offset = 0;
        for (uint32_t out_block_index = 0; out_block_index < num_out_blocks_padded; out_block_index++) {
            uint32_t out_block_h_actual;
            if (extra_out_block && (out_block_index == (num_out_blocks_padded - 1))) {
                out_block_h_actual = out_block_h_last;
            } else {
                out_block_h_actual = out_block_h_normal;
            }

            uint32_t mt_offset = 0;
            for (uint32_t mt = 0; mt < out_block_h_actual; mt++) {
                for (uint32_t nt = 0; nt < per_core_N; ++nt) {
                    cb_out.wait_front(1);
                    const uint32_t l1_read_addr = cb_out.get_read_ptr();
                    noc.async_write(
                        CoreLocalMem<uint32_t>(l1_read_addr),
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
