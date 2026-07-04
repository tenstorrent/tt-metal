// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "tt-metalium/constants.hpp"
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#if defined(MASK_SYNTHESIZE) || defined(NEGATIVE_MASK_SYNTHESIZE)
#include "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/groupnorm_mask_synthesize.hpp"
#endif

void generate_tile_with_packed_bfloat16_values(uint32_t cb_id, uint32_t packed_bf16_value) {
    CircularBuffer cb(cb_id);
    cb.reserve_back(1);
    CoreLocalMem<uint32_t> ptr(cb.get_write_ptr());
    for (uint32_t i = 0; i < 512U; ++i) {
        *ptr++ = packed_bf16_value;
    }
    cb.push_back(1);
}

void kernel_main() {
    constexpr uint32_t TILE_HW = TILE_HW_VAL;
    constexpr bool is_mcast_sender = get_compile_time_arg_val(0) == 1;
    constexpr bool fuse_gamma = get_compile_time_arg_val(1) == 1;
    constexpr bool fuse_beta = get_compile_time_arg_val(2) == 1;

    // Used only if negative mask is passed in kernel, i.e. if define FUSE_NEGATIVE_MASK is defined
    constexpr uint32_t num_cols_tile_gamma_beta = get_compile_time_arg_val(3);

    constexpr uint32_t per_core_N = get_compile_time_arg_val(4);
    constexpr uint32_t per_core_N_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t per_core_N_bytes_with_stride = get_compile_time_arg_val(6);

    constexpr uint32_t num_groups_per_core = get_compile_time_arg_val(7);
    constexpr uint32_t num_batches_per_core = get_compile_time_arg_val(8);
    constexpr uint32_t block_w = get_compile_time_arg_val(9);

    // compile_time_arg 10: size (unused here)
    constexpr uint32_t reduce_factor_w = get_compile_time_arg_val(11);
    constexpr uint32_t reduce_factor_c = get_compile_time_arg_val(12);

    constexpr auto gamma_args = TensorAccessorArgs<13>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    constexpr auto input_mask_args = TensorAccessorArgs<beta_args.next_compile_time_args_offset()>();

    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr = get_arg_val<uint32_t>(2);
    const uint32_t input_mask_addr = get_arg_val<uint32_t>(3);

    // Used only if negative mask is passed in kernel, i.e. if define FUSE_NEGATIVE_MASK is defined
    const uint32_t input_negative_mask_addr = get_arg_val<uint32_t>(4);
    const uint32_t gamma_tile_start_id = get_arg_val<uint32_t>(5);
    const uint32_t beta_tile_start_id = get_arg_val<uint32_t>(6);
    const uint32_t input_mask_tile_start_id = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_gamma_id = tt::CBIndex::c_5;
    constexpr uint32_t cb_beta_id = tt::CBIndex::c_6;
    constexpr uint32_t cb_out0_id = tt::CBIndex::c_16;
    constexpr uint32_t cb_input_mask_id = tt::CBIndex::c_7;
    constexpr uint32_t cb_ones_id = tt::CBIndex::c_26;

    Noc noc;
    CircularBuffer cb_gamma(cb_gamma_id);
    CircularBuffer cb_beta(cb_beta_id);
    CircularBuffer cb_input_mask(cb_input_mask_id);

    const uint32_t single_tile_size_bytes = get_tile_size(cb_gamma_id);
    const uint32_t input_mask_single_tile_size_bytes = get_tile_size(cb_input_mask_id);

    const auto mask = TensorAccessor(input_mask_args, input_mask_addr);

#if defined(MASK_PARTIAL_READ)
    // Logical row 0 of the source TILE-layout mask tensor lands at tile byte
    // offsets [0, face_w_bytes) and [face_bytes, face_bytes + face_w_bytes)
    // — i.e. face 0 row 0 and face 1 row 0. The unpacker for
    // BroadcastType::ROW reads only these 64 bytes per tile (bf16), so we can
    // skip fetching the other 1984 bytes from DRAM. Gated on a non-block-float
    // mask dtype by the program factory; with BFP* masks the per-face shared
    // exponent block changes this layout and the full-tile read path is used.
    const uint32_t input_mask_element_bytes = input_mask_single_tile_size_bytes / TILE_HW;
    const uint32_t input_mask_face_bytes = input_mask_element_bytes * tt::constants::FACE_HW;
    const uint32_t input_mask_face_w_bytes = input_mask_element_bytes * tt::constants::FACE_WIDTH;
#endif

#if defined(FUSE_NEGATIVE_MASK)
    constexpr uint32_t cb_input_negative_mask_id = tt::CBIndex::c_14;
    const uint32_t input_negative_mask_single_tile_size_bytes = get_tile_size(cb_input_negative_mask_id);

    CircularBuffer cb_input_negative_mask(cb_input_negative_mask_id);

    constexpr auto negative_mask_args = TensorAccessorArgs<input_mask_args.next_compile_time_args_offset()>();
    const auto negative_mask_tensor_accessor = TensorAccessor(negative_mask_args, input_negative_mask_addr);

#if defined(NEGATIVE_MASK_PARTIAL_READ)
    const uint32_t neg_mask_element_bytes = input_negative_mask_single_tile_size_bytes / TILE_HW;
    const uint32_t neg_mask_face_bytes = neg_mask_element_bytes * tt::constants::FACE_HW;
    const uint32_t neg_mask_face_w_bytes = neg_mask_element_bytes * tt::constants::FACE_WIDTH;
#endif
#endif

#if defined(MASK_SYNTHESIZE) || defined(NEGATIVE_MASK_SYNTHESIZE)
    // Group sizing for in-kernel mask synthesis. num_cols_per_group is the
    // number of channels per group (e.g. 10 for SDXL C=320, num_groups=32);
    // the row_offset wrapping recurrence mirrors groupnorm_input_mask.cpp:60-72.
    constexpr uint32_t MASK_NUM_COLS_PER_GROUP_V = MASK_NUM_COLS_PER_GROUP;
    constexpr uint32_t MASK_TILE_W = tt::constants::TILE_WIDTH;
    constexpr uint32_t MASK_GROUP_SIZE_MOD_TILE_W =
        (MASK_NUM_COLS_PER_GROUP_V % MASK_TILE_W == 0) ? 0 : (MASK_NUM_COLS_PER_GROUP_V % MASK_TILE_W);
#endif

    for (uint32_t b = 0; b < num_batches_per_core; ++b) {
        uint32_t input_mask_tile_id = input_mask_tile_start_id;
#if defined(FUSE_NEGATIVE_MASK)
        uint32_t input_negative_mask_tile_id = input_mask_tile_start_id;
#endif
#if defined(MASK_SYNTHESIZE) || defined(NEGATIVE_MASK_SYNTHESIZE)
        // start_stride for the first group on this core is 0. Subsequent
        // groups advance row_offset by group_size_mod_tile_w with wrapping.
        uint32_t mask_row_offset = 0;
#endif
        for (uint32_t i = 0; i < num_groups_per_core; ++i) {
            cb_input_mask.reserve_back(block_w);
            uint32_t l1_write_addr_input_mask = cb_input_mask.get_write_ptr();
#if defined(MASK_SYNTHESIZE)
            // Write face 0 row 0 + face 1 row 0 of each of the block_w mask
            // tiles directly, no DRAM read.
            tt::tt_metal::groupnorm::synthesize_group_mask_tiles_bf16(
                l1_write_addr_input_mask,
                mask_row_offset,
                MASK_NUM_COLS_PER_GROUP_V,
                block_w,
                input_mask_single_tile_size_bytes,
                MASK_TILE_W,
                tt::tt_metal::groupnorm::BF16_ONE,
                tt::tt_metal::groupnorm::BF16_ZERO);
#else
            for (uint32_t j = 0; j < block_w; ++j) {
#if defined(MASK_PARTIAL_READ)
                // Read face 0 row 0 (tile bytes [0, face_w_bytes)) and face 1
                // row 0 (tile bytes [face_bytes, face_bytes + face_w_bytes))
                // — the only bytes mul_tiles_bcast_rows actually consumes.
                // Inside the DRAM TILE page, face 1 lives at byte
                // `face_bytes` (typically 512 for bf16), not at `face_w_bytes`
                // — the latter is the row WIDTH, the former is the FACE size.
#ifdef ARCH_BLACKHOLE
                // BH requires 64 B DRAM-read alignment, so each strip is read
                // as a 64 B chunk; the extra 32 B per read overshoots into
                // face row 1 territory which the unpacker never reads.
                noc.async_read(
                    mask,
                    CoreLocalMem<uint32_t>(l1_write_addr_input_mask),
                    input_mask_face_w_bytes * 2,
                    {.page_id = input_mask_tile_id},
                    {});
                noc.async_read(
                    mask,
                    CoreLocalMem<uint32_t>(l1_write_addr_input_mask + input_mask_face_bytes),
                    input_mask_face_w_bytes * 2,
                    {.page_id = input_mask_tile_id, .offset_bytes = input_mask_face_bytes},
                    {});
#else
                noc.async_read(
                    mask,
                    CoreLocalMem<uint32_t>(l1_write_addr_input_mask),
                    input_mask_face_w_bytes,
                    {.page_id = input_mask_tile_id},
                    {});
                noc.async_read(
                    mask,
                    CoreLocalMem<uint32_t>(l1_write_addr_input_mask + input_mask_face_bytes),
                    input_mask_face_w_bytes,
                    {.page_id = input_mask_tile_id, .offset_bytes = input_mask_face_bytes},
                    {});
#endif
#else
                noc.async_read(
                    mask,
                    CoreLocalMem<uint32_t>(l1_write_addr_input_mask),
                    input_mask_single_tile_size_bytes,
                    {.page_id = input_mask_tile_id},
                    {});
#endif
                l1_write_addr_input_mask += input_mask_single_tile_size_bytes;
                input_mask_tile_id += 1;
            }
            noc.async_read_barrier();
#endif  // MASK_SYNTHESIZE
            cb_input_mask.push_back(block_w);

#if defined(FUSE_NEGATIVE_MASK)
            cb_input_negative_mask.reserve_back(block_w);
            uint32_t l1_write_addr_input_negative_mask = cb_input_negative_mask.get_write_ptr();
#if defined(NEGATIVE_MASK_SYNTHESIZE)
            // Negative mask: same start_stride as positive mask but inverted
            // fill values (1s outside the group, 0s inside).
            tt::tt_metal::groupnorm::synthesize_group_mask_tiles_bf16(
                l1_write_addr_input_negative_mask,
                mask_row_offset,
                MASK_NUM_COLS_PER_GROUP_V,
                block_w,
                input_negative_mask_single_tile_size_bytes,
                MASK_TILE_W,
                tt::tt_metal::groupnorm::BF16_ZERO,
                tt::tt_metal::groupnorm::BF16_ONE);
#else
            for (uint32_t j = 0; j < block_w; ++j) {
#if defined(NEGATIVE_MASK_PARTIAL_READ)
#ifdef ARCH_BLACKHOLE
                noc.async_read(
                    negative_mask_tensor_accessor,
                    CoreLocalMem<uint32_t>(l1_write_addr_input_negative_mask),
                    neg_mask_face_w_bytes * 2,
                    {.page_id = input_negative_mask_tile_id},
                    {});
                noc.async_read(
                    negative_mask_tensor_accessor,
                    CoreLocalMem<uint32_t>(l1_write_addr_input_negative_mask + neg_mask_face_bytes),
                    neg_mask_face_w_bytes * 2,
                    {.page_id = input_negative_mask_tile_id, .offset_bytes = neg_mask_face_bytes},
                    {});
#else
                noc.async_read(
                    negative_mask_tensor_accessor,
                    CoreLocalMem<uint32_t>(l1_write_addr_input_negative_mask),
                    neg_mask_face_w_bytes,
                    {.page_id = input_negative_mask_tile_id},
                    {});
                noc.async_read(
                    negative_mask_tensor_accessor,
                    CoreLocalMem<uint32_t>(l1_write_addr_input_negative_mask + neg_mask_face_bytes),
                    neg_mask_face_w_bytes,
                    {.page_id = input_negative_mask_tile_id, .offset_bytes = neg_mask_face_bytes},
                    {});
#endif
#else
                noc.async_read(
                    negative_mask_tensor_accessor,
                    CoreLocalMem<uint32_t>(l1_write_addr_input_negative_mask),
                    input_negative_mask_single_tile_size_bytes,
                    {.page_id = input_negative_mask_tile_id},
                    {});
#endif
                l1_write_addr_input_negative_mask += input_negative_mask_single_tile_size_bytes;
                input_negative_mask_tile_id += 1;
            }
            noc.async_read_barrier();
#endif  // NEGATIVE_MASK_SYNTHESIZE
            cb_input_negative_mask.push_back(block_w);
#endif

#if defined(MASK_SYNTHESIZE) || defined(NEGATIVE_MASK_SYNTHESIZE)
            // Advance row_offset for the next group (same recurrence as
            // groupnorm_input_mask.cpp:64-70).
            mask_row_offset =
                tt::tt_metal::groupnorm::advance_row_offset(mask_row_offset, MASK_GROUP_SIZE_MOD_TILE_W, MASK_TILE_W);
#endif

            if (i == 0 and b == 0) {
                constexpr uint32_t cb_in_2 = tt::CBIndex::c_2;
                dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                    cb_in_2,
                    ckernel::PoolType::AVG,
                    ckernel::ReduceDim::REDUCE_SCALAR,
                    reduce_factor_w>();

                constexpr uint32_t ones = 0x3F803F80;  // 2 packed bfloat16 into 1 uint32_t of value 1.0
                generate_tile_with_packed_bfloat16_values(cb_ones_id, ones);

                if constexpr (is_mcast_sender) {
                    constexpr uint32_t cb_in_4 = tt::CBIndex::c_4;
                    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                        cb_in_4,
                        ckernel::PoolType::AVG,
                        ckernel::ReduceDim::REDUCE_SCALAR,
                        reduce_factor_c>();
                }

                constexpr uint32_t eps_cb_id = tt::CBIndex::c_3;
                const uint32_t eps = get_arg_val<uint32_t>(0);
                generate_bcast_col_scalar(CircularBuffer(eps_cb_id), eps);

                if constexpr (fuse_gamma) {
                    const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma_id);
                    const auto gamma = TensorAccessor(gamma_args, gamma_addr);

                    cb_gamma.reserve_back(num_cols_tile_gamma_beta);
                    uint32_t l1_write_addr_gamma = cb_gamma.get_write_ptr();
                    for (uint32_t w = 0; w < num_cols_tile_gamma_beta; w++) {
                        uint32_t tile_id = gamma_tile_start_id + w;
#ifdef ARCH_BLACKHOLE
                        noc.async_read(
                            gamma,
                            CoreLocalMem<uint32_t>(l1_write_addr_gamma),
                            32 * 2,
                            {.page_id = tile_id},
                            {});
                        noc.async_read_barrier();
                        UnicastEndpoint self_ep;
                        noc.async_read(
                            self_ep,
                            CoreLocalMem<uint32_t>(l1_write_addr_gamma + 512),
                            32,
                            {.noc_x = my_x[0], .noc_y = my_y[0], .addr = l1_write_addr_gamma + 32},
                            {});
#else
                        noc.async_read(
                            gamma,
                            CoreLocalMem<uint32_t>(l1_write_addr_gamma),
                            32,
                            {.page_id = tile_id},
                            {});
                        noc.async_read(
                            gamma,
                            CoreLocalMem<uint32_t>(l1_write_addr_gamma + 512),
                            32,
                            {.page_id = tile_id, .offset_bytes = 32},
                            {});
#endif
                        l1_write_addr_gamma += gamma_tile_bytes;
                    }
                    noc.async_read_barrier();
                    cb_gamma.push_back(num_cols_tile_gamma_beta);
                }

                if constexpr (fuse_beta) {
                    const uint32_t beta_tile_bytes = get_tile_size(cb_beta_id);
                    const auto beta = TensorAccessor(beta_args, beta_addr);

                    uint32_t l1_write_addr_beta = cb_beta.get_write_ptr();
                    cb_beta.reserve_back(num_cols_tile_gamma_beta);
                    for (uint32_t w = 0; w < num_cols_tile_gamma_beta; w++) {
                        uint32_t tile_id = beta_tile_start_id + w;
#ifdef ARCH_BLACKHOLE
                        noc.async_read(
                            beta,
                            CoreLocalMem<uint32_t>(l1_write_addr_beta),
                            32 * 2,
                            {.page_id = tile_id},
                            {});
                        noc.async_read_barrier();
                        UnicastEndpoint self_ep;
                        noc.async_read(
                            self_ep,
                            CoreLocalMem<uint32_t>(l1_write_addr_beta + 512),
                            32,
                            {.noc_x = my_x[0], .noc_y = my_y[0], .addr = l1_write_addr_beta + 32},
                            {});
#else
                        noc.async_read(
                            beta,
                            CoreLocalMem<uint32_t>(l1_write_addr_beta),
                            32,
                            {.page_id = tile_id},
                            {});
                        noc.async_read(
                            beta,
                            CoreLocalMem<uint32_t>(l1_write_addr_beta + 512),
                            32,
                            {.page_id = tile_id, .offset_bytes = 32},
                            {});
#endif
                        l1_write_addr_beta += beta_tile_bytes;
                    }
                    noc.async_read_barrier();
                    cb_beta.push_back(num_cols_tile_gamma_beta);
                }
            }
        }
    }
}
