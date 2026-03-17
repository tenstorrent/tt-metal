// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "experimental/tensor.h"
#include "experimental/endpoints.h"
#include "reshard_writer.hpp"

void kernel_main() {
    constexpr bool is_all_to_all_worker = get_compile_time_arg_val(0) == 1;
    constexpr bool fuse_gamma = get_compile_time_arg_val(1) == 1;
    constexpr bool fuse_beta = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t block_w = get_compile_time_arg_val(3);
    constexpr bool use_welford = get_compile_time_arg_val(4) == 1;
    constexpr auto gamma_args = TensorAccessorArgs<5>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    constexpr uint32_t stick_size = get_compile_time_arg_val(beta_args.next_compile_time_args_offset());
    constexpr bool FLOAT32_DTYPE_GAMMA = get_compile_time_arg_val(beta_args.next_compile_time_args_offset() + 1) == 1;
    constexpr bool FLOAT32_DTYPE_BETA = get_compile_time_arg_val(beta_args.next_compile_time_args_offset() + 2) == 1;

    // Reshard writer
    constexpr uint32_t worker_core_stride_w_bytes =
        get_compile_time_arg_val(beta_args.next_compile_time_args_offset() + 3);
    constexpr uint32_t storage_core_stride_w_bytes =
        get_compile_time_arg_val(beta_args.next_compile_time_args_offset() + 4);
    constexpr uint32_t block_ht = get_compile_time_arg_val(beta_args.next_compile_time_args_offset() + 5);

    const uint32_t gamma_addr = get_arg_val<uint32_t>(3);
    const uint32_t beta_addr = get_arg_val<uint32_t>(4);
    const uint32_t gamma_tile_start_id = get_arg_val<uint32_t>(5);
    const uint32_t beta_tile_start_id = get_arg_val<uint32_t>(6);

    // Reshard writer
#ifndef SKIP_WRITE_BACK
    const uint32_t num_segments_to_write_back = get_arg_val<uint32_t>(7);
    const uint32_t storage_core_start_offset = get_arg_val<uint32_t>(8);
    tt_l1_ptr uint32_t* segment_args = (tt_l1_ptr uint32_t*)(get_arg_addr(9));
#endif

    constexpr uint32_t cb_gamma = get_named_compile_time_arg_val("cb_gamma");
    constexpr uint32_t cb_beta = get_named_compile_time_arg_val("cb_beta");

    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t cb_out_resharded = get_named_compile_time_arg_val("cb_out_resharded");

    experimental::Noc noc;
    experimental::CircularBuffer cb_gamma_obj(cb_gamma);
    experimental::CircularBuffer cb_beta_obj(cb_beta);
    experimental::CircularBuffer cb_out_obj(cb_out);
    experimental::CircularBuffer cb_out_resharded_obj(cb_out_resharded);

    const uint32_t out_single_tile_size_bytes = get_tile_size(cb_out);

    if constexpr (!use_welford) {
        constexpr uint32_t cb_in_2 = get_named_compile_time_arg_val("cb_in_2");
        const uint32_t scalar_w_bits = get_arg_val<uint32_t>(1);
        float scalar_w_f = __builtin_bit_cast(float, scalar_w_bits);
        dataflow_kernel_lib::prepare_reduce_scaler<cb_in_2, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            scalar_w_f);

        constexpr uint32_t eps_cb_id = get_named_compile_time_arg_val("cb_eps");
        const uint32_t eps = get_arg_val<uint32_t>(2);
        generate_bcast_col_scalar(eps_cb_id, eps);

        if constexpr (is_all_to_all_worker) {
            constexpr uint32_t cb_in_4 = get_named_compile_time_arg_val("cb_in_4");
            const uint32_t scalar_c_bits = get_arg_val<uint32_t>(0);
            float scalar_c_f = __builtin_bit_cast(float, scalar_c_bits);
            dataflow_kernel_lib::prepare_reduce_scaler<cb_in_4>(scalar_c_f);
        }
    }

    if constexpr (fuse_gamma) {
        const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
        const auto gamma = TensorAccessor(gamma_args, gamma_addr, stick_size);

        constexpr uint32_t mask_read_tile_face_bytes = FLOAT32_DTYPE_GAMMA ? 64 : 32;
        constexpr uint32_t mask_read_tile_offset_bytes = FLOAT32_DTYPE_GAMMA ? 1024 : 512;

        experimental::UnicastEndpoint local_ep;
        cb_gamma_obj.reserve_back(block_w);
        for (uint32_t w = 0; w < block_w; w++) {
            uint32_t tile_id = gamma_tile_start_id + w;
            noc.async_read(
                gamma,
                cb_gamma_obj,
                mask_read_tile_face_bytes * 2,
                {.page_id = tile_id},
                {.offset_bytes = w * gamma_tile_bytes});
            noc.async_read_barrier();
            noc.async_read(
                local_ep,
                cb_gamma_obj,
                mask_read_tile_face_bytes,
                {.noc_x = my_x[noc.get_noc_id()],
                 .noc_y = my_y[noc.get_noc_id()],
                 .addr = cb_gamma_obj.get_write_ptr() + w * gamma_tile_bytes + mask_read_tile_face_bytes},
                {.offset_bytes = w * gamma_tile_bytes + mask_read_tile_offset_bytes});
        }
        noc.async_read_barrier();
        cb_gamma_obj.push_back(block_w);
    }

    if constexpr (fuse_beta) {
        const uint32_t beta_tile_bytes = get_tile_size(cb_beta);
        const auto beta = TensorAccessor(beta_args, beta_addr, stick_size);

        uint32_t mask_read_tile_face_bytes = FLOAT32_DTYPE_BETA ? 64 : 32;
        uint32_t mask_read_tile_offset_bytes = FLOAT32_DTYPE_BETA ? 1024 : 512;

        experimental::UnicastEndpoint local_ep;
        cb_beta_obj.reserve_back(block_w);
        for (uint32_t w = 0; w < block_w; w++) {
            uint32_t tile_id = beta_tile_start_id + w;
            noc.async_read(
                beta,
                cb_beta_obj,
                mask_read_tile_face_bytes * 2,
                {.page_id = tile_id},
                {.offset_bytes = w * beta_tile_bytes});
            noc.async_read_barrier();
            noc.async_read(
                local_ep,
                cb_beta_obj,
                mask_read_tile_face_bytes,
                {.noc_x = my_x[noc.get_noc_id()],
                 .noc_y = my_y[noc.get_noc_id()],
                 .addr = cb_beta_obj.get_write_ptr() + w * beta_tile_bytes + mask_read_tile_face_bytes},
                {.offset_bytes = w * beta_tile_bytes + mask_read_tile_offset_bytes});
        }
        noc.async_read_barrier();
        cb_beta_obj.push_back(block_w);
    }

#ifndef SKIP_WRITE_BACK
    write_resharded_data(
        noc,
        cb_out_obj,
        cb_out_resharded_obj,
        num_segments_to_write_back,
        storage_core_start_offset,
        segment_args,
        worker_core_stride_w_bytes,
        storage_core_stride_w_bytes,
        block_ht);
#endif
}
