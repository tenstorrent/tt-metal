// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "dataflow_common.hpp"

void kernel_main() {
    Noc noc;

    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(4);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(9);
    constexpr uint32_t num_cores = get_compile_time_arg_val(10);
    constexpr bool out_concat_heads = get_compile_time_arg_val(22) == 1;

    constexpr auto out_args = TensorAccessorArgs<30>();
    constexpr auto cu_window_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto q_offset_args = TensorAccessorArgs<cu_window_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_arg_offset = q_offset_args.next_compile_time_args_offset();
    constexpr uint32_t cb_mask_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_identity_scale_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(cb_arg_offset + 2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(cb_arg_offset + 4);

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;
    constexpr uint32_t out_row_stride = out_concat_heads ? NQH * vDHt : vDHt;
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();

    const auto out_writer = TensorAccessor(out_args, get_arg_val<uint32_t>(0));
    const uint32_t n_steps = get_arg_val<uint32_t>(1);

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_identity_scale_in,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), identity_scalar_packed);
    generate_lightweight_mask_tiles<0u, 0u, cb_mask_in, true, 0u>(noc);

    for (uint32_t s = 0; s < n_steps; ++s) {
        write_block<0>(
            noc,
            out_writer,
            cb_out,
            out_chunk_tiles,
            Sq_chunk_t,
            vDHt,
            get_arg_val<uint32_t>(2 + s),
            tile_bytes,
            barrier_threshold,
            out_row_stride,
            0);
    }
}
