// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

static constexpr int32_t MAX_NUM_DIMENSIONS = 8;

inline uint32_t get_output_grad_tile(
    uint32_t idx,
    uint32_t rank,
    uint32_t* output_grad_dim,
    uint32_t* output_grad_stride,
    uint32_t* input_grad_dim,
    uint32_t* input_grad_stride,
    bool* need_bcast_dim) {
    uint32_t cur_idx[MAX_NUM_DIMENSIONS];

    for (uint32_t i = 0; i < rank; ++i) {
        cur_idx[i] = (need_bcast_dim[i]) ? (0) : ((idx / input_grad_stride[i]) % input_grad_dim[i]);
    }

    uint32_t read_tile_id = 0;
    for (uint32_t i = 0; i < rank; ++i) {
        read_tile_id += (cur_idx[i] * output_grad_stride[i]);
    }

    return read_tile_id;
}

void kernel_main() {
    // compile time args
    constexpr auto input_grad_rank = get_arg(args::input_grad_rank);

    // runtime args
    // input/output/output_grad base addresses are injected by their TensorBindings
    // (TensorAccessor(tensor::name)); no buffer-address RTA is read here.
    const auto decimal = get_arg(args::decimal);

    const auto num_output_tiles = get_arg(args::num_output_tiles);
    const auto start_id = get_arg(args::start_id);

    // The three per-dimension blocks are read as runtime varargs: their count is
    // input_grad_rank (a CTA), so the number of reads varies per instantiation.
    uint32_t vararg_idx = 0;

    uint32_t output_grad_dim[MAX_NUM_DIMENSIONS];
    for (uint32_t i = 0; i < input_grad_rank; ++i) {
        output_grad_dim[i] = get_vararg(vararg_idx++);
    }

    uint32_t input_grad_dim[MAX_NUM_DIMENSIONS];
    for (uint32_t i = 0; i < input_grad_rank; ++i) {
        input_grad_dim[i] = get_vararg(vararg_idx++);
    }

    bool need_bcast_dim[MAX_NUM_DIMENSIONS];
    for (uint32_t i = 0; i < input_grad_rank; ++i) {
        need_bcast_dim[i] = (get_vararg(vararg_idx++) == 1);
    }

    uint32_t output_grad_stride[MAX_NUM_DIMENSIONS];
    output_grad_stride[0] = 1;
    for (uint32_t i = 1; i < input_grad_rank; ++i) {
        output_grad_stride[i] = output_grad_stride[i - 1] * output_grad_dim[i - 1];
    }

    uint32_t input_grad_stride[MAX_NUM_DIMENSIONS];
    input_grad_stride[0] = 1;
    for (uint32_t i = 1; i < input_grad_rank; ++i) {
        input_grad_stride[i] = input_grad_stride[i - 1] * input_grad_dim[i - 1];
    }

    // input
    const auto input_addrg = TensorAccessor(tensor::input);

    // output
    const auto output_addrg = TensorAccessor(tensor::output);

    // output_grad
    const auto output_grad_addrg = TensorAccessor(tensor::output_grad);

    DataflowBuffer dfb_decimal(dfb::decimal);
    fill_cb_with_value(dfb_decimal, decimal);

    Noc noc;
    DataflowBuffer dfb_input(dfb::input);
    DataflowBuffer dfb_output(dfb::output);
    DataflowBuffer dfb_output_grad(dfb::output_grad);
    const auto input_tile_bytes = dfb_input.get_tile_size();
    const auto output_tile_bytes = dfb_output.get_tile_size();
    const auto output_grad_tile_bytes = dfb_output_grad.get_tile_size();

    for (uint32_t i = start_id; i < start_id + num_output_tiles; i++) {
        uint32_t input_tile_id = i;
        auto read_tile_id = get_output_grad_tile(
            i, input_grad_rank, output_grad_dim, output_grad_stride, input_grad_dim, input_grad_stride, need_bcast_dim);

        dfb_input.reserve_back(1);
        noc.async_read(input_addrg, dfb_input, input_tile_bytes, {.page_id = input_tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_input.push_back(1);

        dfb_output.reserve_back(1);
        noc.async_read(output_addrg, dfb_output, output_tile_bytes, {.page_id = read_tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_output.push_back(1);

        dfb_output_grad.reserve_back(1);
        noc.async_read(
            output_grad_addrg, dfb_output_grad, output_grad_tile_bytes, {.page_id = read_tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_output_grad.push_back(1);
    }

}  // void kernel_main()
