// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
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
    // compile-time args
    constexpr uint32_t input_grad_rank = get_arg(args::input_grad_rank);

    // runtime args
    const auto num_output_tiles = get_arg(args::num_output_tiles);
    const auto start_id = get_arg(args::start_id);

    // The three per-dim blocks (each of length input_grad_rank) arrive as positional runtime
    // varargs, read in the same order the host appended them.
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

    constexpr uint32_t onetile = 1;

    // zero tile
    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 0.0f;
    DataflowBuffer dfb_in1_obj(dfb::in1);
    fill_cb_with_value(dfb_in1_obj, scaler.u);

    const auto output_grad_addrg = TensorAccessor(tensor::output_grad);

    Noc noc;
    DataflowBuffer dfb_in0_obj(dfb::in0);
    const auto in0_tile_bytes = dfb_in0_obj.get_tile_size();

    for (uint32_t i = start_id; i < start_id + num_output_tiles; i++) {
        auto read_tile_id = get_output_grad_tile(
            i, input_grad_rank, output_grad_dim, output_grad_stride, input_grad_dim, input_grad_stride, need_bcast_dim);

        dfb_in0_obj.reserve_back(onetile);
        noc.async_read(output_grad_addrg, dfb_in0_obj, in0_tile_bytes, {.page_id = read_tile_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in0_obj.push_back(onetile);
    }
}
