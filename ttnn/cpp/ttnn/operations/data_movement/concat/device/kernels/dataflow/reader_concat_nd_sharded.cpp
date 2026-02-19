// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * ND sharded concat reader: runs on each core, copies this core's input shards
 * (in concat order) into this core's output shard. Uses TensorAccessor for each
 * tensor; reads each input shard page-by-page into a scratch CB and writes to
 * the output shard sequentially.
 */

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "api/tensor/tensor_accessor.h"
#include "concat_nd_sharded_args.hpp"

namespace {

constexpr uint32_t CONCAT_ND_SHARDED_MAX_NUM_INPUTS = ttnn::kernel::CONCAT_ND_SHARDED_MAX_NUM_INPUTS;

// Compile-time layout: [0]=num_input_tensors, [1]=output_page_size, [2..17]=input_page_sizes,
// [18..]=output TensorAccessorArgs, then input0..input15 TensorAccessorArgs.
constexpr uint32_t CT_NUM_INPUTS = 0;
constexpr uint32_t CT_OUTPUT_PAGE_SIZE = 1;
constexpr uint32_t CT_INPUT_PAGE_SIZE_BASE = 2;
constexpr uint32_t CT_TENSOR_ACCESSOR_ARGS_BASE = 18;

}  // namespace

void kernel_main() {
    constexpr uint32_t num_input_tensors = get_compile_time_arg_val(CT_NUM_INPUTS);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(CT_OUTPUT_PAGE_SIZE);

    uint32_t argidx = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(argidx++);
    uint32_t input_addrs[CONCAT_ND_SHARDED_MAX_NUM_INPUTS];
    for (uint32_t i = 0; i < CONCAT_ND_SHARDED_MAX_NUM_INPUTS; ++i) {
        input_addrs[i] = get_arg_val<uint32_t>(argidx++);
    }
    const uint32_t shard_id = get_arg_val<uint32_t>(argidx++);

    // Build TensorAccessorArgs chain: output at 18, then input0..input15
    constexpr auto out_args = TensorAccessorArgs<CT_TENSOR_ACCESSOR_ARGS_BASE>();
    constexpr auto in0_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    constexpr auto in2_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
    constexpr auto in3_args = TensorAccessorArgs<in2_args.next_compile_time_args_offset()>();
    constexpr auto in4_args = TensorAccessorArgs<in3_args.next_compile_time_args_offset()>();
    constexpr auto in5_args = TensorAccessorArgs<in4_args.next_compile_time_args_offset()>();
    constexpr auto in6_args = TensorAccessorArgs<in5_args.next_compile_time_args_offset()>();
    constexpr auto in7_args = TensorAccessorArgs<in6_args.next_compile_time_args_offset()>();
    constexpr auto in8_args = TensorAccessorArgs<in7_args.next_compile_time_args_offset()>();
    constexpr auto in9_args = TensorAccessorArgs<in8_args.next_compile_time_args_offset()>();
    constexpr auto in10_args = TensorAccessorArgs<in9_args.next_compile_time_args_offset()>();
    constexpr auto in11_args = TensorAccessorArgs<in10_args.next_compile_time_args_offset()>();
    constexpr auto in12_args = TensorAccessorArgs<in11_args.next_compile_time_args_offset()>();
    constexpr auto in13_args = TensorAccessorArgs<in12_args.next_compile_time_args_offset()>();
    constexpr auto in14_args = TensorAccessorArgs<in13_args.next_compile_time_args_offset()>();
    constexpr auto in15_args = TensorAccessorArgs<in14_args.next_compile_time_args_offset()>();

    const auto output_accessor = TensorAccessor(out_args, output_addr, output_page_size);

    // Create TensorAccessor for each input and DPRINT: input data (pages), input page size,
    // output data (pages), output page size.
#define CONCAT_ND_DPRINT_INPUT(n)                                                                      \
    do {                                                                                               \
        constexpr uint32_t in_page_size_##n = get_compile_time_arg_val(CT_INPUT_PAGE_SIZE_BASE + (n)); \
        const auto in##n##_accessor = TensorAccessor(in##n##_args, input_addrs[n], in_page_size_##n);  \
        DPRINT << "input " << (n) << ": data (pages)=" << in##n##_accessor.dspec().tensor_volume()     \
               << " page_size=" << in_page_size_##n                                                    \
               << " output data (pages)=" << output_accessor.dspec().tensor_volume()                   \
               << " output page_size=" << output_page_size << ENDL();                                  \
    } while (0)

    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        switch (i) {
            case 0: CONCAT_ND_DPRINT_INPUT(0); break;
            case 1: CONCAT_ND_DPRINT_INPUT(1); break;
            case 2: CONCAT_ND_DPRINT_INPUT(2); break;
            case 3: CONCAT_ND_DPRINT_INPUT(3); break;
            case 4: CONCAT_ND_DPRINT_INPUT(4); break;
            case 5: CONCAT_ND_DPRINT_INPUT(5); break;
            case 6: CONCAT_ND_DPRINT_INPUT(6); break;
            case 7: CONCAT_ND_DPRINT_INPUT(7); break;
            case 8: CONCAT_ND_DPRINT_INPUT(8); break;
            case 9: CONCAT_ND_DPRINT_INPUT(9); break;
            case 10: CONCAT_ND_DPRINT_INPUT(10); break;
            case 11: CONCAT_ND_DPRINT_INPUT(11); break;
            case 12: CONCAT_ND_DPRINT_INPUT(12); break;
            case 13: CONCAT_ND_DPRINT_INPUT(13); break;
            case 14: CONCAT_ND_DPRINT_INPUT(14); break;
            case 15: CONCAT_ND_DPRINT_INPUT(15); break;
            default: break;
        }
    }
#undef CONCAT_ND_DPRINT_INPUT
}
