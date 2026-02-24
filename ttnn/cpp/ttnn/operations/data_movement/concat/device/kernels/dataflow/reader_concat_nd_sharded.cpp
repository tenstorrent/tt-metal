// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * ND sharded concat reader: runs on each core, copies this core's input shards
 * (in concat order) into this core's output shard. Uses TensorAccessor for each
 * tensor; reads each input shard page-by-page into a host-allocated L1 scratch
 * buffer and writes to the output shard sequentially.
 */

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "api/tensor/tensor_accessor.h"
#include "concat_nd_sharded_args.hpp"

namespace {

constexpr uint32_t CONCAT_ND_SHARDED_MAX_NUM_INPUTS = ttnn::kernel::CONCAT_ND_SHARDED_MAX_NUM_INPUTS;

// Scratch L1 address for copy_tensor_data (set from runtime arg: host-allocated L1 buffer).
uint32_t g_copy_scratch_l1_addr = 0;

inline void set_copy_tensor_data_scratch(uint32_t l1_addr) { g_copy_scratch_l1_addr = l1_addr; }

// Copies all data from src to dest starting at byte offset in dest. Returns offset + size in bytes of src data.
// Requires set_copy_tensor_data_scratch() to be called with a valid L1 buffer (at least one page) before use.
template <typename DestDSpec, typename SrcDSpec>
uint32_t copy_tensor_data(
    const TensorAccessor<DestDSpec>& dest,
    const TensorAccessor<SrcDSpec>& src,
    uint32_t offset,
    uint32_t /*absolute_offset_to_start*/,
    uint32_t /*amount_to_write*/,
    uint32_t /*amount_to_skip*/) {
    const uint32_t num_pages = src.dspec().tensor_volume();
    const uint32_t src_page_size = src.page_size;
    const uint32_t total_bytes = num_pages * src_page_size;
    for (uint32_t page_id = 0; page_id < num_pages; ++page_id) {
        const uint32_t dest_byte_offset = offset + page_id * src_page_size;
        const uint32_t dest_page_id = dest_byte_offset / dest.page_size;
        const uint32_t dest_offset_in_page = dest_byte_offset % dest.page_size;
        noc_async_read(src.get_noc_addr(page_id, 0), g_copy_scratch_l1_addr, src_page_size);
        noc_async_read_barrier();
        noc_async_write(g_copy_scratch_l1_addr, dest.get_noc_addr(dest_page_id, dest_offset_in_page), src_page_size);
        noc_async_write_barrier();
    }
    return offset + total_bytes;
}

// Compile-time layout: [0]=num_input_tensors, [1]=output_page_size, [2..17]=input_page_sizes,
// [18..]=output TensorAccessorArgs, then input0..input15 TensorAccessorArgs.
constexpr uint32_t CT_NUM_INPUTS = 0;
constexpr uint32_t CT_CONCAT_DIM = 1;
constexpr uint32_t CT_OUTPUT_PAGE_SIZE = 2;
constexpr uint32_t CT_INPUT_PAGE_SIZE_BASE = 3;
constexpr uint32_t CT_TENSOR_ACCESSOR_ARGS_BASE = 19;

}  // namespace

void kernel_main() {
    constexpr uint32_t num_input_tensors = get_compile_time_arg_val(CT_NUM_INPUTS);
    constexpr uint32_t concat_dim = get_compile_time_arg_val(CT_CONCAT_DIM);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(CT_OUTPUT_PAGE_SIZE);

    uint32_t argidx = 0;
    const uint32_t scratch_l1_addr = get_arg_val<uint32_t>(argidx++);
    set_copy_tensor_data_scratch(scratch_l1_addr);

    const uint32_t output_addr = get_arg_val<uint32_t>(argidx++);
    uint32_t input_addrs[CONCAT_ND_SHARDED_MAX_NUM_INPUTS];
    uint32_t absolute_offset_to_start[CONCAT_ND_SHARDED_MAX_NUM_INPUTS];
    uint32_t amount_to_write[CONCAT_ND_SHARDED_MAX_NUM_INPUTS];
    uint32_t amount_to_skip[CONCAT_ND_SHARDED_MAX_NUM_INPUTS];
    for (uint32_t i = 0; i < CONCAT_ND_SHARDED_MAX_NUM_INPUTS; ++i) {
        input_addrs[i] = get_arg_val<uint32_t>(argidx++);
        absolute_offset_to_start[i] = get_arg_val<uint32_t>(argidx++);
        amount_to_write[i] = get_arg_val<uint32_t>(argidx++);
        amount_to_skip[i] = get_arg_val<uint32_t>(argidx++);
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

    DPRINT << "output: shards=" << output_accessor.dspec().num_banks()
           << " pages=" << output_accessor.dspec().tensor_volume() << " page_size=" << output_page_size
           << " tile_size=" << output_page_size << ENDL();
    {
        const uint32_t out_rank = output_accessor.dspec().rank();
        DPRINT << "output dimensions: rank=" << out_rank;
        for (uint32_t d = 0; d < out_rank; ++d) {
            DPRINT << " dim" << d << "_ts=" << output_accessor.dspec().tensor_shape()[d]
                   << "_ss=" << output_accessor.dspec().shard_shape()[d]
                   << "_sg=" << output_accessor.dspec().shard_grid()[d];
        }
        DPRINT << ENDL();
    }

    // Copy each input to output sequentially; initial offset 0, then use offset returned from previous copy.
#define CONCAT_ND_DPRINT_INPUT(n)                                                                                  \
    do {                                                                                                           \
        const uint32_t initial_offset_##n = offset;                                                                \
        constexpr uint32_t in_page_size_##n = get_compile_time_arg_val(CT_INPUT_PAGE_SIZE_BASE + (n));             \
        const auto in##n##_accessor = TensorAccessor(in##n##_args, input_addrs[n], in_page_size_##n);              \
        DPRINT << "input " << (n) << " source: shards=" << in##n##_accessor.dspec().num_banks()                    \
               << " pages=" << in##n##_accessor.dspec().tensor_volume() << " page_size=" << in_page_size_##n       \
               << " tile_size=" << in_page_size_##n << " absolute_offset_to_start=" << absolute_offset_to_start[n] \
               << " amount_to_write=" << amount_to_write[n] << " amount_to_skip=" << amount_to_skip[n] << ENDL();  \
        {                                                                                                          \
            const uint32_t in_rank_##n = in##n##_accessor.dspec().rank();                                          \
            DPRINT << "input " << (n) << " dimensions: rank=" << in_rank_##n;                                      \
            for (uint32_t d = 0; d < in_rank_##n; ++d) {                                                           \
                DPRINT << " dim" << d << "_ts=" << in##n##_accessor.dspec().tensor_shape()[d]                      \
                       << "_ss=" << in##n##_accessor.dspec().shard_shape()[d]                                      \
                       << "_sg=" << in##n##_accessor.dspec().shard_grid()[d];                                      \
            }                                                                                                      \
            DPRINT << ENDL();                                                                                      \
        }                                                                                                          \
        offset = copy_tensor_data(                                                                                 \
            output_accessor,                                                                                       \
            in##n##_accessor,                                                                                      \
            offset,                                                                                                \
            absolute_offset_to_start[n],                                                                           \
            amount_to_write[n],                                                                                    \
            amount_to_skip[n]);                                                                                    \
        const uint32_t bytes_copied_##n = offset - initial_offset_##n;                                             \
        DPRINT << "input " << (n) << ": copied " << bytes_copied_##n << " bytes to initial offset "                \
               << initial_offset_##n << " (input tensor " << (n) << ")" << ENDL();                                 \
    } while (0)

    uint32_t offset = 0;
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
