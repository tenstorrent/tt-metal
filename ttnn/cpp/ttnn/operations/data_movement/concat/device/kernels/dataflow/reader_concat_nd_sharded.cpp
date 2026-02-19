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
#include "api/tensor/tensor_accessor.h"
#include "concat_nd_sharded_args.hpp"

namespace {

constexpr uint32_t CONCAT_ND_SHARDED_MAX_NUM_INPUTS = ttnn::kernel::CONCAT_ND_SHARDED_MAX_NUM_INPUTS;
constexpr uint32_t NUM_TENSORS = 1u + CONCAT_ND_SHARDED_MAX_NUM_INPUTS;  // output + inputs

// Compile-time: 0 = num_input_tensors, 1..17 = page_size, 18+ = TensorAccessorArgs (output, in0..in15)
constexpr uint32_t CTA_NUM_INPUTS = 0;
constexpr uint32_t CTA_PAGE_SIZE_BASE = 1;
constexpr uint32_t CTA_TENSOR_ACCESSOR_ARGS_START = 18;

// Runtime: 0..16 = buffer addresses (output, in0..in15), 17 = shard_id
constexpr uint32_t RT_SHARD_ID = 17;

constexpr uint32_t CB_ID_SCRATCH = 0;

template <typename OutputAccessor, typename InputAccessor>
void copy_one_input_shard_to_output(
    const OutputAccessor& output_accessor,
    const InputAccessor& input_accessor,
    uint32_t shard_id,
    uint32_t page_size,
    uint32_t& output_page_offset_in_shard) {
    auto input_pages = input_accessor.shard_pages(shard_id);
    for (const auto& page : input_pages) {
        cb_reserve_back(CB_ID_SCRATCH, 1);
        uint32_t l1_addr = get_write_ptr(CB_ID_SCRATCH);
        noc_async_read(page.noc_addr(), l1_addr, page_size);
        noc_async_read_barrier();

        uint64_t out_noc_addr = output_accessor.get_shard_noc_addr(shard_id, output_page_offset_in_shard * page_size);
        noc_async_write(l1_addr, out_noc_addr, page_size);
        noc_async_write_barrier();

        cb_push_back(CB_ID_SCRATCH, 1);
        ++output_page_offset_in_shard;
    }
}

}  // namespace

void kernel_main() {
    const uint32_t num_input_tensors = get_compile_time_arg_val(CTA_NUM_INPUTS);
    const uint32_t page_size = get_compile_time_arg_val(CTA_PAGE_SIZE_BASE);
    const uint32_t shard_id = get_arg_val<uint32_t>(RT_SHARD_ID);

    constexpr auto tensor_accessor_args_tuple =
        make_tensor_accessor_args_tuple<NUM_TENSORS, CTA_TENSOR_ACCESSOR_ARGS_START>();

    auto accessors_tuple = tensor_accessor::detail::make_tensor_accessor_tuple(
        tensor_accessor_args_tuple,
        /*address_rt_arg_index_start=*/0,
        /*page_size_ct_arg_index_start=*/CTA_PAGE_SIZE_BASE);

    const auto& output_accessor = std::get<0>(accessors_tuple);
    uint32_t output_page_offset_in_shard = 0;

    for (uint32_t i = 0; i < num_input_tensors; ++i) {
        switch (i) {
            case 0:
                copy_one_input_shard_to_output(
                    output_accessor, std::get<1>(accessors_tuple), shard_id, page_size, output_page_offset_in_shard);
                break;
            case 1:
                copy_one_input_shard_to_output(
                    output_accessor, std::get<2>(accessors_tuple), shard_id, page_size, output_page_offset_in_shard);
                break;
            case 2:
                copy_one_input_shard_to_output(
                    output_accessor, std::get<3>(accessors_tuple), shard_id, page_size, output_page_offset_in_shard);
                break;
            case 3:
                copy_one_input_shard_to_output(
                    output_accessor, std::get<4>(accessors_tuple), shard_id, page_size, output_page_offset_in_shard);
                break;
            case 4:
                copy_one_input_shard_to_output(
                    output_accessor, std::get<5>(accessors_tuple), shard_id, page_size, output_page_offset_in_shard);
                break;
            case 5:
                copy_one_input_shard_to_output(
                    output_accessor, std::get<6>(accessors_tuple), shard_id, page_size, output_page_offset_in_shard);
                break;
            case 6:
                copy_one_input_shard_to_output(
                    output_accessor, std::get<7>(accessors_tuple), shard_id, page_size, output_page_offset_in_shard);
                break;
            case 7:
                copy_one_input_shard_to_output(
                    output_accessor, std::get<8>(accessors_tuple), shard_id, page_size, output_page_offset_in_shard);
                break;
            case 8:
                copy_one_input_shard_to_output(
                    output_accessor, std::get<9>(accessors_tuple), shard_id, page_size, output_page_offset_in_shard);
                break;
            case 9:
                copy_one_input_shard_to_output(
                    output_accessor, std::get<10>(accessors_tuple), shard_id, page_size, output_page_offset_in_shard);
                break;
            case 10:
                copy_one_input_shard_to_output(
                    output_accessor, std::get<11>(accessors_tuple), shard_id, page_size, output_page_offset_in_shard);
                break;
            case 11:
                copy_one_input_shard_to_output(
                    output_accessor, std::get<12>(accessors_tuple), shard_id, page_size, output_page_offset_in_shard);
                break;
            case 12:
                copy_one_input_shard_to_output(
                    output_accessor, std::get<13>(accessors_tuple), shard_id, page_size, output_page_offset_in_shard);
                break;
            case 13:
                copy_one_input_shard_to_output(
                    output_accessor, std::get<14>(accessors_tuple), shard_id, page_size, output_page_offset_in_shard);
                break;
            case 14:
                copy_one_input_shard_to_output(
                    output_accessor, std::get<15>(accessors_tuple), shard_id, page_size, output_page_offset_in_shard);
                break;
            case 15:
                copy_one_input_shard_to_output(
                    output_accessor, std::get<16>(accessors_tuple), shard_id, page_size, output_page_offset_in_shard);
                break;
            default: break;
        }
    }
}
