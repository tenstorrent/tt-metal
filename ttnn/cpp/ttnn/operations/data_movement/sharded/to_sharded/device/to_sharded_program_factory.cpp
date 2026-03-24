// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "to_sharded_program_factory.hpp"

#include <cmath>

#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"

#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

ToShardedRowMajorProgramFactory::cached_program_t ToShardedRowMajorProgramFactory::create(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = output_tensor;
    // Keep explicit bool init to match legacy behavior which forced it true
    // bool keep_l1_aligned = true;  // operation_attributes.keep_l1_aligned;

    tt::tt_metal::Program program{};

    const auto& output_distribution_spec = output.buffer()->buffer_distribution_spec().value();
    const auto num_output_shards = output_distribution_spec.num_shards();
    const auto bytes_per_element = input.element_size();
    const auto elements_per_tensor_row = input.logical_shape()[-1];
    uint32_t num_input_pages_in_row = 1;
    uint32_t num_output_pages_in_row = 1;
    uint32_t elements_per_output_page = output.logical_shape()[-1];
    uint32_t elements_per_input_page = input.logical_shape()[-1];

    if (input.is_sharded() && input.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        uint32_t input_shard_width =
            (input.shard_spec().has_value() ? input.shard_spec().value().shape[1]
                                            : input.nd_shard_spec().value().shard_shape[-1]);
        num_input_pages_in_row = tt::div_up(elements_per_tensor_row, input_shard_width);
        elements_per_input_page = input_shard_width;
    }
    if (output.is_sharded() &&
        output.memory_config().memory_layout() !=
            TensorMemoryLayout::HEIGHT_SHARDED) {  // This favtory is meant to convert things to ND sharded, but if we
                                                   // decide to make this apply to even converting to legacy sharded,
                                                   // thenm having this branch lets us handle that case too.
        uint32_t output_shard_width =
            (output.shard_spec().has_value() ? output.shard_spec().value().shape[1]
                                             : output.nd_shard_spec().value().shard_shape[-1]);
        num_output_pages_in_row = tt::div_up(elements_per_tensor_row, output_shard_width);
        elements_per_output_page = output_shard_width;
    }

    const auto input_pages_cb_index = tt::CBIndex::c_0;
    const auto output_page_cb_index = tt::CBIndex::c_16;

    // computation of core_grid. To be replaced by get_optimal_worker_cores_for_sharded_tensor API in PR #40452 once
    // that is merged
    std::vector<CoreCoord> ordered_cores_with_data;

    if (!output.memory_config().is_dram()) {
        ordered_cores_with_data = output_distribution_spec.cores_with_data();
    } else {
        TT_FATAL(output.device() != nullptr, "Device pointer must be valid when selecting optimal DRAM worker cores");
        auto all_dram_workers =
            output.device()->get_optimal_dram_bank_to_logical_worker_assignment(NOC::RISCV_0_default);
        const auto dram_banks = output_distribution_spec.cores_with_data();
        ordered_cores_with_data.reserve(dram_banks.size());
        for (const auto& dram_core : dram_banks) {
            const uint32_t dram_channel = output.device()->dram_channel_from_logical_core(dram_core);
            ordered_cores_with_data.push_back(all_dram_workers[dram_channel]);
        }
    }

    // end of core grid computation. Can add option to use custom core grid too (like default path to accomodate
    // subdevices op)

    const auto num_cores = ordered_cores_with_data.size();
    const auto& ordered_cores_with_data_range = CoreRangeSet(ttsl::Span<const CoreCoord>(ordered_cores_with_data));

    // std::cout << "num_cores: " << num_cores << std::endl;
    // Creating CBs
    uint32_t max_number_of_input_pages_overlapping_an_output_page = 1;
    if (elements_per_input_page < elements_per_output_page) {
        max_number_of_input_pages_overlapping_an_output_page =
            static_cast<uint32_t>(elements_per_output_page / elements_per_input_page) + 2;
    } else if (elements_per_input_page > elements_per_output_page) {
        max_number_of_input_pages_overlapping_an_output_page = 2;
    }

    // Configuring the CB that store input pages
    const auto aligned_input_page_size =
        input.buffer()
            ->aligned_page_size();  //  Input page size must be aligned to avoid alignment errors in the reader kernel
    const auto input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::tt_metal::CircularBufferConfig input_pages_cb_config =
        tt::tt_metal::CircularBufferConfig(
            max_number_of_input_pages_overlapping_an_output_page * aligned_input_page_size,
            {{input_pages_cb_index, input_cb_data_format}})
            .set_page_size(input_pages_cb_index, aligned_input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, ordered_cores_with_data_range, input_pages_cb_config);

    // Configuring the CB that stores output pages
    const auto output_page_size =
        output.buffer()->page_size();  // Output page size doesn't need to be aligned, since we are only writing the
                                       // valis nonpadding page data to the output buffer in the kernel
    const auto output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::tt_metal::CircularBufferConfig output_pages_cb_config =
        tt::tt_metal::CircularBufferConfig(1 * output_page_size, {{output_page_cb_index, output_cb_data_format}})
            .set_page_size(output_page_cb_index, output_page_size);
    tt::tt_metal::CreateCircularBuffer(program, ordered_cores_with_data_range, output_pages_cb_config);

    // Reader kernel config with compile-time args
    std::vector<uint32_t> reader_compile_time_args = {
        static_cast<uint32_t>(input_pages_cb_index),
        static_cast<uint32_t>(output_page_cb_index),
        static_cast<uint32_t>(num_output_shards),
        static_cast<uint32_t>(num_cores),
        static_cast<uint32_t>(num_output_pages_in_row),
        static_cast<uint32_t>(num_input_pages_in_row),
        static_cast<uint32_t>(elements_per_output_page),
        static_cast<uint32_t>(bytes_per_element),
        static_cast<uint32_t>(elements_per_input_page),
        static_cast<uint32_t>(elements_per_tensor_row),
    };

    // std::cout << "num_output_shards: " << num_output_shards << std::endl;
    // std::cout << "num_input_shards: " << input.buffer()->buffer_distribution_spec().value().num_shards() <<
    // std::endl; std::cout<<"output_padded_shape: " << output.padded_shape() << std::endl;
    // std::cout<<"input_padded_shape: " << input.padded_shape() << std::endl;
    // std::cout<<"output_logical_shape: " << output.logical_shape() << std::endl;
    // std::cout<<"input_logical_shape: " << input.logical_shape() << std::endl;
    // std::cout << "elements_per_output_page: " << elements_per_output_page << std::endl;
    // std::cout << "elements_per_input_page: " << elements_per_input_page << std::endl;
    // std::cout << "elements_per_tensor_row: " << elements_per_tensor_row << std::endl;
    // std::cout << "bytes_per_element: " << bytes_per_element << std::endl;
    // std::cout << "num_output_pages_in_row: " << num_output_pages_in_row << std::endl;
    // std::cout << "num_input_pages_in_row: " << num_input_pages_in_row << std::endl;
    // std::cout << "num_cores: " << num_cores << std::endl;
    tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/write_to_nd_shard_pages_row_major.cpp",
        ordered_cores_with_data_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Set runtime args
    uint32_t start_shard_id = 0;
    for (const auto& core : ordered_cores_with_data) {
        // Reader run-time args
        std::vector<uint32_t> reader_run_time_args = {
            input.buffer()->address(), output.buffer()->address(), start_shard_id};

        start_shard_id++;
        // Set run-time arg
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_run_time_args);
    }

    return {std::move(program), {reader_kernel_id, ordered_cores_with_data}};
}
void ToShardedRowMajorProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
    const auto& cores = cached_program.shared_variables.cores;
    const auto& input_buffer = tensor_args.input_tensor.buffer();
    const auto& output_buffer = output_tensor.buffer();
    for (const auto& core : cores) {
        auto& runtime_args = runtime_args_by_core[core.x][core.y];
        runtime_args[0] = input_buffer->address();
        runtime_args[1] = output_buffer->address();
    }
}

}  // namespace ttnn::prim
