// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "scatter_program_factory.hpp"

#include "scatter_device_operation_types.hpp"
#include "tt-metalium/device.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>

namespace ttnn::operations::experimental::scatter {

namespace {
constexpr uint32_t BIT_MASK_32 = 32 - 1;

uint64_t ceil32(const uint64_t& number) {
    return ((number & BIT_MASK_32) == 0) ? number : ((number | BIT_MASK_32) + 1);
}

bool is_pow2_min32(const uint64_t& number) { return ((number & (number - 1)) == 0) && number >= 32; }
}  // namespace

// maximal input/index/source/output chunk size, divisible by 32, calculated as follows:
// BH available L1 mem size of nearly 1.5 MB...
// ... divided by 4 to be able to allocate four equally long row chunks (coming from input/index/source/output
// tensors)
// ... divided by 4 to account for 4-byte datum sizes of each tensor (fp32, int32)
// ... minimized by ~20% to account for reserved memory
uint32_t calculate_optimal_chunk_size(IDevice* device) { return ceil32(device->l1_size_per_core() / 4 / 4 * 0.8 - 32); }

ScatterProgramFactory::cached_program_t ScatterProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    using namespace tt::tt_metal;
    using namespace tt::constants;

    Program program{};

    const auto& input_tensor{tensor_args.input_tensor};
    const auto& input_shape{input_tensor.logical_shape()};
    const auto& index_tensor{tensor_args.index_tensor};
    const auto& index_shape{index_tensor.logical_shape()};
    const auto& src_tensor{tensor_args.src_tensor};
    const auto& src_shape{src_tensor.logical_shape()};
    const auto& output_shape{output_tensor.logical_shape()};

    auto input_buffer = input_tensor.buffer();
    auto index_buffer = index_tensor.buffer();
    auto src_buffer = src_tensor.buffer();
    auto output_buffer = output_tensor.buffer();

    const uint32_t input_tensor_is_dram = input_buffer->buffer_type() == BufferType::DRAM;
    const uint32_t index_tensor_is_dram = index_buffer->buffer_type() == BufferType::DRAM;
    const uint32_t src_tensor_is_dram = src_buffer->buffer_type() == BufferType::DRAM;
    const uint32_t output_tensor_is_dram = output_buffer->buffer_type() == BufferType::DRAM;

    auto device = input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    const uint32_t& input_stick_size = input_shape[-1];
    const uint32_t& index_stick_size = index_shape[-1];
    const uint32_t& source_stick_size = src_shape[-1];
    const uint32_t& output_stick_size = output_shape[-1];

    const uint32_t work_units = input_tensor.logical_volume() / input_stick_size;
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, work_units);

    // input dtype byte sizes
    const uint32_t& input_datum_size = input_tensor.element_size();
    const uint32_t& index_datum_size = index_tensor.element_size();
    const uint32_t& source_datum_size = src_tensor.element_size();
    const uint32_t& output_datum_size = output_tensor.element_size();

    // input row byte sizes
    const uint32_t& input_stick_size_bytes = input_stick_size * input_datum_size;
    const uint32_t& index_stick_size_bytes = index_stick_size * index_datum_size;
    const uint32_t& source_stick_size_bytes = source_stick_size * source_datum_size;
    const uint32_t& output_stick_size_bytes = output_stick_size * output_datum_size;

    // check if row byte sizes are at least 32 and a power of 2 (for InterleavedAddrGen)
    const uint32_t is_input_stick_size_bytes_pow2_min_32 = is_pow2_min32(input_stick_size_bytes);
    const uint32_t is_index_stick_size_bytes_pow2_min_32 = is_pow2_min32(index_stick_size_bytes);
    const uint32_t is_source_stick_size_bytes_pow2_min_32 = is_pow2_min32(source_stick_size_bytes);
    const uint32_t is_output_stick_size_bytes_pow2_min_32 = is_pow2_min32(output_stick_size_bytes);

    // for InterleavedAddrGen
    const uint32_t input_stick_size_bytes_log2 =
        is_input_stick_size_bytes_pow2_min_32 ? std::log2(input_stick_size_bytes) : 0;
    const uint32_t index_stick_size_bytes_log2 =
        is_index_stick_size_bytes_pow2_min_32 ? std::log2(index_stick_size_bytes) : 0;
    const uint32_t source_stick_size_bytes_log2 =
        is_source_stick_size_bytes_pow2_min_32 ? std::log2(source_stick_size_bytes) : 0;
    const uint32_t output_stick_size_bytes_log2 =
        is_output_stick_size_bytes_pow2_min_32 ? std::log2(output_stick_size_bytes) : 0;

    // maximal input/index/source/output chunk size, divisible by 32, calculated as follows:
    // BH available L1 mem size of nearly 1.5 MB...
    // ... divided by 4 to be able to allocate four equally long row chunks (coming from input/index/source/output
    // tensors)
    // ... divided by 4 to account for 4-byte datum sizes of each tensor (fp32, int32)
    // ... minimized by ~20% to account for reserved memory
    const uint32_t input_and_output_max_chunk_size = calculate_optimal_chunk_size(input_tensor.device());
    const uint32_t index_and_source_max_chunk_size = input_and_output_max_chunk_size;
    const uint32_t input_and_output_chunk_size = std::min(input_stick_size, input_and_output_max_chunk_size);
    const uint32_t index_and_source_chunk_size = std::min(index_stick_size, index_and_source_max_chunk_size);
    const uint32_t input_and_output_chunk_size_bytes = input_and_output_chunk_size * input_datum_size;
    const uint32_t index_chunk_size_bytes = index_and_source_chunk_size * index_datum_size;
    const uint32_t source_chunk_size_bytes = index_and_source_chunk_size * source_datum_size;

    // pad pages to 32
    const uint32_t input_page_size_bytes = ceil32(input_and_output_chunk_size_bytes);
    const uint32_t index_page_size_bytes = ceil32(index_chunk_size_bytes);
    const uint32_t source_page_size_bytes = ceil32(source_chunk_size_bytes);
    const uint32_t output_page_size_bytes = ceil32(input_and_output_chunk_size_bytes);

    create_cb(program, input_tensor.get_dtype(), ScatterCB::INPUT, all_cores, input_page_size_bytes);
    create_cb(program, index_tensor.get_dtype(), ScatterCB::INDEX, all_cores, index_page_size_bytes);
    create_cb(program, src_tensor.get_dtype(), ScatterCB::SRC, all_cores, source_page_size_bytes);
    create_cb(program, output_tensor.get_dtype(), ScatterCB::DST, all_cores, output_page_size_bytes);

    constexpr const char* reader_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/scatter/device/kernels/dataflow/reader_scatter.cpp";
    constexpr const char* writer_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/scatter/device/kernels/dataflow/writer_scatter.cpp";

    const std::vector<uint32_t> compile_time_args{
        {input_tensor_is_dram,
         index_tensor_is_dram,
         src_tensor_is_dram,
         output_tensor_is_dram,
         input_tensor.buffer()->address(),
         index_tensor.buffer()->address(),
         src_tensor.buffer()->address(),
         output_tensor.buffer()->address(),
         static_cast<uint32_t>(ScatterCB::INPUT),
         static_cast<uint32_t>(ScatterCB::INDEX),
         static_cast<uint32_t>(ScatterCB::SRC),
         static_cast<uint32_t>(ScatterCB::DST),
         input_stick_size,
         index_stick_size,
         source_stick_size,
         output_stick_size,
         input_stick_size_bytes,
         index_stick_size_bytes,
         source_stick_size_bytes,
         output_stick_size_bytes,
         input_stick_size_bytes_log2,
         index_stick_size_bytes_log2,
         source_stick_size_bytes_log2,
         output_stick_size_bytes_log2,
         is_input_stick_size_bytes_pow2_min_32,
         is_index_stick_size_bytes_pow2_min_32,
         is_source_stick_size_bytes_pow2_min_32,
         is_output_stick_size_bytes_pow2_min_32}};

    auto reader_kernel =
        create_kernel(program, reader_kernel_path, all_cores, ReaderDataMovementConfig{compile_time_args});
    auto writer_kernel =
        create_kernel(program, writer_kernel_path, all_cores, WriterDataMovementConfig{compile_time_args});

    const uint32_t& num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t& num_cores_y = compute_with_storage_grid_size.y;
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y);
    uint32_t stick_offset = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core{i / num_cores_y, i % num_cores_y};

        uint32_t sticks_per_core;
        if (core_group_1.contains(core)) {
            sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            sticks_per_core = num_sticks_per_core_group_2;
        } else {
            TT_THROW("Core not in any predefined core range.");
        }

        SetRuntimeArgs(
            program,
            reader_kernel,
            core,
            {input_buffer->address(), index_buffer->address(), src_buffer->address(), stick_offset, sticks_per_core, input_and_output_chunk_size, index_and_source_chunk_size});

        SetRuntimeArgs(program, writer_kernel, core, {output_buffer->address(), stick_offset, sticks_per_core, input_and_output_chunk_size});

        stick_offset += sticks_per_core;
    }

    return {std::move(program), {reader_kernel, writer_kernel, cores}};
}

void ScatterProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;

    auto input_buffer_address = tensor_args.input_tensor.buffer()->address();
    auto index_buffer_address = tensor_args.index_tensor.buffer()->address();
    auto source_buffer_address = tensor_args.src_tensor.buffer()->address();
    auto output_buffer_address = output_tensor.buffer()->address();
    for (const auto& core : cores) {
        auto& reader_runtime_args  = GetRuntimeArgs(program, reader_kernel_id, core);
        auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
        reader_runtime_args[0] = input_buffer_address;
        reader_runtime_args[1] = index_buffer_address;
        reader_runtime_args[2] = source_buffer_address;
        writer_runtime_args[0] = output_buffer_address;
    }
}

CBHandle ScatterProgramFactory::create_cb(
    Program& program,
    const DataType& dtype,
    const ScatterCB& scatter_cb,
    const CoreRangeSet& core_range_set,
    const uint32_t& page_size_bytes) {
    const uint32_t cb_id{static_cast<uint32_t>(scatter_cb)};
    const auto cb_data_format{datatype_to_dataformat_converter(dtype)};
    const auto cb_config{
        CircularBufferConfig{page_size_bytes, {{cb_id, cb_data_format}}}.set_page_size(cb_id, page_size_bytes)};
    return CreateCircularBuffer(program, core_range_set, cb_config);
}

KernelHandle ScatterProgramFactory::create_kernel(
    Program& program,
    const char* kernel_path,
    const CoreRangeSet& core_range_set,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config,
    const std::vector<uint32_t>& runtime_args) {
    auto kernel_id{CreateKernel(program, kernel_path, core_range_set, config)};

    if (!runtime_args.empty()) {
        SetRuntimeArgs(program, kernel_id, core_range_set, runtime_args);
    }

    return kernel_id;
}

}  // namespace ttnn::operations::experimental::scatter
