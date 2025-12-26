// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unique_program_factory.hpp"

#include "../unique_common.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/buffer_types.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/work_split.hpp"
#include "ttnn/operations/experimental/unique/device/unique_device_op.hpp"
#include "ttnn/operations/experimental/unique/unique_cbs.hpp"

namespace ttnn::operations::experimental::unique {

using namespace common;

namespace {
constexpr uint32_t BIT_MASK_64 = 64 - 1;

inline uint64_t ceil64(const uint64_t& number) {
    return ((number & BIT_MASK_64) == 0) ? number : ((number | BIT_MASK_64) + 1);
}

}  // namespace

UniqueProgramFactory::cached_program_t UniqueProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensors) {
    using namespace tt::tt_metal;
    using namespace tt::constants;

    Program program{};

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& first_occurrences_tensor = tensor_args.first_occurrences_tensor;

    const auto& input_dtype = input_tensor.dtype();

    // const auto& dim = args.dim;
    const uint32_t& single_fetch_subchunk_size = args.single_fetch_subchunk_size;
    const auto memory_config = args.memory_config.has_value() ? (*args.memory_config) : input_tensor.memory_config();

    const auto& input_buffer = input_tensor.buffer();
    const auto& first_occurrences_buffer = first_occurrences_tensor.buffer();
    const auto& output_buffer = output_tensors[0].buffer();
    const auto& output_size_buffer = output_tensors[1].buffer();

    const auto input_tensor_buffer_address = input_buffer->address();
    const auto first_occurrences_tensor_buffer_address = first_occurrences_buffer->address();
    const auto output_tensor_buffer_address = output_buffer->address();
    const auto output_size_tensor_buffer_address = output_size_buffer->address();

    // input dtype byte sizes
    const uint32_t& input_datum_size = input_tensor.element_size();
    const uint32_t& first_occurrences_datum_size = input_tensor.element_size();
    const uint32_t& output_datum_size = output_tensors[0].element_size();

    // input row byte sizes
    const uint32_t& input_subchunk_size_bytes = ceil64(single_fetch_subchunk_size * input_datum_size);
    const uint32_t& first_occurrences_subchunk_size_bytes =
        ceil64(single_fetch_subchunk_size * first_occurrences_datum_size);
    const uint32_t& output_subchunk_size_bytes = ceil64(single_fetch_subchunk_size * output_datum_size);

    auto* device = input_tensor.device();
    // const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    std::vector<uint32_t> compile_time_args{
        input_tensor_buffer_address,
        first_occurrences_tensor_buffer_address,
        output_tensor_buffer_address,
        output_size_tensor_buffer_address,
        static_cast<uint32_t>(UniqueCB::INPUT),
        static_cast<uint32_t>(UniqueCB::INPUT_COMPARE),
        static_cast<uint32_t>(UniqueCB::FIRST_OCCURRENCES_READ),
        static_cast<uint32_t>(UniqueCB::FIRST_OCCURRENCES_WRITE),
        static_cast<uint32_t>(UniqueCB::FIRST_OCCURRENCES_OUTPUT),
        static_cast<uint32_t>(UniqueCB::RESULT_ACC),
        static_cast<uint32_t>(UniqueCB::OUTPUT),
        static_cast<uint32_t>(UniqueCB::OUTPUT_SIZE),
        input_tensor.logical_volume(),
        single_fetch_subchunk_size};
    TensorAccessorArgs(*input_buffer).append_to(compile_time_args);
    TensorAccessorArgs(*first_occurrences_buffer).append_to(compile_time_args);
    TensorAccessorArgs(*output_buffer).append_to(compile_time_args);
    TensorAccessorArgs(*output_size_buffer).append_to(compile_time_args);

    const uint32_t subchunks_num =
        input_tensor.logical_volume() / input_tensor.logical_shape()[input_tensor.logical_shape().rank() - 1];
    const auto core_grid = device->compute_with_storage_grid_size();
    const auto
        [num_cores,                       // number of cores utilized
         all_cores,                       // set of all cores used
         core_group_1,                    // Primary core group
         core_group_2,                    // Secondary core group
         num_subchunks_per_core_group_1,  // Number of subchunks each core in the primary group processes
         num_subchunks_per_core_group_2   // Number of subchunks each core in the secondary group processes
    ] = split_work_to_cores(core_grid, subchunks_num);

    const auto single_core = CoreCoord{0, 0};
    const auto core_range_set = CoreRangeSet{single_core};
    create_cb(program, input_dtype, UniqueCB::INPUT, core_range_set, input_subchunk_size_bytes);
    create_cb(program, input_dtype, UniqueCB::INPUT_COMPARE, core_range_set, input_subchunk_size_bytes);
    create_cb(
        program,
        FIRST_OCCURRENCES_TENSOR_DATA_TYPE,
        UniqueCB::FIRST_OCCURRENCES_READ,
        core_range_set,
        first_occurrences_subchunk_size_bytes);
    create_cb(
        program,
        FIRST_OCCURRENCES_TENSOR_DATA_TYPE,
        UniqueCB::FIRST_OCCURRENCES_WRITE,
        core_range_set,
        first_occurrences_subchunk_size_bytes);
    create_cb(
        program,
        FIRST_OCCURRENCES_TENSOR_DATA_TYPE,
        UniqueCB::FIRST_OCCURRENCES_OUTPUT,
        core_range_set,
        first_occurrences_subchunk_size_bytes);
    create_cb(program, input_dtype, UniqueCB::RESULT_ACC, core_range_set, input_subchunk_size_bytes);
    create_cb(program, input_dtype, UniqueCB::OUTPUT, core_range_set, output_subchunk_size_bytes);
    create_cb(
        program,
        FIRST_OCCURRENCES_TENSOR_DATA_TYPE,
        UniqueCB::OUTPUT_SIZE,
        core_range_set,
        OUTPUT_SIZE_TENSOR_SIZE * sizeof(uint32_t));

    const uint32_t subchunks_offset = 0;

    auto reader_kernel_id = create_kernel(
        program,
        READER_KERNEL_PATH,
        core_range_set,
        ReaderDataMovementConfig{compile_time_args},
        {input_tensor_buffer_address, first_occurrences_tensor_buffer_address, subchunks_num, subchunks_offset});
    auto writer_kernel_id = create_kernel(
        program,
        WRITER_KERNEL_PATH,
        core_range_set,
        WriterDataMovementConfig{compile_time_args},
        {first_occurrences_tensor_buffer_address,
         output_tensor_buffer_address,
         output_size_tensor_buffer_address,
         subchunks_num,
         subchunks_offset});

    // SetRuntimeArgs(
    //     program,
    //     reader_kernel_id,
    //     core_range_set,
    //     {input_tensor_buffer_address, first_occurrences_tensor_buffer_address, subchunks_num, subchunks_offset});
    // SetRuntimeArgs(
    //     program,
    //     writer_kernel_id,
    //     core_range_set,
    //     {first_occurrences_tensor_buffer_address,
    //      output_tensor_buffer_address,
    //      output_size_tensor_buffer_address,
    //      subchunks_num,
    //      subchunks_offset});

    return {std::move(program), {reader_kernel_id, writer_kernel_id, {{0, 0}, {0, 0}}}};
}

void UniqueProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensors) {
    const auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;

    auto input_buffer_address = tensor_args.input_tensor.buffer()->address();
    auto first_occurrences_buffer_address = tensor_args.first_occurrences_tensor.buffer()->address();
    auto output_buffer_address = output_tensors[0].buffer()->address();
    auto output_size_buffer_address = output_tensors[1].buffer()->address();
    for (const auto& core : cores) {
        auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
        auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
        reader_runtime_args[0] = input_buffer_address;
        reader_runtime_args[1] = first_occurrences_buffer_address;
        writer_runtime_args[0] = first_occurrences_buffer_address;
        writer_runtime_args[1] = output_buffer_address;
        writer_runtime_args[2] = output_size_buffer_address;
    }
}

CBHandle UniqueProgramFactory::create_cb(
    Program& program,
    const DataType& dtype,
    const UniqueCB& is_in_cb,
    const CoreRangeSet& core_range_set,
    const uint32_t& page_size_bytes) {
    const uint32_t cb_id{static_cast<uint32_t>(is_in_cb)};
    const auto cb_data_format{datatype_to_dataformat_converter(dtype)};
    const auto cb_config{
        CircularBufferConfig{page_size_bytes, {{cb_id, cb_data_format}}}.set_page_size(cb_id, page_size_bytes)};
    return CreateCircularBuffer(program, core_range_set, cb_config);
}

KernelHandle UniqueProgramFactory::create_kernel(
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

}  // namespace ttnn::operations::experimental::unique
