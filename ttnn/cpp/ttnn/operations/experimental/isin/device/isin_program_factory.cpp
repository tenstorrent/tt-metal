// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "isin_program_factory.hpp"

#include "../isin_common.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/buffer_types.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/work_split.hpp"
#include "ttnn/operations/experimental/isin/device/isin_device_op.hpp"
#include "ttnn/operations/experimental/isin/isin_cbs.hpp"

namespace ttnn::operations::experimental::isin {

using namespace common;

namespace {
constexpr uint32_t BIT_MASK_32 = 32 - 1;

inline uint64_t ceil32(const uint64_t& number) {
    return ((number & BIT_MASK_32) == 0) ? number : ((number | BIT_MASK_32) + 1);
}

inline bool is_pow2_min32(const uint64_t& number) { return ((number & (number - 1)) == 0) && number >= 32; }

// inline std::vector<CoreCoord> get_worker_cores(const CoreCoord& compute_grid_size) {
//     //
//     // const CoreRange all_cores_range{
//     //     CoreCoord(0, 0), CoreCoord(compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1)};
//     // const CoreRangeSet all_cores = std::set<CoreRange>({all_cores_range});
// }

}  // namespace

IsInProgramFactory::cached_program_t IsInProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    using namespace tt::tt_metal;
    using namespace tt::constants;

    Program program{};

    const auto& elements_tensor = tensor_args.elements_tensor;
    const auto& test_elements_tensor = tensor_args.test_elements_tensor;

    const auto& elements_dtype = elements_tensor.dtype();
    const auto& test_elements_dtype = test_elements_tensor.dtype();

    const bool& assume_unique = args.assume_unique;
    const bool& invert = args.invert;
    const uint32_t& single_fetch_subchunk_size = args.single_fetch_subchunk_size;
    const auto memory_config = args.memory_config.has_value() ? (*args.memory_config) : elements_tensor.memory_config();

    const auto& elements_buffer = elements_tensor.buffer();
    const auto& test_elements_buffer = test_elements_tensor.buffer();
    const auto& output_buffer = output_tensor.buffer();

    const auto elements_tensor_buffer_address = elements_buffer->address();
    const auto test_elements_tensor_buffer_address = test_elements_buffer->address();
    const auto output_tensor_buffer_address = output_buffer->address();

    // input dtype byte sizes
    const uint32_t& elements_datum_size = elements_tensor.element_size();
    const uint32_t& test_elements_datum_size = test_elements_tensor.element_size();
    const uint32_t& output_datum_size = output_tensor.element_size();

    // input row byte sizes
    const uint32_t& elements_subchunk_size_bytes = single_fetch_subchunk_size * elements_datum_size;
    const uint32_t& test_elements_subchunk_size_bytes = single_fetch_subchunk_size * test_elements_datum_size;
    const uint32_t& output_subchunk_size_bytes = single_fetch_subchunk_size * output_datum_size;

    auto* device = elements_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    std::vector<uint32_t> compile_time_args{
        elements_tensor_buffer_address,
        test_elements_tensor_buffer_address,
        output_tensor_buffer_address,
        static_cast<uint32_t>(IsInCB::ELEMENTS),
        static_cast<uint32_t>(IsInCB::TEST_ELEMENTS),
        static_cast<uint32_t>(IsInCB::OUTPUT),
        elements_tensor.logical_volume(),
        test_elements_tensor.logical_volume(),
        single_fetch_subchunk_size,
        static_cast<uint32_t>(invert)};
    TensorAccessorArgs(*elements_buffer).append_to(compile_time_args);
    TensorAccessorArgs(*test_elements_buffer).append_to(compile_time_args);
    TensorAccessorArgs(*output_buffer).append_to(compile_time_args);

    const uint32_t subchunks_num =
        (elements_tensor.logical_volume() + single_fetch_subchunk_size - 1) / single_fetch_subchunk_size;
    const auto core_grid = device->compute_with_storage_grid_size();
    const auto
        [num_cores,                       // number of cores utilized
         all_cores,                       // set of all cores used
         core_group_1,                    // Primary core group
         core_group_2,                    // Secondary core group
         num_subchunks_per_core_group_1,  // Number of subchunks each core in the primary group processes
         num_subchunks_per_core_group_2   // Number of subchunks each core in the secondary group processes
    ] = split_work_to_cores(core_grid, subchunks_num);
    //
    create_cb(program, elements_dtype, IsInCB::ELEMENTS, all_cores, elements_subchunk_size_bytes);
    create_cb(program, test_elements_dtype, IsInCB::TEST_ELEMENTS, all_cores, test_elements_subchunk_size_bytes);
    create_cb(program, OUTPUT_TENSOR_DATA_TYPE, IsInCB::OUTPUT, all_cores, output_subchunk_size_bytes);

    auto reader_kernel_id =
        create_kernel(program, READER_KERNEL_PATH, all_cores, ReaderDataMovementConfig{compile_time_args});
    auto writer_kernel_id =
        create_kernel(program, WRITER_KERNEL_PATH, all_cores, WriterDataMovementConfig{compile_time_args});

    uint32_t subchunks_offset = 0;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core{i / num_cores_y, i % num_cores_y};
        uint32_t subchunks_per_core;
        if (core_group_1.contains(core)) {
            subchunks_per_core = num_subchunks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            subchunks_per_core = num_subchunks_per_core_group_2;
        } else {
            TT_THROW("Core not in any predefined core range.");
        }

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {elements_tensor_buffer_address,
             test_elements_tensor_buffer_address,
             subchunks_per_core,
             subchunks_offset});
        SetRuntimeArgs(
            program, writer_kernel_id, core, {output_tensor_buffer_address, subchunks_per_core, subchunks_offset});

        subchunks_offset += subchunks_per_core;
    }

    auto cores = grid_to_cores(num_cores, compute_with_storage_grid_size.x, num_cores_y);
    return {std::move(program), {reader_kernel_id, writer_kernel_id, cores}};
}

void IsInProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;

    auto input_buffer_address = tensor_args.elements_tensor.buffer()->address();
    auto test_elements_buffer_address = tensor_args.test_elements_tensor.buffer()->address();
    auto output_buffer_address = output_tensor.buffer()->address();
    for (const auto& core : cores) {
        auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
        auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
        reader_runtime_args[0] = input_buffer_address;
        reader_runtime_args[1] = test_elements_buffer_address;
        writer_runtime_args[0] = output_buffer_address;
    }
}

CBHandle IsInProgramFactory::create_cb(
    Program& program,
    const DataType& dtype,
    const IsInCB& is_in_cb,
    const CoreRangeSet& core_range_set,
    const uint32_t& page_size_bytes) {
    const uint32_t cb_id{static_cast<uint32_t>(is_in_cb)};
    const auto cb_data_format{datatype_to_dataformat_converter(dtype)};
    const auto cb_config{
        CircularBufferConfig{page_size_bytes, {{cb_id, cb_data_format}}}.set_page_size(cb_id, page_size_bytes)};
    return CreateCircularBuffer(program, core_range_set, cb_config);
}

KernelHandle IsInProgramFactory::create_kernel(
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

}  // namespace ttnn::operations::experimental::isin
