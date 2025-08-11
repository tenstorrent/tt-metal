// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "isunique_program_factory.hpp"

#include "../isunique_common.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/buffer_types.hpp"

namespace ttnn::operations::experimental::isunique {

using namespace common;

namespace {
constexpr uint32_t BIT_MASK_32 = 32 - 1;

inline uint64_t ceil32(const uint64_t& number) {
    return ((number & BIT_MASK_32) == 0) ? number : ((number | BIT_MASK_32) + 1);
}

inline bool is_pow2_min32(const uint64_t& number) { return ((number & (number - 1)) == 0) && number >= 32; }

inline std::vector<CoreCoord> get_worker_cores(
    const OptimalIsUniqueConf& optimal_is_unique_conf, const CoreCoord& compute_grid_size) {
    //
    // const CoreRange all_cores_range{
    //     CoreCoord(0, 0), CoreCoord(compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1)};
    // const CoreRangeSet all_cores = std::set<CoreRange>({all_cores_range});
}

}  // namespace

IsUniqueProgramFactory::cached_program_t IsUniqueProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    using namespace tt::tt_metal;
    using namespace tt::constants;

    Program program{};

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.logical_shape();
    const auto& input_dtype = input_tensor.dtype();
    const auto& index_hint_tensor = tensor_args.index_hint_tensor;
    const auto& first_occurrences_tensor = tensor_args.first_occurrences_tensor;
    const auto& first_occurrences_tensor_engaged = first_occurrences_tensor.has_value();

    const bool& invert = args.invert;
    const auto& dim = args.dim.value_or(FIRST_DIMENSION);
    const auto& optimal_isunique_conf = args.optimal_isunique_conf;
    const auto memory_config = args.memory_config.has_value() ? (*args.memory_config) : input_tensor.memory_config();

    const auto& input_buffer = input_tensor.buffer();
    const auto& index_hint_buffer = index_hint_tensor.buffer();
    const auto& output_buffer = output_tensor.buffer();

    const bool is_input_dram = input_buffer->buffer_type() == BufferType::DRAM;
    const bool is_index_hint_dram = index_hint_buffer->buffer_type() == BufferType::DRAM;
    const bool is_first_occurrences_dram = first_occurrences_tensor_engaged
                                               ? (first_occurrences_tensor->buffer()->buffer_type() == BufferType::DRAM)
                                               : false;
    const bool is_output_dram = output_buffer->buffer_type() == BufferType::DRAM;

    const auto input_tensor_buffer_address = input_buffer->address();
    const auto index_hint_tensor_buffer_address = index_hint_buffer->address();
    const auto first_occurrences_tensor_buffer_address =
        first_occurrences_tensor_engaged ? first_occurrences_tensor->buffer()->address() : -1;
    const auto output_tensor_buffer_address = output_buffer->address();

    // input dtype byte sizes
    const uint32_t& input_datum_size = input_tensor.element_size();
    const uint32_t& index_hint_datum_size = index_hint_tensor.element_size();
    const uint32_t& first_occurrences_datum_size =
        first_occurrences_tensor_engaged ? first_occurrences_tensor->element_size() : 0;
    const uint32_t& output_datum_size = output_tensor.element_size();

    const uint32_t& stick_size = optimal_isunique_conf.best_core_data_arrangement.row_size;
    const uint32_t& num_rows = optimal_isunique_conf.best_core_data_arrangement.num_rows;
    const uint32_t& num_cores = optimal_isunique_conf.best_core_data_arrangement.num_cores;

    // input row byte sizes
    const uint32_t& input_stick_size_bytes = stick_size * input_datum_size;
    const uint32_t& index_hint_stick_size_bytes = stick_size * index_hint_datum_size;
    const uint32_t& first_occurrences_stick_size_bytes = stick_size * first_occurrences_datum_size;
    const uint32_t& output_stick_size_bytes = stick_size * output_datum_size;

    // check if row byte sizes are at least 32 and a power of 2 (for InterleavedAddrGen)
    // const uint32_t is_input_stick_size_bytes_pow2_min_32 = is_pow2_min32(input_stick_size_bytes);
    // const uint32_t is_index_hint_stick_size_bytes_pow2_min_32 = is_pow2_min32(index_hint_stick_size_bytes);
    // const uint32_t is_first_occurrences_stick_size_bytes_pow2_min_32 =
    // is_pow2_min32(first_occurrences_stick_size_bytes); const uint32_t is_output_stick_size_bytes_pow2_min_32 =
    // is_pow2_min32(output_stick_size_bytes);

    // for InterleavedAddrGen
    // const uint32_t input_stick_size_bytes_log2 =
    //     is_input_stick_size_bytes_pow2_min_32 ? std::log2(input_stick_size_bytes) : 0;
    // const uint32_t index_hint_stick_size_bytes_log2 =
    //     is_index_hint_stick_size_bytes_pow2_min_32 ? std::log2(index_hint_stick_size_bytes) : 0;
    // const uint32_t first_occurrences_stick_size_bytes_log2 =
    //     is_first_occurrences_stick_size_bytes_pow2_min_32 ? std::log2(first_occurrences_stick_size_bytes) : 0;
    // const uint32_t output_stick_size_bytes_log2 =
    //     is_output_stick_size_bytes_pow2_min_32 ? std::log2(output_stick_size_bytes) : 0;

    auto* device = input_tensor.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const auto worker_cores = get_worker_cores(optimal_isunique_conf, compute_with_storage_grid_size);
    const auto worker_cores_range = CoreRangeSet{worker_cores};
    // const CoreRange all_cores_range{
    //     CoreCoord(0, 0), CoreCoord(compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1)};
    // const CoreRangeSet all_cores = std::set<CoreRange>({all_cores_range});

    create_cb(program, input_dtype, IsUniqueCB::INPUT, worker_cores_range, input_stick_size_bytes);

    create_cb(
        program,
        PREDEFINED_TENSOR_DTYPES.at(IsUniqueCB::INDEX_HINT),
        IsUniqueCB::INDEX_HINT,
        worker_cores_range,
        index_hint_stick_size_bytes);

    if (first_occurrences_tensor_engaged) {
        create_cb(
            program,
            PREDEFINED_TENSOR_DTYPES.at(IsUniqueCB::FIRST_OCCURRENCES),
            IsUniqueCB::FIRST_OCCURRENCES,
            worker_cores_range,
            first_occurrences_stick_size_bytes);
    }

    create_cb(
        program,
        PREDEFINED_TENSOR_DTYPES.at(IsUniqueCB::OUTPUT),
        IsUniqueCB::OUTPUT,
        worker_cores_range,
        output_stick_size_bytes);

    const std::vector<uint32_t> compile_time_args{
        is_input_dram,
        is_index_hint_dram,
        is_first_occurrences_dram,
        is_output_dram,
        input_tensor_buffer_address,
        index_hint_tensor_buffer_address,
        first_occurrences_tensor_buffer_address,
        output_tensor_buffer_address,
        static_cast<std::decay_t<std::underlying_type_t<IsUniqueCB>>>(IsUniqueCB::INPUT),
        static_cast<std::decay_t<std::underlying_type_t<IsUniqueCB>>>(IsUniqueCB::INDEX_HINT),
        static_cast<std::decay_t<std::underlying_type_t<IsUniqueCB>>>(IsUniqueCB::FIRST_OCCURRENCES),
        static_cast<std::decay_t<std::underlying_type_t<IsUniqueCB>>>(IsUniqueCB::OUTPUT),
        stick_size,
        input_stick_size_bytes,
        index_hint_stick_size_bytes,
        first_occurrences_stick_size_bytes,
        output_stick_size_bytes,
        static_cast<uint32_t>(first_occurrences_tensor_engaged),
        num_rows,
        static_cast<uint32_t>(invert),
        dim,
        num_cores,
    };
    TensorAccessorArgs(*input_buffer).append_to(compile_time_args);
    TensorAccessorArgs(*index_hint_buffer).append_to(compile_time_args);
    if (first_occurrences_tensor_engaged) {
        TensorAccessorArgs(*(first_occurrences_tensor->buffer())).append_to(compile_time_args);
    }
    tt::tt_metal::TensorAccessorArgs(*output_buffer).append_to(compile_time_args);

    auto reader_kernel =
        create_kernel(program, READER_KERNEL_PATH, worker_cores_range, ReaderDataMovementConfig{compile_time_args});
    auto writer_kernel =
        create_kernel(program, WRITER_KERNEL_PATH, worker_cores_range, WriterDataMovementConfig{compile_time_args});

    // const uint32_t& num_cores_x = compute_with_storage_grid_size.x;
    // const uint32_t& num_cores_y = compute_with_storage_grid_size.y;
    // auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y);
    // uint32_t stick_offset = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core{i / num_cores_y, i % num_cores_y};

        //     uint32_t sticks_per_core;
        //     if (core_group_1.contains(core)) {
        //         sticks_per_core = num_sticks_per_core_group_1;
        //     } else if (core_group_2.contains(core)) {
        //         sticks_per_core = num_sticks_per_core_group_2;
        //     } else {
        //         TT_THROW("Core not in any predefined core range.");
        //     }

        //     SetRuntimeArgs(
        //         program,
        //         reader_kernel,
        //         core,
        //         {input_buffer->address(), index_buffer->address(), src_buffer->address(), stick_offset,
        //         sticks_per_core, input_and_output_chunk_size, index_and_source_chunk_size});

        //     SetRuntimeArgs(program, writer_kernel, core, {output_buffer->address(), stick_offset, sticks_per_core,
        //     input_and_output_chunk_size});

        //     stick_offset += sticks_per_core;
    }

    return {std::move(program), {reader_kernel, writer_kernel, worker_cores}};
}

}  // namespace ttnn::operations::experimental::isunique
