// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "where_device_operation.hpp"
#include "where_utils.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include <cmath>

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using namespace ttnn::operations::ternary;

template <typename F>
void set_or_update_runtime_arguments(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    tt::tt_metal::KernelHandle compute_kernel_id,
    CoreCoord compute_with_storage_grid_size,
    const WhereDeviceOperation::operation_attributes_t& operation_attributes,
    const WhereDeviceOperation::tensor_args_t& tensor_args,
    WhereDeviceOperation::tensor_return_value_t& output,
    F handle_args) {
    const auto& [predicate_tensor, value_true_tensor, value_false_tensor, optional_output_tensor] = tensor_args;

    WhereVariant variant = operation_attributes.where_variant;

    uint32_t num_output_tiles = output.physical_volume() / output.tensor_spec().tile().get_tile_hw();

    constexpr bool row_major = true;
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);
    constexpr size_t num_writer_args = 3;
    constexpr size_t num_kernel_args = 1;

    // Reader args count depends on variant
    constexpr size_t num_reader_args = 5;
    uint32_t dummy_arg = 0;

    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            handle_args(program, reader_kernel_id, core, std::array<uint32_t, num_reader_args>{0});
            handle_args(program, writer_kernel_id, core, std::array<uint32_t, num_writer_args>{0});
            handle_args(program, compute_kernel_id, core, std::array<uint32_t, num_kernel_args>{0});
            continue;
        }

        // Set reader runtime arguments based on variant
        if (variant == WhereVariant::TTS) {
            // TTS: predicate (arg 0) + value_true tensor (arg 1)
            std::array reader_runtime_args = {
                predicate_tensor.buffer()->address(),
                value_true_tensor.value().buffer()->address(),
                dummy_arg,
                num_tiles_per_core,
                start_tile_id,
            };
            handle_args(program, reader_kernel_id, core, reader_runtime_args);
        } else if (variant == WhereVariant::TST) {
            // TST: predicate (arg 0) + value_false tensor (arg 1, maps to c_1)
            std::array reader_runtime_args = {
                predicate_tensor.buffer()->address(),
                value_false_tensor.value().buffer()->address(),
                dummy_arg,
                num_tiles_per_core,
                start_tile_id,
            };
            handle_args(program, reader_kernel_id, core, reader_runtime_args);
        } else {
            // TTT: predicate (arg 0) + value_true (arg 1) + value_false (arg 2)
            std::array reader_runtime_args = {
                predicate_tensor.buffer()->address(),
                value_true_tensor.value().buffer()->address(),
                value_false_tensor.value().buffer()->address(),
                num_tiles_per_core,
                start_tile_id,
            };
            handle_args(program, reader_kernel_id, core, reader_runtime_args);
        }

        std::array writer_runtime_args = {
            output.buffer()->address(),
            num_tiles_per_core,
            start_tile_id,
        };
        handle_args(program, writer_kernel_id, core, writer_runtime_args);

        // All variants use same compute runtime args now
        if (variant == WhereVariant::TTS) {
            auto bit_cast_scalar =
                pack_scalar_runtime_arg(operation_attributes.value_false_scalar.value(), output.dtype());
            std::array compute_runtime_args = {num_tiles_per_core, bit_cast_scalar};
            handle_args(program, compute_kernel_id, core, compute_runtime_args);
        } else if (variant == WhereVariant::TST) {
            auto bit_cast_scalar =
                pack_scalar_runtime_arg(operation_attributes.value_true_scalar.value(), output.dtype());
            std::array compute_runtime_args = {num_tiles_per_core, bit_cast_scalar};
            handle_args(program, compute_kernel_id, core, compute_runtime_args);
        } else {
            std::array compute_runtime_args = {num_tiles_per_core};
            handle_args(program, compute_kernel_id, core, compute_runtime_args);
        }

        start_tile_id += num_tiles_per_core;
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::ternary {
WhereDeviceOperation::WhereProgramFactory::cached_program_t WhereDeviceOperation::WhereProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& [predicate_tensor, value_true_tensor, value_false_tensor, optional_output_tensor] = tensor_args;

    WhereVariant variant = operation_attributes.where_variant;

    // Use WhereKernelConfig to get the appropriate kernel names
    WhereKernelConfig kernel_config(variant);

    auto program = CreateProgram();

    auto* device = predicate_tensor.device();

    auto predicate_data_format = datatype_to_dataformat_converter(predicate_tensor.dtype());
    // (predicate_tensor.dtype() == DataType::BFLOAT16) ? DataType::UINT16 : predicate_tensor.dtype());

    // Handle data formats based on variant and tensor availability
    DataFormat value_true_data_format, value_false_data_format;
    if (variant == WhereVariant::TTS) {
        // TTS: only value_true tensor exists
        value_true_data_format = datatype_to_dataformat_converter(value_true_tensor.value().dtype());

        // the bfloat16 impl of where_llk uses UINT16 instr set.
        // If the bfloat16 inputs' CBs are set to UINT16 dataformat this will enable us to get 'NaN' for bfloat16 dtype
        // We need to test the impact of this on the composite ops that use where op and on the models, since bfloat16
        // packs nan as inf in all other ops.

        // (value_true_tensor.value().dtype() == DataType::BFLOAT16) ? DataType::UINT16
        //                                                           : value_true_tensor.value().dtype());

        // Use predicate format as fallback for value_false
        value_false_data_format = predicate_data_format;
    } else if (variant == WhereVariant::TST) {
        // TST: only value_false tensor exists
        value_false_data_format = datatype_to_dataformat_converter(value_false_tensor.value().dtype());
        // (value_false_tensor.value().dtype() == DataType::BFLOAT16) ? DataType::UINT16
        //                                                            : value_false_tensor.value().dtype());
        // Use predicate format as fallback for value_true
        value_true_data_format = predicate_data_format;
    } else {
        // TTT: both tensors exist
        value_true_data_format = datatype_to_dataformat_converter(value_true_tensor.value().dtype());
        // (value_true_tensor.value().dtype() == DataType::BFLOAT16) ? DataType::UINT16
        //                                                           : value_true_tensor.value().dtype());
        value_false_data_format = datatype_to_dataformat_converter(value_false_tensor.value().dtype());
        // (value_false_tensor.value().dtype() == DataType::BFLOAT16) ? DataType::UINT16
        //                                                            : value_false_tensor.value().dtype());
    }

    auto output_data_format = datatype_to_dataformat_converter(output.dtype());
    // datatype_to_dataformat_converter((output.dtype() == DataType::BFLOAT16) ? DataType::UINT16 : output.dtype());

    uint32_t predicate_single_tile_size = tt_metal::detail::TileSize(predicate_data_format);
    uint32_t value_true_single_tile_size = tt_metal::detail::TileSize(value_true_data_format);
    uint32_t value_false_single_tile_size = tt_metal::detail::TileSize(value_false_data_format);
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_data_format);

    // we parallelize the computation across the output tiles
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // Number of tiles to store per input CB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;

    // Input buffers - Create predicate CB (always c_0)
    auto [predicate_tensor_cb, predicate_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_0,
        program,
        all_device_cores,
        predicate_single_tile_size,
        num_tiles_per_cb,
        predicate_data_format);  // predicate_tensor

    // Create c_1 based on variant - this is the primary tensor CB
    uint32_t value_true_tensor_cb = 0;
    tt::tt_metal::CBHandle value_true_tensor_cb_handle;
    uint32_t value_false_tensor_cb = 0;
    tt::tt_metal::CBHandle value_false_tensor_cb_handle;

    if (variant == WhereVariant::TTS) {
        // TTS: c_1 = value_true tensor (value_false is scalar)
        auto [cb, cb_handle] = create_cb(
            tt::CBIndex::c_1,
            program,
            all_device_cores,
            value_true_single_tile_size,
            num_tiles_per_cb,
            value_true_data_format);
        value_true_tensor_cb = cb;
        value_true_tensor_cb_handle = cb_handle;
    } else if (variant == WhereVariant::TST) {
        // TST: c_1 = value_false tensor (value_true is scalar)
        auto [cb, cb_handle] = create_cb(
            tt::CBIndex::c_1,
            program,
            all_device_cores,
            value_false_single_tile_size,
            num_tiles_per_cb,
            value_false_data_format);
        value_false_tensor_cb = cb;
        value_false_tensor_cb_handle = cb_handle;
    } else {
        // TTT: c_1 = value_true tensor, c_2 = value_false tensor
        auto [cb1, cb1_handle] = create_cb(
            tt::CBIndex::c_1,
            program,
            all_device_cores,
            value_true_single_tile_size,
            num_tiles_per_cb,
            value_true_data_format);
        value_true_tensor_cb = cb1;
        value_true_tensor_cb_handle = cb1_handle;

        auto [cb2, cb2_handle] = create_cb(
            tt::CBIndex::c_2,
            program,
            all_device_cores,
            value_false_single_tile_size,
            num_tiles_per_cb,
            value_false_data_format);
        value_false_tensor_cb = cb2;
        value_false_tensor_cb_handle = cb2_handle;
    }

    // Output buffer
    auto [output_tensor_cb, output_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_3,
        program,
        all_device_cores,
        output_single_tile_size,
        num_tiles_per_cb,
        output_data_format);  // output

    // READER KERNEL - Use kernel path from utils
    tt_metal::ReaderDataMovementConfig reader_config;
    if (variant == WhereVariant::TTS) {
        // TTS: c_0 = predicate, c_1 = value_true tensor
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t)predicate_tensor_cb, (std::uint32_t)value_true_tensor_cb};
        TensorAccessorArgs(*predicate_tensor.buffer()).append_to(reader_compile_time_args);
        TensorAccessorArgs(*value_true_tensor.value().buffer()).append_to(reader_compile_time_args);
        reader_config = tt_metal::ReaderDataMovementConfig(reader_compile_time_args);

    } else if (variant == WhereVariant::TST) {
        // TST: c_0 = predicate, c_1 = value_false tensor
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t)predicate_tensor_cb, (std::uint32_t)value_false_tensor_cb};
        TensorAccessorArgs(*predicate_tensor.buffer()).append_to(reader_compile_time_args);
        TensorAccessorArgs(*value_false_tensor.value().buffer()).append_to(reader_compile_time_args);
        reader_config = tt_metal::ReaderDataMovementConfig(reader_compile_time_args);

    } else {
        // TTT: c_0 = predicate, c_1 = value_true, c_2 = value_false
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t)predicate_tensor_cb,
            (std::uint32_t)value_true_tensor_cb,
            (std::uint32_t)value_false_tensor_cb};
        TensorAccessorArgs(*predicate_tensor.buffer()).append_to(reader_compile_time_args);
        TensorAccessorArgs(*value_true_tensor.value().buffer()).append_to(reader_compile_time_args);
        TensorAccessorArgs(*value_false_tensor.value().buffer()).append_to(reader_compile_time_args);
        reader_config = tt_metal::ReaderDataMovementConfig(reader_compile_time_args);
    }

    auto reader_kernel_id = tt_metal::CreateKernel(
        program, get_kernel_file_path(kernel_config.reader_kernel), all_device_cores, reader_config);

    // WRITER KERNEL - Use kernel path from utils
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_tensor_cb};
    tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        get_kernel_file_path(kernel_config.writer_kernel),
        all_device_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // COMPUTE KERNEL - Use kernel path from utils
    bool fp32_dest_acc_en = output_data_format == tt::DataFormat::UInt32 ||
                            output_data_format == tt::DataFormat::Int32 ||
                            output_data_format == tt::DataFormat::Float32;

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);

    // c_0 is always predicate
    unpack_to_dest_mode[tt::CBIndex::c_0] = (predicate_tensor.dtype() == DataType::FLOAT32)
                                                ? UnpackToDestMode::UnpackToDestFp32
                                                : UnpackToDestMode::Default;

    // c_1 assignment depends on variant
    if (variant == WhereVariant::TTS) {
        // TTS: c_1 = value_true tensor
        unpack_to_dest_mode[tt::CBIndex::c_1] = (value_true_tensor.value().dtype() == DataType::FLOAT32)
                                                    ? UnpackToDestMode::UnpackToDestFp32
                                                    : UnpackToDestMode::Default;
    } else if (variant == WhereVariant::TST) {
        // TST: c_1 = value_false tensor
        unpack_to_dest_mode[tt::CBIndex::c_1] = (value_false_tensor.value().dtype() == DataType::FLOAT32)
                                                    ? UnpackToDestMode::UnpackToDestFp32
                                                    : UnpackToDestMode::Default;
    } else {
        // TTT: c_1 = value_true tensor, c_2 = value_false tensor
        unpack_to_dest_mode[tt::CBIndex::c_1] = (value_true_tensor.value().dtype() == DataType::FLOAT32)
                                                    ? UnpackToDestMode::UnpackToDestFp32
                                                    : UnpackToDestMode::Default;
        unpack_to_dest_mode[tt::CBIndex::c_2] = (value_false_tensor.value().dtype() == DataType::FLOAT32)
                                                    ? UnpackToDestMode::UnpackToDestFp32
                                                    : UnpackToDestMode::Default;
    }

    // c_3 is always output
    unpack_to_dest_mode[tt::CBIndex::c_3] =
        (output.dtype() == DataType::FLOAT32) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;

    constexpr uint32_t num_tiles_per_cycle = 1;  // we produce 1 output tile per read-compute-write cycle

    // All variants use the same compile args now
    std::vector<uint32_t> compute_kernel_args;
    if (variant == WhereVariant::TTS) {
        auto bit_cast_scalar = pack_scalar_runtime_arg(operation_attributes.value_false_scalar.value(), output.dtype());
        compute_kernel_args = {num_tiles_per_cycle, bit_cast_scalar};
    } else if (variant == WhereVariant::TST) {
        auto bit_cast_scalar = pack_scalar_runtime_arg(operation_attributes.value_true_scalar.value(), output.dtype());
        compute_kernel_args = {num_tiles_per_cycle, bit_cast_scalar};
    } else {
        compute_kernel_args = {num_tiles_per_cycle};
    }

    std::map<std::string, std::string> kernel_defines;
    kernel_defines["WHERE_LLK"] = "where_tile";
    kernel_defines["FILL_LLK"] = "fill_tile";
    if (predicate_tensor.dtype() == DataType::FLOAT32) {
        kernel_defines["WHERE_LLK"] = "where_fp32_tile";
    }
    if (predicate_tensor.dtype() == DataType::INT32) {
        kernel_defines["WHERE_LLK"] = "where_int32_tile";
        kernel_defines["FILL_LLK"] = "fill_tile_int";
        kernel_defines["FILL_WITH_VALUE_INT"] = "1";
    } else {
        kernel_defines["FILL_WITH_VALUE_FLOAT"] = "1";
    }

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        get_kernel_file_path(kernel_config.compute_kernel),
        all_device_cores,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .compile_args = compute_kernel_args,
            .defines = kernel_defines});

    auto set_runtime_args = [](Program& program, KernelHandle kernel_id, CoreCoord core, auto&& args) {
        tt_metal::SetRuntimeArgs(program, kernel_id, core, args);
    };

    CMAKE_UNIQUE_NAMESPACE::set_or_update_runtime_arguments(
        program,
        reader_kernel_id,
        writer_kernel_id,
        compute_kernel_id,
        compute_with_storage_grid_size,
        operation_attributes,
        tensor_args,
        output,
        set_runtime_args);

    return {
        std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id, compute_with_storage_grid_size}};
}

void WhereDeviceOperation::WhereProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto update_args =
        [](tt::tt_metal::Program& program, tt::tt_metal::KernelHandle kernel_id, CoreCoord core, auto&& args) {
            auto& all_args = GetRuntimeArgs(program, kernel_id);
            auto& core_args = all_args.at(core.x).at(core.y);
            std::copy(args.begin(), args.end(), core_args.data());
        };

    CMAKE_UNIQUE_NAMESPACE::set_or_update_runtime_arguments(
        cached_program.program,
        cached_program.shared_variables.reader_kernel_id,
        cached_program.shared_variables.writer_kernel_id,
        cached_program.shared_variables.compute_kernel_id,
        cached_program.shared_variables.compute_with_storage_grid_size,
        operation_attributes,
        tensor_args,
        output,
        update_args);
}

}  // namespace ttnn::operations::ternary
