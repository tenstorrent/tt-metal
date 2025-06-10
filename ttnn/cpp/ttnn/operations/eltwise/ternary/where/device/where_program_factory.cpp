// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "where_device_operation.hpp"
#include "where_utils.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include <cmath>

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using namespace ttnn::operations::ternary;

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> extract_shape_dims(const tt::tt_metal::Tensor& x) {
    const auto& shape = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    return {shape[-4], shape[-3], shape[-2] / tile.get_height(), shape[-1] / tile.get_width()};
}

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
    const auto& [predicate_tensor, value_true_tensor_opt, value_false_tensor_opt, optional_output_tensor] = tensor_args;

    // Handle tensor-tensor-tensor, tensor-tensor-scalar, and tensor-scalar-tensor cases
    bool is_value_true_scalar = !value_true_tensor_opt.has_value();
    bool is_value_false_scalar = !value_false_tensor_opt.has_value();

    // Extract tensor references when available
    const ttnn::Tensor* value_true_tensor_ptr = nullptr;
    const ttnn::Tensor* value_false_tensor_ptr = nullptr;
    if (!is_value_true_scalar) {
        value_true_tensor_ptr = &value_true_tensor_opt.value();
    }
    if (!is_value_false_scalar) {
        value_false_tensor_ptr = &value_false_tensor_opt.value();
    }
    const auto [aN, aC, aHt, aWt] = extract_shape_dims(predicate_tensor);  // Considering all are of same shape

    uint32_t num_output_tiles = output.physical_volume() / output.tensor_spec().tile().get_tile_hw();

    constexpr bool row_major = true;
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);
    constexpr size_t num_reader_args = 5;
    constexpr size_t num_writer_args = 3;
    constexpr size_t num_kernel_args = 1;
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

        // Safe float to uint32_t conversion for scalar cases
        uint32_t value_true_scalar_as_uint = 0;
        uint32_t value_false_scalar_as_uint = 0;

        if (is_value_true_scalar) {
            union {
                float f;
                uint32_t u;
            } converter;
            converter.f = operation_attributes.value_true_scalar.value();

            // Use value_false tensor format for consistency in tensor-scalar-tensor case
            if (value_false_tensor_ptr->dtype() == ttnn::DataType::FLOAT32) {
                // Keep as FP32 for FP32 operations
                value_true_scalar_as_uint = converter.u;
            } else {
                // Convert to BFLOAT16 for BF16 operations
                // Simple FP32 to BF16 conversion: truncate mantissa
                uint32_t fp32_bits = converter.u;
                uint16_t bf16_bits = static_cast<uint16_t>((fp32_bits + 0x8000) >> 16);
                value_true_scalar_as_uint = static_cast<uint32_t>(bf16_bits);
            }
        }

        if (is_value_false_scalar) {
            union {
                float f;
                uint32_t u;
            } converter;
            converter.f = operation_attributes.value_false_scalar.value();

            // Use value_true tensor format for consistency in tensor-tensor-scalar case
            if (value_true_tensor_ptr->dtype() == ttnn::DataType::FLOAT32) {
                // Keep as FP32 for FP32 operations
                value_false_scalar_as_uint = converter.u;
            } else {
                // Convert to BFLOAT16 for BF16 operations
                // Simple FP32 to BF16 conversion: truncate mantissa
                uint32_t fp32_bits = converter.u;
                uint16_t bf16_bits = static_cast<uint16_t>((fp32_bits + 0x8000) >> 16);
                value_false_scalar_as_uint = static_cast<uint32_t>(bf16_bits);
            }
        }

        std::array<uint32_t, 5> reader_runtime_args;
        if (is_value_true_scalar) {
            // tensor-scalar-tensor case
            reader_runtime_args = {
                predicate_tensor.buffer()->address(),
                value_false_tensor_ptr->buffer()->address(),  // value_false tensor address
                value_true_scalar_as_uint,                    // scalar value for value_true
                num_tiles_per_core,
                start_tile_id};
        } else if (is_value_false_scalar) {
            // tensor-tensor-scalar case
            reader_runtime_args = {
                predicate_tensor.buffer()->address(),
                value_true_tensor_ptr->buffer()->address(),
                value_false_scalar_as_uint,  // scalar value for value_false
                num_tiles_per_core,
                start_tile_id};
        } else {
            // tensor-tensor-tensor case
            reader_runtime_args = {
                predicate_tensor.buffer()->address(),
                value_true_tensor_ptr->buffer()->address(),
                value_false_tensor_ptr->buffer()->address(),
                num_tiles_per_core,
                start_tile_id};
        }
        handle_args(program, reader_kernel_id, core, reader_runtime_args);

        std::array writer_runtime_args = {
            output.buffer()->address(),
            num_tiles_per_core,
            start_tile_id,
        };
        handle_args(program, writer_kernel_id, core, writer_runtime_args);

        std::array compute_runtime_args = {num_tiles_per_core};
        handle_args(program, compute_kernel_id, core, compute_runtime_args);

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

    const auto& [predicate_tensor, value_true_tensor_opt, value_false_tensor_opt, optional_output_tensor] = tensor_args;

    // Use where_variant from operation attributes
    const WhereVariant where_variant = operation_attributes.where_variant;
    if (where_variant == WhereVariant::TSS) {
        std::cout << "TSS case" << std::endl;
    }
    if (where_variant == WhereVariant::TST) {
        std::cout << "TST case" << std::endl;
    }
    if (where_variant == WhereVariant::TTS) {
        std::cout << "TTS case" << std::endl;
    }
    if (where_variant == WhereVariant::TTT) {
        std::cout << "TTT case" << std::endl;
    }
    WhereKernelConfig kernel_config(where_variant);

    // Handle tensor-tensor-tensor, tensor-tensor-scalar, and tensor-scalar-tensor cases
    bool is_value_true_scalar = !value_true_tensor_opt.has_value();
    bool is_value_false_scalar = !value_false_tensor_opt.has_value();

    // Extract tensor references when available
    const ttnn::Tensor* value_true_tensor_ptr = nullptr;
    const ttnn::Tensor* value_false_tensor_ptr = nullptr;
    if (!is_value_true_scalar) {
        value_true_tensor_ptr = &value_true_tensor_opt.value();
    }
    if (!is_value_false_scalar) {
        value_false_tensor_ptr = &value_false_tensor_opt.value();
    }
    auto program = CreateProgram();

    auto* device = predicate_tensor.device();

    // auto predicate_data_format = datatype_to_dataformat_converter(predicate_tensor.dtype());
    // auto value_true_data_format = datatype_to_dataformat_converter(value_true_tensor.dtype());
    // auto value_false_data_format = datatype_to_dataformat_converter(value_false_tensor.dtype());
    // auto output_data_format = datatype_to_dataformat_converter(output.dtype());

    auto predicate_data_format = datatype_to_dataformat_converter(
        (predicate_tensor.dtype() == ttnn::DataType::BFLOAT16 || predicate_tensor.dtype() == ttnn::DataType::BFLOAT8_B)
            ? ttnn::DataType::UINT16
            : predicate_tensor.dtype());

    DataFormat value_true_data_format;
    if (is_value_true_scalar) {
        // tensor-scalar-tensor case: use value_false tensor format
        value_true_data_format = datatype_to_dataformat_converter(
            (value_false_tensor_ptr->dtype() == ttnn::DataType::BFLOAT16 ||
             value_false_tensor_ptr->dtype() == ttnn::DataType::BFLOAT8_B)
                ? ttnn::DataType::UINT16
                : value_false_tensor_ptr->dtype());
    } else {
        value_true_data_format = datatype_to_dataformat_converter(
            (value_true_tensor_ptr->dtype() == ttnn::DataType::BFLOAT16 ||
             value_true_tensor_ptr->dtype() == ttnn::DataType::BFLOAT8_B)
                ? ttnn::DataType::UINT16
                : value_true_tensor_ptr->dtype());
    }

    DataFormat value_false_data_format;
    if (is_value_false_scalar) {
        // tensor-tensor-scalar case: use value_true tensor format
        value_false_data_format = datatype_to_dataformat_converter(
            (value_true_tensor_ptr->dtype() == ttnn::DataType::BFLOAT16 ||
             value_true_tensor_ptr->dtype() == ttnn::DataType::BFLOAT8_B)
                ? ttnn::DataType::UINT16
                : value_true_tensor_ptr->dtype());
    } else {
        value_false_data_format = datatype_to_dataformat_converter(
            (value_false_tensor_ptr->dtype() == ttnn::DataType::BFLOAT16 ||
             value_false_tensor_ptr->dtype() == ttnn::DataType::BFLOAT8_B)
                ? ttnn::DataType::UINT16
                : value_false_tensor_ptr->dtype());
    }

    auto output_data_format = datatype_to_dataformat_converter(
        (output.dtype() == ttnn::DataType::BFLOAT16 || output.dtype() == ttnn::DataType::BFLOAT8_B)
            ? ttnn::DataType::UINT16
            : output.dtype());

    uint32_t predicate_single_tile_size = tt_metal::detail::TileSize(predicate_data_format);
    uint32_t value_true_single_tile_size = tt_metal::detail::TileSize(value_true_data_format);
    uint32_t value_false_single_tile_size = tt_metal::detail::TileSize(value_false_data_format);
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_data_format);

    uint32_t num_output_tiles = output.physical_volume() / output.tensor_spec().tile().get_tile_hw();

    // we parallelize the computation across the output tiles
    constexpr bool row_major = true;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // Number of tiles to store per input CB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;

    // Input buffers
    auto [predicate_tensor_cb, predicate_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_0,
        program,
        all_device_cores,
        predicate_single_tile_size,
        num_tiles_per_cb,
        predicate_data_format);  // predicate_tensor
    auto [value_true_tensor_cb, value_true_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_1,
        program,
        all_device_cores,
        value_true_single_tile_size,
        num_tiles_per_cb,
        value_true_data_format);  // value_true_tensor
    auto [value_false_tensor_cb, value_false_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_2,
        program,
        all_device_cores,
        value_false_single_tile_size,
        num_tiles_per_cb,
        value_false_data_format);  // value_false_tensor

    // Output buffer
    auto [output_tensor_cb, output_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_3,
        program,
        all_device_cores,
        output_single_tile_size,
        num_tiles_per_cb,
        output_data_format);  // output

    auto predicate_is_dram =
        static_cast<uint32_t>(predicate_tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    auto value_true_is_dram =
        is_value_true_scalar ? static_cast<uint32_t>(0) :  // No DRAM for scalar case
            static_cast<uint32_t>(value_true_tensor_ptr->buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    auto value_false_is_dram =
        is_value_false_scalar ? static_cast<uint32_t>(0) :  // No DRAM for scalar case
            static_cast<uint32_t>(value_false_tensor_ptr->buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    auto output_is_dram = static_cast<uint32_t>(output.buffer()->buffer_type() == tt_metal::BufferType::DRAM);

    // COMPUTE KERNEL
    bool fp32_dest_acc_en = output_data_format == tt::DataFormat::UInt32 ||
                            output_data_format == tt::DataFormat::Int32 ||
                            output_data_format == tt::DataFormat::Float32;

    // READER KERNEL - Set up defines for FP32 support and dataflow
    std::map<std::string, std::string> reader_defines;
    if (fp32_dest_acc_en) {
        reader_defines["FP32_DEST_ACC_EN"] = "1";
    }

    // Use utility to get reader kernel path
    std::string reader_kernel_path = get_kernel_file_path(kernel_config.reader_kernel);
    std::cout << "reader_kernel_path: " << reader_kernel_path << std::endl;

    KernelHandle reader_kernel_id;

    if (is_value_true_scalar) {
        // tensor-scalar-tensor case
        reader_kernel_id = tt_metal::CreateKernel(
            program,
            reader_kernel_path,
            all_device_cores,
            tt_metal::ReaderDataMovementConfig(
                {predicate_is_dram,
                 predicate_tensor_cb,
                 value_false_is_dram,
                 value_false_tensor_cb,
                 // No DRAM flag needed for scalar, but we still need the CB index
                 static_cast<uint32_t>(0),  // dummy DRAM flag for scalar
                 value_true_tensor_cb},
                reader_defines));
    } else if (is_value_false_scalar) {
        // tensor-tensor-scalar case
        reader_kernel_id = tt_metal::CreateKernel(
            program,
            reader_kernel_path,
            all_device_cores,
            tt_metal::ReaderDataMovementConfig(
                {predicate_is_dram,
                 predicate_tensor_cb,
                 value_true_is_dram,
                 value_true_tensor_cb,
                 // No DRAM flag needed for scalar, but we still need the CB index
                 static_cast<uint32_t>(0),  // dummy DRAM flag for scalar
                 value_false_tensor_cb},
                reader_defines));
    } else {
        // tensor-tensor-tensor case
        reader_kernel_id = tt_metal::CreateKernel(
            program,
            reader_kernel_path,
            all_device_cores,
            tt_metal::ReaderDataMovementConfig(
                {predicate_is_dram,
                 predicate_tensor_cb,
                 value_true_is_dram,
                 value_true_tensor_cb,
                 value_false_is_dram,
                 value_false_tensor_cb},
                reader_defines));
    }

    // WRITER KERNEL
    std::string writer_kernel_path = get_kernel_file_path(kernel_config.writer_kernel);
    std::cout << "writer_kernel_path: " << writer_kernel_path << std::endl;
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        writer_kernel_path,
        all_device_cores,
        tt_metal::WriterDataMovementConfig({output_tensor_cb, output_is_dram}));

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[tt::CBIndex::c_0] = (predicate_tensor.dtype() == ttnn::DataType::FLOAT32)
                                                ? UnpackToDestMode::UnpackToDestFp32
                                                : UnpackToDestMode::Default;

    // Set value_true CB unpack mode
    if (is_value_true_scalar) {
        // tensor-scalar-tensor: match value_false tensor format
        unpack_to_dest_mode[tt::CBIndex::c_1] =
            (value_false_tensor_ptr->dtype() == ttnn::DataType::FLOAT32 ? UnpackToDestMode::UnpackToDestFp32
                                                                        : UnpackToDestMode::Default);
    } else {
        // tensor from value_true_tensor
        unpack_to_dest_mode[tt::CBIndex::c_1] =
            (value_true_tensor_ptr->dtype() == ttnn::DataType::FLOAT32 ? UnpackToDestMode::UnpackToDestFp32
                                                                       : UnpackToDestMode::Default);
    }

    // Set value_false CB unpack mode
    if (is_value_false_scalar) {
        // tensor-tensor-scalar: match value_true tensor format
        unpack_to_dest_mode[tt::CBIndex::c_2] =
            (value_true_tensor_ptr->dtype() == ttnn::DataType::FLOAT32 ? UnpackToDestMode::UnpackToDestFp32
                                                                       : UnpackToDestMode::Default);
    } else {
        // tensor from value_false_tensor
        unpack_to_dest_mode[tt::CBIndex::c_2] =
            (value_false_tensor_ptr->dtype() == ttnn::DataType::FLOAT32 ? UnpackToDestMode::UnpackToDestFp32
                                                                        : UnpackToDestMode::Default);
    }
    unpack_to_dest_mode[tt::CBIndex::c_3] =
        (output.dtype() == ttnn::DataType::FLOAT32) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;

    constexpr uint32_t num_tiles_per_cycle = 1;  // we produce 1 output tile per read-compute-write cycle
    std::vector<uint32_t> compute_kernel_args = {
        num_tiles_per_cycle,
    };

    std::map<std::string, std::string> kernel_defines;
    kernel_defines["WHERE_LLK"] = "where_tile";

    if (predicate_tensor.dtype() == DataType::FLOAT32) {
        kernel_defines["WHERE_LLK"] = "where_fp32_tile";
    } else if (predicate_tensor.dtype() == DataType::INT32) {
        kernel_defines["WHERE_LLK"] = "where_int_tile";
    } else if (predicate_tensor.dtype() == DataType::BFLOAT8_B) {
        // Use FP32 path for BFLOAT8_B to avoid precision issues in condition evaluation
        kernel_defines["WHERE_LLK"] = "where_fp32_tile";
    }

    // Use utility to get compute kernel path
    std::string compute_kernel_path = get_kernel_file_path(kernel_config.compute_kernel);
    std::cout << "compute_kernel_path: " << compute_kernel_path << std::endl;
    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        compute_kernel_path,
        all_device_cores,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .compile_args = compute_kernel_args,
            .defines = kernel_defines});

    auto set_runtime_args = [](tt::tt_metal::Program& program,
                               tt::tt_metal::KernelHandle kernel_id,
                               tt::tt_metal::CoreCoord core,
                               auto&& args) { tt::tt_metal::SetRuntimeArgs(program, kernel_id, core, args); };

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
    auto update_args = [](tt::tt_metal::Program& program,
                          tt::tt_metal::KernelHandle kernel_id,
                          tt::tt_metal::CoreCoord core,
                          auto&& args) {
        auto& all_args = tt::tt_metal::GetRuntimeArgs(program, kernel_id);
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
