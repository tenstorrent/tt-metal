// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ax_plus_b_device_operation.hpp"
#include "ax_plus_b_program_factory_common.hpp"

namespace ttnn::operations::ax_plus_b {

AX_plus_B_DeviceOperation::MultiCore::cached_program_t AX_plus_B_DeviceOperation::MultiCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt::tt_metal;

    Program program{};

    // Multi core uses the device's compute grid
    IDevice* device = tensor_args.tensor_x.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    // Create program configuration
    auto config = AxPlusBProgramFactoryCommon::create_program_config(tensor_args, compute_with_storage_grid_size);

    // Set up data format and circular buffers
    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(tensor_args.tensor_x.dtype());
    const uint32_t single_tile_size = tt::tile_size(cb_data_format);

    AxPlusBProgramFactoryCommon::create_circular_buffers(program, config.all_cores, cb_data_format, single_tile_size);

    // Create all kernels
    auto [reader_kernel_id, compute_kernel_id, writer_kernel_id] =
        AxPlusBProgramFactoryCommon::create_kernels(program, config.all_cores, tensor_args, tensor_return_value);

    // Set runtime arguments for all cores
    AxPlusBProgramFactoryCommon::set_runtime_arguments(
        program, reader_kernel_id, compute_kernel_id, writer_kernel_id, config, tensor_args, tensor_return_value);

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .num_cores = config.num_cores,
         .num_cores_y = config.num_cores_y}};
}

void AX_plus_B_DeviceOperation::MultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    AxPlusBProgramFactoryCommon::update_runtime_arguments(
        program, reader_kernel_id, writer_kernel_id, num_cores, num_cores_y, tensor_args, tensor_return_value);
}

}  // namespace ttnn::operations::ax_plus_b
