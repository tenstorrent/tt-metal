// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/buffers/buffer.hpp"
#include "impl/buffers/circular_buffer_types.hpp"
#include "generic_op_device_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"

#include <cstdint>
#include <iostream>

namespace ttnn::operations::generic {
GenericOpDeviceOperation::GenericProgram::cached_program_t GenericOpDeviceOperation::GenericProgram::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    // // just debug print
    // std::cout << "input buffer 0x" << std::hex << tensor_args.io_tensors.front().buffer()->address() << std::dec << std::endl;
    // std::cout << "output buffer 0x" << std::hex << tensor_args.io_tensors.back().buffer()->address() << std::dec << std::endl;
    // size_t i = 0;
    // for (const auto& tensor : tensor_args.io_tensors) {
    //     std::cout << "[" << i++ << "] 0x=" << tensor.buffer()->address() << std::endl;
    // }
    // // end of debug

    tt::tt_metal::Program program{};

    // create circular buffers
    std::map<uint8_t, tt::tt_metal::CBHandle> cb_handles;
    for (const auto& [buffer_index, circular_buffer_attributes] : operation_attributes.circular_buffer_attributes) {
        tt::tt_metal::CircularBufferConfig cb_config = tt::tt_metal::CircularBufferConfig(
            circular_buffer_attributes.total_size,
            {{buffer_index, circular_buffer_attributes.data_format}})
            .set_page_size(buffer_index, circular_buffer_attributes.page_size);

        // used for sharding to point to the existing buffer
        if (circular_buffer_attributes.set_globally_allocated_address.has_value()) {

            cb_config.set_globally_allocated_address(*tensor_args.io_tensors[circular_buffer_attributes.set_globally_allocated_address.value()].buffer());
        }

        cb_handles[buffer_index] = tt::tt_metal::CreateCircularBuffer(program, circular_buffer_attributes.core_spec, cb_config);
    }

    // create data movement kernels
    std::vector<tt::tt_metal::KernelHandle> data_movement_kernel_ids;

    for (const auto& data_movement_attributes : operation_attributes.data_movement_attributes) {
        data_movement_kernel_ids.push_back(tt::tt_metal::CreateKernel(
            program,
            data_movement_attributes.kernel_path,
            data_movement_attributes.core_spec,
            data_movement_attributes.config));
    }

    // create compute kernels
    for (const auto& compute_attributes : operation_attributes.compute_attributes)
    {
        auto kernel_id = tt::tt_metal::CreateKernel(
            program,
            compute_attributes.kernel_path,
            compute_attributes.core_spec,
            compute_attributes.config);

        for (const auto& [core, args] : compute_attributes.runtime_args_per_core)
        {
            // assuming core is in the compute_attributes.core_spec, would be better to check
            tt::tt_metal::SetRuntimeArgs(program, kernel_id, core, args);
        }
    }

    // initialize data movement runtime arguments
    for (size_t i = 0; i < data_movement_kernel_ids.size(); i++)
    {
        for (const auto& [core, args] : operation_attributes.data_movement_attributes[i].runtime_args_per_core)
        {
            tt::tt_metal::SetRuntimeArgs(program, data_movement_kernel_ids[i], core, args);
        }
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = data_movement_kernel_ids[0],
         .unary_writer_kernel_id = data_movement_kernel_ids[1]}};
}

void GenericOpDeviceOperation::GenericProgram::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    // Not implemented
    const auto& input_tensor = tensor_args.io_tensors.front();
    auto& output_tensor = tensor_args.io_tensors.back();

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();

    // for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
    //     CoreCoord core = {i / num_cores_y, i % num_cores_y};

    //     {
    //         auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
    //         runtime_args[0] = src_buffer->address();
    //     }

    //     {
    //         auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
    //         runtime_args[0] = dst_buffer->address();
    //     }
    // }
}

}  // namespace ttnn::operations::generic
