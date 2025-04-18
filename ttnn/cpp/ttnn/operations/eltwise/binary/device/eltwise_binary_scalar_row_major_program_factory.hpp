// ttnn/cpp/ttnn/operations/eltwise/binary/device/eltwise_binary_scalar_row_major_program_factory.hpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/eltwise/binary/binary_op_types.hpp"
#include "tt_metal/host_api.hpp"

namespace ttnn::operations::binary {

// Forward declaration
struct EltwiseBinaryDeviceOperation;

// This factory creates a program for element-wise binary operations (specifically tensor + scalar)
// where the input and output tensors are in RowMajor layout.
// It avoids intermediate tiling/untiling steps.
// NOTE: This is a simplified single-core version for demonstration.
struct EltwiseBinaryScalarRowMajor {
    // Structure to hold the created program and potentially kernel IDs if needed later
    struct cached_program_t {
        tt::tt_metal::Program program{};
        KernelHandle reader_kernel_id{};
        KernelHandle compute_kernel_id{};
        KernelHandle writer_kernel_id{};
    };

    // The factory function to create the program
    static cached_program_t create(
        const operation_attributes_t& operation_attributes,  // Contains scalar, op_type etc.
        const tensor_args_t& tensor_args,                    // Contains input tensor, maybe output
        tensor_return_value_t& tensor_return_value           // Output tensor reference
    );

    // Function to update runtime arguments (called before each launch)
    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    // Define the operation traits if needed (example, adjust as necessary)
    void validate(const tensor_args_t& tensor_args) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const tensor_args_t& tensor_args) const;
    std::vector<Tensor> create_output_tensors(const tensor_args_t& tensor_args) const;
    // Add other necessary traits if making this a full DeviceOperation
};

}  // namespace ttnn::operations::binary
