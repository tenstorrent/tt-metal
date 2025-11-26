// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "argmax_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::reduction::argmax::program {

// Shared variables for single-core operations (both ROW_MAJOR and TILE)
struct ArgMaxSingleCoreSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    CoreCoord core;
};

// Shared variables for multi-core operation
struct ArgMaxMultiCoreSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id0 = 0;
    tt::tt_metal::KernelHandle reader_kernel_id1 = 0;
    std::vector<CoreCoord> cores_coords0;
    std::vector<CoreCoord> cores_coords1;
    uint32_t num_cores0 = 0;
    uint32_t num_cores1 = 0;
};

// Program factory for single-core ROW_MAJOR layout
struct ArgMaxSingleCoreRowMajorFactory {
    using shared_variables_t = ArgMaxSingleCoreSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

// Program factory for single-core TILE layout
struct ArgMaxSingleCoreTileFactory {
    using shared_variables_t = ArgMaxSingleCoreSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

// Program factory for multi-core ROW_MAJOR layout
struct ArgMaxMultiCoreRowMajorFactory {
    using shared_variables_t = ArgMaxMultiCoreSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::reduction::argmax::program
