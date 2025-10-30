// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ax_plus_b_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::ax_plus_b {

struct ProgramFactoryConfig {
    tt::tt_metal::CoreCoord compute_with_storage_grid_size;
    uint32_t num_cores;
    uint32_t num_cores_y;
    tt::tt_metal::CoreRangeSet all_cores;
    tt::tt_metal::CoreRangeSet core_group_1;
    tt::tt_metal::CoreRangeSet core_group_2;
    uint32_t num_tiles_per_core_group_1;
    uint32_t num_tiles_per_core_group_2;
};

// Common functionality shared between SingleCore and MultiCore implementations
class AxPlusBProgramFactoryCommon {
public:
    // Create the program configuration based on tensor args and grid size
    static ProgramFactoryConfig create_program_config(
        const AX_plus_B_DeviceOperation::tensor_args_t& tensor_args,
        const tt::tt_metal::CoreCoord& compute_with_storage_grid_size);

    // Create circular buffers for the operation
    static void create_circular_buffers(
        tt::tt_metal::Program& program,
        const tt::tt_metal::CoreRangeSet& all_cores,
        tt::DataFormat cb_data_format,
        uint32_t single_tile_size);

    // Create all kernels (reader, compute, writer)
    static std::tuple<tt::tt_metal::KernelHandle, tt::tt_metal::KernelHandle, tt::tt_metal::KernelHandle>
    create_kernels(
        tt::tt_metal::Program& program,
        const tt::tt_metal::CoreRangeSet& all_cores,
        const AX_plus_B_DeviceOperation::tensor_args_t& tensor_args,
        const AX_plus_B_DeviceOperation::tensor_return_value_t& tensor_return_value);

    // Set runtime arguments for all cores
    static void set_runtime_arguments(
        tt::tt_metal::Program& program,
        tt::tt_metal::KernelHandle reader_kernel_id,
        tt::tt_metal::KernelHandle compute_kernel_id,
        tt::tt_metal::KernelHandle writer_kernel_id,
        const ProgramFactoryConfig& config,
        const AX_plus_B_DeviceOperation::tensor_args_t& tensor_args,
        const AX_plus_B_DeviceOperation::tensor_return_value_t& tensor_return_value);

    // Update runtime arguments for cached programs
    static void update_runtime_arguments(
        tt::tt_metal::Program& program,
        tt::tt_metal::KernelHandle reader_kernel_id,
        tt::tt_metal::KernelHandle writer_kernel_id,
        uint32_t num_cores,
        uint32_t num_cores_y,
        const AX_plus_B_DeviceOperation::tensor_args_t& tensor_args,
        const AX_plus_B_DeviceOperation::tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::ax_plus_b
