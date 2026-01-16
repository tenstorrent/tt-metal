// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "transpose_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::data_movement::transpose::program {

struct TransposeHCTiledSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_cores_total{};
    uint32_t num_cores_y{};
    uint32_t num_tiles_per_core_group_1{};
    uint32_t num_tiles_per_core_group_2{};
};

struct TransposeHCTiledProgramFactory {
    using shared_variables_t = TransposeHCTiledSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const transpose::TransposeParams& operation_attributes,
        const transpose::TransposeInputs& tensor_args,
        transpose::tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const transpose::TransposeParams& operation_attributes,
        const transpose::TransposeInputs& tensor_args,
        transpose::tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::data_movement::transpose::program
