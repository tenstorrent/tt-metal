// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/reshape_device_operation_types.hpp"

namespace ttnn::operations::data_movement::reshape {

struct ReshapeRMProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle reader_kernel_id2{};
        bool can_use_dual_kernel{};
        uint32_t num_cores_x{};
        uint32_t num_cores_y{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ReshapeParams& operation_attributes, const ReshapeInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ReshapeParams& operation_attributes,
        const ReshapeInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::data_movement::reshape
