// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/data_movement/sharded/reshard/device/reshard_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct ReshardGenericFactory {
    struct ReshardGenericSharedVariables {
        tt::tt_metal::KernelHandle kernel_id_0{};
        tt::tt_metal::KernelHandle kernel_id_1{};
        tt::tt_metal::CBHandle cb_dst0{};
        CoreCoord grid;
        std::vector<CoreCoord> cores;
    };

    using shared_variables_t = ReshardGenericSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ReshardParams& operation_attributes, const ReshardInputs& tensor_args, Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ReshardParams& operation_attributes,
        const ReshardInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim
