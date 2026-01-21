// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/operations/madd/device/madd_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct MAddProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel{};
        tt::tt_metal::KernelHandle writer_kernel{};
        std::size_t num_cores = 0;
        std::size_t num_cores_y = 0;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const MAddParams& operation_attributes, const MAddArgs& tensor_args, Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const MAddParams& operation_attributes,
        const MAddArgs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim
