// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "transpose_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

#include <tt-metalium/core_coord.hpp>

namespace ttnn::prim {

struct TransposeHCTiledInterleavedSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
};

struct TransposeHCTiledInterleavedProgramFactory {
    using shared_variables_t = TransposeHCTiledInterleavedSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const TransposeParams& operation_attributes, const TransposeInputs& tensor_args, Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const TransposeParams& operation_attributes,
        const TransposeInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::prim
