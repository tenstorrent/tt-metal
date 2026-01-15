// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ema_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::reduction::ema::program {

struct EmaSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    CoreRangeSet all_cores;
};

struct EmaProgramFactory {
    using shared_variables_t = EmaSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const EmaParams& operation_attributes, const EmaInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const EmaParams& operation_attributes,
        const EmaInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::reduction::ema::program
