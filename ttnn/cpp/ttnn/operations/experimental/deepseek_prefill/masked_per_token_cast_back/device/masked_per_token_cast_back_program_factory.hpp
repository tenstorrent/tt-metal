// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "masked_per_token_cast_back_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim::masked_per_token_cast_back {

struct MaskedPerTokenCastBackSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    std::vector<CoreCoord> all_cores_vec;
};

struct MaskedPerTokenCastBackProgramFactory {
    using shared_variables_t = MaskedPerTokenCastBackSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const MaskedPerTokenCastBackParams& operation_attributes,
        const MaskedPerTokenCastBackInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const MaskedPerTokenCastBackParams& operation_attributes,
        const MaskedPerTokenCastBackInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim::masked_per_token_cast_back
