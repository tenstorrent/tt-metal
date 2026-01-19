// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "minimal_matmul_split_device_operation_types.hpp"
#include "minimal_matmul_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct MinimalMatmulSplitProgramFactory {
    // Reuse shared_variables_t from MinimalMatmulProgramFactory
    using shared_variables_t = MinimalMatmulProgramFactory::shared_variables_t;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    using tensor_return_value_t = std::vector<Tensor>;

    static cached_program_t create(
        const MinimalMatmulSplitParams& operation_attributes,
        const MinimalMatmulSplitInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const MinimalMatmulSplitParams& operation_attributes,
        const MinimalMatmulSplitInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
