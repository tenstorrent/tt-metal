// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::compress_pool_decode {

struct CompressPoolDecodeParams;
struct CompressPoolDecodeInputs;

struct CompressPoolDecodeSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    std::vector<CoreCoord> cores;
};

struct CompressPoolDecodeProgramFactory {
    using shared_variables_t = CompressPoolDecodeSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const CompressPoolDecodeParams& operation_attributes,
        const CompressPoolDecodeInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const CompressPoolDecodeParams& operation_attributes,
        const CompressPoolDecodeInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek::compress_pool_decode
