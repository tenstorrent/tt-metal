// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "deepseek_moe_post_combine_tilize_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct DeepseekMoEPostCombineTilizeProgramFactory {
    struct shared_variables_t {
        uint32_t reader_kernel_id;
        uint32_t compute_kernel_id;
        uint32_t writer_kernel_id;
        std::vector<tt::tt_metal::CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const DeepseekMoEPostCombineTilizeParams& operation_attributes,
        const DeepseekMoEPostCombineTilizeInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const DeepseekMoEPostCombineTilizeParams& operation_attributes,
        const DeepseekMoEPostCombineTilizeInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
