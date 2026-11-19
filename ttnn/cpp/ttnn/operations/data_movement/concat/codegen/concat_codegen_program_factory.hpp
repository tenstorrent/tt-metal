// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct ConcatCodegenParams {
    // Already normalized to 0..rank-1.
    uint32_t dim{};
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct ConcatCodegenInputs {
    std::vector<Tensor> input_tensors;
    std::optional<Tensor> optional_output_tensor;
};

struct ConcatCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ConcatCodegenParams& operation_attributes,
        const ConcatCodegenInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
