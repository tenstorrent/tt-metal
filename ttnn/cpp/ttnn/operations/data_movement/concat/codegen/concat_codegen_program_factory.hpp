// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct ConcatCodegenParams {
    uint32_t dim{};
    uint32_t num_inputs{};
    uint32_t stick_size{};
    uint32_t total_out_sticks{};
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct ConcatCodegenInputs {
    std::vector<Tensor> input_tensors;
};

struct ConcatCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ConcatCodegenParams& operation_attributes,
        const ConcatCodegenInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
