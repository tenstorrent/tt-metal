// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include "ttnn/decorators.hpp"

namespace tt {
namespace tt_metal {

enum class RotaryEmbeddingOpParallelizationStrategy { MULTI_CORE };

struct RotaryEmbedding {
    const uint32_t seq_len;
    std::optional<uint32_t> token_idx;
    const MemoryConfig output_mem_config;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;

    RotaryEmbeddingOpParallelizationStrategy get_parallelization_strategy(
        const std::vector<Tensor> &input_tensors) const;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;

    operation::ProgramWithCallbacks create_program(const std::vector<Tensor> &input_tensors,
                                                   std::vector<Tensor> &output_tensors) const;

    const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
};

}  // namespace tt_metal
}  // namespace tt
