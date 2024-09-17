// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::transformer {

struct ScaledDotProductAttentionGQADecode {
    std::vector<uint32_t> cur_pos;
    const std::optional<float> scale;
    const MemoryConfig output_mem_config;
    const std::optional<SDPAProgramConfig> program_config;
    const DeviceComputeKernelConfig compute_kernel_config;
    const uint32_t k_chunk_size;

    void validate(const std::vector<Tensor>& input_tensors) const;

    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;

    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;

    operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

}  // namespace ttnn::operations::transformer
