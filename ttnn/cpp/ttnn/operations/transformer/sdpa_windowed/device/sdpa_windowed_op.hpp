// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::transformer {

constexpr uint32_t cu_window_seqlens_nelements = 1024;
// [INFO] 1024 is large enough for 300DPI images but can be increased if needed
static_assert(cu_window_seqlens_nelements == 1024, "cu_window_seqlens_nelements must be 1024");

struct WindowedScaledDotProductAttention {
    const std::optional<float> scale;
    const tt::tt_metal::MemoryConfig output_mem_config;
    const std::optional<SDPAProgramConfig> program_config;
    const DeviceComputeKernelConfig compute_kernel_config;

    void validate(const std::vector<Tensor>& input_tensors) const;

    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;

    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;

    tt::tt_metal::operation::OpPerformanceModel create_op_performance_model(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;

    tt::tt_metal::operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;

    std::uint32_t get_q_chunk_size() const;
    std::uint32_t get_k_chunk_size() const;
};

}  // namespace ttnn::operations::transformer
