// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct SigmoidGatedRmsNormParams {
    uint32_t batch;
    uint32_t num_heads;
    uint32_t sequence;
    uint32_t value_dim;
    float epsilon;
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    DeviceComputeKernelConfig compute_kernel_config;
    // Gate columns start at this element offset within each row of the gate tensor.
    uint32_t gate_column_offset = 0;
    // Heads [0, activated_heads) take their sigmoid(gate) from activated_gate; the gate holds heads
    // [activated_heads, num_heads) from gate_column_offset.
    uint32_t activated_heads = 0;
};

struct SigmoidGatedRmsNormInputs {
    Tensor input;
    Tensor gate;
    Tensor weight;
    std::optional<Tensor> activated_gate;
};

}  // namespace ttnn::experimental::prim
