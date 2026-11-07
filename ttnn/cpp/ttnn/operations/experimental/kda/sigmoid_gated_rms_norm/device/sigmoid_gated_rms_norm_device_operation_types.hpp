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
    // The worker grid from the origin to spread over; the device's whole grid when unset.
    std::optional<CoreCoord> grid;
};

struct SigmoidGatedRmsNormInputs {
    Tensor input;
    Tensor gate;
    Tensor weight;
    // The last heads' gate columns, from column 0, when the gate holds only the first heads'.
    std::optional<Tensor> gate_tail;
};

}  // namespace ttnn::experimental::prim
