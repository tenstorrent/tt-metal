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
    // L1 prefetch ride-along (kernel_lib/l1_prefetch_dataflow.hpp): prefetch_cores[j] runs job j of the job table once
    // its reads are done. Every job writes inside [prefetch_l1_begin, prefetch_l1_end), which must be free while the
    // norm runs: above its circular buffers and below every L1 buffer.
    std::vector<tt::tt_metal::CoreCoord> prefetch_cores;
    uint32_t prefetch_l1_begin = 0;
    uint32_t prefetch_l1_end = 0;
};

struct SigmoidGatedRmsNormInputs {
    Tensor input;
    Tensor gate;
    Tensor weight;
    std::optional<Tensor> activated_gate;
    // The tensor the prefetch reads, and its per-device job table (ROW_MAJOR UINT32, one 16-word row per job).
    std::optional<Tensor> prefetch_source;
    std::optional<Tensor> prefetch_jobs;
};

}  // namespace ttnn::experimental::prim
