// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "tt-metalium/global_circular_buffer.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"  // for DEFAULT_OUTPUT_MEMORY_CONFIG

namespace ttnn::prim {

struct MatmulParams {
    std::optional<operations::matmul::MatmulProgramConfig> program_config = std::nullopt;
    std::optional<bool> bcast_batch = std::nullopt;
    tt::tt_metal::MemoryConfig output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    std::optional<tt::tt_metal::DataType> output_dtype = std::nullopt;
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
    bool untilize_out = false;
    std::optional<tt::tt_metal::CoreCoord> user_core_coord = std::nullopt;
    std::optional<ttnn::operations::unary::UnaryWithParam> user_fused_activation = std::nullopt;
    bool user_run_batched = false;
    bool transpose_a = false;
    bool transpose_b = false;
    std::optional<tt::tt_metal::Tile> output_tile = std::nullopt;
    std::optional<tt::tt_metal::experimental::GlobalCircularBuffer> global_cb = std::nullopt;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt;
};

struct MatmulInputs {
    std::vector<Tensor> input_tensors;                                // a,b, weights
    std::vector<std::optional<const Tensor>> optional_input_tensors;  // bias
    std::vector<std::optional<Tensor>> optional_output_tensors;       // output
    // Opt-in fused greedy-argmax epilogue for the DRAM-sharded matmul path
    // (Blackhole only). Preallocated UINT32 ROW_MAJOR INTERLEAVED tensor of
    // shape [1, 1, num_dram_banks, 64]: page w receives worker w's per-row
    // (global_col_index, bf16_value_bits) pairs — word 2*r is the winning
    // GLOBAL column index for output row r under bfloat16_greater's
    // sign-magnitude total order with the smallest-index tie-break, word
    // 2*r+1 the winning value's raw bf16 bits (init 0xFF80 = -inf for rows
    // that were never updated). The scan runs on the pack RISC's RVV unit in
    // the pack shadow of the matmul and only covers the LOGICAL width of
    // in1, so padded-vocab tail columns never participate (the -inf mask add
    // callers run today is unnecessary). Providing this tensor turns the
    // fusion ON; the plain path is untouched when it is nullopt.
    std::optional<Tensor> fused_argmax_partials = std::nullopt;
};

}  // namespace ttnn::prim
