// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

// Sizes of one call and the cores it runs on, derived from its inputs, attributes and device.
struct GroupedMatmulLayout {
    uint32_t num_experts = 0;
    uint32_t rows = 0;
    uint32_t k_tiles = 0;
    uint32_t n_tiles = 0;
    // Blocks are dealt round-robin to num_lanes core rows, or with per_expert_lanes lane e takes
    // expert e's blocks, num_lanes = num_experts. The N tiles split over num_columns core columns,
    // column_n_tiles each and last_column_n_tiles on the last.
    uint32_t num_lanes = 0;
    uint32_t num_columns = 0;
    uint32_t column_n_tiles = 0;
    uint32_t last_column_n_tiles = 0;
    // Output tiles a matmul_block call computes, at most what DST holds.
    uint32_t subblock_w = 0;
};

struct GroupedMatmulDeviceOperation {
    struct operation_attributes_t {
        tt::tt_metal::DataType dtype = tt::tt_metal::DataType::BFLOAT16;
        bool transpose_weight = false;
        std::optional<ttnn::operations::unary::UnaryWithParam> activation;
        tt::tt_metal::Layout output_layout = tt::tt_metal::Layout::TILE;
        tt::tt_metal::MemoryConfig memory_config;
        DeviceComputeKernelConfig compute_kernel_config;
        bool per_expert_lanes = false;
    };

    struct tensor_args_t {
        const Tensor& input;
        const Tensor& weight;
        const Tensor& expert_counts;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    // Validates the call and derives its sizes. compute_output_specs calls it first because the
    // framework allocates the output before validate_on_program_cache_miss runs.
    static GroupedMatmulLayout compute_layout(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // Every runtime argument that is not a buffer address comes from a hashed attribute or tensor
    // spec, so the default program hash keeps a cache hit correct; the framework re-binds the buffers.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe

namespace ttnn::prim {

Tensor grouped_matmul(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& expert_counts,
    tt::tt_metal::DataType dtype,
    bool transpose_weight,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& activation,
    tt::tt_metal::Layout output_layout,
    const tt::tt_metal::MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool per_expert_lanes);

}  // namespace ttnn::prim
