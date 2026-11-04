// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

// Sizes of one call and how its work splits over the cores.
struct RoutedCombineLayout {
    uint32_t top_k = 0;
    uint32_t padded_tokens = 0;
    uint32_t hidden_tiles = 0;
    // A unit of work is one 32-token block times one chunk of chunk_tiles output tiles.
    uint32_t chunk_tiles = 0;
    uint32_t num_units = 0;
    uint32_t num_cores = 0;
};

struct RoutedCombineDeviceOperation {
    struct operation_attributes_t {
        uint32_t num_tokens = 0;
        tt::tt_metal::MemoryConfig memory_config;
        DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& expert_rows;
        const Tensor& combine_index;
        const Tensor& combine_weights;
        const Tensor& bias;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    // Validates the call and derives its sizes. compute_output_specs calls it first because the
    // framework allocates the output before validate_on_program_cache_miss runs.
    static RoutedCombineLayout compute_layout(const operation_attributes_t&, const tensor_args_t&);

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

Tensor routed_combine(
    const Tensor& expert_rows,
    const Tensor& combine_index,
    const Tensor& combine_weights,
    const Tensor& bias,
    uint32_t num_tokens,
    const tt::tt_metal::MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
