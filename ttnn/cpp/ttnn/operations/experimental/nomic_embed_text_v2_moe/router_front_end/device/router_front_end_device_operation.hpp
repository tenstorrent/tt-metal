// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

// Sizes of one call.
struct RouterFrontEndLayout {
    uint32_t num_tokens = 0;
    uint32_t hidden_tiles = 0;
    uint32_t num_experts = 0;
    // A unit of work is one 32-token block: its hidden_tiles input tiles, one score tile and 32 rows.
    uint32_t num_blocks = 0;
};

struct RouterFrontEndDeviceOperation {
    struct operation_attributes_t {
        tt::tt_metal::MemoryConfig scores_memory_config;
        tt::tt_metal::MemoryConfig rows_memory_config;
        DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& input;
        const Tensor& weight;
    };

    // scores, rows.
    using spec_return_value_t = std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;

    // Validates the call and derives its sizes. compute_output_specs calls it first because the
    // framework allocates the outputs before validate_on_program_cache_miss runs.
    static RouterFrontEndLayout compute_layout(const operation_attributes_t&, const tensor_args_t&);

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

std::tuple<Tensor, Tensor> router_front_end(
    const Tensor& input,
    const Tensor& weight,
    const tt::tt_metal::MemoryConfig& scores_memory_config,
    const tt::tt_metal::MemoryConfig& rows_memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
