// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::experimental::multi_scale_deformable_attn {

struct MSDAOperation {
    struct operation_attributes_t {
        MemoryConfig output_memory_config;
        // Controls the pixel-coordinate mapping used by the bilinear sampler:
        //   align_corners=false: pixel = (g + 1) * size / 2 - 0.5   (PyTorch / mmcv default)
        //   align_corners=true:  pixel = (g + 1) * (size - 1) / 2
        bool align_corners = false;
    };

    struct tensor_args_t {
        const Tensor& value;
        const Tensor& grid;
        const Tensor& attn;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    // Single-descriptor program factory: build a ProgramDescriptor declaratively.
    // The framework caches the program and patches buffer addresses on cache
    // hits via the buffer_bindings populated by emplace_runtime_args(...).
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::multi_scale_deformable_attn

namespace ttnn::prim {

ttnn::Tensor multi_scale_deformable_attn(
    const Tensor& value,
    const Tensor& grid,
    const Tensor& attn,
    const std::optional<MemoryConfig>& memory_config,
    bool align_corners);

}  // namespace ttnn::prim
