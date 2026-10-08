// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/experimental/modulated_deform_im2col/modulated_deform_im2col.hpp"

namespace ttnn::operations::experimental::modulated_deform_im2col {

// Problem geometry, derived once from the tensor shapes and the conv attributes
// and shared by validation, the output spec and the program factory.
struct Im2colShapes {
    uint32_t batch = 0;
    uint32_t in_h = 0;
    uint32_t in_w = 0;
    uint32_t channels = 0;
    uint32_t out_h = 0;
    uint32_t out_w = 0;
    uint32_t kernel_positions = 0;  // K_h * K_w
    uint32_t rows = 0;              // B * H_out * W_out
};

struct ModulatedDeformIm2colOperation {
    struct operation_attributes_t {
        MemoryConfig output_memory_config;
        std::array<uint32_t, 2> kernel_size;
        std::array<uint32_t, 2> stride;
        std::array<uint32_t, 2> padding;
        std::array<uint32_t, 2> dilation;
    };

    struct tensor_args_t {
        const Tensor& input;
        const Tensor& offset;
        const Tensor& mask;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// Derives and validates the problem geometry; TT_FATAL on any inconsistency.
Im2colShapes derive_shapes(
    const ModulatedDeformIm2colOperation::operation_attributes_t& attrs,
    const ModulatedDeformIm2colOperation::tensor_args_t& args);

}  // namespace ttnn::operations::experimental::modulated_deform_im2col

namespace ttnn::prim {

ttnn::Tensor modulated_deform_im2col(
    const Tensor& input,
    const Tensor& offset,
    const Tensor& mask,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    const std::optional<MemoryConfig>& memory_config);

}  // namespace ttnn::prim
