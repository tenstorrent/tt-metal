// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/experimental/modulated_deform_im2col/modulated_deform_im2col.hpp"

namespace ttnn::operations::experimental::modulated_deform_im2col {

// The reader computes sample positions in Q16.16 (range +-32768) and saturates offsets at this
// many pixels. Input sizes, padding and the kernel span are validated below it, so every
// coordinate stays in range and a saturated offset still lands outside the input.
constexpr uint32_t kOffsetClampPixels = 4096;

inline constexpr std::string_view kKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/modulated_deform_im2col/device/kernels/";
// The widest margin a core's input band takes beyond the kernel's span, and the L1 the band leaves
// free for what the op's program does not account for.
constexpr uint32_t kMaxBandMargin = 4;
constexpr uint32_t kBandL1Reserve = 32 * 1024;
// The most input channels, in tiles, the fused conv samples and multiplies per pass. Wider inputs
// take several passes per kernel position, each with its own slice of the weight, so the corner
// blocks and the two weight slices stay within L1.
constexpr uint32_t kConvMaxChunkTiles = 8;

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
    uint32_t out_channels = 0;      // C_out when the op runs the conv's matmul too, else 0
};

struct ModulatedDeformIm2colOperation {
    struct operation_attributes_t {
        MemoryConfig output_memory_config;
        std::array<uint32_t, 2> kernel_size;
        std::array<uint32_t, 2> stride;
        std::array<uint32_t, 2> padding;
        std::array<uint32_t, 2> dilation;
        // With a weight: a ReLU on the conv's output, after the bias.
        bool relu = false;
    };

    // With a weight, the op runs the whole conv: the columns never leave L1 and the output is
    // (1, 1, B*H_out*W_out, C_out). Without one, it emits the columns.
    struct tensor_args_t {
        const Tensor& input;
        const Tensor& offset_mask;
        const std::optional<const Tensor> weight;
        const std::optional<const Tensor> bias;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output);
    static tt::tt_metal::ProgramDescriptor create_conv_descriptor(
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
    const Tensor& offset_mask,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    const std::optional<MemoryConfig>& memory_config);

ttnn::Tensor modulated_deform_conv2d(
    const Tensor& input,
    const Tensor& offset_mask,
    const Tensor& weight,
    const std::optional<const Tensor>& bias,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    bool relu,
    const std::optional<MemoryConfig>& memory_config);

}  // namespace ttnn::prim
