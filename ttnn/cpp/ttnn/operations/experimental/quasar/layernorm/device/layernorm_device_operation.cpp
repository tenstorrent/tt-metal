// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tile.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim::qsr {

namespace {
// True when the padded shape is the logical shape with only the last two dims rounded up to whole tiles.
bool layernorm_is_minimally_tile_padded(const Tensor& t) {
    const auto& logical = t.logical_shape();
    const auto& padded = t.padded_shape();
    const auto& tile = t.tensor_spec().tile();
    if (logical.rank() != padded.rank()) {
        return false;
    }
    for (int d = 0; d < static_cast<int>(logical.rank()); ++d) {
        uint32_t expected = logical[d];
        if (d == static_cast<int>(logical.rank()) - 1) {
            expected = tt::round_up(expected, tile.get_width());
        } else if (d == static_cast<int>(logical.rank()) - 2) {
            expected = tt::round_up(expected, tile.get_height());
        }
        if (padded[d] != expected) {
            return false;
        }
    }
    return true;
}
}  // namespace

LayerNormDeviceOperation::program_factory_t LayerNormDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return LayerNormWelfordProgramFactory{};
}

void LayerNormDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input;
    const auto& b = tensor_args.residual_input_tensor;
    const auto& gamma = tensor_args.weight;
    const auto& beta = tensor_args.bias;

    // Scope of this port: interleaved TILE input, standard Welford.
    TT_FATAL(
        std::holds_alternative<ttnn::prim::LayerNormDefaultProgramConfig>(operation_attributes.program_config),
        "Quasar layer_norm only supports LayerNormDefaultProgramConfig (interleaved, non-sharded).");
    TT_FATAL(
        std::get<ttnn::prim::LayerNormDefaultProgramConfig>(operation_attributes.program_config).use_welford,
        "Quasar layer_norm only implements the Welford path; set use_welford=True.");
    TT_FATAL(!a.is_sharded(), "Quasar layer_norm does not support sharded input.");
    TT_FATAL(
        !operation_attributes.output_mem_config.is_sharded(), "Quasar layer_norm does not support sharded output.");
    TT_FATAL(a.layout() == Layout::TILE, "Input tensor must have TILE layout, got: {}", a.layout());
    // BFLOAT16 only: Quasar has no BFLOAT8_B, and the pass-1 transpose_tile cannot unpack fp32 to DEST
    // (tt-llk#1559), so FLOAT32 input is not supported yet.
    TT_FATAL(a.dtype() == DataType::BFLOAT16, "Quasar layer_norm: input must be BFLOAT16, got: {}", a.dtype());
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
    TT_FATAL(
        a.tensor_spec().tile() == Tile{},
        "Quasar layer_norm requires standard 32x32 tiles, got {}x{}",
        a.tensor_spec().tile().get_height(),
        a.tensor_spec().tile().get_width());
    // The kernel masks only the last tile of a row (W % 32) and the factory schedules every physical tile, so
    // the padded shape must be the logical shape rounded up to whole tiles and nothing more.
    TT_FATAL(
        layernorm_is_minimally_tile_padded(a),
        "Quasar layer_norm does not support padding beyond the last logical tile (logical {}, padded {})",
        a.logical_shape(),
        a.padded_shape());

    // Scope of the current kernels: no residual add yet.
    TT_FATAL(!b.has_value(), "Quasar layer_norm does not support residual_input_tensor yet.");

    // TILE-layout, standard-tile gamma/beta that cover the input row.
    for (const auto* affine : {&gamma, &beta}) {
        if (!affine->has_value()) {
            continue;
        }
        const auto& t = affine->value();
        TT_FATAL(t.layout() == Layout::TILE, "Quasar layer_norm requires TILE-layout gamma/beta, got: {}", t.layout());
        TT_FATAL(t.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_FATAL(a.device() == t.device(), "Input and gamma/beta tensors must be on same device");
        TT_FATAL(
            t.tensor_spec().tile() == Tile{},
            "Quasar layer_norm requires standard 32x32 gamma/beta tiles, got {}x{}",
            t.tensor_spec().tile().get_height(),
            t.tensor_spec().tile().get_width());
        TT_FATAL(
            t.padded_shape()[-1] == a.padded_shape()[-1] && t.logical_shape()[-1] >= a.logical_shape()[-1],
            "gamma/beta padded width ({}) must match the input padded width ({}) and the logical width ({}) must "
            "cover the input logical width ({})",
            t.padded_shape()[-1],
            a.padded_shape()[-1],
            t.logical_shape()[-1],
            a.logical_shape()[-1]);
    }
    if (gamma.has_value() && beta.has_value()) {
        TT_FATAL(gamma->layout() == beta->layout(), "Gamma and beta must have the same layout!");
    }
}

TensorSpec LayerNormDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Keep the input's page config and alignment: the factory schedules the input's physical tiles.
    const auto& input = tensor_args.input;
    const auto& in_layout = input.tensor_spec().tensor_layout();
    return TensorSpec(
        input.logical_shape(),
        TensorLayout(
            operation_attributes.dtype.value_or(input.dtype()),
            in_layout.get_page_config(),
            operation_attributes.output_mem_config,
            in_layout.get_alignment()));
}

Tensor LayerNormDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

Tensor layer_norm(
    const Tensor& input_tensor,
    float epsilon,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& bias,
    const std::optional<const Tensor>& residual_input_tensor,
    const MemoryConfig& output_mem_config,
    const ttnn::prim::LayerNormProgramConfig& program_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<const Tensor>& recip_tensor,
    const std::optional<DataType>& dtype,
    const ThreadPlan& threads) {
    return ttnn::device_operation::launch<LayerNormDeviceOperation>(
        LayerNormParams{
            .eps = epsilon,
            .output_mem_config = output_mem_config,
            .program_config = program_config,
            .compute_kernel_config = compute_kernel_config,
            .dtype = dtype,
            .threads = threads,
        },
        LayerNormInputs{
            .input = input_tensor,
            .residual_input_tensor = residual_input_tensor,
            .weight = weight,
            .bias = bias,
            .recip_tensor = recip_tensor,
        });
}

}  // namespace ttnn::prim::qsr
