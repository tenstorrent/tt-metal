// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "nemotron3_mamba2_decode_owned_device_operation.hpp"

#include <string_view>
#include <tuple>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {

void check_tile_layout(const Tensor& t, std::string_view name) {
    TT_FATAL(t.layout() == Layout::TILE, "{} must be tile layout", name);
}

}  // namespace

void Nemotron3Mamba2DecodeOwnedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.x;
    const auto& z = tensor_args.z;
    const auto& dt = tensor_args.dt;
    const auto& dt_bias = tensor_args.dt_bias;
    const auto& A_log = tensor_args.A_log;
    const auto& D = tensor_args.D;
    const auto& B_in = tensor_args.B_in;
    const auto& C_in = tensor_args.C_in;
    const auto& ssm_state = tensor_args.ssm_state;

    check_tile_layout(x, "x");
    check_tile_layout(z, "z");
    check_tile_layout(dt, "dt");
    check_tile_layout(dt_bias, "dt_bias");
    check_tile_layout(A_log, "A_log");
    check_tile_layout(D, "D");
    check_tile_layout(B_in, "B_in");
    check_tile_layout(C_in, "C_in");
    check_tile_layout(ssm_state, "ssm_state");

    TT_FATAL(
        ssm_state.dtype() == DataType::FLOAT32, "ssm_state must be fp32 (decision D4 — bf16 recurrent state drifts)");
    TT_FATAL(B_in.padded_shape() == C_in.padded_shape(), "B and C must have the same shape");
    TT_FATAL(x.padded_shape() == z.padded_shape(), "x and z must have the same shape");

    (void)args;
}

Nemotron3Mamba2DecodeOwnedDeviceOperation::spec_return_value_t
Nemotron3Mamba2DecodeOwnedDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& ssm_state = tensor_args.ssm_state;
    const auto& x = tensor_args.x;

    // Output 0: ssm_state_out — same shape + dtype as input ssm_state (fp32, mutated).
    TensorSpec state_spec(
        ssm_state.logical_shape(),
        TensorLayout(
            ssm_state.dtype(),
            PageConfig(Layout::TILE),
            args.output_memory_config.value_or(ssm_state.memory_config())));

    // Output 1: y — same shape + dtype as x (bf16).
    TensorSpec y_spec(
        x.logical_shape(),
        TensorLayout(x.dtype(), PageConfig(Layout::TILE), args.output_memory_config.value_or(x.memory_config())));

    return {state_spec, y_spec};
}

Nemotron3Mamba2DecodeOwnedDeviceOperation::tensor_return_value_t
Nemotron3Mamba2DecodeOwnedDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // ssm_state is mutated in place — output 0 IS the input tensor (shared buffer).
    Tensor ssm_state_out = tensor_args.ssm_state;

    // y: use preallocated if provided, else allocate fresh.
    auto specs = compute_output_specs(args, tensor_args);
    Tensor y_out = tensor_args.preallocated_y.has_value()
                       ? tensor_args.preallocated_y.value()
                       : tt::tt_metal::create_device_tensor(std::get<1>(specs), tensor_args.x.device());

    return {ssm_state_out, y_out};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::tuple<Tensor, Tensor> nemotron3_mamba2_decode_owned(
    const Tensor& x,
    const Tensor& z,
    const Tensor& dt,
    const Tensor& dt_bias,
    const Tensor& A_log,
    const Tensor& D,
    const Tensor& B_in,
    const Tensor& C_in,
    const Tensor& ssm_state,
    bool debug_fill,
    uint32_t debug_mode,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<Tensor>& preallocated_y) {
    using OperationType = ttnn::experimental::prim::Nemotron3Mamba2DecodeOwnedDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .output_memory_config = output_memory_config,
            .debug_fill = debug_fill,
            .debug_mode = debug_mode,
            // Softplus + clamp defaults from Nemotron-3 config.json.
            .softplus_beta_bits = 0x3f800000u,        // 1.0f
            .softplus_beta_recip_bits = 0x3f800000u,  // 1.0f
            .softplus_threshold_bits = 0x41a00000u,   // 20.0f
            .time_step_floor_bits = 0x38d1b717u,      // 1e-4f
            .time_step_max_bits = 0x3dcccccdu,        // 0.1f
        },
        OperationType::tensor_args_t{
            .x = x,
            .z = z,
            .dt = dt,
            .dt_bias = dt_bias,
            .A_log = A_log,
            .D = D,
            .B_in = B_in,
            .C_in = C_in,
            .ssm_state = ssm_state,
            .preallocated_y = preallocated_y,
        });
}

}  // namespace ttnn::prim
