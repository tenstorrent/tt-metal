// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_matmul_device_operation.hpp"
#include "ring_matmul_program_factory.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::experimental::ring_matmul {

RingMatmulDeviceOperation::program_factory_t RingMatmulDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ProgramFactory{};
}

void RingMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensors) {
    const auto& act_tensor = tensors.input_tensor;
    const auto& weight_tensor = tensors.weight_tensor;

    TT_FATAL(
        act_tensor.storage_type() == StorageType::DEVICE && weight_tensor.storage_type() == StorageType::DEVICE,
        "ring_matmul operands must be on device");
    TT_FATAL(act_tensor.device() == weight_tensor.device(), "ring_matmul inputs must reside on the same device");
    TT_FATAL(
        act_tensor.buffer() != nullptr && weight_tensor.buffer() != nullptr,
        "ring_matmul inputs must be allocated in device buffers");

    TT_FATAL(
        act_tensor.layout() == Layout::TILE && weight_tensor.layout() == Layout::TILE,
        "ring_matmul requires TILE layout for activation and weight");

    auto dtype_supported = [](tt::tt_metal::DataType dt) {
        return dt == DataType::BFLOAT16 || dt == DataType::BFLOAT8_B || dt == DataType::BFLOAT4_B ||
               dt == DataType::FLOAT32;
    };
    TT_FATAL(
        dtype_supported(act_tensor.dtype()) && dtype_supported(weight_tensor.dtype()),
        "ring_matmul supports only BFLOAT16, BFLOAT8_B, BFLOAT4_B, and FLOAT32 for inputs");

    TT_FATAL(act_tensor.is_sharded(), "ring_matmul requires sharded activation tensor");

    const auto& a_logical = act_tensor.logical_shape();
    const auto& w_logical = weight_tensor.logical_shape();
    TT_FATAL(a_logical.rank() >= 2 && w_logical.rank() >= 2, "ring_matmul expects rank >= 2 tensors");

    const uint32_t K = a_logical[-1];
    const uint32_t K_w = w_logical[-2];
    TT_FATAL(K == K_w, "ring_matmul inner dimensions must match, got K={} and K_w={}", K, K_w);

    const auto& a_padded = act_tensor.padded_shape();
    const auto& w_padded = weight_tensor.padded_shape();
    TT_FATAL(
        a_padded[-2] % TILE_HEIGHT == 0 && a_padded[-1] % TILE_WIDTH == 0,
        "ring_matmul activation must be tile-aligned");
    TT_FATAL(
        w_padded[-2] % TILE_HEIGHT == 0 && w_padded[-1] % TILE_WIDTH == 0, "ring_matmul weight must be tile-aligned");

    if (attrs.config.has_value()) {
        const auto& cfg = attrs.config.value();
        TT_FATAL(cfg.in0_block_w > 0, "in0_block_w must be > 0");
        TT_FATAL(cfg.out_subblock_h > 0 && cfg.out_subblock_w > 0, "Subblock sizes must be > 0");
        TT_FATAL(cfg.per_core_M > 0 && cfg.per_core_N > 0, "per_core_M and per_core_N must be > 0");
    }
}

void RingMatmulDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensors) {
    // Lighter validation for cache hits
    const auto& act_tensor = tensors.input_tensor;
    const auto& weight_tensor = tensors.weight_tensor;
    TT_FATAL(act_tensor.buffer() != nullptr && weight_tensor.buffer() != nullptr, "Tensors must be allocated");
}

RingMatmulDeviceOperation::spec_return_value_t RingMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensors) {
    const auto& in0_tensor = tensors.input_tensor;
    const auto& in1_tensor = tensors.weight_tensor;
    const auto& in0_shape = in0_tensor.logical_shape();
    const auto& in1_shape = in1_tensor.logical_shape();
    uint32_t N = in1_shape[-1];

    ttnn::Shape output_shape(in0_shape);
    output_shape[-1] = N;

    const auto& memory_config = attrs.output_mem_config.value_or(in0_tensor.memory_config());
    auto dtype = attrs.output_dtype.value_or(in0_tensor.dtype());

    return TensorSpec(output_shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config));
}

RingMatmulDeviceOperation::tensor_return_value_t RingMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensors) {
    auto output_spec = compute_output_specs(attrs, tensors);
    return create_device_tensor(output_spec, tensors.input_tensor.device());
}

std::tuple<RingMatmulDeviceOperation::operation_attributes_t, RingMatmulDeviceOperation::tensor_args_t>
RingMatmulDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    std::optional<unary::UnaryWithParam> fused_activation,
    const std::optional<RingMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const CoreRangeSet& hop_cores,
    const std::optional<tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    uint32_t num_global_cb_receivers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<CoreRangeSet> restricted_cores,
    bool untilize_out) {
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, false, true, true);

    return {
        operation_attributes_t{
            config,
            std::move(fused_activation),
            memory_config,
            dtype,
            kernel_config_val,
            hop_cores,
            global_cb,
            num_global_cb_receivers,
            sub_device_id,
            restricted_cores,
            untilize_out},
        tensor_args_t{input_tensor, weight_tensor}};
}

}  // namespace ttnn::operations::experimental::ring_matmul
