// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_distributed_bw_utils.hpp"

#include <array>

#include <tt_stl/assert.hpp>
#include <tt_stl/small_vector.hpp>
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::normalization::rmsnorm_distributed_bw {
namespace {

Tensor cast_if_needed(const Tensor& tensor, DataType dtype) {
    return tensor.dtype() == dtype ? tensor : ttnn::typecast(tensor, dtype);
}

Tensor to_fp32(const Tensor& tensor) { return cast_if_needed(tensor, DataType::FLOAT32); }

Tensor gained(const Tensor& output_grad_fp32, const Tensor& rms, const std::optional<const Tensor>& weight) {
    if (weight.has_value()) {
        return ttnn::divide(ttnn::multiply(to_fp32(weight.value()), output_grad_fp32), rms);
    }
    return ttnn::divide(output_grad_fp32, rms);
}

// Backward runs in fp32 throughout, so only the two float types round-trip cleanly through
// to_fp32/cast_if_needed. Block formats would silently requantize every intermediate.
void validate_bw_tensor(const Tensor& tensor, std::string_view tensor_name, std::string_view op_name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{}: {} must be on device", op_name, tensor_name);
    TT_FATAL(
        tensor.layout() == Layout::TILE,
        "{}: {} must have TILE layout, got: {}",
        op_name,
        tensor_name,
        tensor.layout());
    TT_FATAL(
        tensor.dtype() == DataType::BFLOAT16 || tensor.dtype() == DataType::FLOAT32,
        "{}: {} must be BFLOAT16 or FLOAT32, got: {}. Typecast it before calling.",
        op_name,
        tensor_name,
        tensor.dtype());
    TT_FATAL(
        !tensor.is_sharded(), "{}: sharded {} is not supported yet; pass an interleaved tensor", op_name, tensor_name);
}

void validate_weight(const Tensor& weight, const Tensor& input, std::string_view op_name) {
    validate_bw_tensor(weight, "weight", op_name);
    TT_FATAL(weight.device() == input.device(), "{}: weight and input must be on the same device", op_name);

    // The gradient formulae broadcast weight over the leading dims and reduce dL/dgamma back to the
    // same shape, so anything but a [1, 1, 1, W] row would silently mean something else.
    const auto& shape = weight.logical_shape();
    TT_FATAL(
        shape.rank() == 4 && shape[0] == 1 && shape[1] == 1 && shape[2] == 1,
        "{}: weight must have shape [1, 1, 1, W], got {}. Reshape it to a single per-channel row.",
        op_name,
        shape);
    TT_FATAL(
        shape[3] == input.logical_shape()[3],
        "{}: weight last dim is {} but the input's local last dim is {}. Shard weight to the local hidden size "
        "rather than passing the full row.",
        op_name,
        shape[3],
        input.logical_shape()[3]);
}

}  // namespace

DeviceComputeKernelConfig resolve_compute_kernel_config(
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config, const Tensor& input) {
    return init_device_compute_kernel_config(
        input.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true);
}

void validate_bw_inputs(
    const Tensor& input,
    const Tensor& output_grad,
    const std::optional<const Tensor>& weight,
    std::string_view op_name) {
    validate_bw_tensor(input, "input", op_name);
    validate_bw_tensor(output_grad, "output_grad", op_name);
    TT_FATAL(input.device() == output_grad.device(), "{}: input and output_grad must be on the same device", op_name);
    TT_FATAL(
        input.logical_shape() == output_grad.logical_shape(),
        "{}: input and output_grad shapes must match, got {} vs {}",
        op_name,
        input.logical_shape(),
        output_grad.logical_shape());
    TT_FATAL(
        input.logical_shape().rank() == 4,
        "{}: input must be rank-4, got rank {}",
        op_name,
        input.logical_shape().rank());

    if (weight.has_value()) {
        validate_weight(weight.value(), input, op_name);
    }
}

void validate_stats_tensor(
    const Tensor& stats, const Tensor& input, std::string_view tensor_name, std::string_view op_name) {
    validate_bw_tensor(stats, tensor_name, op_name);
    TT_FATAL(stats.device() == input.device(), "{}: {} and input must be on the same device", op_name, tensor_name);
    const auto& shape = stats.logical_shape();
    TT_FATAL(shape.rank() == 4, "{}: {} must be rank-4, got rank {}", op_name, tensor_name, shape.rank());

    const uint32_t tile_w = stats.tensor_spec().tile().get_width();
    TT_FATAL(
        shape[3] >= tile_w && shape[3] % tile_w == 0,
        "{}: {} last dim ({}) must be a non-zero multiple of tile width ({}): one tile column per device",
        op_name,
        tensor_name,
        shape[3],
        tile_w);

    // A stats tensor whose leading dims disagree with the shard would broadcast instead of failing,
    // turning a wiring mistake into silently wrong gradients.
    const auto& input_shape = input.logical_shape();
    for (int dim = 0; dim < 3; ++dim) {
        TT_FATAL(
            shape[dim] == input_shape[dim],
            "{}: {} dim {} is {} but the input's is {}; {} must come from this shard's rows",
            op_name,
            tensor_name,
            dim,
            shape[dim],
            input_shape[dim],
            tensor_name);
    }
}

uint32_t num_devices_in_stats(const Tensor& stats) {
    return stats.logical_shape()[3] / stats.tensor_spec().tile().get_width();
}

Tensor mean_from_gathered_stats(
    const Tensor& stats, uint32_t local_width, const DeviceComputeKernelConfig& compute_kernel_config) {
    TT_FATAL(local_width > 0, "local_width must be > 0, got {}", local_width);
    const float full_width = static_cast<float>(local_width) * static_cast<float>(num_devices_in_stats(stats));
    auto summed = ttnn::sum(
        to_fp32(stats),
        /*dim_arg=*/3,
        /*keep_dim=*/true,
        std::nullopt,
        compute_kernel_config);
    return ttnn::multiply(summed, 1.0f / full_width);
}

Tensor rms_from_gathered_stats(
    const Tensor& stats, uint32_t local_width, float epsilon, const DeviceComputeKernelConfig& compute_kernel_config) {
    return ttnn::sqrt(ttnn::add(mean_from_gathered_stats(stats, local_width, compute_kernel_config), epsilon));
}

Tensor x_times_gained(
    const Tensor& input, const Tensor& output_grad, const Tensor& rms, const std::optional<const Tensor>& weight) {
    return ttnn::multiply(to_fp32(input), gained(to_fp32(output_grad), rms, weight), DataType::FLOAT32);
}

Tensor to_stats_layout(const Tensor& tensor) {
    const auto& shape = tensor.logical_shape();
    const uint32_t rank = shape.rank();
    TT_FATAL(tensor.layout() == Layout::TILE, "to_stats_layout requires a TILE tensor, got: {}", tensor.layout());
    const uint32_t tile_w = tensor.tensor_spec().tile().get_width();
    const uint32_t last = shape[rank - 1];
    if (last == tile_w) {
        return tensor;
    }
    TT_FATAL(last < tile_w, "Cannot pad last dim {} down to tile width {}", last, tile_w);

    // The consumer row-reduces the whole tile column, so the columns past the value have to be real zeros.
    ttsl::SmallVector<std::array<uint32_t, 2>> padding(rank, std::array<uint32_t, 2>{0, 0});
    padding.back() = {0, tile_w - last};
    return ttnn::pad(tensor, padding, 0.0f);
}

std::vector<std::optional<Tensor>> apply_backward(
    const Tensor& input,
    const Tensor& output_grad,
    const Tensor& rms,
    const Tensor& scale,
    const std::optional<const Tensor>& weight,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    auto x = to_fp32(input);
    auto dy = to_fp32(output_grad);
    auto g = gained(dy, rms, weight);
    auto dx =
        cast_if_needed(ttnn::subtract(g, ttnn::multiply(x, ttnn::divide(scale, ttnn::square(rms)))), input.dtype());

    std::vector<std::optional<Tensor>> grads{std::move(dx), std::nullopt};
    if (weight.has_value()) {
        auto dgamma = ttnn::sum(
            ttnn::multiply(dy, ttnn::divide(x, rms)),
            ttsl::SmallVector<int>{0, 1, 2},
            /*keep_dim=*/true,
            std::nullopt,
            compute_kernel_config);
        grads[1] = cast_if_needed(dgamma, weight->dtype());
    }
    return grads;
}

}  // namespace ttnn::operations::normalization::rmsnorm_distributed_bw
