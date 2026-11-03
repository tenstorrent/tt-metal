// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_clip_grad_norm.hpp"

#include <algorithm>
#include <cmath>
#include <optional>

#include <tt-metalium/base_types.hpp>
#include "moreh_clip_grad_norm_step1/device/moreh_clip_grad_norm_step1_device_operation.hpp"
#include "moreh_clip_grad_norm_step3/device/moreh_clip_grad_norm_step3_device_operation.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_clip_grad_norm {

inline uint32_t get_num_device_cores(const MeshDevice& device) {
    const auto num_cores_x = static_cast<uint32_t>(device.compute_with_storage_grid_size().x);
    const auto num_cores_y = static_cast<uint32_t>(device.compute_with_storage_grid_size().y);
    return num_cores_x * num_cores_y;
}

// Step 1 and step 3 assign one input per core, so they are launched on chunks of at most one input per core.
template <typename Launch>
void for_each_launch(const std::vector<Tensor>& inputs, Launch&& launch) {
    const auto max_num_inputs = get_num_device_cores(*inputs.at(0).device());
    const auto total_num_inputs = static_cast<uint32_t>(inputs.size());
    for (uint32_t offset = 0; offset < total_num_inputs; offset += max_num_inputs) {
        const auto end = std::min(offset + max_num_inputs, total_num_inputs);
        launch(std::vector<Tensor>(inputs.begin() + offset, inputs.begin() + end), offset);
    }
}

}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm

namespace ttnn {

Tensor moreh_clip_grad_norm(
    const std::vector<Tensor>& inputs,
    float max_norm,
    float norm_type,
    bool error_if_nonfinite,
    const std::optional<const Tensor>& total_norm,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    // Early validation: if error_if_nonfinite is true, check if norm_type itself is NaN
    if (error_if_nonfinite && std::isnan(norm_type)) {
        TT_FATAL(
            false,
            "The total norm of order {} for gradients from `parameters` is non-finite, so it cannot be "
            "clipped. To disable this error and scale the gradients by the non-finite norm anyway, set "
            "`error_if_nonfinite=False`",
            norm_type);
    }

    // The root writes into total_norm if given, which must then have the same shape as its input.
    const auto pow_sum = ttnn::reshape(
        moreh_clip_grad_norm_pow_sum(inputs, norm_type, memory_config, compute_kernel_config),
        total_norm.has_value() ? total_norm->logical_shape() : Shape{1, 1});

    // Sum[Sum[|e|^p]]^(1/p)
    auto output_total_norm = norm_type == 2.0f
                                 ? ttnn::sqrt(pow_sum, /* fast_and_approximate_mode */ false, memory_config, total_norm)
                                 : ttnn::pow(pow_sum, 1.0f / norm_type, memory_config, total_norm);

    moreh_clip_grad_norm_scale(
        inputs, output_total_norm, max_norm, error_if_nonfinite, memory_config, compute_kernel_config);

    return output_total_norm;
}

Tensor moreh_clip_grad_norm_pow_sum(
    const std::vector<Tensor>& inputs,
    float norm_type,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    auto* device = inputs.at(0).device();
    const auto compute_kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4);

    // Run Step 1
    // Sum[|e|^p] of each input, one tile per input
    auto tmp_pow_sum = create_device_tensor(
        tt::tt_metal::TensorSpec(
            Shape{static_cast<uint32_t>(inputs.size()), 1, 1, 1},
            tt::tt_metal::TensorLayout(
                inputs.at(0).dtype(),
                tt::tt_metal::PageConfig(Layout::TILE),
                memory_config.value_or(inputs.at(0).memory_config()))),
        device);
    operations::moreh::moreh_clip_grad_norm::for_each_launch(
        inputs, [&](const std::vector<Tensor>& launch_inputs, uint32_t tile_offset) {
            ttnn::prim::moreh_clip_grad_norm_step1(
                launch_inputs, norm_type, tile_offset, tmp_pow_sum, memory_config, compute_kernel_config_val);
        });

    // Sum[Sum[|e|^p]]
    return ttnn::moreh_sum(
        tmp_pow_sum,
        /* dim */ int64_t{0},
        /* keepdim */ true,
        /* output */ std::nullopt,
        memory_config,
        compute_kernel_config_val);
}

void moreh_clip_grad_norm_scale(
    const std::vector<Tensor>& inputs,
    const Tensor& total_norm,
    float max_norm,
    bool error_if_nonfinite,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    auto* device = inputs.at(0).device();
    const auto compute_kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4);

    if (error_if_nonfinite) {
        // On a mesh, each device holds its own copy of the norm.
        for (const auto& device_total_norm : ttnn::distributed::get_device_tensors(total_norm.cpu())) {
            const auto fp32_total_norm =
                static_cast<float>(tt::tt_metal::host_buffer::get_as<bfloat16>(device_total_norm)[0]);
            TT_FATAL(
                std::isfinite(fp32_total_norm),
                "The total norm for gradients from `parameters` is non-finite, so it cannot be clipped. To disable "
                "this error and scale the gradients by the non-finite norm anyway, set `error_if_nonfinite=False`");
        }
    }

    // max_norm / (total_norm + 1e-6)
    Tensor max_norm_tensor = ttnn::full(Shape({1}), max_norm, inputs.at(0).dtype(), Layout::TILE, *device);
    Tensor added = ttnn::add(total_norm, 1e-6f);
    auto clip_coef = ttnn::div(max_norm_tensor, added);
    // min(clip_coef, 1.0f)
    Tensor scalar = ttnn::full(Shape({1}), 1.0f, inputs.at(0).dtype(), Layout::TILE, *device);
    auto clip_coef_clamped = ttnn::minimum(clip_coef, scalar);
    scalar.deallocate();
    max_norm_tensor.deallocate();

    // Run Step 3
    // Inplace update inputs(inputs *= clip_coef_clamped)
    operations::moreh::moreh_clip_grad_norm::for_each_launch(
        inputs, [&](const std::vector<Tensor>& launch_inputs, uint32_t /*tile_offset*/) {
            ttnn::prim::moreh_clip_grad_norm_step3(
                launch_inputs, clip_coef_clamped, memory_config, compute_kernel_config_val);
        });
}

}  // namespace ttnn
