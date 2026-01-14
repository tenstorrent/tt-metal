// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_clip_grad_norm.hpp"

#include <optional>

#include <tt-metalium/base_types.hpp>
#include "moreh_clip_grad_norm_step1/device/moreh_clip_grad_norm_step1_device_operation.hpp"
#include "moreh_clip_grad_norm_step2/device/moreh_clip_grad_norm_step2_device_operation.hpp"
#include "moreh_clip_grad_norm_step3/device/moreh_clip_grad_norm_step3_device_operation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace {

template <typename OutputDataType, typename InputDataType>
std::vector<OutputDataType> cast_vec(tt::stl::Span<const InputDataType> data_to_convert) {
    std::vector<OutputDataType> converted_data;
    for (auto datum : data_to_convert) {
        if constexpr (std::is_same_v<OutputDataType, float> and std::is_same_v<InputDataType, bfloat16>) {
            converted_data.push_back(static_cast<float>(datum));
        } else if constexpr (std::is_same_v<OutputDataType, uint32_t> and std::is_same_v<InputDataType, bfloat16>) {
            converted_data.push_back((uint32_t)std::bit_cast<uint16_t>(datum));
        } else {
            converted_data.push_back(static_cast<OutputDataType>(datum));
        }
    }
    return converted_data;
}
}  // namespace

namespace ttnn::operations::moreh::moreh_clip_grad_norm {

inline uint32_t get_num_device_cores(IDevice* device) {
    const auto num_cores_x = static_cast<uint32_t>(device->compute_with_storage_grid_size().x);
    const auto num_cores_y = static_cast<uint32_t>(device->compute_with_storage_grid_size().y);
    return num_cores_x * num_cores_y;
}

Tensor MorehClipGradNorm::invoke(
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
    auto* device = inputs.at(0).device();
    const auto compute_kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);

    // Loop variable
    const auto max_num_inputs = get_num_device_cores(device);
    const auto total_num_inputs = static_cast<uint32_t>(inputs.size());
    const auto num_iter = (total_num_inputs + max_num_inputs - 1) / max_num_inputs;
    // Store intermediate reduction of Sum[|e|^p]
    auto tmp_pow_sum = create_device_tensor(
        ttnn::TensorSpec(
            Shape{static_cast<uint32_t>(inputs.size()), 1, 1},
            tt::tt_metal::TensorLayout(
                inputs.at(0).dtype(),
                tt::tt_metal::PageConfig(Layout::TILE),
                memory_config.value_or(inputs.at(0).memory_config()))),
        device);

    // Run Step 1
    // Sum[|e|^p]
    uint32_t tile_offset{0};
    auto num_inputs = total_num_inputs;
    for (uint32_t i = 0; i < num_iter; i++) {
        const auto num_inputs_at_this_iter = std::min(num_inputs, max_num_inputs);

        ttnn::prim::moreh_clip_grad_norm_step1(
            std::vector<Tensor>(inputs.begin() + tile_offset, inputs.begin() + tile_offset + num_inputs_at_this_iter),
            norm_type,
            tile_offset,
            tmp_pow_sum,
            memory_config,
            compute_kernel_config_val);

        if (i < (num_iter - 1)) {
            tile_offset += num_inputs_at_this_iter;
            num_inputs -= num_inputs_at_this_iter;
        }
    }

    // Run Step 2
    // Sum[Sum[|e|^p]]^(1/p)
    auto output_total_norm = ttnn::prim::moreh_clip_grad_norm_step2(
        tmp_pow_sum,
        norm_type,
        total_norm,
        memory_config,
        init_device_compute_kernel_config(inputs.at(0).device()->arch(), compute_kernel_config, MathFidelity::HiFi4));

    if (error_if_nonfinite) {
        const auto fp32_total_norm =
            cast_vec<float>(tt::tt_metal::host_buffer::get_as<bfloat16>(output_total_norm.cpu())).at(0);
        TT_FATAL(
            std::isfinite(fp32_total_norm),
            "The total norm of order {} for gradients from `parameters` is non-finite, so it cannot be "
            "clipped. To disable this error and scale the gradients by the non-finite norm anyway, set "
            "`error_if_nonfinite=False`",
            norm_type);
    }

    // max_norm / (total_norm + 1e-6)
    Tensor max_norm_tensor = ttnn::full(Shape({1}), max_norm, inputs.at(0).dtype(), Layout::TILE, *device);
    Tensor added = ttnn::add(output_total_norm, 1e-6f);
    auto clip_coef = ttnn::div(max_norm_tensor, added);
    // min(clip_coef, 1.0f)
    Tensor scalar = ttnn::full(Shape({1}), 1.0f, inputs.at(0).dtype(), Layout::TILE, *device);
    auto clip_coef_clamped = ttnn::minimum(clip_coef, scalar);
    scalar.deallocate();
    max_norm_tensor.deallocate();

    // Run Step 3
    // Inplace update inputs(inputs *= clip_coef_clamped)
    uint32_t start_input_idx{0};
    num_inputs = total_num_inputs;
    for (uint32_t i = 0; i < num_iter; ++i) {
        const auto num_inputs_at_this_iter = std::min(num_inputs, max_num_inputs);

        auto input_tensors = std::vector<Tensor>(
            inputs.begin() + start_input_idx, inputs.begin() + start_input_idx + num_inputs_at_this_iter);

        ttnn::prim::moreh_clip_grad_norm_step3(
            input_tensors, clip_coef_clamped, memory_config, compute_kernel_config_val);

        if (i < (num_iter - 1)) {
            start_input_idx += num_inputs_at_this_iter;
            num_inputs -= num_inputs_at_this_iter;
        }
    }

    return output_total_norm;
}

}  // namespace ttnn::operations::moreh::moreh_clip_grad_norm
