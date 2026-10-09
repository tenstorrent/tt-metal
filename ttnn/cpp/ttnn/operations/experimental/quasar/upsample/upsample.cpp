// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/upsample/upsample.hpp"

#include <tt-metalium/hal.hpp>

#include "ttnn/operations/experimental/quasar/upsample/device/upsample_device_operation.hpp"

namespace ttnn::operations::experimental::quasar {

ttnn::Tensor upsample(
    const ttnn::Tensor& input_tensor,
    std::variant<int, std::array<int, 2>, float, std::array<float, 2>> scale_factor,
    const std::string& mode,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    const tt::tt_metal::MemoryConfig mem_config = output_mem_config.value_or(input_tensor.memory_config());

    float scale_h = 1.0f;
    float scale_w = 1.0f;
    std::visit(
        [&scale_h, &scale_w](auto&& sf) {
            using T = std::decay_t<decltype(sf)>;
            if constexpr (std::is_same_v<T, int>) {
                scale_h = static_cast<float>(sf);
                scale_w = static_cast<float>(sf);
            } else if constexpr (std::is_same_v<T, std::array<int, 2>>) {
                scale_h = static_cast<float>(sf[0]);
                scale_w = static_cast<float>(sf[1]);
            } else if constexpr (std::is_same_v<T, float>) {
                scale_h = sf;
                scale_w = sf;
            } else if constexpr (std::is_same_v<T, std::array<float, 2>>) {
                scale_h = sf[0];
                scale_w = sf[1];
            } else {
                static_assert(sizeof(T) != 0, "Type check failed.");
            }
        },
        scale_factor);

    const ttnn::DeviceComputeKernelConfig config =
        compute_kernel_config.value_or(ttnn::init_device_compute_kernel_config(
            tt::tt_metal::hal::get_arch(), std::nullopt, tt::tt_metal::MathFidelity::HiFi4));

    return ttnn::prim::qsr::upsample(input_tensor, scale_h, scale_w, mode, mem_config, config);
}

}  // namespace ttnn::operations::experimental::quasar
