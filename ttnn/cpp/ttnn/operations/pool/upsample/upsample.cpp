// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample.hpp"
#include <algorithm>
#include "tt-metalium/shape.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_device_operation.hpp"
#include "tt-metalium/buffer_types.hpp"
#include "tt-metalium/work_split.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/math.hpp>

namespace ttnn::operations::upsample {

ttnn::Tensor upsample(
    const ttnn::Tensor& input_tensor,
    std::variant<int, std::array<int, 2>, float, std::array<float, 2>> scale_factor,
    const std::string& mode,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    tt::tt_metal::MemoryConfig mem_config = output_mem_config.value_or(input_tensor.memory_config());

    // Parse scale factors from variant - extract as floats
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

    // Validation is handled by the device operation's validate_on_program_cache_miss

    auto* device = input_tensor.device();
    TT_FATAL(device != nullptr, "upsample expects an allocated device tensor as input");
    ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(
        ttnn::init_device_compute_kernel_config(device->arch(), std::nullopt, tt::tt_metal::MathFidelity::HiFi4));

    // TODO(nuked-op halo): restore bilinear mode (halo preprocessing + bilinear program factory)
    TT_FATAL(mode != "bilinear", "upsample bilinear mode is not available (halo / sliding_window removed)");

    // For nearest mode, pass to unified prim (device op handles factory selection)
    return ttnn::prim::upsample(input_tensor, scale_h, scale_w, mode, mem_config, config);
}
}  // namespace ttnn::operations::upsample
