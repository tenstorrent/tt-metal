// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include "compute_kernel_config.hpp"
#include "ttnn/device.hpp"

#define DATUMS_PER_ROW (16)

// This parameter is the same for all supported architectures
// Check this invariant when adding new architectures
#define DEST_REGISTER_FULL_SIZE (64 * 16)

namespace ttnn {

DeviceComputeKernelConfig init_device_compute_kernel_config(
    tt::ARCH arch,
    const std::optional<const DeviceComputeKernelConfig>& device_kernel_config,
    const MathFidelity default_fidelity,
    bool default_approx_mode,
    bool default_fp32_acc,
    bool default_l1_acc,
    bool default_dst_full_sync_en,
    ttnn::operations::compute_throttle_utils::ThrottleLevel default_throttle_level) {
    TT_ASSERT(ttnn::device::is_wormhole_or_blackhole(arch), "Only Wormhole and Blackhole architectures are supported");

    if (device_kernel_config.has_value()) {
        return device_kernel_config.value();
    }

    return ComputeKernelConfig{
        .math_fidelity = default_fidelity,
        .math_approx_mode = default_approx_mode,
        .fp32_dest_acc_en = default_fp32_acc,
        .packer_l1_acc = default_l1_acc,
        .dst_full_sync_en = default_dst_full_sync_en,
        .throttle_level = default_throttle_level};
}

bool get_fp32_dest_acc_en(const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    if (not compute_kernel_config.has_value()) {
        return false;
    }
    return compute_kernel_config.value().fp32_dest_acc_en;
}

bool get_dst_full_sync_en(const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    if (not compute_kernel_config.has_value()) {
        return false;
    }
    return compute_kernel_config.value().dst_full_sync_en;
}

MathFidelity get_math_fidelity(const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    if (not compute_kernel_config.has_value()) {
        return MathFidelity::Invalid;
    }
    return compute_kernel_config.value().math_fidelity;
}

ttnn::operations::compute_throttle_utils::ThrottleLevel get_throttle_level(
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    if (not compute_kernel_config.has_value()) {
        return ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE;
    }
    return compute_kernel_config.value().throttle_level;
}

std::tuple<MathFidelity, bool, bool, bool, bool> get_compute_kernel_config_args(
    tt::ARCH arch, const DeviceComputeKernelConfig compute_kernel_config) {
    TT_ASSERT(ttnn::device::is_wormhole_or_blackhole(arch), "Only Wormhole and Blackhole architectures are supported");
    return std::make_tuple(
        compute_kernel_config.math_fidelity,
        compute_kernel_config.math_approx_mode,
        compute_kernel_config.fp32_dest_acc_en,
        compute_kernel_config.packer_l1_acc,
        compute_kernel_config.dst_full_sync_en);
}

uint32_t get_dest_reg_count(
    const DeviceComputeKernelConfig& compute_kernel_config, std::optional<std::array<uint32_t, 2>> tile_shape) {
    uint32_t tile_height;
    uint32_t tile_width;
    if (tile_shape.has_value()) {
        std::array<uint32_t, 2>& shape = tile_shape.value();
        tile_height = shape[0];
        tile_width = shape[1];
    } else {
        tile_height = tt::constants::TILE_HEIGHT;
        tile_width = tt::constants::TILE_WIDTH;
    }
    // Note: if DATUMS_PER_ROW will change in a future architecture, then
    // this code will need to be updated to use an architecture specific value.
    uint32_t available_reg_count = (DEST_REGISTER_FULL_SIZE * DATUMS_PER_ROW) / (tile_width * tile_height);
    if (!compute_kernel_config.dst_full_sync_en) {
        available_reg_count /= 2;
    }
    // Note: using bfloat16 as baseline to be conservative, even
    // if smaller formats could have a larger register count.
    if (compute_kernel_config.fp32_dest_acc_en) {
        available_reg_count /= 2;
    }
    return available_reg_count;
}

}  // namespace ttnn
