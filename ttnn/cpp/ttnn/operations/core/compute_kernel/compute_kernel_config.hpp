// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <tuple>
#include <optional>
#include <umd/device/types/arch.hpp>
#include <tt-metalium/base_types.hpp>
#include "ttnn/operations/compute_throttle_utils.hpp"

namespace ttnn {

// Unified compute kernel configuration for all supported architectures.
struct ComputeKernelConfig {
    MathFidelity math_fidelity = MathFidelity::LoFi;
    bool math_approx_mode = true;
    bool fp32_dest_acc_en = false;
    bool packer_l1_acc = false;
    bool dst_full_sync_en = false;
    ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level =
        ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE;
};

// Type aliases for backward compatibility
using DeviceComputeKernelConfig = ComputeKernelConfig;
using WormholeComputeKernelConfig = ComputeKernelConfig;
using BlackholeComputeKernelConfig = ComputeKernelConfig;

DeviceComputeKernelConfig init_device_compute_kernel_config(
    tt::ARCH arch,
    const std::optional<const DeviceComputeKernelConfig>& device_kernel_config,
    MathFidelity default_fidelity = MathFidelity::LoFi,
    bool default_approx_mode = true,
    bool default_fp32_acc = false,
    bool default_l1_acc = false,
    bool default_dst_full_sync_en = false,
    ttnn::operations::compute_throttle_utils::ThrottleLevel default_throttle_level =
        ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE);

bool get_fp32_dest_acc_en(const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
bool get_dst_full_sync_en(const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
MathFidelity get_math_fidelity(const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
ttnn::operations::compute_throttle_utils::ThrottleLevel get_throttle_level(
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);

std::tuple<MathFidelity, bool, bool, bool, bool> get_compute_kernel_config_args(
    tt::ARCH arch, DeviceComputeKernelConfig compute_kernel_config);

uint32_t get_dest_reg_count(
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<std::array<uint32_t, 2>> tile_shape = std::nullopt);

}  // namespace ttnn
