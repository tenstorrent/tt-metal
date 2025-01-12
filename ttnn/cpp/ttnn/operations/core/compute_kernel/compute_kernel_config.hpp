// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <variant>
#include <tuple>
#include <optional>
#include "umd/device/types/arch.h"
#include "tt_metal/common/base_types.hpp"

namespace ttnn {

struct GrayskullComputeKernelConfig {
    MathFidelity math_fidelity = MathFidelity::LoFi;
    bool math_approx_mode = true;
    bool dst_full_sync_en = false;
};

struct WormholeComputeKernelConfig {
    MathFidelity math_fidelity = MathFidelity::LoFi;
    bool math_approx_mode = true;
    bool fp32_dest_acc_en = false;
    bool packer_l1_acc = false;
    bool dst_full_sync_en = false;
};

using BlackholeComputeKernelConfig = WormholeComputeKernelConfig;

using DeviceComputeKernelConfig = std::variant<GrayskullComputeKernelConfig, WormholeComputeKernelConfig>;

DeviceComputeKernelConfig init_device_compute_kernel_config(
    tt::ARCH arch,
    const std::optional<const DeviceComputeKernelConfig>& device_kernel_config,
    const MathFidelity default_fidelity = MathFidelity::LoFi,
    bool default_approx_mode = true,
    bool default_fp32_acc = false,
    bool default_l1_acc = false,
    bool default_dst_full_sync_en = false);

bool get_fp32_dest_acc_en(const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
MathFidelity get_math_fidelity(const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
tt::ARCH get_arch_from_compute_config(const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);

std::tuple<MathFidelity, bool, bool, bool, bool> get_compute_kernel_config_args(
    tt::ARCH arch, const DeviceComputeKernelConfig compute_kernel_config);

uint32_t get_dest_reg_count(
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<std::array<uint32_t, 2>> tile_shape = std::nullopt);

}  // namespace ttnn
