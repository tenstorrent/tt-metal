// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <ostream>
#include <tuple>
#include <optional>
#include <umd/device/types/arch.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include "ttnn/operations/compute_throttle_utils.hpp"

namespace ttnn {

// Unified compute kernel configuration for all supported architectures.
struct ComputeKernelConfig {
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::LoFi;
    bool math_approx_mode = true;
    bool fp32_dest_acc_en = false;
    bool packer_l1_acc = false;
    bool dst_full_sync_en = false;
    ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level =
        ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE;
};

inline std::ostream& operator<<(std::ostream& os, const ComputeKernelConfig& cfg) {
    os << "ComputeKernelConfig(math_fidelity=" << cfg.math_fidelity << ",math_approx_mode=" << cfg.math_approx_mode
       << ",fp32_dest_acc_en=" << cfg.fp32_dest_acc_en << ",packer_l1_acc=" << cfg.packer_l1_acc
       << ",dst_full_sync_en=" << cfg.dst_full_sync_en << ",throttle_level=" << cfg.throttle_level << ")";
    return os;
}

// Type aliases for backward compatibility
using DeviceComputeKernelConfig = ComputeKernelConfig;
using WormholeComputeKernelConfig = ComputeKernelConfig;
using BlackholeComputeKernelConfig = ComputeKernelConfig;

DeviceComputeKernelConfig init_device_compute_kernel_config(
    tt::ARCH arch,
    const std::optional<const DeviceComputeKernelConfig>& device_kernel_config,
    tt::tt_metal::MathFidelity default_fidelity = tt::tt_metal::MathFidelity::LoFi,
    bool default_approx_mode = true,
    bool default_fp32_acc = false,
    bool default_l1_acc = false,
    bool default_dst_full_sync_en = false,
    ttnn::operations::compute_throttle_utils::ThrottleLevel default_throttle_level =
        ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE);

void verify_numerical_configuration(
    tt::ARCH arch, const std::optional<const DeviceComputeKernelConfig>& user_compute_kernel_config);

bool get_fp32_dest_acc_en(const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
bool get_dst_full_sync_en(const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
tt::tt_metal::MathFidelity get_math_fidelity(const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
ttnn::operations::compute_throttle_utils::ThrottleLevel get_throttle_level(
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);

std::tuple<tt::tt_metal::MathFidelity, bool, bool, bool, bool> get_compute_kernel_config_args(
    tt::ARCH arch, const DeviceComputeKernelConfig& compute_kernel_config);

// Maps the four hardware knobs (math_fidelity, math_approx_mode, fp32_dest_acc_en,
// dst_full_sync_en) of a ComputeKernelConfig to a Metal 2.0 ComputeHardwareConfig. The knobs are
// common to both generations; `arch` selects the matching alternative (ComputeGen2Config on Quasar,
// else ComputeGen1Config) — the config's generation must match the target platform.
// packer_l1_acc and throttle_level are op-side concerns, not translated.
//
// The result's per-DFB unpack_modes table is left default for the program factory to set.
// bfp_pack_precision_mode is likewise left default (rarely set non-default).
//
// The following Gen2-only TEMPORARY fields are also not set here; a use site that needs
// them should set them on the returned config instead:
//   enable_2x_src_register, unpack_to_dest_en
tt::tt_metal::experimental::ComputeHardwareConfig to_compute_hardware_config(
    tt::ARCH arch, const ComputeKernelConfig& config);

uint32_t get_dest_reg_count(
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<std::array<uint32_t, 2>> tile_shape = std::nullopt);

}  // namespace ttnn
