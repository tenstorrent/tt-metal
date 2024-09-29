// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device.hpp"

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

std::tuple<MathFidelity, bool, bool, bool, bool> get_compute_kernel_config_args(tt::ARCH arch, const DeviceComputeKernelConfig compute_kernel_config);

}  // namespace ttnn
