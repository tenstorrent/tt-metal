// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/constants.hpp"
#include "compute_kernel_config.hpp"
#include "ttnn/device.hpp"

#define DATUMS_PER_ROW 16

namespace ttnn {

DeviceComputeKernelConfig init_device_compute_kernel_config(
    tt::ARCH arch,
    const std::optional<const DeviceComputeKernelConfig>& device_kernel_config,
    const MathFidelity default_fidelity,
    bool default_approx_mode,
    bool default_fp32_acc,
    bool default_l1_acc,
    bool default_dst_full_sync_en) {
    DeviceComputeKernelConfig defaultConfig;
    if (device_kernel_config.has_value()) {
        auto compute_kernel_config = device_kernel_config.value();
        std::visit(
            [&](auto&& compute_kernel_config) {
                using T = std::decay_t<decltype(compute_kernel_config)>;
                if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
                    TT_ASSERT(arch == tt::ARCH::GRAYSKULL, "kernel config is not for graykull");
                    MathFidelity math_fidelity = compute_kernel_config.math_fidelity;
                    bool math_approx_mode = compute_kernel_config.math_approx_mode;
                    bool dst_full_sync_en = compute_kernel_config.dst_full_sync_en;
                    defaultConfig = GrayskullComputeKernelConfig{.math_fidelity = math_fidelity,
                                                                 .math_approx_mode = math_approx_mode,
                                                                 .dst_full_sync_en = dst_full_sync_en};
                } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
                    TT_ASSERT(ttnn::device::is_wormhole_or_blackhole(arch),
                              "kernel config is not for wormhole_b0 or blackhole");
                    MathFidelity math_fidelity = compute_kernel_config.math_fidelity;
                    bool math_approx_mode = compute_kernel_config.math_approx_mode;
                    bool fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en;
                    bool packer_l1_acc = compute_kernel_config.packer_l1_acc;
                    bool dst_full_sync_en = compute_kernel_config.dst_full_sync_en;
                    defaultConfig = WormholeComputeKernelConfig{.math_fidelity = math_fidelity,
                                                                .math_approx_mode = math_approx_mode,
                                                                .fp32_dest_acc_en = fp32_dest_acc_en,
                                                                .packer_l1_acc = packer_l1_acc,
                                                                .dst_full_sync_en = dst_full_sync_en};
                } else {
                    TT_THROW("arch not supported");
                }
            },
            compute_kernel_config);
        return defaultConfig;
    } else {
        if (arch == tt::ARCH::GRAYSKULL) {
            return GrayskullComputeKernelConfig{.math_fidelity = default_fidelity,
                                                .math_approx_mode = default_approx_mode};
        } else if (arch == tt::ARCH::WORMHOLE_B0 || arch == tt::ARCH::BLACKHOLE) {
            return WormholeComputeKernelConfig{.math_fidelity = default_fidelity,
                                               .math_approx_mode = default_approx_mode,
                                               .fp32_dest_acc_en = default_fp32_acc,
                                               .packer_l1_acc = default_l1_acc,
                                               .dst_full_sync_en = default_dst_full_sync_en};
        } else {
            TT_THROW("arch not supported");
        }
    }
}

bool get_fp32_dest_acc_en(const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    if (not compute_kernel_config.has_value()) {
        return false;
    }
    return std::visit(
        [](auto&& compute_kernel_config) -> bool {
            using T = std::decay_t<decltype(compute_kernel_config)>;
            if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
                return false;
            } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
                return compute_kernel_config.fp32_dest_acc_en;
            } else {
                TT_THROW("arch not supported");
            }
        },
        compute_kernel_config.value());
}

MathFidelity get_math_fidelity(const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    if (not compute_kernel_config.has_value()) {
        return MathFidelity::Invalid;
    }
    return std::visit(
        [](auto&& compute_kernel_config) -> MathFidelity {
            using T = std::decay_t<decltype(compute_kernel_config)>;
            if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
                return compute_kernel_config.math_fidelity;
            } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
                return compute_kernel_config.math_fidelity;
            } else {
                TT_THROW("arch not supported");
            }
        },
        compute_kernel_config.value());
}

std::tuple<MathFidelity, bool, bool, bool, bool> get_compute_kernel_config_args(
    tt::ARCH arch,
    const DeviceComputeKernelConfig compute_kernel_config) {
    MathFidelity math_fidelity;
    bool math_approx_mode;
    bool fp32_dest_acc_en;
    bool packer_l1_acc;
    bool dst_full_sync_en;

    std::visit(
        [&](auto&& compute_kernel_config) {
            using T = std::decay_t<decltype(compute_kernel_config)>;
            if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
                TT_ASSERT(arch == tt::ARCH::GRAYSKULL, "kernel config is not for graykull");
                math_fidelity = compute_kernel_config.math_fidelity;
                math_approx_mode = compute_kernel_config.math_approx_mode;
                fp32_dest_acc_en = false;
                packer_l1_acc = false;
                dst_full_sync_en = false;
            } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
                TT_ASSERT(ttnn::device::is_wormhole_or_blackhole(arch),
                          "kernel config is not for wormhole_b0 or blackhole");
                math_fidelity = compute_kernel_config.math_fidelity;
                math_approx_mode = compute_kernel_config.math_approx_mode;
                fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en;
                packer_l1_acc = compute_kernel_config.packer_l1_acc;
                dst_full_sync_en = compute_kernel_config.dst_full_sync_en;
            } else {
                TT_THROW("arch not supported");
            }
        },
        compute_kernel_config);

    return std::make_tuple(math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en);
}

uint32_t get_dest_reg_count(const DeviceComputeKernelConfig& compute_kernel_config,
                            std::optional<std::array<uint32_t, 2>> tile_shape) {
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
    std::visit(
        [&](auto&& compute_kernel_config) {
            using T = std::decay_t<decltype(compute_kernel_config)>;
            if (!compute_kernel_config.dst_full_sync_en) {
                available_reg_count /= 2;
            }
            if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
                // Note: using bfloat16 as baseline to be conservative, even
                // if smaller formats could have a larger register count.
                if (compute_kernel_config.fp32_dest_acc_en) {
                    available_reg_count /= 2;
                }
            }
        },
        compute_kernel_config);
    return available_reg_count;
}

}  // namespace ttnn
