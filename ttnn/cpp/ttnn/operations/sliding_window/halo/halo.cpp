// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "halo.hpp"

#include "ttnn/operations/sliding_window/halo/device/halo_device_operation.hpp"

namespace ttnn {

Tensor halo(
    const Tensor& input_tensor,
    const operations::sliding_window::SlidingWindowConfig& config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    uint32_t pad_val,
    bool remote_read,
    bool transpose_mcast,
    bool is_out_tiled,
    bool config_tensors_in_dram,
    const std::optional<Tensor>& prologue_scale,
    const std::optional<Tensor>& prologue_shift,
    const std::optional<Tensor>& halo_boundary,
    const operations::sliding_window::ExternalHaloWindow& external_window) {
    return prim::halo(
        input_tensor,
        config,
        compute_kernel_config,
        pad_val,
        remote_read,
        transpose_mcast,
        is_out_tiled,
        config_tensors_in_dram,
        prologue_scale,
        prologue_shift,
        halo_boundary,
        external_window);
}

}  // namespace ttnn
