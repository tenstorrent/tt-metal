// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {

Tensor halo(
    const Tensor& input_tensor,
    const operations::sliding_window::SlidingWindowConfig& config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    uint32_t pad_val = 0x0,
    bool remote_read = false,
    bool transpose_mcast = true,
    bool is_out_tiled = true,
    bool config_tensors_in_dram = false,
    const std::optional<Tensor>& prologue_scale = std::nullopt,
    const std::optional<Tensor>& prologue_shift = std::nullopt,
    const std::optional<Tensor>& halo_boundary = std::nullopt,
    const operations::sliding_window::ExternalHaloWindow& external_window = {});

}  // namespace ttnn
