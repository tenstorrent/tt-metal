// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {

Tensor halo(
    const Tensor& input_tensor,
    const operations::sliding_window::SlidingWindowConfig& config,
    uint32_t pad_val = 0x0,
    bool remote_read = false,
    bool transpose_mcast = true,
    bool is_out_tiled = true,
    bool config_tensors_in_dram = false);

}  // namespace ttnn
