// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/neighbor_pad_halo/device/neighbor_pad_halo_device_operation_types.hpp"
#include <tt-metalium/sub_device_types.hpp>

namespace ttnn::experimental {

// Fabric halo neighbor-pad: exchanges the H+W boundary rows between H/W-parallel mesh neighbors.
//
// Default (compact) mode: writes only the exchanged halo into `halo_buffer` (compact DRAM,
// pre-allocated by the caller) as [H-top | H-bot | W-left | W-right] and returns it — pure
// transport, bounded by DRAM read + fabric bandwidth.
//
// Padded-output (fold) mode: pass `padded_output` to additionally scatter the full padded
// [.., H+2pH, W+2pW, C] result in the same dispatch — the interior copy overlaps the fabric
// exchange and the border is scattered from the compact staging buffer. This folds what would
// otherwise be a separate scatter op into one launch; the op returns `padded_output`.
//
// Args:
//   input        : input tensor [B, T, H, W, C] row-major bfloat16/float32 (interior; may itself
//                  carry padding — see input_pad_h/w)
//   halo_buffer  : pre-allocated compact DRAM halo buffer [H-top | H-bot | W-left | W-right].
//                  In fold mode this is the internal staging buffer, not the returned tensor.
//   np_padding_h : halo rows per side in the H dimension (typically 1 for k=3)
//   np_padding_w : halo columns per side in the W dimension
//   np_cluster_axis : mesh axis for H-parallel devices
//   np_num_links    : number of fabric links for the H exchange
//   np_topology     : fabric topology (Linear or Ring)
//   h_neighbor_semaphore / barrier_semaphore / w_neighbor_semaphore : NP handshake semaphores
//   np_pad_dim2          : secondary (W-axis) padding dim index (must be > 0 — 2D required)
//   np_pad2_left/right   : padding amounts for the secondary dim
//   np_pad2_cluster_axis : cluster axis for the secondary dim
//   np_pad2_num_links    : links for the secondary dim
//   padding_mode         : "zeros" or "replicate"
//   memory_config        : optional output memory config (defaults to the halo_buffer's)
//   input_pad_h/w        : padding already present on `input`; the reader strides over it and the
//                          halo geometry is reduced to the interior (H_dev, W_dev).
//   padded_output        : opt-in fold-mode target buffer (see above); std::nullopt = compact mode
//   border_only          : fold mode only — scatter just the border, skip the interior copy
//   logical_h/w          : logical extent per device; rows/cols at or beyond it are masked to zero
//                          in-kernel (0 = no masking)
//
// Returns: `padded_output` in fold mode, else the compact halo buffer (written in place).
ttnn::Tensor neighbor_pad_halo(
    const ttnn::Tensor& input,
    const ttnn::Tensor& halo_buffer,
    uint32_t np_padding_h,
    uint32_t np_padding_w,
    uint32_t np_cluster_axis,
    size_t np_num_links,
    ttnn::ccl::Topology np_topology,
    const GlobalSemaphore& h_neighbor_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    const GlobalSemaphore& w_neighbor_semaphore,
    uint32_t np_pad_dim2,
    uint32_t np_pad2_left,
    uint32_t np_pad2_right,
    uint32_t np_pad2_cluster_axis,
    size_t np_pad2_num_links,
    const std::string& padding_mode = "zeros",
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    uint32_t input_pad_h = 0,
    uint32_t input_pad_w = 0,
    const std::optional<ttnn::Tensor>& padded_output = std::nullopt,
    bool border_only = false,
    uint32_t logical_h = 0,
    uint32_t logical_w = 0);

}  // namespace ttnn::experimental
