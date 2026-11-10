// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

struct HaloParams {
    ttnn::operations::sliding_window::SlidingWindowConfig config{};
    ttnn::operations::sliding_window::ParallelConfig parallel_config{};
    uint32_t pad_val = 0;
    bool remote_read = false;
    bool transpose_mcast = false;
    uint32_t max_out_nsticks_per_core = 0;
    uint32_t in_nsticks_per_core = 0;
    tt::tt_metal::MemoryConfig output_memory_config;
    bool is_out_tiled = false;
    bool config_tensors_in_dram = false;
    // Required: caller must pass compute_kernel_config so the halo's untilize
    // compute kernel runs with the correct fp32_dest_acc_en (see issue #43229).
    DeviceComputeKernelConfig compute_kernel_config;
    // y = x * scale + shift, then SiLU, per channel, applied inside the
    // untilize pass (TILE input only); the padding sticks keep the pad value. Default off.
    bool prologue_affine = false;
    // The border padding sticks come from `halo_boundary` (the neighbour
    // devices' boundary rows / columns, zeros at the image edge) instead of the pad value. Default off.
    bool external_halo = false;
    // The window's place in the device image the boundary was built for (a DRAM-sliced conv's width
    // slice); all zero = the whole device image. Flat fields so the program hash tells slices apart.
    uint32_t external_col_offset = 0;
    uint32_t external_boundary_w = 0;
    uint32_t external_dev_pad_left = 0;
    uint32_t external_dev_pad_right = 0;
    ttnn::operations::sliding_window::ExternalHaloWindow external_window() const {
        return {external_col_offset, external_boundary_w, external_dev_pad_left, external_dev_pad_right};
    }
};

struct HaloInputs {
    Tensor input;
    std::optional<Tensor> prologue_scale;  // [1, C] in TILE layout, row 0 holds the values (C = shard width)
    std::optional<Tensor> prologue_shift;
    // [1, 1, batch * sticks_per_image, C] ROW_MAJOR DRAM interleaved, one stick per page (see
    // ExternalHaloSpec)
    std::optional<Tensor> halo_boundary;
};

}  // namespace ttnn::prim
