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
    bool prologue_affine = false;
    bool external_halo = false;
    ttnn::operations::sliding_window::ExternalHaloWindow external_window{};
};

// Optional inputs of halo, each off when unset.
// prologue_scale / prologue_shift ([1, C] TILE, BFLOAT16 TILE input only): the untilize pass writes silu(x * scale +
// shift) instead of x; padding sticks are unchanged. halo_boundary (height-sharded input only): padding sticks come
// from this ROW_MAJOR DRAM tensor instead of pad_val, laid out as ExternalHaloSpec describes; external_window places
// the input inside the image the border belongs to.
struct HaloOptionalInputs {
    std::optional<Tensor> halo_boundary;
    ttnn::operations::sliding_window::ExternalHaloWindow external_window{};
    std::optional<Tensor> prologue_scale;
    std::optional<Tensor> prologue_shift;
};

struct HaloInputs {
    Tensor input;
    std::optional<Tensor> prologue_scale;
    std::optional<Tensor> prologue_shift;
    std::optional<Tensor> halo_boundary;
};

}  // namespace ttnn::prim
