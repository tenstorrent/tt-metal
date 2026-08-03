// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/per_core_allocation_cb.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::matmul_decode {

// The weight's CB aliases the weight buffer's L1 instead of staging a copy, which is what makes the
// resident-weight decode matmul zero-copy -- and is exactly the binding that needs per-core
// treatment when the weight is per-core allocated. See ttnn/per_core_allocation_cb.hpp for why.
//
// `cb_core_ranges` is the set the kernels expect the CB on, which may be wider than the weight's
// shard grid (the merged compute bounding box in the full/partial factories).
inline std::vector<tt::tt_metal::CBDescriptor> make_weight_cb_descriptors(
    const Tensor& weight,
    const CoreRangeSet& cb_core_ranges,
    const tt::tt_metal::CBFormatDescriptor& format,
    uint32_t total_size) {
    return make_per_core_cb_descriptors(
        weight,
        tt::tt_metal::CBDescriptor{
            .total_size = total_size,
            .core_ranges = cb_core_ranges,
            .format_descriptors = {{format}},
            .buffer = weight.buffer(),
        });
}

}  // namespace ttnn::operations::experimental::matmul_decode
