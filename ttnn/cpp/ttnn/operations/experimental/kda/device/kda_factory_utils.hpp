// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim::kda_factory_detail {

tt::tt_metal::ComputeConfigDescriptor kda_compute_cfg(
    tt::ARCH arch, const DeviceComputeKernelConfig& config, bool honor_caller_config = true);

struct KdaPrepWorkDist {
    std::vector<tt::tt_metal::CoreCoord> cores;
    std::vector<uint32_t> wi_start;
    std::vector<uint32_t> wi_count;
    tt::tt_metal::CoreRangeSet core_set;
};

KdaPrepWorkDist distribute_prep(tt::tt_metal::CoreCoord grid, uint32_t total, uint32_t core_cap);

}  // namespace ttnn::experimental::prim::kda_factory_detail
