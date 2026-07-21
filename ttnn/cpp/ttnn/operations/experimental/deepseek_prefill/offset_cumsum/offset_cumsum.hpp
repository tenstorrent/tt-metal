// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::offset_cumsum {

// `use_l1_small_for_semaphores`: route the internal cross-device all-gather's global semaphores to the
// L1_SMALL region instead of main L1. The all-gather creates its sync semaphores internally and keeps
// them resident; in main L1 they pin the L1 floor and clash with the next layer's MLA static CBs. Routing
// them to L1_SMALL keeps them off the main-L1 floor. Requires the device opened with l1_small_size > 0.
std::array<ttnn::Tensor, 3> offset_cumsum(
    const ttnn::Tensor& input_tensor,
    uint32_t cluster_axis,
    uint32_t num_links,
    uint32_t experts_per_chip,
    const ttnn::MemoryConfig& memory_config,
    bool use_l1_small_for_semaphores = false);

}  // namespace ttnn::operations::experimental::deepseek_prefill::offset_cumsum

namespace ttnn {
using operations::experimental::deepseek_prefill::offset_cumsum::offset_cumsum;
}  // namespace ttnn
