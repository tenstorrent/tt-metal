// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::dram_prefetcher {

struct ExecuteDramPrefetcher {
    static ttnn::Tensor invoke(const std::vector<ttnn::Tensor>& tensors);
};

}  // namespace operations::dram_prefetcher

constexpr auto dram_prefetcher = ttnn::register_operation_with_auto_launch_op<
    "ttnn::dram_prefetcher",
    ttnn::operations::dram_prefetcher::ExecuteDramPrefetcher>();

}  // namespace ttnn
