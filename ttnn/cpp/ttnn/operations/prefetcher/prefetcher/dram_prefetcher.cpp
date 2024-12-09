// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher.hpp"
#include <optional>

#include "device/dram_prefetcher_op.hpp"
#include "tt_metal/impl/buffers/global_circular_buffer.hpp"
#include "tt_metal/include/tt_metal/global_circular_buffer.hpp"

namespace ttnn::operations::dram_prefetcher {

ttnn::Tensor ExecuteDramPrefetcher::invoke(
    std::vector<ttnn::Tensor>& tensors,
    const Tensor& tensor_addrs,
    const std::optional<const tt::tt_metal::v1::experimental::GlobalCircularBuffer>& global_cb) {
    operation::run(DramPrefetcher{tensor_addrs, global_cb}, tensors);
    return tensors[0];
}

}  // namespace ttnn::operations::dram_prefetcher
