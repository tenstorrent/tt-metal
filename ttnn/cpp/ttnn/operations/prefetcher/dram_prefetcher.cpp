// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher.hpp"
#include <optional>

#include "device/dram_prefetcher_op.hpp"

namespace ttnn::operations::dram_prefetcher {

ttnn::Tensor ExecuteDramPrefetcher::invoke(const std::vector<ttnn::Tensor>& tensors) {
    auto arch = tensors[0].storage_type() == StorageType::DEVICE
                    ? tensors[0].device()->arch()
                    : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    operation::run(DramPrefetcher{}, {tensors}, {});
    return tensors[0];
}

}  // namespace ttnn::operations::dram_prefetcher
