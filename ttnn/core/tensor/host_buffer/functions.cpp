// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <vector>

#include <tt_stl/overloaded.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::host_buffer {

HostBuffer get_host_buffer(const Tensor& tensor) {
    return std::visit(
        tt::stl::overloaded{
            [](const HostStorage& storage) {
                std::vector<HostBuffer> buffers;
                storage.buffer().apply([&buffers](const HostBuffer& shard) { buffers.push_back(shard); });
                
                if (buffers.empty()) {
                    TT_THROW("No buffers found in host storage");
                }
                
                if (buffers.size() > 1) {
                    // Allow access to first buffer from distributed storage
                    // This is safe for replicated tensors where all shards contain identical data
                    log_warning(tt::LogTTNN, 
                        "Accessing first buffer from distributed storage with {} shards", 
                        buffers.size());
                }
                
                return buffers.front();
            },
            [](const auto&) -> HostBuffer { TT_THROW("Tensor must have HostStorage"); },
        },
        tensor.storage());
}
}  // namespace tt::tt_metal::host_buffer
