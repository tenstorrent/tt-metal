// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include <ttnn/tensor/host_buffer/functions.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_utils.hpp>

namespace tt::tt_metal::host_buffer {

HostBuffer get_host_buffer(const Tensor& tensor) {
    TT_FATAL(is_cpu_tensor(tensor), "Tensor must have HostStorage");
    const auto& storage = tensor.host_storage();
    std::vector<HostBuffer> buffers;
    storage.buffer().apply([&buffers](const HostBuffer& shard) { buffers.push_back(shard); });
    TT_FATAL(
        buffers.size() == 1,
        "Can't get a single buffer from host storage distributed over mesh shape {}",
        storage.buffer().shape());
    return buffers.front();
}
}  // namespace tt::tt_metal::host_buffer
