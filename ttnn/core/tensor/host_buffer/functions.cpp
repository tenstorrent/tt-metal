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
            [](const HostStorage& storage) { return storage.buffer; },
            [](const MultiDeviceHostStorage& storage) {
                TT_FATAL(
                    storage.distributed_buffer().shape().mesh_size() == 1,
                    "Can't get a single buffer from multi device host storage");
                return *storage.get_shard_at_origin();
            },
            [](const auto&) -> HostBuffer { TT_THROW("Tensor must have HostStorage or MultiDeviceHostStorage"); },
        },
        tensor.storage());
}
}  // namespace tt::tt_metal::host_buffer
