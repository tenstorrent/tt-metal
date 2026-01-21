// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/tensor/to_string.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/graph_tracking.hpp>

namespace ttnn {

std::string to_string(const tt::tt_metal::Tensor& tensor) {
    tt::tt_metal::GraphTracker::instance().track_function_start("ttnn::to_string", tensor);

    const auto& shape = tensor.logical_shape();

    if (!tensor.is_allocated()) {
        return fmt::format(
            "{}(<buffer is not allocated>, shape={}, dtype={}, layout={})",
            "ttnn.Tensor",
            shape,
            tensor.dtype(),
            tensor.layout());
    }

    if (std::holds_alternative<tt::tt_metal::DeviceStorage>(tensor.storage())) {
        auto storage = std::get<tt::tt_metal::DeviceStorage>(tensor.storage());
        if (storage.mesh_buffer != nullptr) {
            auto* mesh_device = storage.mesh_buffer->device();

            if (mesh_device->num_devices() == 1) {
                auto cpu_tensor = tensor.cpu();
                return tt::tt_metal::to_string(ttnn::distributed::get_device_tensors(cpu_tensor).at(0));
            }
        }
    }
    auto result = tt::tt_metal::to_string(tensor);
    tt::tt_metal::GraphTracker::instance().track_function_end();
    return result;
}

}  // namespace ttnn
