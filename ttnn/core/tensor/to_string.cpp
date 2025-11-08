#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/graph_tracking.hpp>
#include <tt-metalium/tensor/tensor_impl_wrapper.hpp>
#include "ttnn/distributed/api.hpp"

namespace ttnn {

std::string to_string(const tt::tt_metal::Tensor& tensor) {
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
                return tt::tt_metal::tensor_impl::to_string_wrapper(
                    ttnn::distributed::get_device_tensors(cpu_tensor).at(0));
            }
        }
    }
    return tt::tt_metal::tensor_impl::to_string_wrapper(tensor);
}

}  // namespace ttnn
