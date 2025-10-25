#include "ttnn/tensor/tensor_impl.hpp"
#include "tt-metalium/shape.hpp"
#include "tt-metalium/tensor/types.hpp"

#include "tt-metalium/tensor/layout/tensor_layout.hpp"
#include <tt_stl/span.hpp>
#include <sstream>

template <typename T>
void to_string(
    std::stringstream& ss,
    tt::stl::Span<const T> buffer,
    const tt::tt_metal::Shape& shape,
    const tt::tt_metal::Strides& strides,
    DataType dtype,
    Layout layout) {
    ss << TENSOR_TYPE_STRING << "(";

    if (TTNN_PRINT_OPTIONS.profile == TensorPrintProfile::Empty) {
        ss << "...";
    } else {
        bool use_scientific = should_use_scientific_notation<T>(buffer);
        to_string_row_major<T>(ss, buffer, shape, strides, 0, 0, shape.rank(), 0, use_scientific);
    }
    ss << ", shape=" << fmt::format("{}", shape) << ", dtype=" << fmt::format("{}", dtype)
       << ", layout=" << fmt::format("{}", layout) << ")";
}

}  // namespace detail

template <typename T>
std::string to_string(const Tensor& tensor) {
    const auto& shape = tensor.logical_shape();

    if (!tensor.is_allocated()) {
        return fmt::format(
            "{}(<buffer is not allocated>, shape={}, dtype={}, layout={})",
            detail::TENSOR_TYPE_STRING,
            shape,
            tensor.dtype(),
            tensor.layout());
    }

    auto get_row_major_tensor = [&](const Tensor& tensor) -> Tensor {
        if (tensor.layout() == Layout::ROW_MAJOR) {
            return tensor;
        } else if (tensor.dtype() == DataType::BFLOAT8_B || tensor.dtype() == DataType::BFLOAT4_B) {
            return ttnn::to_layout(ttnn::to_dtype(tensor, DataType::FLOAT32), Layout::ROW_MAJOR);
        } else {
            return ttnn::to_layout(tensor, Layout::ROW_MAJOR);
        }
    };

    auto get_device_buffers = [&](const HostStorage& storage) {
        std::vector<HostBuffer> buffers;
        storage.buffer().apply([&](const HostBuffer& shard) { buffers.push_back(shard); });
        return buffers;
    };

    return std::visit(
        tt::stl::overloaded{
            [&](const HostStorage& storage) -> std::string {
                const Tensor row_major_tensor = get_row_major_tensor(tensor);
                const auto strides = row_major_tensor.tensor_spec().compute_strides();
                const std::vector<HostBuffer> buffers = get_device_buffers(row_major_tensor.host_storage());
                std::stringstream ss;
                for (size_t i = 0; i < buffers.size(); i++) {
                    detail::to_string(ss, buffers[i].view_as<T>(), shape, strides, tensor.dtype(), tensor.layout());
                    if (i + 1 != buffers.size()) {
                        ss << std::endl;
                    }
                }
                return ss.str();
            },
            [&](const DeviceStorage& storage) -> std::string {
                auto cpu_tensor = tensor.cpu();
                if (storage.mesh_buffer == nullptr) {
                    // Use owned buffer path above.
                    return to_string<T>(cpu_tensor);
                }

                auto* mesh_device = storage.mesh_buffer->device();
                if (mesh_device->num_devices() == 1) {
                    return to_string<T>(ttnn::distributed::get_device_tensors(cpu_tensor).at(0));
                }

                const Tensor row_major_tensor = get_row_major_tensor(cpu_tensor);
                const auto strides = row_major_tensor.tensor_spec().compute_strides();
                const auto& coords = storage.coords;
                auto coords_it = coords.begin();
                const std::vector<HostBuffer> buffers = get_device_buffers(row_major_tensor.host_storage());
                std::stringstream ss;
                for (size_t i = 0; i < buffers.size(); i++) {
                    const distributed::MeshCoordinate coord = *coords_it++;
                    if (mesh_device->is_local(coord)) {
                        ss << "device_id: " << mesh_device->get_device(coord)->id() << ", " << coord << std::endl;
                        detail::to_string(ss, buffers[i].view_as<T>(), shape, strides, tensor.dtype(), tensor.layout());
                    }
                    if (i + 1 != buffers.size()) {
                        ss << std::endl;
                    }
                }
                return ss.str();
            }},
        tensor.storage());
}

template std::string to_string<bfloat16>(const Tensor& tensor);
template std::string to_string<float>(const Tensor& tensor);
template std::string to_string<int32_t>(const Tensor& tensor);
template std::string to_string<uint32_t>(const Tensor& tensor);
template std::string to_string<uint16_t>(const Tensor& tensor);
template std::string to_string<uint8_t>(const Tensor& tensor);

template <>
std::string to_string<bfloat8_b>(const Tensor& tensor) {
    return to_string<float>(tensor);
}

template <>
std::string to_string<bfloat4_b>(const Tensor& tensor) {
    return to_string<float>(tensor);
}
