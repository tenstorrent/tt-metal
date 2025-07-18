// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor_impl.hpp"
#include <fmt/format.h>
#include <optional>

#include <sys/mman.h>
#include <unistd.h>

#include "tt-metalium/assert.hpp"
#include "tt-metalium/distributed_host_buffer.hpp"
#include "tt-metalium/host_buffer.hpp"
#include "tt-metalium/memory_pin.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_coord.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_command_queue.hpp"
#include <tt_stl/overloaded.hpp>
#include <tt_stl/span.hpp>
#include "tt-metalium/shape.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_impl_wrapper.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/distributed/api.hpp"

#include <tracy/Tracy.hpp>

using namespace tt::tt_metal;

namespace {

// Threshold for switch for mmap-based allocations to regular allocations.
// Determined empirically using a microbenchmark; see https://github.com/tenstorrent/tt-metal/pull/22959 for details.
constexpr size_t kMmapThresholdBytes = 1 << 20;

// Allocates memory on the host in batch; using either mmap for large allocations or std::vector for small allocations.
using SharedMemoryPtr = std::shared_ptr<void>;
SharedMemoryPtr allocate_host_data(size_t size_bytes) {
    if (size_bytes >= kMmapThresholdBytes) {
        ZoneScopedN("AllocateBufferMmap");
        void* ptr = mmap(nullptr, size_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        TT_FATAL(ptr != MAP_FAILED, "Failed to allocate {} bytes of memory", size_bytes);
        return SharedMemoryPtr(ptr, [size_bytes](void* p) { madvise(p, size_bytes, MADV_FREE); });
    } else {
        auto vec = std::make_shared<std::vector<std::byte>>(size_bytes);
        return SharedMemoryPtr(vec, vec->data());
    }
}

}  // unnamed namespace

namespace tt {

namespace tt_metal {

namespace tensor_impl {

TensorPrintProfile TTNN_TENSOR_PRINT_PROFILE = TensorPrintProfile::Short;

std::ostream& operator<<(std::ostream& os, const DataType& dtype) {
    switch (dtype) {
        case DataType::BFLOAT8_B: os << "bfloat8_b"; break;
        case DataType::BFLOAT4_B: os << "bfloat4_b"; break;
        case DataType::BFLOAT16: os << "bfloat16"; break;
        case DataType::FLOAT32: os << "float32"; break;
        case DataType::UINT8: os << "uint8"; break;
        case DataType::UINT16: os << "uint16"; break;
        case DataType::UINT32: os << "uint32"; break;
        case DataType::INT32: os << "int32"; break;
        default: throw std::invalid_argument("Unknown data type");
    }
    return os;
}

uint32_t element_size_bytes(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16: return sizeof(bfloat16);
        case DataType::FLOAT32: return sizeof(float);
        case DataType::INT32: return sizeof(int32_t);
        case DataType::UINT32: return sizeof(uint32_t);
        case DataType::UINT16: return sizeof(uint16_t);
        case DataType::UINT8: return sizeof(uint8_t);
        case DataType::BFLOAT8_B: return sizeof(std::byte);
        case DataType::BFLOAT4_B: return sizeof(std::byte);
        default: TT_THROW("Unsupported data type");
    }
}

std::shared_ptr<Buffer> allocate_buffer_on_device(IDevice* device, const TensorSpec& tensor_spec) {
    auto buffer_size_bytes = tensor_spec.compute_packed_buffer_size_bytes();
    auto page_size_bytes = tensor_spec.compute_page_size_bytes();
    auto memory_config = tensor_spec.tensor_layout().get_memory_config();

    return Buffer::create(
        device,
        buffer_size_bytes,
        page_size_bytes,
        memory_config.buffer_type(),
        tensor_spec.compute_buffer_sharding_args());
}

std::shared_ptr<distributed::MeshBuffer> allocate_mesh_buffer_on_device(
    distributed::MeshDevice* mesh_device, const TensorSpec& tensor_spec) {
    const auto& memory_config = tensor_spec.tensor_layout().get_memory_config();

    distributed::DeviceLocalBufferConfig device_local_buffer_config{
        .page_size = tensor_spec.compute_page_size_bytes(),
        .buffer_type = memory_config.buffer_type(),
        .sharding_args = tensor_spec.compute_buffer_sharding_args(),
    };

    // Use replicated buffer, which supports both working with individual shards and replicating data across all shards.
    // This is required for the time being, as TTNN has rich multi-device sharding implementation.
    const distributed::ReplicatedBufferConfig replicated_buffer_config{
        .size = tensor_spec.compute_packed_buffer_size_bytes(),
    };

    return distributed::MeshBuffer::create(replicated_buffer_config, device_local_buffer_config, mesh_device);
}

Tensor pad_bfloat8_b(
    const Tensor& tensor,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    float pad_value) {
    auto tile = tensor.tensor_spec().tile();
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and pad
    auto input_packed_data = host_buffer::get_as<uint32_t>(tensor);
    auto input_float_data =
        unpack_bfp8_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);

    auto input_float_buffer = HostBuffer(std::move(input_float_data));
    auto float_tensor = Tensor(
                            std::move(input_float_buffer),
                            TensorSpec(
                                tensor.logical_shape(),
                                TensorLayout::fromPaddedShape(
                                    DataType::FLOAT32,
                                    PageConfig(tensor.layout(), tile),
                                    MemoryConfig{},
                                    tensor.logical_shape(),
                                    tensor.padded_shape())))
                            .pad(output_padded_shape, input_tensor_start, pad_value);

    // Convert back to BFLOAT8_B
    auto output_float_data = host_buffer::get_as<float>(float_tensor);
    auto output_packed_data =
        pack_fp32_vec_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    auto output_uint32_buffer = HostBuffer(std::move(output_packed_data));
    TensorSpec output_spec(
        float_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            DataType::BFLOAT8_B,
            tensor.tensor_spec().page_config(),
            MemoryConfig{},
            float_tensor.logical_shape(),
            float_tensor.padded_shape()));
    return Tensor(std::move(output_uint32_buffer), output_spec);
}

Tensor unpad_bfloat8_b(
    const Tensor& tensor, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) {
    auto tile = tensor.tensor_spec().tile();
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and unpad
    auto input_packed_data = host_buffer::get_as<uint32_t>(tensor);
    auto input_float_data =
        unpack_bfp8_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
    auto input_float_buffer = HostBuffer(std::move(input_float_data));
    auto float_tensor = Tensor(
                            std::move(input_float_buffer),
                            TensorSpec(
                                tensor.logical_shape(),
                                TensorLayout::fromPaddedShape(
                                    DataType::FLOAT32,
                                    PageConfig(tensor.layout(), tile),
                                    MemoryConfig{},
                                    tensor.logical_shape(),
                                    tensor.padded_shape())))
                            .unpad(output_tensor_start, output_tensor_end);

    // Convert back to BFLOAT8_B
    auto output_float_data = host_buffer::get_as<float>(float_tensor);
    auto output_packed_data =
        pack_fp32_vec_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    auto output_uint32_buffer = HostBuffer(std::move(output_packed_data));
    return Tensor(
        std::move(output_uint32_buffer),
        TensorSpec(
            float_tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                DataType::BFLOAT8_B,
                PageConfig(tensor.layout(), tile),
                MemoryConfig{},
                float_tensor.logical_shape(),
                float_tensor.padded_shape())));
}

Tensor pad_bfloat4_b(
    const Tensor& tensor,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    float pad_value) {
    auto tile = tensor.tensor_spec().tile();
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and pad
    auto input_packed_data = host_buffer::get_as<uint32_t>(tensor);
    auto input_float_data =
        unpack_bfp4_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
    auto input_float_buffer = HostBuffer(std::move(input_float_data));
    auto float_tensor = Tensor(
                            std::move(input_float_buffer),
                            TensorSpec(
                                tensor.logical_shape(),
                                TensorLayout::fromPaddedShape(
                                    DataType::FLOAT32,
                                    PageConfig(tensor.layout(), tile),
                                    MemoryConfig{},
                                    tensor.logical_shape(),
                                    tensor.logical_shape())))
                            .pad(output_padded_shape, input_tensor_start, pad_value);

    // Convert back to BFLOAT4_B
    auto output_float_data = host_buffer::get_as<float>(float_tensor);
    auto output_packed_data =
        pack_fp32_vec_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    auto output_uint32_buffer = HostBuffer(std::move(output_packed_data));
    TensorSpec output_spec(
        float_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            DataType::BFLOAT4_B,
            tensor.tensor_spec().page_config(),
            MemoryConfig{},
            float_tensor.logical_shape(),
            float_tensor.padded_shape()));
    return Tensor(std::move(output_uint32_buffer), output_spec);
}

Tensor unpad_bfloat4_b(
    const Tensor& tensor, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) {
    auto tile = tensor.tensor_spec().tile();
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and unpad
    auto input_packed_data = host_buffer::get_as<uint32_t>(tensor);
    auto input_float_data =
        unpack_bfp4_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
    auto input_float_buffer = HostBuffer(std::move(input_float_data));
    auto float_tensor = Tensor(
                            std::move(input_float_buffer),
                            TensorSpec(
                                tensor.logical_shape(),
                                TensorLayout::fromPaddedShape(
                                    DataType::FLOAT32,
                                    PageConfig(tensor.layout(), tile),
                                    MemoryConfig{},
                                    tensor.logical_shape(),
                                    tensor.padded_shape())))
                            .unpad(output_tensor_start, output_tensor_end);

    // Convert back to BFLOAT4_B
    auto output_float_data = host_buffer::get_as<float>(float_tensor);
    auto output_packed_data =
        pack_fp32_vec_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    auto output_uint32_buffer = HostBuffer(std::move(output_packed_data));
    return Tensor(
        std::move(output_uint32_buffer),
        TensorSpec(
            float_tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                DataType::BFLOAT4_B,
                PageConfig(tensor.layout(), tile),
                MemoryConfig{},
                float_tensor.logical_shape(),
                float_tensor.padded_shape())));
}

// ======================================================================================
//                                      .to_string()
// ======================================================================================

namespace detail {

struct DimensionShortener {
    size_t size;
    std::optional<std::size_t> max;

    bool print_parenthesis_and_advance_index_if_reached_half_of_max_and_check_if_loop_is_done(
        std::ostream& ss, std::size_t& index, const std::string& before, const std::string& after) const {
        if (this->max.has_value() and this->size > this->max.value() and index == this->max.value() / 2) {
            ss << before << "...," << after;
            index = this->size - (this->max.value() / 2);
        }
        return index < this->size;
    }
};

inline DimensionShortener get_dimension_shortener(std::size_t size) {
    switch (TTNN_TENSOR_PRINT_PROFILE) {
        case TensorPrintProfile::Empty: return DimensionShortener{size, 0};
        case TensorPrintProfile::Short: return DimensionShortener{size, 4};
        case TensorPrintProfile::Full: return DimensionShortener{size, std::nullopt};
        default: TT_THROW("Unrecognized TTNN_TENSOR_PRINT_PROFILE {}", TTNN_TENSOR_PRINT_PROFILE);
    }
}

inline void print_trailing_comma(std::ostream& ss, std::size_t index, std::size_t size, const std::string& after) {
    if (index < size - 1) {
        ss << "," << after;
    }
}

template <typename T>
inline void print_datum(std::ostream& ss, T datum) {
    if (std::is_integral<T>::value) {
        ss << std::setw(5) << datum;
    } else {
        ss << std::fixed << std::setw(8) << std::setprecision(5) << datum;
    }
}

template <>
inline void print_datum(std::ostream& ss, bfloat16 datum) {
    print_datum(ss, datum.to_float());
}

template <>
inline void print_datum(std::ostream& ss, uint8_t datum) {
    print_datum<uint32_t>(ss, datum);
}

inline constexpr int constexpr_strlen(const char* str) { return *str ? 1 + constexpr_strlen(str + 1) : 0; }

constexpr auto TENSOR_TYPE_STRING = "ttnn.Tensor";
constexpr auto TENSOR_TYPE_STRING_PLUS_OPEN_PARENTHESIS_LENGTH = constexpr_strlen(TENSOR_TYPE_STRING) + 1;

template <typename T>
void to_string_row_major(
    std::stringstream& ss,
    tt::stl::Span<const T> buffer,
    const ttnn::Shape& shape,
    const tt::tt_metal::Strides& strides,
    std::size_t outer_index,
    const std::size_t buffer_offset,
    int64_t rank,
    int64_t dim = 0) {
    auto stride = dim < strides.size() ? strides[dim] : 0;

    std::string spaces = std::string(TENSOR_TYPE_STRING_PLUS_OPEN_PARENTHESIS_LENGTH + dim, ' ');
    std::string before;
    std::string after;
    if (rank == 1) {
        before = " ";
        after = " ";
    } else if (rank == 2) {
        before = spaces + " ";
        after = "\n";
    } else {
        before = spaces + " ";
        after = "\n\n";
    }

    if (dim > 0 and outer_index > 0) {
        ss << spaces;
    }
    if (rank != 0) {
        ss << "[";
    }
    auto dimension_shortener = get_dimension_shortener(rank != 0 ? shape[-rank] : 1);
    for (std::size_t index = 0;
         dimension_shortener.print_parenthesis_and_advance_index_if_reached_half_of_max_and_check_if_loop_is_done(
             ss, index, before, after);
         index++) {
        std::string after_comma;
        if (rank == 1) {
            after_comma = " ";
        } else if (rank == 2) {
            after_comma = "\n";
        } else {
            after_comma = after;
        }

        if (rank > 1) {
            to_string_row_major(ss, buffer, shape, strides, index, buffer_offset + index * stride, rank - 1, dim + 1);
        } else {
            print_datum(ss, buffer[buffer_offset + index]);
        }
        print_trailing_comma(ss, index, rank != 0 ? shape[-rank] : 1, after_comma);
    }
    if (rank != 0) {
        ss << "]";
    }
}

template <typename T>
void to_string(
    std::stringstream& ss,
    tt::stl::Span<const T> buffer,
    const ttnn::Shape& shape,
    const tt::tt_metal::Strides& strides,
    DataType dtype,
    Layout layout) {
    ss << TENSOR_TYPE_STRING << "(";

    if (TTNN_TENSOR_PRINT_PROFILE == TensorPrintProfile::Empty) {
        ss << "...";
    } else {
        to_string_row_major<T>(ss, buffer, shape, strides, 0, 0, shape.rank());
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
                    ss << "device_id: " << mesh_device->get_device(coord)->id() << ", " << coord << std::endl;
                    detail::to_string(ss, buffers[i].view_as<T>(), shape, strides, tensor.dtype(), tensor.layout());
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

// ======================================================================================
//                                      .to_host()
// ======================================================================================

HostBuffer allocate_host_buffer(const TensorSpec& tensor_spec) {
    ZoneScopedN("AllocateBuffer");
    const size_t size_bytes = tensor_spec.compute_packed_buffer_size_bytes();
    switch (tensor_spec.data_type()) {
        case DataType::BFLOAT16: return HostBuffer(std::vector<bfloat16>(size_bytes / sizeof(bfloat16)));
        case DataType::FLOAT32: return HostBuffer(std::vector<float>(size_bytes / sizeof(float)));
        case DataType::INT32: return HostBuffer(std::vector<int32_t>(size_bytes / sizeof(int32_t)));
        case DataType::UINT8: return HostBuffer(std::vector<uint8_t>(size_bytes / sizeof(uint8_t)));
        case DataType::UINT16: return HostBuffer(std::vector<uint16_t>(size_bytes / sizeof(uint16_t)));
        case DataType::BFLOAT4_B:
        case DataType::BFLOAT8_B:
        case DataType::UINT32: return HostBuffer(std::vector<uint32_t>(size_bytes / sizeof(uint32_t)));
        case DataType::INVALID: TT_THROW("Invalid data type");
    }
    TT_THROW("Unreachable");
}

template <typename T>
Tensor to_host_helper(const Tensor& tensor, bool blocking = true, ttnn::QueueId cq_id = ttnn::DefaultQueueId) {
    TT_FATAL(tensor.is_allocated(), "Buffer must be allocated on device!");
    auto device_buffer = tensor.buffer();
    auto device = tensor.device();
    TT_FATAL(device != nullptr, "Need device to be set copy data from device to host!");
    uint32_t size_in_bytes = device_buffer->size();
    std::vector<T> data_vec;
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        data_vec.resize(size_in_bytes / sizeof(T));
        read_data_from_device_buffer<T>(device->command_queue(*cq_id), *device_buffer, data_vec.data(), blocking);
    } else {
        read_data_from_device_buffer<T>(*device_buffer, data_vec);
    }
    auto output_buffer = HostBuffer(std::move(data_vec));
    return Tensor(std::move(output_buffer), tensor.tensor_spec());
}

template <typename T>
Tensor to_host(const Tensor& tensor, bool blocking, ttnn::QueueId cq_id) {
    if (tensor.storage_type() == StorageType::DEVICE) {
        return to_host_helper<T>(tensor, blocking, cq_id);
    } else {
        return tensor;
    }
}

template Tensor to_host<bfloat16>(const Tensor& tensor, bool blocking, ttnn::QueueId cq_id);
template Tensor to_host<float>(const Tensor& tensor, bool blocking, ttnn::QueueId cq_id);
template Tensor to_host<int32_t>(const Tensor& tensor, bool blocking, ttnn::QueueId cq_id);
template Tensor to_host<uint32_t>(const Tensor& tensor, bool blocking, ttnn::QueueId cq_id);
template Tensor to_host<uint16_t>(const Tensor& tensor, bool blocking, ttnn::QueueId cq_id);
template Tensor to_host<uint8_t>(const Tensor& tensor, bool blocking, ttnn::QueueId cq_id);

template <>
Tensor to_host<bfloat4_b>(const Tensor& tensor, bool blocking, ttnn::QueueId cq_id) {
    return to_host<uint32_t>(tensor, blocking, cq_id);
}

template <>
Tensor to_host<bfloat8_b>(const Tensor& tensor, bool blocking, ttnn::QueueId cq_id) {
    return to_host<uint32_t>(tensor, blocking, cq_id);
}

template <typename T>
Tensor to_host_mesh_tensor(const Tensor& tensor, bool blocking, ttnn::QueueId cq_id) {
    TT_FATAL(tensor.is_allocated(), "Buffer must be allocated on device!");
    const auto& storage = tensor.device_storage();
    const auto& mesh_buffer = storage.mesh_buffer;
    ttnn::MeshDevice* device = mesh_buffer->device();
    distributed::MeshCommandQueue& mesh_cq = device->mesh_command_queue(*cq_id);

    // For performance, perform all allocations via DistributedHostBuffer::transform, run from multiple threads.
    auto distributed_host_buffer = DistributedHostBuffer::create(device->shape());
    for (const auto& coord : storage.coords) {
        distributed_host_buffer.emplace_shard(coord, []() { return HostBuffer(); });
    }

    distributed_host_buffer = distributed_host_buffer.transform(
        [&](const HostBuffer&) { return allocate_host_buffer(tensor.tensor_spec()); },
        DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);

    mesh_cq.enqueue_read(mesh_buffer, distributed_host_buffer, /*shards=*/std::nullopt, blocking);

    HostStorage host_storage(std::move(distributed_host_buffer));
    return Tensor(std::move(host_storage), tensor.tensor_spec(), tensor.distributed_tensor_config());
}

template Tensor to_host_mesh_tensor<bfloat16>(const Tensor& tensor, bool blocking, ttnn::QueueId cq_id);
template Tensor to_host_mesh_tensor<float>(const Tensor& tensor, bool blocking, ttnn::QueueId cq_id);
template Tensor to_host_mesh_tensor<int32_t>(const Tensor& tensor, bool blocking, ttnn::QueueId cq_id);
template Tensor to_host_mesh_tensor<uint32_t>(const Tensor& tensor, bool blocking, ttnn::QueueId cq_id);
template Tensor to_host_mesh_tensor<uint16_t>(const Tensor& tensor, bool blocking, ttnn::QueueId cq_id);
template Tensor to_host_mesh_tensor<uint8_t>(const Tensor& tensor, bool blocking, ttnn::QueueId cq_id);

template <>
Tensor to_host_mesh_tensor<bfloat4_b>(const Tensor& tensor, bool blocking, ttnn::QueueId cq_id) {
    return to_host_mesh_tensor<uint32_t>(tensor, blocking, cq_id);
}

template <>
Tensor to_host_mesh_tensor<bfloat8_b>(const Tensor& tensor, bool blocking, ttnn::QueueId cq_id) {
    return to_host_mesh_tensor<uint32_t>(tensor, blocking, cq_id);
}

// ======================================================================================
//                               .to_device() details
// ======================================================================================

template <typename T>
void write_data_to_device_buffer(
    CommandQueue& cq, tt::stl::Span<const T> host_buffer, std::shared_ptr<Buffer> device_buffer) {
    ZoneScoped;
    // TODO(arakhmati): can we use generators in this function to go from `data_to_write` to `uint32_data`?
    // And effectively get rid of any additional allocation
    EnqueueWriteBuffer(cq, device_buffer, host_buffer.data(), false);
}

template <typename T>
void write_data_to_device_buffer(tt::stl::Span<const T> host_buffer, Buffer& device_buffer) {
    ZoneScoped;
    ::detail::WriteToBuffer(
        device_buffer,
        tt::stl::Span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(host_buffer.data()), host_buffer.size() * sizeof(T)));
}

template <typename T>
std::shared_ptr<Buffer> initialize_data_on_device(
    tt::stl::Span<const T> data_to_write,
    IDevice* device,
    const TensorSpec& tensor_spec,
    ttnn::QueueId cq_id = ttnn::DefaultQueueId) {
    ZoneScoped;
    TT_ASSERT(device != nullptr);

    auto device_buffer = allocate_buffer_on_device(device, tensor_spec);

    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        write_data_to_device_buffer<T>(device->command_queue(*cq_id), data_to_write, device_buffer);
    } else {
        write_data_to_device_buffer<T>(data_to_write, *device_buffer);
    }
    return device_buffer;
}

template <typename T>
std::shared_ptr<Buffer> to_device_buffer(
    const Storage& storage, IDevice* device, const TensorSpec& tensor_spec, ttnn::QueueId cq_id) {
    return std::visit(
        tt::stl::overloaded{
            [&device, &tensor_spec, cq_id](const HostStorage& storage) {
                TT_FATAL(
                    storage.buffer().shape() == distributed::MeshShape(1, 1),
                    "Can't get a single buffer from host storage distributed over mesh shape {}",
                    storage.buffer().shape());
                auto data_to_write = storage.buffer().get_shard(distributed::MeshCoordinate(0, 0))->view_as<T>();
                auto expected_packed_buffer_size_bytes = tensor_spec.compute_packed_buffer_size_bytes();
                auto input_size_bytes = data_to_write.size() * sizeof(T);
                TT_FATAL(
                    input_size_bytes == expected_packed_buffer_size_bytes,
                    "Host data with total size {}B does not match expected size {}B of device buffer!",
                    input_size_bytes,
                    expected_packed_buffer_size_bytes);
                return initialize_data_on_device<T>(data_to_write, device, tensor_spec, cq_id);
            },
            [](const auto& s) {
                TT_THROW("Unexpected storage type {}", tt::stl::get_type_name(s));
                return std::shared_ptr<Buffer>();
            }},
        storage);
}

// ======================================================================================
//                                  .to_device()
// ======================================================================================

template <typename T>
Tensor to_device(const Tensor& tensor, IDevice* target_device, const MemoryConfig& memory_config, ttnn::QueueId cq_id) {
    if (auto mesh_device = dynamic_cast<distributed::MeshDevice*>(target_device)) {
        return to_device_mesh_tensor<T>(tensor, mesh_device, memory_config, cq_id);
    }
    TT_FATAL(tensor.storage_type() != StorageType::DEVICE, "Tensor is already on device!");
    TT_FATAL(target_device != nullptr, "Need target device in order to move tensor to device!");

    TensorSpec tensor_spec(
        tensor.logical_shape(), tensor.tensor_spec().tensor_layout().with_memory_config(memory_config));
    auto device_buffer = tensor_impl::to_device_buffer<T>(tensor.storage(), target_device, tensor_spec, cq_id);
    return Tensor(DeviceStorage{device_buffer}, tensor_spec, tensor.distributed_tensor_config());
}

template Tensor to_device<bfloat16>(
    const Tensor& tensor, IDevice* target_device, const MemoryConfig& memory_config, ttnn::QueueId cq_id);
template Tensor to_device<float>(
    const Tensor& tensor, IDevice* target_device, const MemoryConfig& memory_config, ttnn::QueueId cq_id);
template Tensor to_device<int32_t>(
    const Tensor& tensor, IDevice* target_device, const MemoryConfig& memory_config, ttnn::QueueId cq_id);
template Tensor to_device<uint32_t>(
    const Tensor& tensor, IDevice* target_device, const MemoryConfig& memory_config, ttnn::QueueId cq_id);
template Tensor to_device<uint16_t>(
    const Tensor& tensor, IDevice* target_device, const MemoryConfig& memory_config, ttnn::QueueId cq_id);
template Tensor to_device<uint8_t>(
    const Tensor& tensor, IDevice* target_device, const MemoryConfig& memory_config, ttnn::QueueId cq_id);

template <>
Tensor to_device<bfloat4_b>(
    const Tensor& tensor, IDevice* target_device, const MemoryConfig& memory_config, ttnn::QueueId cq_id) {
    return to_device<uint32_t>(tensor, target_device, memory_config, cq_id);
}

template <>
Tensor to_device<bfloat8_b>(
    const Tensor& tensor, IDevice* target_device, const MemoryConfig& memory_config, ttnn::QueueId cq_id) {
    return to_device<uint32_t>(tensor, target_device, memory_config, cq_id);
}

namespace {

DeviceStorage replicate_to_mesh_buffer(
    const HostBuffer& buffer,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    const TensorSpec& tensor_spec,
    ttnn::QueueId cq_id) {
    auto* mesh_device = mesh_buffer->device();
    auto data_to_write = buffer.view_bytes();
    const auto expected_packed_buffer_size_bytes = tensor_spec.compute_packed_buffer_size_bytes();
    const auto input_size_bytes = data_to_write.size();
    TT_FATAL(
        input_size_bytes == expected_packed_buffer_size_bytes,
        "Host data with total size {}B does not match expected size {}B of device buffer!",
        input_size_bytes,
        expected_packed_buffer_size_bytes);

    mesh_device->mesh_command_queue(*cq_id).enqueue_write_mesh_buffer(
        mesh_buffer, data_to_write.data(), /*blocking=*/false);

    std::vector<distributed::MeshCoordinate> coords;
    coords.reserve(mesh_device->shape().mesh_size());
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        coords.push_back(coord);
    }
    return DeviceStorage(mesh_buffer, std::move(coords));
}

DeviceStorage write_to_mesh_buffer(
    const DistributedHostBuffer& distributed_host_buffer,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    ttnn::QueueId cq_id) {
    mesh_buffer->device()->mesh_command_queue(*cq_id).enqueue_write(
        mesh_buffer, distributed_host_buffer, /*blocking=*/false);
    std::vector<distributed::MeshCoordinate> coords;
    coords.reserve(distributed_host_buffer.shard_coords().size());
    std::copy(
        distributed_host_buffer.shard_coords().begin(),
        distributed_host_buffer.shard_coords().end(),
        std::back_inserter(coords));
    return DeviceStorage(mesh_buffer, std::move(coords));
}

}  // namespace

template <typename T>
DeviceStorage to_device_mesh_buffer(
    const Storage& host_storage,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    const TensorSpec& tensor_spec,
    const TensorAttributes& host_tensor_attributes,
    ttnn::QueueId cq_id) {
    return std::visit(
        tt::stl::overloaded{
            [&mesh_buffer, &tensor_spec, cq_id, &host_tensor_attributes](const HostStorage& storage) {
                const auto& host_storage_shape = storage.buffer().shape();
                const auto& mesh_device_shape = mesh_buffer->device()->shape();
                if (host_storage_shape.mesh_size() < mesh_device_shape.mesh_size() &&
                    host_storage_shape == distributed::MeshShape(1, 1)) {
                    // Special case of replicating tensors on 1x1 mesh across the entire mesh device.
                    const auto device_buffer = storage.buffer().get_shard(distributed::MeshCoordinate(0, 0));
                    return replicate_to_mesh_buffer(*device_buffer, mesh_buffer, tensor_spec, cq_id);
                } else {
                    TT_FATAL(
                        host_storage_shape == mesh_device_shape,
                        "Distributed host buffer has different shape {} than the mesh device {}",
                        host_storage_shape,
                        mesh_device_shape);
                    return write_to_mesh_buffer(storage.buffer(), mesh_buffer, cq_id);
                }
            },
            [](const auto& s) -> DeviceStorage { TT_THROW("Unexpected storage type {}", tt::stl::get_type_name(s)); }},
        host_storage);
}

template <typename T>
Tensor to_device_mesh_tensor(
    const Tensor& tensor,
    distributed::MeshDevice* mesh_device,
    const MemoryConfig& memory_config,
    ttnn::QueueId cq_id) {
    if (tensor.storage_type() == StorageType::DEVICE) {
        return tensor;  // Tensor already on device
    }

    TT_FATAL(mesh_device != nullptr, "Need target device in order to move tensor to device!");

    TensorSpec tensor_spec(
        tensor.logical_shape(), tensor.tensor_spec().tensor_layout().with_memory_config(memory_config));

    auto mesh_buffer = allocate_mesh_buffer_on_device(mesh_device, tensor_spec);
    DeviceStorage mesh_storage =
        to_device_mesh_buffer<T>(tensor.storage(), mesh_buffer, tensor_spec, *tensor.tensor_attributes, cq_id);
    return Tensor(std::move(mesh_storage), tensor_spec, tensor.distributed_tensor_config());
}

template <typename T>
void copy_to_host_tensor(const Tensor& device_tensor, Tensor& host_tensor, bool blocking, ttnn::QueueId cq_id) {
    ZoneScoped;
    TT_FATAL(device_tensor.storage_type() == StorageType::DEVICE, "Source tensor is not on device.");
    TT_FATAL(host_tensor.storage_type() == StorageType::HOST, "Destination tensor is not on host.");
    TT_FATAL(device_tensor.is_allocated(), "Buffer must be allocated on device.");

    TT_FATAL(host_tensor.logical_shape() == device_tensor.logical_shape(), "Host tensor has different shape");
    TT_FATAL(host_tensor.dtype() == device_tensor.dtype(), "Host tensor has different dtype");
    TT_FATAL(
        host_tensor.tensor_spec().page_config() == device_tensor.tensor_spec().page_config(),
        "Host tensor has different page config");

    const auto& device_storage = device_tensor.device_storage();
    const auto& mesh_buffer = device_storage.mesh_buffer;
    ttnn::MeshDevice* device = mesh_buffer->device();
    distributed::MeshCommandQueue& mesh_cq = device->mesh_command_queue(*cq_id);

    const auto& distributed_host_buffer = host_tensor.host_storage().buffer();

    // Host tensor must have pre-allocated buffers for all device shards.
    // However, it may have some extra shards. Drop them by "unwrapping" the distributed host buffer, and re-wrapping
    // only for those shards that are actually present on device.
    std::vector<std::pair<distributed::MeshCoordinate, std::optional<HostBuffer>>> shards;
    shards.reserve(device_storage.coords.size());
    for (const auto& device_coord : device_storage.coords) {
        shards.push_back({device_coord, distributed_host_buffer.get_shard(device_coord)});
    }

    DistributedHostBuffer dst_distributed_host_buffer = DistributedHostBuffer::create(device->shape());
    const size_t expected_size_bytes = device_tensor.tensor_spec().compute_packed_buffer_size_bytes();
    for (const auto& [device_coord, host_buffer] : shards) {
        dst_distributed_host_buffer.emplace_shard(device_coord, [&]() {
            // Note the lambda is executed only for host-local shards.
            // If `host_buffer` is nullopt, the data was not correctly allocated on the host.
            TT_FATAL(host_buffer.has_value(), "Host shard for device shard {} is not populated.", device_coord);

            TT_FATAL(
                host_buffer->view_bytes().size() == expected_size_bytes,
                "Host shard for device shard {} has invalid size: {} != {}",
                device_coord,
                host_buffer->view_bytes().size(),
                expected_size_bytes);
            return *host_buffer;
        });
    }

    mesh_cq.enqueue_read(mesh_buffer, dst_distributed_host_buffer, /*shards=*/std::nullopt, blocking);

    host_tensor = Tensor(
        HostStorage(std::move(dst_distributed_host_buffer)),
        device_tensor.tensor_spec(),
        device_tensor.distributed_tensor_config());
}

template <typename T>
void copy_to_device_tensor(const Tensor& host_tensor, Tensor& device_tensor, ttnn::QueueId cq_id) {
    ZoneScoped;
    TT_FATAL(host_tensor.storage_type() == StorageType::HOST, "Source tensor is not on host.");
    TT_FATAL(device_tensor.storage_type() == StorageType::DEVICE, "Destination tensor is not on device.");
    TT_FATAL(device_tensor.is_allocated(), "Buffer must be allocated on device.");

    TT_FATAL(host_tensor.logical_shape() == device_tensor.logical_shape(), "Host tensor has different shape");
    TT_FATAL(host_tensor.dtype() == device_tensor.dtype(), "Host tensor has different dtype");
    TT_FATAL(
        host_tensor.tensor_spec().page_config() == device_tensor.tensor_spec().page_config(),
        "Host tensor has different page config");

    auto mesh_buffer = device_tensor.device_storage().mesh_buffer;

    DeviceStorage mesh_storage = to_device_mesh_buffer<T>(
        host_tensor.storage(), mesh_buffer, device_tensor.tensor_spec(), *host_tensor.tensor_attributes, cq_id);
    device_tensor = Tensor(
        std::move(mesh_storage),
        host_tensor.tensor_spec().with_memory_config(device_tensor.memory_config()),
        host_tensor.distributed_tensor_config());
}

template Tensor to_device_mesh_tensor<bfloat16>(
    const Tensor& tensor,
    distributed::MeshDevice* target_device,
    const MemoryConfig& memory_config,
    ttnn::QueueId cq_id);
template Tensor to_device_mesh_tensor<float>(
    const Tensor& tensor,
    distributed::MeshDevice* target_device,
    const MemoryConfig& memory_config,
    ttnn::QueueId cq_id);
template Tensor to_device_mesh_tensor<int32_t>(
    const Tensor& tensor,
    distributed::MeshDevice* target_device,
    const MemoryConfig& memory_config,
    ttnn::QueueId cq_id);
template Tensor to_device_mesh_tensor<uint32_t>(
    const Tensor& tensor,
    distributed::MeshDevice* target_device,
    const MemoryConfig& memory_config,
    ttnn::QueueId cq_id);
template Tensor to_device_mesh_tensor<uint16_t>(
    const Tensor& tensor,
    distributed::MeshDevice* target_device,
    const MemoryConfig& memory_config,
    ttnn::QueueId cq_id);
template Tensor to_device_mesh_tensor<uint8_t>(
    const Tensor& tensor,
    distributed::MeshDevice* target_device,
    const MemoryConfig& memory_config,
    ttnn::QueueId cq_id);

template <>
Tensor to_device_mesh_tensor<bfloat4_b>(
    const Tensor& tensor,
    distributed::MeshDevice* target_device,
    const MemoryConfig& memory_config,
    ttnn::QueueId cq_id) {
    return to_device_mesh_tensor<uint32_t>(tensor, target_device, memory_config, cq_id);
}

template <>
Tensor to_device_mesh_tensor<bfloat8_b>(
    const Tensor& tensor,
    distributed::MeshDevice* target_device,
    const MemoryConfig& memory_config,
    ttnn::QueueId cq_id) {
    return to_device_mesh_tensor<uint32_t>(tensor, target_device, memory_config, cq_id);
}

template void copy_to_device_tensor<bfloat16>(const Tensor&, Tensor&, ttnn::QueueId);
template void copy_to_device_tensor<float>(const Tensor&, Tensor&, ttnn::QueueId);
template void copy_to_device_tensor<int32_t>(const Tensor&, Tensor&, ttnn::QueueId);
template void copy_to_device_tensor<uint32_t>(const Tensor&, Tensor&, ttnn::QueueId);
template void copy_to_device_tensor<uint16_t>(const Tensor&, Tensor&, ttnn::QueueId);
template void copy_to_device_tensor<uint8_t>(const Tensor&, Tensor&, ttnn::QueueId);
template void copy_to_host_tensor<bfloat16>(const Tensor&, Tensor&, bool, ttnn::QueueId);
template void copy_to_host_tensor<float>(const Tensor&, Tensor&, bool, ttnn::QueueId);
template void copy_to_host_tensor<int32_t>(const Tensor&, Tensor&, bool, ttnn::QueueId);
template void copy_to_host_tensor<uint32_t>(const Tensor&, Tensor&, bool, ttnn::QueueId);
template void copy_to_host_tensor<uint16_t>(const Tensor&, Tensor&, bool, ttnn::QueueId);
template void copy_to_host_tensor<uint8_t>(const Tensor&, Tensor&, bool, ttnn::QueueId);

template <>
void copy_to_device_tensor<bfloat4_b>(const Tensor& host_tensor, Tensor& device_tensor, ttnn::QueueId cq_id) {
    copy_to_device_tensor<uint32_t>(host_tensor, device_tensor, cq_id);
}

template <>
void copy_to_device_tensor<bfloat8_b>(const Tensor& host_tensor, Tensor& device_tensor, ttnn::QueueId cq_id) {
    copy_to_device_tensor<uint32_t>(host_tensor, device_tensor, cq_id);
}

template <>
void copy_to_host_tensor<bfloat4_b>(
    const Tensor& device_tensor, Tensor& host_tensor, bool blocking, ttnn::QueueId cq_id) {
    copy_to_host_tensor<uint32_t>(device_tensor, host_tensor, blocking, cq_id);
}

template <>
void copy_to_host_tensor<bfloat8_b>(
    const Tensor& device_tensor, Tensor& host_tensor, bool blocking, ttnn::QueueId cq_id) {
    copy_to_host_tensor<uint32_t>(device_tensor, host_tensor, blocking, cq_id);
}

// ======================================================================================
//     Helpers for converting between logical <-> physical data with full tensor spec
// ======================================================================================
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// TODO: Remove when we get rid of physical sharding and generalize interleaved and sharded; when we do, directly get
// from TensorLayout
std::array<Shape2D, 2> get_logical_and_physical_shard_shapes(const TensorSpec& tensor_spec) {
    const auto& logical_shape = tensor_spec.logical_shape();
    const auto& padded_shape = tensor_spec.padded_shape();

    // TODO: get_logical_shard_shape always returns shard shape from shard spec, which is not correct in physical mode
    // if there is padding
    if (tensor_spec.memory_config().is_sharded() &&
        ((tensor_spec.memory_config().shard_spec().has_value() &&
          tensor_spec.memory_config().shard_spec().value().mode == ShardMode::LOGICAL) ||
         logical_shape == padded_shape)) {
        return {
            tensor_spec.tensor_layout().get_logical_shard_shape(),
            tensor_spec.tensor_layout().get_physical_shard_shape()};
    }

    Shape2D logical_shard_shape{logical_shape[-2], logical_shape[-1]};
    Shape2D physical_shard_shape = {padded_shape[-2], padded_shape[-1]};
    return {logical_shard_shape, physical_shard_shape};
}

using LogicalPhysicalIdxPairs = std::vector<std::pair<size_t, size_t>>;
using LogicalPhysicalMapping = std::pair<LogicalPhysicalIdxPairs, size_t>;
std::vector<LogicalPhysicalMapping> compute_logical_to_physical_shards_mapping(
    const Shape2D& logical_2d_shape,
    const Shape2D& logical_shard_shape,
    const Shape2D& physical_shard_shape,
    const size_t physical_stride) {
    const auto logical_stride = logical_2d_shape.width();

    const auto [num_shards_height, last_shard_height, num_shards_width, last_shard_width] =
        tt::tt_metal::compute_shard_division_spec(logical_2d_shape, logical_shard_shape);

    std::vector<LogicalPhysicalMapping> logical_physical_mapping(num_shards_height * num_shards_width);

    for (size_t shard_height_idx = 0; shard_height_idx < num_shards_height; shard_height_idx++) {
        for (size_t shard_width_idx = 0; shard_width_idx < num_shards_width; shard_width_idx++) {
            const auto num_shard_rows =
                shard_height_idx == num_shards_height - 1 ? last_shard_height : logical_shard_shape.height();
            const auto num_shard_cols =
                shard_width_idx == num_shards_width - 1 ? last_shard_width : logical_shard_shape.width();

            auto indices = LogicalPhysicalIdxPairs(num_shard_rows);
            const auto logical_start_idx = shard_height_idx * logical_shard_shape.height() * logical_stride +
                                           shard_width_idx * logical_shard_shape.width();
            const auto physical_start_idx = shard_height_idx * physical_shard_shape.height() * physical_stride +
                                            shard_width_idx * physical_shard_shape.width();
            for (size_t i = 0; i < num_shard_rows; i++) {
                indices[i] = {i * logical_stride + logical_start_idx, i * physical_stride + physical_start_idx};
            }

            logical_physical_mapping.push_back((LogicalPhysicalMapping){indices, num_shard_cols});
        }
    }
    return logical_physical_mapping;
};

// Converts a span of logical data to row major physical data.
template <typename T>
std::vector<T> convert_to_row_major_physical_data(
    tt::stl::Span<const T> logical_data, const TensorSpec& tensor_spec, T pad_value) {
    const auto& physical_shape = tensor_spec.physical_shape();
    const size_t physical_stride = physical_shape.width();
    auto [logical_shard_shape, physical_shard_shape] =
        CMAKE_UNIQUE_NAMESPACE::get_logical_and_physical_shard_shapes(tensor_spec);

    std::vector<T> row_major_physical_data(physical_shape.height() * physical_shape.width(), pad_value);

    const auto logical_physical_mapping = CMAKE_UNIQUE_NAMESPACE::compute_logical_to_physical_shards_mapping(
        tensor_spec.logical_2d_shape(), logical_shard_shape, physical_shard_shape, physical_stride);

    for (const auto& [indices, cols] : logical_physical_mapping) {
        for (const auto& [logical_idx_start, physical_idx_start] : indices) {
            for (size_t col = 0; col < cols; col++) {
                row_major_physical_data[physical_idx_start + col] = logical_data[logical_idx_start + col];
            }
        }
    }
    return row_major_physical_data;
}

// Converts a span of row major physical data to logical data.
template <typename T>
std::vector<T> convert_to_logical_data(tt::stl::Span<const T> row_major_physical_data, const TensorSpec& tensor_spec) {
    const auto& logical_2d_shape = tensor_spec.logical_2d_shape();
    const size_t physical_stride = tensor_spec.physical_shape().width();
    auto [logical_shard_shape, physical_shard_shape] =
        CMAKE_UNIQUE_NAMESPACE::get_logical_and_physical_shard_shapes(tensor_spec);

    std::vector<T> logical_data(logical_2d_shape.height() * logical_2d_shape.width(), 0);

    const auto logical_physical_mapping = CMAKE_UNIQUE_NAMESPACE::compute_logical_to_physical_shards_mapping(
        logical_2d_shape, logical_shard_shape, physical_shard_shape, physical_stride);

    for (const auto& [indices, cols] : logical_physical_mapping) {
        for (const auto& [logical_idx_start, physical_idx_start] : indices) {
            for (size_t col = 0; col < cols; col++) {
                logical_data[logical_idx_start + col] = row_major_physical_data[physical_idx_start + col];
            }
        }
    }
    return logical_data;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

template <typename T>
std::vector<T> encode_tensor_data(tt::stl::Span<const T> logical_data, const TensorSpec& tensor_spec, T pad_value) {
    if (logical_data.size() == 0) {
        return {};
    }

    const auto& logical_shape = tensor_spec.logical_shape();
    const auto& physical_shape = tensor_spec.physical_shape();

    TT_FATAL(
        logical_data.size() == logical_shape.volume(),
        "Logical data size {} should be same as volume indicated by logical shape {}",
        logical_data.size(),
        logical_shape);

    // If needed, convert logical data to row major physical data.
    // `row_major_physical_data_span` stores span unconditionally (cheap), while `row_major_physical_data` stores the
    // converted vector only when needed (expensive).
    std::vector<T> row_major_physical_data;
    tt::stl::Span<const T> row_major_physical_data_span;
    if (tensor_spec.logical_2d_shape() != physical_shape) {
        row_major_physical_data =
            CMAKE_UNIQUE_NAMESPACE::convert_to_row_major_physical_data(logical_data, tensor_spec, pad_value);
        row_major_physical_data_span = tt::stl::make_const_span(row_major_physical_data);
    } else {
        row_major_physical_data_span = logical_data;
    }

    TT_FATAL(
        row_major_physical_data_span.size() == physical_shape.height() * physical_shape.width(),
        "Physical data size {} should be same as volume indicated by physical shape {}",
        row_major_physical_data_span.size(),
        physical_shape);

    if (tensor_spec.layout() == Layout::TILE) {
        return tensor_impl::convert_layout_row_major_to_tile(
            physical_shape, tensor_spec.tile(), row_major_physical_data_span);
    } else if (!row_major_physical_data.empty()) {
        // If conversion to physical data was performed, return the row major physical data to avoid extra copy.
        return row_major_physical_data;
    } else {
        // Otherwise, copy the `row_major_physical_data_span`.
        return std::vector<T>(row_major_physical_data_span.begin(), row_major_physical_data_span.end());
    }
}

template std::vector<bfloat16> encode_tensor_data<bfloat16>(
    tt::stl::Span<const bfloat16> logical_data, const TensorSpec& tensor_spec, bfloat16 pad_value);
template std::vector<float> encode_tensor_data<float>(
    tt::stl::Span<const float> logical_data, const TensorSpec& tensor_spec, float pad_value);
template std::vector<int32_t> encode_tensor_data<int32_t>(
    tt::stl::Span<const int32_t> logical_data, const TensorSpec& tensor_spec, int32_t pad_value);
template std::vector<uint32_t> encode_tensor_data<uint32_t>(
    tt::stl::Span<const uint32_t> logical_data, const TensorSpec& tensor_spec, uint32_t pad_value);
template std::vector<uint16_t> encode_tensor_data<uint16_t>(
    tt::stl::Span<const uint16_t> logical_data, const TensorSpec& tensor_spec, uint16_t pad_value);
template std::vector<uint8_t> encode_tensor_data<uint8_t>(
    tt::stl::Span<const uint8_t> logical_data, const TensorSpec& tensor_spec, uint8_t pad_value);

template <typename T>
std::vector<T> decode_tensor_data(tt::stl::Span<const T> physical_data, const TensorSpec& tensor_spec) {
    if (physical_data.size() == 0) {
        return {};
    }

    const auto& physical_shape = tensor_spec.physical_shape();
    TT_FATAL(
        physical_data.size() == physical_shape.height() * physical_shape.width(),
        "Physical data size {} should be same as volume indicated by physical shape {}",
        physical_data.size(),
        physical_shape);

    // If needed, convert physical data to row major physical data.
    // `row_major_physical_data_span` stores span unconditionally (cheap), while `row_major_physical_data` stores the
    // converted vector only when needed (expensive).
    std::vector<T> row_major_physical_data;
    tt::stl::Span<const T> row_major_physical_data_span;
    if (tensor_spec.layout() == Layout::TILE) {
        row_major_physical_data =
            tensor_impl::convert_layout_tile_to_row_major(physical_shape, tensor_spec.tile(), physical_data);
        row_major_physical_data_span = tt::stl::make_const_span(row_major_physical_data);
    } else {
        row_major_physical_data_span = physical_data;
    }

    // Same pattern as the above - `logical_data` is non empty only when the conversion to logical data was performed.
    std::vector<T> logical_data;
    tt::stl::Span<const T> logical_data_span;
    if (const auto& logical_2d_shape = tensor_spec.logical_2d_shape(); logical_2d_shape != physical_shape) {
        logical_data = CMAKE_UNIQUE_NAMESPACE::convert_to_logical_data(row_major_physical_data_span, tensor_spec);
        logical_data_span = tt::stl::make_const_span(logical_data);
    } else {
        logical_data_span = row_major_physical_data_span;
    }

    const auto& logical_shape = tensor_spec.logical_shape();
    TT_FATAL(
        logical_data_span.size() == logical_shape.volume(),
        "Logical data size {} should be same as volume indicated by logical shape {}",
        logical_data_span.size(),
        logical_shape);

    // Check if conversion to logical data was performed, to avoid extra copy upon return.
    if (!logical_data.empty()) {
        return logical_data;
    } else if (!row_major_physical_data.empty()) {
        return row_major_physical_data;
    } else {
        return std::vector<T>(logical_data_span.begin(), logical_data_span.end());
    }
}

bool logical_matches_physical(const TensorSpec& tensor_spec) {
    return tensor_spec.layout() == Layout::ROW_MAJOR && tensor_spec.logical_2d_shape() == tensor_spec.physical_shape();
}

template std::vector<bfloat16> decode_tensor_data<bfloat16>(
    tt::stl::Span<const bfloat16> physical_data, const TensorSpec& tensor_spec);
template std::vector<float> decode_tensor_data<float>(
    tt::stl::Span<const float> physical_data, const TensorSpec& tensor_spec);
template std::vector<int32_t> decode_tensor_data<int32_t>(
    tt::stl::Span<const int32_t> physical_data, const TensorSpec& tensor_spec);
template std::vector<uint32_t> decode_tensor_data<uint32_t>(
    tt::stl::Span<const uint32_t> physical_data, const TensorSpec& tensor_spec);
template std::vector<uint16_t> decode_tensor_data<uint16_t>(
    tt::stl::Span<const uint16_t> physical_data, const TensorSpec& tensor_spec);
template std::vector<uint8_t> decode_tensor_data<uint8_t>(
    tt::stl::Span<const uint8_t> physical_data, const TensorSpec& tensor_spec);

// ======================================================================================
//                                  .to_layout()
// ======================================================================================

template <typename T>
Tensor to_layout(const Tensor& tensor, Layout target_layout) {
    if (tensor.layout() == target_layout) {
        return tensor;
    }

    auto source_layout = tensor.layout();
    auto tile = tensor.tensor_spec().tile();
    auto physical_shape = tensor.tensor_spec().physical_shape();
    auto convert =
        [tile, &physical_shape, source_layout, target_layout](const HostBuffer& input_host_buffer) -> std::vector<T> {
        const auto input_data = input_host_buffer.view_as<T>();
        switch (source_layout) {
            case Layout::ROW_MAJOR:
                TT_FATAL(target_layout == Layout::TILE, "Unsupported layout conversion");
                return convert_layout_row_major_to_tile(physical_shape, tile, input_data);
            case Layout::TILE:
                TT_FATAL(target_layout == Layout::ROW_MAJOR, "Unsupported layout conversion");
                return convert_layout_tile_to_row_major(physical_shape, tile, input_data);
            case Layout::INVALID: TT_THROW("Invalid layout");
        }
        TT_THROW("Unreachable");
    };

    return Tensor(
        tensor.host_storage().transform([&](const HostBuffer& buffer) { return HostBuffer(convert(buffer)); }),
        TensorSpec(
            tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                tensor.dtype(),
                PageConfig(target_layout, tensor.tensor_spec().tile()),
                MemoryConfig{},
                tensor.logical_shape(),
                tensor.padded_shape())),
        tensor.distributed_tensor_config());
}

template Tensor to_layout<bfloat16>(const Tensor& tensor, Layout target_layout);
template Tensor to_layout<float>(const Tensor& tensor, Layout target_layout);
template Tensor to_layout<int32_t>(const Tensor& tensor, Layout target_layout);
template Tensor to_layout<uint32_t>(const Tensor& tensor, Layout target_layout);
template Tensor to_layout<uint16_t>(const Tensor& tensor, Layout target_layout);
template Tensor to_layout<uint8_t>(const Tensor& tensor, Layout target_layout);

template <typename T>
Tensor to_layout_bfloat(const Tensor& tensor, Layout target_layout) {
    static_assert(std::is_same_v<T, bfloat8_b> || std::is_same_v<T, bfloat4_b>, "Invalid type T");
    // TODO: Flip to assert when we remove use cases in python and c++
    if (tensor.layout() != target_layout or tensor.layout() != Layout::TILE) {
        log_warning(
            tt::LogAlways,
            "Tensor layout must be Layout::TILE for bfloat8_b or bfloat4_b! Conversion from {} to {} was not executed!",
            tensor.layout(),
            target_layout);
    }
    return tensor;
}

template <>
Tensor to_layout<bfloat8_b>(const Tensor& tensor, Layout target_layout) {
    return to_layout_bfloat<bfloat8_b>(tensor, target_layout);
}

template <>
Tensor to_layout<bfloat4_b>(const Tensor& tensor, Layout target_layout) {
    return to_layout_bfloat<bfloat4_b>(tensor, target_layout);
}

// ======================================================================================
//                                  .pad() and .unpad()
// ======================================================================================

template <typename T>
Tensor pad(
    const Tensor& tensor,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    float pad_value) {
    TT_FATAL(!is_device_tensor(tensor), "pad only supports host tensors");

    auto pad_value_ = static_cast<T>(pad_value);
    auto input_padded_shape = tensor.padded_shape();
    if (input_padded_shape.rank() < 2) {
        input_padded_shape = input_padded_shape.to_rank(2);
    }
    const auto input_strides = tensor.strides();

    auto pad = [&input_padded_shape, &output_padded_shape, &input_tensor_start, &pad_value_](
                   const HostBuffer& input_host_buffer) {
        const auto input_buffer = input_host_buffer.view_as<T>();
        const auto rank = input_padded_shape.rank();

        auto output_buffer = std::vector<T>(output_padded_shape.volume());
        std::fill(output_buffer.begin(), output_buffer.end(), pad_value_);

        if (input_padded_shape.volume() == 0) {
            return output_buffer;
        }

        if (rank == 1) {
            std::memcpy(
                output_buffer.data() + input_tensor_start[0],
                input_buffer.begin(),
                static_cast<size_t>(input_padded_shape[0]) * sizeof(T));
            return output_buffer;
        }

        // Calculate strides
        auto input_strides = compute_strides(input_padded_shape);
        auto output_strides = compute_strides(output_padded_shape);

        // Process all coordinates except for the last dimension (it's copied with mempcy)
        ttnn::SmallVector<size_t> coords(rank - 1, 0);

        bool processed_all_coords = false;
        while (!processed_all_coords) {
            // Calculate offset for a given coordinate for input and output. Again, last dimension is ignored
            size_t input_idx = 0;
            size_t output_idx = 0;

            for (int i = 0; i < rank - 1; ++i) {
                input_idx += coords[i] * input_strides[i];
                output_idx += (coords[i] + static_cast<size_t>(input_tensor_start[i])) * output_strides[i];
            }

            // Add offset (left padding) for the innermost dimension
            output_idx += static_cast<size_t>(input_tensor_start[rank - 1]) * output_strides[rank - 1];

            // Copy entire input row with memcpy
            std::memcpy(
                output_buffer.data() + output_idx,
                input_buffer.begin() + input_idx,
                static_cast<size_t>(input_padded_shape[rank - 1]) * sizeof(T));

            // Increment coordinates (from right to left), ignore last dimension
            processed_all_coords = true;
            for (int dim = rank - 2; dim >= 0; --dim) {
                coords[dim]++;
                // There are still coordinates to process in dim dimension
                if (coords[dim] < input_padded_shape[dim]) {
                    processed_all_coords = false;
                    break;
                }
                // This dim's coordinate overflowed, reset it and try to increment the next one
                coords[dim] = 0;
            }
        }

        return output_buffer;
    };

    return Tensor(
        tensor.host_storage().transform([&](const HostBuffer& buffer) { return HostBuffer(pad(buffer)); }),
        TensorSpec(
            tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                tensor.dtype(),
                PageConfig(tensor.layout(), tensor.tensor_spec().tile()),
                MemoryConfig{},
                tensor.logical_shape(),
                output_padded_shape)),
        tensor.distributed_tensor_config());
}

template Tensor pad<bfloat16>(
    const Tensor& tensor,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    float pad_value);
template Tensor pad<float>(
    const Tensor& tensor,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    float pad_value);
template Tensor pad<int32_t>(
    const Tensor& tensor,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    float pad_value);
template Tensor pad<uint32_t>(
    const Tensor& tensor,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    float pad_value);
template Tensor pad<uint16_t>(
    const Tensor& tensor,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    float pad_value);
template Tensor pad<uint8_t>(
    const Tensor& tensor,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    float pad_value);

template <>
Tensor pad<bfloat8_b>(
    const Tensor& tensor,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    float pad_value) {
    return pad_bfloat8_b(tensor, output_padded_shape, input_tensor_start, pad_value);
}

template <>
Tensor pad<bfloat4_b>(
    const Tensor& tensor,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    float pad_value) {
    return pad_bfloat4_b(tensor, output_padded_shape, input_tensor_start, pad_value);
}

template <typename T>
Tensor unpad(const Tensor& tensor, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) {
    TT_FATAL(!is_device_tensor(tensor), "unpad only supports host tensors");

    const auto& input_shape = tensor.padded_shape();
    const auto input_strides = tensor.strides();

    // Validate inputs and compute output shape
    ttnn::SmallVector<uint32_t> output_shape;
    for (auto i = 0; i < input_shape.rank(); i++) {
        // Check if tensor start and end indices are within input tensor shape
        TT_ASSERT(output_tensor_start[i] <= input_shape[i]);
        TT_ASSERT(output_tensor_end[i] <= input_shape[i]);
        // Check if start shape is < end shape
        TT_ASSERT(output_tensor_start[i] <= output_tensor_end[i]);
        // Figure out output tensor shape
        output_shape.push_back(output_tensor_end[i] - output_tensor_start[i]);
    }

    auto unpad = [&input_shape, &input_strides, &output_shape, &output_tensor_start, &output_tensor_end](
                     const HostBuffer& input_host_buffer) {
        const auto input_buffer = input_host_buffer.view_as<T>();
        ttnn::SmallVector<uint32_t> input_indices(input_shape.rank(), 0);

        auto flat_output_index = 0;
        auto output_buffer = std::vector<T>(ttnn::Shape(output_shape).volume());

        std::function<void(std::size_t)> unpad_from_tile = [&](std::size_t dim) -> void {
            for (auto i = output_tensor_start[dim]; i < output_tensor_end[dim]; i++) {
                input_indices[dim] = i;
                if (dim == input_shape.rank() - 1) {
                    auto flat_input_index = compute_flat_input_index(input_indices, input_strides);
                    output_buffer[flat_output_index++] = input_buffer[flat_input_index];
                } else {
                    unpad_from_tile(dim + 1);
                }
            }
        };
        unpad_from_tile(0);

        return output_buffer;
    };

    return Tensor(
        tensor.host_storage().transform([&](const HostBuffer& buffer) { return HostBuffer(unpad(buffer)); }),
        TensorSpec(
            ttnn::Shape(output_shape),
            tt::tt_metal::TensorLayout(
                tensor.dtype(),
                tt::tt_metal::PageConfig(tensor.layout(), tensor.tensor_spec().tile()),
                tt::tt_metal::MemoryConfig{})),
        tensor.distributed_tensor_config());
}

template Tensor unpad<bfloat16>(
    const Tensor& tensor, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end);
template Tensor unpad<float>(
    const Tensor& tensor, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end);
template Tensor unpad<int32_t>(
    const Tensor& tensor, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end);
template Tensor unpad<uint32_t>(
    const Tensor& tensor, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end);
template Tensor unpad<uint16_t>(
    const Tensor& tensor, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end);
template Tensor unpad<uint8_t>(
    const Tensor& tensor, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end);

template <>
Tensor unpad<bfloat8_b>(
    const Tensor& tensor, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) {
    return unpad_bfloat8_b(tensor, output_tensor_start, output_tensor_end);
}

template <>
Tensor unpad<bfloat4_b>(
    const Tensor& tensor, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) {
    return unpad_bfloat4_b(tensor, output_tensor_start, output_tensor_end);
}

// ======================================================================================
//                                  .extract_shard()
// ======================================================================================

template <typename T>
Tensor extract_shard(const Tensor& tensor, const uint32_t& core_id) {
    auto buffer = tensor.buffer();
    auto buffer_shard_shape = buffer->shard_spec().shape();
    ttnn::Shape shard_shape({1, 1, buffer_shard_shape[0], buffer_shard_shape[1]});
    std::vector<T> device_data;
    ::detail::ReadShard(*buffer, device_data, core_id);

    auto output_buffer = std::vector<T>(std::move(device_data));
    return Tensor(
        HostBuffer(std::move(output_buffer)),
        shard_shape,
        tensor.dtype(),
        tensor.layout(),
        tensor.tensor_spec().tile());
}

template Tensor extract_shard<bfloat16>(const Tensor& tensor, const uint32_t& core_id);
template Tensor extract_shard<float>(const Tensor& tensor, const uint32_t& core_id);
template Tensor extract_shard<int32_t>(const Tensor& tensor, const uint32_t& core_id);
template Tensor extract_shard<uint32_t>(const Tensor& tensor, const uint32_t& core_id);
template Tensor extract_shard<uint16_t>(const Tensor& tensor, const uint32_t& core_id);
template Tensor extract_shard<uint8_t>(const Tensor& tensor, const uint32_t& core_id);

template <>
Tensor extract_shard<bfloat8_b>(const Tensor& tensor, const uint32_t& core_id) {
    return extract_shard<uint32_t>(tensor, core_id);
}

template <>
Tensor extract_shard<bfloat4_b>(const Tensor& tensor, const uint32_t& core_id) {
    return extract_shard<uint32_t>(tensor, core_id);
}

}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
