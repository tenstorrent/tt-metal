// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>
#include <optional>

#include <sys/mman.h>
#include <unistd.h>

#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"

#include "tt-metalium/shape.hpp"
#include "tt-metalium/math.hpp"
#include "tt-metalium/distributed_host_buffer.hpp"
#include "tt-metalium/host_buffer.hpp"
#include "tt-metalium/memory_pin.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_coord.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_command_queue.hpp"
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>

#include <tt_stl/overloaded.hpp>
#include <tt_stl/span.hpp>
#include <tt_stl/assert.hpp>

#include <tracy/Tracy.hpp>

using namespace tt::tt_metal;

namespace tt::tt_metal::tensor_impl {

PrintOptions TTNN_PRINT_OPTIONS;

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

std::shared_ptr<distributed::MeshBuffer> allocate_device_buffer(
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
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
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
    auto output_float_data = host_buffer::get_as<const float>(float_tensor);
    auto output_packed_data =
        pack_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
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
    const Tensor& tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    const tt::tt_metal::Shape& output_tensor_end) {
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
    auto output_float_data = host_buffer::get_as<const float>(float_tensor);
    auto output_packed_data =
        pack_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
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
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
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
    auto output_float_data = host_buffer::get_as<const float>(float_tensor);
    auto output_packed_data =
        pack_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
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
    const Tensor& tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    const tt::tt_metal::Shape& output_tensor_end) {
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
    auto output_float_data = host_buffer::get_as<const float>(float_tensor);
    auto output_packed_data =
        pack_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
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
    size_t size{};
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
    switch (TTNN_PRINT_OPTIONS.profile) {
        case TensorPrintProfile::Empty: return DimensionShortener{size, 0};
        case TensorPrintProfile::Short: return DimensionShortener{size, 4};
        case TensorPrintProfile::Full: return DimensionShortener{size, std::nullopt};
        default: TT_THROW("Unrecognized TTNN_TENSOR_PRINT_PROFILE {}", TTNN_PRINT_OPTIONS.profile);
    }
}

inline void print_trailing_comma(std::ostream& ss, std::size_t index, std::size_t size, const std::string& after) {
    if (index < size - 1) {
        ss << "," << after;
    }
}

template <typename T>
inline void print_datum(std::ostream& ss, T datum, bool use_scientific = false) {
    if (std::is_integral_v<T>) {
        ss << std::setw(5) << datum;
    } else {
        int precision = TTNN_PRINT_OPTIONS.precision;
        if (use_scientific) {
            // Note: scientific required fixed width + 4 (e+/-AB, e.g. 1.23456e+08)
            ss << std::scientific << std::setw(precision + 7) << std::setprecision(precision) << datum;
        } else {
            ss << std::fixed << std::setw(precision + 3) << std::setprecision(precision) << datum;
        }
    }
}

template <>
inline void print_datum(std::ostream& ss, bfloat16 datum, bool use_scientific) {
    print_datum(ss, static_cast<float>(datum), use_scientific);
}

template <>
inline void print_datum(std::ostream& ss, uint8_t datum, bool use_scientific) {
    print_datum<uint32_t>(ss, datum, use_scientific);
}

// Helper function to determine if scientific notation should be used
template <typename T>
bool should_use_scientific_notation(tt::stl::Span<const T> buffer) {
    if (TTNN_PRINT_OPTIONS.sci_mode == SciMode::Enable) {
        return true;
    }
    if (TTNN_PRINT_OPTIONS.sci_mode == SciMode::Disable) {
        return false;
    }

    // SciMode::Default - auto-detect based on data range
    if constexpr (std::is_integral_v<T>) {
        return false;  // Never use scientific notation for integers
    } else {
        double nonzero_finite_min = std::numeric_limits<double>::max();
        double nonzero_finite_max = std::numeric_limits<double>::lowest();
        bool found_nonzero_finite = false;

        for (const auto& value : buffer) {
            double val = static_cast<double>(value);
            if (std::isfinite(val) && val != 0.0) {
                double abs_val = std::abs(val);
                nonzero_finite_min = std::min(nonzero_finite_min, abs_val);
                nonzero_finite_max = std::max(nonzero_finite_max, abs_val);
                found_nonzero_finite = true;
            }
        }

        if (!found_nonzero_finite) {
            return false;  // No nonzero finite values, don't use scientific notation
        }

        return (nonzero_finite_max / nonzero_finite_min > 1000.0) || (nonzero_finite_max > 1.0e8) ||
               (nonzero_finite_min < 1.0e-4);
    }
}

constexpr int constexpr_strlen(const char* str) { return *str ? 1 + constexpr_strlen(str + 1) : 0; }

constexpr auto TENSOR_TYPE_STRING = "ttnn.Tensor";
constexpr auto TENSOR_TYPE_STRING_PLUS_OPEN_PARENTHESIS_LENGTH = constexpr_strlen(TENSOR_TYPE_STRING) + 1;

template <typename T>
void to_string_row_major(
    std::stringstream& ss,
    tt::stl::Span<const T> buffer,
    const tt::tt_metal::Shape& shape,
    const tt::tt_metal::Strides& strides,
    std::size_t outer_index,
    const std::size_t buffer_offset,
    int64_t rank,
    int64_t dim,
    bool use_scientific) {
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
            to_string_row_major(
                ss, buffer, shape, strides, index, buffer_offset + (index * stride), rank - 1, dim + 1, use_scientific);
        } else {
            print_datum(ss, buffer[buffer_offset + index], use_scientific);
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
Tensor to_layout_impl(const Tensor& tensor, Layout target_layout);

template <typename T>
std::string to_string_impl(const Tensor& tensor) {
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
        }
        if (tensor.dtype() == DataType::BFLOAT8_B || tensor.dtype() == DataType::BFLOAT4_B) {
            return to_layout_impl<T>(tt::tt_metal::to_dtype(tensor, DataType::FLOAT32), Layout::ROW_MAJOR);
        }
        return to_layout_impl<T>(tensor, Layout::ROW_MAJOR);
    };

    auto get_device_buffers = [&](const HostStorage& storage) {
        std::vector<HostBuffer> buffers;
        storage.buffer().apply([&](const HostBuffer& shard) { buffers.push_back(shard); });
        return buffers;
    };

    return std::visit(
        tt::stl::overloaded{
            [&](const HostStorage& /*storage*/) -> std::string {
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
                    return to_string_impl<T>(cpu_tensor);
                }

                auto* mesh_device = storage.mesh_buffer->device();
                // TODO: Uncomment after the distributed tensors migration to tt-metal is complete.
                // if (mesh_device->num_devices() == 1) {
                //     return to_string<T>(ttnn::distributed::get_device_tensors(cpu_tensor).at(0));
                // }

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

template <>
std::string to_string_impl<bfloat8_b>(const Tensor& tensor) {
    return to_string_impl<float>(tensor);
}

template <>
std::string to_string_impl<bfloat4_b>(const Tensor& tensor) {
    return to_string_impl<float>(tensor);
}

std::string to_string(const Tensor& tensor) {
    return dispatch(tensor.dtype(), [&]<typename T>() { return to_string_impl<T>(tensor); });
}

// ======================================================================================
//                                      .to_host()
// ======================================================================================

HostBuffer allocate_host_buffer(const TensorSpec& tensor_spec) {
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

Tensor to_host(const Tensor& tensor, bool blocking, std::optional<tt::tt_metal::QueueId> cq_id) {
    TT_FATAL(tensor.is_allocated(), "Buffer must be allocated on device!");
    const auto& storage = tensor.device_storage();
    const auto& mesh_buffer = storage.mesh_buffer;
    distributed::MeshDevice* device = mesh_buffer->device();

    auto cq_id_int = tt::tt_metal::raw_optional(cq_id);
    distributed::MeshCommandQueue& mesh_cq = device->mesh_command_queue(cq_id_int);

    // For performance, perform all allocations via DistributedHostBuffer::transform, run from multiple threads.
    auto distributed_host_buffer = DistributedHostBuffer::create(device->get_view());

    distributed_host_buffer.emplace_shards(
        storage.coords,
        [&](const distributed::MeshCoordinate&) { return allocate_host_buffer(tensor.tensor_spec()); },
        DistributedHostBuffer::ProcessShardExecutionPolicy::PARALLEL);

    mesh_cq.enqueue_read(mesh_buffer, distributed_host_buffer, /*shards=*/std::nullopt, blocking);

    HostStorage host_storage(std::move(distributed_host_buffer));
    return Tensor(std::move(host_storage), tensor.tensor_spec(), tensor.tensor_topology());
}

// ======================================================================================
//                               .to_device() details
// ======================================================================================

namespace {

DeviceStorage replicate_to_mesh_buffer(
    const HostBuffer& buffer,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    const TensorSpec& tensor_spec,
    std::optional<tt::tt_metal::QueueId> cq_id) {
    auto* mesh_device = mesh_buffer->device();
    auto data_to_write = buffer.view_bytes();
    const auto expected_packed_buffer_size_bytes = tensor_spec.compute_packed_buffer_size_bytes();
    const auto input_size_bytes = data_to_write.size();
    TT_FATAL(
        input_size_bytes == expected_packed_buffer_size_bytes,
        "Host data with total size {}B does not match expected size {}B of device buffer!",
        input_size_bytes,
        expected_packed_buffer_size_bytes);

    std::optional<uint8_t> cq_id_int = cq_id.has_value() ? std::make_optional(cq_id.value().get()) : std::nullopt;
    mesh_device->mesh_command_queue(cq_id_int).enqueue_write_mesh_buffer(
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
    std::optional<tt::tt_metal::QueueId> cq_id) {
    std::optional<uint8_t> cq_id_int = cq_id.has_value() ? std::make_optional(cq_id.value().get()) : std::nullopt;
    mesh_buffer->device()->mesh_command_queue(cq_id_int).enqueue_write(
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

std::pair<DeviceStorage, TensorTopology> to_device_mesh_buffer(
    const Storage& host_storage,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    const TensorSpec& tensor_spec,
    const TensorAttributes& host_tensor_attributes,
    const TensorTopology& tensor_topology,
    std::optional<tt::tt_metal::QueueId> cq_id) {
    return std::visit(
        tt::stl::overloaded{
            [&mesh_buffer, &tensor_spec, cq_id, &host_tensor_attributes, &tensor_topology](
                const HostStorage& storage) -> std::pair<DeviceStorage, TensorTopology> {
                const auto& host_storage_shape = storage.buffer().shape();
                const auto& mesh_device_shape = mesh_buffer->device()->shape();
                if (host_storage_shape.mesh_size() < mesh_device_shape.mesh_size() &&
                    host_storage_shape == distributed::MeshShape(1, 1)) {
                    // Special case of replicating tensors on 1x1 mesh across the entire mesh device.
                    const auto device_buffer = storage.buffer().get_shard(distributed::MeshCoordinate(0, 0));
                    return {
                        replicate_to_mesh_buffer(*device_buffer, mesh_buffer, tensor_spec, cq_id),
                        TensorTopology::create_fully_replicated_tensor_topology(mesh_device_shape)};
                }
                TT_FATAL(
                    host_storage_shape == mesh_device_shape,
                    "Distributed host buffer has different shape {} than the mesh device {}",
                    host_storage_shape,
                    mesh_device_shape);
                return {write_to_mesh_buffer(storage.buffer(), mesh_buffer, cq_id), tensor_topology};
            },
            [](const auto& s) -> std::pair<DeviceStorage, TensorTopology> {
                TT_THROW("Unexpected storage type {}", tt::stl::get_type_name(s));
            }},
        host_storage);
}

Tensor to_device(
    const Tensor& tensor,
    distributed::MeshDevice* mesh_device,
    ttsl::optional_reference<const MemoryConfig> memory_config,
    std::optional<tt::tt_metal::QueueId> cq_id) {
    if (tensor.storage_type() == StorageType::DEVICE) {
        return tensor;  // Tensor already on device
    }

    TT_FATAL(mesh_device != nullptr, "Need target device in order to move tensor to device!");

    std::optional<TensorSpec> tensor_spec_overriden_memory_config;
    if (memory_config) {
        tensor_spec_overriden_memory_config = tensor.tensor_spec().with_memory_config(*memory_config);
    }

    const auto* tensor_spec = tensor_spec_overriden_memory_config.has_value()
                                  ? &tensor_spec_overriden_memory_config.value()
                                  : &tensor.tensor_spec();
    auto mesh_buffer = allocate_device_buffer(mesh_device, *tensor_spec);
    auto [mesh_storage, topology] = to_device_mesh_buffer(
        tensor.storage(), mesh_buffer, *tensor_spec, *tensor.tensor_attributes, tensor.tensor_topology(), cq_id);
    return Tensor(std::move(mesh_storage), *tensor_spec, topology);
}

void copy_to_host(
    const Tensor& device_tensor, Tensor& host_tensor, bool blocking, std::optional<tt::tt_metal::QueueId> cq_id) {
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
    distributed::MeshDevice* device = mesh_buffer->device();

    auto cq_id_int = tt::tt_metal::raw_optional(cq_id);
    distributed::MeshCommandQueue& mesh_cq = device->mesh_command_queue(cq_id_int);

    const auto& distributed_host_buffer = host_tensor.host_storage().buffer();

    // Host tensor must have pre-allocated buffers for all device shards.
    // However, it may have some extra shards. Drop them by "unwrapping" the distributed host buffer, and re-wrapping
    // only for those shards that are actually present on device.
    std::vector<std::pair<distributed::MeshCoordinate, std::optional<HostBuffer>>> shards;
    shards.reserve(device_storage.coords.size());
    for (const auto& device_coord : device_storage.coords) {
        shards.push_back({device_coord, distributed_host_buffer.get_shard(device_coord)});
    }

    DistributedHostBuffer dst_distributed_host_buffer = DistributedHostBuffer::create(device->get_view());
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
        device_tensor.tensor_topology());
}

void copy_to_device(const Tensor& host_tensor, Tensor& device_tensor, std::optional<tt::tt_metal::QueueId> cq_id) {
    TT_FATAL(host_tensor.storage_type() == StorageType::HOST, "Source tensor is not on host.");
    TT_FATAL(device_tensor.storage_type() == StorageType::DEVICE, "Destination tensor is not on device.");
    TT_FATAL(device_tensor.is_allocated(), "Buffer must be allocated on device.");

    TT_FATAL(host_tensor.logical_shape() == device_tensor.logical_shape(), "Host tensor has different shape");
    TT_FATAL(host_tensor.dtype() == device_tensor.dtype(), "Host tensor has different dtype");
    TT_FATAL(
        host_tensor.tensor_spec().page_config() == device_tensor.tensor_spec().page_config(),
        "Host tensor has different page config");

    auto mesh_buffer = device_tensor.device_storage().mesh_buffer;

    auto [mesh_storage, topology] = to_device_mesh_buffer(
        host_tensor.storage(),
        mesh_buffer,
        device_tensor.tensor_spec(),
        *host_tensor.tensor_attributes,
        host_tensor.tensor_topology(),
        cq_id);
    device_tensor = Tensor(
        std::move(mesh_storage), host_tensor.tensor_spec().with_memory_config(device_tensor.memory_config()), topology);
}

// ======================================================================================
//     Helpers for converting between logical <-> physical data with full tensor spec
// ======================================================================================
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// Useful information about how a shard_shape cuts a 2D shape
// - num_shards_height: Number of shards along the height (including partial last shard, if any)
// - last_shard_height: Height of last partial shard (if None, it will be same as full shard shape height)
// - num_shards_width: Number of shards along the width (including partial last shard, if any)
// - last_shard_width: Width of last partial shard (if None, it will be same as full shard shape width)
struct ShardDivisionSpec {
    size_t num_shards_height = 0;
    size_t last_shard_height = 0;
    size_t num_shards_width = 0;
    size_t last_shard_width = 0;
};

ShardDivisionSpec compute_shard_division_spec(const Shape2D& shape, const Shape2D& shard_shape) {
    const auto num_shards_height = tt::div_up(shape.height(), shard_shape.height());
    const auto last_shard_height =
        shape.height() % shard_shape.height() > 0 ? shape.height() % shard_shape.height() : shard_shape.height();
    const auto num_shards_width = tt::div_up(shape.width(), shard_shape.width());
    const auto last_shard_width =
        shape.width() % shard_shape.width() > 0 ? shape.width() % shard_shape.width() : shard_shape.width();

    return ShardDivisionSpec{num_shards_height, last_shard_height, num_shards_width, last_shard_width};
};

// TODO: Remove when we get rid of physical sharding and generalize interleaved and sharded; when we do, directly get
// from TensorLayout
std::array<Shape2D, 2> get_logical_and_physical_shard_shapes(const TensorSpec& tensor_spec) {
    const auto& logical_shape = tensor_spec.logical_shape();
    const auto& padded_shape = tensor_spec.padded_shape();

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
        compute_shard_division_spec(logical_2d_shape, logical_shard_shape);

    std::vector<LogicalPhysicalMapping> logical_physical_mapping{};
    logical_physical_mapping.reserve(num_shards_height * num_shards_width);

    for (size_t shard_height_idx = 0; shard_height_idx < num_shards_height; shard_height_idx++) {
        for (size_t shard_width_idx = 0; shard_width_idx < num_shards_width; shard_width_idx++) {
            const auto num_shard_rows =
                shard_height_idx == num_shards_height - 1 ? last_shard_height : logical_shard_shape.height();
            const auto num_shard_cols =
                shard_width_idx == num_shards_width - 1 ? last_shard_width : logical_shard_shape.width();

            auto indices = LogicalPhysicalIdxPairs(num_shard_rows);
            const auto logical_start_idx = (shard_height_idx * logical_shard_shape.height() * logical_stride) +
                                           (shard_width_idx * logical_shard_shape.width());
            const auto physical_start_idx = (shard_height_idx * physical_shard_shape.height() * physical_stride) +
                                            (shard_width_idx * physical_shard_shape.width());
            for (size_t i = 0; i < num_shard_rows; i++) {
                indices[i] = {(i * logical_stride) + logical_start_idx, (i * physical_stride) + physical_start_idx};
            }

            logical_physical_mapping.emplace_back(indices, num_shard_cols);
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

template <typename T>
std::vector<T> convert_layout_row_major_to_tile(
    const Shape2D& shape, const Tile& tile, tt::stl::Span<const T> data_to_convert) {
    if (shape.width() * shape.height() == 0) {
        return std::vector<T>();
    }
    TT_FATAL(
        (shape.height() % tile.get_tile_shape()[0] == 0 && shape.width() % tile.get_tile_shape()[1] == 0),
        "Unsupported shape for tensor conversion from row-major to tile layout. The tensor shape height and width must "
        "be a multiple of tile height ({}) and width ({}), but the provided shape is {}",
        tile.get_tile_shape()[0],
        tile.get_tile_shape()[1],
        shape);

    auto tile_shape = tile.get_tile_shape();
    auto face_shape = tile.get_face_shape();
    auto transpose_within_face = tile.get_transpose_within_face();
    auto transpose_of_faces = tile.get_transpose_of_faces();

    return convert_layout(
        data_to_convert,
        shape,
        TensorLayoutType::LIN_ROW_MAJOR,
        TensorLayoutType::TILED_NFACES,
        tile_shape,
        face_shape,
        transpose_within_face,
        transpose_of_faces);
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
        return CMAKE_UNIQUE_NAMESPACE::convert_layout_row_major_to_tile(
            physical_shape, tensor_spec.tile(), row_major_physical_data_span);
    }
    if (!row_major_physical_data.empty()) {
        // If conversion to physical data was performed, return the row major physical data to avoid extra copy.
        return row_major_physical_data;
    }  // Otherwise, copy the `row_major_physical_data_span`.
    return std::vector<T>(row_major_physical_data_span.begin(), row_major_physical_data_span.end());
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
    }
    if (!row_major_physical_data.empty()) {
        return row_major_physical_data;
    }
    return std::vector<T>(logical_data_span.begin(), logical_data_span.end());
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
Tensor to_layout_impl(const Tensor& tensor, Layout target_layout) {
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
                return CMAKE_UNIQUE_NAMESPACE::convert_layout_row_major_to_tile(physical_shape, tile, input_data);
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
        tensor.tensor_topology());
}

template <typename T>
Tensor to_layout_bfloat_impl(const Tensor& tensor, Layout target_layout) {
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
Tensor to_layout_impl<bfloat8_b>(const Tensor& tensor, Layout target_layout) {
    return to_layout_bfloat_impl<bfloat8_b>(tensor, target_layout);
}

template <>
Tensor to_layout_impl<bfloat4_b>(const Tensor& tensor, Layout target_layout) {
    return to_layout_bfloat_impl<bfloat4_b>(tensor, target_layout);
}

Tensor to_layout(const Tensor& tensor, Layout target_layout) {
    return dispatch(tensor.dtype(), [&]<typename T>() { return to_layout_impl<T>(tensor, target_layout); });
}

// ======================================================================================
//                                  .pad() and .unpad()
// ======================================================================================

template <typename T>
Tensor pad_impl(
    const Tensor& tensor,
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
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
                input_buffer.data(),
                static_cast<size_t>(input_padded_shape[0]) * sizeof(T));
            return output_buffer;
        }

        // Calculate strides
        auto input_strides = compute_strides(input_padded_shape);
        auto output_strides = compute_strides(output_padded_shape);

        // Process all coordinates except for the last dimension (it's copied with mempcy)
        ttsl::SmallVector<size_t> coords(rank - 1, 0);

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
                input_buffer.data() + input_idx,
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
        tensor.tensor_topology());
}

template <>
Tensor pad_impl<bfloat8_b>(
    const Tensor& tensor,
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
    float pad_value) {
    return pad_bfloat8_b(tensor, output_padded_shape, input_tensor_start, pad_value);
}

template <>
Tensor pad_impl<bfloat4_b>(
    const Tensor& tensor,
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
    float pad_value) {
    return pad_bfloat4_b(tensor, output_padded_shape, input_tensor_start, pad_value);
}

Tensor pad(
    const Tensor& tensor,
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
    float pad_value) {
    return dispatch(tensor.dtype(), [&]<typename T>() {
        return pad_impl<T>(tensor, output_padded_shape, input_tensor_start, pad_value);
    });
}

template <typename T>
Tensor unpad_impl(
    const Tensor& tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    const tt::tt_metal::Shape& output_tensor_end) {
    TT_FATAL(!is_device_tensor(tensor), "unpad only supports host tensors");

    const auto& input_shape = tensor.padded_shape();
    const auto input_strides = compute_strides(input_shape);

    // Validate inputs and compute output shape
    ttsl::SmallVector<uint32_t> output_shape;
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
        ttsl::SmallVector<uint32_t> input_indices(input_shape.rank(), 0);

        auto flat_output_index = 0;
        auto output_buffer = std::vector<T>(tt::tt_metal::Shape(output_shape).volume());

        std::function<void(std::size_t)> unpad_from_tile = [&](std::size_t dim) -> void {
            for (auto i = output_tensor_start[dim]; i < output_tensor_end[dim]; i++) {
                input_indices[dim] = i;
                if (dim == input_shape.rank() - 1) {
                    auto flat_input_index = compute_flat_indices(input_indices, input_strides);
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
            tt::tt_metal::Shape(output_shape),
            tt::tt_metal::TensorLayout(
                tensor.dtype(),
                tt::tt_metal::PageConfig(tensor.layout(), tensor.tensor_spec().tile()),
                tt::tt_metal::MemoryConfig{})),
        tensor.tensor_topology());
}

template <>
Tensor unpad_impl<bfloat8_b>(
    const Tensor& tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    const tt::tt_metal::Shape& output_tensor_end) {
    return unpad_bfloat8_b(tensor, output_tensor_start, output_tensor_end);
}

template <>
Tensor unpad_impl<bfloat4_b>(
    const Tensor& tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    const tt::tt_metal::Shape& output_tensor_end) {
    return unpad_bfloat4_b(tensor, output_tensor_start, output_tensor_end);
}

Tensor unpad(
    const Tensor& tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    const tt::tt_metal::Shape& output_tensor_end) {
    return dispatch(
        tensor.dtype(), [&]<typename T>() { return unpad_impl<T>(tensor, output_tensor_start, output_tensor_end); });
}

// ======================================================================================
//                                  .extract_shard()
// ======================================================================================

template <typename T>
Tensor extract_shard_impl(const Tensor& tensor, const uint32_t& core_id) {
    auto* buffer = tensor.buffer();
    auto buffer_shard_shape = buffer->shard_spec().shape();
    tt::tt_metal::Shape shard_shape({1, 1, buffer_shard_shape[0], buffer_shard_shape[1]});
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

template <>
Tensor extract_shard_impl<bfloat8_b>(const Tensor& tensor, const uint32_t& core_id) {
    return extract_shard_impl<uint32_t>(tensor, core_id);
}

template <>
Tensor extract_shard_impl<bfloat4_b>(const Tensor& tensor, const uint32_t& core_id) {
    return extract_shard_impl<uint32_t>(tensor, core_id);
}

Tensor extract_shard(const Tensor& tensor, const uint32_t& core_id) {
    return dispatch(tensor.dtype(), [&]<typename T>() { return extract_shard_impl<T>(tensor, core_id); });
}

// ======================================================================================
//                                  .to_dtype()
// ======================================================================================

namespace detail {

struct bfloat4_tag {};
struct bfloat8_tag {};

// Preprocess the storage to unpack the bfloat8/4 tiles into float32.
tt::tt_metal::HostStorage preprocess_storage(
    const tt::tt_metal::HostStorage& input_storage, const DataType input_dtype) {
    constexpr bool row_major_output = false;
    constexpr bool is_exp_a = false;

    if (input_dtype == DataType::BFLOAT8_B) {
        return input_storage.transform([&](const tt::tt_metal::HostBuffer& buffer) {
            tt::stl::Span<const uint32_t> uint32_data = buffer.view_as<const uint32_t>();
            auto float_unpacked_data = unpack_bfp8_tiles_into_float_vec(uint32_data, row_major_output, is_exp_a);
            return tt::tt_metal::HostBuffer(std::move(float_unpacked_data));
        });
    }
    if (input_dtype == DataType::BFLOAT4_B) {
        return input_storage.transform([&](const tt::tt_metal::HostBuffer& buffer) {
            tt::stl::Span<const uint32_t> uint32_data = buffer.view_as<const uint32_t>();
            auto float_unpacked_data = unpack_bfp4_tiles_into_float_vec(uint32_data, row_major_output, is_exp_a);
            return tt::tt_metal::HostBuffer(std::move(float_unpacked_data));
        });
    }
    return input_storage;
}

template <typename SrcType, typename DstType>
tt::tt_metal::HostStorage transform_storage(
    const tt::tt_metal::TensorSpec& input_tensor_spec, const tt::tt_metal::HostStorage& input_storage) {
    if constexpr (std::is_same_v<SrcType, DstType>) {
        return input_storage;
    } else if constexpr (std::is_same_v<DstType, bfloat4_tag> || std::is_same_v<DstType, bfloat8_tag>) {
        auto transform_fn = [&](const tt::tt_metal::HostBuffer& buffer) {
            ttsl::Span<const SrcType> data = buffer.view_as<const SrcType>();
            std::vector<SrcType> tilized_data;  // empty if `data` is already in tile layout.
            if (input_tensor_spec.layout() == Layout::ROW_MAJOR) {
                tilized_data = CMAKE_UNIQUE_NAMESPACE::convert_layout_row_major_to_tile(
                    input_tensor_spec.physical_shape(), input_tensor_spec.tile(), data);
                data = ttsl::make_const_span(tilized_data);
            }

            auto float_packed_data = [&]() {
                constexpr bool row_major_input = false;
                constexpr bool is_exp_a = false;
                if constexpr (std::is_same_v<DstType, bfloat8_tag>) {
                    return pack_as_bfp8_tiles(data, row_major_input, is_exp_a, input_tensor_spec.tile());
                } else if constexpr (std::is_same_v<DstType, bfloat4_tag>) {
                    return pack_as_bfp4_tiles(data, row_major_input, is_exp_a, input_tensor_spec.tile());
                } else {
                    static_assert(ttsl::concepts::always_false_v<DstType>, "Unsupported data type");
                }
            }();
            return tt::tt_metal::HostBuffer(std::move(float_packed_data));
        };

        return input_storage.transform(transform_fn);
    } else {
        auto transform_fn = [&](const tt::tt_metal::HostBuffer& buffer) {
            auto data = buffer.view_as<const SrcType>();
            std::vector<DstType> output_vector(data.size());
            std::transform(data.begin(), data.end(), output_vector.begin(), [](SrcType value) {
                return static_cast<DstType>(value);
            });
            return tt::tt_metal::HostBuffer(std::move(output_vector));
        };

        return input_storage.transform(transform_fn);
    }
}

}  // namespace detail

Tensor to_dtype(const Tensor& input_tensor, DataType dtype) {
    const auto src_type = input_tensor.dtype();
    if (src_type == dtype) {
        return input_tensor;
    }

    TT_FATAL(is_cpu_tensor(input_tensor), "to_dtype(...) function only supports host tensors!");

    auto input_storage = detail::preprocess_storage(input_tensor.host_storage(), src_type);

    auto output_storage = [src_type, dst_type = dtype, &input_tensor, &input_storage]() {
        auto with_src_and_dst = [&]<typename SrcType, typename DstType>() {
            return detail::transform_storage<SrcType, DstType>(input_tensor.tensor_spec(), input_storage);
        };

        auto with_src = [dst_type, &with_src_and_dst]<typename SrcType>() {
            switch (dst_type) {
                case DataType::BFLOAT4_B: return with_src_and_dst.operator()<SrcType, detail::bfloat4_tag>();
                case DataType::BFLOAT8_B: return with_src_and_dst.operator()<SrcType, detail::bfloat8_tag>();
                case DataType::FLOAT32: return with_src_and_dst.operator()<SrcType, float>();
                case DataType::BFLOAT16: return with_src_and_dst.operator()<SrcType, bfloat16>();
                case DataType::UINT8: return with_src_and_dst.operator()<SrcType, uint8_t>();
                case DataType::UINT16: return with_src_and_dst.operator()<SrcType, uint16_t>();
                case DataType::UINT32: return with_src_and_dst.operator()<SrcType, uint32_t>();
                case DataType::INT32: return with_src_and_dst.operator()<SrcType, int32_t>();
                case DataType::INVALID: TT_THROW("Unsupported data type conversion requested. Source type is invalid!");
            }
            TT_THROW("Unreachable");
        };

        switch (src_type) {
            case DataType::BFLOAT4_B:
            case DataType::BFLOAT8_B:
            case DataType::FLOAT32: return with_src.operator()<float>();
            case DataType::BFLOAT16: return with_src.operator()<bfloat16>();
            case DataType::UINT8: return with_src.operator()<uint8_t>();
            case DataType::UINT16: return with_src.operator()<uint16_t>();
            case DataType::UINT32: return with_src.operator()<uint32_t>();
            case DataType::INT32: return with_src.operator()<int32_t>();
            case DataType::INVALID: TT_THROW("Unsupported data type conversion requested. Source type is invalid!");
        }
        TT_THROW("Unreachable");
    }();

    const auto layout =
        (dtype == DataType::BFLOAT4_B || dtype == DataType::BFLOAT8_B) ? Layout::TILE : input_tensor.layout();

    auto output_spec = TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout::fromPaddedShape(
            dtype,
            tt::tt_metal::PageConfig(layout, input_tensor.tensor_spec().tile()),
            input_tensor.tensor_spec().memory_config(),
            input_tensor.logical_shape(),
            input_tensor.padded_shape()));

    return Tensor(tt::tt_metal::HostStorage(std::move(output_storage)), output_spec, input_tensor.tensor_topology());
}

}  // namespace tt::tt_metal::tensor_impl
