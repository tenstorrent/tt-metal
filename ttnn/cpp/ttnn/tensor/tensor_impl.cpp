// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor_impl.hpp"
#include <optional>

#include "tt-metalium/mesh_buffer.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/mesh_command_queue.hpp"
#include <tt_stl/overloaded.hpp>
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_impl_wrapper.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/distributed/api.hpp"

using namespace tt::tt_metal;

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
    auto shard_spec_buffer = tensor_spec.compute_shard_spec_buffer();
    auto memory_config = tensor_spec.tensor_layout().get_memory_config();

    return Buffer::create(
        device,
        buffer_size_bytes,
        page_size_bytes,
        memory_config.buffer_type,
        memory_config.memory_layout,
        shard_spec_buffer);
}

std::shared_ptr<distributed::MeshBuffer> allocate_mesh_buffer_on_device(
    distributed::MeshDevice* mesh_device, const TensorSpec& tensor_spec) {
    const auto& memory_config = tensor_spec.tensor_layout().get_memory_config();
    const distributed::DeviceLocalBufferConfig device_local_buffer_config{
        .page_size = tensor_spec.compute_page_size_bytes(),
        .buffer_type = memory_config.buffer_type,
        .buffer_layout = memory_config.memory_layout,
        .shard_parameters = tensor_spec.compute_shard_spec_buffer(),
    };

    // Use replicated buffer, which supports both working with individual shards and replicating data across all shards.
    // This is required for the time being, as TTNN has rich multi-device sharding implementation.
    const distributed::ReplicatedBufferConfig replicated_buffer_config{
        .size = tensor_spec.compute_packed_buffer_size_bytes(),
    };

    return distributed::MeshBuffer::create(replicated_buffer_config, device_local_buffer_config, mesh_device);
}

void validate_on_device_dtype_and_layout(IDevice* device, const ttnn::Shape& shape, DataType dtype, Layout layout) {
    // TODO: Get supported layout and dtypes from device
    auto supported_dtype = [&dtype]() {
        TT_ASSERT(
            (dtype == DataType::UINT32 || dtype == DataType::INT32 || dtype == DataType::FLOAT32 ||
             dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::BFLOAT16 ||
             dtype == DataType::BFLOAT8_B || dtype == DataType::BFLOAT4_B),
            "Only UINT32, INT32, FLOAT32, UINT16, UINT8, BFLOAT16, BFLOAT8_B, or BFLOAT4_B dtypes are supported on "
            "device!");
    };
    auto supported_layout = [&dtype, &layout]() {
        switch (dtype) {
            case DataType::UINT32:
            case DataType::INT32:
            case DataType::FLOAT32:
            case DataType::UINT8:
            case DataType::UINT16:
            case DataType::BFLOAT16: break;
            case DataType::BFLOAT8_B:
            case DataType::BFLOAT4_B:
                TT_ASSERT(layout == Layout::TILE, "Only TILE layout is supported for BFLOAT8_B dtype!");
                break;
            default:
                TT_ASSERT(
                    false,
                    "Only UINT32, INT32, FLOAT32, UINT16, BFLOAT16, BFLOAT8_B, or BFLOAT4_B dtypes are supported on "
                    "device!");
                break;
        }
    };
    supported_dtype();
    supported_layout();
}

Tensor pad_bfloat8_b(
    const Tensor& tensor,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    float pad_value) {
    auto tile = tensor.get_tensor_spec().tile();
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and pad
    auto input_packed_data = owned_buffer::get_as<uint32_t>(tensor).get();
    auto input_float_data =
        unpack_bfp8_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
    auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
    auto float_tensor = Tensor(
                            OwnedStorage{input_float_buffer},
                            TensorSpec(
                                tensor.get_logical_shape(),
                                TensorLayout::fromPaddedShape(
                                    DataType::FLOAT32,
                                    PageConfig(tensor.get_layout(), tile),
                                    MemoryConfig{},
                                    tensor.get_logical_shape(),
                                    tensor.get_padded_shape())))
                            .pad(output_padded_shape, input_tensor_start, pad_value);

    // Convert back to BFLOAT8_B
    auto output_float_data = owned_buffer::get_as<float>(float_tensor).get();
    auto output_packed_data =
        pack_fp32_vec_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
    TensorSpec output_spec(
        float_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            DataType::BFLOAT8_B,
            tensor.get_tensor_spec().page_config(),
            MemoryConfig{},
            float_tensor.logical_shape(),
            float_tensor.padded_shape()));
    return Tensor(std::move(OwnedStorage{std::move(output_uint32_buffer)}), output_spec);
}

Tensor unpad_bfloat8_b(
    const Tensor& tensor, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) {
    auto tile = tensor.get_tensor_spec().tile();
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and unpad
    auto input_packed_data = owned_buffer::get_as<uint32_t>(tensor).get();
    auto input_float_data =
        unpack_bfp8_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
    auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
    auto float_tensor = Tensor(
                            OwnedStorage{input_float_buffer},
                            TensorSpec(
                                tensor.get_logical_shape(),
                                TensorLayout::fromPaddedShape(
                                    DataType::FLOAT32,
                                    PageConfig(tensor.get_layout(), tile),
                                    MemoryConfig{},
                                    tensor.get_logical_shape(),
                                    tensor.get_padded_shape())))
                            .unpad(output_tensor_start, output_tensor_end);

    // Convert back to BFLOAT8_B
    auto output_float_data = owned_buffer::get_as<float>(float_tensor).get();
    auto output_packed_data =
        pack_fp32_vec_as_bfp8_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
    return Tensor(
        std::move(OwnedStorage{std::move(output_uint32_buffer)}),
        TensorSpec(
            float_tensor.get_logical_shape(),
            TensorLayout::fromPaddedShape(
                DataType::BFLOAT8_B,
                PageConfig(tensor.get_layout(), tile),
                MemoryConfig{},
                float_tensor.get_logical_shape(),
                float_tensor.get_padded_shape())));
}

Tensor pad_bfloat4_b(
    const Tensor& tensor,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    float pad_value) {
    auto tile = tensor.get_tensor_spec().tile();
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and pad
    auto input_packed_data = owned_buffer::get_as<uint32_t>(tensor).get();
    auto input_float_data =
        unpack_bfp4_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
    auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
    auto float_tensor = Tensor(
                            OwnedStorage{input_float_buffer},
                            TensorSpec(
                                tensor.get_logical_shape(),
                                TensorLayout::fromPaddedShape(
                                    DataType::FLOAT32,
                                    PageConfig(tensor.get_layout(), tile),
                                    MemoryConfig{},
                                    tensor.get_logical_shape(),
                                    tensor.get_logical_shape())))
                            .pad(output_padded_shape, input_tensor_start, pad_value);

    // Convert back to BFLOAT4_B
    auto output_float_data = owned_buffer::get_as<float>(float_tensor).get();
    auto output_packed_data =
        pack_fp32_vec_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
    TensorSpec output_spec(
        float_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            DataType::BFLOAT4_B,
            tensor.get_tensor_spec().page_config(),
            MemoryConfig{},
            float_tensor.logical_shape(),
            float_tensor.padded_shape()));
    return Tensor(std::move(OwnedStorage{std::move(output_uint32_buffer)}), output_spec);
}

Tensor unpad_bfloat4_b(
    const Tensor& tensor, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) {
    auto tile = tensor.get_tensor_spec().tile();
    // TODO(arakhmati): do not convert to FLOAT32

    // Convert to FLOAT32 tensor and unpad
    auto input_packed_data = owned_buffer::get_as<uint32_t>(tensor).get();
    auto input_float_data =
        unpack_bfp4_tiles_into_float_vec(input_packed_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
    auto input_float_buffer = owned_buffer::create<float>(std::move(input_float_data));
    auto float_tensor = Tensor(
                            OwnedStorage{input_float_buffer},
                            TensorSpec(
                                tensor.get_logical_shape(),
                                TensorLayout::fromPaddedShape(
                                    DataType::FLOAT32,
                                    PageConfig(tensor.get_layout(), tile),
                                    MemoryConfig{},
                                    tensor.get_logical_shape(),
                                    tensor.get_padded_shape())))
                            .unpad(output_tensor_start, output_tensor_end);

    // Convert back to BFLOAT4_B
    auto output_float_data = owned_buffer::get_as<float>(float_tensor).get();
    auto output_packed_data =
        pack_fp32_vec_as_bfp4_tiles(output_float_data, /*row_major_input=*/false, /*is_exp_a=*/false, tile);
    auto output_uint32_buffer = owned_buffer::create<uint32_t>(std::move(output_packed_data));
    return Tensor(
        std::move(OwnedStorage{std::move(output_uint32_buffer)}),
        TensorSpec(
            float_tensor.get_logical_shape(),
            TensorLayout::fromPaddedShape(
                DataType::BFLOAT4_B,
                PageConfig(tensor.get_layout(), tile),
                MemoryConfig{},
                float_tensor.get_logical_shape(),
                float_tensor.get_padded_shape())));
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
        if (this->max.has_value() and this->size >= this->max.value() and index == this->max.value() / 2) {
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

static constexpr auto TAB = "    ";
static constexpr auto TAB_MINUS_1 = "   ";

template <typename BufferType>
void to_string_row_major(
    std::stringstream& ss,
    const BufferType& buffer,
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
    ss << "[";
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
            to_string_row_major<BufferType>(
                ss, buffer, shape, strides, index, buffer_offset + index * stride, rank - 1, dim + 1);
        } else {
            print_datum(ss, buffer[buffer_offset + index]);
        }
        print_trailing_comma(ss, index, rank != 0 ? shape[-rank] : 1, after_comma);
    }
    ss << "]";
}

template <typename BufferType>
std::string to_string(
    const BufferType& buffer,
    const ttnn::Shape& shape,
    const tt::tt_metal::Strides& strides,
    DataType dtype,
    Layout layout) {
    std::stringstream ss;
    ss << TENSOR_TYPE_STRING << "(";

    if (TTNN_TENSOR_PRINT_PROFILE == TensorPrintProfile::Empty) {
        ss << "...";
    } else {
        to_string_row_major<BufferType>(ss, buffer, shape, strides, 0, 0, shape.rank());
    }
    ss << ", shape=" << fmt::format("{}", shape) << ", dtype=" << fmt::format("{}", dtype)
       << ", layout=" << fmt::format("{}", layout) << ")";
    return ss.str();
}

}  // namespace detail

template <typename T>
std::string to_string(
    const Tensor& tensor, std::optional<DataType> original_dtype, std::optional<Layout> original_layout) {
    const auto tile = tensor.get_tensor_spec().tile();
    const auto shape = tensor.get_logical_shape();
    const auto dtype = original_dtype.value_or(tensor.get_dtype());
    const auto layout = original_layout.value_or(tensor.get_layout());

    if (not tensor.is_allocated()) {
        return fmt::format(
            "{}(<buffer is not allocated>, shape={}, dtype={}, layout={})",
            detail::TENSOR_TYPE_STRING,
            shape,
            dtype,
            layout);
    }

    if (is_tensor_on_device(tensor)) {
        return to_string<T>(tensor.cpu(), dtype, layout);
    }

    return std::visit(
        [&](auto&& storage) -> std::string {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage> || std::is_same_v<StorageType, BorrowedStorage>) {
                if (tensor.get_layout() != Layout::ROW_MAJOR) {
                    if (tensor.get_dtype() == DataType::BFLOAT8_B || tensor.get_dtype() == DataType::BFLOAT4_B) {
                        return to_string<float>(ttnn::to_dtype(tensor, DataType::FLOAT32), dtype, layout);
                    }
                    return to_string<T>(
                        ttnn::to_layout(
                            tensor, Layout::ROW_MAJOR, std::nullopt, std::nullopt, static_cast<IDevice*>(nullptr)),
                        dtype,
                        layout);
                }

                const auto strides = tensor.get_tensor_spec().compute_strides();
                if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                    const auto buffer = owned_buffer::get_as<T>(storage.buffer);
                    return detail::to_string(buffer, shape, strides, dtype, layout);
                } else {
                    const auto buffer = borrowed_buffer::get_as<T>(storage.buffer);
                    return detail::to_string(buffer, shape, strides, dtype, layout);
                }
            } else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Cannot print a device tensor!");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                auto devices = get_devices(tensor);
                auto host_tensor = tensor.cpu();
                auto device_index = 0;
                std::stringstream ss;
                apply(host_tensor, [&](const Tensor& device_tensor) {
                    ss << "device_id:" << devices.at(device_index++)->id() << std::endl;
                    ss << to_string<T>(device_tensor) << std::endl;
                });
                return ss.str();
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                std::stringstream ss;
                auto device_tensors = ttnn::distributed::get_tensors_from_multi_device_storage(tensor);
                for (size_t i = 0; i < device_tensors.size(); i++) {
                    ss << to_string<T>(device_tensors[i]);
                    if (i + 1 != device_tensors.size()) {
                        ss << std::endl;
                    }
                }
                return ss.str();
            } else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor.get_storage());
}

template std::string to_string<bfloat16>(
    const Tensor& tensor, std::optional<DataType> original_dtype, std::optional<Layout> original_layout);
template std::string to_string<float>(
    const Tensor& tensor, std::optional<DataType> original_dtype, std::optional<Layout> original_layout);
template std::string to_string<int32_t>(
    const Tensor& tensor, std::optional<DataType> original_dtype, std::optional<Layout> original_layout);
template std::string to_string<uint32_t>(
    const Tensor& tensor, std::optional<DataType> original_dtype, std::optional<Layout> original_layout);
template std::string to_string<uint16_t>(
    const Tensor& tensor, std::optional<DataType> original_dtype, std::optional<Layout> original_layout);
template std::string to_string<uint8_t>(
    const Tensor& tensor, std::optional<DataType> original_dtype, std::optional<Layout> original_layout);

template <>
std::string to_string<bfloat8_b>(
    const Tensor& tensor, std::optional<DataType> original_dtype, std::optional<Layout> original_layout) {
    return to_string<uint32_t>(tensor, original_dtype);
}

template <>
std::string to_string<bfloat4_b>(
    const Tensor& tensor, std::optional<DataType> original_dtype, std::optional<Layout> original_layout) {
    return to_string<uint32_t>(tensor, original_dtype);
}

// ======================================================================================
//                                      .to_host()
// ======================================================================================

template <typename T>
Tensor to_host_helper(const Tensor& tensor, bool blocking = true, ttnn::QueueId cq_id = ttnn::DefaultQueueId) {
    TT_ASSERT(tensor.is_allocated(), "Buffer must be allocated on device!");
    auto device_buffer = tensor.device_buffer();
    auto device = tensor.device();
    TT_ASSERT(device != nullptr && "Need device to be set copy data from device to host!");
    uint32_t size_in_bytes = device_buffer->size();
    std::vector<T> data_vec;
    const char* TT_METAL_SLOW_DISPATCH_MODE = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (TT_METAL_SLOW_DISPATCH_MODE == nullptr) {
        data_vec.resize(size_in_bytes / sizeof(T));
        read_data_from_device_buffer<T>(device->command_queue(*cq_id), device_buffer, data_vec.data(), blocking);
    } else {
        read_data_from_device_buffer<T>(device_buffer, data_vec);
    }
    auto output_buffer = owned_buffer::create<T>(std::move(data_vec));
    return Tensor(OwnedStorage{output_buffer}, tensor.get_tensor_spec());
}

template <typename T>
Tensor to_host(const Tensor& tensor, bool blocking, ttnn::QueueId cq_id) {
    if (tensor.storage_type() == StorageType::DEVICE) {
        return to_host_helper<T>(tensor, blocking, cq_id);
    } else if (tensor.storage_type() == StorageType::MULTI_DEVICE) {
        auto devices = get_devices(tensor);
        Tensor host_tensor(devices.size());
        host_tensor.set_tensor_spec(tensor.get_tensor_spec());
        for (int device_index = 0; device_index < devices.size(); ++device_index) {
            const auto& device = devices[device_index];
            auto shard = get_shard_for_device(tensor, device);
            shard = to_host_helper<T>(shard, blocking, cq_id);
            insert_buffer_and_shape_for_device(device, shard, host_tensor, device_index);
        }
        return host_tensor;
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
Tensor to_host_mesh_tensor(const Tensor& tensor, bool blocking) {
    TT_FATAL(ttnn::distributed::is_mesh_buffer_tensor(tensor), "Tensor is not a mesh buffer tensor!");
    TT_FATAL(tt::tt_metal::detail::InMainThread(), "to_host_mesh_tensor must be called from the main thread");
    const auto& storage = std::get<MultiDeviceStorage>(tensor.get_storage());
    const auto& mesh_buffer = storage.mesh_buffer;
    ttnn::MeshDevice* device = mesh_buffer->device();
    distributed::MeshCommandQueue& mesh_cq = device->mesh_command_queue();
    const auto num_buffers = storage.buffers.size();

    std::vector<distributed::MeshCommandQueue::ShardDataTransfer> shard_data_transfers;
    std::vector<TensorSpec> specs;
    std::vector<OwnedBuffer> buffers;
    specs.reserve(num_buffers);
    buffers.reserve(num_buffers);
    shard_data_transfers.reserve(num_buffers);
    distributed::MeshCoordinateRange coord_range(device->shape());
    auto shard_coord = coord_range.begin();
    for (int id : storage.ordered_device_ids) {
        std::vector<T> host_buffer;
        const auto& shard_tensor_spec = storage.specs.at(id);
        const auto tensor_size_bytes = shard_tensor_spec.compute_packed_buffer_size_bytes();
        host_buffer.resize(tensor_size_bytes / sizeof(T));
        specs.push_back(shard_tensor_spec);
        buffers.push_back(owned_buffer::create<T>(std::move(host_buffer)));

        shard_data_transfers.push_back(distributed::MeshCommandQueue::ShardDataTransfer{
            .shard_coord = *shard_coord,
            .host_data = std::visit([](auto& b) { return reinterpret_cast<T*>(b.data()); }, buffers.back()),
            .region = BufferRegion(0, tensor_size_bytes)});
        ++shard_coord;
    }

    mesh_cq.enqueue_read_shards(shard_data_transfers, mesh_buffer, /*blocking=*/true);

    MultiDeviceHostStorage host_storage(storage.strategy, std::move(buffers), std::move(specs));
    return Tensor(std::move(host_storage), tensor.get_tensor_spec());
}

template Tensor to_host_mesh_tensor<bfloat16>(const Tensor& tensor, bool blocking);
template Tensor to_host_mesh_tensor<float>(const Tensor& tensor, bool blocking);
template Tensor to_host_mesh_tensor<int32_t>(const Tensor& tensor, bool blocking);
template Tensor to_host_mesh_tensor<uint32_t>(const Tensor& tensor, bool blocking);
template Tensor to_host_mesh_tensor<uint16_t>(const Tensor& tensor, bool blocking);
template Tensor to_host_mesh_tensor<uint8_t>(const Tensor& tensor, bool blocking);

template <>
Tensor to_host_mesh_tensor<bfloat4_b>(const Tensor& tensor, bool blocking) {
    return to_host_mesh_tensor<uint32_t>(tensor, blocking);
}

template <>
Tensor to_host_mesh_tensor<bfloat8_b>(const Tensor& tensor, bool blocking) {
    return to_host_mesh_tensor<uint32_t>(tensor, blocking);
}

// ======================================================================================
//                               .to_device() details
// ======================================================================================

template <typename T, template <typename> typename BufferType>
void write_data_to_device_buffer(
    CommandQueue& cq, const BufferType<T>& host_buffer, std::shared_ptr<Buffer> device_buffer) {
    ZoneScoped;
    // TODO(arakhmati): can we use generators in this function to go from `data_to_write` to `uint32_data`?
    // And effectively get rid of any additional allocation
    EnqueueWriteBuffer(cq, device_buffer, host_buffer.data(), false);
}

template <typename T, template <typename> typename BufferType>
void write_data_to_device_buffer(const BufferType<T>& host_buffer, Buffer& device_buffer) {
    ZoneScoped;
    ::detail::WriteToBuffer(
        device_buffer,
        tt::stl::Span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(host_buffer.data()), host_buffer.size() * sizeof(T)));
}

template <typename T, template <typename> typename BufferType>
std::shared_ptr<Buffer> initialize_data_on_device(
    BufferType<T>& data_to_write,
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
            [&device, &tensor_spec, cq_id]<OwnedOrBorrowedStorage StorageType>(const StorageType& storage) {
                auto data_to_write = host_buffer::get_as<T>(storage.buffer);
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
    TT_FATAL(tensor.storage_type() != StorageType::DEVICE, "Tensor is already on device!");
    TT_FATAL(target_device != nullptr, "Need target device in order to move tensor to device!");
    TT_FATAL(tensor.is_allocated(), "Need data to exist in order to move it to device");

    TensorSpec tensor_spec(
        tensor.get_logical_shape(), tensor.get_tensor_spec().tensor_layout().with_memory_config(memory_config));
    auto device_buffer = tensor_impl::to_device_buffer<T>(tensor.get_storage(), target_device, tensor_spec, cq_id);
    return Tensor(DeviceStorage{device_buffer}, tensor_spec);
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

template <typename T, OwnedOrBorrowedStorage StorageType>
MultiDeviceStorage replicate_to_mesh_buffer(
    const StorageType& storage,
    distributed::MeshDevice* mesh_device,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    const TensorSpec& tensor_spec) {
    auto data_to_write = host_buffer::get_as<T>(storage.buffer);
    const auto expected_packed_buffer_size_bytes = tensor_spec.compute_packed_buffer_size_bytes();
    const auto input_size_bytes = data_to_write.size() * sizeof(T);
    TT_FATAL(
        input_size_bytes == expected_packed_buffer_size_bytes,
        "Host data with total size {}B does not match expected size {}B of device buffer!",
        input_size_bytes,
        expected_packed_buffer_size_bytes);

    mesh_device->mesh_command_queue().enqueue_write_mesh_buffer(mesh_buffer, data_to_write.data(), /*blocking=*/false);
    return MultiDeviceStorage(mesh_buffer, tensor_spec);
}

template <typename T>
MultiDeviceStorage shard_to_mesh_buffer(
    const MultiDeviceHostStorage& storage,
    distributed::MeshDevice* mesh_device,
    const std::shared_ptr<distributed::MeshBuffer>& mesh_buffer,
    const TensorSpec& tensor_spec) {
    std::vector<int> ordered_device_ids;
    std::unordered_map<int, std::shared_ptr<Buffer>> buffers;
    std::unordered_map<int, TensorSpec> specs;
    ordered_device_ids.reserve(storage.buffers.size());
    buffers.reserve(storage.buffers.size());
    specs.reserve(storage.buffers.size());

    const auto& mesh_shape = mesh_device->shape();
    TT_FATAL(
        storage.buffers.size() <= mesh_device->num_devices(),
        "Number of host buffers {} exceeds the number of shards {}",
        storage.buffers.size(),
        mesh_device->num_devices());

    std::vector<distributed::MeshCommandQueue::ShardDataTransfer> shard_data_transfers;
    shard_data_transfers.reserve(storage.buffers.size());

    distributed::MeshCoordinateRange coord_range(mesh_shape);
    auto shard_coord = coord_range.begin();
    for (int i = 0; i < storage.buffers.size(); ++shard_coord, i++) {
        TensorSpec shard_tensor_spec(
            storage.specs[i].logical_shape(),
            storage.specs[i].tensor_layout().with_memory_config(tensor_spec.memory_config()));
        const auto& shard_host_buffer = storage.buffers[i];

        const auto& shard_buffer = mesh_buffer->get_device_buffer(*shard_coord);
        ordered_device_ids.push_back(shard_buffer->device()->id());
        buffers.insert({shard_buffer->device()->id(), shard_buffer});
        specs.insert({shard_buffer->device()->id(), shard_tensor_spec});

        auto data_to_write = host_buffer::get_as<T>(shard_host_buffer);
        const auto expected_packed_buffer_size_bytes = shard_tensor_spec.compute_packed_buffer_size_bytes();
        const auto input_size_bytes = data_to_write.size() * sizeof(T);
        TT_FATAL(
            input_size_bytes == expected_packed_buffer_size_bytes,
            "Host data with total size {}B does not match expected size {}B of device buffer!",
            input_size_bytes,
            expected_packed_buffer_size_bytes);
        TT_FATAL(
            expected_packed_buffer_size_bytes <= tensor_spec.compute_packed_buffer_size_bytes(),
            "Shard tensor size exceeds the global tensor size!");
        shard_data_transfers.push_back(distributed::MeshCommandQueue::ShardDataTransfer{
            .shard_coord = *shard_coord,
            .host_data = data_to_write.data(),
            .region = BufferRegion(0, input_size_bytes)});
    }

    mesh_device->mesh_command_queue().enqueue_write_shards(mesh_buffer, shard_data_transfers, /*blocking=*/false);

    return MultiDeviceStorage(
        storage.strategy, std::move(ordered_device_ids), std::move(buffers), std::move(specs), mesh_buffer);
}

template <typename T>
Tensor to_device_mesh_tensor(
    const Tensor& tensor, distributed::MeshDevice* mesh_device, const MemoryConfig& memory_config) {
    TT_FATAL(tt::tt_metal::detail::InMainThread(), "to_device_mesh_tensor must be called from the main thread");
    TT_FATAL(tensor.storage_type() != StorageType::MULTI_DEVICE, "Tensor is already on device!");
    TT_FATAL(mesh_device != nullptr, "Need target device in order to move tensor to device!");
    TT_FATAL(tensor.is_allocated(), "Need data to exist in order to move it to device");

    TensorSpec tensor_spec(
        tensor.get_logical_shape(), tensor.get_tensor_spec().tensor_layout().with_memory_config(memory_config));

    auto mesh_buffer = allocate_mesh_buffer_on_device(mesh_device, tensor_spec);
    MultiDeviceStorage mesh_storage = std::visit(
        tt::stl::overloaded{
            [&mesh_device, &mesh_buffer, &tensor_spec]<OwnedOrBorrowedStorage StorageType>(const StorageType& storage) {
                // Replicate data across devices in a mesh.
                return replicate_to_mesh_buffer<T>(storage, mesh_device, mesh_buffer, tensor_spec);
            },
            [&mesh_device, &mesh_buffer, &tensor_spec](const MultiDeviceHostStorage& storage) {
                // Shard multi device host shards across devices in a mesh..
                return shard_to_mesh_buffer<T>(storage, mesh_device, mesh_buffer, tensor_spec);
            },
            [](const auto& s) -> MultiDeviceStorage {
                TT_THROW("Unexpected storage type {}", tt::stl::get_type_name(s));
            }},
        tensor.get_storage());

    return Tensor(std::move(mesh_storage), tensor_spec);
}

template Tensor to_device_mesh_tensor<bfloat16>(
    const Tensor& tensor, distributed::MeshDevice* target_device, const MemoryConfig& memory_config);
template Tensor to_device_mesh_tensor<float>(
    const Tensor& tensor, distributed::MeshDevice* target_device, const MemoryConfig& memory_config);
template Tensor to_device_mesh_tensor<int32_t>(
    const Tensor& tensor, distributed::MeshDevice* target_device, const MemoryConfig& memory_config);
template Tensor to_device_mesh_tensor<uint32_t>(
    const Tensor& tensor, distributed::MeshDevice* target_device, const MemoryConfig& memory_config);
template Tensor to_device_mesh_tensor<uint16_t>(
    const Tensor& tensor, distributed::MeshDevice* target_device, const MemoryConfig& memory_config);
template Tensor to_device_mesh_tensor<uint8_t>(
    const Tensor& tensor, distributed::MeshDevice* target_device, const MemoryConfig& memory_config);

template <>
Tensor to_device_mesh_tensor<bfloat4_b>(
    const Tensor& tensor, distributed::MeshDevice* target_device, const MemoryConfig& memory_config) {
    return to_device_mesh_tensor<uint32_t>(tensor, target_device, memory_config);
}

template <>
Tensor to_device_mesh_tensor<bfloat8_b>(
    const Tensor& tensor, distributed::MeshDevice* target_device, const MemoryConfig& memory_config) {
    return to_device_mesh_tensor<uint32_t>(tensor, target_device, memory_config);
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
    if (tensor_spec.memory_config().is_sharded() and
        (tensor_spec.memory_config().shard_spec.value().mode == ShardMode::LOGICAL or logical_shape == padded_shape)) {
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
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

template <typename T>
std::vector<T> encode_tensor_data(std::vector<T>&& logical_data, const TensorSpec& tensor_spec) {
    if (logical_data.size() == 0) {
        return {};
    }

    const auto& logical_shape = tensor_spec.logical_shape();
    TT_FATAL(
        logical_data.size() == logical_shape.volume(),
        "Logical data size {} should be same as volume indicated by logical shape {}",
        logical_data.size(),
        logical_shape);

    const auto& physical_shape = tensor_spec.physical_shape();

    auto row_major_physical_data = [&tensor_spec, &physical_shape](auto&& logical_data) {
        const auto& logical_2d_shape = tensor_spec.logical_2d_shape();

        if (logical_2d_shape != physical_shape) {
            auto [logical_shard_shape, physical_shard_shape] =
                CMAKE_UNIQUE_NAMESPACE::get_logical_and_physical_shard_shapes(tensor_spec);

            auto row_major_physical_data = std::vector<T>(physical_shape.height() * physical_shape.width(), 0);

            size_t physical_stride = physical_shape.width();

            const auto logical_physical_mapping = CMAKE_UNIQUE_NAMESPACE::compute_logical_to_physical_shards_mapping(
                logical_2d_shape, logical_shard_shape, physical_shard_shape, physical_stride);

            for (const auto& [indices, cols] : logical_physical_mapping) {
                for (const auto [logical_idx_start, physical_idx_start] : indices) {
                    for (size_t col = 0; col < cols; col++) {
                        row_major_physical_data[physical_idx_start + col] = logical_data[logical_idx_start + col];
                    }
                }
            }
            return row_major_physical_data;
        } else {
            return logical_data;
        }
    }(std::move(logical_data));

    TT_FATAL(
        row_major_physical_data.size() == physical_shape.height() * physical_shape.width(),
        "Physical data size {} should be same as volume indicated by physical shape {}",
        row_major_physical_data.size(),
        physical_shape);

    if (tensor_spec.layout() == Layout::TILE) {
        return tensor_impl::convert_layout_row_major_to_tile(
            physical_shape, tensor_spec.tile(), row_major_physical_data);
    }
    return row_major_physical_data;
}

template std::vector<bfloat16> encode_tensor_data<bfloat16>(
    std::vector<bfloat16>&& logical_data, const TensorSpec& tensor_spec);
template std::vector<float> encode_tensor_data<float>(std::vector<float>&& logical_data, const TensorSpec& tensor_spec);
template std::vector<int32_t> encode_tensor_data<int32_t>(
    std::vector<int32_t>&& logical_data, const TensorSpec& tensor_spec);
template std::vector<uint32_t> encode_tensor_data<uint32_t>(
    std::vector<uint32_t>&& logical_data, const TensorSpec& tensor_spec);
template std::vector<uint16_t> encode_tensor_data<uint16_t>(
    std::vector<uint16_t>&& logical_data, const TensorSpec& tensor_spec);
template std::vector<uint8_t> encode_tensor_data<uint8_t>(
    std::vector<uint8_t>&& logical_data, const TensorSpec& tensor_spec);

template <typename T>
std::vector<T> decode_tensor_data(std::vector<T>&& physical_data, const TensorSpec& tensor_spec) {
    if (physical_data.size() == 0) {
        return {};
    }

    const auto& physical_shape = tensor_spec.physical_shape();
    TT_FATAL(
        physical_data.size() == physical_shape.height() * physical_shape.width(),
        "Physical data size {} should be same as volume indicated by physical shape {}",
        physical_data.size(),
        physical_shape);

    auto row_major_physical_data = [&tensor_spec, &physical_shape](auto&& physical_data) {
        if (tensor_spec.layout() == Layout::TILE) {
            return tensor_impl::convert_layout_tile_to_row_major(physical_shape, tensor_spec.tile(), physical_data);
        } else {
            return physical_data;
        }
    }(std::move(physical_data));

    auto logical_data = [&tensor_spec, &physical_shape](auto&& row_major_physical_data) {
        const auto& logical_2d_shape = tensor_spec.logical_2d_shape();

        if (logical_2d_shape != physical_shape) {
            auto [logical_shard_shape, physical_shard_shape] =
                CMAKE_UNIQUE_NAMESPACE::get_logical_and_physical_shard_shapes(tensor_spec);

            auto logical_data = std::vector<T>(logical_2d_shape.height() * logical_2d_shape.width(), 0);

            size_t physical_stride = physical_shape.width();

            const auto logical_physical_mapping = CMAKE_UNIQUE_NAMESPACE::compute_logical_to_physical_shards_mapping(
                logical_2d_shape, logical_shard_shape, physical_shard_shape, physical_stride);

            for (const auto& [indices, cols] : logical_physical_mapping) {
                for (const auto [logical_idx_start, physical_idx_start] : indices) {
                    for (size_t col = 0; col < cols; col++) {
                        logical_data[logical_idx_start + col] = row_major_physical_data[physical_idx_start + col];
                    }
                }
            }
            return logical_data;
        } else {
            return row_major_physical_data;
        }
    }(std::move(row_major_physical_data));

    const auto& logical_shape = tensor_spec.logical_shape();
    TT_FATAL(
        logical_data.size() == logical_shape.volume(),
        "Logical data size {} should be same as volume indicated by logical shape {}",
        logical_data.size(),
        logical_shape);

    return logical_data;
}

template std::vector<bfloat16> decode_tensor_data<bfloat16>(
    std::vector<bfloat16>&& physical_data, const TensorSpec& tensor_spec);
template std::vector<float> decode_tensor_data<float>(
    std::vector<float>&& physical_data, const TensorSpec& tensor_spec);
template std::vector<int32_t> decode_tensor_data<int32_t>(
    std::vector<int32_t>&& physical_data, const TensorSpec& tensor_spec);
template std::vector<uint32_t> decode_tensor_data<uint32_t>(
    std::vector<uint32_t>&& physical_data, const TensorSpec& tensor_spec);
template std::vector<uint16_t> decode_tensor_data<uint16_t>(
    std::vector<uint16_t>&& physical_data, const TensorSpec& tensor_spec);
template std::vector<uint8_t> decode_tensor_data<uint8_t>(
    std::vector<uint8_t>&& physical_data, const TensorSpec& tensor_spec);

// ======================================================================================
//                                  .to_layout()
// ======================================================================================

template <typename T>
Tensor to_layout(const Tensor& tensor, Layout target_layout) {
    if (tensor.get_layout() == target_layout) {
        return tensor;
    }

    auto source_layout = tensor.get_layout();
    auto tile = tensor.tensor_spec().tile();
    auto physical_shape = tensor.tensor_spec().physical_shape();
    auto convert = [tile, &physical_shape, source_layout, target_layout](const auto& input_data) -> std::vector<T> {
        switch (source_layout) {
            case Layout::ROW_MAJOR:
                if (target_layout == Layout::TILE) {
                    return convert_layout_row_major_to_tile(physical_shape, tile, input_data);
                } else {
                    TT_THROW("Unsupported layout conversion");
                }
                break;
            case Layout::TILE:
                if (target_layout == Layout::ROW_MAJOR) {
                    return convert_layout_tile_to_row_major(physical_shape, tile, input_data);
                } else {
                    TT_THROW("Unsupported layout conversion");
                }
                break;
            default: TT_THROW("Unsupported layout conversion");
        }
    };

    using RetType = std::variant<OwnedStorage, MultiDeviceHostStorage>;
    auto output_storage = std::visit(
        tt::stl::overloaded{
            [&convert, target_layout](const OwnedStorage& storage) -> RetType {
                const auto input_data = owned_buffer::get_as<T>(storage.buffer);
                auto output_buffer = owned_buffer::create<T>(std::move(convert(input_data)));
                return OwnedStorage{output_buffer};
            },
            [&convert, target_layout](const BorrowedStorage& storage) -> RetType {
                const auto input_data = borrowed_buffer::get_as<T>(storage.buffer);
                auto output_buffer = owned_buffer::create<T>(std::move(convert(input_data)));
                return OwnedStorage{output_buffer};
            },
            [&convert, target_layout](const MultiDeviceHostStorage& storage) -> RetType {
                std::vector<OwnedBuffer> output_buffers;
                std::vector<ttnn::TensorSpec> output_specs;
                for (int i = 0; i < storage.num_buffers(); i++) {
                    const auto input_data = owned_buffer::get_as<T>(storage.get_buffer(i));
                    auto output_buffer = owned_buffer::create<T>(std::move(convert(input_data)));
                    output_buffers.push_back(output_buffer);
                    const auto& prev_spec = storage.specs[i];
                    output_specs.push_back(TensorSpec(
                        prev_spec.logical_shape(),
                        TensorLayout::fromPaddedShape(
                            prev_spec.data_type(),
                            PageConfig(target_layout, prev_spec.tile()),
                            MemoryConfig{},
                            prev_spec.logical_shape(),
                            prev_spec.padded_shape())));
                }
                return MultiDeviceHostStorage{storage.strategy, output_buffers, output_specs};
            },
            [](const auto& s) -> RetType { TT_THROW("Unsupported storage type {}", tt::stl::get_type_name(s)); }},
        tensor.get_storage());

    return std::visit(
        [&tensor, &target_layout](auto&& storage) -> Tensor {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (
                !std::is_same_v<StorageType, OwnedStorage> && !std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                raise_unsupported_storage<StorageType>();
            }
            return Tensor(
                storage,
                TensorSpec(
                    tensor.get_logical_shape(),
                    TensorLayout::fromPaddedShape(
                        tensor.get_dtype(),
                        PageConfig(target_layout, tensor.get_tensor_spec().tile()),
                        MemoryConfig{},
                        tensor.get_logical_shape(),
                        tensor.get_padded_shape())));
        },
        output_storage);
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
    if (tensor.get_layout() != target_layout or tensor.get_layout() != Layout::TILE) {
        log_warning(
            tt::LogAlways,
            "Tensor layout must be Layout::TILE for bfloat8_b or bfloat4_b! Conversion from {} to {} was not executed!",
            tensor.get_layout(),
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
    if (ttnn::distributed::is_multi_device_tensor(tensor)) {
        return transform(tensor, [&](const Tensor& device_tensor) {
            return pad<T>(device_tensor, output_padded_shape, input_tensor_start, pad_value);
        });
    }

    auto pad_value_ = static_cast<T>(pad_value);
    auto input_padded_shape = tensor.get_padded_shape();
    if (input_padded_shape.rank() < 2) {
        input_padded_shape = input_padded_shape.to_rank(2);
    }
    const auto input_strides = tensor.strides();
    const auto input_data_type = tensor.get_dtype();

    auto pad =
        [&input_padded_shape, &output_padded_shape, &input_tensor_start, &pad_value_](const auto& input_buffer) {
            auto compute_stride = [](const ttnn::Shape& padded_shape, uint32_t index) {
                uint32_t stride = 1;
                for (auto i = index + 1; i < padded_shape.rank(); i++) {
                    stride *= padded_shape[i];
                }
                return stride;
            };

            ttnn::SmallVector<std::array<uint32_t, 2>> pad_size{};
            ttnn::SmallVector<uint32_t> input_strides{};
            ttnn::SmallVector<uint32_t> output_strides{};
            ttnn::SmallVector<uint32_t> input_indices(input_padded_shape.rank(), 0);

            for (auto index = 0; index < output_padded_shape.rank(); index++) {
                // Check if input tensor fits in output tensor given the input tensor start indices
                TT_ASSERT(
                    input_padded_shape[index] + input_tensor_start[index] <= output_padded_shape[index],
                    "Input tensor is out of bounds");

                // Figure out pad size on each dim
                pad_size.push_back(
                    {input_tensor_start[index],
                     output_padded_shape[index] - input_padded_shape[index] - input_tensor_start[index]});

                input_strides.push_back(compute_stride(input_padded_shape, index));
                output_strides.push_back(compute_stride(output_padded_shape, index));
            }

            auto flat_output_index = 0;
            auto output_buffer = owned_buffer::create<T>(output_padded_shape.volume());
            std::function<void(std::size_t)> pad_to_tile = [&](std::size_t dim) -> void {
                for (auto i = 0; i < pad_size[dim][0] * output_strides[dim]; i++) {
                    output_buffer[flat_output_index++] = pad_value_;
                }

                for (auto i = 0; i < input_padded_shape[dim]; i++) {
                    input_indices[dim] = i;
                    if (dim == input_padded_shape.rank() - 1) {
                        auto flat_input_index = compute_flat_input_index(input_indices, input_strides);
                        output_buffer[flat_output_index++] = input_buffer[flat_input_index];
                    } else {
                        pad_to_tile(dim + 1);
                    }
                }

                for (auto i = 0; i < pad_size[dim][1] * output_strides[dim]; i++) {
                    output_buffer[flat_output_index++] = pad_value_;
                }
            };
            pad_to_tile(0);

            return output_buffer;
        };

    auto output_buffer = std::visit(
        tt::stl::overloaded{
            [&pad]<OwnedOrBorrowedStorage StorageType>(const StorageType& storage) {
                const auto input_data = host_buffer::get_as<T>(storage.buffer);
                return pad(input_data);
            },
            [](const auto& s) -> owned_buffer::Buffer<T> {
                TT_THROW("Unsupported storage type {}", tt::stl::get_type_name(s));
            }},
        tensor.get_storage());
    return Tensor(
        OwnedStorage{output_buffer},
        tensor.get_logical_shape(),
        output_padded_shape,
        tensor.get_dtype(),
        tensor.get_layout(),
        tensor.get_tensor_spec().tile());
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
    const auto input_shape = tensor.get_padded_shape();
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
                     const auto& input_buffer) {
        ttnn::SmallVector<uint32_t> input_indices(input_shape.rank(), 0);

        auto flat_output_index = 0;
        auto output_buffer = owned_buffer::create<T>(ttnn::Shape(output_shape).volume());

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

    auto output_buffer = std::visit(
        tt::stl::overloaded{
            [&unpad]<OwnedOrBorrowedStorage StorageType>(const StorageType& storage) {
                const auto input_data = host_buffer::get_as<T>(storage.buffer);
                return unpad(input_data);
            },
            [](const auto& s) -> owned_buffer::Buffer<T> {
                TT_THROW("Unsupported storage type {}", tt::stl::get_type_name(s));
            }},
        tensor.get_storage());
    return Tensor(
        OwnedStorage{output_buffer},
        ttnn::Shape(output_shape),
        tensor.get_dtype(),
        tensor.get_layout(),
        tensor.get_tensor_spec().tile());
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

    auto output_buffer = owned_buffer::create<T>(std::move(device_data));
    return Tensor(
        OwnedStorage{output_buffer},
        shard_shape,
        tensor.get_dtype(),
        tensor.get_layout(),
        tensor.get_tensor_spec().tile());
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
