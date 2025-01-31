// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/serialization.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>

#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/distributed/types.hpp"

namespace tt::tt_metal {

using MeshDevice = distributed::MeshDevice;

namespace {

struct Padding {
    enum class PadValue { Any, Zero, Infinity, NegativeInfinity };
    struct PadDimension {
        std::size_t front;
        std::size_t back;
    };
    std::size_t rank_;
    std::array<PadDimension, MAX_NUM_DIMENSIONS> pad_dimensions_;
    PadValue pad_value_ = PadValue::Any;
};

struct LegacyShape {
    std::size_t rank_;
    std::array<uint32_t, MAX_NUM_DIMENSIONS> dimensions_;
    Padding padding_;

    LegacyShape() = default;
    LegacyShape(const ttnn::Shape& logical_shape, const ttnn::Shape& padded_shape) {
        rank_ = padded_shape.rank();
        padding_.rank_ = padded_shape.rank();
        for (int index = 0; index < padded_shape.rank(); index++) {
            int shape_index = index + static_cast<int>(logical_shape.size()) - static_cast<int>(padded_shape.size());
            int dimension = shape_index >= 0 ? logical_shape[shape_index] : 1;
            int padded_dimension = padded_shape[index];
            this->dimensions_[index] = padded_dimension;
            this->padding_.pad_dimensions_[index] = {
                .front = 0, .back = static_cast<size_t>(padded_dimension - dimension)};
        }
    }

    ttnn::Shape logical_shape() const {
        ttnn::SmallVector<uint32_t> values(rank_);
        for (size_t i = 0; i < values.size(); i++) {
            auto [front_pad, back_pad] = padding_.pad_dimensions_[i];
            values[i] = dimensions_[i] - front_pad - back_pad;
        }
        return ttnn::Shape(std::move(values));
    }

    ttnn::Shape padded_shape() const {
        ttnn::SmallVector<uint32_t> values(rank_);
        for (size_t i = 0; i < values.size(); i++) {
            values[i] = dimensions_[i];
        }
        return ttnn::Shape(std::move(values));
    }
};

static constexpr std::size_t SENTINEL_VALUE = std::numeric_limits<std::size_t>::max();

void dump_owned_storage(std::ofstream& output_stream, const OwnedStorage& storage) {
    std::visit(
        [&output_stream]<typename T>(const owned_buffer::Buffer<T>& generic_buffer) {
            const auto buffer = owned_buffer::get_as<T>(generic_buffer);
            auto size = buffer.size();
            output_stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
            output_stream.write(reinterpret_cast<const char*>(buffer.begin()), sizeof(T) * size);
        },
        storage.buffer);
}

void dump_borrowed_storage(std::ofstream& output_stream, const BorrowedStorage& storage) {
    std::visit(
        [&output_stream]<typename T>(const borrowed_buffer::Buffer<T>& generic_buffer) {
            const auto buffer = borrowed_buffer::get_as<T>(generic_buffer);
            auto size = buffer.size();
            output_stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
            output_stream.write(reinterpret_cast<const char*>(buffer.begin()), sizeof(T) * size);
        },
        storage.buffer);
}

void dump_multi_device_host_storage(
    std::ofstream& output_stream, const MultiDeviceHostStorage& storage, const DistributedTensorConfig& strategy) {
    std::size_t num_buffers = storage.num_buffers();
    output_stream.write(reinterpret_cast<const char*>(&num_buffers), sizeof(std::size_t));

    // Use the user-specified strategy which defines how it gets distributed when mapped onto multi-device
    output_stream.write(reinterpret_cast<const char*>(&strategy), sizeof(DistributedTensorConfig));

    if (std::holds_alternative<ReplicateTensor>(strategy)) {
        std::visit(
            [&output_stream]<typename T>(const owned_buffer::Buffer<T>& generic_buffer) {
                const auto buffer = owned_buffer::get_as<T>(generic_buffer);
                auto size = buffer.size();
                output_stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
                output_stream.write(reinterpret_cast<const char*>(buffer.begin()), sizeof(T) * size);
            },
            storage.get_buffer(0));
        auto spec = storage.specs.at(0);
        LegacyShape shape(spec.logical_shape(), spec.padded_shape());
        output_stream.write(reinterpret_cast<const char*>(&shape), sizeof(LegacyShape));
    } else {
        for (int i = 0; i < num_buffers; i++) {
            std::visit(
                [&output_stream]<typename T>(const owned_buffer::Buffer<T>& generic_buffer) {
                    const auto buffer = owned_buffer::get_as<T>(generic_buffer);
                    auto size = buffer.size();
                    output_stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
                    output_stream.write(reinterpret_cast<const char*>(buffer.begin()), sizeof(T) * size);
                },
                storage.get_buffer(i));
        }
        for (const auto& spec : storage.specs) {
            LegacyShape shape(spec.logical_shape(), spec.padded_shape());
            output_stream.write(reinterpret_cast<const char*>(&shape), sizeof(LegacyShape));
        }
    }
}

template <typename T>
OwnedStorage load_owned_storage(std::ifstream& input_stream) {
    std::size_t size = 0;
    input_stream.read(reinterpret_cast<char*>(&size), sizeof(std::size_t));
    auto buffer = owned_buffer::create<T>(size);
    input_stream.read(reinterpret_cast<char*>(buffer.begin()), sizeof(T) * size);
    return {buffer};
}

template <typename T>
MultiDeviceHostStorage load_multi_device_host_storage(
    std::ifstream& input_stream, DataType data_type, Layout layout, MeshDevice* mesh_device) {
    std::size_t num_buffers = 0;
    DistributedTensorConfig strategy;
    input_stream.read(reinterpret_cast<char*>(&num_buffers), sizeof(std::size_t));
    input_stream.read(reinterpret_cast<char*>(&strategy), sizeof(DistributedTensorConfig));

    std::vector<OwnedBuffer> buffers;
    std::vector<ttnn::TensorSpec> specs;
    if (std::holds_alternative<ReplicateTensor>(strategy)) {
        std::size_t size = 0;
        input_stream.read(reinterpret_cast<char*>(&size), sizeof(std::size_t));
        auto buffer = owned_buffer::create<T>(size);
        auto shape = LegacyShape{};
        input_stream.read(reinterpret_cast<char*>(buffer.begin()), sizeof(T) * size);
        input_stream.read(reinterpret_cast<char*>(&shape), sizeof(LegacyShape));
        buffers.push_back(buffer);
        TensorSpec spec(
            shape.logical_shape(),
            TensorLayout::fromPaddedShape(
                data_type, PageConfig(layout), MemoryConfig{}, shape.logical_shape(), shape.padded_shape()));
        specs.push_back(spec);

        for (std::size_t i = 1; i < mesh_device->num_devices(); ++i) {
            buffers.push_back(owned_buffer::Buffer<T>{buffer.get_ptr()});
            specs.push_back(spec);
        }

    } else {
        for (std::size_t i = 0; i < num_buffers; ++i) {
            std::size_t size = 0;
            input_stream.read(reinterpret_cast<char*>(&size), sizeof(std::size_t));

            auto buffer = owned_buffer::create<T>(size);
            input_stream.read(reinterpret_cast<char*>(buffer.begin()), sizeof(T) * size);

            buffers.push_back(std::move(buffer));
        }
        for (std::size_t i = 0; i < num_buffers; ++i) {
            auto shape = LegacyShape{};
            input_stream.read(reinterpret_cast<char*>(&shape), sizeof(LegacyShape));
            TensorSpec spec(
                shape.logical_shape(),
                TensorLayout::fromPaddedShape(
                    data_type, PageConfig(layout), MemoryConfig{}, shape.logical_shape(), shape.padded_shape()));
            specs.push_back(spec);
        }
    }

    return {strategy, buffers, specs};
}

OwnedStorage load_owned_storage(std::ifstream& input_stream, DataType data_type) {
    if (data_type == DataType::UINT32 or data_type == DataType::BFLOAT8_B or data_type == DataType::BFLOAT4_B) {
        using T = std::uint32_t;
        return load_owned_storage<T>(input_stream);
    } else if (data_type == DataType::INT32) {
        using T = std::int32_t;
        return load_owned_storage<T>(input_stream);
    } else if (data_type == DataType::UINT8) {
        using T = std::uint8_t;
        return load_owned_storage<T>(input_stream);
    } else if (data_type == DataType::UINT16) {
        using T = std::uint16_t;
        return load_owned_storage<T>(input_stream);
    } else if (data_type == DataType::FLOAT32) {
        using T = float;
        return load_owned_storage<T>(input_stream);
    } else if (data_type == DataType::BFLOAT16) {
        using T = bfloat16;
        return load_owned_storage<T>(input_stream);
    } else {
        TT_THROW("Unsupported DataType");
    }
}

MultiDeviceHostStorage load_multi_device_host_storage(
    std::ifstream& input_stream, DataType data_type, Layout layout, MeshDevice* mesh_device) {
    if (data_type == DataType::UINT32 or data_type == DataType::BFLOAT8_B or data_type == DataType::BFLOAT4_B) {
        using T = std::uint32_t;
        return load_multi_device_host_storage<T>(input_stream, data_type, layout, mesh_device);
    } else if (data_type == DataType::UINT16) {
        using T = std::uint16_t;
        return load_multi_device_host_storage<T>(input_stream, data_type, layout, mesh_device);
    } else if (data_type == DataType::FLOAT32) {
        using T = float;
        return load_multi_device_host_storage<T>(input_stream, data_type, layout, mesh_device);
    } else if (data_type == DataType::BFLOAT16) {
        using T = bfloat16;
        return load_multi_device_host_storage<T>(input_stream, data_type, layout, mesh_device);
    } else {
        TT_THROW("Unsupported DataType");
    }
}

template <typename T>
Storage load_storage(
    std::ifstream& input_stream, DataType data_type, Layout layout, StorageType storage_type, T device) {
    if (storage_type == StorageType::MULTI_DEVICE_HOST or storage_type == StorageType::MULTI_DEVICE) {
        if constexpr (std::is_same_v<T, MeshDevice*>) {
            return load_multi_device_host_storage(input_stream, data_type, layout, device);
        } else {
            TT_THROW("MeshDevice is required for MULTI_DEVICE_HOST storage");
        }
    } else {
        return load_owned_storage(input_stream, data_type);
    }
}

}  // namespace

void dump_tensor(
    const std::string& file_name, const Tensor& tensor, const std::unordered_map<std::string, std::string>& strategy) {
    std::ofstream output_stream(file_name, std::ios::out | std::ios::binary);
    if (not output_stream) {
        throw std::runtime_error(fmt::format("Cannot open \"{}\"", file_name));
    }

    LegacyShape shape(tensor.get_logical_shape(), tensor.get_padded_shape());
    auto data_type = tensor.get_dtype();
    auto layout = tensor.get_layout();
    auto storage_type = tensor.storage_type();

    output_stream.write(reinterpret_cast<const char*>(&SENTINEL_VALUE), sizeof(std::size_t));
    output_stream.write(reinterpret_cast<const char*>(&VERSION_ID), sizeof(std::uint8_t));
    output_stream.write(reinterpret_cast<const char*>(&shape), sizeof(LegacyShape));
    output_stream.write(reinterpret_cast<const char*>(&data_type), sizeof(DataType));
    output_stream.write(reinterpret_cast<const char*>(&layout), sizeof(Layout));
    output_stream.write(reinterpret_cast<const char*>(&storage_type), sizeof(StorageType));

    bool is_on_device = is_tensor_on_device_or_multidevice(tensor);
    bool has_memory_config = is_on_device;
    if (VERSION_ID >= 2) {
        output_stream.write(reinterpret_cast<const char*>(&has_memory_config), sizeof(bool));
        if (has_memory_config) {
            tt::tt_metal::dump_memory_config(output_stream, tensor.memory_config());
        }
    }

    Tensor tensor_to_dump = tensor;
    if (is_on_device) {
        tensor_to_dump = tensor_to_dump.cpu();
    }

    std::visit(
        [&output_stream, &strategy](const auto& storage) {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                dump_owned_storage(output_stream, storage);
            } else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                dump_borrowed_storage(output_stream, storage);
            } else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                auto distribute_config = get_distributed_tensor_config(strategy);
                dump_multi_device_host_storage(output_stream, storage, distribute_config);
            } else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor_to_dump.get_storage());
}

template <typename T>
Tensor load_tensor_helper(const std::string& file_name, T device) {
    std::ifstream input_stream(file_name, std::ios::in | std::ios::binary);
    if (not input_stream) {
        throw std::runtime_error(fmt::format("Cannot open \"{}\"", file_name));
    }

    std::size_t read_sentinel;
    input_stream.read(reinterpret_cast<char*>(&read_sentinel), sizeof(read_sentinel));
    if (read_sentinel == SENTINEL_VALUE) {
        std::uint8_t version_id;
        input_stream.read(reinterpret_cast<char*>(&version_id), sizeof(version_id));

        // Allow only backward compatible versions
        if (version_id > VERSION_ID) {
            throw std::runtime_error(
                fmt::format("Serialized tensor with version_id: {}. Loader version: {}", version_id, VERSION_ID));
        }
        auto shape = LegacyShape{};
        DataType data_type;
        Layout layout;
        StorageType storage_type;
        input_stream.read(reinterpret_cast<char*>(&shape), sizeof(LegacyShape));
        input_stream.read(reinterpret_cast<char*>(&data_type), sizeof(DataType));
        input_stream.read(reinterpret_cast<char*>(&layout), sizeof(Layout));
        input_stream.read(reinterpret_cast<char*>(&storage_type), sizeof(StorageType));

        bool has_memory_config = false;
        MemoryConfig memory_config = MemoryConfig{
            .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::DRAM};

        if (version_id >= 2) {
            input_stream.read(reinterpret_cast<char*>(&has_memory_config), sizeof(bool));
            if (has_memory_config) {
                memory_config = tt::tt_metal::load_memory_config(input_stream);
            }
        }

        auto storage = load_storage(input_stream, data_type, layout, storage_type, device);

        auto tensor = Tensor(
            std::move(storage),
            TensorSpec(
                shape.logical_shape(),
                TensorLayout::fromPaddedShape(
                    data_type, layout, MemoryConfig{}, shape.logical_shape(), shape.padded_shape())));
        if (device != nullptr) {
            tensor = tensor.to(device, memory_config);
        } else if (has_memory_config) {
            tt::log_warning("Memory config is ignored when loading the tensor because device is not provided");
        }
        return tensor;

    } else {
        input_stream.seekg(0, std::ios::beg);  // No sentinel found, assume it's an older format and rewind

        auto shape = LegacyShape{};
        DataType data_type;
        Layout layout;
        input_stream.read(reinterpret_cast<char*>(&shape), sizeof(LegacyShape));
        input_stream.read(reinterpret_cast<char*>(&data_type), sizeof(DataType));
        input_stream.read(reinterpret_cast<char*>(&layout), sizeof(Layout));

        auto storage = load_owned_storage(input_stream, data_type);
        auto tensor = Tensor(
            std::move(storage),
            TensorSpec(
                shape.logical_shape(),
                TensorLayout::fromPaddedShape(
                    data_type, layout, MemoryConfig{}, shape.logical_shape(), shape.padded_shape())));
        if (device != nullptr) {
            tensor = tensor.to(device);
        }
        return tensor;
    }
}

// Explicit instantiations
Tensor load_tensor(const std::string& file_name, IDevice* device) {
    return load_tensor_helper<IDevice*>(file_name, device);
}
Tensor load_tensor(const std::string& file_name, MeshDevice* device) {
    return load_tensor_helper<MeshDevice*>(file_name, device);
}

void dump_memory_config(std::ostream& output_stream, const MemoryConfig& memory_config) {
    output_stream.write(reinterpret_cast<const char*>(&VERSION_ID), sizeof(std::uint8_t));
    output_stream.write(reinterpret_cast<const char*>(&memory_config.memory_layout), sizeof(TensorMemoryLayout));
    output_stream.write(reinterpret_cast<const char*>(&memory_config.buffer_type), sizeof(BufferType));

    bool has_shard_spec = memory_config.shard_spec.has_value();
    output_stream.write(reinterpret_cast<const char*>(&has_shard_spec), sizeof(bool));
    if (has_shard_spec) {
        const auto& shard_spec = memory_config.shard_spec.value();
        const auto& core_ranges = shard_spec.grid.ranges();
        std::size_t num_core_ranges = core_ranges.size();
        output_stream.write(reinterpret_cast<const char*>(&num_core_ranges), sizeof(std::size_t));
        for (const auto& core_range : core_ranges) {
            output_stream.write(reinterpret_cast<const char*>(&core_range), sizeof(CoreRange));
        }
        output_stream.write(reinterpret_cast<const char*>(&shard_spec.shape), sizeof(std::array<uint32_t, 2>));
        output_stream.write(reinterpret_cast<const char*>(&shard_spec.orientation), sizeof(ShardOrientation));
    }
}

void dump_memory_config(const std::string& file_name, const MemoryConfig& memory_config) {
    std::ofstream output_stream(file_name, std::ios::out | std::ios::binary);
    if (not output_stream) {
        throw std::runtime_error(fmt::format("Cannot open \"{}\"", file_name));
    }
    dump_memory_config(output_stream, memory_config);
}

MemoryConfig load_memory_config(std::ifstream& input_stream) {
    std::uint8_t version_id;
    TensorMemoryLayout memory_layout;
    BufferType buffer_type;
    bool has_shard_spec;
    input_stream.read(reinterpret_cast<char*>(&version_id), sizeof(std::uint8_t));

    // Allow only backward compatible versions
    if (version_id > VERSION_ID) {
        throw std::runtime_error(
            fmt::format("Serialized tensor with version_id: {}. Loader version: {}", version_id, VERSION_ID));
    }
    input_stream.read(reinterpret_cast<char*>(&memory_layout), sizeof(TensorMemoryLayout));
    input_stream.read(reinterpret_cast<char*>(&buffer_type), sizeof(BufferType));
    input_stream.read(reinterpret_cast<char*>(&has_shard_spec), sizeof(bool));

    std::optional<ShardSpec> shard_spec = std::nullopt;
    if (has_shard_spec) {
        std::size_t num_core_ranges;
        std::set<CoreRange> core_ranges;
        std::array<uint32_t, 2> shape;
        ShardOrientation orientation;

        input_stream.read(reinterpret_cast<char*>(&num_core_ranges), sizeof(std::size_t));
        for (auto index = 0; index < num_core_ranges; index++) {
            CoreRange core_range{{}, {}};
            input_stream.read(reinterpret_cast<char*>(&core_range), sizeof(CoreRange));
            core_ranges.insert(core_range);
        }
        input_stream.read(reinterpret_cast<char*>(&shape), sizeof(std::array<uint32_t, 2>));
        input_stream.read(reinterpret_cast<char*>(&orientation), sizeof(ShardOrientation));
        if (version_id <= 3) {
            // Read halo for backward compatibility.
            bool halo;
            input_stream.read(reinterpret_cast<char*>(&halo), sizeof(bool));
        }
        shard_spec = {CoreRangeSet{core_ranges}, shape, orientation};
    }
    return MemoryConfig{memory_layout, buffer_type, shard_spec};
}

MemoryConfig load_memory_config(const std::string& file_name) {
    std::ifstream input_stream(file_name, std::ios::in | std::ios::binary);
    if (not input_stream) {
        throw std::runtime_error(fmt::format("Cannot open \"{}\"", file_name));
    }
    return load_memory_config(input_stream);
}

}  // namespace tt::tt_metal
