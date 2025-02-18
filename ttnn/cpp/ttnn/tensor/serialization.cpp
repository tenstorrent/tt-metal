// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/serialization.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>

#include <flatbuffers/flatbuffers.h>

#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/flatbuffer/tensor_types_from_flatbuffer.hpp"
#include "ttnn/tensor/flatbuffer/tensor_types_to_flatbuffer.hpp"

namespace tt::tt_metal {

using MeshDevice = distributed::MeshDevice;

namespace {

struct FileCloser {
    void operator()(FILE* file) const {
        if (file) {
            if (fclose(file) != 0) {
                log_warning("Failed to close file");
            }
        }
    }
};

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

static constexpr uint64_t SENTINEL_VALUE = std::numeric_limits<uint64_t>::max();

void safe_fread(void* buffer, size_t size, size_t count, FILE* file) {
    if (fread(buffer, size, count, file) != count) {
        TT_THROW("Failed to read tensor data, file must be corrupted");
    }
}

void safe_fwrite(const void* buffer, size_t size, size_t count, FILE* file) {
    if (fwrite(buffer, size, count, file) != count) {
        TT_THROW("Failed to write tensor data: file write failed");
    }
}

void dump_tensor_spec(const TensorSpec& tensor_spec, FILE* output_file) {
    flatbuffers::FlatBufferBuilder builder;
    auto flat_spec = ttnn::to_flatbuffer(tensor_spec, builder);
    builder.Finish(flat_spec);
    uint64_t buffer_size = builder.GetSize();
    safe_fwrite(&buffer_size, sizeof(buffer_size), 1, output_file);
    safe_fwrite(builder.GetBufferPointer(), buffer_size, 1, output_file);
}

TensorSpec load_tensor_spec(FILE* input_file) {
    uint64_t bin_size = 0;
    safe_fread(&bin_size, sizeof(bin_size), 1, input_file);
    std::vector<uint8_t> bin(bin_size);
    safe_fread(bin.data(), bin_size, 1, input_file);
    flatbuffers::Verifier verifier(bin.data(), bin_size);
    if (!ttnn::flatbuffer::VerifyTensorSpecBuffer(verifier)) {
        TT_THROW("TensorSpec deserialization failed: invalid buffer");
    }
    auto spec = ttnn::flatbuffer::GetTensorSpec(bin.data());
    return ttnn::from_flatbuffer(spec);
}

void dump_owned_storage(FILE* output_file, const OwnedStorage& storage) {
    std::visit(
        [output_file]<typename T>(const owned_buffer::Buffer<T>& generic_buffer) {
            const auto buffer = owned_buffer::get_as<T>(generic_buffer);
            uint64_t size = buffer.size();
            safe_fwrite(&size, sizeof(size), 1, output_file);
            safe_fwrite(buffer.data(), sizeof(T) * size, 1, output_file);
        },
        storage.buffer);
}

void dump_borrowed_storage(FILE* output_file, const BorrowedStorage& storage) {
    std::visit(
        [output_file]<typename T>(const borrowed_buffer::Buffer<T>& generic_buffer) {
            const auto buffer = borrowed_buffer::get_as<T>(generic_buffer);
            uint64_t size = buffer.size();
            safe_fwrite(&size, sizeof(size), 1, output_file);
            safe_fwrite(buffer.data(), sizeof(T) * size, 1, output_file);
        },
        storage.buffer);
}

void dump_multi_device_host_storage(
    FILE* output_file, const MultiDeviceHostStorage& storage, const DistributedTensorConfig& strategy) {
    uint64_t num_buffers = storage.num_buffers();
    safe_fwrite(&num_buffers, sizeof(num_buffers), 1, output_file);

    // Use the user-specified strategy which defines how it gets distributed when mapped onto multi-device
    safe_fwrite(&strategy, sizeof(strategy), 1, output_file);

    if (std::holds_alternative<ReplicateTensor>(strategy)) {
        std::visit(
            [output_file]<typename T>(const owned_buffer::Buffer<T>& generic_buffer) {
                const auto buffer = owned_buffer::get_as<T>(generic_buffer);
                uint64_t size = buffer.size();
                safe_fwrite(&size, sizeof(size), 1, output_file);
                safe_fwrite(buffer.begin(), sizeof(T) * size, 1, output_file);
            },
            storage.get_buffer(0));
        auto spec = storage.specs.at(0);
        dump_tensor_spec(spec, output_file);
    } else {
        for (int i = 0; i < num_buffers; i++) {
            std::visit(
                [output_file]<typename T>(const owned_buffer::Buffer<T>& generic_buffer) {
                    const auto buffer = owned_buffer::get_as<T>(generic_buffer);
                    uint64_t size = buffer.size();
                    safe_fwrite(&size, sizeof(size), 1, output_file);
                    safe_fwrite(buffer.begin(), sizeof(T) * size, 1, output_file);
                },
                storage.get_buffer(i));
        }
        for (const auto& spec : storage.specs) {
            dump_tensor_spec(spec, output_file);
        }
    }
}

template <typename T>
OwnedStorage load_owned_storage(FILE* input_file) {
    uint64_t size = 0;
    safe_fread(&size, sizeof(size), 1, input_file);
    auto buffer = owned_buffer::create<T>(size);
    safe_fread(buffer.begin(), sizeof(T) * size, 1, input_file);
    return {buffer};
}

template <typename T>
MultiDeviceHostStorage load_multi_device_host_storage(
    FILE* input_file, DataType data_type, Layout layout, MeshDevice* mesh_device, uint8_t version_id) {
    uint64_t num_buffers = 0;
    DistributedTensorConfig strategy;
    safe_fread(&num_buffers, sizeof(num_buffers), 1, input_file);
    safe_fread(&strategy, sizeof(strategy), 1, input_file);

    std::vector<OwnedBuffer> buffers;
    std::vector<ttnn::TensorSpec> specs;
    if (std::holds_alternative<ReplicateTensor>(strategy)) {
        uint64_t size = 0;
        safe_fread(&size, sizeof(size), 1, input_file);
        auto buffer = owned_buffer::create<T>(size);
        safe_fread(buffer.begin(), sizeof(T) * size, 1, input_file);
        buffers.push_back(buffer);
        auto spec = [&] {
            if (version_id >= 5) {
                return load_tensor_spec(input_file);
            }
            auto shape = LegacyShape{};
            safe_fread(&shape, sizeof(shape), 1, input_file);
            return TensorSpec(
                shape.logical_shape(),
                TensorLayout::fromPaddedShape(
                    data_type, PageConfig(layout), MemoryConfig{}, shape.logical_shape(), shape.padded_shape()));
        }();
        specs.push_back(spec);

        for (std::size_t i = 1; i < mesh_device->num_devices(); ++i) {
            buffers.push_back(owned_buffer::Buffer<T>{buffer.get_ptr()});
            specs.push_back(spec);
        }

    } else {
        for (std::size_t i = 0; i < num_buffers; ++i) {
            uint64_t size = 0;
            safe_fread(&size, sizeof(size), 1, input_file);
            auto buffer = owned_buffer::create<T>(size);
            safe_fread(buffer.begin(), sizeof(T) * size, 1, input_file);
            buffers.push_back(std::move(buffer));
        }
        for (std::size_t i = 0; i < num_buffers; ++i) {
            if (version_id >= 5) {
                specs.push_back(load_tensor_spec(input_file));
            } else {
                auto shape = LegacyShape{};
                safe_fread(&shape, sizeof(shape), 1, input_file);
                TensorSpec spec(
                    shape.logical_shape(),
                    TensorLayout::fromPaddedShape(
                        data_type, PageConfig(layout), MemoryConfig{}, shape.logical_shape(), shape.padded_shape()));
                specs.push_back(spec);
            }
        }
    }

    return {strategy, buffers, specs};
}

OwnedStorage load_owned_storage(FILE* input_file, DataType data_type) {
    if (data_type == DataType::UINT32 or data_type == DataType::BFLOAT8_B or data_type == DataType::BFLOAT4_B) {
        using T = std::uint32_t;
        return load_owned_storage<T>(input_file);
    } else if (data_type == DataType::INT32) {
        using T = std::int32_t;
        return load_owned_storage<T>(input_file);
    } else if (data_type == DataType::UINT8) {
        using T = std::uint8_t;
        return load_owned_storage<T>(input_file);
    } else if (data_type == DataType::UINT16) {
        using T = std::uint16_t;
        return load_owned_storage<T>(input_file);
    } else if (data_type == DataType::FLOAT32) {
        using T = float;
        return load_owned_storage<T>(input_file);
    } else if (data_type == DataType::BFLOAT16) {
        using T = bfloat16;
        return load_owned_storage<T>(input_file);
    } else {
        TT_THROW("Unsupported DataType");
    }
}

MultiDeviceHostStorage load_multi_device_host_storage(
    FILE* input_file, DataType data_type, Layout layout, MeshDevice* mesh_device, uint8_t version_id) {
    if (data_type == DataType::UINT32 or data_type == DataType::BFLOAT8_B or data_type == DataType::BFLOAT4_B) {
        using T = std::uint32_t;
        return load_multi_device_host_storage<T>(input_file, data_type, layout, mesh_device, version_id);
    } else if (data_type == DataType::UINT16) {
        using T = std::uint16_t;
        return load_multi_device_host_storage<T>(input_file, data_type, layout, mesh_device, version_id);
    } else if (data_type == DataType::FLOAT32) {
        using T = float;
        return load_multi_device_host_storage<T>(input_file, data_type, layout, mesh_device, version_id);
    } else if (data_type == DataType::BFLOAT16) {
        using T = bfloat16;
        return load_multi_device_host_storage<T>(input_file, data_type, layout, mesh_device, version_id);
    } else {
        TT_THROW("Unsupported DataType");
    }
}

template <typename T>
Storage load_storage(
    FILE* input_file, DataType data_type, Layout layout, StorageType storage_type, T device, uint8_t version_id) {
    if (storage_type == StorageType::MULTI_DEVICE_HOST or storage_type == StorageType::MULTI_DEVICE) {
        if constexpr (std::is_same_v<T, MeshDevice*>) {
            return load_multi_device_host_storage(input_file, data_type, layout, device, version_id);
        } else {
            TT_THROW("MeshDevice is required for MULTI_DEVICE_HOST storage");
        }
    } else {
        return load_owned_storage(input_file, data_type);
    }
}

template <typename T>
Tensor load_tensor_helper_legacy_impl(FILE* input_file, T device, uint8_t version_id) {
    auto shape = LegacyShape{};
    DataType data_type;
    Layout layout;
    StorageType storage_type;
    safe_fread(&shape, sizeof(shape), 1, input_file);
    safe_fread(&data_type, sizeof(data_type), 1, input_file);
    safe_fread(&layout, sizeof(layout), 1, input_file);
    safe_fread(&storage_type, sizeof(storage_type), 1, input_file);

    bool has_memory_config = false;
    MemoryConfig memory_config =
        MemoryConfig{.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::DRAM};

    if (version_id >= 2) {
        safe_fread(&has_memory_config, sizeof(has_memory_config), 1, input_file);
        if (has_memory_config) {
            memory_config = tt::tt_metal::load_memory_config(input_file);
        }
    }

    auto storage = load_storage(input_file, data_type, layout, storage_type, device, version_id);

    auto tensor = Tensor(
        std::move(storage),
        TensorSpec(
            shape.logical_shape(),
            TensorLayout::fromPaddedShape(
                data_type, layout, MemoryConfig{}, shape.logical_shape(), shape.padded_shape())));
    if (device != nullptr) {
        tensor = tensor.to_device(device, memory_config);
    } else if (has_memory_config) {
        tt::log_warning("Memory config is ignored when loading the tensor because device is not provided");
    }
    return tensor;
}

// Used before VERSION_ID was introduced
template <typename T>
Tensor load_tensor_helper_very_legacy_impl(FILE* input_file, T device) {
    auto shape = LegacyShape{};
    DataType data_type;
    Layout layout;
    safe_fread(&shape, sizeof(shape), 1, input_file);
    safe_fread(&data_type, sizeof(data_type), 1, input_file);
    safe_fread(&layout, sizeof(layout), 1, input_file);

    auto storage = load_owned_storage(input_file, data_type);
    auto tensor = Tensor(
        std::move(storage),
        TensorSpec(
            shape.logical_shape(),
            TensorLayout::fromPaddedShape(
                data_type, layout, MemoryConfig{}, shape.logical_shape(), shape.padded_shape())));
    if (device != nullptr) {
        tensor = tensor.to_device(device);
    }
    return tensor;
}

// Used before flatbuffer serialization, aka VERSION_ID < 5
MemoryConfig load_memory_config_legacy_impl(FILE* input_file, uint8_t version_id) {
    TensorMemoryLayout memory_layout;
    BufferType buffer_type;
    bool has_shard_spec;
    safe_fread(&memory_layout, sizeof(memory_layout), 1, input_file);
    safe_fread(&buffer_type, sizeof(buffer_type), 1, input_file);
    safe_fread(&has_shard_spec, sizeof(has_shard_spec), 1, input_file);

    std::optional<ShardSpec> shard_spec = std::nullopt;
    if (has_shard_spec) {
        uint64_t num_core_ranges;
        std::set<CoreRange> core_ranges;
        std::array<uint32_t, 2> shape;
        ShardOrientation orientation;

        safe_fread(&num_core_ranges, sizeof(num_core_ranges), 1, input_file);
        for (auto index = 0; index < num_core_ranges; index++) {
            CoreRange core_range{{}, {}};
            safe_fread(&core_range, sizeof(core_range), 1, input_file);
            core_ranges.insert(core_range);
        }
        safe_fread(&shape, sizeof(shape), 1, input_file);
        safe_fread(&orientation, sizeof(orientation), 1, input_file);
        if (version_id <= 3) {
            // Read halo for backward compatibility.
            bool halo;
            safe_fread(&halo, sizeof(halo), 1, input_file);
        }
        shard_spec = {CoreRangeSet{core_ranges}, shape, orientation};
    }
    return MemoryConfig{memory_layout, buffer_type, shard_spec};
}

template <typename T>
Tensor load_tensor_helper(const std::string& file_name, T device) {
    FILE* input_file = fopen(file_name.c_str(), "rb");
    if (not input_file) {
        TT_THROW("Cannot open \"{}\"", file_name);
    }
    std::unique_ptr<FILE, FileCloser> file_guard(input_file);

    std::size_t read_sentinel;
    safe_fread(&read_sentinel, sizeof(read_sentinel), 1, input_file);
    if (read_sentinel != SENTINEL_VALUE) {
        fseek(input_file, 0, SEEK_SET);
        return load_tensor_helper_very_legacy_impl(input_file, device);
    }

    std::uint8_t version_id = 0;
    safe_fread(&version_id, sizeof(version_id), 1, input_file);
    if (version_id > VERSION_ID) {
        TT_THROW(
            "Version mismatch: the serialized tensor was created with version {} but is being loaded by a loader with "
            "version {}. Please update your saved data or your loader so that both versions match.",
            version_id,
            VERSION_ID);
    }

    if (version_id < 5) {
        return load_tensor_helper_legacy_impl(input_file, device, version_id);
    }

    auto spec = load_tensor_spec(input_file);
    StorageType storage_type = StorageType::OWNED;
    safe_fread(&storage_type, sizeof(storage_type), 1, input_file);
    auto storage = load_storage(input_file, spec.data_type(), spec.layout(), storage_type, device, version_id);
    Tensor tensor(std::move(storage), spec);
    if (device != nullptr) {
        tensor = tensor.to_device(device, spec.memory_config());
    }
    return tensor;
}

}  // namespace

void dump_tensor(
    const std::string& file_name, const Tensor& tensor, const std::unordered_map<std::string, std::string>& strategy) {
    FILE* output_file = fopen(file_name.c_str(), "wb");
    if (not output_file) {
        TT_THROW("Cannot open \"{}\"", file_name);
    }
    std::unique_ptr<FILE, FileCloser> file_guard(output_file);

    safe_fwrite(&SENTINEL_VALUE, sizeof(SENTINEL_VALUE), 1, output_file);
    safe_fwrite(&VERSION_ID, sizeof(VERSION_ID), 1, output_file);

    dump_tensor_spec(tensor.get_tensor_spec(), output_file);

    auto storage_type = tensor.storage_type();
    safe_fwrite(&storage_type, sizeof(storage_type), 1, output_file);

    bool is_on_device = is_tensor_on_device_or_multidevice(tensor);
    Tensor tensor_to_dump = tensor;
    if (is_on_device) {
        tensor_to_dump = tensor_to_dump.cpu();
    }

    std::visit(
        [output_file, &strategy](const auto& storage) {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                dump_owned_storage(output_file, storage);
            } else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                dump_borrowed_storage(output_file, storage);
            } else if constexpr (std::is_same_v<StorageType, DeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceStorage>) {
                TT_THROW("Device storage isn't supported");
            } else if constexpr (std::is_same_v<StorageType, MultiDeviceHostStorage>) {
                auto distribute_config = get_distributed_tensor_config(strategy);
                dump_multi_device_host_storage(output_file, storage, distribute_config);
            } else {
                raise_unsupported_storage<StorageType>();
            }
        },
        tensor_to_dump.get_storage());
}

// Explicit instantiations
Tensor load_tensor(const std::string& file_name, IDevice* device) {
    return load_tensor_helper<IDevice*>(file_name, device);
}
Tensor load_tensor(const std::string& file_name, MeshDevice* device) {
    return load_tensor_helper<MeshDevice*>(file_name, device);
}

void dump_memory_config(FILE* output_file, const MemoryConfig& memory_config) {
    safe_fwrite(&VERSION_ID, sizeof(VERSION_ID), 1, output_file);
    flatbuffers::FlatBufferBuilder builder;
    auto flat_config = ttnn::to_flatbuffer(memory_config, builder);
    builder.Finish(flat_config);
    uint64_t buf_size = builder.GetSize();
    safe_fwrite(&buf_size, sizeof(buf_size), 1, output_file);
    safe_fwrite(builder.GetBufferPointer(), buf_size, 1, output_file);
}

void dump_memory_config(const std::string& file_name, const MemoryConfig& memory_config) {
    FILE* output_file = fopen(file_name.c_str(), "wb");
    if (not output_file) {
        TT_THROW("Cannot open \"{}\"", file_name);
    }
    std::unique_ptr<FILE, FileCloser> file_guard(output_file);
    dump_memory_config(output_file, memory_config);
}

MemoryConfig load_memory_config(FILE* input_file) {
    std::uint8_t version_id;
    safe_fread(&version_id, sizeof(version_id), 1, input_file);

    // Allow only backward compatible versions
    if (version_id > VERSION_ID) {
        TT_THROW(
            "Version mismatch: the serialized memory config was created with version {} but is being loaded by a "
            "loader with version {}. Please update your saved data or your loader so that both versions match.",
            version_id,
            VERSION_ID);
    }

    if (version_id < 5) {
        return load_memory_config_legacy_impl(input_file, version_id);
    }

    uint64_t bin_size = 0;
    safe_fread(&bin_size, sizeof(bin_size), 1, input_file);
    std::vector<uint8_t> bin(bin_size);
    safe_fread(bin.data(), bin_size, 1, input_file);
    flatbuffers::Verifier verifier(bin.data(), bin_size);
    if (!verifier.VerifyBuffer<ttnn::flatbuffer::MemoryConfig>()) {
        TT_THROW("MemoryConfig deserialization failed: invalid buffer");
    }
    auto mem_config = flatbuffers::GetRoot<ttnn::flatbuffer::MemoryConfig>(bin.data());
    return ttnn::from_flatbuffer(mem_config);
}

MemoryConfig load_memory_config(const std::string& file_name) {
    FILE* input_file = fopen(file_name.c_str(), "rb");
    if (not input_file) {
        TT_THROW("Cannot open \"{}\"", file_name);
    }
    std::unique_ptr<FILE, FileCloser> file_guard(input_file);
    return load_memory_config(input_file);
}

}  // namespace tt::tt_metal
