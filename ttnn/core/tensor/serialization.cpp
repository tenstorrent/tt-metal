// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/serialization.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>

#include <flatbuffers/flatbuffers.h>

#include <tt_stl/overloaded.hpp>

#include "distributed/distributed_tensor_config.hpp"
#include "tensor/tensor_spec.hpp"
#include "tt-metalium/mesh_coord.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/distributed/types.hpp"
#include "tensor/flatbuffer/tensor_spec_flatbuffer.hpp"
#include "tensor/flatbuffer/tensor_flatbuffer.hpp"

namespace tt::tt_metal {

using MeshDevice = distributed::MeshDevice;

namespace {

void validate_version(uint8_t version_id) {
    TT_FATAL(
        version_id >= 5,
        "Version {} is no longer supported. Please update your saved data to the supported version (5).",
        version_id);
    TT_FATAL(
        version_id <= VERSION_ID,
        "Version mismatch: the serialized tensor was created with version {} but is "
        "being loaded by a loader with version {}. Please update your saved data or your "
        "loader so that both versions match.",
        version_id,
        VERSION_ID);
}

struct FileCloser {
    void operator()(FILE* file) const {
        if (file) {
            if (fclose(file) != 0) {
                log_warning(tt::LogAlways, "Failed to close file");
            }
        }
    }
};

constexpr uint64_t SENTINEL_VALUE = std::numeric_limits<uint64_t>::max();

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

void dump_host_storage(FILE* output_file, const HostStorage& storage, DataType dtype) {
    // TODO: #16067 - When dumping storage, we should not care about dtype.
    // We should dump the `size` of raw bytes, not the size of logical elements.
    const size_t element_size = [dtype]() {
        switch (dtype) {
            case DataType::BFLOAT16: return sizeof(::bfloat16);
            case DataType::FLOAT32: return sizeof(float);
            case DataType::UINT8: return sizeof(uint8_t);
            case DataType::UINT16: return sizeof(uint16_t);
            case DataType::INT32: return sizeof(int32_t);
            // Block float types are encoded as uint32_t.
            case DataType::BFLOAT8_B:
            case DataType::BFLOAT4_B:
            case DataType::UINT32: return sizeof(uint32_t);
            case DataType::INVALID: TT_THROW("Unsupported DataType");
        }
        TT_THROW("Unreachable");
    }();

    auto raw_bytes = storage.buffer.view_bytes();
    uint64_t size = raw_bytes.size() / element_size;
    safe_fwrite(&size, sizeof(size), 1, output_file);
    safe_fwrite(raw_bytes.data(), raw_bytes.size(), 1, output_file);
}

void dump_multi_device_host_storage(
    FILE* output_file,
    const MultiDeviceHostStorage& storage,
    const DistributedTensorConfig& strategy,
    const TensorSpec& tensor_spec) {
    std::vector<HostBuffer> buffers;
    storage.distributed_buffer().apply([&](const HostBuffer& shard) { buffers.push_back(shard); });

    uint64_t num_buffers = buffers.size();
    safe_fwrite(&num_buffers, sizeof(num_buffers), 1, output_file);

    // Use the user-specified strategy which defines how it gets distributed when mapped onto multi-device
    safe_fwrite(&strategy, sizeof(strategy), 1, output_file);

    if (std::holds_alternative<ReplicateTensor>(strategy)) {
        dump_host_storage(output_file, buffers.front(), tensor_spec.data_type());
        dump_tensor_spec(tensor_spec, output_file);
    } else {
        for (int i = 0; i < num_buffers; i++) {
            dump_host_storage(output_file, buffers[i], tensor_spec.data_type());
        }
        for (int i = 0; i < num_buffers; i++) {
            dump_tensor_spec(tensor_spec, output_file);
        }
    }
}

template <typename T>
HostStorage load_host_storage(FILE* input_file) {
    uint64_t size = 0;
    safe_fread(&size, sizeof(size), 1, input_file);
    std::vector<T> data(size);
    safe_fread(data.data(), sizeof(T) * size, 1, input_file);
    auto buffer = HostBuffer(std::move(data));
    return {buffer};
}

// Helper type to bundle storage and strategy together.
struct DistributedStorage {
    Storage storage;
    DistributedTensorConfig strategy;
};

template <typename T>
DistributedStorage load_multi_device_host_storage(
    FILE* input_file, DataType data_type, Layout layout, MeshDevice* mesh_device) {
    uint64_t num_buffers = 0;
    DistributedTensorConfig strategy;
    safe_fread(&num_buffers, sizeof(num_buffers), 1, input_file);
    safe_fread(&strategy, sizeof(strategy), 1, input_file);

    std::vector<HostBuffer> buffers;
    // Tensor spec was serialized, but now TTNN enforces uniform tensor specs.
    // Load the spec without using it, to correctly read the file.
    auto ignore_spec = [](const TensorSpec&) {};
    if (std::holds_alternative<ReplicateTensor>(strategy)) {
        uint64_t size = 0;
        safe_fread(&size, sizeof(size), 1, input_file);
        std::vector<T> data(size);
        safe_fread(data.data(), sizeof(T) * size, 1, input_file);
        HostBuffer buffer = HostBuffer(std::move(data));
        buffers.push_back(std::move(buffer));
        ignore_spec(load_tensor_spec(input_file));

        auto num_devices = mesh_device ? mesh_device->num_devices() : 1;
        for (std::size_t i = 1; i < num_devices; ++i) {
            buffers.push_back(buffers[0]);
        }

    } else {
        for (std::size_t i = 0; i < num_buffers; ++i) {
            uint64_t size = 0;
            safe_fread(&size, sizeof(size), 1, input_file);
            std::vector<T> data(size);
            safe_fread(data.data(), sizeof(T) * size, 1, input_file);
            auto buffer = HostBuffer(std::move(data));
            buffers.push_back(std::move(buffer));
        }
        for (std::size_t i = 0; i < num_buffers; ++i) {
            ignore_spec(load_tensor_spec(input_file));
        }
    }

    return {MultiDeviceHostStorage{std::move(buffers)}, strategy};
}

HostStorage load_host_storage(FILE* input_file, DataType data_type) {
    if (data_type == DataType::UINT32 or data_type == DataType::BFLOAT8_B or data_type == DataType::BFLOAT4_B) {
        using T = std::uint32_t;
        return load_host_storage<T>(input_file);
    } else if (data_type == DataType::INT32) {
        using T = std::int32_t;
        return load_host_storage<T>(input_file);
    } else if (data_type == DataType::UINT8) {
        using T = std::uint8_t;
        return load_host_storage<T>(input_file);
    } else if (data_type == DataType::UINT16) {
        using T = std::uint16_t;
        return load_host_storage<T>(input_file);
    } else if (data_type == DataType::FLOAT32) {
        using T = float;
        return load_host_storage<T>(input_file);
    } else if (data_type == DataType::BFLOAT16) {
        using T = bfloat16;
        return load_host_storage<T>(input_file);
    } else {
        TT_THROW("Unsupported DataType");
    }
}

DistributedStorage load_multi_device_host_storage(
    FILE* input_file, DataType data_type, Layout layout, MeshDevice* mesh_device) {
    if (data_type == DataType::UINT32 or data_type == DataType::BFLOAT8_B or data_type == DataType::BFLOAT4_B) {
        using T = std::uint32_t;
        return load_multi_device_host_storage<T>(input_file, data_type, layout, mesh_device);
    } else if (data_type == DataType::UINT16) {
        using T = std::uint16_t;
        return load_multi_device_host_storage<T>(input_file, data_type, layout, mesh_device);
    } else if (data_type == DataType::FLOAT32) {
        using T = float;
        return load_multi_device_host_storage<T>(input_file, data_type, layout, mesh_device);
    } else if (data_type == DataType::BFLOAT16) {
        using T = bfloat16;
        return load_multi_device_host_storage<T>(input_file, data_type, layout, mesh_device);
    } else {
        TT_THROW("Unsupported DataType");
    }
}

DistributedStorage load_storage(
    FILE* input_file, DataType data_type, Layout layout, StorageType storage_type, MeshDevice* device) {
    if (storage_type == StorageType::MULTI_DEVICE_HOST or storage_type == StorageType::DEVICE) {
        return load_multi_device_host_storage(input_file, data_type, layout, device);
    }
    return DistributedStorage{load_host_storage(input_file, data_type), ReplicateTensor{}};
}

}  // namespace

Tensor load_tensor(const std::string& file_name, MeshDevice* device) {
    FILE* input_file = fopen(file_name.c_str(), "rb");
    if (not input_file) {
        TT_THROW("Cannot open \"{}\"", file_name);
    }
    std::unique_ptr<FILE, FileCloser> file_guard(input_file);

    std::size_t read_sentinel;
    safe_fread(&read_sentinel, sizeof(read_sentinel), 1, input_file);
    TT_FATAL(
        read_sentinel == SENTINEL_VALUE,
        "Sentinel value is not valid. The tensor data in {} is corrupted and cannot be loaded.",
        file_name);

    std::uint8_t version_id = 0;
    safe_fread(&version_id, sizeof(version_id), 1, input_file);
    validate_version(version_id);

    auto spec = load_tensor_spec(input_file);
    StorageType storage_type = StorageType::HOST;
    safe_fread(&storage_type, sizeof(storage_type), 1, input_file);
    auto storage = load_storage(input_file, spec.data_type(), spec.layout(), storage_type, device);
    Tensor tensor(std::move(storage.storage), spec, storage.strategy);
    if (device != nullptr) {
        tensor = tensor.to_device(device, spec.memory_config());
    }
    return tensor;
}

void dump_tensor(
    const std::string& file_name, const Tensor& tensor, const std::unordered_map<std::string, std::string>& strategy) {
    FILE* output_file = fopen(file_name.c_str(), "wb");
    if (not output_file) {
        TT_THROW("Cannot open \"{}\"", file_name);
    }
    std::unique_ptr<FILE, FileCloser> file_guard(output_file);

    safe_fwrite(&SENTINEL_VALUE, sizeof(SENTINEL_VALUE), 1, output_file);
    safe_fwrite(&VERSION_ID, sizeof(VERSION_ID), 1, output_file);

    dump_tensor_spec(tensor.tensor_spec(), output_file);

    auto storage_type = tensor.storage_type();
    safe_fwrite(&storage_type, sizeof(storage_type), 1, output_file);

    bool is_on_device = is_device_tensor(tensor);
    Tensor tensor_to_dump = tensor;
    if (is_on_device) {
        tensor_to_dump = tensor_to_dump.cpu();
    }

    std::visit(
        tt::stl::overloaded{
            [output_file, dtype = tensor.dtype()](const HostStorage& storage) {
                dump_host_storage(output_file, storage, dtype);
            },
            [output_file, dtype = tensor.dtype()](const DeviceStorage& storage) {
                TT_THROW("Device storage isn't supported");
            },
            [output_file, &strategy, &tensor_spec = tensor.tensor_spec()](const MultiDeviceHostStorage& storage) {
                auto distribute_config = get_distributed_tensor_config(strategy);
                dump_multi_device_host_storage(output_file, storage, distribute_config, tensor_spec);
            },
        },
        tensor_to_dump.storage());
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
    validate_version(version_id);

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

void dump_tensor_flatbuffer(const std::string& file_name, const Tensor& tensor) {
    FILE* output_file = fopen(file_name.c_str(), "wb");
    TT_FATAL(output_file != nullptr, "Cannot open \"{}\"", file_name);
    std::unique_ptr<FILE, FileCloser> file_guard(output_file);

    Tensor cpu_tensor = tensor.cpu();

    flatbuffers::FlatBufferBuilder builder;
    auto tensor_offset = ttnn::to_flatbuffer(cpu_tensor, builder);
    builder.Finish(tensor_offset);

    uint64_t header_size = builder.GetSize();
    safe_fwrite(&header_size, sizeof(header_size), 1, output_file);
    safe_fwrite(builder.GetBufferPointer(), header_size, 1, output_file);

    std::visit(
        tt::stl::overloaded{
            [&output_file](const HostStorage& storage) {
                auto buffer_view = storage.buffer.view_bytes();
                safe_fwrite(buffer_view.data(), buffer_view.size(), 1, output_file);
            },
            [&output_file](const DeviceStorage&) {
                // Unreachable - the tensor should be moved to host at this point.
                TT_THROW("Device storage isn't supported in flatbuffer serialization");
            },
            [&output_file](const MultiDeviceHostStorage& storage) {
                for (const auto& shard : storage.distributed_buffer().shard_coords()) {
                    if (auto buffer = storage.distributed_buffer().get_shard(shard); buffer.has_value()) {
                        auto buffer_view = buffer->view_bytes();
                        safe_fwrite(buffer_view.data(), buffer_view.size(), 1, output_file);
                    }
                }
            }},
        cpu_tensor.storage());
}

Tensor load_tensor_flatbuffer(const std::string& file_name, MeshDevice* device) {
    FILE* input_file = fopen(file_name.c_str(), "rb");
    TT_FATAL(input_file != nullptr, "Cannot open \"{}\"", file_name);
    std::unique_ptr<FILE, FileCloser> file_guard(input_file);

    uint64_t header_size = 0;
    safe_fread(&header_size, sizeof(header_size), 1, input_file);

    std::vector<std::byte> header_buffer(header_size);
    safe_fread(header_buffer.data(), header_size, 1, input_file);

    auto fb_tensor = ttnn::flatbuffer::GetTensor(header_buffer.data());

    // Get current position and seek to end to calculate remaining data size
    off_t current_pos = ftello(input_file);
    TT_FATAL(current_pos != -1, "Failed to get current file position");

    TT_FATAL(fseek(input_file, 0, SEEK_END) == 0, "Failed to seek to end of file");

    off_t end_pos = ftello(input_file);
    TT_FATAL(end_pos != -1, "Failed to get end file position");

    uint64_t data_size = static_cast<uint64_t>(end_pos - current_pos);

    TT_FATAL(fseek(input_file, current_pos, SEEK_SET) == 0, "Failed to seek back to data start");

    std::vector<std::byte> data_buffer(data_size);
    safe_fread(data_buffer.data(), data_size, 1, input_file);

    Tensor tensor = ttnn::from_flatbuffer(fb_tensor, data_buffer);
    if (device != nullptr) {
        tensor = tensor.to_device(device);
    }
    return tensor;
}

}  // namespace tt::tt_metal
