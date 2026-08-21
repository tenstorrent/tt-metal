// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/serialization.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cerrno>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/reflection.h>
#include <flatbuffers/verifier.h>

#include <tt_stl/overloaded.hpp>
#include <tt_stl/cleanup.hpp>

#include "tensor/tensor_spec.hpp"
#include "tensor/flatbuffer/tensor_flatbuffer.hpp"
#include <tt-metalium/distributed_host_buffer.hpp>
#include "ttnn/distributed/host_ccl.hpp"

namespace tt::tt_metal {
namespace {

void safe_fwrite_bytes(
    const void* buffer, size_t bytes, FILE* file, const std::string& filename, std::string_view what) {
    TT_FATAL(bytes > 0, "Expected to write > 0 bytes to file");

    // Use byte-wise fwrite so we can detect partial writes
    const size_t written = fwrite(buffer, /*size=*/1, /*count=*/bytes, file);
    TT_FATAL(
        written == bytes,
        "Failed to write {} to \"{}\": wrote {}/{} bytes (ferror={}, errno={} \"{}\")",
        what,
        filename,
        written,
        bytes,
        ferror(file),
        errno,
        strerror(errno));
}

constexpr std::uint32_t kFlatbufferAlignment = alignof(std::uint64_t);

void safe_pread_bytes(int fd, void* buffer, size_t bytes, uint64_t offset, const std::string& filename) {
    auto* destination = static_cast<std::byte*>(buffer);
    while (bytes != 0) {
        const auto result = pread(fd, destination, bytes, static_cast<off_t>(offset));
        TT_FATAL(
            result > 0,
            "Tensor file \"{}\" is truncated or unreadable at offset {}: errno={} \"{}\"",
            filename,
            offset,
            errno,
            strerror(errno));
        destination += result;
        bytes -= result;
        offset += result;
    }
}

struct TensorbinInput {
    std::string file_name;
    uint64_t data_offset;
    uint64_t data_size;
    ttnn::TensorFlatbufferMetadata metadata;
    std::vector<size_t> canonical_shards;
    std::vector<size_t> alias_pattern;
};

size_t compute_bfloat4_packed_size(const TensorSpec& spec) {
    TT_FATAL(
        spec.data_type() == DataType::BFLOAT4_B && spec.layout() == Layout::TILE,
        "coalesce_tensorbins requires tiled BFLOAT4_B tensorbins");
    const auto physical_shape = spec.physical_shape();
    const auto tile = spec.tile();
    TT_FATAL(
        tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH,
        "coalesce_tensorbins currently supports standard 32x32 BFLOAT4_B tiles only");
    TT_FATAL(
        physical_shape.height() % tile.get_height() == 0 && physical_shape.width() % tile.get_width() == 0,
        "Tensor physical shape {} is not divisible by tile shape {}",
        physical_shape,
        tile.get_tile_shape());
    const size_t height_tiles = physical_shape.height() / tile.get_height();
    const size_t width_tiles = physical_shape.width() / tile.get_width();
    TT_FATAL(
        height_tiles == 0 || width_tiles <= std::numeric_limits<size_t>::max() / height_tiles,
        "Tensor tile count overflows host size");
    const size_t tile_count = height_tiles * width_tiles;
    // Tile::get_tile_size queries MetalContext for the L1 alignment, which would open devices for this host-only API.
    // The standard 32x32 BFP4 tile size is format-defined and available through this constexpr helper.
    constexpr size_t tile_size = tt::tile_size(tt::DataFormat::Bfp4_b);
    TT_FATAL(
        tile_count == 0 || tile_size <= std::numeric_limits<size_t>::max() / tile_count,
        "Tensor packed byte size overflows host size");
    return tile_count * tile_size;
}

TensorbinInput read_tensorbin_metadata(const std::string& file_name) {
    int fd = open(file_name.c_str(), O_RDONLY | O_CLOEXEC);
    TT_FATAL(fd != -1, "Cannot open \"{}\": errno={} \"{}\"", file_name, errno, strerror(errno));
    auto cleanup = ttsl::make_cleanup([fd]() { close(fd); });

    struct stat file_stat{};
    TT_FATAL(fstat(fd, &file_stat) == 0, "Failed to get file stats for \"{}\"", file_name);
    TT_FATAL(file_stat.st_size >= static_cast<off_t>(sizeof(uint64_t)), "Tensor file \"{}\" is too small", file_name);

    uint64_t header_size = 0;
    safe_pread_bytes(fd, &header_size, sizeof(header_size), 0, file_name);
    TT_FATAL(
        header_size < flatbuffers::Verifier::Options().max_size,
        "Tensor header in \"{}\" is too large; the file is corrupt",
        file_name);
    TT_FATAL(
        header_size <= static_cast<uint64_t>(file_stat.st_size) - sizeof(header_size),
        "Tensor file \"{}\" is truncated (header_size={}, file_size={})",
        file_name,
        header_size,
        file_stat.st_size);

    std::vector<uint8_t> header(header_size);
    TT_FATAL(!header.empty(), "Tensor file \"{}\" has an empty header", file_name);
    safe_pread_bytes(fd, header.data(), header.size(), sizeof(header_size), file_name);
    flatbuffers::Verifier verifier(header.data(), header.size());
    TT_FATAL(
        ttnn::flatbuffer::VerifyTensorBuffer(verifier),
        "Cannot validate tensor header in \"{}\"; the file is corrupt",
        file_name);
    auto metadata = ttnn::tensor_metadata_from_flatbuffer(ttnn::flatbuffer::GetTensor(header.data()));
    TT_FATAL(!metadata.shards.empty(), "Tensor file \"{}\" contains no shards", file_name);
    const size_t expected_shard_size = compute_bfloat4_packed_size(metadata.tensor_spec);

    const uint64_t data_offset = sizeof(header_size) + header_size;
    const uint64_t data_size = static_cast<uint64_t>(file_stat.st_size) - data_offset;
    uint64_t expected_offset = 0;
    std::unordered_map<uint64_t, size_t> offset_to_canonical;
    std::vector<size_t> canonical_shards;
    std::vector<size_t> alias_pattern;
    alias_pattern.reserve(metadata.shards.size());
    for (size_t index = 0; index < metadata.shards.size(); ++index) {
        const auto& shard = metadata.shards[index];
        TT_FATAL(shard.size != 0, "Tensor file \"{}\" contains an empty shard", file_name);
        TT_FATAL(
            shard.size == expected_shard_size,
            "Tensor file \"{}\" shard {} size {} does not match tensor spec size {}",
            file_name,
            index,
            shard.size,
            expected_shard_size);
        TT_FATAL(
            shard.offset <= data_size && shard.size <= data_size - shard.offset,
            "Tensor file \"{}\" has a truncated shard at payload offset {} with size {}",
            file_name,
            shard.offset,
            shard.size);
        if (const auto found = offset_to_canonical.find(shard.offset); found != offset_to_canonical.end()) {
            TT_FATAL(
                metadata.shards[found->second].size == shard.size,
                "Tensor file \"{}\" has conflicting aliased shard sizes",
                file_name);
            alias_pattern.push_back(found->second);
        } else {
            TT_FATAL(
                shard.offset == expected_offset,
                "Tensor file \"{}\" has overlapping, out-of-order, or gapped shard payloads",
                file_name);
            offset_to_canonical.emplace(shard.offset, index);
            canonical_shards.push_back(index);
            alias_pattern.push_back(index);
            TT_FATAL(
                shard.size <= std::numeric_limits<uint64_t>::max() - expected_offset,
                "Tensor file \"{}\" shard sizes overflow",
                file_name);
            expected_offset += shard.size;
        }
    }
    TT_FATAL(
        expected_offset == data_size,
        "Tensor file \"{}\" has {} unexpected trailing payload bytes",
        file_name,
        data_size - expected_offset);
    return TensorbinInput{
        .file_name = file_name,
        .data_offset = data_offset,
        .data_size = data_size,
        .metadata = std::move(metadata),
        .canonical_shards = std::move(canonical_shards),
        .alias_pattern = std::move(alias_pattern),
    };
}

void dump_tensor_flatbuffer_impl(const std::string& file_name, const Tensor& tensor, DumpTensorMode mode) {
    Tensor cpu_tensor = tensor.cpu();

    if (mode == DumpTensorMode::DISTRIBUTED_GATHER) {
        // Dump tensor to disk from (global) rank 0 host.
        // Note we use global context as opposed to context embedded to the host-side tensor, since the tensor may
        // already be fully host-local. In this latter case, host buffer context will consist of a single (local) host
        // rank, and each host will attempt to flush the serialized tensor file to disk.
        cpu_tensor = ttnn::distributed::host_ccl::all_gather(cpu_tensor);
        const auto& ctx = distributed::multihost::DistributedContext::get_current_world();
        if (ctx->rank() != tt::tt_metal::distributed::multihost::Rank(0)) {
            ctx->barrier();
            return;
        }
    }

    FILE* output_file = fopen(file_name.c_str(), "wb");
    TT_FATAL(
        output_file != nullptr, "Cannot open \"{}\" for writing: errno={} \"{}\"", file_name, errno, strerror(errno));
    auto cleanup = ttsl::make_cleanup([f = output_file, &file_name]() {
        if (f && fclose(f) != 0) {
            log_warning(tt::LogAlways, "Failed to close \"{}\"", file_name);
        }
    });

    std::vector<HostBuffer> buffers;
    flatbuffers::FlatBufferBuilder builder;
    auto tensor_offset = ttnn::to_flatbuffer(cpu_tensor, builder, buffers);
    // To be able to read flatbuffer data with `mmap` safely, make sure the serialized flatbuffer is aligned to at
    // least 8 bytes, just like `header_size`. Individual `buffers` are aligned according to their element size,
    // which is already what we need for `mmap` to work.
    builder.Align(kFlatbufferAlignment);
    builder.Finish(tensor_offset);

    const uint64_t header_size = builder.GetSize();
    safe_fwrite_bytes(&header_size, sizeof(header_size), output_file, file_name, "tensor header size");
    safe_fwrite_bytes(builder.GetBufferPointer(), header_size, output_file, file_name, "tensor header");

    for (const auto& buffer : buffers) {
        auto buffer_view = buffer.view_bytes();
        TT_FATAL(!buffer_view.empty(), "Unexpected empty buffer during tensor serialization");
        safe_fwrite_bytes(buffer_view.data(), buffer_view.size(), output_file, file_name, "tensor data");
    }

    TT_FATAL(fflush(output_file) == 0, "Failed to flush \"{}\": errno={} \"{}\"", file_name, errno, strerror(errno));

    if (mode == DumpTensorMode::DISTRIBUTED_GATHER) {
        const auto& ctx = distributed::multihost::DistributedContext::get_current_world();
        ctx->barrier();
    }
}

}  // namespace

void dump_tensor_flatbuffer(const std::string& file_name, const Tensor& tensor, DumpTensorMode mode) {
    dump_tensor_flatbuffer_impl(file_name, tensor, mode);
}

Tensor coalesce_tensorbins(const std::vector<std::string>& input_file_names) {
    TT_FATAL(!input_file_names.empty(), "coalesce_tensorbins requires at least one input file");

    std::vector<TensorbinInput> inputs;
    inputs.reserve(input_file_names.size());
    for (const auto& file_name : input_file_names) {
        inputs.push_back(read_tensorbin_metadata(file_name));
    }

    const auto& first = inputs.front();
    const auto& first_spec = first.metadata.tensor_spec;
    TT_FATAL(
        first_spec.data_type() == DataType::BFLOAT4_B,
        "coalesce_tensorbins supports BFLOAT4_B tensorbins only, but \"{}\" has a different dtype",
        first.file_name);
    const auto& first_shape = first_spec.logical_shape();
    TT_FATAL(first_shape.rank() > 0, "Cannot concatenate rank-0 tensorbins");

    uint64_t concatenated_dim0 = 0;
    for (const auto& input : inputs) {
        const auto& spec = input.metadata.tensor_spec;
        const auto& shape = spec.logical_shape();
        TT_FATAL(
            spec.data_type() == DataType::BFLOAT4_B,
            "coalesce_tensorbins supports BFLOAT4_B tensorbins only, but \"{}\" has a different dtype",
            input.file_name);
        TT_FATAL(
            spec.tensor_layout() == first_spec.tensor_layout(),
            "Tensor layout, tile, dtype, or memory config mismatch in \"{}\"",
            input.file_name);
        TT_FATAL(shape.rank() == first_shape.rank(), "Tensor rank mismatch in \"{}\"", input.file_name);
        TT_FATAL(shape[0] != 0, "Dimension 0 must be non-zero in \"{}\"", input.file_name);
        for (size_t dim = 1; dim < shape.rank(); ++dim) {
            TT_FATAL(
                shape[dim] == first_shape[dim],
                "Tensor shape mismatch in \"{}\" at dimension {} ({} != {})",
                input.file_name,
                dim,
                shape[dim],
                first_shape[dim]);
        }
        TT_FATAL(
            input.metadata.mesh_shape == first.metadata.mesh_shape,
            "Distributed mesh shape mismatch in \"{}\"",
            input.file_name);
        TT_FATAL(
            input.metadata.tensor_topology == first.metadata.tensor_topology,
            "Distributed tensor topology mismatch in \"{}\"",
            input.file_name);
        TT_FATAL(
            input.metadata.shards.size() == first.metadata.shards.size(),
            "Shard count mismatch in \"{}\"",
            input.file_name);
        TT_FATAL(
            input.alias_pattern == first.alias_pattern,
            "Replicated shard alias geometry mismatch in \"{}\"",
            input.file_name);
        for (size_t shard_index = 0; shard_index < input.metadata.shards.size(); ++shard_index) {
            TT_FATAL(
                input.metadata.shards[shard_index].mesh_coordinate ==
                    first.metadata.shards[shard_index].mesh_coordinate,
                "Shard mesh coordinate mismatch in \"{}\" at shard {}",
                input.file_name,
                shard_index);
            TT_FATAL(
                static_cast<unsigned __int128>(input.metadata.shards[shard_index].size) * first_shape[0] ==
                    static_cast<unsigned __int128>(first.metadata.shards[shard_index].size) * shape[0],
                "Shard byte geometry in \"{}\" is incompatible with dimension 0 at shard {}",
                input.file_name,
                shard_index);
        }
        TT_FATAL(
            shape[0] <= std::numeric_limits<uint32_t>::max() - concatenated_dim0,
            "Concatenated dimension 0 exceeds the maximum supported shape");
        concatenated_dim0 += shape[0];
    }

    auto output_shape = first_shape;
    output_shape[0] = static_cast<uint32_t>(concatenated_dim0);

    auto output_layout = first_spec.tensor_layout();
    const auto& input_memory_config = first_spec.memory_config();
    TT_FATAL(
        !input_memory_config.created_with_nd_shard_spec() &&
            input_memory_config.memory_layout() != TensorMemoryLayout::ND_SHARDED,
        "coalesce_tensorbins does not support ND-sharded tensorbins");
    if (input_memory_config.is_sharded()) {
        TT_FATAL(
            input_memory_config.shard_spec().has_value(),
            "Legacy-sharded tensorbin memory config must contain a ShardSpec");
        auto output_shard_spec = *input_memory_config.shard_spec();
        const auto scaled_shard_height = static_cast<unsigned __int128>(output_shard_spec.shape[0]) * concatenated_dim0;
        TT_FATAL(
            scaled_shard_height % first_shape[0] == 0 &&
                scaled_shard_height / first_shape[0] <= std::numeric_limits<uint32_t>::max(),
            "Coalesced legacy shard height exceeds the supported range");
        output_shard_spec.shape[0] = static_cast<uint32_t>(scaled_shard_height / first_shape[0]);

        MemoryConfig output_memory_config(
            input_memory_config.memory_layout(), input_memory_config.buffer_type(), std::move(output_shard_spec));
        if (experimental::per_core_allocation::is_per_core_allocation(input_memory_config)) {
            experimental::per_core_allocation::set_per_core_allocation(output_memory_config, true);
        }
        output_layout = TensorLayout::restore_from_serialized(
            first_spec.data_type(),
            first_spec.page_config(),
            output_memory_config,
            first_spec.tensor_layout().get_alignment());
    }

    TensorSpec output_spec(output_shape, std::move(output_layout));
    const uint64_t output_shard_size = compute_bfloat4_packed_size(output_spec);
    TT_FATAL(
        output_shard_size % sizeof(uint32_t) == 0,
        "BFLOAT4_B coalesced shard size must be aligned to uint32_t storage");
    uint64_t output_payload_size = 0;
    for (const size_t canonical_index : first.canonical_shards) {
        uint64_t concatenated_shard_size = 0;
        for (const auto& input : inputs) {
            const uint64_t size = input.metadata.shards[canonical_index].size;
            TT_FATAL(
                size <= std::numeric_limits<uint64_t>::max() - concatenated_shard_size,
                "Coalesced shard size exceeds the supported limit");
            concatenated_shard_size += size;
        }
        TT_FATAL(
            concatenated_shard_size == output_shard_size,
            "Concatenated shard size {} does not match output tensor spec size {}; dimension 0 must have compatible "
            "packing/alignment",
            concatenated_shard_size,
            output_shard_size);
        TT_FATAL(
            output_shard_size <= std::numeric_limits<uint64_t>::max() - output_payload_size,
            "Coalesced tensor payload exceeds the supported limit");
        output_payload_size += output_shard_size;
    }
    TT_FATAL(
        output_payload_size <= std::numeric_limits<size_t>::max(), "Coalesced tensor payload is too large to allocate");
    auto output_payload = std::shared_ptr<uint32_t>(
        new uint32_t[output_payload_size / sizeof(uint32_t)], std::default_delete<uint32_t[]>());
    auto* output_bytes = reinterpret_cast<std::byte*>(output_payload.get());
    std::vector<uint64_t> canonical_output_offsets(first.metadata.shards.size());
    uint64_t output_offset = 0;
    for (const size_t canonical_index : first.canonical_shards) {
        canonical_output_offsets[canonical_index] = output_offset;
        output_offset += output_shard_size;
    }
    auto shard_write_offsets = canonical_output_offsets;
    for (const auto& input : inputs) {
        int input_fd = open(input.file_name.c_str(), O_RDONLY | O_CLOEXEC);
        TT_FATAL(
            input_fd != -1,
            "Cannot reopen \"{}\" while coalescing: errno={} \"{}\"",
            input.file_name,
            errno,
            strerror(errno));
        auto input_cleanup = ttsl::make_cleanup([input_fd]() { close(input_fd); });
        for (const size_t canonical_index : first.canonical_shards) {
            const auto& input_shard = input.metadata.shards[canonical_index];
            safe_pread_bytes(
                input_fd,
                output_bytes + shard_write_offsets[canonical_index],
                input_shard.size,
                input.data_offset + input_shard.offset,
                input.file_name);
            shard_write_offsets[canonical_index] += input_shard.size;
        }
    }
    for (const size_t canonical_index : first.canonical_shards) {
        TT_FATAL(
            shard_write_offsets[canonical_index] == canonical_output_offsets[canonical_index] + output_shard_size,
            "Coalesced shard {} was not filled exactly",
            canonical_index);
    }

    auto distributed_buffer = DistributedHostBuffer::create(
        first.metadata.mesh_shape,
        first.metadata.mesh_shape,
        distributed::MeshCoordinate::zero_coordinate(first.metadata.mesh_shape.dims()),
        /*context=*/nullptr);
    for (size_t shard_index = 0; shard_index < first.metadata.shards.size(); ++shard_index) {
        const size_t canonical_index = first.alias_pattern[shard_index];
        const uint64_t shard_offset = canonical_output_offsets[canonical_index];
        const auto coordinate = first.metadata.shards[shard_index].mesh_coordinate;
        distributed_buffer.emplace_shard(coordinate, [output_payload, shard_offset, output_shard_size]() {
            return HostBuffer(
                ttsl::Span<uint32_t>(
                    output_payload.get() + shard_offset / sizeof(uint32_t), output_shard_size / sizeof(uint32_t)),
                MemoryPin(output_payload));
        });
    }
    return Tensor(
        HostTensor::from_buffer(std::move(distributed_buffer), std::move(output_spec), first.metadata.tensor_topology));
}

Tensor alias_coalesced_tensor(const Tensor& packed_device_tensor, const Tensor& template_host_tensor) {
    TT_FATAL(
        packed_device_tensor.storage_type() == StorageType::DEVICE,
        "alias_coalesced_tensor requires a device tensor as its first argument");
    TT_FATAL(
        template_host_tensor.storage_type() == StorageType::HOST,
        "alias_coalesced_tensor requires a host tensor as its template");
    TT_FATAL(packed_device_tensor.is_allocated(), "Packed device tensor must be allocated");

    const auto& packed_spec = packed_device_tensor.tensor_spec();
    const auto& template_spec = template_host_tensor.tensor_spec();
    TT_FATAL(
        packed_spec.data_type() == template_spec.data_type() &&
            packed_spec.page_config() == template_spec.page_config() &&
            packed_spec.tensor_layout().get_alignment() == template_spec.tensor_layout().get_alignment(),
        "Packed and template tensors must have identical dtype, page config, tile, and alignment");
    TT_FATAL(
        packed_device_tensor.tensor_topology() == template_host_tensor.tensor_topology(),
        "Packed and template tensors must have identical distributed topology");

    const auto& packed_shape = packed_spec.logical_shape();
    const auto& template_shape = template_spec.logical_shape();
    TT_FATAL(packed_shape.rank() == template_shape.rank(), "Packed and template tensor ranks must match");
    TT_FATAL(template_shape.rank() > 0 && template_shape[0] != 0, "Template tensor dimension 0 must be non-zero");
    TT_FATAL(
        packed_shape[0] >= template_shape[0] && packed_shape[0] % template_shape[0] == 0,
        "Packed tensor dimension 0 must be an integer multiple of the template");
    for (size_t dim = 1; dim < packed_shape.rank(); ++dim) {
        TT_FATAL(
            packed_shape[dim] == template_shape[dim], "Packed and template tensor shapes differ at dimension {}", dim);
    }

    const auto& packed_memory = packed_spec.memory_config();
    const auto& template_memory = template_spec.memory_config();
    TT_FATAL(
        packed_memory.memory_layout() == template_memory.memory_layout() &&
            packed_memory.buffer_type() == template_memory.buffer_type(),
        "Packed and template tensors must have identical memory layout and buffer type");
    TT_FATAL(
        !packed_memory.created_with_nd_shard_spec() && !template_memory.created_with_nd_shard_spec() &&
            packed_memory.memory_layout() != TensorMemoryLayout::ND_SHARDED,
        "alias_coalesced_tensor does not support ND-sharded tensors");
    if (packed_memory.is_sharded()) {
        TT_FATAL(
            packed_memory.shard_spec().has_value() && template_memory.shard_spec().has_value(),
            "Legacy-sharded packed and template tensors must contain ShardSpec");
        const auto& packed_shard = *packed_memory.shard_spec();
        const auto& template_shard = *template_memory.shard_spec();
        TT_FATAL(
            packed_shard.grid == template_shard.grid && packed_shard.orientation == template_shard.orientation &&
                packed_shard.shape[1] == template_shard.shape[1] && packed_shard.shape[0] >= template_shard.shape[0] &&
                packed_shard.shape[0] % template_shard.shape[0] == 0,
            "Packed and template tensors have incompatible legacy shard geometry");
    } else {
        TT_FATAL(
            packed_memory == template_memory,
            "Unsharded packed and template tensors must have identical memory config");
    }

    const auto& packed_buffer = packed_device_tensor.device_storage().get_mesh_buffer();
    TT_FATAL(
        packed_buffer.global_layout() == distributed::MeshBufferLayout::REPLICATED,
        "Packed tensor must use replicated MeshBuffer storage");
    const size_t template_shard_bytes = template_spec.compute_packed_buffer_size_bytes();
    const size_t packed_shard_bytes = packed_buffer.device_local_size();
    TT_FATAL(
        template_shard_bytes != 0 && packed_shard_bytes >= template_shard_bytes &&
            packed_shard_bytes % template_shard_bytes == 0,
        "Packed per-device shard size {} must be an integer multiple of template size {}",
        packed_shard_bytes,
        template_shard_bytes);

    const auto& packed_local_config = packed_buffer.device_local_config();
    distributed::DeviceLocalBufferConfig alias_local_config{
        .page_size = template_spec.compute_page_size_bytes(),
        .buffer_type = template_memory.buffer_type(),
        .sharding_args = template_spec.compute_buffer_sharding_args(),
        .bottom_up = packed_local_config.bottom_up,
        .sub_device_id = packed_local_config.sub_device_id,
    };
    const distributed::ReplicatedBufferConfig alias_global_config{
        .size = static_cast<DeviceAddr>(template_shard_bytes),
    };
    auto alias_mesh_buffer = distributed::MeshBuffer::create(
        alias_global_config, alias_local_config, packed_buffer.device(), packed_buffer.address());
    TT_FATAL(
        alias_mesh_buffer->address() == packed_buffer.address(),
        "Coalesced tensor alias address {} does not match packed address {}",
        alias_mesh_buffer->address(),
        packed_buffer.address());

    MeshTensor alias_mesh_tensor =
        MeshTensor::from_buffer(std::move(*alias_mesh_buffer), template_spec, template_host_tensor.tensor_topology());
    DeviceStorage alias_storage(packed_device_tensor.device_storage(), std::move(alias_mesh_tensor));
    Tensor alias(std::move(alias_storage));
    TT_FATAL(
        alias.device_storage().get_mesh_buffer().address() == packed_buffer.address(),
        "Returned alias does not preserve packed base address");
    return alias;
}

Tensor load_tensor_flatbuffer(const std::string& file_name, distributed::MeshDevice* device) {
    int fd = open(file_name.c_str(), O_RDONLY | O_CLOEXEC);
    TT_FATAL(fd != -1, "Cannot open \"{}\": errno={} \"{}\"", file_name, errno, strerror(errno));
    auto cleanup = ttsl::make_cleanup([fd]() { close(fd); });

    struct stat file_stat{};
    TT_FATAL(fstat(fd, &file_stat) == 0, "Failed to get file stats for \"{}\"", file_name);
    size_t file_size = file_stat.st_size;
    TT_FATAL(file_size >= sizeof(uint64_t), "Tensor file \"{}\" is too small to be valid", file_name);

    // Mmap the file to read tensor data lazily.
    void* mmap_addr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    TT_FATAL(mmap_addr != MAP_FAILED, "Failed to mmap file \"{}\": {}", file_name, strerror(errno));

    std::shared_ptr<void> mmap_ptr(mmap_addr, [file_size](void* addr) { munmap(addr, file_size); });
    MemoryPin memory_pin(mmap_ptr);

    auto* file_data = static_cast<std::byte*>(mmap_addr);
    uint64_t header_size = 0;
    std::memcpy(&header_size, file_data, sizeof(header_size));
    TT_FATAL(
        sizeof(header_size) + header_size <= file_size,
        "Tensor file \"{}\" is truncated or corrupt (header_size={}, file_size={})",
        file_name,
        header_size,
        file_size);

    const auto* header_start = reinterpret_cast<const std::uint8_t*>(file_data) + sizeof(header_size);
    TT_FATAL(
        header_size < flatbuffers::Verifier::Options().max_size,
        "Tensor header size is too large; this most likely indicates data corruption.");
    flatbuffers::Verifier verifier(header_start, header_size);
    TT_FATAL(
        ttnn::flatbuffer::VerifyTensorBuffer(verifier),
        "Cannot validate tensor data; this most likely indicates data corruption.");
    const auto* fb_tensor = ttnn::flatbuffer::GetTensor(header_start);

    const uint64_t data_offset = sizeof(header_size) + header_size;
    const uint64_t data_size = file_size - data_offset;

    std::byte* data_region = file_data + data_offset;
    TT_FATAL(
        (reinterpret_cast<uintptr_t>(data_region) & (kFlatbufferAlignment - 1)) == 0,
        "Tensor data pointer must be 8-byte aligned!");

    Tensor tensor = ttnn::from_flatbuffer(fb_tensor, ttsl::Span<std::byte>(data_region, data_size), memory_pin);
    if (device != nullptr) {
        tensor = tensor.to_device(device, tensor.tensor_spec().memory_config());
    }
    return tensor;
}

}  // namespace tt::tt_metal
