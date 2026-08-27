// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/wavelet/device/wavelet_program_utils.hpp"

#include <filesystem>
#include <limits>
#include <utility>

#include <tt_stl/assert.hpp>

#include "tt-metalium/buffer.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/program_descriptors.hpp"
#include "tt-metalium/workload_descriptor.hpp"

namespace ttnn::prim::wavelet_program_utils {
namespace {

struct UploadedMetadataOwner {
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> buffer;
    std::vector<uint32_t> payload;
};

}  // namespace

uint32_t checked_u32(const size_t value, const char* label) {
    TT_FATAL(
        value <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()), "{} {} overflows uint32_t", label, value);
    return static_cast<uint32_t>(value);
}

uint32_t worker_core_count(tt::tt_metal::distributed::MeshDevice& mesh_device, const char* empty_grid_error) {
    const auto grid = mesh_device.compute_with_storage_grid_size();
    const uint32_t core_count = static_cast<uint32_t>(grid.x * grid.y);
    TT_FATAL(core_count > 0, "{}", empty_grid_error);
    return core_count;
}

std::vector<tt::tt_metal::CoreCoord> select_row_major_cores(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const uint32_t active_core_count,
    const char* invalid_core_count_error) {
    const auto grid = mesh_device.compute_with_storage_grid_size();
    TT_FATAL(
        active_core_count > 0 && active_core_count <= static_cast<uint32_t>(grid.x * grid.y),
        "{}",
        invalid_core_count_error);
    return tt::tt_metal::grid_to_cores(
        active_core_count, static_cast<uint32_t>(grid.x), static_cast<uint32_t>(grid.y), true);
}

tt::tt_metal::CoreRangeSet core_range_set(const std::vector<tt::tt_metal::CoreCoord>& cores) {
    std::vector<tt::tt_metal::CoreRange> ranges;
    ranges.reserve(cores.size());
    for (const auto& core : cores) {
        ranges.emplace_back(core);
    }
    return tt::tt_metal::CoreRangeSet(std::move(ranges)).merge_ranges();
}

std::vector<CoreChunkWork> partition_chunk_work(
    const std::vector<tt::tt_metal::CoreCoord>& cores, const uint32_t chunk_count, const char* operation_name) {
    TT_FATAL(!cores.empty(), "{} chunk partition requires cores", operation_name);
    TT_FATAL(chunk_count >= cores.size(), "{} active core count exceeds chunk count", operation_name);

    const uint32_t core_count = checked_u32(cores.size(), "Wavelet core count");
    const uint32_t base_chunks = chunk_count / core_count;
    const uint32_t extra_chunks = chunk_count % core_count;
    uint32_t chunk_begin = 0;
    std::vector<CoreChunkWork> work;
    work.reserve(cores.size());
    for (uint32_t core_index = 0; core_index < core_count; ++core_index) {
        const uint32_t count = base_chunks + (core_index < extra_chunks ? 1U : 0U);
        work.push_back(CoreChunkWork{
            .core = cores[core_index],
            .chunk_begin = chunk_begin,
            .chunk_count = count,
        });
        chunk_begin += count;
    }
    TT_FATAL(chunk_begin == chunk_count, "{} chunk partition is incomplete", operation_name);
    return work;
}

std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_replicated_dram_pages(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const size_t physical_page_count, const uint32_t page_bytes) {
    TT_FATAL(physical_page_count > 0, "Wavelet DRAM buffer requires at least one page");
    return tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{
            .size = static_cast<uint64_t>(physical_page_count) * page_bytes,
        },
        tt::tt_metal::distributed::DeviceLocalBufferConfig{
            .page_size = page_bytes,
            .buffer_type = tt::tt_metal::BufferType::DRAM,
            .bottom_up = false,
        },
        &mesh_device);
}

std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> upload_replicated_dram_metadata(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const size_t physical_page_count,
    const uint32_t page_bytes,
    std::vector<uint32_t> payload,
    tt::tt_metal::WorkloadDescriptor& workload,
    const char* payload_name) {
    auto buffer = create_replicated_dram_pages(mesh_device, physical_page_count, page_bytes);
    const size_t physical_words = static_cast<size_t>(buffer->get_backing_buffer()->size()) / sizeof(uint32_t);
    TT_FATAL(
        payload.size() <= physical_words,
        "{} has {} words but its device buffer holds only {}",
        payload_name,
        payload.size(),
        physical_words);
    payload.resize(physical_words, 0);

    auto owner = std::make_shared<UploadedMetadataOwner>(UploadedMetadataOwner{
        .buffer = buffer,
        .payload = std::move(payload),
    });
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
        mesh_device.mesh_command_queue(), owner->buffer, owner->payload, false);
    workload.buffers.push_back({owner, buffer->get_backing_buffer()});
    return buffer;
}

void add_generated_scheme_include_path(tt::tt_metal::KernelDescriptor& descriptor) {
    const std::filesystem::path include_root = TTNN_WAVELET_GENERATED_INCLUDE_ROOT;
    if (std::filesystem::is_directory(include_root)) {
        descriptor.compiler_include_paths.push_back(include_root);
    }
}

void append_program_to_mesh_ranges(
    tt::tt_metal::WorkloadDescriptor& workload,
    tt::tt_metal::ProgramDescriptor descriptor,
    const MeshCoordinateRangeSet& tensor_coords,
    const char* empty_range_error) {
    const auto& ranges = tensor_coords.ranges();
    TT_FATAL(!ranges.empty(), "{}", empty_range_error);
    for (size_t index = 0; index + 1 < ranges.size(); ++index) {
        workload.programs.push_back({ranges[index], descriptor});
    }
    workload.programs.push_back({ranges.back(), std::move(descriptor)});
}

}  // namespace ttnn::prim::wavelet_program_utils
