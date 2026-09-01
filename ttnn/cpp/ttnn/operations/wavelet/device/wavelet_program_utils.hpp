// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "tt-metalium/core_coord.hpp"
#include "ttnn/distributed/types.hpp"

namespace tt::tt_metal {

struct KernelDescriptor;
struct ProgramDescriptor;
struct WorkloadDescriptor;

namespace distributed {

class MeshBuffer;
class MeshDevice;

}  // namespace distributed

}  // namespace tt::tt_metal

namespace ttnn::prim::wavelet_program_utils {

struct CoreChunkWork {
    tt::tt_metal::CoreCoord core;
    uint32_t chunk_begin{0};
    uint32_t chunk_count{0};
};

[[nodiscard]] uint32_t checked_u32(size_t value, const char* label);

[[nodiscard]] uint32_t worker_core_count(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const char* empty_grid_error);

[[nodiscard]] std::vector<tt::tt_metal::CoreCoord> select_row_major_cores(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    uint32_t active_core_count,
    const char* invalid_core_count_error);

[[nodiscard]] tt::tt_metal::CoreRangeSet core_range_set(const std::vector<tt::tt_metal::CoreCoord>& cores);

[[nodiscard]] std::vector<CoreChunkWork> partition_chunk_work(
    const std::vector<tt::tt_metal::CoreCoord>& cores, uint32_t chunk_count, const char* operation_name);

[[nodiscard]] std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_replicated_dram_pages(
    tt::tt_metal::distributed::MeshDevice& mesh_device, size_t physical_page_count, uint32_t page_bytes);

[[nodiscard]] std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> upload_replicated_dram_metadata(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    size_t physical_page_count,
    uint32_t page_bytes,
    std::vector<uint32_t> payload,
    tt::tt_metal::WorkloadDescriptor& workload,
    const char* payload_name);

void add_generated_scheme_include_path(tt::tt_metal::KernelDescriptor& descriptor);

void append_program_to_mesh_ranges(
    tt::tt_metal::WorkloadDescriptor& workload,
    tt::tt_metal::ProgramDescriptor descriptor,
    const MeshCoordinateRangeSet& tensor_coords,
    const char* empty_range_error);

}  // namespace ttnn::prim::wavelet_program_utils
