// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/wavelet/device/wavelet_1d_operation_impl.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tt_stl/assert.hpp>
#include <utility>
#include <vector>

#include "tt-logger/tt-logger.hpp"
#include "tt-metalium/allocator.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/circular_buffer_constants.h"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/program_descriptors.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/tile.hpp"
#include "tt-metalium/workload_descriptor.hpp"
#include "ttnn/operations/wavelet/common/storage_contract.hpp"
#include "ttnn/operations/wavelet/common/wavelet_host.hpp"
#include "ttnn/operations/wavelet/device/protocol/lwt_config.hpp"
#include "ttnn/operations/wavelet/device/wavelet_program_utils.hpp"
#include "ttnn/operations/wavelet/device/wavelet_tensor_validation.hpp"
#include "ttnn/operations/wavelet/generated/wavelet_schemes/scheme_dispatch.hpp"
#include "ttnn/operations/wavelet/planner/inverse_plan.hpp"
#include "ttnn/operations/wavelet/planner/l1_accounting.hpp"
#include "ttnn/operations/wavelet/planner/policy.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

using namespace operations::wavelet;
using wavelet_program_utils::add_generated_scheme_include_path;
using wavelet_program_utils::checked_u32;
using wavelet_program_utils::core_range_set;
using wavelet_program_utils::CoreChunkWork;
using wavelet_tensor_validation::validate_input_memory_config;
using wavelet_tensor_validation::validate_output_memory_config;

namespace {

constexpr tt::DataFormat kDataFormat = tt::DataFormat::Float32;

constexpr const char* kLwtReaderKernelPath = "ttnn/cpp/ttnn/operations/wavelet/device/kernels/dataflow/lwt_reader.cpp";
constexpr const char* kLwtWriterKernelPath = "ttnn/cpp/ttnn/operations/wavelet/device/kernels/dataflow/lwt_writer.cpp";
constexpr const char* kLwtComputeKernelPath = "ttnn/cpp/ttnn/operations/wavelet/device/kernels/compute/lwt_compute.cpp";

constexpr uint32_t kSrcTile0Cb = tt::CBIndex::c_0;
constexpr uint32_t kSrcTile1Cb = tt::CBIndex::c_1;
constexpr uint32_t kBaseTileCb = tt::CBIndex::c_2;
constexpr uint32_t kOutputCb = tt::CBIndex::c_16;
constexpr uint32_t kSrcCacheCb = tt::CBIndex::c_3;
constexpr uint32_t kInterleaveCb = tt::CBIndex::c_4;
constexpr uint32_t kSyncCb = tt::CBIndex::c_5;
constexpr uint32_t kReaderConfigCb = tt::CBIndex::c_6;
constexpr uint32_t kWriterConfigCb = tt::CBIndex::c_7;
constexpr uint32_t kWorkspaceACb = tt::CBIndex::c_8;
constexpr uint32_t kWorkspaceBCb = tt::CBIndex::c_9;
constexpr uint32_t kWorkspaceScratchCb = tt::CBIndex::c_10;
constexpr uint32_t kTileGroupBuffering = 2;
constexpr uint32_t kIlwtInterleaveBatchSticks = 96;
constexpr uint32_t kAlignedNocMaxRouteCount = 5;
constexpr uint32_t kAlignedNocMinGroupsPerChunk = 2;
constexpr uint32_t kWormholeSingleGroupLwtTileMinRouteCount = 4;
constexpr uint32_t kWormholeSingleGroupIlwtTileMinRouteCount = 7;
constexpr uint32_t kDefaultL1SignalBudgetBytes = 768 * 1024;

static_assert(
    kIlwtInterleaveBatchSticks <= device_protocol::kIlwtGroupOutputElements / kStickWidth,
    "ILWT interleave batch must fit in one output group");
static_assert(static_cast<uint32_t>(StorageSlot::kA) == 0);
static_assert(static_cast<uint32_t>(StorageSlot::kB) == 1);
static_assert(static_cast<uint32_t>(StorageSlot::kScratch) == 2);

struct LwtWorkingBuffers {
    tt::tt_metal::Buffer* final_even{};
    tt::tt_metal::Buffer* final_odd{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> route_config;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> chunk_config;
    std::vector<tt::tt_metal::CoreCoord> cores;

    [[nodiscard]] uint32_t slot_id(const StorageSlot slot) const noexcept { return static_cast<uint32_t>(slot); }
};

struct IlwtWorkingBuffers {
    tt::tt_metal::Buffer* output{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> route_config;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> chunk_config;
    std::vector<tt::tt_metal::CoreCoord> cores;

    [[nodiscard]] uint32_t slot_id(const StorageSlot slot) const noexcept { return static_cast<uint32_t>(slot); }
};

struct Logical1DShape {
    uint32_t batch_count{1};
    uint32_t length{0};
    bool rank_four{false};
};

[[nodiscard]] Logical1DShape logical_1d_shape(const Tensor& tensor, const char* tensor_name) {
    const auto& shape = tensor.logical_shape();
    if (shape.rank() == 2) {
        TT_FATAL(
            shape[1] == kStickWidth,
            "{} stick-native rank-2 shape requires W == {}, got {}",
            tensor_name,
            kStickWidth,
            shape[1]);
        return Logical1DShape{
            .batch_count = 1,
            .length = checked_u32(static_cast<uint64_t>(shape[0]) * shape[1], tensor_name),
            .rank_four = false,
        };
    }
    if (shape.rank() == 1) {
        return Logical1DShape{
            .batch_count = 1,
            .length = checked_u32(shape[0], tensor_name),
            .rank_four = false,
        };
    }
    TT_FATAL(
        shape.rank() == 4,
        "{} must have shape [S,32], [W], [B,1,S,32], or [B,1,1,W], got rank {}",
        tensor_name,
        shape.rank());
    TT_FATAL(shape[0] > 0, "{} batch dimension must be positive", tensor_name);
    TT_FATAL(shape[1] == 1, "{} requires C == 1, got {}", tensor_name, shape[1]);
    if (shape[3] == kStickWidth && shape[2] > 1) {
        return Logical1DShape{
            .batch_count = checked_u32(shape[0], "1D wavelet batch count"),
            .length = checked_u32(static_cast<uint64_t>(shape[2]) * shape[3], tensor_name),
            .rank_four = true,
        };
    }
    TT_FATAL(shape[2] == 1, "{} requires H == 1, got {}", tensor_name, shape[2]);
    return Logical1DShape{
        .batch_count = checked_u32(shape[0], "1D wavelet batch count"),
        .length = checked_u32(shape[3], tensor_name),
        .rank_four = true,
    };
}

[[nodiscard]] Logical1DShape logical_1d_signal_shape(const Tensor& tensor, const char* tensor_name) {
    const auto& shape = tensor.logical_shape();
    if (shape.rank() == 1) {
        return Logical1DShape{
            .batch_count = 1,
            .length = checked_u32(shape[0], tensor_name),
            .rank_four = false,
        };
    }
    TT_FATAL(shape.rank() == 4, "{} must have shape [W] or [B,1,1,W], got rank {}", tensor_name, shape.rank());
    TT_FATAL(shape[0] > 0, "{} batch dimension must be positive", tensor_name);
    TT_FATAL(shape[1] == 1, "{} requires C == 1, got {}", tensor_name, shape[1]);
    TT_FATAL(shape[2] == 1, "{} requires H == 1, got {}", tensor_name, shape[2]);
    return Logical1DShape{
        .batch_count = checked_u32(shape[0], "1D wavelet batch count"),
        .length = checked_u32(shape[3], tensor_name),
        .rank_four = true,
    };
}

[[nodiscard]] uint32_t pages_per_batch_item(const Tensor& tensor, const uint32_t batch_count, const char* tensor_name) {
    TT_FATAL(batch_count > 0, "{} batch count must be positive", tensor_name);
    const uint64_t physical_bytes = static_cast<uint64_t>(tensor.physical_volume()) * sizeof(float);
    TT_FATAL(physical_bytes % batch_count == 0, "{} physical volume is not divisible by its batch count", tensor_name);
    const uint64_t bytes_per_batch = physical_bytes / batch_count;
    const uint32_t page_size = tensor.buffer()->page_size();
    TT_FATAL(
        page_size > 0 && bytes_per_batch % page_size == 0,
        "{} physical batch stride {} bytes is not page aligned to {} bytes",
        tensor_name,
        bytes_per_batch,
        page_size);
    return checked_u32(bytes_per_batch / page_size, "1D wavelet pages per batch item");
}

[[nodiscard]] uint32_t ilwt_interleave_batch_sticks(const tt::ARCH architecture) {
    return architecture == tt::ARCH::WORMHOLE_B0 ? kIlwtInterleaveBatchSticks : 1U;
}

[[nodiscard]] bool supports_hybrid_tile_mirror(const tt::ARCH architecture, const WorkspaceLayout layout) {
    return architecture == tt::ARCH::WORMHOLE_B0 && layout == WorkspaceLayout::kRowMajor;
}

[[nodiscard]] uint32_t tile_mirror_elements(const uint32_t workspace_elements, const bool hybrid_tile_mirror) {
    return hybrid_tile_mirror
               ? checked_u32(
                     ceil_div(workspace_elements, static_cast<uint32_t>(device_protocol::kLwtGroupOutputElements)) *
                         device_protocol::kLwtGroupOutputElements,
                     "hybrid tile mirror elements")
               : 0U;
}

[[nodiscard]] uint32_t output_group_count(const size_t output_length) {
    return checked_u32(
        ceil_div(output_length, static_cast<size_t>(device_protocol::kLwtGroupOutputElements)), "LWT group count");
}

[[nodiscard]] uint32_t planner_signal_budget_bytes(
    const uint32_t available_bytes,
    const ArchitecturePolicy& policy,
    const bool hybrid_tile_mirror,
    const uint32_t interleave_batch_sticks) {
    const uint64_t fixed_bytes =
        checked_l1_allocation_bytes(0, 0, 0, interleave_batch_sticks, policy.l1_scratch_bytes, available_bytes);
    constexpr uint64_t mirror_rounding_reserve =
        uint64_t{3} * (device_protocol::kLwtGroupOutputElements - 1U) * sizeof(float);
    const uint64_t physical_workspace_multiplier = hybrid_tile_mirror ? 2U : 1U;
    const uint64_t rounding_reserve = hybrid_tile_mirror ? mirror_rounding_reserve : 0U;
    TT_FATAL(
        available_bytes >= fixed_bytes + rounding_reserve + 3 * device_protocol::kStickBytes,
        "LWT requires at least {} bytes of free per-core L1 after external L1 tensor allocation, but only {} remain",
        fixed_bytes + rounding_reserve + 3 * device_protocol::kStickBytes,
        available_bytes);
    const uint64_t capacity_limited_budget =
        (available_bytes - fixed_bytes - rounding_reserve) / physical_workspace_multiplier;
    return checked_u32(std::min<uint64_t>(kDefaultL1SignalBudgetBytes, capacity_limited_budget), "LWT signal budget");
}

[[nodiscard]] std::optional<WorkspaceLayout> workspace_layout_override() { return std::nullopt; }

[[nodiscard]] bool prefer_tile_native_workspace(const LwtExecutionPlan& plan, const tt::ARCH architecture) {
    TT_FATAL(!plan.chunks.empty(), "LWT workspace selection requires at least one chunk");
    if (architecture == tt::ARCH::BLACKHOLE) {
        return true;
    }

    if (architecture == tt::ARCH::WORMHOLE_B0 && plan.chunks.size() > 1 && plan.groups_per_chunk == 1 &&
        plan.chunks.front().routes.size() >= kWormholeSingleGroupLwtTileMinRouteCount) {
        return true;
    }

    uint32_t predict_update_count = 0;
    uint32_t aligned_base_count = 0;
    for (const auto& route : plan.chunks.front().routes) {
        if (!is_predict_update_step(route.type)) {
            continue;
        }
        ++predict_update_count;
        aligned_base_count += route.base_offset_elements == 0 ? 1U : 0U;
    }
    return predict_update_count > 0 && 2U * aligned_base_count >= predict_update_count;
}

[[nodiscard]] bool prefer_tile_native_inverse_workspace(const IlwtExecutionPlan& plan, const tt::ARCH architecture) {
    TT_FATAL(!plan.chunks.empty(), "ILWT workspace selection requires at least one chunk");
    if (architecture == tt::ARCH::BLACKHOLE) {
        const uint32_t route_count = checked_u32(plan.chunks.front().routes.size(), "ILWT route count");
        const bool row_major_crossover = (plan.output_groups_per_chunk >= 2 && route_count <= 2) ||
                                         (plan.output_groups_per_chunk >= 3 && route_count <= 3);
        return !row_major_crossover;
    }
    return architecture == tt::ARCH::WORMHOLE_B0 && plan.chunks.size() > 1 && plan.output_groups_per_chunk == 1 &&
           plan.chunks.front().routes.size() >= kWormholeSingleGroupIlwtTileMinRouteCount;
}

template <typename Plan>
[[nodiscard]] bool prefer_aligned_row_major_noc_staging(
    const Plan& plan, const uint32_t groups_per_chunk, const bool hybrid_tile_mirror) {
    TT_FATAL(!plan.chunks.empty(), "LWT NoC staging selection requires at least one chunk");
    return hybrid_tile_mirror && groups_per_chunk >= kAlignedNocMinGroupsPerChunk &&
           plan.chunks.front().routes.size() <= kAlignedNocMaxRouteCount;
}

void add_circular_buffer(
    tt::tt_metal::ProgramDescriptor& descriptor,
    const tt::tt_metal::CoreRangeSet& cores,
    const uint32_t cb_index,
    const uint32_t entry_count,
    const uint32_t page_bytes) {
    descriptor.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = entry_count * page_bytes,
        .core_ranges = cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_index),
            .data_format = kDataFormat,
            .page_size = page_bytes,
        }}},
    });
}

void add_workspace_circular_buffers(
    tt::tt_metal::ProgramDescriptor& descriptor,
    const tt::tt_metal::CoreRangeSet& cores,
    const uint32_t workspace_elements,
    const bool hybrid_tile_mirror) {
    TT_FATAL(workspace_elements > 0, "LWT workspace must contain at least one element");
    const uint32_t mirror_elements = tile_mirror_elements(workspace_elements, hybrid_tile_mirror);
    const uint32_t slot_elements =
        checked_u32(static_cast<size_t>(workspace_elements) + mirror_elements, "hybrid workspace slot elements");
    TT_FATAL(
        slot_elements % kStickWidth == 0,
        "LWT workspace length {} is not a multiple of the {}-element stick width",
        slot_elements,
        kStickWidth);
    const uint32_t slot_sticks = slot_elements / kStickWidth;
    for (const uint32_t cb_index : {kWorkspaceACb, kWorkspaceBCb, kWorkspaceScratchCb}) {
        add_circular_buffer(descriptor, cores, cb_index, slot_sticks, device_protocol::kStickBytes);
    }
}

void add_narrow_tile_circular_buffer(
    tt::tt_metal::ProgramDescriptor& descriptor,
    const tt::tt_metal::CoreRangeSet& cores,
    const uint32_t cb_index,
    const uint32_t entry_count) {
    descriptor.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = entry_count * device_protocol::kLwtNarrowTileBytes,
        .core_ranges = cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_index),
            .data_format = kDataFormat,
            .page_size = device_protocol::kLwtNarrowTileBytes,
            .tile = tt::tt_metal::TileDescriptor{32, 16, false},
        }}},
    });
}

[[nodiscard]] uint32_t resolve_output_address(const LwtWorkingBuffers& buffers, const RouteOutputRef output) {
    switch (output.storage) {
        case RouteOutputStorage::kWorkspaceSlot: return buffers.slot_id(output.slot);
        case RouteOutputStorage::kFinalEvenDram: return 0;
        case RouteOutputStorage::kFinalOddDram: return 1;
    }
    TT_THROW("Unsupported LWT output storage");
}

[[nodiscard]] uint32_t resolve_workspace_address(const LwtWorkingBuffers& buffers, const StreamRef stream) {
    return buffers.slot_id(stream.slot);
}

[[nodiscard]] std::vector<uint32_t> build_chunk_config_words(const LwtExecutionPlan& plan) {
    std::vector<uint32_t> words(std::max(plan.chunks.size(), size_t{1}) * device_protocol::kLwtChunkConfigWordCount, 0);
    for (size_t chunk_index = 0; chunk_index < plan.chunks.size(); ++chunk_index) {
        const auto& chunk = plan.chunks[chunk_index];
        const size_t offset = chunk_index * device_protocol::kLwtChunkConfigWordCount;
        words[offset + device_protocol::kLwtInitialEvenBegin] =
            checked_u32(chunk.initial_even.begin, "initial even begin");
        words[offset + device_protocol::kLwtInitialEvenLength] =
            checked_u32(chunk.initial_even.length(), "initial even length");
        words[offset + device_protocol::kLwtInitialOddBegin] =
            checked_u32(chunk.initial_odd.begin, "initial odd begin");
        words[offset + device_protocol::kLwtInitialOddLength] =
            checked_u32(chunk.initial_odd.length(), "initial odd length");
    }
    return words;
}

[[nodiscard]] std::vector<uint32_t> build_route_config_words(
    const LwtExecutionPlan& plan, const LwtWorkingBuffers& buffers) {
    TT_FATAL(!plan.chunks.empty(), "LWT plan has no chunks");
    const size_t route_count = plan.chunks.front().routes.size();
    std::vector<uint32_t> words(
        std::max(plan.chunks.size() * route_count, size_t{1}) * device_protocol::kRouteConfigWordCount, 0);

    for (size_t chunk_index = 0; chunk_index < plan.chunks.size(); ++chunk_index) {
        const auto& chunk = plan.chunks[chunk_index];
        std::array<bool, 3> tile_mirror_valid{};
        TT_FATAL(chunk.routes.size() == route_count, "LWT chunks have inconsistent route counts");
        for (size_t route_index = 0; route_index < route_count; ++route_index) {
            const auto& route = chunk.routes[route_index];
            const uint32_t output_offset = checked_u32(route.output_offset_elements, "LWT output offset");
            const size_t word_offset =
                (chunk_index * route_count + route_index) * device_protocol::kRouteConfigWordCount;
            words[word_offset + device_protocol::kRouteType] = static_cast<uint32_t>(route.type);
            words[word_offset + device_protocol::kRouteSourceAddr] = resolve_workspace_address(buffers, route.source);
            words[word_offset + device_protocol::kRouteSourceLength] =
                checked_u32(route.source_storage_length, "LWT source storage end");
            words[word_offset + device_protocol::kRouteBaseAddr] = resolve_workspace_address(buffers, route.base);
            words[word_offset + device_protocol::kRouteBaseLength] =
                checked_u32(route.base_storage_length, "LWT base storage end");
            words[word_offset + device_protocol::kRouteOutputAddr] = resolve_output_address(buffers, route.output);
            words[word_offset + device_protocol::kRouteOutputLength] =
                checked_u32(route.output_length, "LWT output length");
            words[word_offset + device_protocol::kRouteSourceOffset] =
                checked_u32(route.source_offset_elements, "LWT source offset");
            words[word_offset + device_protocol::kRouteBaseOffset] =
                checked_u32(route.base_offset_elements, "LWT base offset");
            words[word_offset + device_protocol::kRouteSourceLeftPad] = route.source_left_pad_elements;
            words[word_offset + device_protocol::kRouteOutputOffset] = output_offset;
            words[word_offset + device_protocol::kRouteGroupCount] = output_group_count(route.output_length);
            uint32_t route_flags = 0;
            if (route.output.storage == RouteOutputStorage::kFinalEvenDram) {
                route_flags = device_protocol::kRouteFlagFinalDram | device_protocol::kRouteFlagFinalEven;
            } else if (route.output.storage == RouteOutputStorage::kFinalOddDram) {
                route_flags = device_protocol::kRouteFlagFinalDram | device_protocol::kRouteFlagFinalOdd;
            }
            route_flags |= tile_mirror_valid[static_cast<size_t>(route.source.slot)]
                               ? device_protocol::kRouteFlagSourceTileMirror
                               : 0U;
            route_flags |= tile_mirror_valid[static_cast<size_t>(route.base.slot)]
                               ? device_protocol::kRouteFlagBaseTileMirror
                               : 0U;
            if (route.output.storage == RouteOutputStorage::kWorkspaceSlot) {
                const bool produces_tile_mirror = output_offset % device_protocol::kLwtGroupOutputElements == 0;
                route_flags |= produces_tile_mirror ? device_protocol::kRouteFlagOutputTileMirror : 0U;
                tile_mirror_valid[static_cast<size_t>(route.output.slot)] = produces_tile_mirror;
            }
            words[word_offset + device_protocol::kRouteFlags] = route_flags;
        }
    }
    return words;
}

[[nodiscard]] tt::tt_metal::KernelDescriptor::RTArgList reader_runtime_args(
    const LwtExecutionPlan& plan,
    const LwtWorkingBuffers& buffers,
    const tt::tt_metal::Buffer& input_buffer,
    const CoreChunkWork& work,
    const uint32_t chunks_per_sample,
    const uint32_t input_pages_per_sample) {
    tt::tt_metal::KernelDescriptor::RTArgList args;
    args.reserve(10);
    args.push_back(const_cast<tt::tt_metal::Buffer*>(&input_buffer));
    args.push_back(checked_u32(plan.full_plan.preprocess_layout.input.length, "LWT input length"));
    args.push_back(plan.full_plan.preprocess_layout.pad_config.left);
    args.push_back(buffers.slot_id(StorageSlot::kA));
    args.push_back(buffers.slot_id(StorageSlot::kB));
    args.push_back(buffers.chunk_config->get_backing_buffer());
    args.push_back(buffers.route_config->get_backing_buffer());
    args.push_back(work.chunk_begin);
    args.push_back(work.chunk_count);
    args.push_back(checked_u32(plan.chunks.front().routes.size(), "LWT route count"));
    args.push_back(checked_u32(plan.workspace_elements * sizeof(float), "LWT tile mirror offset"));
    args.push_back(chunks_per_sample);
    args.push_back(input_pages_per_sample);
    return args;
}

[[nodiscard]] tt::tt_metal::KernelDescriptor::RTArgList writer_runtime_args(
    const LwtExecutionPlan& plan,
    const LwtWorkingBuffers& buffers,
    const CoreChunkWork& work,
    const uint32_t chunks_per_sample,
    const uint32_t output_pages_per_sample) {
    tt::tt_metal::KernelDescriptor::RTArgList args;
    args.reserve(6);
    args.push_back(buffers.route_config->get_backing_buffer());
    args.push_back(work.chunk_begin);
    args.push_back(work.chunk_count);
    args.push_back(checked_u32(plan.chunks.front().routes.size(), "LWT route count"));
    args.push_back(buffers.final_even);
    args.push_back(buffers.final_odd);
    args.push_back(checked_u32(plan.workspace_elements * sizeof(float), "LWT tile mirror offset"));
    args.push_back(chunks_per_sample);
    args.push_back(output_pages_per_sample);
    return args;
}

[[nodiscard]] std::vector<uint32_t> compute_runtime_args(const LwtExecutionPlan& plan, const CoreChunkWork& work) {
    const size_t route_count = plan.chunks.front().routes.size();
    std::vector<uint32_t> args;
    args.reserve(1 + static_cast<size_t>(work.chunk_count) * route_count);
    args.push_back(work.chunk_count);
    for (uint32_t local_chunk = 0; local_chunk < work.chunk_count; ++local_chunk) {
        const auto& chunk = plan.chunks[(work.chunk_begin + local_chunk) % plan.chunks.size()];
        for (const auto& route : chunk.routes) {
            args.push_back(output_group_count(route.output_length));
        }
    }
    return args;
}

[[nodiscard]] tt::tt_metal::ProgramDescriptor create_forward_program_descriptor(
    const tt::tt_metal::CoreRangeSet& cores,
    const tt::tt_metal::Buffer& input_buffer,
    const LwtWorkingBuffers& buffers,
    const LwtExecutionPlan& plan,
    const WorkspaceLayout workspace_layout,
    const bool hybrid_tile_mirror,
    const bool row_major_noc_staging,
    const BoundaryMode boundary_mode,
    const char* compute_scheme_header,
    const char* compute_scheme_type,
    const std::vector<CoreChunkWork>& work,
    const uint32_t chunks_per_sample,
    const uint32_t input_pages_per_sample,
    const uint32_t output_pages_per_sample,
    const uint32_t l1_alignment_bytes) {
    tt::tt_metal::ProgramDescriptor descriptor;
    add_narrow_tile_circular_buffer(descriptor, cores, kSrcTile0Cb, 2 * kTileGroupBuffering);
    add_narrow_tile_circular_buffer(descriptor, cores, kSrcTile1Cb, 2 * kTileGroupBuffering);
    add_narrow_tile_circular_buffer(descriptor, cores, kBaseTileCb, 3 * kTileGroupBuffering);
    add_narrow_tile_circular_buffer(descriptor, cores, kOutputCb, 3 * kTileGroupBuffering);
    add_circular_buffer(
        descriptor, cores, kSrcCacheCb, device_protocol::kLwtCacheStickCount, device_protocol::kStickBytes);
    add_circular_buffer(descriptor, cores, kInterleaveCb, 1, device_protocol::kStickBytes);
    add_circular_buffer(descriptor, cores, kSyncCb, 1, l1_alignment_bytes);
    add_circular_buffer(descriptor, cores, kReaderConfigCb, 1, device_protocol::kRouteConfigPageBytes);
    add_circular_buffer(descriptor, cores, kWriterConfigCb, 1, device_protocol::kRouteConfigPageBytes);
    add_workspace_circular_buffers(descriptor, cores, plan.workspace_elements, hybrid_tile_mirror);

    const auto& config_buffer = *buffers.route_config->get_backing_buffer();

    std::vector<uint32_t> reader_compile_args = {
        kReaderConfigCb,
        kSrcTile0Cb,
        kSrcTile1Cb,
        kBaseTileCb,
        kSrcCacheCb,
        kSyncCb,
        static_cast<uint32_t>(workspace_layout == WorkspaceLayout::kTileNative),
        0U,
        static_cast<uint32_t>(boundary_mode),
        input_buffer.page_size(),
        static_cast<uint32_t>(row_major_noc_staging),
        static_cast<uint32_t>(hybrid_tile_mirror),
        kWorkspaceACb,
        kWorkspaceBCb,
        kWorkspaceScratchCb,
    };
    tt::tt_metal::TensorAccessorArgs(config_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_args);

    std::vector<uint32_t> writer_compile_args = {
        kWriterConfigCb,
        kOutputCb,
        kSyncCb,
        1U,
        static_cast<uint32_t>(workspace_layout == WorkspaceLayout::kTileNative),
        0U,
        kInterleaveCb,
        buffers.final_even->page_size(),
        1U,
        static_cast<uint32_t>(hybrid_tile_mirror),
        kWorkspaceACb,
        kWorkspaceBCb,
        kWorkspaceScratchCb,
    };
    tt::tt_metal::TensorAccessorArgs(config_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(*buffers.final_even).append_to(writer_compile_args);

    const std::vector<uint32_t> compute_compile_args = {kSrcTile0Cb, kSrcTile1Cb, kBaseTileCb, kOutputCb};
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    unpack_to_dest_mode[kSrcTile0Cb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kSrcTile1Cb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kBaseTileCb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;

    tt::tt_metal::KernelDescriptor reader_descriptor;
    reader_descriptor.kernel_source = kLwtReaderKernelPath;
    reader_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_descriptor.core_ranges = cores;
    reader_descriptor.compile_time_args = std::move(reader_compile_args);
    reader_descriptor.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor writer_descriptor;
    writer_descriptor.kernel_source = kLwtWriterKernelPath;
    writer_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_descriptor.core_ranges = cores;
    writer_descriptor.compile_time_args = std::move(writer_compile_args);
    writer_descriptor.config = tt::tt_metal::WriterConfigDescriptor{};

    tt::tt_metal::KernelDescriptor compute_descriptor;
    compute_descriptor.kernel_source = kLwtComputeKernelPath;
    compute_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_descriptor.core_ranges = cores;
    compute_descriptor.compile_time_args = compute_compile_args;
    compute_descriptor.defines = {
        {"LWT_SCHEME_HEADER", compute_scheme_header},
        {"LWT_SCHEME_TYPE", compute_scheme_type},
        {"LWT_INLINE_TERMINAL_SCALE", "1"},
    };
    add_generated_scheme_include_path(compute_descriptor);
    compute_descriptor.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .unpack_to_dest_mode = unpack_to_dest_mode,
    };

    for (const auto& core_work : work) {
        reader_descriptor.emplace_runtime_args(
            core_work.core,
            reader_runtime_args(plan, buffers, input_buffer, core_work, chunks_per_sample, input_pages_per_sample));
        tt::tt_metal::KernelDescriptor::RTArgList compute_args;
        compute_args.append(compute_runtime_args(plan, core_work));
        compute_descriptor.emplace_runtime_args(core_work.core, compute_args);
        writer_descriptor.emplace_runtime_args(
            core_work.core, writer_runtime_args(plan, buffers, core_work, chunks_per_sample, output_pages_per_sample));
    }

    descriptor.kernels.push_back(std::move(reader_descriptor));
    descriptor.kernels.push_back(std::move(compute_descriptor));
    descriptor.kernels.push_back(std::move(writer_descriptor));
    return descriptor;
}

[[nodiscard]] uint32_t resolve_workspace_address(const IlwtWorkingBuffers& buffers, const StreamRef stream) {
    return buffers.slot_id(stream.slot);
}

[[nodiscard]] std::vector<uint32_t> build_inverse_chunk_config_words(
    const IlwtExecutionPlan& plan, const IlwtWorkingBuffers& buffers) {
    std::vector<uint32_t> words(std::max(plan.chunks.size(), size_t{1}) * device_protocol::kLwtChunkConfigWordCount, 0);
    for (size_t chunk_index = 0; chunk_index < plan.chunks.size(); ++chunk_index) {
        const auto& chunk = plan.chunks[chunk_index];
        const size_t offset = chunk_index * device_protocol::kLwtChunkConfigWordCount;
        words[offset + device_protocol::kIlwtApproximationBegin] =
            checked_u32(chunk.canonical_approximation.begin, "ILWT approximation begin");
        words[offset + device_protocol::kIlwtApproximationLength] =
            checked_u32(chunk.canonical_approximation.length(), "ILWT approximation length");
        words[offset + device_protocol::kIlwtDetailBegin] =
            checked_u32(chunk.canonical_detail.begin, "ILWT detail begin");
        words[offset + device_protocol::kIlwtDetailLength] =
            checked_u32(chunk.canonical_detail.length(), "ILWT detail length");
        words[offset + device_protocol::kIlwtFinalEvenAddr] = resolve_workspace_address(buffers, chunk.final_even);
        words[offset + device_protocol::kIlwtFinalEvenStorageLength] =
            checked_u32(chunk.final_even_storage_length, "ILWT final even storage length");
        words[offset + device_protocol::kIlwtFinalEvenOffset] =
            checked_u32(chunk.final_even_offset_elements, "ILWT final even offset");
        words[offset + device_protocol::kIlwtFinalEvenBegin] =
            checked_u32(chunk.reconstructed_even.begin, "ILWT final even begin");
        words[offset + device_protocol::kIlwtFinalOddAddr] = resolve_workspace_address(buffers, chunk.final_odd);
        words[offset + device_protocol::kIlwtFinalOddStorageLength] =
            checked_u32(chunk.final_odd_storage_length, "ILWT final odd storage length");
        words[offset + device_protocol::kIlwtFinalOddOffset] =
            checked_u32(chunk.final_odd_offset_elements, "ILWT final odd offset");
        words[offset + device_protocol::kIlwtFinalOddBegin] =
            checked_u32(chunk.reconstructed_odd.begin, "ILWT final odd begin");
        words[offset + device_protocol::kIlwtOutputBegin] = checked_u32(chunk.output_signal.begin, "ILWT output begin");
        words[offset + device_protocol::kIlwtOutputLength] =
            checked_u32(chunk.output_signal.length(), "ILWT output length");
    }
    return words;
}

[[nodiscard]] std::vector<uint32_t> build_inverse_route_config_words(
    const IlwtExecutionPlan& plan, const IlwtWorkingBuffers& buffers) {
    TT_FATAL(!plan.chunks.empty(), "ILWT plan has no chunks");
    const size_t route_count = plan.chunks.front().routes.size();
    std::vector<uint32_t> words(
        std::max(plan.chunks.size() * route_count, size_t{1}) * device_protocol::kRouteConfigWordCount, 0);
    for (size_t chunk_index = 0; chunk_index < plan.chunks.size(); ++chunk_index) {
        const auto& chunk = plan.chunks[chunk_index];
        std::array<bool, 3> tile_mirror_valid{};
        TT_FATAL(chunk.routes.size() == route_count, "ILWT chunks have inconsistent route counts");
        for (size_t route_index = 0; route_index < route_count; ++route_index) {
            const auto& route = chunk.routes[route_index];
            TT_FATAL(
                route.output.storage == RouteOutputStorage::kWorkspaceSlot,
                "ILWT intermediate route must target a local workspace slot");
            const size_t word_offset =
                (chunk_index * route_count + route_index) * device_protocol::kRouteConfigWordCount;
            words[word_offset + device_protocol::kRouteType] = static_cast<uint32_t>(route.type);
            words[word_offset + device_protocol::kRouteSourceAddr] = resolve_workspace_address(buffers, route.source);
            words[word_offset + device_protocol::kRouteSourceLength] =
                checked_u32(route.source_storage_length, "ILWT source storage length");
            words[word_offset + device_protocol::kRouteBaseAddr] = resolve_workspace_address(buffers, route.base);
            words[word_offset + device_protocol::kRouteBaseLength] =
                checked_u32(route.base_storage_length, "ILWT base storage length");
            words[word_offset + device_protocol::kRouteOutputAddr] =
                resolve_workspace_address(buffers, StreamRef{.slot = route.output.slot});
            words[word_offset + device_protocol::kRouteOutputLength] =
                checked_u32(route.output_length, "ILWT output length");
            words[word_offset + device_protocol::kRouteSourceOffset] =
                checked_u32(route.source_offset_elements, "ILWT source offset");
            words[word_offset + device_protocol::kRouteBaseOffset] =
                checked_u32(route.base_offset_elements, "ILWT base offset");
            words[word_offset + device_protocol::kRouteSourceLeftPad] = route.source_left_pad_elements;
            words[word_offset + device_protocol::kRouteOutputOffset] = 0;
            words[word_offset + device_protocol::kRouteGroupCount] = output_group_count(route.output_length);
            uint32_t route_flags = plan.final_interleave_direct && route_index + 1 == route_count
                                       ? device_protocol::kRouteFlagIlwtFinalInterleave
                                       : 0U;
            route_flags |= tile_mirror_valid[static_cast<size_t>(route.source.slot)]
                               ? device_protocol::kRouteFlagSourceTileMirror
                               : 0U;
            route_flags |= tile_mirror_valid[static_cast<size_t>(route.base.slot)]
                               ? device_protocol::kRouteFlagBaseTileMirror
                               : 0U;
            route_flags |= device_protocol::kRouteFlagOutputTileMirror;
            tile_mirror_valid[static_cast<size_t>(route.output.slot)] = true;
            words[word_offset + device_protocol::kRouteFlags] = route_flags;
        }
    }
    return words;
}

[[nodiscard]] tt::tt_metal::KernelDescriptor::RTArgList inverse_reader_runtime_args(
    const IlwtExecutionPlan& plan,
    const IlwtWorkingBuffers& buffers,
    const tt::tt_metal::Buffer& approximation_buffer,
    const tt::tt_metal::Buffer& detail_buffer,
    const CoreChunkWork& work,
    const uint32_t chunks_per_sample,
    const uint32_t input_pages_per_sample) {
    tt::tt_metal::KernelDescriptor::RTArgList args;
    args.reserve(10);
    args.push_back(const_cast<tt::tt_metal::Buffer*>(&approximation_buffer));
    args.push_back(const_cast<tt::tt_metal::Buffer*>(&detail_buffer));
    args.push_back(checked_u32(plan.full_plan.coefficient_length, "ILWT coefficient length"));
    args.push_back(buffers.slot_id(StorageSlot::kA));
    args.push_back(buffers.slot_id(StorageSlot::kB));
    args.push_back(buffers.chunk_config->get_backing_buffer());
    args.push_back(buffers.route_config->get_backing_buffer());
    args.push_back(work.chunk_begin);
    args.push_back(work.chunk_count);
    args.push_back(checked_u32(plan.chunks.front().routes.size(), "ILWT route count"));
    args.push_back(checked_u32(plan.workspace_elements * sizeof(float), "ILWT tile mirror offset"));
    args.push_back(chunks_per_sample);
    args.push_back(input_pages_per_sample);
    return args;
}

[[nodiscard]] tt::tt_metal::KernelDescriptor::RTArgList inverse_writer_runtime_args(
    const IlwtExecutionPlan& plan,
    const IlwtWorkingBuffers& buffers,
    const CoreChunkWork& work,
    const uint32_t chunks_per_sample,
    const uint32_t output_pages_per_sample) {
    tt::tt_metal::KernelDescriptor::RTArgList args;
    args.reserve(7);
    args.push_back(buffers.route_config->get_backing_buffer());
    args.push_back(work.chunk_begin);
    args.push_back(work.chunk_count);
    args.push_back(checked_u32(plan.chunks.front().routes.size(), "ILWT route count"));
    args.push_back(buffers.chunk_config->get_backing_buffer());
    args.push_back(buffers.output);
    args.push_back(plan.full_plan.forward_trace.preprocess_layout.pad_config.left);
    args.push_back(checked_u32(plan.workspace_elements * sizeof(float), "ILWT tile mirror offset"));
    args.push_back(chunks_per_sample);
    args.push_back(output_pages_per_sample);
    return args;
}

[[nodiscard]] std::vector<uint32_t> inverse_compute_runtime_args(
    const IlwtExecutionPlan& plan, const CoreChunkWork& work) {
    const size_t route_count = plan.chunks.front().routes.size();
    std::vector<uint32_t> args;
    args.reserve(1 + static_cast<size_t>(work.chunk_count) * route_count);
    args.push_back(work.chunk_count);
    for (uint32_t local_chunk = 0; local_chunk < work.chunk_count; ++local_chunk) {
        const auto& chunk = plan.chunks[(work.chunk_begin + local_chunk) % plan.chunks.size()];
        for (const auto& route : chunk.routes) {
            args.push_back(output_group_count(route.output_length));
        }
    }
    return args;
}

[[nodiscard]] tt::tt_metal::ProgramDescriptor create_inverse_program_descriptor(
    const tt::tt_metal::CoreRangeSet& cores,
    const tt::tt_metal::Buffer& approximation_buffer,
    const tt::tt_metal::Buffer& detail_buffer,
    const IlwtWorkingBuffers& buffers,
    const IlwtExecutionPlan& plan,
    const WorkspaceLayout workspace_layout,
    const bool hybrid_tile_mirror,
    const bool row_major_noc_staging,
    const uint32_t interleave_batch_sticks,
    const char* compute_scheme_header,
    const char* compute_scheme_type,
    const std::vector<CoreChunkWork>& work,
    const uint32_t chunks_per_sample,
    const uint32_t input_pages_per_sample,
    const uint32_t output_pages_per_sample,
    const uint32_t l1_alignment_bytes) {
    tt::tt_metal::ProgramDescriptor descriptor;
    add_narrow_tile_circular_buffer(descriptor, cores, kSrcTile0Cb, 2 * kTileGroupBuffering);
    add_narrow_tile_circular_buffer(descriptor, cores, kSrcTile1Cb, 2 * kTileGroupBuffering);
    add_narrow_tile_circular_buffer(descriptor, cores, kBaseTileCb, 3 * kTileGroupBuffering);
    add_narrow_tile_circular_buffer(descriptor, cores, kOutputCb, 3 * kTileGroupBuffering);
    add_circular_buffer(
        descriptor, cores, kSrcCacheCb, device_protocol::kLwtCacheStickCount, device_protocol::kStickBytes);
    add_circular_buffer(descriptor, cores, kInterleaveCb, interleave_batch_sticks, device_protocol::kStickBytes);
    add_circular_buffer(descriptor, cores, kSyncCb, 1, l1_alignment_bytes);
    add_circular_buffer(descriptor, cores, kReaderConfigCb, 1, device_protocol::kRouteConfigPageBytes);
    add_circular_buffer(descriptor, cores, kWriterConfigCb, 1, device_protocol::kRouteConfigPageBytes);
    add_workspace_circular_buffers(descriptor, cores, plan.workspace_elements, hybrid_tile_mirror);

    const auto& config_buffer = *buffers.route_config->get_backing_buffer();
    std::vector<uint32_t> reader_compile_args = {
        kReaderConfigCb,
        kSrcTile0Cb,
        kSrcTile1Cb,
        kBaseTileCb,
        kSrcCacheCb,
        kSyncCb,
        static_cast<uint32_t>(workspace_layout == WorkspaceLayout::kTileNative),
        1U,
        static_cast<uint32_t>(BoundaryMode::kSymmetric),
        approximation_buffer.page_size(),
        static_cast<uint32_t>(row_major_noc_staging),
        static_cast<uint32_t>(hybrid_tile_mirror),
        kWorkspaceACb,
        kWorkspaceBCb,
        kWorkspaceScratchCb,
    };
    tt::tt_metal::TensorAccessorArgs(config_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(approximation_buffer).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(detail_buffer).append_to(reader_compile_args);

    std::vector<uint32_t> writer_compile_args = {
        kWriterConfigCb,
        kOutputCb,
        kSyncCb,
        1U,
        static_cast<uint32_t>(workspace_layout == WorkspaceLayout::kTileNative),
        1U,
        kInterleaveCb,
        buffers.output->page_size(),
        interleave_batch_sticks,
        static_cast<uint32_t>(hybrid_tile_mirror),
        kWorkspaceACb,
        kWorkspaceBCb,
        kWorkspaceScratchCb,
    };
    tt::tt_metal::TensorAccessorArgs(config_buffer).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(*buffers.output).append_to(writer_compile_args);

    const std::vector<uint32_t> compute_compile_args = {kSrcTile0Cb, kSrcTile1Cb, kBaseTileCb, kOutputCb};
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    unpack_to_dest_mode[kSrcTile0Cb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kSrcTile1Cb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[kBaseTileCb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;

    tt::tt_metal::KernelDescriptor reader_descriptor;
    reader_descriptor.kernel_source = kLwtReaderKernelPath;
    reader_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_descriptor.core_ranges = cores;
    reader_descriptor.compile_time_args = std::move(reader_compile_args);
    reader_descriptor.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor writer_descriptor;
    writer_descriptor.kernel_source = kLwtWriterKernelPath;
    writer_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_descriptor.core_ranges = cores;
    writer_descriptor.compile_time_args = std::move(writer_compile_args);
    writer_descriptor.config = tt::tt_metal::WriterConfigDescriptor{};

    tt::tt_metal::KernelDescriptor compute_descriptor;
    compute_descriptor.kernel_source = kLwtComputeKernelPath;
    compute_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_descriptor.core_ranges = cores;
    compute_descriptor.compile_time_args = compute_compile_args;
    compute_descriptor.defines = {
        {"ILWT_SCHEME_HEADER", compute_scheme_header},
        {"ILWT_SCHEME_TYPE", compute_scheme_type},
        {"ILWT_INLINE_INVERSE_SCALE", "1"},
    };
    add_generated_scheme_include_path(compute_descriptor);
    compute_descriptor.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .unpack_to_dest_mode = unpack_to_dest_mode,
    };

    for (const auto& core_work : work) {
        reader_descriptor.emplace_runtime_args(
            core_work.core,
            inverse_reader_runtime_args(
                plan,
                buffers,
                approximation_buffer,
                detail_buffer,
                core_work,
                chunks_per_sample,
                input_pages_per_sample));
        tt::tt_metal::KernelDescriptor::RTArgList compute_args;
        compute_args.append(inverse_compute_runtime_args(plan, core_work));
        compute_descriptor.emplace_runtime_args(core_work.core, compute_args);
        writer_descriptor.emplace_runtime_args(
            core_work.core,
            inverse_writer_runtime_args(plan, buffers, core_work, chunks_per_sample, output_pages_per_sample));
    }

    descriptor.kernels.push_back(std::move(reader_descriptor));
    descriptor.kernels.push_back(std::move(compute_descriptor));
    descriptor.kernels.push_back(std::move(writer_descriptor));
    return descriptor;
}

}  // namespace

namespace {

void validate_1d_tensor(const Tensor& tensor, const char* tensor_name) {
    wavelet_tensor_validation::validate_device_tensor(tensor, tensor_name);
    TT_FATAL(tensor.layout() == Layout::ROW_MAJOR, "{} must use ROW_MAJOR layout", tensor_name);
    const Logical1DShape shape = logical_1d_shape(tensor, tensor_name);
    TT_FATAL(shape.length > 0, "{} must be non-empty", tensor_name);
    validate_input_memory_config(tensor.memory_config(), tensor_name);

    // Stick-native shapes expose their complete allocated capacity, including
    // unspecified final-stick lanes, so validate the complete physical volume.
    const uint64_t physical_bytes = static_cast<uint64_t>(tensor.physical_volume()) * sizeof(float);
    TT_FATAL(
        tensor.buffer()->size() >= physical_bytes,
        "{} physical buffer has {} bytes but the physical volume requires at least {} bytes",
        tensor_name,
        tensor.buffer()->size(),
        physical_bytes);
    static_cast<void>(make_architecture_policy(tensor.device()->arch()));
}

[[nodiscard]] tt::tt_metal::TensorSpec output_spec_1d(
    const Logical1DShape& input_shape, const uint32_t length, const MemoryConfig& memory_config) {
    const uint32_t stick_count = checked_u32(ceil_div(length, kStickWidth), "1D wavelet output stick count");
    const Shape output_shape = input_shape.rank_four ? Shape({input_shape.batch_count, 1, stick_count, kStickWidth})
                                                     : Shape({stick_count, kStickWidth});
    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            DataType::FLOAT32,
            tt::tt_metal::PageConfig(Layout::ROW_MAJOR),
            memory_config,
            tt::tt_metal::Alignment{kStickWidth}));
}

void validate_preallocated_output(
    const Tensor& output,
    const tt::tt_metal::TensorSpec& expected_spec,
    const tt::tt_metal::distributed::MeshDevice* expected_device,
    const char* output_name) {
    validate_1d_tensor(output, output_name);
    wavelet_tensor_validation::validate_preallocated_output_placement(output, expected_device, output_name);
    TT_FATAL(
        output.logical_shape() == expected_spec.logical_shape(),
        "{} logical shape does not match the wavelet output specification",
        output_name);
    TT_FATAL(
        output.tensor_spec().compute_page_size_bytes() == expected_spec.compute_page_size_bytes(),
        "{} page size {} does not match the required {} bytes",
        output_name,
        output.tensor_spec().compute_page_size_bytes(),
        expected_spec.compute_page_size_bytes());
    TT_FATAL(
        output.buffer()->size() >= expected_spec.compute_packed_buffer_size_bytes(),
        "{} buffer has {} bytes but requires at least {} bytes",
        output_name,
        output.buffer()->size(),
        expected_spec.compute_packed_buffer_size_bytes());
}

template <typename Scheme>
[[nodiscard]] LwtExecutionPlan make_forward_execution_plan(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const size_t input_length,
    const BoundaryMode boundary_mode,
    const uint32_t available_l1_bytes) {
    const SignalBuffer input{
        .length = input_length,
        .stick_width = kStickWidth,
        .element_size_bytes = sizeof(float),
    };
    LiftingForwardPlan full_plan = make_forward_lifting_plan<Scheme>(input, boundary_mode);
    TT_FATAL(
        full_plan.preprocess_layout.padded_length() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "LWT padded input length exceeds the device signed-index range");

    const uint32_t max_cores =
        wavelet_program_utils::worker_core_count(mesh_device, "LWT requires at least one hardware worker core");
    const ArchitecturePolicy architecture_policy = make_architecture_policy(mesh_device.arch());
    const std::optional<WorkspaceLayout> workspace_override = workspace_layout_override();
    const WorkspaceLayout initial_layout = workspace_override.value_or(WorkspaceLayout::kRowMajor);
    const bool initial_hybrid_tile_mirror =
        supports_hybrid_tile_mirror(architecture_policy.architecture, initial_layout);
    const uint32_t signal_budget_bytes =
        planner_signal_budget_bytes(available_l1_bytes, architecture_policy, initial_hybrid_tile_mirror, 1U);
    LwtExecutionPlan plan =
        make_lwt_execution_plan(std::move(full_plan), max_cores, signal_budget_bytes, initial_layout);
    const bool tile_native_preferred = prefer_tile_native_workspace(plan, architecture_policy.architecture);
    const bool hybrid_has_steady_state = plan.groups_per_chunk >= kAlignedNocMinGroupsPerChunk;
    if (!workspace_override.has_value() && tile_native_preferred &&
        (!initial_hybrid_tile_mirror || !hybrid_has_steady_state)) {
        plan = make_lwt_execution_plan(
            std::move(plan.full_plan), max_cores, signal_budget_bytes, WorkspaceLayout::kTileNative);
    }
    const bool hybrid_tile_mirror =
        supports_hybrid_tile_mirror(architecture_policy.architecture, plan.workspace_layout);
    static_cast<void>(checked_l1_allocation_bytes(
        plan.workspace_elements,
        plan.max_workspace_elements,
        tile_mirror_elements(plan.workspace_elements, hybrid_tile_mirror),
        1U,
        architecture_policy.l1_scratch_bytes,
        available_l1_bytes));
    return plan;
}

template <typename Scheme>
[[nodiscard]] IlwtExecutionPlan make_inverse_execution_plan(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const uint32_t original_length,
    const size_t coefficient_length,
    const BoundaryMode boundary_mode,
    const uint32_t available_l1_bytes) {
    const std::optional<WorkspaceLayout> workspace_override = workspace_layout_override();
    const ArchitecturePolicy architecture_policy = make_architecture_policy(mesh_device.arch(), workspace_override);
    const uint32_t interleave_batch_sticks = ilwt_interleave_batch_sticks(architecture_policy.architecture);
    const bool initial_hybrid_tile_mirror =
        supports_hybrid_tile_mirror(architecture_policy.architecture, architecture_policy.ilwt_layout);
    const uint32_t signal_budget_bytes = planner_signal_budget_bytes(
        available_l1_bytes, architecture_policy, initial_hybrid_tile_mirror, interleave_batch_sticks);
    TT_FATAL(architecture_policy.inverse_scale_inline, "ILWT must preserve inline FP32 inverse scaling");
    LiftingInversePlan full_plan =
        make_inverse_lifting_plan<Scheme>(original_length, coefficient_length, boundary_mode);
    TT_FATAL(
        full_plan.forward_trace.preprocess_layout.padded_length() <=
            static_cast<size_t>(std::numeric_limits<int32_t>::max()),
        "ILWT padded signal length exceeds the device signed-index range");
    IlwtExecutionPlan plan = make_ilwt_execution_plan(
        std::move(full_plan),
        wavelet_program_utils::worker_core_count(mesh_device, "LWT requires at least one hardware worker core"),
        signal_budget_bytes,
        architecture_policy.ilwt_layout,
        architecture_policy.final_interleave_direct);
    const WorkspaceLayout preferred_layout =
        prefer_tile_native_inverse_workspace(plan, architecture_policy.architecture) ? WorkspaceLayout::kTileNative
                                                                                     : WorkspaceLayout::kRowMajor;
    if (!workspace_override.has_value() && plan.workspace_layout != preferred_layout) {
        const ArchitecturePolicy preferred_policy =
            make_architecture_policy(architecture_policy.architecture, preferred_layout);
        plan = make_ilwt_execution_plan(
            std::move(plan.full_plan),
            wavelet_program_utils::worker_core_count(mesh_device, "LWT requires at least one hardware worker core"),
            signal_budget_bytes,
            preferred_layout,
            preferred_policy.final_interleave_direct);
    }
    const bool hybrid_tile_mirror =
        supports_hybrid_tile_mirror(architecture_policy.architecture, plan.workspace_layout);
    static_cast<void>(checked_l1_allocation_bytes(
        plan.workspace_elements,
        plan.max_workspace_elements,
        tile_mirror_elements(plan.workspace_elements, hybrid_tile_mirror),
        interleave_batch_sticks,
        architecture_policy.l1_scratch_bytes,
        available_l1_bytes));
    return plan;
}

template <typename Scheme>
[[nodiscard]] tt::tt_metal::WorkloadDescriptor build_forward_workload(
    const Lwt1DParams& operation_attributes,
    const Lwt1DInputs& tensor_args,
    std::tuple<Tensor, Tensor>& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords) {
    auto& mesh_device = *tensor_args.input.device();
    const auto& input_buffer = *tensor_args.input.buffer();
    const Logical1DShape input_shape = logical_1d_signal_shape(tensor_args.input, "DWT input");
    LwtExecutionPlan plan = make_forward_execution_plan<Scheme>(
        mesh_device, input_shape.length, operation_attributes.boundary_mode, operation_attributes.available_l1_bytes);
    const ArchitecturePolicy architecture_policy = make_architecture_policy(mesh_device.arch());
    const bool hybrid_tile_mirror =
        supports_hybrid_tile_mirror(architecture_policy.architecture, plan.workspace_layout);
    const bool row_major_noc_staging =
        prefer_aligned_row_major_noc_staging(plan, plan.groups_per_chunk, hybrid_tile_mirror);

    tt::tt_metal::WorkloadDescriptor workload;
    const uint32_t chunks_per_sample = checked_u32(plan.chunks.size(), "LWT chunks per sample");
    const uint32_t total_work_items =
        checked_u32(static_cast<size_t>(chunks_per_sample) * input_shape.batch_count, "LWT total batch work items");
    std::vector<tt::tt_metal::CoreCoord> cores = wavelet_program_utils::select_row_major_cores(
        mesh_device,
        std::min(
            wavelet_program_utils::worker_core_count(mesh_device, "LWT requires at least one hardware worker core"),
            total_work_items),
        "LWT active core count exceeds the worker grid");

    static_assert(executable_step_count<Scheme>() > 0, "DWT schemes require an executable terminal scale step");
    constexpr size_t expected_route_count = executable_step_count<Scheme>() - 1U;
    for (const auto& chunk : plan.chunks) {
        TT_FATAL(
            chunk.routes.size() == expected_route_count,
            "DWT planner produced {} routes, but the kernel ABI requires {}",
            chunk.routes.size(),
            expected_route_count);
    }
    const size_t route_count = plan.chunks.front().routes.size();
    auto route_config = wavelet_program_utils::create_replicated_dram_pages(
        mesh_device, std::max(plan.chunks.size() * route_count, size_t{1}), device_protocol::kRouteConfigPageBytes);
    auto chunk_config = wavelet_program_utils::create_replicated_dram_pages(
        mesh_device, std::max(plan.chunks.size(), size_t{1}), device_protocol::kLwtChunkConfigPageBytes);
    LwtWorkingBuffers buffers{
        .final_even = std::get<0>(tensor_return_value).buffer(),
        .final_odd = std::get<1>(tensor_return_value).buffer(),
        .route_config = route_config,
        .chunk_config = chunk_config,
        .cores = std::move(cores),
    };

    constexpr int32_t canonical_start = static_cast<int32_t>(Scheme::tap_size / 2);
    const int32_t final_even_delta = plan.full_plan.final_even_shift - canonical_start;
    const int32_t final_odd_delta = plan.full_plan.final_odd_shift - canonical_start;
    TT_FATAL(
        final_even_delta <= 0 && static_cast<int64_t>(plan.full_plan.final_even_length) + final_even_delta >=
                                     static_cast<int64_t>(plan.full_plan.output_length),
        "LWT approximation stream does not cover the canonical coefficient interval");
    TT_FATAL(
        final_odd_delta <= 0 && static_cast<int64_t>(plan.full_plan.final_odd_length) + final_odd_delta >=
                                    static_cast<int64_t>(plan.full_plan.output_length),
        "LWT detail stream does not cover the canonical coefficient interval");
    const std::vector<uint32_t> chunk_words = build_chunk_config_words(plan);
    const std::vector<uint32_t> route_words = build_route_config_words(plan, buffers);
    buffers.chunk_config = wavelet_program_utils::upload_replicated_dram_metadata(
        mesh_device,
        std::max(plan.chunks.size(), size_t{1}),
        device_protocol::kLwtChunkConfigPageBytes,
        chunk_words,
        workload,
        "Wavelet metadata payload");
    buffers.route_config = wavelet_program_utils::upload_replicated_dram_metadata(
        mesh_device,
        std::max(plan.chunks.size() * route_count, size_t{1}),
        device_protocol::kRouteConfigPageBytes,
        route_words,
        workload,
        "Wavelet metadata payload");

    const std::vector<CoreChunkWork> work =
        wavelet_program_utils::partition_chunk_work(buffers.cores, total_work_items, "LWT");
    const auto [min_work, max_work] = std::minmax_element(
        work.begin(), work.end(), [](const auto& lhs, const auto& rhs) { return lhs.chunk_count < rhs.chunk_count; });
    log_debug(
        tt::LogOp,
        "ttnn::dwt batch scheduler: B={}, chunks_per_sample={}, total_work_items={}, active_cores={}, "
        "work_items_per_core={}..{}, max_per_core_workspace_bytes={}, arch={}, scheme={}, layout={}, "
        "routes={}, groups_per_chunk={}, dependency_overhead={}, hybrid_tile_mirror={}, row_major_noc_staging={}",
        input_shape.batch_count,
        chunks_per_sample,
        total_work_items,
        buffers.cores.size(),
        min_work->chunk_count,
        max_work->chunk_count,
        3 *
            static_cast<uint64_t>(
                plan.workspace_elements + tile_mirror_elements(plan.workspace_elements, hybrid_tile_mirror)) *
            sizeof(float),
        static_cast<uint32_t>(architecture_policy.architecture),
        Scheme::name,
        static_cast<uint32_t>(plan.workspace_layout),
        route_count,
        plan.groups_per_chunk,
        plan.max_dependency_overhead,
        hybrid_tile_mirror,
        row_major_noc_staging);
    const uint32_t input_pages_per_sample =
        pages_per_batch_item(tensor_args.input, input_shape.batch_count, "LWT input");
    const uint32_t output_pages_per_sample =
        pages_per_batch_item(std::get<0>(tensor_return_value), input_shape.batch_count, "LWT approximation output");
    const uint32_t l1_alignment_bytes = mesh_device.allocator()->get_alignment(tt::tt_metal::BufferType::L1);
    auto descriptor = create_forward_program_descriptor(
        core_range_set(buffers.cores),
        input_buffer,
        buffers,
        plan,
        plan.workspace_layout,
        hybrid_tile_mirror,
        row_major_noc_staging,
        operation_attributes.boundary_mode,
        Scheme::compute_scheme_header,
        Scheme::compute_scheme_type,
        work,
        chunks_per_sample,
        input_pages_per_sample,
        output_pages_per_sample,
        l1_alignment_bytes);
    wavelet_program_utils::append_program_to_mesh_ranges(
        workload, std::move(descriptor), tensor_coords, "Wavelet workload has no mesh coordinate range");
    return workload;
}

template <typename Scheme>
[[nodiscard]] tt::tt_metal::WorkloadDescriptor build_inverse_workload(
    const Ilwt1DParams& operation_attributes,
    const Ilwt1DInputs& tensor_args,
    Tensor& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords) {
    auto& mesh_device = *tensor_args.approximation.device();
    const auto& approximation_buffer = *tensor_args.approximation.buffer();
    const auto& detail_buffer = *tensor_args.detail.buffer();
    const Logical1DShape coefficient_shape = logical_1d_shape(tensor_args.approximation, "IDWT approximation");
    IlwtExecutionPlan plan = make_inverse_execution_plan<Scheme>(
        mesh_device,
        operation_attributes.original_length,
        coefficient_shape.length,
        operation_attributes.boundary_mode,
        operation_attributes.available_l1_bytes);
    using InverseScheme = typename Scheme::inverse;
    const ArchitecturePolicy architecture_policy = make_architecture_policy(mesh_device.arch());
    const uint32_t interleave_batch_sticks = ilwt_interleave_batch_sticks(architecture_policy.architecture);
    const bool hybrid_tile_mirror =
        supports_hybrid_tile_mirror(architecture_policy.architecture, plan.workspace_layout);
    const bool row_major_noc_staging =
        prefer_aligned_row_major_noc_staging(plan, plan.output_groups_per_chunk, hybrid_tile_mirror);

    tt::tt_metal::WorkloadDescriptor workload;
    const uint32_t chunks_per_sample = checked_u32(plan.chunks.size(), "ILWT chunks per sample");
    const uint32_t total_work_items = checked_u32(
        static_cast<size_t>(chunks_per_sample) * coefficient_shape.batch_count, "ILWT total batch work items");
    std::vector<tt::tt_metal::CoreCoord> cores = wavelet_program_utils::select_row_major_cores(
        mesh_device,
        std::min(
            wavelet_program_utils::worker_core_count(mesh_device, "LWT requires at least one hardware worker core"),
            total_work_items),
        "ILWT active core count exceeds the worker grid");

    static_assert(
        executable_step_count<InverseScheme>() >= 2,
        "IDWT schemes require executable terminal and inverse scale steps");
    constexpr size_t expected_route_count = executable_step_count<InverseScheme>() - 2U;
    for (const auto& chunk : plan.chunks) {
        TT_FATAL(
            chunk.routes.size() == expected_route_count,
            "IDWT planner produced {} routes, but the kernel ABI requires {}",
            chunk.routes.size(),
            expected_route_count);
    }
    const size_t route_count = plan.chunks.front().routes.size();
    auto route_config = wavelet_program_utils::create_replicated_dram_pages(
        mesh_device, std::max(plan.chunks.size() * route_count, size_t{1}), device_protocol::kRouteConfigPageBytes);
    auto chunk_config = wavelet_program_utils::create_replicated_dram_pages(
        mesh_device, std::max(plan.chunks.size(), size_t{1}), device_protocol::kLwtChunkConfigPageBytes);
    IlwtWorkingBuffers buffers{
        .output = tensor_return_value.buffer(),
        .route_config = route_config,
        .chunk_config = chunk_config,
        .cores = std::move(cores),
    };

    const std::vector<uint32_t> chunk_words = build_inverse_chunk_config_words(plan, buffers);
    const std::vector<uint32_t> route_words = build_inverse_route_config_words(plan, buffers);
    buffers.chunk_config = wavelet_program_utils::upload_replicated_dram_metadata(
        mesh_device,
        std::max(plan.chunks.size(), size_t{1}),
        device_protocol::kLwtChunkConfigPageBytes,
        chunk_words,
        workload,
        "Wavelet metadata payload");
    buffers.route_config = wavelet_program_utils::upload_replicated_dram_metadata(
        mesh_device,
        std::max(plan.chunks.size() * route_count, size_t{1}),
        device_protocol::kRouteConfigPageBytes,
        route_words,
        workload,
        "Wavelet metadata payload");

    const std::vector<CoreChunkWork> work =
        wavelet_program_utils::partition_chunk_work(buffers.cores, total_work_items, "ILWT");
    const auto [min_work, max_work] = std::minmax_element(
        work.begin(), work.end(), [](const auto& lhs, const auto& rhs) { return lhs.chunk_count < rhs.chunk_count; });
    log_debug(
        tt::LogOp,
        "ttnn::idwt batch scheduler: B={}, chunks_per_sample={}, total_work_items={}, active_cores={}, "
        "work_items_per_core={}..{}, max_per_core_workspace_bytes={}, arch={}, scheme={}, layout={}, "
        "routes={}, groups_per_chunk={}, dependency_overhead={}, hybrid_tile_mirror={}, row_major_noc_staging={}, "
        "interleave_batch_sticks={}",
        coefficient_shape.batch_count,
        chunks_per_sample,
        total_work_items,
        buffers.cores.size(),
        min_work->chunk_count,
        max_work->chunk_count,
        3 *
            static_cast<uint64_t>(
                plan.workspace_elements + tile_mirror_elements(plan.workspace_elements, hybrid_tile_mirror)) *
            sizeof(float),
        static_cast<uint32_t>(architecture_policy.architecture),
        InverseScheme::name,
        static_cast<uint32_t>(plan.workspace_layout),
        route_count,
        plan.output_groups_per_chunk,
        plan.max_dependency_overhead,
        hybrid_tile_mirror,
        row_major_noc_staging,
        interleave_batch_sticks);
    const uint32_t input_pages_per_sample =
        pages_per_batch_item(tensor_args.approximation, coefficient_shape.batch_count, "ILWT approximation");
    const uint32_t output_pages_per_sample =
        pages_per_batch_item(tensor_return_value, coefficient_shape.batch_count, "ILWT output");
    const uint32_t l1_alignment_bytes = mesh_device.allocator()->get_alignment(tt::tt_metal::BufferType::L1);
    auto descriptor = create_inverse_program_descriptor(
        core_range_set(buffers.cores),
        approximation_buffer,
        detail_buffer,
        buffers,
        plan,
        plan.workspace_layout,
        hybrid_tile_mirror,
        row_major_noc_staging,
        interleave_batch_sticks,
        InverseScheme::compute_scheme_header,
        InverseScheme::compute_scheme_type,
        work,
        chunks_per_sample,
        input_pages_per_sample,
        output_pages_per_sample,
        l1_alignment_bytes);
    wavelet_program_utils::append_program_to_mesh_ranges(
        workload, std::move(descriptor), tensor_coords, "Wavelet workload has no mesh coordinate range");
    return workload;
}

void validate_forward_inputs(const Lwt1DParams& operation_attributes, const Lwt1DInputs& tensor_args) {
    validate_1d_tensor(tensor_args.input, "DWT input");
    const Logical1DShape input_shape = logical_1d_signal_shape(tensor_args.input, "DWT input");
    validate_output_memory_config(operation_attributes.output_memory_config, "DWT");
    TT_FATAL(operation_attributes.scheme_id != SchemeId::kUnknown, "DWT received an invalid wavelet scheme identifier");
    TT_FATAL(
        is_supported_lwt_boundary_mode(operation_attributes.boundary_mode),
        "DWT received an unsupported boundary mode");
    TT_FATAL(
        !boundary_mode_requires_multiple_samples(operation_attributes.boundary_mode) || input_shape.length > 1,
        "DWT reflect and antireflect modes require an input length greater than one");

    const auto expected_specs = detail::compute_lwt_1d_output_specs(operation_attributes, tensor_args);
    if (tensor_args.preallocated_outputs.has_value()) {
        validate_preallocated_output(
            std::get<0>(*tensor_args.preallocated_outputs),
            std::get<0>(expected_specs),
            tensor_args.input.device(),
            "DWT approximation output");
        validate_preallocated_output(
            std::get<1>(*tensor_args.preallocated_outputs),
            std::get<1>(expected_specs),
            tensor_args.input.device(),
            "DWT detail output");
        wavelet_tensor_validation::validate_distinct_buffers(
            std::get<0>(*tensor_args.preallocated_outputs),
            std::get<1>(*tensor_args.preallocated_outputs),
            "DWT approximation and detail outputs must not alias");
        wavelet_tensor_validation::validate_distinct_buffers(
            std::get<0>(*tensor_args.preallocated_outputs), tensor_args.input, "DWT outputs must not alias the input");
        wavelet_tensor_validation::validate_distinct_buffers(
            std::get<1>(*tensor_args.preallocated_outputs), tensor_args.input, "DWT outputs must not alias the input");
    }
}

void validate_inverse_inputs(const Ilwt1DParams& operation_attributes, const Ilwt1DInputs& tensor_args) {
    validate_1d_tensor(tensor_args.approximation, "IDWT approximation input");
    validate_1d_tensor(tensor_args.detail, "IDWT detail input");
    const Logical1DShape coefficient_shape = logical_1d_shape(tensor_args.approximation, "IDWT approximation input");
    validate_output_memory_config(operation_attributes.output_memory_config, "IDWT");
    wavelet_tensor_validation::validate_same_device(
        tensor_args.detail,
        tensor_args.approximation.device(),
        "IDWT approximation and detail inputs must be on the same device");
    TT_FATAL(
        tensor_args.approximation.logical_shape() == tensor_args.detail.logical_shape(),
        "IDWT approximation and detail inputs must have identical shapes");
    wavelet_tensor_validation::validate_distinct_buffers(
        tensor_args.approximation, tensor_args.detail, "IDWT approximation and detail inputs must not alias");
    TT_FATAL(operation_attributes.original_length > 0, "IDWT original_length must be greater than zero");
    TT_FATAL(
        operation_attributes.scheme_id != SchemeId::kUnknown, "IDWT received an invalid wavelet scheme identifier");
    TT_FATAL(
        is_supported_lwt_boundary_mode(operation_attributes.boundary_mode),
        "IDWT received an unsupported boundary mode");
    TT_FATAL(
        !boundary_mode_requires_multiple_samples(operation_attributes.boundary_mode) ||
            operation_attributes.original_length > 1,
        "IDWT reflect and antireflect modes require original_length greater than one");

    const uint32_t expected_coefficient_length =
        dwt_coefficient_length(operation_attributes.original_length, operation_attributes.scheme_id);
    const bool coefficient_length_valid =
        coefficient_shape.length == expected_coefficient_length ||
        (coefficient_shape.length >= expected_coefficient_length && coefficient_shape.length % kStickWidth == 0 &&
         coefficient_shape.length - expected_coefficient_length < kStickWidth);
    TT_FATAL(
        coefficient_length_valid,
        "IDWT coefficient length {} does not match expected length {} for original length {}",
        coefficient_shape.length,
        expected_coefficient_length,
        operation_attributes.original_length);

    if (tensor_args.preallocated_output.has_value()) {
        validate_preallocated_output(
            *tensor_args.preallocated_output,
            detail::compute_ilwt_1d_output_spec(operation_attributes, tensor_args),
            tensor_args.approximation.device(),
            "IDWT output");
        wavelet_tensor_validation::validate_distinct_buffers(
            *tensor_args.preallocated_output, tensor_args.approximation, "IDWT output must not alias an input");
        wavelet_tensor_validation::validate_distinct_buffers(
            *tensor_args.preallocated_output, tensor_args.detail, "IDWT output must not alias an input");
    }
}

}  // namespace

namespace detail {

tt::tt_metal::WorkloadDescriptor create_lwt_1d_workload(
    const Lwt1DParams& operation_attributes,
    const Lwt1DInputs& tensor_args,
    Lwt1DOutputs& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords) {
    return dispatch_scheme(operation_attributes.scheme_id, [&]<typename Scheme>() {
        return build_forward_workload<Scheme>(operation_attributes, tensor_args, tensor_return_value, tensor_coords);
    });
}

void validate_lwt_1d(const Lwt1DParams& operation_attributes, const Lwt1DInputs& tensor_args) {
    validate_forward_inputs(operation_attributes, tensor_args);
}

Lwt1DOutputSpecs compute_lwt_1d_output_specs(const Lwt1DParams& operation_attributes, const Lwt1DInputs& tensor_args) {
    const Logical1DShape input_shape = logical_1d_signal_shape(tensor_args.input, "DWT input");
    const uint32_t coefficient_length = dwt_coefficient_length(input_shape.length, operation_attributes.scheme_id);
    auto spec = output_spec_1d(input_shape, coefficient_length, operation_attributes.output_memory_config);
    return {spec, spec};
}

Lwt1DOutputs create_lwt_1d_output_tensors(const Lwt1DParams& operation_attributes, const Lwt1DInputs& tensor_args) {
    if (tensor_args.preallocated_outputs.has_value()) {
        return *tensor_args.preallocated_outputs;
    }
    auto specs = compute_lwt_1d_output_specs(operation_attributes, tensor_args);
    return {
        create_device_tensor(std::get<0>(specs), tensor_args.input.device()),
        create_device_tensor(std::get<1>(specs), tensor_args.input.device()),
    };
}

tt::tt_metal::WorkloadDescriptor create_ilwt_1d_workload(
    const Ilwt1DParams& operation_attributes,
    const Ilwt1DInputs& tensor_args,
    Tensor& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords) {
    return dispatch_scheme(operation_attributes.scheme_id, [&]<typename Scheme>() {
        return build_inverse_workload<Scheme>(operation_attributes, tensor_args, tensor_return_value, tensor_coords);
    });
}

void validate_ilwt_1d(const Ilwt1DParams& operation_attributes, const Ilwt1DInputs& tensor_args) {
    validate_inverse_inputs(operation_attributes, tensor_args);
}

tt::tt_metal::TensorSpec compute_ilwt_1d_output_spec(
    const Ilwt1DParams& operation_attributes, const Ilwt1DInputs& tensor_args) {
    return output_spec_1d(
        logical_1d_shape(tensor_args.approximation, "IDWT approximation"),
        operation_attributes.original_length,
        operation_attributes.output_memory_config);
}

Tensor create_ilwt_1d_output_tensor(const Ilwt1DParams& operation_attributes, const Ilwt1DInputs& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return *tensor_args.preallocated_output;
    }
    return create_device_tensor(
        compute_ilwt_1d_output_spec(operation_attributes, tensor_args), tensor_args.approximation.device());
}

}  // namespace detail

}  // namespace ttnn::prim
