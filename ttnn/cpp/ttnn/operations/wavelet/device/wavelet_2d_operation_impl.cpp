// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/wavelet/device/wavelet_2d_operation_impl.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tt-logger/tt-logger.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/circular_buffer_constants.h"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/program_descriptors.hpp"
#include "tt-metalium/shape.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/tile.hpp"
#include "tt-metalium/workload_descriptor.hpp"
#include "ttnn/operations/wavelet/common/wavelet_host.hpp"
#include "ttnn/operations/wavelet/device/wavelet_program_utils.hpp"
#include "ttnn/operations/wavelet/device/wavelet_tensor_validation.hpp"
#include "ttnn/operations/wavelet/generated/wavelet_schemes/scheme_dispatch.hpp"
#include "ttnn/operations/wavelet/planner/inverse_plan_2d.hpp"
#include "ttnn/operations/wavelet/planner/plan_2d.hpp"
#include "ttnn/operations/wavelet/planner/policy.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

using namespace operations::wavelet;
using wavelet_program_utils::checked_u32;
using wavelet_program_utils::core_range_set;
using wavelet_program_utils::CoreChunkWork;
using wavelet_tensor_validation::validate_input_memory_config;
using wavelet_tensor_validation::validate_output_memory_config;

namespace {

constexpr tt::DataFormat kDataFormat = tt::DataFormat::Float32;
constexpr uint32_t kTileBytes = device_protocol::kLwt2DFullTileBytes;
constexpr uint32_t kSource0Cb = tt::CBIndex::c_0;
constexpr uint32_t kSource1Cb = tt::CBIndex::c_1;
constexpr uint32_t kBaseCb = tt::CBIndex::c_2;
constexpr uint32_t kSyncCb = tt::CBIndex::c_3;
constexpr uint32_t kReaderChunkConfigCb = tt::CBIndex::c_4;
constexpr uint32_t kWriterBandConfigCb = tt::CBIndex::c_7;
constexpr uint32_t kNocScratchCb = tt::CBIndex::c_8;
constexpr uint32_t kRouteZeroCb = tt::CBIndex::c_9;
constexpr uint32_t kOutputCb = tt::CBIndex::c_16;
constexpr uint32_t kTileBuffering = 2;
constexpr uint32_t kConfigNocAlignmentBytes = 64;

constexpr const char* kReaderKernel = "ttnn/cpp/ttnn/operations/wavelet/device/kernels/dataflow/lwt_2d_reader.cpp";
constexpr const char* kComputeKernel = "ttnn/cpp/ttnn/operations/wavelet/device/kernels/compute/lwt_2d_compute.cpp";
constexpr const char* kWriterKernel = "ttnn/cpp/ttnn/operations/wavelet/device/kernels/dataflow/lwt_2d_writer.cpp";

[[nodiscard]] constexpr uint32_t split_scratch_tile_count(const BoundaryMode boundary_mode, const bool inverse) {
    return !inverse && boundary_mode == BoundaryMode::kSymmetric ? device_protocol::kLwt2DSymmetricSplitScratchTileCount
                                                                 : device_protocol::kLwt2DSplitScratchTileCount;
}

struct WorkingBuffers2D {
    std::array<uint32_t, device_protocol::kLwt2DPlaneCount> plane_tile_counts{};
    std::array<tt::tt_metal::Buffer*, device_protocol::kLwt2DBandCount> outputs{};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> chunk_config;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> route_config;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> band_config;
    std::vector<tt::tt_metal::CoreCoord> cores;
};

struct Logical2DShape {
    uint32_t batch_count{1};
    uint32_t height{0};
    uint32_t width{0};
    bool rank_four{false};
};

[[nodiscard]] Logical2DShape logical_2d_shape(const Tensor& tensor, const char* tensor_name) {
    const auto& shape = tensor.logical_shape();
    if (shape.rank() == 2) {
        return Logical2DShape{
            .batch_count = 1,
            .height = checked_u32(shape[0], tensor_name),
            .width = checked_u32(shape[1], tensor_name),
            .rank_four = false,
        };
    }
    TT_FATAL(shape.rank() == 4, "{} must have shape [H,W] or [B,1,H,W], got rank {}", tensor_name, shape.rank());
    TT_FATAL(shape[0] > 0, "{} batch dimension must be positive", tensor_name);
    TT_FATAL(shape[1] == 1, "{} requires C == 1, got {}", tensor_name, shape[1]);
    return Logical2DShape{
        .batch_count = checked_u32(shape[0], "2D wavelet batch count"),
        .height = checked_u32(shape[2], tensor_name),
        .width = checked_u32(shape[3], tensor_name),
        .rank_four = true,
    };
}

[[nodiscard]] uint32_t tile_pages_per_batch_item(
    const Tensor& tensor, const uint32_t batch_count, const char* tensor_name) {
    TT_FATAL(batch_count > 0, "{} batch count must be positive", tensor_name);
    const uint64_t physical_elements = tensor.physical_volume();
    TT_FATAL(
        physical_elements % batch_count == 0, "{} physical volume is not divisible by its batch count", tensor_name);
    const uint64_t elements_per_batch = physical_elements / batch_count;
    TT_FATAL(
        elements_per_batch % (kTileHeight2D * kTileWidth2D) == 0,
        "{} physical batch stride is not tile aligned",
        tensor_name);
    return checked_u32(elements_per_batch / (kTileHeight2D * kTileWidth2D), "2D wavelet tiles per batch item");
}

[[nodiscard]] uint32_t noc_scratch_tile_count(
    const BoundaryMode boundary_mode, const bool inverse, const size_t route_count) {
    TT_FATAL(
        route_count <= std::numeric_limits<size_t>::max() / (2 * device_protocol::kLwt2DRouteConfigPageBytes),
        "2D route-config scratch size overflows size_t");
    const size_t route_config_bytes = route_count * device_protocol::kLwt2DRouteConfigPageBytes;
    const size_t route_config_tile_count = ceil_div(2 * route_config_bytes, static_cast<size_t>(kTileBytes));
    const size_t tile_count =
        std::max(static_cast<size_t>(split_scratch_tile_count(boundary_mode, inverse)), route_config_tile_count);
    TT_FATAL(
        tile_count <= device_protocol::kLwt2DSplitScratchTileCount,
        "2D route descriptors require {} scratch tiles, exceeding the {}-tile accounted split-scratch budget",
        tile_count,
        device_protocol::kLwt2DSplitScratchTileCount);
    return checked_u32(tile_count, "2D NoC scratch tile count");
}

void add_cb(
    tt::tt_metal::ProgramDescriptor& descriptor,
    const tt::tt_metal::CoreRangeSet& cores,
    const uint32_t cb,
    const uint32_t pages,
    const uint32_t page_bytes,
    const bool tile) {
    descriptor.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = pages * page_bytes,
        .core_ranges = cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb),
            .data_format = kDataFormat,
            .page_size = page_bytes,
            .tile = tile ? std::optional{tt::tt_metal::TileDescriptor{32, 32, false}} : std::nullopt,
        }}},
    });
}

[[nodiscard]] uint32_t route_tile_count(const Lwt2DRoutePlan& route) {
    if (route.output.empty()) {
        return 0;
    }
    const size_t height = plan_2d_detail::aligned_interval_span(route.output.y, kTileHeight2D, "2D route tile height");
    const size_t width = plan_2d_detail::aligned_interval_span(route.output.x, kTileWidth2D, "2D route tile width");
    return checked_u32((height / kTileHeight2D) * (width / kTileWidth2D), "2D route tile count");
}

[[nodiscard]] std::vector<uint32_t> plane_offsets(const WorkingBuffers2D& buffers) {
    std::vector<uint32_t> args(2 * device_protocol::kLwt2DPlaneCount, 0);
    uint32_t plane_offset = 0;
    for (size_t slot = 0; slot < buffers.plane_tile_counts.size(); ++slot) {
        args[slot] = plane_offset;
        TT_FATAL(
            buffers.plane_tile_counts[slot] <= (std::numeric_limits<uint32_t>::max() - plane_offset) / kTileBytes,
            "2D workspace plane offsets overflow uint32_t");
        plane_offset += buffers.plane_tile_counts[slot] * kTileBytes;
    }
    return args;
}

template <typename Plan>
void replace_plane_tile_counts_with_widths(std::vector<uint32_t>& args, const Plan& plan) {
    for (size_t slot = 0; slot < device_protocol::kLwt2DPlaneCount; ++slot) {
        args[device_protocol::kLwt2DPlaneCount + slot] = plan.allocated_plane_widths_elements[slot] / kTileWidth2D;
    }
}

[[nodiscard]] tt::tt_metal::KernelDescriptor::RTArgList reader_args(
    const tt::tt_metal::Buffer& input,
    const Lwt2DExecutionPlan& plan,
    const WorkingBuffers2D& buffers,
    const CoreChunkWork& work,
    const uint32_t chunks_per_sample,
    const uint32_t input_tiles_per_sample) {
    tt::tt_metal::KernelDescriptor::RTArgList args;
    args.push_back(const_cast<tt::tt_metal::Buffer*>(&input));
    args.append({
        checked_u32(plan.input_height, "2D input height"),
        checked_u32(plan.input_width, "2D input width"),
        checked_u32(plan.tiling.input.storage.width / kTileWidth2D, "2D input tile columns"),
        plan.y_plan.preprocess_layout.pad_config.left,
        plan.x_plan.preprocess_layout.pad_config.left,
    });
    std::vector<uint32_t> planes = plane_offsets(buffers);
    replace_plane_tile_counts_with_widths(planes, plan);
    args.append(planes);
    args.push_back(static_cast<uint32_t>(buffers.chunk_config->get_backing_buffer()->address()));
    args.push_back(static_cast<uint32_t>(buffers.route_config->get_backing_buffer()->address()));
    args.push_back(work.chunk_begin);
    args.push_back(work.chunk_count);
    args.push_back(checked_u32(plan.chunks.front().routes.size(), "2D route count"));
    args.push_back(chunks_per_sample);
    args.push_back(input_tiles_per_sample);
    return args;
}

[[nodiscard]] tt::tt_metal::KernelDescriptor::RTArgList writer_args(
    const Lwt2DExecutionPlan& plan,
    const WorkingBuffers2D& buffers,
    const CoreChunkWork& work,
    const uint32_t chunks_per_sample,
    const uint32_t output_tiles_per_sample) {
    std::vector<uint32_t> plane_args = plane_offsets(buffers);
    replace_plane_tile_counts_with_widths(plane_args, plan);
    tt::tt_metal::KernelDescriptor::RTArgList args;
    args.append(plane_args);
    args.push_back(static_cast<uint32_t>(buffers.route_config->get_backing_buffer()->address()));
    args.push_back(static_cast<uint32_t>(buffers.band_config->get_backing_buffer()->address()));
    for (auto* output : buffers.outputs) {
        args.push_back(output);
    }
    args.push_back(checked_u32(plan.tiling.band.storage.width / kTileWidth2D, "2D band tile columns"));
    args.push_back(work.chunk_begin);
    args.push_back(work.chunk_count);
    args.push_back(checked_u32(plan.chunks.front().routes.size(), "2D route count"));
    args.push_back(chunks_per_sample);
    args.push_back(output_tiles_per_sample);
    return args;
}

[[nodiscard]] uint32_t encoded_i32(const int64_t value, const char* label) {
    TT_FATAL(
        value >= std::numeric_limits<int32_t>::min() && value <= std::numeric_limits<int32_t>::max(),
        "{} exceeds int32_t",
        label);
    return static_cast<uint32_t>(static_cast<int32_t>(value));
}

[[nodiscard]] tt::tt_metal::KernelDescriptor::RTArgList inverse_reader_args(
    const std::array<const tt::tt_metal::Buffer*, device_protocol::kLwt2DBandCount>& bands,
    const Ilwt2DExecutionPlan& plan,
    const WorkingBuffers2D& buffers,
    const CoreChunkWork& work,
    const uint32_t chunks_per_sample,
    const uint32_t input_tiles_per_sample) {
    const auto& y_forward = plan.y_plan.forward_trace;
    const auto& x_forward = plan.x_plan.forward_trace;
    const int64_t y_canonical_start = static_cast<int64_t>(y_forward.preprocess_layout.pad_config.left + 1) / 2;
    const int64_t x_canonical_start = static_cast<int64_t>(x_forward.preprocess_layout.pad_config.left + 1) / 2;
    tt::tt_metal::KernelDescriptor::RTArgList args;
    for (const auto* band : bands) {
        args.push_back(const_cast<tt::tt_metal::Buffer*>(band));
    }
    args.push_back(checked_u32(plan.band_height, "2D ILWT band height"));
    args.push_back(checked_u32(plan.band_width, "2D ILWT band width"));
    args.push_back(checked_u32(plan.tiling.band.storage.width / kTileWidth2D, "2D ILWT band tile columns"));
    args.push_back(encoded_i32(y_canonical_start - y_forward.final_even_shift, "2D ILWT y-even offset"));
    args.push_back(encoded_i32(y_canonical_start - y_forward.final_odd_shift, "2D ILWT y-odd offset"));
    args.push_back(encoded_i32(x_canonical_start - x_forward.final_even_shift, "2D ILWT x-even offset"));
    args.push_back(encoded_i32(x_canonical_start - x_forward.final_odd_shift, "2D ILWT x-odd offset"));
    std::vector<uint32_t> planes = plane_offsets(buffers);
    replace_plane_tile_counts_with_widths(planes, plan);
    args.append(planes);
    args.push_back(static_cast<uint32_t>(buffers.chunk_config->get_backing_buffer()->address()));
    args.push_back(static_cast<uint32_t>(buffers.route_config->get_backing_buffer()->address()));
    args.push_back(work.chunk_begin);
    args.push_back(work.chunk_count);
    args.push_back(checked_u32(plan.chunks.front().routes.size(), "2D ILWT route count"));
    args.push_back(chunks_per_sample);
    args.push_back(input_tiles_per_sample);
    return args;
}

[[nodiscard]] tt::tt_metal::KernelDescriptor::RTArgList inverse_writer_args(
    const Ilwt2DExecutionPlan& plan,
    const WorkingBuffers2D& buffers,
    const CoreChunkWork& work,
    const uint32_t chunks_per_sample,
    const uint32_t output_tiles_per_sample) {
    std::vector<uint32_t> plane_args = plane_offsets(buffers);
    replace_plane_tile_counts_with_widths(plane_args, plan);
    tt::tt_metal::KernelDescriptor::RTArgList args;
    args.append(plane_args);
    args.push_back(static_cast<uint32_t>(buffers.route_config->get_backing_buffer()->address()));
    args.push_back(static_cast<uint32_t>(buffers.band_config->get_backing_buffer()->address()));
    for (uint32_t band = 0; band < device_protocol::kLwt2DBandCount; ++band) {
        args.push_back(buffers.outputs[0]);
    }
    args.push_back(checked_u32(plan.tiling.input.storage.width / kTileWidth2D, "2D ILWT output tile columns"));
    args.push_back(work.chunk_begin);
    args.push_back(work.chunk_count);
    args.push_back(checked_u32(plan.chunks.front().routes.size(), "2D ILWT route count"));
    args.push_back(plan.y_plan.forward_trace.preprocess_layout.pad_config.left);
    args.push_back(plan.x_plan.forward_trace.preprocess_layout.pad_config.left);
    args.push_back(chunks_per_sample);
    args.push_back(output_tiles_per_sample);
    return args;
}

template <typename Plan>
[[nodiscard]] std::vector<uint32_t> compute_args(const Plan& plan, const CoreChunkWork& work) {
    const size_t route_count = plan.chunks.front().routes.size();
    const size_t packed_words_per_chunk = ceil_div(route_count, static_cast<size_t>(4));
    std::vector<uint32_t> args;
    args.reserve(1 + static_cast<size_t>(work.chunk_count) * packed_words_per_chunk);
    args.push_back(work.chunk_count);
    for (uint32_t local_chunk = 0; local_chunk < work.chunk_count; ++local_chunk) {
        const Lwt2DChunkPlan& chunk = plan.chunks[(work.chunk_begin + local_chunk) % plan.chunks.size()];
        for (size_t route_begin = 0; route_begin < route_count; route_begin += 4) {
            uint32_t packed_counts = 0;
            const size_t route_end = std::min(route_begin + 4, route_count);
            for (size_t route_index = route_begin; route_index < route_end; ++route_index) {
                const uint32_t count = route_tile_count(chunk.routes[route_index]);
                TT_FATAL(
                    count <= std::numeric_limits<uint8_t>::max(), "2D route tile count {} exceeds packed uint8", count);
                packed_counts |= count << (8 * (route_index - route_begin));
            }
            args.push_back(packed_counts);
        }
    }
    return args;
}

[[nodiscard]] tt::tt_metal::ProgramDescriptor create_program_descriptor(
    const tt::tt_metal::CoreRangeSet& cores,
    const std::array<const tt::tt_metal::Buffer*, device_protocol::kLwt2DBandCount>& inputs,
    const WorkingBuffers2D& buffers,
    const char* compute_scheme_header,
    const char* compute_scheme_type,
    const BoundaryMode boundary_mode,
    const bool compact_boundary_code,
    const bool inverse,
    const uint32_t scratch_tile_count) {
    tt::tt_metal::ProgramDescriptor descriptor;
    const uint32_t scratch_bytes = scratch_tile_count * kTileBytes;
    uint32_t workspace_tiles = 0;
    for (const uint32_t plane_tiles : buffers.plane_tile_counts) {
        TT_FATAL(
            plane_tiles <= std::numeric_limits<uint32_t>::max() - workspace_tiles,
            "2D workspace tile count overflows uint32_t");
        workspace_tiles += plane_tiles;
    }
    add_cb(descriptor, cores, kSource0Cb, kTileBuffering, kTileBytes, true);
    add_cb(descriptor, cores, kSource1Cb, kTileBuffering, kTileBytes, true);
    add_cb(descriptor, cores, kBaseCb, kTileBuffering, kTileBytes, true);
    add_cb(descriptor, cores, kOutputCb, kTileBuffering, kTileBytes, true);
    add_cb(descriptor, cores, kSyncCb, 1, kConfigNocAlignmentBytes, false);
    add_cb(descriptor, cores, kReaderChunkConfigCb, 1, device_protocol::kLwt2DChunkConfigPageBytes, false);
    add_cb(descriptor, cores, kWriterBandConfigCb, 1, device_protocol::kLwt2DBandConfigPageBytes, false);
    add_cb(descriptor, cores, kNocScratchCb, scratch_tile_count, kTileBytes, true);
    add_cb(descriptor, cores, kRouteZeroCb, 1, kTileBytes, true);
    add_cb(descriptor, cores, device_protocol::kLwt2DWorkspaceCb, workspace_tiles, kTileBytes, true);
    std::vector<uint32_t> reader_compile_args = {
        kSource0Cb,
        kSource1Cb,
        kBaseCb,
        kSyncCb,
        kReaderChunkConfigCb,
        kNocScratchCb,
        kRouteZeroCb,
    };
    if (inverse) {
        for (const auto* input : inputs) {
            TT_FATAL(input != nullptr, "2D ILWT input buffer must be allocated");
            tt::tt_metal::TensorAccessorArgs(*input).append_to(reader_compile_args);
        }
    } else {
        TT_FATAL(inputs.front() != nullptr, "2D LWT input buffer must be allocated");
        tt::tt_metal::TensorAccessorArgs(*inputs.front()).append_to(reader_compile_args);
    }
    tt::tt_metal::TensorAccessorArgs(*buffers.chunk_config->get_backing_buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(*buffers.route_config->get_backing_buffer()).append_to(reader_compile_args);
    reader_compile_args.push_back(static_cast<uint32_t>(boundary_mode));
    reader_compile_args.push_back(scratch_bytes);
    tt::tt_metal::KernelDescriptor::Defines reader_defines;
    if (inverse) {
        reader_defines.emplace_back("ILWT_2D", "1");
    }
    if (compact_boundary_code) {
        reader_defines.emplace_back(inverse ? "ILWT_2D_COMPACT_BOUNDARY_CODE" : "LWT_2D_COMPACT_BOUNDARY_CODE", "1");
    }
    tt::tt_metal::KernelDescriptor reader_descriptor;
    reader_descriptor.kernel_source = kReaderKernel;
    reader_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_descriptor.core_ranges = cores;
    reader_descriptor.compile_time_args = std::move(reader_compile_args);
    reader_descriptor.defines = std::move(reader_defines);
    reader_descriptor.config = tt::tt_metal::ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_compile_args = {
        kOutputCb,
        kSyncCb,
        kWriterBandConfigCb,
        kNocScratchCb,
    };
    tt::tt_metal::TensorAccessorArgs(*buffers.route_config->get_backing_buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(*buffers.band_config->get_backing_buffer()).append_to(writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(*buffers.outputs.front()).append_to(writer_compile_args);
    writer_compile_args.push_back(scratch_bytes);
    tt::tt_metal::KernelDescriptor::Defines writer_defines;
    if (inverse) {
        writer_defines.emplace_back("ILWT_2D", "1");
    }
    tt::tt_metal::KernelDescriptor writer_descriptor;
    writer_descriptor.kernel_source = kWriterKernel;
    writer_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_descriptor.core_ranges = cores;
    writer_descriptor.compile_time_args = std::move(writer_compile_args);
    writer_descriptor.defines = std::move(writer_defines);
    writer_descriptor.config = tt::tt_metal::WriterConfigDescriptor{};

    std::vector<tt::tt_metal::UnpackToDestMode> unpack_modes(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    unpack_modes[kSource0Cb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[kSource1Cb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    unpack_modes[kBaseCb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    tt::tt_metal::KernelDescriptor::Defines compute_defines;
    if (inverse) {
        compute_defines.emplace_back("ILWT_2D_SCHEME_HEADER", compute_scheme_header);
        compute_defines.emplace_back("ILWT_2D_SCHEME_TYPE", compute_scheme_type);
        compute_defines.emplace_back("ILWT_2D", "1");
    } else {
        compute_defines.emplace_back("LWT_2D_SCHEME_HEADER", compute_scheme_header);
        compute_defines.emplace_back("LWT_2D_SCHEME_TYPE", compute_scheme_type);
    }
    tt::tt_metal::KernelDescriptor compute_descriptor;
    compute_descriptor.kernel_source = kComputeKernel;
    compute_descriptor.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_descriptor.core_ranges = cores;
    compute_descriptor.compile_time_args = {kSource0Cb, kSource1Cb, kBaseCb, kOutputCb};
    compute_descriptor.defines = std::move(compute_defines);
    compute_descriptor.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .unpack_to_dest_mode = unpack_modes,
    };

    descriptor.kernels.push_back(std::move(reader_descriptor));
    descriptor.kernels.push_back(std::move(compute_descriptor));
    descriptor.kernels.push_back(std::move(writer_descriptor));
    return descriptor;
}

}  // namespace

namespace {

constexpr uint32_t kL1SignalBudgetBytes2D = 768 * 1024;
constexpr size_t kCompactBoundaryRouteThreshold = 52;

void validate_2d_tensor(const Tensor& tensor, const char* tensor_name) {
    wavelet_tensor_validation::validate_device_tensor(tensor, tensor_name);
    TT_FATAL(tensor.layout() == Layout::TILE, "{} must use TILE layout", tensor_name);
    const Logical2DShape shape = logical_2d_shape(tensor, tensor_name);
    TT_FATAL(shape.height > 0 && shape.width > 0, "{} height and width must be positive", tensor_name);
    const auto tile = tensor.tensor_spec().tile();
    TT_FATAL(
        tile.get_height() == kTileHeight2D && tile.get_width() == kTileWidth2D,
        "{} must use standard 32x32 TTNN tiles",
        tensor_name);
    validate_input_memory_config(tensor.memory_config(), tensor_name);

    const size_t padded_height = round_up(static_cast<size_t>(shape.height), kTileHeight2D);
    const size_t padded_width = round_up(static_cast<size_t>(shape.width), kTileWidth2D);
    TT_FATAL(
        padded_height <= std::numeric_limits<size_t>::max() / padded_width / sizeof(float),
        "{} per-batch physical size calculation overflows size_t",
        tensor_name);
    const size_t bytes_per_batch = padded_height * padded_width * sizeof(float);
    TT_FATAL(
        bytes_per_batch == 0 ||
            static_cast<size_t>(shape.batch_count) <= std::numeric_limits<size_t>::max() / bytes_per_batch,
        "{} batched physical size calculation overflows size_t",
        tensor_name);
    const size_t required_bytes = static_cast<size_t>(shape.batch_count) * bytes_per_batch;
    TT_FATAL(
        tensor.buffer()->size() >= required_bytes,
        "{} physical buffer has {} bytes but its padded tile shape requires {}",
        tensor_name,
        tensor.buffer()->size(),
        required_bytes);
    static_cast<void>(make_architecture_policy(tensor.device()->arch()));
}

[[nodiscard]] tt::tt_metal::TensorSpec output_spec_2d(
    const Logical2DShape& input_shape, const uint32_t height, const uint32_t width, const MemoryConfig& memory_config) {
    const Shape output_shape =
        input_shape.rank_four ? Shape({input_shape.batch_count, 1, height, width}) : Shape({height, width});
    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(DataType::FLOAT32, tt::tt_metal::PageConfig(Layout::TILE), memory_config));
}

void validate_preallocated_output_2d(
    const Tensor& output,
    const tt::tt_metal::TensorSpec& expected_spec,
    const tt::tt_metal::distributed::MeshDevice* expected_device,
    const char* output_name) {
    validate_2d_tensor(output, output_name);
    wavelet_tensor_validation::validate_preallocated_output_placement(output, expected_device, output_name);
    TT_FATAL(
        output.tensor_spec() == expected_spec,
        "{} tensor spec does not match the wavelet output specification",
        output_name);
}

template <typename Scheme>
[[nodiscard]] Lwt2DExecutionPlan make_forward_plan_2d(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const uint32_t height,
    const uint32_t width,
    const BoundaryMode boundary_mode,
    const uint32_t available_l1_bytes) {
    const uint64_t l1_budget_bytes = std::min<uint64_t>(kL1SignalBudgetBytes2D, available_l1_bytes);
    Lwt2DExecutionPlan plan = make_lwt_2d_execution_plan<Scheme>(
        height,
        width,
        wavelet_program_utils::worker_core_count(mesh_device, "2D wavelet transforms require at least one worker core"),
        l1_budget_bytes,
        boundary_mode,
        true,
        true,
        Lwt2DRouteDomainPolicy::kExact);
    validate_lwt_2d_tiling_contract(plan.tiling);
    TT_FATAL(
        plan.input_height <= static_cast<size_t>(std::numeric_limits<int32_t>::max() / 2) &&
            plan.input_width <= static_cast<size_t>(std::numeric_limits<int32_t>::max() / 2),
        "2D LWT input dimensions exceed the signed boundary-index range");
    TT_FATAL(
        plan.allocated_l1_bytes <= available_l1_bytes,
        "2D LWT allocation requires {} L1 bytes but only {} remain below allocator-managed L1 tensors",
        plan.allocated_l1_bytes,
        available_l1_bytes);
    return plan;
}

template <typename Scheme>
[[nodiscard]] Ilwt2DExecutionPlan make_inverse_plan_2d(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const uint32_t height,
    const uint32_t width,
    const BoundaryMode boundary_mode,
    const uint32_t available_l1_bytes) {
    const uint64_t l1_budget_bytes = std::min<uint64_t>(kL1SignalBudgetBytes2D, available_l1_bytes);
    const ArchitecturePolicy architecture_policy = make_architecture_policy(mesh_device.arch());
    Ilwt2DExecutionPlan plan = make_ilwt_2d_execution_plan<Scheme>(
        height,
        width,
        wavelet_program_utils::worker_core_count(mesh_device, "2D wavelet transforms require at least one worker core"),
        l1_budget_bytes,
        boundary_mode,
        architecture_policy.inverse_2d_coordination_penalty_cycles_per_core);
    TT_FATAL(!plan.chunks.empty(), "2D ILWT requires at least one planned chunk");
    TT_FATAL(
        plan.output_height <= static_cast<size_t>(std::numeric_limits<int32_t>::max() / 2) &&
            plan.output_width <= static_cast<size_t>(std::numeric_limits<int32_t>::max() / 2),
        "2D ILWT output dimensions exceed the signed boundary-index range");
    TT_FATAL(
        plan.allocated_l1_bytes <= available_l1_bytes,
        "2D ILWT allocation requires {} L1 bytes but only {} remain below allocator-managed L1 tensors",
        plan.allocated_l1_bytes,
        available_l1_bytes);
    return plan;
}

template <typename Plan>
void bind_compute_args_2d(
    tt::tt_metal::KernelDescriptor& descriptor, const Plan& plan, const CoreChunkWork& core_work) {
    tt::tt_metal::KernelDescriptor::RTArgList args;
    args.append(compute_args(plan, core_work));
    descriptor.emplace_runtime_args(core_work.core, args);
}

template <typename Scheme>
[[nodiscard]] tt::tt_metal::WorkloadDescriptor build_forward_workload_2d(
    const Lwt2DParams& operation_attributes,
    const Lwt2DInputs& tensor_args,
    std::tuple<Tensor, Tensor, Tensor, Tensor>& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords) {
    auto& mesh_device = *tensor_args.input.device();
    const auto& input_buffer = *tensor_args.input.buffer();
    const Logical2DShape input_shape = logical_2d_shape(tensor_args.input, "2D DWT input");
    const std::array<const tt::tt_metal::Buffer*, device_protocol::kLwt2DBandCount> input_buffers = {
        &input_buffer, &input_buffer, &input_buffer, &input_buffer};
    Lwt2DExecutionPlan plan = make_forward_plan_2d<Scheme>(
        mesh_device,
        input_shape.height,
        input_shape.width,
        operation_attributes.boundary_mode,
        operation_attributes.available_l1_bytes);

    const uint32_t chunks_per_sample = checked_u32(plan.chunks.size(), "2D LWT chunks per sample");
    const uint32_t total_work_items =
        checked_u32(static_cast<size_t>(chunks_per_sample) * input_shape.batch_count, "2D LWT total batch work items");
    std::vector<tt::tt_metal::CoreCoord> cores = wavelet_program_utils::select_row_major_cores(
        mesh_device,
        std::min(
            wavelet_program_utils::worker_core_count(
                mesh_device, "2D wavelet transforms require at least one worker core"),
            total_work_items),
        "2D LWT active core count exceeds the worker grid");
    tt::tt_metal::WorkloadDescriptor workload;
    constexpr size_t expected_route_count = 4U * Scheme::num_steps;
    for (const auto& chunk : plan.chunks) {
        TT_FATAL(
            chunk.routes.size() == expected_route_count,
            "2D DWT planner produced {} routes, but the kernel ABI requires {}",
            chunk.routes.size(),
            expected_route_count);
    }
    const size_t route_count = plan.chunks.front().routes.size();
    const uint32_t scratch_tile_count = noc_scratch_tile_count(operation_attributes.boundary_mode, false, route_count);
    const size_t config_capacity = static_cast<size_t>(scratch_tile_count) * kTileBytes / 2;
    TT_FATAL(
        route_count * device_protocol::kLwt2DRouteConfigPageBytes <= config_capacity,
        "2D LWT {} route descriptors require {} bytes, exceeding the {}-byte per-RISC preload region",
        route_count,
        route_count * device_protocol::kLwt2DRouteConfigPageBytes,
        config_capacity);

    auto chunk_config = wavelet_program_utils::upload_replicated_dram_metadata(
        mesh_device,
        plan.chunks.size(),
        device_protocol::kLwt2DChunkConfigPageBytes,
        build_lwt_2d_chunk_config_words(plan),
        workload,
        "2D wavelet metadata");
    auto route_config = wavelet_program_utils::upload_replicated_dram_metadata(
        mesh_device,
        plan.chunks.size() * route_count,
        device_protocol::kLwt2DRouteConfigPageBytes,
        build_lwt_2d_route_config_words(plan),
        workload,
        "2D wavelet metadata");
    auto band_config = wavelet_program_utils::upload_replicated_dram_metadata(
        mesh_device,
        plan.chunks.size(),
        device_protocol::kLwt2DBandConfigPageBytes,
        build_lwt_2d_band_config_words(plan),
        workload,
        "2D wavelet metadata");

    WorkingBuffers2D buffers{
        .plane_tile_counts = {},
        .outputs =
            {
                std::get<0>(tensor_return_value).buffer(),
                std::get<1>(tensor_return_value).buffer(),
                std::get<2>(tensor_return_value).buffer(),
                std::get<3>(tensor_return_value).buffer(),
            },
        .chunk_config = std::move(chunk_config),
        .route_config = std::move(route_config),
        .band_config = std::move(band_config),
        .cores = std::move(cores),
    };
    for (size_t slot = 0; slot < buffers.plane_tile_counts.size(); ++slot) {
        buffers.plane_tile_counts[slot] =
            checked_u32(plan.allocated_plane_slot_bytes[slot] / kTileBytes, "2D LWT workspace plane tiles");
    }
    const ArchitecturePolicy architecture_policy = make_architecture_policy(mesh_device.arch());
    const bool compact_boundary_code = architecture_policy.compact_2d_reader ||
                                       route_count >= kCompactBoundaryRouteThreshold ||
                                       operation_attributes.boundary_mode == BoundaryMode::kAntireflect;
    auto descriptor = create_program_descriptor(
        core_range_set(buffers.cores),
        input_buffers,
        buffers,
        Scheme::compute_scheme_header,
        Scheme::compute_scheme_type,
        operation_attributes.boundary_mode,
        compact_boundary_code,
        false,
        scratch_tile_count);
    const std::vector<CoreChunkWork> work =
        wavelet_program_utils::partition_chunk_work(buffers.cores, total_work_items, "2D LWT");
    const auto [min_work, max_work] = std::minmax_element(
        work.begin(), work.end(), [](const auto& lhs, const auto& rhs) { return lhs.chunk_count < rhs.chunk_count; });
    log_debug(
        tt::LogOp,
        "ttnn::dwt_2d batch scheduler: B={}, chunks_per_sample={}, total_work_items={}, active_cores={}, "
        "work_items_per_core={}..{}, max_per_core_workspace_bytes={}",
        input_shape.batch_count,
        chunks_per_sample,
        total_work_items,
        buffers.cores.size(),
        min_work->chunk_count,
        max_work->chunk_count,
        plan.allocated_l1_bytes);
    const uint32_t input_tiles_per_sample =
        tile_pages_per_batch_item(tensor_args.input, input_shape.batch_count, "2D LWT input");
    const uint32_t output_tiles_per_sample =
        tile_pages_per_batch_item(std::get<0>(tensor_return_value), input_shape.batch_count, "2D LWT output");
    for (const auto& core_work : work) {
        descriptor.kernels[0].emplace_runtime_args(
            core_work.core,
            reader_args(input_buffer, plan, buffers, core_work, chunks_per_sample, input_tiles_per_sample));
        bind_compute_args_2d(descriptor.kernels[1], plan, core_work);
        descriptor.kernels[2].emplace_runtime_args(
            core_work.core, writer_args(plan, buffers, core_work, chunks_per_sample, output_tiles_per_sample));
    }
    wavelet_program_utils::append_program_to_mesh_ranges(
        workload, std::move(descriptor), tensor_coords, "2D wavelet workload has no mesh coordinate range");
    return workload;
}

template <typename Scheme>
[[nodiscard]] tt::tt_metal::WorkloadDescriptor build_inverse_workload_2d(
    const Ilwt2DParams& operation_attributes,
    const Ilwt2DInputs& tensor_args,
    Tensor& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords) {
    auto& mesh_device = *tensor_args.ll.device();
    const Logical2DShape band_shape = logical_2d_shape(tensor_args.ll, "2D ILWT LL input");
    const std::array<const tt::tt_metal::Buffer*, device_protocol::kLwt2DBandCount> band_buffers = {
        tensor_args.ll.buffer(), tensor_args.lh.buffer(), tensor_args.hl.buffer(), tensor_args.hh.buffer()};
    Ilwt2DExecutionPlan plan = make_inverse_plan_2d<Scheme>(
        mesh_device,
        operation_attributes.output_height,
        operation_attributes.output_width,
        operation_attributes.boundary_mode,
        operation_attributes.available_l1_bytes);

    const uint32_t chunks_per_sample = checked_u32(plan.chunks.size(), "2D ILWT chunks per sample");
    const uint32_t total_work_items =
        checked_u32(static_cast<size_t>(chunks_per_sample) * band_shape.batch_count, "2D ILWT total batch work items");
    std::vector<tt::tt_metal::CoreCoord> cores = wavelet_program_utils::select_row_major_cores(
        mesh_device,
        std::min(
            wavelet_program_utils::worker_core_count(
                mesh_device, "2D wavelet transforms require at least one worker core"),
            total_work_items),
        "2D ILWT active core count exceeds the worker grid");
    tt::tt_metal::WorkloadDescriptor workload;
    using InverseScheme = typename Scheme::inverse;
    constexpr size_t expected_route_count = 4U * InverseScheme::num_steps;
    for (const auto& chunk : plan.chunks) {
        TT_FATAL(
            chunk.routes.size() == expected_route_count,
            "2D IDWT planner produced {} routes, but the kernel ABI requires {}",
            chunk.routes.size(),
            expected_route_count);
    }
    const size_t route_count = plan.chunks.front().routes.size();
    const uint32_t scratch_tile_count = noc_scratch_tile_count(operation_attributes.boundary_mode, true, route_count);
    const size_t config_capacity = static_cast<size_t>(scratch_tile_count) * kTileBytes / 2;
    TT_FATAL(
        route_count * device_protocol::kLwt2DRouteConfigPageBytes <= config_capacity,
        "2D ILWT route descriptors exceed the per-RISC preload region");

    auto chunk_config = wavelet_program_utils::upload_replicated_dram_metadata(
        mesh_device,
        plan.chunks.size(),
        device_protocol::kLwt2DChunkConfigPageBytes,
        build_ilwt_2d_chunk_config_words(plan),
        workload,
        "2D wavelet metadata");
    auto route_config = wavelet_program_utils::upload_replicated_dram_metadata(
        mesh_device,
        plan.chunks.size() * route_count,
        device_protocol::kLwt2DRouteConfigPageBytes,
        build_ilwt_2d_route_config_words(plan),
        workload,
        "2D wavelet metadata");
    auto band_config = wavelet_program_utils::upload_replicated_dram_metadata(
        mesh_device,
        plan.chunks.size(),
        device_protocol::kLwt2DBandConfigPageBytes,
        build_ilwt_2d_band_config_words(plan),
        workload,
        "2D wavelet metadata");

    WorkingBuffers2D buffers{
        .plane_tile_counts = {},
        .outputs =
            {
                tensor_return_value.buffer(),
                tensor_return_value.buffer(),
                tensor_return_value.buffer(),
                tensor_return_value.buffer(),
            },
        .chunk_config = std::move(chunk_config),
        .route_config = std::move(route_config),
        .band_config = std::move(band_config),
        .cores = std::move(cores),
    };
    for (size_t slot = 0; slot < buffers.plane_tile_counts.size(); ++slot) {
        buffers.plane_tile_counts[slot] =
            checked_u32(plan.allocated_plane_slot_bytes[slot] / kTileBytes, "2D ILWT workspace plane tiles");
    }
    const ArchitecturePolicy architecture_policy = make_architecture_policy(mesh_device.arch());
    auto descriptor = create_program_descriptor(
        core_range_set(buffers.cores),
        band_buffers,
        buffers,
        InverseScheme::compute_scheme_header,
        InverseScheme::compute_scheme_type,
        operation_attributes.boundary_mode,
        architecture_policy.compact_2d_reader,
        true,
        scratch_tile_count);
    const std::vector<CoreChunkWork> work =
        wavelet_program_utils::partition_chunk_work(buffers.cores, total_work_items, "2D ILWT");
    const auto [min_work, max_work] = std::minmax_element(
        work.begin(), work.end(), [](const auto& lhs, const auto& rhs) { return lhs.chunk_count < rhs.chunk_count; });
    log_debug(
        tt::LogOp,
        "ttnn::idwt_2d batch scheduler: B={}, chunks_per_sample={}, total_work_items={}, active_cores={}, "
        "work_items_per_core={}..{}, max_per_core_workspace_bytes={}",
        band_shape.batch_count,
        chunks_per_sample,
        total_work_items,
        buffers.cores.size(),
        min_work->chunk_count,
        max_work->chunk_count,
        plan.allocated_l1_bytes);
    const uint32_t input_tiles_per_sample =
        tile_pages_per_batch_item(tensor_args.ll, band_shape.batch_count, "2D ILWT input band");
    const uint32_t output_tiles_per_sample =
        tile_pages_per_batch_item(tensor_return_value, band_shape.batch_count, "2D ILWT output");
    for (const auto& core_work : work) {
        descriptor.kernels[0].emplace_runtime_args(
            core_work.core,
            inverse_reader_args(band_buffers, plan, buffers, core_work, chunks_per_sample, input_tiles_per_sample));
        bind_compute_args_2d(descriptor.kernels[1], plan, core_work);
        descriptor.kernels[2].emplace_runtime_args(
            core_work.core, inverse_writer_args(plan, buffers, core_work, chunks_per_sample, output_tiles_per_sample));
    }
    wavelet_program_utils::append_program_to_mesh_ranges(
        workload, std::move(descriptor), tensor_coords, "2D wavelet workload has no mesh coordinate range");
    return workload;
}

void validate_forward_inputs_2d(const Lwt2DParams& operation_attributes, const Lwt2DInputs& tensor_args) {
    validate_2d_tensor(tensor_args.input, "2D DWT input");
    const Logical2DShape input_shape = logical_2d_shape(tensor_args.input, "2D DWT input");
    validate_output_memory_config(operation_attributes.output_memory_config, "2D DWT");
    TT_FATAL(
        operation_attributes.scheme_id != SchemeId::kUnknown, "2D DWT received an invalid wavelet scheme identifier");
    TT_FATAL(
        is_supported_lwt_boundary_mode(operation_attributes.boundary_mode),
        "2D DWT received an unsupported boundary mode");
    TT_FATAL(
        !boundary_mode_requires_multiple_samples(operation_attributes.boundary_mode) ||
            (input_shape.height > 1 && input_shape.width > 1),
        "2D DWT reflect and antireflect modes require both dimensions greater than one");

    if (tensor_args.preallocated_outputs.has_value()) {
        const auto specs = detail::compute_lwt_2d_output_specs(operation_attributes, tensor_args);
        const std::array<tt::tt_metal::TensorSpec, 4> expected = {
            std::get<0>(specs), std::get<1>(specs), std::get<2>(specs), std::get<3>(specs)};
        for (size_t index = 0; index < expected.size(); ++index) {
            validate_preallocated_output_2d(
                (*tensor_args.preallocated_outputs)[index],
                expected[index],
                tensor_args.input.device(),
                "2D DWT preallocated output");
            wavelet_tensor_validation::validate_distinct_buffers(
                (*tensor_args.preallocated_outputs)[index],
                tensor_args.input,
                "2D DWT outputs must not alias the input");
            for (size_t previous = 0; previous < index; ++previous) {
                wavelet_tensor_validation::validate_distinct_buffers(
                    (*tensor_args.preallocated_outputs)[index],
                    (*tensor_args.preallocated_outputs)[previous],
                    "2D DWT output bands must not alias each other");
            }
        }
    }
}

void validate_inverse_inputs_2d(const Ilwt2DParams& operation_attributes, const Ilwt2DInputs& tensor_args) {
    const std::array<const Tensor*, 4> inputs = {&tensor_args.ll, &tensor_args.lh, &tensor_args.hl, &tensor_args.hh};
    for (const auto* input : inputs) {
        validate_2d_tensor(*input, "2D IDWT input band");
        wavelet_tensor_validation::validate_same_device(
            *input, tensor_args.ll.device(), "All 2D IDWT bands must be on the same device");
        TT_FATAL(
            input->logical_shape() == tensor_args.ll.logical_shape(), "All 2D IDWT bands must have identical shapes");
    }
    for (size_t index = 0; index < inputs.size(); ++index) {
        for (size_t previous = 0; previous < index; ++previous) {
            wavelet_tensor_validation::validate_distinct_buffers(
                *inputs[index], *inputs[previous], "2D IDWT input bands must not alias");
        }
    }
    validate_output_memory_config(operation_attributes.output_memory_config, "2D IDWT");
    TT_FATAL(
        operation_attributes.output_height > 0 && operation_attributes.output_width > 0,
        "2D IDWT output_shape dimensions must be positive");
    TT_FATAL(
        operation_attributes.scheme_id != SchemeId::kUnknown, "2D IDWT received an invalid wavelet scheme identifier");
    TT_FATAL(
        is_supported_lwt_boundary_mode(operation_attributes.boundary_mode),
        "2D IDWT received an unsupported boundary mode");
    TT_FATAL(
        !boundary_mode_requires_multiple_samples(operation_attributes.boundary_mode) ||
            (operation_attributes.output_height > 1 && operation_attributes.output_width > 1),
        "2D IDWT reflect and antireflect modes require both output dimensions greater than one");

    const auto& info = scheme_info(operation_attributes.scheme_id);
    const uint64_t expected_height =
        (static_cast<uint64_t>(operation_attributes.output_height) + info.tap_size - 1) / 2;
    const uint64_t expected_width = (static_cast<uint64_t>(operation_attributes.output_width) + info.tap_size - 1) / 2;
    const Logical2DShape band_shape = logical_2d_shape(tensor_args.ll, "2D IDWT LL input");
    TT_FATAL(
        band_shape.height == expected_height && band_shape.width == expected_width,
        "2D IDWT band shape {}x{} does not match expected shape {}x{} for output {}x{}",
        band_shape.height,
        band_shape.width,
        expected_height,
        expected_width,
        operation_attributes.output_height,
        operation_attributes.output_width);

    if (tensor_args.preallocated_output.has_value()) {
        validate_preallocated_output_2d(
            *tensor_args.preallocated_output,
            detail::compute_ilwt_2d_output_spec(operation_attributes, tensor_args),
            tensor_args.ll.device(),
            "2D IDWT output");
        for (const auto* input : inputs) {
            wavelet_tensor_validation::validate_distinct_buffers(
                *tensor_args.preallocated_output, *input, "2D IDWT output must not alias an input band");
        }
    }
}

}  // namespace

namespace detail {

tt::tt_metal::WorkloadDescriptor create_lwt_2d_workload(
    const Lwt2DParams& operation_attributes,
    const Lwt2DInputs& tensor_args,
    Lwt2DOutputs& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords) {
    return dispatch_scheme(operation_attributes.scheme_id, [&]<typename Scheme>() {
        return build_forward_workload_2d<Scheme>(operation_attributes, tensor_args, tensor_return_value, tensor_coords);
    });
}

void validate_lwt_2d(const Lwt2DParams& operation_attributes, const Lwt2DInputs& tensor_args) {
    validate_forward_inputs_2d(operation_attributes, tensor_args);
}

Lwt2DOutputSpecs compute_lwt_2d_output_specs(const Lwt2DParams& operation_attributes, const Lwt2DInputs& tensor_args) {
    const auto& info = scheme_info(operation_attributes.scheme_id);
    const Logical2DShape input_shape = logical_2d_shape(tensor_args.input, "2D DWT input");
    const uint64_t height = (static_cast<uint64_t>(input_shape.height) + info.tap_size - 1) / 2;
    const uint64_t width = (static_cast<uint64_t>(input_shape.width) + info.tap_size - 1) / 2;
    TT_FATAL(
        height <= std::numeric_limits<uint32_t>::max() && width <= std::numeric_limits<uint32_t>::max(),
        "2D DWT output dimensions exceed the device uint32 range");
    auto spec = output_spec_2d(
        input_shape,
        static_cast<uint32_t>(height),
        static_cast<uint32_t>(width),
        operation_attributes.output_memory_config);
    return {spec, spec, spec, spec};
}

Lwt2DOutputs create_lwt_2d_output_tensors(const Lwt2DParams& operation_attributes, const Lwt2DInputs& tensor_args) {
    if (tensor_args.preallocated_outputs.has_value()) {
        const auto& outputs = *tensor_args.preallocated_outputs;
        return {outputs[0], outputs[1], outputs[2], outputs[3]};
    }
    auto specs = compute_lwt_2d_output_specs(operation_attributes, tensor_args);
    return {
        create_device_tensor(std::get<0>(specs), tensor_args.input.device()),
        create_device_tensor(std::get<1>(specs), tensor_args.input.device()),
        create_device_tensor(std::get<2>(specs), tensor_args.input.device()),
        create_device_tensor(std::get<3>(specs), tensor_args.input.device()),
    };
}

tt::tt_metal::WorkloadDescriptor create_ilwt_2d_workload(
    const Ilwt2DParams& operation_attributes,
    const Ilwt2DInputs& tensor_args,
    Tensor& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords) {
    return dispatch_scheme(operation_attributes.scheme_id, [&]<typename Scheme>() {
        return build_inverse_workload_2d<Scheme>(operation_attributes, tensor_args, tensor_return_value, tensor_coords);
    });
}

void validate_ilwt_2d(const Ilwt2DParams& operation_attributes, const Ilwt2DInputs& tensor_args) {
    validate_inverse_inputs_2d(operation_attributes, tensor_args);
}

tt::tt_metal::TensorSpec compute_ilwt_2d_output_spec(
    const Ilwt2DParams& operation_attributes, const Ilwt2DInputs& tensor_args) {
    return output_spec_2d(
        logical_2d_shape(tensor_args.ll, "2D ILWT LL input"),
        operation_attributes.output_height,
        operation_attributes.output_width,
        operation_attributes.output_memory_config);
}

Tensor create_ilwt_2d_output_tensor(const Ilwt2DParams& operation_attributes, const Ilwt2DInputs& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return *tensor_args.preallocated_output;
    }
    return create_device_tensor(
        compute_ilwt_2d_output_spec(operation_attributes, tensor_args), tensor_args.ll.device());
}

}  // namespace detail

}  // namespace ttnn::prim
