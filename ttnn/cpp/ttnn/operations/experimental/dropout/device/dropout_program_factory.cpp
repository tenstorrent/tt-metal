// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <string_view>

#include "dropout_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::experimental::prim {
namespace {
constexpr auto kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/dataflow/writer_dropout_interleaved_start_id.cpp";
constexpr auto kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/dataflow/reader_dropout_interleaved_start_id.cpp";
constexpr auto kComputeKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/compute/dropout_kernel.cpp";

constexpr auto kSrc0CbIndex = tt::CBIndex::c_0;
constexpr auto kOutputCbIndex = tt::CBIndex::c_2;

constexpr uint32_t kNumInputTiles = 2;
constexpr uint32_t kNumOutputTiles = 2;

// Offsets the seed by the device ID so each device of a mesh draws a different mask.
// Single-sourced: used by DropoutMeshWorkloadFactory::create_descriptor (cache miss) and by
// override_runtime_arguments (cache hit), which must reproduce the same seed bit-for-bit.
uint32_t per_device_seed(
    const DropoutParams& args,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate,
    const ttnn::Tensor& input_tensor) {
    auto* device = input_tensor.device();
    return args.seed +
           (mesh_dispatch_coordinate.has_value() ? device->get_device(*mesh_dispatch_coordinate)->id() : device->id());
}

}  // namespace

using namespace tt::constants;

/**
 *   Helper struct to hold references to all kernels we create,
 *        used during runtime argument setup.
 */
struct DropoutKernels {
    tt::tt_metal::KernelDescriptor reader;
    tt::tt_metal::KernelDescriptor writer;
    tt::tt_metal::KernelDescriptor compute_group_1;
    std::optional<tt::tt_metal::KernelDescriptor> compute_group_2;
};

/**
 *   Create and configure a circular buffer descriptor.
 */
inline void create_circular_buffer(
    tt::tt_metal::ProgramDescriptor& descriptor,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t single_tile_size,
    uint32_t num_tiles) {
    using namespace tt::tt_metal;

    descriptor.cbs.push_back(CBDescriptor{
        .total_size = num_tiles * single_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_index,
            .data_format = data_format,
            .page_size = single_tile_size,
        }}},
    });
}

/**
 *   Create a reader kernel descriptor with the given compile-time arguments.
 */
inline tt::tt_metal::KernelDescriptor create_reader_kernel(
    const tt::tt_metal::CoreRangeSet& core_ranges,
    tt::tt_metal::KernelDescriptor::CompileTimeArgs&& compile_time_args,
    std::string_view kernel_path) {
    using namespace tt::tt_metal;

    KernelDescriptor descriptor;
    descriptor.kernel_source = kernel_path;
    descriptor.source_type = KernelDescriptor::SourceType::FILE_PATH;
    descriptor.core_ranges = core_ranges;
    descriptor.compile_time_args = std::move(compile_time_args);
    descriptor.config = ReaderConfigDescriptor{};
    return descriptor;
}

/**
 *   Create a writer kernel descriptor with the given compile-time arguments.
 */
inline tt::tt_metal::KernelDescriptor create_writer_kernel(
    const tt::tt_metal::CoreRangeSet& core_ranges,
    tt::tt_metal::KernelDescriptor::CompileTimeArgs&& compile_time_args,
    std::string_view kernel_path) {
    using namespace tt::tt_metal;

    KernelDescriptor descriptor;
    descriptor.kernel_source = kernel_path;
    descriptor.source_type = KernelDescriptor::SourceType::FILE_PATH;
    descriptor.core_ranges = core_ranges;
    descriptor.compile_time_args = std::move(compile_time_args);
    descriptor.config = WriterConfigDescriptor{};
    return descriptor;
}

/**
 * Create a compute kernel descriptor (for dropout) with the given compile-time arguments.
 */
inline tt::tt_metal::KernelDescriptor create_compute_kernel(
    const tt::tt_metal::CoreRangeSet& core_ranges,
    tt::tt_metal::KernelDescriptor::CompileTimeArgs&& compile_time_args,
    std::string_view kernel_path,
    bool math_approx_mode) {
    using namespace tt::tt_metal;

    KernelDescriptor descriptor;
    descriptor.kernel_source = kernel_path;
    descriptor.source_type = KernelDescriptor::SourceType::FILE_PATH;
    descriptor.core_ranges = core_ranges;
    descriptor.compile_time_args = std::move(compile_time_args);
    descriptor.config = ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .dst_full_sync_en = false,
        .math_approx_mode = math_approx_mode,
    };
    return descriptor;
}

// Work split used by create_descriptor (cache miss) and override_runtime_arguments (cache hit).
struct DropoutCoreSplit {
    uint32_t num_cores = 0;
    uint32_t num_cores_y = 0;
    tt::tt_metal::CoreRangeSet all_cores;
    tt::tt_metal::CoreRangeSet core_group_1;
    tt::tt_metal::CoreRangeSet core_group_2;
    uint32_t num_tiles_per_core_group_1 = 0;
    uint32_t num_tiles_per_core_group_2 = 0;
};

DropoutCoreSplit dropout_core_split(const Tensor& input) {
    auto grid = input.device()->compute_with_storage_grid_size();
    uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(grid, num_tiles);
    return {
        num_cores,
        grid.y,
        all_cores,
        core_group_1,
        core_group_2,
        num_tiles_per_core_group_1,
        num_tiles_per_core_group_2};
}

// Per-core slice of the work split.
struct DropoutCoreWork {
    tt::tt_metal::CoreCoord core;
    uint32_t num_tiles = 0;
    uint32_t tile_offset = 0;
    bool in_group_1 = false;  // otherwise in core_group_2
};

// Walks the cores in the exact order create_descriptor emplaces runtime args for them. Shared with
// override_runtime_arguments so the per-core layout the cache-hit patch writes cannot drift from the
// one the cache-miss build baked.
template <typename Fn>
void for_each_dropout_core(const DropoutCoreSplit& split, const Fn& fn) {
    for (uint32_t i = 0, num_tiles_written = 0; i < split.num_cores; i++) {
        const tt::tt_metal::CoreCoord core = {i / split.num_cores_y, i % split.num_cores_y};
        const bool in_group_1 = split.core_group_1.contains(core);
        TT_FATAL(
            in_group_1 || split.core_group_2.contains(core),
            "Core ({}, {}) is not in the specified core ranges",
            core.x,
            core.y);
        const uint32_t num_tiles = in_group_1 ? split.num_tiles_per_core_group_1 : split.num_tiles_per_core_group_2;

        fn(DropoutCoreWork{core, num_tiles, num_tiles_written, in_group_1});

        num_tiles_written += num_tiles;
    }
}

/**
 * Set up the runtime arguments for the relevant kernels (reader, writer, compute G1, compute G2)
 *        for each core in the grid.
 */
inline void assign_per_core_runtime_args(
    DropoutKernels& kernels,
    tt::tt_metal::Buffer* src_buffer,
    tt::tt_metal::Buffer* dst_buffer,
    const DropoutCoreSplit& split,
    uint32_t seed) {
    using namespace tt::tt_metal;

    kernels.reader.runtime_args.reserve(split.num_cores);
    kernels.writer.runtime_args.reserve(split.num_cores);
    kernels.compute_group_1.runtime_args.reserve(split.num_cores);
    if (kernels.compute_group_2.has_value()) {
        kernels.compute_group_2->runtime_args.reserve(split.num_cores);
    }

    for_each_dropout_core(split, [&](const DropoutCoreWork& work) {
        // Compute kernel: (seed)
        if (work.in_group_1) {
            kernels.compute_group_1.runtime_args.emplace_back(work.core, KernelDescriptor::CoreRuntimeArgs{seed});
        } else {
            TT_FATAL(kernels.compute_group_2.has_value(), "Core group 2 descriptor should be present");
            kernels.compute_group_2->runtime_args.emplace_back(work.core, KernelDescriptor::CoreRuntimeArgs{seed});
        }

        // Reader kernel: (src_addr, number_of_tiles, offset_in_tiles).  src/dst go in as Buffer*
        // bindings so this cache-miss build resolves their current addresses; on a cache hit
        // override_runtime_arguments re-applies them (correct for the input==output in-place case).
        kernels.reader.emplace_runtime_args(work.core, {src_buffer, work.num_tiles, work.tile_offset});

        // Writer kernel: (dst_addr, number_of_tiles, offset_in_tiles)
        kernels.writer.emplace_runtime_args(work.core, {dst_buffer, work.num_tiles, work.tile_offset});
    });
}

namespace {
// Kernel indices: positions in the `descriptor.kernels` push order at the end of create_descriptor
// (reader, writer, compute group 1, then compute group 2 only when core_group_2 is non-empty).
// Single-sourced here, next to the pushes that define the order, so the cache-miss push order and the
// cache-hit GetRuntimeArgs indices in override_runtime_arguments stay one edit apart.
enum : uint32_t { kReaderIdx, kWriterIdx, kComputeGroup1Idx, kComputeGroup2Idx };
}  // namespace

tt::tt_metal::ProgramDescriptor DropoutProgramFactory::create_descriptor(
    const DropoutParams& args, const DropoutInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;

    ProgramDescriptor descriptor{};

    tt::DataFormat data_fmt_in = datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat data_fmt_out = datatype_to_dataformat_converter(output.dtype());

    uint32_t single_tile_size_in = tt::tile_size(data_fmt_in);
    uint32_t single_tile_size_out = tt::tile_size(data_fmt_out);

    // Kept whole so it can be handed to for_each_dropout_core, the walk shared with the cache-hit patch.
    const auto split = dropout_core_split(input);
    const auto& all_cores = split.all_cores;
    const auto& core_group_1 = split.core_group_1;
    const auto& core_group_2 = split.core_group_2;
    const uint32_t num_tiles_per_core_group_1 = split.num_tiles_per_core_group_1;
    const uint32_t num_tiles_per_core_group_2 = split.num_tiles_per_core_group_2;

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------
    create_circular_buffer(descriptor, all_cores, kSrc0CbIndex, data_fmt_in, single_tile_size_in, kNumInputTiles);

    create_circular_buffer(descriptor, all_cores, kOutputCbIndex, data_fmt_out, single_tile_size_out, kNumOutputTiles);

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------
    auto* src_buffer = input.buffer();
    KernelDescriptor::CompileTimeArgs reader_compile_args = {static_cast<uint32_t>(kSrc0CbIndex)};
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(reader_compile_args);

    auto* dst_buffer = output.buffer();
    KernelDescriptor::CompileTimeArgs writer_compile_args = {static_cast<uint32_t>(kOutputCbIndex)};
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(writer_compile_args);

    DropoutKernels kernels{
        .reader = create_reader_kernel(all_cores, std::move(reader_compile_args), kReaderKernelPath),
        .writer = create_writer_kernel(all_cores, std::move(writer_compile_args), kWriterKernelPath),
    };

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for dropout
    // -------------------------------------------------------------------------
    uint32_t uscale = std::bit_cast<uint32_t>(args.scale);

    // Convert probability (args.prob) to integer representation
    uint32_t prob_int = static_cast<uint32_t>(static_cast<double>(INT_MAX) * args.prob);

    // Group 1 compile-time arguments
    std::vector<uint32_t> compute_group_1_args = {
        num_tiles_per_core_group_1,  // per_core_block_cnt
        1,                           // per_core_block_size
        prob_int,                    // prob
        uscale                       // scale
    };

    bool math_approx_mode = false;

    kernels.compute_group_1 =
        create_compute_kernel(core_group_1, std::move(compute_group_1_args), kComputeKernelPath, math_approx_mode);

    // Group 2 (if present) compile-time arguments
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            num_tiles_per_core_group_2,  // per_core_block_cnt
            1,                           // per_core_block_size
            prob_int,                    // prob
            uscale                       // scale
        };

        kernels.compute_group_2 =
            create_compute_kernel(core_group_2, std::move(compute_group_2_args), kComputeKernelPath, math_approx_mode);
    }

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    assign_per_core_runtime_args(kernels, src_buffer, dst_buffer, split, args.seed);

    // -------------------------------------------------------------------------
    // 6) Return the fully configured descriptor
    // -------------------------------------------------------------------------
    descriptor.kernels.push_back(std::move(kernels.reader));           // kReaderIdx
    descriptor.kernels.push_back(std::move(kernels.writer));           // kWriterIdx
    descriptor.kernels.push_back(std::move(kernels.compute_group_1));  // kComputeGroup1Idx
    if (kernels.compute_group_2.has_value()) {
        descriptor.kernels.push_back(std::move(*kernels.compute_group_2));  // kComputeGroup2Idx
    }

    return descriptor;
}

tt::tt_metal::ProgramDescriptor DropoutMeshWorkloadFactory::create_descriptor(
    const DropoutParams& args,
    const DropoutInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_ASSERT(args.use_per_device_seed, "DropoutMeshWorkloadFactory should only be used if per-device seed is used.");
    DropoutParams effective_args = args;
    effective_args.seed = per_device_seed(args, mesh_dispatch_coordinate, tensor_args.input);
    return DropoutProgramFactory::create_descriptor(effective_args, tensor_args, output);
}

void DropoutProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const DropoutParams& operation_attributes,
    const DropoutInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    using namespace tt::tt_metal;

    // Kernel indices (kReaderIdx/kWriterIdx/kComputeGroup{1,2}Idx) are shared with create_descriptor's
    // `descriptor.kernels` push order -- see the enum defined next to those pushes.
    const auto& input = tensor_args.input;

    // `seed` is excluded from the program hash, so the cached program still carries the first miss's
    // seed and it must be re-derived here exactly as the selected factory derives it:
    // DropoutMeshWorkloadFactory offsets it by the dispatch coordinate's device id.
    const uint32_t seed = operation_attributes.use_per_device_seed
                              ? per_device_seed(operation_attributes, mesh_dispatch_coordinate, input)
                              : operation_attributes.seed;

    // override_runtime_arguments supersedes resolve_bindings, so the buffer addresses the descriptor
    // emplaced as Buffer* bindings are ours to re-apply. Each address comes from its own tensor, which
    // is what makes the in-place case (input == output, so both addresses equal) correct.
    const uint32_t src_addr = input.buffer()->address();
    const uint32_t dst_addr = tensor_return_value.buffer()->address();

    // Everything else the descriptor emplaced (tile counts/offsets) derives from the input spec and the
    // compute grid, both fixed on a cache hit; rewritten anyway since the shared walk already has them.
    // Both CBs are program-local (no .buffer/.tensor binding), so there is no CB address to re-point.
    // Hoist the per-kernel lookup: the whole-kernel overload hands back the [x][y] grid, so this costs
    // one lookup per kernel instead of one per core per kernel (same amortisation apply_resolved_bindings does).
    const DropoutCoreSplit split = dropout_core_split(input);
    auto& reader_grid = GetRuntimeArgs(program, kReaderIdx);
    auto& writer_grid = GetRuntimeArgs(program, kWriterIdx);
    auto& compute_group_1_grid = GetRuntimeArgs(program, kComputeGroup1Idx);
    auto* compute_group_2_grid =
        split.core_group_2.ranges().empty() ? nullptr : &GetRuntimeArgs(program, kComputeGroup2Idx);

    for_each_dropout_core(split, [&](const DropoutCoreWork& work) {
        auto& reader_args = reader_grid[work.core.x][work.core.y];
        reader_args[0] = src_addr;
        reader_args[1] = work.num_tiles;
        reader_args[2] = work.tile_offset;

        auto& writer_args = writer_grid[work.core.x][work.core.y];
        writer_args[0] = dst_addr;
        writer_args[1] = work.num_tiles;
        writer_args[2] = work.tile_offset;

        TT_FATAL(work.in_group_1 || compute_group_2_grid != nullptr, "Core group 2 kernel should be present");
        auto& compute_grid = work.in_group_1 ? compute_group_1_grid : *compute_group_2_grid;
        compute_grid[work.core.x][work.core.y][0] = seed;
    });
}

void DropoutMeshWorkloadFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const DropoutParams& operation_attributes,
    const DropoutInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    DropoutProgramFactory::override_runtime_arguments(
        program, operation_attributes, tensor_args, tensor_return_value, mesh_dispatch_coordinate);
}

}  // namespace ttnn::experimental::prim
