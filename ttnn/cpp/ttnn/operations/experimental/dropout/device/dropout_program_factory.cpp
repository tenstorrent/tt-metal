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

// Overrides the seed with a per-device seed by using the device ID as an offset.
DropoutParams override_per_device_seed(
    const DropoutParams& args,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate,
    const ttnn::Tensor& input_tensor) {
    DropoutParams args_with_per_device_seed = args;
    if (mesh_dispatch_coordinate.has_value()) {
        args_with_per_device_seed.seed += input_tensor.device()->get_device(*mesh_dispatch_coordinate)->id();
    } else {
        args_with_per_device_seed.seed += input_tensor.device()->id();
    }
    return args_with_per_device_seed;
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

/**
 * Set up the runtime arguments for the relevant kernels (reader, writer, compute G1, compute G2)
 *        for each core in the grid.
 */
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

inline void assign_per_core_runtime_args(
    DropoutKernels& kernels,
    tt::tt_metal::Buffer* src_buffer,
    tt::tt_metal::Buffer* dst_buffer,
    uint32_t num_cores,
    uint32_t num_cores_y,
    uint32_t num_tiles_per_core_group_1,
    uint32_t num_tiles_per_core_group_2,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2,
    uint32_t seed) {
    using namespace tt::tt_metal;

    kernels.reader.runtime_args.reserve(num_cores);
    kernels.writer.runtime_args.reserve(num_cores);
    kernels.compute_group_1.runtime_args.reserve(num_cores);
    if (kernels.compute_group_2.has_value()) {
        kernels.compute_group_2->runtime_args.reserve(num_cores);
    }

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Determine how many tiles this core will process
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        if (core_group_1.contains(core)) {
            kernels.compute_group_1.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{seed});
        } else if (core_group_2.contains(core)) {
            TT_FATAL(kernels.compute_group_2.has_value(), "Core group 2 descriptor should be present");
            kernels.compute_group_2->runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{seed});
        } else {
            TT_THROW("Core not in specified core ranges.");
        }
        // Reader kernel: (src_addr, number_of_tiles, offset_in_tiles).  src/dst go in as Buffer*
        // bindings for the cache-miss build; their addresses are re-applied on every cache hit by
        // override_runtime_arguments() (correct for the input==output in-place case).
        kernels.reader.emplace_runtime_args(core, {src_buffer, num_tiles_per_core, num_tiles_written});

        // Writer kernel: (dst_addr, number_of_tiles, offset_in_tiles)
        kernels.writer.emplace_runtime_args(core, {dst_buffer, num_tiles_per_core, num_tiles_written});

        num_tiles_written += num_tiles_per_core;
    }
}

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

    auto
        [num_cores,
         num_cores_y,
         all_cores,
         core_group_1,
         core_group_2,
         num_tiles_per_core_group_1,
         num_tiles_per_core_group_2] = dropout_core_split(input);

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
    assign_per_core_runtime_args(
        kernels,
        src_buffer,
        dst_buffer,
        num_cores,
        num_cores_y,
        num_tiles_per_core_group_1,
        num_tiles_per_core_group_2,
        core_group_1,
        core_group_2,
        args.seed);

    // -------------------------------------------------------------------------
    // 6) Return the fully configured descriptor
    // -------------------------------------------------------------------------
    descriptor.kernels.push_back(std::move(kernels.reader));
    descriptor.kernels.push_back(std::move(kernels.writer));
    descriptor.kernels.push_back(std::move(kernels.compute_group_1));
    if (kernels.compute_group_2.has_value()) {
        descriptor.kernels.push_back(std::move(*kernels.compute_group_2));
    }

    return descriptor;
}

tt::tt_metal::ProgramDescriptor DropoutMeshWorkloadFactory::create_descriptor(
    const DropoutParams& args,
    const DropoutInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_ASSERT(args.use_per_device_seed, "DropoutMeshWorkloadFactory should only be used if per-device seed is used.");
    const auto effective_args = override_per_device_seed(args, mesh_dispatch_coordinate, tensor_args.input);
    return DropoutProgramFactory::create_descriptor(effective_args, tensor_args, output);
}

namespace {
// Re-apply every per-dispatch arg (seed and the src/dst addresses) to the cached program, mirroring
// exactly what assign_per_core_runtime_args baked on the cache-miss build. override replaces
// resolve_bindings, so the addresses are ours to re-apply too. Kernel push order: reader(0),
// writer(1), compute_group_1(2), compute_group_2(3 if present).
void apply_dropout_runtime_args(
    tt::tt_metal::Program& program, const DropoutInputs& tensor_args, Tensor& output, uint32_t seed) {
    using namespace tt::tt_metal;

    constexpr uint32_t kReaderIdx = 0;
    constexpr uint32_t kWriterIdx = 1;
    constexpr uint32_t kComputeGroup1Idx = 2;
    constexpr uint32_t kComputeGroup2Idx = 3;

    const auto& input = tensor_args.input;
    [[maybe_unused]] const auto
        [num_cores, num_cores_y, all_cores, core_group_1, core_group_2, tpc_group_1, tpc_group_2] =
            dropout_core_split(input);

    const uint32_t src_addr = input.buffer()->address();
    const uint32_t dst_addr = output.buffer()->address();

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = core_group_1.contains(core) ? tpc_group_1 : tpc_group_2;

        auto& reader_args = GetRuntimeArgs(program, kReaderIdx, core);
        reader_args[0] = src_addr;
        reader_args[1] = num_tiles_per_core;
        reader_args[2] = num_tiles_written;

        auto& writer_args = GetRuntimeArgs(program, kWriterIdx, core);
        writer_args[0] = dst_addr;
        writer_args[1] = num_tiles_per_core;
        writer_args[2] = num_tiles_written;

        if (core_group_1.contains(core)) {
            GetRuntimeArgs(program, kComputeGroup1Idx, core)[0] = seed;
        } else if (core_group_2.contains(core)) {
            GetRuntimeArgs(program, kComputeGroup2Idx, core)[0] = seed;
        }

        num_tiles_written += num_tiles_per_core;
    }
}
}  // namespace

void DropoutProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const DropoutParams& args,
    const DropoutInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    apply_dropout_runtime_args(program, tensor_args, output, args.seed);
}

void DropoutMeshWorkloadFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const DropoutParams& args,
    const DropoutInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    const uint32_t seed = override_per_device_seed(args, mesh_dispatch_coordinate, tensor_args.input).seed;
    apply_dropout_runtime_args(program, tensor_args, output, seed);
}

}  // namespace ttnn::experimental::prim
