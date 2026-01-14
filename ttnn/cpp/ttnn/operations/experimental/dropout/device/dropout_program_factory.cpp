// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "dropout_program_factory.hpp"

#include "dropout_device_operation_types.hpp"
#include "tt-metalium/mesh_workload.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::dropout::program {
namespace {
constexpr auto kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/dataflow/writer_dropout_interleaved_start_id.cpp";
constexpr auto kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/dataflow/reader_dropout_interleaved_start_id.cpp";
constexpr auto kComputeKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/compute/dropout_kernel.cpp";
constexpr auto kSeedIdx = 0;
constexpr auto kDstBufferIdx = 0;
constexpr auto kSrcBufferIdx = 0;

constexpr auto kSrc0CbIndex = tt::CBIndex::c_0;
constexpr auto kOutputCbIndex = tt::CBIndex::c_2;

constexpr uint32_t kNumInputTiles = 2;
constexpr uint32_t kNumOutputTiles = 2;

// Overrides the seed with a per-device seed by using the device ID as an offset.
operation_attributes_t override_per_device_seed(
    const operation_attributes_t& args, const ttnn::MeshCoordinate& mesh_coord, const ttnn::Tensor& input_tensor) {
    operation_attributes_t args_with_per_device_seed = args;
    args_with_per_device_seed.seed += input_tensor.device()->get_device(mesh_coord)->id();
    return args_with_per_device_seed;
}

}  // namespace

using namespace tt::constants;

/**
 *   Helper struct to hold references to all kernels we create,
 *        used during runtime argument setup.
 */
struct DropoutKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute_group_1;
    tt::tt_metal::KernelHandle compute_group_2;
};

/**
 *   Create and configure a circular buffer, returning both the configuration and the handle.
 */
inline tt::tt_metal::CBHandle create_circular_buffer(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t single_tile_size,
    uint32_t num_tiles) {
    using namespace tt::tt_metal;

    CircularBufferConfig cb_config = CircularBufferConfig(num_tiles * single_tile_size, {{cb_index, data_format}})
                                         .set_page_size(cb_index, single_tile_size);

    auto cb_handle = CreateCircularBuffer(program, core_ranges, cb_config);
    return cb_handle;
}

/**
 *   Create a reader kernel with the given compile-time arguments.
 */
inline tt::tt_metal::KernelHandle create_reader_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::string& kernel_path) {
    using namespace tt::tt_metal;

    return CreateKernel(program, kernel_path, core_ranges, ReaderDataMovementConfig(compile_time_args));
}

/**
 *   Create a writer kernel with the given compile-time arguments.
 */
inline tt::tt_metal::KernelHandle create_writer_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::string& kernel_path) {
    using namespace tt::tt_metal;

    return CreateKernel(program, kernel_path, core_ranges, WriterDataMovementConfig(compile_time_args));
}

/**
 * Create a compute kernel (for dropout) with the given compile-time arguments.
 */
inline tt::tt_metal::KernelHandle create_compute_kernel(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    const std::vector<uint32_t>& compile_time_args,
    const std::string& kernel_path,
    bool math_approx_mode) {
    using namespace tt::tt_metal;

    return CreateKernel(
        program,
        kernel_path,
        core_ranges,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = math_approx_mode,
            .compile_args = compile_time_args});
}

/**
 * Set up the runtime arguments for the 4 relevant kernels (reader, writer, compute G1, compute G2)
 *        for each core in the grid.
 */
inline void assign_per_core_runtime_args(
    tt::tt_metal::Program& program,
    const DropoutKernels& kernels,
    const tt::tt_metal::Buffer* src_buffer,
    const tt::tt_metal::Buffer* dst_buffer,
    uint32_t num_cores,
    uint32_t num_cores_y,
    uint32_t num_tiles_per_core_group_1,
    uint32_t num_tiles_per_core_group_2,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2,
    uint32_t seed) {
    using namespace tt::tt_metal;

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
            SetRuntimeArgs(program, kernels.compute_group_1, core, {seed});
        } else if (core_group_2.contains(core)) {
            SetRuntimeArgs(program, kernels.compute_group_2, core, {seed});
        } else {
            TT_THROW("Core not in specified core ranges.");
        }
        // Reader kernel: (src_addr, number_of_tiles, offset_in_tiles)
        SetRuntimeArgs(program, kernels.reader, core, {src_buffer->address(), num_tiles_per_core, num_tiles_written});

        // Writer kernel: (dst_addr, number_of_tiles, offset_in_tiles)
        SetRuntimeArgs(program, kernels.writer, core, {dst_buffer->address(), num_tiles_per_core, num_tiles_written});

        num_tiles_written += num_tiles_per_core;
    }
}

DropoutProgramFactory::cached_program_t DropoutProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;
    auto* device = input.device();

    tt::tt_metal::Program program{};

    tt::DataFormat data_fmt_in = datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat data_fmt_out = datatype_to_dataformat_converter(output.dtype());

    uint32_t single_tile_size_in = tt::tile_size(data_fmt_in);
    uint32_t single_tile_size_out = tt::tile_size(data_fmt_out);

    uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    // -------------------------------------------------------------------------
    // 2) Create and configure circular buffers
    // -------------------------------------------------------------------------

    create_circular_buffer(program, all_cores, kSrc0CbIndex, data_fmt_in, single_tile_size_in, kNumInputTiles);

    create_circular_buffer(program, all_cores, kOutputCbIndex, data_fmt_out, single_tile_size_out, kNumOutputTiles);

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------
    auto* src_buffer = input.buffer();
    std::vector<uint32_t> reader_compile_args = {static_cast<uint32_t>(kSrc0CbIndex)};
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(reader_compile_args);

    auto* dst_buffer = output.buffer();
    std::vector<uint32_t> writer_compile_args = {static_cast<uint32_t>(kOutputCbIndex)};
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(writer_compile_args);

    DropoutKernels kernels{};
    kernels.reader = create_reader_kernel(program, all_cores, reader_compile_args, kReaderKernelPath);

    kernels.writer = create_writer_kernel(program, all_cores, writer_compile_args, kWriterKernelPath);

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
        create_compute_kernel(program, core_group_1, compute_group_1_args, kComputeKernelPath, math_approx_mode);

    // Group 2 (if present) compile-time arguments
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_group_2_args = {
            num_tiles_per_core_group_2,  // per_core_block_cnt
            1,                           // per_core_block_size
            prob_int,                    // prob
            uscale                       // scale
        };

        kernels.compute_group_2 =
            create_compute_kernel(program, core_group_2, compute_group_2_args, kComputeKernelPath, math_approx_mode);
    }

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    assign_per_core_runtime_args(
        program,
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
    // 6) Return the fully configured program & relevant shared variables
    // -------------------------------------------------------------------------
    return cached_program_t{
        std::move(program),
        {/* dropout_reader_kernel_id  = */ kernels.reader,
         /* dropout_writer_kernel_id  = */ kernels.writer,
         /* dropout_kernel_group_1_id = */ kernels.compute_group_1,
         /* dropout_kernel_group_2_id = */ kernels.compute_group_2,
         /* core_group_1              = */ core_group_1,
         /* core_group_2              = */ core_group_2,
         /* num_cores                 = */ num_cores,
         /* num_cores_y               = */ num_cores_y}};
}

void DropoutProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    Tensor& output) {
    using namespace tt::tt_metal;

    auto& shared_vars = cached_program.shared_variables;
    auto& dropout_reader_kernel = shared_vars.dropout_reader_kernel_id;
    auto& dropout_writer_kernel = shared_vars.dropout_writer_kernel_id;
    auto& dropout_group_1_kernel = shared_vars.dropout_kernel_group_1_id;
    auto& dropout_group_2_kernel = shared_vars.dropout_kernel_group_2_id;
    auto& core_group_1 = shared_vars.core_group_1;
    auto& core_group_2 = shared_vars.core_group_2;
    auto& program = cached_program.program;

    uint32_t num_cores = shared_vars.num_cores;
    uint32_t num_cores_y = shared_vars.num_cores_y;

    const auto& input = tensor_args.input;
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    // Only seed/address arguments need updating here; tile counts remain the same as in create().
    auto& reader_runtime_args = GetRuntimeArgs(program, dropout_reader_kernel);
    auto& writer_runtime_args = GetRuntimeArgs(program, dropout_writer_kernel);
    auto& group_1_runtime_args = GetRuntimeArgs(program, dropout_group_1_kernel);
    // we need to initialize it with something, but if group 2 is  empty it will be used in the loop
    auto& group_2_runtime_args =
        core_group_2.ranges().empty() ? group_1_runtime_args : GetRuntimeArgs(program, dropout_group_2_kernel);

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Update the source address for the reader kernel
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[kSrcBufferIdx] = src_buffer->address();
        }
        // Update the destination address for the writer kernel
        {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[kDstBufferIdx] = dst_buffer->address();
        }
        // Update the seed for the compute kernels
        if (core_group_1.contains(core)) {
            auto& runtime_args = group_1_runtime_args[core.x][core.y];
            runtime_args[kSeedIdx] = operation_attributes.seed;
        } else if (core_group_2.contains(core)) {
            auto& runtime_args = group_2_runtime_args[core.x][core.y];
            runtime_args[kSeedIdx] = operation_attributes.seed;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }
    }
}

DropoutMeshWorkloadFactory::cached_mesh_workload_t DropoutMeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    Tensor& output) {
    TT_ASSERT(args.use_per_device_seed, "DropoutMeshWorkloadFactory should only be used if per-device seed is used.");

    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& mesh_coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : mesh_coord_range) {
            const ttnn::MeshCoordinateRange mesh_coord_range{mesh_coord, mesh_coord};
            auto single_device_program = DropoutProgramFactory::create(
                override_per_device_seed(args, mesh_coord, tensor_args.input), tensor_args, output);
            shared_variables[mesh_coord_range] = std::move(single_device_program.shared_variables);
            workload.add_program(mesh_coord_range, std::move(single_device_program.program));
        }
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

void DropoutMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    TT_ASSERT(args.use_per_device_seed, "DropoutMeshWorkloadFactory should only be used if per-device seed is used.");

    for (auto& [mesh_coord_range, program] : cached_workload.workload.get_programs()) {
        auto cached_program_proxy = DropoutProgramFactory::cached_program_t::proxy(
            program, cached_workload.shared_variables.at(mesh_coord_range));
        DropoutProgramFactory::override_runtime_arguments(
            cached_program_proxy,
            override_per_device_seed(args, mesh_coord_range.start_coord(), tensor_args.input),
            tensor_args,
            tensor_return_value);
    }
}

}  // namespace ttnn::operations::experimental::dropout::program
