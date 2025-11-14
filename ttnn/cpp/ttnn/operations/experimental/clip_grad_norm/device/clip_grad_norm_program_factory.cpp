// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "clip_grad_norm_program_factory.hpp"
#include "clip_grad_norm_device_operation_types.hpp"
#include "tt-metalium/mesh_workload.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::experimental::clip_grad_norm::program {
namespace {

constexpr auto kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/clip_grad_norm/device/kernels/dataflow/"
    "writer_clip_grad_norm_interleaved_start_id.cpp";
constexpr auto kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/clip_grad_norm/device/kernels/dataflow/reader_clip_grad_norm.cpp";
constexpr auto kComputeKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/clip_grad_norm/device/kernels/compute/clip_grad_norm_kernel.cpp";

constexpr auto kSrc0CbIndex = tt::CBIndex::c_0;
constexpr auto kOutputCbIndex = tt::CBIndex::c_2;
constexpr auto kScalerCbIndex = tt::CBIndex::c_1;
constexpr auto kNormPartialCbIndex = tt::CBIndex::c_3;   // Per-core partial norm (compute → reader)
constexpr auto kNormGlobalCbIndex = tt::CBIndex::c_5;    // Global norm (sender → all cores)
constexpr auto kNormExternalCbIndex = tt::CBIndex::c_6;  // All partials for reduction (sender only)
constexpr auto kTempCbIndex = tt::CBIndex::c_4;

constexpr uint32_t kNumInputTiles = 2;
constexpr uint32_t kNumOutputTiles = 2;
constexpr uint32_t kNumScalerTiles = 1;
constexpr uint32_t kNumNormPartialTiles = 2;  // Need 2 tiles: 1 for compute to write, 1 for reader to wait on
constexpr uint32_t kNumNormGlobalTiles = 1;   // Single tile for global norm
constexpr uint32_t kNumTempTiles = 1;

struct ClipGradNormKernels {
    tt::tt_metal::KernelHandle reader;
    tt::tt_metal::KernelHandle writer;
    tt::tt_metal::KernelHandle compute_group_1;
    tt::tt_metal::KernelHandle compute_group_2;
};

struct AllReduceConfig {
    std::vector<uint32_t> remote_noc_x;
    std::vector<uint32_t> remote_noc_y;
    uint32_t mcast_start_x;
    uint32_t mcast_start_y;
    uint32_t mcast_end_x;
    uint32_t mcast_end_y;
    tt::tt_metal::CoreCoord sender_physical;
};

inline void create_circular_buffer(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& core_ranges,
    uint32_t cb_index,
    tt::DataFormat data_format,
    uint32_t single_tile_size,
    uint32_t num_tiles) {
    using namespace tt::tt_metal;
    CircularBufferConfig cb_config(num_tiles * single_tile_size, {{cb_index, data_format}});
    cb_config.set_page_size(cb_index, single_tile_size);
    CreateCircularBuffer(program, core_ranges, cb_config);
}

inline void create_circular_buffers(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& all_cores,
    const tt::tt_metal::CoreRangeSet& sender_cores,
    uint32_t num_cores,
    tt::DataFormat data_fmt_in,
    tt::DataFormat data_fmt_out,
    uint32_t single_tile_size_in,
    uint32_t single_tile_size_out) {
    create_circular_buffer(program, all_cores, kSrc0CbIndex, data_fmt_in, single_tile_size_in, kNumInputTiles);
    create_circular_buffer(program, all_cores, kOutputCbIndex, data_fmt_out, single_tile_size_out, kNumOutputTiles);
    create_circular_buffer(program, all_cores, kScalerCbIndex, data_fmt_in, single_tile_size_in, kNumScalerTiles);
    create_circular_buffer(
        program, all_cores, kNormPartialCbIndex, data_fmt_in, single_tile_size_in, kNumNormPartialTiles);
    create_circular_buffer(
        program, all_cores, kNormGlobalCbIndex, data_fmt_in, single_tile_size_in, kNumNormGlobalTiles);
    create_circular_buffer(program, sender_cores, kNormExternalCbIndex, data_fmt_in, single_tile_size_in, num_cores);
    create_circular_buffer(program, all_cores, kTempCbIndex, data_fmt_in, single_tile_size_in, kNumTempTiles);
}

inline AllReduceConfig prepare_all_reduce_config(
    const tt::tt_metal::CoreRangeSet& all_cores, tt::tt_metal::IDevice* device) {
    using namespace tt::tt_metal;

    AllReduceConfig config;
    auto bbox = all_cores.bounding_box();
    config.mcast_start_x = bbox.start_coord.x;
    config.mcast_start_y = bbox.start_coord.y;
    config.mcast_end_x = bbox.end_coord.x;
    config.mcast_end_y = bbox.end_coord.y;

    auto cores_vec = corerange_to_cores(all_cores, std::nullopt);
    for (const auto& core : cores_vec) {
        if (core.x == 0 && core.y == 0) {
            continue;
        }
        auto physical_core = device->worker_core_from_logical_core(core);
        config.remote_noc_x.push_back(physical_core.x);
        config.remote_noc_y.push_back(physical_core.y);
    }

    config.sender_physical = device->worker_core_from_logical_core({0, 0});
    return config;
}

inline std::vector<uint32_t> build_reader_runtime_args(
    const tt::tt_metal::Buffer* src_buffer,
    uint32_t num_tiles_per_core,
    uint32_t start_id,
    const tt::tt_metal::CoreCoord& core,
    const AllReduceConfig& all_reduce_config) {
    std::vector<uint32_t> args = {src_buffer->address(), num_tiles_per_core, start_id};

    bool is_sender = (core.x == 0 && core.y == 0);
    args.push_back(is_sender ? 1 : 0);
    args.push_back(all_reduce_config.mcast_start_x);
    args.push_back(all_reduce_config.mcast_start_y);
    args.push_back(all_reduce_config.mcast_end_x);
    args.push_back(all_reduce_config.mcast_end_y);

    if (is_sender) {
        args.insert(args.end(), all_reduce_config.remote_noc_x.begin(), all_reduce_config.remote_noc_x.end());
        args.insert(args.end(), all_reduce_config.remote_noc_y.begin(), all_reduce_config.remote_noc_y.end());
    } else {
        args.push_back(all_reduce_config.sender_physical.x);
        args.push_back(all_reduce_config.sender_physical.y);
        args.insert(args.end(), all_reduce_config.remote_noc_x.size(), 0);
        args.insert(args.end(), all_reduce_config.remote_noc_y.size(), 0);
    }

    return args;
}

inline std::vector<uint32_t> build_compute_runtime_args(
    const tt::tt_metal::Buffer* src_buffer,
    const tt::tt_metal::Buffer* dst_buffer,
    const operation_attributes_t& args,
    uint32_t num_cores,
    uint32_t core_id) {
    uint32_t umax_norm = std::bit_cast<uint32_t>(args.max_norm);
    uint32_t ueps = std::bit_cast<uint32_t>(args.eps);

    return {
        src_buffer->address(),
        dst_buffer->address(),
        umax_norm,
        ueps,
        num_cores,
        core_id,
        0  // sync_addr (unused)
    };
}

inline ClipGradNormKernels create_kernels(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& all_cores,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2,
    const tt::tt_metal::Buffer* src_buffer,
    const tt::tt_metal::Buffer* dst_buffer,
    uint32_t num_cores,
    uint32_t num_tiles_per_core_group_1,
    uint32_t num_tiles_per_core_group_2,
    const operation_attributes_t& args,
    tt::tt_metal::IDevice* device) {
    using namespace tt::tt_metal;

    ClipGradNormKernels kernels;

    auto reduce_sender_semaphore_id = CreateSemaphore(program, all_cores, 0);
    auto reduce_receiver_semaphore_id = CreateSemaphore(program, all_cores, 0);

    uint32_t up = std::bit_cast<uint32_t>(args.p);
    uint32_t umax_norm = std::bit_cast<uint32_t>(args.max_norm);
    uint32_t ueps = std::bit_cast<uint32_t>(args.eps);

    std::vector<uint32_t> reader_compile_args = {static_cast<uint32_t>(kSrc0CbIndex)};
    TensorAccessorArgs(src_buffer).append_to(reader_compile_args);
    reader_compile_args.push_back(reduce_receiver_semaphore_id);
    reader_compile_args.push_back(reduce_sender_semaphore_id);
    reader_compile_args.push_back(num_cores);
    reader_compile_args.push_back(up);

    NOC reader_noc = detail::preferred_noc_for_dram_read(device->arch());
    NOC writer_noc = detail::preferred_noc_for_dram_write(device->arch());

    ReaderDataMovementConfig reader_config(reader_compile_args);
    reader_config.noc = reader_noc;
    kernels.reader = CreateKernel(program, kReaderKernelPath, all_cores, reader_config);

    std::vector<uint32_t> writer_compile_args = {static_cast<uint32_t>(kOutputCbIndex)};
    TensorAccessorArgs(dst_buffer).append_to(writer_compile_args);
    WriterDataMovementConfig writer_config(writer_compile_args);
    writer_config.noc = writer_noc;
    kernels.writer = CreateKernel(program, kWriterKernelPath, all_cores, writer_config);

    if (!core_group_1.ranges().empty()) {
        // Compute kernel expects: per_core_block_cnt, per_core_block_dim, umax_norm, up, ueps
        std::vector<uint32_t> compute_args = {
            1,                           // per_core_block_cnt (single block)
            num_tiles_per_core_group_1,  // per_core_block_dim (tiles per core)
            umax_norm,
            up,
            ueps};
        kernels.compute_group_1 = CreateKernel(
            program,
            kComputeKernelPath,
            core_group_1,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false,
                .compile_args = compute_args});
    }

    if (!core_group_2.ranges().empty()) {
        // Compute kernel expects: per_core_block_cnt, per_core_block_dim, umax_norm, up, ueps
        std::vector<uint32_t> compute_args = {
            1,                           // per_core_block_cnt (single block)
            num_tiles_per_core_group_2,  // per_core_block_dim (tiles per core)
            umax_norm,
            up,
            ueps};
        kernels.compute_group_2 = CreateKernel(
            program,
            kComputeKernelPath,
            core_group_2,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false,
                .compile_args = compute_args});
    }

    return kernels;
}

inline void assign_runtime_args(
    tt::tt_metal::Program& program,
    const ClipGradNormKernels& kernels,
    const tt::tt_metal::CoreRangeSet& all_cores,
    const tt::tt_metal::CoreRangeSet& core_group_1,
    const tt::tt_metal::CoreRangeSet& core_group_2,
    const tt::tt_metal::Buffer* src_buffer,
    const tt::tt_metal::Buffer* dst_buffer,
    uint32_t num_cores,
    uint32_t num_cores_y,
    uint32_t num_tiles_per_core_group_1,
    uint32_t num_tiles_per_core_group_2,
    const operation_attributes_t& args,
    tt::tt_metal::IDevice* device) {
    using namespace tt::tt_metal;

    AllReduceConfig all_reduce_config = prepare_all_reduce_config(all_cores, device);

    uint32_t num_tiles_written = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core =
            core_group_1.contains(core) ? num_tiles_per_core_group_1 : num_tiles_per_core_group_2;

        auto reader_args =
            build_reader_runtime_args(src_buffer, num_tiles_per_core, num_tiles_written, core, all_reduce_config);
        SetRuntimeArgs(program, kernels.reader, core, reader_args);

        SetRuntimeArgs(program, kernels.writer, core, {dst_buffer->address(), num_tiles_per_core, num_tiles_written});

        auto compute_args = build_compute_runtime_args(src_buffer, dst_buffer, args, num_cores, i);

        if (core_group_1.contains(core)) {
            SetRuntimeArgs(program, kernels.compute_group_1, core, compute_args);
        } else {
            SetRuntimeArgs(program, kernels.compute_group_2, core, compute_args);
        }

        num_tiles_written += num_tiles_per_core;
    }
}

}  // namespace

ClipGradNormProgramFactory::cached_program_t ClipGradNormProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    auto* device = input.device();
    Program program;

    auto data_fmt_in = datatype_to_dataformat_converter(input.dtype());
    auto data_fmt_out = datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size_in = tile_size(data_fmt_in);
    uint32_t single_tile_size_out = tile_size(data_fmt_out);
    uint32_t num_tiles = input.physical_volume() / constants::TILE_HW;

    auto grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(grid_size, num_tiles);

    CoreRangeSet sender_cores(CoreRange({0, 0}, {0, 0}));

    create_circular_buffers(
        program,
        all_cores,
        sender_cores,
        num_cores,
        data_fmt_in,
        data_fmt_out,
        single_tile_size_in,
        single_tile_size_out);

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();

    auto kernels = create_kernels(
        program,
        all_cores,
        core_group_1,
        core_group_2,
        src_buffer,
        dst_buffer,
        num_cores,
        num_tiles_per_core_group_1,
        num_tiles_per_core_group_2,
        args,
        device);

    assign_runtime_args(
        program,
        kernels,
        all_cores,
        core_group_1,
        core_group_2,
        src_buffer,
        dst_buffer,
        num_cores,
        grid_size.y,
        num_tiles_per_core_group_1,
        num_tiles_per_core_group_2,
        args,
        device);

    return cached_program_t{
        std::move(program),
        {kernels.reader,
         kernels.writer,
         kernels.compute_group_1,
         kernels.compute_group_2,
         core_group_1,
         core_group_2,
         num_cores,
         grid_size.y}};
}

void ClipGradNormProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt::tt_metal;

    auto& shared_vars = cached_program.shared_variables;
    auto& program = cached_program.program;

    auto src_buffer = tensor_args.input.buffer();
    auto dst_buffer = output.buffer();

    auto& reader_args = GetRuntimeArgs(program, shared_vars.clip_grad_norm_reader_kernel_id);
    auto& writer_args = GetRuntimeArgs(program, shared_vars.clip_grad_norm_writer_kernel_id);
    auto& compute_group_1_args = GetRuntimeArgs(program, shared_vars.clip_grad_norm_kernel_group_1_id);
    auto& compute_group_2_args = GetRuntimeArgs(program, shared_vars.clip_grad_norm_kernel_group_2_id);

    for (uint32_t i = 0; i < shared_vars.num_cores; i++) {
        CoreCoord core = {i / shared_vars.num_cores_y, i % shared_vars.num_cores_y};

        reader_args[core.x][core.y][0] = src_buffer->address();
        writer_args[core.x][core.y][0] = dst_buffer->address();

        if (shared_vars.core_group_1.contains(core)) {
            compute_group_1_args[core.x][core.y][0] = src_buffer->address();
            compute_group_1_args[core.x][core.y][1] = dst_buffer->address();
        } else {
            compute_group_2_args[core.x][core.y][0] = src_buffer->address();
            compute_group_2_args[core.x][core.y][1] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::operations::experimental::clip_grad_norm::program
