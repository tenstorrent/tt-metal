// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_v3_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::normalization::group_norm_v3::program {

using namespace tt::constants;
using namespace tt::tt_metal;

namespace {
// Define buffer depth for each circular buffer
constexpr uint32_t CB_DEPTH = 2;

constexpr auto kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/reader_groupnorm_v3.cpp";
constexpr auto kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/writer_groupnorm_v3.cpp";
constexpr auto kComputeKernelPath =
    "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm_v3.cpp";

}  // namespace

GroupNormV3ProgramFactory::cached_program_t GroupNormV3ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    Program program = Program();

    const auto& input = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;

    const auto& in_data_format = datatype_to_dataformat_converter(input.dtype());
    const auto& out_data_format = datatype_to_dataformat_converter(output.dtype());

    if (gamma.has_value() && beta.has_value()) {
        TT_FATAL(gamma.value().dtype() == beta.value().dtype(), "Gamma and beta must have the same dtype");
    }

    const auto& core_grid = operation_attributes.core_grid;
    const auto num_groups = operation_attributes.num_groups;
    const auto chunk_size = operation_attributes.chunk_size;

    const auto total_available_cores = core_grid.x * core_grid.y;
    TT_FATAL(
        num_groups <= total_available_cores,
        "Number of groups ({}) must be <= the number of cores ({})",
        num_groups,
        total_available_cores);

    // Get a subset of cores to use
    std::vector<CoreCoord> cores_used_coords = grid_to_cores(num_groups, core_grid.x, core_grid.y);
    CoreRangeSet cores_used = CoreRangeSet(cores_used_coords);

    // Create circular buffers
    // Input CB
    const uint32_t src_cb_index = tt::CBIndex::c_0;
    const uint32_t src_page_size = chunk_size * tt::datum_size(in_data_format);
    const uint32_t src_tile_size = tt::tile_size(in_data_format);
    const uint32_t src_tiles_per_page = src_page_size / src_tile_size;
    const uint32_t src_cb_size = src_page_size * CB_DEPTH;
    const auto src_cb_config =
        CircularBufferConfig(src_cb_size, {{src_cb_index, in_data_format}}).set_page_size(src_cb_index, src_tile_size);
    CreateCircularBuffer(program, cores_used, src_cb_config);

    // Output CB
    const uint32_t dst_cb_index = tt::CBIndex::c_1;
    const uint32_t dst_page_size = chunk_size * tt::datum_size(out_data_format);
    const uint32_t dst_tile_size = tt::tile_size(out_data_format);
    const uint32_t dst_tiles_per_page = dst_page_size / dst_tile_size;
    const uint32_t dst_cb_size = dst_page_size * CB_DEPTH;
    const auto dst_cb_config =
        CircularBufferConfig(dst_cb_size, {{dst_cb_index, out_data_format}}).set_page_size(dst_cb_index, dst_tile_size);
    CreateCircularBuffer(program, cores_used, dst_cb_config);

    // Sum CB
    const uint32_t sum_cb_index = tt::CBIndex::c_2;
    const uint32_t sum_cb_size = tt::tile_size(in_data_format);
    const auto sum_cb_config =
        CircularBufferConfig(sum_cb_size, {{sum_cb_index, in_data_format}}).set_page_size(sum_cb_index, sum_cb_size);
    CreateCircularBuffer(program, cores_used, sum_cb_config);

    // Mean CB
    const uint32_t mean_cb_index = tt::CBIndex::c_3;
    const uint32_t mean_cb_size = tt::tile_size(in_data_format);
    const auto mean_cb_config = CircularBufferConfig(mean_cb_size, {{mean_cb_index, in_data_format}})
                                    .set_page_size(mean_cb_index, mean_cb_size);
    CreateCircularBuffer(program, cores_used, mean_cb_config);

    // sum of (x - E[x])^2 CB
    const uint32_t varsum_cb_index = tt::CBIndex::c_4;
    const uint32_t varsum_cb_size = tt::tile_size(in_data_format);
    const auto varsum_cb_config = CircularBufferConfig(varsum_cb_size, {{varsum_cb_index, in_data_format}})
                                      .set_page_size(varsum_cb_index, varsum_cb_size);
    CreateCircularBuffer(program, cores_used, varsum_cb_config);

    // Variance CB
    const uint32_t variance_cb_index = tt::CBIndex::c_5;
    const uint32_t variance_cb_size = tt::tile_size(in_data_format);
    const auto variance_cb_config = CircularBufferConfig(variance_cb_size, {{variance_cb_index, in_data_format}})
                                        .set_page_size(variance_cb_index, variance_cb_size);
    CreateCircularBuffer(program, cores_used, variance_cb_config);

    // Sum scaler CB
    const uint32_t sum_scaler_cb_index = tt::CBIndex::c_6;
    const uint32_t sum_scaler_cb_size = tt::tile_size(in_data_format);
    const auto sum_scaler_cb_config = CircularBufferConfig(sum_scaler_cb_size, {{sum_scaler_cb_index, in_data_format}})
                                          .set_page_size(sum_scaler_cb_index, sum_scaler_cb_size);
    CreateCircularBuffer(program, cores_used, sum_scaler_cb_config);

    // Mean scaler CB
    const uint32_t mean_scaler_cb_index = tt::CBIndex::c_7;
    const uint32_t mean_scaler_cb_size = tt::tile_size(in_data_format);
    const auto mean_scaler_cb_config =
        CircularBufferConfig(mean_scaler_cb_size, {{mean_scaler_cb_index, in_data_format}})
            .set_page_size(mean_scaler_cb_index, mean_scaler_cb_size);
    CreateCircularBuffer(program, cores_used, mean_scaler_cb_config);

    const auto N = input.logical_shape()[0];
    const auto C = input.logical_shape()[1];
    const auto H = input.logical_shape()[2];
    const auto W = input.logical_shape()[3];

    const auto pages_per_batch = C * H * W / chunk_size;
    const auto pages_per_group = pages_per_batch / num_groups;

    const auto& device = input.device();

    bool src_is_dram = input.memory_config().buffer_type() == BufferType::DRAM;
    bool dst_is_dram = output.memory_config().buffer_type() == BufferType::DRAM;

    auto reader_kernel = CreateKernel(
        program,
        kReaderKernelPath,
        cores_used,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = detail::preferred_noc_for_dram_read(device->arch()),
            .compile_args =
                {
                    src_is_dram,
                    src_tiles_per_page,
                    src_page_size,
                    pages_per_group,
                    pages_per_batch,
                    N  // num_batches
                },
            .defines = {},
        });

    for (uint32_t i = 0; i < cores_used_coords.size(); ++i) {
        CoreCoord core = cores_used_coords.at(i);
        SetRuntimeArgs(program, reader_kernel, core, {input.buffer()->address(), i * pages_per_group});
    }

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);
    CreateKernel(
        program,
        kComputeKernelPath,
        cores_used,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args =
                {src_tiles_per_page,
                 pages_per_group,
                 N,  // num_batches
                 dst_tiles_per_page},
            .defines = {}});

    auto writer_kernel = CreateKernel(
        program,
        kWriterKernelPath,
        cores_used,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = detail::preferred_noc_for_dram_write(device->arch()),
            .compile_args =
                {
                    dst_is_dram,
                    dst_tiles_per_page,
                    dst_page_size,
                    pages_per_group,
                    pages_per_batch,
                    N,  // num_batches
                },
            .defines = {}});

    for (uint32_t i = 0; i < cores_used_coords.size(); ++i) {
        CoreCoord core = cores_used_coords.at(i);
        SetRuntimeArgs(program, writer_kernel, core, {output.buffer()->address(), i * pages_per_group});
    }

    return cached_program_t{
        std::move(program),
        {.cores_used_coords = std::move(cores_used_coords),
         .reader_kernel = reader_kernel,
         .writer_kernel = writer_kernel,
         .pages_per_group = pages_per_group}};
}

void GroupNormV3ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    const auto& shared_vars = cached_program.shared_variables;

    const auto& cores_used_coords = shared_vars.cores_used_coords;
    const auto& reader_kernel = shared_vars.reader_kernel;
    const auto& writer_kernel = shared_vars.writer_kernel;
    const auto pages_per_group = shared_vars.pages_per_group;

    auto src_address = tensor_args.input.buffer()->address();
    auto dst_address = output.buffer()->address();

    for (uint32_t i = 0; i < cores_used_coords.size(); ++i) {
        CoreCoord core = cores_used_coords.at(i);

        auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel, core);
        reader_runtime_args[0] = src_address;
        reader_runtime_args[1] = i * pages_per_group;

        auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel, core);
        writer_runtime_args[0] = dst_address;
        writer_runtime_args[1] = i * pages_per_group;
    }
}

}  // namespace ttnn::operations::normalization::group_norm_v3::program
