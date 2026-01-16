// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_layernorm_post_all_gather_welford_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/math.hpp"

#include <optional>
#include <string>
#include <variant>

using uint32_t = std::uint32_t;
using namespace tt::constants;

namespace ttnn::operations::experimental::transformer::dit_layernorm::program {

PostAllGatherWelfordProgramFactory::cached_program_t PostAllGatherWelfordProgramFactory::create(
    const PostAllGatherOperationAttributes& operation_attributes,
    const PostAllGatherTensorArgs& tensor_args,
    Tensor& output) {
    using tt::tt_metal::CircularBufferConfig;

    const auto& a = tensor_args.input;
    const auto& stats = tensor_args.stats;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;

    const auto& shape = a.padded_shape();
    const uint32_t W = shape[-1], H = shape[-2];
    const uint32_t HW = H * W;
    const uint32_t NC = a.physical_volume() / HW;

    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Ht = H / TILE_HEIGHT;
    const uint32_t stats_tiles_cols = stats.padded_shape()[-1] / TILE_WIDTH;
    const uint32_t tile_cols_per_device = 2;  // sum and sumsq
    const uint32_t num_devices = stats_tiles_cols / tile_cols_per_device;
    TT_FATAL(num_devices > 0, "Number of devices must be greater than 0");
    TT_FATAL(
        num_devices * tile_cols_per_device == stats_tiles_cols, "Number of devices must divide number of stats tiles");

    uint32_t num_tile_rows = NC * Ht;

    IDevice* device = a.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    const uint32_t dst_reg_count = get_dest_reg_count(operation_attributes.compute_kernel_config);
    uint32_t block_size = dst_reg_count;  // stride by dest regs, like fused rmsnorm

    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat stats_data_format = tt::tt_metal::datatype_to_dataformat_converter(stats.dtype());
    tt::DataFormat out_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat gamma_cb_data_format = gamma.has_value()
                                              ? tt::tt_metal::datatype_to_dataformat_converter(gamma.value().dtype())
                                              : tt::DataFormat::Float16_b;
    tt::DataFormat beta_cb_data_format = beta.has_value()
                                             ? tt::tt_metal::datatype_to_dataformat_converter(beta.value().dtype())
                                             : tt::DataFormat::Float16_b;
    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t stats_single_tile_size = tt::tile_size(stats_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t bfloat16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    uint32_t gamma_single_tile_size = tt::tile_size(gamma_cb_data_format);
    uint32_t beta_single_tile_size = tt::tile_size(beta_cb_data_format);

    auto a_addr = a.buffer()->address();
    auto stats_addr = stats.buffer()->address();
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;
    auto dst_addr = output.buffer()->address();

    // Size circular buffers in terms of dst_reg_count (like fused rmsnorm)
    const uint32_t double_buffer = 2;
    const uint32_t in0_tiles = dst_reg_count * double_buffer;
    const uint32_t in1_tiles = stats_tiles_cols * double_buffer;
    const uint32_t in2_tiles = gamma.has_value() ? tt::round_up(Wt, dst_reg_count) : 0;
    const uint32_t in3_tiles = beta.has_value() ? tt::round_up(Wt, dst_reg_count) : 0;
    const uint32_t in4_tiles = 1;  // epsilon

    const uint32_t intermed0_tiles = tile_cols_per_device;
    const uint32_t intermed1_tiles = 1;
    const uint32_t intermed2_tiles = dst_reg_count;
    const uint32_t out0_tiles = dst_reg_count * double_buffer;

    auto grid_size = device->compute_with_storage_grid_size();
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid_size, num_tile_rows, true);

    auto cores = corerange_to_cores(all_cores, std::nullopt);

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)block_size,
        (std::uint32_t)stats_tiles_cols,
    };

    uint32_t gamma_page_size = 0;
    uint32_t gamma_is_row_major = 0;
    uint32_t beta_is_row_major = 0;
    if (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) {
        gamma_page_size = gamma.value().padded_shape()[-1] * gamma.value().element_size();
        bool gamma_page_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(gamma_page_size);
        TT_FATAL(gamma_page_size_is_power_of_two, "Only power of 2 gammas are supported");
        gamma_is_row_major = 1;
    } else if (gamma.has_value() and gamma.value().layout() == Layout::TILE) {
        gamma_page_size = gamma_single_tile_size;
    }
    uint32_t beta_page_size = 0;
    if (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR) {
        beta_page_size = beta.value().padded_shape()[-1] * beta.value().element_size();
        bool beta_page_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(beta_page_size);
        TT_FATAL(beta_page_size_is_power_of_two, "Only power of 2 betas are supported");
        beta_is_row_major = 1;
    } else if (beta.has_value() and beta.value().layout() == Layout::TILE) {
        beta_page_size = beta_single_tile_size;
    }
    uint32_t gamma_element_size = gamma.has_value() ? gamma.value().element_size() : 0;
    uint32_t beta_element_size = beta.has_value() ? beta.value().element_size() : 0;

    reader_compile_time_args.push_back((std::uint32_t)gamma_page_size);
    reader_compile_time_args.push_back((std::uint32_t)beta_page_size);
    reader_compile_time_args.push_back((std::uint32_t)gamma_is_row_major);
    reader_compile_time_args.push_back((std::uint32_t)beta_is_row_major);
    reader_compile_time_args.push_back((std::uint32_t)Wt);
    reader_compile_time_args.push_back((std::uint32_t)gamma_element_size);
    reader_compile_time_args.push_back((std::uint32_t)beta_element_size);

    tt::tt_metal::TensorAccessorArgs(a.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(stats.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(gamma.has_value() ? gamma.value().buffer() : nullptr)
        .append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(beta.has_value() ? beta.value().buffer() : nullptr)
        .append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)Wt, (std::uint32_t)block_size};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> compute_defines;
    if (gamma.has_value()) {
        reader_defines["FUSE_GAMMA"] = "1";
    }
    if (beta.has_value()) {
        reader_defines["FUSE_BETA"] = "1";
    }

    auto reader_kernels_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_post_all_gather/device/kernels/dataflow/"
        "reader_layernorm_postallgather_dit.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    auto writer_kernels_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_post_all_gather/device/kernels/dataflow/"
        "writer_layernorm_postallgather_dit.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_args = {Wt, W, block_size, num_devices, gamma.has_value(), beta.has_value()};

    const auto* compute_kernel_file =
        "ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_post_all_gather/device/kernels/compute/"
        "layernorm_post_allgather_welford.cpp";
    auto compute_config = tt::tt_metal::ComputeConfig{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
        .compile_args = compute_args,
        .defines = compute_defines};
    auto compute_kernels_id = tt::tt_metal::CreateKernel(program, compute_kernel_file, all_cores, compute_config);

    /**
     * c_0 -> a
     * c_1 -> stats
     * c_2 -> gamma
     * c_3 -> beta
     * c_4 -> epsilon
     * c_5 -> [mean(x**2), mean(x)] stats reduced
     * c_6 -> 1/sqrt(var + epsilon)
     * c_7 -> a intermediate
     * c_8 -> output
     */

    // A input
    CircularBufferConfig cb_inp_config =
        CircularBufferConfig(in0_tiles * in_single_tile_size, {{tt::CBIndex::c_0, in_data_format}})
            .set_page_size(tt::CBIndex::c_0, in_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_inp_config);
    // Stats input
    CircularBufferConfig cb_stats_config =
        CircularBufferConfig(in1_tiles * stats_single_tile_size, {{tt::CBIndex::c_1, stats_data_format}})
            .set_page_size(tt::CBIndex::c_1, stats_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_stats_config);
    if (gamma.has_value()) {
        // Gamma input
        CircularBufferConfig cb_gamma_config =
            CircularBufferConfig(in2_tiles * gamma_single_tile_size, {{tt::CBIndex::c_2, gamma_cb_data_format}})
                .set_page_size(tt::CBIndex::c_2, gamma_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_gamma_config);
    }
    if (beta.has_value()) {
        // Beta input
        CircularBufferConfig cb_beta_config =
            CircularBufferConfig(in3_tiles * beta_single_tile_size, {{tt::CBIndex::c_3, beta_cb_data_format}})
                .set_page_size(tt::CBIndex::c_3, beta_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_beta_config);
    }
    // Epsilon input
    CircularBufferConfig cb_eps_config =
        CircularBufferConfig(in4_tiles * bfloat16_tile_size, {{tt::CBIndex::c_4, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_4, bfloat16_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_eps_config);
    // stats reduced
    CircularBufferConfig cb_intermed0_config =
        CircularBufferConfig(intermed0_tiles * single_tile_size, {{tt::CBIndex::c_5, cb_data_format}})
            .set_page_size(tt::CBIndex::c_5, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed0_config);
    // recip_sqrt_var
    CircularBufferConfig cb_intermed4_config =
        CircularBufferConfig(intermed1_tiles * single_tile_size, {{tt::CBIndex::c_6, cb_data_format}})
            .set_page_size(tt::CBIndex::c_6, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed4_config);
    // a intermediate
    CircularBufferConfig cb_intermed5_config =
        CircularBufferConfig(intermed2_tiles * single_tile_size, {{tt::CBIndex::c_7, cb_data_format}})
            .set_page_size(tt::CBIndex::c_7, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_intermed5_config);
    // output
    CircularBufferConfig cb_out0_config =
        CircularBufferConfig(out0_tiles * out_single_tile_size, {{tt::CBIndex::c_8, out_data_format}})
            .set_page_size(tt::CBIndex::c_8, out_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out0_config);

    union {
        float f;
        uint32_t u;
    } e{};
    e.f = operation_attributes.eps;  // epsilon

    uint32_t curr_row = 0;
    for (const auto& core : cores) {
        uint32_t num_tile_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        uint32_t tile_row_start = curr_row;
        uint32_t tile_row_end = tile_row_start + num_tile_rows_per_core;

        TT_FATAL(tile_row_start < tile_row_end, "Tile row start must be less than tile row end");
        TT_FATAL(tile_row_end <= num_tile_rows, "Tile row end must be less than or equal to number of tile rows");

        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_kernels_id,
            core,
            {
                a_addr,
                stats_addr,
                gamma_dram_addr,
                beta_dram_addr,
                tile_row_start,
                tile_row_end,
                e.u,
            });
        tt::tt_metal::SetRuntimeArgs(program, compute_kernels_id, core, {num_tile_rows_per_core});
        tt::tt_metal::SetRuntimeArgs(program, writer_kernels_id, core, {dst_addr, tile_row_start, tile_row_end});
        curr_row += num_tile_rows_per_core;
    }

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernels_id, .writer_kernel_id = writer_kernels_id, .cores = cores}};
}

void PostAllGatherWelfordProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const PostAllGatherOperationAttributes&,
    const PostAllGatherTensorArgs& tensor_args,
    Tensor& output) {
    auto& shared_vars = cached_program.shared_variables;
    auto& program = cached_program.program;

    const auto input_addr = tensor_args.input.buffer()->address();
    const auto stats_addr = tensor_args.stats.buffer()->address();
    const bool has_gamma = tensor_args.gamma.has_value();
    const bool has_beta = tensor_args.beta.has_value();
    const auto gamma_addr = has_gamma ? tensor_args.gamma.value().buffer()->address() : 0;
    const auto beta_addr = has_beta ? tensor_args.beta.value().buffer()->address() : 0;
    const auto output_addr = output.buffer()->address();

    auto& reader_runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(program, shared_vars.reader_kernel_id);
    auto& writer_runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(program, shared_vars.writer_kernel_id);

    for (const auto& core : shared_vars.cores) {
        {
            auto& reader_args = reader_runtime_args_by_core.at(core.x).at(core.y);

            reader_args[0] = input_addr;
            reader_args[1] = stats_addr;
            if (has_gamma) {
                reader_args[2] = gamma_addr;
            }
            if (has_beta) {
                reader_args[3] = beta_addr;
            }
        }

        {
            auto& writer_args = writer_runtime_args_by_core.at(core.x).at(core.y);
            writer_args[0] = output_addr;
        }
    }
}

}  // namespace ttnn::operations::experimental::transformer::dit_layernorm::program
