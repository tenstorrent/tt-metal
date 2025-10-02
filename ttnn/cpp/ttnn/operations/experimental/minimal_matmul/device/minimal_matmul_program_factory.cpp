// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "minimal_matmul_device_operation.hpp"
#include "minimal_matmul_program_factory.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include <algorithm>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::minimal_matmul::detail {

tt::tt_metal::operation::ProgramWithCallbacks minimal_matmul_factory(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const MinimalMatmulConfig& config,
    const Tensor& output_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    auto grid_size = config.compute_with_storage_grid_size;
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    auto num_cores = core_grid.size();

    auto input_tensor_shape = input_tensor.logical_shape();
    auto weight_tensor_shape = weight_tensor.logical_shape();
    uint32_t M = input_tensor_shape[0];
    uint32_t K = input_tensor_shape[1];
    uint32_t N = weight_tensor_shape[1];

    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto input_tile_size = tt::tile_size(data_format);

    auto device = input_tensor.device();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    auto intermediate_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    auto intermediate_tile_size = tt::tile_size(intermediate_data_format);

    // bool use_bias = bias_tensor.has_value();
    /**
     * A: M_tiles x K_tiles
     * A subdivided into M_block x K_block tiles
     */

    uint32_t M_tiles = M / tt::constants::TILE_HEIGHT;
    uint32_t K_tiles = K / tt::constants::TILE_WIDTH;
    uint32_t N_tiles = N / tt::constants::TILE_WIDTH;

    /**
     * TODO: Pick optimal subblock sizes, hardcoded to 2x4 or 1x4 or 2x2
     */
    uint32_t subblock_h = config.subblock_h;
    uint32_t subblock_w = config.subblock_w;

    uint32_t M_block = config.M_block_size;
    uint32_t K_block = config.K_block_size;
    uint32_t N_block = config.N_block_size;

    uint32_t in0_block_num_tiles = M_block * K_block;
    uint32_t in1_block_num_tiles = K_block * N_block;
    uint32_t out_block_num_tiles = M_block * N_block;

    const uint32_t double_buffer_factor = 2;
    uint32_t in0_cb_num_tiles = in0_block_num_tiles * double_buffer_factor;
    uint32_t in1_cb_num_tiles = in1_block_num_tiles * double_buffer_factor;
    uint32_t out_cb_num_tiles = out_block_num_tiles * double_buffer_factor;
    uint32_t interm_cb_num_tiles = out_block_num_tiles;  // not double buffered

    // Create circular buffers for vol2col, weights, bias and matmul intermediates
    uint32_t next_cb_index = tt::CBIndex::c_0;
    uint32_t in0_cb_id = next_cb_index++;
    tt::tt_metal::create_cb(in0_cb_id, program, core_grid, input_tile_size, in0_cb_num_tiles, data_format);

    uint32_t in1_cb_id = next_cb_index++;
    tt::tt_metal::create_cb(in1_cb_id, program, core_grid, input_tile_size, in1_cb_num_tiles, data_format);

    uint32_t out_cb_id = next_cb_index++;
    tt::tt_metal::create_cb(out_cb_id, program, core_grid, input_tile_size, out_cb_num_tiles, data_format);

    uint32_t intermediate_cb_id = next_cb_index++;
    tt::tt_metal::create_cb(
        intermediate_cb_id, program, core_grid, intermediate_tile_size, interm_cb_num_tiles, intermediate_data_format);

    log_info(tt::LogOp, "in0_cb_id: {}", in0_cb_id);
    log_info(tt::LogOp, "in1_cb_id: {}", in1_cb_id);
    log_info(tt::LogOp, "out_cb_id: {}", out_cb_id);
    log_info(tt::LogOp, "intermediate_cb_id: {}", intermediate_cb_id);
    log_info(tt::LogOp, "M_tiles: {}", M_tiles);
    log_info(tt::LogOp, "K_tiles: {}", K_tiles);
    log_info(tt::LogOp, "N_tiles: {}", N_tiles);
    log_info(tt::LogOp, "M_block: {}", M_block);
    log_info(tt::LogOp, "K_block: {}", K_block);
    log_info(tt::LogOp, "N_block: {}", N_block);
    log_info(tt::LogOp, "subblock_h: {}", subblock_h);
    log_info(tt::LogOp, "subblock_w: {}", subblock_w);
    log_info(tt::LogOp, "input_tile_size: {}", input_tile_size);
    log_info(tt::LogOp, "intermediate_tile_size: {}", intermediate_tile_size);
    log_info(tt::LogOp, "intermediate_data_format: {}", intermediate_data_format);
    log_info(tt::LogOp, "in0_cb_num_tiles: {}", in0_cb_num_tiles);
    log_info(tt::LogOp, "in1_cb_num_tiles: {}", in1_cb_num_tiles);
    log_info(tt::LogOp, "out_cb_num_tiles: {}", out_cb_num_tiles);
    log_info(tt::LogOp, "interm_cb_num_tiles: {}", interm_cb_num_tiles);

    // Get environment variables to optionally skip data movement kernels
    bool skip_in0 = false;
    bool skip_in1 = false;
    bool skip_out = false;
    if (const char* env_p = std::getenv("TT_MM_SKIP_IN0")) {
        skip_in0 = std::string(env_p) == "1";
    }
    if (const char* env_p = std::getenv("TT_MM_SKIP_IN1")) {
        skip_in1 = std::string(env_p) == "1";
    }
    if (const char* env_p = std::getenv("TT_MM_SKIP_OUT")) {
        skip_out = std::string(env_p) == "1";
    }
    std::map<std::string, std::string> defines;
    if (skip_in0) {
        defines["SKIP_IN0"] = "1";
        log_warning(tt::LogOp, "Skipping in0 data movement! PCC will be wrong!");
    }
    if (skip_in1) {
        defines["SKIP_IN1"] = "1";
        log_warning(tt::LogOp, "Skipping in1 data movement! PCC will be wrong!");
    }
    if (skip_out) {
        defines["SKIP_OUT"] = "1";
        log_warning(tt::LogOp, "Skipping out data movement! PCC will be wrong!");
    }

    std::vector<uint32_t> reader_compile_time_args = {
        M_tiles, K_tiles, N_tiles, M_block, K_block, N_block, input_tile_size};
    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_compile_time_args);

    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0.cpp",
        core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, defines));

    // const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
    // const uint32_t in0_block_w = matmul_K_t;

    // const uint32_t out_subblock_w = std::min(matmul_N_t, dst_size);
    // TT_FATAL(matmul_N_t % out_subblock_w == 0, "matmul_N_t must be divisible by out_subblock_w");
    // // If out_subblock_w is full row of output, scale subblock_h so volume = dst_size. Otherwise it's 1 to maintain
    // // row-major intermediate buffer.
    // const uint32_t out_subblock_h =
    //     (out_subblock_w == matmul_N_t) ? (std::min(matmul_M_t, dst_size / out_subblock_w)) : 1;

    // const uint32_t in0_num_subblocks = matmul_M_t / out_subblock_h;
    // const uint32_t in1_num_subblocks = matmul_N_t / out_subblock_w;

    std::vector<uint32_t> compute_compile_time_args = {
        M_tiles, K_tiles, N_tiles, M_block, K_block, N_block, subblock_h, subblock_w};

    // auto compute_kernels_id = CreateKernel(
    CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args = {
        M_tiles, K_tiles, N_tiles, M_block, K_block, N_block, input_tile_size};

    tt::tt_metal::TensorAccessorArgs(*weight_tensor.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_compile_time_args);

    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_out.cpp",
        core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, defines));

    uint32_t input_addr = input_tensor.buffer()->address();
    uint32_t weight_addr = weight_tensor.buffer()->address();
    uint32_t out_addr = output_tensor.buffer()->address();

    auto cores = corerange_to_cores(core_grid, num_cores, true);
    std::vector<std::vector<uint32_t>> reader_args_per_core(num_cores);
    std::vector<std::vector<uint32_t>> compute_args_per_core(num_cores);
    std::vector<std::vector<uint32_t>> writer_args_per_core(num_cores);

    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        // CoreCoord core = cores.at(core_id);

        // Store runtime args for later use
        reader_args_per_core[core_id] = {input_addr};

        writer_args_per_core[core_id] = {weight_addr, out_addr};
    }
    SetRuntimeArgs(program, reader_kernels_id, cores, reader_args_per_core);
    SetRuntimeArgs(program, writer_kernels_id, cores, writer_args_per_core);

    auto override_runtime_arguments_callback =
        [num_cores, cores, reader_kernels_id, writer_kernels_id](
            const void* operation,
            tt::tt_metal::Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto& reader_args_by_core = GetRuntimeArgs(program, reader_kernels_id);
            auto& writer_args_by_core = GetRuntimeArgs(program, writer_kernels_id);

            auto input_addr = input_tensors.at(0).buffer()->address();
            auto weight_addr = input_tensors.at(1).buffer()->address();
            auto output_addr = output_tensors.at(0).buffer()->address();

            for (uint32_t i = 0; i < num_cores; ++i) {
                CoreCoord core = cores.at(i);
                auto& reader_args = reader_args_by_core[core.x][core.y];
                auto& writer_args = writer_args_by_core[core.x][core.y];
                reader_args[0] = input_addr;
                writer_args[0] = weight_addr;
                writer_args[1] = output_addr;
            }
        };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::minimal_matmul::detail
