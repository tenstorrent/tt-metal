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

    uint32_t M_block_tiles = config.M_block_size;
    uint32_t K_block_tiles = config.K_block_size;
    uint32_t N_block_tiles = config.N_block_size;

    uint32_t M_blocks = M_tiles / M_block_tiles;
    uint32_t N_blocks = N_tiles / N_block_tiles;

    uint32_t M_blocks_per_core = M_blocks / grid_size.y;
    uint32_t N_blocks_per_core = N_blocks / grid_size.x;

    uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    const uint32_t double_buffer_factor = 2;
    uint32_t in0_cb_num_tiles = in0_block_num_tiles * double_buffer_factor;
    uint32_t in1_cb_num_tiles = in1_block_num_tiles * double_buffer_factor;
    uint32_t out_cb_num_tiles = out_block_num_tiles * double_buffer_factor;
    uint32_t interm_cb_num_tiles = out_block_num_tiles;  // not double buffered

    auto in0_sender_cores = CoreRange({0, 0}, {0, grid_size.y - 1});
    auto in0_receiver_cores = CoreRange({1, 0}, {grid_size.x - 1, grid_size.y - 1});
    auto in1_sender_cores = CoreRange({0, 0}, {grid_size.x - 1, 0});
    auto in1_receiver_cores = CoreRange({0, 1}, {grid_size.x - 1, grid_size.y - 1});

    auto in0_mcast_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in0_mcast_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in1_mcast_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in1_mcast_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);

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
    log_info(tt::LogOp, "M_block_tiles: {}", M_block_tiles);
    log_info(tt::LogOp, "K_block_tiles: {}", K_block_tiles);
    log_info(tt::LogOp, "N_block_tiles: {}", N_block_tiles);
    log_info(tt::LogOp, "M_blocks: {}", M_blocks);
    log_info(tt::LogOp, "N_blocks: {}", N_blocks);
    log_info(tt::LogOp, "M_blocks_per_core: {}", M_blocks_per_core);
    log_info(tt::LogOp, "N_blocks_per_core: {}", N_blocks_per_core);
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

    uint32_t input_addr = input_tensor.buffer()->address();
    uint32_t weight_addr = weight_tensor.buffer()->address();
    uint32_t out_addr = output_tensor.buffer()->address();

    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;

    auto cores = corerange_to_cores(core_grid, num_cores, true);

    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        CoreCoord core = cores.at(core_id);

        CoreCoord left_core = {(std::size_t)0, (std::size_t)core.y};
        CoreCoord left_core_plus_one = {(std::size_t)1, (std::size_t)core.y};
        CoreCoord right_core = {(std::size_t)grid_size.x - 1, (std::size_t)core.y};
        CoreCoord top_core = {(std::size_t)core.x, (std::size_t)0};
        CoreCoord top_core_plus_one = {(std::size_t)core.x, (std::size_t)1};
        CoreCoord bottom_core = {(std::size_t)core.x, (std::size_t)grid_size.y - 1};

        auto left_core_physical = device->worker_core_from_logical_core(left_core);
        auto left_core_plus_one_physical = device->worker_core_from_logical_core(left_core_plus_one);
        auto right_core_physical = device->worker_core_from_logical_core(right_core);
        auto top_core_physical = device->worker_core_from_logical_core(top_core);
        auto top_core_plus_one_physical = device->worker_core_from_logical_core(top_core_plus_one);
        auto bottom_core_physical = device->worker_core_from_logical_core(bottom_core);
        uint32_t in0_idx = core.y;
        uint32_t in1_idx = core.x;

        auto in0_mcast_sender = left_core_physical;
        auto in1_mcast_sender = top_core_physical;

        auto in0_mcast_start = left_core_plus_one_physical;
        auto in0_mcast_end = right_core_physical;

        auto in1_mcast_start = bottom_core_physical;
        auto in1_mcast_end = top_core_plus_one_physical;

        if (in1_idx == 0) {
            // in0 sender
            std::vector<uint32_t> in0_sender_compile_time_args = {
                M_blocks_per_core * core.y,
                M_blocks_per_core * (core.y + 1) - 1,
                K_tiles,
                N_blocks_per_core * core.x,
                N_blocks_per_core * (core.x + 1) - 1,
                M_block_tiles,
                K_block_tiles,
                N_block_tiles,
                input_tile_size,
                in0_mcast_sender_semaphore_id,
                in0_mcast_receiver_semaphore_id,
                grid_size.y - 1};
            tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(in0_sender_compile_time_args);
            auto in0_sender_kernels_id = CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp",
                core,
                tt::tt_metal::ReaderDataMovementConfig(in0_sender_compile_time_args, defines));
            reader_kernel_ids.push_back(in0_sender_kernels_id);

            std::vector<uint32_t> in0_sender_args = {
                input_addr,
                (std::uint32_t)in0_mcast_start.x,  // in0_mcast_dest_noc_start_x
                (std::uint32_t)in0_mcast_start.y,  // in0_mcast_dest_noc_start_y
                (std::uint32_t)in0_mcast_end.x,    // in0_mcast_dest_noc_end_x
                (std::uint32_t)in0_mcast_end.y,    // in0_mcast_dest_noc_end_y
            };
            SetRuntimeArgs(program, in0_sender_kernels_id, core, in0_sender_args);
        } else {
            // in0 receiver
            std::vector<uint32_t> in0_receiver_compile_time_args = {
                M_blocks_per_core * core.y,
                M_blocks_per_core * (core.y + 1) - 1,
                K_tiles,
                N_blocks_per_core * core.x,
                N_blocks_per_core * (core.x + 1) - 1,
                M_block_tiles,
                K_block_tiles,
                N_block_tiles,
                input_tile_size,
                in0_mcast_sender_semaphore_id,
                in0_mcast_receiver_semaphore_id};
            auto in0_receiver_kernels_id = CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_receiver.cpp",
                core,
                tt::tt_metal::ReaderDataMovementConfig(in0_receiver_compile_time_args, defines));
            reader_kernel_ids.push_back(in0_receiver_kernels_id);

            std::vector<uint32_t> in0_receiver_args = {
                (std::uint32_t)in0_mcast_sender.x,  // in0_mcast_sender_noc_x
                (std::uint32_t)in0_mcast_sender.y   // in0_mcast_sender_noc_y
            };
            SetRuntimeArgs(program, in0_receiver_kernels_id, core, in0_receiver_args);
        }

        if (in0_idx == 0) {
            // in1 sender
            std::vector<uint32_t> in1_sender_compile_time_args = {
                M_blocks_per_core * core.y,
                M_blocks_per_core * (core.y + 1) - 1,
                K_tiles,
                N_tiles,
                N_blocks_per_core * core.x,
                N_blocks_per_core * (core.x + 1) - 1,
                M_block_tiles,
                K_block_tiles,
                N_block_tiles,
                input_tile_size,
                in1_mcast_sender_semaphore_id,
                in1_mcast_receiver_semaphore_id,
                grid_size.x - 1};
            tt::tt_metal::TensorAccessorArgs(*weight_tensor.buffer()).append_to(in1_sender_compile_time_args);
            tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(in1_sender_compile_time_args);
            auto in1_sender_kernels_id = CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_sender_out.cpp",
                core,
                tt::tt_metal::WriterDataMovementConfig(in1_sender_compile_time_args, defines));
            writer_kernel_ids.push_back(in1_sender_kernels_id);

            std::vector<uint32_t> in1_sender_args = {
                weight_addr,
                out_addr,
                (std::uint32_t)in1_mcast_start.x,  // in1_mcast_dest_noc_start_x
                (std::uint32_t)in1_mcast_start.y,  // in1_mcast_dest_noc_start_y
                (std::uint32_t)in1_mcast_end.x,    // in1_mcast_dest_noc_end_x
                (std::uint32_t)in1_mcast_end.y,    // in1_mcast_dest_noc_end_y
            };
            SetRuntimeArgs(program, in1_sender_kernels_id, core, in1_sender_args);
        } else {
            // in1 receiver
            std::vector<uint32_t> in1_receiver_compile_time_args = {
                M_blocks_per_core * core.y,
                M_blocks_per_core * (core.y + 1) - 1,
                K_tiles,
                N_tiles,
                N_blocks_per_core * core.x,
                N_blocks_per_core * (core.x + 1) - 1,
                M_block_tiles,
                K_block_tiles,
                N_block_tiles,
                input_tile_size,
                in1_mcast_sender_semaphore_id,
                in1_mcast_receiver_semaphore_id};
            tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(in1_receiver_compile_time_args);
            auto in1_receiver_kernels_id = CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_receiver_out.cpp",
                core,
                tt::tt_metal::WriterDataMovementConfig(in1_receiver_compile_time_args, defines));
            writer_kernel_ids.push_back(in1_receiver_kernels_id);

            std::vector<uint32_t> in1_receiver_args = {
                out_addr,
                (std::uint32_t)in1_mcast_sender.x,  // in1_mcast_sender_noc_x
                (std::uint32_t)in1_mcast_sender.y,  // in1_mcast_sender_noc_y
            };
            SetRuntimeArgs(program, in1_receiver_kernels_id, core, in1_receiver_args);

            std::vector<uint32_t> compute_compile_time_args = {
                M_blocks_per_core * core.y,
                M_blocks_per_core * (core.y + 1) - 1,
                K_tiles,
                N_blocks_per_core * core.x,
                N_blocks_per_core * (core.x + 1) - 1,
                M_block_tiles,
                K_block_tiles,
                N_block_tiles,
                subblock_h,
                subblock_w};

            // auto compute_kernels_id = CreateKernel(
            CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp",
                core,
                tt::tt_metal::ComputeConfig{
                    .math_fidelity = math_fidelity,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .math_approx_mode = math_approx_mode,
                    .compile_args = compute_compile_time_args});
        }
    }

    auto override_runtime_arguments_callback =
        [num_cores, cores, reader_kernel_ids, writer_kernel_ids](
            const void* operation,
            tt::tt_metal::Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto input_addr = input_tensors.at(0).buffer()->address();
            auto weight_addr = input_tensors.at(1).buffer()->address();
            auto output_addr = output_tensors.at(0).buffer()->address();

            for (uint32_t i = 0; i < num_cores; ++i) {
                CoreCoord core = cores.at(i);
                auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_ids[i]);
                auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_ids[i]);
                uint32_t in0_idx = core.y;
                uint32_t in1_idx = core.x;
                auto& reader_args = reader_runtime_args[core.x][core.y];
                if (in1_idx == 0) {
                    reader_args[0] = input_addr;
                }
                auto& writer_args = writer_runtime_args[core.x][core.y];
                if (in0_idx == 0) {
                    writer_args[0] = weight_addr;
                    writer_args[1] = output_addr;
                } else {
                    writer_args[0] = output_addr;
                }
            }
        };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::minimal_matmul::detail
