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
    auto in0_valid_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, VALID);
    auto in1_mcast_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in1_mcast_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in1_valid_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, VALID);

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

    tt::tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt::tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    // tt::tt_metal::NOC in0_split_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    // tt::tt_metal::NOC in1_split_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());

    auto cores = corerange_to_cores(core_grid, num_cores, true);

    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        CoreCoord core = cores.at(core_id);
        uint32_t in0_idx = core.y;
        uint32_t in1_idx = core.x;

        CoreCoord left_core = {(std::size_t)0, (std::size_t)core.y};
        CoreCoord top_core = {(std::size_t)core.x, (std::size_t)0};

        std::vector<CoreCoord> in0_core_order;
        in0_core_order.push_back(left_core);
        uint32_t in0_core_order_index = 0;
        for (uint32_t in0_worker_idx = 1; in0_worker_idx < grid_size.x; in0_worker_idx++) {
            CoreCoord in0_worker_core = {(std::size_t)0, (std::size_t)core.y};
            if (in0_noc == tt::tt_metal::NOC::NOC_1) {
                in0_worker_core.x = grid_size.x - in0_worker_idx;
            } else {
                in0_worker_core.x = in0_worker_idx;
            }
            if (in0_worker_core.x == core.x) {
                in0_core_order_index = in0_worker_idx;
            }
            in0_core_order.push_back(in0_worker_core);
        }
        std::vector<CoreCoord> in1_core_order;
        in1_core_order.push_back(top_core);
        uint32_t in1_core_order_index = 0;
        for (uint32_t in1_worker_idx = 1; in1_worker_idx < grid_size.y; in1_worker_idx++) {
            CoreCoord in1_worker_core = {(std::size_t)core.x, (std::size_t)0};
            if (in1_noc == tt::tt_metal::NOC::NOC_0) {
                in1_worker_core.y = grid_size.y - in1_worker_idx;
            } else {
                in1_worker_core.y = in1_worker_idx;
            }
            if (in1_worker_core.y == core.y) {
                in1_core_order_index = in1_worker_idx;
            }
            in1_core_order.push_back(in1_worker_core);
        }
        auto in0_prev_core = in0_core_order.at((std::size_t)std::max((int32_t)in0_core_order_index - 1, 0));
        auto in0_next_core =
            in0_core_order.at((std::size_t)std::min((size_t)in0_core_order_index + 1, grid_size.x - 1));
        auto in1_prev_core = in1_core_order.at((std::size_t)std::max((int32_t)in1_core_order_index - 1, 0));
        auto in1_next_core =
            in1_core_order.at((std::size_t)std::min((size_t)in1_core_order_index + 1, grid_size.y - 1));

        auto in0_prev_core_physical = device->worker_core_from_logical_core(in0_prev_core);
        ;
        auto in0_next_core_physical = device->worker_core_from_logical_core(in0_next_core);
        ;
        auto in1_prev_core_physical = device->worker_core_from_logical_core(in1_prev_core);
        ;
        auto in1_next_core_physical = device->worker_core_from_logical_core(in1_next_core);
        ;

        // in0 sender
        std::vector<uint32_t> in0_sender_compile_time_args = {
            (in1_idx == 0) ? true : false,
            (in1_idx == in0_core_order.at(grid_size.x - 1).x) ? true : false,
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
            in0_valid_semaphore_id,
            1};
        tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(in0_sender_compile_time_args);
        auto in0_sender_kernels_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp",
            core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = in0_noc,
                .compile_args = in0_sender_compile_time_args,
                .defines = defines});
        reader_kernel_ids.push_back(in0_sender_kernels_id);

        std::vector<uint32_t> in0_sender_args = {
            input_addr,
            (std::uint32_t)in0_next_core_physical.x,  // in0_dest_noc_x
            (std::uint32_t)in0_next_core_physical.y,  // in0_dest_noc_y
            (std::uint32_t)in0_prev_core_physical.x,  // in0_sender_noc_x
            (std::uint32_t)in0_prev_core_physical.y,  // in0_sender_noc_y
        };
        SetRuntimeArgs(program, in0_sender_kernels_id, core, in0_sender_args);

        // in1 sender
        std::vector<uint32_t> in1_sender_compile_time_args = {
            (in0_idx == 0) ? true : false,
            (in0_idx == in1_core_order.at(grid_size.y - 1).y) ? true : false,
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
            in1_valid_semaphore_id,
            1};
        tt::tt_metal::TensorAccessorArgs(*weight_tensor.buffer()).append_to(in1_sender_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(in1_sender_compile_time_args);
        auto in1_sender_kernels_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_sender_out.cpp",
            core,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = in1_noc,
                .compile_args = in1_sender_compile_time_args,
                .defines = defines});
        writer_kernel_ids.push_back(in1_sender_kernels_id);

        std::vector<uint32_t> in1_sender_args = {
            weight_addr,
            out_addr,
            (std::uint32_t)in1_next_core_physical.x,  // in1_dest_noc_x
            (std::uint32_t)in1_next_core_physical.y,  // in1_dest_noc_y
            (std::uint32_t)in1_prev_core_physical.x,  // in1_sender_noc_x
            (std::uint32_t)in1_prev_core_physical.y,  // in1_sender_noc_y
        };
        SetRuntimeArgs(program, in1_sender_kernels_id, core, in1_sender_args);

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
