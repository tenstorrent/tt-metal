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
#include <tuple>

namespace ttnn::operations::experimental::minimal_matmul::detail {

static inline std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> determine_default_block_sizes(
    uint32_t M, uint32_t K, uint32_t N, bool fp32_dest_acc_en) {
    (void)K;  // K not used for determining defaults currently
    uint32_t M_block_tiles = 8;
    uint32_t K_block_tiles = 8;
    uint32_t N_block_tiles = 8;

    uint32_t subblock_h = 2;
    uint32_t subblock_w = 2;
    if (!fp32_dest_acc_en) {
        if (N >= M) {
            subblock_h = 2;
            subblock_w = 4;
        } else {
            subblock_h = 4;
            subblock_w = 2;
        }
    }

    return {M_block_tiles, K_block_tiles, N_block_tiles, subblock_h, subblock_w};
}

tt::tt_metal::operation::ProgramWithCallbacks minimal_matmul_factory(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<unary::UnaryWithParam>& fused_activation,
    const std::optional<const MinimalMatmulConfig>& config,
    const Tensor& output_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    auto device = input_tensor.device();

    if (!config.has_value()) {
        log_debug(tt::LogOp, "No config provided, using default block sizes and core grid");
    }

    auto grid_size =
        config.has_value() ? config.value().compute_with_storage_grid_size : device->compute_with_storage_grid_size();
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    auto num_cores = core_grid.size();

    bool use_bias = bias_tensor.has_value();

    /**
     * Determine dataformats, compute kernel config
     */
    auto in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto in0_tile_size = tt::tile_size(in0_data_format);
    auto in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(weight_tensor.dtype());
    auto in1_tile_size = tt::tile_size(in1_data_format);
    auto output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    auto out_tile_size = tt::tile_size(output_data_format);

    auto in2_data_format =
        use_bias ? tt::tt_metal::datatype_to_dataformat_converter(bias_tensor.value().dtype()) : in1_data_format;
    auto in2_tile_size = tt::tile_size(in2_data_format);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    // Intermediate CB dataformat is the same datatype as DST register.
    auto intermediate_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    auto intermediate_tile_size = tt::tile_size(intermediate_data_format);

    /**
     * in0: M_tiles x K_tiles
     * in0 is divided into blocks, which are M_block_tiles x K_block_tiles
     *
     * in1: K_tiles x N_tiles
     * in1 is divided into blocks, which are K_block_tiles x N_block_tiles
     *
     * output: M_tiles x N_tiles
     * output is divided into blocks, which are M_block_tiles x N_block_tiles
     *
     * Blocks are further subdivided into subblocks. The output block is subdivided into subblock_h x subblock_w
     * subblocks. The in0 and in1 blocks are accordingly subdivided on M and N.
     */

    auto in0_tensor_shape = input_tensor.padded_shape();
    auto in1_tensor_shape = weight_tensor.padded_shape();
    uint32_t M = in0_tensor_shape[0];
    uint32_t K = in0_tensor_shape[1];
    uint32_t N = in1_tensor_shape[1];

    uint32_t M_tiles = M / tt::constants::TILE_HEIGHT;
    uint32_t K_tiles = K / tt::constants::TILE_WIDTH;
    uint32_t N_tiles = N / tt::constants::TILE_WIDTH;

    auto [default_M_block_tiles, default_K_block_tiles, default_N_block_tiles, default_subblock_h, default_subblock_w] =
        determine_default_block_sizes(M, K, N, fp32_dest_acc_en);

    /**
     * TODO: Pick optimal subblock sizes. Currently a simple default is used.
     */
    uint32_t subblock_h = config.has_value() ? config.value().subblock_h : default_subblock_h;
    uint32_t subblock_w = config.has_value() ? config.value().subblock_w : default_subblock_w;

    uint32_t M_block_tiles = config.has_value() ? config.value().M_block_size : default_M_block_tiles;
    uint32_t K_block_tiles = config.has_value() ? config.value().K_block_size : default_K_block_tiles;
    uint32_t N_block_tiles = config.has_value() ? config.value().N_block_size : default_N_block_tiles;

    /**
     * We originally saw that for non-square outputs, N > M was significantly faster than M > N.
     * This is because originally, the in0 DM kernel was responsible for reading in0 and writing output.
     * When M > N, the in0 DM kernel has more data to read on top of its responsibility to write output.
     *
     * An optimization is to have the DM kernel with less data to read handle writes, and transpose the core_grid
     * to keep NOC usage consistent. With this optimization, N > M performance is symmetric with M > N.
     *
     * The smaller input read and mcast is always across a row of cores (x, y): (0, core_y) -> (grid_size.x-1, core_y)
     * The larger input read and mcast is always across a column of cores (x, y): (core_x, 0) -> (core_x. grid_size.y-1)
     *
     * Output is always written by DM reading the smaller input.
     *
     * Small input + output DM always runs on RISCV_1, NOC_1
     * Large input DM always runs on RISCV_0, NOC_0
     */

    auto small_input_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    auto small_input_risc = tt::tt_metal::DataMovementProcessor::RISCV_1;
    auto large_input_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    auto large_input_risc = tt::tt_metal::DataMovementProcessor::RISCV_0;

    // Transpose core grid if the output is wide (M > N)
    // If transpose core grid, we parallelize M on cores_x and N on cores_y and swap the NOCs and RISCVs
    bool transpose_core_grid = M > N;

    auto in0_noc = transpose_core_grid ? large_input_noc : small_input_noc;
    auto in0_risc = transpose_core_grid ? large_input_risc : small_input_risc;
    uint32_t in0_parallel_axis_cores = transpose_core_grid ? grid_size.x : grid_size.y;

    auto in1_noc = transpose_core_grid ? small_input_noc : large_input_noc;
    auto in1_risc = transpose_core_grid ? small_input_risc : large_input_risc;
    uint32_t in1_parallel_axis_cores = transpose_core_grid ? grid_size.y : grid_size.x;

    /**
     * In order to enable arbitrary blockings, we pad the input dims to the nearest multiple of block size.
     *
     * In addition, we "pad" up the number of blocks in M, K, N in order to divide by the number of cores.
     */
    uint32_t padded_M_tiles = tt::round_up(M_tiles, M_block_tiles);
    uint32_t padded_N_tiles = tt::round_up(N_tiles, N_block_tiles);
    uint32_t padded_K_tiles = tt::round_up(K_tiles, K_block_tiles);

    uint32_t M_blocks = padded_M_tiles / M_block_tiles;
    uint32_t N_blocks = padded_N_tiles / N_block_tiles;
    uint32_t K_blocks = padded_K_tiles / K_block_tiles;

    uint32_t M_blocks_per_core = tt::div_up(M_blocks, in0_parallel_axis_cores);
    uint32_t N_blocks_per_core = tt::div_up(N_blocks, in1_parallel_axis_cores);

    uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;
    uint32_t in2_block_num_tiles = N_block_tiles;

    const uint32_t double_buffer_factor = 2;
    uint32_t in0_cb_num_tiles = in0_block_num_tiles * double_buffer_factor;
    uint32_t in1_cb_num_tiles = in1_block_num_tiles * double_buffer_factor;
    // TODO: consider not double buffering the output
    uint32_t out_cb_num_tiles = out_block_num_tiles * double_buffer_factor;
    uint32_t interm_cb_num_tiles = out_block_num_tiles;  // not double buffered
    uint32_t in2_cb_num_tiles = in2_block_num_tiles;     // not double buffered

    auto core_0_0 = CoreCoord{0, 0};
    auto core_0_1 = CoreCoord{0, 1};
    auto core_1_0 = CoreCoord{1, 0};
    auto core_endx_0 = CoreCoord{grid_size.x - 1, 0};
    auto core_0_endy = CoreCoord{0, grid_size.y - 1};
    auto core_endx_endy = CoreCoord{grid_size.x - 1, grid_size.y - 1};

    auto in0_sender_cores = CoreRange(core_0_0, transpose_core_grid ? core_endx_0 : core_0_endy);
    auto in0_receiver_cores = CoreRange(transpose_core_grid ? core_0_1 : core_1_0, core_endx_endy);
    auto in1_sender_cores = CoreRange(core_0_0, transpose_core_grid ? core_0_endy : core_endx_0);
    auto in1_receiver_cores = CoreRange(transpose_core_grid ? core_1_0 : core_0_1, core_endx_endy);

    auto in0_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in0_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in0_valid_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, VALID);
    auto in1_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in1_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in1_valid_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, VALID);

    uint32_t in0_cb_id = tt::CBIndex::c_0;
    tt::tt_metal::create_cb(in0_cb_id, program, core_grid, in0_tile_size, in0_cb_num_tiles, in0_data_format);

    uint32_t in1_cb_id = tt::CBIndex::c_1;
    tt::tt_metal::create_cb(in1_cb_id, program, core_grid, in1_tile_size, in1_cb_num_tiles, in1_data_format);

    uint32_t out_cb_id = tt::CBIndex::c_2;
    tt::tt_metal::create_cb(out_cb_id, program, core_grid, out_tile_size, out_cb_num_tiles, output_data_format);

    uint32_t intermediate_cb_id = tt::CBIndex::c_3;
    tt::tt_metal::create_cb(
        intermediate_cb_id, program, core_grid, intermediate_tile_size, interm_cb_num_tiles, intermediate_data_format);

    if (use_bias) {
        uint32_t in2_cb_id = tt::CBIndex::c_4;
        tt::tt_metal::create_cb(in2_cb_id, program, core_grid, in2_tile_size, in2_cb_num_tiles, in2_data_format);
    }

    log_debug(tt::LogOp, "in0_cb_id: {}", in0_cb_id);
    log_debug(tt::LogOp, "in1_cb_id: {}", in1_cb_id);
    log_debug(tt::LogOp, "out_cb_id: {}", out_cb_id);
    log_debug(tt::LogOp, "intermediate_cb_id: {}", intermediate_cb_id);
    log_debug(tt::LogOp, "M_tiles: {}", M_tiles);
    log_debug(tt::LogOp, "padded_M_tiles: {}", padded_M_tiles);
    log_debug(tt::LogOp, "K_tiles: {}", K_tiles);
    log_debug(tt::LogOp, "padded_K_tiles: {}", padded_K_tiles);
    log_debug(tt::LogOp, "N_tiles: {}", N_tiles);
    log_debug(tt::LogOp, "padded_N_tiles: {}", padded_N_tiles);
    log_debug(tt::LogOp, "M_block_tiles: {}", M_block_tiles);
    log_debug(tt::LogOp, "K_block_tiles: {}", K_block_tiles);
    log_debug(tt::LogOp, "N_block_tiles: {}", N_block_tiles);
    log_debug(tt::LogOp, "M_blocks: {}", M_blocks);
    log_debug(tt::LogOp, "N_blocks: {}", N_blocks);
    log_debug(tt::LogOp, "M_blocks_per_core: {}", M_blocks_per_core);
    log_debug(tt::LogOp, "N_blocks_per_core: {}", N_blocks_per_core);
    log_debug(tt::LogOp, "subblock_h: {}", subblock_h);
    log_debug(tt::LogOp, "subblock_w: {}", subblock_w);
    log_debug(tt::LogOp, "in0_tile_size: {}", in0_tile_size);
    log_debug(tt::LogOp, "in1_tile_size: {}", in1_tile_size);
    log_debug(tt::LogOp, "out_tile_size: {}", out_tile_size);
    log_debug(tt::LogOp, "in2_tile_size: {}", in2_tile_size);
    log_debug(tt::LogOp, "intermediate_tile_size: {}", intermediate_tile_size);
    log_debug(tt::LogOp, "intermediate_data_format: {}", intermediate_data_format);
    log_debug(tt::LogOp, "in0_cb_num_tiles: {}", in0_cb_num_tiles);
    log_debug(tt::LogOp, "in1_cb_num_tiles: {}", in1_cb_num_tiles);
    log_debug(tt::LogOp, "out_cb_num_tiles: {}", out_cb_num_tiles);
    log_debug(tt::LogOp, "interm_cb_num_tiles: {}", interm_cb_num_tiles);

    std::map<std::string, std::string> defines;
    if (use_bias) {
        defines["FUSE_BIAS"] = "1";
    }

    uint32_t in0_addr = input_tensor.buffer()->address();
    uint32_t in1_addr = weight_tensor.buffer()->address();
    uint32_t in2_addr = use_bias ? bias_tensor.value().buffer()->address() : 0;
    uint32_t out_addr = output_tensor.buffer()->address();

    /**
     * Create kernels
     */

    bool in0_is_output_writer = !transpose_core_grid;
    bool in1_is_output_writer = transpose_core_grid;

    std::vector<uint32_t> in0_sender_compile_time_args = {
        M_tiles,
        padded_M_tiles,
        K_tiles,
        padded_K_tiles,
        N_tiles,
        padded_N_tiles,
        M_block_tiles,
        K_block_tiles,
        N_block_tiles,
        in0_tile_size,
        out_tile_size,
        in2_tile_size,
        in0_sender_semaphore_id,
        in0_receiver_semaphore_id,
        in0_valid_semaphore_id,
        in0_is_output_writer,
        true  // is_injector_core
    };
    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(in0_sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(in0_sender_compile_time_args);
    if (use_bias) {
        tt::tt_metal::TensorAccessorArgs(*bias_tensor.value().buffer()).append_to(in0_sender_compile_time_args);
    }
    auto in0_sender_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp",
        in0_sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = in0_risc, .noc = in0_noc, .compile_args = in0_sender_compile_time_args, .defines = defines});

    std::vector<uint32_t> in0_receiver_compile_time_args = {
        M_tiles,
        padded_M_tiles,
        K_tiles,
        padded_K_tiles,
        N_tiles,
        padded_N_tiles,
        M_block_tiles,
        K_block_tiles,
        N_block_tiles,
        in0_tile_size,
        out_tile_size,
        in2_tile_size,
        in0_sender_semaphore_id,
        in0_receiver_semaphore_id,
        in0_valid_semaphore_id,
        in0_is_output_writer,
        false  // is_injector_core
    };
    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(in0_receiver_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(in0_receiver_compile_time_args);
    if (use_bias) {
        tt::tt_metal::TensorAccessorArgs(*bias_tensor.value().buffer()).append_to(in0_receiver_compile_time_args);
    }
    auto in0_receiver_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp",
        in0_receiver_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = in0_risc, .noc = in0_noc, .compile_args = in0_receiver_compile_time_args, .defines = defines});

    std::vector<uint32_t> in1_sender_compile_time_args = {
        M_tiles,
        padded_M_tiles,
        K_tiles,
        padded_K_tiles,
        N_tiles,
        padded_N_tiles,
        M_block_tiles,
        K_block_tiles,
        N_block_tiles,
        in1_tile_size,
        out_tile_size,
        in2_tile_size,
        in1_sender_semaphore_id,
        in1_receiver_semaphore_id,
        in1_valid_semaphore_id,
        in1_is_output_writer,
        true  // is_injector_core
    };
    tt::tt_metal::TensorAccessorArgs(*weight_tensor.buffer()).append_to(in1_sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(in1_sender_compile_time_args);
    if (use_bias) {
        tt::tt_metal::TensorAccessorArgs(*bias_tensor.value().buffer()).append_to(in1_sender_compile_time_args);
    }
    auto in1_sender_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_sender_out.cpp",
        in1_sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = in1_risc, .noc = in1_noc, .compile_args = in1_sender_compile_time_args, .defines = defines});

    std::vector<uint32_t> in1_receiver_compile_time_args = {
        M_tiles,
        padded_M_tiles,
        K_tiles,
        padded_K_tiles,
        N_tiles,
        padded_N_tiles,
        M_block_tiles,
        K_block_tiles,
        N_block_tiles,
        in1_tile_size,
        out_tile_size,
        in2_tile_size,
        in1_sender_semaphore_id,
        in1_receiver_semaphore_id,
        in1_valid_semaphore_id,
        in1_is_output_writer,
        false  // is_injector_core
    };
    tt::tt_metal::TensorAccessorArgs(*weight_tensor.buffer()).append_to(in1_receiver_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(in1_receiver_compile_time_args);
    if (use_bias) {
        tt::tt_metal::TensorAccessorArgs(*bias_tensor.value().buffer()).append_to(in1_receiver_compile_time_args);
    }
    auto in1_receiver_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_sender_out.cpp",
        in1_receiver_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = in1_risc, .noc = in1_noc, .compile_args = in1_receiver_compile_time_args, .defines = defines});

    std::vector<uint32_t> compute_compile_time_args = {
        K_blocks, M_block_tiles, K_block_tiles, N_block_tiles, subblock_h, subblock_w};

    auto compute_defines = defines;
    std::map<std::string, std::string> compute_activation_defines;
    if (fused_activation.has_value()) {
        compute_activation_defines = ttnn::operations::unary::utils::get_defines(
            fused_activation.value().op_type,
            fused_activation.value().params,
            "ACTIVATION",
            "fused_act_dst_id",
            output_tensor.dtype());
    }
    compute_defines.merge(compute_activation_defines);
    auto compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
            .defines = compute_defines});

    /**
     * The receiver writer cores defer their writes in order to reduce NOC congestion.
     * Further, the amount of K_blocks they defer by depends on their core coordinate.
     * If we have core_grid.x cores, we'd want to evenly stride the K_blocks they defer by.
     * For first pass, it's easy enough to use core_grid.x
     */
    uint32_t k_blocks_per_core =
        tt::div_up(K_blocks, (transpose_core_grid ? in1_parallel_axis_cores : in0_parallel_axis_cores));

    auto cores = corerange_to_cores(core_grid, num_cores, true);

    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        CoreCoord core = cores.at(core_id);
        uint32_t in0_idx = transpose_core_grid ? core.x : core.y;
        uint32_t in1_idx = transpose_core_grid ? core.y : core.x;

        CoreCoord left_core = {(std::size_t)0, (std::size_t)core.y};
        CoreCoord top_core = {(std::size_t)core.x, (std::size_t)0};

        std::vector<CoreCoord> in0_core_order;
        in0_core_order.push_back(transpose_core_grid ? top_core : left_core);
        uint32_t in0_core_order_index = 0;
        for (uint32_t in0_worker_idx = 1; in0_worker_idx < in1_parallel_axis_cores; in0_worker_idx++) {
            CoreCoord in0_worker_core = core;
            size_t& in0_coord_to_modify = transpose_core_grid ? in0_worker_core.y : in0_worker_core.x;
            if (in0_noc == tt::tt_metal::NOC::NOC_1) {
                in0_coord_to_modify = in1_parallel_axis_cores - in0_worker_idx;
            } else {
                in0_coord_to_modify = in0_worker_idx;
            }
            if (in0_coord_to_modify == (transpose_core_grid ? core.y : core.x)) {
                in0_core_order_index = in0_worker_idx;
            }
            in0_core_order.push_back(in0_worker_core);
        }
        std::vector<CoreCoord> in1_core_order;
        in1_core_order.push_back(transpose_core_grid ? left_core : top_core);
        uint32_t in1_core_order_index = 0;
        for (uint32_t in1_worker_idx = 1; in1_worker_idx < in0_parallel_axis_cores; in1_worker_idx++) {
            CoreCoord in1_worker_core = core;
            size_t& in1_coord_to_modify = transpose_core_grid ? in1_worker_core.x : in1_worker_core.y;
            if (in1_noc == tt::tt_metal::NOC::NOC_0) {
                in1_coord_to_modify = in1_worker_idx;
            } else {
                in1_coord_to_modify = in0_parallel_axis_cores - in1_worker_idx;
            }
            if (in1_coord_to_modify == (transpose_core_grid ? core.x : core.y)) {
                in1_core_order_index = in1_worker_idx;
            }
            in1_core_order.push_back(in1_worker_core);
        }
        auto in0_prev_core = in0_core_order.at((std::size_t)std::max((int32_t)in0_core_order_index - 1, 0));
        auto in0_next_core = in0_core_order.at(
            (std::size_t)std::min((size_t)in0_core_order_index + 1, (size_t)in1_parallel_axis_cores - 1));
        auto in1_prev_core = in1_core_order.at((std::size_t)std::max((int32_t)in1_core_order_index - 1, 0));
        auto in1_next_core = in1_core_order.at(
            (std::size_t)std::min((size_t)in1_core_order_index + 1, (size_t)in0_parallel_axis_cores - 1));

        auto in0_prev_core_physical = device->worker_core_from_logical_core(in0_prev_core);
        ;
        auto in0_next_core_physical = device->worker_core_from_logical_core(in0_next_core);
        ;
        auto in1_prev_core_physical = device->worker_core_from_logical_core(in1_prev_core);
        ;
        auto in1_next_core_physical = device->worker_core_from_logical_core(in1_next_core);
        ;

        // uint32_t M_start_block = std::min(M_blocks_per_core * in0_idx, M_blocks - 1);
        // uint32_t M_end_block = std::min(M_blocks_per_core * (in0_idx + 1) - 1, M_blocks - 1);
        // uint32_t N_start_block = std::min(N_blocks_per_core * in1_idx, N_blocks - 1);
        // uint32_t N_end_block = std::min(N_blocks_per_core * (in1_idx + 1) - 1, N_blocks - 1);

        // NOTE: I do not use std::min here because even if a core doesn't need to process a block, the mcast core needs
        // its ACK
        uint32_t M_start_block = M_blocks_per_core * in0_idx;
        uint32_t M_end_block = M_blocks_per_core * (in0_idx + 1) - 1;
        uint32_t N_start_block = N_blocks_per_core * in1_idx;
        uint32_t N_end_block = N_blocks_per_core * (in1_idx + 1) - 1;

        // Defer write to K block with same coordinate as core
        // The writer receiver cores always have core.x > 0
        uint32_t defer_write_k_block = core.y * k_blocks_per_core;
        defer_write_k_block = std::min(defer_write_k_block, K_blocks - 1);

        bool is_in0_sink = core == in0_core_order.at(in1_parallel_axis_cores - 1);
        bool is_in1_sink = core == in1_core_order.at(in0_parallel_axis_cores - 1);

        if (in1_idx == 0) {
            // in0 sender
            std::vector<uint32_t> in0_sender_args = {
                in0_addr,
                out_addr,
                in2_addr,
                is_in0_sink,
                (std::uint32_t)in0_next_core_physical.x,  // in0_dest_noc_x
                (std::uint32_t)in0_next_core_physical.y,  // in0_dest_noc_y
                (std::uint32_t)in0_prev_core_physical.x,  // in0_sender_noc_x
                (std::uint32_t)in0_prev_core_physical.y,  // in0_sender_noc_y
                M_start_block,
                M_end_block,
                N_start_block,
                N_end_block,
                defer_write_k_block,
            };
            SetRuntimeArgs(program, in0_sender_kernels_id, core, in0_sender_args);
        } else {
            // in0 receiver
            std::vector<uint32_t> in0_receiver_args = {
                in0_addr,
                out_addr,
                in2_addr,
                is_in0_sink,
                (std::uint32_t)in0_next_core_physical.x,  // in0_dest_noc_x
                (std::uint32_t)in0_next_core_physical.y,  // in0_dest_noc_y
                (std::uint32_t)in0_prev_core_physical.x,  // in0_sender_noc_x
                (std::uint32_t)in0_prev_core_physical.y,  // in0_sender_noc_y
                M_start_block,
                M_end_block,
                N_start_block,
                N_end_block,
                defer_write_k_block,
            };
            SetRuntimeArgs(program, in0_receiver_kernels_id, core, in0_receiver_args);
        }

        if (in0_idx == 0) {
            // in1 sender
            std::vector<uint32_t> in1_sender_args = {
                in1_addr,
                out_addr,
                in2_addr,
                is_in1_sink,
                (std::uint32_t)in1_next_core_physical.x,  // in1_dest_noc_x
                (std::uint32_t)in1_next_core_physical.y,  // in1_dest_noc_y
                (std::uint32_t)in1_prev_core_physical.x,  // in1_sender_noc_x
                (std::uint32_t)in1_prev_core_physical.y,  // in1_sender_noc_y
                M_start_block,
                M_end_block,
                N_start_block,
                N_end_block,
                defer_write_k_block,
            };
            SetRuntimeArgs(program, in1_sender_kernels_id, core, in1_sender_args);
        } else {
            // in1 receiver
            std::vector<uint32_t> in1_receiver_args = {
                in1_addr,
                out_addr,
                in2_addr,
                is_in1_sink,
                (std::uint32_t)in1_next_core_physical.x,  // in1_dest_noc_x
                (std::uint32_t)in1_next_core_physical.y,  // in1_dest_noc_y
                (std::uint32_t)in1_prev_core_physical.x,  // in1_sender_noc_x
                (std::uint32_t)in1_prev_core_physical.y,  // in1_sender_noc_y
                M_start_block,
                M_end_block,
                N_start_block,
                N_end_block,
                defer_write_k_block,
            };
            SetRuntimeArgs(program, in1_receiver_kernels_id, core, in1_receiver_args);
        }

        std::vector<uint32_t> compute_runtime_args = {
            M_start_block,
            M_end_block,
            N_start_block,
            N_end_block,
        };
        SetRuntimeArgs(program, compute_kernels_id, core, compute_runtime_args);
    }

    auto override_runtime_arguments_callback =
        [num_cores,
         cores,
         in0_sender_kernels_id,
         in0_receiver_kernels_id,
         in1_sender_kernels_id,
         in1_receiver_kernels_id,
         transpose_core_grid](
            const void* operation,
            tt::tt_metal::Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto in0_addr = input_tensors.at(0).buffer()->address();
            auto in1_addr = input_tensors.at(1).buffer()->address();
            auto output_addr = output_tensors.at(0).buffer()->address();
            auto in2_addr =
                optional_input_tensors.at(0).has_value() ? optional_input_tensors.at(0).value().buffer()->address() : 0;

            auto& in0_sender_runtime_args = GetRuntimeArgs(program, in0_sender_kernels_id);
            auto& in0_receiver_runtime_args = GetRuntimeArgs(program, in0_receiver_kernels_id);
            auto& in1_sender_runtime_args = GetRuntimeArgs(program, in1_sender_kernels_id);
            auto& in1_receiver_runtime_args = GetRuntimeArgs(program, in1_receiver_kernels_id);

            for (uint32_t i = 0; i < num_cores; ++i) {
                CoreCoord core = cores.at(i);
                uint32_t in0_idx = transpose_core_grid ? core.x : core.y;
                uint32_t in1_idx = transpose_core_grid ? core.y : core.x;
                if (in1_idx == 0) {
                    auto& in0_sender_args = in0_sender_runtime_args[core.x][core.y];
                    in0_sender_args[0] = in0_addr;
                    in0_sender_args[1] = output_addr;
                    in0_sender_args[2] = in2_addr;
                } else {
                    auto& in0_receiver_args = in0_receiver_runtime_args[core.x][core.y];
                    in0_receiver_args[1] = output_addr;
                    in0_receiver_args[2] = in2_addr;
                }
                if (in0_idx == 0) {
                    auto& in1_sender_args = in1_sender_runtime_args[core.x][core.y];
                    in1_sender_args[0] = in1_addr;
                    in1_sender_args[1] = output_addr;
                    in1_sender_args[2] = in2_addr;
                } else {
                    auto& in1_receiver_args = in1_receiver_runtime_args[core.x][core.y];
                    in1_receiver_args[1] = output_addr;
                    in1_receiver_args[2] = in2_addr;
                }
            }
        };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::experimental::minimal_matmul::detail
