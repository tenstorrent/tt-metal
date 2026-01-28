// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d_program_factory.hpp"
#include "conv3d_device_operation_types.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include <algorithm>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::experimental::prim {

Conv3dProgramFactory::cached_program_t Conv3dProgramFactory::create(
    const Conv3dParams& operation_attributes, const Conv3dInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;
    const auto& bias_tensor = tensor_args.bias_tensor;
    const auto& output_tensor = tensor_return_value;

    // Extract config from operation_attributes
    const auto& config = operation_attributes.config;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    auto grid_size = config.compute_with_storage_grid_size;
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    auto num_cores = core_grid.size();
    /*
    First implementation just performs vol2col on a single core.
    */

    auto input_tensor_shape = input_tensor.logical_shape();
    uint32_t N = input_tensor_shape[0];
    uint32_t T_in = input_tensor_shape[1];
    uint32_t H_in = input_tensor_shape[2];
    uint32_t W_in = input_tensor_shape[3];
    uint32_t C_in = input_tensor_shape[4];
    auto [T_out, H_out, W_out] = detail::compute_output_dims(
        T_in, H_in, W_in, operation_attributes.padding, operation_attributes.stride, operation_attributes.kernel_size);
    uint32_t C_out = operation_attributes.output_channels;

    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto dtype_bytes = input_tensor.element_size();
    auto tile_size = tt::tile_size(data_format);

    bool use_bias = bias_tensor.has_value();

    /* Shapes/sizes needed in the kernel
        Reader does volume2column to convert some `T_block x H_block x W_block` of activation
        to `T_block x H_block x W_block, kD x kH x kW x C_in` patches.
        Compute takes this `num_patches x patch_size` CB and tilizes it.

        Writer reads the weights of size `kD x kH x kW x C_in, C_out`, tilized.
        Writer reads the bias of size `1, C_out`, tilized.
        Compute runs matmul on `patches @ kernel` and adds bias.
        Compute untilizes the result.
        Writer writes the result to the output tensor.


    Padding/tilizing constraints:
        - ceil(num_patches / TILE_HEIGHT) is number of tile rows of matmul
        - `kD x kH x kW x C_in` of the kernel weight is padded to tile size (since it's tilized)
            and must be padded with zeros so the MM result is correct.
    */

    // If C_out_block is set, use it. Otherwise, use the full number of output channels.
    uint32_t C_out_block = config.C_out_block > 0 ? config.C_out_block : C_out;
    uint32_t C_in_block = config.C_in_block > 0 ? config.C_in_block : C_in;

    uint32_t patch_size = operation_attributes.kernel_size[0] * operation_attributes.kernel_size[1] *
                          operation_attributes.kernel_size[2] * C_in_block;
    uint32_t num_patches = config.T_out_block * config.H_out_block * config.W_out_block;

    uint32_t C_in_num_blocks = tt::div_up(C_in, C_in_block);
    TT_FATAL(C_in_num_blocks * C_in_block == C_in, "C_in_num_blocks * C_in_block must equal C_in");
    uint32_t C_out_num_blocks = tt::div_up(C_out, C_out_block);
    TT_FATAL(C_out_num_blocks * C_out_block == C_out, "C_out_num_blocks * C_out_block must equal C_out");

    uint32_t matmul_M_t = tt::div_up(num_patches, tt::constants::TILE_HEIGHT);
    uint32_t matmul_K_t = tt::div_up(patch_size, tt::constants::TILE_WIDTH);
    uint32_t matmul_N_t = tt::div_up(C_out_block, tt::constants::TILE_WIDTH);

    uint32_t num_patches_tile_padded = tt::round_up(num_patches, tt::constants::TILE_HEIGHT);

    // NOTE: Should this be padded up to tile_size for tilize_block?
    uint32_t patch_size_bytes =
        tt::round_up(patch_size, tt::constants::TILE_WIDTH) * dtype_bytes;  // bytes per patch row
    // NOTE: Also padded up to tile size
    uint32_t C_out_block_bytes = C_out_block * dtype_bytes;  // bytes per output channel row
    uint32_t C_in_block_bytes = C_in_block * dtype_bytes;    // bytes per input channel row

    log_debug(tt::LogOp, "Block sizes:");
    log_debug(tt::LogOp, "  T_out_block: {}", config.T_out_block);
    log_debug(tt::LogOp, "  H_out_block: {}", config.H_out_block);
    log_debug(tt::LogOp, "  W_out_block: {}", config.W_out_block);
    log_debug(tt::LogOp, "  C_out_block: {}", C_out_block);
    log_debug(tt::LogOp, "  C_out_num_blocks: {}", C_out_num_blocks);
    log_debug(tt::LogOp, "Patch size: {}", patch_size);
    log_debug(tt::LogOp, "Num patches: {}", num_patches);
    log_debug(tt::LogOp, "Patch size bytes: {}", patch_size_bytes);
    log_debug(tt::LogOp, "C_out block bytes: {}", C_out_block_bytes);
    log_debug(tt::LogOp, "Num patches tile padded: {}", num_patches_tile_padded);
    log_debug(tt::LogOp, "Matmul M_t: {}", matmul_M_t);
    log_debug(tt::LogOp, "Matmul K_t: {}", matmul_K_t);
    log_debug(tt::LogOp, "Matmul N_t: {}", matmul_N_t);
    // Log CB sizes
    log_debug(tt::LogOp, "CB vol2col_rm: page_size={} bytes, num_pages={}", patch_size_bytes, num_patches);

    log_debug(tt::LogOp, "CB vol2col_tiled: page_size={} bytes, num_pages={}", tile_size, matmul_M_t * matmul_K_t);

    log_debug(tt::LogOp, "CB weight_tiled: page_size={} bytes, num_pages={}", tile_size, matmul_K_t * matmul_N_t);

    log_debug(
        tt::LogOp, "CB matmul_interm_tiled: page_size={} bytes, num_pages={}", tile_size, matmul_M_t * matmul_N_t);

    log_debug(
        tt::LogOp, "CB matmul_result_rm: page_size={} bytes, num_pages={}", C_out_block_bytes, num_patches_tile_padded);

    // Create circular buffers for vol2col, weights, bias and matmul intermediates
    uint32_t next_cb_index = tt::CBIndex::c_0;

    uint32_t cb_vol2col_rm_id = next_cb_index++;
    tt::tt_metal::create_cb(cb_vol2col_rm_id, program, core_grid, patch_size_bytes, num_patches, data_format);

    uint32_t cb_vol2col_tiled_id = next_cb_index++;
    tt::tt_metal::create_cb(cb_vol2col_tiled_id, program, core_grid, tile_size, matmul_M_t * matmul_K_t, data_format);

    uint32_t cb_weight_tiled_id = next_cb_index++;
    tt::tt_metal::create_cb(cb_weight_tiled_id, program, core_grid, tile_size, matmul_K_t * matmul_N_t, data_format);

    uint32_t cb_matmul_interm_tiled_id = next_cb_index++;
    tt::tt_metal::create_cb(
        cb_matmul_interm_tiled_id, program, core_grid, tile_size, matmul_M_t * matmul_N_t, data_format);

    // NOTE: Most kernels create RM CB with tile_size pages and num_tile number of pages.
    // Using stick pages led to PCC issues.
    uint32_t cb_matmul_result_rm_id = next_cb_index++;
    tt::tt_metal::create_cb(
        cb_matmul_result_rm_id,
        program,
        core_grid,
        tile_size,
        matmul_M_t * matmul_N_t,  // untilize will write padded rows, so this must be sized to avoid overflowing CB
        data_format);

    uint32_t cb_reduction_tiled_id =
        32;  // Invalid value for cb index since there is only 32 of them and the indices go from 0 to 31
    uint32_t cb_worker_ack_back_id =
        32;  // Invalid value for cb index since there is only 32 of them and the indices go from 0 to 31
    if (C_in_num_blocks > 1) {
        // Implies reduction step
        cb_reduction_tiled_id = next_cb_index++;
        tt::tt_metal::create_cb(
            cb_reduction_tiled_id, program, core_grid, tile_size, matmul_M_t * matmul_N_t, data_format);

        cb_worker_ack_back_id = next_cb_index++;
        tt::tt_metal::create_cb(cb_worker_ack_back_id, program, core_grid, tile_size, 1, data_format);
    }

    uint32_t cb_bias_tiled_id =
        32;  // Invalid value for cb index since there is only 32 of them and the indices go from 0 to 31
    if (use_bias) {
        cb_bias_tiled_id = next_cb_index++;
        tt::tt_metal::create_cb(cb_bias_tiled_id, program, core_grid, tile_size, matmul_N_t, data_format);
    }

    bool is_padding_zeros = operation_attributes.padding_mode == "zeros";

    uint32_t in_row_size_bytes = input_tensor.buffer()->aligned_page_size();
    uint32_t out_row_size_bytes = output_tensor.buffer()->aligned_page_size();

    log_debug(tt::LogOp, "Input tensor shape: N={}, T={}, H={}, W={}, C={}", N, T_in, H_in, W_in, C_in);
    log_debug(tt::LogOp, "Output tensor shape: T={}, H={}, W={}, C={}", T_out, H_out, W_out, C_out);
    log_debug(
        tt::LogOp,
        "Kernel size: {}x{}x{}",
        operation_attributes.kernel_size[0],
        operation_attributes.kernel_size[1],
        operation_attributes.kernel_size[2]);
    log_debug(
        tt::LogOp,
        "Stride: {}x{}x{}",
        operation_attributes.stride[0],
        operation_attributes.stride[1],
        operation_attributes.stride[2]);
    log_debug(
        tt::LogOp,
        "Padding: {}x{}x{}",
        operation_attributes.padding[0],
        operation_attributes.padding[1],
        operation_attributes.padding[2]);
    log_debug(tt::LogOp, "Groups: {}", operation_attributes.groups);
    log_debug(tt::LogOp, "Patch size: {}", patch_size);
    log_debug(tt::LogOp, "Input row size (bytes): {}", in_row_size_bytes);
    log_debug(tt::LogOp, "Output row size (bytes): {}", out_row_size_bytes);
    log_debug(tt::LogOp, "Data format: {}", data_format);

    // Set up semaphore for synchronization. It is dual-purpose.
    // On the reducer core, it tracks the number of workers that are done with an output block.
    // On the worker core, it is a valid bit indicating the worker can continue.
    auto semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, 0);

    std::vector<uint32_t> reader_compile_time_args = {
        cb_vol2col_rm_id,
        N,
        T_in,
        H_in,
        W_in,
        C_in,
        T_out,
        H_out,
        W_out,
        C_out,
        operation_attributes.padding[0],
        operation_attributes.padding[1],
        operation_attributes.padding[2],
        operation_attributes.kernel_size[0],
        operation_attributes.kernel_size[1],
        operation_attributes.kernel_size[2],
        config.T_out_block,
        config.H_out_block,
        config.W_out_block,
        C_out_num_blocks,
        in_row_size_bytes,
        C_in_block_bytes,
        out_row_size_bytes,
        is_padding_zeros,
        semaphore_id,
        operation_attributes.stride[0],
        operation_attributes.stride[1],
        operation_attributes.stride[2]};
    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_compile_time_args);

    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/reader_vol2col.cpp",
        core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Matmul parameters
    auto* device = input_tensor.device();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
    const uint32_t in0_block_w = matmul_K_t;

    const uint32_t out_subblock_w = std::min(matmul_N_t, dst_size);
    TT_FATAL(matmul_N_t % out_subblock_w == 0, "matmul_N_t must be divisible by out_subblock_w");
    // If out_subblock_w is full row of output, scale subblock_h so volume = dst_size. Otherwise it's 1 to maintain
    // row-major intermediate buffer.
    const uint32_t out_subblock_h =
        (out_subblock_w == matmul_N_t) ? (std::min(matmul_M_t, dst_size / out_subblock_w)) : 1;

    const uint32_t in0_num_subblocks = matmul_M_t / out_subblock_h;
    const uint32_t in1_num_subblocks = matmul_N_t / out_subblock_w;

    log_debug(tt::LogOp, "Matmul parameters:");
    log_debug(tt::LogOp, "  matmul_M_t: {}", matmul_M_t);
    log_debug(tt::LogOp, "  matmul_K_t: {}", matmul_K_t);
    log_debug(tt::LogOp, "  matmul_N_t: {}", matmul_N_t);
    log_debug(tt::LogOp, "  dst_size: {}", dst_size);
    log_debug(tt::LogOp, "  in0_block_w: {}", in0_block_w);
    log_debug(tt::LogOp, "  out_subblock_w: {}", out_subblock_w);
    log_debug(tt::LogOp, "  out_subblock_h: {}", out_subblock_h);
    log_debug(tt::LogOp, "  in0_num_subblocks: {}", in0_num_subblocks);
    log_debug(tt::LogOp, "  in1_num_subblocks: {}", in1_num_subblocks);

    std::vector<uint32_t> compute_compile_time_args = {
        cb_vol2col_rm_id,
        cb_vol2col_tiled_id,
        cb_weight_tiled_id,
        cb_bias_tiled_id,
        cb_matmul_interm_tiled_id,
        cb_matmul_result_rm_id,
        cb_reduction_tiled_id,
        cb_worker_ack_back_id,
        N,
        num_patches,
        matmul_M_t,
        matmul_K_t,
        matmul_N_t,
        (uint32_t)use_bias,
        T_out,
        H_out,
        W_out,
        config.T_out_block,
        config.H_out_block,
        config.W_out_block,
        C_out_num_blocks,
        in0_num_subblocks,
        in1_num_subblocks,
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
        semaphore_id};

    auto compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/compute.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args = {
        cb_matmul_result_rm_id,
        cb_weight_tiled_id,
        cb_bias_tiled_id,
        cb_matmul_interm_tiled_id,
        cb_reduction_tiled_id,
        cb_worker_ack_back_id,
        N,
        T_out,
        H_out,
        W_out,
        config.T_out_block,
        config.H_out_block,
        config.W_out_block,
        C_out_num_blocks,
        matmul_M_t,
        matmul_K_t,
        matmul_N_t,
        num_patches_tile_padded,
        out_row_size_bytes,
        C_out_block_bytes,
        (uint32_t)use_bias,
        semaphore_id};
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*weight_tensor.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(bias_tensor.has_value() ? bias_tensor.value().buffer() : nullptr)
        .append_to(writer_compile_time_args);

    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/writer.cpp",
        core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    uint32_t input_addr = input_tensor.buffer()->address();
    uint32_t out_addr = output_tensor.buffer()->address();
    uint32_t weight_addr = weight_tensor.buffer()->address();
    uint32_t bias_addr = bias_tensor.has_value() ? bias_tensor.value().buffer()->address() : 0;

    /**
     * Compute parallelism for multi-core.
     * We now parallelize across C_in as the outermost dimension, followed by
     * C_out, T_out, H_out, and W_out dimensions. Cores working on the same output block
     * but different C_in ranges will need to synchronize for reduction.
     */

    // Calculate number of blocks along each dimension
    uint32_t T_out_blocks = tt::div_up(T_out, config.T_out_block);
    uint32_t H_out_blocks = tt::div_up(H_out, config.H_out_block);
    uint32_t W_out_blocks = tt::div_up(W_out, config.W_out_block);

    // Define parallelization factors for each dimension
    // C_in is the outermost parallelization dimension
    uint32_t c_in_parallel_factor = std::min(C_in_num_blocks, (uint32_t)num_cores);

    // Remaining cores per output block
    uint32_t cores_per_output = std::max(1u, (uint32_t)(num_cores / c_in_parallel_factor));

    // Distribute output parallelism across dimensions
    uint32_t c_out_parallel_factor = std::min(C_out_num_blocks, cores_per_output);
    uint32_t remaining_parallel = cores_per_output / c_out_parallel_factor;

    uint32_t t_out_parallel_factor = std::min(T_out_blocks, remaining_parallel);
    remaining_parallel = remaining_parallel / t_out_parallel_factor;

    uint32_t h_out_parallel_factor = std::min(H_out_blocks, remaining_parallel);
    remaining_parallel = remaining_parallel / h_out_parallel_factor;

    uint32_t w_out_parallel_factor = std::min(W_out_blocks, remaining_parallel);

    // Calculate total output blocks that will be processed in parallel
    uint32_t total_output_parallel =
        c_out_parallel_factor * t_out_parallel_factor * h_out_parallel_factor * w_out_parallel_factor;

    // Verify parallelization is valid
    TT_FATAL(
        c_in_parallel_factor * total_output_parallel <= num_cores,
        "Parallelism must not exceed number of cores. Got {}, expected at most {}.",
        c_in_parallel_factor * total_output_parallel,
        num_cores);

    log_debug(tt::LogOp, "Parallelization scheme:");
    log_debug(tt::LogOp, "C_in_num_blocks: {}, C_in_parallel_factor: {}", C_in_num_blocks, c_in_parallel_factor);
    log_debug(tt::LogOp, "C_out_parallel_factor: {}", c_out_parallel_factor);
    log_debug(tt::LogOp, "T_out_parallel_factor: {}", t_out_parallel_factor);
    log_debug(tt::LogOp, "H_out_parallel_factor: {}", h_out_parallel_factor);
    log_debug(tt::LogOp, "W_out_parallel_factor: {}", w_out_parallel_factor);
    log_debug(tt::LogOp, "Total output parallel blocks: {}", total_output_parallel);

    // Calculate blocks per core using ceiling division
    const uint32_t c_in_per_core = tt::div_up(C_in_num_blocks, c_in_parallel_factor);
    const uint32_t c_out_per_core = tt::div_up(C_out_num_blocks, c_out_parallel_factor);
    const uint32_t t_out_per_core = tt::div_up(T_out_blocks, t_out_parallel_factor);
    const uint32_t h_out_per_core = tt::div_up(H_out_blocks, h_out_parallel_factor);
    const uint32_t w_out_per_core = tt::div_up(W_out_blocks, w_out_parallel_factor);

    // Track cores that need to perform reduction together
    std::vector<std::vector<uint32_t>> reduction_groups(total_output_parallel);

    // First loop: Calculate runtime args and build reduction groups
    std::vector<std::vector<uint32_t>> reader_args_per_core(num_cores);
    std::vector<std::vector<uint32_t>> compute_args_per_core(num_cores);
    std::vector<std::vector<uint32_t>> writer_args_per_core(num_cores);
    std::vector<uint32_t> reducer_core_ids(total_output_parallel, UINT32_MAX);
    std::vector<std::vector<uint32_t>> worker_core_ids(total_output_parallel);

    // Track physical coordinates for reducers and workers
    std::vector<uint32_t> reducer_core_physical_xs(total_output_parallel);
    std::vector<uint32_t> reducer_core_physical_ys(total_output_parallel);
    std::vector<std::vector<uint32_t>> worker_core_physical_xs(total_output_parallel);
    std::vector<std::vector<uint32_t>> worker_core_physical_ys(total_output_parallel);

    auto cores = corerange_to_cores(core_grid, num_cores, true);

    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        CoreCoord core = cores.at(core_id);

        // First, determine which output block and which C_in range this core handles
        uint32_t output_idx = core_id % total_output_parallel;
        uint32_t c_in_idx = core_id / total_output_parallel;

        // Decompose output_idx into (c_out, t_out, h_out, w_out) coordinates
        uint32_t c_out_idx = output_idx / (t_out_parallel_factor * h_out_parallel_factor * w_out_parallel_factor);
        uint32_t remaining = output_idx % (t_out_parallel_factor * h_out_parallel_factor * w_out_parallel_factor);

        uint32_t t_out_idx = remaining / (h_out_parallel_factor * w_out_parallel_factor);
        remaining = remaining % (h_out_parallel_factor * w_out_parallel_factor);

        uint32_t h_out_idx = remaining / w_out_parallel_factor;
        uint32_t w_out_idx = remaining % w_out_parallel_factor;

        // Calculate reduction group ID (same as output_idx)
        uint32_t reduction_group_id = output_idx;

        // Calculate block ranges
        uint32_t c_in_block_start = c_in_idx * c_in_per_core;
        uint32_t c_in_block_end = std::min(c_in_block_start + c_in_per_core, C_in_num_blocks);

        uint32_t c_out_block_start = c_out_idx * c_out_per_core;
        uint32_t c_out_block_end = std::min(c_out_block_start + c_out_per_core, C_out_num_blocks);

        uint32_t t_out_block_start = t_out_idx * t_out_per_core;
        uint32_t t_out_block_end = std::min(t_out_block_start + t_out_per_core, T_out_blocks);

        uint32_t h_out_block_start = h_out_idx * h_out_per_core;
        uint32_t h_out_block_end = std::min(h_out_block_start + h_out_per_core, H_out_blocks);

        uint32_t w_out_block_start = w_out_idx * w_out_per_core;
        uint32_t w_out_block_end = std::min(w_out_block_start + w_out_per_core, W_out_blocks);

        // Calculate actual indices
        uint32_t t_out_start = t_out_block_start * config.T_out_block;
        uint32_t t_out_end = std::min(t_out_block_end * config.T_out_block, T_out);

        uint32_t h_out_start = h_out_block_start * config.H_out_block;
        uint32_t h_out_end = std::min(h_out_block_end * config.H_out_block, H_out);

        uint32_t w_out_start = w_out_block_start * config.W_out_block;
        uint32_t w_out_end = std::min(w_out_block_end * config.W_out_block, W_out);

        // Check if this core has actual work to do
        bool has_work = (c_in_block_end > c_in_block_start) && (c_out_block_end > c_out_block_start) &&
                        (t_out_end > t_out_start) && (h_out_end > h_out_start) && (w_out_end > w_out_start);

        bool is_reducer = has_work && c_in_idx == 0;

        // Only include in reduction group if there's actual work to do
        if (has_work) {
            // Add this core to its reduction group
            reduction_groups[reduction_group_id].push_back(core_id);

            // Track reducer and worker cores
            if (is_reducer) {
                reducer_core_ids[reduction_group_id] = core_id;
                // Get physical coordinates for reducer core
                auto reducer_core_physical = device->worker_core_from_logical_core(core);
                reducer_core_physical_xs[reduction_group_id] = (uint32_t)reducer_core_physical.x;
                reducer_core_physical_ys[reduction_group_id] = (uint32_t)reducer_core_physical.y;
            } else {
                worker_core_ids[reduction_group_id].push_back(core_id);
                // Get physical coordinates for worker core
                auto worker_core_physical = device->worker_core_from_logical_core(core);
                worker_core_physical_xs[reduction_group_id].push_back((uint32_t)worker_core_physical.x);
                worker_core_physical_ys[reduction_group_id].push_back((uint32_t)worker_core_physical.y);
            }
        }

        log_debug(
            tt::LogOp,
            "Core {},{}: C_in=[{},{}), C_out=[{},{}), T_out=[{},{}), H_out=[{},{}), W_out=[{},{}), "
            "ReductionGroup={}, C_in_idx={}, HasWork={}, IsReducer={}",
            core.x,
            core.y,
            c_in_block_start,
            c_in_block_end,
            c_out_block_start,
            c_out_block_end,
            t_out_start,
            t_out_end,
            h_out_start,
            h_out_end,
            w_out_start,
            w_out_end,
            reduction_group_id,
            c_in_idx,
            has_work,
            is_reducer);

        // Store runtime args for later use
        reader_args_per_core[core_id] = {
            input_addr,
            c_in_block_start,
            c_in_block_end,
            c_out_block_start,
            c_out_block_end,
            t_out_start,
            t_out_end,
            h_out_start,
            h_out_end,
            w_out_start,
            w_out_end,
        };

        compute_args_per_core[core_id] = {
            c_in_block_start,
            c_in_block_end,
            c_out_block_start,
            c_out_block_end,
            t_out_start,
            t_out_end,
            h_out_start,
            h_out_end,
            w_out_start,
            w_out_end,
            (uint32_t)is_reducer};

        writer_args_per_core[core_id] = {
            out_addr,
            weight_addr,
            bias_addr,
            c_in_block_start,
            c_in_block_end,
            c_out_block_start,
            c_out_block_end,
            t_out_start,
            t_out_end,
            h_out_start,
            h_out_end,
            w_out_start,
            w_out_end,
            (uint32_t)is_reducer};
    }

    // Log reduction groups information
    for (uint32_t group_id = 0; group_id < reduction_groups.size(); group_id++) {
        const auto& group = reduction_groups[group_id];
        if (!group.empty()) {
            std::string cores_str;
            for (uint32_t core_id : group) {
                CoreCoord core = cores.at(core_id);
                if (!cores_str.empty()) {
                    cores_str += ", ";
                }
                cores_str += "(" + std::to_string(core.x) + "," + std::to_string(core.y) + ")";
            }

            [[maybe_unused]] CoreCoord reducer_core = {
                reducer_core_ids[group_id] % grid_size.x, reducer_core_ids[group_id] / grid_size.x};

            log_debug(
                tt::LogOp,
                "Reduction Group {}: {} cores [{}], Reducer: ({},{})",
                group_id,
                group.size(),
                cores_str,
                reducer_core.x,
                reducer_core.y);
        }
    }

    // Second loop: Set runtime args with reducer and worker information
    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        CoreCoord core = cores.at(core_id);
        uint32_t output_idx = core_id % total_output_parallel;
        uint32_t reduction_group_id = output_idx;

        auto& reader_args = reader_args_per_core[core_id];
        auto& compute_args = compute_args_per_core[core_id];
        auto& writer_args = writer_args_per_core[core_id];

        // Get is_reducer value from the stored arguments
        [[maybe_unused]] bool is_reducer = (writer_args[13] == 1);

        // Add reducer core coordinates
        if (reducer_core_ids[reduction_group_id] != UINT32_MAX) {
            writer_args.push_back(reducer_core_physical_xs[reduction_group_id]);
            writer_args.push_back(reducer_core_physical_ys[reduction_group_id]);
        }

        // Add worker cores count
        uint32_t num_workers = worker_core_ids[reduction_group_id].size();
        compute_args.push_back(num_workers);
        writer_args.push_back(num_workers);

        // Add all worker core coordinates to runtime args
        writer_args.insert(
            writer_args.end(),
            worker_core_physical_xs[reduction_group_id].begin(),
            worker_core_physical_xs[reduction_group_id].end());
        writer_args.insert(
            writer_args.end(),
            worker_core_physical_ys[reduction_group_id].begin(),
            worker_core_physical_ys[reduction_group_id].end());

        // Prepare worker cores string for logging
        std::string worker_cores_str;
        for (uint32_t i = 0; i < num_workers; i++) {
            if (!worker_cores_str.empty()) {
                worker_cores_str += ", ";
            }
            worker_cores_str += "(" + std::to_string(worker_core_physical_xs[reduction_group_id][i]) + "," +
                                std::to_string(worker_core_physical_ys[reduction_group_id][i]) + ")";
        }
        log_debug(
            tt::LogOp,
            "Core ({},{}): IsReducer={}, ReductionGroup={}, ReducerCore=({},{}), Workers=[{}]",
            core.x,
            core.y,
            is_reducer,
            reduction_group_id,
            reducer_core_physical_xs[reduction_group_id],
            reducer_core_physical_ys[reduction_group_id],
            worker_cores_str);

        // Set runtime args
        SetRuntimeArgs(program, reader_kernels_id, core, reader_args);
        SetRuntimeArgs(program, compute_kernels_id, core, compute_args);
        SetRuntimeArgs(program, writer_kernels_id, core, writer_args);
    }

    // Return cached program with shared variables
    return cached_program_t{
        std::move(program),
        {/* num_cores = */ num_cores,
         /* cores = */ cores,
         /* grid_size = */ grid_size,
         /* reader_kernels_id = */ reader_kernels_id,
         /* writer_kernels_id = */ writer_kernels_id,
         /* compute_kernels_id = */ compute_kernels_id}};
}

void Conv3dProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const Conv3dParams& /*operation_attributes*/,
    const Conv3dInputs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt::tt_metal;

    auto& shared_vars = cached_program.shared_variables;
    auto& reader_kernels_id = shared_vars.reader_kernels_id;
    auto& writer_kernels_id = shared_vars.writer_kernels_id;
    auto& num_cores = shared_vars.num_cores;
    auto& cores = shared_vars.cores;
    auto& program = cached_program.program;

    auto& reader_args_by_core = GetRuntimeArgs(program, reader_kernels_id);
    auto& writer_args_by_core = GetRuntimeArgs(program, writer_kernels_id);

    auto input_addr = tensor_args.input_tensor.buffer()->address();
    auto weight_addr = tensor_args.weight_tensor.buffer()->address();
    auto output_addr = tensor_return_value.buffer()->address();
    auto bias_addr = tensor_args.bias_tensor.has_value() ? tensor_args.bias_tensor.value().buffer()->address() : 0;

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = cores.at(i);
        auto& reader_args = reader_args_by_core[core.x][core.y];
        auto& writer_args = writer_args_by_core[core.x][core.y];
        reader_args[0] = input_addr;
        writer_args[0] = output_addr;
        writer_args[1] = weight_addr;
        writer_args[2] = bias_addr;
    }
}

}  // namespace ttnn::experimental::prim
