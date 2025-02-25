// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d_device_operation.hpp"
#include "conv3d_program_factory.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include <algorithm>

namespace ttnn::operations::conv::conv3d::detail {

operation::ProgramWithCallbacks conv3d_factory(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const Conv3dConfig& config,
    const Tensor& output_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    Program program = CreateProgram();

    auto grid_size = config.compute_with_storage_grid_size;
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    auto num_cores = core_grid.size();
    /*
    First implementation just performs vol2col on a single core.
    */

    auto input_tensor_shape = input_tensor.get_logical_shape();
    uint32_t N = input_tensor_shape[0];
    uint32_t T_in = input_tensor_shape[1];
    uint32_t H_in = input_tensor_shape[2];
    uint32_t W_in = input_tensor_shape[3];
    uint32_t C_in = input_tensor_shape[4];
    auto [T_out, H_out, W_out] = detail::compute_output_dims(T_in, H_in, W_in, config.padding, config.kernel_size);
    uint32_t C_out = config.output_channels;

    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    auto dtype_bytes = input_tensor.element_size();
    auto tile_size = tt::tt_metal::detail::TileSize(data_format);

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

    uint32_t patch_size = config.kernel_size[0] * config.kernel_size[1] * config.kernel_size[2] * C_in;
    uint32_t num_patches = config.T_out_block * config.H_out_block * config.W_out_block;

    // If C_out_block is set, use it. Otherwise, use the full number of output channels.
    uint32_t C_out_block = config.C_out_block > 0 ? config.C_out_block : C_out;
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

    log_info("Block sizes:");
    log_info("  T_out_block: {}", config.T_out_block);
    log_info("  H_out_block: {}", config.H_out_block);
    log_info("  W_out_block: {}", config.W_out_block);
    log_info("  C_out_block: {}", C_out_block);
    log_info("  C_out_num_blocks: {}", C_out_num_blocks);
    log_info("Patch size: {}", patch_size);
    log_info("Num patches: {}", num_patches);
    log_info("Patch size bytes: {}", patch_size_bytes);
    log_info("C_out block bytes: {}", C_out_block_bytes);
    log_info("Num patches tile padded: {}", num_patches_tile_padded);
    log_info("Matmul M_t: {}", matmul_M_t);
    log_info("Matmul K_t: {}", matmul_K_t);
    log_info("Matmul N_t: {}", matmul_N_t);
    // Log CB sizes
    log_info("CB vol2col_rm: page_size={} bytes, num_pages={}", patch_size_bytes, num_patches);

    log_info("CB vol2col_tiled: page_size={} bytes, num_pages={}", tile_size, matmul_M_t * matmul_K_t);

    log_info("CB weight_tiled: page_size={} bytes, num_pages={}", tile_size, matmul_K_t * matmul_N_t);

    log_info("CB matmul_interm_tiled: page_size={} bytes, num_pages={}", tile_size, matmul_M_t * matmul_N_t);

    log_info("CB matmul_result_rm: page_size={} bytes, num_pages={}", C_out_block_bytes, num_patches_tile_padded);

    uint32_t cb_vol2col_rm_id = tt::CBIndex::c_0;
    uint32_t cb_vol2col_tiled_id = tt::CBIndex::c_1;
    uint32_t cb_weight_tiled_id = tt::CBIndex::c_2;
    uint32_t cb_matmul_interm_tiled_id = tt::CBIndex::c_3;
    uint32_t cb_matmul_result_rm_id = tt::CBIndex::c_4;
    uint32_t cb_bias_tiled_id = tt::CBIndex::c_5;

    // Create circular buffers for vol2col, weights, bias and matmul intermediates
    auto [_, cb_vol2col_rm_handle] =
        tt::tt_metal::create_cb(cb_vol2col_rm_id, program, core_grid, patch_size_bytes, num_patches, data_format);

    auto [__, cb_vol2col_tiled_handle] = tt::tt_metal::create_cb(
        cb_vol2col_tiled_id, program, core_grid, tile_size, matmul_M_t * matmul_K_t, data_format);

    auto [___, cb_weight_tiled_handle] = tt::tt_metal::create_cb(
        cb_weight_tiled_id, program, core_grid, tile_size, matmul_K_t * matmul_N_t, data_format);

    auto [_____, cb_matmul_interm_tiled_handle] = tt::tt_metal::create_cb(
        cb_matmul_interm_tiled_id, program, core_grid, tile_size, matmul_M_t * matmul_N_t, data_format);

    // NOTE: Most kernels create RM CB with tile_size pages and num_tile number of pages.
    // Using stick pages led to PCC issues.
    auto [______, cb_matmul_result_rm_handle] = tt::tt_metal::create_cb(
        cb_matmul_result_rm_id,
        program,
        core_grid,
        tile_size,
        matmul_M_t * matmul_N_t,  // untilize will write padded rows, so this must be sized to avoid overflowing CB
        data_format);

    if (use_bias) {
        auto [____, cb_bias_tiled_handle] =
            tt::tt_metal::create_cb(cb_bias_tiled_id, program, core_grid, tile_size, matmul_N_t, data_format);
    }

    bool is_padding_zeros = config.padding_mode == "zeros";

    uint32_t in_row_size_bytes = input_tensor.buffer()->aligned_page_size();
    uint32_t out_row_size_bytes = output_tensor.buffer()->aligned_page_size();

    tt::log_info("Input tensor shape: N={}, T={}, H={}, W={}, C={}", N, T_in, H_in, W_in, C_in);
    tt::log_info("Output tensor shape: T={}, H={}, W={}, C={}", T_out, H_out, W_out, C_out);
    tt::log_info("Kernel size: {}x{}x{}", config.kernel_size[0], config.kernel_size[1], config.kernel_size[2]);
    tt::log_info("Stride: {}x{}x{}", config.stride[0], config.stride[1], config.stride[2]);
    tt::log_info("Padding: {}x{}x{}", config.padding[0], config.padding[1], config.padding[2]);
    tt::log_info("Groups: {}", config.groups);
    tt::log_info("Patch size: {}", patch_size);
    tt::log_info("Input row size (bytes): {}", in_row_size_bytes);
    tt::log_info("Output row size (bytes): {}", out_row_size_bytes);
    tt::log_info("Data format: {}", data_format);

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
        config.padding[0],
        config.padding[1],
        config.padding[2],
        config.kernel_size[0],
        config.kernel_size[1],
        config.kernel_size[2],
        config.T_out_block,
        config.H_out_block,
        config.W_out_block,
        C_out_num_blocks,
        in_row_size_bytes,
        out_row_size_bytes,
        is_padding_zeros,
    };

    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/conv/conv3d/device/kernels/reader_vol2col.cpp",
        core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Matmul parameters
    IDevice* device = input_tensor.device();
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

    log_info("Matmul parameters:");
    log_info("  matmul_M_t: {}", matmul_M_t);
    log_info("  matmul_K_t: {}", matmul_K_t);
    log_info("  matmul_N_t: {}", matmul_N_t);
    log_info("  dst_size: {}", dst_size);
    log_info("  in0_block_w: {}", in0_block_w);
    log_info("  out_subblock_w: {}", out_subblock_w);
    log_info("  out_subblock_h: {}", out_subblock_h);
    log_info("  in0_num_subblocks: {}", in0_num_subblocks);
    log_info("  in1_num_subblocks: {}", in1_num_subblocks);

    std::vector<uint32_t> compute_compile_time_args = {
        cb_vol2col_rm_id,
        cb_vol2col_tiled_id,
        cb_weight_tiled_id,
        cb_bias_tiled_id,
        cb_matmul_interm_tiled_id,
        cb_matmul_result_rm_id,
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
        out_subblock_w};

    auto compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/conv/conv3d/device/kernels/compute.cpp",
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
    };

    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/conv/conv3d/device/kernels/writer.cpp",
        core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    uint32_t input_addr = input_tensor.buffer()->address();
    uint32_t out_addr = output_tensor.buffer()->address();
    uint32_t weight_addr = weight_tensor.buffer()->address();
    uint32_t bias_addr = bias_tensor.has_value() ? bias_tensor.value().buffer()->address() : 0;

    /**
     * Compute parallelism for multi-core.
     * Distribute work across C_out, T_out, H_out, and W_out dimensions.
     */

    // Calculate number of blocks along each dimension
    uint32_t T_out_blocks = tt::div_up(T_out, config.T_out_block);
    uint32_t H_out_blocks = tt::div_up(H_out, config.H_out_block);
    uint32_t W_out_blocks = tt::div_up(W_out, config.W_out_block);

    // Define parallelization factors for each dimension
    uint32_t c_out_parallel_factor = std::min(C_out_num_blocks, (uint32_t)num_cores);
    uint32_t t_out_parallel_factor = std::min((uint32_t)(num_cores / c_out_parallel_factor), T_out_blocks);
    uint32_t h_out_parallel_factor =
        std::min((uint32_t)(num_cores / (c_out_parallel_factor * t_out_parallel_factor)), H_out_blocks);
    uint32_t w_out_parallel_factor = std::min(
        (uint32_t)(num_cores / (c_out_parallel_factor * t_out_parallel_factor * h_out_parallel_factor)), W_out_blocks);

    TT_FATAL(
        c_out_parallel_factor * t_out_parallel_factor * h_out_parallel_factor * w_out_parallel_factor <= num_cores,
        "Parallelism must not exceed number of cores. Got {}, expected at most {}.",
        c_out_parallel_factor * t_out_parallel_factor * h_out_parallel_factor * w_out_parallel_factor,
        num_cores);

    log_info("Parallelization scheme:");
    log_info("C_out_parallel_factor: {}", c_out_parallel_factor);
    log_info("T_out_parallel_factor: {}", t_out_parallel_factor);
    log_info("H_out_parallel_factor: {}", h_out_parallel_factor);
    log_info("W_out_parallel_factor: {}", w_out_parallel_factor);

    // Calculate blocks per core using ceiling division
    const uint32_t c_out_per_core = tt::div_up(C_out_num_blocks, c_out_parallel_factor);
    const uint32_t t_out_per_core = tt::div_up(T_out_blocks, t_out_parallel_factor);
    const uint32_t h_out_per_core = tt::div_up(H_out_blocks, h_out_parallel_factor);
    const uint32_t w_out_per_core = tt::div_up(W_out_blocks, w_out_parallel_factor);

    // Set runtime args for each core
    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        CoreCoord core = {core_id % grid_size.x, core_id / grid_size.x};

        // Calculate start/end ranges using mathematical mapping
        uint32_t c_idx = core_id / (t_out_parallel_factor * h_out_parallel_factor * w_out_parallel_factor);
        uint32_t t_idx = (core_id / (h_out_parallel_factor * w_out_parallel_factor)) % t_out_parallel_factor;
        uint32_t h_idx = (core_id / w_out_parallel_factor) % h_out_parallel_factor;
        uint32_t w_idx = core_id % w_out_parallel_factor;

        uint32_t c_out_block_start = std::min(c_idx * c_out_per_core, C_out_num_blocks);
        uint32_t c_out_block_end = std::min(c_out_block_start + c_out_per_core, C_out_num_blocks);
        uint32_t t_out_block_start = std::min(t_idx * t_out_per_core, T_out_blocks);
        uint32_t t_out_block_end = std::min(t_out_block_start + t_out_per_core, T_out_blocks);
        uint32_t h_out_block_start = std::min(h_idx * h_out_per_core, H_out_blocks);
        uint32_t h_out_block_end = std::min(h_out_block_start + h_out_per_core, H_out_blocks);
        uint32_t w_out_block_start = std::min(w_idx * w_out_per_core, W_out_blocks);
        uint32_t w_out_block_end = std::min(w_out_block_start + w_out_per_core, W_out_blocks);

        // Convert block indices to actual indices
        // Note that in C_out, we just use the block index rather than the absolute index
        uint32_t t_out_start = std::min(t_out_block_start * config.T_out_block, T_out);
        uint32_t t_out_end = std::min(t_out_block_end * config.T_out_block, T_out);
        uint32_t h_out_start = std::min(h_out_block_start * config.H_out_block, H_out);
        uint32_t h_out_end = std::min(h_out_block_end * config.H_out_block, H_out);
        uint32_t w_out_start = std::min(w_out_block_start * config.W_out_block, W_out);
        uint32_t w_out_end = std::min(w_out_block_end * config.W_out_block, W_out);

        log_info(
            "Core {},{}: C_out=[{},{}), T_out=[{},{}), H_out=[{},{}), W_out=[{},{})",
            core.x,
            core.y,
            c_out_block_start,
            c_out_block_end,
            t_out_start,
            t_out_end,
            h_out_start,
            h_out_end,
            w_out_start,
            w_out_end);

        // Set runtime args
        SetRuntimeArgs(
            program,
            reader_kernels_id,
            core,
            {input_addr,
             c_out_block_start,
             c_out_block_end,
             t_out_start,
             t_out_end,
             h_out_start,
             h_out_end,
             w_out_start,
             w_out_end});

        SetRuntimeArgs(
            program,
            compute_kernels_id,
            core,
            {c_out_block_start,
             c_out_block_end,
             t_out_start,
             t_out_end,
             h_out_start,
             h_out_end,
             w_out_start,
             w_out_end});

        SetRuntimeArgs(
            program,
            writer_kernels_id,
            core,
            {out_addr,
             weight_addr,
             bias_addr,
             c_out_block_start,
             c_out_block_end,
             t_out_start,
             t_out_end,
             h_out_start,
             h_out_end,
             w_out_start,
             w_out_end});
    }

    auto override_runtime_arguments_callback =
        [num_cores, grid_size, reader_kernels_id, writer_kernels_id](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto& reader_args_by_core = GetRuntimeArgs(program, reader_kernels_id);
            auto& writer_args_by_core = GetRuntimeArgs(program, writer_kernels_id);

            auto input_addr = input_tensors.at(0).buffer()->address();
            auto weight_addr = input_tensors.at(1).buffer()->address();
            auto output_addr = output_tensors.at(0).buffer()->address();
            auto bias_addr =
                optional_input_tensors.at(0).has_value() ? optional_input_tensors.at(0).value().buffer()->address() : 0;

            for (uint32_t i = 0; i < num_cores; ++i) {
                CoreCoord core = {i % grid_size.x, i / grid_size.x};
                auto& reader_args = reader_args_by_core[core.x][core.y];
                auto& writer_args = writer_args_by_core[core.x][core.y];
                reader_args[0] = input_addr;
                writer_args[0] = output_addr;
                writer_args[1] = weight_addr;
                writer_args[2] = bias_addr;
            }
        };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::conv::conv3d::detail
