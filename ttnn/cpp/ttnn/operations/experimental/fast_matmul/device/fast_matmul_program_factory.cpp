// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fast_matmul_program_factory.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include "tt-metalium/data_types.hpp"
#include "ttnn/operations/cb_utils.hpp"

#include <algorithm>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tuple>
#include <utility>
#include <vector>

#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

namespace {

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> determine_default_block_sizes(
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

// Append tensor accessors in a consistent order
void append_accessors(
    std::vector<uint32_t>& args,
    const Tensor& main_tensor,
    const std::vector<Tensor>& output_tensors,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<const Tensor>& ag_input_tensor = std::nullopt) {
    tt::tt_metal::TensorAccessorArgs(*main_tensor.buffer()).append_to(args);
    for (const auto& output_tensor : output_tensors) {
        tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(args);
    }
    if (bias_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*bias_tensor.value().buffer()).append_to(args);
    }
    if (ag_input_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*ag_input_tensor.value().buffer()).append_to(args);
    }
}

}  // namespace

// SHARED IMPLEMENTATION - works with vector of output tensors (exposed for fast_matmul_split)
FastMatmulProgramFactory::shared_variables_t fast_matmul_factory_helper_common(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<operations::unary::UnaryWithParam>& fused_activation,
    const std::optional<const FastMatmulConfig>& config,
    const std::vector<Tensor>& output_tensors,
    const DeviceComputeKernelConfig& compute_kernel_config,
    [[maybe_unused]] uint32_t N_chunks) {
    auto* device = input_tensor.device();

    if (!config.has_value()) {
        log_debug(tt::LogOp, "No config provided, using default block sizes and core grid");
    }

    auto core_grid = input_tensor.shard_spec()->grid;
    auto num_cores = core_grid.size();

    bool use_bias = bias_tensor.has_value();

    /**
     * Determine dataformats, compute kernel config
     */
    auto in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto in0_tile_size = tt::tile_size(in0_data_format);
    auto in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(weight_tensor.dtype());
    auto in1_tile_size = tt::tile_size(in1_data_format);
    auto output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensors[0].dtype());
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
    // Fold activation (LHS) upper dimensions into rows: M_total = prod(upper dims) * M
    uint32_t K = in0_tensor_shape[-1];
    uint32_t M = input_tensor.physical_volume() / K;
    uint32_t N = in1_tensor_shape[-1];

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
    K_block_tiles = 2;  // std::min(K_block_tiles, K_tiles);

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

    auto input_shard_spec = input_tensor.shard_spec().value();
    auto weights_shard_spec = weight_tensor.shard_spec().value();
    auto output_shard_spec = output_tensors[0].shard_spec().value();

    auto input_shard_shape = input_shard_spec.shape;
    auto weights_shard_shape = weights_shard_spec.shape;
    auto output_shard_shape = output_shard_spec.shape;

    log_debug(tt::LogOp, "Input shard shape: {}", input_shard_shape);
    log_debug(tt::LogOp, "Weights shard shape: {}", weights_shard_shape);
    log_debug(tt::LogOp, "Output shard shape: {}", output_shard_shape);

    TT_FATAL(
        input_shard_shape[0] == weights_shard_shape[1],
        "Shard spec shapes are incompatible for matmul: input shard shape {}, weights shard shape {}",
        input_shard_shape,
        weights_shard_shape);

    uint32_t in0_parallel_axis_cores = num_cores;

    uint32_t in1_parallel_axis_cores = 1;

    /**
     * We pad the input dimensions to the nearest multiple of the parallelization factor.
     *
     * Each core is assigned a certain number of tiles in M and N to compute.
     * Within a core, tiles are blocked by M_block_tiles and N_block_tiles.
     * Most output blocks are the full block size, but the last block in M or N can be partial.
     */

    uint32_t padded_M_tiles = tt::round_up(M_tiles, in0_parallel_axis_cores);
    uint32_t padded_N_tiles = tt::round_up(N_tiles, in1_parallel_axis_cores);
    uint32_t padded_K_tiles = tt::round_up(K_tiles, K_block_tiles);

    uint32_t M_tiles_per_core = padded_M_tiles / in0_parallel_axis_cores;
    uint32_t N_tiles_per_core = padded_N_tiles / in1_parallel_axis_cores;

    M_block_tiles = std::min(M_block_tiles, M_tiles_per_core);
    N_block_tiles = std::min(N_block_tiles, N_tiles_per_core);

    uint32_t K_blocks = padded_K_tiles / K_block_tiles;

    uint32_t M_blocks_per_core = tt::div_up(M_tiles_per_core, M_block_tiles);
    uint32_t N_blocks_per_core = tt::div_up(N_tiles_per_core, N_block_tiles);

    log_debug(tt::LogOp, "M_tiles_per_core: {}", M_tiles_per_core);
    log_debug(tt::LogOp, "N_tiles_per_core: {}", N_tiles_per_core);
    log_debug(tt::LogOp, "M_blocks_per_core: {}", M_blocks_per_core);
    log_debug(tt::LogOp, "N_blocks_per_core: {}", N_blocks_per_core);

    uint32_t in0_cb_num_tiles =
        input_shard_shape[0] * input_shard_shape[1] / (tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH);
    uint32_t in1_cb_num_tiles =
        weights_shard_shape[0] * weights_shard_shape[1] / (tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH);

    // TODO: consider not double buffering the output
    uint32_t out_cb_num_tiles =
        output_shard_shape[0] * output_shard_shape[1] / (tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH);

    uint32_t in0_cb_id = tt::CBIndex::c_0;
    auto in_cb_handle = std::get<1>(tt::tt_metal::create_cb(
        in0_cb_id, program, core_grid, in0_tile_size, in0_cb_num_tiles, in0_data_format, input_tensor.buffer()));

    uint32_t in1_cb_id = tt::CBIndex::c_1;
    auto weight_cb_handle = std::get<1>(tt::tt_metal::create_cb(
        in1_cb_id, program, core_grid, in1_tile_size, in1_cb_num_tiles, in1_data_format, weight_tensor.buffer()));

    uint32_t out_cb_id = tt::CBIndex::c_2;
    auto out_cb_handle = std::get<1>(tt::tt_metal::create_cb(
        out_cb_id,
        program,
        core_grid,
        out_tile_size,
        out_cb_num_tiles,
        output_data_format,
        output_tensors[0].buffer()));

    log_debug(tt::LogOp, "in0_cb_id: {}", in0_cb_id);
    log_debug(tt::LogOp, "in1_cb_id: {}", in1_cb_id);
    log_debug(tt::LogOp, "out_cb_id: {}", out_cb_id);
    log_debug(tt::LogOp, "M_tiles: {}", M_tiles);
    log_debug(tt::LogOp, "padded_M_tiles: {}", padded_M_tiles);
    log_debug(tt::LogOp, "K_tiles: {}", K_tiles);
    log_debug(tt::LogOp, "padded_K_tiles: {}", padded_K_tiles);
    log_debug(tt::LogOp, "N_tiles: {}", N_tiles);
    log_debug(tt::LogOp, "padded_N_tiles: {}", padded_N_tiles);
    log_debug(tt::LogOp, "M_block_tiles: {}", M_block_tiles);
    log_debug(tt::LogOp, "K_block_tiles: {}", K_block_tiles);
    log_debug(tt::LogOp, "N_block_tiles: {}", N_block_tiles);
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

    std::map<std::string, std::string> defines;
    std::map<std::string, std::string> in0_injector_defines;
    if (use_bias) {
        defines["FUSE_BIAS"] = "1";
    }

    uint32_t in0_addr = input_tensor.buffer()->address();
    uint32_t in1_addr = weight_tensor.buffer()->address();
    uint32_t in2_addr = use_bias ? bias_tensor.value().buffer()->address() : 0;
    // Note: Dataflow kernels can take a variable number of output tensors.
    // They are appended as a variable-length array at the end of the runtime-args:
    //   - for in0 output-writer cores the first output address is at index 13
    //   - for in1 output-writer cores the first output address is at index 12

    /**
     * Create kernels
     */

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
        M_blocks_per_core,
        N_blocks_per_core,
        in0_tile_size,
        out_tile_size,
        in2_tile_size,
    };

    auto in_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/fast_matmul/device/kernels/dm_in.cpp",
        core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::RISCV_0_default,
            .compile_args = in0_sender_compile_time_args});

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
        M_blocks_per_core,
        N_blocks_per_core,
        in0_tile_size,
        out_tile_size,
        in2_tile_size,
    };
    append_accessors(in0_receiver_compile_time_args, input_tensor, output_tensors, bias_tensor);

    auto out_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/fast_matmul/device/kernels/dm_out.cpp",
        core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::RISCV_1_default,
            .compile_args = in0_receiver_compile_time_args,
            .defines = defines});

    std::vector<uint32_t> compute_compile_time_args = {
        K_blocks,
        M_block_tiles,
        K_block_tiles,
        N_block_tiles,
        M_blocks_per_core,
        N_blocks_per_core,
        subblock_h,
        subblock_w};

    auto compute_defines = defines;
    std::map<std::string, std::string> compute_activation_defines;
    if (fused_activation.has_value()) {
        compute_activation_defines = ttnn::operations::unary::utils::get_defines(
            fused_activation.value().op_type,
            fused_activation.value().params,
            "ACTIVATION",
            "fused_act_dst_id",
            output_tensors[0].dtype());
    }
    compute_defines.merge(compute_activation_defines);
    auto compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/fast_matmul/device/kernels/compute.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
            .defines = compute_defines});

    auto cores = corerange_to_cores(core_grid, num_cores, true);

    // NOTE: Uniform per-core M/N ranges are required for DM forward handshakes to match across links.
    // If neighboring cores along a forwarding chain iterate different (M,N) counts, the sender can wait
    // for requests that the receiver will never issue, leading to deadlock. Keep the original uniform
    // div_up-based ranges for M and N.

    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        CoreCoord core = cores.at(core_id);
        uint32_t in0_idx = core_id;
        uint32_t in1_idx = 0;

        /**
         * NOTE: Some cores are doing unnecessary work, on blocks which are processed just to make
         * the total number of blocks divisible by the number of cores.
         * We can't yet get rid of these blocks, since the receiver cores must ack
         * all blocks that sender cores are expected to send.
         */
        uint32_t M_start_tile = M_tiles_per_core * in0_idx;
        uint32_t M_end_tile = M_tiles_per_core * (in0_idx + 1);
        uint32_t N_start_tile = N_tiles_per_core * in1_idx;
        uint32_t N_end_tile = N_tiles_per_core * (in1_idx + 1);

        // log_info(tt::LogOp, "core_id: {}, M_start_tile: {}, M_end_tile: {}, N_start_tile: {}, N_end_tile: {}",
        // core_id, M_start_tile, M_end_tile, N_start_tile, N_end_tile);

        // Defer write to K block with same coordinate as core
        // The writer receiver cores always have core.x > 0

        std::vector<uint32_t> in0_args = {
            in0_addr,
            in2_addr,
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
        };
        // Add output addresses at the end (unified layout for both regular and split)
        for (const auto& output_tensor : output_tensors) {
            in0_args.push_back(output_tensor.buffer()->address());
        }

        SetRuntimeArgs(program, in_kernel_id, core, in0_args);

        std::vector<uint32_t> in1_args = {
            in1_addr,
            in2_addr,
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
        };
        // Add output addresses at the end (unified layout for both regular and split)
        for (const auto& output_tensor : output_tensors) {
            in1_args.push_back(output_tensor.buffer()->address());
        }

        SetRuntimeArgs(program, out_kernel_id, core, in1_args);

        std::vector<uint32_t> compute_runtime_args = {
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
        };
        SetRuntimeArgs(program, compute_kernels_id, core, compute_runtime_args);
    }

    return FastMatmulProgramFactory::shared_variables_t{
        .cbs = std::array<tt::tt_metal::CBHandle, 3>{in_cb_handle, weight_cb_handle, out_cb_handle}};
}

// Legacy wrapper for single output tensor (backward compatibility)
FastMatmulProgramFactory::shared_variables_t fast_matmul_factory_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<operations::unary::UnaryWithParam>& fused_activation,
    const std::optional<const FastMatmulConfig>& config,
    const Tensor& output_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    // Wrap single output in vector and call shared implementation
    std::vector<Tensor> output_tensors = {output_tensor};
    return fast_matmul_factory_helper_common(
        program,
        input_tensor,
        weight_tensor,
        bias_tensor,
        fused_activation,
        config,
        output_tensors,
        compute_kernel_config,
        1  // N_chunks = 1 for regular fast_matmul
    );
}

FastMatmulProgramFactory::cached_program_t FastMatmulProgramFactory::create(
    const FastMatmulParams& operation_attributes, const FastMatmulInputs& tensor_args, Tensor& tensor_return_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    auto shared_vars = fast_matmul_factory_helper(
        program,
        tensor_args.input_tensor,
        tensor_args.weight_tensor,
        tensor_args.bias_tensor,
        operation_attributes.fused_activation,
        operation_attributes.config,
        tensor_return_value,
        operation_attributes.compute_kernel_config);

    return {std::move(program), std::move(shared_vars)};
}

// Common helper for override_runtime_arguments - works with both single and multiple output tensors
void override_runtime_arguments_common(
    FastMatmulProgramFactory::cached_program_t& cached_program,
    const Buffer& in0_buffer,
    const Buffer& in1_buffer,
    const Buffer& output_buffer) {
    auto& program = cached_program.program;
    auto& override_variables = cached_program.shared_variables;

    UpdateDynamicCircularBufferAddress(program, override_variables.cbs[0], in0_buffer);
    UpdateDynamicCircularBufferAddress(program, override_variables.cbs[1], in1_buffer);
    UpdateDynamicCircularBufferAddress(program, override_variables.cbs[2], output_buffer);
}

void FastMatmulProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const FastMatmulParams& /*operation_attributes*/,
    const FastMatmulInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto in0_buffer = tensor_args.input_tensor.buffer();
    auto in1_buffer = tensor_args.weight_tensor.buffer();

    auto out_buffer = tensor_return_value.buffer();
    override_runtime_arguments_common(cached_program, *in0_buffer, *in1_buffer, *out_buffer);
}

}  // namespace ttnn::experimental::prim
