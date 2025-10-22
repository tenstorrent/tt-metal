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
#include <utility>
#include <vector>

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

// Build a linear order of cores along one axis for data movement, plus index of the current core
static inline std::pair<std::vector<CoreCoord>, uint32_t> build_core_order_for_axis(
    const CoreCoord& core,
    bool transpose_core_grid,
    uint32_t axis_length,
    tt::tt_metal::NOC noc,
    bool axis_is_x_when_not_transposed,
    const CoreCoord& initial_endpoint) {
    std::vector<CoreCoord> order;
    order.reserve(axis_length);
    order.push_back(initial_endpoint);

    // Determine which coordinate of the current core defines its position along this axis
    const size_t current_axis_value = transpose_core_grid ? (axis_is_x_when_not_transposed ? core.y : core.x)
                                                          : (axis_is_x_when_not_transposed ? core.x : core.y);

    // Direction along the axis: increasing for NOC_0, decreasing for NOC_1
    const bool increasing = (noc == tt::tt_metal::NOC::NOC_0);

    uint32_t index_of_current = 0;  // default to 0 if axis_length == 1
    for (uint32_t worker_idx = 1; worker_idx < axis_length; ++worker_idx) {
        CoreCoord worker_core = core;
        size_t& coord_to_modify = transpose_core_grid ? (axis_is_x_when_not_transposed ? worker_core.y : worker_core.x)
                                                      : (axis_is_x_when_not_transposed ? worker_core.x : worker_core.y);

        coord_to_modify = increasing ? worker_idx : (axis_length - worker_idx);
        if (coord_to_modify == current_axis_value) {
            index_of_current = worker_idx;
        }
        order.push_back(worker_core);
    }
    return {order, index_of_current};
}

static inline CoreCoord clamped_prev(const std::vector<CoreCoord>& order, uint32_t index) {
    return order.at(index == 0 ? 0 : index - 1);
}

static inline CoreCoord clamped_next(const std::vector<CoreCoord>& order, uint32_t index) {
    const uint32_t last = static_cast<uint32_t>(order.size() - 1);
    return order.at(index >= last ? last : index + 1);
}

// Append tensor accessors in a consistent order
static inline void append_accessors(
    std::vector<uint32_t>& args,
    const Tensor& main_tensor,
    const Tensor& output_tensor,
    const std::optional<const Tensor>& bias_tensor) {
    tt::tt_metal::TensorAccessorArgs(*main_tensor.buffer()).append_to(args);
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(args);
    if (bias_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*bias_tensor.value().buffer()).append_to(args);
    }
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

    uint32_t in0_addr = input_tensor.buffer()->address();
    uint32_t in1_addr = weight_tensor.buffer()->address();
    uint32_t in2_addr = use_bias ? bias_tensor.value().buffer()->address() : 0;
    uint32_t out_addr = output_tensor.buffer()->address();

    std::map<std::string, std::string> defines;
    if (use_bias) {
        defines["FUSE_BIAS"] = "1";
    }

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
    uint32_t M = in0_tensor_shape[-2];
    uint32_t K = in0_tensor_shape[-1];
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

    // Aspect-ratio based partitioning for highly non-square outputs
    // We partition along the longer output axis into multiple independent sub-grids.
    const uint32_t min_tiles = std::max(1u, std::min(M_tiles, N_tiles));
    const uint32_t max_tiles = std::max(M_tiles, N_tiles);
    const double aspect_ratio = static_cast<double>(max_tiles) / static_cast<double>(min_tiles);
    const bool output_is_wide = (N_tiles >= M_tiles);
    const bool should_partition = aspect_ratio > 2.0;

    if (should_partition) {
        // Decide number of partitions along the grid dimension that parallels the shorter output axis:
        // - If output is wide (M < N), we partition along Y (cores.y)
        // - If output is tall (N < M), we partition along X (cores.x)
        const uint32_t grid_dim_partition = output_is_wide ? grid_size.y : grid_size.x;
        const uint32_t ar_floor = static_cast<uint32_t>(aspect_ratio);
        const uint32_t upper_bound = std::min(grid_dim_partition, ar_floor);
        uint32_t partitions = 1;
        for (uint32_t candidate = upper_bound; candidate >= 1; --candidate) {
            if ((grid_dim_partition % candidate) == 0u) {
                partitions = candidate;
                break;
            }
        }

        log_info(
            tt::LogOp,
            "Partitioning enabled: aspect_ratio: {} partitions: {} split_dim: {}",
            aspect_ratio,
            partitions,
            output_is_wide ? "Y" : "X");

        // Create per-partition kernels and state
        std::vector<tt::tt_metal::KernelHandle> in0_sender_kernels_ids;
        std::vector<tt::tt_metal::KernelHandle> in0_receiver_kernels_ids;
        std::vector<tt::tt_metal::KernelHandle> in1_sender_kernels_ids;
        std::vector<tt::tt_metal::KernelHandle> in1_receiver_kernels_ids;
        std::vector<tt::tt_metal::KernelHandle> compute_kernels_ids;
        std::vector<std::vector<CoreCoord>> partition_cores;
        std::vector<CoreCoord> partition_starts;
        std::vector<uint32_t> partition_local_grid_size_x;
        std::vector<uint32_t> partition_local_grid_size_y;
        std::vector<bool> in0_receiver_exists;
        std::vector<bool> in1_receiver_exists;

        in0_sender_kernels_ids.reserve(partitions);
        in0_receiver_kernels_ids.reserve(partitions);
        in1_sender_kernels_ids.reserve(partitions);
        in1_receiver_kernels_ids.reserve(partitions);
        compute_kernels_ids.reserve(partitions);
        partition_cores.reserve(partitions);

        // Split the longer axis tiles across partitions (ceil)
        const uint32_t longer_tiles = output_is_wide ? N_tiles : M_tiles;
        // Ensure that the longer tiles divides by the number of partitions and the grid size of the axis that is not
        // partitioned
        const uint32_t padded_longer_tiles =
            tt::round_up(longer_tiles, partitions * (output_is_wide ? grid_size.x : grid_size.y));
        const uint32_t tiles_per_partition = tt::div_up(padded_longer_tiles, partitions);

        for (uint32_t p = 0; p < partitions; ++p) {
            // Local core grid for this partition:
            // - Wide (M < N): slice Y into equal stripes -> grid_x = grid_size.x, grid_y = grid_size.y / partitions
            // - Tall (N < M): slice X into equal stripes -> grid_x = grid_size.x / partitions, grid_y = grid_size.y
            const uint32_t local_grid_size_y = output_is_wide ? (grid_size.y / partitions) : grid_size.y;
            const uint32_t local_grid_size_x = output_is_wide ? grid_size.x : (grid_size.x / partitions);
            const uint32_t y0 = output_is_wide ? (p * local_grid_size_y) : 0u;
            const uint32_t x0 = output_is_wide ? 0u : (p * local_grid_size_x);
            CoreCoord start = CoreCoord{x0, y0};
            CoreCoord end = CoreCoord{x0 + local_grid_size_x - 1, y0 + local_grid_size_y - 1};
            auto partition_core_grid = CoreRange(start, end);
            auto local_num_cores = partition_core_grid.size();

            // Transpose decision follows global rule: transpose when M > N (tall)
            const bool local_transpose_core_grid = (M > N);

            // Local parallel axis core counts (following original mapping rules on the local grid)
            const uint32_t local_in0_parallel_axis_cores =
                local_transpose_core_grid ? local_grid_size_x : local_grid_size_y;
            const uint32_t local_in1_parallel_axis_cores =
                local_transpose_core_grid ? local_grid_size_y : local_grid_size_x;

            // Partition slice on the longer axis
            const uint32_t part_start_tile = p * tiles_per_partition;
            const uint32_t part_end_tile = (p + 1) * tiles_per_partition;
            const uint32_t part_len_tiles = part_end_tile - part_start_tile;

            // Local logical M/N in tiles for this partition
            const uint32_t local_M_tiles = output_is_wide ? M_tiles : part_len_tiles;
            const uint32_t local_N_tiles = output_is_wide ? part_len_tiles : N_tiles;

            // Local paddings and per-core ranges
            const uint32_t local_M_tiles_per_core = tt::div_up(local_M_tiles, local_in0_parallel_axis_cores);
            const uint32_t local_N_tiles_per_core = tt::div_up(local_N_tiles, local_in1_parallel_axis_cores);

            const uint32_t padded_K_tiles = tt::round_up(K_tiles, K_block_tiles);
            const uint32_t K_blocks = padded_K_tiles / K_block_tiles;
            const uint32_t M_blocks_per_core = tt::div_up(local_M_tiles_per_core, M_block_tiles);
            const uint32_t N_blocks_per_core = tt::div_up(local_N_tiles_per_core, N_block_tiles);

            // Assign NOCs/RISCs per local transpose
            auto small_input_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
            auto small_input_risc = tt::tt_metal::DataMovementProcessor::RISCV_1;
            auto large_input_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
            auto large_input_risc = tt::tt_metal::DataMovementProcessor::RISCV_0;

            auto in0_noc = local_transpose_core_grid ? large_input_noc : small_input_noc;
            auto in0_risc = local_transpose_core_grid ? large_input_risc : small_input_risc;
            auto in1_noc = local_transpose_core_grid ? small_input_noc : large_input_noc;
            auto in1_risc = local_transpose_core_grid ? small_input_risc : large_input_risc;

            // CB sizes (same formulae as original but local per-core counts)
            uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
            uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
            uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;
            uint32_t in2_block_num_tiles = N_block_tiles;

            const uint32_t double_buffer_factor = 2;
            uint32_t in0_cb_num_tiles = in0_block_num_tiles * double_buffer_factor;
            uint32_t in1_cb_num_tiles = in1_block_num_tiles * double_buffer_factor;
            uint32_t out_cb_num_tiles = out_block_num_tiles * double_buffer_factor;
            uint32_t interm_cb_num_tiles = out_block_num_tiles;
            uint32_t in2_cb_num_tiles = in2_block_num_tiles;

            // Semaphores per partition
            auto in0_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, partition_core_grid, INVALID);
            auto in0_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, partition_core_grid, INVALID);
            auto in0_valid_semaphore_id = tt::tt_metal::CreateSemaphore(program, partition_core_grid, VALID);
            auto in1_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, partition_core_grid, INVALID);
            auto in1_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, partition_core_grid, INVALID);
            auto in1_valid_semaphore_id = tt::tt_metal::CreateSemaphore(program, partition_core_grid, VALID);

            // CBs per partition
            uint32_t in0_cb_id = tt::CBIndex::c_0;
            tt::tt_metal::create_cb(
                in0_cb_id, program, partition_core_grid, in0_tile_size, in0_cb_num_tiles, in0_data_format);
            uint32_t in1_cb_id = tt::CBIndex::c_1;
            tt::tt_metal::create_cb(
                in1_cb_id, program, partition_core_grid, in1_tile_size, in1_cb_num_tiles, in1_data_format);
            uint32_t out_cb_id = tt::CBIndex::c_2;
            tt::tt_metal::create_cb(
                out_cb_id, program, partition_core_grid, out_tile_size, out_cb_num_tiles, output_data_format);
            uint32_t intermediate_cb_id = tt::CBIndex::c_3;
            tt::tt_metal::create_cb(
                intermediate_cb_id,
                program,
                partition_core_grid,
                intermediate_tile_size,
                interm_cb_num_tiles,
                intermediate_data_format);
            if (use_bias) {
                uint32_t in2_cb_id = tt::CBIndex::c_4;
                tt::tt_metal::create_cb(
                    in2_cb_id, program, partition_core_grid, in2_tile_size, in2_cb_num_tiles, in2_data_format);
            }

            // Compile-time args for DM kernels (local sizes and counts)
            bool in0_is_output_writer = !local_transpose_core_grid;
            bool in1_is_output_writer = local_transpose_core_grid;
            log_info(tt::LogOp, "in0_is_output_writer: {}", in0_is_output_writer);
            log_info(tt::LogOp, "in1_is_output_writer: {}", in1_is_output_writer);

            std::vector<uint32_t> in0_sender_compile_time_args = {
                M_tiles,
                M_tiles,
                K_tiles,
                padded_K_tiles,
                N_tiles,
                N_tiles,
                M_block_tiles,
                K_block_tiles,
                N_block_tiles,
                M_blocks_per_core,
                N_blocks_per_core,
                in0_tile_size,
                out_tile_size,
                in2_tile_size,
                in0_sender_semaphore_id,
                in0_receiver_semaphore_id,
                in0_valid_semaphore_id,
                in0_is_output_writer,
                true};
            append_accessors(in0_sender_compile_time_args, input_tensor, output_tensor, bias_tensor);

            // Build local sender/receiver core ranges within this partition
            CoreCoord local_core_0_0 = start;
            CoreCoord local_core_endx_0 = CoreCoord{start.x + local_grid_size_x - 1, start.y};
            CoreCoord local_core_0_endy = CoreCoord{start.x, start.y + local_grid_size_y - 1};
            CoreCoord local_core_1_0 = CoreCoord{start.x + (local_grid_size_x > 1 ? 1 : 0), start.y};
            CoreCoord local_core_0_1 = CoreCoord{start.x, start.y + (local_grid_size_y > 1 ? 1 : 0)};
            CoreCoord local_core_endx_endy = end;

            auto in0_sender_cores =
                CoreRange(local_core_0_0, local_transpose_core_grid ? local_core_endx_0 : local_core_0_endy);
            auto in0_receiver_cores =
                CoreRange(local_transpose_core_grid ? local_core_0_1 : local_core_1_0, local_core_endx_endy);

            log_info(tt::LogOp, "Creating in0 sender kernel with cores: {}", in0_sender_cores);
            auto in0_sender_kernels_id = CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp",
                in0_sender_cores,
                tt::tt_metal::DataMovementConfig{
                    .processor = in0_risc,
                    .noc = in0_noc,
                    .compile_args = in0_sender_compile_time_args,
                    .defines = defines});

            std::vector<uint32_t> in0_receiver_compile_time_args = in0_sender_compile_time_args;
            in0_receiver_compile_time_args[18] = false;  // is_injector_core = false
            // Conditionally create in0 receiver kernel only if needed
            // in0 forwards along X when transposed, else along Y
            bool create_in0_receiver = local_transpose_core_grid ? (local_grid_size_y > 1) : (local_grid_size_x > 1);
            tt::tt_metal::KernelHandle in0_receiver_kernels_id = 0;
            if (create_in0_receiver) {
                log_info(tt::LogOp, "Creating in0 receiver kernel with cores: {}", in0_receiver_cores);
                in0_receiver_kernels_id = CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp",
                    in0_receiver_cores,
                    tt::tt_metal::DataMovementConfig{
                        .processor = in0_risc,
                        .noc = in0_noc,
                        .compile_args = in0_receiver_compile_time_args,
                        .defines = defines});
            }

            std::vector<uint32_t> in1_sender_compile_time_args = {
                M_tiles,
                M_tiles,
                K_tiles,
                padded_K_tiles,
                N_tiles,
                N_tiles,
                M_block_tiles,
                K_block_tiles,
                N_block_tiles,
                M_blocks_per_core,
                N_blocks_per_core,
                in1_tile_size,
                out_tile_size,
                in2_tile_size,
                in1_sender_semaphore_id,
                in1_receiver_semaphore_id,
                in1_valid_semaphore_id,
                in1_is_output_writer,
                true};
            append_accessors(in1_sender_compile_time_args, weight_tensor, output_tensor, bias_tensor);
            log_info(
                tt::LogOp,
                "Creating in1 sender kernel with cores: {}",
                CoreRange(local_core_0_0, local_transpose_core_grid ? local_core_0_endy : local_core_endx_0));
            auto in1_sender_kernels_id = CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_sender_out.cpp",
                CoreRange(local_core_0_0, local_transpose_core_grid ? local_core_0_endy : local_core_endx_0),
                tt::tt_metal::DataMovementConfig{
                    .processor = in1_risc,
                    .noc = in1_noc,
                    .compile_args = in1_sender_compile_time_args,
                    .defines = defines});

            std::vector<uint32_t> in1_receiver_compile_time_args = in1_sender_compile_time_args;
            in1_receiver_compile_time_args[18] = false;  // is_injector_core = false
            // Conditionally create in1 receiver kernel only if needed
            // in1 forwards along Y when transposed, else along X
            bool create_in1_receiver = local_transpose_core_grid ? (local_grid_size_x > 1) : (local_grid_size_y > 1);
            tt::tt_metal::KernelHandle in1_receiver_kernels_id = 0;
            if (create_in1_receiver) {
                log_info(
                    tt::LogOp,
                    "Creating in1 receiver kernel with cores: {}",
                    CoreRange(local_transpose_core_grid ? local_core_1_0 : local_core_0_1, local_core_endx_endy));
                in1_receiver_kernels_id = CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_sender_out.cpp",
                    CoreRange(local_transpose_core_grid ? local_core_1_0 : local_core_0_1, local_core_endx_endy),
                    tt::tt_metal::DataMovementConfig{
                        .processor = in1_risc,
                        .noc = in1_noc,
                        .compile_args = in1_receiver_compile_time_args,
                        .defines = defines});
            }

            // Compute kernel per partition
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
                    output_tensor.dtype());
            }
            compute_defines.merge(compute_activation_defines);
            auto compute_kernels_id = CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp",
                partition_core_grid,
                tt::tt_metal::ComputeConfig{
                    .math_fidelity = math_fidelity,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .math_approx_mode = math_approx_mode,
                    .compile_args = compute_compile_time_args,
                    .defines = compute_defines});

            // Save kernel ids
            in0_sender_kernels_ids.push_back(in0_sender_kernels_id);
            in0_receiver_kernels_ids.push_back(in0_receiver_kernels_id);
            in1_sender_kernels_ids.push_back(in1_sender_kernels_id);
            in1_receiver_kernels_ids.push_back(in1_receiver_kernels_id);
            compute_kernels_ids.push_back(compute_kernels_id);
            in0_receiver_exists.push_back(create_in0_receiver);
            in1_receiver_exists.push_back(create_in1_receiver);

            // Build per-partition runtime args
            auto cores = corerange_to_cores(partition_core_grid, local_num_cores, true);
            partition_cores.push_back(cores);
            partition_starts.push_back(start);
            partition_local_grid_size_x.push_back(local_grid_size_x);
            partition_local_grid_size_y.push_back(local_grid_size_y);

            // K-block write deferral based on local parallel axis
            uint32_t k_blocks_per_core = tt::div_up(
                K_blocks, (local_transpose_core_grid ? local_in1_parallel_axis_cores : local_in0_parallel_axis_cores));

            // Endpoints for order building relative to local grid
            // No-op placeholders preserved intentionally for symmetry with non-partitioned path

            for (uint32_t core_id = 0; core_id < local_num_cores; ++core_id) {
                CoreCoord core = cores.at(core_id);
                CoreCoord left_core = {start.x, core.y};
                CoreCoord top_core = {core.x, start.y};

                auto [in0_core_order, in0_core_order_index] = build_core_order_for_axis(
                    core,
                    local_transpose_core_grid,
                    local_in1_parallel_axis_cores,
                    in0_noc,
                    /*axis_is_x_when_not_transposed=*/true,
                    /*initial_endpoint=*/(local_transpose_core_grid ? top_core : left_core));

                auto [in1_core_order, in1_core_order_index] = build_core_order_for_axis(
                    core,
                    local_transpose_core_grid,
                    local_in0_parallel_axis_cores,
                    in1_noc,
                    /*axis_is_x_when_not_transposed=*/false,
                    /*initial_endpoint=*/(local_transpose_core_grid ? left_core : top_core));

                auto in0_prev_core = clamped_prev(in0_core_order, in0_core_order_index);
                auto in0_next_core = clamped_next(in0_core_order, in0_core_order_index);
                auto in1_prev_core = clamped_prev(in1_core_order, in1_core_order_index);
                auto in1_next_core = clamped_next(in1_core_order, in1_core_order_index);

                log_info(
                    tt::LogOp,
                    "in0_core_order_index: {}, in1_core_order_index: {}",
                    in0_core_order_index,
                    in1_core_order_index);

                auto in0_prev_core_physical = device->worker_core_from_logical_core(in0_prev_core);
                auto in0_next_core_physical = device->worker_core_from_logical_core(in0_next_core);
                auto in1_prev_core_physical = device->worker_core_from_logical_core(in1_prev_core);
                auto in1_next_core_physical = device->worker_core_from_logical_core(in1_next_core);

                // Per-core M/N ranges with global partition offsets applied on the longer axis
                // uint32_t M_start_tile = local_M_tiles_per_core * in0_idx + (output_is_wide ? 0u : part_start_tile);
                // uint32_t M_end_tile = local_M_tiles_per_core * (in0_idx + 1) + (output_is_wide ? 0u :
                // part_start_tile); uint32_t N_start_tile = local_N_tiles_per_core * in1_idx + (output_is_wide ?
                // part_start_tile : 0u); uint32_t N_end_tile = local_N_tiles_per_core * (in1_idx + 1) + (output_is_wide
                // ? part_start_tile : 0u); DEBUG
                uint32_t relative_index_in0 = in1_core_order_index;  // This is the index of the core on the core_grid
                                                                     // axis that in0 is parallel over
                uint32_t relative_index_in1 = in0_core_order_index;  // This is the index of the core on the core_grid
                                                                     // axis that in1 is parallel over
                uint32_t M_start_tile =
                    relative_index_in0 * local_M_tiles_per_core + (output_is_wide ? 0u : part_start_tile);
                uint32_t M_end_tile = M_start_tile + local_M_tiles_per_core;
                uint32_t N_start_tile =
                    relative_index_in1 * local_N_tiles_per_core + (output_is_wide ? part_start_tile : 0u);
                uint32_t N_end_tile = N_start_tile + local_N_tiles_per_core;

                uint32_t defer_write_k_block = (local_transpose_core_grid ? core.x : core.y) * k_blocks_per_core;
                defer_write_k_block = std::min(defer_write_k_block, K_blocks - 1);

                bool is_in0_sink = core == in0_core_order.back();
                bool is_in1_sink = core == in1_core_order.back();
                // log_info(tt::LogOp, "in0_core_order: {}", in0_core_order);
                // log_info(tt::LogOp, "in1_core_order: {}", in1_core_order);

                std::vector<uint32_t> in0_args = {
                    in0_addr,
                    out_addr,
                    in2_addr,
                    is_in0_sink,
                    (std::uint32_t)in0_next_core_physical.x,
                    (std::uint32_t)in0_next_core_physical.y,
                    (std::uint32_t)in0_prev_core_physical.x,
                    (std::uint32_t)in0_prev_core_physical.y,
                    M_start_tile,
                    M_end_tile,
                    N_start_tile,
                    N_end_tile,
                    defer_write_k_block,
                };
                const bool is_in0_sender =
                    local_transpose_core_grid ? core.y == local_core_0_0.y : core.x == local_core_0_0.x;
                if (is_in0_sender) {
                    SetRuntimeArgs(program, in0_sender_kernels_id, core, in0_args);
                } else {
                    SetRuntimeArgs(program, in0_receiver_kernels_id, core, in0_args);
                }

                std::vector<uint32_t> in1_args = {
                    in1_addr,
                    out_addr,
                    in2_addr,
                    is_in1_sink,
                    (std::uint32_t)in1_next_core_physical.x,
                    (std::uint32_t)in1_next_core_physical.y,
                    (std::uint32_t)in1_prev_core_physical.x,
                    (std::uint32_t)in1_prev_core_physical.y,
                    M_start_tile,
                    M_end_tile,
                    N_start_tile,
                    N_end_tile,
                    defer_write_k_block,
                };
                const bool is_in1_sender =
                    local_transpose_core_grid ? core.x == local_core_0_0.x : core.y == local_core_0_0.y;
                if (is_in1_sender) {
                    SetRuntimeArgs(program, in1_sender_kernels_id, core, in1_args);
                } else {
                    SetRuntimeArgs(program, in1_receiver_kernels_id, core, in1_args);
                }

                log_info(
                    tt::LogOp,
                    "Setting runtime args for core: {} in0_sender: {} in1_sender: {} in0_sink: {} in1_sink: {}",
                    core,
                    is_in0_sender,
                    is_in1_sender,
                    is_in0_sink,
                    is_in1_sink);

                log_info(
                    tt::LogOp,
                    "M_start_tile: {}, M_end_tile: {}, N_start_tile: {}, N_end_tile: {}",
                    M_start_tile,
                    M_end_tile,
                    N_start_tile,
                    N_end_tile);

                std::vector<uint32_t> compute_runtime_args = {
                    M_start_tile,
                    M_end_tile,
                    N_start_tile,
                    N_end_tile,
                };
                SetRuntimeArgs(program, compute_kernels_id, core, compute_runtime_args);
            }
        }

        const bool partitions_transpose = (M > N);
        auto override_runtime_arguments_callback =
            [partitions,
             partition_cores,
             partition_starts,
             partition_local_grid_size_x,
             partition_local_grid_size_y,
             in0_sender_kernels_ids,
             in0_receiver_kernels_ids,
             in1_sender_kernels_ids,
             in1_receiver_kernels_ids,
             in0_receiver_exists,
             in1_receiver_exists,
             partitions_transpose](
                const void* operation,
                tt::tt_metal::Program& program,
                const std::vector<Tensor>& input_tensors,
                const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                const std::vector<Tensor>& output_tensors) {
                auto in0_addr = input_tensors.at(0).buffer()->address();
                auto in1_addr = input_tensors.at(1).buffer()->address();
                auto output_addr = output_tensors.at(0).buffer()->address();
                auto in2_addr = optional_input_tensors.at(0).has_value()
                                    ? optional_input_tensors.at(0).value().buffer()->address()
                                    : 0;

                for (uint32_t p = 0; p < partitions; ++p) {
                    auto& in0_sender_runtime_args = GetRuntimeArgs(program, in0_sender_kernels_ids[p]);
                    auto& in1_sender_runtime_args = GetRuntimeArgs(program, in1_sender_kernels_ids[p]);

                    const auto& cores = partition_cores[p];
                    const CoreCoord start = partition_starts[p];
                    for (const auto& core : cores) {
                        // Compute local indices within the partition to decide sender vs receiver
                        uint32_t local_x = core.x - start.x;
                        uint32_t local_y = core.y - start.y;
                        uint32_t in0_idx = partitions_transpose ? local_x : local_y;
                        uint32_t in1_idx = partitions_transpose ? local_y : local_x;

                        if (in1_idx == 0) {
                            auto& args = in0_sender_runtime_args[core.x][core.y];
                            args[0] = in0_addr;
                            args[1] = output_addr;
                            args[2] = in2_addr;
                        } else if (in0_receiver_exists[p]) {
                            auto& in0_receiver_runtime_args = GetRuntimeArgs(program, in0_receiver_kernels_ids[p]);
                            auto& args = in0_receiver_runtime_args[core.x][core.y];
                            args[1] = output_addr;
                            args[2] = in2_addr;
                        }

                        if (in0_idx == 0) {
                            auto& args = in1_sender_runtime_args[core.x][core.y];
                            args[0] = in1_addr;
                            args[1] = output_addr;
                            args[2] = in2_addr;
                        } else if (in1_receiver_exists[p]) {
                            auto& in1_receiver_runtime_args = GetRuntimeArgs(program, in1_receiver_kernels_ids[p]);
                            auto& args = in1_receiver_runtime_args[core.x][core.y];
                            args[1] = output_addr;
                            args[2] = in2_addr;
                        }
                    }
                }
            };
        return {
            .program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
    }

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

    uint32_t K_blocks = padded_K_tiles / K_block_tiles;

    uint32_t M_blocks_per_core = tt::div_up(M_tiles_per_core, M_block_tiles);
    uint32_t N_blocks_per_core = tt::div_up(N_tiles_per_core, N_block_tiles);

    log_debug(tt::LogOp, "M_tiles_per_core: {}", M_tiles_per_core);
    log_debug(tt::LogOp, "N_tiles_per_core: {}", N_tiles_per_core);
    log_debug(tt::LogOp, "M_blocks_per_core: {}", M_blocks_per_core);
    log_debug(tt::LogOp, "N_blocks_per_core: {}", N_blocks_per_core);

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
        M_blocks_per_core,
        N_blocks_per_core,
        in0_tile_size,
        out_tile_size,
        in2_tile_size,
        in0_sender_semaphore_id,
        in0_receiver_semaphore_id,
        in0_valid_semaphore_id,
        in0_is_output_writer,
        true  // is_injector_core
    };
    append_accessors(in0_sender_compile_time_args, input_tensor, output_tensor, bias_tensor);

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
        M_blocks_per_core,
        N_blocks_per_core,
        in0_tile_size,
        out_tile_size,
        in2_tile_size,
        in0_sender_semaphore_id,
        in0_receiver_semaphore_id,
        in0_valid_semaphore_id,
        in0_is_output_writer,
        false  // is_injector_core
    };
    append_accessors(in0_receiver_compile_time_args, input_tensor, output_tensor, bias_tensor);

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
        M_blocks_per_core,
        N_blocks_per_core,
        in1_tile_size,
        out_tile_size,
        in2_tile_size,
        in1_sender_semaphore_id,
        in1_receiver_semaphore_id,
        in1_valid_semaphore_id,
        in1_is_output_writer,
        true  // is_injector_core
    };
    append_accessors(in1_sender_compile_time_args, weight_tensor, output_tensor, bias_tensor);
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
        M_blocks_per_core,
        N_blocks_per_core,
        in1_tile_size,
        out_tile_size,
        in2_tile_size,
        in1_sender_semaphore_id,
        in1_receiver_semaphore_id,
        in1_valid_semaphore_id,
        in1_is_output_writer,
        false  // is_injector_core
    };
    append_accessors(in1_receiver_compile_time_args, weight_tensor, output_tensor, bias_tensor);
    auto in1_receiver_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_sender_out.cpp",
        in1_receiver_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = in1_risc, .noc = in1_noc, .compile_args = in1_receiver_compile_time_args, .defines = defines});

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

    // NOTE: Uniform per-core M/N ranges are required for DM forward handshakes to match across links.
    // If neighboring cores along a forwarding chain iterate different (M,N) counts, the sender can wait
    // for requests that the receiver will never issue, leading to deadlock. Keep the original uniform
    // div_up-based ranges for M and N.

    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        CoreCoord core = cores.at(core_id);
        uint32_t in0_idx = transpose_core_grid ? core.x : core.y;
        uint32_t in1_idx = transpose_core_grid ? core.y : core.x;

        CoreCoord left_core = {(std::size_t)0, (std::size_t)core.y};
        CoreCoord top_core = {(std::size_t)core.x, (std::size_t)0};

        auto [in0_core_order, in0_core_order_index] = build_core_order_for_axis(
            core,
            transpose_core_grid,
            in1_parallel_axis_cores,
            in0_noc,
            /*axis_is_x_when_not_transposed=*/true,
            /*initial_endpoint=*/(transpose_core_grid ? top_core : left_core));

        auto [in1_core_order, in1_core_order_index] = build_core_order_for_axis(
            core,
            transpose_core_grid,
            in0_parallel_axis_cores,
            in1_noc,
            /*axis_is_x_when_not_transposed=*/false,
            /*initial_endpoint=*/(transpose_core_grid ? left_core : top_core));

        auto in0_prev_core = clamped_prev(in0_core_order, in0_core_order_index);
        auto in0_next_core = clamped_next(in0_core_order, in0_core_order_index);
        auto in1_prev_core = clamped_prev(in1_core_order, in1_core_order_index);
        auto in1_next_core = clamped_next(in1_core_order, in1_core_order_index);

        auto in0_prev_core_physical = device->worker_core_from_logical_core(in0_prev_core);
        auto in0_next_core_physical = device->worker_core_from_logical_core(in0_next_core);
        auto in1_prev_core_physical = device->worker_core_from_logical_core(in1_prev_core);
        auto in1_next_core_physical = device->worker_core_from_logical_core(in1_next_core);

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
        uint32_t defer_write_k_block = core.y * k_blocks_per_core;
        defer_write_k_block = std::min(defer_write_k_block, K_blocks - 1);

        bool is_in0_sink = core == in0_core_order.back();
        bool is_in1_sink = core == in1_core_order.back();

        std::vector<uint32_t> in0_args = {
            in0_addr,
            out_addr,
            in2_addr,
            is_in0_sink,
            (std::uint32_t)in0_next_core_physical.x,  // in0_dest_noc_x
            (std::uint32_t)in0_next_core_physical.y,  // in0_dest_noc_y
            (std::uint32_t)in0_prev_core_physical.x,  // in0_sender_noc_x
            (std::uint32_t)in0_prev_core_physical.y,  // in0_sender_noc_y
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
            defer_write_k_block,
        };

        if (in1_idx == 0) {
            // in0 sender
            SetRuntimeArgs(program, in0_sender_kernels_id, core, in0_args);
        } else {
            // in0 receiver
            SetRuntimeArgs(program, in0_receiver_kernels_id, core, in0_args);
        }

        std::vector<uint32_t> in1_args = {
            in1_addr,
            out_addr,
            in2_addr,
            is_in1_sink,
            (std::uint32_t)in1_next_core_physical.x,  // in1_dest_noc_x
            (std::uint32_t)in1_next_core_physical.y,  // in1_dest_noc_y
            (std::uint32_t)in1_prev_core_physical.x,  // in1_sender_noc_x
            (std::uint32_t)in1_prev_core_physical.y,  // in1_sender_noc_y
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
            defer_write_k_block,
        };
        if (in0_idx == 0) {
            // in1 sender
            SetRuntimeArgs(program, in1_sender_kernels_id, core, in1_args);
        } else {
            // in1 receiver
            SetRuntimeArgs(program, in1_receiver_kernels_id, core, in1_args);
        }

        std::vector<uint32_t> compute_runtime_args = {
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
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
