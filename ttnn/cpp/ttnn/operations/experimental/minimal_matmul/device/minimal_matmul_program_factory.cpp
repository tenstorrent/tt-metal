// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "minimal_matmul_program_factory.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"

#include <algorithm>
#include <cstdlib>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tuple>
#include <utility>
#include <vector>

#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

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

// Build a linear order of cores along one axis for data movement, plus index of the current core
std::pair<std::vector<CoreCoord>, uint32_t> build_core_order_for_axis(
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

CoreCoord clamped_prev(const std::vector<CoreCoord>& order, uint32_t index) {
    return order.at(index == 0 ? 0 : index - 1);
}

CoreCoord clamped_next(const std::vector<CoreCoord>& order, uint32_t index) {
    const uint32_t last = static_cast<uint32_t>(order.size() - 1);
    return order.at(index >= last ? last : index + 1);
}

// Append tensor accessors in a consistent order
void append_accessors(
    std::vector<uint32_t>& args,
    const Tensor& main_tensor,
    const std::vector<Tensor>& output_tensors,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<const Tensor>& ag_input_tensor = std::nullopt,
    const std::optional<const Tensor>& ternary_a_tensor = std::nullopt,
    const std::optional<const Tensor>& ternary_b_tensor = std::nullopt) {
    tt::tt_metal::TensorAccessorArgs(*main_tensor.buffer()).append_to(args);
    for (const auto& output_tensor : output_tensors) {
        tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(args);
    }
    if (bias_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*bias_tensor.value().buffer()).append_to(args);
    }
    // AG input must come before ternary to match kernel accessor order
    if (ag_input_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*ag_input_tensor.value().buffer()).append_to(args);
    }
    if (ternary_a_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*ternary_a_tensor.value().buffer()).append_to(args);
    }
    if (ternary_b_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*ternary_b_tensor.value().buffer()).append_to(args);
    }
}

}  // namespace

// SHARED IMPLEMENTATION - works with vector of output tensors (exposed for minimal_matmul_split)
MinimalMatmulProgramFactory::shared_variables_t minimal_matmul_factory_helper_common(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<operations::unary::UnaryWithParam>& fused_activation,
    const std::optional<const MinimalMatmulConfig>& config,
    const std::vector<Tensor>& output_tensors,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler>& fused_op_signaler,
    uint32_t N_chunks,
    std::optional<float> fused_ternary_scalar,
    const std::optional<const Tensor>& fused_ternary_input_a,
    const std::optional<const Tensor>& fused_ternary_input_b,
    std::optional<ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler> srs_fused_op_signaler,
    bool fuse_swiglu) {
    (void)fused_ternary_scalar;  // Scalar not needed in dataflow kernel, only in compute kernel
    auto* device = input_tensor.device();

    bool fuse_op = fused_op_signaler.has_value();

    if (!config.has_value()) {
        log_debug(tt::LogOp, "No config provided, using default block sizes and core grid");
    }

    auto grid_size =
        config.has_value() ? config.value().compute_with_storage_grid_size : device->compute_with_storage_grid_size();
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    auto num_cores = core_grid.size();

    bool use_bias = bias_tensor.has_value();
    bool use_fused_ternary = fused_ternary_input_a.has_value() && fused_ternary_input_b.has_value();

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

    // Compute N_tiles_per_chunk for splitting
    const uint32_t N_tiles_per_chunk = N_tiles / N_chunks;

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
    // When fusing with strided reduce scatter, transposing is disabled
    // because it resulted in slightly lower performance on a case of interest.
    // (This can be revisited if needed.)
    const bool fuse_srs = srs_fused_op_signaler.has_value();
    bool transpose_core_grid = M > N && !fuse_srs;

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
    uint32_t padded_K_tiles = tt::round_up(K_tiles, K_block_tiles);

    uint32_t padded_N_tiles;
    uint32_t N_tiles_per_core;
    if (fuse_swiglu) {
        // Partition on gate/up PAIRS (= output tiles), so every core's weight-tile range is
        // 2 * (pairs per core): even, and never splitting a pair across cores.
        uint32_t out_N_tiles = N_tiles / 2;
        uint32_t padded_out_N_tiles = tt::round_up(out_N_tiles, in1_parallel_axis_cores);
        padded_N_tiles = 2 * padded_out_N_tiles;
        N_tiles_per_core = 2 * (padded_out_N_tiles / in1_parallel_axis_cores);
    } else {
        padded_N_tiles = tt::round_up(N_tiles, in1_parallel_axis_cores);
        N_tiles_per_core = padded_N_tiles / in1_parallel_axis_cores;
    }

    uint32_t M_tiles_per_core = padded_M_tiles / in0_parallel_axis_cores;

    uint32_t K_blocks = padded_K_tiles / K_block_tiles;

    uint32_t M_blocks_per_core = tt::div_up(M_tiles_per_core, M_block_tiles);
    uint32_t N_blocks_per_core = tt::div_up(N_tiles_per_core, N_block_tiles);

    if (fuse_swiglu) {
        // The gate/up tile pairs are interleaved along N (gate=2p, up=2p+1). Every core's
        // N range and every N block must start on an even tile and span an even number of
        // tiles so a pair is never split across cores or blocks.
        TT_FATAL(
            N_tiles % 2 == 0 && N_tiles_per_core % 2 == 0 && N_block_tiles % 2 == 0,
            "minimal_matmul fuse_swiglu requires N_tiles ({}), N_tiles_per_core ({}) and N_block_tiles ({}) all even",
            N_tiles,
            N_tiles_per_core,
            N_block_tiles);
    }

    log_debug(tt::LogOp, "M_tiles_per_core: {}", M_tiles_per_core);
    log_debug(tt::LogOp, "N_tiles_per_core: {}", N_tiles_per_core);
    log_debug(tt::LogOp, "M_blocks_per_core: {}", M_blocks_per_core);
    log_debug(tt::LogOp, "N_blocks_per_core: {}", N_blocks_per_core);

    uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;
    uint32_t in2_block_num_tiles = N_block_tiles;

    // Sub-chunk (M-row band) count for the fused AG in0 delivery; only meaningful when fusing AG.
    // Parsed here so the in1 scratch CB, the divisibility checks, and the per-kernel define all agree
    // on one value.
    uint32_t in0_sub_chunks = 1;
    if (fuse_op) {
        if (const char* e = std::getenv("IN0_SUB_CHUNKS")) {
            long v = std::strtol(e, nullptr, 10);
            if (v > 1) {
                in0_sub_chunks = static_cast<uint32_t>(v);
            }
        }
    }
    // Band-interleave: process a forward remote k-block and the following backward one one-band-at-a-time
    // (fwd.b0, bwd.b0, ...) instead of draining each whole. Needs two k-blocks resident in the in1 scratch
    // at once, so it also drives the scratch CB size below.
    bool interleave_bands = false;
    if (fuse_op && in0_sub_chunks > 1) {
        if (const char* e = std::getenv("AG_INTERLEAVE_BANDS")) {
            interleave_bands = (e[0] == '1');
        }
    }
    // Number of leading (self/local) k-block positions this device owns (see kernels). Placeholder
    // K_blocks when not fusing / not banding (value only consumed on the IN0_SUB_CHUNKS > 1 path).
    uint32_t num_local_k_blocks = K_blocks;

    const uint32_t double_buffer_factor = 2;
    uint32_t in0_cb_num_tiles = in0_block_num_tiles * double_buffer_factor;
    uint32_t in1_cb_num_tiles = in1_block_num_tiles * double_buffer_factor;
    // TODO: consider not double buffering the output
    // SwiGLU emits half the N tiles per block (one per gate/up pair), so the output CB only
    // needs to hold half a block. The intermediate CB still holds the full (2N) block.
    uint32_t out_block_num_tiles_written = fuse_swiglu ? (out_block_num_tiles / 2) : out_block_num_tiles;
    uint32_t out_cb_num_tiles = out_block_num_tiles_written * double_buffer_factor;
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

    {
        // Scratch holds one K_block x N_block so the in1 injector can read it once and re-present it to
        // compute per M-row band without re-reading DRAM (see dm_in1_sender_out.cpp). Single-buffered.
        // Created unconditionally: the in1 dataflow always reads through it (nb == 1 when not banding).
        // Band-interleave needs the forward and backward k-blocks of a pair resident at once, so it holds
        // two blocks.
        uint32_t in1_scratch_cb_id = tt::CBIndex::c_7;
        tt::tt_metal::create_cb(
            in1_scratch_cb_id,
            program,
            core_grid,
            in1_tile_size,
            in1_block_num_tiles * (interleave_bands ? 2u : 1u),
            in1_data_format);
    }

    // Two-NoC output-write split (env AG_SPLIT_OUTPUT_WRITE=1): the whole-block post-loop write is split
    // across M-rows so dm_in1 writes the low rows on NOC_1 and dm_in0 writes the high rows on NOC_0. Both DMs
    // are idle at the write (reads/mcasts done), so no input contention and no dynamic-NoC needed. Only the
    // plain (copy) epilogue with a single output and one M-block per core (defer_write always false).
    // AG_SPLIT_NOC1_PCT (0..100, default 50) sets the percent of rows going to NOC_1 (dm_in1); the rest go to
    // NOC_0. Lower values shift more onto NOC_0, which does not also carry the in1 mcast.
    bool split_output_write = false;
    uint32_t split_noc1_pct = 50;
    {
        const char* se = std::getenv("AG_SPLIT_OUTPUT_WRITE");
        if (se != nullptr && se[0] == '1' && N_chunks == 1 && !use_bias && !use_fused_ternary && !fuse_swiglu &&
            M_blocks_per_core == 1 && M_block_tiles > 1) {
            split_output_write = true;
            if (const char* p = std::getenv("AG_SPLIT_NOC1_PCT")) {
                long v = std::strtol(p, nullptr, 10);
                if (v < 0) {
                    v = 0;
                }
                if (v > 100) {
                    v = 100;
                }
                split_noc1_pct = static_cast<uint32_t>(v);
            }
        }
    }

    uint32_t out_cb_id = tt::CBIndex::c_2;
    tt::tt_metal::create_cb(out_cb_id, program, core_grid, out_tile_size, out_cb_num_tiles, output_data_format);
    if (split_output_write) {
        // Second output CB (c_8): the high-row half drained by dm_in0 on NOC_0. Full-block size upper-bounds
        // the high half; total out-CB L1 stays modest.
        uint32_t out_cb_b_id = tt::CBIndex::c_8;
        tt::tt_metal::create_cb(
            out_cb_b_id, program, core_grid, out_tile_size, out_block_num_tiles, output_data_format);
    }

    uint32_t intermediate_cb_id = tt::CBIndex::c_3;
    tt::tt_metal::create_cb(
        intermediate_cb_id, program, core_grid, intermediate_tile_size, interm_cb_num_tiles, intermediate_data_format);

    if (use_bias) {
        uint32_t in2_cb_id = tt::CBIndex::c_4;
        tt::tt_metal::create_cb(in2_cb_id, program, core_grid, in2_tile_size, in2_cb_num_tiles, in2_data_format);
    }

    // Create circular buffers for fused ternary inputs
    if (use_fused_ternary) {
        uint32_t ternary_a_cb_id = tt::CBIndex::c_5;
        uint32_t ternary_c_cb_id = tt::CBIndex::c_6;

        // Fused ternary input A - circular buffer c_5
        auto ternary_a_data_format =
            tt::tt_metal::datatype_to_dataformat_converter(fused_ternary_input_a.value().dtype());
        auto ternary_a_tile_size = tt::tile_size(ternary_a_data_format);

        TT_FATAL(ternary_a_tile_size == in1_tile_size, "ternary_a_tile_size must be equal to in1_tile_size");
        TT_FATAL(ternary_a_data_format == in1_data_format, "ternary_a_data_format must be equal to in1_data_format");
        uint32_t ternary_a_cb_num_tiles = out_block_num_tiles;  // Same as output block, not double buffered

        tt::tt_metal::create_cb(
            ternary_a_cb_id, program, core_grid, ternary_a_tile_size, ternary_a_cb_num_tiles, ternary_a_data_format);

        // Fused ternary input C - circular buffer c_6
        auto ternary_c_data_format =
            tt::tt_metal::datatype_to_dataformat_converter(fused_ternary_input_b.value().dtype());
        auto ternary_c_tile_size = tt::tile_size(ternary_c_data_format);
        uint32_t ternary_c_cb_num_tiles = N_block_tiles;  // Single row (like bias), broadcast across M

        tt::tt_metal::create_cb(
            ternary_c_cb_id, program, core_grid, ternary_c_tile_size, ternary_c_cb_num_tiles, ternary_c_data_format);

        log_debug(tt::LogOp, "ternary_a_cb_id: {}", ternary_a_cb_id);
        log_debug(tt::LogOp, "ternary_c_cb_id: {}", ternary_c_cb_id);
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

    std::map<std::string, std::string> defines;
    if (use_bias) {
        defines["FUSE_BIAS"] = "1";
    }

    if (fuse_swiglu) {
        defines["FUSE_SWIGLU"] = "1";
    }

    if (use_fused_ternary) {
        defines["FUSE_TERNARY"] = "1";

        // Workaround for LLK bug (https://github.com/tenstorrent/tt-llk/issues/1338)
        // - If ternary_b / gate is float32 then use unary_bcast (row broadcast) + mul_binary_tile (accurate)
        // - If ternary_b / gate is bfloat16 then use mul_tiles_bcast (row broadcast) (workaround)
        if (fused_ternary_input_b.value().dtype() == DataType::FLOAT32) {
            defines["TERNARY_B_IS_FLOAT32"] = "1";
        }
    }

    if (fuse_op) {
        // Create semaphores
        fused_op_signaler->init_fused_op(program, device, in0_sender_cores);
        defines["FUSE_AG"] = "1";
        // Stream the in0 read in this many M-row bands (parsed above), matching the AG's per-band
        // delivery/signal. On the IN0_SUB_CHUNKS > 1 path the matmul also matmuls + forwards per band
        // (see compute.cpp / dm_in0_sender.cpp / dm_in1_sender_out.cpp). Must equal the AG program's
        // IN0_SUB_CHUNKS so the per-band signal counts match.
        defines["IN0_SUB_CHUNKS"] = std::to_string(in0_sub_chunks);
        if (in0_sub_chunks > 1) {
            // Every band occupies a uniform in0 CB slot of (M_block_tiles / in0_sub_chunks) rows, so a
            // ragged band (height not a multiple of subblock_h, e.g. a partial final M-block) still
            // reserves a slot that tiles the CB exactly -- no mid-block fifo wrap. matmul reads the full
            // subblock_h into the slot's slack rows and packs only the real rows. Two divisibility
            // invariants make that exact:
            //   1. in0_sub_chunks | M_block_tiles           -- uniform slot is a whole number of rows
            //   2. subblock_h | (M_block_tiles/in0_sub_chunks) -- the deep read never runs past the slot
            // Plus every band must be non-empty: balanced_band yields a zero band only when a block has
            // fewer rows than in0_sub_chunks, and the in1 sender / signal aggregator fire IN0_SUB_CHUNKS
            // times unconditionally -- a zero band would desync them against the in0/compute early-exit
            // and deadlock. The smallest block is the (possibly partial) last one.
            uint32_t last_m_block_tiles = M_tiles_per_core - (M_blocks_per_core - 1) * M_block_tiles;
            TT_FATAL(
                last_m_block_tiles >= in0_sub_chunks,
                "smallest M block ({} tiles) must be >= IN0_SUB_CHUNKS ({}) so every M-row band is "
                "non-empty",
                last_m_block_tiles,
                in0_sub_chunks);
            TT_FATAL(
                (M_block_tiles % in0_sub_chunks) == 0,
                "IN0_SUB_CHUNKS ({}) must divide M_block_tiles ({})",
                in0_sub_chunks,
                M_block_tiles);
            TT_FATAL(
                ((M_block_tiles / in0_sub_chunks) % subblock_h) == 0,
                "subblock_h ({}) must divide the per-band slot M_block_tiles/IN0_SUB_CHUNKS ({}/{} = {})",
                subblock_h,
                M_block_tiles,
                in0_sub_chunks,
                M_block_tiles / in0_sub_chunks);
            // Count this device's local (self) k-blocks = the leading schedule positions the AG delivers
            // whole (never sub-chunked). Mirrors compute_device_chunk_stats for start_ring_index. v1
            // requires K_block_tiles-aligned device boundaries: a straddling (co-owned) k-block would
            // break the local-first band schedule, so assert none exist.
            uint32_t my_chip = fused_op_signaler->start_ring_index;
            uint32_t in_Wt = fused_op_signaler->input_tensor_Wt;
            uint32_t curr_device = 0;
            uint32_t curr_device_end = in_Wt - 1;
            uint32_t my_count = 0;
            for (uint32_t kb = 0; kb < K_blocks; kb++) {
                uint32_t kb_end = (kb + 1) * K_block_tiles - 1;
                if (kb_end < curr_device_end) {
                    if (curr_device == my_chip) {
                        my_count++;
                    }
                } else if (kb_end == curr_device_end) {
                    if (curr_device == my_chip) {
                        my_count++;
                    }
                    curr_device++;
                    curr_device_end = (curr_device + 1) * in_Wt - 1;
                } else {
                    TT_FATAL(
                        false,
                        "IN0_SUB_CHUNKS > 1 requires K_block_tiles ({}) aligned device boundaries "
                        "(input_tensor_Wt = {}); a straddling k-block is not supported",
                        K_block_tiles,
                        in_Wt);
                }
            }
            num_local_k_blocks = my_count;
        }
        // Consume the middle forward/backward k-blocks 1-backward-1-forward instead of grouped.
        // Env override lets A/B runs disable it (AG_ALTERNATE_MIDDLE=0) without a rebuild-time edit.
        const char* alt_middle_env = std::getenv("AG_ALTERNATE_MIDDLE");
        defines["AG_ALTERNATE_MIDDLE"] = (alt_middle_env != nullptr) ? alt_middle_env : "1";
        // Band-interleave a forward remote k-block with the following backward one (see dm_in0_sender.cpp).
        // Assigned to the shared defines map so in0/in1 senders and compute all agree on the paired path.
        if (interleave_bands) {
            defines["AG_INTERLEAVE_BANDS"] = "1";
        }
        // Timing-isolation knob: SKIP_IN0_DRAM_READ=1 makes the in0 injector skip the DRAM/local read
        // (semaphores still fire) to measure the read's contribution. PCC is garbage under this flag.
        const char* skip_in0_read_env = std::getenv("SKIP_IN0_DRAM_READ");
        if (skip_in0_read_env != nullptr && skip_in0_read_env[0] == '1') {
            defines["SKIP_IN0_DRAM_READ"] = "1";
        }
    }

    uint32_t srs_fuse_signaler_sync_semaphore_id = 0;
    if (fuse_srs) {
        defines["SRS_FUSE_OP_SIGNALER"] = "1";
        srs_fuse_signaler_sync_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, 0);
    }

    std::vector<CoreCoord> all_worker_cores_noc;
    if (fuse_srs) {
        all_worker_cores_noc.reserve(num_cores);
        auto all_cores_tmp = corerange_to_cores(core_grid, num_cores, true);
        for (const auto& c : all_cores_tmp) {
            all_worker_cores_noc.push_back(device->worker_core_from_logical_core(c));
        }
    }

    uint32_t in0_addr = input_tensor.buffer()->address();
    uint32_t in1_addr = weight_tensor.buffer()->address();
    uint32_t in2_addr = use_bias ? bias_tensor.value().buffer()->address() : 0;
    // Note: Dataflow kernels can take a variable number of output tensors.
    // They are appended as a variable-length array at the end of the runtime-args:
    //   - for in0 output-writer cores the first output address is at index 13
    //   - for in1 output-writer cores the first output address is at index 12
    uint32_t in3_addr = (fuse_op && fused_op_signaler->read_local_slice_from_input)
                            ? fused_op_signaler->ag_input.value().buffer()->address()
                            : 0;
    auto in3_data_format =
        (fuse_op && fused_op_signaler->read_local_slice_from_input)
            ? tt::tt_metal::datatype_to_dataformat_converter(fused_op_signaler->ag_input.value().dtype())
            : in1_data_format;

    auto in3_tile_size = tt::tile_size(in3_data_format);

    /**
     * Create kernels
     */

    // Under the two-NoC split both DMs write (dm_in1 the low rows on NOC_1, dm_in0 the high rows on NOC_0);
    // otherwise exactly one writes.
    bool in0_is_output_writer = split_output_write ? true : !transpose_core_grid;
    bool in1_is_output_writer = split_output_write ? true : transpose_core_grid;

    // Per-DM-family defines. Under the split: dm_in1 writes rows [0, split) from c_2; dm_in0 drains c_8 and
    // writes [split, M). Both get the same split percent so their ranges line up with compute's copy.
    auto in0_defines = defines;
    auto in1_defines = defines;
    if (split_output_write) {
        in1_defines["SPLIT_OUTPUT_WRITE"] = "1";
        in1_defines["AG_SPLIT_NOC1_PCT"] = std::to_string(split_noc1_pct);
        in0_defines["SPLIT_OUTPUT_WRITE"] = "1";
        in0_defines["AG_OUT_WRITE_CB"] = std::to_string(static_cast<uint32_t>(tt::CBIndex::c_8));
        in0_defines["AG_SPLIT_NOC1_PCT"] = std::to_string(split_noc1_pct);
    }
    // dm_in0 injector (read-local) variant layers READ_FROM_LOCAL_INPUT on top of the in0 defines.
    auto in0_injector_defines = in0_defines;
    if (fuse_op && fused_op_signaler->read_local_slice_from_input) {
        in0_injector_defines["READ_FROM_LOCAL_INPUT"] = "1";
    }

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
        true,               // is_injector_core
        N_chunks,           // N_chunks
        N_tiles_per_chunk,  // N_tiles_per_chunk
        in3_tile_size,
    };
    append_accessors(
        in0_sender_compile_time_args,
        input_tensor,
        output_tensors,
        bias_tensor,
        (fuse_op && fused_op_signaler->read_local_slice_from_input) ? fused_op_signaler->ag_input : std::nullopt,
        fused_ternary_input_a,
        fused_ternary_input_b);
    auto in0_sender_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp",
        in0_sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = in0_risc,
            .noc = in0_noc,
            .compile_args = in0_sender_compile_time_args,
            .defines = in0_injector_defines});

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
        false,              // is_injector_core
        N_chunks,           // N_chunks
        N_tiles_per_chunk,  // N_tiles_per_chunk
        in3_tile_size,
    };
    append_accessors(
        in0_receiver_compile_time_args,
        input_tensor,
        output_tensors,
        bias_tensor,
        std::nullopt,  // no ag_input for in0_receiver
        fused_ternary_input_a,
        fused_ternary_input_b);

    auto in0_receiver_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp",
        in0_receiver_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = in0_risc,
            .noc = in0_noc,
            .compile_args = in0_receiver_compile_time_args,
            .defines = in0_defines});

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
        true,               // is_injector_core
        N_chunks,           // N_chunks
        N_tiles_per_chunk,  // N_tiles_per_chunk
    };
    append_accessors(
        in1_sender_compile_time_args,
        weight_tensor,
        output_tensors,
        bias_tensor,
        std::nullopt,  // no ag_input for in1_sender
        fused_ternary_input_a,
        fused_ternary_input_b);

    auto in1_sender_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_sender_out.cpp",
        in1_sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = in1_risc,
            .noc = in1_noc,
            .compile_args = in1_sender_compile_time_args,
            .defines = in1_defines});

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
        false,              // is_injector_core
        N_chunks,           // N_chunks
        N_tiles_per_chunk,  // N_tiles_per_chunk
    };
    append_accessors(
        in1_receiver_compile_time_args,
        weight_tensor,
        output_tensors,
        bias_tensor,
        std::nullopt,  // no ag_input for in1_receiver
        fused_ternary_input_a,
        fused_ternary_input_b);

    auto in1_receiver_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_sender_out.cpp",
        in1_receiver_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = in1_risc,
            .noc = in1_noc,
            .compile_args = in1_receiver_compile_time_args,
            .defines = in1_defines});

    std::vector<uint32_t> compute_compile_time_args = {
        K_blocks,
        M_block_tiles,
        K_block_tiles,
        N_block_tiles,
        M_blocks_per_core,
        N_blocks_per_core,
        subblock_h,
        subblock_w,
        num_local_k_blocks};

    auto compute_defines = defines;
    if (split_output_write) {
        compute_defines["SPLIT_OUTPUT_WRITE"] = "1";
        compute_defines["OUT_CB_B"] = std::to_string(static_cast<uint32_t>(tt::CBIndex::c_8));
        compute_defines["AG_SPLIT_NOC1_PCT"] = std::to_string(split_noc1_pct);
    }
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
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, compute_defines, ttnn::get_throttle_level(compute_kernel_config));
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

    uint32_t max_defer_write_k_block = 0;
    for (const auto& c : cores) {
        uint32_t dwk = std::min(static_cast<uint32_t>(c.y) * k_blocks_per_core, K_blocks - 1);
        max_defer_write_k_block = std::max(max_defer_write_k_block, dwk);
    }

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

        // Defer write to K block with same coordinate as core
        // The writer receiver cores always have core.x > 0
        uint32_t defer_write_k_block = std::min(static_cast<uint32_t>(core.y) * k_blocks_per_core, K_blocks - 1);

        bool is_in0_sink = core == in0_core_order.back();
        bool is_in1_sink = core == in1_core_order.back();

        std::vector<uint32_t> in0_args = {
            in0_addr,
            in2_addr,
            in3_addr,
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
            max_defer_write_k_block,
            num_local_k_blocks,
        };
        // Add ternary addresses if present (after defer_write_k_block, before output addresses)
        if (use_fused_ternary) {
            in0_args.push_back(fused_ternary_input_a.value().buffer()->address());
            in0_args.push_back(fused_ternary_input_b.value().buffer()->address());
            uint32_t ternary_b_M_tiles = fused_ternary_input_b.value().padded_shape()[-2] / tt::constants::TILE_HEIGHT;
            in0_args.push_back(ternary_b_M_tiles == 1 ? 1u : 0u);  // broadcast_ternary_b
        }
        // Add output addresses at the end (unified layout for both regular and split)
        for (const auto& output_tensor : output_tensors) {
            in0_args.push_back(output_tensor.buffer()->address());
        }
        if (fuse_op) {
            fused_op_signaler->push_matmul_fused_op_rt_args(in0_args, padded_K_tiles / K_block_tiles, K_block_tiles);
        }
        if (fuse_srs) {
            in0_args.push_back(static_cast<uint32_t>(num_cores));
            in0_args.push_back(static_cast<uint32_t>(core_id));
            in0_args.push_back(static_cast<uint32_t>(srs_fuse_signaler_sync_semaphore_id));
            for (const auto& noc_core : all_worker_cores_noc) {
                in0_args.push_back(static_cast<uint32_t>(noc_core.x));
                in0_args.push_back(static_cast<uint32_t>(noc_core.y));
            }
            in0_args.push_back(static_cast<uint32_t>(srs_fused_op_signaler->num_fused_op_cores_to_signal));
            for (const auto& noc_core : srs_fused_op_signaler->fused_op_receiver_cores_noc) {
                in0_args.push_back(static_cast<uint32_t>(noc_core.x));
                in0_args.push_back(static_cast<uint32_t>(noc_core.y));
            }
            in0_args.push_back(static_cast<uint32_t>(srs_fused_op_signaler->fused_op_receiver_signal_semaphore));
            in0_args.push_back(1);  // mcast_signal_op_cores
        }
        if (in1_idx == 0) {
            // in0 sender
            SetRuntimeArgs(program, in0_sender_kernels_id, core, in0_args);
        } else {
            // in0 receiver
            SetRuntimeArgs(program, in0_receiver_kernels_id, core, in0_args);
        }

        std::vector<uint32_t> in1_args = {
            in1_addr,
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
            max_defer_write_k_block,
            num_local_k_blocks,
        };
        // Add ternary addresses if present (after defer_write_k_block, before output addresses)
        if (use_fused_ternary) {
            in1_args.push_back(fused_ternary_input_a.value().buffer()->address());
            in1_args.push_back(fused_ternary_input_b.value().buffer()->address());
            uint32_t ternary_b_M_tiles = fused_ternary_input_b.value().padded_shape()[-2] / tt::constants::TILE_HEIGHT;
            in1_args.push_back(ternary_b_M_tiles == 1 ? 1u : 0u);  // broadcast_ternary_b
        }
        // Add output addresses at the end (unified layout for both regular and split)
        for (const auto& output_tensor : output_tensors) {
            in1_args.push_back(output_tensor.buffer()->address());
        }
        if (fuse_op) {
            fused_op_signaler->push_matmul_fused_op_rt_args(in1_args, padded_K_tiles / K_block_tiles, K_block_tiles);
        }
        if (fuse_srs) {
            in1_args.push_back(static_cast<uint32_t>(num_cores));
            in1_args.push_back(static_cast<uint32_t>(core_id));
            in1_args.push_back(static_cast<uint32_t>(srs_fuse_signaler_sync_semaphore_id));
            for (const auto& noc_core : all_worker_cores_noc) {
                in1_args.push_back(static_cast<uint32_t>(noc_core.x));
                in1_args.push_back(static_cast<uint32_t>(noc_core.y));
            }
            in1_args.push_back(static_cast<uint32_t>(srs_fused_op_signaler->num_fused_op_cores_to_signal));
            for (const auto& noc_core : srs_fused_op_signaler->fused_op_receiver_cores_noc) {
                in1_args.push_back(static_cast<uint32_t>(noc_core.x));
                in1_args.push_back(static_cast<uint32_t>(noc_core.y));
            }
            in1_args.push_back(static_cast<uint32_t>(srs_fused_op_signaler->fused_op_receiver_signal_semaphore));
            in1_args.push_back(1);  // mcast_signal_op_cores
        }
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
        if (use_fused_ternary) {
            compute_runtime_args.push_back(*reinterpret_cast<const uint32_t*>(&fused_ternary_scalar.value()));
            uint32_t ternary_b_M_tiles = fused_ternary_input_b.value().padded_shape()[-2] / tt::constants::TILE_HEIGHT;
            compute_runtime_args.push_back(ternary_b_M_tiles == 1 ? 1u : 0u);  // broadcast_ternary_b
        }
        SetRuntimeArgs(program, compute_kernels_id, core, compute_runtime_args);
    }

    return MinimalMatmulProgramFactory::shared_variables_t{
        num_cores,
        cores,
        in0_sender_kernels_id,
        in0_receiver_kernels_id,
        in1_sender_kernels_id,
        in1_receiver_kernels_id,
        compute_kernels_id,
        transpose_core_grid,
        fuse_op && fused_op_signaler->read_local_slice_from_input};
}

MinimalMatmulProgramFactory::shared_variables_t minimal_matmul_factory_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<operations::unary::UnaryWithParam>& fused_activation,
    const std::optional<const MinimalMatmulConfig>& config,
    const Tensor& output_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler>& fused_op_signaler,
    std::optional<ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler>& srs_fused_op_signaler,
    bool fuse_swiglu) {
    std::vector<Tensor> output_tensors = {output_tensor};
    return minimal_matmul_factory_helper_common(
        program,
        input_tensor,
        weight_tensor,
        bias_tensor,
        fused_activation,
        config,
        output_tensors,
        compute_kernel_config,
        fused_op_signaler,
        1,  // N_chunks = 1 for regular minimal_matmul
        std::nullopt,
        std::nullopt,
        std::nullopt,
        srs_fused_op_signaler,
        fuse_swiglu);
}

MinimalMatmulProgramFactory::cached_program_t MinimalMatmulProgramFactory::create(
    const MinimalMatmulParams& operation_attributes,
    const MinimalMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler> empty_fused_op_signaler;
    std::optional<ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler> empty_srs_fused_op_signaler;

    auto shared_vars = minimal_matmul_factory_helper_common(
        program,
        tensor_args.input_tensor,
        tensor_args.weight_tensor,
        tensor_args.bias_tensor,
        operation_attributes.fused_activation,
        operation_attributes.config,
        tensor_return_value,
        operation_attributes.compute_kernel_config,
        empty_fused_op_signaler,
        static_cast<uint32_t>(operation_attributes.chunks),
        operation_attributes.fused_ternary_scalar,
        tensor_args.fused_ternary_input_a,
        tensor_args.fused_ternary_input_b,
        empty_srs_fused_op_signaler,
        operation_attributes.fuse_swiglu);

    return {std::move(program), std::move(shared_vars)};
}

// Common helper for override_runtime_arguments - works with both single and multiple output tensors
void MinimalMatmulProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const MinimalMatmulParams& operation_attributes,
    const MinimalMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    auto& program = cached_program.program;
    auto& override_variables = cached_program.shared_variables;

    auto& in0_sender_runtime_args = GetRuntimeArgs(program, override_variables.in0_sender_kernels_id);
    auto& in0_receiver_runtime_args = GetRuntimeArgs(program, override_variables.in0_receiver_kernels_id);
    auto& in1_sender_runtime_args = GetRuntimeArgs(program, override_variables.in1_sender_kernels_id);
    auto& in1_receiver_runtime_args = GetRuntimeArgs(program, override_variables.in1_receiver_kernels_id);
    auto& compute_runtime_args = GetRuntimeArgs(program, override_variables.compute_kernels_id);

    // RT args layout for in0: [in0_addr, in2_addr, in3_addr, is_sink, noc_coords(4), tile_ranges(4),
    //   defer_write_k_block, max_defer_write_k_block,
    //   [optional: ternary_a_addr, ternary_b_addr, broadcast_ternary_b], out_addrs(N)...]
    // RT args layout for in1: [in1_addr, in2_addr, is_sink, noc_coords(4), tile_ranges(4),
    //   defer_write_k_block, max_defer_write_k_block,
    //   [optional: ternary_a_addr, ternary_b_addr, broadcast_ternary_b], out_addrs(N)...]
    constexpr uint32_t in0_in0_addr_idx = 0;
    constexpr uint32_t in0_in2_addr_idx = 1;
    constexpr uint32_t in0_in3_addr_idx = 2;
    constexpr uint32_t in0_ternary_a_addr_idx = 15;  // After max_defer_write_k_block (13), num_local_k_blocks (14)
    constexpr uint32_t in0_ternary_b_addr_idx = 16;

    constexpr uint32_t in1_in0_addr_idx = 0;
    constexpr uint32_t in1_bias_addr_idx = 1;
    constexpr uint32_t in1_ternary_a_addr_idx = 14;  // After max_defer_write_k_block (12), num_local_k_blocks (13)
    constexpr uint32_t in1_ternary_b_addr_idx = 15;

    // Check if ternary addresses are present
    bool has_fused_ternary =
        tensor_args.fused_ternary_input_a.has_value() && tensor_args.fused_ternary_input_b.has_value();
    // Output addresses start after max_defer_write_k_block, num_local_k_blocks, and optional ternary addresses.
    // in0: max_defer(13), num_local(14) -> outputs at 15 (+3 for ternary a/b/broadcast = 18).
    // in1: max_defer(12), num_local(13) -> outputs at 14 (+3 for ternary = 17).
    uint32_t in0_out_addr_start_idx = has_fused_ternary ? 18 : 15;
    uint32_t in1_out_addr_start_idx = has_fused_ternary ? 17 : 14;

    for (uint32_t i = 0; i < override_variables.num_cores; ++i) {
        CoreCoord core = override_variables.cores.at(i);
        uint32_t in0_idx = override_variables.transpose_core_grid ? core.x : core.y;
        uint32_t in1_idx = override_variables.transpose_core_grid ? core.y : core.x;

        if (in1_idx == 0) {
            auto& in0_sender_args = in0_sender_runtime_args[core.x][core.y];

            in0_sender_args[in0_in0_addr_idx] = tensor_args.input_tensor.buffer()->address();
            in0_sender_args[in0_in2_addr_idx] =
                tensor_args.bias_tensor.has_value() ? tensor_args.bias_tensor.value().buffer()->address() : 0;
            in0_sender_args[in0_in3_addr_idx] = tensor_args.optional_input_tensor.has_value() &&
                                                        cached_program.shared_variables.read_local_slice_from_input
                                                    ? tensor_args.optional_input_tensor.value().buffer()->address()
                                                    : 0;
            // Update ternary addresses if present
            if (has_fused_ternary) {
                in0_sender_args[in0_ternary_a_addr_idx] = tensor_args.fused_ternary_input_a.value().buffer()->address();
                in0_sender_args[in0_ternary_b_addr_idx] = tensor_args.fused_ternary_input_b.value().buffer()->address();
            }
            // Update N output addresses at the end
            for (size_t out_idx = 0; out_idx < tensor_return_value.size(); ++out_idx) {
                in0_sender_args[in0_out_addr_start_idx + out_idx] = tensor_return_value[out_idx].buffer()->address();
            }
        } else {
            auto& in0_receiver_args = in0_receiver_runtime_args[core.x][core.y];
            in0_receiver_args[in0_in2_addr_idx] =
                tensor_args.bias_tensor.has_value() ? tensor_args.bias_tensor.value().buffer()->address() : 0;
            // Update ternary addresses if present
            if (has_fused_ternary) {
                in0_receiver_args[in0_ternary_a_addr_idx] =
                    tensor_args.fused_ternary_input_a.value().buffer()->address();
                in0_receiver_args[in0_ternary_b_addr_idx] =
                    tensor_args.fused_ternary_input_b.value().buffer()->address();
            }
            // Update N output addresses at the end
            for (size_t out_idx = 0; out_idx < tensor_return_value.size(); ++out_idx) {
                in0_receiver_args[in0_out_addr_start_idx + out_idx] = tensor_return_value[out_idx].buffer()->address();
            }
        }

        if (in0_idx == 0) {
            auto& in1_sender_args = in1_sender_runtime_args[core.x][core.y];
            in1_sender_args[in1_in0_addr_idx] = tensor_args.weight_tensor.buffer()->address();
            in1_sender_args[in1_bias_addr_idx] =
                tensor_args.bias_tensor.has_value() ? tensor_args.bias_tensor.value().buffer()->address() : 0;
            // Update ternary addresses if present
            if (has_fused_ternary) {
                in1_sender_args[in1_ternary_a_addr_idx] = tensor_args.fused_ternary_input_a.value().buffer()->address();
                in1_sender_args[in1_ternary_b_addr_idx] = tensor_args.fused_ternary_input_b.value().buffer()->address();
            }
            // Update N output addresses at the end
            for (size_t out_idx = 0; out_idx < tensor_return_value.size(); ++out_idx) {
                in1_sender_args[in1_out_addr_start_idx + out_idx] = tensor_return_value[out_idx].buffer()->address();
            }
        } else {
            auto& in1_receiver_args = in1_receiver_runtime_args[core.x][core.y];
            in1_receiver_args[in1_bias_addr_idx] =
                tensor_args.bias_tensor.has_value() ? tensor_args.bias_tensor.value().buffer()->address() : 0;
            // Update ternary addresses if present
            if (has_fused_ternary) {
                in1_receiver_args[in1_ternary_a_addr_idx] =
                    tensor_args.fused_ternary_input_a.value().buffer()->address();
                in1_receiver_args[in1_ternary_b_addr_idx] =
                    tensor_args.fused_ternary_input_b.value().buffer()->address();
            }
            // Update N output addresses at the end
            for (size_t out_idx = 0; out_idx < tensor_return_value.size(); ++out_idx) {
                in1_receiver_args[in1_out_addr_start_idx + out_idx] = tensor_return_value[out_idx].buffer()->address();
            }
        }
    }

    // Update compute kernel runtime args for scalar
    for (uint32_t i = 0; i < override_variables.num_cores; ++i) {
        CoreCoord core = override_variables.cores.at(i);
        auto& compute_args = compute_runtime_args[core.x][core.y];

        // Compute RT args: [M_start, M_end, N_start, N_end, [optional: scalar]]
        // If ternary is present and scalar arg exists, update it at index 4
        if (has_fused_ternary && operation_attributes.fused_ternary_scalar.has_value()) {
            float scalar = operation_attributes.fused_ternary_scalar.value();
            uint32_t scalar_as_uint = *reinterpret_cast<const uint32_t*>(&scalar);
            compute_args[4] = scalar_as_uint;
        }
    }
}

}  // namespace ttnn::experimental::prim
