// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "minimal_matmul_fabric_bound_program_factory.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <algorithm>
#include <bit>
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

// Lowest semaphore id free on every core of `core_ranges`, the same id CreateSemaphore picks; Program{desc}
// rejects an id past the per-core semaphore limit.
uint32_t add_fabric_bound_semaphore(
    tt::tt_metal::ProgramDescriptor& desc, const CoreRangeSet& core_ranges, uint32_t initial_value) {
    uint32_t semaphore_id = 0;
    while (std::any_of(
        desc.semaphores.begin(), desc.semaphores.end(), [&](const tt::tt_metal::SemaphoreDescriptor& semaphore) {
            return semaphore.id == semaphore_id && semaphore.core_type == tt::CoreType::WORKER &&
                   semaphore.core_ranges.intersects(core_ranges);
        })) {
        semaphore_id++;
    }
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = core_ranges,
        .initial_value = initial_value,
    });
    return semaphore_id;
}

void add_fabric_bound_cb(
    tt::tt_metal::ProgramDescriptor& desc,
    uint32_t cb_id,
    const CoreRangeSet& core_ranges,
    uint32_t page_size,
    uint32_t num_pages,
    tt::DataFormat data_format) {
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = num_pages * page_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_id),
            .data_format = data_format,
            .page_size = page_size,
        }}},
    });
}

template <typename F>
void for_each_core_runtime_args(tt::tt_metal::Program& program, tt::tt_metal::KernelHandle kernel_index, F&& patch) {
    auto& runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(program, kernel_index);
    for (auto& runtime_args_column : runtime_args_by_core) {
        for (auto& runtime_args : runtime_args_column) {
            if (runtime_args.size() > 0) {
                patch(runtime_args);
            }
        }
    }
}

}  // namespace

// SHARED IMPLEMENTATION - works with vector of output tensors
void minimal_matmul_fabric_bound_factory_helper_common(
    tt::tt_metal::ProgramDescriptor& desc,
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
    namespace layout = minimal_matmul_fabric_bound_layout;
    using tt::tt_metal::ComputeConfigDescriptor;
    using tt::tt_metal::DataMovementConfigDescriptor;
    using tt::tt_metal::KernelDescriptor;

    const uint32_t first_kernel_index = desc.kernels.size();
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

    // Transpose core grid if the output is wide (M > N) If transpose core grid
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
        // Partition on gate/up PAIRS (= output tiles), so every core's weight-tile range is 2 *
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
        // The gate/up tile pairs are interleaved along N (gate=2p, up=2p+1)
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

    // Sub-chunk (M-row band) count for the fused AG in0 delivery
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
    bool interleave_bands = fuse_op && in0_sub_chunks > 1;
    // Number of leading (self/local) k-block positions this device owns (see kernels)
    uint32_t num_local_k_blocks = K_blocks;

    const uint32_t double_buffer_factor = 2;
    uint32_t in0_cb_num_tiles = in0_block_num_tiles * double_buffer_factor;
    uint32_t in1_cb_num_tiles = in1_block_num_tiles * double_buffer_factor;
    // SwiGLU emits half the N tiles per block (one per gate/up pair)
    uint32_t out_block_num_tiles_written = fuse_swiglu ? (out_block_num_tiles / 2) : out_block_num_tiles;
    // out_cb_num_tiles is sized below, after split_output_write is known: the two-NoC split path only runs
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

    const CoreRangeSet core_grid_set(core_grid);
    auto in0_sender_semaphore_id = add_fabric_bound_semaphore(desc, core_grid_set, INVALID);
    auto in0_receiver_semaphore_id = add_fabric_bound_semaphore(desc, core_grid_set, INVALID);
    auto in0_valid_semaphore_id = add_fabric_bound_semaphore(desc, core_grid_set, VALID);
    auto in1_sender_semaphore_id = add_fabric_bound_semaphore(desc, core_grid_set, INVALID);
    auto in1_receiver_semaphore_id = add_fabric_bound_semaphore(desc, core_grid_set, INVALID);
    auto in1_valid_semaphore_id = add_fabric_bound_semaphore(desc, core_grid_set, VALID);

    // CB index of every circular buffer this helper creates, exposed to all five kernels as named compile-time args
    KernelDescriptor::NamedCompileTimeArgs cb_named_args;

    uint32_t in0_cb_id = tt::CBIndex::c_0;
    add_fabric_bound_cb(desc, in0_cb_id, core_grid_set, in0_tile_size, in0_cb_num_tiles, in0_data_format);
    cb_named_args.emplace_back("cb_in0", in0_cb_id);

    uint32_t in1_cb_id = tt::CBIndex::c_1;
    add_fabric_bound_cb(desc, in1_cb_id, core_grid_set, in1_tile_size, in1_cb_num_tiles, in1_data_format);
    cb_named_args.emplace_back("cb_in1", in1_cb_id);

    {
        // Scratch holds one K_block x N_block so the in1 injector can read it once and re-present it to compute
        uint32_t in1_scratch_cb_id = tt::CBIndex::c_7;
        add_fabric_bound_cb(
            desc,
            in1_scratch_cb_id,
            core_grid_set,
            in1_tile_size,
            in1_block_num_tiles * (interleave_bands ? 2u : 1u),
            in1_data_format);
        cb_named_args.emplace_back("cb_in1_scratch", in1_scratch_cb_id);
    }

    // Two-NoC output-write split: the whole-block post-loop write is split across M-rows so dm_in1 writes the low
    const uint32_t split_noc1_pct = 50;
    // Works with bias
    bool split_output_write = !use_fused_ternary && !fuse_swiglu && M_blocks_per_core == 1 && M_block_tiles > 1;
    // Interleaved two-NoC output write: replace the contiguous [0, split_rows) / [split_rows
    const bool interleaved_output_write = true;

    // Output CB double-buffering only earns its L1 when there is a *next* M-block for compute to work
    uint32_t out_cb_num_tiles = out_block_num_tiles_written * (M_blocks_per_core == 1 ? 1u : double_buffer_factor);

    uint32_t out_cb_id = tt::CBIndex::c_2;
    add_fabric_bound_cb(desc, out_cb_id, core_grid_set, out_tile_size, out_cb_num_tiles, output_data_format);
    cb_named_args.emplace_back("cb_out", out_cb_id);
    if (split_output_write) {
        // Second output CB (c_8): the high-row half drained by dm_in0 on NOC_0
        uint32_t out_cb_b_id = tt::CBIndex::c_8;
        add_fabric_bound_cb(desc, out_cb_b_id, core_grid_set, out_tile_size, out_block_num_tiles, output_data_format);
        cb_named_args.emplace_back("cb_out_b", out_cb_b_id);
    }

    uint32_t intermediate_cb_id = tt::CBIndex::c_3;
    add_fabric_bound_cb(
        desc, intermediate_cb_id, core_grid_set, intermediate_tile_size, interm_cb_num_tiles, intermediate_data_format);
    cb_named_args.emplace_back("cb_intermediate", intermediate_cb_id);

    if (use_bias) {
        uint32_t in2_cb_id = tt::CBIndex::c_4;
        add_fabric_bound_cb(desc, in2_cb_id, core_grid_set, in2_tile_size, in2_cb_num_tiles, in2_data_format);
        cb_named_args.emplace_back("cb_bias", in2_cb_id);
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

        add_fabric_bound_cb(
            desc, ternary_a_cb_id, core_grid_set, ternary_a_tile_size, ternary_a_cb_num_tiles, ternary_a_data_format);
        cb_named_args.emplace_back("cb_ternary_a", ternary_a_cb_id);

        // Fused ternary input C - circular buffer c_6
        auto ternary_c_data_format =
            tt::tt_metal::datatype_to_dataformat_converter(fused_ternary_input_b.value().dtype());
        auto ternary_c_tile_size = tt::tile_size(ternary_c_data_format);
        uint32_t ternary_c_cb_num_tiles = N_block_tiles;  // Single row (like bias), broadcast across M

        add_fabric_bound_cb(
            desc, ternary_c_cb_id, core_grid_set, ternary_c_tile_size, ternary_c_cb_num_tiles, ternary_c_data_format);
        cb_named_args.emplace_back("cb_ternary_b", ternary_c_cb_id);

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
        if (fused_ternary_input_b.value().dtype() == DataType::FLOAT32) {
            defines["TERNARY_B_IS_FLOAT32"] = "1";
        }
    }

    if (fuse_op) {
        // Descriptor form of MinimalMatmulFusedOpSignaler::init_fused_op (MULTI mode): every in0 injector core is
        // signaled, and each gets 2 * num_ag_workers + 1 receiver semaphores laid out [backward..., forward..., self].
        fused_op_signaler->fused_op_signaler_mode = ttnn::experimental::ccl::FusedOpSignalerMode::MULTI;
        fused_op_signaler->fused_op_receiver_cores_noc.clear();
        for (const auto& core :
             tt::tt_metal::grid_to_cores(in0_sender_cores.start_coord, in0_sender_cores.end_coord, true)) {
            fused_op_signaler->fused_op_receiver_cores_noc.push_back(device->worker_core_from_logical_core(core));
        }
        const CoreRangeSet in0_sender_core_set(in0_sender_cores);
        const uint32_t num_signal_semaphores = (2 * fused_op_signaler->num_ag_workers) + 1;
        for (uint32_t i = 0; i < num_signal_semaphores; i++) {
            fused_op_signaler->fused_op_receiver_signal_semaphores.push_back(
                add_fabric_bound_semaphore(desc, in0_sender_core_set, 0));
        }
        fused_op_signaler->num_fused_op_cores_to_signal = fused_op_signaler->fused_op_receiver_cores_noc.size();
        fused_op_signaler->initialized_fused_op = true;
        defines["FUSE_AG"] = "1";
        // Stream the in0 read in this many M-row bands (parsed above), matching the AG's per-band delivery/signal
        defines["IN0_SUB_CHUNKS"] = std::to_string(in0_sub_chunks);
        if (in0_sub_chunks > 1) {
            // Every band occupies a uniform in0 CB slot of (M_block_tiles / in0_sub_chunks) rows
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
            // Count this device's local (self) k-blocks = the leading schedule positions the AG delivers whole
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
        // Consume the middle forward/backward k-blocks 1-backward-1-forward instead of grouped
        defines["AG_ALTERNATE_MIDDLE"] = "1";
        // Band-interleave a forward remote k-block with the following backward one (see dm_in0_sender.cpp)
        if (interleave_bands) {
            defines["AG_INTERLEAVE_BANDS"] = "1";
        }
    }

    uint32_t srs_fuse_signaler_sync_semaphore_id = 0;
    if (fuse_srs) {
        defines["SRS_FUSE_OP_SIGNALER"] = "1";
        srs_fuse_signaler_sync_semaphore_id = add_fabric_bound_semaphore(desc, core_grid_set, 0);
    }

    std::vector<CoreCoord> all_worker_cores_noc;
    if (fuse_srs) {
        all_worker_cores_noc.reserve(num_cores);
        auto all_cores_tmp = corerange_to_cores(core_grid, num_cores, true);
        for (const auto& c : all_cores_tmp) {
            all_worker_cores_noc.push_back(device->worker_core_from_logical_core(c));
        }
    }

    tt::tt_metal::Buffer* in0_buffer = input_tensor.buffer();
    tt::tt_metal::Buffer* in1_buffer = weight_tensor.buffer();
    TT_FATAL(in0_buffer != nullptr, "minimal_matmul fabric-bound input (in0) tensor buffer is null");
    TT_FATAL(in1_buffer != nullptr, "minimal_matmul fabric-bound weight (in1) tensor buffer is null");
    // An absent optional tensor binds a null buffer, which the kernel sees as address 0
    tt::tt_metal::Buffer* in2_buffer = use_bias ? bias_tensor.value().buffer() : nullptr;
    TT_FATAL(!use_bias || in2_buffer != nullptr, "minimal_matmul fabric-bound bias tensor buffer is null");
    // Note: Dataflow kernels can take a variable number of output tensors
    tt::tt_metal::Buffer* in3_buffer = (fuse_op && fused_op_signaler->read_local_slice_from_input)
                                           ? fused_op_signaler->ag_input.value().buffer()
                                           : nullptr;
    TT_FATAL(
        !(fuse_op && fused_op_signaler->read_local_slice_from_input) || in3_buffer != nullptr,
        "minimal_matmul fabric-bound all-gather input tensor buffer is null");
    if (use_fused_ternary) {
        TT_FATAL(
            fused_ternary_input_a.value().buffer() != nullptr,
            "minimal_matmul fabric-bound fused_ternary_input_a buffer is null");
        TT_FATAL(
            fused_ternary_input_b.value().buffer() != nullptr,
            "minimal_matmul fabric-bound fused_ternary_input_b buffer is null");
    }
    for (const auto& output_tensor : output_tensors) {
        TT_FATAL(output_tensor.buffer() != nullptr, "minimal_matmul fabric-bound output tensor buffer is null");
    }
    auto in3_data_format =
        (fuse_op && fused_op_signaler->read_local_slice_from_input)
            ? tt::tt_metal::datatype_to_dataformat_converter(fused_op_signaler->ag_input.value().dtype())
            : in1_data_format;

    auto in3_tile_size = tt::tile_size(in3_data_format);

    /**
     * Create kernels
     */

    // Under the two-NoC split both DMs write (dm_in1 the low rows on NOC_1, dm_in0 the high rows on NOC_0)
    bool in0_is_output_writer = split_output_write ? true : !transpose_core_grid;
    bool in1_is_output_writer = split_output_write ? true : transpose_core_grid;

    // Per-DM-family defines
    auto in0_defines = defines;
    auto in1_defines = defines;
    if (split_output_write) {
        in1_defines["SPLIT_OUTPUT_WRITE"] = "1";
        in1_defines["AG_SPLIT_NOC1_PCT"] = std::to_string(split_noc1_pct);
        in0_defines["SPLIT_OUTPUT_WRITE"] = "1";
        in0_defines["AG_OUT_WRITE_CB"] = std::to_string(static_cast<uint32_t>(tt::CBIndex::c_8));
        in0_defines["AG_SPLIT_NOC1_PCT"] = std::to_string(split_noc1_pct);
        if (interleaved_output_write && N_chunks == 1) {
            in1_defines["AGMM_INTERLEAVED_OUTPUT_WRITE"] = "1";
            in0_defines["AGMM_INTERLEAVED_OUTPUT_WRITE"] = "1";
        }
    }
    // dm_in0 injector (read-local) variant layers READ_FROM_LOCAL_INPUT on top of the in0 defines.
    auto in0_injector_defines = in0_defines;
    if (fuse_op && fused_op_signaler->read_local_slice_from_input) {
        in0_injector_defines["READ_FROM_LOCAL_INPUT"] = "1";
    }

    const auto make_dm_kernel = [&](const char* kernel_source,
                                    const CoreRange& cores,
                                    std::vector<uint32_t> compile_time_args,
                                    const std::map<std::string, std::string>& kernel_defines,
                                    tt::tt_metal::DataMovementProcessor processor,
                                    tt::tt_metal::NOC noc) {
        KernelDescriptor kernel;
        kernel.kernel_source = kernel_source;
        kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
        kernel.core_ranges = CoreRangeSet(cores);
        kernel.compile_time_args = std::move(compile_time_args);
        kernel.named_compile_time_args = cb_named_args;
        kernel.defines = {kernel_defines.begin(), kernel_defines.end()};
        kernel.config = DataMovementConfigDescriptor{.processor = processor, .noc = noc};
        return kernel;
    };

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
    auto in0_sender_kernel = make_dm_kernel(
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/fabric_bound_dm_in0_sender.cpp",
        in0_sender_cores,
        std::move(in0_sender_compile_time_args),
        in0_injector_defines,
        in0_risc,
        in0_noc);

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

    auto in0_receiver_kernel = make_dm_kernel(
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/fabric_bound_dm_in0_sender.cpp",
        in0_receiver_cores,
        std::move(in0_receiver_compile_time_args),
        in0_defines,
        in0_risc,
        in0_noc);

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

    auto in1_sender_kernel = make_dm_kernel(
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/fabric_bound_dm_in1_sender_out.cpp",
        in1_sender_cores,
        std::move(in1_sender_compile_time_args),
        in1_defines,
        in1_risc,
        in1_noc);

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

    auto in1_receiver_kernel = make_dm_kernel(
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/fabric_bound_dm_in1_sender_out.cpp",
        in1_receiver_cores,
        std::move(in1_receiver_compile_time_args),
        in1_defines,
        in1_risc,
        in1_noc);

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
        if (interleaved_output_write && N_chunks == 1) {
            compute_defines["AGMM_INTERLEAVED_OUTPUT_WRITE"] = "1";
        }
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
    KernelDescriptor compute_kernel;
    compute_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/fabric_bound_compute.cpp";
    compute_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel.core_ranges = core_grid_set;
    compute_kernel.compile_time_args = std::move(compute_compile_time_args);
    compute_kernel.named_compile_time_args = cb_named_args;
    compute_kernel.defines = {compute_defines.begin(), compute_defines.end()};
    compute_kernel.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };

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

    const uint32_t broadcast_ternary_b =
        (use_fused_ternary && fused_ternary_input_b.value().padded_shape()[-2] / tt::constants::TILE_HEIGHT == 1) ? 1u
                                                                                                                  : 0u;

    // NOTE: Uniform per-core M/N ranges are required for DM forward handshakes to match across links

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

        // Defer write to K block with same coordinate as core The writer receiver cores always have core.x > 0
        uint32_t defer_write_k_block = std::min(static_cast<uint32_t>(core.y) * k_blocks_per_core, K_blocks - 1);

        bool is_in0_sink = core == in0_core_order.back();
        bool is_in1_sink = core == in1_core_order.back();

        // Ternary addresses go after num_local_k_blocks and before the output addresses
        const auto append_ternary_and_outputs = [&](KernelDescriptor::RTArgList& args) {
            if (use_fused_ternary) {
                args.push_back(fused_ternary_input_a.value().buffer());
                args.push_back(fused_ternary_input_b.value().buffer());
                args.push_back(broadcast_ternary_b);
            }
            // Add output addresses at the end (unified layout for both regular and split)
            for (const auto& output_tensor : output_tensors) {
                args.push_back(output_tensor.buffer());
            }
        };
        const auto append_signaler_args = [&](KernelDescriptor::RTArgList& args) {
            std::vector<uint32_t> signaler_args;
            if (fuse_op) {
                fused_op_signaler->push_matmul_fused_op_rt_args(
                    signaler_args, padded_K_tiles / K_block_tiles, K_block_tiles);
            }
            if (fuse_srs) {
                signaler_args.push_back(static_cast<uint32_t>(num_cores));
                signaler_args.push_back(static_cast<uint32_t>(core_id));
                signaler_args.push_back(static_cast<uint32_t>(srs_fuse_signaler_sync_semaphore_id));
                for (const auto& noc_core : all_worker_cores_noc) {
                    signaler_args.push_back(static_cast<uint32_t>(noc_core.x));
                    signaler_args.push_back(static_cast<uint32_t>(noc_core.y));
                }
                signaler_args.push_back(static_cast<uint32_t>(srs_fused_op_signaler->num_fused_op_cores_to_signal));
                for (const auto& noc_core : srs_fused_op_signaler->fused_op_receiver_cores_noc) {
                    signaler_args.push_back(static_cast<uint32_t>(noc_core.x));
                    signaler_args.push_back(static_cast<uint32_t>(noc_core.y));
                }
                signaler_args.push_back(
                    static_cast<uint32_t>(srs_fused_op_signaler->fused_op_receiver_signal_semaphore));
                signaler_args.push_back(1);  // mcast_signal_op_cores
            }
            args.append(signaler_args);
        };

        KernelDescriptor::RTArgList in0_args;
        in0_args.push_back(in0_buffer);
        in0_args.push_back(in2_buffer);
        in0_args.push_back(in3_buffer);
        in0_args.append({
            static_cast<uint32_t>(is_in0_sink),
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
        });
        append_ternary_and_outputs(in0_args);
        append_signaler_args(in0_args);
        if (in1_idx == 0) {
            // in0 sender
            in0_sender_kernel.emplace_runtime_args(core, in0_args);
        } else {
            // in0 receiver
            in0_receiver_kernel.emplace_runtime_args(core, in0_args);
        }

        KernelDescriptor::RTArgList in1_args;
        in1_args.push_back(in1_buffer);
        in1_args.push_back(in2_buffer);
        in1_args.append({
            static_cast<uint32_t>(is_in1_sink),
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
        });
        append_ternary_and_outputs(in1_args);
        append_signaler_args(in1_args);
        if (in0_idx == 0) {
            // in1 sender
            in1_sender_kernel.emplace_runtime_args(core, in1_args);
        } else {
            // in1 receiver
            in1_receiver_kernel.emplace_runtime_args(core, in1_args);
        }

        std::vector<uint32_t> compute_runtime_args = {
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
        };
        if (use_fused_ternary) {
            // The scalar is part of the program hash (matmul_struct), so it never needs a cache-hit patch
            static_assert(sizeof(float) == sizeof(uint32_t), "fused_ternary_scalar is passed as one 32-bit arg");
            compute_runtime_args.push_back(std::bit_cast<uint32_t>(fused_ternary_scalar.value()));
            compute_runtime_args.push_back(broadcast_ternary_b);
        }
        compute_kernel.runtime_args.emplace_back(core, std::move(compute_runtime_args));
    }

    TT_FATAL(
        desc.kernels.size() == first_kernel_index,
        "minimal_matmul fabric-bound kernels must be appended contiguously from index {}",
        first_kernel_index);
    desc.kernels.push_back(std::move(in0_sender_kernel));
    desc.kernels.push_back(std::move(in0_receiver_kernel));
    desc.kernels.push_back(std::move(in1_sender_kernel));
    desc.kernels.push_back(std::move(in1_receiver_kernel));
    desc.kernels.push_back(std::move(compute_kernel));
    TT_FATAL(
        desc.kernels.size() == first_kernel_index + layout::kNumKernels,
        "minimal_matmul fabric-bound helper must append exactly {} kernels",
        layout::kNumKernels);
}

void minimal_matmul_fabric_bound_patch_runtime_args(
    tt::tt_metal::Program& program,
    uint32_t first_kernel_index,
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<const Tensor>& ag_input_tensor,
    const std::optional<const Tensor>& fused_ternary_input_a,
    const std::optional<const Tensor>& fused_ternary_input_b,
    const std::vector<Tensor>& output_tensors) {
    namespace layout = minimal_matmul_fabric_bound_layout;

    const uint32_t input_address = input_tensor.buffer()->address();
    const uint32_t weight_address = weight_tensor.buffer()->address();
    const uint32_t bias_address = bias_tensor.has_value() ? bias_tensor.value().buffer()->address() : 0;
    const uint32_t ag_input_address = ag_input_tensor.has_value() ? ag_input_tensor.value().buffer()->address() : 0;
    const bool has_fused_ternary = fused_ternary_input_a.has_value() && fused_ternary_input_b.has_value();
    const uint32_t ternary_a_address = has_fused_ternary ? fused_ternary_input_a.value().buffer()->address() : 0;
    const uint32_t ternary_b_address = has_fused_ternary ? fused_ternary_input_b.value().buffer()->address() : 0;
    std::vector<uint32_t> output_addresses;
    output_addresses.reserve(output_tensors.size());
    for (const auto& output_tensor : output_tensors) {
        output_addresses.push_back(output_tensor.buffer()->address());
    }

    const auto patch_ternary_and_outputs = [&](auto& args, uint32_t fixed_args) {
        uint32_t output_start = fixed_args;
        if (has_fused_ternary) {
            args[fixed_args + layout::kTernaryAAddrOffset] = ternary_a_address;
            args[fixed_args + layout::kTernaryBAddrOffset] = ternary_b_address;
            output_start += layout::kTernaryArgs;
        }
        for (size_t out_idx = 0; out_idx < output_addresses.size(); ++out_idx) {
            args[output_start + out_idx] = output_addresses[out_idx];
        }
    };

    // Senders and receivers share one argument layout, so both get every address.
    for (const uint32_t kernel : {layout::kIn0SenderKernel, layout::kIn0ReceiverKernel}) {
        for_each_core_runtime_args(program, first_kernel_index + kernel, [&](auto& in0_args) {
            in0_args[layout::kIn0InputAddrArg] = input_address;
            in0_args[layout::kIn0BiasAddrArg] = bias_address;
            in0_args[layout::kIn0AgInputAddrArg] = ag_input_address;
            patch_ternary_and_outputs(in0_args, layout::kIn0FixedArgs);
        });
    }
    for (const uint32_t kernel : {layout::kIn1SenderKernel, layout::kIn1ReceiverKernel}) {
        for_each_core_runtime_args(program, first_kernel_index + kernel, [&](auto& in1_args) {
            in1_args[layout::kIn1WeightAddrArg] = weight_address;
            in1_args[layout::kIn1BiasAddrArg] = bias_address;
            patch_ternary_and_outputs(in1_args, layout::kIn1FixedArgs);
        });
    }
}

}  // namespace ttnn::experimental::prim
