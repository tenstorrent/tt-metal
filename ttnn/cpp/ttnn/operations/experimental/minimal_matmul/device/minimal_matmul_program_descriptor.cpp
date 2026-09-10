// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Program construction for the standalone minimal_matmul op.
//
// This is the ProgramDescriptor translation of minimal_matmul_factory_helper_common in
// minimal_matmul_program_factory.cpp. That legacy Program&-based helper still exists, but is now
// reached only by the CCL composites that build a fused matmul + reduce-scatter program
// (minimal_matmul_strided_reduce_scatter_async).
//

#include "minimal_matmul_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <algorithm>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::experimental::prim {

using tt::tt_metal::Buffer;
using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::DataMovementConfigDescriptor;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::SemaphoreDescriptor;

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
    const std::optional<Tensor>& bias_tensor,
    const std::optional<Tensor>& in3_tensor = std::nullopt,
    const std::optional<Tensor>& ternary_a_tensor = std::nullopt,
    const std::optional<Tensor>& ternary_b_tensor = std::nullopt) {
    tt::tt_metal::TensorAccessorArgs(*main_tensor.buffer()).append_to(args);
    for (const auto& output_tensor : output_tensors) {
        tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(args);
    }
    if (bias_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*bias_tensor.value().buffer()).append_to(args);
    }
    // The in0 second source must come before ternary to match kernel accessor order
    if (in3_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*in3_tensor.value().buffer()).append_to(args);
    }
    if (ternary_a_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*ternary_a_tensor.value().buffer()).append_to(args);
    }
    if (ternary_b_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*ternary_b_tensor.value().buffer()).append_to(args);
    }
}

// The define builders below (and throttle_mm_perf) work in terms of std::map; KernelDescriptor
// wants a vector of pairs.
KernelDescriptor::Defines to_defines(const std::map<std::string, std::string>& defines) {
    return KernelDescriptor::Defines(defines.begin(), defines.end());
}

Buffer* buffer_or_null(const std::optional<Tensor>& tensor) {
    return tensor.has_value() ? tensor.value().buffer() : nullptr;
}

// Runtime-arg layout, shared by create_descriptor and override_runtime_arguments so the two
// cannot drift. Mirrors the kernels' get_arg_val order in dm_in0_sender.cpp / dm_in1_sender_out.cpp:
//
//   in0: [in0, in2, in3, is_sink, noc(4), tile_ranges(4), defer, max_defer,
//         (ternary_a, ternary_b, broadcast_b)?, out(N)...]
//   in1: [in1, in2,      is_sink, noc(4), tile_ranges(4), defer, max_defer,
//         (ternary_a, ternary_b, broadcast_b)?, out(N)...]
constexpr uint32_t kIn0BufferIdx = 0;
constexpr uint32_t kIn0BiasIdx = 1;
constexpr uint32_t kIn0SecondSourceIdx = 2;
constexpr uint32_t kIn0FixedArgCount = 14;

constexpr uint32_t kIn1BufferIdx = 0;
constexpr uint32_t kIn1BiasIdx = 1;
constexpr uint32_t kIn1FixedArgCount = 13;

constexpr uint32_t kTernaryArgCount = 3;  // ternary_a, ternary_b, broadcast_b

// Kernel indices are the descriptor's push order at the bottom of create_descriptor.
constexpr uint32_t kIn0SenderKernel = 0;
constexpr uint32_t kIn0ReceiverKernel = 1;
constexpr uint32_t kIn1SenderKernel = 2;
constexpr uint32_t kIn1ReceiverKernel = 3;

}  // namespace

ProgramDescriptor MinimalMatmulDeviceOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    const Tensor& weight_tensor = tensor_args.weight_tensor;
    const std::optional<Tensor>& bias_tensor = tensor_args.bias_tensor;
    const std::optional<Tensor>& optional_input_tensor = tensor_args.optional_input_tensor;
    const std::optional<Tensor>& fused_ternary_input_a = tensor_args.fused_ternary_input_a;
    const std::optional<Tensor>& fused_ternary_input_b = tensor_args.fused_ternary_input_b;
    const std::vector<Tensor>& output_tensors = tensor_return_value;

    const auto& fused_activation = operation_attributes.fused_activation;
    const auto& config = operation_attributes.config;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;
    const bool fuse_swiglu = operation_attributes.fuse_swiglu;
    const uint32_t N_chunks = static_cast<uint32_t>(operation_attributes.chunks);

    auto* device = input_tensor.device();

    // Fused concat (concat-free): in0's K is sourced from input_tensor (prefix K-tiles) then
    // optional_input_tensor (suffix), via the in0 second-source (in3) read path, instead of a
    // materialized concat. The split point is input_tensor's own K width.
    const bool two_input_split = optional_input_tensor.has_value();

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
    // Fold activation (LHS) upper dimensions into rows: M_total = prod(upper dims) * M.
    // M is derived from input_tensor's OWN K width (K_in); for fused concat that's only the prefix
    // half, so the matmul contraction K must instead span the full weight K (both concat halves).
    uint32_t K_in = in0_tensor_shape[-1];
    uint32_t M = input_tensor.physical_volume() / K_in;
    uint32_t K = two_input_split ? static_cast<uint32_t>(in1_tensor_shape[-2]) : K_in;
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

    ProgramDescriptor desc;

    /**
     * Semaphores. The ids are the sequential ids CreateSemaphore would have handed out, and are
     * passed to the kernels as compile-time args below; descriptor and kernel agree on the literal.
     */
    constexpr uint32_t in0_sender_semaphore_id = 0;
    constexpr uint32_t in0_receiver_semaphore_id = 1;
    constexpr uint32_t in0_valid_semaphore_id = 2;
    constexpr uint32_t in1_sender_semaphore_id = 3;
    constexpr uint32_t in1_receiver_semaphore_id = 4;
    constexpr uint32_t in1_valid_semaphore_id = 5;

    for (auto [id, initial_value] : {
             std::pair{in0_sender_semaphore_id, INVALID},
             std::pair{in0_receiver_semaphore_id, INVALID},
             std::pair{in0_valid_semaphore_id, VALID},
             std::pair{in1_sender_semaphore_id, INVALID},
             std::pair{in1_receiver_semaphore_id, INVALID},
             std::pair{in1_valid_semaphore_id, VALID},
         }) {
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = CoreRangeSet(core_grid),
            .initial_value = initial_value});
    }

    /**
     * Circular buffers. None are tensor-backed or globally allocated, so none need patching on a
     * program-cache hit.
     */
    auto push_cb = [&desc, &core_grid](uint32_t cb_id, uint32_t page_size, uint32_t num_pages, tt::DataFormat format) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_pages * page_size,
            .core_ranges = CoreRangeSet(core_grid),
            .format_descriptors = {CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_id),
                .data_format = format,
                .page_size = page_size,
            }}});
    };

    uint32_t in0_cb_id = tt::CBIndex::c_0;
    push_cb(in0_cb_id, in0_tile_size, in0_cb_num_tiles, in0_data_format);

    uint32_t in1_cb_id = tt::CBIndex::c_1;
    push_cb(in1_cb_id, in1_tile_size, in1_cb_num_tiles, in1_data_format);

    uint32_t out_cb_id = tt::CBIndex::c_2;
    push_cb(out_cb_id, out_tile_size, out_cb_num_tiles, output_data_format);

    uint32_t intermediate_cb_id = tt::CBIndex::c_3;
    push_cb(intermediate_cb_id, intermediate_tile_size, interm_cb_num_tiles, intermediate_data_format);

    if (use_bias) {
        uint32_t in2_cb_id = tt::CBIndex::c_4;
        push_cb(in2_cb_id, in2_tile_size, in2_cb_num_tiles, in2_data_format);
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

        push_cb(ternary_a_cb_id, ternary_a_tile_size, ternary_a_cb_num_tiles, ternary_a_data_format);

        // Fused ternary input C - circular buffer c_6
        auto ternary_c_data_format =
            tt::tt_metal::datatype_to_dataformat_converter(fused_ternary_input_b.value().dtype());
        auto ternary_c_tile_size = tt::tile_size(ternary_c_data_format);
        uint32_t ternary_c_cb_num_tiles = N_block_tiles;  // Single row (like bias), broadcast across M

        push_cb(ternary_c_cb_id, ternary_c_tile_size, ternary_c_cb_num_tiles, ternary_c_data_format);

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

    // Fused concatenation of in0: only the in0 SENDER reads from the two source buffers; the
    // receiver gets the assembled block by mcast and needs no split.
    std::map<std::string, std::string> in0_concat_defines;
    if (two_input_split) {
        in0_concat_defines = defines;
        in0_concat_defines["IN0_VIRTUAL_CONCAT"] = "1";
        // Split point = input_tensor's own K width (prefix half), in tiles.
        in0_concat_defines["IN0_K_SPLIT_TILES"] = std::to_string(K_in / tt::constants::TILE_WIDTH);
    }

    // in3 is the second in0 source buffer: for fused concat, the second concat half supplied via
    // optional_input_tensor.
    auto in3_data_format = two_input_split
                               ? tt::tt_metal::datatype_to_dataformat_converter(optional_input_tensor.value().dtype())
                               : in1_data_format;
    auto in3_tile_size = tt::tile_size(in3_data_format);

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
        true,               // is_injector_core
        N_chunks,           // N_chunks
        N_tiles_per_chunk,  // N_tiles_per_chunk
        in3_tile_size,
    };
    // The in0 sender's second source: (fused concat) optional_input_tensor.
    std::optional<Tensor> in0_sender_in3_tensor;
    if (two_input_split) {
        in0_sender_in3_tensor = optional_input_tensor.value();
    }
    append_accessors(
        in0_sender_compile_time_args,
        input_tensor,
        output_tensors,
        bias_tensor,
        in0_sender_in3_tensor,
        fused_ternary_input_a,
        fused_ternary_input_b);

    KernelDescriptor in0_sender_kernel{
        .kernel_source = "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = CoreRangeSet(in0_sender_cores),
        .compile_time_args = in0_sender_compile_time_args,
        .defines = to_defines(two_input_split ? in0_concat_defines : defines),
        .config = DataMovementConfigDescriptor{.processor = in0_risc, .noc = in0_noc}};

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
        std::nullopt,  // no second in0 source for in0_receiver
        fused_ternary_input_a,
        fused_ternary_input_b);

    KernelDescriptor in0_receiver_kernel{
        .kernel_source = "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = CoreRangeSet(in0_receiver_cores),
        .compile_time_args = in0_receiver_compile_time_args,
        .defines = to_defines(defines),
        .config = DataMovementConfigDescriptor{.processor = in0_risc, .noc = in0_noc}};

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
        std::nullopt,  // no second in0 source for in1_sender
        fused_ternary_input_a,
        fused_ternary_input_b);

    KernelDescriptor in1_sender_kernel{
        .kernel_source = "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_sender_out.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = CoreRangeSet(in1_sender_cores),
        .compile_time_args = in1_sender_compile_time_args,
        .defines = to_defines(defines),
        .config = DataMovementConfigDescriptor{.processor = in1_risc, .noc = in1_noc}};

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
        std::nullopt,  // no second in0 source for in1_receiver
        fused_ternary_input_a,
        fused_ternary_input_b);

    KernelDescriptor in1_receiver_kernel{
        .kernel_source = "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_sender_out.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = CoreRangeSet(in1_receiver_cores),
        .compile_time_args = in1_receiver_compile_time_args,
        .defines = to_defines(defines),
        .config = DataMovementConfigDescriptor{.processor = in1_risc, .noc = in1_noc}};

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
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, compute_defines, ttnn::get_throttle_level(compute_kernel_config));

    KernelDescriptor compute_kernel{
        .kernel_source = "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp",
        .source_type = KernelDescriptor::SourceType::FILE_PATH,
        .core_ranges = CoreRangeSet(core_grid),
        .compile_time_args = compute_compile_time_args,
        .defines = to_defines(compute_defines),
        .config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode}};

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

    // Buffer pointers, not addresses: each one pushed below registers a BufferBinding that the
    // framework re-patches on a program-cache hit. An absent optional tensor passes a null
    // Buffer*, for which the framework emits 0 and registers no binding - matching the legacy
    // helper, which wrote a literal 0 into the same slot.
    Buffer* in0_buffer = input_tensor.buffer();
    Buffer* in1_buffer = weight_tensor.buffer();
    Buffer* in2_buffer = buffer_or_null(bias_tensor);
    Buffer* in3_buffer = two_input_split ? optional_input_tensor.value().buffer() : nullptr;
    Buffer* ternary_a_buffer = use_fused_ternary ? fused_ternary_input_a.value().buffer() : nullptr;
    Buffer* ternary_b_buffer = use_fused_ternary ? fused_ternary_input_b.value().buffer() : nullptr;

    uint32_t ternary_b_broadcast = 0u;
    if (use_fused_ternary) {
        uint32_t ternary_b_M_tiles = fused_ternary_input_b.value().padded_shape()[-2] / tt::constants::TILE_HEIGHT;
        ternary_b_broadcast = ternary_b_M_tiles == 1 ? 1u : 0u;
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

        KernelDescriptor::RTArgList in0_args;
        in0_args.push_back(in0_buffer);
        in0_args.push_back(in2_buffer);
        in0_args.push_back(in3_buffer);
        in0_args.push_back(static_cast<uint32_t>(is_in0_sink));
        in0_args.push_back((std::uint32_t)in0_next_core_physical.x);  // in0_dest_noc_x
        in0_args.push_back((std::uint32_t)in0_next_core_physical.y);  // in0_dest_noc_y
        in0_args.push_back((std::uint32_t)in0_prev_core_physical.x);  // in0_sender_noc_x
        in0_args.push_back((std::uint32_t)in0_prev_core_physical.y);  // in0_sender_noc_y
        in0_args.push_back(M_start_tile);
        in0_args.push_back(M_end_tile);
        in0_args.push_back(N_start_tile);
        in0_args.push_back(N_end_tile);
        in0_args.push_back(defer_write_k_block);
        in0_args.push_back(max_defer_write_k_block);
        // Add ternary addresses if present (after defer_write_k_block, before output addresses)
        if (use_fused_ternary) {
            in0_args.push_back(ternary_a_buffer);
            in0_args.push_back(ternary_b_buffer);
            in0_args.push_back(ternary_b_broadcast);
        }
        // Add output addresses at the end (unified layout for both regular and split)
        for (const auto& output_tensor : output_tensors) {
            in0_args.push_back(output_tensor.buffer());
        }
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
        in1_args.push_back(static_cast<uint32_t>(is_in1_sink));
        in1_args.push_back((std::uint32_t)in1_next_core_physical.x);  // in1_dest_noc_x
        in1_args.push_back((std::uint32_t)in1_next_core_physical.y);  // in1_dest_noc_y
        in1_args.push_back((std::uint32_t)in1_prev_core_physical.x);  // in1_sender_noc_x
        in1_args.push_back((std::uint32_t)in1_prev_core_physical.y);  // in1_sender_noc_y
        in1_args.push_back(M_start_tile);
        in1_args.push_back(M_end_tile);
        in1_args.push_back(N_start_tile);
        in1_args.push_back(N_end_tile);
        in1_args.push_back(defer_write_k_block);
        in1_args.push_back(max_defer_write_k_block);
        // Add ternary addresses if present (after defer_write_k_block, before output addresses)
        if (use_fused_ternary) {
            in1_args.push_back(ternary_a_buffer);
            in1_args.push_back(ternary_b_buffer);
            in1_args.push_back(ternary_b_broadcast);
        }
        // Add output addresses at the end (unified layout for both regular and split)
        for (const auto& output_tensor : output_tensors) {
            in1_args.push_back(output_tensor.buffer());
        }
        if (in0_idx == 0) {
            // in1 sender
            in1_sender_kernel.emplace_runtime_args(core, in1_args);
        } else {
            // in1 receiver
            in1_receiver_kernel.emplace_runtime_args(core, in1_args);
        }

        // No buffers in the compute args, so no bindings are needed here.
        std::vector<uint32_t> compute_runtime_args = {
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
        };
        if (use_fused_ternary) {
            // fused_ternary_scalar is part of the program hash, so a cache hit guarantees this
            // value is unchanged; it does not need re-applying per dispatch.
            compute_runtime_args.push_back(
                *reinterpret_cast<const uint32_t*>(&operation_attributes.fused_ternary_scalar.value()));
            compute_runtime_args.push_back(ternary_b_broadcast);
        }
        compute_kernel.runtime_args.emplace_back(core, std::move(compute_runtime_args));
    }

    desc.kernels.push_back(std::move(in0_sender_kernel));
    desc.kernels.push_back(std::move(in0_receiver_kernel));
    desc.kernels.push_back(std::move(in1_sender_kernel));
    desc.kernels.push_back(std::move(in1_receiver_kernel));
    desc.kernels.push_back(std::move(compute_kernel));

    // Cache-miss-only guard that the arg lists above still match the layout constants
    // override_runtime_arguments indexes with. Cheap here, and it turns an arg-index drift into a
    // loud failure instead of a stale address.
    const uint32_t ternary_args = use_fused_ternary ? kTernaryArgCount : 0;
    const uint32_t expected_in0_args = kIn0FixedArgCount + ternary_args + output_tensors.size();
    const uint32_t expected_in1_args = kIn1FixedArgCount + ternary_args + output_tensors.size();
    for (auto [kernel_idx, expected] : {
             std::pair{kIn0SenderKernel, expected_in0_args},
             std::pair{kIn0ReceiverKernel, expected_in0_args},
             std::pair{kIn1SenderKernel, expected_in1_args},
             std::pair{kIn1ReceiverKernel, expected_in1_args},
         }) {
        const auto& runtime_args = desc.kernels[kernel_idx].runtime_args;
        TT_FATAL(
            runtime_args.empty() || runtime_args.front().second.size() == expected,
            "minimal_matmul descriptor kernel {} emitted {} runtime args but the layout constants "
            "used by override_runtime_arguments expect {}",
            kernel_idx,
            runtime_args.front().second.size(),
            expected);
    }

    return desc;
}

// Surgical cache-hit refresh. Everything except the buffer addresses is derived from hashed
// inputs, so a cache hit guarantees it is already correct and is deliberately left untouched.
//
// Exists purely for dispatch cost: apply_resolved_bindings costs one GetRuntimeArgs lookup per (kernel, core) - 260
// here - vs. five hoisted grid references, ~7% cheaper. Bindings stay declared for the cache miss, then ignored.
void MinimalMatmulDeviceOperation::ProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    const std::optional<Tensor>& bias_tensor = tensor_args.bias_tensor;
    const bool two_input_split = tensor_args.optional_input_tensor.has_value();
    const bool use_fused_ternary =
        tensor_args.fused_ternary_input_a.has_value() && tensor_args.fused_ternary_input_b.has_value();

    // Only two structural facts are needed to walk the cores, and both derive from hashed inputs,
    // so they match the cache-miss dispatch by construction. Deliberately NOT re-deriving the
    // block sizes, core order or per-core tile ranges: recomputing a full work split on every hit
    // is what the descriptor migration is trying to avoid.
    auto* device = input_tensor.device();
    const auto grid_size = operation_attributes.config.has_value()
                               ? operation_attributes.config.value().compute_with_storage_grid_size
                               : device->compute_with_storage_grid_size();

    const uint32_t K_in = input_tensor.padded_shape()[-1];
    const uint32_t M = input_tensor.physical_volume() / K_in;
    const uint32_t N = tensor_args.weight_tensor.padded_shape()[-1];
    const bool transpose_core_grid = M > N;

    const uint32_t in0_addr = input_tensor.buffer()->address();
    const uint32_t in1_addr = tensor_args.weight_tensor.buffer()->address();
    const uint32_t in2_addr = bias_tensor.has_value() ? bias_tensor.value().buffer()->address() : 0;
    const uint32_t in3_addr = two_input_split ? tensor_args.optional_input_tensor.value().buffer()->address() : 0;

    const uint32_t ternary_args = use_fused_ternary ? kTernaryArgCount : 0;
    const uint32_t in0_ternary_a_idx = kIn0FixedArgCount;
    const uint32_t in1_ternary_a_idx = kIn1FixedArgCount;
    const uint32_t in0_out_addr_start = kIn0FixedArgCount + ternary_args;
    const uint32_t in1_out_addr_start = kIn1FixedArgCount + ternary_args;

    uint32_t ternary_a_addr = 0;
    uint32_t ternary_b_addr = 0;
    if (use_fused_ternary) {
        ternary_a_addr = tensor_args.fused_ternary_input_a.value().buffer()->address();
        ternary_b_addr = tensor_args.fused_ternary_input_b.value().buffer()->address();
    }

    // Hoisted grid references (pitfall 5: `auto&`, not `auto` - the by-value form deep-copies the
    // whole per-core arg grid). Taking these five once, rather than one lookup per (kernel, core),
    // is the entire point of this override.
    auto& in0_sender_runtime_args = GetRuntimeArgs(program, kIn0SenderKernel);
    auto& in0_receiver_runtime_args = GetRuntimeArgs(program, kIn0ReceiverKernel);
    auto& in1_sender_runtime_args = GetRuntimeArgs(program, kIn1SenderKernel);
    auto& in1_receiver_runtime_args = GetRuntimeArgs(program, kIn1ReceiverKernel);

    auto patch_tail = [&](auto& args, uint32_t ternary_a_idx, uint32_t out_addr_start) {
        if (use_fused_ternary) {
            args[ternary_a_idx] = ternary_a_addr;
            args[ternary_a_idx + 1] = ternary_b_addr;
        }
        for (size_t out_idx = 0; out_idx < tensor_return_value.size(); ++out_idx) {
            args[out_addr_start + out_idx] = tensor_return_value[out_idx].buffer()->address();
        }
    };

    for (uint32_t y = 0; y < grid_size.y; ++y) {
        for (uint32_t x = 0; x < grid_size.x; ++x) {
            const uint32_t in0_idx = transpose_core_grid ? x : y;
            const uint32_t in1_idx = transpose_core_grid ? y : x;

            auto& in0_args = (in1_idx == 0 ? in0_sender_runtime_args : in0_receiver_runtime_args)[x][y];
            in0_args[kIn0BufferIdx] = in0_addr;
            in0_args[kIn0BiasIdx] = in2_addr;
            in0_args[kIn0SecondSourceIdx] = in3_addr;
            patch_tail(in0_args, in0_ternary_a_idx, in0_out_addr_start);

            auto& in1_args = (in0_idx == 0 ? in1_sender_runtime_args : in1_receiver_runtime_args)[x][y];
            in1_args[kIn1BufferIdx] = in1_addr;
            in1_args[kIn1BiasIdx] = in2_addr;
            patch_tail(in1_args, in1_ternary_a_idx, in1_out_addr_start);
        }
    }
}

}  // namespace ttnn::experimental::prim
