// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_minimal_matmul_async_device_operation.hpp"
#include "all_gather_minimal_matmul_async_program_factory.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <algorithm>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tuple>
#include <utility>
#include <vector>

namespace detail {

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

void fabric_mux_connection_ct_args(
    const uint32_t num_workers_per_link,
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    std::vector<uint32_t>& worker_ct_args) {
    worker_ct_args.push_back(mux_kernel_config.get_num_buffers(channel_type));  // fabric_mux_num_buffers_per_channel
    worker_ct_args.push_back(
        mux_kernel_config.get_buffer_size_bytes(channel_type));        // fabric_mux_channel_buffer_size_bytes
    worker_ct_args.push_back(mux_kernel_config.get_status_address());  // fabric_mux_status_address
    worker_ct_args.push_back(
        mux_kernel_config.get_termination_signal_address());  // fabric_mux_termination_signal_address
    worker_ct_args.push_back(num_workers_per_link);           // num_mux_clients
}

void fabric_mux_connection_rt_args(
    const bool mux_connection_valid,
    const bool is_termination_master,
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    const CoreCoord& mux_virtual_core,
    const uint32_t worker_id,
    const CoreCoord& worker_logical_core,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    tt::tt_metal::Program& program,
    CoreCoord termination_master_virtual_core,
    uint32_t num_mux_clients,
    uint32_t termination_sync_id,
    std::vector<uint32_t>& worker_rt_args) {
    worker_rt_args.push_back(mux_connection_valid);   // mux_connection_valid
    worker_rt_args.push_back(is_termination_master);  // is_termination_master
    worker_rt_args.push_back(mux_virtual_core.x);     // fabric_mux_x
    worker_rt_args.push_back(mux_virtual_core.y);     // fabric_mux_y
    worker_rt_args.push_back(
        mux_kernel_config.get_channel_base_address(channel_type, worker_id));  // fabric_mux_channel_base_address
    worker_rt_args.push_back(
        mux_kernel_config.get_connection_info_address(channel_type, worker_id));  // fabric_mux_connection_info_address
    worker_rt_args.push_back(mux_kernel_config.get_connection_handshake_address(
        channel_type, worker_id));  // fabric_mux_connection_handshake_address
    worker_rt_args.push_back(
        mux_kernel_config.get_flow_control_address(channel_type, worker_id));  // fabric_mux_flow_control_address
    worker_rt_args.push_back(
        mux_kernel_config.get_buffer_index_address(channel_type, worker_id));  // fabric_mux_buffer_index_address
    worker_rt_args.push_back(
        mux_kernel_config.get_channel_credits_stream_id(channel_type, worker_id));  // fabric_mux_channel_id
    worker_rt_args.push_back(termination_sync_id);  // termination_sync_address (shared, uniform L1 addr)
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // local_fabric_mux_status_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // local_flow_control_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // local_teardown_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // local_buffer_index_address
    worker_rt_args.push_back(termination_master_virtual_core.x);                    // termination_master_noc_x
    worker_rt_args.push_back(termination_master_virtual_core.y);                    // termination_master_noc_y
    worker_rt_args.push_back(num_mux_clients);                                      // num_mux_clients (this mux)
}

// Append tensor accessors in a consistent order
static inline void append_accessors(
    std::vector<uint32_t>& args,
    const ttnn::Tensor& main_tensor,
    const std::vector<ttnn::Tensor>& output_tensors,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const ttnn::Tensor& ag_input_tensor,
    const std::optional<const ttnn::Tensor>& ternary_a_tensor = std::nullopt,
    const std::optional<const ttnn::Tensor>& ternary_b_tensor = std::nullopt,
    bool is_injector_core = false,
    const std::optional<const ttnn::Tensor>& fsdp_local_weight_tensor = std::nullopt) {
    tt::tt_metal::TensorAccessorArgs(*main_tensor.buffer()).append_to(args);
    for (const auto& output_tensor : output_tensors) {
        tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(args);
    }
    if (bias_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*bias_tensor.value().buffer()).append_to(args);
    }
    if (is_injector_core) {
        tt::tt_metal::TensorAccessorArgs(*ag_input_tensor.buffer()).append_to(args);
    }
    if (fsdp_local_weight_tensor.has_value()) {
        // FSDP-sharded local weight (for in1 kernel reading its own K-slice from DRAM)
        tt::tt_metal::TensorAccessorArgs(*fsdp_local_weight_tensor.value().buffer()).append_to(args);
    }
    if (ternary_a_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*ternary_a_tensor.value().buffer()).append_to(args);
    }
    if (ternary_b_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*ternary_b_tensor.value().buffer()).append_to(args);
    }
}

ttnn::experimental::prim::AllGatherMinimalMatmulAsyncProgramFactory::shared_variables_t
all_gather_minimal_matmul_async_factory_helper(
    tt::tt_metal::Program& program,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& fused_activation,
    const std::optional<const ttnn::experimental::prim::MinimalMatmulConfig>& config,
    const std::vector<ttnn::Tensor>& mm_output_tensors,
    const ttnn::Tensor& ag_output_tensor,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    const ttnn::MeshCoordinate& sender_device_coord,
    const std::optional<ttnn::MeshCoordinate>& forward_coord,
    const std::optional<ttnn::MeshCoordinate>& backward_coord,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<ttnn::GlobalSemaphore>& semaphore,
    //    const std::optional<ttnn::GlobalSemaphore>& barrier_semaphore,
    //    bool using_persistent_buffers,
    const bool force_transpose,
    const uint32_t num_workers_per_link,
    const uint32_t num_buffers_per_channel,
    uint32_t N_chunks,
    std::optional<float> fused_ternary_scalar,
    const std::optional<const ttnn::Tensor>& fused_ternary_input_a,
    const std::optional<const ttnn::Tensor>& fused_ternary_input_b,
    // FSDP fusion args (all unset/default when not fused)
    const std::optional<const ttnn::Tensor>& persistent_weight_buffer,
    const std::optional<ttnn::MeshCoordinate>& fsdp_forward_coord,
    const std::optional<ttnn::MeshCoordinate>& fsdp_backward_coord,
    uint32_t fsdp_ring_size,
    uint32_t fsdp_ring_index,
    const std::vector<ttnn::GlobalSemaphore>& fsdp_semaphore,
    ttnn::ccl::Topology fsdp_topology) {
    auto* device = input_tensor.device();

    if (!config.has_value()) {
        log_debug(tt::LogOp, "No config provided, using default block sizes and core grid");
    }

    auto grid_size =
        config.has_value() ? config.value().compute_with_storage_grid_size : device->compute_with_storage_grid_size();
    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    auto num_cores = core_grid.size();

    bool use_bias = bias_tensor.has_value();
    // Derive from scalar presence, matching validate (which guarantees tensors are also present).
    bool use_fused_ternary = fused_ternary_scalar.has_value();

    /**
     * Determine dataformats, compute kernel config
     */
    auto in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(ag_output_tensor.dtype());
    auto in0_tile_size = tt::tile_size(in0_data_format);
    auto in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(weight_tensor.dtype());
    auto in1_tile_size = tt::tile_size(in1_data_format);
    auto output_data_format = tt::tt_metal::datatype_to_dataformat_converter(mm_output_tensors[0].dtype());
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

    auto in0_tensor_shape = ag_output_tensor.padded_shape();
    auto in1_tensor_shape = weight_tensor.padded_shape();
    // Fold activation (LHS) upper dimensions into rows: M_total = prod(upper dims) * M
    uint32_t K = in0_tensor_shape[-1];
    uint32_t M = ag_output_tensor.physical_volume() / K;
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
    bool transpose_core_grid = force_transpose ? true : (M > N);

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

    // K is sharded equally across devices (validated upstream: K_tiles % ring_size == 0).
    // Within a device, K_per_device tiles are processed in K_blocks_per_device blocks (div_up).
    // The last block per device may be a "tail" block of K_block_tail_tiles < K_block_tiles
    // when K_block_tiles does not divide K_tiles_per_device; otherwise tail == K_block_tiles.
    uint32_t K_tiles_per_device = K_tiles / ring_size;
    uint32_t K_blocks_per_device = tt::div_up(K_tiles_per_device, K_block_tiles);
    uint32_t K_block_tail_tiles = K_tiles_per_device - (K_blocks_per_device - 1) * K_block_tiles;
    uint32_t K_blocks = K_blocks_per_device * ring_size;

    uint32_t M_tiles_per_core = padded_M_tiles / in0_parallel_axis_cores;
    uint32_t N_tiles_per_core = padded_N_tiles / in1_parallel_axis_cores;

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

    // CB sizing: fixed depth-2 for in0/in1, single-buffered out/interm/in2.
    //
    // This reverts the OOM-causing part of #44982 — the adaptive L1-budget sizing that grew
    // in0_cb depth up to 8x to fill L1 for perf. That budget is estimated from
    // lowest_occupied_compute_l1_address() (which falls back to the full l1_size_per_core when
    // no L1 buffers are placed yet) minus a fixed 64 KB margin; on systems where that estimate
    // is too optimistic, the depth-8 in0_cb exceeds available L1 and OOMs. Fixed depth-2 is
    // sized purely from the matmul blocking and is safe on all systems.
    //
    // The PR's single-buffered output is KEPT (not reverted to 2x): it is what lets large-N
    // shapes such as chunks=3 QKV fit in L1; re-doubling it OOMs those here.
    const uint32_t double_buffer_factor = 2;
    uint32_t in0_cb_num_tiles = in0_block_num_tiles * double_buffer_factor;
    uint32_t in1_cb_num_tiles = in1_block_num_tiles * double_buffer_factor;
    uint32_t out_cb_num_tiles = out_block_num_tiles;     // single-buffered
    uint32_t interm_cb_num_tiles = out_block_num_tiles;  // not double buffered
    uint32_t in2_cb_num_tiles = in2_block_num_tiles;     // not double buffered

    auto core_0_0 = CoreCoord{0, 0};
    auto core_0_1 = CoreCoord{0, 1};
    auto core_1_0 = CoreCoord{1, 0};
    auto core_endx_0 = CoreCoord{grid_size.x - 1, 0};
    auto core_0_endy = CoreCoord{0, grid_size.y - 1};
    auto core_endx_endy = CoreCoord{grid_size.x - 1, grid_size.y - 1};
    auto core_endx_2_endy = CoreCoord{grid_size.x - 3, grid_size.y - 1};
    auto core_endx_endy_2 = CoreCoord{grid_size.x - 1, grid_size.y - 3};
    auto core_0_endy_1 = CoreCoord{0, grid_size.y - 2};
    auto core_endx_1_0 = CoreCoord{grid_size.x - 2, 0};

    auto in0_sender_cores = CoreRange(core_0_0, transpose_core_grid ? core_endx_0 : core_0_endy);
    auto in0_receiver_cores_no_fabric =
        transpose_core_grid ? CoreRange(core_0_1, core_endx_endy_2) : CoreRange(core_1_0, core_endx_2_endy);
    auto in0_receiver_cores_fabric =
        transpose_core_grid ? CoreRange(core_0_endy_1, core_endx_endy) : CoreRange(core_endx_1_0, core_endx_endy);
    auto in1_sender_cores = CoreRange(core_0_0, transpose_core_grid ? core_0_endy : core_endx_0);
    auto in1_receiver_cores = CoreRange(transpose_core_grid ? core_1_0 : core_0_1, core_endx_endy);

    auto in0_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in0_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in0_valid_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, VALID);
    auto in1_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in1_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, INVALID);
    auto in1_valid_semaphore_id = tt::tt_metal::CreateSemaphore(program, core_grid, VALID);

    // Mux termination-sync semaphores. Created ONCE on the full grid so every mux client
    // (master and peers) sees this semaphore at the SAME L1 address — required because peers
    // increment it on the master's core. Per-core creation diverges when a core also carries
    // the other operand's mux semaphores (the in0/in1 sender overlap), which deadlocked teardown.
    // Separate ids for in0 vs in1 so an overlap core's two muxes don't share a termination slot.
    auto in0_term_sync_id = tt::tt_metal::CreateSemaphore(program, core_grid, 0);
    auto in1_term_sync_id = tt::tt_metal::CreateSemaphore(program, core_grid, 0);

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

    // Mux
    auto [num_targets_forward, num_targets_backward] =
        ttnn::ccl::get_forward_backward_line_mcast_distance(ring_size, ring_index, topology, false);
    auto [unicast_forward_args, unicast_backward_args] = ttnn::ccl::get_forward_backward_line_unicast_configuration(
        sender_device_coord, forward_coord, backward_coord, device);

    auto full_grid_size = device->compute_with_storage_grid_size();
    // Place the fsdp muxes in the last column (packed along the in1/Y axis) only when the matmul
    // actually leaves that column free — i.e. a transpose grid narrower than the physical grid (the
    // 7x8-on-an-8-wide-device case). On a full-width grid the last column is a worker column, so
    // fall back to the original bottom-row mux layout to avoid colliding with the matmul.
    const bool fsdp_mux_in_column = transpose_core_grid && (grid_size.x < full_grid_size.x);
    // The in0 sender axis is grouped by num_workers_per_link; each group owns one mux per
    // direction, so it must form exactly num_links groups (the last group may be short when the
    // axis isn't a multiple of num_workers_per_link — the per-mux client-count clamp handles that).
    {
        uint32_t in0_axis = transpose_core_grid ? grid_size.x : grid_size.y;
        TT_FATAL(
            (in0_axis + num_workers_per_link - 1) / num_workers_per_link == num_links,
            "The in0 sender axis ({}) must form exactly num_links ({}) groups of num_workers_per_link ({})",
            in0_axis,
            num_links,
            num_workers_per_link);
    }
    uint32_t num_mux_cores = num_links * 2;  // 2 being the number of directions
    TT_FATAL(
        (transpose_core_grid ? full_grid_size.y : full_grid_size.x) >= num_mux_cores,
        "The are not enough cores for the number of mux cores requested");

    // In-column fsdp mux row for a given (group, dir): dir SWAPPED (flip) AND the result shifted up by
    // one cyclically within the 2*num_links packed rows. The flip swaps which direction owns the
    // even vs odd row; the up-shift then moves both of a group's muxes to rows <= the group base so
    // every in1 write (hot tail and cold rank-0) is same-row or upward on NOC_1 — no "down" write
    // anywhere. Cost: the hot tail write becomes up-1/up-2 instead of same-row, and group 0's muxes
    // wrap to the top rows. Must stay in lockstep with the fsdp mux create/RT-args/sender loops.
    const auto fsdp_mux_col_row = [num_mux_cores](uint32_t group, uint32_t dir) {
        return (group * 2 + (1 - dir) + num_mux_cores - 1) % num_mux_cores;
    };

    // Scheme 4 single-row interleave: when the fsdp ring is fused and BOTH axes are uni-rings
    // (Linear), every device creates exactly one mux per (ring, link) — so all 8 muxes fit on the
    // single row R = full_grid_size.y - 1, interleaved by parity: in0 link g -> ODD col (nwpl*g + 1),
    // fsdp link g -> EVEN col (nwpl*g). This frees the fsdp column AND the second mux row the old
    // fallback consumed, letting the matmul reclaim the full 8x8. Collision-free for nwpl==2 (4 links
    // exactly fill cols 0..7). The create / RT-args / sender-wiring sites all route mux placement
    // through these helpers so they stay in lockstep.
    //
    // in0 on the ODD column (not even) is deliberate: in0's worker->mux write runs on NOC_0 (prefers
    // +x), and a group's chain-tail sender sits at the group's columns {2g, 2g+1}. Placing the mux at
    // 2g+1 keeps that write on the +x (with-grain) side; placing it at 2g (the group's left edge)
    // forced the odd-column sender to write -x against NOC_0, which measured ~11% slower on the fused
    // op. fsdp takes the even column; its relay write is column-matched (vertical) and m==0-gated, so
    // the +x/-x grain doesn't apply to it.
    const bool single_row_muxes = persistent_weight_buffer.has_value() && topology == ttnn::ccl::Topology::Linear &&
                                  fsdp_topology == ttnn::ccl::Topology::Linear;
    TT_FATAL(
        !single_row_muxes || num_workers_per_link == 2,
        "Scheme-4 single-row mux interleave assumes num_workers_per_link==2 (got {})",
        num_workers_per_link);
    const uint32_t single_mux_row = full_grid_size.y - 1;
    const auto in0_mux_logical = [&](uint32_t link, uint32_t dir) -> CoreCoord {
        if (single_row_muxes) {
            return CoreCoord(num_workers_per_link * link + 1, single_mux_row);  // odd col 2g+1 (NOC_0 +x-aligned)
        }
        uint32_t x = (num_workers_per_link * (link + 1)) - (1 - dir);
        if (x >= full_grid_size.x) {
            x -= full_grid_size.x;
        }
        return CoreCoord(x, full_grid_size.y - 1);
    };
    const auto fsdp_mux_logical = [&](uint32_t link, uint32_t dir) -> CoreCoord {
        if (single_row_muxes) {
            return CoreCoord(num_workers_per_link * link, single_mux_row);  // even col 2g
        }
        if (fsdp_mux_in_column) {
            return CoreCoord(full_grid_size.x - 1, fsdp_mux_col_row(link, dir));
        }
        uint32_t x = (num_workers_per_link * (link + 1)) - (1 - dir);
        if (x >= full_grid_size.x) {
            x -= full_grid_size.x;
        }
        x = (x == 0) ? (full_grid_size.x - 1) : (x - 1);
        return CoreCoord(x, full_grid_size.y - 2);
    };

    // Uni-ring (Linear): each device relays through exactly ONE mux direction — rank>0 (has a
    // backward neighbor) short-hops backward (dir=1); rank 0 (no backward neighbor) long-wraps
    // forward (dir=0). Create/wire only that direction so interior devices stop allocating an
    // opposite-direction mux they never send through (collapses the 64/68/72 core spread to a
    // flat 64). The same validity flag that already lets line-END devices skip their absent
    // direction now applies to every device. Ring is genuinely bidirectional, so keep the
    // neighbor-existence gate there.
    const uint32_t in0_uni_dir = backward_coord.has_value() ? 1u : 0u;
    const auto mux_connection_valid = [&backward_coord, &forward_coord, &topology, in0_uni_dir](const uint32_t dir) {
        if (topology == ttnn::ccl::Topology::Linear) {
            return dir == in0_uni_dir;
        }
        return (dir && backward_coord.has_value()) || (!dir && forward_coord.has_value());
    };

    std::vector<CoreRange> mux_core_ranges;
    for (uint32_t mux_id = 0; mux_id < num_mux_cores; ++mux_id) {
        uint32_t dir = mux_id % 2;  // 2 being the number of directions
        if (mux_connection_valid(dir)) {
            uint32_t link = mux_id / 2;  // 2 is the num directions
            mux_core_ranges.emplace_back(in0_mux_logical(link, dir));
        }
    }
    CoreRangeSet mux_core_range_set = CoreRangeSet(mux_core_ranges);

    const uint32_t l1_unreserved_base_address =
        device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;
    auto num_full_size_channels = num_workers_per_link;
    auto num_header_only_channels = 0;
    size_t buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_per_channel,
        0,
        buffer_size_bytes_full_size_channel,
        mux_base_l1_address);

    // all gather
    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = in0_tile_size;

    // scatter-write currently only supports 4 distinct noc addresses
    uint32_t max_target_noc_addresses_per_packet = 4;

    // for bfloat8_b, tile_num_per_link=6, we would need to send 2 packages, but they can be of size 3 instead of 4
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t num_tiles_to_write_per_packet = std::min(max_target_noc_addresses_per_packet, num_pages_per_packet);

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
    log_debug(tt::LogOp, "K_tiles_per_device: {}", K_tiles_per_device);
    log_debug(tt::LogOp, "K_blocks_per_device: {}", K_blocks_per_device);
    log_debug(tt::LogOp, "K_block_tail_tiles: {}", K_block_tail_tiles);
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
    std::map<std::string, std::string> in0_defines;
    std::map<std::string, std::string> in0_injector_defines;
    std::map<std::string, std::string> in0_fabric_defines;
    if (use_bias) {
        defines["FUSE_BIAS"] = "1";
    }
    if (use_fused_ternary) {
        defines["FUSE_TERNARY"] = "1";

        // Workaround for LLK bug (https://github.com/tenstorrent/tt-llk/issues/1338)
        // - If ternary_b / gate is float32 then use unary_bcast (row broadcast) + mul_binary_tile (accurate)
        // - If ternary_b / gate is bfloat16 then use mul_tiles_bcast (row broadcast) (workaround)
        if (fused_ternary_input_b.value().dtype() == ttnn::DataType::FLOAT32) {
            defines["TERNARY_B_IS_FLOAT32"] = "1";
        }
        constexpr uint32_t one_f32_bits = 0x3F800000u;
        uint32_t scalar_bits = 0;
        std::memcpy(&scalar_bits, &fused_ternary_scalar.value(), sizeof(uint32_t));
        if (scalar_bits == one_f32_bits) {
            defines["ADDCMUL_SCALAR_IS_ONE"] = "1";
        }
    }
    in0_defines = defines;
    in0_defines["READ_FROM_LOCAL_INPUT"] = "1";
    in0_defines["IS_IN0"] = "1";
    in0_fabric_defines = in0_defines;
    in0_fabric_defines["USE_MUX"] = "1";
    auto in1_defines = defines;
    if (persistent_weight_buffer.has_value()) {
        in1_defines["FSDP_FUSED"] = "1";
        in1_defines["IS_IN1"] = "1";
        in1_defines["USE_MUX"] = "1";
    }

    // Linear uni-ring routing: Dev 0's forward unicast routes N-1 hops to Dev N-1
    // (rather than 1 hop to Dev 1). fabric_set_unicast_route<false>(hdr, distance)
    // sends the packet to the device `distance` hops away, with no intermediate
    // deliveries.
    if (topology == ttnn::ccl::Topology::Linear && ring_index == 0) {
        unicast_forward_args[1] = ring_size - 1;  // distance_in_hops = N-1
    }

    const bool fsdp_fused = persistent_weight_buffer.has_value();

    // Skewed rank for compute_actual_k_block's `my_rank` (the consume rooting). With the (a+b)
    // skewed sharding, device (tp=ring_index, fsdp=fsdp_ring_index) holds local K-stripe
    // (ring_index + fsdp_ring_index) mod ring_size for BOTH operands, so both in0 (tp uni-ring)
    // and in1 (fsdp uni-ring) must root their K-walk there. Non-fsdp path: fsdp_ring_index==0 so
    // this is just ring_index. NOTE: routing/mcast still use the *physical* ring_index, not this.
    const uint32_t skewed_rank = fsdp_fused ? (ring_index + fsdp_ring_index) % ring_size : ring_index;

    // FSDP-axis ring metrics and fabric mux setup. Mirrors the in0 mux block (above) but
    // uses fsdp_ring_size / fsdp_ring_index / fsdp_topology and lives on row full_grid_size.y - 2.
    uint32_t fsdp_num_targets_forward = 0;
    uint32_t fsdp_num_targets_backward = 0;
    std::array<uint32_t, 2> fsdp_unicast_forward_args{};
    std::array<uint32_t, 2> fsdp_unicast_backward_args{};
    tt::tt_fabric::FabricMuxConfig fsdp_mux_kernel_config = mux_kernel_config;  // overwritten below if fsdp_fused
    tt::tt_metal::KernelHandle fsdp_mux_kernel_id{};
    // Uni-ring (Linear): mirror the in0 gate on the fsdp axis — create/wire only the single
    // direction each device relays through (rank>0 backward dir=1, rank 0 forward dir=0). Ring stays
    // bidirectional via the neighbor-existence gate.
    const uint32_t fsdp_uni_dir = fsdp_backward_coord.has_value() ? 1u : 0u;
    const auto fsdp_mux_connection_valid =
        [&fsdp_backward_coord, &fsdp_forward_coord, &fsdp_topology, fsdp_uni_dir](const uint32_t dir) {
            if (fsdp_topology == ttnn::ccl::Topology::Linear) {
                return dir == fsdp_uni_dir;
            }
            return (dir && fsdp_backward_coord.has_value()) || (!dir && fsdp_forward_coord.has_value());
        };
    if (fsdp_fused) {
        std::tie(fsdp_num_targets_forward, fsdp_num_targets_backward) =
            ttnn::ccl::get_forward_backward_line_mcast_distance(fsdp_ring_size, fsdp_ring_index, fsdp_topology, false);
        std::tie(fsdp_unicast_forward_args, fsdp_unicast_backward_args) =
            ttnn::ccl::get_forward_backward_line_unicast_configuration(
                sender_device_coord, fsdp_forward_coord, fsdp_backward_coord, device);

        // FSDP uni-ring routing (mirrors the in0 override above): fsdp Dev 0's forward unicast
        // routes N-1 hops to fsdp Dev N-1 to close the virtual ring.
        if (fsdp_topology == ttnn::ccl::Topology::Linear && fsdp_ring_index == 0) {
            fsdp_unicast_forward_args[1] = fsdp_ring_size - 1;  // distance_in_hops = N-1
        }

        std::vector<CoreRange> fsdp_mux_core_ranges;
        for (uint32_t mux_id = 0; mux_id < num_mux_cores; ++mux_id) {
            uint32_t dir = mux_id % 2;
            if (fsdp_mux_connection_valid(dir)) {
                uint32_t link = mux_id / 2;
                // Transpose grid (in1 chain runs along X): the fsdp muxes live in the LAST column
                // (freed by the 7-wide matmul), packed along the in1 sender axis (Y) at rows
                // 0..2*num_links-1. Each row's chain-tail sender then reaches its mux with a short
                // horizontal NOC_1 hop, instead of the old long vertical write down to a bottom mux
                // row (which ran against NOC_1's up/left bias). Non-transpose keeps the bottom-row
                // layout with the -1 NOC_1 shift.
                fsdp_mux_core_ranges.emplace_back(fsdp_mux_logical(link, dir));
            }
        }
        CoreRangeSet fsdp_mux_core_range_set = CoreRangeSet(fsdp_mux_core_ranges);

        fsdp_mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
            num_full_size_channels,
            num_header_only_channels,
            num_buffers_per_channel,
            0,
            buffer_size_bytes_full_size_channel,
            mux_base_l1_address);

        if (!fsdp_mux_core_ranges.empty()) {
            fsdp_mux_kernel_id = tt::tt_metal::CreateKernel(
                program,
                "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
                fsdp_mux_core_range_set,
                tt::tt_metal::DataMovementConfig{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt::tt_metal::NOC::RISCV_1_default,
                    .compile_args = fsdp_mux_kernel_config.get_fabric_mux_compile_time_args(),
                    .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});
        }
    }
    uint32_t in0_addr = ag_output_tensor.buffer()->address();
    // When FSDP fusion is active, in1 reads from the (op-managed) persistent_weight_buffer
    // — which holds the gathered weight [K_full, N_local] — rather than the FSDP-sharded
    // local weight slice. The FSDP gather kernel populates this buffer before in1 reads.
    uint32_t in1_addr =
        fsdp_fused ? persistent_weight_buffer.value().buffer()->address() : weight_tensor.buffer()->address();
    uint32_t in2_addr = use_bias ? bias_tensor.value().buffer()->address() : 0;
    // Note: Dataflow kernels can take a variable number of output tensors.
    // They are appended as a variable-length array at the end of the runtime-args:
    uint32_t in3_addr = input_tensor.buffer()->address();
    auto in3_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto in3_tile_size = tt::tile_size(in3_data_format);

    /**
     * Create kernels
     */

    bool in0_is_output_writer = transpose_core_grid;
    bool in1_is_output_writer = !transpose_core_grid;

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
        in0_is_output_writer,
        true,
        ring_size,
        skewed_rank,  // my_rank (skewed for fsdp; == ring_index otherwise)
        in3_tile_size,
        num_tiles_to_write_per_packet,
        num_targets_forward,
        num_targets_backward,
        static_cast<uint32_t>(topology),
        N_chunks,           // N_chunks
        N_tiles_per_chunk,  // N_tiles_per_chunk
        K_tiles_per_device,
        K_block_tail_tiles,
    };
    append_accessors(
        in0_sender_compile_time_args,
        ag_output_tensor,
        mm_output_tensors,
        bias_tensor,
        input_tensor,
        fused_ternary_input_a,
        fused_ternary_input_b,
        true);
    auto in0_sender_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/dm_in0_sender.cpp",
        in0_sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = in0_risc,
            .noc = in0_noc,
            .compile_args = in0_sender_compile_time_args,
            .defines = in0_defines});

    std::vector<uint32_t> in0_receiver_no_fabric_compile_time_args = {
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
        in0_is_output_writer,
        false,  // is_injector_core
        ring_size,
        skewed_rank,  // my_rank (skewed for fsdp; == ring_index otherwise)
        in3_tile_size,
        num_tiles_to_write_per_packet,
        num_targets_forward,
        num_targets_backward,
        static_cast<uint32_t>(topology),
        N_chunks,           // N_chunks
        N_tiles_per_chunk,  // N_tiles_per_chunk
        K_tiles_per_device,
        K_block_tail_tiles,
    };
    append_accessors(
        in0_receiver_no_fabric_compile_time_args,
        ag_output_tensor,
        mm_output_tensors,
        bias_tensor,
        input_tensor,
        fused_ternary_input_a,
        fused_ternary_input_b,
        true);

    auto in0_receiver_no_fabric_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/dm_in0_sender.cpp",
        in0_receiver_cores_no_fabric,
        tt::tt_metal::DataMovementConfig{
            .processor = in0_risc,
            .noc = in0_noc,
            .compile_args = in0_receiver_no_fabric_compile_time_args,
            .defines = in0_defines});

    std::vector<uint32_t> in0_receiver_fabric_compile_time_args = {
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
        in0_is_output_writer,
        false,  // is_injector_core
        ring_size,
        skewed_rank,  // my_rank (skewed for fsdp; == ring_index otherwise)
        in3_tile_size,
        num_tiles_to_write_per_packet,
        num_targets_forward,
        num_targets_backward,
        static_cast<uint32_t>(topology),
        N_chunks,           // N_chunks
        N_tiles_per_chunk,  // N_tiles_per_chunk
        K_tiles_per_device,
        K_block_tail_tiles,
    };
    fabric_mux_connection_ct_args(
        num_workers_per_link,
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mux_kernel_config,
        in0_receiver_fabric_compile_time_args);
    in0_receiver_fabric_compile_time_args.insert(
        in0_receiver_fabric_compile_time_args.end(), unicast_forward_args.begin(), unicast_forward_args.end());
    in0_receiver_fabric_compile_time_args.insert(
        in0_receiver_fabric_compile_time_args.end(), unicast_backward_args.begin(), unicast_backward_args.end());
    append_accessors(
        in0_receiver_fabric_compile_time_args,
        ag_output_tensor,
        mm_output_tensors,
        bias_tensor,
        input_tensor,
        fused_ternary_input_a,
        fused_ternary_input_b,
        true);

    auto in0_receiver_fabric_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/dm_in0_sender.cpp",
        in0_receiver_cores_fabric,
        tt::tt_metal::DataMovementConfig{
            .processor = in0_risc,
            .noc = in0_noc,
            .compile_args = in0_receiver_fabric_compile_time_args,
            .defines = in0_fabric_defines});

    // For in1's primary accessor: when fsdp_fused, the kernel reads from PWB (gathered weight),
    // so the main accessor describes the PWB. When not fsdp_fused, it describes the weight tensor
    // as before. The FSDP-sharded local weight (the original `weight_tensor`) gets a separate
    // accessor appended after bias.
    const ttnn::Tensor& in1_primary_tensor = fsdp_fused ? persistent_weight_buffer.value() : weight_tensor;
    std::optional<const ttnn::Tensor> in1_fsdp_local_weight =
        fsdp_fused ? std::optional<const ttnn::Tensor>{weight_tensor} : std::nullopt;

    auto build_in1_ct_args = [&](bool is_injector) {
        std::vector<uint32_t> ct = {
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
            in1_is_output_writer,
            (uint32_t)is_injector,  // is_injector_core
            ring_size,
            skewed_rank,                      // my_rank (skewed for fsdp; == ring_index otherwise)
            N_chunks,                         // N_chunks
            N_tiles_per_chunk,                // N_tiles_per_chunk
            static_cast<uint32_t>(topology),  // topology
            K_tiles_per_device,
            K_block_tail_tiles,
        };
        // FSDP-only CT args. Order matches the kernel's parse order under FSDP_FUSED.
        if (fsdp_fused) {
            ct.push_back(fsdp_ring_size);
            ct.push_back(fsdp_ring_index);
            ct.push_back(fsdp_num_targets_forward);
            ct.push_back(fsdp_num_targets_backward);
            ct.push_back(static_cast<uint32_t>(fsdp_topology));
            fabric_mux_connection_ct_args(
                num_workers_per_link,
                tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                fsdp_mux_kernel_config,
                ct);
            ct.insert(ct.end(), fsdp_unicast_forward_args.begin(), fsdp_unicast_forward_args.end());
            ct.insert(ct.end(), fsdp_unicast_backward_args.begin(), fsdp_unicast_backward_args.end());
        }
        append_accessors(
            ct,
            in1_primary_tensor,
            mm_output_tensors,
            bias_tensor,
            input_tensor,
            fused_ternary_input_a,
            fused_ternary_input_b,
            /*is_injector_core=*/false,
            in1_fsdp_local_weight);
        return ct;
    };

    std::vector<uint32_t> in1_sender_compile_time_args = build_in1_ct_args(/*is_injector=*/true);
    auto in1_sender_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/"
        "dm_in1_sender_out.cpp",
        in1_sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = in1_risc,
            .noc = in1_noc,
            .compile_args = in1_sender_compile_time_args,
            .defines = in1_defines});

    std::vector<uint32_t> in1_receiver_compile_time_args = build_in1_ct_args(/*is_injector=*/false);
    auto in1_receiver_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/"
        "dm_in1_sender_out.cpp",
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
        subblock_w};

    auto compute_defines = defines;
    if (topology == ttnn::ccl::Topology::Linear) {
        compute_defines["IS_LINEAR"] = "1";
    }
    std::map<std::string, std::string> compute_activation_defines;
    if (fused_activation.has_value()) {
        compute_activation_defines = ttnn::operations::unary::utils::get_defines(
            fused_activation.value().op_type,
            fused_activation.value().params,
            "ACTIVATION",
            "fused_act_dst_id",
            mm_output_tensors[0].dtype());
    }
    compute_defines.merge(compute_activation_defines);
    auto compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/compute.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
            .defines = compute_defines});

    tt::tt_metal::KernelHandle mux_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        mux_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

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

    for (uint32_t mux_id = 0; mux_id < num_mux_cores; ++mux_id) {
        uint32_t dir = mux_id % 2;  // 2 being the number of directions
        if (mux_connection_valid(dir)) {
            uint32_t link = mux_id / 2;  // 2 is the num directions
            auto mux_logical_core = in0_mux_logical(link, dir);

            std::vector<uint32_t> mux_rt_args = {};
            const auto src_node_id = device->get_fabric_node_id(sender_device_coord);
            if (dir) {  // forward
                const auto dst_node_id = device->get_fabric_node_id(backward_coord.value());
                mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                    src_node_id, dst_node_id, link, program, {mux_logical_core});
            } else {
                const auto dst_node_id = device->get_fabric_node_id(forward_coord.value());
                mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                    src_node_id, dst_node_id, link, program, {mux_logical_core});
            }
            tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);
        }
    }

    // FSDP mux RT args. Same iteration pattern as in0 mux above, but on row full_grid_size.y - 2
    // and using FSDP-axis neighbor coords.
    if (fsdp_fused) {
        for (uint32_t mux_id = 0; mux_id < num_mux_cores; ++mux_id) {
            uint32_t dir = mux_id % 2;
            if (fsdp_mux_connection_valid(dir)) {
                uint32_t link = mux_id / 2;
                // Match the create-loop placement via the shared helper.
                CoreCoord fsdp_mux_logical_core = fsdp_mux_logical(link, dir);

                std::vector<uint32_t> fsdp_mux_rt_args;
                const auto src_node_id = device->get_fabric_node_id(sender_device_coord);
                if (dir) {
                    const auto dst_node_id = device->get_fabric_node_id(fsdp_backward_coord.value());
                    fsdp_mux_rt_args = fsdp_mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link, program, {fsdp_mux_logical_core});
                } else {
                    const auto dst_node_id = device->get_fabric_node_id(fsdp_forward_coord.value());
                    fsdp_mux_rt_args = fsdp_mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link, program, {fsdp_mux_logical_core});
                }
                tt::tt_metal::SetRuntimeArgs(program, fsdp_mux_kernel_id, {fsdp_mux_logical_core}, fsdp_mux_rt_args);
            }
        }
    }

    // Set common runtime args (same for all cores, updated in override_runtime_arguments)
    // in0 common args: [in0_addr, in2_addr, in3_addr, sem_backward, sem_forward, [ternary_a, ternary_b],
    // output_addrs...]
    {
        std::vector<uint32_t> in0_common_args = {
            in0_addr,
            in2_addr,
            in3_addr,
            semaphore.at(0).address(),
            semaphore.at(1).address(),
        };
        if (use_fused_ternary) {
            in0_common_args.push_back(fused_ternary_input_a.value().buffer()->address());
            in0_common_args.push_back(fused_ternary_input_b.value().buffer()->address());
            uint32_t ternary_b_M_tiles = fused_ternary_input_b.value().padded_shape()[-2] / tt::constants::TILE_HEIGHT;
            in0_common_args.push_back(ternary_b_M_tiles == 1 ? 1u : 0u);  // broadcast_ternary_b
        }
        for (const auto& mm_output_tensor : mm_output_tensors) {
            in0_common_args.push_back(mm_output_tensor.buffer()->address());
        }
        tt::tt_metal::SetCommonRuntimeArgs(program, in0_sender_kernels_id, in0_common_args);
        tt::tt_metal::SetCommonRuntimeArgs(program, in0_receiver_fabric_kernels_id, in0_common_args);
        tt::tt_metal::SetCommonRuntimeArgs(program, in0_receiver_no_fabric_kernels_id, in0_common_args);
    }

    // in1 common args: [in1_addr, in2_addr, [ternary_a, ternary_b, broadcast_ternary_b],
    //                   [local_weight_addr, fsdp_sem_backward, fsdp_sem_forward], output_addrs...]
    {
        std::vector<uint32_t> in1_common_args = {
            in1_addr,
            in2_addr,
        };
        if (use_fused_ternary) {
            in1_common_args.push_back(fused_ternary_input_a.value().buffer()->address());
            in1_common_args.push_back(fused_ternary_input_b.value().buffer()->address());
            uint32_t ternary_b_M_tiles = fused_ternary_input_b.value().padded_shape()[-2] / tt::constants::TILE_HEIGHT;
            in1_common_args.push_back(ternary_b_M_tiles == 1 ? 1u : 0u);  // broadcast_ternary_b
        }
        if (fsdp_fused) {
            // local_weight_addr: the FSDP-sharded weight (injector reads its own K-slice from this).
            // fsdp_sem_backward / fsdp_sem_forward: per-direction counters that the injector waits on
            //   for remote K-slices to land in the local PWB (mirrors in0's out_ready_sem pair).
            in1_common_args.push_back(weight_tensor.buffer()->address());
            in1_common_args.push_back(fsdp_semaphore.at(0).address());
            in1_common_args.push_back(fsdp_semaphore.at(1).address());
        }
        for (const auto& mm_output_tensor : mm_output_tensors) {
            in1_common_args.push_back(mm_output_tensor.buffer()->address());
        }
        tt::tt_metal::SetCommonRuntimeArgs(program, in1_sender_kernels_id, in1_common_args);
        tt::tt_metal::SetCommonRuntimeArgs(program, in1_receiver_kernels_id, in1_common_args);
    }

    // compute common args: [scalar, broadcast_ternary_b] (only if fused ternary)
    if (use_fused_ternary) {
        uint32_t ternary_b_M_tiles = fused_ternary_input_b.value().padded_shape()[-2] / tt::constants::TILE_HEIGHT;
        std::vector<uint32_t> compute_common_args = {
            *reinterpret_cast<const uint32_t*>(&fused_ternary_scalar.value()),
            ternary_b_M_tiles == 1 ? 1u : 0u,  // broadcast_ternary_b
        };
        tt::tt_metal::SetCommonRuntimeArgs(program, compute_kernels_id, compute_common_args);
    }

    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        CoreCoord core = cores.at(core_id);
        CoreCoord virtual_core = device->worker_core_from_logical_core(core);
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
        uint32_t defer_write_k_block = core.y * k_blocks_per_core;
        defer_write_k_block = std::min(defer_write_k_block, K_blocks - 1);

        bool is_in0_sink = core == in0_core_order.back();
        bool is_in1_sink = core == in1_core_order.back();

        auto in0_injector_virtual_core = device->worker_core_from_logical_core(in0_core_order.front());
        // Per-core args only (common values set via SetCommonRuntimeArgs above)
        std::vector<uint32_t> in0_args = {
            is_in0_sink,
            (std::uint32_t)in0_next_core_physical.x,  // in0_dest_noc_x
            (std::uint32_t)in0_next_core_physical.y,  // in0_dest_noc_y
            (std::uint32_t)in0_prev_core_physical.x,  // in0_sender_noc_x
            (std::uint32_t)in0_prev_core_physical.y,  // in0_sender_noc_y
            in0_sender_semaphore_id,
            in0_receiver_semaphore_id,
            in0_valid_semaphore_id,
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
            defer_write_k_block,
            virtual_core.x,
            virtual_core.y,
            in0_injector_virtual_core.x,
            in0_injector_virtual_core.y,
            in0_core_order_index,
            in0_core_order.size()};
        if (in0_core_order_index > (in0_core_order.size() - 3)) {
            uint32_t worker_idx = in0_idx % num_workers_per_link;
            auto last_in0_core = in0_core_order.back();

            // Actual client count on this core's mux: the senders share a mux per group of
            // num_workers_per_link along the in0 sender axis. The last group is short when the
            // axis isn't a multiple of num_workers_per_link, so clamp to the axis size.
            uint32_t in0_group_base = in0_idx - worker_idx;
            uint32_t in0_mux_clients =
                std::min(in0_group_base + num_workers_per_link, in0_parallel_axis_cores) - in0_group_base;

            // Each fabric-sender core only registers as a client of the mux for the SINGLE
            // direction it actually sends in. (Previously both directions were registered, with
            // the unused one returning nullptr from build_and_connect — wasting 5 semaphores per
            // core for nothing.)
            // Core at size-2 → backward fabric sender → backward mux only.
            // Core at size-1 → forward  fabric sender → forward  mux only.
            const bool is_in0_backward_sender = (in0_core_order_index == (in0_core_order.size() - 2));
            if (is_in0_backward_sender) {
                auto termination_master_logical_core_backward =
                    transpose_core_grid ? CoreCoord(in0_idx - worker_idx, last_in0_core.y - 1)
                                        : CoreCoord(last_in0_core.x - 1, in0_idx - worker_idx);
                CoreCoord termination_master_virtual_core_backward =
                    device->worker_core_from_logical_core(termination_master_logical_core_backward);

                auto mux_logical_core_backward = in0_mux_logical(in0_idx / num_workers_per_link, /*dir=*/0);
                CoreCoord mux_virtual_core_backward = device->worker_core_from_logical_core(mux_logical_core_backward);
                fabric_mux_connection_rt_args(
                    mux_connection_valid(0),
                    !(in0_idx % num_workers_per_link),  // termination master at worker_idx 0
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    mux_virtual_core_backward,
                    worker_idx,
                    core,
                    mux_kernel_config,
                    program,
                    termination_master_virtual_core_backward,
                    in0_mux_clients,
                    in0_term_sync_id,
                    in0_args);
            } else {
                // Forward fabric sender (in0_core_order_index == size - 1).
                auto termination_master_logical_core_forward = transpose_core_grid
                                                                   ? CoreCoord(in0_idx - worker_idx, last_in0_core.y)
                                                                   : CoreCoord(last_in0_core.x, in0_idx - worker_idx);
                CoreCoord termination_master_virtual_core_forward =
                    device->worker_core_from_logical_core(termination_master_logical_core_forward);

                auto mux_logical_core_forward = in0_mux_logical(in0_idx / num_workers_per_link, /*dir=*/1);
                CoreCoord mux_virtual_core_forward = device->worker_core_from_logical_core(mux_logical_core_forward);
                fabric_mux_connection_rt_args(
                    mux_connection_valid(1),
                    !(in0_idx % num_workers_per_link),  // termination master at worker_idx 0
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    mux_virtual_core_forward,
                    worker_idx,
                    core,
                    mux_kernel_config,
                    program,
                    termination_master_virtual_core_forward,
                    in0_mux_clients,
                    in0_term_sync_id,
                    in0_args);
            }
        }
        if (in0_core_order_index == 0) {
            // in0 sender
            SetRuntimeArgs(program, in0_sender_kernels_id, core, in0_args);
        } else if (in0_core_order_index > (in0_core_order.size() - 3)) {
            // in0 receiver fabric
            SetRuntimeArgs(program, in0_receiver_fabric_kernels_id, core, in0_args);
        } else {
            // in0 receiver no fabric
            SetRuntimeArgs(program, in0_receiver_no_fabric_kernels_id, core, in0_args);
        }

        // Per-core args only (common values set via SetCommonRuntimeArgs above)
        auto in1_injector_virtual_core = device->worker_core_from_logical_core(in1_core_order.front());
        std::vector<uint32_t> in1_args = {
            is_in1_sink,
            (std::uint32_t)in1_next_core_physical.x,  // in1_dest_noc_x
            (std::uint32_t)in1_next_core_physical.y,  // in1_dest_noc_y
            (std::uint32_t)in1_prev_core_physical.x,  // in1_sender_noc_x
            (std::uint32_t)in1_prev_core_physical.y,  // in1_sender_noc_y
            in1_sender_semaphore_id,
            in1_receiver_semaphore_id,
            in1_valid_semaphore_id,
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
            defer_write_k_block,
        };
        // FSDP fabric senders are the in1 chain tail — the last two cores of in1_core_order, one per
        // direction (size-2 backward, size-1 forward). For a transpose grid the in1 chain runs along
        // X, so the tail relays a short horizontal NOC_1 hop to its mux in the freed last column (see
        // the fsdp mux create loop); for non-transpose the tail relays to the bottom-row mux. Each
        // mux is shared by a group of num_workers_per_link consecutive rows along the in1 sender axis.
        uint32_t fsdp_worker_idx = in1_idx % num_workers_per_link;
        uint32_t fsdp_group_base = in1_idx - fsdp_worker_idx;
        uint32_t fsdp_backward_sender_index = static_cast<uint32_t>(in1_core_order.size()) - 2;
        uint32_t fsdp_forward_sender_index = static_cast<uint32_t>(in1_core_order.size()) - 1;
        // Scheme 3 (single-row muxes): relay from the in1-chain core in the SAME column as the fsdp
        // mux (odd col 2g+1), so the sender->mux write drops straight down the mux column instead of
        // zig-zagging from the chain tail (col 6/7). All cores in a row hold the identical weight
        // block / consume order, so any column works for relay-what-you-consume. Only the surviving
        // uni-ring direction connects; point it at the column-matched core, disable the other.
        // fsdp_uni_dir==0 -> backward sender relays (Dev 0 chain head); ==1 -> forward sender relays.
        constexpr uint32_t NO_FSDP_SENDER = 0xFFFFFFFFu;
        if (single_row_muxes) {
            const uint32_t fsdp_mux_col = num_workers_per_link * (in1_idx / num_workers_per_link);
            uint32_t col_matched_index = fsdp_forward_sender_index;  // fallback (tail) if not found
            for (uint32_t i = 0; i < in1_core_order.size(); ++i) {
                if (in1_core_order[i].x == fsdp_mux_col) {
                    col_matched_index = i;
                    break;
                }
            }
            if (fsdp_uni_dir == 0) {
                fsdp_backward_sender_index = col_matched_index;
                fsdp_forward_sender_index = NO_FSDP_SENDER;
            } else {
                fsdp_forward_sender_index = col_matched_index;
                fsdp_backward_sender_index = NO_FSDP_SENDER;
            }
        }

        // FSDP-only per-core args: kernel parses these under FSDP_FUSED. The two sender indices tell
        // the kernel which cores in this row's chain relay (backward / forward direction).
        if (fsdp_fused) {
            in1_args.push_back(virtual_core.x);
            in1_args.push_back(virtual_core.y);
            in1_args.push_back(in1_injector_virtual_core.x);
            in1_args.push_back(in1_injector_virtual_core.y);
            in1_args.push_back(in1_core_order_index);
            in1_args.push_back(in1_core_order.size());
            in1_args.push_back(fsdp_backward_sender_index);
            in1_args.push_back(fsdp_forward_sender_index);
        }
        // Wire the FSDP mux RT args only on the two fabric-sender cores (one per direction). Each
        // call creates 5 semaphores per direction, so we cap the per-core semaphore count by
        // skipping non-fabric cores. The kernel correspondingly skips mux parsing on those cores.
        const bool is_in1_backward_sender = (in1_core_order_index == fsdp_backward_sender_index);
        const bool is_in1_forward_sender = (in1_core_order_index == fsdp_forward_sender_index);
        const bool is_in1_fabric_sender = is_in1_backward_sender || is_in1_forward_sender;
        if (fsdp_fused && is_in1_fabric_sender) {
            uint32_t worker_idx = fsdp_worker_idx;
            auto last_in1_core = in1_core_order.back();

            // Actual client count on this core's FSDP mux. in1 senders share a mux per group of
            // num_workers_per_link along the in1 sender axis (in1_parallel_axis_cores). When that
            // axis isn't a multiple of num_workers_per_link the last group is short (e.g. an odd
            // core_grid_y with nwpl=2 → 1 client), so num_mux_clients must be clamped to the axis
            // — otherwise that mux's termination master waits forever for a non-existent peer.
            uint32_t in1_group_base = fsdp_group_base;
            uint32_t in1_mux_clients =
                std::min(in1_group_base + num_workers_per_link, in1_parallel_axis_cores) - in1_group_base;

            // Each FSDP fabric-sender core only registers as a client of the mux for the SINGLE
            // direction it actually sends in (backward / forward). The termination master is the
            // group's worker-0 client, which now lives in the mux's column at the group-base row.
            if (is_in1_backward_sender) {
                // Backward sender = chain tail core_order[size-2]. Transpose: its mux is in the last
                // column at the group's backward row ((group)*2 + 0); the sender reaches it with a
                // horizontal NOC_1 hop. Non-transpose: the original bottom-row mux with the -1 shift.
                // The termination master is the group's worker-0 client — the backward sender of the
                // group-base row, which sits in the chain-tail column on the in1 axis.
                auto second_last_in1_core = in1_core_order[in1_core_order.size() - 2];
                CoreCoord fsdp_mux_logical_backward = fsdp_mux_logical(in1_idx / num_workers_per_link, /*dir=*/0);
                // Term master = the group's worker-0 client. The layout follows the GRID orientation,
                // not the mux placement: a transpose grid (in1 chain along X) indexes the client by its
                // chain-tail column + group-base row; non-transpose swaps the axes. Gating on
                // fsdp_mux_in_column was wrong for the single-row case (transpose grid, column==false),
                // which picked the swapped non-transpose form and pointed at non-client cores -> the mux
                // never terminated. Mirror the in0 term-master, which already gates on transpose.
                // Scheme 3: clients now sit in the mux's column (not the chain tail), so the
                // worker-0 term master is the column-matched core at the group-base row.
                CoreCoord fsdp_term_master_logical_backward =
                    single_row_muxes
                        ? CoreCoord(num_workers_per_link * (in1_idx / num_workers_per_link), in1_idx - worker_idx)
                    : transpose_core_grid ? CoreCoord(second_last_in1_core.x, in1_idx - worker_idx)
                                          : CoreCoord(in1_idx - worker_idx, second_last_in1_core.y);
                CoreCoord fsdp_mux_virtual_backward = device->worker_core_from_logical_core(fsdp_mux_logical_backward);
                CoreCoord fsdp_term_master_virtual_backward =
                    device->worker_core_from_logical_core(fsdp_term_master_logical_backward);
                fabric_mux_connection_rt_args(
                    fsdp_mux_connection_valid(0),
                    !(in1_idx % num_workers_per_link),
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    fsdp_mux_virtual_backward,
                    worker_idx,
                    core,
                    fsdp_mux_kernel_config,
                    program,
                    fsdp_term_master_virtual_backward,
                    in1_mux_clients,
                    in1_term_sync_id,
                    in1_args);
            } else {
                // Forward fabric sender (in1_core_order_index == size - 1) = chain tail back().
                // Transpose: mux in the last column at the group's forward row ((group)*2 + 1).
                // Non-transpose: original bottom-row mux with the -1 shift. Term master = the group's
                // worker-0 forward sender (chain tail) at the group-base row.
                CoreCoord fsdp_mux_logical_forward = fsdp_mux_logical(in1_idx / num_workers_per_link, /*dir=*/1);
                CoreCoord fsdp_term_master_logical_forward =
                    single_row_muxes
                        ? CoreCoord(num_workers_per_link * (in1_idx / num_workers_per_link), in1_idx - worker_idx)
                    : transpose_core_grid ? CoreCoord(last_in1_core.x, in1_idx - worker_idx)
                                          : CoreCoord(in1_idx - worker_idx, last_in1_core.y);
                CoreCoord fsdp_mux_virtual_forward = device->worker_core_from_logical_core(fsdp_mux_logical_forward);
                CoreCoord fsdp_term_master_virtual_forward =
                    device->worker_core_from_logical_core(fsdp_term_master_logical_forward);
                fabric_mux_connection_rt_args(
                    fsdp_mux_connection_valid(1),
                    !(in1_idx % num_workers_per_link),
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    fsdp_mux_virtual_forward,
                    worker_idx,
                    core,
                    fsdp_mux_kernel_config,
                    program,
                    fsdp_term_master_virtual_forward,
                    in1_mux_clients,
                    in1_term_sync_id,
                    in1_args);
            }
        }
        if (in1_core_order_index == 0) {
            // in1 sender
            SetRuntimeArgs(program, in1_sender_kernels_id, core, in1_args);
        } else {
            // in1 receiver
            SetRuntimeArgs(program, in1_receiver_kernels_id, core, in1_args);
        }

        // Per-core compute args (scalar is in common args)
        std::vector<uint32_t> compute_runtime_args = {
            M_start_tile,
            M_end_tile,
            N_start_tile,
            N_end_tile,
        };
        SetRuntimeArgs(program, compute_kernels_id, core, compute_runtime_args);
    }

    return {
        num_cores,
        cores,
        in0_sender_kernels_id,
        in0_receiver_fabric_kernels_id,
        in0_receiver_no_fabric_kernels_id,
        in1_sender_kernels_id,
        in1_receiver_kernels_id,
        compute_kernels_id,
        transpose_core_grid,
        transpose_core_grid ? grid_size.y : grid_size.x};
}

}  // namespace detail

namespace ttnn::experimental::prim {

AllGatherMinimalMatmulAsyncProgramFactory::cached_mesh_workload_t
AllGatherMinimalMatmulAsyncProgramFactory::create_mesh_workload(
    const AllGatherMinimalMatmulAsyncParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AllGatherMinimalMatmulAsyncInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

void AllGatherMinimalMatmulAsyncProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherMinimalMatmulAsyncParams& attributes,
    const AllGatherMinimalMatmulAsyncInputs& tensor_args,
    std::vector<ttnn::Tensor>& output_tensor) {
    // Derive has_fused_ternary from scalar presence, matching validate and create_at.
    // validate guarantees that scalar and tensors are always provided together.
    bool has_fused_ternary = attributes.fused_ternary_scalar.has_value();

    // Output layout: [0]=ag_output, [1]=persistent_weight_buffer (if FSDP), then chunk outputs
    const size_t mm_outputs_start = 1 + (attributes.fsdp_cluster_axis.has_value() ? 1 : 0);

    // Build in0 common args: [in0_addr, in2_addr, in3_addr, sem_backward, sem_forward, [ternary], output_addrs...]
    std::vector<uint32_t> in0_common = {
        output_tensor.at(0).buffer()->address(),
        tensor_args.bias_tensor.has_value() ? tensor_args.bias_tensor.value().buffer()->address() : 0,
        tensor_args.input_tensor.buffer()->address(),
        attributes.semaphore.at(0).address(),
        attributes.semaphore.at(1).address(),
    };
    if (has_fused_ternary) {
        in0_common.push_back(tensor_args.fused_ternary_input_a.value().buffer()->address());
        in0_common.push_back(tensor_args.fused_ternary_input_b.value().buffer()->address());
        uint32_t ternary_b_M_tiles =
            tensor_args.fused_ternary_input_b.value().padded_shape()[-2] / tt::constants::TILE_HEIGHT;
        in0_common.push_back(ternary_b_M_tiles == 1 ? 1u : 0u);  // broadcast_ternary_b
    }
    for (size_t i = mm_outputs_start; i < output_tensor.size(); ++i) {
        in0_common.push_back(output_tensor[i].buffer()->address());
    }

    // Build in1 common args: [in1_addr, in2_addr, [ternary_a, ternary_b, broadcast_ternary_b], output_addrs...]
    // When FSDP fusion is active, in1 reads from the gathered persistent_weight_buffer
    // (output_tensor[1]) instead of from the FSDP-sharded weight_tensor directly.
    const bool fsdp_fused_override = attributes.fsdp_cluster_axis.has_value();
    uint32_t in1_in_addr =
        fsdp_fused_override ? output_tensor.at(1).buffer()->address() : tensor_args.weight_tensor.buffer()->address();
    std::vector<uint32_t> in1_common = {
        in1_in_addr,
        tensor_args.bias_tensor.has_value() ? tensor_args.bias_tensor.value().buffer()->address() : 0,
    };
    if (has_fused_ternary) {
        in1_common.push_back(tensor_args.fused_ternary_input_a.value().buffer()->address());
        in1_common.push_back(tensor_args.fused_ternary_input_b.value().buffer()->address());
        uint32_t ternary_b_M_tiles =
            tensor_args.fused_ternary_input_b.value().padded_shape()[-2] / tt::constants::TILE_HEIGHT;
        in1_common.push_back(ternary_b_M_tiles == 1 ? 1u : 0u);  // broadcast_ternary_b
    }
    if (fsdp_fused_override) {
        in1_common.push_back(tensor_args.weight_tensor.buffer()->address());
        in1_common.push_back(attributes.fsdp_semaphore.at(0).address());
        in1_common.push_back(attributes.fsdp_semaphore.at(1).address());
    }
    for (size_t i = mm_outputs_start; i < output_tensor.size(); ++i) {
        in1_common.push_back(output_tensor[i].buffer()->address());
    }

    // Build compute common args: [scalar, broadcast_ternary_b] (only if fused ternary)
    uint32_t scalar_as_uint = 0;
    uint32_t broadcast_ternary_b_uint = 1;  // default broadcast
    if (has_fused_ternary) {
        float scalar = attributes.fused_ternary_scalar.value();
        scalar_as_uint = *reinterpret_cast<const uint32_t*>(&scalar);
        uint32_t ternary_b_M_tiles =
            tensor_args.fused_ternary_input_b.value().padded_shape()[-2] / tt::constants::TILE_HEIGHT;
        broadcast_ternary_b_uint = ternary_b_M_tiles == 1 ? 1u : 0u;
    }

    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        auto& shared_variables = cached_workload.shared_variables.at(range);

        // Update in0 common args (no per-core loop needed)
        auto& in0_sender_common = tt::tt_metal::GetCommonRuntimeArgs(program, shared_variables.in0_sender_kernels_id);
        auto& in0_receiver_fabric_common =
            tt::tt_metal::GetCommonRuntimeArgs(program, shared_variables.in0_receiver_fabric_kernels_id);
        auto& in0_receiver_no_fabric_common =
            tt::tt_metal::GetCommonRuntimeArgs(program, shared_variables.in0_receiver_no_fabric_kernels_id);
        for (size_t i = 0; i < in0_common.size(); ++i) {
            in0_sender_common[i] = in0_common[i];
            in0_receiver_fabric_common[i] = in0_common[i];
            in0_receiver_no_fabric_common[i] = in0_common[i];
        }

        // Update in1 common args
        auto& in1_sender_common = tt::tt_metal::GetCommonRuntimeArgs(program, shared_variables.in1_sender_kernels_id);
        auto& in1_receiver_common =
            tt::tt_metal::GetCommonRuntimeArgs(program, shared_variables.in1_receiver_kernels_id);
        for (size_t i = 0; i < in1_common.size(); ++i) {
            in1_sender_common[i] = in1_common[i];
            in1_receiver_common[i] = in1_common[i];
        }

        // Update compute common args
        if (has_fused_ternary) {
            auto& compute_common = tt::tt_metal::GetCommonRuntimeArgs(program, shared_variables.compute_kernels_id);
            compute_common[0] = scalar_as_uint;
            compute_common[1] = broadcast_ternary_b_uint;
        }
    }
}

ttnn::device_operation::CachedProgram<AllGatherMinimalMatmulAsyncProgramFactory::shared_variables_t>
all_gather_minimal_matmul_async_factory(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& fused_activation,
    const std::optional<const MinimalMatmulConfig>& config,
    const std::vector<ttnn::Tensor>& mm_output_tensors,
    const ttnn::Tensor& ag_output_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<ttnn::GlobalSemaphore>& semaphore,
    // const std::optional<ttnn::GlobalSemaphore>& barrier_semaphore,
    // bool using_persistent_buffers,
    const bool force_transpose,
    const uint32_t num_workers_per_link,
    const uint32_t num_buffers_per_channel,
    uint32_t N_chunks,
    std::optional<float> fused_ternary_scalar,
    const std::optional<const Tensor>& fused_ternary_input_a,
    const std::optional<const Tensor>& fused_ternary_input_b,
    const std::optional<const Tensor>& persistent_weight_buffer,
    const std::optional<MeshCoordinate>& fsdp_forward_coord,
    const std::optional<MeshCoordinate>& fsdp_backward_coord,
    uint32_t fsdp_ring_size,
    uint32_t fsdp_ring_index,
    const std::vector<ttnn::GlobalSemaphore>& fsdp_semaphore,
    ttnn::ccl::Topology fsdp_topology) {
    tt::tt_metal::Program program{};

    return {
        std::move(program),
        ::detail::all_gather_minimal_matmul_async_factory_helper(
            program,
            input_tensor,
            weight_tensor,
            bias_tensor,
            fused_activation,
            config,
            mm_output_tensors,
            ag_output_tensor,
            compute_kernel_config,
            sender_device_coord,
            forward_coord,
            backward_coord,
            num_links,
            ring_size,
            ring_index,
            topology,
            semaphore,
            // barrier_semaphore,
            // using_persistent_buffers,
            force_transpose,
            num_workers_per_link,
            num_buffers_per_channel,
            N_chunks,
            fused_ternary_scalar,
            fused_ternary_input_a,
            fused_ternary_input_b,
            persistent_weight_buffer,
            fsdp_forward_coord,
            fsdp_backward_coord,
            fsdp_ring_size,
            fsdp_ring_index,
            fsdp_semaphore,
            fsdp_topology)};
}

ttnn::device_operation::CachedProgram<AllGatherMinimalMatmulAsyncProgramFactory::shared_variables_t>
AllGatherMinimalMatmulAsyncProgramFactory::create_at(
    const AllGatherMinimalMatmulAsyncParams& attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const AllGatherMinimalMatmulAsyncInputs& tensor_args,
    std::vector<ttnn::Tensor>& output_tensor) {
    uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, attributes.cluster_axis);

    std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, 1, attributes.topology, attributes.cluster_axis);

    std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, -1, attributes.topology, attributes.cluster_axis);

    // FSDP neighbor lookup along fsdp_cluster_axis (only when FSDP fusion is active).
    std::optional<MeshCoordinate> fsdp_forward_coord;
    std::optional<MeshCoordinate> fsdp_backward_coord;
    uint32_t fsdp_ring_index = 0;
    if (attributes.fsdp_cluster_axis.has_value()) {
        fsdp_ring_index = ttnn::ccl::get_linearized_index_from_physical_coord(
            tensor_args.input_tensor, mesh_coordinate, attributes.fsdp_cluster_axis);
        fsdp_forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor, mesh_coordinate, 1, attributes.fsdp_topology, attributes.fsdp_cluster_axis);
        fsdp_backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_tensor, mesh_coordinate, -1, attributes.fsdp_topology, attributes.fsdp_cluster_axis);
    }

    const auto& act_tensor = tensor_args.input_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;
    const auto& bias_tensor = tensor_args.bias_tensor;
    const auto& ag_output_tensor = output_tensor.at(0);

    // Output layout: [ag_output, (optional: persistent_weight_buffer), chunks...]
    // Strip ag_output (always) and persistent_weight_buffer (if FSDP) from mm_output_tensors.
    size_t mm_outputs_start = 1 + (attributes.fsdp_cluster_axis.has_value() ? 1 : 0);
    std::optional<ttnn::Tensor> pwb_opt;
    if (attributes.fsdp_cluster_axis.has_value()) {
        pwb_opt = output_tensor.at(1);
    }

    return all_gather_minimal_matmul_async_factory(
        act_tensor,
        weight_tensor,
        bias_tensor,
        attributes.fused_activation,
        attributes.config,
        std::vector<ttnn::Tensor>(output_tensor.begin() + mm_outputs_start, output_tensor.end()),
        ag_output_tensor,
        attributes.compute_kernel_config,
        mesh_coordinate,
        forward_coord,
        backward_coord,
        attributes.num_links,
        attributes.ring_size,
        device_index,
        attributes.topology,
        attributes.semaphore,
        // attributes.barrier_semaphore,
        // attributes.using_persistent_buffers,
        attributes.force_transpose,
        attributes.num_workers_per_link,
        attributes.num_buffers_per_channel,
        attributes.chunks,
        attributes.fused_ternary_scalar,
        tensor_args.fused_ternary_input_a,
        tensor_args.fused_ternary_input_b,
        pwb_opt,
        fsdp_forward_coord,
        fsdp_backward_coord,
        attributes.fsdp_ring_size,
        fsdp_ring_index,
        attributes.fsdp_semaphore,
        attributes.fsdp_topology);
}

}  // namespace ttnn::experimental::prim
