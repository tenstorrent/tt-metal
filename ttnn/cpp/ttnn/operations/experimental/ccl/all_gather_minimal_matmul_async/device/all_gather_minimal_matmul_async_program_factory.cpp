// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // termination_sync_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // local_fabric_mux_status_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // local_flow_control_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // local_teardown_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));   // local_buffer_index_address
    worker_rt_args.push_back(termination_master_virtual_core.x);                    // termination_master_noc_x
    worker_rt_args.push_back(termination_master_virtual_core.y);                    // termination_master_noc_y
}

// Append tensor accessors in a consistent order
static inline void append_accessors(
    std::vector<uint32_t>& args,
    const ttnn::Tensor& main_tensor,
    const ttnn::Tensor& output_tensor,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const ttnn::Tensor& ag_input_tensor,
    bool is_injector_core = false) {
    tt::tt_metal::TensorAccessorArgs(*main_tensor.buffer()).append_to(args);
    tt::tt_metal::TensorAccessorArgs(*output_tensor.buffer()).append_to(args);
    if (bias_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*bias_tensor.value().buffer()).append_to(args);
    }
    if (is_injector_core) {
        tt::tt_metal::TensorAccessorArgs(*ag_input_tensor.buffer()).append_to(args);
    }
}

ttnn::experimental::prim::AllGatherMinimalMatmulAsyncProgramFactory::shared_variables_t
all_gather_minimal_matmul_async_factory_helper(
    tt::tt_metal::Program& program,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& fused_activation,
    const std::optional<const ttnn::experimental::prim::AllGatherMinimalMatmulAsyncConfig>& config,
    const ttnn::Tensor& mm_output_tensor,
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
    const uint32_t num_buffers_per_channel) {
    auto* device = input_tensor.device();

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
    auto in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(ag_output_tensor.dtype());
    auto in0_tile_size = tt::tile_size(in0_data_format);
    auto in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(weight_tensor.dtype());
    auto in1_tile_size = tt::tile_size(in1_data_format);
    auto output_data_format = tt::tt_metal::datatype_to_dataformat_converter(mm_output_tensor.dtype());
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

    // Mux
    auto [num_targets_forward, num_targets_backward] =
        ttnn::ccl::get_forward_backward_line_mcast_distance(ring_size, ring_index, topology, false);
    auto [unicast_forward_args, unicast_backward_args] = ttnn::ccl::get_forward_backward_line_unicast_configuration(
        topology, sender_device_coord, forward_coord, backward_coord, device);

    auto full_grid_size = device->compute_with_storage_grid_size();
    TT_FATAL(
        !((transpose_core_grid ? grid_size.x : grid_size.y) % num_links),
        "The number of in0 rows must be a multiple of num_links");
    uint32_t num_mux_cores = num_links * 2;  // 2 being the number of directions
    TT_FATAL(
        (transpose_core_grid ? full_grid_size.y : full_grid_size.x) >= num_mux_cores,
        "The are not enough cores for the number of mux cores requested");

    const auto mux_connection_valid = [&backward_coord, &forward_coord](const uint32_t dir) {
        return (dir && backward_coord.has_value()) || (!dir && forward_coord.has_value());
    };

    std::vector<CoreRange> mux_core_ranges;
    for (uint32_t mux_id = 0; mux_id < num_mux_cores; ++mux_id) {
        uint32_t dir = mux_id % 2;  // 2 being the number of directions
        if (mux_connection_valid(dir)) {
            uint32_t link = mux_id / 2;  // 2 is the num directions
            uint32_t mux_x_index = (num_workers_per_link * (link + 1)) - (1 - dir);
            if (mux_x_index >= full_grid_size.x) {
                mux_x_index = mux_x_index - full_grid_size.x;
            }
            auto mux_logical_core = CoreCoord(mux_x_index, full_grid_size.y - 1);
            mux_core_ranges.emplace_back(mux_logical_core);
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

    // scatter-write currently only supports 2 distinct noc addresses
    uint32_t max_target_noc_addresses_per_packet = 2;

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
    in0_defines = defines;
    in0_defines["READ_FROM_LOCAL_INPUT"] = "1";
    in0_defines["IS_IN0"] = "1";
    in0_fabric_defines = in0_defines;
    in0_fabric_defines["USE_MUX"] = "1";

    uint32_t in0_addr = ag_output_tensor.buffer()->address();
    uint32_t in1_addr = weight_tensor.buffer()->address();
    uint32_t in2_addr = use_bias ? bias_tensor.value().buffer()->address() : 0;
    uint32_t out_addr = mm_output_tensor.buffer()->address();
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
        ring_index,
        in3_tile_size,
        num_tiles_to_write_per_packet,
        num_targets_forward,
        num_targets_backward,
        static_cast<uint32_t>(topology),
    };
    append_accessors(in0_sender_compile_time_args, ag_output_tensor, mm_output_tensor, bias_tensor, input_tensor, true);
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
        ring_index,
        in3_tile_size,
        num_tiles_to_write_per_packet,
        num_targets_forward,
        num_targets_backward,
        static_cast<uint32_t>(topology),
    };
    append_accessors(
        in0_receiver_no_fabric_compile_time_args, ag_output_tensor, mm_output_tensor, bias_tensor, input_tensor, true);

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
        ring_index,
        in3_tile_size,
        num_tiles_to_write_per_packet,
        num_targets_forward,
        num_targets_backward,
        static_cast<uint32_t>(topology),
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
        in0_receiver_fabric_compile_time_args, ag_output_tensor, mm_output_tensor, bias_tensor, input_tensor, true);

    auto in0_receiver_fabric_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/dm_in0_sender.cpp",
        in0_receiver_cores_fabric,
        tt::tt_metal::DataMovementConfig{
            .processor = in0_risc,
            .noc = in0_noc,
            .compile_args = in0_receiver_fabric_compile_time_args,
            .defines = in0_fabric_defines});

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
        in1_is_output_writer,
        true,  // is_injector_core
        ring_size,
        ring_index,
    };
    append_accessors(in1_sender_compile_time_args, weight_tensor, mm_output_tensor, bias_tensor, input_tensor, false);
    auto in1_sender_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/"
        "dm_in1_sender_out.cpp",
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
        in1_is_output_writer,
        false,  // is_injector_core
        ring_size,
        ring_index,
    };
    append_accessors(in1_receiver_compile_time_args, weight_tensor, mm_output_tensor, bias_tensor, input_tensor, false);
    auto in1_receiver_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/"
        "dm_in1_sender_out.cpp",
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
            mm_output_tensor.dtype());
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

    // mux kernel
    auto mux_kernel_id = tt::tt_metal::CreateKernel(
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
            uint32_t mux_x_index = (num_workers_per_link * (link + 1)) - (1 - dir);
            if (mux_x_index >= full_grid_size.x) {
                mux_x_index = mux_x_index - full_grid_size.x;
            }
            auto mux_logical_core = CoreCoord(mux_x_index, full_grid_size.y - 1);

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

        // log_info(tt::LogOp, "core_id: {}, M_start_tile: {}, M_end_tile: {}, N_start_tile: {}, N_end_tile: {}",
        // core_id, M_start_tile, M_end_tile, N_start_tile, N_end_tile);

        // Defer write to K block with same coordinate as core
        // The writer receiver cores always have core.x > 0
        uint32_t defer_write_k_block = core.y * k_blocks_per_core;
        defer_write_k_block = std::min(defer_write_k_block, K_blocks - 1);

        bool is_in0_sink = core == in0_core_order.back();
        bool is_in1_sink = core == in1_core_order.back();

        auto in0_injector_virtual_core = device->worker_core_from_logical_core(in0_core_order.front());
        std::vector<uint32_t> in0_args = {
            in0_addr,
            out_addr,
            in2_addr,
            in3_addr,
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
            semaphore.at(0).address(),
            semaphore.at(1).address(),
            in0_core_order_index,
            in0_core_order.size()};
        if (in0_core_order_index > (in0_core_order.size() - 3)) {
            uint32_t worker_idx = in0_idx % num_workers_per_link;
            auto last_in0_core = in0_core_order.back();
            auto termination_master_logical_core_backward = transpose_core_grid
                                                                ? CoreCoord(in0_idx - worker_idx, last_in0_core.y - 1)
                                                                : CoreCoord(last_in0_core.x - 1, in0_idx - worker_idx);
            CoreCoord termination_master_virtual_core_backward =
                device->worker_core_from_logical_core(termination_master_logical_core_backward);

            // in0 backward sender
            uint32_t mux_core_index_backward =
                ((in0_idx / num_workers_per_link) * num_workers_per_link) + (num_workers_per_link - 1);
            if (mux_core_index_backward >= full_grid_size.x) {
                mux_core_index_backward = mux_core_index_backward - full_grid_size.x;
            }
            auto mux_logical_core_backward = CoreCoord(mux_core_index_backward, full_grid_size.y - 1);
            CoreCoord mux_virtual_core_backward = device->worker_core_from_logical_core(mux_logical_core_backward);
            fabric_mux_connection_rt_args(
                mux_connection_valid(0),
                (in0_core_order_index == (in0_core_order.size() - 2)) && !(in0_idx % num_workers_per_link),
                tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                mux_virtual_core_backward,
                worker_idx,
                core,
                mux_kernel_config,
                program,
                termination_master_virtual_core_backward,
                in0_args);

            auto termination_master_logical_core_forward = transpose_core_grid
                                                               ? CoreCoord(in0_idx - worker_idx, last_in0_core.y)
                                                               : CoreCoord(last_in0_core.x, in0_idx - worker_idx);
            CoreCoord termination_master_virtual_core_forward =
                device->worker_core_from_logical_core(termination_master_logical_core_forward);

            // in0 forward sender
            uint32_t mux_core_index_forward =
                ((in0_idx / num_workers_per_link) * num_workers_per_link) + num_workers_per_link;
            if (mux_core_index_forward >= full_grid_size.x) {
                mux_core_index_forward = mux_core_index_forward - full_grid_size.x;
            }
            auto mux_logical_core_forward = CoreCoord(mux_core_index_forward, full_grid_size.y - 1);
            CoreCoord mux_virtual_core_forward = device->worker_core_from_logical_core(mux_logical_core_forward);
            fabric_mux_connection_rt_args(
                mux_connection_valid(1),
                (in0_core_order_index == (in0_core_order.size() - 1)) && !(in0_idx % num_workers_per_link),
                tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                mux_virtual_core_forward,
                worker_idx,
                core,
                mux_kernel_config,
                program,
                termination_master_virtual_core_forward,
                in0_args);
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

        std::vector<uint32_t> in1_args = {
            in1_addr,
            out_addr,
            in2_addr,
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
        if (in1_core_order_index == 0) {
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

    return {
        num_cores,
        cores,
        in0_sender_kernels_id,
        in0_receiver_fabric_kernels_id,
        in0_receiver_no_fabric_kernels_id,
        in1_sender_kernels_id,
        in1_receiver_kernels_id,
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
    const auto& out_ready_semaphore_backward = attributes.semaphore.at(0);
    const auto& out_ready_semaphore_forward = attributes.semaphore.at(1);

    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        auto& shared_variables = cached_workload.shared_variables.at(range);

        auto in0_addr = output_tensor.at(0).buffer()->address();
        auto in1_addr = tensor_args.weight_tensor.buffer()->address();
        auto output_addr = output_tensor.at(1).buffer()->address();
        auto in2_addr = tensor_args.bias_tensor.has_value() ? tensor_args.bias_tensor.value().buffer()->address() : 0;
        auto in3_addr = tensor_args.input_tensor.buffer()->address();
        auto& in0_sender_runtime_args = GetRuntimeArgs(program, shared_variables.in0_sender_kernels_id);
        auto& in0_receiver_no_fabric_runtime_args =
            GetRuntimeArgs(program, shared_variables.in0_receiver_no_fabric_kernels_id);
        auto& in0_receiver_fabric_runtime_args =
            GetRuntimeArgs(program, shared_variables.in0_receiver_fabric_kernels_id);
        auto& in1_sender_runtime_args = GetRuntimeArgs(program, shared_variables.in1_sender_kernels_id);
        auto& in1_receiver_runtime_args = GetRuntimeArgs(program, shared_variables.in1_receiver_kernels_id);
        auto in1_size = shared_variables.in1_size;

        for (uint32_t i = 0; i < shared_variables.num_cores; ++i) {
            CoreCoord core = shared_variables.cores.at(i);
            uint32_t in0_idx = shared_variables.transpose_core_grid ? core.x : core.y;
            uint32_t in1_idx = shared_variables.transpose_core_grid ? core.y : core.x;
            if (in1_idx == 0) {
                auto& in0_sender_args = in0_sender_runtime_args[core.x][core.y];
                in0_sender_args[0] = in0_addr;
                in0_sender_args[1] = output_addr;
                in0_sender_args[2] = in2_addr;
                in0_sender_args[3] = in3_addr;
                in0_sender_args[21] = out_ready_semaphore_backward.address();
                in0_sender_args[22] = out_ready_semaphore_forward.address();
            } else if (in1_idx > (in1_size - 3)) {
                auto& in0_receiver_args = in0_receiver_fabric_runtime_args[core.x][core.y];
                in0_receiver_args[0] = in0_addr;
                in0_receiver_args[1] = output_addr;
                in0_receiver_args[2] = in2_addr;
                in0_receiver_args[3] = in3_addr;
                in0_receiver_args[21] = out_ready_semaphore_backward.address();
                in0_receiver_args[22] = out_ready_semaphore_forward.address();
            } else {
                auto& in0_receiver_args = in0_receiver_no_fabric_runtime_args[core.x][core.y];
                in0_receiver_args[0] = in0_addr;
                in0_receiver_args[1] = output_addr;
                in0_receiver_args[2] = in2_addr;
                in0_receiver_args[3] = in3_addr;
                in0_receiver_args[21] = out_ready_semaphore_backward.address();
                in0_receiver_args[22] = out_ready_semaphore_forward.address();
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
    }
}

ttnn::device_operation::CachedProgram<AllGatherMinimalMatmulAsyncProgramFactory::shared_variables_t>
all_gather_minimal_matmul_async_factory(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& fused_activation,
    const std::optional<const AllGatherMinimalMatmulAsyncConfig>& config,
    const ttnn::Tensor& mm_output_tensor,
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
    const uint32_t num_buffers_per_channel) {
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
            mm_output_tensor,
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
            num_buffers_per_channel)};
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

    const auto& act_tensor = tensor_args.input_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;
    const auto& bias_tensor = tensor_args.bias_tensor;
    const auto& ag_output_tensor = output_tensor.at(0);
    const auto& mm_output_tensor = output_tensor.at(1);

    return all_gather_minimal_matmul_async_factory(
        act_tensor,
        weight_tensor,
        bias_tensor,
        attributes.fused_activation,
        attributes.config,
        mm_output_tensor,
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
        attributes.num_buffers_per_channel);
}

}  // namespace ttnn::experimental::prim
