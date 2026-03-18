// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_program_factory.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"

#include <algorithm>
#include <optional>
#include <cmath>
#include <string>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// Appends 5 compile-time args needed for a fabric MUX client worker kernel.
static void fabric_mux_connection_ct_args(
    const uint32_t num_workers_per_link,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    std::vector<uint32_t>& worker_ct_args) {
    auto channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
    worker_ct_args.push_back(mux_kernel_config.get_num_buffers(channel_type));
    worker_ct_args.push_back(mux_kernel_config.get_buffer_size_bytes(channel_type));
    worker_ct_args.push_back(mux_kernel_config.get_status_address());
    worker_ct_args.push_back(mux_kernel_config.get_termination_signal_address());
    worker_ct_args.push_back(num_workers_per_link);  // num_mux_clients
}

// Appends 17 runtime args for a fabric MUX client worker.
// Creates 5 semaphores on worker_logical_core for the connection state.
static void fabric_mux_connection_rt_args(
    const bool mux_connection_valid,
    const bool is_termination_master,
    const CoreCoord& mux_logical_core,
    const uint32_t worker_id,
    const CoreCoord& worker_logical_core,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    tt::tt_metal::Program& program,
    const CoreCoord& termination_master_logical_core,
    tt::tt_metal::IDevice* device,
    std::vector<uint32_t>& worker_rt_args) {
    auto channel_type = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
    const CoreCoord mux_virtual_core = device->worker_core_from_logical_core(mux_logical_core);
    const CoreCoord termination_master_virtual_core =
        device->worker_core_from_logical_core(termination_master_logical_core);

    worker_rt_args.push_back(static_cast<uint32_t>(mux_connection_valid));
    worker_rt_args.push_back(static_cast<uint32_t>(is_termination_master));
    worker_rt_args.push_back(mux_virtual_core.x);
    worker_rt_args.push_back(mux_virtual_core.y);
    const uint8_t ch_id = static_cast<uint8_t>(worker_id);
    worker_rt_args.push_back(mux_kernel_config.get_channel_base_address(channel_type, ch_id));
    worker_rt_args.push_back(mux_kernel_config.get_connection_info_address(channel_type, ch_id));
    worker_rt_args.push_back(mux_kernel_config.get_connection_handshake_address(channel_type, ch_id));
    worker_rt_args.push_back(mux_kernel_config.get_flow_control_address(channel_type, ch_id));
    worker_rt_args.push_back(mux_kernel_config.get_buffer_index_address(channel_type, ch_id));
    worker_rt_args.push_back(mux_kernel_config.get_channel_credits_stream_id(channel_type, ch_id));
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));  // termination_sync_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));  // local_fabric_mux_status_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));  // local_flow_control_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));  // local_teardown_address
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));  // local_buffer_index_address
    worker_rt_args.push_back(termination_master_virtual_core.x);
    worker_rt_args.push_back(termination_master_virtual_core.y);
}

}  // namespace

ExpRingJointSDPAProgramFactory::cached_mesh_workload_t ExpRingJointSDPAProgramFactory::create_mesh_workload(
    const ExpRingJointSDPAParams& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ExpRingJointSDPAInputs& tensor_args,
    ExpRingJointSDPAResult& output_tensors) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_vars;

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(args, coord, tensor_args, output_tensors);
        mesh_workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_vars.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_vars)};
}

ExpRingJointSDPAProgramFactory::cached_program_t ExpRingJointSDPAProgramFactory::create_at(
    const ExpRingJointSDPAParams& args,
    const ttnn::MeshCoordinate& coord,
    const ExpRingJointSDPAInputs& tensor_args,
    ExpRingJointSDPAResult& output_tensors) {
    /*
    The QKV inputs are fractured on the sequence dimension across ring_size.
    The sequence length comes in padded such that it is divisible by `TILE_HEIGHT * ring_size`.
    Therefore each device has `padded_N / ring_size` local tokens.

    Naming:
        - padded_N: the global, padded sequence length
        - local_padded_N: the local shard of the padded sequence length. local_padded_N = padded_N / ring_size
        - logical_n: the logical global sequence length. logical_n <= padded_N.
        - L: the logical joint sequence length

    input_tensor_q: B x NH x local_padded_N x DH
    input_tensor_k: B x NH x local_padded_N x DH
    input_tensor_v: B x NH x local_padded_N x DH

    gathered_input_tensor_k: B x NH x padded_N x DH
    gathered_input_tensor_v: B x NH x padded_N x DH

    joint_tensor_q: B x NH x L x DH
    joint_tensor_k: B x NH x L x DH
    joint_tensor_v: B x NH x L x DH

    output_tensor: B x NH x local_padded_N x DH
    joint_output_tensor: B x NH x L x DH


    The algorithm is roughly described below.
    - for each ring iteration:
        - read a Q chunk from input_tensor_q
        - for each KV chunk in local_padded_N:
            - on the first ring iteration, read from local input_tensor_k and input_tensor_v
            - otherwise, read from gathered_input_tensor_k and gathered_input_tensor_v
            - on the last ring iteration, also read from joint_tensor_k and joint_tensor_v
            - if the KV chunk is from the non-joint input and contains the global token index (logical_n - 1), generate
    a mask
            - else if the KV chunk is from non-joint input and contains the local token index (local_padded_N - 1),
    generate an attention mask
            - else if the KV chunk is from the joint input and contains the local token index (L - 1), generate a mask
            - compute attention
        - write the output Q chunk
        - if this is not the first ring iteration, do the LSE update.
    */

    log_debug(tt::LogOp, "DEBUG: create_at is called");

    const auto& input_tensor_q = tensor_args.input_q;
    const auto& input_tensor_k = tensor_args.input_k;
    const auto& input_tensor_v = tensor_args.input_v;

    const auto& joint_tensor_q = tensor_args.joint_q;
    const auto& joint_tensor_k = tensor_args.joint_k;
    const auto& joint_tensor_v = tensor_args.joint_v;

    const auto& gathered_input_tensor_k = tensor_args.gathered_k;
    const auto& gathered_input_tensor_v = tensor_args.gathered_v;

    auto& output_tensor = output_tensors[EXP_RING_JOINT_SDPA_OUTPUT_IDX];
    auto& joint_output_tensor = output_tensors[EXP_RING_JOINT_SDPA_JOINT_OUTPUT_IDX];
    auto& stats_output_tensor = output_tensors[EXP_RING_JOINT_SDPA_STATS_OUTPUT_IDX];

    std::size_t q_chunk_size = args.get_q_chunk_size();
    std::size_t k_chunk_size = args.get_k_chunk_size();

    tt::tt_metal::Program program{};

    auto* mesh_device = input_tensor_q.device();
    uint32_t device_index = ccl::get_linearized_index_from_physical_coord(
        input_tensor_q, coord, args.cluster_axis);

    std::optional<MeshCoordinate> forward_coord = ccl::get_physical_neighbor_from_physical_coord(
        input_tensor_q,
        coord,
        1,
        args.topology,
        args.cluster_axis);

    std::optional<MeshCoordinate> backward_coord = ccl::get_physical_neighbor_from_physical_coord(
        input_tensor_q,
        coord,
        -1,
        args.topology,
        args.cluster_axis);

    auto scale = args.scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.logical_shape()[-1]));
    }

    std::optional<ttnn::prim::RingSDPAFusedOpSignaler> sdpa_fused_op_signaler = ttnn::prim::RingSDPAFusedOpSignaler();

    auto [num_targets_forward, num_targets_backward, dynamic_alternate] = ccl::get_forward_backward_configuration(
        args.ring_size, device_index, args.topology);
    if (args.topology == ttnn::ccl::Topology::Ring && device_index % 2 == 0) {
        std::swap(num_targets_forward, num_targets_backward);
    }

    // Minimally use matmul fused op signaler
    sdpa_fused_op_signaler->init_all_gather(args.ring_size, device_index);

    const auto& q_shape = input_tensor_q.logical_shape();
    const auto& k_shape = gathered_input_tensor_k.logical_shape();
    const auto& joint_q_shape = joint_tensor_q.logical_shape();
    const uint32_t B = q_shape[0], NH = q_shape[1], local_padded_N = q_shape[2], DH = q_shape[3];
    const uint32_t padded_N = k_shape[2];
    const uint32_t L = joint_q_shape[2];

    const uint32_t local_padded_Nt = local_padded_N / tt::constants::TILE_HEIGHT;
    const uint32_t padded_Nt = padded_N / tt::constants::TILE_HEIGHT;
    // Find unpadded sequence lengths in tiles
    const uint32_t Lt = tt::div_up(L, tt::constants::TILE_HEIGHT);
    const uint32_t DHt = DH / tt::constants::TILE_WIDTH;
    const uint32_t logical_nt = tt::div_up(static_cast<uint32_t>(args.logical_n), tt::constants::TILE_HEIGHT);

    /*
    For non-causal case we must provide a padded mask if the K sequence length has been padded
    Note that we dont have this issue in non-causal case if Q is padded, since those pad tokens
    don't affect attention of unpadded tokens.
    In causal case, the causal mask takes care of masking K pad tokens.
    */

    const uint32_t Sq_chunk_t = q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t Sk_chunk_t = k_chunk_size / tt::constants::TILE_HEIGHT;

    // Lightweight mask: only needed when any K/joint dimension has padding that doesn't fill a chunk.
    const bool local_n_has_padding = (local_padded_Nt % Sk_chunk_t) != 0;
    const bool global_n_has_padding = (args.logical_n % (Sk_chunk_t * tt::constants::TILE_HEIGHT)) != 0;
    const bool joint_has_padding = L > 0 && (L % (Sk_chunk_t * tt::constants::TILE_HEIGHT)) != 0;
    const bool needs_lightweight_mask = local_n_has_padding || global_n_has_padding || joint_has_padding;

    // Partial tile support when padding boundary falls inside a tile.
    const uint32_t global_n_partial_col = args.logical_n % tt::constants::TILE_HEIGHT;
    const uint32_t joint_l_partial_col = L % tt::constants::TILE_HEIGHT;
    const uint32_t partial_mask_tiles = (global_n_partial_col != 0 ? 1 : 0) + (joint_l_partial_col != 0 ? 1 : 0);
    // Single CB holds: 1 neginf tile + up to 2 partial mask tiles
    const uint32_t total_lightweight_mask_tiles = 1 + partial_mask_tiles;

    const uint32_t num_local_q_chunks = tt::div_up(local_padded_N, q_chunk_size);
    const uint32_t num_joint_q_chunks = tt::div_up(L, q_chunk_size);
    const uint32_t num_q_chunks = num_local_q_chunks + num_joint_q_chunks;
    const uint32_t num_local_k_chunks = tt::div_up(local_padded_N, k_chunk_size);
    const uint32_t num_joint_k_chunks = tt::div_up(L, k_chunk_size);

    log_debug(tt::LogOp, "B: {}", B);
    log_debug(tt::LogOp, "NH: {}", NH);
    log_debug(tt::LogOp, "L: {}", L);
    log_debug(tt::LogOp, "DH: {}", DH);

    // Log padded dimensions
    log_debug(tt::LogOp, "local_padded_N: {}", local_padded_N);
    log_debug(tt::LogOp, "padded_N: {}", padded_N);
    log_debug(tt::LogOp, "L: {}", L);

    // Log tile dimensions
    log_debug(tt::LogOp, "DHt: {}", DHt);
    log_debug(tt::LogOp, "local_padded_Nt: {}", local_padded_Nt);
    log_debug(tt::LogOp, "padded_Nt: {}", padded_Nt);
    log_debug(tt::LogOp, "Lt: {}", Lt);

    // Log chunking parameters
    log_debug(tt::LogOp, "Sq_chunk_t: {}", Sq_chunk_t);
    log_debug(tt::LogOp, "Sk_chunk_t: {}", Sk_chunk_t);
    log_debug(tt::LogOp, "num_local_q_chunks: {}", num_local_q_chunks);
    log_debug(tt::LogOp, "num_joint_q_chunks: {}", num_joint_q_chunks);
    log_debug(tt::LogOp, "q_chunk_size: {}", q_chunk_size);
    log_debug(tt::LogOp, "k_chunk_size: {}", k_chunk_size);
    log_debug(tt::LogOp, "num_q_chunks: {}", num_q_chunks);
    log_debug(tt::LogOp, "num_local_k_chunks: {}", num_local_k_chunks);
    log_debug(tt::LogOp, "num_joint_k_chunks: {}", num_joint_k_chunks);

    IDevice* device = input_tensor_q.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(mesh_device->arch(), args.compute_kernel_config);

    CoreCoord grid_size = args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size
                                                          : mesh_device->compute_with_storage_grid_size();
    bool exp_approx_mode =
        args.program_config.has_value()
            ? (args.program_config->exp_approx_mode.has_value() ? args.program_config->exp_approx_mode.value() : true)
            : true;

    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    uint32_t num_cores = grid_size.x * grid_size.y;

    // Init fused op signaler
    sdpa_fused_op_signaler->init_fused_op(program, mesh_device, core_grid);

    // Override the fused-op semaphore with the global out-ready semaphore so that
    // the reader injector waits on the same semaphore that MUX writers increment.
    const uint32_t out_ready_global_sem_addr = args.semaphore[0].address();
    sdpa_fused_op_signaler->fused_op_receiver_signal_semaphores[0] = out_ready_global_sem_addr;
    sdpa_fused_op_signaler->fused_op_receiver_signal_semaphores[1] = out_ready_global_sem_addr;

    log_debug(tt::LogOp, "num_cores: {}", num_cores);
    log_debug(
        tt::LogOp, "mesh_device->compute_with_storage_grid_size(): {}", mesh_device->compute_with_storage_grid_size());
    log_debug(tt::LogOp, "grid_size: {}", grid_size);

    TT_FATAL(
        num_cores <= mesh_device->compute_with_storage_grid_size().x * mesh_device->compute_with_storage_grid_size().y,
        "Provided grid must not contain more cores than the device. Got {} cores, expected at most {} cores.",
        num_cores,
        mesh_device->compute_with_storage_grid_size().x * mesh_device->compute_with_storage_grid_size().y);

    /**
     * This parallelization scheme is efficient because it divides the global work,
     * the total number of Q chunks across all batches and heads, evenly across the cores.
     *
     */
    const uint32_t all_heads_num_q_chunks = B * NH * num_q_chunks;
    const uint32_t max_q_per_core = tt::div_up(all_heads_num_q_chunks, num_cores);

    const uint32_t q_buffer_factor = (max_q_per_core > 1) ? 2 : 1;

    log_debug(tt::LogOp, "max_q_per_core: {}", max_q_per_core);

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t q_tiles = Sq_chunk_t * DHt * q_buffer_factor;
    uint32_t k_tiles = Sk_chunk_t * DHt * 2;  // double buffer
    uint32_t v_tiles = Sk_chunk_t * DHt * 2;  // double buffer
    uint32_t qk_tiles = Sq_chunk_t * Sk_chunk_t;
    uint32_t out_im_tiles = Sq_chunk_t * DHt;
    uint32_t out0_t = Sq_chunk_t * DHt;
    uint32_t scale_tiles = 1;
    uint32_t statistics_tiles = Sq_chunk_t;  // Single column of values in each iteration

    // log all values
    log_debug(tt::LogOp, "q_tiles: {}", q_tiles);
    log_debug(tt::LogOp, "k_tiles: {}", k_tiles);
    log_debug(tt::LogOp, "v_tiles: {}", v_tiles);
    log_debug(tt::LogOp, "qk_tiles: {}", qk_tiles);
    log_debug(tt::LogOp, "out0_t: {}", out0_t);
    log_debug(tt::LogOp, "scale_tiles: {}", scale_tiles);
    log_debug(tt::LogOp, "statistics_tiles: {}", statistics_tiles);

    // Host code is responsible for determining matmul configuration
    const uint32_t dst_size = ttnn::get_dest_reg_count(args.compute_kernel_config);
    const uint32_t qk_in0_block_w = DHt;
    auto [qk_out_subblock_h, qk_out_subblock_w] =
        detail::determine_largest_subblock_size(Sq_chunk_t, Sk_chunk_t, dst_size);

    TT_FATAL(
        Sq_chunk_t % qk_out_subblock_h == 0,
        "Sq_chunk_t ({}) must be divisible by qk_out_subblock_h ({})",
        Sq_chunk_t,
        qk_out_subblock_h);
    const uint32_t qk_in0_num_subblocks = Sq_chunk_t / qk_out_subblock_h;
    const uint32_t qk_in1_num_subblocks = Sk_chunk_t / qk_out_subblock_w;
    const uint32_t qk_num_blocks = DHt / qk_in0_block_w;

    // now for out0
    const uint32_t out_in0_block_w = Sk_chunk_t;

    auto [out_out_subblock_h, out_out_subblock_w] = detail::determine_largest_subblock_size(Sq_chunk_t, DHt, dst_size);

    const uint32_t out_in0_num_subblocks = Sq_chunk_t / out_out_subblock_h;
    const uint32_t out_in1_num_subblocks = DHt / out_out_subblock_w;
    const uint32_t out_num_blocks = Sk_chunk_t / out_in0_block_w;

    // Streaming compute v2: eliminates row buffers via cb_push_back_hold_wr_ptr.
    // Ring joint has no causal/mask/sink/sliding/chunked flags — gating is simpler.
    // Streaming v2 requires q_num_subblocks > 1 (Sq_chunk_t > subblock_h) because the Phase 2
    // pipeline assumes at least one q_subblock iteration for correct softmax drain + SALAD overlap.
    const bool use_streaming_compute = !fp32_dest_acc_en && qk_out_subblock_h <= 2 &&
                                       Sk_chunk_t % (8 / qk_out_subblock_h) == 0 && qk_in0_num_subblocks > 1;
    log_debug(tt::LogOp, "use_streaming_compute: {}", use_streaming_compute);

    // log all values
    log_debug(tt::LogOp, "dst_size: {}", dst_size);
    log_debug(tt::LogOp, "qk_in0_block_w: {}", qk_in0_block_w);
    log_debug(tt::LogOp, "qk_out_subblock_w: {}", qk_out_subblock_w);
    log_debug(tt::LogOp, "qk_out_subblock_h: {}", qk_out_subblock_h);
    log_debug(tt::LogOp, "qk_in0_num_subblocks: {}", qk_in0_num_subblocks);
    log_debug(tt::LogOp, "qk_in1_num_subblocks: {}", qk_in1_num_subblocks);
    log_debug(tt::LogOp, "qk_num_blocks: {}", qk_num_blocks);
    log_debug(tt::LogOp, "out_in0_block_w: {}", out_in0_block_w);
    log_debug(tt::LogOp, "out_out_subblock_w: {}", out_out_subblock_w);
    log_debug(tt::LogOp, "out_out_subblock_h: {}", out_out_subblock_h);
    log_debug(tt::LogOp, "out_in0_num_subblocks: {}", out_in0_num_subblocks);
    log_debug(tt::LogOp, "out_in1_num_subblocks: {}", out_in1_num_subblocks);
    log_debug(tt::LogOp, "out_num_blocks: {}", out_num_blocks);

    // Determine granularity for statistics computation
    // Each granularity must evenly divide its tile count to avoid dropping tiles
    const uint32_t stats_granularity = detail::find_valid_granularity(Sq_chunk_t, dst_size);
    const uint32_t sub_exp_granularity = detail::find_valid_granularity(Sk_chunk_t, dst_size);
    const uint32_t mul_bcast_granularity = detail::find_valid_granularity(Sq_chunk_t * Sk_chunk_t, dst_size);
    const uint32_t dht_granularity = detail::find_valid_granularity(DHt, dst_size);
    const uint32_t reduce_granularity = detail::find_valid_granularity(Sq_chunk_t, dst_size / 2);

    // Log these
    log_debug(tt::LogOp, "stats_granularity: {}", stats_granularity);
    log_debug(tt::LogOp, "sub_exp_granularity: {}", sub_exp_granularity);
    log_debug(tt::LogOp, "mul_bcast_granularity: {}", mul_bcast_granularity);
    log_debug(tt::LogOp, "dht_granularity: {}", dht_granularity);
    log_debug(tt::LogOp, "reduce_granularity: {}", reduce_granularity);

    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    class bfloat16 bfloat_identity_scalar(1.0f);
    uint32_t packed_identity_scalar = pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});

    union {
        float f;
        uint32_t u;
    } scale_union{};
    scale_union.f = scale.value_or(1.0f);

    // log scale
    log_debug(tt::LogOp, "scale: {}", scale_union.f);

    std::vector<uint32_t> reader_compile_time_args = {
        B,
        NH,
        DHt,
        Sq_chunk_t,
        Sk_chunk_t,
        local_padded_N,
        local_padded_Nt,
        padded_Nt,
        static_cast<uint32_t>(args.logical_n),
        logical_nt,
        Lt,
        L,
        num_local_q_chunks,
        num_joint_q_chunks,
        num_local_k_chunks,
        num_joint_k_chunks,
        num_q_chunks,
        args.ring_size,
        qk_out_subblock_h};

    TensorAccessorArgs(input_tensor_q.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(gathered_input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(gathered_input_tensor_v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(joint_tensor_q.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(joint_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(joint_tensor_v.buffer()).append_to(reader_compile_time_args);

    /**
     * Create semaphores used for L1-L1 store-and-forward of KV between cores.
     */
    auto sender_semaphore_id = CreateSemaphore(program, core_grid, INVALID);
    auto receiver_semaphore_id = CreateSemaphore(program, core_grid, INVALID);
    auto valid_semaphore_id = CreateSemaphore(program, core_grid, VALID);

    // Append semaphore ids to reader compile-time args (must match reader kernel expectations)
    const auto sem_args_offset = reader_compile_time_args.size();
    reader_compile_time_args.push_back(sender_semaphore_id);
    reader_compile_time_args.push_back(receiver_semaphore_id);
    reader_compile_time_args.push_back(valid_semaphore_id);
    reader_compile_time_args.push_back(0);  // mcast_enabled placeholder (patched after chain construction)

    std::vector<uint32_t> writer_compile_time_args = {
        B,
        NH,
        DHt,
        Sq_chunk_t,
        Sk_chunk_t,
        local_padded_N,
        local_padded_Nt,
        padded_Nt,
        args.logical_n,
        logical_nt,
        Lt,
        L,
        num_local_q_chunks,
        num_joint_q_chunks,
        num_local_k_chunks,
        num_joint_k_chunks,
        num_q_chunks,
        packed_identity_scalar,
        scale_union.u,
        args.ring_size,
        global_n_partial_col,
        joint_l_partial_col,
        (std::uint32_t)use_streaming_compute,
    };

    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(joint_output_tensor.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(stats_output_tensor.buffer()).append_to(writer_compile_time_args);

    // Early format check: when all data formats are identical, reconfig calls can be skipped.
    const tt::DataFormat q_df_early = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_q.dtype());
    const tt::DataFormat k_df_early = tt::tt_metal::datatype_to_dataformat_converter(gathered_input_tensor_k.dtype());
    const tt::DataFormat v_df_early = tt::tt_metal::datatype_to_dataformat_converter(gathered_input_tensor_v.dtype());
    const tt::DataFormat out_df_early = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    const tt::DataFormat im_df_early = tt::DataFormat::Float16_b;
    const tt::DataFormat mask_df_early = tt::DataFormat::Float16_b;
    const bool uniform_dataformat =
        (q_df_early == k_df_early && q_df_early == v_df_early && q_df_early == out_df_early &&
         q_df_early == mask_df_early && q_df_early == im_df_early);

    std::vector<uint32_t> compute_compile_time_args = {
        B,
        NH,
        DHt,
        Sq_chunk_t,
        Sk_chunk_t,
        local_padded_N,
        local_padded_Nt,
        padded_Nt,
        args.logical_n,
        logical_nt,
        Lt,
        L,
        num_local_q_chunks,
        num_joint_q_chunks,
        num_local_k_chunks,
        num_joint_k_chunks,
        num_q_chunks,
        args.ring_size,
        qk_in0_block_w,
        qk_out_subblock_w,
        qk_out_subblock_h,
        qk_in0_num_subblocks,
        qk_in1_num_subblocks,
        qk_num_blocks,
        out_in0_block_w,
        out_out_subblock_w,
        out_out_subblock_h,
        out_in0_num_subblocks,
        out_in1_num_subblocks,
        out_num_blocks,
        scale_union.u,
        (std::uint32_t)use_streaming_compute,
        global_n_partial_col,
        joint_l_partial_col,
        (std::uint32_t)uniform_dataformat,
    };

    std::map<std::string, std::string> defines;
    defines["STATS_GRANULARITY"] = std::to_string(stats_granularity);
    defines["SUB_EXP_GRANULARITY"] = std::to_string(sub_exp_granularity);
    defines["MUL_BCAST_GRANULARITY"] = std::to_string(mul_bcast_granularity);
    defines["DHT_GRANULARITY"] = std::to_string(dht_granularity);
    defines["REDUCE_GRANULARITY"] = std::to_string(reduce_granularity);
    defines["EXP_APPROX_MODE"] = std::to_string(exp_approx_mode);

    // NOTE: CreateKernel calls are deferred until after chain construction so that
    // the mcast_enabled compile-time arg can be determined first.

    // Create circular buffers

    tt::DataFormat q_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_q.dtype());
    tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(gathered_input_tensor_k.dtype());
    tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(gathered_input_tensor_v.dtype());
    // Lightweight mask: both streaming and non-streaming paths use Float16_b
    // to support L1-accumulation and avoid Bfp4_b precision loss.
    tt::DataFormat mask_df = tt::DataFormat::Float16_b;
    tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::DataFormat scalar_df = tt::DataFormat::Float16_b;
    tt::DataFormat im_df = tt::DataFormat::Float16_b;  // need to disable fp32 cbs (Issue #13364) fp32_dest_acc_en ?
                                                       // tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat stats_df = im_df;

    uint32_t q_tile_size = tt::tile_size(q_df);
    uint32_t k_tile_size = tt::tile_size(k_df);
    uint32_t v_tile_size = tt::tile_size(v_df);
    uint32_t mask_tile_size = tt::tile_size(mask_df);
    uint32_t out_tile_size = tt::tile_size(out_df);
    uint32_t scalar_tile_size = tt::tile_size(scalar_df);
    uint32_t im_tile_size = tt::tile_size(im_df);
    uint32_t stats_tile_size = tt::tile_size(stats_df);

    log_debug(tt::LogOp, "q_data_format: {}", q_df);
    log_debug(tt::LogOp, "k_data_format: {}", k_df);
    log_debug(tt::LogOp, "v_data_format: {}", v_df);
    log_debug(tt::LogOp, "mask_data_format: {}", mask_df);
    log_debug(tt::LogOp, "out_data_format: {}", out_df);
    log_debug(tt::LogOp, "scalar_data_format: {}", scalar_df);
    log_debug(tt::LogOp, "intermediate_data_format: {}", im_df);
    log_debug(tt::LogOp, "statistics_data_format: {}", stats_df);

    // Q input
    auto c_in0_config = CircularBufferConfig(q_tiles * q_tile_size, {{tt::CBIndex::c_0, q_df}})
                            .set_page_size(tt::CBIndex::c_0, q_tile_size);

    CreateCircularBuffer(program, core_grid, c_in0_config);
    // K and V input CBs.
    // Fabric writer columns get overlapping handles (c_1+c_14 for K, c_2+c_15 for V) so both
    // compute and writer can pop independently from the same address space.
    // Non-fabric columns get single-handle CBs to avoid the unused second handle blocking
    // space reclamation.
    CoreRange non_fabric_core_range({0, 0}, {grid_size.x - 3, grid_size.y - 1});
    CoreRange fabric_core_range({grid_size.x - 2, 0}, {grid_size.x - 1, grid_size.y - 1});
    {
        // DANGEROUS
        // K input: non-fabric cores (single handle)
        auto c_in1_config = CircularBufferConfig(k_tiles * k_tile_size, {{tt::CBIndex::c_1, k_df}})
                                .set_page_size(tt::CBIndex::c_1, k_tile_size);
        CreateCircularBuffer(program, non_fabric_core_range, c_in1_config);
        // K input: fabric cores (overlapping handles for compute + writer)
        uint32_t k_cbs[] = {tt::CBIndex::c_1, tt::CBIndex::c_14};
        tt::tt_metal::create_cb(k_cbs, program, fabric_core_range, k_tile_size, k_tiles, k_df);
    }
    {
        // V input: non-fabric cores (single handle)
        auto c_in2_config = CircularBufferConfig(v_tiles * v_tile_size, {{tt::CBIndex::c_2, v_df}})
                                .set_page_size(tt::CBIndex::c_2, v_tile_size);
        CreateCircularBuffer(program, non_fabric_core_range, c_in2_config);
        // V input: fabric cores (overlapping handles for compute + writer)
        uint32_t v_cbs[] = {tt::CBIndex::c_2, tt::CBIndex::c_15};
        tt::tt_metal::create_cb(v_cbs, program, fabric_core_range, v_tile_size, v_tiles, v_df);
    }

    // Lightweight mask: single CB holds 1 neginf tile + up to 2 partial mask tiles
    if (needs_lightweight_mask) {
        auto c_in3_config =
            CircularBufferConfig(total_lightweight_mask_tiles * mask_tile_size, {{tt::CB::c_in3, mask_df}})
                .set_page_size(tt::CB::c_in3, mask_tile_size);
        CreateCircularBuffer(program, core_grid, c_in3_config);
    }

    // scale input
    auto c_in4_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{tt::CBIndex::c_4, scalar_df}})
                            .set_page_size(tt::CBIndex::c_4, scalar_tile_size);
    CreateCircularBuffer(program, core_grid, c_in4_config);

    // identity scale input
    auto c_in5_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{tt::CBIndex::c_5, scalar_df}})
                            .set_page_size(tt::CBIndex::c_5, scalar_tile_size);
    CreateCircularBuffer(program, core_grid, c_in5_config);

    // stats input
    auto c_in6_config = CircularBufferConfig(statistics_tiles * im_tile_size, {{tt::CBIndex::c_6, im_df}})
                            .set_page_size(tt::CBIndex::c_6, im_tile_size);
    CreateCircularBuffer(program, core_grid, c_in6_config);

    // previous block output as input
    auto c_in7_config = CircularBufferConfig(out_im_tiles * out_tile_size, {{tt::CBIndex::c_7, out_df}})
                            .set_page_size(tt::CBIndex::c_7, out_tile_size);
    CreateCircularBuffer(program, core_grid, c_in7_config);

    // column identity input
    auto c_in8_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{tt::CBIndex::c_8, scalar_df}})
                            .set_page_size(tt::CBIndex::c_8, scalar_tile_size);
    CreateCircularBuffer(program, core_grid, c_in8_config);

    // cb_qk_im
    auto c_intermed0_config = CircularBufferConfig(qk_tiles * im_tile_size, {{tt::CBIndex::c_24, im_df}})
                                  .set_page_size(tt::CBIndex::c_24, im_tile_size);
    CreateCircularBuffer(program, core_grid, c_intermed0_config);

    // cb_out_im
    auto c_intermed1_config = CircularBufferConfig(out_im_tiles * im_tile_size, {{tt::CBIndex::c_25, im_df}})
                                  .set_page_size(tt::CBIndex::c_25, im_tile_size);
    CreateCircularBuffer(program, core_grid, c_intermed1_config);

    // cb_out_accumulate_im
    auto c_intermed2_config = CircularBufferConfig(out_im_tiles * im_tile_size, {{tt::CBIndex::c_26, im_df}})
                                  .set_page_size(tt::CBIndex::c_26, im_tile_size);
    CreateCircularBuffer(program, core_grid, c_intermed2_config);

    // cb_cur_max
    auto c_intermed3_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_27, stats_df}})
                                  .set_page_size(tt::CBIndex::c_27, stats_tile_size);
    CreateCircularBuffer(program, core_grid, c_intermed3_config);

    // cb_prev_max
    auto c_intermed4_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_28, stats_df}})
                                  .set_page_size(tt::CBIndex::c_28, stats_tile_size);
    CreateCircularBuffer(program, core_grid, c_intermed4_config);

    // cb_cur_sum
    auto c_intermed5_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_29, stats_df}})
                                  .set_page_size(tt::CBIndex::c_29, stats_tile_size);
    CreateCircularBuffer(program, core_grid, c_intermed5_config);

    // cb_prev_sum
    auto c_intermed6_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_30, stats_df}})
                                  .set_page_size(tt::CBIndex::c_30, stats_tile_size);
    CreateCircularBuffer(program, core_grid, c_intermed6_config);

    // cb_exp_max_diff
    auto c_intermed7_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_31, stats_df}})
                                  .set_page_size(tt::CBIndex::c_31, stats_tile_size);
    CreateCircularBuffer(program, core_grid, c_intermed7_config);

    // Output
    auto c_out0_config = CircularBufferConfig(out0_t * out_tile_size, {{tt::CBIndex::c_16, out_df}})
                             .set_page_size(tt::CBIndex::c_16, out_tile_size);
    CreateCircularBuffer(program, core_grid, c_out0_config);

    // stats output
    auto c_out1_config = CircularBufferConfig(statistics_tiles * im_tile_size, {{tt::CBIndex::c_17, im_df}})
                             .set_page_size(tt::CBIndex::c_17, im_tile_size);
    CreateCircularBuffer(program, core_grid, c_out1_config);

    // Streaming compute v2: 1-tile recip scratch CB (c_9) for normalize_row_streaming.
    // c_4 is used by cb_scale_in in ring joint, so we use c_9 instead.
    if (use_streaming_compute) {
        auto c_recip_scratch_config = CircularBufferConfig(1 * im_tile_size, {{tt::CBIndex::c_9, im_df}})
                                          .set_page_size(tt::CBIndex::c_9, im_tile_size);
        CreateCircularBuffer(program, core_grid, c_recip_scratch_config);
    }

    // Deferred norm: sum save/restore CBs for multi Q-chunk DRAM round-trip.
    // cb_sum_out (c_10) = compute pushes sum for writer to save to DRAM.
    // cb_sum_in (c_11) = writer pushes restored sum from DRAM for compute to read.
    if (use_streaming_compute) {
        auto c_sum_out_config =
            CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_10, stats_df}})
                .set_page_size(tt::CBIndex::c_10, stats_tile_size);
        CreateCircularBuffer(program, core_grid, c_sum_out_config);

        auto c_sum_in_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_11, stats_df}})
                                   .set_page_size(tt::CBIndex::c_11, stats_tile_size);
        CreateCircularBuffer(program, core_grid, c_sum_in_config);
    }

    uint32_t q_addr = input_tensor_q.buffer()->address();
    uint32_t k_addr = input_tensor_k.buffer()->address();
    uint32_t v_addr = input_tensor_v.buffer()->address();
    uint32_t gathered_k_addr = gathered_input_tensor_k.buffer()->address();
    uint32_t gathered_v_addr = gathered_input_tensor_v.buffer()->address();
    uint32_t joint_q_addr = joint_tensor_q.buffer()->address();
    uint32_t joint_k_addr = joint_tensor_k.buffer()->address();
    uint32_t joint_v_addr = joint_tensor_v.buffer()->address();
    uint32_t out_addr = output_tensor.buffer()->address();
    uint32_t joint_out_addr = joint_output_tensor.buffer()->address();
    uint32_t stats_addr = stats_output_tensor.buffer()->address();

    /**
     * Build chain selection for store-and-forward across cores per (batch, head).
     */
    struct CoreHeadWork {
        uint32_t batch = 0;
        uint32_t head = 0;
        uint32_t q_chunk_start = 0;
        uint32_t q_chunk_count = 0;
    };

    struct CoreWork {
        CoreCoord logical_core;
        CoreCoord physical_core;
        uint32_t global_q_start = 0;
        uint32_t global_q_count = 0;
        std::vector<CoreHeadWork> head_work;
    };

    struct HeadSegmentRef {
        uint32_t core_idx = 0;
        uint32_t head_work_index = 0;
    };

    struct CoreChainInfo {
        bool participates = false;
        bool is_injector = false;
        bool is_sink = false;
        uint32_t batch = 0;
        uint32_t head = 0;
        uint32_t q_chunk_start = 0;
        uint32_t q_chunk_count = 0;
        CoreCoord prev_physical = CoreCoord{0, 0};
        CoreCoord next_physical = CoreCoord{0, 0};
        uint32_t next_core_q_chunks = 0;
        bool use_mcast = false;
        uint32_t mcast_num_dests = 0;
        uint32_t mcast_sender_wait = 0;
    };

    std::vector<CoreWork> core_work(num_cores);
    std::vector<CoreChainInfo> core_chain_info(num_cores);
    const uint32_t total_heads = B * NH;
    std::vector<std::vector<HeadSegmentRef>> head_segments(total_heads);

    // Evenly distribute flat global q chunks across cores
    const uint32_t total_q_chunks = B * NH * num_q_chunks;
    const uint32_t base_chunks_per_core = (num_cores == 0) ? 0 : (total_q_chunks / num_cores);
    const uint32_t extra_chunks = (num_cores == 0) ? 0 : (total_q_chunks % num_cores);

    log_info(
        tt::LogOp,
        "[ExpRingJointSDPA] grid={}x{}={} cores, B={}, NH={}, num_q_chunks={}({} local+{} joint), "
        "base_chunks_per_core={} (+{} extras)",
        grid_size.x,
        grid_size.y,
        num_cores,
        B,
        NH,
        num_q_chunks,
        num_local_q_chunks,
        num_joint_q_chunks,
        base_chunks_per_core,
        extra_chunks);
    uint32_t next_global_chunk = 0;

    auto decode_flat_chunk = [&](uint32_t flat_chunk_index) {
        const uint32_t head_span = num_q_chunks;
        const uint32_t head_index = head_span == 0 ? 0 : (flat_chunk_index / head_span);
        const uint32_t q_chunk = head_span == 0 ? 0 : (flat_chunk_index % head_span);
        const uint32_t batch = (NH == 0) ? 0 : (head_index / NH);
        const uint32_t head = (NH == 0) ? 0 : (head_index % NH);
        return std::tuple<uint32_t, uint32_t, uint32_t>{batch, head, q_chunk};
    };

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};
        uint32_t chunk_count = base_chunks_per_core + ((i < extra_chunks) ? 1 : 0);
        if (next_global_chunk >= total_q_chunks) {
            chunk_count = 0;
        } else if (chunk_count > total_q_chunks - next_global_chunk) {
            chunk_count = total_q_chunks - next_global_chunk;
        }

        auto& work = core_work.at(i);
        work.logical_core = core;
        work.physical_core = device->worker_core_from_logical_core(core);
        work.global_q_start = next_global_chunk;
        work.global_q_count = chunk_count;

        uint32_t remaining = chunk_count;
        uint32_t flat_chunk = next_global_chunk;
        while (remaining > 0) {
            auto [batch_idx, head_idx, q_chunk_idx] = decode_flat_chunk(flat_chunk);
            uint32_t chunk_capacity_in_head = num_q_chunks - q_chunk_idx;
            uint32_t chunk_take = std::min(remaining, chunk_capacity_in_head);

            work.head_work.push_back(CoreHeadWork{
                .batch = batch_idx,
                .head = head_idx,
                .q_chunk_start = q_chunk_idx,
                .q_chunk_count = chunk_take,
            });

            if (!head_segments.empty()) {
                uint32_t head_id = (batch_idx * NH) + head_idx;
                if (head_id < head_segments.size()) {
                    head_segments[head_id].push_back(HeadSegmentRef{
                        .core_idx = i, .head_work_index = static_cast<uint32_t>(work.head_work.size() - 1)});
                }
            }

            remaining -= chunk_take;
            flat_chunk += chunk_take;
        }

        next_global_chunk += chunk_count;
    }

    // Construct chains: for each head that spans >= 2 cores, pick first core
    // with single head segment as injector. Linear forward traversal only —
    // no wrap-around (wrapping back would pull in straddling cores whose
    // q_iter_local is inflated by prior-head work, causing deadlock).
    // Injector reselection for DRAM channel spreading is deferred to the
    // mcast eligibility pass below.
    for (auto& segments : head_segments) {
        if (segments.size() < 2) {
            continue;
        }

        std::optional<std::size_t> chain_start_idx;
        for (std::size_t idx = 0; idx + 1 < segments.size(); ++idx) {
            const auto& seg = segments.at(idx);
            const auto& work = core_work.at(seg.core_idx);
            if (work.global_q_count == 0) {
                continue;
            }
            if (work.head_work.size() == 1) {
                chain_start_idx = idx;
                break;
            }
        }

        if (!chain_start_idx.has_value()) {
            continue;
        }

        const std::size_t start = chain_start_idx.value();
        for (std::size_t idx = start; idx < segments.size(); ++idx) {
            const auto& seg = segments.at(idx);
            const uint32_t core_idx = seg.core_idx;
            const auto& hw = core_work.at(core_idx).head_work.at(seg.head_work_index);
            auto& chain = core_chain_info.at(core_idx);

            chain.participates = true;
            chain.batch = hw.batch;
            chain.head = hw.head;
            chain.q_chunk_start = hw.q_chunk_start;
            chain.q_chunk_count = hw.q_chunk_count;

            if (idx == start) {
                chain.is_injector = true;
            }
            if (idx == segments.size() - 1) {
                chain.is_sink = true;
            }

            if (idx > start) {
                const uint32_t prev_core_idx = segments.at(idx - 1).core_idx;
                chain.prev_physical = core_work.at(prev_core_idx).physical_core;
            }
            if (idx + 1 < segments.size()) {
                const uint32_t next_core_idx = segments.at(idx + 1).core_idx;
                chain.next_physical = core_work.at(next_core_idx).physical_core;
                const auto& next_hw = core_work.at(next_core_idx).head_work.at(segments.at(idx + 1).head_work_index);
                chain.next_core_q_chunks = next_hw.q_chunk_count;
            }
        }
    }

    // Log chain summary
    {
        uint32_t num_chains = 0;
        std::vector<uint32_t> chain_len_counts(num_cores + 1, 0);
        for (uint32_t hi = 0; hi < total_heads; ++hi) {
            const auto& segs = head_segments[hi];
            const uint32_t batch_id = hi / NH, head_id = hi % NH;
            uint32_t chain_len = 0;
            for (const auto& seg : segs) {
                const auto& ci = core_chain_info[seg.core_idx];
                if (ci.participates && ci.batch == batch_id && ci.head == head_id) {
                    chain_len++;
                }
            }
            if (chain_len >= 2) {
                num_chains++;
                chain_len_counts[chain_len]++;
            }
        }
        std::string hist_str;
        for (uint32_t len = 2; len <= num_cores; ++len) {
            if (chain_len_counts[len] > 0) {
                hist_str += std::to_string(chain_len_counts[len]) + "x" + std::to_string(len) + "-core ";
            }
        }
        log_info(
            tt::LogOp,
            "[ExpRingJointSDPA] {} chains ({})",
            num_chains,
            hist_str.empty() ? "none" : hist_str);
    }

    // Third pass: Check multicast eligibility and configure mcast for eligible chains
    uint32_t mcast_chains = 0;
    {
        struct McastCandidate {
            std::vector<uint32_t> core_indices;
            uint32_t ref_q_chunks;
        };
        std::vector<McastCandidate> candidates;
        bool all_eligible = true;

        for (uint32_t head_id = 0; head_id < head_segments.size(); ++head_id) {
            const auto& segments = head_segments[head_id];
            if (segments.size() < 2) {
                continue;
            }

            std::vector<uint32_t> chain_core_indices;
            for (const auto& seg : segments) {
                if (seg.core_idx < core_chain_info.size() && core_chain_info[seg.core_idx].participates &&
                    core_chain_info[seg.core_idx].batch == (head_id / NH) &&
                    core_chain_info[seg.core_idx].head == (head_id % NH)) {
                    chain_core_indices.push_back(seg.core_idx);
                }
            }

            if (chain_core_indices.size() < 2) {
                continue;
            }

            // Eligibility condition 1: All physical cores share the same Y coordinate
            const uint32_t ref_y = core_work[chain_core_indices[0]].physical_core.y;
            bool same_row = true;
            for (size_t ci = 1; ci < chain_core_indices.size(); ++ci) {
                if (core_work[chain_core_indices[ci]].physical_core.y != ref_y) {
                    same_row = false;
                    break;
                }
            }

            if (!same_row) {
                all_eligible = false;
                log_debug(tt::LogOp, "Head {}: mcast ineligible - cores span multiple rows", head_id);
                break;
            }

            // Eligibility condition 2: no non-chain worker cores inside the mcast rectangle.
            uint32_t min_x = core_work[chain_core_indices[0]].physical_core.x;
            uint32_t max_x = min_x;
            for (const auto& ci : chain_core_indices) {
                uint32_t x = core_work[ci].physical_core.x;
                min_x = std::min(min_x, x);
                max_x = std::max(max_x, x);
            }

            bool has_gap = false;
            for (uint32_t ci = 0; ci < num_cores; ++ci) {
                const auto& phys = core_work[ci].physical_core;
                if (phys.y == ref_y && phys.x >= min_x && phys.x <= max_x) {
                    bool in_chain = false;
                    for (const auto& chain_ci : chain_core_indices) {
                        if (chain_ci == ci) {
                            in_chain = true;
                            break;
                        }
                    }
                    if (!in_chain) {
                        has_gap = true;
                        break;
                    }
                }
            }

            if (has_gap) {
                all_eligible = false;
                log_debug(
                    tt::LogOp, "Head {}: mcast ineligible - non-chain worker core inside mcast rectangle", head_id);
                break;
            }

            // Eligibility condition 3: All chain cores must have the same q_chunk_count.
            const uint32_t ref_q_chunks = core_chain_info[chain_core_indices[0]].q_chunk_count;
            bool uniform_q_mcast = true;
            for (size_t ci = 1; ci < chain_core_indices.size(); ++ci) {
                if (core_chain_info[chain_core_indices[ci]].q_chunk_count != ref_q_chunks) {
                    uniform_q_mcast = false;
                    break;
                }
            }

            if (!uniform_q_mcast) {
                all_eligible = false;
                log_debug(tt::LogOp, "Head {}: mcast ineligible - mixed q_chunk_counts", head_id);
                break;
            }

            candidates.push_back(McastCandidate{std::move(chain_core_indices), ref_q_chunks});
        }

        if (all_eligible && !candidates.empty()) {
            mcast_chains = candidates.size();
            // Track injector physical X columns for DRAM channel spreading
            std::vector<uint32_t> injector_phys_x;
            for (const auto& cand : candidates) {
                const uint32_t chain_size = cand.core_indices.size();
                const uint32_t num_receivers = chain_size - 1;

                // Find current injector
                uint32_t injector_idx = cand.core_indices[0];
                for (const auto& ci : cand.core_indices) {
                    if (core_chain_info[ci].is_injector) {
                        injector_idx = ci;
                        break;
                    }
                }

                // Reselect injector for DRAM channel spreading: pick the core
                // whose physical X is furthest from all previously chosen injectors.
                {
                    uint32_t best_idx = injector_idx;
                    uint32_t best_dist = 0;
                    for (const auto& ci : cand.core_indices) {
                        const uint32_t phys_x = core_work[ci].physical_core.x;
                        uint32_t min_dist = UINT32_MAX;
                        for (uint32_t ix : injector_phys_x) {
                            uint32_t d = (phys_x > ix) ? (phys_x - ix) : (ix - phys_x);
                            min_dist = std::min(min_dist, d);
                        }
                        if (min_dist > best_dist) {
                            best_dist = min_dist;
                            best_idx = ci;
                        }
                    }
                    if (best_idx != injector_idx) {
                        // Clear old injector, set new one
                        core_chain_info[injector_idx].is_injector = false;
                        core_chain_info[injector_idx].is_sink = true;
                        core_chain_info[best_idx].is_injector = true;
                        core_chain_info[best_idx].is_sink = false;
                        injector_idx = best_idx;
                    }
                }
                injector_phys_x.push_back(core_work[injector_idx].physical_core.x);

                uint32_t min_x = core_work[cand.core_indices[0]].physical_core.x;
                uint32_t max_x = min_x;
                for (size_t ci = 1; ci < cand.core_indices.size(); ++ci) {
                    uint32_t x = core_work[cand.core_indices[ci]].physical_core.x;
                    min_x = std::min(min_x, x);
                    max_x = std::max(max_x, x);
                }
                const uint32_t injector_y = core_work[injector_idx].physical_core.y;
                const CoreCoord rect_start = CoreCoord{min_x, injector_y};
                const CoreCoord rect_end = CoreCoord{max_x, injector_y};

                const uint32_t injector_x = core_work[injector_idx].physical_core.x;
                const bool injector_inside_rect = (injector_x > min_x && injector_x < max_x);
                const uint32_t mcast_num_dests = injector_inside_rect ? chain_size : num_receivers;

                auto& injector_chain = core_chain_info[injector_idx];
                injector_chain.use_mcast = true;
                injector_chain.prev_physical = rect_start;
                injector_chain.next_physical = rect_end;
                injector_chain.mcast_num_dests = mcast_num_dests;
                injector_chain.mcast_sender_wait = num_receivers;
                injector_chain.next_core_q_chunks = cand.ref_q_chunks;

                for (const auto& ci : cand.core_indices) {
                    if (ci == injector_idx) {
                        continue;
                    }
                    auto& receiver_chain = core_chain_info[ci];
                    receiver_chain.use_mcast = true;
                    receiver_chain.prev_physical = core_work[injector_idx].physical_core;
                    receiver_chain.next_physical = CoreCoord{0, 0};
                    receiver_chain.next_core_q_chunks = 0;
                    receiver_chain.is_sink = true;
                }

                log_debug(
                    tt::LogOp,
                    "Head: mcast enabled - {} receivers, injector core {} (phys_x={}), num_dests={} -> rect "
                    "({},{}) to ({},{})",
                    num_receivers,
                    injector_idx,
                    core_work[injector_idx].physical_core.x,
                    mcast_num_dests,
                    rect_start.x,
                    rect_start.y,
                    rect_end.x,
                    rect_end.y);
            }
        }

        log_debug(
            tt::LogOp,
            "Multicast eligibility: {}/{} chains using mcast (all-or-nothing)",
            mcast_chains,
            static_cast<uint32_t>(candidates.size()));
    }

    log_info(
        tt::LogOp,
        "[ExpRingJointSDPA] mcast: {} ({}/{} chains)",
        mcast_chains > 0 ? "ENABLED" : "DISABLED",
        mcast_chains,
        mcast_chains > 0 ? mcast_chains : static_cast<uint32_t>(
            std::count_if(core_chain_info.begin(), core_chain_info.end(), [](const CoreChainInfo& c) { return c.is_injector; })));

    // Update mcast_enabled compile-time arg now that chain construction is complete
    reader_compile_time_args[sem_args_offset + 3] = (mcast_chains > 0) ? 1 : 0;

    // Map (batch * NH + head) -> injector physical coordinates for MUX writer signaling
    std::unordered_map<uint32_t, CoreCoord> injector_physical_by_head;
    for (uint32_t ci = 0; ci < num_cores; ++ci) {
        if (core_chain_info[ci].is_injector) {
            uint32_t head_key = core_chain_info[ci].batch * NH + core_chain_info[ci].head;
            injector_physical_by_head[head_key] = core_work[ci].physical_core;
        }
    }

    // ---- Fabric MUX config (needed for writer kernel CT args below) ----
    // Hardcoded positions: backward-direction MUX at (11,0), (11,5); forward-direction MUX at (11,4), (11,9).
    const std::vector<CoreCoord> mux_backward_logical_cores = {{11, 0}, {11, 5}};
    const std::vector<CoreCoord> mux_forward_logical_cores = {{11, 4}, {11, 9}};

    const uint32_t l1_unreserved_base_address =
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const uint32_t num_mux_full_size_channels = args.num_workers_per_link;
    const uint32_t num_mux_header_only_channels = 0;
    const uint32_t num_mux_buffers_per_channel = args.num_buffers_per_channel;
    const size_t mux_buffer_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
        num_mux_full_size_channels,
        num_mux_header_only_channels,
        num_mux_buffers_per_channel,
        0,
        mux_buffer_size_bytes,
        l1_unreserved_base_address);

    // Create kernels (deferred until after chain construction for mcast_enabled flag)
    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/exp_ring_joint_reader.cpp",
        core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, defines));

    // Non-fabric writer: columns 0..(grid_size.x-3)
    // grid_size.x-2 and grid_size.x-1 are fabric MUX client columns
    CoreRange non_fabric_writer_range({0, 0}, {grid_size.x - 3, grid_size.y - 1});
    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/exp_ring_joint_writer.cpp",
        non_fabric_writer_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, defines));

    // Fabric writer: columns grid_size.x-2 and grid_size.x-1 (backward and forward MUX clients)
    CoreRange fabric_writer_range({grid_size.x - 2, 0}, {grid_size.x - 1, grid_size.y - 1});
    auto writer_fabric_compile_time_args = writer_compile_time_args;
    fabric_mux_connection_ct_args(args.num_workers_per_link, mux_kernel_config, writer_fabric_compile_time_args);

    // All-gather CT args for the fabric writer (integrated K/V all-gather on MUX client columns)
    const uint32_t ag_kv_scratch_cb_id = tt::CBIndex::c_12;
    const uint32_t ag_pkt_hdr_cb_id = tt::CBIndex::c_13;
    const uint32_t ag_page_size = input_tensor_k.buffer()->page_size();
    const size_t ag_packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t ag_packet_size_in_pages =
        std::min(static_cast<uint32_t>(ag_packet_size_bytes / ag_page_size), uint32_t{2});

    writer_fabric_compile_time_args.push_back(device_index);
    writer_fabric_compile_time_args.push_back(ag_packet_size_in_pages);
    writer_fabric_compile_time_args.push_back(ag_page_size);
    writer_fabric_compile_time_args.push_back(ag_pkt_hdr_cb_id);
    writer_fabric_compile_time_args.push_back(ag_kv_scratch_cb_id);
    writer_fabric_compile_time_args.push_back(num_targets_forward);
    writer_fabric_compile_time_args.push_back(num_targets_backward);
    writer_fabric_compile_time_args.push_back(static_cast<uint32_t>(args.topology));
    TensorAccessorArgs(input_tensor_k.buffer()).append_to(writer_fabric_compile_time_args);
    TensorAccessorArgs(input_tensor_v.buffer()).append_to(writer_fabric_compile_time_args);
    TensorAccessorArgs(gathered_input_tensor_k.buffer()).append_to(writer_fabric_compile_time_args);
    TensorAccessorArgs(gathered_input_tensor_v.buffer()).append_to(writer_fabric_compile_time_args);

    auto writer_fabric_defines = defines;
    writer_fabric_defines["USE_MUX"] = "1";
    auto writer_fabric_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/exp_ring_joint_writer.cpp",
        fabric_writer_range,
        tt::tt_metal::WriterDataMovementConfig(writer_fabric_compile_time_args, writer_fabric_defines));

    // K/V staging scratch CB and packet header CB for all-gather on fabric writer cores
    const tt::DataFormat ag_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_k.dtype());
    const uint32_t ag_cb_pages = 3 * ag_packet_size_in_pages;
    tt::tt_metal::CreateCircularBuffer(
        program,
        fabric_writer_range,
        tt::tt_metal::CircularBufferConfig(ag_cb_pages * ag_page_size, {{ag_kv_scratch_cb_id, ag_df}})
            .set_page_size(ag_kv_scratch_cb_id, ag_page_size));
    const uint32_t ag_pkt_hdr_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CreateCircularBuffer(
        program,
        fabric_writer_range,
        tt::tt_metal::CircularBufferConfig(8 * ag_pkt_hdr_size, {{ag_pkt_hdr_cb_id, tt::DataFormat::RawUInt32}})
            .set_page_size(ag_pkt_hdr_cb_id, ag_pkt_hdr_size));

    auto compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/exp_ring_joint_sdpa.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
            .defines = defines});

    // ---- AllGatherFusedOpSignaler for integrated all-gather on MUX client columns 9-10 ----
    auto all_gather_fused_op_signaler = ttnn::experimental::ccl::AllGatherFusedOpSignaler();
    all_gather_fused_op_signaler.init_fused_op(
        sdpa_fused_op_signaler->fused_op_receiver_cores_noc,
        sdpa_fused_op_signaler->fused_op_receiver_signal_semaphores,
        sdpa_fused_op_signaler->fused_op_signaler_mode);

    // Build backward and forward termination master core sets (1 per link per direction)
    // Backward masters: row 0 of both MUX client columns (top half = backward direction).
    // Forward masters:  row num_workers_per_link of both MUX client columns (bottom half = forward).
    // Link is determined by column: col grid_size.x-2 = link 0, col grid_size.x-1 = link 1.
    std::vector<CoreCoord> ag_backward_master_cores, ag_forward_master_cores;
    std::set<CoreRange> ag_backward_master_ranges, ag_forward_master_ranges;
    for (uint32_t col_offset = 0; col_offset < 2; ++col_offset) {
        CoreCoord bwd_master = {grid_size.x - 2 + col_offset, 0};
        CoreCoord fwd_master = {grid_size.x - 2 + col_offset, args.num_workers_per_link};
        ag_backward_master_cores.push_back(bwd_master);
        ag_forward_master_cores.push_back(fwd_master);
        ag_backward_master_ranges.insert(CoreRange(bwd_master));
        ag_forward_master_ranges.insert(CoreRange(fwd_master));
    }
    auto fused_op_signaler_backward = all_gather_fused_op_signaler;
    auto fused_op_signaler_forward = all_gather_fused_op_signaler;
    // Pass the full direction-half range across both MUX client columns so that
    // CreateSemaphore inside init_all_gather allocates the AG sync semaphore on ALL
    // workers in each direction group.  This ensures every core (both term-masters
    // and non-masters) has the same number of semaphores allocated before
    // fabric_mux_connection_rt_args runs, keeping termination_sync IDs consistent.
    CoreRange all_backward_clients({grid_size.x - 2, 0}, {grid_size.x - 1, args.num_workers_per_link - 1});
    CoreRange all_forward_clients({grid_size.x - 2, args.num_workers_per_link}, {grid_size.x - 1, grid_size.y - 1});
    fused_op_signaler_backward.init_all_gather(
        program, mesh_device, CoreRangeSet({all_backward_clients}), ag_backward_master_cores);
    fused_op_signaler_forward.init_all_gather(
        program, mesh_device, CoreRangeSet({all_forward_clients}), ag_forward_master_cores);

    // K/V tensor shape info for all-gather RT args
    const auto& ag_input_shape = input_tensor_k.padded_shape();
    const auto& ag_output_shape = gathered_input_tensor_k.padded_shape();
    TT_ASSERT(!(ag_input_shape[3] % tt::constants::TILE_WIDTH));
    TT_ASSERT(!(ag_output_shape[3] % tt::constants::TILE_WIDTH));
    const uint32_t ag_input_Wt = ag_input_shape[3] / tt::constants::TILE_WIDTH;
    const uint32_t ag_input_Ht = ag_input_shape[2] / tt::constants::TILE_HEIGHT;
    const uint32_t ag_output_Wt = ag_output_shape[3] / tt::constants::TILE_WIDTH;
    const uint32_t ag_output_Ht = ag_output_shape[2] / tt::constants::TILE_HEIGHT;
    const uint32_t ag_batch_head_count = ag_input_shape[0] * ag_input_shape[1];
    const uint32_t ag_single_bh_num_tiles = input_tensor_k.buffer()->num_pages() / ag_batch_head_count;

    // Track the RT arg offset for all-gather args on fabric writer master cores
    uint32_t writer_fabric_ag_rt_offset = 0;
    bool ag_rt_offset_set = false;

    // Track the RT arg offset for the fused-op global semaphore address in reader args
    uint32_t reader_fused_op_sem_rt_offset = 0;

    // Set reader rt args
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        // Prefer the computed even distribution above for chain construction
        const auto& work = core_work.at(i);
        uint32_t global_q_start = work.global_q_start;
        uint32_t global_q_end = work.global_q_start + work.global_q_count;

        // Direction: top half of rows = backward (0), bottom half = forward (1)
        const uint32_t direction = (core.y < args.num_workers_per_link) ? 0 : 1;

        // log the above
        log_debug(tt::LogOp, "core: {}", i);
        log_debug(tt::LogOp, "x={},y={}", core.x, core.y);
        log_debug(tt::LogOp, "global_q_start: {}", global_q_start);
        log_debug(tt::LogOp, "global_q_end: {}", global_q_end);

        std::vector<uint32_t> reader_args = {
            q_addr,
            k_addr,
            v_addr,
            gathered_k_addr,
            gathered_v_addr,
            joint_q_addr,
            joint_k_addr,
            joint_v_addr,
            global_q_start,
            global_q_end,
        };
        // Append chain runtime args for store-and-forward
        const auto& chain = core_chain_info.at(i);

        log_debug(
            tt::LogOp,
            "core logical=({},{})->phys=({},{}), q=[{},{}), chain={{part:{}, inj:{}, sink:{}, "
            "b:{}, h:{}, q_start:{}, q_cnt:{}, next_cnt:{}}}",
            core.x,
            core.y,
            core_work.at(i).physical_core.x,
            core_work.at(i).physical_core.y,
            global_q_start,
            global_q_end,
            chain.participates,
            chain.is_injector,
            chain.is_sink,
            chain.batch,
            chain.head,
            chain.q_chunk_start,
            chain.q_chunk_count,
            chain.next_core_q_chunks);

        reader_args.push_back(static_cast<uint32_t>(chain.participates));
        reader_args.push_back(static_cast<uint32_t>(chain.is_injector));
        reader_args.push_back(static_cast<uint32_t>(chain.is_sink));
        reader_args.push_back(chain.batch);
        reader_args.push_back(chain.head);
        reader_args.push_back(chain.q_chunk_start);
        reader_args.push_back(chain.q_chunk_count);
        reader_args.push_back(static_cast<uint32_t>(chain.prev_physical.x));
        reader_args.push_back(static_cast<uint32_t>(chain.prev_physical.y));
        reader_args.push_back(static_cast<uint32_t>(chain.next_physical.x));
        reader_args.push_back(static_cast<uint32_t>(chain.next_physical.y));
        reader_args.push_back(chain.next_core_q_chunks);
        reader_args.push_back(chain.mcast_num_dests);
        reader_args.push_back(chain.mcast_sender_wait);

        // Determine if this core's writer has a valid MUX connection (for reader-side forwarding)
        const bool is_mux_client = (core.x >= grid_size.x - 2);
        bool is_mux_writer_valid = false;
        if (is_mux_client) {
            const uint32_t half_within_col = core.y / args.num_workers_per_link;
            const bool is_backward = (half_within_col == 0);
            const uint32_t link = (core.x == grid_size.x - 1) ? 1 : 0;
            const bool link_in_range = (link < args.num_links) && (link < mux_backward_logical_cores.size()) &&
                                       (link < mux_forward_logical_cores.size());
            if (link_in_range) {
                const bool valid = is_backward ? backward_coord.has_value() : forward_coord.has_value();
                is_mux_writer_valid = valid;
            }
        }
        reader_args.push_back(static_cast<uint32_t>(is_mux_writer_valid));

        // Inject fused-op synchronization RT args (AllGather) here; it will append to reader_args
        // The semaphore address is the 4th value pushed (index = current size + 3)
        reader_fused_op_sem_rt_offset = reader_args.size() + 3;
        sdpa_fused_op_signaler->push_ring_sdpa_fused_op_rt_args(reader_args, direction);

        SetRuntimeArgs(program, reader_kernels_id, core, reader_args);

        // Writer args
        std::vector<uint32_t> writer_args = {
            out_addr,
            joint_out_addr,
            stats_addr,
            global_q_start,
            global_q_end,
        };
        sdpa_fused_op_signaler->push_ring_sdpa_fused_op_rt_args(writer_args, direction);

        if (is_mux_client) {
            // Direction is determined by row half: top half = backward, bottom half = forward.
            // Link is determined by column: col grid_size.x-2 = link 0, col grid_size.x-1 = link 1.
            const uint32_t half_within_col = core.y / args.num_workers_per_link;
            const bool is_backward = (half_within_col == 0);
            const uint32_t link = (core.x == grid_size.x - 1) ? 1 : 0;
            const uint32_t worker_idx = core.y % args.num_workers_per_link;
            const bool is_term_master = (worker_idx == 0);
            const CoreCoord termination_master_logical = {core.x, half_within_col * args.num_workers_per_link};

            const bool link_in_range = (link < args.num_links) && (link < mux_backward_logical_cores.size()) &&
                                       (link < mux_forward_logical_cores.size());
            if (link_in_range) {
                const CoreCoord& mux_core =
                    is_backward ? mux_backward_logical_cores[link] : mux_forward_logical_cores[link];
                const bool valid = is_backward ? backward_coord.has_value() : forward_coord.has_value();
                fabric_mux_connection_rt_args(
                    valid,
                    is_term_master,
                    mux_core,
                    worker_idx,
                    core,
                    mux_kernel_config,
                    program,
                    termination_master_logical,
                    device,
                    writer_args);
            } else {
                // link index out of range or invalid direction — append a disconnected MUX connection
                // Still need valid semaphore addresses for the 5 semaphore fields
                writer_args.push_back(0);                                    // mux_connection_valid = false
                writer_args.push_back(0);                                    // is_termination_master
                writer_args.push_back(0);                                    // mux_x
                writer_args.push_back(0);                                    // mux_y
                writer_args.push_back(0);                                    // channel_base_address
                writer_args.push_back(0);                                    // connection_info_address
                writer_args.push_back(0);                                    // connection_handshake_address
                writer_args.push_back(0);                                    // flow_control_address
                writer_args.push_back(0);                                    // buffer_index_address
                writer_args.push_back(0);                                    // channel_credits_stream_id
                writer_args.push_back(CreateSemaphore(program, {core}, 0));  // termination_sync
                writer_args.push_back(CreateSemaphore(program, {core}, 0));  // local_fabric_mux_status
                writer_args.push_back(CreateSemaphore(program, {core}, 0));  // local_flow_control
                writer_args.push_back(CreateSemaphore(program, {core}, 0));  // local_teardown
                writer_args.push_back(CreateSemaphore(program, {core}, 0));  // local_buffer_index
                writer_args.push_back(0);                                    // termination_master_noc_x
                writer_args.push_back(0);                                    // termination_master_noc_y
            }

            // MUX writer RT args: out_ready_sem, injector coords, AG params, op signaler
            if (link_in_range) {
                if (!ag_rt_offset_set) {
                    writer_fabric_ag_rt_offset = writer_args.size();
                    ag_rt_offset_set = true;
                }

                const uint32_t out_ready_sem_addr = args.semaphore[0].address();
                writer_args.push_back(out_ready_sem_addr);

                // Find the injector core for this MUX writer's (batch, head)
                const auto& mux_head_work = core_work.at(i).head_work;
                CoreCoord injector_physical = {0, 0};
                if (mux_head_work.empty()) {
                    log_warning(
                        tt::LogOp,
                        "MUX writer core ({},{}) has no head_work; cannot determine injector",
                        core.x,
                        core.y);
                } else {
                    uint32_t head_key = mux_head_work[0].batch * NH + mux_head_work[0].head;
                    auto it = injector_physical_by_head.find(head_key);
                    if (it != injector_physical_by_head.end()) {
                        injector_physical = it->second;
                    } else {
                        log_warning(
                            tt::LogOp,
                            "MUX writer core ({},{}) batch={} head={}: no injector found in chain info",
                            core.x,
                            core.y,
                            mux_head_work[0].batch,
                            mux_head_work[0].head);
                    }
                }
                writer_args.push_back(static_cast<uint32_t>(injector_physical.x));
                writer_args.push_back(static_cast<uint32_t>(injector_physical.y));
                writer_args.push_back(args.num_links);   // num_muxes_in_direction
                writer_args.push_back(link);              // my_mux_index

                // Direction: top half of rows = backward (0), bottom half = forward (1).
                const uint32_t ag_direction = is_backward ? 0 : 1;

                // Tile range for this link
                const uint32_t base_tiles = ag_single_bh_num_tiles / args.num_links;
                const uint32_t remainder = ag_single_bh_num_tiles % args.num_links;
                const uint32_t ag_tile_id_start = link * base_tiles + std::min(link, remainder);
                const uint32_t ag_tile_id_end = (link + 1) * base_tiles + std::min(link + 1, remainder);

                writer_args.push_back(ag_direction);
                writer_args.push_back(ag_input_Wt);
                writer_args.push_back(ag_input_Ht);
                writer_args.push_back(ag_output_Wt);
                writer_args.push_back(ag_output_Ht);
                writer_args.push_back(static_cast<uint32_t>(args.dim));  // gather_dim
                writer_args.push_back(ag_batch_head_count);
                writer_args.push_back(ag_tile_id_start);
                writer_args.push_back(ag_tile_id_end);
                writer_args.push_back(static_cast<uint32_t>(args.ring_size));
                writer_args.push_back(k_addr);
                writer_args.push_back(v_addr);
                writer_args.push_back(gathered_k_addr);
                writer_args.push_back(gathered_v_addr);

                if (ag_direction == 1) {
                    fused_op_signaler_forward.push_all_gather_fused_op_rt_args(writer_args, args.num_links, link, 1);
                } else {
                    fused_op_signaler_backward.push_all_gather_fused_op_rt_args(writer_args, args.num_links, link, 0);
                }
            }
            SetRuntimeArgs(program, writer_fabric_kernels_id, core, writer_args);
        } else {
            SetRuntimeArgs(program, writer_kernels_id, core, writer_args);
        }

        // Compute args
        std::vector<uint32_t> compute_args = {
            global_q_start,
            global_q_end,
        };
        sdpa_fused_op_signaler->push_ring_sdpa_fused_op_rt_args(compute_args, direction);
        SetRuntimeArgs(program, compute_kernels_id, core, compute_args);
    }

    // ---- Fabric MUX cores ----
    std::vector<CoreRange> mux_core_ranges;
    for (uint32_t link = 0; link < args.num_links; ++link) {
        if (backward_coord.has_value()) {
            mux_core_ranges.emplace_back(mux_backward_logical_cores[link]);
        }
        if (forward_coord.has_value()) {
            mux_core_ranges.emplace_back(mux_forward_logical_cores[link]);
        }
    }
    CoreRangeSet mux_core_range_set(mux_core_ranges);

    tt::tt_metal::KernelHandle ccl_mux_kernel_id{};
    if (!mux_core_ranges.empty()) {
        ccl_mux_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
            mux_core_range_set,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_1_default,
                .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
                .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

        const auto src_node_id = mesh_device->get_fabric_node_id(coord);
        for (uint32_t link = 0; link < args.num_links; ++link) {
            if (backward_coord.has_value()) {
                const auto dst_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
                auto mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                    src_node_id, dst_node_id, link, program, {mux_backward_logical_cores[link]});
                tt::tt_metal::SetRuntimeArgs(
                    program, ccl_mux_kernel_id, {mux_backward_logical_cores[link]}, mux_rt_args);
            }
            if (forward_coord.has_value()) {
                const auto dst_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
                auto mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                    src_node_id, dst_node_id, link, program, {mux_forward_logical_cores[link]});
                tt::tt_metal::SetRuntimeArgs(
                    program, ccl_mux_kernel_id, {mux_forward_logical_cores[link]}, mux_rt_args);
            }
        }
    }

    return cached_program_t{
        std::move(program),
        {.num_cores = num_cores,
         .grid_size = grid_size,
         .reader_kernels_id = reader_kernels_id,
         .writer_kernels_id = writer_kernels_id,
         .writer_fabric_kernels_id = writer_fabric_kernels_id,
         .compute_kernels_id = compute_kernels_id,
         .writer_fabric_ag_rt_offset = writer_fabric_ag_rt_offset,
         .reader_fused_op_sem_rt_offset = reader_fused_op_sem_rt_offset,
         .ccl_mux_kernel_id = ccl_mux_kernel_id,
         .ccl_mux_backward_cores = mux_backward_logical_cores,
         .ccl_mux_forward_cores = mux_forward_logical_cores}};
}

void ExpRingJointSDPAProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const ExpRingJointSDPAParams& args,
    const ExpRingJointSDPAInputs& tensor_args,
    ExpRingJointSDPAResult& output_tensors) {
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        // Get addresses for regular tensors
        auto* q_buffer = tensor_args.input_q.buffer();
        auto* k_buffer = tensor_args.input_k.buffer();
        auto* v_buffer = tensor_args.input_v.buffer();
        auto* gathered_k_buffer = tensor_args.gathered_k.buffer();
        auto* gathered_v_buffer = tensor_args.gathered_v.buffer();
        auto* joint_q_buffer = tensor_args.joint_q.buffer();
        auto* joint_k_buffer = tensor_args.joint_k.buffer();
        auto* joint_v_buffer = tensor_args.joint_v.buffer();

        // Get addresses for output tensors
        auto* out_buffer = output_tensors[EXP_RING_JOINT_SDPA_OUTPUT_IDX].buffer();
        auto* joint_out_buffer = output_tensors[EXP_RING_JOINT_SDPA_JOINT_OUTPUT_IDX].buffer();
        auto* stats_buffer = output_tensors[EXP_RING_JOINT_SDPA_STATS_OUTPUT_IDX].buffer();

        uint32_t q_addr = q_buffer->address();
        uint32_t k_addr = k_buffer->address();
        uint32_t v_addr = v_buffer->address();
        uint32_t gathered_k_addr = gathered_k_buffer->address();
        uint32_t gathered_v_addr = gathered_v_buffer->address();
        uint32_t joint_q_addr = joint_q_buffer->address();
        uint32_t joint_k_addr = joint_k_buffer->address();
        uint32_t joint_v_addr = joint_v_buffer->address();
        uint32_t out_addr = out_buffer->address();
        uint32_t joint_out_addr = joint_out_buffer->address();
        uint32_t stats_addr = stats_buffer->address();

        const uint32_t out_ready_sem_addr = args.semaphore[0].address();

        auto& reader_args_by_core = GetRuntimeArgs(program, shared_vars.reader_kernels_id);
        auto& writer_args_by_core = GetRuntimeArgs(program, shared_vars.writer_kernels_id);
        auto& writer_fabric_args_by_core = GetRuntimeArgs(program, shared_vars.writer_fabric_kernels_id);

        for (uint32_t i = 0; i < shared_vars.num_cores; ++i) {
            CoreCoord core = {i % shared_vars.grid_size.x, i / shared_vars.grid_size.x};

            auto& reader_args = reader_args_by_core[core.x][core.y];

            // Update reader args
            reader_args[0] = q_addr;
            reader_args[1] = k_addr;
            reader_args[2] = v_addr;
            reader_args[3] = gathered_k_addr;
            reader_args[4] = gathered_v_addr;
            reader_args[5] = joint_q_addr;
            reader_args[6] = joint_k_addr;
            reader_args[7] = joint_v_addr;

            // Update fused-op global semaphore address (used by injector readers)
            if (shared_vars.reader_fused_op_sem_rt_offset > 0) {
                reader_args[shared_vars.reader_fused_op_sem_rt_offset] = out_ready_sem_addr;
            }

            // Update writer args — fabric clients (last 2 columns) use writer_fabric_kernels_id
            const bool is_fabric_client = (core.x >= shared_vars.grid_size.x - 2);
            if (is_fabric_client) {
                auto& writer_args = writer_fabric_args_by_core[core.x][core.y];
                writer_args[0] = out_addr;
                writer_args[1] = joint_out_addr;
                writer_args[2] = stats_addr;
                // Update addresses for MUX writers with link_in_range
                if (shared_vars.writer_fabric_ag_rt_offset > 0 &&
                    writer_args.size() > shared_vars.writer_fabric_ag_rt_offset) {
                    writer_args[shared_vars.writer_fabric_ag_rt_offset + 0] = out_ready_sem_addr;
                    writer_args[shared_vars.writer_fabric_ag_rt_offset + 15] = k_addr;
                    writer_args[shared_vars.writer_fabric_ag_rt_offset + 16] = v_addr;
                    writer_args[shared_vars.writer_fabric_ag_rt_offset + 17] = gathered_k_addr;
                    writer_args[shared_vars.writer_fabric_ag_rt_offset + 18] = gathered_v_addr;
                }
            } else {
                auto& writer_args = writer_args_by_core[core.x][core.y];
                writer_args[0] = out_addr;
                writer_args[1] = joint_out_addr;
                writer_args[2] = stats_addr;
            }
        }
    }
}

}  // namespace ttnn::prim
