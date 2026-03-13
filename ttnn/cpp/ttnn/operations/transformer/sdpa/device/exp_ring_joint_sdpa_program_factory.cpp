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
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

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

    uint32_t forward_writes_expected, backward_writes_expected;
    if (args.topology == ttnn::ccl::Topology::Linear) {
        forward_writes_expected = num_targets_backward;
        backward_writes_expected = num_targets_forward;
    } else {
        TT_FATAL(args.topology == ttnn::ccl::Topology::Ring, "Topology must be Linear or Ring");
        forward_writes_expected = num_targets_forward;
        backward_writes_expected = num_targets_backward;
    }
    // Minimally use matmul fused op signaler
    sdpa_fused_op_signaler->init_all_gather(
        args.ring_size,
        device_index,
        forward_writes_expected,
        backward_writes_expected);

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
    // K input
    auto c_in1_config = CircularBufferConfig(k_tiles * k_tile_size, {{tt::CBIndex::c_1, k_df}})
                            .set_page_size(tt::CBIndex::c_1, k_tile_size);
    CreateCircularBuffer(program, core_grid, c_in1_config);
    // V input
    auto c_in2_config = CircularBufferConfig(v_tiles * v_tile_size, {{tt::CBIndex::c_2, v_df}})
                            .set_page_size(tt::CBIndex::c_2, v_tile_size);
    CreateCircularBuffer(program, core_grid, c_in2_config);

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

    // Create kernels (deferred until after chain construction for mcast_enabled flag)
    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/exp_ring_joint_reader.cpp",
        core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, defines));

    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/exp_ring_joint_writer.cpp",
        core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, defines));

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

    // Set reader rt args
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        // Prefer the computed even distribution above for chain construction
        const auto& work = core_work.at(i);
        uint32_t global_q_start = work.global_q_start;
        uint32_t global_q_end = work.global_q_start + work.global_q_count;

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

        // Inject fused-op synchronization RT args (AllGather) here; it will append to reader_args
        sdpa_fused_op_signaler->push_ring_sdpa_fused_op_rt_args(reader_args);

        SetRuntimeArgs(program, reader_kernels_id, core, reader_args);

        // Writer args
        std::vector<uint32_t> writer_args = {
            out_addr,
            joint_out_addr,
            stats_addr,
            global_q_start,
            global_q_end,
        };
        sdpa_fused_op_signaler->push_ring_sdpa_fused_op_rt_args(writer_args);
        SetRuntimeArgs(program, writer_kernels_id, core, writer_args);

        // Compute args
        std::vector<uint32_t> compute_args = {
            global_q_start,
            global_q_end,
        };
        sdpa_fused_op_signaler->push_ring_sdpa_fused_op_rt_args(compute_args);
        SetRuntimeArgs(program, compute_kernels_id, core, compute_args);
    }

    // ---- Inline CCL (all-gather) kernel creation ----
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> all_gather_fused_op_signaler =
        ttnn::experimental::ccl::AllGatherFusedOpSignaler();

    all_gather_fused_op_signaler->init_fused_op(
        sdpa_fused_op_signaler->fused_op_receiver_cores_noc,
        sdpa_fused_op_signaler->fused_op_receiver_signal_semaphores,
        sdpa_fused_op_signaler->fused_op_signaler_mode);

    const bool fuse_op = true;

    // Build input/output tensor lists for the all-gather
    std::vector<Tensor> all_gather_input_tensors = {input_tensor_k, input_tensor_v};
    std::vector<Tensor> all_gather_output_tensors = {gathered_input_tensor_k, gathered_input_tensor_v};
    const uint32_t ccl_num_inputs = all_gather_input_tensors.size();

    const auto& op_config = ttnn::ccl::CCLOpConfig(all_gather_input_tensors, all_gather_output_tensors, args.topology);

    // Choose CCL worker cores: 4 cores/link = [mux_bwd, worker_bwd, mux_fwd, worker_fwd]
    const auto [sender_worker_core_range, sender_worker_cores] = ttnn::ccl::choose_worker_cores(
        args.num_links,
        4 /*num_cores_per_link*/,
        mesh_device,
        args.sub_device_id,
        args.ccl_core_grid_offset,
        std::nullopt,
        args.core_allocation_strategy);

    // Per link: [mux_bwd, worker_bwd, mux_fwd, worker_fwd]
    std::set<CoreRange> sender_forward_core_ranges;
    std::set<CoreRange> sender_backward_core_ranges;
    std::set<CoreRange> mux_forward_core_ranges;
    std::set<CoreRange> mux_backward_core_ranges;
    for (int l = 0; l < static_cast<int>(args.num_links); l++) {
        CoreCoord mux_backward_core = sender_worker_cores[l * 4 + 0];
        CoreCoord worker_backward_core = sender_worker_cores[l * 4 + 1];
        CoreCoord mux_forward_core = sender_worker_cores[l * 4 + 2];
        CoreCoord worker_forward_core = sender_worker_cores[l * 4 + 3];
        mux_backward_core_ranges.insert(CoreRange(mux_backward_core));
        sender_backward_core_ranges.insert(CoreRange(worker_backward_core));
        mux_forward_core_ranges.insert(CoreRange(mux_forward_core));
        sender_forward_core_ranges.insert(CoreRange(worker_forward_core));
    }

    // L1 scratch CBs
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    const uint32_t max_scatter_write_pages = 2;
    const uint32_t num_pages_per_packet =
        std::min(static_cast<uint32_t>(packet_size_bytes / l1_scratch_cb_page_size_bytes), max_scatter_write_pages);
    const uint32_t cb_num_pages = 3 * num_pages_per_packet;  // triple buffering
    const tt::DataFormat ccl_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_k.dtype());

    uint32_t sender_forward_cb_index = tt::CB::c_in0;
    tt::tt_metal::CreateCircularBuffer(
        program,
        sender_forward_core_ranges,
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{sender_forward_cb_index, ccl_df}})
            .set_page_size(sender_forward_cb_index, l1_scratch_cb_page_size_bytes));

    uint32_t sender_backward_cb_index = tt::CB::c_in2;
    tt::tt_metal::CreateCircularBuffer(
        program,
        sender_backward_core_ranges,
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{sender_backward_cb_index, ccl_df}})
            .set_page_size(sender_backward_cb_index, l1_scratch_cb_page_size_bytes));

    const auto reserved_packet_header_forward_CB_index = tt::CB::c_in1;
    static constexpr auto num_packet_headers_storable = 8;
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CreateCircularBuffer(
        program,
        sender_forward_core_ranges,
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_forward_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_forward_CB_index, packet_header_size_bytes));

    const auto reserved_packet_header_backward_CB_index = tt::CB::c_in1;
    tt::tt_metal::CreateCircularBuffer(
        program,
        sender_backward_core_ranges,
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_backward_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_backward_CB_index, packet_header_size_bytes));

    // Tensor info
    const auto ccl_input_tensor_num_pages = all_gather_input_tensors[0].buffer()->num_pages();
    const auto ccl_input_tensor_shape = all_gather_input_tensors[0].padded_shape();
    const auto ccl_output_tensor_shape = all_gather_output_tensors[0].padded_shape();
    const uint32_t tiles_to_write_per_packet = 1;

    // Fused-op signalers
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_forward =
        all_gather_fused_op_signaler.value();
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_backward =
        all_gather_fused_op_signaler.value();
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_sender_workers =
        all_gather_fused_op_signaler.value();

    // Init fused op on all sets of CCL worker cores
    {
        auto sender_workers_forward = corerange_to_cores(sender_forward_core_ranges, std::nullopt, true);
        auto sender_workers_backward = corerange_to_cores(sender_backward_core_ranges, std::nullopt, true);
        fused_op_signaler_forward->init_all_gather(
            program, mesh_device, sender_forward_core_ranges, sender_workers_forward);
        fused_op_signaler_backward->init_all_gather(
            program, mesh_device, sender_backward_core_ranges, sender_workers_backward);
        fused_op_signaler_sender_workers->init_all_gather(
            program, mesh_device, sender_forward_core_ranges, sender_workers_forward);
    }

    // ---- Fabric MUX setup ----
    const size_t max_packet_size_bytes = static_cast<size_t>(num_pages_per_packet) * l1_scratch_cb_page_size_bytes;
    const size_t mux_base_l1_address = mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    constexpr uint8_t mux_num_buffers = 8;  // FabricMuxConfig default_num_buffers
    tt::tt_fabric::FabricMuxConfig forward_mux_config(
        /*num_full_size_channels=*/1,
        /*num_header_only_channels=*/0,
        /*num_buffers_full_size_channel=*/mux_num_buffers,
        /*num_buffers_header_only_channel=*/0,
        /*buffer_size_bytes_full_size_channel=*/max_packet_size_bytes,
        /*base_l1_address=*/mux_base_l1_address,
        tt::CoreType::WORKER);
    tt::tt_fabric::FabricMuxConfig backward_mux_config(
        /*num_full_size_channels=*/1,
        /*num_header_only_channels=*/0,
        /*num_buffers_full_size_channel=*/mux_num_buffers,
        /*num_buffers_header_only_channel=*/0,
        /*buffer_size_bytes_full_size_channel=*/max_packet_size_bytes,
        /*base_l1_address=*/mux_base_l1_address,
        tt::CoreType::WORKER);

    auto ccl_mux_forward_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        mux_forward_core_ranges,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = forward_mux_config.get_fabric_mux_compile_time_args(),
        });
    auto ccl_mux_backward_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        mux_backward_core_ranges,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = backward_mux_config.get_fabric_mux_compile_time_args(),
        });

    // Set mux RT args per link
    for (uint32_t link = 0; link < args.num_links; link++) {
        CoreCoord mux_forward_logical = sender_worker_cores[link * 4 + 2];
        CoreCoord mux_backward_logical = sender_worker_cores[link * 4 + 0];

        if (forward_coord.has_value()) {
            const auto src_node_id = mesh_device->get_fabric_node_id(coord);
            const auto dst_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
            auto mux_fwd_rt_args = forward_mux_config.get_fabric_mux_run_time_args(
                src_node_id, dst_node_id, link, program, {mux_forward_logical});
            tt::tt_metal::SetRuntimeArgs(program, ccl_mux_forward_kernel_id, {mux_forward_logical}, mux_fwd_rt_args);
        }
        if (backward_coord.has_value()) {
            const auto src_node_id = mesh_device->get_fabric_node_id(coord);
            const auto dst_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
            auto mux_bwd_rt_args = backward_mux_config.get_fabric_mux_run_time_args(
                src_node_id, dst_node_id, link, program, {mux_backward_logical});
            tt::tt_metal::SetRuntimeArgs(program, ccl_mux_backward_kernel_id, {mux_backward_logical}, mux_bwd_rt_args);
        }
    }

    // Forward reader kernel
    auto sender_reader_forward_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    sender_reader_forward_kernel_config.compile_args = {
        device_index,
        sender_forward_cb_index,
        num_pages_per_packet,
        op_config.get_page_size(),
        num_targets_forward,
        num_targets_backward,
        static_cast<uint32_t>(args.topology),
        tiles_to_write_per_packet,
        ccl_num_inputs,
        1,          // direction (forward)
        fuse_op,
    };
    for (uint32_t i = 0; i < ccl_num_inputs; i++) {
        sender_reader_forward_kernel_config.compile_args.push_back(op_config.get_page_size());
    }
    for (uint32_t i = 0; i < ccl_num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(all_gather_input_tensors[i].buffer())
            .append_to(sender_reader_forward_kernel_config.compile_args);
    }
    for (uint32_t i = 0; i < ccl_num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(all_gather_output_tensors[i].buffer())
            .append_to(sender_reader_forward_kernel_config.compile_args);
    }
    auto ccl_reader_forward_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_all_gather_reader.cpp",
        sender_forward_core_ranges,
        sender_reader_forward_kernel_config);

    // Forward writer kernel
    auto sender_writer_forward_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    sender_writer_forward_kernel_config.compile_args = {
        device_index,
        reserved_packet_header_forward_CB_index,
        num_packet_headers_storable,
        sender_forward_cb_index,
        num_pages_per_packet,
        op_config.get_page_size(),
        num_targets_forward,
        num_targets_backward,
        dynamic_alternate,
        fuse_op,
        static_cast<uint32_t>(args.topology),
        tiles_to_write_per_packet,
        ccl_num_inputs,
        1,          // direction (forward)
    };
    for (uint32_t i = 0; i < ccl_num_inputs; i++) {
        sender_writer_forward_kernel_config.compile_args.push_back(op_config.get_page_size());
    }
    for (uint32_t i = 0; i < ccl_num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(all_gather_output_tensors[i].buffer())
            .append_to(sender_writer_forward_kernel_config.compile_args);
    }
    ttnn::ccl::fabric_mux_connection_ct_args(
        /*num_workers_per_direction=*/1,
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        forward_mux_config,
        sender_writer_forward_kernel_config.compile_args);
    sender_writer_forward_kernel_config.defines["USE_WORKER_MUX"] = "1";
    auto ccl_writer_forward_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_all_gather_writer.cpp",
        sender_forward_core_ranges,
        sender_writer_forward_kernel_config);

    // Backward reader kernel
    auto sender_reader_backward_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    sender_reader_backward_kernel_config.compile_args = {
        device_index,
        sender_backward_cb_index,
        num_pages_per_packet,
        op_config.get_page_size(),
        num_targets_forward,
        num_targets_backward,
        static_cast<uint32_t>(args.topology),
        tiles_to_write_per_packet,
        ccl_num_inputs,
        0,          // direction (backward)
        fuse_op,
    };
    for (uint32_t i = 0; i < ccl_num_inputs; i++) {
        sender_reader_backward_kernel_config.compile_args.push_back(op_config.get_page_size());
    }
    for (uint32_t i = 0; i < ccl_num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(all_gather_input_tensors[i].buffer())
            .append_to(sender_reader_backward_kernel_config.compile_args);
    }
    for (uint32_t i = 0; i < ccl_num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(all_gather_output_tensors[i].buffer())
            .append_to(sender_reader_backward_kernel_config.compile_args);
    }
    auto ccl_reader_backward_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_all_gather_reader.cpp",
        sender_backward_core_ranges,
        sender_reader_backward_kernel_config);

    // Backward writer kernel
    auto sender_writer_backward_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    sender_writer_backward_kernel_config.compile_args = {
        device_index,
        reserved_packet_header_backward_CB_index,
        num_packet_headers_storable,
        sender_backward_cb_index,
        num_pages_per_packet,
        op_config.get_page_size(),
        num_targets_forward,
        num_targets_backward,
        dynamic_alternate,
        fuse_op,
        static_cast<uint32_t>(args.topology),
        tiles_to_write_per_packet,
        ccl_num_inputs,
        0,          // direction (backward)
    };
    for (uint32_t i = 0; i < ccl_num_inputs; i++) {
        sender_writer_backward_kernel_config.compile_args.push_back(op_config.get_page_size());
    }
    for (uint32_t i = 0; i < ccl_num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(all_gather_output_tensors[i].buffer())
            .append_to(sender_writer_backward_kernel_config.compile_args);
    }
    ttnn::ccl::fabric_mux_connection_ct_args(
        /*num_workers_per_direction=*/1,
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        backward_mux_config,
        sender_writer_backward_kernel_config.compile_args);
    sender_writer_backward_kernel_config.defines["USE_WORKER_MUX"] = "1";
    auto ccl_writer_backward_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_all_gather_writer.cpp",
        sender_backward_core_ranges,
        sender_writer_backward_kernel_config);

    // CCL kernel runtime args
    const uint32_t batch_head_size = ccl_input_tensor_shape[0] * ccl_input_tensor_shape[1];
    const uint32_t single_batch_head_num_pages = ccl_input_tensor_num_pages / batch_head_size;

    TT_ASSERT(!(ccl_input_tensor_shape[3] % tt::constants::TILE_WIDTH));
    TT_ASSERT(!(ccl_output_tensor_shape[3] % tt::constants::TILE_WIDTH));
    const uint32_t ccl_input_tensor_Wt = ccl_input_tensor_shape[3] / tt::constants::TILE_WIDTH;
    const uint32_t ccl_input_tensor_Ht = ccl_input_tensor_shape[2] / tt::constants::TILE_WIDTH;
    const uint32_t ccl_output_tensor_Wt = ccl_output_tensor_shape[3] / tt::constants::TILE_WIDTH;
    const uint32_t ccl_output_tensor_Ht = ccl_output_tensor_shape[2] / tt::constants::TILE_WIDTH;

    uint32_t ccl_reader_sender_rt_offset = 0;
    uint32_t ccl_writer_sender_rt_offset = 0;

    for (uint32_t link = 0; link < args.num_links; link++) {
        const uint32_t base_pages_per_worker = single_batch_head_num_pages / args.num_links;
        const uint32_t remainder = single_batch_head_num_pages % args.num_links;
        const uint32_t input_tile_id_start = (link * base_pages_per_worker) + std::min(link, remainder);
        const uint32_t input_tile_id_end = ((link + 1) * base_pages_per_worker) + std::min(link + 1, remainder);

        std::vector<uint32_t> reader_forward_rt_args = {
            ccl_input_tensor_Wt,
            ccl_input_tensor_Ht,
            ccl_output_tensor_Wt,
            ccl_output_tensor_Ht,
            static_cast<uint32_t>(args.dim),
            batch_head_size,
            input_tile_id_start,
            input_tile_id_end,
            static_cast<uint32_t>(args.ring_size),
            args.semaphore.at(1).address(),
        };
        ccl_reader_sender_rt_offset = reader_forward_rt_args.size();
        for (uint32_t input_idx = 0; input_idx < ccl_num_inputs; input_idx++) {
            reader_forward_rt_args.push_back(all_gather_input_tensors[input_idx].buffer()->address());
        }
        for (uint32_t input_idx = 0; input_idx < ccl_num_inputs; input_idx++) {
            reader_forward_rt_args.push_back(all_gather_output_tensors[input_idx].buffer()->address());
        }
        // Per-link core assignments: [mux_bwd(0), worker_bwd(1), mux_fwd(2), worker_fwd(3)]
        const CoreCoord worker_forward_logical = sender_worker_cores[link * 4 + 3];
        const CoreCoord worker_backward_logical = sender_worker_cores[link * 4 + 1];
        const CoreCoord mux_forward_logical_core = sender_worker_cores[link * 4 + 2];
        const CoreCoord mux_backward_logical_core = sender_worker_cores[link * 4 + 0];

        fused_op_signaler_forward->push_all_gather_fused_op_rt_args(reader_forward_rt_args, args.num_links, link, 1);
        tt::tt_metal::SetRuntimeArgs(
            program, ccl_reader_forward_kernel_id, {worker_forward_logical}, reader_forward_rt_args);

        std::vector<uint32_t> reader_backward_rt_args = {
            ccl_input_tensor_Wt,
            ccl_input_tensor_Ht,
            ccl_output_tensor_Wt,
            ccl_output_tensor_Ht,
            static_cast<uint32_t>(args.dim),
            batch_head_size,
            input_tile_id_start,
            input_tile_id_end,
            static_cast<uint32_t>(args.ring_size),
            args.semaphore.at(0).address(),
        };
        for (uint32_t input_idx = 0; input_idx < ccl_num_inputs; input_idx++) {
            reader_backward_rt_args.push_back(all_gather_input_tensors[input_idx].buffer()->address());
        }
        for (uint32_t input_idx = 0; input_idx < ccl_num_inputs; input_idx++) {
            reader_backward_rt_args.push_back(all_gather_output_tensors[input_idx].buffer()->address());
        }
        fused_op_signaler_backward->push_all_gather_fused_op_rt_args(reader_backward_rt_args, args.num_links, link, 0);
        tt::tt_metal::SetRuntimeArgs(
            program, ccl_reader_backward_kernel_id, {worker_backward_logical}, reader_backward_rt_args);

        const CoreCoord sender_forward_worker_core = mesh_device->worker_core_from_logical_core(worker_forward_logical);
        const CoreCoord sender_backward_worker_core =
            mesh_device->worker_core_from_logical_core(worker_backward_logical);

        std::vector<uint32_t> writer_forward_rt_args = {
            ccl_input_tensor_Wt,
            ccl_input_tensor_Ht,
            ccl_output_tensor_Wt,
            ccl_output_tensor_Ht,
            static_cast<uint32_t>(args.dim),
            batch_head_size,
            input_tile_id_start,
            input_tile_id_end,
            sender_forward_worker_core.x,
            sender_forward_worker_core.y,
            static_cast<uint32_t>(args.ring_size),
            args.semaphore.at(1).address(),
        };
        ccl_writer_sender_rt_offset = writer_forward_rt_args.size();
        for (uint32_t input_idx = 0; input_idx < ccl_num_inputs; input_idx++) {
            writer_forward_rt_args.push_back(all_gather_output_tensors[input_idx].buffer()->address());
        }
        // MUX RT args: forward writer (direction=1) uses mux_backward to send to backward neighbor
        {
            const CoreCoord mux_bwd_virtual = mesh_device->worker_core_from_logical_core(mux_backward_logical_core);
            const CoreCoord worker_fwd_virtual = sender_forward_worker_core;
            ttnn::ccl::fabric_mux_connection_rt_args(
                /*mux_connection_valid=*/backward_coord.has_value(),
                /*is_termination_master=*/true,
                tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                mux_bwd_virtual,
                /*worker_id=*/0,
                worker_forward_logical,
                backward_mux_config,
                program,
                worker_fwd_virtual,
                writer_forward_rt_args,
                std::nullopt);
        }
        fused_op_signaler_sender_workers->push_all_gather_fused_op_rt_args(
            writer_forward_rt_args, args.num_links, link, 1);
        tt::tt_metal::SetRuntimeArgs(
            program, ccl_writer_forward_kernel_id, worker_forward_logical, writer_forward_rt_args);

        std::vector<uint32_t> writer_backward_rt_args = {
            ccl_input_tensor_Wt,
            ccl_input_tensor_Ht,
            ccl_output_tensor_Wt,
            ccl_output_tensor_Ht,
            static_cast<uint32_t>(args.dim),
            batch_head_size,
            input_tile_id_start,
            input_tile_id_end,
            sender_backward_worker_core.x,
            sender_backward_worker_core.y,
            static_cast<uint32_t>(args.ring_size),
            args.semaphore.at(0).address(),
        };
        for (uint32_t input_idx = 0; input_idx < ccl_num_inputs; input_idx++) {
            writer_backward_rt_args.push_back(all_gather_output_tensors[input_idx].buffer()->address());
        }
        // MUX RT args: backward writer (direction=0) uses mux_forward to send to forward neighbor
        {
            const CoreCoord mux_fwd_virtual = mesh_device->worker_core_from_logical_core(mux_forward_logical_core);
            const CoreCoord worker_bwd_virtual = sender_backward_worker_core;
            ttnn::ccl::fabric_mux_connection_rt_args(
                /*mux_connection_valid=*/forward_coord.has_value(),
                /*is_termination_master=*/true,
                tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                mux_fwd_virtual,
                /*worker_id=*/0,
                worker_backward_logical,
                forward_mux_config,
                program,
                worker_bwd_virtual,
                writer_backward_rt_args,
                std::nullopt);
        }
        fused_op_signaler_sender_workers->push_all_gather_fused_op_rt_args(writer_backward_rt_args, 1, 0, 0);
        tt::tt_metal::SetRuntimeArgs(
            program, ccl_writer_backward_kernel_id, worker_backward_logical, writer_backward_rt_args);
    }

    return cached_program_t{
        std::move(program),
        {.num_cores = num_cores,
         .grid_size = grid_size,
         .reader_kernels_id = reader_kernels_id,
         .writer_kernels_id = writer_kernels_id,
         .compute_kernels_id = compute_kernels_id,
         .ccl_reader_forward_kernel_id = ccl_reader_forward_kernel_id,
         .ccl_writer_forward_kernel_id = ccl_writer_forward_kernel_id,
         .ccl_reader_backward_kernel_id = ccl_reader_backward_kernel_id,
         .ccl_writer_backward_kernel_id = ccl_writer_backward_kernel_id,
         .ccl_mux_forward_kernel_id = ccl_mux_forward_kernel_id,
         .ccl_mux_backward_kernel_id = ccl_mux_backward_kernel_id,
         .ccl_worker_cores = sender_worker_cores,
         .ccl_num_inputs = ccl_num_inputs,
         .ccl_reader_sender_rt_offset = ccl_reader_sender_rt_offset,
         .ccl_writer_sender_rt_offset = ccl_writer_sender_rt_offset,
         .ccl_num_links = args.num_links}};
}

void ExpRingJointSDPAProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const ExpRingJointSDPAParams& args,
    const ExpRingJointSDPAInputs& tensor_args,
    ExpRingJointSDPAResult& output_tensors) {
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        // Inline CCL (all-gather) runtime argument overrides
        const std::vector<Tensor> ccl_input_tensors = {tensor_args.input_k, tensor_args.input_v};
        const std::vector<Tensor> ccl_output_tensors = {tensor_args.gathered_k, tensor_args.gathered_v};
        const auto& semaphore = args.semaphore;
        const auto& ccl_sender_worker_cores = shared_vars.ccl_worker_cores;
        const auto& ccl_num_inputs = shared_vars.ccl_num_inputs;
        const auto& reader_sender_rt_offset = shared_vars.ccl_reader_sender_rt_offset;
        const auto& writer_sender_rt_offset = shared_vars.ccl_writer_sender_rt_offset;
        const auto& ccl_num_links = shared_vars.ccl_num_links;

        auto& worker_reader_sender_forward_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.ccl_reader_forward_kernel_id);
        auto& worker_writer_sender_forward_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.ccl_writer_forward_kernel_id);
        auto& worker_reader_sender_backward_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.ccl_reader_backward_kernel_id);
        auto& worker_writer_sender_backward_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.ccl_writer_backward_kernel_id);

        for (int link = 0; link < static_cast<int>(ccl_num_links); link++) {
            // Per-link layout: [mux_bwd(0), worker_bwd(1), mux_fwd(2), worker_fwd(3)]
            const auto& worker_fwd_core = ccl_sender_worker_cores[link * 4 + 3];
            const auto& worker_bwd_core = ccl_sender_worker_cores[link * 4 + 1];
            auto& worker_reader_sender_forward_runtime_args =
                worker_reader_sender_forward_runtime_args_by_core[worker_fwd_core.x][worker_fwd_core.y];
            auto& worker_reader_sender_backward_runtime_args =
                worker_reader_sender_backward_runtime_args_by_core[worker_bwd_core.x][worker_bwd_core.y];
            auto& worker_writer_sender_forward_runtime_args =
                worker_writer_sender_forward_runtime_args_by_core[worker_fwd_core.x][worker_fwd_core.y];
            auto& worker_writer_sender_backward_runtime_args =
                worker_writer_sender_backward_runtime_args_by_core[worker_bwd_core.x][worker_bwd_core.y];

            worker_reader_sender_forward_runtime_args[9] = semaphore.at(1).address();
            worker_reader_sender_backward_runtime_args[9] = semaphore.at(0).address();
            worker_writer_sender_forward_runtime_args[11] = semaphore.at(1).address();
            worker_writer_sender_backward_runtime_args[11] = semaphore.at(0).address();

            for (uint32_t input_idx = 0; input_idx < ccl_num_inputs; input_idx++) {
                worker_reader_sender_forward_runtime_args[reader_sender_rt_offset + input_idx] =
                    ccl_input_tensors[input_idx].buffer()->address();
                worker_reader_sender_forward_runtime_args[reader_sender_rt_offset + ccl_num_inputs + input_idx] =
                    ccl_output_tensors[input_idx].buffer()->address();
                worker_reader_sender_backward_runtime_args[reader_sender_rt_offset + input_idx] =
                    ccl_input_tensors[input_idx].buffer()->address();
                worker_reader_sender_backward_runtime_args[reader_sender_rt_offset + ccl_num_inputs + input_idx] =
                    ccl_output_tensors[input_idx].buffer()->address();
                worker_writer_sender_forward_runtime_args[writer_sender_rt_offset + input_idx] =
                    ccl_output_tensors[input_idx].buffer()->address();
                worker_writer_sender_backward_runtime_args[writer_sender_rt_offset + input_idx] =
                    ccl_output_tensors[input_idx].buffer()->address();
            }
        }

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

        auto& reader_args_by_core = GetRuntimeArgs(program, shared_vars.reader_kernels_id);
        auto& writer_args_by_core = GetRuntimeArgs(program, shared_vars.writer_kernels_id);

        for (uint32_t i = 0; i < shared_vars.num_cores; ++i) {
            CoreCoord core = {i % shared_vars.grid_size.x, i / shared_vars.grid_size.x};

            auto& reader_args = reader_args_by_core[core.x][core.y];
            auto& writer_args = writer_args_by_core[core.x][core.y];

            // Update reader args
            reader_args[0] = q_addr;
            reader_args[1] = k_addr;
            reader_args[2] = v_addr;
            reader_args[3] = gathered_k_addr;
            reader_args[4] = gathered_v_addr;
            reader_args[5] = joint_q_addr;
            reader_args[6] = joint_k_addr;
            reader_args[7] = joint_v_addr;

            // Update writer args
            writer_args[0] = out_addr;
            writer_args[1] = joint_out_addr;
            writer_args[2] = stats_addr;
        }
    }
}

}  // namespace ttnn::prim
