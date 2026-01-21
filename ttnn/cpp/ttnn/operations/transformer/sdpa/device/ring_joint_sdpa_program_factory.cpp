// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.hpp"

#include <optional>
#include <cmath>
#include <string>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

RingJointSDPAProgramFactory::cached_mesh_workload_t RingJointSDPAProgramFactory::create_mesh_workload(
    const RingJointSDPAParams& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const RingJointSDPAInputs& tensor_args,
    RingJointSDPAResult& output_tensors) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_vars;

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(args, coord, tensor_args, output_tensors);
        mesh_workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_vars.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_vars)};
}

RingJointSDPAProgramFactory::cached_program_t RingJointSDPAProgramFactory::create_at(
    const RingJointSDPAParams& args,
    const ttnn::MeshCoordinate& coord,
    const RingJointSDPAInputs& tensor_args,
    RingJointSDPAResult& output_tensors) {
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

    auto& output_tensor = output_tensors.output;
    auto& joint_output_tensor = output_tensors.joint_output;
    auto& lse_output_tensor = output_tensors.lse_output;

    std::size_t q_chunk_size = args.get_q_chunk_size();
    std::size_t k_chunk_size = args.get_k_chunk_size();

    tt::tt_metal::Program program{};

    auto* mesh_device = input_tensor_q.device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(coord) : nullptr;

    std::vector<IDevice*> devices_to_use = {};
    // User specified the cluster-axis. Derive devices based on the current coordinate
    // and the cluster-axis.
    const auto& mesh_view = input_tensor_q.device()->get_view();
    devices_to_use = (args.all_gather_operation_attributes.cluster_axis.value() == 0)
                         ? mesh_view.get_devices_on_column(coord[1])
                         : mesh_view.get_devices_on_row(coord[0]);

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < args.all_gather_operation_attributes.ring_size; ++i) {
        if (devices_to_use.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
            } else if (args.all_gather_operation_attributes.topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices_to_use.at(args.all_gather_operation_attributes.ring_size - 1);
            }
            if (i != args.all_gather_operation_attributes.ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
            } else if (args.all_gather_operation_attributes.topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices_to_use.at(0);
            }
        }
    }

    auto scale = args.scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.logical_shape()[-1]));
    }

    std::optional<ttnn::prim::RingSDPAFusedOpSignaler> sdpa_fused_op_signaler = ttnn::prim::RingSDPAFusedOpSignaler();

    auto [num_targets_forward, num_targets_backward, dynamic_alternate] = ccl::get_forward_backward_configuration(
        args.all_gather_operation_attributes.ring_size, device_index, args.all_gather_operation_attributes.topology);

    // This is how ring_joint_sdpa expects the number of forward and backward writes
    uint32_t forward_writes_expected, backward_writes_expected;
    if (args.all_gather_operation_attributes.topology == ttnn::ccl::Topology::Linear) {
        forward_writes_expected = num_targets_backward;
        backward_writes_expected = num_targets_forward;
    } else {
        TT_FATAL(
            args.all_gather_operation_attributes.topology == ttnn::ccl::Topology::Ring,
            "Topology must be Linear or Ring");
        forward_writes_expected = num_targets_forward - 1;
        backward_writes_expected = num_targets_backward - 1;
    }
    // Minimally use matmul fused op signaler
    sdpa_fused_op_signaler->init_all_gather(
        args.all_gather_operation_attributes.ring_size,
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
    const uint32_t q_per_core = tt::div_up(all_heads_num_q_chunks, num_cores);

    const uint32_t q_buffer_factor = (q_per_core > 1) ? 2 : 1;

    log_debug(tt::LogOp, "q_per_core: {}", q_per_core);

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t q_tiles = Sq_chunk_t * DHt * q_buffer_factor;
    uint32_t k_tiles = Sk_chunk_t * DHt * 2;  // double buffer
    uint32_t v_tiles = Sk_chunk_t * DHt * 2;  // double buffer
    uint32_t mask_tiles = Sq_chunk_t * Sk_chunk_t;
    uint32_t qk_tiles = Sq_chunk_t * Sk_chunk_t;
    uint32_t out_im_tiles = Sq_chunk_t * DHt;
    uint32_t out0_t = Sq_chunk_t * DHt;
    uint32_t scale_tiles = 1;
    uint32_t statistics_tiles = Sq_chunk_t;  // Single column of values in each iteration

    // log all values
    log_debug(tt::LogOp, "q_tiles: {}", q_tiles);
    log_debug(tt::LogOp, "k_tiles: {}", k_tiles);
    log_debug(tt::LogOp, "v_tiles: {}", v_tiles);
    log_debug(tt::LogOp, "mask_tiles: {}", mask_tiles);
    log_debug(tt::LogOp, "qk_tiles: {}", qk_tiles);
    log_debug(tt::LogOp, "out0_t: {}", out0_t);
    log_debug(tt::LogOp, "scale_tiles: {}", scale_tiles);
    log_debug(tt::LogOp, "statistics_tiles: {}", statistics_tiles);

    // Host code is responsible for determining matmul configuration
    const uint32_t dst_size = ttnn::get_dest_reg_count(args.compute_kernel_config);
    const uint32_t qk_in0_block_w = DHt;
    // max of Sk_chunk_t and dst_size
    const uint32_t qk_out_subblock_w = std::min(Sk_chunk_t, dst_size);
    // If qk_out_subblock_w is full row of output, scale subblock_h so volume = dst_size. Otherwise it's 1 to maintain
    // row-major intermediate buffer.
    const uint32_t qk_out_subblock_h =
        (qk_out_subblock_w == Sk_chunk_t) ? (std::min(Sq_chunk_t, dst_size / qk_out_subblock_w)) : 1;

    const uint32_t qk_in0_num_subblocks = Sq_chunk_t / qk_out_subblock_h;
    const uint32_t qk_in1_num_subblocks = Sk_chunk_t / qk_out_subblock_w;
    const uint32_t qk_num_blocks = DHt / qk_in0_block_w;

    // now for out0
    const uint32_t out_in0_block_w = Sk_chunk_t;
    const uint32_t out_out_subblock_w = std::min(DHt, dst_size);
    const uint32_t out_out_subblock_h =
        (out_out_subblock_w == DHt) ? (std::min(Sq_chunk_t, dst_size / out_out_subblock_w)) : 1;

    const uint32_t out_in0_num_subblocks = Sq_chunk_t / out_out_subblock_h;
    const uint32_t out_in1_num_subblocks = DHt / out_out_subblock_w;
    const uint32_t out_num_blocks = Sk_chunk_t / out_in0_block_w;

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
    const uint32_t stats_granularity = std::min(Sq_chunk_t, dst_size);
    // Find log2 of stats_granularity using std
    const uint32_t log2_stats_granularity = std::log2(stats_granularity);
    // Assert that this is a power of 2
    TT_FATAL(
        stats_granularity == (1 << log2_stats_granularity),
        "stats_granularity must be a power of 2. Got {}.",
        stats_granularity);

    const uint32_t sub_exp_granularity = std::min(Sk_chunk_t, dst_size);
    const uint32_t log2_sub_exp_granularity = std::log2(sub_exp_granularity);
    TT_FATAL(
        sub_exp_granularity == (1 << log2_sub_exp_granularity),
        "sub_exp_granularity must be a power of 2. Got {}.",
        sub_exp_granularity);

    const uint32_t mul_bcast_granularity = std::min(Sq_chunk_t * Sk_chunk_t, dst_size);
    const uint32_t log2_mul_bcast_granularity = std::log2(mul_bcast_granularity);
    TT_FATAL(
        mul_bcast_granularity == (1 << log2_mul_bcast_granularity),
        "mul_bcast_granularity must be a power of 2. Got {}.",
        mul_bcast_granularity);

    uint32_t dht_granularity = std::min(DHt, dst_size);
    uint32_t log2_dht_granularity = std::log2(dht_granularity);
    // Sometimes DHt is not a power of 2, so granularity should be 1
    if (dht_granularity != (1 << log2_dht_granularity)) {
        dht_granularity = 1;
        log2_dht_granularity = 0;
    }
    TT_FATAL(
        dht_granularity == (1 << log2_dht_granularity),
        "dht_granularity must be a power of 2. Got {}.",
        dht_granularity);

    // Reduce ops can use granularity of dst_size/2
    const uint32_t reduce_granularity = std::min(Sq_chunk_t, dst_size / 2);
    const uint32_t log2_reduce_granularity = std::log2(reduce_granularity);
    TT_FATAL(
        reduce_granularity == (1 << log2_reduce_granularity),
        "reduce_granularity must be a power of 2. Got {}.",
        reduce_granularity);

    // Log these
    log_debug(tt::LogOp, "stats_granularity: {}", stats_granularity);
    log_debug(tt::LogOp, "log2_stats_granularity: {}", log2_stats_granularity);
    log_debug(tt::LogOp, "sub_exp_granularity: {}", sub_exp_granularity);
    log_debug(tt::LogOp, "log2_sub_exp_granularity: {}", log2_sub_exp_granularity);
    log_debug(tt::LogOp, "mul_bcast_granularity: {}", mul_bcast_granularity);
    log_debug(tt::LogOp, "log2_mul_bcast_granularity: {}", log2_mul_bcast_granularity);
    log_debug(tt::LogOp, "dht_granularity: {}", dht_granularity);
    log_debug(tt::LogOp, "log2_dht_granularity: {}", log2_dht_granularity);
    log_debug(tt::LogOp, "reduce_granularity: {}", reduce_granularity);
    log_debug(tt::LogOp, "log2_reduce_granularity: {}", log2_reduce_granularity);

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
        args.all_gather_operation_attributes.ring_size};

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
    reader_compile_time_args.push_back(sender_semaphore_id);
    reader_compile_time_args.push_back(receiver_semaphore_id);
    reader_compile_time_args.push_back(valid_semaphore_id);

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
        args.all_gather_operation_attributes.ring_size};

    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(joint_output_tensor.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(lse_output_tensor.buffer()).append_to(writer_compile_time_args);

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
        args.all_gather_operation_attributes.ring_size,
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
        scale_union.u};

    std::map<std::string, std::string> defines;
    defines["STATS_GRANULARITY"] = std::to_string(stats_granularity);
    defines["LOG2_STATS_GRANULARITY"] = std::to_string(log2_stats_granularity);
    defines["SUB_EXP_GRANULARITY"] = std::to_string(sub_exp_granularity);
    defines["LOG2_SUB_EXP_GRANULARITY"] = std::to_string(log2_sub_exp_granularity);
    defines["MUL_BCAST_GRANULARITY"] = std::to_string(mul_bcast_granularity);
    defines["LOG2_MUL_BCAST_GRANULARITY"] = std::to_string(log2_mul_bcast_granularity);
    defines["DHT_GRANULARITY"] = std::to_string(dht_granularity);
    defines["LOG2_DHT_GRANULARITY"] = std::to_string(log2_dht_granularity);
    defines["REDUCE_GRANULARITY"] = std::to_string(reduce_granularity);
    defines["LOG2_REDUCE_GRANULARITY"] = std::to_string(log2_reduce_granularity);
    defines["EXP_APPROX_MODE"] = std::to_string(exp_approx_mode);

    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_reader.cpp",
        core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, defines));

    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer.cpp",
        core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, defines));

    auto compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/ring_joint_sdpa.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
            .defines = defines});

    // Create circular buffers

    tt::DataFormat q_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_q.dtype());
    tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(gathered_input_tensor_k.dtype());
    tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(gathered_input_tensor_v.dtype());
    tt::DataFormat mask_df = tt::DataFormat::Bfp4_b;
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

    // attn_mask input
    auto c_in3_config = CircularBufferConfig(mask_tiles * mask_tile_size, {{tt::CB::c_in3, mask_df}})
                            .set_page_size(tt::CB::c_in3, mask_tile_size);
    CreateCircularBuffer(program, core_grid, c_in3_config);

    // scale input
    auto c_in4_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{tt::CBIndex::c_4, scalar_df}})
                            .set_page_size(tt::CBIndex::c_4, scalar_tile_size);
    CreateCircularBuffer(program, core_grid, c_in4_config);

    // identity scale input
    auto c_in5_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{tt::CBIndex::c_5, scalar_df}})
                            .set_page_size(tt::CBIndex::c_5, scalar_tile_size);
    CreateCircularBuffer(program, core_grid, c_in5_config);

    // lse input
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

    // lse output
    auto c_out1_config = CircularBufferConfig(statistics_tiles * im_tile_size, {{tt::CBIndex::c_17, im_df}})
                             .set_page_size(tt::CBIndex::c_17, im_tile_size);
    CreateCircularBuffer(program, core_grid, c_out1_config);

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
    uint32_t lse_addr = lse_output_tensor.buffer()->address();

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
    };

    std::vector<CoreWork> core_work(num_cores);
    std::vector<CoreChainInfo> core_chain_info(num_cores);
    const uint32_t total_heads = B * NH;
    std::vector<std::vector<HeadSegmentRef>> head_segments(total_heads);

    // Evenly distribute flat global q chunks across cores
    const uint32_t total_q_chunks = B * NH * num_q_chunks;
    const uint32_t base_chunks_per_core = (num_cores == 0) ? 0 : (total_q_chunks / num_cores);
    const uint32_t extra_chunks = (num_cores == 0) ? 0 : (total_q_chunks % num_cores);
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

    // Construct chains: for each head that spans >= 2 cores, pick first core with single head segment as injector
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

        // Inject fused-op synchronization RT args (AllGather) here; it will append to reader_args
        sdpa_fused_op_signaler->push_ring_sdpa_fused_op_rt_args(reader_args);

        SetRuntimeArgs(program, reader_kernels_id, core, reader_args);

        // Writer args
        std::vector<uint32_t> writer_args = {
            out_addr,
            joint_out_addr,
            lse_addr,
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

    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> all_gather_fused_op_signaler =
        ttnn::experimental::ccl::AllGatherFusedOpSignaler();

    all_gather_fused_op_signaler->init_fused_op(
        sdpa_fused_op_signaler->fused_op_receiver_cores_noc,
        sdpa_fused_op_signaler->fused_op_receiver_signal_semaphores,
        sdpa_fused_op_signaler->fused_op_signaler_mode);

    std::vector<Tensor> all_gather_input_tensors = {
        input_tensor_k,
        input_tensor_v,
    };
    std::vector<Tensor> all_gather_output_tensors = {
        gathered_input_tensor_k,
        gathered_input_tensor_v,
    };
    auto all_gather_shared_variables = ring_attention_all_gather_async_multi_core_with_workers_helper(
        program,  // Must pass ring_joint_sdpa's program
        all_gather_input_tensors,
        target_device,
        forward_device,
        backward_device,
        all_gather_output_tensors,
        args.all_gather_operation_attributes.dim,
        args.all_gather_operation_attributes.num_links,
        args.all_gather_operation_attributes.ring_size,
        device_index,
        args.all_gather_operation_attributes.topology,
        args.all_gather_operation_attributes.semaphore,
        args.all_gather_operation_attributes.sub_device_id,
        all_gather_fused_op_signaler,
        args.ccl_core_grid_offset);

    return cached_program_t{
        std::move(program),
        {num_cores, grid_size, reader_kernels_id, writer_kernels_id, compute_kernels_id, all_gather_shared_variables}};
}

void RingJointSDPAProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const RingJointSDPAParams& args,
    const RingJointSDPAInputs& tensor_args,
    RingJointSDPAResult& output_tensors) {
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        ring_attention_all_gather_async_multicore_with_workers_override_runtime_arguments(
            shared_vars.all_gather_shared_variables,
            program,
            {tensor_args.input_k, tensor_args.input_v},       /*input_tensors*/
            {tensor_args.gathered_k, tensor_args.gathered_v}, /*output_tensors*/
            args.all_gather_operation_attributes.semaphore);

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
        auto* out_buffer = output_tensors.output.buffer();
        auto* joint_out_buffer = output_tensors.joint_output.buffer();
        auto* lse_buffer = output_tensors.lse_output.buffer();

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
        uint32_t lse_addr = lse_buffer->address();

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
            writer_args[2] = lse_addr;
        }
    }
}

}  // namespace ttnn::prim
