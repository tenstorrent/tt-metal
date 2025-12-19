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

namespace ttnn::operations::transformer::sdpa::ring_joint_sdpa::program {

RingJointSDPAProgramFactory::cached_mesh_workload_t RingJointSDPAProgramFactory::create_mesh_workload(
    const operation_attributes_t& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensors) {
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
    const operation_attributes_t& args,
    const ttnn::MeshCoordinate& coord,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensors) {
    /*
    Q: B x NH x N/num_devices x DH
    K: B x NH x N x DH
    V: B x NH x N x DH

    Q_joint: B x NH x L x DH
    K_joint: B x NH x L x DH
    V_joint: B x NH x L x DH

    logical_n is the unpadded length of the gathered tensor. depending on device id, Q logical length
    may be less than padded length. K, V are gathered, so logical_n tells the true length of K and V.
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
    devices_to_use = (args.all_gather_struct.cluster_axis.value() == 0) ? mesh_view.get_devices_on_column(coord[1])
                                                                        : mesh_view.get_devices_on_row(coord[0]);

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < args.all_gather_struct.ring_size; ++i) {
        if (devices_to_use.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
            } else if (args.all_gather_struct.topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices_to_use.at(args.all_gather_struct.ring_size - 1);
            }
            if (i != args.all_gather_struct.ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
            } else if (args.all_gather_struct.topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices_to_use.at(0);
            }
        }
    }

    auto scale = args.scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.logical_shape()[-1]));
    }

    std::optional<detail::RingSDPAFusedOpSignaler> sdpa_fused_op_signaler = detail::RingSDPAFusedOpSignaler();

    auto [num_targets_forward, num_targets_backward, dynamic_alternate] = ccl::get_forward_backward_configuration(
        args.all_gather_struct.ring_size, device_index, args.all_gather_struct.topology);

    // This is how ring_joint_sdpa expects the number of forward and backward writes
    uint32_t forward_writes_expected, backward_writes_expected;
    if (args.all_gather_struct.topology == ttnn::ccl::Topology::Linear) {
        forward_writes_expected = num_targets_backward;
        backward_writes_expected = num_targets_forward;
    } else {
        TT_FATAL(args.all_gather_struct.topology == ttnn::ccl::Topology::Ring, "Topology must be Linear or Ring");
        forward_writes_expected = num_targets_forward - 1;
        backward_writes_expected = num_targets_backward - 1;
    }
    // Minimally use matmul fused op signaler
    sdpa_fused_op_signaler->init_all_gather(
        args.all_gather_struct.ring_size, device_index, forward_writes_expected, backward_writes_expected);

    const auto& q_shape = input_tensor_q.logical_shape();
    const auto& k_shape = gathered_input_tensor_k.logical_shape();
    const auto& joint_q_shape = joint_tensor_q.logical_shape();
    const uint32_t B = q_shape[0], NH = q_shape[1], local_N = q_shape[2], DH = q_shape[3];
    const uint32_t global_N = k_shape[2];
    const uint32_t L = joint_q_shape[2];

    // Calculate padded sequence length. Both N_local and L may need to be padded to chunk size.
    const uint32_t padded_Lq = tt::round_up(L, q_chunk_size);
    const uint32_t padded_Lk = tt::round_up(L, k_chunk_size);

    const uint32_t local_Nt = local_N / tt::constants::TILE_HEIGHT;
    const uint32_t global_Nt = global_N / tt::constants::TILE_HEIGHT;
    // Find unpadded sequence lengths in tiles
    const uint32_t logical_Lt = tt::div_up(L, tt::constants::TILE_HEIGHT);
    const uint32_t padded_Lqt = padded_Lq / tt::constants::TILE_HEIGHT;
    const uint32_t padded_Lkt = padded_Lk / tt::constants::TILE_HEIGHT;

    // Compute kernel operates on concatenated Q and K
    const uint32_t cat_Sq = local_N + padded_Lq;
    const uint32_t cat_Sk = global_N + padded_Lk;

    [[maybe_unused]] const uint32_t cat_Sqt = cat_Sq / tt::constants::TILE_HEIGHT;
    const uint32_t cat_Skt = cat_Sk / tt::constants::TILE_HEIGHT;
    const uint32_t DHt = DH / tt::constants::TILE_WIDTH;

    // Kernel will need to know the tile-based shapes of both sets of tensors
    // to create a representation of the concatenated tensors.

    // const std::vector<uint32_t> q_tile_shape = {B, NH, padded_Nqt, DHt};
    // const std::vector<uint32_t> k_tile_shape = {B, NH, padded_Nkt, DHt};
    // const std::vector<uint32_t> joint_q_tile_shape = {B, NH, padded_Lqt, DHt};
    // const std::vector<uint32_t> joint_k_tile_shape = {B, NH, padded_Lkt, DHt};

    /*
    For non-causal case we must provide a padded mask if the K sequence length has been padded
    Note that we dont have this issue in non-causal case if Q is padded, since those pad tokens
    don't affect attention of unpadded tokens.
    In causal case, the causal mask takes care of masking K pad tokens.
    */
    const bool use_joint_mask = (args.logical_n != global_N) || (padded_Lk != L);

    const uint32_t Sq_chunk_t = q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t Sk_chunk_t = k_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t q_num_chunks = cat_Sq / q_chunk_size;
    [[maybe_unused]] const uint32_t k_num_chunks = cat_Sk / k_chunk_size;
    const uint32_t N_k_num_chunks_local = local_N / k_chunk_size;
    const uint32_t L_k_num_chunks = padded_Lk / k_chunk_size;
    const uint32_t global_logical_NK_chunks = tt::div_up(args.logical_n, k_chunk_size);
    const uint32_t global_padded_NK_chunks = global_N / k_chunk_size;

    log_debug(tt::LogOp, "B: {}", B);
    log_debug(tt::LogOp, "NH: {}", NH);
    log_debug(tt::LogOp, "N: {}", local_N);
    log_debug(tt::LogOp, "L: {}", L);
    log_debug(tt::LogOp, "DH: {}", DH);

    // Log padded dimensions
    log_debug(tt::LogOp, "padded_Lq: {}", padded_Lq);
    log_debug(tt::LogOp, "padded_Lk: {}", padded_Lk);
    log_debug(tt::LogOp, "padded_Lqt: {}", padded_Lqt);
    log_debug(tt::LogOp, "padded_Lkt: {}", padded_Lkt);

    // Log tile dimensions
    log_debug(tt::LogOp, "DHt: {}", DHt);
    log_debug(tt::LogOp, "local_Nt: {}", local_Nt);
    log_debug(tt::LogOp, "global_Nt: {}", global_Nt);
    log_debug(tt::LogOp, "logical_Lt: {}", logical_Lt);
    log_debug(tt::LogOp, "logical_n: {}", args.logical_n);
    log_debug(tt::LogOp, "global_N: {}", global_N);

    // Log chunking parameters
    log_debug(tt::LogOp, "Sq_chunk_t: {}", Sq_chunk_t);
    log_debug(tt::LogOp, "Sk_chunk_t: {}", Sk_chunk_t);
    log_debug(tt::LogOp, "q_chunk_size: {}", q_chunk_size);
    log_debug(tt::LogOp, "k_chunk_size: {}", k_chunk_size);
    log_debug(tt::LogOp, "q_num_chunks: {}", q_num_chunks);
    log_debug(tt::LogOp, "k_num_chunks: {}", k_num_chunks);

    // Log concatenated dimensions
    log_debug(tt::LogOp, "cat_Sq: {}", cat_Sq);
    log_debug(tt::LogOp, "cat_Sk: {}", cat_Sk);
    log_debug(tt::LogOp, "cat_Sqt: {}", cat_Sqt);
    log_debug(tt::LogOp, "cat_Skt: {}", cat_Skt);

    log_debug(tt::LogOp, "use_joint_mask: {}", use_joint_mask);

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
     * the total number of Q chunks processed, evenly across the cores.
     *
     */
    const uint32_t global_q_chunks = B * NH * q_num_chunks;
    const uint32_t q_per_core = tt::div_up(global_q_chunks, num_cores);

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
    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
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
        local_Nt,
        global_Nt,
        logical_Lt,
        padded_Lqt,
        padded_Lkt,
        num_cores,
        args.ring_size,
        N_k_num_chunks_local,
        L_k_num_chunks,
        global_logical_NK_chunks,
        global_padded_NK_chunks,
        q_num_chunks};

    TensorAccessorArgs(input_tensor_q.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(gathered_input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(gathered_input_tensor_v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(joint_tensor_q.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(joint_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(joint_tensor_v.buffer()).append_to(reader_compile_time_args);

    // Calculate which K chunks contain the mask boundaries
    // If a tensor does not require masking, set to MAX_UINT32. This avoids a
    // bug in the mask generation code, which would mask a full, valid chunk
    // with -inf.
    const uint32_t mask_chunk_0 = (args.logical_n != global_N) ? (args.logical_n / k_chunk_size)
                                                               : (uint32_t)(-1);  // idx of last chunk in first sequence
    const uint32_t mask_chunk_1 =
        (padded_Lk != L) ? (cat_Skt / Sk_chunk_t) - 1 : (uint32_t)(-1);  // idx of last chunk in second sequence

    log_debug(tt::LogOp, "mask_chunk_0: {}", mask_chunk_0);
    log_debug(tt::LogOp, "mask_chunk_1: {}", mask_chunk_1);

    std::vector<uint32_t> writer_compile_time_args = {
        B,
        NH,
        DHt,
        Sq_chunk_t,
        Sk_chunk_t,
        local_Nt,
        global_Nt,
        logical_Lt,
        padded_Lqt,
        padded_Lkt,
        args.logical_n,
        L,
        num_cores,
        packed_identity_scalar,
        scale_union.u,
        (uint32_t)use_joint_mask,
        mask_chunk_0,
        mask_chunk_1,
        args.ring_size,
        N_k_num_chunks_local,
        L_k_num_chunks,
        global_logical_NK_chunks,
        global_padded_NK_chunks,
        q_num_chunks};

    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(joint_output_tensor.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(lse_output_tensor.buffer()).append_to(writer_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {
        B,
        NH,
        cat_Skt,
        DHt,
        Sq_chunk_t,
        Sk_chunk_t,
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
        (uint32_t)use_joint_mask,
        mask_chunk_0,
        mask_chunk_1,
        args.ring_size,
        N_k_num_chunks_local,
        L_k_num_chunks,
        global_logical_NK_chunks,
        global_padded_NK_chunks,
        q_num_chunks,
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

    // Only create mask buffer if it's going to be used
    if (use_joint_mask) {
        // attn_mask input
        auto c_in3_config = CircularBufferConfig(mask_tiles * mask_tile_size, {{tt::CB::c_in3, mask_df}})
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

    // Set reader rt args
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        uint32_t global_q_start = q_per_core * i;
        uint32_t global_q_end = global_q_start + q_per_core;

        // clamp all to max values for non-even partitioning
        global_q_start = std::min(global_q_start, global_q_chunks);
        global_q_end = std::min(global_q_end, global_q_chunks);

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
    auto all_gather_program_with_callbacks = ring_attention_all_gather_async_multi_core_with_workers_helper(
        program,  // Must pass ring_joint_sdpa's program
        all_gather_input_tensors,
        target_device,
        forward_device,
        backward_device,
        all_gather_output_tensors,
        args.all_gather_struct.dim,
        args.all_gather_struct.num_links,
        args.all_gather_struct.ring_size,
        device_index,
        args.all_gather_struct.topology,
        args.all_gather_struct.semaphore,
        args.all_gather_struct.sub_device_id,
        all_gather_fused_op_signaler,
        args.ccl_core_grid_offset);

    return cached_program_t{
        std::move(all_gather_program_with_callbacks.program),
        {num_cores,
         grid_size,
         reader_kernels_id,
         writer_kernels_id,
         compute_kernels_id,
         all_gather_program_with_callbacks.override_runtime_arguments_callback}};
}

void RingJointSDPAProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensors) {
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        if (shared_vars.all_gather_callback.has_value()) {
            shared_vars.all_gather_callback.value()(
                &args.all_gather_struct,
                program,
                {tensor_args.input_k, tensor_args.input_v},      /*input_tensors*/
                {},                                              /*optional_input_tensors*/
                {tensor_args.gathered_k, tensor_args.gathered_v} /*output_tensors*/
            );
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

}  // namespace ttnn::operations::transformer::sdpa::ring_joint_sdpa::program
