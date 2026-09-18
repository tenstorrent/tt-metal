// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/joint_sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"

#include <bit>
#include <cstdlib>
#include <climits>
#include <cmath>
#include <map>
#include <string>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

// implementation of softmax with optional scale/mask (see the header for input_tensor more detailed description)
ProgramDescriptor JointSDPADeviceOperation::JointSDPAProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& args = operation_attributes;
    auto& output_tensors = tensor_return_value;
    /*
    Q: B x NH x N x DH
    K: B x NH x N x DH
    V: B x NH x N x DH

    Q_joint: B x NH x L x DH
    K_joint: B x NH x L x DH
    V_joint: B x NH x L x DH
    */

    const Tensor& input_tensor_q = tensor_args.input_q;
    const Tensor& input_tensor_k = tensor_args.input_k;
    const Tensor& input_tensor_v = tensor_args.input_v;
    const Tensor& joint_tensor_q = tensor_args.joint_q;
    const Tensor& joint_tensor_k = tensor_args.joint_k;
    const Tensor& joint_tensor_v = tensor_args.joint_v;
    const Tensor& output_tensor = output_tensors[JOINT_SDPA_OUTPUT_IDX];
    const Tensor& joint_output_tensor = output_tensors[JOINT_SDPA_JOINT_OUTPUT_IDX];

    std::size_t q_chunk_size = args.get_q_chunk_size();
    std::size_t k_chunk_size = args.get_k_chunk_size();

    const auto& q_shape = input_tensor_q.logical_shape();
    const auto& joint_q_shape = joint_tensor_q.logical_shape();
    const uint32_t B = q_shape[0], NH = q_shape[1], N = q_shape[2], DH = q_shape[3];
    const uint32_t L = joint_q_shape[2];

    // Calculate padded sequence length
    const uint32_t padded_Nq = tt::round_up(N, q_chunk_size);
    const uint32_t padded_Nk = tt::round_up(N, k_chunk_size);
    const uint32_t padded_Lq = tt::round_up(L, q_chunk_size);
    const uint32_t padded_Lk = tt::round_up(L, k_chunk_size);

    const uint32_t padded_Nqt = padded_Nq / TILE_HEIGHT;
    const uint32_t padded_Nkt = padded_Nk / TILE_HEIGHT;
    const uint32_t padded_Lqt = padded_Lq / TILE_HEIGHT;
    const uint32_t padded_Lkt = padded_Lk / TILE_HEIGHT;

    // Find unpadded sequence lengths in tiles
    const uint32_t valid_Nt = tt::div_up(N, TILE_HEIGHT);
    const uint32_t valid_Lt = tt::div_up(L, TILE_HEIGHT);

    // Compute kernel operates on concatenated Q and K
    const uint32_t cat_Sq = padded_Nq + padded_Lq;
    const uint32_t cat_Sk = padded_Nk + padded_Lk;

    [[maybe_unused]] const uint32_t cat_Sqt = cat_Sq / TILE_HEIGHT;
    const uint32_t cat_Skt = cat_Sk / TILE_HEIGHT;
    const uint32_t DHt = DH / TILE_WIDTH;

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
    const bool use_joint_mask = (padded_Nk != N) || (padded_Lk != L);

    const uint32_t Sq_chunk_t = q_chunk_size / TILE_HEIGHT;
    const uint32_t Sk_chunk_t = k_chunk_size / TILE_HEIGHT;
    const uint32_t q_num_chunks = cat_Sq / q_chunk_size;
    const uint32_t k_num_chunks = cat_Sk / k_chunk_size;

    log_debug(tt::LogOp, "B: {}", B);
    log_debug(tt::LogOp, "NH: {}", NH);
    log_debug(tt::LogOp, "N: {}", N);
    log_debug(tt::LogOp, "L: {}", L);
    log_debug(tt::LogOp, "DH: {}", DH);

    // Log padded dimensions
    log_debug(tt::LogOp, "padded_Nq: {}", padded_Nq);
    log_debug(tt::LogOp, "padded_Nk: {}", padded_Nk);
    log_debug(tt::LogOp, "padded_Lq: {}", padded_Lq);
    log_debug(tt::LogOp, "padded_Lk: {}", padded_Lk);
    log_debug(tt::LogOp, "padded_Nqt: {}", padded_Nqt);
    log_debug(tt::LogOp, "padded_Nkt: {}", padded_Nkt);
    log_debug(tt::LogOp, "padded_Lqt: {}", padded_Lqt);
    log_debug(tt::LogOp, "padded_Lkt: {}", padded_Lkt);

    // Log tile dimensions
    log_debug(tt::LogOp, "DHt: {}", DHt);
    log_debug(tt::LogOp, "valid_Nt: {}", valid_Nt);
    log_debug(tt::LogOp, "valid_Lt: {}", valid_Lt);

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

    IDevice* device = input_tensor_q.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), args.compute_kernel_config);

    CoreCoord grid_size = args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size
                                                          : device->compute_with_storage_grid_size();
    bool exp_approx_mode =
        args.program_config.has_value()
            ? (args.program_config->exp_approx_mode.has_value() ? args.program_config->exp_approx_mode.value() : true)
            : true;

    auto core_grid = CoreRangeSet(CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1}));
    uint32_t num_cores = grid_size.x * grid_size.y;

    TT_FATAL(
        num_cores <= device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y,
        "Provided grid must not contain more cores than the device. Got {} cores, expected at most {} cores.",
        num_cores,
        device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y);

    // Parallelization scheme
    // We will choose parallelization factors for batch, num_heads, and q_seq_len in that order
    uint32_t batch_parallel_factor = std::min(B, num_cores);
    uint32_t nh_parallel_factor = std::min(num_cores / batch_parallel_factor, NH);
    uint32_t q_parallel_factor = std::min(num_cores / (batch_parallel_factor * nh_parallel_factor), q_num_chunks);

    TT_FATAL(
        batch_parallel_factor * nh_parallel_factor * q_parallel_factor <= num_cores,
        "Parallelism must not exceed number of cores. Got {}, expected at most {}.",
        batch_parallel_factor * nh_parallel_factor * q_parallel_factor,
        num_cores);

    log_debug(tt::LogOp, "Parallelization scheme:");
    log_debug(tt::LogOp, "batch_parallel_factor: {}", batch_parallel_factor);
    log_debug(tt::LogOp, "nh_parallel_factor: {}", nh_parallel_factor);
    log_debug(tt::LogOp, "q_parallel_factor: {}", q_parallel_factor);

    // Ceiling divide to allow for non-perfect divisions
    const uint32_t batch_per_core = tt::div_up(B, batch_parallel_factor);
    const uint32_t nh_per_core = tt::div_up(NH, nh_parallel_factor);
    const uint32_t q_per_core = tt::div_up(q_num_chunks, q_parallel_factor);

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

    // ---- Streaming compute (sdpa_standard_v2, same pipeline as the plain SDPA kernel) ----
    // The concatenated K = [N spatial | L joint]. Streaming treats it as one sequence whose valid
    // tile count is padded_Nkt + valid_Lt, so the spatial part must be k_chunk aligned (no interior
    // padding); only the joint tail may be partial. It is masked by the lightweight palette
    // [neginf, partial-col] exactly like a padded plain SDPA.
    const bool streaming_env_disabled = std::getenv("TT_JOINT_SDPA_NO_STREAMING") != nullptr;
    const bool use_streaming_compute = !fp32_dest_acc_en && !streaming_env_disabled && (padded_Nk == N);
    const uint32_t valid_Skt = padded_Nkt + valid_Lt;
    const uint32_t k_partial_col = use_streaming_compute ? (L % TILE_HEIGHT) : 0;
    const bool use_streaming_padded_mask =
        use_streaming_compute && ((valid_Skt % Sk_chunk_t != 0) || (k_partial_col != 0));
    log_debug(
        tt::LogOp,
        "joint sdpa use_streaming_compute: {} valid_Skt: {} k_partial_col: {} padded_mask: {}",
        use_streaming_compute,
        valid_Skt,
        k_partial_col,
        use_streaming_padded_mask);
    const uint32_t qk_in0_block_w = DHt;
    auto [qk_out_subblock_h, qk_out_subblock_w] =
        detail::determine_largest_subblock_size(Sq_chunk_t, Sk_chunk_t, dst_size);

    const uint32_t qk_in0_num_subblocks = Sq_chunk_t / qk_out_subblock_h;
    const uint32_t qk_in1_num_subblocks = Sk_chunk_t / qk_out_subblock_w;
    const uint32_t qk_num_blocks = DHt / qk_in0_block_w;

    // now for out0
    const uint32_t out_in0_block_w = Sk_chunk_t;

    auto [out_out_subblock_h, out_out_subblock_w] = detail::determine_largest_subblock_size(
        Sq_chunk_t, DHt, dst_size, /*max_subblock_h=*/use_streaming_compute ? 2 : UINT32_MAX);
    // Writer must drain cb_out with the same row-group cadence compute pushes.
    const uint32_t writer_out_row_group_h =
        use_streaming_compute
            ? ttnn::transformer::sdpa::streaming_qktv_h(out_out_subblock_h, out_out_subblock_w, dst_size, Sq_chunk_t)
            : out_out_subblock_h;
    if (use_streaming_compute) {
        // Streaming: cb_out shrinks to a 2-slot row-group ping-pong.
        out0_t = detail::streaming_cb_out_tiles(out_out_subblock_h, out_out_subblock_w, dst_size, Sq_chunk_t, DHt);
        TT_FATAL(
            Sq_chunk_t % out_out_subblock_h == 0,
            "Streaming cb_out drain requires Sq_chunk_t ({}) divisible by out_out_subblock_h ({})",
            Sq_chunk_t,
            out_out_subblock_h);
    }

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

    const uint32_t scale_packed = std::bit_cast<uint32_t>(args.scale);

    // log scale
    log_debug(tt::LogOp, "scale: {}", args.scale);

    std::vector<uint32_t> reader_compile_time_args = {
        B,
        NH,
        DHt,
        Sq_chunk_t,
        Sk_chunk_t,
        k_num_chunks,
        valid_Nt,
        valid_Lt,
        padded_Nqt,
        padded_Nkt,
        padded_Lqt,
        padded_Lkt,
        num_cores,
    };
    TensorAccessorArgs(input_tensor_q.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(joint_tensor_q.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(joint_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(joint_tensor_v.buffer()).append_to(reader_compile_time_args);

    // ---- KV store-and-forward chain (unicast) ----
    // Cores sharing one (batch, head) re-read the same K/V chunks from DRAM once per Q chunk; at
    // long N this op is DRAM-bandwidth bound. Like the plain SDPA reader, chain those cores so the
    // injector reads from DRAM and each core forwards the chunk to the next over the NoC. Only when
    // every core owns exactly one (batch, head) so the chain applies to all of the core's work.
    const bool kv_chain_env_disabled = std::getenv("TT_JOINT_SDPA_NO_KV_CHAIN") != nullptr;
    const bool kv_chain_enabled = !kv_chain_env_disabled && batch_per_core == 1 && nh_per_core == 1;
    const uint32_t sender_semaphore_id = 0;
    const uint32_t receiver_semaphore_id = 1;
    const uint32_t valid_semaphore_id = 2;
    reader_compile_time_args.push_back(static_cast<uint32_t>(kv_chain_enabled));
    reader_compile_time_args.push_back(sender_semaphore_id);
    reader_compile_time_args.push_back(receiver_semaphore_id);
    reader_compile_time_args.push_back(valid_semaphore_id);

    // Calculate which K chunks contain the mask boundaries
    // If a tensor does not require masking, set to MAX_UINT32. This avoids a
    // bug in the mask generation code, which would mask a full, valid chunk
    // with -inf.
    const uint32_t mask_chunk_0 =
        (padded_Nk != N) ? (padded_Nkt / Sk_chunk_t) - 1 : UINT32_MAX;  // idx of last chunk in first sequence
    const uint32_t mask_chunk_1 =
        (padded_Lk != L) ? (cat_Skt / Sk_chunk_t) - 1 : UINT32_MAX;  // idx of last chunk in second sequence

    std::vector<uint32_t> writer_compile_time_args = {
        B,
        NH,
        DHt,
        Sq_chunk_t,
        Sk_chunk_t,
        k_num_chunks,
        valid_Nt,
        valid_Lt,
        padded_Nqt,
        padded_Nkt,
        padded_Lqt,
        padded_Lkt,
        N,
        L,
        num_cores,
        packed_identity_scalar,
        scale_packed,
        static_cast<uint32_t>(use_joint_mask),
        mask_chunk_0,
        mask_chunk_1,
        static_cast<uint32_t>(use_streaming_compute),  // arg 20: row-grouped cb_out drain + lightweight mask
        writer_out_row_group_h,                        // arg 21: drain group height
        k_partial_col,                                 // arg 22: joint K partial-tile col (0 = none)
    };
    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(joint_output_tensor.buffer()).append_to(writer_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {
        B,
        NH,
        cat_Skt,
        DHt,
        Sq_chunk_t,
        Sk_chunk_t,
        k_num_chunks,
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
        static_cast<uint32_t>(use_joint_mask),
        mask_chunk_0,
        mask_chunk_1,
        scale_packed,
        static_cast<uint32_t>(use_streaming_compute),      // arg 23
        valid_Skt,                                         // arg 24: valid concatenated K tiles
        k_partial_col,                                     // arg 25
        static_cast<uint32_t>(tt::CBIndex::c_8),           // arg 26: recip scratch CB
        q_num_chunks,                                      // arg 27
        static_cast<uint32_t>(use_streaming_padded_mask),  // arg 28
    };

    std::map<std::string, std::string> defines_map;
    defines_map["STATS_GRANULARITY"] = std::to_string(stats_granularity);
    defines_map["SUB_EXP_GRANULARITY"] = std::to_string(sub_exp_granularity);
    defines_map["MUL_BCAST_GRANULARITY"] = std::to_string(mul_bcast_granularity);
    defines_map["DHT_GRANULARITY"] = std::to_string(dht_granularity);
    defines_map["REDUCE_GRANULARITY"] = std::to_string(reduce_granularity);
    defines_map["EXP_APPROX_MODE"] = std::to_string(exp_approx_mode);

    KernelDescriptor::Defines defines(defines_map.begin(), defines_map.end());

    // ---- Circular buffers ----

    ProgramDescriptor desc;

    tt::DataFormat q_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_q.dtype());
    tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_k.dtype());
    tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_v.dtype());
    // Streaming reads the lightweight palette as Float16_b (no block-float decode); legacy uses Bfp4_b.
    tt::DataFormat mask_df = use_streaming_compute ? tt::DataFormat::Float16_b : tt::DataFormat::Bfp4_b;
    tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::DataFormat scalar_df =
        (input_tensor_q.dtype() == DataType::FLOAT32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
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
    desc.cbs.push_back(CBDescriptor{
        .total_size = q_tiles * q_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
            .data_format = q_df,
            .page_size = q_tile_size,
        }}},
    });

    // K input
    desc.cbs.push_back(CBDescriptor{
        .total_size = k_tiles * k_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),
            .data_format = k_df,
            .page_size = k_tile_size,
        }}},
    });

    // V input
    desc.cbs.push_back(CBDescriptor{
        .total_size = v_tiles * v_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_2),
            .data_format = v_df,
            .page_size = v_tile_size,
        }}},
    });

    // Only create mask buffer if it's going to be used. Streaming always gets the lightweight
    // palette CB ([neginf, partial-col?]); legacy gets the full Sq x Sk generated mask.
    if (use_streaming_compute || use_joint_mask) {
        const uint32_t mask_cb_tiles = use_streaming_compute ? (1 + (k_partial_col > 0 ? 1 : 0)) : mask_tiles;
        // attn_mask input
        desc.cbs.push_back(CBDescriptor{
            .total_size = mask_cb_tiles * mask_tile_size,
            .core_ranges = core_grid,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
                .data_format = mask_df,
                .page_size = mask_tile_size,
            }}},
        });
    }

    // Streaming compute v2: 1-tile recip scratch CB for normalize_row_streaming.
    if (use_streaming_compute) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = im_tile_size,
            .core_ranges = core_grid,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_8),
                .data_format = im_df,
                .page_size = im_tile_size,
            }}},
        });
    }
    // identity scalar input
    desc.cbs.push_back(CBDescriptor{
        .total_size = scale_tiles * scalar_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_5),
            .data_format = scalar_df,
            .page_size = scalar_tile_size,
        }}},
    });

    // identity column input
    desc.cbs.push_back(CBDescriptor{
        .total_size = scale_tiles * scalar_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_7),
            .data_format = scalar_df,
            .page_size = scalar_tile_size,
        }}},
    });

    // cb_qk_im
    desc.cbs.push_back(CBDescriptor{
        .total_size = qk_tiles * im_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_24),
            .data_format = im_df,
            .page_size = im_tile_size,
        }}},
    });

    // cb_out_im
    desc.cbs.push_back(CBDescriptor{
        .total_size = out_im_tiles * im_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_25),
            .data_format = im_df,
            .page_size = im_tile_size,
        }}},
    });

    // cb_out_accumulate_im
    desc.cbs.push_back(CBDescriptor{
        .total_size = out_im_tiles * im_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_26),
            .data_format = im_df,
            .page_size = im_tile_size,
        }}},
    });

    // cb_cur_max
    desc.cbs.push_back(CBDescriptor{
        .total_size = statistics_tiles * stats_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_27),
            .data_format = stats_df,
            .page_size = stats_tile_size,
        }}},
    });

    // cb_prev_max
    desc.cbs.push_back(CBDescriptor{
        .total_size = statistics_tiles * stats_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_28),
            .data_format = stats_df,
            .page_size = stats_tile_size,
        }}},
    });

    // cb_cur_sum
    desc.cbs.push_back(CBDescriptor{
        .total_size = statistics_tiles * stats_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_29),
            .data_format = stats_df,
            .page_size = stats_tile_size,
        }}},
    });

    // cb_prev_sum
    desc.cbs.push_back(CBDescriptor{
        .total_size = statistics_tiles * stats_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_30),
            .data_format = stats_df,
            .page_size = stats_tile_size,
        }}},
    });

    // cb_exp_max_diff
    desc.cbs.push_back(CBDescriptor{
        .total_size = statistics_tiles * stats_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_31),
            .data_format = stats_df,
            .page_size = stats_tile_size,
        }}},
    });

    // Output
    desc.cbs.push_back(CBDescriptor{
        .total_size = out0_t * out_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_16),
            .data_format = out_df,
            .page_size = out_tile_size,
        }}},
    });

    // ---- Kernels ----

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/joint_reader.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_grid;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = defines;
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/joint_writer.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_grid;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.defines = defines;
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/joint_sdpa.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_grid;
    compute_desc.compile_time_args = compute_compile_time_args;
    compute_desc.defines = defines;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    auto* const q_buf = input_tensor_q.buffer();
    auto* const k_buf = input_tensor_k.buffer();
    auto* const v_buf = input_tensor_v.buffer();
    auto* const joint_q_buf = joint_tensor_q.buffer();
    auto* const joint_k_buf = joint_tensor_k.buffer();
    auto* const joint_v_buf = joint_tensor_v.buffer();
    auto* const out_buf = output_tensor.buffer();
    auto* const joint_out_buf = joint_output_tensor.buffer();

    // Per-core chain metadata. Core i owns (batch, head) group g = i / q_parallel_factor and the
    // q range [(i % q_parallel_factor) * q_per_core, ...). The chain for group g is its cores with
    // non-empty q ranges in index order: the first is the injector (reads DRAM), the last the sink.
    // q counts are non-increasing along the chain, so a sender forwards only for the receiver's
    // iteration count (next_core_q_chunks) and never over-produces.
    struct JointChainInfo {
        bool participates = false;
        bool is_injector = false;
        bool is_sink = false;
        CoreCoord prev_physical{0, 0};
        CoreCoord next_physical{0, 0};
        uint32_t next_core_q_chunks = 0;
    };
    std::vector<JointChainInfo> chain_info(num_cores);
    auto core_q_count = [&](uint32_t i) -> uint32_t {
        const uint32_t qs = std::min((i % q_parallel_factor) * q_per_core, q_num_chunks);
        const uint32_t qe = std::min(qs + q_per_core, q_num_chunks);
        const uint32_t bs = std::min((i / (nh_parallel_factor * q_parallel_factor)) * batch_per_core, B);
        const uint32_t ns = std::min(((i / q_parallel_factor) % nh_parallel_factor) * nh_per_core, NH);
        return (bs < B && ns < NH) ? (qe - qs) : 0u;
    };
    if (kv_chain_enabled) {
        const uint32_t num_groups = num_cores / q_parallel_factor;
        for (uint32_t g = 0; g < num_groups; ++g) {
            std::vector<uint32_t> members;
            for (uint32_t j = 0; j < q_parallel_factor; ++j) {
                const uint32_t i = g * q_parallel_factor + j;
                if (i < num_cores && core_q_count(i) > 0) {
                    members.push_back(i);
                }
            }
            if (members.size() < 2) {
                continue;
            }
            for (std::size_t pos = 0; pos < members.size(); ++pos) {
                auto& ci = chain_info[members[pos]];
                ci.participates = true;
                ci.is_injector = (pos == 0);
                ci.is_sink = (pos + 1 == members.size());
                if (pos > 0) {
                    const uint32_t p = members[pos - 1];
                    ci.prev_physical = device->worker_core_from_logical_core({p % grid_size.x, p / grid_size.x});
                }
                if (pos + 1 < members.size()) {
                    const uint32_t n = members[pos + 1];
                    ci.next_physical = device->worker_core_from_logical_core({n % grid_size.x, n / grid_size.x});
                    ci.next_core_q_chunks = core_q_count(n);
                }
            }
        }
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = sender_semaphore_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = core_grid,
            .initial_value = INVALID,
        });
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = receiver_semaphore_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = core_grid,
            .initial_value = INVALID,
        });
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = valid_semaphore_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = core_grid,
            .initial_value = VALID,
        });
    }

    // Set reader rt args
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        uint32_t local_batch_start = (i / (nh_parallel_factor * q_parallel_factor)) * batch_per_core;
        uint32_t local_batch_end = local_batch_start + batch_per_core;
        uint32_t local_nh_start = ((i / q_parallel_factor) % nh_parallel_factor) * nh_per_core;
        uint32_t local_nh_end = local_nh_start + nh_per_core;
        uint32_t local_q_start = (i % q_parallel_factor) * q_per_core;
        uint32_t local_q_end = local_q_start + q_per_core;

        // clamp all to max values for non-even partitioning
        local_batch_start = std::min(local_batch_start, B);
        local_batch_end = std::min(local_batch_end, B);
        local_nh_start = std::min(local_nh_start, NH);
        local_nh_end = std::min(local_nh_end, NH);
        local_q_start = std::min(local_q_start, q_num_chunks);
        local_q_end = std::min(local_q_end, q_num_chunks);

        // log the above
        log_debug(tt::LogOp, "core: {}", i);
        log_debug(tt::LogOp, "x={},y={}", core.x, core.y);
        log_debug(tt::LogOp, "local_batch_start: {}", local_batch_start);
        log_debug(tt::LogOp, "local_batch_end: {}", local_batch_end);
        log_debug(tt::LogOp, "local_nh_start: {}", local_nh_start);
        log_debug(tt::LogOp, "local_nh_end: {}", local_nh_end);
        log_debug(tt::LogOp, "local_q_start: {}", local_q_start);
        log_debug(tt::LogOp, "local_q_end: {}", local_q_end);

        reader_desc.emplace_runtime_args(
            core,
            {q_buf,
             k_buf,
             v_buf,
             joint_q_buf,
             joint_k_buf,
             joint_v_buf,
             local_batch_start,
             local_batch_end,
             local_nh_start,
             local_nh_end,
             local_q_start,
             local_q_end,
             static_cast<uint32_t>(chain_info[i].participates),
             static_cast<uint32_t>(chain_info[i].is_injector),
             static_cast<uint32_t>(chain_info[i].is_sink),
             static_cast<uint32_t>(chain_info[i].prev_physical.x),
             static_cast<uint32_t>(chain_info[i].prev_physical.y),
             static_cast<uint32_t>(chain_info[i].next_physical.x),
             static_cast<uint32_t>(chain_info[i].next_physical.y),
             chain_info[i].next_core_q_chunks});

        // Writer args
        writer_desc.emplace_runtime_args(
            core,
            {out_buf,
             joint_out_buf,
             local_batch_start,
             local_batch_end,
             local_nh_start,
             local_nh_end,
             local_q_start,
             local_q_end});

        // Compute args
        compute_desc.emplace_runtime_args(
            core, {local_batch_start, local_batch_end, local_nh_start, local_nh_end, local_q_start, local_q_end});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
