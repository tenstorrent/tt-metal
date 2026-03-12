//
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_profile_program_factory.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"

#include <optional>
#include <cmath>
#include <string>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

RingJointSDPAProfileProgramFactory::cached_program_t RingJointSDPAProfileProgramFactory::create(
    const RingJointSDPAProfileParams& args,
    const RingJointSDPAProfileInputs& tensor_args,
    RingJointSDPAProfileResult& output_tensors) {
    /*
    The profile op simulates one device in a ring, reading from pre-staged KV buffers.

    Naming:
        - padded_N: the global, padded sequence length
        - local_padded_N: the local shard of the padded sequence length. local_padded_N = padded_N / ring_size
        - logical_n: the logical global sequence length. logical_n <= padded_N.
        - L: the logical joint sequence length

    input_tensor_q: B x NH x local_padded_N x DH
    input_tensor_k: B x NH x local_padded_N x DH
    input_tensor_v: B x NH x local_padded_N x DH

    gathered_input_tensor_k: B x NH x padded_N x DH (pre-staged in arrival order)
    gathered_input_tensor_v: B x NH x padded_N x DH (pre-staged in arrival order)

    joint_tensor_q: B x NH x L x DH
    joint_tensor_k: B x NH x L x DH
    joint_tensor_v: B x NH x L x DH

    output_tensor: B x NH x local_padded_N x DH
    joint_output_tensor: B x NH x L x DH
    */

    log_debug(tt::LogOp, "DEBUG: ring_joint_sdpa_profile create is called");

    const auto& input_tensor_q = tensor_args.input_q;
    const auto& input_tensor_k = tensor_args.input_k;
    const auto& input_tensor_v = tensor_args.input_v;

    // Check if joint tensors are provided
    const bool use_joint_tensors =
        tensor_args.joint_q.has_value() && tensor_args.joint_k.has_value() && tensor_args.joint_v.has_value();
    uint32_t L = 0;  // Default joint sequence length
    if (use_joint_tensors) {
        const auto& joint_q_shape = tensor_args.joint_q.value().logical_shape();
        L = joint_q_shape[2];
    }

    const auto& gathered_input_tensor_k = tensor_args.gathered_k;
    const auto& gathered_input_tensor_v = tensor_args.gathered_v;

    auto& output_tensor = output_tensors.output;
    auto& lse_output_tensor = output_tensors.lse_output;

    // Check if joint output tensor exists
    const bool has_joint_output = output_tensors.joint_output.has_value();

    std::size_t q_chunk_size = args.get_q_chunk_size();
    std::size_t k_chunk_size = args.get_k_chunk_size();

    tt::tt_metal::Program program{};

    IDevice* device = input_tensor_q.device();

    auto scale = args.scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.logical_shape()[-1]));
    }

    const auto& q_shape = input_tensor_q.logical_shape();
    const auto& k_shape = gathered_input_tensor_k.logical_shape();
    const auto& v_shape = gathered_input_tensor_v.logical_shape();

    log_debug(tt::LogOp, "q_shape: {}", q_shape);
    log_debug(tt::LogOp, "k_shape (gathered): {}", k_shape);
    log_debug(tt::LogOp, "v_shape (gathered): {}", v_shape);
    log_debug(tt::LogOp, "ring_index: {}", args.ring_index);
    log_debug(tt::LogOp, "is_causal: {}", args.is_causal);
    log_debug(tt::LogOp, "is_balanced: {}", args.is_balanced);

    const uint32_t B = q_shape[0], NH = q_shape[1], NHK = k_shape[1], local_padded_N = q_shape[2], DH = q_shape[3];
    const uint32_t padded_N = k_shape[2];
    const uint32_t vDH = v_shape[3];

    const uint32_t local_padded_Nt = local_padded_N / tt::constants::TILE_HEIGHT;
    const uint32_t padded_Nt = padded_N / tt::constants::TILE_HEIGHT;
    const uint32_t Lt = tt::div_up(L, tt::constants::TILE_HEIGHT);
    const uint32_t DHt = DH / tt::constants::TILE_WIDTH;
    const uint32_t vDHt = vDH / tt::constants::TILE_WIDTH;
    const uint32_t logical_nt = tt::div_up(static_cast<uint32_t>(args.logical_n), tt::constants::TILE_HEIGHT);

    const uint32_t Sq_chunk_t = q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t Sk_chunk_t = k_chunk_size / tt::constants::TILE_HEIGHT;

    // Lightweight mask: only needed when any K/joint dimension has padding that doesn't fill a chunk.
    const bool local_n_has_padding = (local_padded_Nt % Sk_chunk_t) != 0;
    const bool global_n_has_padding = (args.logical_n % (Sk_chunk_t * tt::constants::TILE_HEIGHT)) != 0;
    const bool joint_has_padding = L > 0 && (L % (Sk_chunk_t * tt::constants::TILE_HEIGHT)) != 0;
    const bool needs_lightweight_mask =
        (local_n_has_padding || global_n_has_padding || joint_has_padding) && !args.is_causal;

    // Partial tile support when padding boundary falls inside a tile.
    const uint32_t global_n_partial_col = args.logical_n % tt::constants::TILE_HEIGHT;
    const uint32_t joint_l_partial_col = L % tt::constants::TILE_HEIGHT;
    const uint32_t partial_mask_tiles = (global_n_partial_col != 0 ? 1 : 0) + (joint_l_partial_col != 0 ? 1 : 0);
    const uint32_t total_lightweight_mask_tiles = 1 + partial_mask_tiles;

    const uint32_t num_local_q_chunks = tt::div_up(local_padded_N, q_chunk_size);
    const uint32_t num_joint_q_chunks = tt::div_up(L, q_chunk_size);
    const uint32_t num_q_chunks = num_local_q_chunks + num_joint_q_chunks;
    const uint32_t num_local_k_chunks = tt::div_up(local_padded_N, k_chunk_size);
    const uint32_t num_joint_k_chunks = tt::div_up(L, k_chunk_size);

    log_debug(tt::LogOp, "B: {}", B);
    log_debug(tt::LogOp, "NH: {}", NH);
    log_debug(tt::LogOp, "NHK: {}", NHK);
    log_debug(tt::LogOp, "L: {}", L);
    log_debug(tt::LogOp, "DH: {}", DH);
    log_debug(tt::LogOp, "vDH: {}", vDH);
    log_debug(tt::LogOp, "local_padded_N: {}", local_padded_N);
    log_debug(tt::LogOp, "padded_N: {}", padded_N);
    log_debug(tt::LogOp, "DHt: {}", DHt);
    log_debug(tt::LogOp, "vDHt: {}", vDHt);
    log_debug(tt::LogOp, "local_padded_Nt: {}", local_padded_Nt);
    log_debug(tt::LogOp, "padded_Nt: {}", padded_Nt);
    log_debug(tt::LogOp, "Lt: {}", Lt);
    log_debug(tt::LogOp, "Sq_chunk_t: {}", Sq_chunk_t);
    log_debug(tt::LogOp, "Sk_chunk_t: {}", Sk_chunk_t);
    log_debug(tt::LogOp, "num_local_q_chunks: {}", num_local_q_chunks);
    log_debug(tt::LogOp, "num_joint_q_chunks: {}", num_joint_q_chunks);
    log_debug(tt::LogOp, "q_chunk_size: {}", q_chunk_size);
    log_debug(tt::LogOp, "k_chunk_size: {}", k_chunk_size);
    log_debug(tt::LogOp, "num_q_chunks: {}", num_q_chunks);
    log_debug(tt::LogOp, "num_local_k_chunks: {}", num_local_k_chunks);
    log_debug(tt::LogOp, "num_joint_k_chunks: {}", num_joint_k_chunks);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), args.compute_kernel_config);

    CoreCoord grid_size = args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size
                                                          : device->compute_with_storage_grid_size();
    bool exp_approx_mode =
        args.program_config.has_value()
            ? (args.program_config->exp_approx_mode.has_value() ? args.program_config->exp_approx_mode.value() : true)
            : true;

    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    uint32_t num_cores = grid_size.x * grid_size.y;

    log_debug(tt::LogOp, "num_cores: {}", num_cores);
    log_debug(tt::LogOp, "grid_size: {}", grid_size);

    TT_FATAL(
        num_cores <= device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y,
        "Provided grid must not contain more cores than the device. Got {} cores, expected at most {} cores.",
        num_cores,
        device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y);

    const uint32_t all_heads_num_q_chunks = B * NH * num_q_chunks;
    const uint32_t q_per_core = tt::div_up(all_heads_num_q_chunks, num_cores);

    const uint32_t q_buffer_factor = (q_per_core > 1) ? 2 : 1;

    log_debug(tt::LogOp, "q_per_core: {}", q_per_core);

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel
    uint32_t q_tiles = Sq_chunk_t * DHt * q_buffer_factor;
    uint32_t k_tiles = Sk_chunk_t * DHt * 2;   // double buffer
    uint32_t v_tiles = Sk_chunk_t * vDHt * 2;  // double buffer
    uint32_t mask_tiles = Sq_chunk_t * Sk_chunk_t;
    uint32_t qk_tiles = Sq_chunk_t * Sk_chunk_t;
    uint32_t out_im_tiles = Sq_chunk_t * vDHt;
    uint32_t out0_t = Sq_chunk_t * vDHt;
    uint32_t scale_tiles = 1;
    uint32_t statistics_tiles = Sq_chunk_t;

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
    auto [qk_out_subblock_h, qk_out_subblock_w] =
        detail::determine_largest_subblock_size(Sq_chunk_t, Sk_chunk_t, dst_size);

    const uint32_t qk_in0_num_subblocks = Sq_chunk_t / qk_out_subblock_h;
    const uint32_t qk_in1_num_subblocks = Sk_chunk_t / qk_out_subblock_w;
    const uint32_t qk_num_blocks = DHt / qk_in0_block_w;

    // now for out0
    const uint32_t out_in0_block_w = Sk_chunk_t;

    auto [out_out_subblock_h, out_out_subblock_w] = detail::determine_largest_subblock_size(Sq_chunk_t, vDHt, dst_size);

    const uint32_t out_in0_num_subblocks = Sq_chunk_t / out_out_subblock_h;
    const uint32_t out_in1_num_subblocks = vDHt / out_out_subblock_w;
    const uint32_t out_num_blocks = Sk_chunk_t / out_in0_block_w;

    // Streaming compute v2: eliminates row buffers via cb_push_back_hold_wr_ptr.
    const bool use_streaming_compute = !fp32_dest_acc_en && qk_out_subblock_h <= 2 &&
                                       Sk_chunk_t % (8 / qk_out_subblock_h) == 0 && qk_in0_num_subblocks > 1 &&
                                       !args.is_causal;
    log_debug(tt::LogOp, "use_streaming_compute: {}", use_streaming_compute);

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
    const uint32_t stats_granularity = detail::find_valid_granularity(Sq_chunk_t, dst_size);
    const uint32_t sub_exp_granularity = detail::find_valid_granularity(Sk_chunk_t, dst_size);
    const uint32_t mul_bcast_granularity = detail::find_valid_granularity(Sq_chunk_t * Sk_chunk_t, dst_size);
    uint32_t dht_granularity = std::min({DHt, vDHt, dst_size});
    while (dht_granularity > 1 && (DHt % dht_granularity != 0 || vDHt % dht_granularity != 0)) {
        dht_granularity--;
    }
    const uint32_t reduce_granularity = detail::find_valid_granularity(Sq_chunk_t, dst_size / 2);

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

    log_debug(tt::LogOp, "scale: {}", scale_union.f);

    std::vector<uint32_t> reader_compile_time_args = {
        B,
        NH,
        NHK,
        DHt,
        vDHt,
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
        static_cast<uint32_t>(args.ring_size),
        static_cast<uint32_t>(args.ring_index),  // Profile: ring_index as compile-time arg
        args.is_causal,
        args.is_balanced};

    TensorAccessorArgs(input_tensor_q.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(gathered_input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(gathered_input_tensor_v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(tensor_args.joint_q.has_value() ? tensor_args.joint_q->buffer() : nullptr)
        .append_to(reader_compile_time_args);
    TensorAccessorArgs(tensor_args.joint_k.has_value() ? tensor_args.joint_k->buffer() : nullptr)
        .append_to(reader_compile_time_args);
    TensorAccessorArgs(tensor_args.joint_v.has_value() ? tensor_args.joint_v->buffer() : nullptr)
        .append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        B,
        NH,
        NHK,
        DHt,
        vDHt,
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
        packed_identity_scalar,
        scale_union.u,
        static_cast<uint32_t>(args.ring_size),
        static_cast<uint32_t>(args.ring_index),  // Profile: ring_index as compile-time arg
        global_n_partial_col,
        joint_l_partial_col,
        args.is_causal,
        args.is_balanced};

    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(has_joint_output ? output_tensors.joint_output.value().buffer() : nullptr)
        .append_to(writer_compile_time_args);
    TensorAccessorArgs(lse_output_tensor.buffer()).append_to(writer_compile_time_args);

    // Early format check: when all data formats are identical, reconfig calls can be skipped.
    const tt::DataFormat q_df_early = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_q.dtype());
    const tt::DataFormat k_df_early = tt::tt_metal::datatype_to_dataformat_converter(gathered_input_tensor_k.dtype());
    const tt::DataFormat v_df_early = tt::tt_metal::datatype_to_dataformat_converter(gathered_input_tensor_v.dtype());
    const tt::DataFormat out_df_early = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    const tt::DataFormat im_df_early = tt::DataFormat::Float16_b;
    const tt::DataFormat mask_df_early = (args.is_causal ? tt::DataFormat::Bfp4_b : tt::DataFormat::Float16_b);
    const bool uniform_dataformat =
        (q_df_early == k_df_early && q_df_early == v_df_early && q_df_early == out_df_early &&
         q_df_early == mask_df_early && q_df_early == im_df_early);

    std::vector<uint32_t> compute_compile_time_args = {
        B,
        NH,
        NHK,
        DHt,
        vDHt,
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
        static_cast<uint32_t>(args.ring_size),
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
        args.is_causal,
        args.is_balanced};

    std::map<std::string, std::string> defines;
    defines["STATS_GRANULARITY"] = std::to_string(stats_granularity);
    defines["SUB_EXP_GRANULARITY"] = std::to_string(sub_exp_granularity);
    defines["MUL_BCAST_GRANULARITY"] = std::to_string(mul_bcast_granularity);
    defines["DHT_GRANULARITY"] = std::to_string(dht_granularity);
    defines["REDUCE_GRANULARITY"] = std::to_string(reduce_granularity);
    defines["EXP_APPROX_MODE"] = std::to_string(exp_approx_mode);

    // Profile kernels: simplified reader/writer without sync
    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_profile_reader.cpp",
        core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, defines));

    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_profile_writer.cpp",
        core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, defines));

    // Reuse the exact same compute kernel
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

    tt::DataFormat mask_df = (args.is_causal ? tt::DataFormat::Bfp4_b : tt::DataFormat::Float16_b);
    tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::DataFormat scalar_df = tt::DataFormat::Float16_b;
    tt::DataFormat im_df = tt::DataFormat::Float16_b;
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

    if (args.is_causal) {
        // attn_mask input
        auto c_in3_config = CircularBufferConfig(mask_tiles * mask_tile_size, {{tt::CB::c_in3, mask_df}})
                                .set_page_size(tt::CB::c_in3, mask_tile_size);
        CreateCircularBuffer(program, core_grid, c_in3_config);
    } else {
        // Lightweight mask: single CB holds 1 neginf tile + up to 2 partial mask tiles
        if (needs_lightweight_mask) {
            auto c_in3_config =
                CircularBufferConfig(total_lightweight_mask_tiles * mask_tile_size, {{tt::CB::c_in3, mask_df}})
                    .set_page_size(tt::CB::c_in3, mask_tile_size);
            CreateCircularBuffer(program, core_grid, c_in3_config);
        }
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

    // Streaming compute v2: 1-tile recip scratch CB (c_9) for normalize_row_streaming.
    if (use_streaming_compute) {
        auto c_recip_scratch_config = CircularBufferConfig(1 * im_tile_size, {{tt::CBIndex::c_9, im_df}})
                                          .set_page_size(tt::CBIndex::c_9, im_tile_size);
        CreateCircularBuffer(program, core_grid, c_recip_scratch_config);
    }

    uint32_t q_addr = input_tensor_q.buffer()->address();
    uint32_t k_addr = input_tensor_k.buffer()->address();
    uint32_t v_addr = input_tensor_v.buffer()->address();
    uint32_t gathered_k_addr = gathered_input_tensor_k.buffer()->address();
    uint32_t gathered_v_addr = gathered_input_tensor_v.buffer()->address();
    uint32_t joint_q_addr = use_joint_tensors ? tensor_args.joint_q.value().buffer()->address() : 0;
    uint32_t joint_k_addr = use_joint_tensors ? tensor_args.joint_k.value().buffer()->address() : 0;
    uint32_t joint_v_addr = use_joint_tensors ? tensor_args.joint_v.value().buffer()->address() : 0;
    uint32_t out_addr = output_tensor.buffer()->address();
    uint32_t joint_out_addr = has_joint_output ? output_tensors.joint_output.value().buffer()->address() : 0;
    uint32_t lse_addr = lse_output_tensor.buffer()->address();

    // Set runtime args for each core
    const uint32_t total_q_chunks = B * NH * num_q_chunks;
    const uint32_t base_chunks_per_core = (num_cores == 0) ? 0 : (total_q_chunks / num_cores);
    const uint32_t extra_chunks = (num_cores == 0) ? 0 : (total_q_chunks % num_cores);
    uint32_t next_global_chunk = 0;

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        uint32_t chunk_count = base_chunks_per_core + ((i < extra_chunks) ? 1 : 0);
        if (next_global_chunk >= total_q_chunks) {
            chunk_count = 0;
        } else if (chunk_count > total_q_chunks - next_global_chunk) {
            chunk_count = total_q_chunks - next_global_chunk;
        }

        uint32_t global_q_start = next_global_chunk;
        uint32_t global_q_end = next_global_chunk + chunk_count;

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
        SetRuntimeArgs(program, reader_kernels_id, core, reader_args);

        // Writer args
        std::vector<uint32_t> writer_args = {
            out_addr,
            joint_out_addr,
            lse_addr,
            global_q_start,
            global_q_end,
        };
        SetRuntimeArgs(program, writer_kernels_id, core, writer_args);

        // Compute args
        std::vector<uint32_t> compute_args = {
            global_q_start,
            global_q_end,
            static_cast<uint32_t>(args.ring_size),
            static_cast<uint32_t>(args.ring_index),  // Profile: pass ring_index to compute
            // Expected inputs for fused_op_indexer: ring_size, ring_index, forward_writes, backward_writes
            // For profile, we simulate all writes completed
            static_cast<uint32_t>(args.ring_size) - 1 -
                static_cast<uint32_t>(args.ring_index),  // forward_writes_expected
            static_cast<uint32_t>(args.ring_index),      // backward_writes_expected
        };
        SetRuntimeArgs(program, compute_kernels_id, core, compute_args);

        next_global_chunk += chunk_count;
    }

    return cached_program_t{
        std::move(program), {num_cores, grid_size, reader_kernels_id, writer_kernels_id, compute_kernels_id}};
}

void RingJointSDPAProfileProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    [[maybe_unused]] const RingJointSDPAProfileParams& args,
    const RingJointSDPAProfileInputs& tensor_args,
    RingJointSDPAProfileResult& output_tensors) {
    auto& shared_vars = cached_program.shared_variables;
    auto& program = cached_program.program;

    // Get addresses for regular tensors
    auto* q_buffer = tensor_args.input_q.buffer();
    auto* k_buffer = tensor_args.input_k.buffer();
    auto* v_buffer = tensor_args.input_v.buffer();
    auto* gathered_k_buffer = tensor_args.gathered_k.buffer();
    auto* gathered_v_buffer = tensor_args.gathered_v.buffer();

    // Check if joint tensors are provided
    const bool use_joint_tensors =
        tensor_args.joint_q.has_value() && tensor_args.joint_k.has_value() && tensor_args.joint_v.has_value();
    const bool has_joint_output = output_tensors.joint_output.has_value();

    // Get joint buffer pointers conditionally
    auto* joint_q_buffer = use_joint_tensors ? tensor_args.joint_q.value().buffer() : nullptr;
    auto* joint_k_buffer = use_joint_tensors ? tensor_args.joint_k.value().buffer() : nullptr;
    auto* joint_v_buffer = use_joint_tensors ? tensor_args.joint_v.value().buffer() : nullptr;

    // Get addresses for output tensors
    auto* out_buffer = output_tensors.output.buffer();
    auto* joint_out_buffer = has_joint_output ? output_tensors.joint_output.value().buffer() : nullptr;
    auto* lse_buffer = output_tensors.lse_output.buffer();

    uint32_t q_addr = q_buffer->address();
    uint32_t k_addr = k_buffer->address();
    uint32_t v_addr = v_buffer->address();
    uint32_t gathered_k_addr = gathered_k_buffer->address();
    uint32_t gathered_v_addr = gathered_v_buffer->address();
    uint32_t joint_q_addr = use_joint_tensors ? joint_q_buffer->address() : 0;
    uint32_t joint_k_addr = use_joint_tensors ? joint_k_buffer->address() : 0;
    uint32_t joint_v_addr = use_joint_tensors ? joint_v_buffer->address() : 0;
    uint32_t out_addr = out_buffer->address();
    uint32_t joint_out_addr = has_joint_output ? joint_out_buffer->address() : 0;
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

}  // namespace ttnn::prim
