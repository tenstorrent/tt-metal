// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sdpa_program_factory.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/math.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <optional>
#include <string>
#include <cmath>
#include <tuple>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::transformer::sdpa::program {

SDPAProgramFactory::cached_program_t SDPAProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor_q = tensor_args.q;
    const auto& input_tensor_k = tensor_args.k;
    const auto& input_tensor_v = operation_attributes.use_mla ? tensor_args.k : tensor_args.v.value_or(tensor_args.k);
    const auto& output_tensor = tensor_return_value;
    const auto& attn_mask = tensor_args.attn_mask;
    const auto& page_table = tensor_args.page_table;
    const auto& attention_sink = tensor_args.attention_sink;
    auto scale = operation_attributes.scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.padded_shape()[-1]));
    }
    const bool is_causal = operation_attributes.is_causal;
    const auto& chunk_start_idx = operation_attributes.chunk_start_idx;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;
    auto program_config = operation_attributes.program_config;
    const bool use_mla = operation_attributes.use_mla;
    const uint32_t head_dim_v = operation_attributes.head_dim_v.value_or(input_tensor_q.logical_shape()[3]);
    const auto& sliding_window_size = operation_attributes.sliding_window_size;

    std::size_t q_chunk_size =
        operation_attributes.program_config ? operation_attributes.program_config->q_chunk_size : 32;
    std::size_t k_chunk_size =
        operation_attributes.program_config ? operation_attributes.program_config->k_chunk_size : 32;

    /*
    Q: B x NQH x S x DH
    K: B x NKH x DH x S
    V: B x NKH x S x DH
    attn_mask: B x NQH x S x S
    */

    const auto& q_shape = input_tensor_q.logical_shape();
    const auto& k_shape = input_tensor_k.logical_shape();
    const uint32_t B = q_shape[0], NQH = q_shape[1], Sq = q_shape[2], DH = q_shape[3];
    const uint32_t NKH = k_shape[1];

    // Paged cache parameters when in chunked mode
    bool is_chunked = chunk_start_idx.has_value();
    // In chunked mode, we only need to process K/V up to chunk_start_idx + Sq
    const uint32_t Sk = is_chunked ? (chunk_start_idx.value() + Sq) : k_shape[2];

    /*
    Note about tensor shapes:
    SDPA inputs may be padded on the sequence length dimension. In addition,
    q_chunk_size and k_chunk_size don't have to divide the valid sequence length.
    Internally, the kernels pad tensors up to nearest multiple of the larger chunk size
    and handle masking pad tokens when appropriate.
    */

    // Calculate padded sequence length
    const uint32_t padded_Sq = std::ceil((float)Sq / q_chunk_size) * q_chunk_size;
    const uint32_t padded_Sk = std::ceil((float)Sk / k_chunk_size) * k_chunk_size;

    const uint32_t Sqt = padded_Sq / TILE_HEIGHT;
    const uint32_t Skt = padded_Sk / TILE_HEIGHT;
    const uint32_t DHt = DH / TILE_WIDTH;
    const uint32_t vDHt = use_mla ? head_dim_v / TILE_WIDTH : DHt;

    const uint32_t valid_Sqt = std::ceil((float)Sq / TILE_HEIGHT);
    const uint32_t valid_Skt = std::ceil((float)Sk / TILE_HEIGHT);
    /*
    For non-causal case we must provide a padded mask if the K sequence length has been padded
    Note that we dont have this issue in non-causal case if Q is padded, since those pad tokens
    don't affect attention of unpadded tokens.
    In causal case, the causal mask takes care of masking K pad tokens.
    */
    const bool use_padded_mask = (!is_causal) && (padded_Sk != Sk);

    const uint32_t Sq_chunk_t = q_chunk_size / TILE_HEIGHT;
    const uint32_t Sk_chunk_t = k_chunk_size / TILE_HEIGHT;
    const uint32_t q_num_chunks = padded_Sq / q_chunk_size;
    const uint32_t k_num_chunks = padded_Sk / k_chunk_size;
    const bool use_provided_mask = attn_mask.has_value();

    // log_info all of the above
    log_info(tt::LogOp, "B: {}", B);
    log_info(tt::LogOp, "NQH: {}", NQH);

    log_info(tt::LogOp, "Sq: {}", Sq);
    log_info(tt::LogOp, "Sk: {}", Sk);
    log_info(tt::LogOp, "padded_Sq: {}", padded_Sq);
    log_info(tt::LogOp, "padded_Sk: {}", padded_Sk);
    log_info(tt::LogOp, "valid_Sqt: {}", valid_Sqt);
    log_info(tt::LogOp, "valid_Skt: {}", valid_Skt);
    log_info(tt::LogOp, "DH: {}", DH);
    log_info(tt::LogOp, "Sqt: {}", Sqt);
    log_info(tt::LogOp, "Skt: {}", Skt);
    log_info(tt::LogOp, "DHt: {}", DHt);
    log_info(tt::LogOp, "vDHt: {}", vDHt);
    log_info(tt::LogOp, "Sq_chunk_t: {}", Sq_chunk_t);
    log_info(tt::LogOp, "Sk_chunk_t: {}", Sk_chunk_t);
    log_info(tt::LogOp, "q_chunk_size: {}", q_chunk_size);
    log_info(tt::LogOp, "k_chunk_size: {}", k_chunk_size);
    log_info(tt::LogOp, "q_num_chunks: {}", q_num_chunks);
    log_info(tt::LogOp, "k_num_chunks: {}", k_num_chunks);
    log_info(tt::LogOp, "NKH: {}", NKH);
    log_info(tt::LogOp, "sliding_window_size: {}", sliding_window_size.has_value() ? sliding_window_size.value() : 0);

    // In chunked prefill mode, the offset of Q in terms of Q chunks
    uint32_t chunked_q_chunk_offset = 0;
    uint32_t block_size = 0;
    uint32_t block_size_t = 0;
    [[maybe_unused]] uint32_t max_blocks_per_seq = 0;
    uint32_t page_table_stick_size = 0;
    tt::DataFormat page_table_df = tt::DataFormat::Int32;

    if (is_chunked) {
        chunked_q_chunk_offset = chunk_start_idx.value() / q_chunk_size;
        const auto& page_table_tensor = page_table.value();
        block_size = k_shape[2];  // K's sequence dimension represents block size
        block_size_t = block_size / TILE_HEIGHT;
        max_blocks_per_seq = page_table_tensor.padded_shape()[1];
        page_table_stick_size = page_table_tensor.buffer()->aligned_page_size();
        TT_FATAL(
            page_table_stick_size % 32 == 0,
            "page table page size in bytes must be a multiple of 32 due to address alignment");

        TT_FATAL(
            page_table_stick_size % 32 == 0,
            "page table page size in bytes must be a multiple of 32 due to address alignment");
    }
    // Log page table info
    log_info(tt::LogOp, "is_chunked: {}", is_chunked);
    if (is_chunked) {
        log_info(tt::LogOp, "block_size: {}", block_size);
        log_info(tt::LogOp, "block_size_t: {}", block_size_t);
        log_info(tt::LogOp, "max_blocks_per_seq: {}", max_blocks_per_seq);
        log_info(tt::LogOp, "page_table_stick_size: {}", page_table_stick_size);
        log_info(tt::LogOp, "page_table_df: {}", page_table_df);
    }

    Program program = CreateProgram();

    IDevice* device = input_tensor_q.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    auto* q_buffer = input_tensor_q.buffer();
    auto* k_buffer = input_tensor_k.buffer();
    auto* v_buffer = use_mla ? input_tensor_k.buffer() : input_tensor_v.buffer();
    auto* mask_buffer = attn_mask.has_value() ? attn_mask.value().buffer() : nullptr;
    auto* attention_sink_buffer = attention_sink.has_value() ? attention_sink.value().buffer() : nullptr;

    auto* out0_buffer = output_tensor.buffer();

    bool use_attention_sink = attention_sink.has_value();

    CoreCoord grid_size = program_config.has_value() ? program_config->compute_with_storage_grid_size
                                                     : device->compute_with_storage_grid_size();
    bool exp_approx_mode =
        program_config.has_value()
            ? (program_config->exp_approx_mode.has_value() ? program_config->exp_approx_mode.value() : true)
            : true;

    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    uint32_t num_cores = grid_size.x * grid_size.y;

    TT_FATAL(
        num_cores <= device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y,
        "Provided grid must not contain more cores than the device. Got {} cores, expected at most {} cores.",
        num_cores,
        device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y);

    const uint32_t total_q_chunks = B * NQH * q_num_chunks;
    const uint32_t max_q_chunks_per_core = (num_cores == 0) ? 0 : ((total_q_chunks + num_cores - 1) / num_cores);
    const uint32_t q_buffer_factor = (max_q_chunks_per_core > 1) ? 2 : 1;

    log_info(tt::LogOp, "total_q_chunks: {}", total_q_chunks);
    log_info(tt::LogOp, "max_q_chunks_per_core: {}", max_q_chunks_per_core);

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t q_tiles = Sq_chunk_t * DHt * q_buffer_factor;
    uint32_t k_tiles = Sk_chunk_t * DHt * 2;            // double buffer
    uint32_t v_tiles = Sk_chunk_t * vDHt * 2;           // double buffer
    uint32_t mask_tiles = Sq_chunk_t * Sk_chunk_t * 2;  // double buffer
    uint32_t qk_tiles = Sq_chunk_t * Sk_chunk_t;
    uint32_t out_im_tiles = Sq_chunk_t * vDHt;
    uint32_t out0_t = Sq_chunk_t * vDHt;
    uint32_t scale_tiles = 1;
    uint32_t statistics_tiles = Sq_chunk_t;  // Single column of values in each iteration
    uint32_t attention_sink_tiles = use_attention_sink ? Sq_chunk_t : 0;  // One column vector per Q chunk

    // log all values
    log_info(tt::LogOp, "q_tiles: {}", q_tiles);
    log_info(tt::LogOp, "k_tiles: {}", k_tiles);
    log_info(tt::LogOp, "v_tiles: {}", v_tiles);
    log_info(tt::LogOp, "mask_tiles: {}", mask_tiles);
    log_info(tt::LogOp, "qk_tiles: {}", qk_tiles);
    log_info(tt::LogOp, "out0_t: {}", out0_t);
    log_info(tt::LogOp, "scale_tiles: {}", scale_tiles);
    log_info(tt::LogOp, "statistics_tiles: {}", statistics_tiles);
    log_info(tt::LogOp, "attention_sink_tiles: {}", attention_sink_tiles);

    // Host code is responsible for determining matmul configuration
    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
    const uint32_t qk_in0_block_w = DHt;
    // max of Sk_chunk_t and dst_size
    uint32_t qk_out_subblock_w = std::min(Sk_chunk_t, dst_size);
    // If qk_out_subblock_w is full row of output, scale subblock_h so volume = dst_size. Otherwise it's 1 to maintain
    // row-major intermediate buffer.
    uint32_t qk_out_subblock_h =
        (qk_out_subblock_w == Sk_chunk_t) ? (std::min(Sq_chunk_t, dst_size / qk_out_subblock_w)) : 1;

    if (qk_out_subblock_w == dst_size && qk_out_subblock_h == 1 && Sk_chunk_t % 2 == 0 && Sq_chunk_t % 2 == 0) {
        // Hacky, try to get 2x4 output subblock if possible to optimize matmul util.
        qk_out_subblock_w = qk_out_subblock_w / 2;
        qk_out_subblock_h = 2;
    }

    const uint32_t qk_in0_num_subblocks = Sq_chunk_t / qk_out_subblock_h;
    const uint32_t qk_in1_num_subblocks = Sk_chunk_t / qk_out_subblock_w;
    const uint32_t qk_num_blocks = DHt / qk_in0_block_w;

    // now for out0
    const uint32_t out_in0_block_w = Sk_chunk_t;
    const uint32_t out_out_subblock_w = std::min(vDHt, dst_size);
    const uint32_t out_out_subblock_h =
        (out_out_subblock_w == vDHt) ? (std::min(Sq_chunk_t, dst_size / out_out_subblock_w)) : 1;

    const uint32_t out_in0_num_subblocks = Sq_chunk_t / out_out_subblock_h;
    const uint32_t out_in1_num_subblocks = vDHt / out_out_subblock_w;
    const uint32_t out_num_blocks = Sk_chunk_t / out_in0_block_w;

    // log all values
    log_info(tt::LogOp, "dst_size: {}", dst_size);
    log_info(tt::LogOp, "qk_in0_block_w: {}", qk_in0_block_w);
    log_info(tt::LogOp, "qk_out_subblock_w: {}", qk_out_subblock_w);
    log_info(tt::LogOp, "qk_out_subblock_h: {}", qk_out_subblock_h);
    log_info(tt::LogOp, "qk_in0_num_subblocks: {}", qk_in0_num_subblocks);
    log_info(tt::LogOp, "qk_in1_num_subblocks: {}", qk_in1_num_subblocks);
    log_info(tt::LogOp, "qk_num_blocks: {}", qk_num_blocks);
    log_info(tt::LogOp, "out_in0_block_w: {}", out_in0_block_w);
    log_info(tt::LogOp, "out_out_subblock_w: {}", out_out_subblock_w);
    log_info(tt::LogOp, "out_out_subblock_h: {}", out_out_subblock_h);
    log_info(tt::LogOp, "out_in0_num_subblocks: {}", out_in0_num_subblocks);
    log_info(tt::LogOp, "out_in1_num_subblocks: {}", out_in1_num_subblocks);
    log_info(tt::LogOp, "out_num_blocks: {}", out_num_blocks);

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
    log_info(tt::LogOp, "stats_granularity: {}", stats_granularity);
    log_info(tt::LogOp, "log2_stats_granularity: {}", log2_stats_granularity);
    log_info(tt::LogOp, "sub_exp_granularity: {}", sub_exp_granularity);
    log_info(tt::LogOp, "log2_sub_exp_granularity: {}", log2_sub_exp_granularity);
    log_info(tt::LogOp, "mul_bcast_granularity: {}", mul_bcast_granularity);
    log_info(tt::LogOp, "log2_mul_bcast_granularity: {}", log2_mul_bcast_granularity);
    log_info(tt::LogOp, "dht_granularity: {}", dht_granularity);
    log_info(tt::LogOp, "log2_dht_granularity: {}", log2_dht_granularity);
    log_info(tt::LogOp, "reduce_granularity: {}", reduce_granularity);
    log_info(tt::LogOp, "log2_reduce_granularity: {}", log2_reduce_granularity);

    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    class bfloat16 bfloat_identity_scalar(1.0f);
    uint32_t packed_identity_scalar = pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});

    union {
        float f;
        uint32_t u;
    } scale_union{};
    scale_union.f = scale.value_or(1.0f);

    /**
     * Create semaphores for L1-L1 sharing of KV
     */
    auto sender_semahpore_id = CreateSemaphore(program, core_grid, INVALID);
    auto receiver_semahpore_id = CreateSemaphore(program, core_grid, INVALID);
    auto valid_semahpore_id = CreateSemaphore(program, core_grid, VALID);

    std::vector<uint32_t> reader_compile_time_args = {// interleaved accessor args
                                                      B,
                                                      NQH,
                                                      NKH,
                                                      Sqt,
                                                      Skt,
                                                      valid_Sqt,
                                                      valid_Skt,
                                                      DHt,
                                                      vDHt,
                                                      Sq_chunk_t,
                                                      q_num_chunks,
                                                      Sk_chunk_t,
                                                      k_num_chunks,
                                                      num_cores,
                                                      (std::uint32_t)is_causal,
                                                      (std::uint32_t)use_provided_mask,
                                                      (std::uint32_t)use_padded_mask,
                                                      (uint32_t)is_chunked,
                                                      block_size_t,
                                                      page_table_stick_size,
                                                      (std::uint32_t)use_attention_sink,
                                                      max_q_chunks_per_core,
                                                      sender_semahpore_id,
                                                      receiver_semahpore_id,
                                                      valid_semahpore_id};

    TensorAccessorArgs(input_tensor_q.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(attn_mask.has_value() ? attn_mask->buffer() : nullptr).append_to(reader_compile_time_args);
    TensorAccessorArgs(page_table.has_value() ? page_table->buffer() : nullptr).append_to(reader_compile_time_args);
    TensorAccessorArgs(attention_sink.has_value() ? attention_sink->buffer() : nullptr)
        .append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        B,
        NQH,
        NKH,
        Sqt,
        valid_Sqt,
        Sk,
        DHt,
        vDHt,
        Sq_chunk_t,
        q_num_chunks,
        Sk_chunk_t,
        k_num_chunks,
        packed_identity_scalar,
        scale_union.u,
        num_cores,
        (std::uint32_t)is_causal,
        (std::uint32_t)use_provided_mask,
        (std::uint32_t)use_padded_mask,
        (uint32_t)is_chunked,
        sliding_window_size.value_or(0),
    };

    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {
        // matmul args
        B,
        NQH,
        NKH,
        Skt,
        DHt,
        vDHt,
        Sq_chunk_t,
        q_num_chunks,
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
        num_cores,
        (std::uint32_t)is_causal,
        (std::uint32_t)use_provided_mask,
        (std::uint32_t)use_padded_mask,
        (uint32_t)is_chunked,
        scale_union.u,
        sliding_window_size.value_or(0),
        (std::uint32_t)use_attention_sink,
    };

    TensorAccessorArgs(output_tensor.buffer()).append_to(compute_compile_time_args);

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
    uint32_t balanced_q_parallel = 0;
    log_info(tt::LogOp, "BALANCED_Q_PARALLEL: {}", balanced_q_parallel);

    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp",
        core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, defines));

    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/writer_interleaved.cpp",
        core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, defines));

    auto compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
            .defines = defines});

    // Create circular buffers

    tt::DataFormat q_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_q.dtype());
    tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_k.dtype());
    tt::DataFormat v_df;
    if (use_mla) {
        v_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_k.dtype());
    } else {
        v_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_v.dtype());
    }
    tt::DataFormat mask_df = attn_mask.has_value()
                                 ? tt::tt_metal::datatype_to_dataformat_converter(attn_mask.value().dtype())
                                 : tt::DataFormat::Bfp4_b;
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

    log_info(tt::LogOp, "q_data_format: {}", q_df);
    log_info(tt::LogOp, "k_data_format: {}", k_df);
    log_info(tt::LogOp, "v_data_format: {}", v_df);
    log_info(tt::LogOp, "mask_data_format: {}", mask_df);
    log_info(tt::LogOp, "out_data_format: {}", out_df);
    log_info(tt::LogOp, "scalar_data_format: {}", scalar_df);
    log_info(tt::LogOp, "intermediate_data_format: {}", im_df);
    log_info(tt::LogOp, "statistics_data_format: {}", stats_df);

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
    if (use_provided_mask or is_causal or use_padded_mask) {
        // attn_mask input
        auto c_in3_config = CircularBufferConfig(mask_tiles * mask_tile_size, {{tt::CBIndex::c_3, mask_df}})
                                .set_page_size(tt::CBIndex::c_3, mask_tile_size);
        CreateCircularBuffer(program, core_grid, c_in3_config);
    }

    // identity scalar input
    auto c_in5_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{tt::CBIndex::c_5, scalar_df}})
                            .set_page_size(tt::CBIndex::c_5, scalar_tile_size);
    CreateCircularBuffer(program, core_grid, c_in5_config);
    // identity column input
    auto c_in7_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{tt::CBIndex::c_7, scalar_df}})
                            .set_page_size(tt::CBIndex::c_7, scalar_tile_size);
    CreateCircularBuffer(program, core_grid, c_in7_config);

    if (is_chunked) {
        auto c_in6_config = CircularBufferConfig(page_table_stick_size, {{tt::CBIndex::c_6, page_table_df}})
                                .set_page_size(tt::CBIndex::c_6, page_table_stick_size);
        CreateCircularBuffer(program, core_grid, c_in6_config);
    }

    // Create attention sink buffer if provided
    if (use_attention_sink) {
        tt::DataFormat sink_df = tt::tt_metal::datatype_to_dataformat_converter(attention_sink.value().dtype());
        uint32_t sink_tile_size = tt::tile_size(sink_df);
        // cb_attention_sink (CBIndex::c_4)
        log_info(tt::LogOp, "attention_sink_tiles: {}", attention_sink_tiles);
        log_info(tt::LogOp, "sink_tile_size: {}", sink_tile_size);
        log_info(tt::LogOp, "sink_df: {}", sink_df);
        auto c_in4_config = CircularBufferConfig(attention_sink_tiles * sink_tile_size, {{tt::CBIndex::c_4, sink_df}})
                                .set_page_size(tt::CBIndex::c_4, sink_tile_size);
        CreateCircularBuffer(program, core_grid, c_in4_config);
    }

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

    uint32_t q_addr = q_buffer->address();
    uint32_t k_addr = k_buffer->address();
    uint32_t v_addr = v_buffer->address();
    uint32_t mask_addr = attn_mask.has_value() ? mask_buffer->address() : 0;
    uint32_t attention_sink_addr = attention_sink.has_value() ? attention_sink_buffer->address() : 0;
    uint32_t out_addr = out0_buffer->address();

    uint32_t num_phases = 1;
    uint32_t read_offset = 0;
    uint32_t write_offset = 0;

    struct CoreHeadWork {
        uint32_t batch = 0;
        uint32_t head = 0;
        uint32_t q_chunk_start = 0;
        uint32_t q_chunk_count = 0;
    };

    struct CoreWork {
        CoreCoord logical_core{};
        CoreCoord physical_core{};
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
        CoreCoord sender_physical = CoreCoord{0, 0};
        CoreCoord receiver_bbox_start = CoreCoord{0, 0};
        CoreCoord receiver_bbox_end = CoreCoord{0, 0};
        uint32_t receiver_count = 0;
    };

    struct ChainRectangle {
        bool initialized = false;
        uint32_t min_x = 0;
        uint32_t max_x = 0;
        uint32_t min_y = 0;
        uint32_t max_y = 0;
        uint32_t count = 0;
    };

    std::vector<CoreWork> core_work(num_cores);
    std::vector<CoreChainInfo> core_chain_info(num_cores);
    const uint32_t total_heads = B * NQH;
    std::vector<std::vector<HeadSegmentRef>> head_segments(total_heads);

    auto try_extend_rectangle = [](ChainRectangle& rect, const CoreCoord& coord) -> bool {
        uint32_t new_min_x = coord.x;
        uint32_t new_max_x = coord.x;
        uint32_t new_min_y = coord.y;
        uint32_t new_max_y = coord.y;
        uint32_t new_count = 1;
        if (rect.initialized) {
            new_min_x = std::min((size_t)rect.min_x, (size_t)coord.x);
            new_max_x = std::max((size_t)rect.max_x, (size_t)coord.x);
            new_min_y = std::min((size_t)rect.min_y, (size_t)coord.y);
            new_max_y = std::max((size_t)rect.max_y, (size_t)coord.y);
            new_count = rect.count + 1;
        }

        const uint32_t width = new_max_x - new_min_x + 1;
        const uint32_t height = new_max_y - new_min_y + 1;
        if ((width * height) != new_count) {
            return false;
        }

        rect.initialized = true;
        rect.min_x = new_min_x;
        rect.max_x = new_max_x;
        rect.min_y = new_min_y;
        rect.max_y = new_max_y;
        rect.count = new_count;
        return true;
    };

    const uint32_t base_chunks_per_core = (num_cores == 0) ? 0 : (total_q_chunks / num_cores);
    const uint32_t extra_chunks = (num_cores == 0) ? 0 : (total_q_chunks % num_cores);
    uint32_t next_global_chunk = 0;

    auto decode_flat_chunk = [&](uint32_t flat_chunk_index) {
        const uint32_t head_span = q_num_chunks;
        const uint32_t head_index = head_span == 0 ? 0 : (flat_chunk_index / head_span);
        const uint32_t q_chunk = head_span == 0 ? 0 : (flat_chunk_index % head_span);
        const uint32_t batch = (NQH == 0) ? 0 : (head_index / NQH);
        const uint32_t head = (NQH == 0) ? 0 : (head_index % NQH);
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
            uint32_t chunk_capacity_in_head = q_num_chunks - q_chunk_idx;
            uint32_t chunk_take = std::min(remaining, chunk_capacity_in_head);

            work.head_work.push_back(CoreHeadWork{
                .batch = batch_idx,
                .head = head_idx,
                .q_chunk_start = q_chunk_idx,
                .q_chunk_count = chunk_take,
            });

            if (!head_segments.empty()) {
                uint32_t head_id = batch_idx * NQH + head_idx;
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

    for (uint32_t head_id = 0; head_id < head_segments.size(); ++head_id) {
        auto& segments = head_segments[head_id];
        if (segments.size() < 2) {
            continue;
        }

        std::size_t idx = 0;
        while (idx < segments.size()) {
            const auto& sender_segment = segments.at(idx);
            auto& sender_work = core_work.at(sender_segment.core_idx);
            if (sender_work.global_q_count == 0 || sender_work.head_work.size() != 1) {
                ++idx;
                continue;
            }

            const auto& sender_head_work = sender_work.head_work.at(sender_segment.head_work_index);
            if (sender_head_work.q_chunk_count == 0) {
                ++idx;
                continue;
            }

            std::vector<std::size_t> receiver_segment_indices;
            ChainRectangle receiver_rectangle{};
            std::size_t lookahead = idx + 1;

            while (lookahead < segments.size()) {
                const auto& candidate_segment = segments.at(lookahead);
                auto& candidate_work = core_work.at(candidate_segment.core_idx);

                if (candidate_work.global_q_count == 0 || candidate_work.head_work.size() != 1) {
                    break;
                }

                const auto& candidate_head_work = candidate_work.head_work.at(candidate_segment.head_work_index);
                if (candidate_head_work.q_chunk_count != sender_head_work.q_chunk_count) {
                    break;
                }

                if (!try_extend_rectangle(receiver_rectangle, candidate_work.physical_core)) {
                    break;
                }

                receiver_segment_indices.push_back(lookahead);
                ++lookahead;
            }

            if (receiver_segment_indices.empty()) {
                ++idx;
                continue;
            }

            auto& sender_chain = core_chain_info.at(sender_segment.core_idx);
            sender_chain.participates = true;
            sender_chain.is_injector = true;
            sender_chain.is_sink = false;
            sender_chain.batch = sender_head_work.batch;
            sender_chain.head = sender_head_work.head;
            sender_chain.q_chunk_start = sender_head_work.q_chunk_start;
            sender_chain.q_chunk_count = sender_head_work.q_chunk_count;
            sender_chain.sender_physical = sender_work.physical_core;
            sender_chain.receiver_bbox_start = {receiver_rectangle.min_x, receiver_rectangle.min_y};
            sender_chain.receiver_bbox_end = {receiver_rectangle.max_x, receiver_rectangle.max_y};
            sender_chain.receiver_count = receiver_rectangle.count;

            for (const auto receiver_segment_index : receiver_segment_indices) {
                const auto& receiver_segment = segments.at(receiver_segment_index);
                auto& receiver_work = core_work.at(receiver_segment.core_idx);
                const auto& receiver_head_work = receiver_work.head_work.at(receiver_segment.head_work_index);

                auto& receiver_chain = core_chain_info.at(receiver_segment.core_idx);
                receiver_chain.participates = true;
                receiver_chain.is_injector = false;
                receiver_chain.is_sink = true;
                receiver_chain.batch = receiver_head_work.batch;
                receiver_chain.head = receiver_head_work.head;
                receiver_chain.q_chunk_start = receiver_head_work.q_chunk_start;
                receiver_chain.q_chunk_count = receiver_head_work.q_chunk_count;
                receiver_chain.sender_physical = sender_work.physical_core;
                receiver_chain.receiver_bbox_start = {receiver_rectangle.min_x, receiver_rectangle.min_y};
                receiver_chain.receiver_bbox_end = {receiver_rectangle.max_x, receiver_rectangle.max_y};
                receiver_chain.receiver_count = receiver_rectangle.count;
            }

            idx = receiver_segment_indices.back() + 1;
        }
    }

    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& work = core_work.at(i);
        log_info(
            tt::LogOp,
            "Core {} logical=({}, {}) physical=({}, {}) global_q=[{}:{}) head_segments={}",
            i,
            work.logical_core.x,
            work.logical_core.y,
            work.physical_core.x,
            work.physical_core.y,
            work.global_q_start,
            work.global_q_start + work.global_q_count,
            work.head_work.size());
        for (const auto& segment : work.head_work) {
            log_info(
                tt::LogOp,
                "  head segment: batch={} head={} q_range=[{}:{})",
                segment.batch,
                segment.head,
                segment.q_chunk_start,
                segment.q_chunk_start + segment.q_chunk_count);
        }
        if (core_chain_info.at(i).participates) {
            const auto& chain = core_chain_info.at(i);
            log_info(
                tt::LogOp,
                "  chain participation: batch={} head={} sender={} receiver={} q_range=[{}:{}) sender_coord=({}, {}) "
                "bbox=[({}, {})-({}, {})] receivers={}",
                chain.batch,
                chain.head,
                chain.is_injector,
                chain.is_sink,
                chain.q_chunk_start,
                chain.q_chunk_start + chain.q_chunk_count,
                chain.sender_physical.x,
                chain.sender_physical.y,
                chain.receiver_bbox_start.x,
                chain.receiver_bbox_start.y,
                chain.receiver_bbox_end.x,
                chain.receiver_bbox_end.y,
                chain.receiver_count);
        }
    }

    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& work = core_work.at(i);
        const auto& chain = core_chain_info.at(i);
        const auto& core = work.logical_core;

        SetRuntimeArgs(
            program,
            reader_kernels_id,
            core,
            {
                q_addr,
                k_addr,
                v_addr,
                mask_addr,
                is_chunked ? page_table.value().buffer()->address() : 0,
                attention_sink_addr,
                i,
                work.global_q_start,
                work.global_q_count,
                num_phases,
                chunked_q_chunk_offset,
                read_offset,
                static_cast<uint32_t>(chain.participates),
                static_cast<uint32_t>(chain.is_injector),
                static_cast<uint32_t>(chain.is_sink),
                chain.batch,
                chain.head,
                chain.q_chunk_start,
                chain.q_chunk_count,
                static_cast<uint32_t>(chain.sender_physical.x),
                static_cast<uint32_t>(chain.sender_physical.y),
                static_cast<uint32_t>(chain.receiver_bbox_start.x),
                static_cast<uint32_t>(chain.receiver_bbox_start.y),
                static_cast<uint32_t>(chain.receiver_bbox_end.x),
                static_cast<uint32_t>(chain.receiver_bbox_end.y),
                chain.receiver_count,
            });

        SetRuntimeArgs(
            program,
            writer_kernels_id,
            core,
            {out_addr, i, work.global_q_start, work.global_q_count, num_phases, chunked_q_chunk_offset, write_offset});

        SetRuntimeArgs(
            program,
            compute_kernels_id,
            core,
            {i, work.global_q_start, work.global_q_count, num_phases, chunked_q_chunk_offset});
    }

    return cached_program_t{
        std::move(program),
        {
            .reader_kernels_id = reader_kernels_id,
            .writer_kernels_id = writer_kernels_id,
            .compute_kernels_id = compute_kernels_id,
            .grid_size = grid_size,
            .num_cores = num_cores,
            .is_chunked = is_chunked,
            .q_chunk_size = q_chunk_size,
            .use_mla = use_mla,
        }};
}

void SDPAProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& shared_vars = cached_program.shared_variables;
    auto& program = cached_program.program;

    const bool is_chunked = operation_attributes.chunk_start_idx.has_value();
    const bool use_mla = operation_attributes.use_mla;
    std::size_t q_chunk_size =
        operation_attributes.program_config ? operation_attributes.program_config->q_chunk_size : 32;

    auto *q_buffer = tensor_args.q.buffer();
    auto *k_buffer = tensor_args.k.buffer();
    auto *v_buffer = use_mla ? tensor_args.k.buffer() : tensor_args.v.value_or(tensor_args.k).buffer();
    auto *mask_buffer = tensor_args.attn_mask.has_value() ? tensor_args.attn_mask->buffer() : nullptr;
    auto *attention_sink_buffer =
        tensor_args.attention_sink.has_value() ? tensor_args.attention_sink->buffer() : nullptr;

    auto *out0_buffer = tensor_return_value.buffer();
    uint32_t q_addr = q_buffer->address();
    uint32_t k_addr = k_buffer->address();
    uint32_t v_addr = v_buffer->address();
    uint32_t mask_addr = mask_buffer != nullptr ? mask_buffer->address() : 0;
    uint32_t attention_sink_addr = attention_sink_buffer != nullptr ? attention_sink_buffer->address() : 0;
    uint32_t out_addr = out0_buffer->address();

    uint32_t page_table_addr = 0;
    uint32_t chunked_q_chunk_offset = 0;
    if (is_chunked) {
        page_table_addr = tensor_args.page_table.value().buffer()->address();
        chunked_q_chunk_offset = operation_attributes.chunk_start_idx.value() / q_chunk_size;
    }

    auto& reader_args_by_core = GetRuntimeArgs(program, shared_vars.reader_kernels_id);
    auto& writer_args_by_core = GetRuntimeArgs(program, shared_vars.writer_kernels_id);
    auto& compute_args_by_core = GetRuntimeArgs(program, shared_vars.compute_kernels_id);

    const auto& grid_size = shared_vars.grid_size;
    const auto num_cores = shared_vars.num_cores;

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        auto& reader_args = reader_args_by_core[core.x][core.y];
        auto& writer_args = writer_args_by_core[core.x][core.y];
        auto& compute_args = compute_args_by_core[core.x][core.y];

        reader_args[0] = q_addr;
        reader_args[1] = k_addr;
        reader_args[2] = v_addr;
        reader_args[3] = mask_addr;
        reader_args[4] = page_table_addr;
        reader_args[5] = attention_sink_addr;
        reader_args[10] = chunked_q_chunk_offset;

        writer_args[0] = out_addr;
        writer_args[5] = chunked_q_chunk_offset;

        compute_args[4] = chunked_q_chunk_offset;
    }
}

}  // namespace ttnn::operations::transformer::sdpa::program
