// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sdpa_program_factory.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/math.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <optional>
#include <string>
#include <cmath>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

// Chain management structures for KV store-and-forward optimization
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
    uint32_t mcast_num_dests = 0;    // num_dests for mcast API (includes self if injector inside rect)
    uint32_t mcast_sender_wait = 0;  // number of actual receivers that signal back (always chain_size - 1)
};

SDPAProgramFactory::cached_program_t SDPAProgramFactory::create(
    const SDPAParams& operation_attributes, const SDPAInputs& tensor_args, Tensor& tensor_return_value) {
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
    attn_mask: B x NQH x S x S  or  B x 1 x S x S
    */

    const auto& q_shape = input_tensor_q.logical_shape();
    const auto& k_shape = input_tensor_k.logical_shape();
    const uint32_t B = q_shape[0], NQH = q_shape[1], Sq = q_shape[2], DH = q_shape[3];
    const uint32_t NKH = k_shape[1];

    // Paged cache parameters when in chunked mode
    const bool flexible_chunked = operation_attributes.chunk_start_idx_tensor.has_value();
    const bool is_chunked_legacy = chunk_start_idx.has_value() && !flexible_chunked;
    const bool is_chunked = is_chunked_legacy || flexible_chunked;
    // For flexible chunked: max prefix length = page_table num_pages * block_size (from K/V layout).
    uint32_t max_prefix_tokens_flexible = 0;
    if (is_chunked && flexible_chunked) {
        const uint32_t block_size_for_sk = k_shape[2];
        const uint32_t max_blocks = page_table.value().padded_shape()[1];
        max_prefix_tokens_flexible = max_blocks * block_size_for_sk;
    }
    // In chunked mode: legacy uses chunk_start_idx + Sq; flexible uses Sq + max prefix from page table.
    const uint32_t Sk = is_chunked
                            ? (flexible_chunked ? (Sq + max_prefix_tokens_flexible) : (chunk_start_idx.value() + Sq))
                            : k_shape[2];

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
    For non-causal case with Q/K padding:
    - If user provides a mask: reader reads unpadded mask and fills padded K positions with -inf
    - If no mask provided: writer generates a mask with 0 for valid K and -inf for padded K
    In causal case, the causal mask naturally handles masking of padded K tokens.
    */
    const bool use_padded_mask = (!is_causal) && ((padded_Sk != Sk) || (padded_Sq != Sq));

    const uint32_t Sq_chunk_t = q_chunk_size / TILE_HEIGHT;
    const uint32_t Sk_chunk_t = k_chunk_size / TILE_HEIGHT;
    const uint32_t q_num_chunks = padded_Sq / q_chunk_size;
    const uint32_t k_num_chunks = padded_Sk / k_chunk_size;
    const bool use_provided_mask = attn_mask.has_value();
    const bool broadcast_provided_mask_batch = use_provided_mask ? (attn_mask.value().logical_shape()[0] == 1) : false;
    const bool broadcast_provided_mask_heads = use_provided_mask ? (attn_mask.value().logical_shape()[1] == 1) : false;

    // log_debug all of the above
    log_debug(tt::LogOp, "B: {}", B);
    log_debug(tt::LogOp, "NQH: {}", NQH);

    log_debug(tt::LogOp, "Sq: {}", Sq);
    log_debug(tt::LogOp, "Sk: {}", Sk);
    log_debug(tt::LogOp, "padded_Sq: {}", padded_Sq);
    log_debug(tt::LogOp, "padded_Sk: {}", padded_Sk);
    log_debug(tt::LogOp, "valid_Sqt: {}", valid_Sqt);
    log_debug(tt::LogOp, "valid_Skt: {}", valid_Skt);
    log_debug(tt::LogOp, "DH: {}", DH);
    log_debug(tt::LogOp, "Sqt: {}", Sqt);
    log_debug(tt::LogOp, "Skt: {}", Skt);
    log_debug(tt::LogOp, "DHt: {}", DHt);
    log_debug(tt::LogOp, "vDHt: {}", vDHt);
    log_debug(tt::LogOp, "Sq_chunk_t: {}", Sq_chunk_t);
    log_debug(tt::LogOp, "Sk_chunk_t: {}", Sk_chunk_t);
    log_debug(tt::LogOp, "q_chunk_size: {}", q_chunk_size);
    log_debug(tt::LogOp, "k_chunk_size: {}", k_chunk_size);
    log_debug(tt::LogOp, "q_num_chunks: {}", q_num_chunks);
    log_debug(tt::LogOp, "k_num_chunks: {}", k_num_chunks);
    log_debug(tt::LogOp, "NKH: {}", NKH);
    log_debug(tt::LogOp, "sliding_window_size: {}", sliding_window_size.has_value() ? sliding_window_size.value() : 0);

    // In chunked prefill mode, the offset of Q in terms of Q chunks
    uint32_t chunked_q_chunk_offset = 0;
    uint32_t block_size = 0;
    uint32_t block_size_t = 0;
    [[maybe_unused]] uint32_t max_blocks_per_seq = 0;
    uint32_t page_table_stick_size = 0;
    tt::DataFormat page_table_df = tt::DataFormat::Int32;

    if (is_chunked) {
        if (is_chunked_legacy) {
            // chunk_start_idx must be a multiple of q_chunk_size (validated in sdpa_device_operation.cpp)
            chunked_q_chunk_offset = chunk_start_idx.value() / q_chunk_size;
        }
        // else: flexible_chunked - chunked_q_chunk_offset set inside of the op
        const auto& page_table_tensor = page_table.value();
        block_size = k_shape[2];  // K's sequence dimension represents block size
        block_size_t = block_size / TILE_HEIGHT;
        if (flexible_chunked) {
            max_blocks_per_seq = page_table_tensor.padded_shape()[1];
            page_table_stick_size = max_blocks_per_seq * sizeof(int32_t);
            TT_FATAL(page_table_stick_size % 32 == 0, "page table stick size must be a multiple of 32");
        } else {
            max_blocks_per_seq = page_table_tensor.padded_shape()[1];
            page_table_stick_size = page_table_tensor.buffer()->aligned_page_size();
            TT_FATAL(
                page_table_stick_size % 32 == 0,
                "page table page size in bytes must be a multiple of 32 due to address alignment");
        }
    }
    // Log page table info
    log_debug(tt::LogOp, "is_chunked: {}", is_chunked);
    if (is_chunked) {
        log_debug(tt::LogOp, "block_size: {}", block_size);
        log_debug(tt::LogOp, "block_size_t: {}", block_size_t);
        log_debug(tt::LogOp, "max_blocks_per_seq: {}", max_blocks_per_seq);
        log_debug(tt::LogOp, "page_table_stick_size: {}", page_table_stick_size);
        log_debug(tt::LogOp, "page_table_df: {}", page_table_df);
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

    // Parallelization scheme
    // We will choose parallelization factors for batch, num_heads, and q_seq_len in that order
    uint32_t batch_parallel_factor = std::min(B, num_cores);
    uint32_t nh_parallel_factor = std::min(num_cores / batch_parallel_factor, NQH);
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
    const uint32_t batch_per_core = (B + batch_parallel_factor - 1) / batch_parallel_factor;
    const uint32_t nh_per_core = (NQH + nh_parallel_factor - 1) / nh_parallel_factor;
    const uint32_t q_per_core = (q_num_chunks + q_parallel_factor - 1) / q_parallel_factor;

    const uint32_t q_buffer_factor = (q_per_core > 1) ? 2 : 1;

    log_debug(tt::LogOp, "q_per_core: {}", q_per_core);

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
    log_debug(tt::LogOp, "q_tiles: {}", q_tiles);
    log_debug(tt::LogOp, "k_tiles: {}", k_tiles);
    log_debug(tt::LogOp, "v_tiles: {}", v_tiles);
    log_debug(tt::LogOp, "mask_tiles: {}", mask_tiles);
    log_debug(tt::LogOp, "qk_tiles: {}", qk_tiles);
    log_debug(tt::LogOp, "out0_t: {}", out0_t);
    log_debug(tt::LogOp, "scale_tiles: {}", scale_tiles);
    log_debug(tt::LogOp, "statistics_tiles: {}", statistics_tiles);
    log_debug(tt::LogOp, "attention_sink_tiles: {}", attention_sink_tiles);

    // Host code is responsible for determining matmul configuration
    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
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
    // DHT_GRANULARITY is used in the kernel with both DHt and vDHt as the cols parameter,
    // so the granularity must evenly divide both to avoid dropping tiles.
    uint32_t dht_granularity = std::min({DHt, vDHt, dst_size});
    while (dht_granularity > 1 && (DHt % dht_granularity != 0 || vDHt % dht_granularity != 0)) {
        dht_granularity--;
    }
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
                                                      (std::uint32_t)broadcast_provided_mask_batch,
                                                      (std::uint32_t)broadcast_provided_mask_heads,
                                                      (std::uint32_t)use_padded_mask,
                                                      (uint32_t)is_chunked,
                                                      block_size_t,
                                                      page_table_stick_size,
                                                      (std::uint32_t)use_attention_sink,
                                                      qk_out_subblock_h};

    // Placeholder semaphore IDs for KV chain forwarding (will be filled later if enabled)
    // Add these BEFORE TensorAccessorArgs to keep indexing consistent with kernel expectations
    const auto sem_args_offset = reader_compile_time_args.size();
    reader_compile_time_args.push_back(0);  // sender_semaphore_id placeholder
    reader_compile_time_args.push_back(0);  // receiver_semaphore_id placeholder
    reader_compile_time_args.push_back(0);  // valid_semaphore_id placeholder
    reader_compile_time_args.push_back(0);  // mcast_enabled placeholder

    TensorAccessorArgs(input_tensor_q.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(attn_mask.has_value() ? attn_mask->buffer() : nullptr).append_to(reader_compile_time_args);
    TensorAccessorArgs(page_table.has_value() ? page_table->buffer() : nullptr).append_to(reader_compile_time_args);
    TensorAccessorArgs(attention_sink.has_value() ? attention_sink->buffer() : nullptr)
        .append_to(reader_compile_time_args);
    TensorAccessorArgs(flexible_chunked ? operation_attributes.chunk_start_idx_tensor.value().buffer() : nullptr)
        .append_to(reader_compile_time_args);

    // Create semaphores for KV chain forwarding BEFORE kernel compilation (non-causal only)
    // This must happen before CreateKernel so the actual semaphore IDs are in the compile-time args
    uint32_t sender_semaphore_id = 0;
    uint32_t receiver_semaphore_id = 0;
    uint32_t valid_semaphore_id = 0;

    if (!is_causal) {
        sender_semaphore_id = CreateSemaphore(program, core_grid, INVALID);
        receiver_semaphore_id = CreateSemaphore(program, core_grid, INVALID);
        valid_semaphore_id = CreateSemaphore(program, core_grid, VALID);

        // Update the placeholder compile-time args with actual semaphore IDs
        reader_compile_time_args[sem_args_offset + 0] = sender_semaphore_id;
        reader_compile_time_args[sem_args_offset + 1] = receiver_semaphore_id;
        reader_compile_time_args[sem_args_offset + 2] = valid_semaphore_id;

        log_debug(
            tt::LogOp,
            "KV chain forwarding enabled - created semaphores: sender={}, receiver={}, valid={}",
            sender_semaphore_id,
            receiver_semaphore_id,
            valid_semaphore_id);
    }

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
    defines["SUB_EXP_GRANULARITY"] = std::to_string(sub_exp_granularity);
    defines["MUL_BCAST_GRANULARITY"] = std::to_string(mul_bcast_granularity);
    defines["DHT_GRANULARITY"] = std::to_string(dht_granularity);
    defines["REDUCE_GRANULARITY"] = std::to_string(reduce_granularity);
    defines["EXP_APPROX_MODE"] = std::to_string(exp_approx_mode);
    uint32_t balanced_q_parallel =
        (is_causal && (q_per_core * q_parallel_factor == q_num_chunks) && (q_per_core % 2 == 0));
    if (balanced_q_parallel) {
        defines["BALANCED_Q_PARALLEL"] = "1";
    }

    log_debug(tt::LogOp, "BALANCED_Q_PARALLEL: {}", balanced_q_parallel);

    // NOTE: CreateKernel calls are deferred until after chain construction so that
    // the mcast_enabled compile-time arg can be determined first.

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
    if (flexible_chunked) {
        constexpr uint32_t chunk_start_idx_page_size = 32;
        auto c_chunk_start_compute_config =
            CircularBufferConfig(chunk_start_idx_page_size, {{tt::CBIndex::c_8, tt::DataFormat::Int32}})
                .set_page_size(tt::CBIndex::c_8, chunk_start_idx_page_size);
        CreateCircularBuffer(program, core_grid, c_chunk_start_compute_config);
        auto c_chunk_start_writer_config =
            CircularBufferConfig(chunk_start_idx_page_size, {{tt::CBIndex::c_9, tt::DataFormat::Int32}})
                .set_page_size(tt::CBIndex::c_9, chunk_start_idx_page_size);
        CreateCircularBuffer(program, core_grid, c_chunk_start_writer_config);
    }

    // Create attention sink buffer if provided
    if (use_attention_sink) {
        tt::DataFormat sink_df = tt::tt_metal::datatype_to_dataformat_converter(attention_sink.value().dtype());
        uint32_t sink_tile_size = tt::tile_size(sink_df);
        // cb_attention_sink (CBIndex::c_4)
        log_debug(tt::LogOp, "attention_sink_tiles: {}", attention_sink_tiles);
        log_debug(tt::LogOp, "sink_tile_size: {}", sink_tile_size);
        log_debug(tt::LogOp, "sink_df: {}", sink_df);
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

    // Note: Semaphores for KV chain forwarding are now created earlier (before kernel compilation)
    // to ensure the actual semaphore IDs are available in the compile-time args

    uint32_t q_addr = q_buffer->address();
    uint32_t k_addr = k_buffer->address();
    uint32_t v_addr = v_buffer->address();
    uint32_t mask_addr = attn_mask.has_value() ? mask_buffer->address() : 0;
    uint32_t attention_sink_addr = attention_sink.has_value() ? attention_sink_buffer->address() : 0;
    uint32_t out_addr = out0_buffer->address();

    uint32_t num_phases = 1;
    uint32_t read_offset = 0;
    uint32_t write_offset = 0;

    // Build chain topology for KV forwarding (non-causal only)
    std::vector<CoreWork> core_work(num_cores);
    std::vector<CoreChainInfo> core_chain_info(num_cores);
    const uint32_t total_heads = B * NQH;
    std::vector<std::vector<HeadSegmentRef>> head_segments;
    uint32_t mcast_chains = 0;

    if (!is_causal && !is_chunked) {
        head_segments.resize(total_heads);

        log_debug(tt::LogOp, "=== Building KV chain forwarding topology ===");
        log_debug(tt::LogOp, "Total heads (B * NQH): {}", total_heads);
        log_debug(tt::LogOp, "Q chunks per head: {}", q_num_chunks);
        log_debug(tt::LogOp, "Grid size: {}x{} = {} cores", grid_size.x, grid_size.y, num_cores);

        // First pass: Record work distribution for each core
        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord core = {i % grid_size.x, i / grid_size.x};

            uint32_t local_batch_start = (i / (nh_parallel_factor * q_parallel_factor)) * batch_per_core;
            uint32_t local_batch_end = local_batch_start + batch_per_core;
            uint32_t local_nh_start = ((i / q_parallel_factor) % nh_parallel_factor) * nh_per_core;
            uint32_t local_nh_end = local_nh_start + nh_per_core;
            uint32_t local_q_start = (i % q_parallel_factor) * q_per_core;
            uint32_t local_q_end = local_q_start + q_per_core;

            // Clamp to max values
            local_batch_start = std::min(local_batch_start, B);
            local_batch_end = std::min(local_batch_end, B);
            local_nh_start = std::min(local_nh_start, NQH);
            local_nh_end = std::min(local_nh_end, NQH);
            local_q_start = std::min(local_q_start, q_num_chunks);
            local_q_end = std::min(local_q_end, q_num_chunks);

            auto& work = core_work[i];
            work.logical_core = core;
            work.physical_core = device->worker_core_from_logical_core(core);

            // Track each (batch, head, q_chunk_range) this core handles
            for (uint32_t b = local_batch_start; b < local_batch_end; ++b) {
                for (uint32_t h = local_nh_start; h < local_nh_end; ++h) {
                    uint32_t q_count = local_q_end - local_q_start;
                    if (q_count > 0) {
                        work.head_work.push_back(CoreHeadWork{
                            .batch = b,
                            .head = h,
                            .q_chunk_start = local_q_start,
                            .q_chunk_count = q_count,
                        });

                        uint32_t head_id = (b * NQH) + h;
                        if (head_id < head_segments.size()) {
                            head_segments[head_id].push_back(HeadSegmentRef{
                                .core_idx = i, .head_work_index = static_cast<uint32_t>(work.head_work.size() - 1)});
                        }
                    }
                }
            }

            if (!work.head_work.empty()) {
                log_debug(
                    tt::LogOp, "Core {} ({}): handles {} head segments", i, work.physical_core, work.head_work.size());
            }
        }

        // Second pass: Build chains for heads spanning multiple cores
        uint32_t chains_built = 0;
        uint32_t chains_skipped = 0;

        for (uint32_t head_id = 0; head_id < head_segments.size(); ++head_id) {
            auto& segments = head_segments[head_id];
            if (segments.size() < 2) {
                continue;  // No chain needed for single core
            }

            // Find chain start (injector), rotating preferred position per head to
            // spread injectors across different physical columns for DRAM BW.
            const std::size_t preferred_start = head_id % segments.size();
            std::optional<std::size_t> chain_start_idx;

            // First pass: prefer cores handling only one head segment
            for (std::size_t offset = 0; offset < segments.size(); ++offset) {
                std::size_t idx = (preferred_start + offset) % segments.size();
                const auto& seg = segments[idx];
                const auto& work = core_work[seg.core_idx];

                if (seg.head_work_index >= work.head_work.size()) {
                    continue;
                }
                if (core_chain_info[seg.core_idx].participates) {
                    continue;
                }
                if (work.head_work.size() == 1) {
                    chain_start_idx = idx;
                    break;
                }
            }

            // If no single-segment core found, try any core not in a chain
            if (!chain_start_idx.has_value()) {
                for (std::size_t offset = 0; offset < segments.size(); ++offset) {
                    std::size_t idx = (preferred_start + offset) % segments.size();
                    const auto& seg = segments[idx];
                    if (!core_chain_info[seg.core_idx].participates) {
                        chain_start_idx = idx;
                        break;
                    }
                }
            }

            if (!chain_start_idx.has_value()) {
                chains_skipped++;
                log_debug(
                    tt::LogOp,
                    "Head {} spans {} cores but no valid chain start found (conflicts)",
                    head_id,
                    segments.size());
                continue;  // Can't build chain for this head
            }

            // Build the chain from start to end
            const std::size_t start = chain_start_idx.value();
            uint32_t batch = segments[start].core_idx < core_work.size()
                                 ? core_work[segments[start].core_idx].head_work[segments[start].head_work_index].batch
                                 : 0;
            uint32_t head = head_id % NQH;

            log_debug(
                tt::LogOp,
                "Building chain for head {} (batch={}, head={}): {} segments starting at idx {}",
                head_id,
                batch,
                head,
                segments.size(),
                start);

            // Build chain in wrap order: start, start+1, ..., N-1, 0, 1, ..., start-1
            // Exclude segments with different q_chunk_count than the injector (uneven tail).
            std::vector<std::size_t> chain_order;
            const auto& start_seg = segments[start];
            const uint32_t ref_q_count =
                core_work[start_seg.core_idx].head_work[start_seg.head_work_index].q_chunk_count;
            for (std::size_t step = 0; step < segments.size(); ++step) {
                std::size_t idx = (start + step) % segments.size();
                const auto& seg = segments[idx];
                const uint32_t core_idx = seg.core_idx;

                if (core_idx >= core_work.size() || seg.head_work_index >= core_work[core_idx].head_work.size()) {
                    continue;
                }

                if (core_chain_info[core_idx].participates) {
                    log_debug(
                        tt::LogOp,
                        "WARNING: Core {} already participates in chain (batch={}, head={}), skipping rest",
                        core_idx,
                        core_chain_info[core_idx].batch,
                        core_chain_info[core_idx].head);
                    break;
                }

                // Skip cores with different q_chunk_count (e.g. uneven tail)
                const auto& hw = core_work[core_idx].head_work[seg.head_work_index];
                if (hw.q_chunk_count != ref_q_count) {
                    continue;
                }

                chain_order.push_back(idx);
            }

            for (std::size_t pos = 0; pos < chain_order.size(); ++pos) {
                const std::size_t idx = chain_order[pos];
                const auto& seg = segments[idx];
                const uint32_t core_idx = seg.core_idx;
                const auto& hw = core_work[core_idx].head_work[seg.head_work_index];
                auto& chain = core_chain_info[core_idx];

                chain.participates = true;
                chain.batch = hw.batch;
                chain.head = hw.head;
                chain.q_chunk_start = hw.q_chunk_start;
                chain.q_chunk_count = hw.q_chunk_count;

                if (pos == 0) {
                    chain.is_injector = true;
                }
                if (pos == chain_order.size() - 1) {
                    chain.is_sink = true;
                }

                // Set prev core coordinates (previous in wrap order)
                if (pos > 0) {
                    const uint32_t prev_core_idx = segments[chain_order[pos - 1]].core_idx;
                    if (prev_core_idx < core_work.size()) {
                        chain.prev_physical = core_work[prev_core_idx].physical_core;
                    }
                }

                // Set next core coordinates and q_chunk count (next in wrap order)
                if (pos + 1 < chain_order.size()) {
                    const std::size_t next_idx = chain_order[pos + 1];
                    const uint32_t next_core_idx = segments[next_idx].core_idx;
                    if (next_core_idx < core_work.size() &&
                        segments[next_idx].head_work_index < core_work[next_core_idx].head_work.size()) {
                        chain.next_physical = core_work[next_core_idx].physical_core;
                        const auto& next_hw = core_work[next_core_idx].head_work[segments[next_idx].head_work_index];
                        chain.next_core_q_chunks = next_hw.q_chunk_count;
                    }
                }

                log_debug(
                    tt::LogOp,
                    "  Core {} in chain: injector={}, sink={}, q_chunks={}, prev={}, next={}",
                    core_idx,
                    chain.is_injector,
                    chain.is_sink,
                    chain.q_chunk_count,
                    chain.prev_physical,
                    chain.next_physical);
            }

            chains_built++;
        }

        log_debug(
            tt::LogOp,
            "Chain construction complete: {} chains built, {} skipped due to conflicts",
            chains_built,
            chains_skipped);

        // Third pass: Check multicast eligibility — all-or-nothing policy.
        // First, check if ALL multi-core chains are eligible. Only if every chain
        // qualifies do we configure mcast (compile-time decision for the kernel).
        struct McastCandidate {
            std::vector<uint32_t> core_indices;
            uint32_t ref_q_chunks;
        };
        std::vector<McastCandidate> candidates;
        bool all_eligible = true;
        uint32_t total_multi_core_chains = 0;

        for (uint32_t head_id = 0; head_id < head_segments.size(); ++head_id) {
            auto& segments = head_segments[head_id];
            if (segments.size() < 2) {
                continue;
            }

            // Collect chain core indices that actually participate in this head's chain
            std::vector<uint32_t> chain_core_indices;
            for (const auto& seg : segments) {
                if (seg.core_idx < core_chain_info.size() && core_chain_info[seg.core_idx].participates &&
                    core_chain_info[seg.core_idx].batch == (head_id / NQH) &&
                    core_chain_info[seg.core_idx].head == (head_id % NQH)) {
                    chain_core_indices.push_back(seg.core_idx);
                }
            }

            if (chain_core_indices.size() < 2) {
                continue;
            }

            total_multi_core_chains++;

            // Check eligibility condition 1: All physical cores share the same Y coordinate
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

            // Note: Physical X contiguity is NOT required. Harvested (non-worker) cores
            // in the multicast rectangle safely discard the data.

            // Check eligibility condition 2: All chain cores have equal q_chunk_count
            const uint32_t ref_q_chunks = core_chain_info[chain_core_indices[0]].q_chunk_count;
            bool equal_q_chunks = true;
            for (size_t ci = 1; ci < chain_core_indices.size(); ++ci) {
                if (core_chain_info[chain_core_indices[ci]].q_chunk_count != ref_q_chunks) {
                    equal_q_chunks = false;
                    break;
                }
            }

            if (!equal_q_chunks) {
                all_eligible = false;
                log_debug(tt::LogOp, "Head {}: mcast ineligible - unequal q_chunk_count across chain", head_id);
                break;
            }

            candidates.push_back(McastCandidate{std::move(chain_core_indices), ref_q_chunks});
        }

        // Only configure mcast if ALL multi-core chains are eligible (all-or-nothing)
        if (all_eligible && !candidates.empty()) {
            mcast_chains = candidates.size();
            for (const auto& cand : candidates) {
                const uint32_t chain_size = cand.core_indices.size();
                const uint32_t num_receivers = chain_size - 1;

                // Find the injector (may not be at index 0 due to rotation)
                uint32_t injector_idx = cand.core_indices[0];
                for (const auto& ci : cand.core_indices) {
                    if (core_chain_info[ci].is_injector) {
                        injector_idx = ci;
                        break;
                    }
                }

                // Mcast rect covers the full row (min to max physical X across all chain cores).
                // The mcast API excludes the source from destinations automatically.
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

                // When the injector is geometrically inside the mcast rect (not at min or max X),
                // the hardware counts it as a destination slot, so num_dests must include it.
                const uint32_t injector_x = core_work[injector_idx].physical_core.x;
                const bool injector_inside_rect = (injector_x > min_x && injector_x < max_x);
                const uint32_t mcast_num_dests = injector_inside_rect ? chain_size : num_receivers;

                // Configure injector
                auto& injector_chain = core_chain_info[injector_idx];
                injector_chain.use_mcast = true;
                injector_chain.prev_physical = rect_start;  // mcast rect start
                injector_chain.next_physical = rect_end;    // mcast rect end
                injector_chain.mcast_num_dests = mcast_num_dests;
                injector_chain.mcast_sender_wait = num_receivers;
                injector_chain.next_core_q_chunks = cand.ref_q_chunks;

                // Configure receivers (all non-injector cores)
                for (const auto& ci : cand.core_indices) {
                    if (ci == injector_idx) {
                        continue;
                    }
                    auto& receiver_chain = core_chain_info[ci];
                    receiver_chain.use_mcast = true;
                    receiver_chain.prev_physical = core_work[injector_idx].physical_core;
                    receiver_chain.is_sink = true;
                }

                log_debug(
                    tt::LogOp,
                    "Head: mcast enabled - {} receivers, injector core {} (phys_x={}), num_dests={} -> rect ({},{}) to "
                    "({},{})",
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

        log_info(
            tt::LogOp,
            "Multicast eligibility: {}/{} chains using mcast (all-or-nothing)",
            mcast_chains,
            total_multi_core_chains);
    }

    // Update mcast_enabled compile-time arg now that chain construction is complete
    reader_compile_time_args[sem_args_offset + 3] = (mcast_chains > 0) ? 1 : 0;

    // Create kernels (deferred until after chain construction for mcast_enabled flag)
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

    // Set reader rt args
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        // log_debug(tt::LogOp, "core: {} getting runtime args for idx {i}", core, i);
        uint32_t local_batch_start = (i / (nh_parallel_factor * q_parallel_factor)) * batch_per_core;
        uint32_t local_batch_end = local_batch_start + batch_per_core;
        uint32_t local_nh_start = ((i / q_parallel_factor) % nh_parallel_factor) * nh_per_core;
        uint32_t local_nh_end = local_nh_start + nh_per_core;
        uint32_t local_q_start = (i % q_parallel_factor) * q_per_core;
        uint32_t local_q_end = local_q_start + q_per_core;

        // clamp all to max values for non-even partitioning
        local_batch_start = std::min(local_batch_start, B);
        local_batch_end = std::min(local_batch_end, B);
        local_nh_start = std::min(local_nh_start, NQH);
        local_nh_end = std::min(local_nh_end, NQH);
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

        // Get chain info for this core
        const auto& chain = core_chain_info[i];

        std::vector<uint32_t> reader_args = {
            q_addr,
            k_addr,
            v_addr,
            mask_addr,
            is_chunked ? page_table.value().buffer()->address() : 0,
            attention_sink_addr,
            flexible_chunked ? operation_attributes.chunk_start_idx_tensor.value().buffer()->address() : 0,
            i,
            local_batch_start,
            local_batch_end,
            local_nh_start,
            local_nh_end,
            local_q_start,
            local_q_end,
            num_phases,
            chunked_q_chunk_offset,
            read_offset  // read_offset
        };

        // Add chain metadata for non-causal case
        if (!is_causal) {
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
        }

        SetRuntimeArgs(program, reader_kernels_id, core, reader_args);
        SetRuntimeArgs(
            program,
            writer_kernels_id,
            core,
            {out_addr,
             i,
             local_batch_start,
             local_batch_end,
             local_nh_start,
             local_nh_end,
             local_q_start,
             local_q_end,
             num_phases,
             static_cast<uint32_t>(flexible_chunked ? 1 : 0),
             chunked_q_chunk_offset,
             write_offset});  // write_offset
        SetRuntimeArgs(
            program,
            compute_kernels_id,
            core,
            {i,
             local_batch_start,
             local_batch_end,
             local_nh_start,
             local_nh_end,
             local_q_start,
             local_q_end,
             num_phases,
             static_cast<uint32_t>(flexible_chunked ? 1 : 0),
             chunked_q_chunk_offset});
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
    const SDPAParams& operation_attributes,
    const SDPAInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& shared_vars = cached_program.shared_variables;
    auto& program = cached_program.program;

    const bool flexible_chunked = operation_attributes.chunk_start_idx_tensor.has_value();
    const bool is_chunked = operation_attributes.chunk_start_idx.has_value() || flexible_chunked;
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
    uint32_t chunk_start_idx_addr = 0;
    const uint32_t use_chunk_start_idx_tensor = flexible_chunked ? 1 : 0;
    if (is_chunked) {
        page_table_addr = tensor_args.page_table.value().buffer()->address();
        if (!flexible_chunked) {
            // chunk_start_idx must be a multiple of q_chunk_size (validated in sdpa_device_operation.cpp)
            chunked_q_chunk_offset = operation_attributes.chunk_start_idx.value() / q_chunk_size;
        } else {
            chunk_start_idx_addr = operation_attributes.chunk_start_idx_tensor.value().buffer()->address();
        }
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
        reader_args[6] = chunk_start_idx_addr;
        reader_args[15] = chunked_q_chunk_offset;

        writer_args[0] = out_addr;
        writer_args[9] = use_chunk_start_idx_tensor;
        writer_args[10] = chunked_q_chunk_offset;

        compute_args[8] = use_chunk_start_idx_tensor;
        compute_args[9] = chunked_q_chunk_offset;
    }
}

}  // namespace ttnn::prim
