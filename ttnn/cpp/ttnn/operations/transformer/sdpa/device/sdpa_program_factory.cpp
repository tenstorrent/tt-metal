// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_program_factory.hpp"
#include "sdpa_op.hpp"

#include <optional>
#include <cmath>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::transformer::detail {

// implementation of softmax with optional scale/mask (see the header for input_tensor more detailed description)
operation::ProgramWithCallbacks sdpa_multi_core(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const Tensor& input_tensor_v,
    const Tensor& output_tensor,
    const std::optional<const Tensor>& attn_mask,
    const std::optional<const Tensor>& page_table,
    const std::optional<int64_t>& chunk_start_idx,
    std::optional<float> scale,
    bool is_causal,
    std::size_t q_chunk_size,
    std::size_t k_chunk_size,
    DeviceComputeKernelConfig compute_kernel_config,
    std::optional<SDPAProgramConfig> program_config) {
    /*
    Q: B x NQH x S x DH
    K: B x NKH x DH x S
    V: B x NKH x S x DH
    attn_mask: B x NQH x S x S
    */

    const auto q_shape = input_tensor_q.logical_shape();
    const auto k_shape = input_tensor_k.logical_shape();
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
    log_debug(tt::LogOp, "Sq_chunk_t: {}", Sq_chunk_t);
    log_debug(tt::LogOp, "Sk_chunk_t: {}", Sk_chunk_t);
    log_debug(tt::LogOp, "q_chunk_size: {}", q_chunk_size);
    log_debug(tt::LogOp, "k_chunk_size: {}", k_chunk_size);
    log_debug(tt::LogOp, "q_num_chunks: {}", q_num_chunks);
    log_debug(tt::LogOp, "k_num_chunks: {}", k_num_chunks);
    log_debug(tt::LogOp, "NKH: {}", NKH);

    // In chunked prefill mode, the offset of Q in terms of Q chunks
    uint32_t chunked_q_chunk_offset = 0;
    uint32_t block_size = 0;
    uint32_t block_size_t = 0;
    uint32_t max_blocks_per_seq = 0;
    uint32_t page_table_stick_size = 0;
    bool page_table_is_dram = true;
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
        page_table_is_dram = page_table_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;

        TT_FATAL(
            page_table_stick_size % 32 == 0,
            "page table page size in bytes must be a multiple of 32 due to address alignment");
    }
    // Log page table info
    log_debug(tt::LogOp, "is_chunked: {}", is_chunked);
    if (is_chunked) {
        log_debug(tt::LogOp, "block_size: {}", block_size);
        log_debug(tt::LogOp, "block_size_t: {}", block_size_t);
        log_debug(tt::LogOp, "max_blocks_per_seq: {}", max_blocks_per_seq);
        log_debug(tt::LogOp, "page_table_stick_size: {}", page_table_stick_size);
        log_debug(tt::LogOp, "page_table_is_dram: {}", page_table_is_dram);
        log_debug(tt::LogOp, "page_table_df: {}", page_table_df);
    }

    Program program = CreateProgram();

    IDevice* device = input_tensor_q.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    auto q_buffer = input_tensor_q.buffer();
    auto k_buffer = input_tensor_k.buffer();
    auto v_buffer = input_tensor_v.buffer();
    auto mask_buffer = attn_mask.has_value() ? attn_mask.value().buffer() : nullptr;

    auto out0_buffer = output_tensor.buffer();

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
    uint32_t v_tiles = Sk_chunk_t * DHt * 2;            // double buffer
    uint32_t mask_tiles = Sq_chunk_t * Sk_chunk_t * 2;  // double buffer
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

    // Log these
    log_debug(tt::LogOp, "stats_granularity: {}", stats_granularity);
    log_debug(tt::LogOp, "log2_stats_granularity: {}", log2_stats_granularity);
    log_debug(tt::LogOp, "sub_exp_granularity: {}", sub_exp_granularity);
    log_debug(tt::LogOp, "log2_sub_exp_granularity: {}", log2_sub_exp_granularity);
    log_debug(tt::LogOp, "mul_bcast_granularity: {}", mul_bcast_granularity);
    log_debug(tt::LogOp, "log2_mul_bcast_granularity: {}", log2_mul_bcast_granularity);
    log_debug(tt::LogOp, "dht_granularity: {}", dht_granularity);
    log_debug(tt::LogOp, "log2_dht_granularity: {}", log2_dht_granularity);

    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    class bfloat16 bfloat_identity_scalar(1.0f);
    uint32_t packed_identity_scalar = pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});

    union {
        float f;
        uint32_t u;
    } scale_union;
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
                                                      Sq_chunk_t,
                                                      q_num_chunks,
                                                      Sk_chunk_t,
                                                      k_num_chunks,
                                                      num_cores,
                                                      (std::uint32_t)is_causal,
                                                      (std::uint32_t)use_provided_mask,
                                                      (std::uint32_t)use_padded_mask,
                                                      (uint32_t)is_chunked,
                                                      (uint32_t)page_table_is_dram,
                                                      block_size_t,
                                                      page_table_stick_size};

    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        B,
        NQH,
        NKH,
        Sqt,
        valid_Sqt,
        Sk,
        DHt,
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
    };

    std::vector<uint32_t> compute_compile_time_args = {
        // matmul args
        B,
        NQH,
        NKH,
        Skt,
        DHt,
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
    };

    std::map<string, string> defines;
    defines["STATS_GRANULARITY"] = std::to_string(stats_granularity);
    defines["LOG2_STATS_GRANULARITY"] = std::to_string(log2_stats_granularity);
    defines["SUB_EXP_GRANULARITY"] = std::to_string(sub_exp_granularity);
    defines["LOG2_SUB_EXP_GRANULARITY"] = std::to_string(log2_sub_exp_granularity);
    defines["MUL_BCAST_GRANULARITY"] = std::to_string(mul_bcast_granularity);
    defines["LOG2_MUL_BCAST_GRANULARITY"] = std::to_string(log2_mul_bcast_granularity);
    defines["DHT_GRANULARITY"] = std::to_string(dht_granularity);
    defines["LOG2_DHT_GRANULARITY"] = std::to_string(log2_dht_granularity);
    defines["EXP_APPROX_MODE"] = std::to_string(exp_approx_mode);
    uint32_t balanced_q_parallel =
        (is_causal && (q_per_core * q_parallel_factor == q_num_chunks) && (q_per_core % 2 == 0));
    if (balanced_q_parallel) {
        defines["BALANCED_Q_PARALLEL"] = "1";
    }

    log_debug(tt::LogOp, "BALANCED_Q_PARALLEL: {}", balanced_q_parallel);

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
    tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_v.dtype());
    tt::DataFormat mask_df = attn_mask.has_value()
                                 ? tt::tt_metal::datatype_to_dataformat_converter(attn_mask.value().dtype())
                                 : tt::DataFormat::Bfp4_b;
    tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::DataFormat scalar_df = tt::DataFormat::Float16_b;
    tt::DataFormat im_df = tt::DataFormat::Float16_b;  // need to disable fp32 cbs (Issue #13364) fp32_dest_acc_en ?
                                                       // tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat stats_df = im_df;

    uint32_t q_tile_size = tt::tt_metal::detail::TileSize(q_df);
    uint32_t k_tile_size = tt::tt_metal::detail::TileSize(k_df);
    uint32_t v_tile_size = tt::tt_metal::detail::TileSize(v_df);
    uint32_t mask_tile_size = tt::tt_metal::detail::TileSize(mask_df);
    uint32_t out_tile_size = tt::tt_metal::detail::TileSize(out_df);
    uint32_t scalar_tile_size = tt::tt_metal::detail::TileSize(scalar_df);
    uint32_t im_tile_size = tt::tt_metal::detail::TileSize(im_df);
    uint32_t stats_tile_size = tt::tt_metal::detail::TileSize(stats_df);

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

    auto cb_in0_id = CreateCircularBuffer(program, core_grid, c_in0_config);
    // K input
    auto c_in1_config = CircularBufferConfig(k_tiles * k_tile_size, {{tt::CBIndex::c_1, k_df}})
                            .set_page_size(tt::CBIndex::c_1, k_tile_size);
    auto cb_in1_id = CreateCircularBuffer(program, core_grid, c_in1_config);
    // V input
    auto c_in2_config = CircularBufferConfig(v_tiles * v_tile_size, {{tt::CBIndex::c_2, v_df}})
                            .set_page_size(tt::CBIndex::c_2, v_tile_size);
    auto cb_in2_id = CreateCircularBuffer(program, core_grid, c_in2_config);

    // Only create mask buffer if it's going to be used
    if (use_provided_mask or is_causal or use_padded_mask) {
        // attn_mask input
        auto c_in3_config = CircularBufferConfig(mask_tiles * mask_tile_size, {{tt::CBIndex::c_3, mask_df}})
                                .set_page_size(tt::CBIndex::c_3, mask_tile_size);
        auto cb_in3_id = CreateCircularBuffer(program, core_grid, c_in3_config);
    }

    // scale input
    auto c_in4_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{tt::CBIndex::c_4, scalar_df}})
                            .set_page_size(tt::CBIndex::c_4, scalar_tile_size);
    auto cb_in4_id = CreateCircularBuffer(program, core_grid, c_in4_config);

    // identity scale input
    auto c_in5_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{tt::CBIndex::c_5, scalar_df}})
                            .set_page_size(tt::CBIndex::c_5, scalar_tile_size);
    auto cb_in5_id = CreateCircularBuffer(program, core_grid, c_in5_config);

    if (is_chunked) {
        auto c_in6_config = CircularBufferConfig(page_table_stick_size, {{tt::CBIndex::c_6, page_table_df}})
                                .set_page_size(tt::CBIndex::c_6, page_table_stick_size);
        auto cb_in6_id = CreateCircularBuffer(program, core_grid, c_in6_config);
    }

    // cb_qk_im
    auto c_intermed0_config = CircularBufferConfig(qk_tiles * im_tile_size, {{tt::CBIndex::c_24, im_df}})
                                  .set_page_size(tt::CBIndex::c_24, im_tile_size);
    auto cb_intermed0_id = CreateCircularBuffer(program, core_grid, c_intermed0_config);

    // cb_out_im
    auto c_intermed1_config = CircularBufferConfig(out_im_tiles * im_tile_size, {{tt::CBIndex::c_25, im_df}})
                                  .set_page_size(tt::CBIndex::c_25, im_tile_size);
    auto cb_intermed1_id = CreateCircularBuffer(program, core_grid, c_intermed1_config);

    // cb_out_accumulate_im
    auto c_intermed2_config = CircularBufferConfig(out_im_tiles * im_tile_size, {{tt::CBIndex::c_26, im_df}})
                                  .set_page_size(tt::CBIndex::c_26, im_tile_size);
    auto cb_intermed2_id = CreateCircularBuffer(program, core_grid, c_intermed2_config);

    // cb_cur_max
    auto c_intermed3_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_27, stats_df}})
                                  .set_page_size(tt::CBIndex::c_27, stats_tile_size);
    auto cb_intermed3_id = CreateCircularBuffer(program, core_grid, c_intermed3_config);

    // cb_prev_max
    auto c_intermed4_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_28, stats_df}})
                                  .set_page_size(tt::CBIndex::c_28, stats_tile_size);
    auto cb_intermed4_id = CreateCircularBuffer(program, core_grid, c_intermed4_config);

    // cb_cur_sum
    auto c_intermed5_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_29, stats_df}})
                                  .set_page_size(tt::CBIndex::c_29, stats_tile_size);
    auto cb_intermed5_id = CreateCircularBuffer(program, core_grid, c_intermed5_config);

    // cb_prev_sum
    auto c_intermed6_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_30, stats_df}})
                                  .set_page_size(tt::CBIndex::c_30, stats_tile_size);
    auto cb_intermed6_id = CreateCircularBuffer(program, core_grid, c_intermed6_config);

    // cb_exp_max_diff
    auto c_intermed7_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_31, stats_df}})
                                  .set_page_size(tt::CBIndex::c_31, stats_tile_size);
    auto cb_intermed7_id = CreateCircularBuffer(program, core_grid, c_intermed7_config);

    // Output
    auto c_out0_config = CircularBufferConfig(out0_t * out_tile_size, {{tt::CBIndex::c_16, out_df}})
                             .set_page_size(tt::CBIndex::c_16, out_tile_size);

    auto cb_out0_id = CreateCircularBuffer(program, core_grid, c_out0_config);

    uint32_t q_addr = q_buffer->address();
    uint32_t k_addr = k_buffer->address();
    uint32_t v_addr = v_buffer->address();
    uint32_t mask_addr = attn_mask.has_value() ? mask_buffer->address() : 0;
    uint32_t out_addr = out0_buffer->address();

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

        SetRuntimeArgs(
            program,
            reader_kernels_id,
            core,
            {q_addr,
             k_addr,
             v_addr,
             mask_addr,
             is_chunked ? page_table.value().buffer()->address() : 0,
             i,
             local_batch_start,
             local_batch_end,
             local_nh_start,
             local_nh_end,
             local_q_start,
             local_q_end,
             chunked_q_chunk_offset});
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
             chunked_q_chunk_offset});
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
             chunked_q_chunk_offset});
    }

    auto override_runtime_arguments_callback =
        [num_cores, grid_size, reader_kernels_id, writer_kernels_id, compute_kernels_id, is_chunked, q_chunk_size](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto q_buffer = input_tensors.at(0).buffer();
            auto k_buffer = input_tensors.at(1).buffer();
            auto v_buffer = input_tensors.at(2).buffer();
            auto mask_buffer =
                optional_input_tensors.at(0).has_value() ? optional_input_tensors.at(0).value().buffer() : nullptr;

            auto out0_buffer = output_tensors.at(0).buffer();
            uint32_t q_addr = q_buffer->address();
            uint32_t k_addr = k_buffer->address();
            uint32_t v_addr = v_buffer->address();
            uint32_t mask_addr = mask_buffer != nullptr ? mask_buffer->address() : 0;
            uint32_t out_addr = out0_buffer->address();

            uint32_t page_table_addr = 0;
            uint32_t chunked_q_chunk_offset = 0;
            if (is_chunked) {
                page_table_addr = optional_input_tensors.at(1).value().buffer()->address();
                chunked_q_chunk_offset =
                    static_cast<const ScaledDotProductAttention*>(operation)->chunk_start_idx.value() / q_chunk_size;
            }

            auto& reader_args_by_core = GetRuntimeArgs(program, reader_kernels_id);
            auto& writer_args_by_core = GetRuntimeArgs(program, writer_kernels_id);
            auto& compute_args_by_core = GetRuntimeArgs(program, compute_kernels_id);
            // Set reader rt args
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
                reader_args[12] = chunked_q_chunk_offset;

                writer_args[0] = out_addr;
                writer_args[8] = chunked_q_chunk_offset;

                compute_args[7] = chunked_q_chunk_offset;
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::transformer::detail
