// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_distributed_sdpa_program_factory.hpp"

#include <cmath>
#include <optional>
#include <string>
#include <unordered_map>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::transformer::ring_distributed_sdpa::program {

// Ring-distributed SDPA program factory
RingDistributedSdpaMeshWorkloadFactory::cached_program_t RingDistributedSdpaMeshWorkloadFactory::create_at(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const Tensor& input_tensor_v,
    const Tensor& output_tensor,
    uint32_t ring_size,
    uint32_t ring_id,
    std::optional<float> scale,
    std::size_t q_chunk_size,
    std::size_t k_chunk_size,
    DeviceComputeKernelConfig compute_kernel_config,
    std::optional<SDPAProgramConfig> program_config,
    const std::optional<Tensor>& page_table,
    std::optional<int64_t> chunk_start_idx) {
    /*
    Q: B x NQH x S*ring_size x DH
    K: B x NKH x DH x S
    V: B x NKH x S x DH
    */
    Program program = CreateProgram();

    IDevice* device = input_tensor_q.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    auto* q_buffer = input_tensor_q.buffer();
    auto* k_buffer = input_tensor_k.buffer();
    auto* v_buffer = input_tensor_v.buffer();

    auto* out0_buffer = output_tensor.buffer();

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

    const auto& q_shape = input_tensor_q.logical_shape();
    const auto& k_shape = input_tensor_k.logical_shape();
    const uint32_t B = q_shape[0], NQH = q_shape[1], Sq = q_shape[2] / (2 * ring_size), DH = q_shape[3];
    const uint32_t NKH = k_shape[1];

    // define chunk_1 and chunk_2
    const uint32_t chunk_1 = ring_id;
    const uint32_t chunk_2 = (2 * ring_size) - ring_id - 1;

    bool is_chunked = chunk_start_idx.has_value();

    // Extract paged KV cache parameters first, before calculating Sk
    uint32_t block_size = 0;
    uint32_t block_size_t = 0;
    [[maybe_unused]] uint32_t max_blocks_per_seq = 0;
    uint32_t page_table_stick_size = 0;
    tt::DataFormat page_table_df = tt::DataFormat::Int32;

    // Calculate KV sequence length: when using paged KV, k_shape[2] is block_size, not sequence length
    // The full KV sequence length is: chunk_start_idx + full_Q_seq_len
    // In ring distributed, q_shape[2] is the full Q sequence length (before ring distribution)
    // So full_Q_seq_len = q_shape[2] = Sq * 2 * ring_size
    uint32_t Sk;
    if (is_chunked) {
        const auto& page_table_tensor = page_table.value();
        block_size = k_shape[2];  // K's sequence dimension represents block size
        block_size_t = block_size / TILE_HEIGHT;
        max_blocks_per_seq = page_table_tensor.padded_shape()[1];
        page_table_stick_size = page_table_tensor.buffer()->aligned_page_size();
        TT_FATAL(
            page_table_stick_size % 32 == 0,
            "page table page size in bytes must be a multiple of 32 due to address alignment");

        // Full KV sequence length = chunk_start_idx + full_Q_seq_len
        // q_shape[2] is the full Q sequence length (before ring distribution)
        Sk = chunk_start_idx.value() + q_shape[2];
    } else {
        Sk = k_shape[2];
    }

    // Calculate padded sequence length
    const uint32_t padded_Sq = std::ceil((float)Sq / q_chunk_size) * q_chunk_size;
    const uint32_t padded_Sk = std::ceil((float)Sk / k_chunk_size) * k_chunk_size;

    const uint32_t Sqt = padded_Sq / TILE_HEIGHT;
    const uint32_t Skt = padded_Sk / TILE_HEIGHT;
    const uint32_t DHt = DH / TILE_WIDTH;
    const uint32_t vDHt = DHt;

    const uint32_t valid_Sqt = std::ceil((float)Sq / TILE_HEIGHT);
    const uint32_t valid_Skt = std::ceil((float)Sk / TILE_HEIGHT);

    const uint32_t Sq_chunk_t = q_chunk_size / TILE_HEIGHT;
    const uint32_t Sk_chunk_t = k_chunk_size / TILE_HEIGHT;
    const uint32_t q_num_chunks = padded_Sq / q_chunk_size;
    const uint32_t k_num_chunks = padded_Sk / k_chunk_size;

    // Calculate chunk offsets for ring distribution
    uint32_t chunked_q_chunk_offset_phase_1 = (chunk_1 * Sq) / q_chunk_size;
    uint32_t chunked_q_chunk_offset_phase_2 = (chunk_2 * Sq) / q_chunk_size;

    // Add chunk_start_idx offset for prefix caching
    // The offset should be relative to the full Q sequence, so we add chunk_start_idx / q_chunk_size
    if (is_chunked) {
        chunked_q_chunk_offset_phase_1 += chunk_start_idx.value() / q_chunk_size;
        chunked_q_chunk_offset_phase_2 += chunk_start_idx.value() / q_chunk_size;
    }

    // Parallelization scheme
    // We will choose parallelization factors for batch, num_heads, and q_seq_len in that order
    uint32_t batch_parallel_factor = std::min(B, num_cores);
    uint32_t nh_parallel_factor = std::min(num_cores / batch_parallel_factor, NQH);
    uint32_t q_parallel_factor = std::min(num_cores / (batch_parallel_factor * nh_parallel_factor), q_num_chunks);

    // Ceiling divide to allow for non-perfect divisions
    const uint32_t batch_per_core = (B + batch_parallel_factor - 1) / batch_parallel_factor;
    const uint32_t nh_per_core = (NQH + nh_parallel_factor - 1) / nh_parallel_factor;
    const uint32_t q_per_core = (q_num_chunks + q_parallel_factor - 1) / q_parallel_factor;

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t q_tiles = Sq_chunk_t * DHt * 2;
    uint32_t k_tiles = Sk_chunk_t * DHt * 2;            // double buffer
    uint32_t v_tiles = Sk_chunk_t * vDHt * 2;           // double buffer
    uint32_t mask_tiles = Sq_chunk_t * Sk_chunk_t * 2;  // double buffer
    uint32_t qk_tiles = Sq_chunk_t * Sk_chunk_t;
    uint32_t out_im_tiles = Sq_chunk_t * vDHt;
    uint32_t out0_t = Sq_chunk_t * vDHt;
    uint32_t scale_tiles = 1;
    uint32_t statistics_tiles = Sq_chunk_t;  // Single column of values in each iteration

    // log all values

    // Host code is responsible for determining matmul configuration
    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
    const uint32_t qk_in0_block_w = DHt;
    // max of Sk_chunk_t and dst_size
    uint32_t qk_out_subblock_w = std::min(Sk_chunk_t, dst_size);
    // If qk_out_subblock_w is full row of output, scale subblock_h so volume = dst_size. Otherwise it's 1 to maintain
    // row-major intermediate buffer.
    uint32_t qk_out_subblock_h =
        (qk_out_subblock_w == Sk_chunk_t) ? (std::min(Sq_chunk_t, dst_size / qk_out_subblock_w)) : 1;

    if (qk_out_subblock_w == dst_size && qk_out_subblock_h == 1 && Sk_chunk_t % 2 == 0) {
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

    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    class bfloat16 bfloat_identity_scalar(1.0f);
    uint32_t packed_identity_scalar = pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});

    union {
        float f;
        uint32_t u;
    } scale_union{};
    scale_union.f = scale.value_or(1.0f);

    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        B,
        NQH,
        NKH,
        Sqt,
        Skt,
        valid_Sqt * 2 * ring_size,
        valid_Skt,
        DHt,
        vDHt,
        Sq_chunk_t,
        q_num_chunks,
        Sk_chunk_t,
        k_num_chunks,
        num_cores,
        true,                  //(std::uint32_t)is_causal,
        false,                 //(std::uint32_t)use_provided_mask,
        false,                 //(std::uint32_t)broadcast_provided_mask_heads,
        false,                 //(std::uint32_t)use_padded_mask,
        (uint32_t)is_chunked,  //(uint32_t)is_chunked,
        block_size_t,
        page_table_stick_size,
        0  // use_attention_sink
    };
    TensorAccessorArgs(input_tensor_q.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs().append_to(reader_compile_time_args);  // mask tensor (not used in ring)
    TensorAccessorArgs(page_table.has_value() ? page_table->buffer() : nullptr)
        .append_to(reader_compile_time_args);                  // page table
    TensorAccessorArgs().append_to(reader_compile_time_args);  // attention sink (not used in ring)

    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        B,
        NQH,
        NKH,
        Sqt,
        valid_Sqt * 2,
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
        true,   //(std::uint32_t)is_causal,
        false,  //(std::uint32_t)use_provided_mask,
        false,  //(std::uint32_t)use_padded_mask,
        true,   //(uint32_t)is_chunked,
        0,      //(uint32_t)sliding_window_size,
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
        true,   //(std::uint32_t)is_causal,
        false,  //(std::uint32_t)use_provided_mask,
        false,  //(std::uint32_t)use_padded_mask,
        true,   //(uint32_t)is_chunked,
        scale_union.u,
        0,  //(uint32_t)sliding_window_size,
        0,  //(std::uint32_t)use_attention_sink,
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
    uint32_t balanced_q_parallel = (q_per_core * q_parallel_factor == q_num_chunks) && (q_per_core % 2 == 0);
    if (balanced_q_parallel) {
        defines["BALANCED_Q_PARALLEL"] = "1";
    }

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
    auto c_in3_config = CircularBufferConfig(mask_tiles * mask_tile_size, {{tt::CBIndex::c_3, mask_df}})
                            .set_page_size(tt::CBIndex::c_3, mask_tile_size);
    CreateCircularBuffer(program, core_grid, c_in3_config);

    // identity scalar input
    auto c_in5_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{tt::CBIndex::c_5, scalar_df}})
                            .set_page_size(tt::CBIndex::c_5, scalar_tile_size);
    CreateCircularBuffer(program, core_grid, c_in5_config);
    // identity column input
    auto c_in7_config = CircularBufferConfig(scale_tiles * scalar_tile_size, {{tt::CBIndex::c_7, scalar_df}})
                            .set_page_size(tt::CBIndex::c_7, scalar_tile_size);
    CreateCircularBuffer(program, core_grid, c_in7_config);

    // page table circular buffer (only when using paged KV)
    if (is_chunked) {
        auto c_in6_config = CircularBufferConfig(page_table_stick_size, {{tt::CBIndex::c_6, page_table_df}})
                                .set_page_size(tt::CBIndex::c_6, page_table_stick_size);
        CreateCircularBuffer(program, core_grid, c_in6_config);
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
    uint32_t out_addr = out0_buffer->address();

    // Set reader rt args
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};
        uint32_t core_id = i;

        uint32_t read_offset_phase_1 = chunk_1 * Sqt;
        uint32_t read_offset_phase_2 = chunk_2 * Sqt;
        uint32_t write_offset_phase_1 = 0;
        uint32_t write_offset_phase_2 = Sqt;

        uint32_t local_batch_start = (core_id / (nh_parallel_factor * q_parallel_factor)) * batch_per_core;
        uint32_t local_batch_end = local_batch_start + batch_per_core;
        uint32_t local_nh_start = ((core_id / q_parallel_factor) % nh_parallel_factor) * nh_per_core;
        uint32_t local_nh_end = local_nh_start + nh_per_core;
        uint32_t local_q_start = (core_id % q_parallel_factor) * q_per_core;
        uint32_t local_q_end = local_q_start + q_per_core;

        // clamp all to max values for non-even partitioning
        local_batch_start = std::min(local_batch_start, B);
        local_batch_end = std::min(local_batch_end, B);
        local_nh_start = std::min(local_nh_start, NQH);
        local_nh_end = std::min(local_nh_end, NQH);
        local_q_start = std::min(local_q_start, q_num_chunks);
        local_q_end = std::min(local_q_end, q_num_chunks);

        uint32_t page_table_addr = page_table.has_value() ? page_table->buffer()->address() : 0;
        SetRuntimeArgs(
            program,
            reader_kernels_id,
            core,
            {q_addr,
             k_addr,
             v_addr,
             0,  // mask_addr,
             page_table_addr,
             0,  // attention sink addr,
             i,
             local_batch_start,
             local_batch_end,
             local_nh_start,
             local_nh_end,
             local_q_start,
             local_q_end,
             2,
             chunked_q_chunk_offset_phase_1,
             read_offset_phase_1,
             chunked_q_chunk_offset_phase_2,
             read_offset_phase_2});
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
             2,
             chunked_q_chunk_offset_phase_1,
             write_offset_phase_1,
             chunked_q_chunk_offset_phase_2,
             write_offset_phase_2});
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
             2,
             chunked_q_chunk_offset_phase_1,
             chunked_q_chunk_offset_phase_2});
    }

    return cached_program_t(
        std::move(program),
        shared_variables_t{
            .num_cores = num_cores,
            .grid_size = grid_size,
            .reader_kernel_id = reader_kernels_id,
            .writer_kernel_id = writer_kernels_id,
            .compute_kernel_id = compute_kernels_id,
        });
}

RingDistributedSdpaMeshWorkloadFactory::cached_mesh_workload_t
RingDistributedSdpaMeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    const auto& input_tensor_q = tensor_args.q;
    const auto& input_tensor_k = tensor_args.k;
    const auto& input_tensor_v = tensor_args.v;

    const auto q_chunk_size =
        operation_attributes.program_config.has_value() ? operation_attributes.program_config->q_chunk_size : 32;
    const auto k_chunk_size =
        operation_attributes.program_config.has_value() ? operation_attributes.program_config->k_chunk_size : 32;

    // Determine ring_id: use provided value or infer from device coordinate
    uint32_t curr_ring_id;
    // Create programs for each coordinate in tensor_coords
    for (const auto& mesh_coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : mesh_coord_range) {
            if (operation_attributes.ring_id.has_value()) {
                // Use explicitly provided ring_id
                curr_ring_id = operation_attributes.ring_id.value();
            } else {
                // Infer ring_id from device coordinate (similar to ring_joint_sdpa)
                auto* mesh_device = input_tensor_q.device();
                IDevice* target_device = mesh_device ? mesh_device->get_device(mesh_coord) : input_tensor_q.device();

                // Ensure mesh_device is not null before dereferencing
                TT_FATAL(mesh_device != nullptr, "Mesh device must not be null when inferring ring_id");
                const auto& mesh_view = mesh_device->get_view();
                std::vector<IDevice*> devices_to_use;
                // For simplicity, assume ring is along the first axis (adjust as needed)
                if (mesh_view.shape()[0] == operation_attributes.ring_size) {
                    devices_to_use = mesh_view.get_devices_on_column(mesh_coord[1]);
                } else if (mesh_view.shape()[1] == operation_attributes.ring_size) {
                    devices_to_use = mesh_view.get_devices_on_row(mesh_coord[0]);
                } else {
                    TT_FATAL(
                        false,
                        "Ring size {} doesn't match mesh dimensions [{}, {}]",
                        operation_attributes.ring_size,
                        mesh_view.shape()[0],
                        mesh_view.shape()[1]);
                }

                // Find ring_id (device index in the ring)
                curr_ring_id = 0;
                for (uint32_t i = 0; i < operation_attributes.ring_size; ++i) {
                    if (devices_to_use.at(i) == target_device) {
                        curr_ring_id = i;
                        break;
                    }
                }
                log_debug(tt::LogOp, "Inferred ring_id: {} for device_id: {}", curr_ring_id, target_device->id());
            }
            // Create a program for this specific coordinate using the base factory
            const ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};
            auto cached_program = create_at(
                input_tensor_q,
                input_tensor_k,
                input_tensor_v,
                tensor_return_value,
                operation_attributes.ring_size,
                curr_ring_id,
                operation_attributes.scale,
                q_chunk_size,
                k_chunk_size,
                operation_attributes.compute_kernel_config,
                operation_attributes.program_config,
                tensor_args.page_table,
                operation_attributes.chunk_start_idx);
            shared_variables[single_coord_range] = cached_program.shared_variables;
            mesh_workload.add_program(single_coord_range, std::move(cached_program.program));
        }
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
}

void RingDistributedSdpaMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // Update runtime arguments for each program in the mesh workload
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        const auto& reader_kernels_id = shared_vars.reader_kernel_id;
        const auto& writer_kernels_id = shared_vars.writer_kernel_id;
        const auto num_cores = shared_vars.num_cores;
        const auto& grid_size = shared_vars.grid_size;

        auto* q_buffer = tensor_args.q.buffer();
        auto* k_buffer = tensor_args.k.buffer();
        auto* v_buffer = tensor_args.v.buffer();

        auto* out0_buffer = tensor_return_value.buffer();
        uint32_t q_addr = q_buffer->address();
        uint32_t k_addr = k_buffer->address();
        uint32_t v_addr = v_buffer->address();
        uint32_t out_addr = out0_buffer->address();
        uint32_t page_table_addr = tensor_args.page_table.has_value() ? tensor_args.page_table->buffer()->address() : 0;

        auto& reader_args_by_core = GetRuntimeArgs(program, reader_kernels_id);
        auto& writer_args_by_core = GetRuntimeArgs(program, writer_kernels_id);

        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord core = {i % grid_size.x, i / grid_size.x};

            auto& reader_args = reader_args_by_core[core.x][core.y];
            auto& writer_args = writer_args_by_core[core.x][core.y];

            reader_args[0] = q_addr;
            reader_args[1] = k_addr;
            reader_args[2] = v_addr;
            reader_args[4] = page_table_addr;  // Update page_table_addr (index 4 is after mask_addr)

            writer_args[0] = out_addr;
        }
    }
}
}  // namespace ttnn::operations::transformer::ring_distributed_sdpa::program
