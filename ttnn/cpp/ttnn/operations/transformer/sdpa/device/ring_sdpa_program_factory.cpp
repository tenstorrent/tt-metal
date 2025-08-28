// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_sdpa_program_factory.hpp"
#include "ring_sdpa_op.hpp"

#include <optional>
#include <string>
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

// Ring-distributed SDPA program factory
operation::ProgramWithCallbacks ring_sdpa_multi_core(
    const Tensor& input_tensor_q,
    const Tensor& input_tensor_k,
    const Tensor& input_tensor_v,
    const Tensor& output_tensor,
    uint32_t ring_size,
    uint32_t ring_id,
    std::optional<float> scale,
    bool is_causal,
    std::size_t q_chunk_size,
    std::size_t k_chunk_size,
    DeviceComputeKernelConfig compute_kernel_config,
    std::optional<SDPAProgramConfig> program_config) {
    /*
    Ring-Distributed SDPA:
    Q: B x NQH x S x DH (global shape)
    K: B x NKH x S x DH (global shape - each device reads full K for now)
    V: B x NKH x S x DH (global shape - each device reads full V for now)
    Output: B x NQH x local_S x DH (local shape - device processes subset of queries)

    Ring distribution: Each device processes 2 chunks of Q
    - Chunk 1: ring_id
    - Chunk 2: (2*ring_size-1) - ring_id
    This balances early (cheap) and late (expensive) queries across devices.
    */

    const auto& q_shape = input_tensor_q.logical_shape();
    const auto& k_shape = input_tensor_k.logical_shape();
    const uint32_t B = q_shape[0], NQH = q_shape[1], Sq = q_shape[2], DH = q_shape[3];
    const uint32_t NKH = k_shape[1];

    // Ring-distributed SDPA is always causal with full sequences
    const uint32_t Sk = Sq;  // Causal requirement

    // Calculate ring distribution parameters
    const uint32_t global_chunk_size_positions = Sq / (2 * ring_size);  // Size of each Q chunk in positions
    const uint32_t global_chunk_size =
        std::max(1u, global_chunk_size_positions / TILE_HEIGHT);  // Convert to tile units
    const uint32_t local_num_chunks = 2;                          // Each device processes exactly 2 chunks
    const uint32_t local_seq_len =
        local_num_chunks * global_chunk_size_positions;  // Local Q sequence length in positions

    log_info(tt::LogOp, "Ring distribution parameters:");
    log_info(tt::LogOp, "ring_size: {}, ring_id: {}", ring_size, ring_id);
    log_info(tt::LogOp, "global_chunk_size: {} positions = {} tiles", global_chunk_size_positions, global_chunk_size);
    log_info(tt::LogOp, "local_seq_len: {}", local_seq_len);

    // Calculate which global chunks this device processes
    uint32_t first_chunk_id = ring_id;
    uint32_t second_chunk_id = (2 * ring_size - 1) - ring_id;

    log_info(tt::LogOp, "Device {} processes chunks: {} and {}", ring_id, first_chunk_id, second_chunk_id);

    /*
    Note about tensor shapes:
    Ring-distributed SDPA requires sequences divisible by 2*ring_size.
    Internally, we work with the local sequence length (local_seq_len = 2 * global_chunk_size_positions)
    and map iterations to the two global chunk ranges.
    */

    // Calculate padded sequence lengths based on local processing
    const uint32_t padded_local_Sq = std::ceil((float)local_seq_len / q_chunk_size) * q_chunk_size;
    const uint32_t padded_Sk = std::ceil((float)Sk / k_chunk_size) * k_chunk_size;  // Still need full K

    const uint32_t local_Sqt = padded_local_Sq / TILE_HEIGHT;
    const uint32_t Skt = padded_Sk / TILE_HEIGHT;
    const uint32_t DHt = DH / TILE_WIDTH;
    const uint32_t vDHt = DHt;  // No MLA support in ring distribution for now

    const uint32_t valid_local_Sqt = std::ceil((float)local_seq_len / TILE_HEIGHT);
    const uint32_t valid_Skt = std::ceil((float)Sk / TILE_HEIGHT);

    const uint32_t Sq_chunk_t = q_chunk_size / TILE_HEIGHT;
    const uint32_t Sk_chunk_t = k_chunk_size / TILE_HEIGHT;
    const uint32_t local_q_num_chunks = padded_local_Sq / q_chunk_size;  // Local chunks for this device
    const uint32_t k_num_chunks = padded_Sk / k_chunk_size;              // Global K chunks (all devices need same)

    // Log all parameters
    log_info(tt::LogOp, "Shape parameters:");
    log_info(tt::LogOp, "B: {}, NQH: {}, Sq (global): {}, DH: {}", B, NQH, Sq, DH);
    log_info(
        tt::LogOp, "local_seq_len: {}, padded_local_Sq: {}, padded_Sk: {}", local_seq_len, padded_local_Sq, padded_Sk);
    log_info(tt::LogOp, "local_q_num_chunks: {}, k_num_chunks: {}", local_q_num_chunks, k_num_chunks);
    log_info(tt::LogOp, "Sq_chunk_t: {}, Sk_chunk_t: {}", Sq_chunk_t, Sk_chunk_t);

    Program program = CreateProgram();
    IDevice* device = input_tensor_q.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    auto q_buffer = input_tensor_q.buffer();
    auto k_buffer = input_tensor_k.buffer();
    auto v_buffer = input_tensor_v.buffer();
    auto out0_buffer = output_tensor.buffer();

    CoreCoord grid_size = {1, 1};  // program_config.has_value() ? program_config->compute_with_storage_grid_size
                                   //: device->compute_with_storage_grid_size();
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

    // Parallelization scheme - adapted for local processing
    // We parallelize over batch, num_heads, and local Q chunks
    uint32_t batch_parallel_factor = std::min(B, num_cores);
    uint32_t nh_parallel_factor = std::min(num_cores / batch_parallel_factor, NQH);
    uint32_t q_parallel_factor = std::min(num_cores / (batch_parallel_factor * nh_parallel_factor), local_q_num_chunks);

    TT_FATAL(
        batch_parallel_factor * nh_parallel_factor * q_parallel_factor <= num_cores,
        "Parallelism must not exceed number of cores. Got {}, expected at most {}.",
        batch_parallel_factor * nh_parallel_factor * q_parallel_factor,
        num_cores);

    log_info(tt::LogOp, "Ring parallelization scheme:");
    log_info(tt::LogOp, "batch_parallel_factor: {}", batch_parallel_factor);
    log_info(tt::LogOp, "nh_parallel_factor: {}", nh_parallel_factor);
    log_info(tt::LogOp, "q_parallel_factor: {} (for local chunks)", q_parallel_factor);

    // Calculate work per core
    const uint32_t batch_per_core = (B + batch_parallel_factor - 1) / batch_parallel_factor;
    const uint32_t nh_per_core = (NQH + nh_parallel_factor - 1) / nh_parallel_factor;
    const uint32_t local_q_per_core = (local_q_num_chunks + q_parallel_factor - 1) / q_parallel_factor;

    const uint32_t q_buffer_factor = (local_q_per_core > 1) ? 2 : 1;

    log_info(tt::LogOp, "Work distribution:");
    log_info(
        tt::LogOp,
        "batch_per_core: {}, nh_per_core: {}, local_q_per_core: {}",
        batch_per_core,
        nh_per_core,
        local_q_per_core);

    // Calculate CB tile counts (adjusted for ring distribution causal attention)
    uint32_t q_tiles = Sq_chunk_t * DHt * q_buffer_factor;

    // For ring distribution, calculate max K/V chunks needed for causal attention
    // The highest chunk (second_chunk_id) needs K/V chunks [0..second_chunk_id], so (second_chunk_id + 1) total chunks
    uint32_t max_k_chunks_needed = second_chunk_id + 1;
    uint32_t k_tiles =
        max_k_chunks_needed * Sk_chunk_t * DHt * 2;  // buffer for all needed K chunks with double buffering
    uint32_t v_tiles =
        max_k_chunks_needed * Sk_chunk_t * vDHt * 2;  // buffer for all needed V chunks with double buffering
    // For ring SDPA, mask CB needs to hold masks for all K chunks that the largest Q chunk attends to
    // In causal attention, the last Q chunk (second_chunk_id) attends to the most K chunks
    uint32_t max_k_chunks_for_any_q = second_chunk_id + 1;  // Q chunk attends to chunks [0, second_chunk_id]
    uint32_t qk_tiles = max_k_chunks_for_any_q * Sq_chunk_t * Sk_chunk_t;

    log_info(tt::LogOp, "Ring SDPA mask CB sizing:");
    log_info(
        tt::LogOp,
        "second_chunk_id: {}, max_k_chunks_for_any_q: {}, qk_tiles: {}",
        second_chunk_id,
        max_k_chunks_for_any_q,
        qk_tiles);
    uint32_t out_im_tiles = Sq_chunk_t * vDHt;
    uint32_t out0_t = Sq_chunk_t * vDHt;
    uint32_t scale_tiles = 1;
    uint32_t statistics_tiles = Sq_chunk_t;

    log_info(tt::LogOp, "Circular buffer sizes:");
    log_info(tt::LogOp, "max_k_chunks_needed: {}, second_chunk_id: {}", max_k_chunks_needed, second_chunk_id);
    log_info(tt::LogOp, "q_tiles: {}, k_tiles: {}, v_tiles: {}", q_tiles, k_tiles, v_tiles);
    log_info(tt::LogOp, "qk_tiles: {}, out0_t: {}, statistics_tiles: {}", qk_tiles, out0_t, statistics_tiles);

    // Scale computation
    float scale_val = scale.value_or(1.0f / std::sqrt(static_cast<float>(DH)));
    bfloat16 scale_bf16 = bfloat16(scale_val);
    uint32_t scale_fp32 = std::bit_cast<uint32_t>(scale_val);

    log_info(tt::LogOp, "scale_val: {}", scale_val);

    // Matmul configuration (same as regular SDPA)
    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;

    // QK matmul config
    const uint32_t qk_in0_block_w = DHt;
    uint32_t qk_out_subblock_w = std::min(Sk_chunk_t, dst_size);
    uint32_t qk_out_subblock_h =
        (qk_out_subblock_w == Sk_chunk_t) ? (std::min(Sq_chunk_t, dst_size / qk_out_subblock_w)) : 1;

    if (qk_out_subblock_w == dst_size && qk_out_subblock_h == 1 && Sk_chunk_t % 2 == 0) {
        qk_out_subblock_w = qk_out_subblock_w / 2;
        qk_out_subblock_h = 2;
    }

    const uint32_t qk_in0_num_subblocks = Sq_chunk_t / qk_out_subblock_h;
    const uint32_t qk_in1_num_subblocks = Sk_chunk_t / qk_out_subblock_w;
    const uint32_t qk_num_blocks = DHt / qk_in0_block_w;

    // Output matmul config
    const uint32_t out_in0_block_w = Sk_chunk_t;
    const uint32_t out_out_subblock_w = std::min(vDHt, dst_size);
    const uint32_t out_out_subblock_h =
        (out_out_subblock_w == vDHt) ? (std::min(Sq_chunk_t, dst_size / out_out_subblock_w)) : 1;
    const uint32_t out_in0_num_subblocks = Sq_chunk_t / out_out_subblock_h;
    const uint32_t out_in1_num_subblocks = vDHt / out_out_subblock_w;
    const uint32_t out_num_blocks = Sk_chunk_t / out_in0_block_w;

    // Calculate scalar values needed by writer kernel (same as regular SDPA)
    class bfloat16 bfloat_identity_scalar(1.0f);
    uint32_t packed_identity_scalar = pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});

    union {
        float f;
        uint32_t u;
    } scale_union{};
    scale_union.f = scale.value_or(1.0f);

    // Compile-time arguments
    std::vector<uint32_t> reader_compile_time_args = {
        B,
        NQH,
        NKH,
        Skt,
        DHt,
        vDHt,
        Sq_chunk_t,
        local_q_num_chunks,
        Sk_chunk_t,
        k_num_chunks,
        valid_local_Sqt,
        valid_Skt,
        static_cast<uint32_t>(is_causal),
        ring_size,
        ring_id,  // New: ring parameters
        first_chunk_id,
        second_chunk_id,
        global_chunk_size  // New: ring distribution parameters
    };

    std::vector<uint32_t> writer_compile_time_args = {
        B,
        NQH,
        NKH,
        Skt,
        DHt,
        vDHt,  // 0-5
        Sq_chunk_t,
        local_q_num_chunks,  // 6-7
        Sk_chunk_t,
        k_num_chunks,                      // 8-9
        valid_Skt,                         // 10
        static_cast<uint32_t>(is_causal),  // 11
        ring_size,
        ring_id,  // 12-13
        first_chunk_id,
        second_chunk_id,
        global_chunk_size,  // 14-16
        packed_identity_scalar,
        scale_union.u  // 17-18: scalar values for generate functions
    };

    std::vector<uint32_t> compute_compile_time_args = {
        B,
        NQH,
        NKH,
        Skt,
        DHt,
        vDHt,  // 0-5
        Sq_chunk_t,
        local_q_num_chunks,
        Sk_chunk_t,
        k_num_chunks,  // 6-9
        qk_in0_block_w,
        qk_out_subblock_w,
        qk_out_subblock_h,
        qk_in0_num_subblocks,
        qk_in1_num_subblocks,
        qk_num_blocks,  // 10-15
        out_in0_block_w,
        out_out_subblock_w,
        out_out_subblock_h,
        out_in0_num_subblocks,
        out_in1_num_subblocks,
        out_num_blocks,  // 16-21
        num_cores,       // 22
        scale_fp32,      // 23 (removed is_causal since ring SDPA is always causal)
        ring_size,
        ring_id,  // 24-25: ring parameters
        first_chunk_id,
        second_chunk_id  // 26-27: ring chunk IDs
    };

    // Calculate granularities for SFPU operations (similar to regular SDPA)
    const uint32_t stats_granularity = std::min(Sq_chunk_t, dst_size);
    const uint32_t log2_stats_granularity = std::log2(stats_granularity);
    TT_ASSERT(
        stats_granularity == (1 << log2_stats_granularity),
        "stats_granularity must be a power of 2. Got {}.",
        stats_granularity);

    const uint32_t sub_exp_granularity = std::min(Sk_chunk_t, dst_size);
    const uint32_t log2_sub_exp_granularity = std::log2(sub_exp_granularity);
    TT_ASSERT(
        sub_exp_granularity == (1 << log2_sub_exp_granularity),
        "sub_exp_granularity must be a power of 2. Got {}.",
        sub_exp_granularity);

    uint32_t dht_granularity = std::min(DHt, dst_size);
    uint32_t log2_dht_granularity = std::log2(dht_granularity);
    // Sometimes DHt is not a power of 2, so granularity should be 1
    if (dht_granularity != (1 << log2_dht_granularity)) {
        dht_granularity = 1;
        log2_dht_granularity = 0;
    }
    TT_ASSERT(
        dht_granularity == (1 << log2_dht_granularity),
        "dht_granularity must be a power of 2. Got {}.",
        dht_granularity);

    log_info(tt::LogOp, "stats_granularity: {}", stats_granularity);
    log_info(tt::LogOp, "log2_stats_granularity: {}", log2_stats_granularity);
    log_info(tt::LogOp, "sub_exp_granularity: {}", sub_exp_granularity);
    log_info(tt::LogOp, "log2_sub_exp_granularity: {}", log2_sub_exp_granularity);
    log_info(tt::LogOp, "dht_granularity: {}", dht_granularity);
    log_info(tt::LogOp, "log2_dht_granularity: {}", log2_dht_granularity);

    // Create defines for kernels
    std::map<std::string, std::string> defines;
    if (exp_approx_mode) {
        defines["EXP_APPROX_MODE"] = "1";
    }

    // Enable ring distribution in kernels (mutually exclusive with BALANCED_Q_PARALLEL)
    defines["RING_Q_DISTRIBUTION"] = "1";
    // Note: BALANCED_Q_PARALLEL should NOT be defined when using ring distribution
    // Ring distribution provides superior global load balancing across all devices

    // Add granularity defines for SFPU operations
    defines["STATS_GRANULARITY"] = std::to_string(stats_granularity);
    defines["LOG2_STATS_GRANULARITY"] = std::to_string(log2_stats_granularity);
    defines["SUB_EXP_GRANULARITY"] = std::to_string(sub_exp_granularity);
    defines["LOG2_SUB_EXP_GRANULARITY"] = std::to_string(log2_sub_exp_granularity);
    defines["DHT_GRANULARITY"] = std::to_string(dht_granularity);
    defines["LOG2_DHT_GRANULARITY"] = std::to_string(log2_dht_granularity);

    log_info(tt::LogOp, "Creating kernels with ring distribution...");

    auto reader_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_reader.cpp",
        core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, defines));

    auto writer_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_writer.cpp",
        core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, defines));

    auto compute_kernels_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/ring_sdpa.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
            .defines = defines});

    // Create circular buffers (same pattern as regular SDPA)
    tt::DataFormat q_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_q.dtype());
    tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_k.dtype());
    tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_v.dtype());
    tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());

    // Q input CB
    auto c_in0_config = CircularBufferConfig(q_tiles * tt::tt_metal::detail::TileSize(q_df), {{tt::CBIndex::c_0, q_df}})
                            .set_page_size(tt::CBIndex::c_0, tt::tt_metal::detail::TileSize(q_df));
    CreateCircularBuffer(program, core_grid, c_in0_config);

    // K input CB
    auto c_in1_config = CircularBufferConfig(k_tiles * tt::tt_metal::detail::TileSize(k_df), {{tt::CBIndex::c_1, k_df}})
                            .set_page_size(tt::CBIndex::c_1, tt::tt_metal::detail::TileSize(k_df));
    CreateCircularBuffer(program, core_grid, c_in1_config);

    // V input CB
    auto c_in2_config = CircularBufferConfig(v_tiles * tt::tt_metal::detail::TileSize(v_df), {{tt::CBIndex::c_2, v_df}})
                            .set_page_size(tt::CBIndex::c_2, tt::tt_metal::detail::TileSize(v_df));
    CreateCircularBuffer(program, core_grid, c_in2_config);

    // Causal mask CB (will be generated internally) - use BFP4_b format for mask generation
    auto c_in3_config = CircularBufferConfig(
                            qk_tiles * tt::tt_metal::detail::TileSize(tt::DataFormat::Bfp4_b),
                            {{tt::CBIndex::c_3, tt::DataFormat::Bfp4_b}})
                            .set_page_size(tt::CBIndex::c_3, tt::tt_metal::detail::TileSize(tt::DataFormat::Bfp4_b));
    CreateCircularBuffer(program, core_grid, c_in3_config);

    // Scale CB
    auto c_in5_config = CircularBufferConfig(
                            scale_tiles * tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b),
                            {{tt::CBIndex::c_5, tt::DataFormat::Float16_b}})
                            .set_page_size(tt::CBIndex::c_5, tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b));
    CreateCircularBuffer(program, core_grid, c_in5_config);

    // Identity column CB for final reduction
    auto c_in7_config = CircularBufferConfig(
                            statistics_tiles * tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b),
                            {{tt::CBIndex::c_7, tt::DataFormat::Float16_b}})
                            .set_page_size(tt::CBIndex::c_7, tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b));
    CreateCircularBuffer(program, core_grid, c_in7_config);

    // Output CB
    auto c_out_config =
        CircularBufferConfig(out0_t * tt::tt_metal::detail::TileSize(out_df), {{tt::CBIndex::c_16, out_df}})
            .set_page_size(tt::CBIndex::c_16, tt::tt_metal::detail::TileSize(out_df));
    CreateCircularBuffer(program, core_grid, c_out_config);

    // Intermediate CBs
    auto c_qk_config = CircularBufferConfig(
                           qk_tiles * tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b),
                           {{tt::CBIndex::c_24, tt::DataFormat::Float16_b}})
                           .set_page_size(tt::CBIndex::c_24, tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b));
    CreateCircularBuffer(program, core_grid, c_qk_config);

    auto c_out_im_a_config =
        CircularBufferConfig(
            out_im_tiles * tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b),
            {{tt::CBIndex::c_25, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_25, tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b));
    CreateCircularBuffer(program, core_grid, c_out_im_a_config);

    auto c_out_im_b_config =
        CircularBufferConfig(
            out_im_tiles * tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b),
            {{tt::CBIndex::c_26, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_26, tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b));
    CreateCircularBuffer(program, core_grid, c_out_im_b_config);

    // Statistics CBs
    auto c_max_a_config =
        CircularBufferConfig(
            statistics_tiles * tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b),
            {{tt::CBIndex::c_27, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_27, tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b));
    CreateCircularBuffer(program, core_grid, c_max_a_config);

    auto c_max_b_config =
        CircularBufferConfig(
            statistics_tiles * tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b),
            {{tt::CBIndex::c_28, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_28, tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b));
    CreateCircularBuffer(program, core_grid, c_max_b_config);

    auto c_sum_a_config =
        CircularBufferConfig(
            statistics_tiles * tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b),
            {{tt::CBIndex::c_29, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_29, tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b));
    CreateCircularBuffer(program, core_grid, c_sum_a_config);

    auto c_sum_b_config =
        CircularBufferConfig(
            statistics_tiles * tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b),
            {{tt::CBIndex::c_30, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_30, tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b));
    CreateCircularBuffer(program, core_grid, c_sum_b_config);

    auto c_exp_max_diff_config =
        CircularBufferConfig(
            statistics_tiles * tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b),
            {{tt::CBIndex::c_31, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_31, tt::tt_metal::detail::TileSize(tt::DataFormat::Float16_b));
    CreateCircularBuffer(program, core_grid, c_exp_max_diff_config);

    // Set runtime arguments for each core
    uint32_t q_addr = q_buffer->address();
    uint32_t k_addr = k_buffer->address();
    uint32_t v_addr = v_buffer->address();
    uint32_t out_addr = out0_buffer->address();

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        // Calculate local ranges for this core (within device's local chunks)
        uint32_t local_batch_start = (i / (nh_parallel_factor * q_parallel_factor)) * batch_per_core;
        uint32_t local_batch_end = local_batch_start + batch_per_core;
        uint32_t local_nh_start = ((i / q_parallel_factor) % nh_parallel_factor) * nh_per_core;
        uint32_t local_nh_end = local_nh_start + nh_per_core;
        uint32_t local_q_start = (i % q_parallel_factor) * local_q_per_core;  // Local chunk indices
        uint32_t local_q_end = local_q_start + local_q_per_core;

        // Clamp to valid ranges
        local_batch_start = std::min(local_batch_start, B);
        local_batch_end = std::min(local_batch_end, B);
        local_nh_start = std::min(local_nh_start, NQH);
        local_nh_end = std::min(local_nh_end, NQH);
        local_q_start = std::min(local_q_start, local_q_num_chunks);
        local_q_end = std::min(local_q_end, local_q_num_chunks);

        log_info(tt::LogOp, "Core {} runtime args:", i);
        log_info(tt::LogOp, "  batch range: [{}, {})", local_batch_start, local_batch_end);
        log_info(tt::LogOp, "  nh range: [{}, {})", local_nh_start, local_nh_end);
        log_info(tt::LogOp, "  local q range: [{}, {}) (local chunk indices)", local_q_start, local_q_end);

        SetRuntimeArgs(
            program,
            reader_kernels_id,
            core,
            {q_addr,
             k_addr,
             v_addr,
             0,  // No attention mask (causal mask generated internally)
             0,  // No page table
             i,
             local_batch_start,
             local_batch_end,
             local_nh_start,
             local_nh_end,
             local_q_start,
             local_q_end,
             0});  // No chunked offset

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
             0});

        SetRuntimeArgs(
            program,
            compute_kernels_id,
            core,
            {i, local_batch_start, local_batch_end, local_nh_start, local_nh_end, local_q_start, local_q_end, 0});
    }

    // Override runtime arguments callback (simplified - no chunked mode support for now)
    auto override_runtime_arguments_callback =
        [num_cores, grid_size, reader_kernels_id, writer_kernels_id, compute_kernels_id](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto q_buffer = input_tensors.at(0).buffer();
            auto k_buffer = input_tensors.at(1).buffer();
            auto v_buffer = input_tensors.at(2).buffer();
            auto out0_buffer = output_tensors.at(0).buffer();

            uint32_t q_addr = q_buffer->address();
            uint32_t k_addr = k_buffer->address();
            uint32_t v_addr = v_buffer->address();
            uint32_t out_addr = out0_buffer->address();

            auto& reader_args_by_core = GetRuntimeArgs(program, reader_kernels_id);
            auto& writer_args_by_core = GetRuntimeArgs(program, writer_kernels_id);

            for (uint32_t i = 0; i < num_cores; ++i) {
                CoreCoord core = {i % grid_size.x, i / grid_size.x};

                auto& reader_args = reader_args_by_core[core.x][core.y];
                auto& writer_args = writer_args_by_core[core.x][core.y];

                reader_args[0] = q_addr;
                reader_args[1] = k_addr;
                reader_args[2] = v_addr;
                writer_args[0] = out_addr;
            }
        };

    return {std::move(program), override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::transformer::detail
