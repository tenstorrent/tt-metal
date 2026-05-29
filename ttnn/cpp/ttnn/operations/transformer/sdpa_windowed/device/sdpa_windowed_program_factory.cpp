// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa_windowed/device/sdpa_windowed_device_operation.hpp"

#include <bit>
#include <cmath>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

// Implementation of windowed SDPA
// [INFO] This implementation is based on the multi-core implementation of SDPA in the tt-metal repository.
// read: Because the cu_window_seqlens tensor is small, we naively load the whole tensor into a circular buffer on each
// core and then use the window indexes to compute tiles of attn_masks.
// compute and write: mostly the same as the multi-core implementation of SDPA.
// [INFO] a natural thought for potentially faster implementation is to only compute the SDPA within each windown as
// defined by cu_window_seqlens; this should be driven by performance analysis results of whole ML model
ProgramDescriptor WindowedScaledDotProductAttentionDeviceOperation::WindowedSDPAProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor_q = tensor_args.q;
    const auto& input_tensor_k = tensor_args.k;
    const auto& input_tensor_v = tensor_args.v;
    const auto& cu_window_seqlens = tensor_args.cu_window_seqlens;
    auto& output_tensor = tensor_return_value;
    auto scale = operation_attributes.scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.padded_shape()[-1]));
    }

    const auto& program_config = operation_attributes.program_config;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;
    std::size_t q_chunk_size =
        operation_attributes.program_config ? operation_attributes.program_config->q_chunk_size : 32;
    std::size_t k_chunk_size =
        operation_attributes.program_config ? operation_attributes.program_config->k_chunk_size : 32;
    /*
    Q: B x NQH x S x DH
    K: B x NKH x S x DH
    V: B x NKH x S x DH

    cu_window_seqlens defines windows for attention computation
    */

    const auto& q_shape = input_tensor_q.logical_shape();
    const auto& k_shape = input_tensor_k.logical_shape();
    const uint32_t B = q_shape[0], NQH = q_shape[1], Sq = q_shape[2], DH = q_shape[3];
    const uint32_t NKH = k_shape[1];
    const uint32_t Sk = k_shape[2];

    // Calculate padded sequence lengths
    const uint32_t padded_Sq = std::ceil(static_cast<float>(Sq) / q_chunk_size) * q_chunk_size;
    const uint32_t padded_Sk = std::ceil(static_cast<float>(Sk) / k_chunk_size) * k_chunk_size;

    const uint32_t Sqt = padded_Sq / TILE_HEIGHT;
    const uint32_t Skt = padded_Sk / TILE_HEIGHT;
    const uint32_t DHt = DH / TILE_WIDTH;

    const uint32_t valid_Sqt = std::ceil(static_cast<float>(Sq) / TILE_HEIGHT);
    const uint32_t valid_Skt = std::ceil(static_cast<float>(Sk) / TILE_HEIGHT);

    const uint32_t Sq_chunk_t = q_chunk_size / TILE_HEIGHT;
    const uint32_t Sk_chunk_t = k_chunk_size / TILE_HEIGHT;
    const uint32_t q_num_chunks = padded_Sq / q_chunk_size;
    const uint32_t k_num_chunks = padded_Sk / k_chunk_size;

    const uint32_t cu_window_seqlens_eles = cu_window_seqlens.logical_shape()[0];

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
    log_debug(tt::LogOp, "cu_windows_seqlens.size(): {}", cu_window_seqlens_eles);

    auto* device = input_tensor_q.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    auto* q_buffer = input_tensor_q.buffer();
    auto* k_buffer = input_tensor_k.buffer();
    auto* v_buffer = input_tensor_v.buffer();
    auto* cu_window_seqlens_buffer = cu_window_seqlens.buffer();
    auto* out0_buffer = output_tensor.buffer();

    CoreCoord grid_size = program_config.has_value() ? program_config->compute_with_storage_grid_size
                                                     : device->compute_with_storage_grid_size();
    bool exp_approx_mode =
        program_config.has_value()
            ? (program_config->exp_approx_mode.has_value() ? program_config->exp_approx_mode.value() : true)
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

    log_debug(tt::LogOp, "batch_per_core: {}", batch_per_core);
    log_debug(tt::LogOp, "nh_per_core: {}", nh_per_core);
    log_debug(tt::LogOp, "q_per_core: {}", q_per_core);

    // Tile capacity counts for CBs
    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t q_tiles = Sq_chunk_t * DHt * q_buffer_factor;
    uint32_t k_tiles = Sk_chunk_t * DHt * 2;  // double buffer
    uint32_t v_tiles = Sk_chunk_t * DHt * 2;  // double buffer
    uint32_t qk_ntiles_per_chunk = Sq_chunk_t * Sk_chunk_t;
    uint32_t out_im_tiles = Sq_chunk_t * DHt;
    uint32_t out0_t = Sq_chunk_t * DHt;
    uint32_t scale_tiles = 1;
    uint32_t statistics_tiles = Sq_chunk_t;  // Single column of values in each iteration

    // log all values
    log_debug(tt::LogOp, "q_tiles: {}", q_tiles);
    log_debug(tt::LogOp, "k_tiles: {}", k_tiles);
    log_debug(tt::LogOp, "v_tiles: {}", v_tiles);
    log_debug(tt::LogOp, "qk_tiles_per_chunk: {}", qk_ntiles_per_chunk);
    log_debug(tt::LogOp, "out0_t: {}", out0_t);
    log_debug(tt::LogOp, "scale_tiles: {}", scale_tiles);
    log_debug(tt::LogOp, "statistics_tiles: {}", statistics_tiles);
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
    const uint32_t log2_stats_granularity = static_cast<uint32_t>(std::log2(stats_granularity));
    TT_FATAL(
        stats_granularity == (1u << log2_stats_granularity),
        "stats_granularity must be a power of 2. Got {}.",
        stats_granularity);

    const uint32_t sub_exp_granularity = std::min(Sk_chunk_t, dst_size);
    const uint32_t log2_sub_exp_granularity = static_cast<uint32_t>(std::log2(sub_exp_granularity));
    TT_FATAL(
        sub_exp_granularity == (1u << log2_sub_exp_granularity),
        "sub_exp_granularity must be a power of 2. Got {}.",
        sub_exp_granularity);

    const uint32_t mul_bcast_granularity = std::min(Sq_chunk_t * Sk_chunk_t, dst_size);
    const uint32_t log2_mul_bcast_granularity = static_cast<uint32_t>(std::log2(mul_bcast_granularity));
    TT_FATAL(
        mul_bcast_granularity == (1u << log2_mul_bcast_granularity),
        "mul_bcast_granularity must be a power of 2. Got {}.",
        mul_bcast_granularity);

    uint32_t dht_granularity = std::min(DHt, dst_size);
    uint32_t log2_dht_granularity = static_cast<uint32_t>(std::log2(dht_granularity));
    // Sometimes DHt is not a power of 2, so granularity should be 1
    if (dht_granularity != (1u << log2_dht_granularity)) {
        dht_granularity = 1;
        log2_dht_granularity = 0;
    }
    TT_FATAL(
        dht_granularity == (1u << log2_dht_granularity),
        "dht_granularity must be a power of 2. Got {}.",
        dht_granularity);

    // Reduce ops can use granularity of dst_size/2
    const uint32_t reduce_granularity = std::min(Sq_chunk_t, dst_size / 2);
    const uint32_t log2_reduce_granularity = static_cast<uint32_t>(std::log2(reduce_granularity));
    TT_FATAL(
        reduce_granularity == (1u << log2_reduce_granularity),
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

    const uint32_t scale_packed = std::bit_cast<uint32_t>(scale.value_or(1.0f));

    std::vector<uint32_t> reader_compile_time_args = {
        B,
        NQH,
        NKH,
        Sqt,
        Skt,
        valid_Sqt,
        valid_Skt,
        DHt,
        Sq_chunk_t,
        Sk_chunk_t,
        num_cores,
    };
    TensorAccessorArgs(*q_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*k_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*v_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*cu_window_seqlens_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        B,
        NQH,
        Sqt,
        valid_Sqt,
        DHt,
        Sq_chunk_t,
        q_num_chunks,
        Sk_chunk_t,
        packed_identity_scalar,
        num_cores,
    };
    TensorAccessorArgs(*out0_buffer).append_to(writer_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {
        Skt,
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
        scale_packed,
    };

    KernelDescriptor::Defines defines;
    defines.emplace_back("STATS_GRANULARITY", std::to_string(stats_granularity));
    defines.emplace_back("LOG2_STATS_GRANULARITY", std::to_string(log2_stats_granularity));
    defines.emplace_back("SUB_EXP_GRANULARITY", std::to_string(sub_exp_granularity));
    defines.emplace_back("LOG2_SUB_EXP_GRANULARITY", std::to_string(log2_sub_exp_granularity));
    defines.emplace_back("MUL_BCAST_GRANULARITY", std::to_string(mul_bcast_granularity));
    defines.emplace_back("LOG2_MUL_BCAST_GRANULARITY", std::to_string(log2_mul_bcast_granularity));
    defines.emplace_back("DHT_GRANULARITY", std::to_string(dht_granularity));
    defines.emplace_back("LOG2_DHT_GRANULARITY", std::to_string(log2_dht_granularity));
    defines.emplace_back("REDUCE_GRANULARITY", std::to_string(reduce_granularity));
    defines.emplace_back("LOG2_REDUCE_GRANULARITY", std::to_string(log2_reduce_granularity));
    defines.emplace_back("EXP_APPROX_MODE", std::to_string(static_cast<uint32_t>(exp_approx_mode)));

    // Create circular buffers
    tt::DataFormat q_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_q.dtype());
    tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_k.dtype());
    tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_v.dtype());
    tt::DataFormat cu_window_seqlens_df = tt::tt_metal::datatype_to_dataformat_converter(cu_window_seqlens.dtype());
    tt::DataFormat mask_df = tt::DataFormat::Bfp4_b;
    tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::DataFormat scalar_df =
        (input_tensor_q.dtype() == DataType::FLOAT32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat im_df = tt::DataFormat::Float16_b;  // need to disable fp32 cbs (Issue #13364) fp32_dest_acc_en ?
                                                       // tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat stats_df = im_df;

    uint32_t q_tile_size = tt::tile_size(q_df);
    uint32_t k_tile_size = tt::tile_size(k_df);
    uint32_t v_tile_size = tt::tile_size(v_df);
    uint32_t out_tile_size = tt::tile_size(out_df);
    uint32_t scalar_tile_size = tt::tile_size(scalar_df);
    uint32_t im_tile_size = tt::tile_size(im_df);
    uint32_t stats_tile_size = tt::tile_size(stats_df);
    uint32_t cu_window_seqlens_tile_size = tt::tile_size(cu_window_seqlens_df);

    log_debug(tt::LogOp, "q_data_format: {}", q_df);
    log_debug(tt::LogOp, "k_data_format: {}", k_df);
    log_debug(tt::LogOp, "v_data_format: {}", v_df);
    log_debug(tt::LogOp, "mask_data_format: {}", mask_df);
    log_debug(tt::LogOp, "out_data_format: {}", out_df);
    log_debug(tt::LogOp, "scalar_data_format: {}", scalar_df);
    log_debug(tt::LogOp, "intermediate_data_format: {}", im_df);
    log_debug(tt::LogOp, "statistics_data_format: {}", stats_df);

    ProgramDescriptor desc;

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

    // cu_window_seqlens input
    // [INFO] cu_window_seqlens is a small 1D tensor, so we can set the page size to the size of all the tiles
    uint32_t cu_window_seqlens_page_size = cu_window_seqlens_tile_size;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cu_window_seqlens_page_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_3),
            .data_format = cu_window_seqlens_df,
            .page_size = cu_window_seqlens_page_size,
        }}},
    });

    // cb_mask_in
    uint32_t mask_tile_size = tt::tile_size(mask_df);
    uint32_t mask_ntiles = qk_ntiles_per_chunk * 2;  // double buffer
    desc.cbs.push_back(CBDescriptor{
        .total_size = mask_ntiles * mask_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_4),
            .data_format = mask_df,
            .page_size = mask_tile_size,
        }}},
    });

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

    desc.cbs.push_back(CBDescriptor{
        .total_size = scale_tiles * scalar_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_7),
            .data_format = scalar_df,
            .page_size = scalar_tile_size,
        }}},
    });

    // Intermediate buffers
    // cb_qk_im
    desc.cbs.push_back(CBDescriptor{
        .total_size = qk_ntiles_per_chunk * im_tile_size,
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

    // Reader kernel
    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa_windowed/device/kernels/dataflow/reader_windowed.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_grid;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = defines;
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer kernel
    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa_windowed/device/kernels/dataflow/writer_windowed.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_grid;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.defines = defines;
    writer_desc.config = WriterConfigDescriptor{};

    // Compute kernel
    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa_windowed/device/kernels/compute/sdpa_windowed.cpp";
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

    // Set per-core runtime args
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
            {
                q_buffer,
                k_buffer,
                v_buffer,
                cu_window_seqlens_buffer,
                cu_window_seqlens_eles,
                i,
                local_batch_start,
                local_batch_end,
                local_nh_start,
                local_nh_end,
                local_q_start,
                local_q_end,
            });

        // Writer args
        writer_desc.emplace_runtime_args(
            core,
            {
                out0_buffer,
                i,
                local_batch_start,
                local_batch_end,
                local_nh_start,
                local_nh_end,
                local_q_start,
                local_q_end,
            });

        // Compute args
        compute_desc.emplace_runtime_args(
            core,
            {
                i,
                local_batch_start,
                local_batch_end,
                local_nh_start,
                local_nh_end,
                local_q_start,
                local_q_end,
            });
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
