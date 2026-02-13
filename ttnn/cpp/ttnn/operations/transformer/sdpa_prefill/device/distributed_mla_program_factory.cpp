// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "distributed_mla_program_factory.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/math.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <optional>
#include <string>
#include <cmath>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::transformer::sdpa_prefill {

DistributedMlaMeshWorkloadFactory::cached_mesh_workload_t DistributedMlaMeshWorkloadFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

DistributedMlaMeshWorkloadFactory::cached_program_t DistributedMlaMeshWorkloadFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // Calculate device order using physical coordinate (using the Q tensor and cluster axis)
    uint32_t device_order = ttnn::ccl::get_linearized_index_from_physical_coord(
        tensor_args.q, mesh_coordinate, operation_attributes.cluster_axis);

    log_info(
        tt::LogOp,
        "Device at coordinate has linearized index: {} (cluster_axis: {})",
        device_order,
        operation_attributes.cluster_axis.has_value() ? operation_attributes.cluster_axis.value() : 0);

    // Extract tensors from inputs (similar to original SDPA)
    const auto& input_tensor_q = tensor_args.q;
    const auto& input_tensor_k = tensor_args.k;
    const auto& input_tensor_v = tensor_args.v;
    const auto& output_tensor = tensor_return_value;
    const auto& attn_mask = tensor_args.attn_mask;
    const auto& page_table = tensor_args.page_table;
    const auto& attention_sink = tensor_args.attention_sink;

    // Extract operation attributes (similar to original SDPA)
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

    // Extract chunk sizes from program config (like original SDPA)
    std::size_t q_chunk_size = program_config ? program_config->q_chunk_size : 32;
    std::size_t k_chunk_size = program_config ? program_config->k_chunk_size : 32;

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

    // Paged cache parameters when in chunked mode (copied from original SDPA)
    const bool flexible_chunked = tensor_args.chunk_start_idx_tensor.has_value();
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

    // Calculate padded sequence length (copied from original SDPA)
    const uint32_t padded_Sq = std::ceil((float)Sq / q_chunk_size) * q_chunk_size;
    const uint32_t padded_Sk = std::ceil((float)Sk / k_chunk_size) * k_chunk_size;

    const uint32_t Sqt = padded_Sq / TILE_HEIGHT;
    const uint32_t Skt = padded_Sk / TILE_HEIGHT;
    const uint32_t DHt = DH / TILE_WIDTH;
    const uint32_t vDHt = use_mla ? head_dim_v / TILE_WIDTH : DHt;

    const uint32_t valid_Sqt = std::ceil((float)Sq / TILE_HEIGHT);
    const uint32_t valid_Skt = std::ceil((float)Sk / TILE_HEIGHT);
    const bool use_padded_mask = (!is_causal) && ((padded_Sk != Sk) || (padded_Sq != Sq));

    const uint32_t Sq_chunk_t = q_chunk_size / TILE_HEIGHT;
    const uint32_t Sk_chunk_t = k_chunk_size / TILE_HEIGHT;
    const uint32_t q_num_chunks = padded_Sq / q_chunk_size;  // This is already per device
    const uint32_t k_num_chunks = padded_Sk / k_chunk_size;
    const bool use_provided_mask = attn_mask.has_value();
    const bool broadcast_provided_mask_batch = use_provided_mask ? (attn_mask.value().logical_shape()[0] == 1) : false;
    const bool broadcast_provided_mask_heads = use_provided_mask ? (attn_mask.value().logical_shape()[1] == 1) : false;

    // Log debug information (copied from original SDPA)
    log_debug(tt::LogOp, "B: {}", B);
    log_debug(tt::LogOp, "NQH: {}", NQH);
    log_debug(tt::LogOp, "Sq: {}", Sq);
    log_debug(tt::LogOp, "Sk: {}", Sk);
    log_debug(tt::LogOp, "padded_Sq: {}", padded_Sq);
    log_debug(tt::LogOp, "padded_Sk: {}", padded_Sk);
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

    // KEY DIFFERENCE: Calculate chunked Q offset based on device order
    // Since Q is already sharded per device, q_num_chunks is already per device
    // Each device processes its own q_num_chunks, and starts from device_order * q_num_chunks offset
    uint32_t chunked_q_chunk_offset = device_order * q_num_chunks;

    // Add prefix offset if in chunked mode (like original SDPA)
    if (is_chunked && chunk_start_idx.has_value()) {
        chunked_q_chunk_offset += chunk_start_idx.value() / q_chunk_size;
    }

    log_info(
        tt::LogOp,
        "Device {}: Q offset = {}, Q chunks per device = {}",
        device_order,
        chunked_q_chunk_offset,
        q_num_chunks);

    // Paged cache setup (copied from original SDPA)
    uint32_t block_size = 0;
    uint32_t block_size_t = 0;
    [[maybe_unused]] uint32_t max_blocks_per_seq = 0;
    uint32_t page_table_stick_size = 0;
    tt::DataFormat page_table_df = tt::DataFormat::Int32;

    if (is_chunked) {
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

    // Parallelization scheme (copied from original SDPA)
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

    // Circular buffer tile capacity calculations (copied from original SDPA)
    uint32_t q_tiles = Sq_chunk_t * DHt * q_buffer_factor;
    uint32_t k_tiles = Sk_chunk_t * DHt * 2;            // double buffer
    uint32_t v_tiles = Sk_chunk_t * vDHt * 2;           // double buffer
    uint32_t mask_tiles = Sq_chunk_t * Sk_chunk_t * 2;  // double buffer
    uint32_t qk_tiles = Sq_chunk_t * Sk_chunk_t;
    uint32_t out_im_tiles = Sq_chunk_t * vDHt;
    uint32_t out0_t = Sq_chunk_t * vDHt;
    uint32_t scale_tiles = 1;
    uint32_t statistics_tiles = Sq_chunk_t;                               // Single column of values in each iteration
    uint32_t attention_sink_tiles = use_attention_sink ? Sq_chunk_t : 0;  // One column vector per Q chunk

    // Matmul configuration calculations (copied from original SDPA)
    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
    const uint32_t qk_in0_block_w = DHt;
    uint32_t qk_out_subblock_w = std::min(Sk_chunk_t, dst_size);
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

    // Granularity calculations for compute operations (copied from original SDPA)
    const uint32_t stats_granularity = std::min(Sq_chunk_t, dst_size);
    const uint32_t log2_stats_granularity = std::log2(stats_granularity);
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

    // Compile-time arguments setup (copied from original SDPA)
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
                                                      (std::uint32_t)use_attention_sink};

    // Placeholder semaphore IDs (for potential future use)
    reader_compile_time_args.push_back(0);  // sender_semaphore_id placeholder
    reader_compile_time_args.push_back(0);  // receiver_semaphore_id placeholder
    reader_compile_time_args.push_back(0);  // valid_semaphore_id placeholder

    TensorAccessorArgs(input_tensor_q.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(attn_mask.has_value() ? attn_mask->buffer() : nullptr).append_to(reader_compile_time_args);
    TensorAccessorArgs(page_table.has_value() ? page_table->buffer() : nullptr).append_to(reader_compile_time_args);
    TensorAccessorArgs(attention_sink.has_value() ? attention_sink->buffer() : nullptr)
        .append_to(reader_compile_time_args);
    TensorAccessorArgs(flexible_chunked ? tensor_args.chunk_start_idx_tensor.value().buffer() : nullptr)
        .append_to(reader_compile_time_args);

    // KEY: Force is_chunked = true for writer to enable offset application (like ring distributed)
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
        true,  // FORCE is_chunked = true to enable offset application
        sliding_window_size.value_or(0),
    };

    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);

    // KEY: Force is_chunked = true for compute to enable offset application (like ring distributed)
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
        true,  // FORCE is_chunked = true to enable offset application
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
    uint32_t balanced_q_parallel =
        (is_causal && (q_per_core * q_parallel_factor == q_num_chunks) && (q_per_core % 2 == 0));
    if (balanced_q_parallel) {
        defines["BALANCED_Q_PARALLEL"] = "1";
    }

    log_debug(tt::LogOp, "BALANCED_Q_PARALLEL: {}", balanced_q_parallel);

    // Create kernels (using original SDPA kernels)
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

    // Create circular buffers (copied from original SDPA)
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
        auto c_in4_config = CircularBufferConfig(attention_sink_tiles * sink_tile_size, {{tt::CBIndex::c_4, sink_df}})
                                .set_page_size(tt::CBIndex::c_4, sink_tile_size);
        CreateCircularBuffer(program, core_grid, c_in4_config);
    }

    // Intermediate circular buffers
    auto c_intermed0_config = CircularBufferConfig(qk_tiles * im_tile_size, {{tt::CBIndex::c_24, im_df}})
                                  .set_page_size(tt::CBIndex::c_24, im_tile_size);
    CreateCircularBuffer(program, core_grid, c_intermed0_config);

    auto c_intermed1_config = CircularBufferConfig(out_im_tiles * im_tile_size, {{tt::CBIndex::c_25, im_df}})
                                  .set_page_size(tt::CBIndex::c_25, im_tile_size);
    CreateCircularBuffer(program, core_grid, c_intermed1_config);

    auto c_intermed2_config = CircularBufferConfig(out_im_tiles * im_tile_size, {{tt::CBIndex::c_26, im_df}})
                                  .set_page_size(tt::CBIndex::c_26, im_tile_size);
    CreateCircularBuffer(program, core_grid, c_intermed2_config);

    auto c_intermed3_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_27, stats_df}})
                                  .set_page_size(tt::CBIndex::c_27, stats_tile_size);
    CreateCircularBuffer(program, core_grid, c_intermed3_config);

    auto c_intermed4_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_28, stats_df}})
                                  .set_page_size(tt::CBIndex::c_28, stats_tile_size);
    CreateCircularBuffer(program, core_grid, c_intermed4_config);

    auto c_intermed5_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_29, stats_df}})
                                  .set_page_size(tt::CBIndex::c_29, stats_tile_size);
    CreateCircularBuffer(program, core_grid, c_intermed5_config);

    auto c_intermed6_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_30, stats_df}})
                                  .set_page_size(tt::CBIndex::c_30, stats_tile_size);
    CreateCircularBuffer(program, core_grid, c_intermed6_config);

    auto c_intermed7_config = CircularBufferConfig(statistics_tiles * stats_tile_size, {{tt::CBIndex::c_31, stats_df}})
                                  .set_page_size(tt::CBIndex::c_31, stats_tile_size);
    CreateCircularBuffer(program, core_grid, c_intermed7_config);

    // Output
    auto c_out0_config = CircularBufferConfig(out0_t * out_tile_size, {{tt::CBIndex::c_16, out_df}})
                             .set_page_size(tt::CBIndex::c_16, out_tile_size);
    CreateCircularBuffer(program, core_grid, c_out0_config);

    // Set up runtime arguments for all cores
    uint32_t q_addr = q_buffer->address();
    uint32_t k_addr = k_buffer->address();
    uint32_t v_addr = v_buffer->address();
    uint32_t mask_addr = attn_mask.has_value() ? mask_buffer->address() : 0;
    uint32_t attention_sink_addr = attention_sink.has_value() ? attention_sink_buffer->address() : 0;
    uint32_t out_addr = out0_buffer->address();

    uint32_t num_phases = 1;
    uint32_t read_offset = 0;
    uint32_t write_offset = 0;

    // Runtime arguments setup for all cores (simplified from original SDPA)
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

        std::vector<uint32_t> reader_args = {
            q_addr,
            k_addr,
            v_addr,
            mask_addr,
            0,  // page_table_addr (will be filled if needed)
            attention_sink_addr,
            0,  // chunk_start_idx_addr (will be filled if needed)
            i,
            local_batch_start,
            local_batch_end,
            local_nh_start,
            local_nh_end,
            local_q_start,
            local_q_end,
            num_phases,
            chunked_q_chunk_offset,
            read_offset};

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
             write_offset});
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

    // Store device order in shared variables
    shared_variables_t shared_vars{};
    shared_vars.device_order = device_order;

    return cached_program_t(std::move(program), std::move(shared_vars));
}

void DistributedMlaMeshWorkloadFactory::override_runtime_arguments(
    [[maybe_unused]] cached_mesh_workload_t& cached_workload,
    [[maybe_unused]] const operation_attributes_t& operation_attributes,
    [[maybe_unused]] const tensor_args_t& tensor_args,
    [[maybe_unused]] tensor_return_value_t& tensor_return_value) {
    // TODO: Implement runtime argument updates if needed
}

}  // namespace ttnn::operations::transformer::sdpa_prefill
