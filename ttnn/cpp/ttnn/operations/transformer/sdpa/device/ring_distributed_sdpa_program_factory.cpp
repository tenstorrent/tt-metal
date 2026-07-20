// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_distributed_sdpa_device_operation.hpp"
#include "sdpa_interleaved_cb_ids.hpp"
#include "sdpa_subblock_utils.hpp"

#include <bit>
#include <cmath>
#include <map>
#include <optional>
#include <string>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// Resolve ring_id either from explicit override or by inferring from the mesh coordinate.
uint32_t resolve_ring_id(
    const RingDistributedSDPAParams& operation_attributes,
    const RingDistributedSDPAInputs& tensor_args,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    if (operation_attributes.ring_id.has_value()) {
        // Use explicitly provided ring_id
        return operation_attributes.ring_id.value();
    }

    // Infer ring_id directly from the mesh coordinate. ring_size constrains the ring to one
    // of the two mesh axes, so the index along that axis IS the ring_id - no need to look up
    // the device by coord (which uses the deprecated MeshDevice::get_device(MeshCoordinate)
    // and is multi-host-unsafe).
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "mesh_dispatch_coordinate must be provided when ring_id is not explicitly set");
    auto* mesh_device = tensor_args.q.device();
    TT_FATAL(mesh_device != nullptr, "Mesh device must not be null when inferring ring_id");

    const auto& mesh_coord = mesh_dispatch_coordinate.value();
    const auto& mesh_view = mesh_device->get_view();
    if (mesh_view.shape()[0] == operation_attributes.ring_size) {
        // Ring is along axis 0 (rows of the mesh). ring_id = row index.
        const uint32_t curr_ring_id = mesh_coord[0];
        log_debug(tt::LogOp, "Inferred ring_id (axis 0): {}", curr_ring_id);
        return curr_ring_id;
    }
    if (mesh_view.shape()[1] == operation_attributes.ring_size) {
        // Ring is along axis 1 (columns of the mesh). ring_id = column index.
        const uint32_t curr_ring_id = mesh_coord[1];
        log_debug(tt::LogOp, "Inferred ring_id (axis 1): {}", curr_ring_id);
        return curr_ring_id;
    }
    TT_FATAL(
        false,
        "Ring size {} doesn't match mesh dimensions [{}, {}]",
        operation_attributes.ring_size,
        mesh_view.shape()[0],
        mesh_view.shape()[1]);
    return 0;  // unreachable; satisfies non-void return.
}

// Ring-distributed SDPA per-coord program build. Pulled into an anonymous-namespace helper
// so create_workload_descriptor() can loop coords and reuse this body verbatim.
ProgramDescriptor build_ring_distributed_sdpa_program_descriptor(
    const RingDistributedSDPAParams& operation_attributes,
    const RingDistributedSDPAInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    /*
    Q: B x NQH x S*ring_size x DH
    K: B x NKH x DH x S
    V: B x NKH x S x DH
    */
    const Tensor& input_tensor_q = tensor_args.q;
    const Tensor& input_tensor_k = tensor_args.k;
    const Tensor& input_tensor_v = tensor_args.v;
    const Tensor& output_tensor = tensor_return_value;
    const std::optional<Tensor>& page_table = tensor_args.page_table;

    const uint32_t ring_size = operation_attributes.ring_size;
    const uint32_t ring_id = resolve_ring_id(operation_attributes, tensor_args, mesh_dispatch_coordinate);
    const std::optional<float>& scale = operation_attributes.scale;
    const std::size_t q_chunk_size =
        operation_attributes.program_config.has_value() ? operation_attributes.program_config->q_chunk_size : 32;
    const std::size_t k_chunk_size =
        operation_attributes.program_config.has_value() ? operation_attributes.program_config->k_chunk_size : 32;
    const DeviceComputeKernelConfig& compute_kernel_config = operation_attributes.compute_kernel_config;
    const std::optional<operations::transformer::SDPAProgramConfig>& program_config =
        operation_attributes.program_config;
    const std::optional<int64_t>& chunk_start_idx = operation_attributes.chunk_start_idx;

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

    auto core_grid = CoreRangeSet(CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1}));
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
    const uint32_t padded_Sq = std::ceil(static_cast<float>(Sq) / q_chunk_size) * q_chunk_size;
    const uint32_t padded_Sk = std::ceil(static_cast<float>(Sk) / k_chunk_size) * k_chunk_size;

    const uint32_t Sqt = padded_Sq / TILE_HEIGHT;
    const uint32_t Skt = padded_Sk / TILE_HEIGHT;
    const uint32_t DHt = DH / TILE_WIDTH;
    const uint32_t vDHt = DHt;

    const uint32_t valid_Sqt = std::ceil(static_cast<float>(Sq) / TILE_HEIGHT);
    const uint32_t valid_Skt = std::ceil(static_cast<float>(Sk) / TILE_HEIGHT);

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

    // Global Q scheduling: distribute the flat B*NQH*q_num_chunks Q-chunk space evenly across cores
    // (one linear range per core). Ring is always causal, so pair-distribute when q_num_chunks is
    // even — that keeps the shared zigzag remap balancing light/heavy work across cores. The same
    // (global_q_start, global_q_count) range is walked once per ring phase (num_phases=2 below).
    const uint32_t total_q_chunks = B * NQH * q_num_chunks;
    const bool global_q_pair_distribute = (q_num_chunks % 2 == 0);
    uint32_t global_q_base_chunks_per_core = 0;
    uint32_t global_q_cores_doing_extra = 0;
    uint32_t global_q_extra_chunks_per_core = 0;
    if (global_q_pair_distribute) {
        const uint32_t total_pairs = total_q_chunks / 2;
        global_q_base_chunks_per_core = (total_pairs / num_cores) * 2;
        global_q_cores_doing_extra = total_pairs % num_cores;
        global_q_extra_chunks_per_core = 2;
    } else {
        global_q_base_chunks_per_core = total_q_chunks / num_cores;
        global_q_cores_doing_extra = total_q_chunks % num_cores;
        global_q_extra_chunks_per_core = 1;
    }

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t q_tiles = Sq_chunk_t * DHt * 2;
    uint32_t k_tiles = Sk_chunk_t * DHt * 2;            // double buffer
    uint32_t v_tiles = Sk_chunk_t * vDHt * 2;           // double buffer
    uint32_t mask_tiles = 2;                            // lightweight: neginf + causal diagonal
    uint32_t qk_tiles = Sq_chunk_t * Sk_chunk_t;
    uint32_t out_im_tiles = Sq_chunk_t * vDHt;
    uint32_t out0_t = Sq_chunk_t * vDHt;
    uint32_t scale_tiles = 1;
    uint32_t statistics_tiles = Sq_chunk_t;  // Single column of values in each iteration

    // log all values

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

    // Determine granularity for statistics computation
    // Each granularity must evenly divide its tile count to avoid dropping tiles
    const uint32_t stats_granularity = detail::find_valid_granularity(Sq_chunk_t, dst_size);
    const uint32_t sub_exp_granularity = detail::find_valid_granularity(Sk_chunk_t, dst_size);
    const uint32_t mul_bcast_granularity = detail::find_valid_granularity(Sq_chunk_t * Sk_chunk_t, dst_size);
    const uint32_t dht_granularity = detail::find_valid_granularity(DHt, dst_size);
    const uint32_t reduce_granularity = detail::find_valid_granularity(Sq_chunk_t, dst_size / 2);

    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    class bfloat16 bfloat_identity_scalar(1.0f);
    uint32_t packed_identity_scalar = pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});

    const uint32_t scale_packed = std::bit_cast<uint32_t>(scale.value_or(1.0f));

    constexpr bool use_zigzag_balancing = true;

    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        B,
        NQH,
        NKH,
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
        true,                               //(std::uint32_t)is_causal,
        false,                              //(std::uint32_t)use_provided_mask,
        false,                              //(std::uint32_t)broadcast_provided_mask_batch,
        false,                              //(std::uint32_t)broadcast_provided_mask_heads,
        false,                              //(std::uint32_t)use_padded_mask,
        static_cast<uint32_t>(is_chunked),  //(uint32_t)is_chunked,
        block_size_t,
        page_table_stick_size,
        0,                  // use_attention_sink
        0,                  // use_mla
        0,                  // mla_kv_overlap
        qk_out_subblock_h,  // qk_subblock_h
        0,                  // sliding_window_size (ring uses no sliding window)
        0                   // use_streaming_compute (ring uses legacy compute)
    };
    // Semaphore placeholders (not used in ring, but kernel expects them at indices 29-32)
    reader_compile_time_args.push_back(0);  // sender_semaphore_id
    reader_compile_time_args.push_back(0);  // receiver_semaphore_id
    reader_compile_time_args.push_back(0);  // valid_semaphore_id
    reader_compile_time_args.push_back(0);  // mcast_enabled
    reader_compile_time_args.push_back(static_cast<uint32_t>(use_zigzag_balancing));  // arg 33

    TensorAccessorArgs(input_tensor_q.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs().append_to(reader_compile_time_args);  // mask tensor (not used in ring)
    TensorAccessorArgs(page_table.has_value() ? page_table->buffer() : nullptr)
        .append_to(reader_compile_time_args);                  // page table
    TensorAccessorArgs().append_to(reader_compile_time_args);  // attention sink (not used in ring)
    TensorAccessorArgs().append_to(reader_compile_time_args);  // chunk_start_idx_tensor (ring has no flexible chunked)

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
        scale_packed,
        num_cores,
        true,   //(std::uint32_t)is_causal,
        false,  //(std::uint32_t)use_provided_mask,
        false,  //(std::uint32_t)use_padded_mask,
        true,   //(uint32_t)is_chunked,
        0,      //(uint32_t)sliding_window_size,
        1,      // arg 20: lightweight causal mask
        0,      // arg 21: use_streaming_compute — always false for ring distributed (causal)
        0,      // arg 22: out_subblock_h — unused when streaming is off
        0,      // arg 23: k_partial_col — non-streaming, no partial mask emitted
        static_cast<uint32_t>(use_zigzag_balancing),  // arg 24
        0,  // arg 25: use_windowed_mask — ring never uses windowed (block-diagonal) attention
        0,  // arg 26: return_lse — ring never emits LSE (mirrors the regular factory arg layout)
    };
    // out accessor, then the cu_window accessor, then the LSE accessor chained right after it (mirrors
    // the regular factory so the writer's accessor offset chain stays intact). Ring is never windowed and
    // never emits LSE → nullptr placeholder accessors.
    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs().append_to(writer_compile_time_args);
    TensorAccessorArgs().append_to(writer_compile_time_args);

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
        scale_packed,
        0,          //(uint32_t)sliding_window_size,
        0,          //(std::uint32_t)use_attention_sink,
        0,          //(std::uint32_t)use_streaming_compute - always false for ring distributed (causal)
        valid_Skt,  // arg 31: unpadded K tiles for streaming padded_k_tiles
        0u,         // arg 32: k_partial_col - unused on ring's non-streaming path
        static_cast<uint32_t>(use_zigzag_balancing),  // arg 33: unified zigzag remap
        0,  // arg 34: return_lse — ring never emits LSE (mirrors the regular factory arg layout)
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

    tt::DataFormat mask_df = tt::DataFormat::Float16_b;
    tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::DataFormat scalar_df =
        (input_tensor_q.dtype() == DataType::FLOAT32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat im_df = tt::DataFormat::Float16_b;  // need to disable fp32 cbs (Issue #13364) fp32_dest_acc_en ?
                                                       // tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat stats_df = im_df;
    // salad_correct_fused inits mul_bcast_cols with out CB and applies it to sum CB too —
    // both must share the same data format for the unpack config to be correct.
    TT_ASSERT(im_df == stats_df, "SDPA fused SALAD correction requires out and sum CBs to share data format");

    uint32_t q_tile_size = tt::tile_size(q_df);
    uint32_t k_tile_size = tt::tile_size(k_df);
    uint32_t v_tile_size = tt::tile_size(v_df);
    uint32_t mask_tile_size = tt::tile_size(mask_df);
    uint32_t out_tile_size = tt::tile_size(out_df);
    uint32_t scalar_tile_size = tt::tile_size(scalar_df);
    uint32_t im_tile_size = tt::tile_size(im_df);
    uint32_t stats_tile_size = tt::tile_size(stats_df);

    sdpa_cb::CBIds cb_ids;
    uint32_t next_cb_index = 0;
    const auto allocate_cb = [&](uint32_t page_size_bytes, uint32_t num_pages, tt::DataFormat data_format) -> uint32_t {
        const uint32_t cb_index = next_cb_index++;
        desc.cbs.push_back(CBDescriptor{
            .total_size = page_size_bytes * num_pages,
            .core_ranges = core_grid,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_index),
                .data_format = data_format,
                .page_size = page_size_bytes,
            }}},
        });
        return cb_index;
    };
    const auto allocate_tile_cb = [&](uint32_t num_tiles, uint32_t tile_size, tt::DataFormat data_format) -> uint32_t {
        return allocate_cb(tile_size, num_tiles, data_format);
    };

    cb_ids.q_in = allocate_tile_cb(q_tiles, q_tile_size, q_df);
    // Ring is never windowed, but the writer's windowed block (gated by `if constexpr`) is still compiled
    // in a non-template function, so get_tile_size(cb_cu_window_in) must resolve to a valid CB id rather
    // than the `inactive` sentinel (which constexpr-faults on unpack_tile_size[-1]). Mirror the regular
    // factory and point it at q_in.
    cb_ids.cu_window_seqlens = cb_ids.q_in;
    cb_ids.k_in = allocate_tile_cb(k_tiles, k_tile_size, k_df);
    cb_ids.v_in = allocate_tile_cb(v_tiles, v_tile_size, v_df);
    cb_ids.mask_in = allocate_tile_cb(mask_tiles, mask_tile_size, mask_df);
    cb_ids.identity_scale_in = allocate_tile_cb(scale_tiles, scalar_tile_size, scalar_df);
    cb_ids.col_identity = allocate_tile_cb(scale_tiles, scalar_tile_size, scalar_df);

    // page table circular buffer (only when using paged KV)
    if (is_chunked) {
        cb_ids.page_table = allocate_cb(page_table_stick_size, 1, page_table_df);
    }

    cb_ids.qk_im = allocate_tile_cb(qk_tiles, im_tile_size, im_df);
    cb_ids.out_im_A = allocate_tile_cb(out_im_tiles, im_tile_size, im_df);
    cb_ids.out_im_B = allocate_tile_cb(out_im_tiles, im_tile_size, im_df);
    cb_ids.max_A = allocate_tile_cb(statistics_tiles, stats_tile_size, stats_df);
    cb_ids.max_B = allocate_tile_cb(statistics_tiles, stats_tile_size, stats_df);
    cb_ids.sum_A = allocate_tile_cb(statistics_tiles, stats_tile_size, stats_df);
    cb_ids.sum_B = allocate_tile_cb(statistics_tiles, stats_tile_size, stats_df);
    cb_ids.exp_max_diff = allocate_tile_cb(statistics_tiles, stats_tile_size, stats_df);
    cb_ids.out = allocate_tile_cb(out0_t, out_tile_size, out_df);
    // T6: ring never emits LSE, but the shared writer/compute kernels reference cb_lse_out/cb_scale_in
    // in their (runtime-discarded) return_lse branches. Point them at valid ids so get_tile_size/format
    // lookups stay well-formed, mirroring the regular factory's fallback.
    cb_ids.lse_out = cb_ids.out;
    cb_ids.scale_in = cb_ids.identity_scale_in;

    const auto reader_cb_compile_time_args = cb_ids.reader_compile_time_args();
    const auto writer_cb_compile_time_args = cb_ids.writer_compile_time_args();
    const auto compute_cb_compile_time_args = cb_ids.compute_compile_time_args();
    reader_compile_time_args.insert(
        reader_compile_time_args.end(), reader_cb_compile_time_args.begin(), reader_cb_compile_time_args.end());
    writer_compile_time_args.insert(
        writer_compile_time_args.end(), writer_cb_compile_time_args.begin(), writer_cb_compile_time_args.end());
    compute_compile_time_args.insert(
        compute_compile_time_args.end(), compute_cb_compile_time_args.begin(), compute_cb_compile_time_args.end());
    TensorAccessorArgs(output_tensor.buffer()).append_to(compute_compile_time_args);

    // ---- Kernels ----

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_grid;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = defines;
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/writer_interleaved.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_grid;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.defines = defines;
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp";
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

    // page_table address must be patched on cache hit; use BufferBinding.
    auto* page_table_buffer = page_table.has_value() ? page_table->buffer() : nullptr;

    // Set reader rt args
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        uint32_t read_offset_phase_1 = chunk_1 * Sqt;
        uint32_t read_offset_phase_2 = chunk_2 * Sqt;
        uint32_t write_offset_phase_1 = 0;
        uint32_t write_offset_phase_2 = Sqt;

        // Per-core slice of the flat B*NQH*q_num_chunks Q-chunk space (walked once per phase).
        uint32_t global_q_start = i * global_q_base_chunks_per_core +
                                  std::min(i, global_q_cores_doing_extra) * global_q_extra_chunks_per_core;
        uint32_t global_q_count =
            global_q_base_chunks_per_core + ((i < global_q_cores_doing_extra) ? global_q_extra_chunks_per_core : 0u);
        if (global_q_start >= total_q_chunks) {
            global_q_start = total_q_chunks;
            global_q_count = 0;
        } else if (global_q_start + global_q_count > total_q_chunks) {
            global_q_count = total_q_chunks - global_q_start;
        }

        reader_desc.emplace_runtime_args(
            core,
            {q_buffer,
             k_buffer,
             v_buffer,
             static_cast<Buffer*>(nullptr),  // mask
             page_table_buffer,
             static_cast<Buffer*>(nullptr),  // attention_sink
             static_cast<Buffer*>(nullptr),  // chunk_start_idx (ring has none)
             i,
             2u,
             chunked_q_chunk_offset_phase_1,
             read_offset_phase_1,
             chunked_q_chunk_offset_phase_2,
             read_offset_phase_2,
             global_q_start,
             global_q_count});

        writer_desc.emplace_runtime_args(
            core,
            {out0_buffer,
             i,
             2u,
             0u,  // use_chunk_start_idx_tensor (ring has no chunk_start_idx_tensor)
             chunked_q_chunk_offset_phase_1,
             write_offset_phase_1,
             chunked_q_chunk_offset_phase_2,
             write_offset_phase_2,
             global_q_start,
             global_q_count,
             0u,    // arg 10: cu_window_seqlens_addr — unused (ring is never windowed)
             0u,    // arg 11: cu_window_seqlens_eles — unused (ring is never windowed)
             0u});  // arg 12: lse_addr — unused (ring never emits LSE); kernel reads it unconditionally

        compute_desc.emplace_runtime_args(
            core,
            {i,
             2u,
             0u,  // use_chunk_start_idx_tensor (ring has no chunk_start_idx_tensor)
             chunked_q_chunk_offset_phase_1,
             chunked_q_chunk_offset_phase_2,
             global_q_start,
             global_q_count});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace

// Ring-distributed SDPA returns a WorkloadDescriptor with one ProgramDescriptor per coord:
// ring_id is inferred from the coord, so each coord builds a distinct descriptor.
WorkloadDescriptor RingDistributedSdpaDeviceOperation::RingDistributedSdpaProgramFactory::create_workload_descriptor(
    const RingDistributedSDPAParams& operation_attributes,
    const RingDistributedSDPAInputs& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    WorkloadDescriptor wd;
    const auto coords = tensor_coords.coords();
    wd.programs.reserve(coords.size());
    for (const auto& coord : coords) {
        auto desc = build_ring_distributed_sdpa_program_descriptor(
            operation_attributes, tensor_args, tensor_return_value, coord);
        wd.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return wd;
}

}  // namespace ttnn::prim
