// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/sliding_window_geometry.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/math.hpp"
#include <hostdevcommon/common_values.hpp>
#include <bit>
#include <map>
#include <optional>
#include <string>
#include <cmath>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

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

namespace {

// Select the mask data format: user-provided mask dtype, or Float16_b for streaming (avoids Bfp4_b precision loss),
// or Bfp4_b for legacy path.
tt::DataFormat select_mask_dataformat(const std::optional<Tensor>& attn_mask, bool use_streaming_compute) {
    if (attn_mask.has_value()) {
        return tt::tt_metal::datatype_to_dataformat_converter(attn_mask.value().dtype());
    }
    return use_streaming_compute ? tt::DataFormat::Float16_b : tt::DataFormat::Bfp4_b;
}

// Streaming compute (v2) handles every SDPA variant; only fp32 dest-accumulate falls back to the
// legacy compute kernel.
bool can_use_streaming_compute(bool fp32_dest_acc_en) { return !fp32_dest_acc_en; }

uint32_t lightweight_mask_tile_count(bool is_causal, bool has_sliding_window, bool has_k_partial_mask) {
    uint32_t tiles = 1;  // neginf
    if (has_sliding_window) {
        tiles += kSlidingWindowEdgeTiles;
    } else if (is_causal) {
        tiles++;  // causal diagonal
    }
    if (has_k_partial_mask) {
        tiles++;  // partial K tile
    }
    return tiles;
}

// Compute the largest granularity that evenly divides both DHt and vDHt (up to dst_size).
uint32_t compute_dht_granularity(uint32_t DHt, uint32_t vDHt, uint32_t dst_size) {
    uint32_t g = std::min({DHt, vDHt, dst_size});
    while (g > 1 && (DHt % g != 0 || vDHt % g != 0)) {
        g--;
    }
    return g;
}

// Resolve exp_approx_mode from program config, defaulting to true.
bool get_exp_approx_mode(const std::optional<ttnn::operations::transformer::SDPAProgramConfig>& program_config) {
    if (program_config.has_value() && program_config->exp_approx_mode.has_value()) {
        return program_config->exp_approx_mode.value();
    }
    return true;
}

// Effective (num_kv_heads_k, num_kv_heads_v, block_size) for an HMA-shared paged buffer.
// Apply PagedCacheGeometryOverride only when !use_mla: MLA never passes overrides (validated
// upstream), and applying num_kv_heads to V under MLA would skip the elems/block check.
struct EffectiveKvGeometry {
    uint32_t nkh = 0;
    uint32_t nvh = 0;
    uint32_t block_size = 0;
};

EffectiveKvGeometry resolve_effective_kv_geometry(
    const ttnn::operations::transformer::PagedCacheGeometryOverride& geo,
    bool use_mla,
    uint32_t k_num_heads,
    uint32_t v_num_heads,
    uint32_t k_block_size) {
    if (use_mla || !geo.active()) {
        return {k_num_heads, v_num_heads, k_block_size};
    }
    return {geo.num_kv_heads, geo.num_kv_heads, geo.block_size};
}

// Chunked prefill parameters collected from page table layout.
struct ChunkedParams {
    uint32_t chunked_q_chunk_offset = 0;
    uint32_t block_size = 0;
    uint32_t block_size_t = 0;
    uint32_t max_blocks_per_seq = 0;
    uint32_t page_table_stick_size = 0;
};

// Compute chunked prefill parameters from the page table tensor.
ChunkedParams compute_chunked_params(
    bool is_chunked,
    bool is_chunked_legacy,
    bool flexible_chunked,
    const std::optional<int64_t>& chunk_start_idx,
    const std::optional<Tensor>& page_table,
    uint32_t k_seq_dim,
    std::size_t q_chunk_size) {
    ChunkedParams p;
    if (!is_chunked) {
        return p;
    }
    if (is_chunked_legacy) {
        p.chunked_q_chunk_offset = chunk_start_idx.value() / q_chunk_size;
    }
    const auto& page_table_tensor = page_table.value();
    p.block_size = k_seq_dim;
    p.block_size_t = p.block_size / TILE_HEIGHT;
    if (flexible_chunked) {
        p.max_blocks_per_seq = page_table_tensor.padded_shape()[1];
        p.page_table_stick_size = p.max_blocks_per_seq * sizeof(int32_t);
        TT_FATAL(p.page_table_stick_size % 32 == 0, "page table stick size must be a multiple of 32");
    } else {
        p.max_blocks_per_seq = page_table_tensor.padded_shape()[1];
        p.page_table_stick_size = page_table_tensor.buffer()->aligned_page_size();
        TT_FATAL(
            p.page_table_stick_size % 32 == 0,
            "page table page size in bytes must be a multiple of 32 due to address alignment");
    }
    return p;
}

tt::DataFormat fp32_dest_intermediate_dataformat(bool fp32_dest_acc_en) {
    return fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
}

uint32_t attention_sink_tile_count(bool use_attention_sink, bool use_streaming_compute, uint32_t q_chunk_tiles) {
    if (!use_attention_sink) {
        return 0;
    }
    return use_streaming_compute ? 1 : q_chunk_tiles;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts SDPAOperation::SDPAProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // Windowed (block-diagonal) attention reuses the regular reader/writer/compute kernels. The mask is
    // synthesized on-device in the writer from cu_window_seqlens (reader streams Q/K/V only) and consumed
    // by the compute via the provided-mask path. Like regular SDPA it honors the streaming-vs-standard
    // selection: streaming kernel when fp32_dest_acc_en is false (Blackhole default), standard otherwise.
    const bool is_windowed = operation_attributes.is_windowed;
    const auto& input_tensor_q = tensor_args.q;
    const auto& input_tensor_k = tensor_args.k;
    const auto& input_tensor_v = tensor_args.v.value_or(tensor_args.k);
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
    const auto& program_config = operation_attributes.program_config;
    const bool use_mla = operation_attributes.use_mla;
    const bool mla_kv_overlap = use_mla && !tensor_args.v.has_value();
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
    const auto& v_shape = input_tensor_v.logical_shape();
    const uint32_t B = q_shape[0], NQH = q_shape[1], Sq = q_shape[2], DH = q_shape[3];
    // Geometry overrides for an HMA-shared paged buffer (see PagedCacheGeometryOverride): when
    // the paged K/V cache was allocated for a different layer's view, the reader must address it
    // with this call's num_kv_heads / block_size (Q already drives head_dim via DHt) rather than
    // the cache's declared shape. Unset ⇒ the cache's own num_kv_heads / block_size. The reader
    // computes physical tile ids manually from these as compile-time args
    // (dataflow_common.hpp virtual_seq_tile_id_to_physical_tile_id).
    const auto kv_geo = resolve_effective_kv_geometry(
        operation_attributes.paged_cache_geometry, use_mla, k_shape[1], v_shape[1], k_shape[2]);
    const uint32_t NKH = kv_geo.nkh;
    const uint32_t NVH = kv_geo.nvh;
    const uint32_t effective_kv_block_size = kv_geo.block_size;

    // In flash mla prefill, we have to support the case where NKH != NVH
    // We are calling op with the following shapes:
    // q - [B, NHQ, Sq, DH_qk]
    // k - [B, 1, Sk, DH_qk]
    // v - [B, NVH, Sk, DH_v]
    // k head is in latent space, and is reused across all q heads

    // Paged cache parameters when in chunked mode
    const bool flexible_chunked = operation_attributes.chunk_start_idx_tensor.has_value();
    const bool is_chunked_legacy = chunk_start_idx.has_value() && !flexible_chunked;
    const bool is_chunked = is_chunked_legacy || flexible_chunked;
    // For flexible chunked: max prefix length = page_table num_pages * block_size (from K/V layout).
    uint32_t max_prefix_tokens_flexible = 0;
    if (is_chunked && flexible_chunked) {
        const uint32_t block_size_for_sk = effective_kv_block_size;
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
    const uint32_t padded_Sq = std::ceil(static_cast<float>(Sq) / q_chunk_size) * q_chunk_size;
    const uint32_t padded_Sk = std::ceil(static_cast<float>(Sk) / k_chunk_size) * k_chunk_size;

    const uint32_t Sqt = padded_Sq / TILE_HEIGHT;
    const uint32_t Skt = padded_Sk / TILE_HEIGHT;
    const uint32_t DHt = DH / TILE_WIDTH;
    const uint32_t vDHt = use_mla ? head_dim_v / TILE_WIDTH : DHt;

    const uint32_t valid_Sqt = std::ceil(static_cast<float>(Sq) / TILE_HEIGHT);
    const uint32_t valid_Skt = std::ceil(static_cast<float>(Sk) / TILE_HEIGHT);
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
    // Windowed mode synthesizes the mask in the writer; the compute consumes it through the provided-mask
    // path even though there is no attn_mask tensor (and the reader does not read one).
    const bool compute_use_provided_mask = use_provided_mask || is_windowed;
    // Windowed masks are complete dense masks synthesized from cu_window_seqlens. They already cover padding
    // positions outside the final boundary, so the generic generated-padding-mask paths must stay disabled.
    const bool generated_padding_mask = use_padded_mask && !is_windowed;

    // log_debug all of the above
    log_debug(tt::LogOp, "B: {}", B);
    log_debug(tt::LogOp, "NQH: {}", NQH);
    log_debug(tt::LogOp, "NVH: {}", NVH);
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

    const auto chunked = compute_chunked_params(
        is_chunked,
        is_chunked_legacy,
        flexible_chunked,
        chunk_start_idx,
        page_table,
        effective_kv_block_size,
        q_chunk_size);
    const uint32_t chunked_q_chunk_offset = chunked.chunked_q_chunk_offset;
    const uint32_t block_size = chunked.block_size;
    const uint32_t block_size_t = chunked.block_size_t;
    [[maybe_unused]] const uint32_t max_blocks_per_seq = chunked.max_blocks_per_seq;
    const uint32_t page_table_stick_size = chunked.page_table_stick_size;
    const tt::DataFormat page_table_df = tt::DataFormat::Int32;
    // Log page table info
    log_debug(tt::LogOp, "is_chunked: {}", is_chunked);
    if (is_chunked) {
        log_debug(tt::LogOp, "block_size: {}", block_size);
        log_debug(tt::LogOp, "block_size_t: {}", block_size_t);
        log_debug(tt::LogOp, "max_blocks_per_seq: {}", max_blocks_per_seq);
        log_debug(tt::LogOp, "page_table_stick_size: {}", page_table_stick_size);
        log_debug(tt::LogOp, "page_table_df: {}", page_table_df);
    }

    IDevice* device = input_tensor_q.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    bool use_attention_sink = attention_sink.has_value();

    CoreCoord grid_size = program_config.has_value() ? program_config->compute_with_storage_grid_size
                                                     : device->compute_with_storage_grid_size();
    const bool exp_approx_mode = get_exp_approx_mode(program_config);

    auto core_grid = CoreRangeSet(CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1}));
    uint32_t num_cores = grid_size.x * grid_size.y;

    TT_FATAL(
        num_cores <= device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y,
        "Provided grid must not contain more cores than the device. Got {} cores, expected at most {} cores.",
        num_cores,
        device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y);

    TT_FATAL(num_cores > 0, "SDPA requires a non-empty core grid; got num_cores=0.");

    // Global Q scheduling is the single-chip default: distribute the flat B*NQH*q_num_chunks
    // Q-chunk space evenly across cores. Pair-distribute when causal + even q_num_chunks so every
    // core gets balanced light/heavy work after the shared zigzag remap (CT 31/24/34 to kernels).
    const uint32_t total_q_chunks = B * NQH * q_num_chunks;
    const bool global_q_pair_distribute = is_causal && (q_num_chunks % 2 == 0);
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
    const uint32_t max_global_q_chunks_per_core =
        global_q_base_chunks_per_core + (global_q_cores_doing_extra > 0 ? global_q_extra_chunks_per_core : 0);

    const uint32_t q_buffer_factor = (max_global_q_chunks_per_core > 1) ? 2 : 1;

    // Host code is responsible for determining matmul configuration
    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
    const uint32_t qk_in0_block_w = DHt;

    auto [qk_out_subblock_h, qk_out_subblock_w] =
        detail::determine_largest_subblock_size(Sq_chunk_t, Sk_chunk_t, dst_size);

    const bool use_streaming_compute = can_use_streaming_compute(fp32_dest_acc_en);

    const bool has_sliding_window = sliding_window_size.value_or(0) != 0;
    // A user-provided dense mask on the streaming path takes its own per-chunk apply
    // (apply_provided_mask_streaming) and must win over the structured lightweight palette: forcing
    // lightweight_mask false (via !use_provided_mask below) routes mask_in sizing/dtype to the
    // full Sq×Sk provided-mask branch instead of the 1–4-tile palette.
    const bool lightweight_causal = is_causal && !use_provided_mask && !is_windowed && !has_sliding_window;
    const bool lightweight_streaming_mask = use_streaming_compute && !use_provided_mask && !is_windowed &&
                                            (is_causal || has_sliding_window || generated_padding_mask);
    const bool lightweight_mask = lightweight_causal || lightweight_streaming_mask;
    // Non-causal partial-tile K (Sk % TILE != 0) needs a partial-tile mask in mask_in.
    // Not used for a dense provided mask (the reader neginf-fills padded positions in the mask).
    const uint32_t k_partial_col =
        (use_streaming_compute && generated_padding_mask && !use_provided_mask && (Sk % TILE_HEIGHT != 0))
            ? (Sk % TILE_HEIGHT)
            : 0;
    const bool lw_partial_active = (k_partial_col > 0);
    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t q_tiles = Sq_chunk_t * DHt * q_buffer_factor;
    uint32_t k_tiles = Sk_chunk_t * DHt * 2;   // double buffer
    uint32_t v_tiles = Sk_chunk_t * vDHt * 2;  // double buffer
    uint32_t mask_tiles = lightweight_mask
                              ? lightweight_mask_tile_count(is_causal, has_sliding_window, lw_partial_active)
                              : Sq_chunk_t * Sk_chunk_t * 2;  // double buffer
    uint32_t qk_tiles = Sq_chunk_t * Sk_chunk_t;
    uint32_t out_im_tiles = Sq_chunk_t * vDHt;
    uint32_t out0_t = Sq_chunk_t * vDHt;  // finalized below once out_out_subblock_h is known
    uint32_t scale_tiles = 1;
    uint32_t statistics_tiles = Sq_chunk_t;  // Single column of values in each iteration
    // Streaming compute broadcasts the per-head scalar directly; legacy compute consumes one
    // expanded first-column tile per Q row.
    uint32_t attention_sink_tiles = attention_sink_tile_count(use_attention_sink, use_streaming_compute, Sq_chunk_t);

    // log all values
    log_debug(tt::LogOp, "q_tiles: {}", q_tiles);
    log_debug(tt::LogOp, "k_tiles: {}", k_tiles);
    log_debug(tt::LogOp, "v_tiles: {}", v_tiles);
    log_debug(tt::LogOp, "mask_tiles: {}", mask_tiles);
    log_debug(tt::LogOp, "qk_tiles: {}", qk_tiles);
    log_debug(tt::LogOp, "scale_tiles: {}", scale_tiles);
    log_debug(tt::LogOp, "statistics_tiles: {}", statistics_tiles);
    log_debug(tt::LogOp, "attention_sink_tiles: {}", attention_sink_tiles);

    const uint32_t qk_in0_num_subblocks = Sq_chunk_t / qk_out_subblock_h;
    const uint32_t qk_in1_num_subblocks = Sk_chunk_t / qk_out_subblock_w;
    const uint32_t qk_num_blocks = DHt / qk_in0_block_w;

    // now for out0
    const uint32_t out_in0_block_w = Sk_chunk_t;

    auto [out_out_subblock_h, out_out_subblock_w] =
        detail::determine_largest_subblock_size(Sq_chunk_t, vDHt, dst_size, use_streaming_compute ? 2 : UINT32_MAX);

    const uint32_t out_in0_num_subblocks = Sq_chunk_t / out_out_subblock_h;
    const uint32_t out_in1_num_subblocks = vDHt / out_out_subblock_w;
    const uint32_t out_num_blocks = Sk_chunk_t / out_in0_block_w;

    // Streaming: shrink the out DFB to a 2-slot ping-pong (see sdpa_subblock_utils.hpp).
    if (use_streaming_compute) {
        out0_t = detail::streaming_cb_out_tiles(out_out_subblock_h, out_out_subblock_w, dst_size, Sq_chunk_t, vDHt);
        TT_FATAL(
            Sq_chunk_t % out_out_subblock_h == 0,
            "Streaming out drain requires Sq_chunk_t ({}) divisible by out_out_subblock_h ({})",
            Sq_chunk_t,
            out_out_subblock_h);
    }
    log_debug(tt::LogOp, "out0_t: {}", out0_t);
    log_debug(tt::LogOp, "use_streaming_compute: {}", use_streaming_compute);

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
    const uint32_t dht_granularity = compute_dht_granularity(DHt, vDHt, dst_size);
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

    const uint32_t scale_packed = std::bit_cast<uint32_t>(scale.value_or(1.0f));

    const bool use_zigzag_balancing = is_causal;

    // ---- Metal 2.0 named resources ----

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    const DFBSpecName Q_IN{"q_in"};
    const DFBSpecName K_IN{"k_in"};
    const DFBSpecName V_IN{"v_in"};
    const DFBSpecName MASK_IN{"mask_in"};
    const DFBSpecName CU_WINDOW{"cu_window_seqlens"};
    const DFBSpecName IDENTITY_SCALE_IN{"identity_scale_in"};
    const DFBSpecName COL_IDENTITY{"col_identity"};
    const DFBSpecName PAGE_TABLE{"page_table"};
    const DFBSpecName CHUNK_START_IDX_COMPUTE{"chunk_start_idx_compute"};
    const DFBSpecName CHUNK_START_IDX_WRITER{"chunk_start_idx_writer"};
    const DFBSpecName ATTENTION_SINK{"attention_sink"};
    const DFBSpecName RECIP_SCRATCH{"recip_scratch"};
    const DFBSpecName QK_IM{"qk_im"};
    const DFBSpecName OUT_IM_A{"out_im_A"};
    const DFBSpecName OUT_IM_B{"out_im_B"};
    const DFBSpecName MAX_A{"max_A"};
    const DFBSpecName MAX_B{"max_B"};
    const DFBSpecName SUM_A{"sum_A"};
    const DFBSpecName SUM_B{"sum_B"};
    const DFBSpecName EXP_MAX_DIFF{"exp_max_diff"};
    const DFBSpecName OUT{"out"};
    // Windowed K-range narrowing (#54492): the reader's own cu_window copy, the {k_lo,k_hi} ctrl CB
    // (reader -> compute), and the per-device Q-offset tensor CB.
    const DFBSpecName WINDOWED_CU_READER{"windowed_cu_reader"};
    const DFBSpecName WINDOWED_K_RANGE{"windowed_k_range"};
    const DFBSpecName WINDOWED_Q_OFFSET{"windowed_q_offset"};

    const TensorParamName T_Q_IN{"q_in"};
    const TensorParamName T_K_IN{"k_in"};
    const TensorParamName T_V_IN{"v_in"};
    const TensorParamName T_MASK{"mask"};
    const TensorParamName T_PAGE_TABLE{"page_table"};
    const TensorParamName T_ATTENTION_SINK{"attention_sink"};
    const TensorParamName T_CHUNK_START_IDX{"chunk_start_idx"};
    const TensorParamName T_OUT{"out"};
    const TensorParamName T_CU_WINDOW{"cu_window_seqlens"};
    const TensorParamName T_WINDOWED_Q_OFFSET{"windowed_q_token_offset"};

    const SemaphoreSpecName SEM_SENDER{"sender"};
    const SemaphoreSpecName SEM_RECEIVER{"receiver"};
    const SemaphoreSpecName SEM_VALID{"valid"};

    // Conditional-binding predicates (mirror the legacy CB-allocation / kernel-touch conditions).
    const bool kv_chain = !is_causal;
    const bool needs_mask_cb =
        use_provided_mask || is_causal || generated_padding_mask || sliding_window_size.value_or(0) > 0 || is_windowed;
    const bool writer_produces_mask = needs_mask_cb && !use_provided_mask;

    // Shared kernel defines (granularities + exp approx mode).
    KernelSpec::CompilerOptions::Defines base_defines;
    base_defines.insert({"STATS_GRANULARITY", std::to_string(stats_granularity)});
    base_defines.insert({"SUB_EXP_GRANULARITY", std::to_string(sub_exp_granularity)});
    base_defines.insert({"MUL_BCAST_GRANULARITY", std::to_string(mul_bcast_granularity)});
    base_defines.insert({"DHT_GRANULARITY", std::to_string(dht_granularity)});
    base_defines.insert({"REDUCE_GRANULARITY", std::to_string(reduce_granularity)});
    base_defines.insert({"EXP_APPROX_MODE", std::to_string(exp_approx_mode)});
    log_debug(tt::LogOp, "use_zigzag_balancing: {}", use_zigzag_balancing);

    tt::DataFormat q_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_q.dtype());
    tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_k.dtype());
    tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_v.dtype());
    // Windowed mask is generated on-device. Float16_b so it works on both the streaming path (which does
    // not decode block-float masks) and the standard path; windowed_mask_gen.hpp fills the right format.
    tt::DataFormat mask_df =
        is_windowed ? tt::DataFormat::Float16_b : select_mask_dataformat(attn_mask, use_streaming_compute);
    tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::DataFormat scalar_df =
        (input_tensor_q.dtype() == DataType::FLOAT32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat im_df =
        tt::DataFormat::Float16_b;  // Keep most intermediates in bf16 to save L1; opt-in fp32 per-CB below.
    tt::DataFormat stats_df = im_df;
    tt::DataFormat qk_im_df = fp32_dest_intermediate_dataformat(fp32_dest_acc_en);
    tt::DataFormat sum_df = fp32_dest_intermediate_dataformat(fp32_dest_acc_en);
    // salad_correct_fused inits mul_bcast_cols with out CB and applies it to sum CB too —
    // both must share the same data format for the unpack config to be correct.
    TT_ASSERT(
        !use_streaming_compute || sum_df == im_df,
        "SDPA fused SALAD correction requires out and sum CBs to share data format");

    uint32_t q_tile_size = tt::tile_size(q_df);
    uint32_t k_tile_size = tt::tile_size(k_df);
    uint32_t v_tile_size = tt::tile_size(v_df);
    uint32_t out_tile_size = tt::tile_size(out_df);
    uint32_t scalar_tile_size = tt::tile_size(scalar_df);
    uint32_t im_tile_size = tt::tile_size(im_df);
    uint32_t stats_tile_size = tt::tile_size(stats_df);
    uint32_t qk_im_tile_size = tt::tile_size(qk_im_df);
    uint32_t sum_tile_size = tt::tile_size(sum_df);

    log_debug(tt::LogOp, "q_data_format: {}", q_df);
    log_debug(tt::LogOp, "k_data_format: {}", k_df);
    log_debug(tt::LogOp, "v_data_format: {}", v_df);
    log_debug(tt::LogOp, "mask_data_format: {}", mask_df);
    log_debug(tt::LogOp, "out_data_format: {}", out_df);
    log_debug(tt::LogOp, "scalar_data_format: {}", scalar_df);
    log_debug(tt::LogOp, "intermediate_data_format: {}", im_df);
    log_debug(tt::LogOp, "statistics_data_format: {}", stats_df);
    log_debug(tt::LogOp, "qk_im_data_format: {}", qk_im_df);
    log_debug(tt::LogOp, "sum_data_format: {}", sum_df);

    // ---- Dataflow buffers ----

    Group<DataflowBufferSpec> dfbs = {
        DataflowBufferSpec{
            .unique_id = Q_IN, .entry_size = q_tile_size, .num_entries = q_tiles, .data_format_metadata = q_df},
        DataflowBufferSpec{
            .unique_id = K_IN, .entry_size = k_tile_size, .num_entries = k_tiles, .data_format_metadata = k_df},
        DataflowBufferSpec{
            .unique_id = V_IN, .entry_size = v_tile_size, .num_entries = v_tiles, .data_format_metadata = v_df},
        DataflowBufferSpec{
            .unique_id = IDENTITY_SCALE_IN,
            .entry_size = scalar_tile_size,
            .num_entries = scale_tiles,
            .data_format_metadata = scalar_df},
        DataflowBufferSpec{
            .unique_id = COL_IDENTITY,
            .entry_size = scalar_tile_size,
            .num_entries = scale_tiles,
            .data_format_metadata = scalar_df},
        DataflowBufferSpec{
            .unique_id = QK_IM,
            .entry_size = qk_im_tile_size,
            .num_entries = qk_tiles,
            .data_format_metadata = qk_im_df},
        DataflowBufferSpec{
            .unique_id = OUT_IM_A,
            .entry_size = im_tile_size,
            .num_entries = out_im_tiles,
            .data_format_metadata = im_df},
        DataflowBufferSpec{
            .unique_id = OUT_IM_B,
            .entry_size = im_tile_size,
            .num_entries = out_im_tiles,
            .data_format_metadata = im_df},
        DataflowBufferSpec{
            .unique_id = MAX_A,
            .entry_size = stats_tile_size,
            .num_entries = statistics_tiles,
            .data_format_metadata = stats_df},
        DataflowBufferSpec{
            .unique_id = MAX_B,
            .entry_size = stats_tile_size,
            .num_entries = statistics_tiles,
            .data_format_metadata = stats_df},
        DataflowBufferSpec{
            .unique_id = SUM_A,
            .entry_size = sum_tile_size,
            .num_entries = statistics_tiles,
            .data_format_metadata = sum_df},
        DataflowBufferSpec{
            .unique_id = SUM_B,
            .entry_size = sum_tile_size,
            .num_entries = statistics_tiles,
            .data_format_metadata = sum_df},
        DataflowBufferSpec{
            .unique_id = EXP_MAX_DIFF,
            .entry_size = stats_tile_size,
            .num_entries = statistics_tiles,
            .data_format_metadata = stats_df},
        DataflowBufferSpec{
            .unique_id = OUT, .entry_size = out_tile_size, .num_entries = out0_t, .data_format_metadata = out_df},
    };
    if (needs_mask_cb) {
        // Lightweight mask: Float16_b palette; legacy: full Sq×Sk matrix in mask_df.
        const tt::DataFormat actual_mask_df = lightweight_mask ? tt::DataFormat::Float16_b : mask_df;
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = MASK_IN,
            .entry_size = tt::tile_size(actual_mask_df),
            .num_entries = mask_tiles,
            .data_format_metadata = actual_mask_df});
    }
    // Windowed (block-diagonal) runtime values and #54492 K-range-narrowing buffers.
    uint32_t cu_window_seqlens_eles = 0;
    const uint32_t windowed_q_token_offset = operation_attributes.windowed_q_token_offset;
    const bool windowed_q_offset_present = tensor_args.windowed_q_token_offset_tensor.has_value();
    if (is_windowed) {
        const auto& cu = tensor_args.cu_window_seqlens.value();
        const tt::DataFormat cu_df = tt::tt_metal::datatype_to_dataformat_converter(cu.dtype());
        // Writer's cu_window copy (windowed-mask generation).
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = CU_WINDOW,
            .entry_size = tt::tile_size(cu_df),
            .num_entries = 1,
            .data_format_metadata = cu_df});
        cu_window_seqlens_eles = cu.logical_shape()[-1];
        // K-range narrowing (#54492): the reader's OWN cu_window copy (a second producer on the writer's
        // CB is illegal), plus a small reader->compute ctrl CB carrying each Q chunk's {k_lo, k_hi},
        // double-buffered so the reader can run a Q chunk ahead.
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = WINDOWED_CU_READER,
            .entry_size = tt::tile_size(cu_df),
            .num_entries = 1,
            .data_format_metadata = cu_df});
        constexpr uint32_t k_range_page_size = 16;
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = WINDOWED_K_RANGE,
            .entry_size = k_range_page_size,
            .num_entries = 2,
            .data_format_metadata = tt::DataFormat::Int32});
        // Per-device Q-offset tensor CB (only when the offset arrives as a tensor).
        if (windowed_q_offset_present) {
            const auto& off = tensor_args.windowed_q_token_offset_tensor.value();
            const tt::DataFormat off_df = tt::tt_metal::datatype_to_dataformat_converter(off.dtype());
            dfbs.push_back(DataflowBufferSpec{
                .unique_id = WINDOWED_Q_OFFSET,
                .entry_size = tt::tile_size(off_df),
                .num_entries = 1,
                .data_format_metadata = off_df});
        }
    }
    if (is_chunked) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = PAGE_TABLE,
            .entry_size = page_table_stick_size,
            .num_entries = 1,
            .data_format_metadata = page_table_df});
    }
    if (flexible_chunked) {
        constexpr uint32_t chunk_start_idx_page_size = 32;
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = CHUNK_START_IDX_COMPUTE,
            .entry_size = chunk_start_idx_page_size,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::Int32});
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = CHUNK_START_IDX_WRITER,
            .entry_size = chunk_start_idx_page_size,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::Int32});
    }
    if (use_attention_sink) {
        const tt::DataFormat sink_df = tt::tt_metal::datatype_to_dataformat_converter(attention_sink.value().dtype());
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = ATTENTION_SINK,
            .entry_size = tt::tile_size(sink_df),
            .num_entries = attention_sink_tiles,
            .data_format_metadata = sink_df});
    }
    if (use_streaming_compute) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = RECIP_SCRATCH, .entry_size = im_tile_size, .num_entries = 1, .data_format_metadata = im_df});
    }

    // ---- Tensor parameters ----

    Group<TensorParameter> tensor_params = {
        TensorParameter{.unique_id = T_Q_IN, .spec = input_tensor_q.tensor_spec()},
        TensorParameter{.unique_id = T_K_IN, .spec = input_tensor_k.tensor_spec()},
        TensorParameter{.unique_id = T_V_IN, .spec = input_tensor_v.tensor_spec()},
        TensorParameter{.unique_id = T_OUT, .spec = output_tensor.tensor_spec()},
    };
    if (use_provided_mask) {
        tensor_params.push_back(TensorParameter{.unique_id = T_MASK, .spec = attn_mask.value().tensor_spec()});
    }
    if (is_chunked) {
        tensor_params.push_back(TensorParameter{.unique_id = T_PAGE_TABLE, .spec = page_table.value().tensor_spec()});
    }
    if (use_attention_sink) {
        tensor_params.push_back(
            TensorParameter{.unique_id = T_ATTENTION_SINK, .spec = attention_sink.value().tensor_spec()});
    }
    if (flexible_chunked) {
        tensor_params.push_back(TensorParameter{
            .unique_id = T_CHUNK_START_IDX, .spec = tensor_args.chunk_start_idx_tensor.value().tensor_spec()});
    }
    if (is_windowed) {
        tensor_params.push_back(
            TensorParameter{.unique_id = T_CU_WINDOW, .spec = tensor_args.cu_window_seqlens.value().tensor_spec()});
        if (windowed_q_offset_present) {
            tensor_params.push_back(TensorParameter{
                .unique_id = T_WINDOWED_Q_OFFSET,
                .spec = tensor_args.windowed_q_token_offset_tensor.value().tensor_spec()});
        }
    }

    // ---- Semaphores (KV chain forwarding, non-causal only) ----
    // sender / receiver default to 0 (INVALID); valid initializes to VALID (non-zero: WH/BH only, a
    // deprecated capability slated for removal, mirrored here to preserve the legacy op's behavior).
    Group<SemaphoreSpec> sems;
    if (!is_causal) {
        sems.push_back(SemaphoreSpec{.unique_id = SEM_SENDER, .target_nodes = core_grid});
        sems.push_back(SemaphoreSpec{.unique_id = SEM_RECEIVER, .target_nodes = core_grid});
        SemaphoreSpec valid_sem{.unique_id = SEM_VALID, .target_nodes = core_grid};
        valid_sem.advanced_options.initial_value = VALID;
        sems.push_back(valid_sem);
    }

    uint32_t num_phases = 1;
    uint32_t read_offset = 0;
    uint32_t write_offset = 0;

    // Defense-in-depth: kernels place global_q runtime args past the max phase_2 slot,
    // but the host-side compute/writer arg packing only zeros the phase_2 slots when
    // num_phases==1. Any future change that raises num_phases on this path must rethink
    // the layout — assert early so the failure is loud rather than a slot reinterpretation.
    TT_FATAL(num_phases == 1, "Single-chip SDPA assumes num_phases == 1 under global Q scheduling");

    // Build chain topology for KV forwarding (non-causal only)
    std::vector<CoreWork> core_work(num_cores);
    std::vector<CoreChainInfo> core_chain_info(num_cores);
    const uint32_t total_heads = B * NQH;
    std::vector<std::vector<HeadSegmentRef>> head_segments;
    uint32_t mcast_chains = 0;

    // Windowed is excluded like sliding-window: both narrow the per-Q-chunk K range, and chains
    // lock-step-forward K between cores whose Q chunks now need DIFFERENT K ranges — the semaphore
    // handshake counts diverge and the cores deadlock. Narrowing saves far more K reads than
    // forwarding did.
    if (!is_causal && !is_chunked && !has_sliding_window && !is_windowed) {
        head_segments.resize(total_heads);

        log_debug(tt::LogOp, "=== Building KV chain forwarding topology ===");
        log_debug(tt::LogOp, "Total heads (B * NQH): {}", total_heads);
        log_debug(tt::LogOp, "Q chunks per head: {}", q_num_chunks);
        log_debug(tt::LogOp, "Grid size: {}x{} = {} cores", grid_size.x, grid_size.y, num_cores);

        // First pass: Record work distribution for each core
        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord core = {i % grid_size.x, i / grid_size.x};

            auto& work = core_work[i];
            work.logical_core = core;
            work.physical_core = device->worker_core_from_logical_core(core);

            auto push_head_work = [&](uint32_t nb, uint32_t nh, uint32_t q_start, uint32_t q_count) {
                if (q_count == 0) {
                    return;
                }
                work.head_work.push_back(CoreHeadWork{
                    .batch = nb,
                    .head = nh,
                    .q_chunk_start = q_start,
                    .q_chunk_count = q_count,
                });
                const uint32_t head_id = (nb * NQH) + nh;
                if (head_id < head_segments.size()) {
                    head_segments[head_id].push_back(HeadSegmentRef{
                        .core_idx = i, .head_work_index = static_cast<uint32_t>(work.head_work.size() - 1)});
                }
            };

            // Walk the core's [g_start, g_start + g_count) linear range and split into
            // contiguous (nb, nq, q_chunk_range) segments. Non-causal here (chain section is
            // !is_causal), so the zigzag remap is off and the decompose is identity.
            uint32_t g_start = i * global_q_base_chunks_per_core +
                               std::min(i, global_q_cores_doing_extra) * global_q_extra_chunks_per_core;
            uint32_t g_count = global_q_base_chunks_per_core +
                               ((i < global_q_cores_doing_extra) ? global_q_extra_chunks_per_core : 0u);
            if (g_start >= total_q_chunks) {
                g_start = total_q_chunks;
                g_count = 0;
            } else if (g_start + g_count > total_q_chunks) {
                g_count = total_q_chunks - g_start;
            }
            work.global_q_start = g_start;
            work.global_q_count = g_count;

            uint32_t cursor = g_start;
            const uint32_t g_end = g_start + g_count;
            while (cursor < g_end) {
                const uint32_t nb = cursor / (NQH * q_num_chunks);
                const uint32_t nq = (cursor / q_num_chunks) % NQH;
                const uint32_t q_in_head = cursor % q_num_chunks;
                const uint32_t remaining_in_head = q_num_chunks - q_in_head;
                const uint32_t remaining_in_range = g_end - cursor;
                const uint32_t span = std::min(remaining_in_head, remaining_in_range);
                push_head_work(nb, nq, q_in_head, span);
                cursor += span;
            }

            if (!work.head_work.empty()) {
                log_debug(
                    tt::LogOp, "Core {} ({}): handles {} head segments", i, work.physical_core, work.head_work.size());
            }
        }

        // Second pass: Build chains for heads spanning multiple cores
        uint32_t chains_built = 0;
        uint32_t chains_skipped = 0;
        // Track injector physical X columns for DRAM channel spreading
        std::vector<uint32_t> injector_phys_x;
        injector_phys_x.reserve(head_segments.size());

        for (uint32_t head_id = 0; head_id < head_segments.size(); ++head_id) {
            auto& segments = head_segments[head_id];
            if (segments.size() < 2) {
                continue;  // No chain needed for single core
            }

            // Find first non-conflicting single-segment core as chain start.
            // Exclude the last segment: it must remain as a chain tail since the
            // wrap-around build below needs at least one segment after start.
            std::optional<std::size_t> chain_start_idx;
            for (std::size_t idx = 0; idx + 1 < segments.size(); ++idx) {
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
            if (!chain_start_idx.has_value()) {
                for (std::size_t idx = 0; idx + 1 < segments.size(); ++idx) {
                    const auto& seg = segments[idx];
                    if (!core_chain_info[seg.core_idx].participates) {
                        chain_start_idx = idx;
                        break;
                    }
                }
            }

            if (!chain_start_idx.has_value()) {
                chains_skipped++;
                continue;
            }

            const std::size_t start = chain_start_idx.value();

            // Build chain in wrap order: start, start+1, ..., N-1, 0, 1, ..., start-1.
            // Break on conflict (core already in a different chain).
            std::vector<std::size_t> chain_order;
            chain_order.reserve(segments.size());
            for (std::size_t step = 0; step < segments.size(); ++step) {
                std::size_t idx = (start + step) % segments.size();
                const auto& seg = segments[idx];
                const uint32_t core_idx = seg.core_idx;
                if (core_idx >= core_work.size() || seg.head_work_index >= core_work[core_idx].head_work.size()) {
                    continue;
                }
                if (core_chain_info[core_idx].participates) {
                    break;
                }
                chain_order.push_back(idx);
            }

            if (chain_order.size() < 2) {
                chains_skipped++;
                continue;
            }

            // Check if all chain cores have the same q_chunk_count.
            // Mixed q_chunk_count chains are safe in unicast mode when sorted in
            // descending q_chunk_count order: the kernel's should_forward condition
            // guards on (q_iter < next_core_q_chunks), so a heavier sender only
            // forwards for the lighter receiver's iteration count, and the receiver
            // receives for all of its own iterations.  Mcast mode requires uniform
            // q_chunk_count (checked separately in the mcast eligibility pass).
            const uint32_t ref_q = core_work[segments[chain_order[0]].core_idx]
                                       .head_work[segments[chain_order[0]].head_work_index]
                                       .q_chunk_count;
            bool uniform_q = true;
            for (std::size_t i = 1; i < chain_order.size(); ++i) {
                const auto& seg = segments[chain_order[i]];
                if (core_work[seg.core_idx].head_work[seg.head_work_index].q_chunk_count != ref_q) {
                    uniform_q = false;
                    break;
                }
            }

            if (uniform_q) {
                // All cores have equal q_chunk_count — safe to pick any injector.
                // Choose the core whose physical X is furthest from existing
                // injectors to spread DRAM reads across channels.
                std::size_t best_pos = 0;
                uint32_t best_dist = 0;
                for (std::size_t pos = 0; pos < chain_order.size(); ++pos) {
                    const uint32_t phys_x = core_work[segments[chain_order[pos]].core_idx].physical_core.x;
                    uint32_t min_dist = UINT32_MAX;
                    for (uint32_t ix : injector_phys_x) {
                        uint32_t d = (phys_x > ix) ? (phys_x - ix) : (ix - phys_x);
                        min_dist = std::min(min_dist, d);
                    }
                    if (min_dist > best_dist) {
                        best_dist = min_dist;
                        best_pos = pos;
                    }
                }
                if (best_pos != 0) {
                    std::swap(chain_order[0], chain_order[best_pos]);
                }
            } else {
                // Mixed q_chunk_counts — sort descending so heavier cores come first.
                // Each sender forwards only for min(own_q_iters, next_core_q_chunks)
                // iterations, so a heavier sender safely serves a lighter receiver.
                // Stable sort preserves physical topology where q_counts are equal.
                std::stable_sort(chain_order.begin(), chain_order.end(), [&](std::size_t a, std::size_t b) {
                    const auto& seg_a = segments[a];
                    const auto& seg_b = segments[b];
                    return core_work[seg_a.core_idx].head_work[seg_a.head_work_index].q_chunk_count >
                           core_work[seg_b.core_idx].head_work[seg_b.head_work_index].q_chunk_count;
                });
            }

            const auto& inj_seg = segments[chain_order[0]];
            injector_phys_x.push_back(core_work[inj_seg.core_idx].physical_core.x);
            uint32_t batch = core_work[inj_seg.core_idx].head_work[inj_seg.head_work_index].batch;
            uint32_t head = head_id % NQH;

            log_debug(
                tt::LogOp,
                "Building chain for head {} (batch={}, head={}): {} cores, uniform_q={}, injector phys_x={}",
                head_id,
                batch,
                head,
                chain_order.size(),
                uniform_q,
                core_work[inj_seg.core_idx].physical_core.x);

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
        candidates.reserve(head_segments.size());
        bool all_eligible = true;
        uint32_t total_multi_core_chains = 0;

        for (uint32_t head_id = 0; head_id < head_segments.size(); ++head_id) {
            auto& segments = head_segments[head_id];
            if (segments.size() < 2) {
                continue;
            }

            // Collect chain core indices that actually participate in this head's chain
            std::vector<uint32_t> chain_core_indices;
            chain_core_indices.reserve(segments.size());
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

            // Eligibility condition 2: no non-chain worker cores inside the mcast rectangle.
            // The mcast rectangle spans [min_x, max_x] on the same row. Any active worker
            // core in that range receives the multicast — if it's not part of the chain,
            // its K/V CB and semaphores get corrupted. This can happen when the q_chunk_count
            // filter creates "gaps" by excluding mid-chain segments with uneven tail work.
            uint32_t min_x = core_work[chain_core_indices[0]].physical_core.x;
            uint32_t max_x = min_x;
            for (const auto& ci : chain_core_indices) {
                uint32_t x = core_work[ci].physical_core.x;
                min_x = std::min(min_x, x);
                max_x = std::max(max_x, x);
            }

            bool has_gap = false;
            for (const auto& seg : segments) {
                if (seg.core_idx >= core_work.size()) {
                    continue;
                }
                const auto& phys = core_work[seg.core_idx].physical_core;
                if (phys.y == ref_y && phys.x >= min_x && phys.x <= max_x) {
                    bool in_chain = false;
                    for (const auto& ci : chain_core_indices) {
                        if (ci == seg.core_idx) {
                            in_chain = true;
                            break;
                        }
                    }
                    if (!in_chain) {
                        has_gap = true;
                        break;
                    }
                }
            }

            if (has_gap) {
                all_eligible = false;
                log_debug(
                    tt::LogOp, "Head {}: mcast ineligible - non-chain worker core inside mcast rectangle", head_id);
                break;
            }

            // Eligibility condition 3: All chain cores must have the same q_chunk_count.
            // Mcast uses a single sender_wait count — receivers that finish their Q loop
            // early won't signal back, causing the injector to deadlock waiting for
            // missing semaphores.  (Unicast chains handle mixed q via descending sort,
            // but mcast requires strict uniformity.)
            const uint32_t ref_q_chunks = core_chain_info[chain_core_indices[0]].q_chunk_count;
            bool uniform_q_mcast = true;
            for (size_t ci = 1; ci < chain_core_indices.size(); ++ci) {
                if (core_chain_info[chain_core_indices[ci]].q_chunk_count != ref_q_chunks) {
                    uniform_q_mcast = false;
                    break;
                }
            }

            if (!uniform_q_mcast) {
                all_eligible = false;
                log_debug(tt::LogOp, "Head {}: mcast ineligible - mixed q_chunk_counts", head_id);
                break;
            }

            // Defensive: crash in all builds if a non-uniform chain slips past the check above.
            for (const auto& ci : chain_core_indices) {
                TT_FATAL(
                    core_chain_info[ci].q_chunk_count == ref_q_chunks,
                    "Mcast chain for head {} has non-uniform q_chunk_count: core {} has {} vs ref {}",
                    head_id,
                    ci,
                    core_chain_info[ci].q_chunk_count,
                    ref_q_chunks);
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

                const uint32_t mcast_num_dests = num_receivers;

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
                    receiver_chain.next_physical = CoreCoord{0, 0};
                    receiver_chain.next_core_q_chunks = 0;
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

        log_debug(
            tt::LogOp,
            "Multicast eligibility: {}/{} chains using mcast (all-or-nothing)",
            mcast_chains,
            total_multi_core_chains);
    }

    // mcast is enabled iff chain construction produced any mcast chains.
    const uint32_t mcast_enabled_val = (mcast_chains > 0) ? 1 : 0;

    // ---- Kernels ----

    KernelSpec::CompileTimeArgs reader_cta = {
        {"B", B},
        {"NQH", NQH},
        {"NKH", NKH},
        {"NVH", NVH},
        {"Sqt", Sqt},
        {"Skt", Skt},
        {"valid_Sqt", valid_Sqt},
        {"valid_Skt", valid_Skt},
        {"DHt", DHt},
        {"vDHt", vDHt},
        {"Sq_chunk_t", Sq_chunk_t},
        {"q_num_chunks", q_num_chunks},
        {"Sk_chunk_t", Sk_chunk_t},
        {"k_num_chunks", k_num_chunks},
        {"num_cores", num_cores},
        {"is_causal", static_cast<uint32_t>(is_causal)},
        {"use_provided_mask", static_cast<uint32_t>(use_provided_mask)},
        {"broadcast_provided_mask_batch", static_cast<uint32_t>(broadcast_provided_mask_batch)},
        {"broadcast_provided_mask_heads", static_cast<uint32_t>(broadcast_provided_mask_heads)},
        {"use_padded_mask", static_cast<uint32_t>(generated_padding_mask)},
        {"is_chunked", static_cast<uint32_t>(is_chunked)},
        {"block_size_t", block_size_t},
        {"page_table_stick_size", page_table_stick_size},
        {"use_attention_sink", static_cast<uint32_t>(use_attention_sink)},
        {"use_mla", static_cast<uint32_t>(use_mla)},
        {"mla_kv_overlap", static_cast<uint32_t>(mla_kv_overlap)},
        {"qk_subblock_h", qk_out_subblock_h},
        {"sliding_window_size", sliding_window_size.value_or(0)},
        {"use_streaming_compute", static_cast<uint32_t>(use_streaming_compute)},
        {"mcast_enabled", mcast_enabled_val},
        {"use_zigzag_balancing", static_cast<uint32_t>(use_zigzag_balancing)},
        {"use_windowed_narrowing", static_cast<uint32_t>(is_windowed)},
    };

    Group<DFBBinding> reader_dfbs = {
        DFBBinding{.dfb_spec_name = Q_IN, .accessor_name = "q_in", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = K_IN, .accessor_name = "k_in", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = V_IN, .accessor_name = "v_in", .endpoint_type = DFBEndpointType::PRODUCER},
    };
    Group<TensorBinding> reader_tensors = {
        TensorBinding{.tensor_parameter_name = T_Q_IN, .accessor_name = "q_in"},
        TensorBinding{.tensor_parameter_name = T_K_IN, .accessor_name = "k_in"},
        TensorBinding{.tensor_parameter_name = T_V_IN, .accessor_name = "v_in"},
    };
    Group<SemaphoreBinding> reader_sems;
    Group<std::string> reader_rta_names = {
        "core_id",
        "num_phases",
        "chunked_q_chunk_offset_phase_1",
        "read_offset_phase_1",
        "global_q_start",
        "global_q_count"};
    KernelSpec::CompilerOptions::Defines reader_defines = base_defines;
    if (use_provided_mask) {
        reader_dfbs.push_back(DFBBinding{
            .dfb_spec_name = MASK_IN, .accessor_name = "mask_in", .endpoint_type = DFBEndpointType::PRODUCER});
        reader_tensors.push_back(TensorBinding{.tensor_parameter_name = T_MASK, .accessor_name = "mask"});
        reader_defines.insert({"READER_PRODUCES_MASK", "1"});
    }
    if (use_attention_sink) {
        reader_dfbs.push_back(DFBBinding{
            .dfb_spec_name = ATTENTION_SINK,
            .accessor_name = "attention_sink",
            .endpoint_type = DFBEndpointType::PRODUCER});
        reader_tensors.push_back(
            TensorBinding{.tensor_parameter_name = T_ATTENTION_SINK, .accessor_name = "attention_sink"});
        reader_defines.insert({"USE_ATTENTION_SINK", "1"});
    }
    if (is_chunked) {
        // Page table is filled and read within the reader — self-loop.
        reader_dfbs.push_back(DFBBinding{
            .dfb_spec_name = PAGE_TABLE, .accessor_name = "page_table", .endpoint_type = DFBEndpointType::PRODUCER});
        reader_dfbs.push_back(DFBBinding{
            .dfb_spec_name = PAGE_TABLE, .accessor_name = "page_table", .endpoint_type = DFBEndpointType::CONSUMER});
        reader_tensors.push_back(TensorBinding{.tensor_parameter_name = T_PAGE_TABLE, .accessor_name = "page_table"});
        reader_defines.insert({"IS_CHUNKED", "1"});
    }
    if (flexible_chunked) {
        reader_dfbs.push_back(DFBBinding{
            .dfb_spec_name = CHUNK_START_IDX_COMPUTE,
            .accessor_name = "chunk_start_idx_compute",
            .endpoint_type = DFBEndpointType::PRODUCER});
        reader_dfbs.push_back(DFBBinding{
            .dfb_spec_name = CHUNK_START_IDX_WRITER,
            .accessor_name = "chunk_start_idx_writer",
            .endpoint_type = DFBEndpointType::PRODUCER});
        reader_tensors.push_back(
            TensorBinding{.tensor_parameter_name = T_CHUNK_START_IDX, .accessor_name = "chunk_start_idx"});
        reader_defines.insert({"FLEXIBLE_CHUNKED", "1"});
    }
    if (kv_chain) {
        reader_sems = {
            SemaphoreBinding{.semaphore_spec_name = SEM_SENDER, .accessor_name = "sender"},
            SemaphoreBinding{.semaphore_spec_name = SEM_RECEIVER, .accessor_name = "receiver"},
            SemaphoreBinding{.semaphore_spec_name = SEM_VALID, .accessor_name = "valid"},
        };
        reader_defines.insert({"KV_CHAIN", "1"});
        for (const char* n :
             {"is_chain_participant",
              "is_injector",
              "is_sink",
              "chain_batch",
              "chain_head",
              "prev_physical_x",
              "prev_physical_y",
              "next_physical_x",
              "next_physical_y",
              "next_core_q_chunks",
              "mcast_num_dests",
              "mcast_sender_wait"}) {
            reader_rta_names.push_back(n);
        }
    }
    if (is_windowed) {
        // #54492 K-range narrowing: the reader keeps its own cu_window copy (self-loop) and produces
        // the {k_lo, k_hi} ctrl CB consumed by compute. It binds cu_window_seqlens under its own
        // accessor, and the per-device Q-offset tensor when supplied (read into the cu copy's landing
        // spot — no separate CB on the reader side).
        reader_dfbs.push_back(DFBBinding{
            .dfb_spec_name = WINDOWED_CU_READER,
            .accessor_name = "windowed_cu_reader",
            .endpoint_type = DFBEndpointType::PRODUCER});
        reader_dfbs.push_back(DFBBinding{
            .dfb_spec_name = WINDOWED_CU_READER,
            .accessor_name = "windowed_cu_reader",
            .endpoint_type = DFBEndpointType::CONSUMER});
        reader_dfbs.push_back(DFBBinding{
            .dfb_spec_name = WINDOWED_K_RANGE,
            .accessor_name = "windowed_k_range",
            .endpoint_type = DFBEndpointType::PRODUCER});
        reader_tensors.push_back(TensorBinding{.tensor_parameter_name = T_CU_WINDOW, .accessor_name = "cu_window_reader"});
        reader_defines.insert({"USE_WINDOWED_NARROWING", "1"});
        reader_rta_names.push_back("cu_window_seqlens_eles");
        reader_rta_names.push_back("windowed_q_tok_offset");
        if (windowed_q_offset_present) {
            reader_tensors.push_back(
                TensorBinding{.tensor_parameter_name = T_WINDOWED_Q_OFFSET, .accessor_name = "windowed_q_offset"});
            reader_defines.insert({"WINDOWED_Q_OFFSET_TENSOR", "1"});
        }
    }

    KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved_metal2.cpp",
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings = reader_dfbs,
        .semaphore_bindings = reader_sems,
        .tensor_bindings = reader_tensors,
        .compile_time_args = reader_cta,
        .runtime_arg_schema = {.runtime_arg_names = reader_rta_names},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec::CompileTimeArgs writer_cta = {
        {"B", B},
        {"NQH", NQH},
        {"NKH", NKH},
        {"Sqt", Sqt},
        {"valid_Sqt", valid_Sqt},
        {"unpadded_Sk", Sk},
        {"DHt", DHt},
        {"vDHt", vDHt},
        {"Sq_chunk_t", Sq_chunk_t},
        {"q_num_chunks", q_num_chunks},
        {"Sk_chunk_t", Sk_chunk_t},
        {"k_num_chunks", k_num_chunks},
        {"identity_scalar_packed", packed_identity_scalar},
        {"scale_val", scale_packed},
        {"num_cores", num_cores},
        {"is_causal", static_cast<uint32_t>(is_causal)},
        {"use_provided_mask", static_cast<uint32_t>(use_provided_mask)},
        {"use_padded_mask", static_cast<uint32_t>(generated_padding_mask)},
        {"is_chunked", static_cast<uint32_t>(is_chunked)},
        {"sliding_window_size", sliding_window_size.value_or(0)},
        {"use_lightweight_mask", static_cast<uint32_t>(lightweight_mask)},
        {"use_streaming_compute", static_cast<uint32_t>(use_streaming_compute)},
        {"out_subblock_h", out_out_subblock_h},
        {"k_partial_col", k_partial_col},
        {"use_zigzag_balancing", static_cast<uint32_t>(use_zigzag_balancing)},
        {"use_windowed_mask", static_cast<uint32_t>(is_windowed)},
    };

    Group<DFBBinding> writer_dfbs = {
        DFBBinding{
            .dfb_spec_name = IDENTITY_SCALE_IN,
            .accessor_name = "identity_scale_in",
            .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{
            .dfb_spec_name = COL_IDENTITY, .accessor_name = "col_identity", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
    };
    Group<TensorBinding> writer_tensors = {
        TensorBinding{.tensor_parameter_name = T_OUT, .accessor_name = "out"},
    };
    KernelSpec::CompilerOptions::Defines writer_defines = base_defines;
    Group<std::string> writer_rta_names = {
        "core_id",
        "num_phases",
        "use_chunk_start_idx_tensor",
        "chunk_start_t_in_q_chunks_phase_1",
        "write_offset_phase_1",
        "chunk_start_t_in_q_chunks_phase_2",
        "write_offset_phase_2",
        "global_q_start",
        "global_q_count",
        "cu_window_seqlens_eles"};
    if (writer_produces_mask) {
        writer_dfbs.push_back(DFBBinding{
            .dfb_spec_name = MASK_IN, .accessor_name = "mask_in", .endpoint_type = DFBEndpointType::PRODUCER});
        writer_defines.insert({"WRITER_PRODUCES_MASK", "1"});
    }
    if (is_windowed) {
        // cu_window_seqlens is filled and read within the writer — self-loop.
        writer_dfbs.push_back(DFBBinding{
            .dfb_spec_name = CU_WINDOW,
            .accessor_name = "cu_window_seqlens",
            .endpoint_type = DFBEndpointType::PRODUCER});
        writer_dfbs.push_back(DFBBinding{
            .dfb_spec_name = CU_WINDOW,
            .accessor_name = "cu_window_seqlens",
            .endpoint_type = DFBEndpointType::CONSUMER});
        writer_tensors.push_back(
            TensorBinding{.tensor_parameter_name = T_CU_WINDOW, .accessor_name = "cu_window_seqlens"});
        writer_defines.insert({"USE_WINDOWED_MASK", "1"});
        // #54492: per-Q-chunk K narrowing origin. Scalar via a named RTA; per-device via a dedicated
        // self-loop CB fed by the windowed_q_token_offset tensor (writer reads it into its own CB).
        writer_rta_names.push_back("windowed_q_tok_offset");
        if (windowed_q_offset_present) {
            writer_dfbs.push_back(DFBBinding{
                .dfb_spec_name = WINDOWED_Q_OFFSET,
                .accessor_name = "windowed_q_offset",
                .endpoint_type = DFBEndpointType::PRODUCER});
            writer_dfbs.push_back(DFBBinding{
                .dfb_spec_name = WINDOWED_Q_OFFSET,
                .accessor_name = "windowed_q_offset",
                .endpoint_type = DFBEndpointType::CONSUMER});
            writer_tensors.push_back(
                TensorBinding{.tensor_parameter_name = T_WINDOWED_Q_OFFSET, .accessor_name = "windowed_q_offset"});
            writer_defines.insert({"WINDOWED_Q_OFFSET_TENSOR", "1"});
        }
    }
    if (flexible_chunked) {
        writer_dfbs.push_back(DFBBinding{
            .dfb_spec_name = CHUNK_START_IDX_WRITER,
            .accessor_name = "chunk_start_idx_writer",
            .endpoint_type = DFBEndpointType::CONSUMER});
        writer_defines.insert({"FLEXIBLE_CHUNKED", "1"});
    }

    KernelSpec writer{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/writer_interleaved_metal2.cpp",
        .compiler_options = {.defines = writer_defines},
        .dfb_bindings = writer_dfbs,
        .tensor_bindings = writer_tensors,
        .compile_time_args = writer_cta,
        .runtime_arg_schema = {.runtime_arg_names = writer_rta_names},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    KernelSpec::CompileTimeArgs compute_cta = {
        {"B", B},
        {"NQH", NQH},
        {"NKH", NKH},
        {"Skt", Skt},
        {"DHt", DHt},
        {"vDHt", vDHt},
        {"Sq_chunk_t", Sq_chunk_t},
        {"q_num_chunks", q_num_chunks},
        {"Sk_chunk_t", Sk_chunk_t},
        {"k_num_chunks", k_num_chunks},
        {"qk_in0_block_w", qk_in0_block_w},
        {"qk_subblock_w", qk_out_subblock_w},
        {"qk_subblock_h", qk_out_subblock_h},
        {"qk_in0_num_subblocks", qk_in0_num_subblocks},
        {"qk_in1_num_subblocks", qk_in1_num_subblocks},
        {"qk_num_blocks", qk_num_blocks},
        {"out_in0_block_w", out_in0_block_w},
        {"out_subblock_w", out_out_subblock_w},
        {"out_subblock_h", out_out_subblock_h},
        {"out_in0_num_subblocks", out_in0_num_subblocks},
        {"out_in1_num_subblocks", out_in1_num_subblocks},
        {"out_num_blocks", out_num_blocks},
        {"num_cores", num_cores},
        {"is_causal", static_cast<uint32_t>(is_causal)},
        {"use_provided_mask", static_cast<uint32_t>(compute_use_provided_mask)},
        {"use_padded_mask", static_cast<uint32_t>(generated_padding_mask)},
        {"is_chunked", static_cast<uint32_t>(is_chunked)},
        {"scale_fp32", scale_packed},
        {"sliding_window_size", sliding_window_size.value_or(0)},
        {"use_attention_sink", static_cast<uint32_t>(use_attention_sink)},
        {"use_streaming_compute", static_cast<uint32_t>(use_streaming_compute)},
        {"valid_Skt", valid_Skt},
        {"k_partial_col", k_partial_col},
        {"use_zigzag_balancing", static_cast<uint32_t>(use_zigzag_balancing)},
        {"use_windowed_narrowing", static_cast<uint32_t>(is_windowed)},
    };

    Group<DFBBinding> compute_dfbs = {
        DFBBinding{.dfb_spec_name = Q_IN, .accessor_name = "q_in", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = K_IN, .accessor_name = "k_in", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = V_IN, .accessor_name = "v_in", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{
            .dfb_spec_name = IDENTITY_SCALE_IN,
            .accessor_name = "identity_scale_in",
            .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{
            .dfb_spec_name = COL_IDENTITY, .accessor_name = "col_identity", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
        // Compute-only intermediates: self-loop.
        DFBBinding{.dfb_spec_name = QK_IM, .accessor_name = "qk_im", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = QK_IM, .accessor_name = "qk_im", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = OUT_IM_A, .accessor_name = "out_im_A", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = OUT_IM_A, .accessor_name = "out_im_A", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = OUT_IM_B, .accessor_name = "out_im_B", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = OUT_IM_B, .accessor_name = "out_im_B", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = MAX_A, .accessor_name = "max_A", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = MAX_A, .accessor_name = "max_A", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = MAX_B, .accessor_name = "max_B", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = MAX_B, .accessor_name = "max_B", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = SUM_A, .accessor_name = "sum_A", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = SUM_A, .accessor_name = "sum_A", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = SUM_B, .accessor_name = "sum_B", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = SUM_B, .accessor_name = "sum_B", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{
            .dfb_spec_name = EXP_MAX_DIFF, .accessor_name = "exp_max_diff", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{
            .dfb_spec_name = EXP_MAX_DIFF, .accessor_name = "exp_max_diff", .endpoint_type = DFBEndpointType::CONSUMER},
    };
    KernelSpec::CompilerOptions::Defines compute_defines = base_defines;
    if (needs_mask_cb) {
        compute_dfbs.push_back(DFBBinding{
            .dfb_spec_name = MASK_IN, .accessor_name = "mask_in", .endpoint_type = DFBEndpointType::CONSUMER});
        compute_defines.insert({"HAS_MASK", "1"});
    }
    if (use_attention_sink) {
        compute_dfbs.push_back(DFBBinding{
            .dfb_spec_name = ATTENTION_SINK,
            .accessor_name = "attention_sink",
            .endpoint_type = DFBEndpointType::CONSUMER});
        compute_defines.insert({"USE_ATTENTION_SINK", "1"});
    }
    if (flexible_chunked) {
        compute_dfbs.push_back(DFBBinding{
            .dfb_spec_name = CHUNK_START_IDX_COMPUTE,
            .accessor_name = "chunk_start_idx_compute",
            .endpoint_type = DFBEndpointType::CONSUMER});
        compute_defines.insert({"FLEXIBLE_CHUNKED", "1"});
    }
    if (use_streaming_compute) {
        compute_dfbs.push_back(DFBBinding{
            .dfb_spec_name = RECIP_SCRATCH,
            .accessor_name = "recip_scratch",
            .endpoint_type = DFBEndpointType::PRODUCER});
        compute_dfbs.push_back(DFBBinding{
            .dfb_spec_name = RECIP_SCRATCH,
            .accessor_name = "recip_scratch",
            .endpoint_type = DFBEndpointType::CONSUMER});
        compute_defines.insert({"USE_STREAMING_COMPUTE", "1"});
    }
    if (is_windowed) {
        // #54492 K-range narrowing: compute consumes the reader-produced {k_lo, k_hi} ctrl CB.
        compute_dfbs.push_back(DFBBinding{
            .dfb_spec_name = WINDOWED_K_RANGE,
            .accessor_name = "windowed_k_range",
            .endpoint_type = DFBEndpointType::CONSUMER});
        compute_defines.insert({"USE_WINDOWED_NARROWING", "1"});
    }

    auto compute_hw = ttnn::to_compute_hardware_config(device->arch(), compute_kernel_config);
    if (fp32_dest_acc_en) {
        // qk_im / sum_A / sum_B are Float32 when enable_32_bit_dest is on; the validator requires an
        // explicit unpack_modes entry for each Float32 DFB the compute consumes. Legacy defaulted to
        // UnpackToSrc (no explicit unpack_to_dest_mode), preserved here.
        auto& gen1 = std::get<ComputeGen1Config>(compute_hw);
        gen1.unpack_modes.insert({QK_IM, tt::tt_metal::UnpackMode::UnpackToSrc});
        gen1.unpack_modes.insert({SUM_A, tt::tt_metal::UnpackMode::UnpackToSrc});
        gen1.unpack_modes.insert({SUM_B, tt::tt_metal::UnpackMode::UnpackToSrc});
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa_metal2.cpp",
        .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = compute_dfbs,
        .compile_time_args = compute_cta,
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"core_id",
                  "num_phases",
                  "use_chunk_start_idx_tensor",
                  "chunked_q_chunk_offset_phase_1",
                  "chunked_q_chunk_offset_phase_2",
                  "global_q_start",
                  "global_q_count"}},
        .hw_config = compute_hw,
    };

    // ---- ProgramSpec ----

    ProgramSpec spec{
        .name = "sdpa",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = dfbs,
        .semaphores = sems,
        .tensor_parameters = tensor_params,
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = core_grid}},
    };

    // ---- ProgramRunArgs ----

    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = READER};
    KernelRunArgs writer_run{.kernel = WRITER};
    KernelRunArgs compute_run{.kernel = COMPUTE};

    const uint32_t use_chunk_start_idx_tensor = flexible_chunked ? 1u : 0u;

    for (uint32_t i = 0; i < num_cores; ++i) {
        NodeCoord node = {i % grid_size.x, i / grid_size.x};

        // Global Q scheduling per-core range: contiguous slice of the flat (B, NQH, q_num_chunks) space.
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

        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            node,
            {{"core_id", i},
             {"num_phases", num_phases},
             {"chunked_q_chunk_offset_phase_1", chunked_q_chunk_offset},
             {"read_offset_phase_1", read_offset},
             {"global_q_start", global_q_start},
             {"global_q_count", global_q_count}});
        if (kv_chain) {
            const auto& chain = core_chain_info[i];
            AddRuntimeArgsForNode(
                reader_run.runtime_arg_values,
                node,
                {{"is_chain_participant", static_cast<uint32_t>(chain.participates)},
                 {"is_injector", static_cast<uint32_t>(chain.is_injector)},
                 {"is_sink", static_cast<uint32_t>(chain.is_sink)},
                 {"chain_batch", chain.batch},
                 {"chain_head", chain.head},
                 {"prev_physical_x", static_cast<uint32_t>(chain.prev_physical.x)},
                 {"prev_physical_y", static_cast<uint32_t>(chain.prev_physical.y)},
                 {"next_physical_x", static_cast<uint32_t>(chain.next_physical.x)},
                 {"next_physical_y", static_cast<uint32_t>(chain.next_physical.y)},
                 {"next_core_q_chunks", chain.next_core_q_chunks},
                 {"mcast_num_dests", chain.mcast_num_dests},
                 {"mcast_sender_wait", chain.mcast_sender_wait}});
        }

        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            node,
            {{"core_id", i},
             {"num_phases", num_phases},
             {"use_chunk_start_idx_tensor", use_chunk_start_idx_tensor},
             {"chunk_start_t_in_q_chunks_phase_1", chunked_q_chunk_offset},
             {"write_offset_phase_1", write_offset},
             {"chunk_start_t_in_q_chunks_phase_2", 0u},
             {"write_offset_phase_2", 0u},
             {"global_q_start", global_q_start},
             {"global_q_count", global_q_count},
             {"cu_window_seqlens_eles", cu_window_seqlens_eles}});

        // Windowed K-range narrowing (#54492): the reader and writer each resolve the per-Q-chunk
        // K range from cu_window_seqlens + the (possibly per-device) Q origin; both need the element
        // count and scalar origin. Addresses arrive via TensorBindings, not runtime args.
        if (is_windowed) {
            AddRuntimeArgsForNode(
                reader_run.runtime_arg_values,
                node,
                {{"cu_window_seqlens_eles", cu_window_seqlens_eles},
                 {"windowed_q_tok_offset", windowed_q_token_offset}});
            AddRuntimeArgsForNode(
                writer_run.runtime_arg_values, node, {{"windowed_q_tok_offset", windowed_q_token_offset}});
        }

        AddRuntimeArgsForNode(
            compute_run.runtime_arg_values,
            node,
            {{"core_id", i},
             {"num_phases", num_phases},
             {"use_chunk_start_idx_tensor", use_chunk_start_idx_tensor},
             {"chunked_q_chunk_offset_phase_1", chunked_q_chunk_offset},
             {"chunked_q_chunk_offset_phase_2", 0u},
             {"global_q_start", global_q_start},
             {"global_q_count", global_q_count}});
    }

    run_args.kernel_run_args = {reader_run, writer_run, compute_run};

    // Bind V against the persistent io tensor (input_tensor_v is a value_or temporary whose reference
    // would dangle after this function returns); it aliases K in the MLA kv-overlap case.
    const auto& v_binding_tensor = tensor_args.v.has_value() ? tensor_args.v.value() : tensor_args.k;
    run_args.tensor_args = {
        {T_Q_IN, input_tensor_q.mesh_tensor()},
        {T_K_IN, input_tensor_k.mesh_tensor()},
        {T_V_IN, v_binding_tensor.mesh_tensor()},
        {T_OUT, output_tensor.mesh_tensor()},
    };
    if (use_provided_mask) {
        run_args.tensor_args.insert({T_MASK, attn_mask.value().mesh_tensor()});
    }
    if (is_chunked) {
        run_args.tensor_args.insert({T_PAGE_TABLE, page_table.value().mesh_tensor()});
    }
    if (use_attention_sink) {
        run_args.tensor_args.insert({T_ATTENTION_SINK, attention_sink.value().mesh_tensor()});
    }
    if (flexible_chunked) {
        run_args.tensor_args.insert({T_CHUNK_START_IDX, tensor_args.chunk_start_idx_tensor.value().mesh_tensor()});
    }
    if (is_windowed) {
        run_args.tensor_args.insert({T_CU_WINDOW, tensor_args.cu_window_seqlens.value().mesh_tensor()});
        if (windowed_q_offset_present) {
            run_args.tensor_args.insert(
                {T_WINDOWED_Q_OFFSET, tensor_args.windowed_q_token_offset_tensor.value().mesh_tensor()});
        }
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
