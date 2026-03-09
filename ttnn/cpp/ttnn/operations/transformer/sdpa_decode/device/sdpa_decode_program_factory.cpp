// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode_program_factory.hpp"

#include <optional>
#include <string>
#include <cmath>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

SdpaDecodeProgramFactory::cached_program_t SdpaDecodeProgramFactory::create(
    const SdpaDecodeParams& operation_attributes, const SdpaDecodeInputs& tensor_args, Tensor& tensor_return_value) {
    // ========== Input Tensors ==========
    const auto& input_tensor_q = tensor_args.q;
    const auto& input_tensor_k = tensor_args.k;
    const auto& cur_pos_tensor = tensor_args.cur_pos_tensor;
    const auto& page_table_tensor = tensor_args.page_table_tensor;
    const auto& attn_mask = tensor_args.attn_mask;
    const auto& attention_sink = tensor_args.attention_sink;
    const auto& output_tensor = tensor_return_value;

    // ========== Operation Attributes ==========
    const bool use_mla = operation_attributes.use_mla.value_or(false);
    const bool is_causal = operation_attributes.is_causal;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;
    const auto& program_config = operation_attributes.program_config;
    const uint32_t k_chunk_size = operation_attributes.k_chunk_size;
    const uint32_t head_dim_v = operation_attributes.head_dim_v.value_or(0);
    const auto& cur_pos_ids = operation_attributes.cur_pos;
    const float scale =
        operation_attributes.scale.value_or(1.0f / std::sqrt(static_cast<float>(input_tensor_q.padded_shape()[-1])));
    const uint32_t sliding_window_size = operation_attributes.sliding_window_size.value_or(0);
    bool share_cache = operation_attributes.share_cache.value_or(false);

    // V tensor: use K if MLA (V is subset of K), otherwise require explicit V
    TT_FATAL(use_mla || tensor_args.v.has_value(), "V tensor must be provided when MLA is disabled.");
    const auto& input_tensor_v = tensor_args.v.value_or(input_tensor_k);

    // ========== Device & Program ==========
    IDevice* device = input_tensor_q.device();
    Program program = CreateProgram();

    // ========== Feature Flags ==========
    const bool is_paged_attention = page_table_tensor.has_value();
    const bool is_q_sharded = input_tensor_q.is_sharded();
    const bool is_output_sharded = output_tensor.is_sharded();
    const bool tilize_q = input_tensor_q.layout() == Layout::ROW_MAJOR;
    const bool use_cur_pos_tensor = cur_pos_tensor.has_value();
    const bool use_attention_mask = attn_mask.has_value();
    const bool use_attention_sink = attention_sink.has_value();
    // ========== Tensor Shapes ==========
    auto q_shape = input_tensor_q.padded_shape();
    q_shape[2] = tt::round_up(q_shape[2], tt::constants::TILE_HEIGHT);
    const auto& q_shape_unpadded = input_tensor_q.logical_shape();
    const auto& k_shape = input_tensor_k.padded_shape();
    const auto& v_shape = input_tensor_v.padded_shape();

    // ========== Core Dimensions ==========
    // B = batch size, PNH = padded num Q heads, S = sequence length, DH = head dim
    uint32_t B = q_shape[1];
    uint32_t PNH = q_shape[2];
    uint32_t S = k_shape[2];
    uint32_t DH = k_shape[3];
    uint32_t vDH = use_mla ? head_dim_v : v_shape[3];
    uint32_t Bkv = k_shape[0];
    uint32_t num_kv_heads = k_shape[1];
    uint32_t num_q_heads = q_shape_unpadded[2];
    uint32_t page_block_size_t = 0;
    uint32_t q_heads_parallel_factor = 1;
    uint32_t original_block_size = 0;
    bool has_block_padding = false;

    // Handle paged attention sequence length
    if (is_paged_attention) {
        B = page_table_tensor->is_sharded() ? page_table_tensor->padded_shape()[0] /
                                                  page_table_tensor->memory_config().shard_spec()->grid.num_cores()
                                            : page_table_tensor->padded_shape()[0];
        uint32_t block_size = k_shape[2];
        original_block_size = input_tensor_k.logical_shape()[2];
        page_block_size_t = block_size / TILE_HEIGHT;
        S = page_table_tensor.value().padded_shape()[-1] * S;
        has_block_padding = original_block_size < TILE_HEIGHT;
    }

    // ========== Q Sharding & MLA Parallelization ==========
    // Q is "locally available" when sharded for MLA with data replicated across all worker cores.
    //   Replicated layout:   Q shape = (1, 1, B * num_q_heads * num_cores_per_head, D) — batch folded into dim 2.
    //   Non-replicated layout: Q shape = (1, B, num_q_heads, D) — batch in dim 1.
    bool q_locally_available = false;
    if (is_q_sharded && use_mla) {
        const uint32_t q_shard_height = input_tensor_q.memory_config().shard_spec()->shape[0];
        const uint32_t max_cores = program_config.has_value() ? program_config->max_cores_per_head_batch : 16;
        const uint32_t num_q_shards = input_tensor_q.memory_config().shard_spec()->grid.num_cores();
        const uint32_t num_groups = num_q_shards / max_cores;
        q_heads_parallel_factor = num_groups / B;
        q_locally_available = (q_shape[2] == B * q_shard_height * q_heads_parallel_factor * max_cores);
        if (q_locally_available) {
            num_q_heads = q_heads_parallel_factor * q_shard_height;
            PNH = num_q_heads;
        } else {
            q_heads_parallel_factor = std::max(1u, (num_q_heads + q_shard_height - 1) / q_shard_height);
        }
        B *= q_heads_parallel_factor;
        TT_FATAL(
            q_heads_parallel_factor == 1 || num_kv_heads == 1,
            "Q head parallelization (factor={}) requires num_kv_heads=1, got {}",
            q_heads_parallel_factor,
            num_kv_heads);
    }
    if (share_cache) {
        TT_FATAL(B % Bkv == 0, "Batch dim in Q must be divisible by batch dim in KV if sharing cache");
    }

    // ========== Tile Dimensions ==========
    const uint32_t St = S / TILE_HEIGHT;
    const uint32_t DHt = DH / TILE_WIDTH;
    const uint32_t vDHt = vDH / TILE_WIDTH;
    const uint32_t PNHt = PNH / q_heads_parallel_factor / TILE_HEIGHT;
    const uint32_t Sk_chunk_t = k_chunk_size / TILE_HEIGHT;

    // ========== Grid & Core Configuration ==========
    CoreCoord grid_size = program_config.has_value() ? program_config->compute_with_storage_grid_size
                                                     : device->compute_with_storage_grid_size();
    const uint32_t num_cores_available = grid_size.x * grid_size.y;
    const uint32_t num_cores_in_grid =
        device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;

    bool on_subcoregrid = false;
    CoreRangeSet core_grid;
    if (program_config.has_value() && program_config->sub_core_grids.has_value()) {
        core_grid = program_config->sub_core_grids.value();
        TT_FATAL(
            core_grid.num_cores() == num_cores_available,
            "sub_core_grids cores ({}) must match compute_with_storage_grid_size ({})",
            core_grid.num_cores(),
            num_cores_available);
        on_subcoregrid = true;
    } else {
        core_grid = CoreRangeSet(std::vector{CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1})});
    }

    TT_FATAL(
        num_cores_available <= num_cores_in_grid,
        "Cores available ({}) exceeds grid size ({})",
        num_cores_available,
        num_cores_in_grid);
    TT_FATAL(num_cores_available >= B, "Cores available ({}) must be >= batch size ({})", num_cores_available, B);

    // ========== Core Allocation ==========
    const uint32_t max_cores_per_head =
        program_config.has_value() ? program_config->max_cores_per_head_batch : num_cores_available;
    const uint32_t max_num_cores_for_compute = max_cores_per_head * B * num_kv_heads;
    const uint32_t num_cores_per_batch = std::min(num_cores_available, max_num_cores_for_compute) / B;
    const uint32_t num_cores_per_head = std::max(1u, num_cores_per_batch / num_kv_heads);
    const uint32_t num_heads_per_core = std::max(1u, (uint32_t)std::ceil((float)num_kv_heads / num_cores_per_batch));
    const uint32_t num_reducer_cores = num_kv_heads * B / num_heads_per_core;
    const uint32_t num_output_cores = B;
    const uint32_t num_active_cores = num_cores_per_head * num_kv_heads * B / num_heads_per_core;

    // ========== Compute Kernel Config ==========
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);
    const bool exp_approx_mode = program_config.has_value() && program_config->exp_approx_mode.has_value()
                                     ? program_config->exp_approx_mode.value()
                                     : true;

    // ========== Buffer Pointers & Metadata ==========
    auto* q_buffer = input_tensor_q.buffer();
    auto* k_buffer = input_tensor_k.buffer();
    auto* v_buffer = input_tensor_v.buffer();
    auto* out_buffer = output_tensor.buffer();

    // Optional tensor buffers and metadata
    Buffer* cur_pos_buffer = use_cur_pos_tensor ? cur_pos_tensor.value().buffer() : nullptr;
    Buffer* page_table_buffer = is_paged_attention ? page_table_tensor.value().buffer() : nullptr;
    const bool is_cur_pos_tensor_sharded = use_cur_pos_tensor && cur_pos_tensor.value().is_sharded();
    const bool is_page_table_sharded = is_paged_attention && page_table_tensor.value().is_sharded();
    const uint32_t cur_pos_stick_size = cur_pos_buffer ? cur_pos_buffer->aligned_page_size() : 0;
    const uint32_t page_table_stick_size = page_table_buffer ? page_table_buffer->aligned_page_size() : 0;
    const tt::DataFormat cur_pos_df = use_cur_pos_tensor
                                          ? tt_metal::datatype_to_dataformat_converter(cur_pos_tensor.value().dtype())
                                          : tt::DataFormat::Invalid;
    const tt::DataFormat page_table_df =
        is_paged_attention ? tt_metal::datatype_to_dataformat_converter(page_table_tensor.value().dtype())
                           : tt::DataFormat::Invalid;

    // ========== Tree Reduction Setup ==========
    // For n cores, need ceil(log2(n)) rounds
    const uint32_t num_tree_reduction_rounds = num_cores_per_head > 1 ? 32 - __builtin_clz(num_cores_per_head - 1) : 0;
    TT_FATAL(
        num_tree_reduction_rounds <= MAX_TREE_REDUCTION_ROUNDS,
        "Tree reduction max {} rounds ({} cores/head), got {} cores/head",
        MAX_TREE_REDUCTION_ROUNDS,
        1 << MAX_TREE_REDUCTION_ROUNDS,
        num_cores_per_head);
    TT_FATAL(
        (num_cores_per_head >= 1 && num_heads_per_core == 1) || (num_cores_per_head == 1 && num_heads_per_core >= 1),
        "Invalid core assignment: cores_per_head={}, heads_per_core={}",
        num_cores_per_head,
        num_heads_per_core);

    // ========== Group Indexing Mode ==========
    // A core group can be laid out in either row-major or column-major order on the core grid.
    // By default core groups are laid out in row-major order. But when Q heads is parallelized,
    // column-major group indexing is used to keep batch groups spatially close for efficient K multicast along columns.
    const bool use_col_major_group_indexing =
        (q_heads_parallel_factor > 1) && (grid_size.y >= num_cores_per_head) && !on_subcoregrid && q_locally_available;
    uint32_t num_group_rows = 0;
    uint32_t num_group_cols = 0;
    uint32_t num_groups_total = 0;
    if (use_col_major_group_indexing) {
        num_groups_total = num_active_cores / num_cores_per_head;
        num_group_rows = grid_size.x / num_cores_per_head;
        num_group_cols = num_groups_total / num_group_rows;
        TT_FATAL(
            num_group_cols % q_heads_parallel_factor == 0,
            "num_group_cols must be divisible by q_heads_parallel_factor");
        TT_FATAL(
            num_heads_per_core == 1, "Column major allocation of core groups is only supported for num kv heads = 1");
        TT_FATAL(
            num_active_cores % num_cores_per_head == 0,
            "num_active_cores must be divisible by num_cores_per_head for even distribution.");
        TT_FATAL(grid_size.x % num_cores_per_head == 0, "grid_size.x must be divisible by num_cores_per_head");
        TT_FATAL(
            num_groups_total == B,
            "num_groups_total must be equal to B (for q heads parallel factor > 1, B is number of virtual batches)");
        TT_FATAL(
            num_group_cols * num_group_rows == num_groups_total,
            "num_group_cols * num_group_rows must be equal to num_groups_total");
    }

    // ========== Core Group Assignment ==========
    // Core layout depends on sharding and indexing mode:
    // - Spatial indexing: simple linear order, spatial index computes batch/head from 2D position
    // - Q-sharded (no spatial): reorder so reducers at i % num_cores_per_batch == 0
    // - Neither: simple linear order with linear indexing
    std::vector<CoreCoord> core_group;
    std::vector<CoreCoord> core_group_idle;
    core_group.reserve(num_active_cores);
    core_group_idle.reserve(num_cores_available - num_active_cores);

    if (on_subcoregrid) {
        TT_FATAL(is_q_sharded || is_output_sharded, "Subcoregrids require sharded Q or output");
        auto cores_vec = corerange_to_cores(core_grid, num_cores_available, true);
        uint32_t reducer_idx = 0, worker_idx = num_output_cores;
        for (uint32_t i = 0; i < num_cores_available; ++i) {
            bool is_reducer = (i % num_cores_per_batch == 0) && (reducer_idx < num_output_cores);
            CoreCoord core = is_reducer ? cores_vec[reducer_idx++] : cores_vec[worker_idx++];
            (i < num_active_cores ? core_group : core_group_idle).push_back(core);
        }
    } else if ((is_q_sharded || is_output_sharded) && !use_col_major_group_indexing) {
        // Q/output sharded without row major group assignment: reorder cores so reducers are at batch boundaries
        // This ensures i % num_cores_per_batch == 0 identifies output/reducer cores
        uint32_t reducer_idx = 0, worker_idx = num_output_cores;
        for (uint32_t i = 0; i < num_cores_available; ++i) {
            CoreCoord core;
            if ((i % num_cores_per_batch == 0) && (reducer_idx < num_output_cores)) {
                core = {reducer_idx % grid_size.x, reducer_idx / grid_size.x};
                reducer_idx++;
            } else {
                core = {worker_idx % grid_size.x, worker_idx / grid_size.x};
                worker_idx++;
            }
            (i < num_active_cores ? core_group : core_group_idle).push_back(core);
        }
    } else {
        // Q in DRAM, no sharding: simple linear assignment
        for (uint32_t i = 0; i < num_cores_available; ++i) {
            CoreCoord core = {i % grid_size.x, i / grid_size.x};
            (i < num_active_cores ? core_group : core_group_idle).push_back(core);
        }
    }

    // ========== Physical Core Coordinate Maps ==========
    // Col-major group index for reducer/output cores
    // Guard: if num_cores_per_head > grid_size.x, groups don't fit in a row, so clamp to 1
    auto get_col_major_group_idx = [&](uint32_t row_major_idx) -> uint32_t {
        uint32_t group_row = row_major_idx / num_group_rows;
        uint32_t group_col = row_major_idx % num_group_rows;
        return (group_col * num_group_rows) + group_row;
    };

    // Reducer cores (one per KV head group)
    // With num_kv_heads=1, num_reducer_cores = B = num_output_cores (one reducer per batch)
    std::vector<uint32_t> reduce_core_physical_xs(num_reducer_cores);
    std::vector<uint32_t> reduce_core_physical_ys(num_reducer_cores);
    uint32_t reducer_count = 0;
    for (uint32_t i = 0; i < num_active_cores; ++i) {
        if (i % num_cores_per_head != 0) {
            continue;
        }
        auto physical = device->worker_core_from_logical_core(core_group[i]);
        // Reducer index: for single KV head case, reducer index = batch index
        // For multi KV head, would need: batch * num_kv_heads + head_within_batch
        uint32_t idx = use_col_major_group_indexing ? get_col_major_group_idx(reducer_count) : reducer_count;
        TT_FATAL(idx < num_reducer_cores, "Reducer spatial index {} out of bounds (max {})", idx, num_reducer_cores);
        reduce_core_physical_xs[idx] = physical.x;
        reduce_core_physical_ys[idx] = physical.y;
        reducer_count++;
    }

    // Output cores (one per batch)
    std::vector<uint32_t> output_core_physical_xs(num_output_cores);
    std::vector<uint32_t> output_core_physical_ys(num_output_cores);
    uint32_t output_count = 0;
    for (uint32_t i = 0; i < num_active_cores; ++i) {
        if (i % num_cores_per_batch != 0) {
            continue;
        }
        auto physical = device->worker_core_from_logical_core(core_group[i]);
        uint32_t idx = use_col_major_group_indexing ? get_col_major_group_idx(output_count) : output_count;
        TT_FATAL(idx < num_output_cores, "Output spatial index {} out of bounds (max {})", idx, num_output_cores);
        output_core_physical_xs[idx] = physical.x;
        output_core_physical_ys[idx] = physical.y;
        output_count++;
    }

    // All active cores (for tree reduction lookups)
    std::vector<uint32_t> reduction_group_core_xs(num_active_cores);
    std::vector<uint32_t> reduction_group_core_ys(num_active_cores);
    for (uint32_t i = 0; i < num_active_cores; ++i) {
        auto physical = device->worker_core_from_logical_core(core_group[i]);
        reduction_group_core_xs[i] = physical.x;
        reduction_group_core_ys[i] = physical.y;
    }

    log_debug(
        tt::LogOp,
        "Column-major group indexing: enabled={}, cores_per_head={}, groups_per_row={}",
        use_col_major_group_indexing,
        num_cores_per_head,
        num_group_rows);

    // ========== Compute Configuration ==========
    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
    const uint32_t max_dynamic_chunk_size = dst_size;
    const uint32_t Sk_chunk_t_cb_size = Sk_chunk_t == 0 ? max_dynamic_chunk_size : Sk_chunk_t;

    // Matmul block/subblock configuration for QK
    const uint32_t qk_in0_block_w = DHt;
    const uint32_t qk_num_blocks = 1;
    uint32_t qk_out_subblock_w = 0, qk_out_subblock_h = 0, qk_in0_num_subblocks = 0, qk_in1_num_subblocks = 0;
    if (Sk_chunk_t > 0) {
        qk_out_subblock_w = std::min(Sk_chunk_t, dst_size);
        qk_out_subblock_h = (qk_out_subblock_w == Sk_chunk_t) ? std::min(PNHt, dst_size / qk_out_subblock_w) : 1;
        qk_in0_num_subblocks = PNHt / qk_out_subblock_h;
        qk_in1_num_subblocks = Sk_chunk_t / qk_out_subblock_w;
    }

    // Matmul block/subblock configuration for output (QK * V)
    uint32_t out_in0_block_w = Sk_chunk_t > 0 ? Sk_chunk_t : 0;
    uint32_t out_num_blocks = Sk_chunk_t > 0 ? 1 : 0;
    const uint32_t out_out_subblock_w = std::min(vDHt, dst_size);
    const uint32_t out_out_subblock_h =
        (out_out_subblock_w == vDHt) ? std::min(PNHt, dst_size / out_out_subblock_w) : 1;
    const uint32_t out_in0_num_subblocks = PNHt / out_out_subblock_h;
    const uint32_t out_in1_num_subblocks = vDHt / out_out_subblock_w;

    // DHt granularity for compute loops (must be power of 2)
    uint32_t dht_granularity = std::min(DHt, dst_size);
    uint32_t log2_dht_granularity = std::log2(dht_granularity);
    if (dht_granularity != (1u << log2_dht_granularity)) {
        dht_granularity = 1;
        log2_dht_granularity = 0;
    }

    // ========== Tile Counts for Circular Buffers ==========
    const uint32_t q_tiles = PNHt * DHt;
    const uint32_t k_tiles = Sk_chunk_t_cb_size * DHt * 2;   // double buffer
    const uint32_t v_tiles = Sk_chunk_t_cb_size * vDHt * 2;  // double buffer
    const uint32_t qk_tiles = PNHt * Sk_chunk_t_cb_size;
    const uint32_t out_tiles = PNHt * vDHt;
    const uint32_t scale_tiles = 1;
    const uint32_t statistics_tiles = PNHt;
    const uint32_t intermed_output_tiles = (out_tiles + 2 * PNHt) * (num_cores_per_head - 1);

    // ========== Data Formats ==========
    const tt::DataFormat q_df = tt_metal::datatype_to_dataformat_converter(input_tensor_q.dtype());
    const tt::DataFormat k_df = tt_metal::datatype_to_dataformat_converter(input_tensor_k.dtype());
    const tt::DataFormat v_df = tt_metal::datatype_to_dataformat_converter(input_tensor_v.dtype());
    const tt::DataFormat out_df = tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    const tt::DataFormat mask_df = use_attention_mask
                                       ? tt_metal::datatype_to_dataformat_converter(attn_mask.value().dtype())
                                       : tt::DataFormat::Float16_b;
    const tt::DataFormat scalar_df = tt::DataFormat::Float16_b;
    const tt::DataFormat im_df = tt::DataFormat::Float16_b;
    const tt::DataFormat stats_df = tt::DataFormat::Float16_b;

    // ========== Tile Configurations ==========
    const auto half_tile = tt::tt_metal::Tile({16, 32});
    const auto full_tile = tt::tt_metal::Tile({32, 32});
    const bool use_half_tile = is_causal && num_q_heads <= 16 && q_df == tt::DataFormat::Float16_b;
    const auto q_tile = use_half_tile ? half_tile : full_tile;
    const auto k_tile = full_tile;
    const auto v_tile = full_tile;
    const auto mask_tile = use_half_tile ? half_tile : full_tile;
    const auto out_tile = full_tile;
    const auto scalar_tile = use_half_tile ? half_tile : full_tile;
    const auto im_tile = use_half_tile ? half_tile : full_tile;
    const auto stats_tile = use_half_tile ? half_tile : full_tile;
    const uint32_t q_tile_size = q_tile.get_tile_size(q_df);
    const uint32_t k_tile_size = k_tile.get_tile_size(k_df);
    const uint32_t v_tile_size = v_tile.get_tile_size(v_df);
    const uint32_t mask_tile_size = mask_tile.get_tile_size(mask_df);
    const uint32_t out_tile_size = out_tile.get_tile_size(out_df);
    const uint32_t scalar_tile_size = scalar_tile.get_tile_size(scalar_df);
    const uint32_t im_tile_size = im_tile.get_tile_size(im_df);
    const uint32_t stats_tile_size = stats_tile.get_tile_size(stats_df);

    // ========== Debug Logging ==========
    log_debug(tt::LogOp, "Dimensions: B={}, PNH={}, S={}, DH={}, vDH={}, Bkv={}", B, PNH, S, DH, vDH, Bkv);
    log_debug(tt::LogOp, "Tiles: St={}, DHt={}, vDHt={}, PNHt={}, Sk_chunk_t={}", St, DHt, vDHt, PNHt, Sk_chunk_t);
    log_debug(
        tt::LogOp, "Heads: kv={}, q={}, q_parallel_factor={}", num_kv_heads, num_q_heads, q_heads_parallel_factor);
    log_debug(
        tt::LogOp,
        "Cores: available={}, active={}, per_batch={}, per_head={}, reducers={}, outputs={}",
        num_cores_available,
        num_active_cores,
        num_cores_per_batch,
        num_cores_per_head,
        num_reducer_cores,
        num_output_cores);
    log_debug(tt::LogOp, "Tree reduction: {} rounds", num_tree_reduction_rounds);
    log_debug(
        tt::LogOp,
        "Flags: paged={}, q_sharded={}, q_local={}, mask={}, sink={}, half_tile={}",
        is_paged_attention,
        is_q_sharded,
        q_locally_available,
        use_attention_mask,
        use_attention_sink,
        use_half_tile);

    // Print reducer core coordinates
    log_debug(tt::LogOp, "Reducer cores ({}):", num_reducer_cores);
    for (uint32_t i = 0; i < num_reducer_cores; ++i) {
        log_debug(
            tt::LogOp, "  reducer[{}]: physical=({}, {})", i, reduce_core_physical_xs[i], reduce_core_physical_ys[i]);
    }

    // Print output core coordinates
    log_debug(tt::LogOp, "Output cores ({}):", num_output_cores);
    for (uint32_t i = 0; i < num_output_cores; ++i) {
        log_debug(
            tt::LogOp, "  output[{}]: physical=({}, {})", i, output_core_physical_xs[i], output_core_physical_ys[i]);
    }

    // Print reduction group core coordinates
    log_debug(tt::LogOp, "Reduction group cores ({}):", num_active_cores);
    for (uint32_t i = 0; i < num_active_cores; ++i) {
        log_debug(
            tt::LogOp, "  group[{}]: physical=({}, {})", i, reduction_group_core_xs[i], reduction_group_core_ys[i]);
    }

    // ========== Circular Buffer Creation ==========
    // Unified helper: tile!=nullptr sets tile_dims, buffer!=nullptr sets globally allocated address
    auto create_cb = [&](CBIndex idx,
                         uint32_t total_size,
                         tt::DataFormat df,
                         uint32_t page_size,
                         const tt::tt_metal::Tile* tile = nullptr,
                         Buffer* buffer = nullptr) -> CBHandle {
        auto config = CircularBufferConfig(total_size, {{idx, df}}).set_page_size(idx, page_size);
        if (tile != nullptr) {
            config.set_tile_dims(idx, *tile);
        }
        if (buffer != nullptr) {
            config.set_globally_allocated_address(*buffer);
        }
        return CreateCircularBuffer(program, core_grid, config);
    };

    // Input CBs
    auto cb_q_in_id = create_cb(
        CBIndex::c_0, q_tiles * q_tile_size, q_df, q_tile_size, &q_tile, q_locally_available ? q_buffer : nullptr);
    create_cb(CBIndex::c_1, k_tiles * k_tile_size, k_df, k_tile_size);                        // K input
    create_cb(CBIndex::c_2, v_tiles * v_tile_size, v_df, v_tile_size);                        // V input
    create_cb(CBIndex::c_3, qk_tiles * mask_tile_size, mask_df, mask_tile_size, &mask_tile);  // attn_mask
    if (use_attention_sink) {
        create_cb(CBIndex::c_4, statistics_tiles * stats_tile_size, stats_df, stats_tile_size, &stats_tile);
    }
    create_cb(CBIndex::c_5, scale_tiles * scalar_tile_size, scalar_df, scalar_tile_size, &scalar_tile);   // scale
    create_cb(CBIndex::c_6, statistics_tiles * stats_tile_size, stats_df, stats_tile_size, &stats_tile);  // m_in
    create_cb(CBIndex::c_7, statistics_tiles * stats_tile_size, stats_df, stats_tile_size, &stats_tile);  // l_in

    // Optional input CBs (cur_pos and page_table - raw data, no tile dims)
    CBHandle cb_cur_pos_id = 0;
    if (use_cur_pos_tensor) {
        cb_cur_pos_id = create_cb(
            CBIndex::c_8,
            cur_pos_stick_size,
            cur_pos_df,
            cur_pos_stick_size,
            nullptr,
            is_cur_pos_tensor_sharded ? cur_pos_buffer : nullptr);
    }
    CBHandle cb_page_table_id = 0;
    if (is_paged_attention) {
        uint32_t page_table_cb_size = is_page_table_sharded ? B * page_table_stick_size : page_table_stick_size;
        cb_page_table_id = create_cb(
            CBIndex::c_9,
            page_table_cb_size,
            page_table_df,
            page_table_stick_size,
            nullptr,
            is_page_table_sharded ? page_table_buffer : nullptr);
    }
    create_cb(CBIndex::c_10, q_tiles * q_tile_size, q_df, q_tile_size, &q_tile);  // tilized Q

    // Scalar/identity CBs
    const uint32_t col_identity_tile_size = full_tile.get_tile_size(scalar_df);
    create_cb(CBIndex::c_11, scale_tiles * col_identity_tile_size, scalar_df, col_identity_tile_size, &full_tile);
    create_cb(CBIndex::c_12, scale_tiles * scalar_tile_size, scalar_df, scalar_tile_size, &scalar_tile);
    if (sliding_window_size > 0) {
        create_cb(CBIndex::c_13, qk_tiles * mask_tile_size, mask_df, mask_tile_size, &mask_tile);
    }
    // Block padding mask (when block_size < TILE_HEIGHT, masks zero-padded rows in each K tile)
    if (has_block_padding) {
        create_cb(CBIndex::c_14, qk_tiles * mask_tile_size, mask_df, mask_tile_size, &mask_tile);
    }

    // Intermediate CBs
    create_cb(CBIndex::c_24, qk_tiles * im_tile_size, im_df, im_tile_size, &im_tile);
    create_cb(CBIndex::c_25, out_tiles * im_tile_size, im_df, im_tile_size, &im_tile);
    create_cb(CBIndex::c_26, out_tiles * im_tile_size, im_df, im_tile_size, &im_tile);
    create_cb(CBIndex::c_27, statistics_tiles * stats_tile_size, stats_df, stats_tile_size, &stats_tile);
    create_cb(CBIndex::c_28, statistics_tiles * stats_tile_size, stats_df, stats_tile_size, &stats_tile);
    create_cb(CBIndex::c_29, statistics_tiles * stats_tile_size, stats_df, stats_tile_size, &stats_tile);
    create_cb(CBIndex::c_30, statistics_tiles * stats_tile_size, stats_df, stats_tile_size, &stats_tile);
    create_cb(CBIndex::c_31, statistics_tiles * stats_tile_size, stats_df, stats_tile_size, &stats_tile);
    create_cb(CBIndex::c_21, statistics_tiles * stats_tile_size, stats_df, stats_tile_size, &stats_tile);
    create_cb(CBIndex::c_22, statistics_tiles * stats_tile_size, stats_df, stats_tile_size, &stats_tile);
    create_cb(CBIndex::c_23, out_tiles * im_tile_size, im_df, im_tile_size, &im_tile);

    // Output CBs
    create_cb(CBIndex::c_16, out_tiles * stats_tile_size, stats_df, stats_tile_size, &stats_tile);
    create_cb(CBIndex::c_17, statistics_tiles * stats_tile_size, stats_df, stats_tile_size, &stats_tile);
    create_cb(CBIndex::c_18, statistics_tiles * stats_tile_size, stats_df, stats_tile_size, &stats_tile);
    if (intermed_output_tiles > 0) {
        create_cb(CBIndex::c_19, intermed_output_tiles * stats_tile_size, stats_df, stats_tile_size, &stats_tile);
    }
    auto cb_out_final_id = create_cb(
        CBIndex::c_20,
        out_tiles * out_tile_size,
        out_df,
        out_tile_size,
        &out_tile,
        is_output_sharded ? out_buffer : nullptr);

    // ========== Kernel Scalars ==========
    const bfloat16 bfloat_identity_scalar(1.0f);
    const bfloat16 bfloat_zero_scalar(0.0f);
    const uint32_t packed_identity_scalar =
        pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});
    const uint32_t packed_zero_scalar = pack_two_bfloat16_into_uint32({bfloat_zero_scalar, bfloat_zero_scalar});

    union {
        float f;
        uint32_t u;
    } scale_union{};
    scale_union.f = scale;

    // ========== Semaphores ==========
    auto reducer_semaphore_id = tt_metal::CreateSemaphore(program, core_grid, 0);
    auto output_semaphore_id = tt_metal::CreateSemaphore(program, core_grid, 0);
    auto k_mcast_semaphore_id = tt_metal::CreateSemaphore(program, core_grid, 0);

    // If q is sharded, directly read in q_chunk_size_bytes if q is row major or tilized but with full tiles
    // If q is tilized and want to use tiny tiles, this is ignored since we need to skip bottom half of tiles
    const uint32_t q_chunk_size_bytes =
        q_tiles * (tilize_q ? num_q_heads * TILE_WIDTH * input_tensor_q.element_size() : q_tile_size);
    const uint32_t reuse_k = (tensor_args.v.has_value() ? 0 : 1);

    // ========== Compile Time Arguments ==========
    std::vector<uint32_t> reader_compile_time_args_common = {
        B,
        PNHt,
        St,
        DHt,
        vDHt,
        Sk_chunk_t,
        num_active_cores,
        is_q_sharded,
        num_cores_per_batch,
        k_chunk_size,
        cur_pos_stick_size,
        (uint32_t)is_paged_attention,
        num_kv_heads,
        page_block_size_t,
        Bkv,
        q_heads_parallel_factor,
        num_cores_per_head,
        num_heads_per_core,
        num_output_cores,
        is_causal,
        use_attention_mask,
        use_attention_sink,
        max_dynamic_chunk_size,
        tilize_q,
        reuse_k,
        use_half_tile,
        q_chunk_size_bytes,
        is_cur_pos_tensor_sharded,
        is_page_table_sharded,
        full_tile.get_tile_size(q_df),
        sliding_window_size,
        original_block_size,
        k_mcast_semaphore_id,
        (uint32_t)q_locally_available,
        (uint32_t)use_col_major_group_indexing,  // use_k_mcast
    };
    tt_metal::TensorAccessorArgs(input_tensor_q.buffer()).append_to(reader_compile_time_args_common);
    tt_metal::TensorAccessorArgs(input_tensor_k.buffer()).append_to(reader_compile_time_args_common);
    tt_metal::TensorAccessorArgs(input_tensor_v.buffer()).append_to(reader_compile_time_args_common);
    tt_metal::TensorAccessorArgs(attn_mask ? attn_mask->buffer() : nullptr).append_to(reader_compile_time_args_common);
    tt_metal::TensorAccessorArgs(cur_pos_tensor ? cur_pos_tensor->buffer() : nullptr)
        .append_to(reader_compile_time_args_common);
    tt_metal::TensorAccessorArgs(page_table_tensor ? page_table_tensor->buffer() : nullptr)
        .append_to(reader_compile_time_args_common);
    if (use_attention_sink) {
        tt_metal::TensorAccessorArgs(*attention_sink->buffer()).append_to(reader_compile_time_args_common);
    } else {
        reader_compile_time_args_common.push_back(0);
    }

    std::vector<uint32_t> writer_compile_time_args_common = {
        B,
        PNHt,
        St,
        DHt,
        vDHt,
        Sk_chunk_t,
        packed_identity_scalar,
        packed_zero_scalar,
        scale_union.u,
        num_cores_per_batch,
        num_active_cores,
        reducer_semaphore_id,
        output_semaphore_id,
        is_output_sharded,
        k_chunk_size,
        num_q_heads,
        num_kv_heads,
        num_cores_per_head,
        num_heads_per_core,
        num_reducer_cores,
        num_output_cores,
        output_tensor.element_size(),
        is_causal,
        max_dynamic_chunk_size,
        q_heads_parallel_factor,
        sliding_window_size,
        num_tree_reduction_rounds,
        original_block_size,
    };
    tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args_common);

    std::vector<uint32_t> compute_compile_time_args_common = {
        St,
        DHt,
        vDHt,
        PNHt,
        Sk_chunk_t,
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
        num_cores_per_batch,
        k_chunk_size,
        num_cores_per_head,
        num_heads_per_core,
        is_causal,
        use_attention_mask,
        use_attention_sink,
        max_dynamic_chunk_size,
        tilize_q,
        q_heads_parallel_factor,
        use_half_tile,
        scale_union.u,
        sliding_window_size,
        num_tree_reduction_rounds,
        original_block_size,
    };

    // ========== Compute Defines ==========
    std::map<std::string, std::string> compute_defines;
    compute_defines["EXP_APPROX_MODE"] = std::to_string(exp_approx_mode);
    compute_defines["DHT_GRANULARITY"] = std::to_string(dht_granularity);
    compute_defines["LOG2_DHT_GRANULARITY"] = std::to_string(log2_dht_granularity);

    if (Sk_chunk_t > 0) {
        auto add_granularity = [&](const char* name, uint32_t value) {
            uint32_t log2_val = std::log2(value);
            TT_FATAL(value == (1u << log2_val), "{} ({}) must be power of 2", name, value);
            compute_defines[name] = std::to_string(value);
            compute_defines[std::string("LOG2_") + name] = std::to_string(log2_val);
        };
        add_granularity("SUB_EXP_GRANULARITY", std::min(Sk_chunk_t, dst_size));
        add_granularity("MUL_BCAST_GRANULARITY", std::min(PNHt * Sk_chunk_t, dst_size));
        add_granularity("STATS_GRANULARITY", std::min(Sk_chunk_t, dst_size));
    } else {
        compute_defines["DYNAMIC_CHUNK_SIZE"] = "1";
    }

    // ========== Kernel Creation ==========
    const std::string kernel_path = "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/";

    auto compute_kernels_id = CreateKernel(
        program,
        kernel_path + "compute/sdpa_flash_decode.cpp",
        core_grid,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args_common,
            .defines = compute_defines});

    auto reader_kernels_id = CreateKernel(
        program,
        kernel_path + "dataflow/reader_decode_all.cpp",
        core_grid,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args_common));

    auto writer_kernels_id = CreateKernel(
        program,
        kernel_path + "dataflow/writer_decode_all.cpp",
        core_grid,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args_common));

    // ========== Buffer Addresses for Runtime Args ==========
    const uint32_t q_addr = q_buffer->address();
    const uint32_t k_addr = k_buffer->address();
    const uint32_t v_addr = v_buffer->address();
    const uint32_t out_addr = out_buffer->address();
    const uint32_t pos_addr = use_cur_pos_tensor ? cur_pos_tensor.value().buffer()->address() : 0;
    const uint32_t page_table_addr = is_paged_attention ? page_table_tensor.value().buffer()->address() : 0;
    const uint32_t attn_mask_addr = use_attention_mask ? attn_mask.value().buffer()->address() : 0;
    const uint32_t attention_sink_addr = use_attention_sink ? attention_sink.value().buffer()->address() : 0;

    // ========== Runtime Arguments ==========
    for (uint32_t i = 0; i < num_active_cores; ++i) {
        CoreCoord core = core_group[i];
        bool do_k_mcast = false;
        uint32_t mcast_x = 0, mcast_y0 = 0, mcast_y1 = 0, num_dests = 0;
        uint32_t cur_batch = 0, cur_head = 0, core_num_in_reduce = 0, core_num_in_output = 0;
        if (use_col_major_group_indexing) {
            uint32_t group_idx = i / num_cores_per_head;          // row-major group index
            uint32_t group_row = group_idx / num_group_rows;      // which row of groups (0 to grid_size.y-1)
            uint32_t group_col = group_idx % num_group_rows;      // which column of groups
            cur_batch = group_col * num_group_cols + group_row;   // column-major: batches go down columns first
            cur_head = 0;                                         // single KV head when using this indexing
            core_num_in_reduce =
                i % num_cores_per_head;               // position within the reduction group (0 to num_cores_per_head-1)
            core_num_in_output = core_num_in_reduce;  // same as reduce for single head
            do_k_mcast = (core.y % q_heads_parallel_factor == 0);
            num_dests = q_heads_parallel_factor - 1;
            if (do_k_mcast && num_dests > 0) {
                auto phys_start = device->worker_core_from_logical_core(CoreCoord{core.x, core.y + 1});
                auto phys_end = device->worker_core_from_logical_core(CoreCoord{core.x, core.y + num_dests});
                mcast_x = phys_start.x;
                mcast_y0 = phys_start.y;
                mcast_y1 = phys_end.y;
            }
        } else {
            cur_head = (i % num_cores_per_batch) / num_cores_per_head;
            cur_batch = i / num_cores_per_batch;
            core_num_in_reduce = i % num_cores_per_head;
            core_num_in_output = i % num_cores_per_batch;
        }
        uint32_t worker_id_for_reduce = (num_cores_per_head == 0) ? UINT32_MAX : core_num_in_reduce - 1;
        uint32_t worker_id_for_output = (core_num_in_output == 0) ? UINT32_MAX : core_num_in_output - 1;
        bool do_reduce = (worker_id_for_reduce == UINT32_MAX);
        bool do_output = (worker_id_for_output == UINT32_MAX);
        uint32_t cur_pos =
            (use_cur_pos_tensor || !is_causal) ? -1 : cur_pos_ids.at((uint32_t)(cur_batch / q_heads_parallel_factor));

        // Compute tree reduction parameters for this core
        TreeReductionParams tree_params = get_tree_reduction_params(core_num_in_reduce, num_cores_per_head);

        log_debug(tt::LogOp, "---- core_id: {}, coord: {} ----", i, core);
        log_debug(tt::LogOp, "worker_id_for_reduce: {}", worker_id_for_reduce);
        log_debug(tt::LogOp, "worker_id_for_output: {}", worker_id_for_output);
        log_debug(tt::LogOp, "do_reduce: {}", do_reduce);
        log_debug(tt::LogOp, "do_output: {}", do_output);
        log_debug(tt::LogOp, "cur_head: {}", cur_head);
        log_debug(tt::LogOp, "cur_batch: {}", cur_batch);
        log_debug(tt::LogOp, "core_num_in_reduce: {}", core_num_in_reduce);
        log_debug(tt::LogOp, "core_num_in_output: {}", core_num_in_output);
        log_debug(tt::LogOp, "cur_pos: {}", cur_pos);
        log_debug(tt::LogOp, "tree_params.is_root: {}", tree_params.is_root);
        log_debug(tt::LogOp, "tree_params.parent_core_in_group: {}", tree_params.parent_core_in_group);
        log_debug(tt::LogOp, "tree_params.send_at_round: {}", tree_params.send_at_round);
        log_debug(tt::LogOp, "tree_params.num_children: {}", tree_params.num_children);
        log_debug(tt::LogOp, "tree_params.my_active_rounds: {}", tree_params.my_active_rounds);
        log_debug(tt::LogOp, "do_k_mcast: {}", do_k_mcast);
        log_debug(tt::LogOp, "mcast_x: {}", mcast_x);
        log_debug(tt::LogOp, "mcast_y0: {}", mcast_y0);
        log_debug(tt::LogOp, "mcast_y1: {}", mcast_y1);
        log_debug(tt::LogOp, "num_dests: {}", num_dests);

        // Calculate base index for this reduction group's cores in the physical coordinate arrays
        // reduction_group_core_xs/ys are populated in row-major order (by linear index i)
        // So we use 'i' directly to find the start of this core's reduction group
        uint32_t reduction_group_base_idx;
        if (use_col_major_group_indexing) {
            // For column-major indexing: the group starts at (i / num_cores_per_head) * num_cores_per_head
            reduction_group_base_idx = (i / num_cores_per_head) * num_cores_per_head;
        } else {
            reduction_group_base_idx = (cur_batch * num_cores_per_batch) + (cur_head * num_cores_per_head);
        }
        log_debug(tt::LogOp, "reduction_group_base_idx: {}", reduction_group_base_idx);
        // reader runtime args
        std::vector<uint32_t> reader_rt_args = {
            q_addr,
            k_addr,
            v_addr,
            pos_addr,
            page_table_addr,
            attn_mask_addr,
            attention_sink_addr,
            page_table_stick_size,
            do_reduce,
            do_output,
            cur_head,
            cur_batch,
            core_num_in_reduce,
            core_num_in_output,
            cur_pos,
            do_k_mcast,
            mcast_x,
            mcast_y0,
            mcast_y1,
            num_dests,
        };
        reader_rt_args.insert(reader_rt_args.end(), output_core_physical_xs.begin(), output_core_physical_xs.end());
        reader_rt_args.insert(reader_rt_args.end(), output_core_physical_ys.begin(), output_core_physical_ys.end());

        // writer runtime args (do_reduce is NOT included — writer doesn't use it)
        std::vector<uint32_t> writer_rt_args = {
            out_addr,
            worker_id_for_reduce,
            worker_id_for_output,
            do_reduce,
            do_output,
            cur_head,
            cur_batch,
            core_num_in_reduce,
            core_num_in_output,
            cur_pos,
            // Tree reduction parameters
            tree_params.is_root ? 1u : 0u,
            tree_params.parent_core_in_group,
            tree_params.send_at_round,
            tree_params.num_children,
            tree_params.my_active_rounds,
            reduction_group_base_idx,
        };
        // Add children_per_round array (MAX_TREE_REDUCTION_ROUNDS elements)
        for (unsigned int children : tree_params.children_per_round) {
            writer_rt_args.push_back(children);
        }
        for (uint32_t c = 0; c < num_cores_per_head; ++c) {
            writer_rt_args.push_back(reduction_group_core_xs[reduction_group_base_idx + c]);
        }
        // Then add the y coordinates for all cores in this reduction group
        for (uint32_t c = 0; c < num_cores_per_head; ++c) {
            writer_rt_args.push_back(reduction_group_core_ys[reduction_group_base_idx + c]);
        }
        writer_rt_args.insert(writer_rt_args.end(), reduce_core_physical_xs.begin(), reduce_core_physical_xs.end());
        writer_rt_args.insert(writer_rt_args.end(), reduce_core_physical_ys.begin(), reduce_core_physical_ys.end());
        writer_rt_args.insert(writer_rt_args.end(), output_core_physical_xs.begin(), output_core_physical_xs.end());
        writer_rt_args.insert(writer_rt_args.end(), output_core_physical_ys.begin(), output_core_physical_ys.end());

        // compute runtime args
        std::vector<uint32_t> compute_rt_args = {
            do_reduce,
            do_output,
            cur_head,
            cur_batch,
            core_num_in_reduce,
            core_num_in_output,
            cur_pos,
            // Tree reduction parameters for compute
            tree_params.is_root ? 1u : 0u,
            tree_params.parent_core_in_group,
            tree_params.send_at_round,
            tree_params.num_children,
            tree_params.my_active_rounds,
        };
        // Add children_per_round array for compute
        for (unsigned int children : tree_params.children_per_round) {
            compute_rt_args.push_back(children);
        }
        SetRuntimeArgs(program, reader_kernels_id, core, reader_rt_args);
        SetRuntimeArgs(program, writer_kernels_id, core, writer_rt_args);
        SetRuntimeArgs(program, compute_kernels_id, core, compute_rt_args);
    }
    if (num_active_cores < num_cores_available) {
        log_debug(tt::LogOp, "idle cores {}", core_group_idle.size());
        // Set the rest of the cores to idle
        for (auto core : core_group_idle) {
            log_debug(tt::LogOp, "Setting core {} to idle", core);

            // Reader runtime args
            // Base args (20): includes K-mcast args [do_k_mcast, mcast_x, mcast_y0, mcast_y1, num_dests]
            std::vector<uint32_t> reader_rt_args(20, 0);

            // Writer runtime args - need to match the size with tree reduction params
            // Base args (10) + tree params (6) + children_per_round (MAX_TREE_REDUCTION_ROUNDS) + group coords
            // (2*num_cores_per_head)
            // + reducer coords + output coords
            std::vector<uint32_t> writer_rt_args(10 + 6 + MAX_TREE_REDUCTION_ROUNDS + (2 * num_cores_per_head), 0);

            // Compute runtime args - 65 indicates idle core
            // Base args (7) + tree params (5) + children_per_round (MAX_TREE_REDUCTION_ROUNDS)
            std::vector<uint32_t> compute_rt_args(7 + 5 + MAX_TREE_REDUCTION_ROUNDS, 0);
            compute_rt_args[0] = 65;  // Idle marker

            SetRuntimeArgs(program, reader_kernels_id, core, reader_rt_args);
            SetRuntimeArgs(program, writer_kernels_id, core, writer_rt_args);
            SetRuntimeArgs(program, compute_kernels_id, core, compute_rt_args);
        }
    }

    return cached_program_t{
        std::move(program),
        {.num_active_cores = num_active_cores,
         .core_group = core_group,
         .reader_kernels_id = reader_kernels_id,
         .writer_kernels_id = writer_kernels_id,
         .compute_kernels_id = compute_kernels_id,
         .num_cores_per_batch = num_cores_per_batch,
         .num_cores_per_head = num_cores_per_head,
         .num_output_cores = num_output_cores,
         .cb_q_in_id = cb_q_in_id,
         .cb_cur_pos_id = cb_cur_pos_id,
         .cb_page_table_id = cb_page_table_id,
         .is_q_sharded = is_q_sharded,
         .is_output_sharded = is_output_sharded,
         .q_locally_available = q_locally_available,
         .cb_out_final_id = cb_out_final_id,
         .B = B,
         .q_heads_parallel_factor = q_heads_parallel_factor,
         .use_cur_pos_tensor = use_cur_pos_tensor,
         .use_attention_mask = use_attention_mask,
         .use_attention_sink = use_attention_sink,
         .is_paged_attention = is_paged_attention,
         .is_causal = is_causal,
         .use_mla = use_mla,
         .use_col_major_group_indexing = use_col_major_group_indexing,
         .grid_size = grid_size,
         .num_group_rows = num_group_rows,
         .num_group_cols = num_group_cols}};
}

void SdpaDecodeProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const SdpaDecodeParams& operation_attributes,
    const SdpaDecodeInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;

    const auto& shared_variables = cached_program.shared_variables;
    const auto& num_active_cores = shared_variables.num_active_cores;
    const auto& core_group = shared_variables.core_group;
    const auto& reader_kernels_id = shared_variables.reader_kernels_id;
    const auto& writer_kernels_id = shared_variables.writer_kernels_id;
    const auto& compute_kernels_id = shared_variables.compute_kernels_id;
    const auto& num_cores_per_batch = shared_variables.num_cores_per_batch;
    const auto& num_cores_per_head = shared_variables.num_cores_per_head;
    const auto& cb_q_in_id = shared_variables.cb_q_in_id;
    const auto& cb_cur_pos_id = shared_variables.cb_cur_pos_id;
    const auto& cb_page_table_id = shared_variables.cb_page_table_id;
    const auto& is_output_sharded = shared_variables.is_output_sharded;
    const auto& q_locally_available = shared_variables.q_locally_available;
    const auto& cb_out_final_id = shared_variables.cb_out_final_id;
    const auto& q_heads_parallel_factor = shared_variables.q_heads_parallel_factor;
    const auto& cur_pos_ids = operation_attributes.cur_pos;
    const bool use_cur_pos_tensor = shared_variables.use_cur_pos_tensor;
    const bool use_attention_mask = shared_variables.use_attention_mask;
    const bool use_attention_sink = shared_variables.use_attention_sink;
    const bool is_paged_attention = shared_variables.is_paged_attention;
    const bool is_causal = shared_variables.is_causal;
    const bool use_col_major_group_indexing = shared_variables.use_col_major_group_indexing;
    const auto& grid_size = shared_variables.grid_size;
    const uint32_t num_group_cols = shared_variables.num_group_cols;

    auto* q_buffer = tensor_args.q.buffer();
    auto* k_buffer = tensor_args.k.buffer();
    auto* v_buffer = k_buffer;

    if (tensor_args.v.has_value()) {
        v_buffer = tensor_args.v.value().buffer();
    }

    auto* out_buffer = tensor_return_value.buffer();

    uint32_t q_addr = q_buffer->address();
    uint32_t k_addr = k_buffer->address();
    uint32_t v_addr = v_buffer->address();
    uint32_t out_addr = out_buffer->address();

    const auto& cur_pos_tensor = tensor_args.cur_pos_tensor;
    const auto& page_table_tensor = tensor_args.page_table_tensor;
    uint32_t pos_addr = use_cur_pos_tensor ? cur_pos_tensor.value().buffer()->address() : 0;

    uint32_t page_table_addr = is_paged_attention ? page_table_tensor.value().buffer()->address() : 0;
    uint32_t attn_mask_addr = use_attention_mask ? tensor_args.attn_mask.value().buffer()->address() : 0;
    uint32_t attention_sink_addr = use_attention_sink ? tensor_args.attention_sink.value().buffer()->address() : 0;
    auto* page_table_buffer = is_paged_attention ? page_table_tensor.value().buffer() : nullptr;
    uint32_t page_table_stick_size = is_paged_attention ? page_table_buffer->aligned_page_size() : 0;

    IDevice* dev = tensor_args.q.device();

    auto& reader_args_by_core = GetRuntimeArgs(program, reader_kernels_id);
    auto& writer_args_by_core = GetRuntimeArgs(program, writer_kernels_id);
    auto& compute_args_by_core = GetRuntimeArgs(program, compute_kernels_id);

    // Set rt args
    for (uint32_t i = 0; i < num_active_cores; ++i) {
        CoreCoord core = core_group[i];
        bool do_k_mcast = false;
        uint32_t mcast_x = 0, mcast_y0 = 0, mcast_y1 = 0, num_dests = 0;
        uint32_t cur_batch = 0, cur_head = 0, core_num_in_reduce = 0, core_num_in_output = 0;

        if (use_col_major_group_indexing) {
            uint32_t num_groups_per_row = std::max(1u, (uint32_t)(grid_size.x / num_cores_per_head));
            uint32_t group_idx = i / num_cores_per_head;          // row-major group index
            uint32_t group_row = group_idx / num_groups_per_row;  // which row of groups (0 to grid_size.y-1)
            uint32_t group_col = group_idx % num_groups_per_row;  // which column of groups
            cur_batch = group_col * num_group_cols + group_row;   // column-major: batches go down columns first
            cur_head = 0;                                         // single KV head when using this indexing
            core_num_in_reduce = i % num_cores_per_head;          // position within the reduction group
            core_num_in_output = core_num_in_reduce;              // same as reduce for single head
            do_k_mcast = (core.y % q_heads_parallel_factor == 0);
            num_dests = q_heads_parallel_factor - 1;
            if (do_k_mcast && num_dests > 0) {
                auto phys_start = dev->worker_core_from_logical_core(CoreCoord{core.x, core.y + 1});
                auto phys_end = dev->worker_core_from_logical_core(CoreCoord{core.x, core.y + num_dests});
                mcast_x = phys_start.x;
                mcast_y0 = phys_start.y;
                mcast_y1 = phys_end.y;
            }
        } else {
            cur_head = (i % num_cores_per_batch) / num_cores_per_head;
            cur_batch = i / num_cores_per_batch;
            core_num_in_reduce = i % num_cores_per_head;
            core_num_in_output = i % num_cores_per_batch;
        }

        uint32_t worker_id_for_reduce = (num_cores_per_head == 0) ? UINT32_MAX : core_num_in_reduce - 1;
        uint32_t worker_id_for_output = (core_num_in_output == 0) ? UINT32_MAX : core_num_in_output - 1;
        bool do_reduce = (worker_id_for_reduce == UINT32_MAX);
        bool do_output = (worker_id_for_output == UINT32_MAX);

        uint32_t cur_pos =
            (use_cur_pos_tensor || !is_causal) ? -1 : cur_pos_ids.at((uint32_t)(cur_batch / q_heads_parallel_factor));
        uint32_t reduction_group_base_idx;

        TreeReductionParams tree_params = get_tree_reduction_params(core_num_in_reduce, num_cores_per_head);
        if (use_col_major_group_indexing) {
            // For column-major indexing: the group starts at (i / num_cores_per_head) * num_cores_per_head
            reduction_group_base_idx = (i / num_cores_per_head) * num_cores_per_head;
        } else {
            reduction_group_base_idx = (cur_batch * num_cores_per_batch) + (cur_head * num_cores_per_head);
        }
        auto& reader_args = reader_args_by_core[core.x][core.y];
        auto& writer_args = writer_args_by_core[core.x][core.y];
        auto& compute_args = compute_args_by_core[core.x][core.y];

        // reader runtime args
        uint32_t arg_idx = 0;
        reader_args[arg_idx++] = q_addr;
        reader_args[arg_idx++] = k_addr;
        reader_args[arg_idx++] = v_addr;
        reader_args[arg_idx++] = pos_addr;
        reader_args[arg_idx++] = page_table_addr;
        reader_args[arg_idx++] = attn_mask_addr;
        reader_args[arg_idx++] = attention_sink_addr;
        reader_args[arg_idx++] = page_table_stick_size;
        reader_args[arg_idx++] = do_reduce;
        reader_args[arg_idx++] = do_output;
        reader_args[arg_idx++] = cur_head;
        reader_args[arg_idx++] = cur_batch;
        reader_args[arg_idx++] = core_num_in_reduce;
        reader_args[arg_idx++] = core_num_in_output;
        reader_args[arg_idx++] = cur_pos;

        reader_args[arg_idx++] = do_k_mcast;  // do_k_mcast
        reader_args[arg_idx++] = mcast_x;     // mcast_x
        reader_args[arg_idx++] = mcast_y0;    // mcast_y0
        reader_args[arg_idx++] = mcast_y1;    // mcast_y1
        reader_args[arg_idx++] = num_dests;   // num_dests

        // writer runtime args
        arg_idx = 0;
        writer_args[arg_idx++] = out_addr;
        writer_args[arg_idx++] = worker_id_for_reduce;
        writer_args[arg_idx++] = worker_id_for_output;
        writer_args[arg_idx++] = do_reduce;
        writer_args[arg_idx++] = do_output;
        writer_args[arg_idx++] = cur_head;
        writer_args[arg_idx++] = cur_batch;
        writer_args[arg_idx++] = core_num_in_reduce;
        writer_args[arg_idx++] = core_num_in_output;
        writer_args[arg_idx++] = cur_pos;
        writer_args[arg_idx++] = tree_params.is_root ? 1u : 0u;
        writer_args[arg_idx++] = tree_params.parent_core_in_group;
        writer_args[arg_idx++] = tree_params.send_at_round;
        writer_args[arg_idx++] = tree_params.num_children;
        writer_args[arg_idx++] = tree_params.my_active_rounds;
        writer_args[arg_idx++] = reduction_group_base_idx;
        // Add children_per_round array (MAX_TREE_REDUCTION_ROUNDS elements)
        for (unsigned int children : tree_params.children_per_round) {
            writer_args[arg_idx++] = children;
        }

        // compute runtime args
        arg_idx = 0;
        compute_args[arg_idx++] = do_reduce;
        compute_args[arg_idx++] = do_output;
        compute_args[arg_idx++] = cur_head;
        compute_args[arg_idx++] = cur_batch;
        compute_args[arg_idx++] = core_num_in_reduce;
        compute_args[arg_idx++] = core_num_in_output;
        compute_args[arg_idx++] = cur_pos;
        // Tree reduction parameters for compute
        compute_args[arg_idx++] = tree_params.is_root ? 1u : 0u;
        compute_args[arg_idx++] = tree_params.parent_core_in_group;
        compute_args[arg_idx++] = tree_params.send_at_round;
        compute_args[arg_idx++] = tree_params.num_children;
        compute_args[arg_idx++] = tree_params.my_active_rounds;
        // Add children_per_round array for compute
        for (unsigned int children : tree_params.children_per_round) {
            compute_args[arg_idx++] = children;
        }
    }
    if (q_locally_available) {
        UpdateDynamicCircularBufferAddress(program, cb_q_in_id, *q_buffer);
    }
    if (use_cur_pos_tensor && cur_pos_tensor.value().is_sharded()) {
        UpdateDynamicCircularBufferAddress(program, cb_cur_pos_id, *cur_pos_tensor.value().buffer());
    }
    if (is_paged_attention && page_table_tensor.value().is_sharded()) {
        UpdateDynamicCircularBufferAddress(program, cb_page_table_id, *page_table_tensor.value().buffer());
    }
    if (is_output_sharded) {
        UpdateDynamicCircularBufferAddress(program, cb_out_final_id, *out_buffer);
    }
}

}  // namespace ttnn::prim
