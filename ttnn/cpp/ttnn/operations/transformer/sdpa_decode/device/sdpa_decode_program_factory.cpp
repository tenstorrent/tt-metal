// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_decode_device_operation.hpp"

#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-logger/tt-logger.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

SdpaDecodeDeviceOperation::program_factory_t SdpaDecodeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return SdpaDecodeProgramFactory{};
}

ttnn::device_operation::ProgramArtifacts SdpaDecodeDeviceOperation::SdpaDecodeProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
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
    // capacity_t is in TILE rows (= cache_position_modulo / TILE_HEIGHT); 0 = unbounded.
    // Validator enforces cache_position_modulo % effective_block_size == 0, so the
    // tile-aligned divide is exact. Sets the kernel's compile-time wrap modulus on
    // every page_table lookup so a bounded sliding-window cache can be indexed by
    // absolute positions.
    const uint32_t cache_position_modulo = operation_attributes.cache_position_modulo.value_or(0);
    const uint32_t capacity_t = cache_position_modulo / TILE_HEIGHT;
    const bool share_cache = operation_attributes.share_cache.value_or(false);

    // V tensor: use K if MLA (V is subset of K), otherwise require explicit V
    TT_FATAL(use_mla || tensor_args.v.has_value(), "V tensor must be provided when MLA is disabled.");
    const auto& input_tensor_v = tensor_args.v.value_or(input_tensor_k);

    // ========== Device ==========
    IDevice* device = input_tensor_q.device();

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
    //
    // With an active PagedCacheGeometryOverride, this call reads a K/V cache allocated
    // for a different layer's shape; Q's last dim drives DH. Without it, k_shape[3] is used
    // and the strict q.head_dim == k.head_dim check in validate keeps legacy callers
    // byte-identical. Overrides are rejected under MLA in validate; do not apply them when
    // use_mla is true (same asymmetry fix as chunked prefill).
    const auto& geo = operation_attributes.paged_cache_geometry;
    const bool apply_geometry_override = !use_mla && geo.active();
    uint32_t B = q_shape[1];
    uint32_t PNH = q_shape[2];
    uint32_t S = k_shape[2];
    uint32_t DH = apply_geometry_override ? q_shape[3] : k_shape[3];
    uint32_t vDH = use_mla ? head_dim_v : (apply_geometry_override ? q_shape[3] : v_shape[3]);
    uint32_t Bkv = k_shape[0];
    const uint32_t Bmask = attn_mask.has_value() ? attn_mask->padded_shape()[0] : Bkv;
    // num_kv_heads from the cache view by default, or from the explicit override
    // when an HMA cross-group caller is reading a buffer allocated for a different
    // layer's spec (e.g. Gemma4-26B-A4B sliding kv=8 cache read by a full kv=2
    // layer). The override drives the kernel's per-block stride and head-parallel
    // reduction grid the same way the legacy cache shape did.
    uint32_t num_kv_heads = apply_geometry_override ? geo.num_kv_heads : k_shape[1];
    TT_FATAL(num_kv_heads > 0, "num_kv_heads must be > 0");
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
        uint32_t block_size = apply_geometry_override ? geo.block_size : k_shape[2];
        // original_block_size gates the sub-tile padding mask. With an override active,
        // validate already enforces it's a multiple of TILE_HEIGHT (no padding path).
        original_block_size = apply_geometry_override ? block_size : input_tensor_k.logical_shape()[2];
        page_block_size_t = block_size / TILE_HEIGHT;
        // kv_seq_len = max_num_blocks_per_seq * effective block_size.
        S = page_table_tensor.value().padded_shape()[-1] * block_size;
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
    TT_FATAL(max_cores_per_head > 0, "max_cores_per_head_batch must be > 0");
    const uint32_t max_num_cores_for_compute = max_cores_per_head * B * num_kv_heads;
    const uint32_t num_cores_per_batch_uncapped = std::min(num_cores_available, max_num_cores_for_compute) / B;
    const uint32_t num_cores_per_head = std::max(1u, num_cores_per_batch_uncapped / num_kv_heads);
    uint32_t num_heads_per_core =
        std::max(1u, static_cast<uint32_t>(std::ceil(static_cast<float>(num_kv_heads) / num_cores_per_batch_uncapped)));
    while (num_kv_heads % num_heads_per_core != 0) {
        num_heads_per_core++;
    }
    const uint32_t num_cores_per_batch = num_cores_per_head * num_kv_heads / num_heads_per_core;
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
    // (Buffer* pointers no longer flow through runtime args — every tensor address is delivered by a
    // TensorParameter binding. We still read the tensors' page-size / data-format metadata below.)
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
    std::vector<uint32_t> reduction_group_core_xs;
    std::vector<uint32_t> reduction_group_core_ys;
    reduction_group_core_xs.reserve(num_active_cores);
    reduction_group_core_ys.reserve(num_active_cores);
    for (uint32_t i = 0; i < num_active_cores; ++i) {
        auto physical = device->worker_core_from_logical_core(core_group[i]);
        reduction_group_core_xs.push_back(physical.x);
        reduction_group_core_ys.push_back(physical.y);
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
    const uint32_t Sk_chunk_t_dfb_size = Sk_chunk_t == 0 ? max_dynamic_chunk_size : Sk_chunk_t;

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
    uint32_t log2_dht_granularity = static_cast<uint32_t>(std::log2(dht_granularity));
    if (dht_granularity != (1u << log2_dht_granularity)) {
        dht_granularity = 1;
        log2_dht_granularity = 0;
    }

    // ========== Tile Counts for Circular Buffers ==========
    const uint32_t q_tiles = PNHt * DHt;
    const uint32_t k_tiles = Sk_chunk_t_dfb_size * DHt * 2;   // double buffer
    const uint32_t v_tiles = Sk_chunk_t_dfb_size * vDHt * 2;  // double buffer
    const uint32_t qk_tiles = PNHt * Sk_chunk_t_dfb_size;
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
    const tt::DataFormat scalar_df =
        (input_tensor_q.dtype() == DataType::FLOAT32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
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
    const uint32_t col_identity_tile_size = full_tile.get_tile_size(scalar_df);

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

    // ========== Kernel Scalars ==========
    const bfloat16 bfloat_identity_scalar(1.0f);
    const bfloat16 bfloat_zero_scalar(0.0f);
    const uint32_t packed_identity_scalar =
        pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});
    const uint32_t packed_zero_scalar = pack_two_bfloat16_into_uint32({bfloat_zero_scalar, bfloat_zero_scalar});
    const uint32_t scale_packed = std::bit_cast<uint32_t>(scale);

    // If q is sharded, directly read in q_chunk_size_bytes if q is row major or tilized but with full tiles
    // If q is tilized and want to use tiny tiles, this is ignored since we need to skip bottom half of tiles
    const uint32_t q_chunk_size_bytes =
        q_tiles * (tilize_q ? num_q_heads * TILE_WIDTH * input_tensor_q.element_size() : q_tile_size);
    const uint32_t reuse_k = (tensor_args.v.has_value() ? 0 : 1);

    // ======================================================================================
    //  Metal 2.0 spec construction
    // ======================================================================================

    // ---- Kernel / resource names ----
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    const TensorParamName Q{"q"};
    const TensorParamName K{"k"};
    const TensorParamName VP{"v"};
    const TensorParamName CUR_POS{"cur_pos"};
    const TensorParamName PAGE_TABLE{"page_table"};
    const TensorParamName ATTN_MASK{"attn_mask"};
    const TensorParamName ATTN_SINK{"attn_sink"};
    const TensorParamName OUT_T{"out"};

    const SemaphoreSpecName SEM_REDUCER{"reducer"};
    const SemaphoreSpecName SEM_OUTPUT{"output"};
    const SemaphoreSpecName SEM_K_MCAST{"k_mcast"};

    const DFBSpecName DFB_Q_IN{"q_in"};
    const DFBSpecName DFB_K_IN{"k_in"};
    const DFBSpecName DFB_V_IN{"v_in"};
    const DFBSpecName DFB_MASK_IN{"mask_in"};
    const DFBSpecName DFB_ATTN_SINK{"attention_sink"};
    const DFBSpecName DFB_IDENTITY_SCALE{"identity_scale_in"};
    const DFBSpecName DFB_M_IN{"m_in"};
    const DFBSpecName DFB_L_IN{"l_in"};
    const DFBSpecName DFB_WRITER_CUR_POS{"writer_cur_pos"};
    const DFBSpecName DFB_PAGE_TABLE{"page_table"};
    const DFBSpecName DFB_Q_RM{"q_rm"};
    const DFBSpecName DFB_COL_IDENTITY{"col_identity"};
    const DFBSpecName DFB_ZERO_IN{"zero_in"};
    const DFBSpecName DFB_SLIDING_MASK{"sliding_window_mask_in"};
    const DFBSpecName DFB_BLOCK_PAD_MASK{"block_pad_mask"};
    const DFBSpecName DFB_COMPUTE_CUR_POS{"compute_cur_pos"};
    const DFBSpecName DFB_OUT_O{"out_o"};
    const DFBSpecName DFB_OUT_M{"out_m"};
    const DFBSpecName DFB_OUT_L{"out_l"};
    const DFBSpecName DFB_INTERMED_OUT{"intermed_out"};
    const DFBSpecName DFB_OUT{"out"};
    const DFBSpecName DFB_QK_IM{"qk_im"};
    const DFBSpecName DFB_OUT_IM{"out_im"};
    const DFBSpecName DFB_OUT_ACC_IM{"out_accumulate_im"};
    const DFBSpecName DFB_MAX_1{"max_1"};
    const DFBSpecName DFB_MAX_2{"max_2"};
    const DFBSpecName DFB_SUM_1{"sum_1"};
    const DFBSpecName DFB_SUM_2{"sum_2"};
    const DFBSpecName DFB_EXP_MAX_DIFF{"exp_max_diff"};
    const DFBSpecName DFB_PREV_SUM_2{"prev_sum_2"};
    const DFBSpecName DFB_EXP_MAX_DIFF_2{"exp_max_diff_2"};
    const DFBSpecName DFB_OUT_ACC_IM_2{"out_accumulate_im_2"};

    // ---- DFB specs + per-kernel endpoint bindings ----
    Group<DataflowBufferSpec> dfbs;
    Group<DFBBinding> reader_dfb;
    Group<DFBBinding> writer_dfb;
    Group<DFBBinding> compute_dfb;

    auto add_dfb = [&](const DFBSpecName& name,
                       uint32_t entry_size,
                       uint32_t num_entries,
                       tt::DataFormat df,
                       const tt::tt_metal::Tile* tile,
                       std::optional<TensorParamName> borrowed = std::nullopt,
                       bool multi = false) {
        DataflowBufferSpec s{
            .unique_id = name,
            .entry_size = entry_size,
            .num_entries = num_entries,
            .data_format_metadata = df,
        };
        if (tile != nullptr) {
            s.tile_format_metadata = *tile;
        }
        if (borrowed.has_value()) {
            s.borrowed_from = borrowed;
        }
        if (multi) {
            s.advanced_options.allow_instance_multi_binding = true;
        }
        dfbs.push_back(std::move(s));
    };
    auto bind = [](Group<DFBBinding>& g, const DFBSpecName& name, std::string accessor, DFBEndpointType role) {
        g.push_back(DFBBinding{
            .dfb_spec_name = name,
            .accessor_name = std::move(accessor),
            .endpoint_type = role,
        });
    };

    // c_0 q_in / c_10 q_rm — config-flip.
    //   tilize_q  : reader fills q_rm (P); compute tilizes q_rm->q_in (q_in self-loop), consumes q_rm.
    //   !tilize_q : reader fills q_in (P); compute consumes q_in. q_rm not used (dropped).
    add_dfb(
        DFB_Q_IN,
        q_tile_size,
        q_tiles,
        q_df,
        &q_tile,
        q_locally_available ? std::optional<TensorParamName>(Q) : std::nullopt);
    if (tilize_q) {
        // c_10 (q_rm) is never borrowed — matching legacy add_cb(c_10, ...) which passes no buffer. Only c_0
        // (the tilize output / borrowed-Q input) borrows Q under q_locally_available, as legacy does. (A
        // borrow here would alias the tilize source and output when q_locally_available && tilize_q — a
        // combination MLA never produces, since MLA Q is tiled — but the port must not add a borrow legacy
        // lacks.)
        add_dfb(DFB_Q_RM, q_tile_size, q_tiles, q_df, &q_tile);
        bind(reader_dfb, DFB_Q_RM, "q_rm", DFBEndpointType::PRODUCER);
        bind(compute_dfb, DFB_Q_RM, "q_rm", DFBEndpointType::CONSUMER);
        bind(compute_dfb, DFB_Q_IN, "q_in", DFBEndpointType::PRODUCER);
        bind(compute_dfb, DFB_Q_IN, "q_in", DFBEndpointType::CONSUMER);
    } else {
        bind(reader_dfb, DFB_Q_IN, "q_in", DFBEndpointType::PRODUCER);
        bind(compute_dfb, DFB_Q_IN, "q_in", DFBEndpointType::CONSUMER);
    }

    // c_1 k_in / c_2 v_in — reader produces, compute consumes.
    add_dfb(DFB_K_IN, k_tile_size, k_tiles, k_df, nullptr);
    bind(reader_dfb, DFB_K_IN, "k_in", DFBEndpointType::PRODUCER);
    bind(compute_dfb, DFB_K_IN, "k_in", DFBEndpointType::CONSUMER);
    add_dfb(DFB_V_IN, v_tile_size, v_tiles, v_df, nullptr);
    bind(reader_dfb, DFB_V_IN, "v_in", DFBEndpointType::PRODUCER);
    bind(compute_dfb, DFB_V_IN, "v_in", DFBEndpointType::CONSUMER);

    // c_3 mask_in — producer depends on config; compute always references it (matmul mask arg).
    add_dfb(DFB_MASK_IN, mask_tile_size, qk_tiles, mask_df, &mask_tile);
    if (is_causal) {
        bind(writer_dfb, DFB_MASK_IN, "mask_in", DFBEndpointType::PRODUCER);
        bind(compute_dfb, DFB_MASK_IN, "mask_in", DFBEndpointType::CONSUMER);
    } else if (use_attention_mask) {
        bind(reader_dfb, DFB_MASK_IN, "mask_in", DFBEndpointType::PRODUCER);
        bind(compute_dfb, DFB_MASK_IN, "mask_in", DFBEndpointType::CONSUMER);
    } else {
        // No mask produced/consumed via FIFO; compute only references the handle. Self-loop on compute.
        bind(compute_dfb, DFB_MASK_IN, "mask_in", DFBEndpointType::PRODUCER);
        bind(compute_dfb, DFB_MASK_IN, "mask_in", DFBEndpointType::CONSUMER);
    }

    // c_4 attention_sink — reader produces, compute consumes (conditional).
    if (use_attention_sink) {
        add_dfb(DFB_ATTN_SINK, stats_tile_size, statistics_tiles, stats_df, &stats_tile);
        bind(reader_dfb, DFB_ATTN_SINK, "attention_sink", DFBEndpointType::PRODUCER);
        bind(compute_dfb, DFB_ATTN_SINK, "attention_sink", DFBEndpointType::CONSUMER);
    }

    // c_5 identity_scale — writer produces, compute consumes.
    add_dfb(DFB_IDENTITY_SCALE, scalar_tile_size, scale_tiles, scalar_df, &scalar_tile);
    bind(writer_dfb, DFB_IDENTITY_SCALE, "identity_scale_in", DFBEndpointType::PRODUCER);
    bind(compute_dfb, DFB_IDENTITY_SCALE, "identity_scale_in", DFBEndpointType::CONSUMER);

    // c_6 m_in / c_7 l_in — writer produces (receives child stats), compute consumes.
    add_dfb(DFB_M_IN, stats_tile_size, statistics_tiles, stats_df, &stats_tile);
    bind(writer_dfb, DFB_M_IN, "m_in", DFBEndpointType::PRODUCER);
    bind(compute_dfb, DFB_M_IN, "m_in", DFBEndpointType::CONSUMER);
    add_dfb(DFB_L_IN, stats_tile_size, statistics_tiles, stats_df, &stats_tile);
    bind(writer_dfb, DFB_L_IN, "l_in", DFBEndpointType::PRODUCER);
    bind(compute_dfb, DFB_L_IN, "l_in", DFBEndpointType::CONSUMER);

    // c_8 writer_cur_pos / c_15 compute_cur_pos — reader produces, writer/compute consume (conditional).
    if (use_cur_pos_tensor) {
        add_dfb(
            DFB_WRITER_CUR_POS,
            cur_pos_stick_size,
            1,
            cur_pos_df,
            nullptr,
            is_cur_pos_tensor_sharded ? std::optional<TensorParamName>(CUR_POS) : std::nullopt);
        bind(reader_dfb, DFB_WRITER_CUR_POS, "writer_cur_pos", DFBEndpointType::PRODUCER);
        bind(writer_dfb, DFB_WRITER_CUR_POS, "cur_pos", DFBEndpointType::CONSUMER);
        add_dfb(DFB_COMPUTE_CUR_POS, cur_pos_stick_size, 1, cur_pos_df, nullptr);
        bind(reader_dfb, DFB_COMPUTE_CUR_POS, "compute_cur_pos", DFBEndpointType::PRODUCER);
        bind(compute_dfb, DFB_COMPUTE_CUR_POS, "cur_pos", DFBEndpointType::CONSUMER);
    }

    // c_9 page_table — reader fills + raw-reads its own buffer (self-loop) (conditional).
    if (is_paged_attention) {
        uint32_t page_table_num_entries = is_page_table_sharded ? B : 1;
        add_dfb(
            DFB_PAGE_TABLE,
            page_table_stick_size,
            page_table_num_entries,
            page_table_df,
            nullptr,
            is_page_table_sharded ? std::optional<TensorParamName>(PAGE_TABLE) : std::nullopt);
        bind(reader_dfb, DFB_PAGE_TABLE, "page_table", DFBEndpointType::PRODUCER);
        bind(reader_dfb, DFB_PAGE_TABLE, "page_table", DFBEndpointType::CONSUMER);
    }

    // c_11 col_identity — writer produces; no consumer in sdpa_decode's compute (dead-but-kept).
    // (It is consumed by sdpa *prefill*'s matmul_reduce, so this reads as carried-over dead code from a
    // matmul-based reduce path. Removing it is an ops-team cleanup, not a port drop, so it is preserved
    // here as a single-toucher self-loop with zero functional effect.)
    add_dfb(DFB_COL_IDENTITY, col_identity_tile_size, scale_tiles, scalar_df, &full_tile);
    bind(writer_dfb, DFB_COL_IDENTITY, "col_identity", DFBEndpointType::PRODUCER);
    bind(writer_dfb, DFB_COL_IDENTITY, "col_identity", DFBEndpointType::CONSUMER);

    // c_12 zero_in — writer produces, compute consumes.
    add_dfb(DFB_ZERO_IN, scalar_tile_size, scale_tiles, scalar_df, &scalar_tile);
    bind(writer_dfb, DFB_ZERO_IN, "zero_in", DFBEndpointType::PRODUCER);
    bind(compute_dfb, DFB_ZERO_IN, "zero_in", DFBEndpointType::CONSUMER);

    // c_13 sliding_window_mask — writer produces, compute consumes (conditional).
    if (sliding_window_size > 0) {
        add_dfb(DFB_SLIDING_MASK, mask_tile_size, qk_tiles, mask_df, &mask_tile);
        bind(writer_dfb, DFB_SLIDING_MASK, "sliding_window_mask_in", DFBEndpointType::PRODUCER);
        bind(compute_dfb, DFB_SLIDING_MASK, "sliding_window_mask_in", DFBEndpointType::CONSUMER);
    }

    // c_14 block_pad_mask — writer produces, compute consumes (conditional).
    if (has_block_padding) {
        add_dfb(DFB_BLOCK_PAD_MASK, mask_tile_size, qk_tiles, mask_df, &mask_tile);
        bind(writer_dfb, DFB_BLOCK_PAD_MASK, "block_pad_mask", DFBEndpointType::PRODUCER);
        bind(compute_dfb, DFB_BLOCK_PAD_MASK, "block_pad_mask", DFBEndpointType::CONSUMER);
    }

    // c_16 out_o/out_worker — tree-reduction multi-binding (writer P+C, compute P+C).
    add_dfb(DFB_OUT_O, stats_tile_size, out_tiles, stats_df, &stats_tile, std::nullopt, /*multi=*/true);
    bind(writer_dfb, DFB_OUT_O, "out_o", DFBEndpointType::PRODUCER);
    bind(writer_dfb, DFB_OUT_O, "out_worker", DFBEndpointType::CONSUMER);
    bind(compute_dfb, DFB_OUT_O, "out_o", DFBEndpointType::PRODUCER);
    bind(compute_dfb, DFB_OUT_O, "out_o", DFBEndpointType::CONSUMER);

    // c_17 out_m / c_18 out_l — compute produces, writer consumes.
    add_dfb(DFB_OUT_M, stats_tile_size, statistics_tiles, stats_df, &stats_tile);
    bind(compute_dfb, DFB_OUT_M, "out_m", DFBEndpointType::PRODUCER);
    bind(writer_dfb, DFB_OUT_M, "out_m", DFBEndpointType::CONSUMER);
    add_dfb(DFB_OUT_L, stats_tile_size, statistics_tiles, stats_df, &stats_tile);
    bind(compute_dfb, DFB_OUT_L, "out_l", DFBEndpointType::PRODUCER);
    bind(writer_dfb, DFB_OUT_L, "out_l", DFBEndpointType::CONSUMER);

    // c_19 intermed_out — writer raw cross-core read/write (self-loop) (conditional).
    if (intermed_output_tiles > 0) {
        add_dfb(DFB_INTERMED_OUT, stats_tile_size, intermed_output_tiles, stats_df, &stats_tile);
        bind(writer_dfb, DFB_INTERMED_OUT, "intermed_out", DFBEndpointType::PRODUCER);
        bind(writer_dfb, DFB_INTERMED_OUT, "intermed_out", DFBEndpointType::CONSUMER);
    }

    // c_20 out — compute produces, writer consumes (final output shard).
    add_dfb(
        DFB_OUT,
        out_tile_size,
        out_tiles,
        out_df,
        &out_tile,
        is_output_sharded ? std::optional<TensorParamName>(OUT_T) : std::nullopt);
    bind(compute_dfb, DFB_OUT, "out", DFBEndpointType::PRODUCER);
    bind(writer_dfb, DFB_OUT, "out", DFBEndpointType::CONSUMER);

    // c_21..c_31 compute intermediates — compute self-loop.
    auto add_compute_intermediate = [&](const DFBSpecName& name,
                                        std::string accessor,
                                        uint32_t entry,
                                        uint32_t n,
                                        tt::DataFormat df,
                                        const tt::tt_metal::Tile* tile) {
        add_dfb(name, entry, n, df, tile);
        bind(compute_dfb, name, accessor, DFBEndpointType::PRODUCER);
        bind(compute_dfb, name, std::move(accessor), DFBEndpointType::CONSUMER);
    };
    add_compute_intermediate(DFB_QK_IM, "qk_im", im_tile_size, qk_tiles, im_df, &im_tile);
    add_compute_intermediate(DFB_OUT_IM, "out_im", im_tile_size, out_tiles, im_df, &im_tile);
    add_compute_intermediate(DFB_OUT_ACC_IM, "out_accumulate_im", im_tile_size, out_tiles, im_df, &im_tile);
    add_compute_intermediate(DFB_MAX_1, "max_1", stats_tile_size, statistics_tiles, stats_df, &stats_tile);
    add_compute_intermediate(DFB_MAX_2, "max_2", stats_tile_size, statistics_tiles, stats_df, &stats_tile);
    add_compute_intermediate(DFB_SUM_1, "sum_1", stats_tile_size, statistics_tiles, stats_df, &stats_tile);
    add_compute_intermediate(DFB_SUM_2, "sum_2", stats_tile_size, statistics_tiles, stats_df, &stats_tile);
    add_compute_intermediate(
        DFB_EXP_MAX_DIFF, "exp_max_diff", stats_tile_size, statistics_tiles, stats_df, &stats_tile);
    add_compute_intermediate(DFB_PREV_SUM_2, "prev_sum_2", stats_tile_size, statistics_tiles, stats_df, &stats_tile);
    add_compute_intermediate(
        DFB_EXP_MAX_DIFF_2, "exp_max_diff_2", stats_tile_size, statistics_tiles, stats_df, &stats_tile);
    add_compute_intermediate(DFB_OUT_ACC_IM_2, "out_accumulate_im_2", im_tile_size, out_tiles, im_df, &im_tile);

    // ---- Tensor parameters + bindings ----
    Group<TensorParameter> tensor_params;
    Group<TensorBinding> reader_tensors;
    Group<TensorBinding> writer_tensors;
    auto add_tensor = [&](const TensorParamName& name, const ttnn::Tensor& t) {
        tensor_params.push_back(TensorParameter{.unique_id = name, .spec = t.tensor_spec()});
    };
    // Q — reader binds via TensorAccessor except when locally available (borrowed into q_in/q_rm).
    add_tensor(Q, input_tensor_q);
    if (!q_locally_available) {
        reader_tensors.push_back(TensorBinding{.tensor_parameter_name = Q, .accessor_name = "q"});
    }
    add_tensor(K, input_tensor_k);
    reader_tensors.push_back(TensorBinding{.tensor_parameter_name = K, .accessor_name = "k"});
    // V — only an independent tensor read when !reuse_k. Under reuse_k (MLA) V rides on K's L1.
    if (!reuse_k) {
        add_tensor(VP, tensor_args.v.value());
        reader_tensors.push_back(TensorBinding{.tensor_parameter_name = VP, .accessor_name = "v"});
    }
    if (use_cur_pos_tensor && !is_cur_pos_tensor_sharded) {
        add_tensor(CUR_POS, cur_pos_tensor.value());
        reader_tensors.push_back(TensorBinding{.tensor_parameter_name = CUR_POS, .accessor_name = "cur_pos"});
    } else if (use_cur_pos_tensor) {
        // sharded: borrowed into c_8; borrow keeps the parameter "used" with no TensorBinding.
        add_tensor(CUR_POS, cur_pos_tensor.value());
    }
    if (is_paged_attention && !is_page_table_sharded) {
        add_tensor(PAGE_TABLE, page_table_tensor.value());
        reader_tensors.push_back(TensorBinding{.tensor_parameter_name = PAGE_TABLE, .accessor_name = "page_table"});
    } else if (is_paged_attention) {
        // sharded: borrowed into c_9.
        add_tensor(PAGE_TABLE, page_table_tensor.value());
    }
    if (use_attention_mask) {
        add_tensor(ATTN_MASK, attn_mask.value());
        reader_tensors.push_back(TensorBinding{.tensor_parameter_name = ATTN_MASK, .accessor_name = "attn_mask"});
    }
    if (use_attention_sink) {
        add_tensor(ATTN_SINK, attention_sink.value());
        reader_tensors.push_back(TensorBinding{.tensor_parameter_name = ATTN_SINK, .accessor_name = "attention_sink"});
    }
    add_tensor(OUT_T, output_tensor);
    if (!is_output_sharded) {
        writer_tensors.push_back(TensorBinding{.tensor_parameter_name = OUT_T, .accessor_name = "out"});
    }
    // Note: when output is sharded, c_20 borrows from OUT_T; the writer still writes the shard via the
    // borrowed DFB / peer reads. The writer's out_writer TensorAccessor is only used on the !sharded path,
    // so it is bound only there. On the sharded path OUT_T stays "used" via the borrow.

    // ---- Semaphores + bindings ----
    Group<SemaphoreSpec> semaphores;
    Group<SemaphoreBinding> reader_sems;
    Group<SemaphoreBinding> writer_sems;
    // Faithful to legacy: kernels, DFBs, and semaphores are placed on the FULL core grid, and idle
    // cores (core_group_idle) are marked with the do_reduce==65 sentinel and early-return in every
    // kernel. (Legacy keyed idle off addr==0 runtime args; Metal 2.0 injects buffer addresses via
    // TensorBindings, so there is no addr RTA to zero — the compute kernel's ==65 idle marker is
    // reused across reader/writer/compute instead.)
    const NodeRangeSet full_node_set = core_grid;
    semaphores.push_back(SemaphoreSpec{.unique_id = SEM_REDUCER, .target_nodes = full_node_set});
    semaphores.push_back(SemaphoreSpec{.unique_id = SEM_OUTPUT, .target_nodes = full_node_set});
    semaphores.push_back(SemaphoreSpec{.unique_id = SEM_K_MCAST, .target_nodes = full_node_set});
    reader_sems.push_back(SemaphoreBinding{.semaphore_spec_name = SEM_K_MCAST, .accessor_name = "k_mcast"});
    writer_sems.push_back(SemaphoreBinding{.semaphore_spec_name = SEM_REDUCER, .accessor_name = "reducer"});
    writer_sems.push_back(SemaphoreBinding{.semaphore_spec_name = SEM_OUTPUT, .accessor_name = "output"});

    // ---- Named compile-time args ----
    KernelSpec::CompileTimeArgs reader_cta{
        {"B", B},
        {"PNHt", PNHt},
        {"St", St},
        {"DHt", DHt},
        {"vDHt", vDHt},
        {"Sk_chunk_t", Sk_chunk_t},
        {"num_cores", num_active_cores},
        {"is_q_sharded", static_cast<uint32_t>(is_q_sharded)},
        {"num_cores_per_batch", num_cores_per_batch},
        {"k_chunk_size", k_chunk_size},
        {"index_stick_size_B", cur_pos_stick_size},
        {"num_kv_heads", num_kv_heads},
        {"block_size_t", page_block_size_t},
        {"Bkv", Bkv},
        {"q_heads_parallel_factor", q_heads_parallel_factor},
        {"num_cores_per_head", num_cores_per_head},
        {"num_heads_per_core", num_heads_per_core},
        {"num_output_cores", num_output_cores},
        {"max_dynamic_chunk_size", max_dynamic_chunk_size},
        {"reuse_k", reuse_k},
        {"use_half_tile", static_cast<uint32_t>(use_half_tile)},
        {"q_chunk_size_bytes", q_chunk_size_bytes},
        {"sliding_window_size", sliding_window_size},
        {"original_block_size", original_block_size},
        {"Bmask", Bmask},
        {"capacity_t", capacity_t},
        {"use_k_mcast", static_cast<uint32_t>(use_col_major_group_indexing)},
    };
    KernelSpec::CompileTimeArgs writer_cta{
        {"B", B},
        {"PNHt", PNHt},
        {"St", St},
        {"DHt", DHt},
        {"vDHt", vDHt},
        {"Sk_chunk_t", Sk_chunk_t},
        {"identity_scalar_packed", packed_identity_scalar},
        {"zero_scalar_packed", packed_zero_scalar},
        {"scale_val", scale_packed},
        {"num_cores_per_batch", num_cores_per_batch},
        {"num_cores", num_active_cores},
        {"k_chunk_size", k_chunk_size},
        {"num_q_heads", num_q_heads},
        {"num_kv_heads", num_kv_heads},
        {"num_cores_per_head", num_cores_per_head},
        {"num_heads_per_core", num_heads_per_core},
        {"num_reducer_cores", num_reducer_cores},
        {"num_output_cores", num_output_cores},
        {"ELEMENT_SIZE", static_cast<uint32_t>(output_tensor.element_size())},
        {"max_dynamic_chunk_size", max_dynamic_chunk_size},
        {"q_heads_parallel_factor", q_heads_parallel_factor},
        {"sliding_window_size", sliding_window_size},
        {"num_tree_reduction_rounds", num_tree_reduction_rounds},
        {"original_block_size", original_block_size},
    };
    KernelSpec::CompileTimeArgs compute_cta{
        {"St", St},
        {"DHt", DHt},
        {"vDHt", vDHt},
        {"Sq_chunk_t", PNHt},
        {"Sk_chunk_t", Sk_chunk_t},
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
        {"num_cores_per_head", num_cores_per_head},
        {"num_heads_per_core", num_heads_per_core},
        {"max_dynamic_chunk_size", max_dynamic_chunk_size},
        {"q_heads_parallel_factor", q_heads_parallel_factor},
        {"use_half_tile", static_cast<uint32_t>(use_half_tile)},
        {"scale_fp32", scale_packed},
        {"sliding_window_size", sliding_window_size},
        {"num_tree_reduction_rounds", num_tree_reduction_rounds},
        {"original_block_size", original_block_size},
    };

    // ---- Preprocessor defines (config-gated resource references) ----
    KernelSpec::CompilerOptions::Defines reader_defines;
    KernelSpec::CompilerOptions::Defines writer_defines;
    KernelSpec::CompilerOptions::Defines compute_defines;
    auto set_flag = [](KernelSpec::CompilerOptions::Defines& d, const char* name, bool on) {
        if (on) {
            d.insert({name, "1"});
        }
    };
    // Reader
    set_flag(reader_defines, "IS_CAUSAL", is_causal);
    set_flag(reader_defines, "USE_ATTENTION_MASK", use_attention_mask);
    set_flag(reader_defines, "USE_ATTENTION_SINK", use_attention_sink);
    set_flag(reader_defines, "IS_PAGED_ATTENTION", is_paged_attention);
    set_flag(reader_defines, "USE_CUR_POS_TENSOR", use_cur_pos_tensor);
    set_flag(reader_defines, "TILIZE_Q", tilize_q);
    set_flag(reader_defines, "Q_LOCALLY_AVAILABLE", q_locally_available);
    set_flag(reader_defines, "IS_CUR_POS_TENSOR_SHARDED", is_cur_pos_tensor_sharded);
    set_flag(reader_defines, "IS_PAGE_TABLE_SHARDED", is_page_table_sharded);
    set_flag(reader_defines, "REUSE_K", reuse_k != 0);
    // Writer
    set_flag(writer_defines, "IS_CAUSAL", is_causal);
    set_flag(writer_defines, "IS_OUT_SHARDED", is_output_sharded);
    set_flag(writer_defines, "USE_CUR_POS_TENSOR", use_cur_pos_tensor);
    set_flag(writer_defines, "SLIDING_WINDOW", sliding_window_size > 0);
    set_flag(writer_defines, "HAS_INTERMED_OUT", intermed_output_tiles > 0);
    // Compute
    compute_defines.insert({"EXP_APPROX_MODE", std::to_string(exp_approx_mode)});
    compute_defines.insert({"DHT_GRANULARITY", std::to_string(dht_granularity)});
    compute_defines.insert({"LOG2_DHT_GRANULARITY", std::to_string(log2_dht_granularity)});
    if (Sk_chunk_t > 0) {
        auto add_granularity = [&](const char* name, uint32_t value) {
            uint32_t log2_val = static_cast<uint32_t>(std::log2(value));
            TT_FATAL(value == (1u << log2_val), "{} ({}) must be power of 2", name, value);
            compute_defines.insert({name, std::to_string(value)});
            compute_defines.insert({std::string("LOG2_") + name, std::to_string(log2_val)});
        };
        add_granularity("SUB_EXP_GRANULARITY", std::min(Sk_chunk_t, dst_size));
        add_granularity("MUL_BCAST_GRANULARITY", std::min(PNHt * Sk_chunk_t, dst_size));
        add_granularity("STATS_GRANULARITY", std::min(Sk_chunk_t, dst_size));
    } else {
        compute_defines.insert({"DYNAMIC_CHUNK_SIZE", "1"});
    }
    set_flag(compute_defines, "IS_CAUSAL", is_causal);
    set_flag(compute_defines, "USE_ATTENTION_MASK", use_attention_mask);
    set_flag(compute_defines, "USE_ATTENTION_SINK", use_attention_sink);
    set_flag(compute_defines, "TILIZE_Q", tilize_q);
    set_flag(compute_defines, "SLIDING_WINDOW", sliding_window_size > 0);
    set_flag(compute_defines, "HAS_BLOCK_PADDING", has_block_padding);
    set_flag(compute_defines, "USE_CUR_POS_TENSOR", use_cur_pos_tensor);

    // HAS_BLOCK_PADDING is derived kernel-side from IS_PAGED_ATTENTION + original_block_size on reader/writer;
    // emit it directly on the writer too (it gates c_14 there).
    set_flag(writer_defines, "HAS_BLOCK_PADDING", has_block_padding);

    // ---- Compute hardware config (Style A: resolve the op's ComputeKernelConfig, translate) ----
    ttnn::ComputeKernelConfig resolved_config{
        .math_fidelity = math_fidelity,
        .math_approx_mode = math_approx_mode,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .packer_l1_acc = packer_l1_acc,
        .dst_full_sync_en = dst_full_sync_en,
    };
    auto compute_hw = ttnn::to_compute_hardware_config(device->arch(), resolved_config);
    // unpack_modes: Metal 2.0 requires an explicit entry for each Float32-format DFB a compute kernel
    // consumes when enable_32_bit_dest is set. Legacy set no unpack_to_dest_mode (all Default = UnpackToSrc),
    // so every required entry is UnpackToSrc. The trigger is the DFB's format, not the tensor dtype.
    if (fp32_dest_acc_en) {
        auto& modes = unpack_modes(compute_hw);
        auto maybe_unpack = [&](const DFBSpecName& name, tt::DataFormat df, bool bound) {
            if (bound && df == tt::DataFormat::Float32) {
                modes.insert({name, tt::tt_metal::UnpackMode::UnpackToSrc});
            }
        };
        maybe_unpack(DFB_Q_IN, q_df, true);
        maybe_unpack(DFB_K_IN, k_df, true);
        maybe_unpack(DFB_V_IN, v_df, true);
        maybe_unpack(DFB_MASK_IN, mask_df, true);
        maybe_unpack(DFB_IDENTITY_SCALE, scalar_df, true);
        maybe_unpack(DFB_ZERO_IN, scalar_df, true);
        maybe_unpack(DFB_Q_RM, q_df, tilize_q);
        maybe_unpack(DFB_SLIDING_MASK, mask_df, sliding_window_size > 0);
        maybe_unpack(DFB_BLOCK_PAD_MASK, mask_df, has_block_padding);
    }

    // ---- Kernel specs ----
    const std::string kernel_path = "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/";
    KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path(kernel_path + "dataflow/reader_decode_all.cpp"),
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings = std::move(reader_dfb),
        .semaphore_bindings = std::move(reader_sems),
        .tensor_bindings = std::move(reader_tensors),
        .compile_time_args = std::move(reader_cta),
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"page_table_page_size",
                  "do_reduce",
                  "do_output",
                  "cur_head_group",
                  "cur_batch",
                  "core_num_in_reduce",
                  "core_num_in_output",
                  "cur_pos_arg",
                  "do_k_mcast",
                  "mcast_x",
                  "mcast_y0",
                  "mcast_y1",
                  "num_dests"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        .advanced_options = {.num_runtime_varargs = 2 * num_output_cores},
    };
    KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path(kernel_path + "dataflow/writer_decode_all.cpp"),
        .compiler_options = {.defines = std::move(writer_defines)},
        .dfb_bindings = std::move(writer_dfb),
        .semaphore_bindings = std::move(writer_sems),
        .tensor_bindings = std::move(writer_tensors),
        .compile_time_args = std::move(writer_cta),
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"worker_id_for_reduce",
                  "worker_id_for_output",
                  "do_reduce",
                  "do_output",
                  "cur_head_group",
                  "cur_batch",
                  "core_num_in_reduce",
                  "core_num_in_output",
                  "cur_pos_arg",
                  "is_tree_root",
                  "parent_core_in_group",
                  "send_at_round",
                  "num_children",
                  "my_active_rounds",
                  "reduction_group_base_idx",
                  "children_per_round_0",
                  "children_per_round_1",
                  "children_per_round_2",
                  "children_per_round_3",
                  "children_per_round_4",
                  "children_per_round_5"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
        .advanced_options =
            {.num_runtime_varargs = 2 * num_cores_per_head + 2 * num_reducer_cores + 2 * num_output_cores},
    };
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::filesystem::path(kernel_path + "compute/sdpa_flash_decode.cpp"),
        .compiler_options = {.defines = std::move(compute_defines), .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings = std::move(compute_dfb),
        .compile_time_args = std::move(compute_cta),
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"do_reduce",
                  "do_output",
                  "cur_head",
                  "cur_batch",
                  "core_num_in_reduce",
                  "core_num_in_output",
                  "cur_pos_arg",
                  "is_tree_root",
                  "parent_core_in_group",
                  "send_at_round",
                  "num_children",
                  "my_active_rounds",
                  "children_per_round_0",
                  "children_per_round_1",
                  "children_per_round_2",
                  "children_per_round_3",
                  "children_per_round_4",
                  "children_per_round_5"}},
        .hw_config = std::move(compute_hw),
    };

    // ---- Per-node runtime arg values (name-first) + varargs ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_ra{.kernel = READER};
    KernelRunArgs writer_ra{.kernel = WRITER};
    KernelRunArgs compute_ra{.kernel = COMPUTE};

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
        uint32_t cur_pos = (use_cur_pos_tensor || !is_causal)
                               ? UINT32_MAX
                               : cur_pos_ids.at(static_cast<uint32_t>(cur_batch / q_heads_parallel_factor));

        // Compute tree reduction parameters for this core
        TreeReductionParams tree_params = get_tree_reduction_params(core_num_in_reduce, num_cores_per_head);

        uint32_t reduction_group_base_idx = 0;
        if (use_col_major_group_indexing) {
            reduction_group_base_idx = (i / num_cores_per_head) * num_cores_per_head;
        } else {
            reduction_group_base_idx = (cur_batch * num_cores_per_batch) + (cur_head * num_cores_per_head);
        }

        // Reader named RTAs
        AddRuntimeArgsForNode(
            reader_ra.runtime_arg_values,
            core,
            {{"page_table_page_size", page_table_stick_size},
             {"do_reduce", static_cast<uint32_t>(do_reduce)},
             {"do_output", static_cast<uint32_t>(do_output)},
             {"cur_head_group", cur_head},
             {"cur_batch", cur_batch},
             {"core_num_in_reduce", core_num_in_reduce},
             {"core_num_in_output", core_num_in_output},
             {"cur_pos_arg", cur_pos},
             {"do_k_mcast", static_cast<uint32_t>(do_k_mcast)},
             {"mcast_x", mcast_x},
             {"mcast_y0", mcast_y0},
             {"mcast_y1", mcast_y1},
             {"num_dests", num_dests}});
        // Reader varargs: all_output_noc_x[num_output_cores] ++ all_output_noc_y[num_output_cores]
        {
            std::vector<uint32_t> va;
            va.reserve(2 * num_output_cores);
            va.insert(va.end(), output_core_physical_xs.begin(), output_core_physical_xs.end());
            va.insert(va.end(), output_core_physical_ys.begin(), output_core_physical_ys.end());
            reader_ra.advanced_options.runtime_varargs[core] = std::move(va);
        }

        // Writer named RTAs
        AddRuntimeArgsForNode(
            writer_ra.runtime_arg_values,
            core,
            {{"worker_id_for_reduce", worker_id_for_reduce},
             {"worker_id_for_output", worker_id_for_output},
             {"do_reduce", static_cast<uint32_t>(do_reduce)},
             {"do_output", static_cast<uint32_t>(do_output)},
             {"cur_head_group", cur_head},
             {"cur_batch", cur_batch},
             {"core_num_in_reduce", core_num_in_reduce},
             {"core_num_in_output", core_num_in_output},
             {"cur_pos_arg", cur_pos},
             {"is_tree_root", tree_params.is_root ? 1u : 0u},
             {"parent_core_in_group", tree_params.parent_core_in_group},
             {"send_at_round", tree_params.send_at_round},
             {"num_children", tree_params.num_children},
             {"my_active_rounds", tree_params.my_active_rounds},
             {"reduction_group_base_idx", reduction_group_base_idx},
             {"children_per_round_0", tree_params.children_per_round[0]},
             {"children_per_round_1", tree_params.children_per_round[1]},
             {"children_per_round_2", tree_params.children_per_round[2]},
             {"children_per_round_3", tree_params.children_per_round[3]},
             {"children_per_round_4", tree_params.children_per_round[4]},
             {"children_per_round_5", tree_params.children_per_round[5]}});
        // Writer varargs: reduction_group_core_xs/ys (num_cores_per_head each) ++ all_reducer x/y ++ all_output x/y
        {
            std::vector<uint32_t> va;
            va.reserve(2 * num_cores_per_head + 2 * num_reducer_cores + 2 * num_output_cores);
            for (uint32_t c = 0; c < num_cores_per_head; ++c) {
                va.push_back(reduction_group_core_xs[reduction_group_base_idx + c]);
            }
            for (uint32_t c = 0; c < num_cores_per_head; ++c) {
                va.push_back(reduction_group_core_ys[reduction_group_base_idx + c]);
            }
            va.insert(va.end(), reduce_core_physical_xs.begin(), reduce_core_physical_xs.end());
            va.insert(va.end(), reduce_core_physical_ys.begin(), reduce_core_physical_ys.end());
            va.insert(va.end(), output_core_physical_xs.begin(), output_core_physical_xs.end());
            va.insert(va.end(), output_core_physical_ys.begin(), output_core_physical_ys.end());
            writer_ra.advanced_options.runtime_varargs[core] = std::move(va);
        }

        // Compute named RTAs
        AddRuntimeArgsForNode(
            compute_ra.runtime_arg_values,
            core,
            {{"do_reduce", static_cast<uint32_t>(do_reduce)},
             {"do_output", static_cast<uint32_t>(do_output)},
             {"cur_head", cur_head},
             {"cur_batch", cur_batch},
             {"core_num_in_reduce", core_num_in_reduce},
             {"core_num_in_output", core_num_in_output},
             {"cur_pos_arg", cur_pos},
             {"is_tree_root", tree_params.is_root ? 1u : 0u},
             {"parent_core_in_group", tree_params.parent_core_in_group},
             {"send_at_round", tree_params.send_at_round},
             {"num_children", tree_params.num_children},
             {"my_active_rounds", tree_params.my_active_rounds},
             {"children_per_round_0", tree_params.children_per_round[0]},
             {"children_per_round_1", tree_params.children_per_round[1]},
             {"children_per_round_2", tree_params.children_per_round[2]},
             {"children_per_round_3", tree_params.children_per_round[3]},
             {"children_per_round_4", tree_params.children_per_round[4]},
             {"children_per_round_5", tree_params.children_per_round[5]}});
    }

    // Idle cores: placed on the full grid (matching legacy) but do no work. Every named RTA gets a
    // value so the schema is satisfied on these nodes, do_reduce==65 marks them idle so each kernel
    // early-returns before touching bindings/semaphores, and the varargs the idle kernels never read
    // are zero-filled to the active layout.
    for (const CoreCoord& core : core_group_idle) {
        AddRuntimeArgsForNode(
            reader_ra.runtime_arg_values,
            core,
            {{"page_table_page_size", 0u},
             {"do_reduce", 65u},  // Idle marker
             {"do_output", 0u},
             {"cur_head_group", 0u},
             {"cur_batch", 0u},
             {"core_num_in_reduce", 0u},
             {"core_num_in_output", 0u},
             {"cur_pos_arg", 0u},
             {"do_k_mcast", 0u},
             {"mcast_x", 0u},
             {"mcast_y0", 0u},
             {"mcast_y1", 0u},
             {"num_dests", 0u}});
        reader_ra.advanced_options.runtime_varargs[core] = std::vector<uint32_t>(2 * num_output_cores, 0);

        AddRuntimeArgsForNode(
            writer_ra.runtime_arg_values,
            core,
            {{"worker_id_for_reduce", 0u},
             {"worker_id_for_output", 0u},
             {"do_reduce", 65u},  // Idle marker
             {"do_output", 0u},
             {"cur_head_group", 0u},
             {"cur_batch", 0u},
             {"core_num_in_reduce", 0u},
             {"core_num_in_output", 0u},
             {"cur_pos_arg", 0u},
             {"is_tree_root", 0u},
             {"parent_core_in_group", 0u},
             {"send_at_round", 0u},
             {"num_children", 0u},
             {"my_active_rounds", 0u},
             {"reduction_group_base_idx", 0u},
             {"children_per_round_0", 0u},
             {"children_per_round_1", 0u},
             {"children_per_round_2", 0u},
             {"children_per_round_3", 0u},
             {"children_per_round_4", 0u},
             {"children_per_round_5", 0u}});
        writer_ra.advanced_options.runtime_varargs[core] =
            std::vector<uint32_t>(2 * num_cores_per_head + 2 * num_reducer_cores + 2 * num_output_cores, 0);

        AddRuntimeArgsForNode(
            compute_ra.runtime_arg_values,
            core,
            {{"do_reduce", 65u},  // Idle marker
             {"do_output", 0u},
             {"cur_head", 0u},
             {"cur_batch", 0u},
             {"core_num_in_reduce", 0u},
             {"core_num_in_output", 0u},
             {"cur_pos_arg", 0u},
             {"is_tree_root", 0u},
             {"parent_core_in_group", 0u},
             {"send_at_round", 0u},
             {"num_children", 0u},
             {"my_active_rounds", 0u},
             {"children_per_round_0", 0u},
             {"children_per_round_1", 0u},
             {"children_per_round_2", 0u},
             {"children_per_round_3", 0u},
             {"children_per_round_4", 0u},
             {"children_per_round_5", 0u}});
    }
    run_args.kernel_run_args = {std::move(reader_ra), std::move(writer_ra), std::move(compute_ra)};

    // ---- Tensor args (one per declared TensorParameter) ----
    run_args.tensor_args.insert({Q, TensorArgument{input_tensor_q.mesh_tensor()}});
    run_args.tensor_args.insert({K, TensorArgument{input_tensor_k.mesh_tensor()}});
    if (!reuse_k) {
        // Bind the ORIGINAL V (not the value_or copy) so the framework matches it by MeshTensor identity.
        run_args.tensor_args.insert({VP, TensorArgument{tensor_args.v.value().mesh_tensor()}});
    }
    if (use_cur_pos_tensor) {
        run_args.tensor_args.insert({CUR_POS, TensorArgument{cur_pos_tensor.value().mesh_tensor()}});
    }
    if (is_paged_attention) {
        run_args.tensor_args.insert({PAGE_TABLE, TensorArgument{page_table_tensor.value().mesh_tensor()}});
    }
    if (use_attention_mask) {
        run_args.tensor_args.insert({ATTN_MASK, TensorArgument{attn_mask.value().mesh_tensor()}});
    }
    if (use_attention_sink) {
        run_args.tensor_args.insert({ATTN_SINK, TensorArgument{attention_sink.value().mesh_tensor()}});
    }
    run_args.tensor_args.insert({OUT_T, TensorArgument{output_tensor.mesh_tensor()}});

    // ---- Assemble ----
    ProgramSpec spec{
        .name = "sdpa_decode",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dfbs),
        .semaphores = std::move(semaphores),
        .tensor_parameters = std::move(tensor_params),
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {READER, WRITER, COMPUTE},
            .target_nodes = full_node_set,
        }},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
