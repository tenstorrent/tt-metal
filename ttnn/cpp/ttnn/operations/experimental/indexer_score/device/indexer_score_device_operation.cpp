// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_device_operation.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_device.hpp>  // q.device()->get_view().shape() for block-cyclic sp derivation

#include "kernels/indexer_score_work_split.hpp"  // banded grid mapping (banded_core_count)
#include "ttnn/operations/ccl/ccl_common.hpp"    // get_linearized_index_from_physical_coord

namespace ttnn::operations::experimental::indexer_score {

namespace {
// Largest linearized index of q's devices along the given mesh axis (0 on a single device). Single source
// for the worst-case window check (max_chunk_start) and the host-side chunk_start deduction, so a future
// change to the coord/linearization semantics can't desync the deduced base from the validated window.
uint32_t max_linearized_rank(const Tensor& q, std::optional<uint32_t> axis) {
    uint32_t max_rank = 0;
    if (q.device_storage().get_coords().size() > 1) {
        for (const auto& coord : q.device_storage().get_coords()) {
            max_rank = std::max(max_rank, ttnn::ccl::get_linearized_index_from_physical_coord(q, coord, axis));
        }
    }
    return max_rank;
}

// Fullest device's chunk_start (= its causal-window end - Sq). Used by the worst-case window check.
//   * block-cyclic: the devices collectively cover the whole global chunk [chunk_start, +sp*chunk_local),
//     so the fullest window reaches chunk_start + sp*chunk_local no matter how SP×TP slices it. (For the
//     SP-only case Sq == chunk_local, so this equals the old chunk_start + (sp-1)*chunk_local.)
//   * contiguous (incl. a size-1 SP axis, stored as no block-cyclic): each device's row 0 sits at
//     chunk_start + rank*Sq, where rank is the SP rank PLUS the TP sub-shard rank. The two are
//     mutually-exclusive-nonzero here (a TP sub-shard needs block_cyclic_sp_axis with sp==1, which forces
//     SP rank 0; no sub-shard -> TP rank 0), mirroring device_causal_geometry's no-block-cyclic branch.
uint32_t max_chunk_start(const operation_attributes_t& attrs, const Tensor& q, uint32_t Sq) {
    if (attrs.block_cyclic.has_value()) {
        return attrs.chunk_start_idx + attrs.block_cyclic->sp * attrs.block_cyclic->chunk_local - Sq;
    }
    const uint32_t tp_rank = attrs.tp_axis().has_value() ? max_linearized_rank(q, attrs.tp_axis()) : 0u;
    return attrs.chunk_start_idx + (max_linearized_rank(q, attrs.sp_axis()) + tp_rank) * Sq;
}

// Miss-only checks: hash-pinned (placement, non-indexed k batch shape) so they can't differ on a hit. The
// slot/kv_len values that do differ on a hit live in validate_runtime_values.
void validate_static(const operation_attributes_t& attrs, const tensor_args_t& t) {
    const auto& q = t.q;
    const auto& k = t.k;
    const auto& w = t.weights;
    // On device + allocated. q/weights interleaved; k may be interleaved OR ND-sharded DRAM (a sharded k
    // pins its bank mapping via the hashed spec).
    for (const auto& [tp, name, allow_sharded] :
         {std::tuple{&q, "q", false}, std::tuple{&k, "k", true}, std::tuple{&w, "weights", false}}) {
        TT_FATAL(tp->storage_type() == StorageType::DEVICE, "indexer_score {} must be on device", name);
        TT_FATAL(tp->buffer() != nullptr, "indexer_score {} must have an allocated buffer", name);
        if (!allow_sharded) {
            TT_FATAL(!tp->is_sharded(), "indexer_score {} must use interleaved memory", name);
        }
    }
    TT_FATAL(
        q.device() == k.device() && q.device() == w.device(), "indexer_score q, k, weights must be on the same device");

    // Non-indexed k must be single-slot [1,1,T,D] (the indexed slot < B check is runtime -> below).
    if (!attrs.cache_batch_idx.has_value()) {
        const uint32_t kB = k.logical_shape()[0];
        TT_FATAL(kB == 1, "indexer_score k batch must be 1 unless cache_batch_idx is set (got {})", kB);
    }
}

// The slot/kv_len values: hashed only by has_value(), so they can differ on a hit. Run on miss AND hit.
void validate_runtime_values(const operation_attributes_t& attrs, const tensor_args_t& t) {
    const auto& k = t.k;

    // Indexed KV cache: cache_batch_idx picks a slot of [B,1,T,D] k; an out-of-range slot offsets pages OOB.
    if (attrs.cache_batch_idx.has_value()) {
        const uint32_t kB = k.logical_shape()[0];
        TT_FATAL(
            attrs.cache_batch_idx.value() < kB,
            "indexer_score cache_batch_idx ({}) must be < k batch slots ({})",
            attrs.cache_batch_idx.value(),
            kB);
    }

    // Runtime KV length: kv_len <= T is the valid prefix this dispatch (not hashed -> re-checked here). The
    // chunk-window-vs-kv_len bound lives in validate_chunk_start.
    if (attrs.kv_len.has_value()) {
        const uint32_t T = k.logical_shape()[2];
        const uint32_t kv_len = attrs.kv_len.value();
        TT_FATAL(kv_len % tt::constants::TILE_WIDTH == 0, "indexer_score kv_len {} must be tile-aligned", kv_len);
        TT_FATAL(
            kv_len > 0 && kv_len <= T,
            "indexer_score kv_len {} must be in (0, T={}] (the allocated k length)",
            kv_len,
            T);
        // Block-max-pool writes whole blocks: the writer emits valid_blocks = valid_tiles / block_tiles (floor),
        // so a kv_len that lands mid-block would drop the partially-valid boundary block (compute pools it
        // correctly, but the writer never stores it, leaving a stale output column). block_size is compile-time
        // and kv_len is a runtime value re-checked on hit, so the cross-check lives here (runs miss AND hit).
        if (attrs.block_size > 0) {
            TT_FATAL(
                kv_len % attrs.block_size == 0,
                "indexer_score kv_len {} must be a multiple of block_size {} when block-max-pooling (a kv_len "
                "splitting a block drops the boundary block's score)",
                kv_len,
                attrs.block_size);
        }
    }
}

// Runs on miss AND hit (chunk_start is hash-excluded). All chunk_start checks in one place: base alignment
// and the fullest device's window against T and (when set) kv_len.
void validate_chunk_start(const operation_attributes_t& attrs, const tensor_args_t& t) {
    const uint32_t Sq = t.q.logical_shape()[2];
    const uint32_t T = t.k.logical_shape()[2];
    TT_FATAL(
        attrs.chunk_start_idx % tt::constants::TILE_WIDTH == 0,
        "chunk_start_idx {} must be tile-aligned",
        attrs.chunk_start_idx);
    const uint32_t max_cs = max_chunk_start(attrs, t.q, Sq);
    TT_FATAL(
        max_cs + Sq <= T,
        "fullest-device chunk window [{}, {}+{}) exceeds T={} (base={}, per-rank stride Sq={})",
        max_cs,
        max_cs,
        Sq,
        T,
        attrs.chunk_start_idx,
        Sq);
    // Causal window must also stay inside the valid prefix.
    if (attrs.kv_len.has_value()) {
        const uint32_t kv_len = attrs.kv_len.value();
        TT_FATAL(
            max_cs + Sq <= kv_len,
            "fullest-device causal window [{}, {}+{}) exceeds kv_len={} (cannot attend past the valid keys; "
            "base={}, per-rank stride Sq={})",
            max_cs,
            max_cs,
            Sq,
            kv_len,
            attrs.chunk_start_idx,
            Sq);
    }
}
// Block-cyclic layout (sp derived from block_cyclic_sp_axis, global chunk = sp*block_cyclic_chunk_local):
// sp/chunk_local are hashed and bake invP's divisors into the reader, so the per-shard chunk must be
// tile-aligned and tile the cache evenly -- miss-only (a hit can't differ). Sq <= chunk_local bounds a
// device's queries to at most one cache-slab crossing (one straddle).
void validate_block_cyclic(const operation_attributes_t& attrs, const tensor_args_t& t) {
    if (!attrs.block_cyclic.has_value()) {
        return;
    }
    const uint32_t sp = attrs.block_cyclic->sp;
    const uint32_t chunk_local = attrs.block_cyclic->chunk_local;
    const uint32_t chunk_global = sp * chunk_local;
    const uint32_t T = t.k.logical_shape()[2];
    const uint32_t Sq = t.q.logical_shape()[2];
    TT_FATAL(sp >= 1, "block-cyclic sp must be >= 1 (got {})", sp);
    TT_FATAL(
        chunk_local > 0 && chunk_local % tt::constants::TILE_WIDTH == 0,
        "block_cyclic_chunk_local ({}) must be > 0 and tile-aligned",
        chunk_local);
    TT_FATAL(
        T % chunk_global == 0,
        "global chunk {} (= sp*chunk_local) must divide T {} (the cache must be a whole number of global chunks)",
        chunk_global,
        T);
    TT_FATAL(
        Sq <= chunk_local,
        "Sq ({}) must be <= block_cyclic_chunk_local ({}); a device's queries may cross at most one cache-slab "
        "boundary (one straddle). A coarser indexer SP (Sq > chunk_local) is not yet supported.",
        Sq,
        chunk_local);
}
}  // namespace

IndexerScoreDeviceOperation::program_factory_t IndexerScoreDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::IndexerScoreProgramFactory{};
}

void IndexerScoreDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // chunk_start, cache slot, kv_len are hash-excluded runtime values -> re-checked on hits.
    validate_runtime_values(attrs, tensor_args);
    validate_chunk_start(attrs, tensor_args);
}

void IndexerScoreDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& q = tensor_args.q;
    const auto& k = tensor_args.k;
    const auto& w = tensor_args.weights;

    const auto& q_shape = q.logical_shape();
    const auto& k_shape = k.logical_shape();
    const auto& w_shape = w.logical_shape();

    // Blackhole-only: the compute kernel relies on BH fast-untilize + custom BH LLK paths. Enforce in C++
    // so a Wormhole caller fails cleanly instead of hanging at launch.
    const tt::ARCH arch = tt::tt_metal::hal::get_arch();
    TT_FATAL(arch == tt::ARCH::BLACKHOLE, "indexer_score is only supported on Blackhole, got {}", arch);

    // The custom blocked bcast-col LLK + half-sync 8-head subblock are validated only for bf16 DEST in
    // half-sync; reject the unvalidated knobs loudly (a silent path could drop the -inf mask).
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        ttnn::get_compute_kernel_config_args(arch, attrs.compute_kernel_config);
    TT_FATAL(
        !fp32_dest_acc_en,
        "indexer_score requires fp32_dest_acc_en=false (bf16 DEST; the custom LLK is not validated for fp32 DEST)");
    TT_FATAL(!dst_full_sync_en, "indexer_score requires dst_full_sync_en=false (the kernel is built half-sync)");

    // Placement/layout/same-device and the non-indexed k batch shape are hash-pinned (miss only); the
    // slot/kv_len runtime values are re-checked every dispatch.
    validate_static(attrs, tensor_args);
    validate_runtime_values(attrs, tensor_args);
    validate_block_cyclic(attrs, tensor_args);

    // Shapes: q [B, Hi, Sq, D], k [B, 1, T, D] (single shared head), weights [B, Hi, Sq, 1].
    TT_FATAL(q_shape.rank() == 4 && k_shape.rank() == 4, "q, k must be rank 4");
    TT_FATAL(k_shape[1] == 1, "k must be single-head [B, 1, T, D], got {} heads", k_shape[1]);
    TT_FATAL(q_shape[3] == k_shape[3], "q head dim {} != k head dim {}", q_shape[3], k_shape[3]);
    TT_FATAL(q_shape[0] == 1, "q batch 1 only, got {}", q_shape[0]);
    // The learned-gate weights are validated only when present; MSA synthesizes a constant gate in-kernel
    // (the weights handle is an unused q placeholder), so its shape/dtype carry no meaning.
    if (!attrs.synthesize_gate) {
        TT_FATAL(w_shape.rank() == 4, "weights must be rank 4");
        TT_FATAL(
            w_shape[1] == q_shape[1] && w_shape[2] == q_shape[2] && w_shape[3] == 1,
            "weights must be [B, Hi, Sq, 1] matching q [B, Hi, Sq, D]");
        TT_FATAL(q_shape[0] == w_shape[0], "q/weights batch mismatch ({} vs {})", q_shape[0], w_shape[0]);
        TT_FATAL(w.dtype() == DataType::BFLOAT16, "weights must be bfloat16 (got {})", w.dtype());
        TT_FATAL(w.layout() == Layout::TILE, "weights must be TILE layout");
    }

    // q/k are matmul inputs (never packed), so each may be bfp8_b (halves BW); both bfp8 -> LoFi, any bf16
    // -> HiFi2.
    TT_FATAL(
        q.dtype() == DataType::BFLOAT16 || q.dtype() == DataType::BFLOAT8_B,
        "q must be bfloat16 or bfloat8_b (got {})",
        q.dtype());
    TT_FATAL(
        k.dtype() == DataType::BFLOAT16 || k.dtype() == DataType::BFLOAT8_B,
        "k must be bfloat16 or bfloat8_b (got {})",
        k.dtype());
    TT_FATAL(q.layout() == Layout::TILE && k.layout() == Layout::TILE, "q, k must be TILE layout");

    // Tile alignment and the causal chunk window.
    const uint32_t Hi = q_shape[1];
    const uint32_t Sq = q_shape[2];
    const uint32_t D = q_shape[3];
    const uint32_t T = k_shape[2];
    TT_FATAL(
        Sq % tt::constants::TILE_HEIGHT == 0 && T % tt::constants::TILE_WIDTH == 0 &&
            D % tt::constants::TILE_WIDTH == 0,
        "Sq {}, T {}, D {} must be tile-aligned",
        Sq,
        T,
        D);
    validate_chunk_start(attrs, tensor_args);  // base/stride alignment + worst-device window

    // Work-unit knobs (elements, tile-aligned); see IndexerScoreProgramConfig.
    const auto& cfg = attrs.program_config;
    TT_FATAL(
        cfg.q_chunk_size % tt::constants::TILE_HEIGHT == 0 && cfg.k_chunk_size % tt::constants::TILE_WIDTH == 0,
        "q_chunk_size {} / k_chunk_size {} must be tile-aligned",
        cfg.q_chunk_size,
        cfg.k_chunk_size);
    const uint32_t QC = cfg.q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t KC = cfg.k_chunk_size / tt::constants::TILE_WIDTH;
    const uint32_t HB = resolve_head_group(cfg, Hi);
    TT_FATAL(
        QC > 0 && (Sq / tt::constants::TILE_HEIGHT) % QC == 0,
        "q_chunk_size {} must divide Sq {}",
        cfg.q_chunk_size,
        Sq);
    TT_FATAL(KC > 0 && KC <= T / tt::constants::TILE_WIDTH, "k_chunk_size {} out of range (T={})", cfg.k_chunk_size, T);
    // KC need not divide Tt: the last unit is then partial (compute runs a full KC strip, writer clips).
    TT_FATAL(HB > 0 && Hi % HB == 0, "head_group_size {} must divide Hi {}", HB, Hi);

    // num_groups: 1 sums all heads (DSA/GLM); G>1 emits G per-group planes (M3). G>1 reuses the all-resident
    // full-strip path one group at a time (the streaming/per-column fallback is not wired for groups).
    const uint32_t G = attrs.num_groups;
    TT_FATAL(G >= 1 && Hi % G == 0, "num_groups {} must be >= 1 and divide Hi {}", G, Hi);
    if (G > 1) {
        TT_FATAL(HB == Hi, "num_groups {}>1 requires all heads resident (head_group_size 0 or Hi); got HB={}", G, HB);
        TT_FATAL(
            KC >= 2,
            "num_groups {}>1 requires k_chunk_size>=64 (the full-strip path); got k_chunk_size={}",
            G,
            cfg.k_chunk_size);
    }

    // block_size: 0 = no pooling; >0 = block-max-pool -> [B,G,Sq,T/block_size] (M3). Guards fire only when
    // pooling. A block must align to T and the k-chunk boundary, and blocks-per-unit must fit TILE_HEIGHT.
    const uint32_t bs = attrs.block_size;
    if (bs > 0) {
        TT_FATAL(
            bs % tt::constants::TILE_WIDTH == 0,
            "block_size {} must be a multiple of {}",
            bs,
            tt::constants::TILE_WIDTH);
        TT_FATAL(T % bs == 0, "block_size {} must divide T {}", bs, T);
        TT_FATAL(
            cfg.k_chunk_size % bs == 0,
            "block_size {} must divide k_chunk_size {} (a block can't straddle a work unit)",
            bs,
            cfg.k_chunk_size);
        // No partial last unit: every unit emits exactly blocks_per_unit aligned block-scores.
        const uint32_t Tt = T / tt::constants::TILE_WIDTH;
        TT_FATAL(Tt % KC == 0, "block_size pooling requires k_chunk_size {} to divide T {}", cfg.k_chunk_size, T);
        const uint32_t block_tiles = bs / tt::constants::TILE_WIDTH;
        const uint32_t blocks_per_unit = KC / block_tiles;
        // The writer writes each unit's bf16 output slice at offset unit*blocks_per_unit*2; Blackhole's 16 B
        // write/source alignment requires blocks_per_unit a multiple of 8. Net: k_chunk_size/block_size in
        // {8,16,24,32}.
        constexpr uint32_t kBf16AlignBlocks = 8;  // 16 B / 2 B
        TT_FATAL(
            blocks_per_unit % kBf16AlignBlocks == 0,
            "block_size pooling needs k_chunk_size/block_size (= {}) to be a multiple of {} so each unit's "
            "row-major output slice is 16 B-aligned; set k_chunk_size to a multiple of {}*block_size",
            blocks_per_unit,
            kBf16AlignBlocks,
            kBf16AlignBlocks);
        TT_FATAL(
            blocks_per_unit <= tt::constants::TILE_HEIGHT,
            "blocks per work unit (k_chunk_size/block_size = {}) must be <= {}; reduce k_chunk_size",
            blocks_per_unit,
            tt::constants::TILE_HEIGHT);
    }
}

IndexerScoreDeviceOperation::spec_return_value_t IndexerScoreDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& q_shape = tensor_args.q.logical_shape();
    const auto& k_shape = tensor_args.k.logical_shape();
    // score [B, num_groups, Sq, T_out], row-major bf16. num_groups==1 = head-summed (DSA/GLM); >1 = one
    // plane per GQA group (M3). T_out = T, or T/block_size when block-max-pooling.
    const uint32_t T_out = attrs.block_size ? k_shape[2] / attrs.block_size : k_shape[2];
    ttnn::Shape out_shape({q_shape[0], attrs.num_groups, q_shape[2], T_out});
    return TensorSpec(
        out_shape,
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), tensor_args.q.memory_config()));
}

IndexerScoreDeviceOperation::tensor_return_value_t IndexerScoreDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.q.device());
}

// Matmul-FLOP performance model: report ideal matmul cycles = num_mul_adds / (cores * peak), so the
// profiler's ideal/actual ratio equals the math-util test's mm_flops / (cores * device_cycles * peak).
// Every term matches the test: causal-valid output tiles only, actual cores, fidelity-scaled BH peak.
tt::tt_metal::operation::OpPerformanceModelGeneral<IndexerScoreDeviceOperation::tensor_return_value_t>
IndexerScoreDeviceOperation::create_op_performance_model(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& q = tensor_args.q;
    const auto& k = tensor_args.k;
    const tt::tt_metal::operation::Tensors input_tensors = {q, k, tensor_args.weights};

    // Matmul throughput model is Blackhole-specific (the only validated arch).
    const tt::ARCH arch = q.device()->arch();
    if (arch != tt::ARCH::BLACKHOLE) {
        return tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t>(input_tensors, output, 0);
    }

    const auto& q_shape = q.logical_shape();
    const auto& k_shape = k.logical_shape();
    const uint32_t B = q_shape[0];
    const uint32_t Hi = q_shape[1];
    const uint32_t D = q_shape[3];
    const uint32_t Sqt = q_shape[2] / tt::constants::TILE_HEIGHT;
    const uint32_t Tt = k_shape[2] / tt::constants::TILE_WIDTH;
    const uint32_t chunk_t = attrs.chunk_start_idx / tt::constants::TILE_WIDTH;

    // Causal-valid output tiles V = sum_rows min(kv_len_tiles, chunk_t + row + 1) (masked future excluded;
    // matches the test's sp7_valid_tiles()). kv_len caps per-row valid columns; nullopt == full Tt.
    const uint32_t kv_len_tiles = attrs.kv_len.has_value() ? attrs.kv_len.value() / tt::constants::TILE_WIDTH : Tt;
    uint64_t valid_tiles = 0;
    for (uint32_t s = 0; s < Sqt; ++s) {
        valid_tiles += std::min<uint64_t>(kv_len_tiles, (uint64_t)chunk_t + s + 1);
    }

    // Per valid 32x32 tile per head: (32*32) outputs x 2*D FLOPs; summed over heads/tiles/batch
    // (matches the test's indexer_mm_flops()).
    const uint64_t num_mul_adds =
        2ull * valid_tiles * Hi * B * static_cast<uint64_t>(tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH) * D;

    // Cores used: the banded schedule's (group_rows x row_blocks) x cols rectangle, via the same helper the
    // factory uses (block-split included), so this equals tracy's CORE COUNT.
    const uint32_t QC = attrs.program_config.q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t KC = attrs.program_config.k_chunk_size / tt::constants::TILE_WIDTH;
    const uint32_t group_count = Sqt / QC;
    const uint32_t band_count = units_in_group(KC, Tt);  // ceil(Tt/KC); shared with the factory
    const auto grid = q.device()->compute_with_storage_grid_size();
    const uint64_t num_cores = banded_core_count(group_count, band_count, grid.x, grid.y);

    // Blackhole matmul peak: 4096 mul-adds/cycle/core at LoFi, scaled by the fidelity multiplier.
    const auto math_fidelity = std::get<0>(ttnn::get_compute_kernel_config_args(arch, attrs.compute_kernel_config));
    constexpr uint64_t tensix_mul_adds_per_cycle_lofi = 4096;
    const int ideal_compute_cycles = static_cast<int>(std::ceil(
        (static_cast<double>(num_mul_adds) / static_cast<double>(num_cores * tensix_mul_adds_per_cycle_lofi)) *
        static_cast<double>(tt::tt_metal::operation::OpPerformanceModel::fidelity_multiplier(math_fidelity))));

    return tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t>(
        input_tensors, output, ideal_compute_cycles);
}

std::tuple<IndexerScoreDeviceOperation::operation_attributes_t, IndexerScoreDeviceOperation::tensor_args_t>
IndexerScoreDeviceOperation::invoke(
    const Tensor& q,
    const Tensor& k,
    const Tensor& weights,
    uint32_t chunk_start_idx,
    bool apply_relu,
    uint32_t num_groups,
    uint32_t block_size,
    bool synthesize_gate,
    float gate_scale,
    const IndexerScoreProgramConfig& program_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<uint32_t> cache_batch_idx,
    std::optional<uint32_t> kv_len,
    std::vector<uint32_t> seq_shard_axes,
    std::optional<BlockCyclicLayout> block_cyclic) {
    return {
        operation_attributes_t{
            .chunk_start_idx = chunk_start_idx,
            .seq_shard_axes = std::move(seq_shard_axes),
            .apply_relu = apply_relu,
            .num_groups = num_groups,
            .block_size = block_size,
            .synthesize_gate = synthesize_gate,
            .gate_scale = gate_scale,
            .program_config = program_config,
            .compute_kernel_config = compute_kernel_config,
            .cache_batch_idx = cache_batch_idx,
            .kv_len = kv_len,
            .block_cyclic = block_cyclic},
        tensor_args_t{.q = q, .k = k, .weights = weights}};
}

}  // namespace ttnn::operations::experimental::indexer_score

namespace ttnn::experimental {

namespace {

// seq_shard_axes names the mesh axes the query seq is sharded over, outermost (SP ring) -> innermost (TP
// sub-shard). Decompose to the (SP, TP) roles the validation + causal geometry use: {} -> (none, none) =
// linear device order; {sp} -> (sp, none); {sp, tp} -> (sp, tp). allow_subshard is false for MSA (no TP
// sub-shard -> at most one axis).
std::pair<std::optional<uint32_t>, std::optional<uint32_t>> split_seq_shard_axes(
    const std::vector<uint32_t>& axes, bool allow_subshard) {
    const uint32_t max_axes = allow_subshard ? 2u : 1u;
    TT_FATAL(
        axes.size() <= max_axes,
        "indexer_score: seq_shard_axes takes at most {} axis/axes [SP{}], got {}",
        max_axes,
        allow_subshard ? ", TP" : "",
        axes.size());
    return {
        axes.empty() ? std::nullopt : std::optional<uint32_t>(axes[0]),
        axes.size() >= 2 ? std::optional<uint32_t>(axes[1]) : std::nullopt};
}

// Shared launch path for both frontends: resolve the compute-kernel config from the matmul-input dtypes,
// then pack and launch the one device op. The flavour knobs are decided by the caller below.
ttnn::Tensor launch_indexer_score(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& weights,
    std::optional<uint32_t> chunk_start_idx,
    bool apply_relu,
    uint32_t num_groups,
    uint32_t block_size,
    bool synthesize_gate,
    float gate_scale,
    const ttnn::operations::experimental::indexer_score::IndexerScoreProgramConfig& program_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<uint32_t> cache_batch_idx,
    std::optional<uint32_t> kv_len,
    std::vector<uint32_t> seq_shard_axes,
    bool allow_subshard,
    std::optional<uint32_t> block_cyclic_sp_axis,
    std::optional<uint32_t> block_cyclic_chunk_local) {
    // Decompose the seq-shard axes into the SP/TP roles the validation + causal geometry below reason about.
    const auto [cluster_axis, seq_subshard_axis] = split_seq_shard_axes(seq_shard_axes, allow_subshard);
    using OperationType = ttnn::operations::experimental::indexer_score::IndexerScoreDeviceOperation;
    using ttnn::operations::experimental::indexer_score::BlockCyclicLayout;

    const uint32_t Sq = q.logical_shape()[2];

    // Block-cyclic (per-SP-shard) K layout -- interface matches ttnn.transformer.sparse_sdpa: the caller
    // names the MESH AXIS the cache was striped over (block_cyclic_sp_axis) and passes the per-shard chunk
    // length (block_cyclic_chunk_local); `sp` is DERIVED from the mesh shape on that axis, so a caller cannot
    // pass an sp that disagrees with the device. Both set together or neither. sp == 1 (single chip / size-1
    // axis) is the identity permutation, represented as no block-cyclic layout.
    TT_FATAL(
        block_cyclic_sp_axis.has_value() == block_cyclic_chunk_local.has_value(),
        "indexer_score: block_cyclic_sp_axis and block_cyclic_chunk_local must both be set or both unset "
        "(got sp_axis={}, chunk_local={})",
        block_cyclic_sp_axis.has_value(),
        block_cyclic_chunk_local.has_value());
    // seq_shard_axes[1] (2D TP sub-shard) only means anything with a block-cyclic layout; reject a stray one.
    TT_FATAL(
        !seq_subshard_axis.has_value() || block_cyclic_sp_axis.has_value(),
        "indexer_score: seq_shard_axes[1] (TP sub-shard) requires a block-cyclic layout "
        "(block_cyclic_sp_axis/chunk_local)");
    std::optional<BlockCyclicLayout> block_cyclic = std::nullopt;
    if (block_cyclic_sp_axis.has_value()) {
        const auto mesh_shape = q.device()->get_view().shape();
        const uint32_t sp_axis = *block_cyclic_sp_axis;
        TT_FATAL(
            sp_axis < mesh_shape.dims(),
            "indexer_score: block_cyclic_sp_axis ({}) out of range for mesh rank {}",
            sp_axis,
            mesh_shape.dims());
        const uint32_t sp = mesh_shape[sp_axis];
        const uint32_t tp = static_cast<uint32_t>(mesh_shape.mesh_size()) / sp;  // remaining (TP) device count
        const uint32_t chunk_local = *block_cyclic_chunk_local;
        // chunk_local is one of exactly two legal values (the cross-check sparse_sdpa also applies): q's
        // per-chip seq length (seq sharded only on the SP axis) or tp*q_isl (seq also sliced across the TP
        // axis, post-reshard). Anything else is a producer bug.
        TT_FATAL(
            chunk_local == Sq || chunk_local == Sq * tp,
            "indexer_score: block_cyclic_chunk_local ({}) must be q_isl ({}) or tp*q_isl ({})",
            chunk_local,
            Sq,
            Sq * tp);
        // Seq sharded across BOTH axes (chunk_local == tp*q_isl, tp > 1) needs the second axis's seq offset.
        // Two ways to supply it:
        //   (a) seq_shard_axes=[] -> the flat row-major device rank (a*B+b) folds BOTH axes in, so the LINEAR
        //       chunk_start is each device's position -- but that linear form is only exact for a slab-aligned
        //       (boundary_chip == 0) start; mid-slab (rotated) starts drift (see device_causal_geometry).
        //   (b) seq_shard_axes=[SP, TP] -> the EXACT block-cyclic geometry (mirroring rotated_chip_positions)
        //       adds the tp_rank*Sq sub-offset. Rotation-exact.
        // A lone SP axis (seq_shard_axes=[SP]) would miss the TP offset entirely -- reject that.
        const bool both_axes = (chunk_local == Sq * tp && tp > 1);
        TT_FATAL(
            !(both_axes && cluster_axis.has_value() && !seq_subshard_axis.has_value()),
            "indexer_score: block_cyclic_chunk_local == tp*q_isl (tp={} > 1) with seq_shard_axes=[SP] needs the "
            "TP axis too (seq_shard_axes=[SP, TP]) so the second axis's seq offset is applied; or pass "
            "seq_shard_axes=[] (flat linearization).",
            tp);
        if (seq_subshard_axis.has_value()) {
            TT_FATAL(
                cluster_axis.has_value() && both_axes,
                "indexer_score: seq_shard_axes TP axis needs the SP axis present (seq_shard_axes=[SP, TP]) and a "
                "2D seq shard (block_cyclic_chunk_local == tp*q_isl); got has_sp_axis={}, chunk_local={}, Sq*tp={}",
                cluster_axis.has_value(),
                chunk_local,
                Sq * tp);
            TT_FATAL(
                *seq_subshard_axis < mesh_shape.dims() && *seq_subshard_axis != *cluster_axis,
                "indexer_score: seq_shard_axes TP axis ({}) must be an in-range mesh axis distinct from the SP "
                "axis ({})",
                *seq_subshard_axis,
                *cluster_axis);
        }
        // Store {sp, chunk_local} (matching sparse_sdpa's BlockCyclicLayout); the factory derives the global
        // chunk (sp*chunk_local) and the invP tile divisors from these.
        if (sp > 1) {
            block_cyclic = BlockCyclicLayout{.sp = sp, .chunk_local = chunk_local};
        }
    }

    // base = the absolute chunk_start of this op's rank 0. Omit it -> deduce the start of the gathered chunk:
    //   * block-cyclic: the gathered chunk IS the global chunk (sp*chunk_local) -> base = T - chunk.
    //   * contiguous: the gathered chunk is seq_ring*Sq. Normally seq_ring is the SP ring; for the identity
    //     block-cyclic SP=1 + TP sub-shard it is the TP ring instead.
    // The deduced window ends at T (incompatible with a growing kv_len < T -- pass chunk_start_idx there).
    uint32_t base = 0;
    if (chunk_start_idx.has_value()) {
        base = *chunk_start_idx;
    } else {
        const uint32_t T = k.logical_shape()[2];
        if (block_cyclic.has_value()) {
            const uint32_t chunk = block_cyclic->sp * block_cyclic->chunk_local;
            TT_FATAL(
                T >= chunk,
                "indexer_score: cannot deduce chunk_start_idx -- T={} < global chunk={}. Pass chunk_start_idx "
                "explicitly if K does not equal history + the gathered chunk.",
                T,
                chunk);
            base = T - chunk;
        } else {
            // seq_ring = max_rank + 1 (get_linearized_index returns coord-min; get_topological_dimension would
            // over-count on a nonzero-offset sub-mesh). A TP sub-shard is possible here only for SP=1, whose
            // block-cyclic permutation is stored as contiguous, so TP is the sequence-bearing axis.
            const auto seq_axis = seq_subshard_axis.has_value() ? seq_subshard_axis : cluster_axis;
            const uint32_t seq_ring =
                ttnn::operations::experimental::indexer_score::max_linearized_rank(q, seq_axis) + 1;
            TT_FATAL(
                T >= seq_ring * Sq,
                "indexer_score: cannot deduce chunk_start_idx -- T={} < seq_ring({})*Sq({}). Pass chunk_start_idx "
                "explicitly if K does not equal history + the gathered query chunk.",
                T,
                seq_ring,
                Sq);
            base = T - seq_ring * Sq;
        }
    }

    // Default math_fidelity follows the matmul-input dtypes (both bfp8 -> LoFi, else HiFi2); a caller config
    // overrides per field. fp32-dest acc / full-sync default off (the only modes the custom LLK is validated for).
    const bool both_bfp8 = q.dtype() == DataType::BFLOAT8_B && k.dtype() == DataType::BFLOAT8_B;
    const auto resolved = ttnn::init_device_compute_kernel_config(
        q.device()->arch(),
        compute_kernel_config,
        /*default_fidelity=*/both_bfp8 ? tt::tt_metal::MathFidelity::LoFi : tt::tt_metal::MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false,
        /*default_dst_full_sync_en=*/false);
    auto [operation_attributes, tensor_args] = OperationType::invoke(
        q,
        k,
        weights,
        base,
        apply_relu,
        num_groups,
        block_size,
        synthesize_gate,
        gate_scale,
        program_config,
        resolved,
        cache_batch_idx,
        kv_len,
        std::move(seq_shard_axes),
        block_cyclic);
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace

ttnn::Tensor indexer_score_dsa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& weights,
    std::optional<uint32_t> chunk_start_idx,
    const ttnn::operations::experimental::indexer_score::IndexerScoreProgramConfig& program_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<uint32_t> cache_batch_idx,
    std::optional<uint32_t> kv_len,
    const std::optional<std::vector<uint32_t>>& seq_shard_axes,
    std::optional<uint32_t> block_cyclic_sp_axis,
    std::optional<uint32_t> block_cyclic_chunk_local) {
    // DSA/GLM: relu, learned per-head gates, one head-summed plane, no pooling. Reads its real weights tensor.
    return launch_indexer_score(
        q,
        k,
        weights,
        chunk_start_idx,
        /*apply_relu=*/true,
        /*num_groups=*/1,
        /*block_size=*/0,
        /*synthesize_gate=*/false,
        /*gate_scale=*/1.0f,
        program_config,
        compute_kernel_config,
        cache_batch_idx,
        kv_len,
        seq_shard_axes.value_or(std::vector<uint32_t>{}),
        /*allow_subshard=*/true,
        block_cyclic_sp_axis,
        block_cyclic_chunk_local);
}

ttnn::Tensor indexer_score_msa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    uint32_t num_groups,
    std::optional<uint32_t> chunk_start_idx,
    float scale,
    uint32_t block_size,
    const ttnn::operations::experimental::indexer_score::IndexerScoreProgramConfig& program_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<uint32_t> cache_batch_idx,
    std::optional<uint32_t> kv_len,
    const std::optional<std::vector<uint32_t>>& seq_shard_axes,
    std::optional<uint32_t> block_cyclic_sp_axis,
    std::optional<uint32_t> block_cyclic_chunk_local) {
    // M3 has no learned gates, only a 1/sqrt(d) scale. Rather than materialize a constant [B,Hi,Sq,1] gate
    // tensor (an extra fill op dispatched every call), the reader fills cb_w with `scale` in L1 in-kernel
    // (synthesize_gate); q is passed as the unused weights placeholder so the op infra still has a valid
    // on-device tensor. MSA fixes apply_relu=false; num_groups/block_size are selection knobs. The
    // persistent-KV-cache knobs (cache_batch_idx/kv_len) are the same runtime, hash-excluded pass-throughs as
    // DSA -- the device op and all 3 kernels are mode-agnostic for them (the fused-streaming K read applies
    // the indexed-slot offset, and pooled kv_len is guarded to a block boundary in validate).
    return launch_indexer_score(
        q,
        k,
        /*weights=*/q,
        chunk_start_idx,
        /*apply_relu=*/false,
        num_groups,
        block_size,
        /*synthesize_gate=*/true,
        /*gate_scale=*/scale,
        program_config,
        compute_kernel_config,
        cache_batch_idx,
        kv_len,
        seq_shard_axes.value_or(std::vector<uint32_t>{}),
        /*allow_subshard=*/false,  // MSA has no TP sub-shard
        block_cyclic_sp_axis,
        block_cyclic_chunk_local);
}

}  // namespace ttnn::experimental
