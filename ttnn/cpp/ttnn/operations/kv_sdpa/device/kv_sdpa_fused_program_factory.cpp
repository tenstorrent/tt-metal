// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kv_sdpa_device_operation.hpp"

#include <algorithm>
#include <numeric>

#include "tt-metalium/constants.hpp"
#include "tt-metalium/work_split.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"

namespace ttnn::operations::kv_sdpa {

using namespace tt;
using namespace tt::tt_metal;

// FlashFused: one core per Q head; the compute kernel calls the production transformer-SDPA
// sdpa_standard() online-softmax routine (matches production speed). MQA: every core's reader reads
// the single KV head, chunk-by-chunk. Interleaved I/O for now; sharded folds come next.
ProgramDescriptor KvSdpaDeviceOperation::FlashFused::create_descriptor(
    const operation_attributes_t& attrs, const tensor_args_t& ta, tensor_return_value_t& out) {
    IDevice* device = ta.q.device();

    const auto qdf = datatype_to_dataformat_converter(ta.q.dtype());
    const auto kdf = datatype_to_dataformat_converter(ta.k.dtype());
    const auto vdf = datatype_to_dataformat_converter(ta.v.dtype());
    const auto odf = datatype_to_dataformat_converter(out.dtype());
    constexpr auto bf16 = tt::DataFormat::Float16_b;  // im/stats/scalar all bf16 (matches prod sdpa)

    const auto qtile = ta.q.tensor_spec().tile();
    const auto ktile = ta.k.tensor_spec().tile();
    const auto vtile = ta.v.tensor_spec().tile();
    const auto otile = out.tensor_spec().tile();
    const uint32_t q_ts = qtile.get_tile_size(qdf);
    const uint32_t k_ts = ktile.get_tile_size(kdf);
    const uint32_t v_ts = vtile.get_tile_size(vdf);
    const uint32_t o_ts = otile.get_tile_size(odf);
    const uint32_t bf16_ts = qtile.get_tile_size(bf16);

    const auto& qs = ta.q.padded_shape();
    const auto& ks = ta.k.padded_shape();
    const uint32_t NQH = qs[1];
    const uint32_t NKH = ks[1];
    const uint32_t DHt = qs[3] / tt::constants::TILE_WIDTH;
    const uint32_t vDHt = DHt;
    const uint32_t group = NQH / NKH;
    // Two-source K/V: an optional resident prefix (past_k/past_v) and the new/suffix K/V (k/v). They
    // may use DIFFERENT tile heights (e.g. a 32x32 bf8 prefix + a 16x32 bf8 suffix); the reader feeds
    // each into its own CB pair at its own geometry and the compute runs one flash loop over both,
    // sharing the online-softmax state. The caller need not pre-concatenate.
    const bool has_past = ta.past_k.has_value();
    // Prefix tile geometry (own dtype/tile; falls back to the suffix geometry when there is no past).
    const auto pktile = has_past ? ta.past_k->tensor_spec().tile() : ktile;
    const auto pvtile = has_past ? ta.past_v->tensor_spec().tile() : vtile;
    const auto pkdf = has_past ? datatype_to_dataformat_converter(ta.past_k->dtype()) : kdf;
    const auto pvdf = has_past ? datatype_to_dataformat_converter(ta.past_v->dtype()) : vdf;
    const uint32_t pk_ts = pktile.get_tile_size(pkdf);
    const uint32_t pv_ts = pvtile.get_tile_size(pvdf);
    // KV tile counts derive from each tensor's actual tile height (tiny tiles may be < 32 tall).
    const uint32_t suffix_Kt = ks[2] / ktile.get_height();
    const uint32_t prefix_Kt = has_past ? (ta.past_k->padded_shape()[2] / pktile.get_height()) : 0;
    // The two-source flash compute has no mask path (pi0 uses non-causal full attention with no mask).
    TT_FATAL(!ta.mask.has_value(), "kv_sdpa FlashFused two-source path does not support an attention mask");

    // KV chunk size (tiles per flash chunk). This op is compute-bound on the per-chunk fixed overhead
    // (matmul re-init + the reduce/exp reconfig_data_format churn in sdpa_inner_loop), so we want the
    // FEWEST chunks: pick the largest divisor of Kt whose double-buffered per-chunk K/V CBs still fit a
    // modest L1 budget. Any divisor is correctness-safe -- the compute derives its subblock/granularity
    // from Sk_chunk_t at build time (determine_largest_subblock_size / find_valid_granularity both fall
    // back gracefully), and dividing Kt exactly keeps every chunk full (no ragged tail to mask).
    // E.g. Kt=33 now picks 11 (3 chunks) instead of the old {4,3,2}-capped 3 (11 chunks).
    // Cap chunk tiles so cb_k_in/cb_v_in (each Sk_chunk_t*DHt*2 tiles, double-buffered) stay bounded in
    // L1 regardless of head_dim: keep Sk_chunk_t*DHt <= 128 tiles per (single-buffered) K/V chunk.
    const uint32_t max_chunk_tiles = std::max<uint32_t>(1u, attrs.max_kv_chunk_tiles / DHt);
    auto pick_chunk = [&](uint32_t kt) -> uint32_t {
        uint32_t sc = 1;
        for (uint32_t cand = std::min(kt, max_chunk_tiles); cand >= 1; --cand) {
            if (kt % cand == 0) {
                sc = cand;
                break;
            }
        }
        return sc;
    };
    const uint32_t suffix_Sk_chunk_t = pick_chunk(suffix_Kt);
    const uint32_t suffix_num_chunks = suffix_Sk_chunk_t == 0 ? 0 : suffix_Kt / suffix_Sk_chunk_t;
    // Split-KV (flash decode): S cores per Q head, each taking prefix_Kt/S of the prefix. Clamp to 1
    // unless there is a prefix that divides evenly -- the suffix is not split (it is far shorter, and
    // it stays on the reducer so no split has mixed geometry).
    uint32_t kv_splits = std::max<uint32_t>(1u, attrs.kv_splits);
    if (!has_past || prefix_Kt == 0 || prefix_Kt % kv_splits != 0) {
        kv_splits = 1;
    }
    // Prefix chunk geometry is PER SPLIT, so every split has the same chunk count and the compute's
    // per-phase block sizing stays a compile-time constant.
    const uint32_t prefix_Kt_per_split = has_past ? prefix_Kt / kv_splits : 0;
    const uint32_t prefix_Sk_chunk_t = has_past ? pick_chunk(prefix_Kt_per_split) : 1;
    const uint32_t prefix_num_chunks = has_past ? prefix_Kt_per_split / prefix_Sk_chunk_t : 0;
    const uint32_t num_children = kv_splits - 1;

    // Subblock widths must fit the DST register budget (get_dest_reg_count halves it for
    // fp32_dest_acc, and again without dst_full_sync). Derive them from dst_size like the production
    // transformer SDPA rather than assuming the full head_dim fits (Sq_chunk_t == 1 here). Per phase.
    const auto ckc = ttnn::init_device_compute_kernel_config(
        device->arch(), attrs.compute_kernel_config, MathFidelity::HiFi2, false, false, false);
    const uint32_t dst_size = ttnn::get_dest_reg_count(ckc);
    const uint32_t suffix_qk_subblock_w =
        ttnn::prim::detail::determine_largest_subblock_size(1, suffix_Sk_chunk_t, dst_size).second;
    const uint32_t prefix_qk_subblock_w =
        ttnn::prim::detail::determine_largest_subblock_size(1, prefix_Sk_chunk_t, dst_size).second;
    const uint32_t out_subblock_w = ttnn::prim::detail::determine_largest_subblock_size(1, vDHt, dst_size).second;

    const uint32_t Sq_chunk_t = 1;
    const uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    // cb_qk_im / out CBs are shared by both phases; size them for the larger chunk.
    const uint32_t max_Sk_chunk_t = std::max(prefix_num_chunks ? prefix_Sk_chunk_t : 0u, suffix_Sk_chunk_t);
    const uint32_t out_tiles = Sq_chunk_t * vDHt;
    // sub_exp/mul-bcast DST unroll runs over Sk_chunk (differs per phase); the granularity must divide
    // BOTH phases' chunk sizes. Use their gcd when a prefix exists, else the suffix chunk.
    const uint32_t gran_base = prefix_num_chunks ? std::gcd(prefix_Sk_chunk_t, suffix_Sk_chunk_t) : suffix_Sk_chunk_t;

    // Core layout: NQH * kv_splits cores, split s of head h at core_vec[h * kv_splits + s]. Split 0 of
    // each head is that head's reducer (it owns the output row and merges the other splits' partials).
    const uint32_t num_cores = NQH * kv_splits;
    const CoreRangeSet cores =
        tt::tt_metal::num_cores_to_corerangeset(num_cores, device->compute_with_storage_grid_size(), /*row_wise=*/true);
    const auto core_vec = corerange_to_cores(cores, std::nullopt, true);
    TT_FATAL(
        core_vec.size() >= num_cores,
        "kv_sdpa: kv_splits={} needs {} cores for {} Q heads, but only {} are available",
        kv_splits,
        num_cores,
        NQH,
        core_vec.size());

    ProgramDescriptor desc;
    auto add_cb = [&](uint32_t idx, uint32_t ntiles, tt::DataFormat df, uint32_t ts, const auto& tile) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = ntiles * ts,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = idx, .data_format = df, .page_size = ts, .tile = TileDescriptor{tile}}}}});
    };
    using C = CBIndex;
    add_cb(C::c_0, q_chunk_tiles, qdf, q_ts, qtile);                 // cb_q_in
    add_cb(C::c_1, suffix_Sk_chunk_t * DHt * 2, kdf, k_ts, ktile);   // cb_k_in (suffix, double-buffered)
    add_cb(C::c_2, suffix_Sk_chunk_t * vDHt * 2, vdf, v_ts, vtile);  // cb_v_in (suffix)
    // Prefix K/V CBs (own tile geometry). When there is no past, prefix_num_chunks==0 so the compute
    // never touches them; declare a minimal placeholder so the CB index is valid.
    const uint32_t pk_cb_tiles = has_past ? prefix_Sk_chunk_t * DHt * 2 : 1;
    add_cb(C::c_8, pk_cb_tiles, pkdf, pk_ts, pktile);                    // cb_k_prefix
    add_cb(C::c_9, pk_cb_tiles, pvdf, pv_ts, pvtile);                    // cb_v_prefix
    add_cb(C::c_5, 1, bf16, bf16_ts, qtile);                             // cb_identity_scale_in
    add_cb(C::c_7, 1, bf16, bf16_ts, qtile);                             // cb_col_identity
    add_cb(C::c_24, Sq_chunk_t * max_Sk_chunk_t, bf16, bf16_ts, qtile);  // cb_qk_im
    add_cb(C::c_25, out_tiles, bf16, bf16_ts, qtile);                // cb_out_im_A
    add_cb(C::c_26, out_tiles, bf16, bf16_ts, qtile);                // cb_out_im_B
    add_cb(C::c_27, Sq_chunk_t, bf16, bf16_ts, qtile);               // cb_max_A
    add_cb(C::c_28, Sq_chunk_t, bf16, bf16_ts, qtile);               // cb_max_B
    add_cb(C::c_29, Sq_chunk_t, bf16, bf16_ts, qtile);               // cb_sum_A
    add_cb(C::c_30, Sq_chunk_t, bf16, bf16_ts, qtile);               // cb_sum_B
    add_cb(C::c_31, Sq_chunk_t, bf16, bf16_ts, qtile);               // cb_exp_max_diff
    add_cb(C::c_16, out_tiles, odf, o_ts, otile);                    // cb_out (interleaved output)
    // ---- Split-KV reduction CBs (declared on every core; untouched when kv_splits == 1) ----
    // A child's partial state lands in the reducer's cb_intermed as a (l, m, o) block per child slot;
    // the reducer's writer copies each slot into cb_l_in/cb_m_in/cb_out_o for the compute to merge.
    const uint32_t partial_block_tiles = out_tiles + 2 * Sq_chunk_t;  // o, then l and m
    // Sized only when a reduction actually happens. Declaring them at full size unconditionally cost
    // ~31 bf16 tiles (~62 KB) of L1 per core on the kv_splits == 1 path that never touches them, and
    // that alone moved the single-layer denoise 0.116 -> 0.129 ms. The indices must remain declared
    // (the writer reads get_tile_size(cb_intermed) unconditionally), so keep a 1-tile placeholder.
    const bool red = num_children > 0;
    const uint32_t stat_cb = red ? Sq_chunk_t : 1;
    const uint32_t out_cb = red ? out_tiles : 1;
    add_cb(C::c_10, red ? num_children * partial_block_tiles : 1, bf16, bf16_ts, qtile);  // cb_intermed
    add_cb(C::c_11, stat_cb, bf16, bf16_ts, qtile);                                       // cb_l_in  (child sum)
    add_cb(C::c_12, stat_cb, bf16, bf16_ts, qtile);                                       // cb_m_in  (child max)
    add_cb(C::c_13, out_cb, bf16, bf16_ts, qtile);                                        // cb_out_o (child out)
    add_cb(C::c_14, stat_cb, bf16, bf16_ts, qtile);                                       // cb_prev_sum_2
    add_cb(C::c_15, out_cb, bf16, bf16_ts, qtile);                                        // cb_out_accumulate_im_2
    add_cb(C::c_17, stat_cb, bf16, bf16_ts, qtile);                                       // cb_exp_max_diff_2
    // cb_partial_out: a worker's own (l, m, o) staged for its writer to push to the reducer.
    add_cb(C::c_18, red ? partial_block_tiles : 1, bf16, bf16_ts, qtile);

    // One semaphore per reducer, counting arrived children. Declared over all cores (workers never
    // read theirs); the reducer waits for it to reach num_children.
    constexpr uint32_t reducer_sem_id = 0;
    if (num_children > 0) {
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = reducer_sem_id, .core_type = CoreType::WORKER, .core_ranges = cores, .initial_value = 0});
    }

    // 1.0 packed as bf16 for the reduce/bcast scalars (identity_scalar_packed in prod writer).
    constexpr uint32_t identity_scalar_packed = 0x3F803F80u;

    // ---- Two kernel variants: [0] = reducer cores (split 0 of each head), [1] = worker cores ----
    // Every split-KV per-core value (this core's suffix chunk count, whether it reduces, how many
    // children it merges) is a COMPILE-TIME arg of its variant, not a runtime arg. That matters a lot:
    // passing them as runtime args cost ~12% even at kv_splits == 1 (0.116 -> 0.131 ms) because the
    // flash loops stopped being constexpr-bounded and lost constant propagation into
    // flash_accumulate_chunk. Same shape as nlp_create_qkv_heads_rope's qk_cores / v_cores split.
    // When kv_splits == 1 only variant [0] is emitted, with args identical to the pre-split program.
    std::vector<CoreRange> red_ranges, wrk_ranges;
    for (uint32_t h = 0; h < NQH; ++h) {
        for (uint32_t s = 0; s < kv_splits; ++s) {
            const CoreCoord c = core_vec[h * kv_splits + s];
            (s == 0 ? red_ranges : wrk_ranges).emplace_back(c, c);
        }
    }
    const CoreRangeSet role_cores[2] = {CoreRangeSet(red_ranges), CoreRangeSet(wrk_ranges)};
    const uint32_t num_variants = num_children > 0 ? 2u : 1u;

    Buffer* pk_buf = has_past ? ta.past_k->buffer() : ta.k.buffer();
    Buffer* pv_buf = has_past ? ta.past_v->buffer() : ta.v.buffer();

    // ---- Reader ----
    auto reader_cta_for = [&](bool is_red) {
        KernelDescriptor::CompileTimeArgs c = {
            NQH,
            DHt,
            prefix_Kt,
            prefix_Sk_chunk_t,
            prefix_num_chunks,
            suffix_Kt,
            suffix_Sk_chunk_t,
            is_red ? suffix_num_chunks : 0u,  // only the reducer reads the suffix
            (uint32_t)has_past};
        TensorAccessorArgs(*ta.q.buffer()).append_to(c);
        TensorAccessorArgs(*ta.k.buffer()).append_to(c);
        TensorAccessorArgs(*ta.v.buffer()).append_to(c);
        // Always append the prefix accessors so the reader's compile-time offsets are valid; when
        // there is no real past they alias k/v and the reader never reads them (has_past gates use).
        TensorAccessorArgs(*pk_buf).append_to(c);
        TensorAccessorArgs(*pv_buf).append_to(c);
        return c;
    };
    KernelDescriptor readers[2];
    for (uint32_t r = 0; r < num_variants; ++r) {
        readers[r].kernel_source = "ttnn/cpp/ttnn/operations/kv_sdpa/device/kernels/dataflow/reader_fused.cpp";
        readers[r].source_type = KernelDescriptor::SourceType::FILE_PATH;
        readers[r].core_ranges = role_cores[r];
        readers[r].compile_time_args = reader_cta_for(r == 0);
        readers[r].config =
            DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_0};
    }

    // ---- Writer (generates the sdpa scalars + drains cb_out to this core's Q head of the output) ----
    auto writer_cta_for = [&](bool is_red) {
        KernelDescriptor::CompileTimeArgs c = {
            DHt, identity_scalar_packed, reducer_sem_id, partial_block_tiles, (uint32_t)is_red, num_children};
        TensorAccessorArgs(*out.buffer()).append_to(c);
        return c;
    };
    KernelDescriptor writers[2];
    for (uint32_t r = 0; r < num_variants; ++r) {
        writers[r].kernel_source = "ttnn/cpp/ttnn/operations/kv_sdpa/device/kernels/dataflow/writer_fused.cpp";
        writers[r].source_type = KernelDescriptor::SourceType::FILE_PATH;
        writers[r].core_ranges = role_cores[r];
        writers[r].compile_time_args = writer_cta_for(r == 0);
        writers[r].config =
            DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_1};
    }

    // ---- Compute ----
    KernelDescriptor compute{};
    compute.kernel_source = "ttnn/cpp/ttnn/operations/kv_sdpa/device/kernels/compute/flash_fused.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = {
        DHt,
        attrs.scale_bits,
        prefix_num_chunks,
        prefix_Sk_chunk_t,
        prefix_qk_subblock_w,
        out_subblock_w,
        suffix_num_chunks,
        suffix_Sk_chunk_t,
        suffix_qk_subblock_w,
        out_subblock_w};
    // Granularity defines are DST-loop unroll factors (compute_common.hpp): each must be <= dst_size
    // and divide its tile count, or the DST overflows / trailing tiles are dropped. Derive them from
    // dst_size like the production SDPA (Sq_chunk_t == 1 here, so stats/reduce counts are 1).
    compute.defines = {
        {"STATS_GRANULARITY", std::to_string(ttnn::prim::detail::find_valid_granularity(1, dst_size))},
        {"SUB_EXP_GRANULARITY", std::to_string(ttnn::prim::detail::find_valid_granularity(gran_base, dst_size))},
        {"MUL_BCAST_GRANULARITY", std::to_string(ttnn::prim::detail::find_valid_granularity(gran_base, dst_size))},
        {"DHT_GRANULARITY", std::to_string(ttnn::prim::detail::find_valid_granularity(DHt, dst_size))},
        {"REDUCE_GRANULARITY", std::to_string(ttnn::prim::detail::find_valid_granularity(1, dst_size / 2))},
        {"EXP_APPROX_MODE", "0"},
        // QK-scores tile face count for reduce_block_max_row (compute_streaming.hpp): 2 for a 16x32
        // tiny Q tile (one face-row), 4 for a full 32x32 tile. Defaults to 4 in the kernel, which
        // mis-walks the faces on a tiny tile and corrupts the online-softmax max/SALAD combine
        // (PCC regression); derive it from the Q operand tile height.
        {"QK_NUM_FACES", std::to_string(qtile.get_height() < tt::constants::TILE_HEIGHT ? 2u : 4u)}};
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = ckc.math_fidelity,
        .fp32_dest_acc_en = ckc.fp32_dest_acc_en,
        .dst_full_sync_en = ckc.dst_full_sync_en,
        .math_approx_mode = ckc.math_approx_mode};

    // Only truly per-CORE values stay runtime args: the prefix slice offset, this head's output row,
    // and the reducer's NOC coords / slot index. Everything role-dependent is a compile-time arg of
    // the variant assigned to that core set.
    KernelDescriptor computes[2];
    for (uint32_t r = 0; r < num_variants; ++r) {
        computes[r] = compute;
        computes[r].core_ranges = role_cores[r];
        computes[r].compile_time_args[6] = (r == 0) ? suffix_num_chunks : 0u;  // per-core suffix chunks
        computes[r].compile_time_args.push_back(r == 0 ? 1u : 0u);             // is_reducer
        computes[r].compile_time_args.push_back(num_children);
    }

    for (uint32_t h = 0; h < NQH; ++h) {
        const uint32_t kv_head = h / group;
        const CoreCoord reducer_phys = device->worker_core_from_logical_core(core_vec[h * kv_splits]);
        for (uint32_t s = 0; s < kv_splits; ++s) {
            const CoreCoord core = core_vec[h * kv_splits + s];
            const uint32_t r = (s == 0) ? 0u : 1u;
            const uint32_t prefix_tile_start = s * prefix_Kt_per_split;
            readers[r].emplace_runtime_args(
                core, {ta.q.buffer(), ta.k.buffer(), ta.v.buffer(), h, kv_head, pk_buf, pv_buf, prefix_tile_start});
            writers[r].emplace_runtime_args(
                core,
                {out.buffer(),
                 h,
                 (uint32_t)reducer_phys.x,
                 (uint32_t)reducer_phys.y,
                 // Workers occupy slots 0..num_children-1 of the reducer's cb_intermed.
                 s == 0 ? 0u : (s - 1)});
        }
    }
    for (uint32_t r = 0; r < num_variants; ++r) {
        desc.kernels.push_back(std::move(readers[r]));
        desc.kernels.push_back(std::move(writers[r]));
        desc.kernels.push_back(std::move(computes[r]));
    }
    return desc;
}

}  // namespace ttnn::operations::kv_sdpa
