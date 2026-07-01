// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bluestein.cpp — arbitrary-N DFT via Bluestein's chirp-Z transform.
//
// Per-call device dispatch chain (B = 1, length N → length N):
//
//   1.  complex_mul(x, chirp_n)              — pre-twiddle, shape (B, N)
//   2.  pad to (B, M),  trailing zeros       — zero_pad_to_m (rebank) or ttnn::pad
//   3.  fft (forward, length M)              — ttnn::experimental::fft
//   4.  complex_mul(A, B)                    — convolution multiply, (B, M)
//   5.  ifft (length M)                      — ttnn::experimental::ifft
//   6.  extract first N elements             — trim_to_n (rebank) or ttnn::slice
//   7.  complex_mul(c, chirp_k)              — post-twiddle, shape (B, N)
//
// Steps 2 and 6 switch to rebank_rm-based helpers (zero_pad_to_m / trim_to_n)
// when the output row exceeds 64 KB and N%1024==0.  This avoids ttnn::pad's
// 16× async CB and ttnn::slice's 32× CB, both of which overflow the 1.5 MB
// L1 for M ≥ 131072 (fp32).
//
// Steps 1 and 7 (complex_mul_safe) use a 1024-element chunked path when
// P = N > 1024.  If P%1024 ≠ 0, the function falls back to ttnn::pad (Case B),
// which allocates CB ≈ 17 × P_pad × elem_bytes.  This is safe for P < ~16 K
// (fp32) or ~32 K (bf16), but overflows L1 for larger non-1024-aligned N.
// Callers must restrict to P%1024==0 when P*elem_bytes > 64 KB.
//
// Step 3 / 5 each lower to either the SingleTileStockham factory
// (M ≤ 1024) or fft_two_pass (1024 < M ≤ 1M).  Chirp_n, chirp_k, and
// B = FFT(b_cyc) are pre-computed and cached per (device, N, dtype) —
// see device/bluestein_host.hpp.

#include "ttnn/operations/experimental/fft/bluestein.hpp"

#include "ttnn/operations/experimental/fft/complex_mul.hpp"
#include "ttnn/operations/experimental/fft/fft.hpp"
#include "ttnn/operations/experimental/fft/device/bluestein_host.hpp"
#include "ttnn/operations/experimental/fft/device/rebank_rm_device_operation.hpp"
#include "ttnn/operations/experimental/fft/device/rebank_rm_merge_device_operation.hpp"

#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/creation/creation.hpp"  // ttnn::zeros for the imag input fallback

#include "ttnn/distributed/types.hpp"             // MeshDevice
#include "ttnn/types.hpp"

#include <algorithm>
#include <array>
#include <vector>

namespace ttnn::operations::experimental {

namespace {

// Source-page threshold above which ttnn::reshape allocates a CB equal to the
// full source row (e.g. (1,131072)→(128,1024) uses 4×1 MB CB → L1 overflow).
// Matches kRebankThresholdBytes in fft.cpp (kept separate to avoid Unity collision).
constexpr uint32_t kBluesteinRebankThreshold = 64u * 1024u;

// Page-shrinking reshape: (B, src_cols) → (B × src_cols/new_cols, new_cols).
//
// For large source pages (> 64 KB), ttnn::reshape overflows L1, so we use
// rebank_rm (CB = 8 KB).  rebank_rm requires the input last-dim to be a
// power of 2.  When src_cols is NOT a power of 2, we:
//   1. Column-concat a zero block to reach the next pow-2 (CB = 2×pow2×ebytes).
//   2. Call rebank_rm on the pow-2-aligned tensor (CB = 8 KB).
//   3. Row-slice off the extra rows introduced by the zero-padding (CB = 256 KB).
// For small source pages the standard ttnn::reshape is used (metadata or small CB).
static ttnn::Tensor shrink_reshape(
    const ttnn::Tensor& t, uint32_t new_cols)
{
    const auto& s = t.padded_shape();
    const uint32_t src_cols = static_cast<uint32_t>(s[-1]);
    const uint32_t elem_bytes =
        (t.dtype() == tt::tt_metal::DataType::BFLOAT16) ? 2u : 4u;

    if (src_cols * elem_bytes <= kBluesteinRebankThreshold) {
        // Small source page: standard reshape (metadata-only or small-CB kernel).
        uint32_t total = 1u;
        for (int d = 0; d < static_cast<int>(s.size()); ++d)
            total *= static_cast<uint32_t>(s[d]);
        const uint32_t new_rows = total / new_cols;
        return ttnn::reshape(t,
            ttnn::Shape{ttnn::SmallVector<uint32_t>{new_rows, new_cols}});
    }

    // Large source page: prefer rebank_rm (DRAM-to-DRAM, CB = 8 KB).
    // Compute next pow-2 ≥ src_cols (needed for the pow-2 check and the concat fallback).
    uint32_t src_pow2 = 1u;
    while (src_pow2 < src_cols) src_pow2 <<= 1u;

    if (src_pow2 == src_cols) {
        // src_cols is exactly a power of 2: rebank directly.
        return ttnn::prim::rebank_rm(t, new_cols);
    }
    // Non-pow-2: use rebank_rm directly only when the concat-to-pow2 fallback would
    // overflow L1 (concat CB = 2 * src_pow2 * elem > ~1 MB).
    // For small non-pow2 N (e.g. 64512 → src_pow2=65536 → CB=512KB) the concat path
    // is safe and avoids dispatch-core-placement issues that arise when the rebank_rm
    // work-unit count (B × N/1024) cannot be tiled into the device's physical grid.
    if (src_cols % new_cols == 0u && (uint64_t)src_pow2 * elem_bytes > (1u << 19u)) {
        return ttnn::prim::rebank_rm(t, new_cols);
    }

    // src_cols is not divisible by new_cols OR the concat path is safe.
    // Zero-pad to the next pow-2 (which is a multiple of new_cols).

    // Compute B (product of all leading dims).
    uint32_t B_total = 1u;
    for (int d = 0; d < static_cast<int>(s.size()) - 1; ++d)
        B_total *= static_cast<uint32_t>(s[d]);

    auto* dev = t.device();
    TT_FATAL(dev != nullptr, "shrink_reshape: tensor has no device.");
    const auto mc = t.memory_config();

    // Append (B_total, src_pow2 - src_cols) zeros via concat along dim=1.
    // CB = 2 × src_pow2 × elem_bytes ≤ 1 MB (for src_pow2 ≤ 131072 fp32).
    auto zeros_tail = ttnn::zeros(
        ttnn::Shape{ttnn::SmallVector<uint32_t>{B_total, src_pow2 - src_cols}},
        t.dtype(), t.layout(), std::ref(*dev), mc);
    auto t_pow2 = ttnn::concat({t, zeros_tail}, /*dim=*/1);

    // rebank_rm: (B_total, src_pow2) → (B_total × src_pow2/new_cols, new_cols).
    // src_pow2 IS a power of 2.  CB = 8 KB.
    auto rebankd = ttnn::prim::rebank_rm(t_pow2, new_cols);

    // Trim the extra rows added by the zero-padding.
    const uint32_t n_want = B_total * (src_cols / new_cols);
    const uint32_t n_have = B_total * (src_pow2 / new_cols);
    if (n_have > n_want) {
        const ttnn::SmallVector<uint32_t> beg = {0u, 0u};
        const ttnn::SmallVector<uint32_t> end = {n_want, new_cols};
        const ttnn::SmallVector<uint32_t> stp = {1u, 1u};
        rebankd = ttnn::slice(rebankd, beg, end, stp, mc);
    }
    return rebankd;
}

// complex_mul_safe: element-wise complex multiply for any last-dim size.
//
// The complex_mul kernel has a hard cap of P ≤ 1024 (one tile row), and its
// CB scales as 4 inputs × 2 (dbl-buf) × B_eff × 1024 × elem_bytes.  For
// large B_eff (= B × P/1024) that exceeds the 1.5 MB L1 limit.
//
// Strategy for P > 1024:
//   1. Rebank (B, P) → (B·P/1024, 1024) via rebank_rm when the source page
//      is large (avoids the multi-MB reshape CB).
//   2. If B_eff = B·P/1024 is within the safe limit (b_safe), one complex_mul
//      call suffices.  Otherwise, slice the rebankd tensor into b_safe-row
//      blocks, call complex_mul once per block, and concat the results.
//   3. Reshape the (B·P/1024, 1024) result back to (B, P) — page-growing
//      reshape, CB ≤ 1 MB, always within L1.
//
// b_safe derivation (CB ≤ 1 MB):
//   fp32: b_safe = 1 MB / (4 × 2 × 1024 × 4 B) = 32 rows
//   bf16: b_safe = 1 MB / (4 × 2 × 1024 × 2 B) = 64 rows
//
// Case B (P not divisible by 1024): pad to next multiple of 1024 first,
// apply the above, then slice back to (B, P).
static std::tuple<ttnn::Tensor, ttnn::Tensor> complex_mul_chunked(
    const ttnn::Tensor& ar, const ttnn::Tensor& ai,
    const ttnn::Tensor& br, const ttnn::Tensor& bi,
    uint32_t B, uint32_t P_col)
{
    // P_col must be a multiple of 1024 on entry.
    const uint32_t nchunks   = P_col / 1024u;
    const uint32_t total_rows = B * nchunks;

    const uint32_t elem_bytes =
        (ar.dtype() == tt::tt_metal::DataType::BFLOAT16) ? 2u : 4u;
    // Max rows per complex_mul to keep CB ≤ 1 MB.
    const uint32_t b_safe = (1u << 20u) / (8u * 1024u * elem_bytes); // 32/64

    // Rebank (B, P_col) → (total_rows, 1024) with tiny CB.
    auto ar_f = shrink_reshape(ar, 1024u);
    auto ai_f = shrink_reshape(ai, 1024u);
    auto br_f = shrink_reshape(br, 1024u);
    auto bi_f = shrink_reshape(bi, 1024u);

    const auto mc = ar.memory_config();

    if (total_rows <= b_safe) {
        // Small enough: one complex_mul call.
        auto [cr, ci] = complex_mul(ar_f, ai_f, br_f, bi_f);
        const auto orig = ttnn::Shape{ttnn::SmallVector<uint32_t>{B, P_col}};
        return {ttnn::reshape(cr, orig), ttnn::reshape(ci, orig)};
    }

    // Large: loop in b_safe-row blocks, collect results, concat.
    std::vector<ttnn::Tensor> cr_vec, ci_vec;
    cr_vec.reserve((total_rows + b_safe - 1u) / b_safe);
    ci_vec.reserve(cr_vec.capacity());

    for (uint32_t start = 0u; start < total_rows; start += b_safe) {
        const uint32_t end_r = std::min(start + b_safe, total_rows);
        const ttnn::SmallVector<uint32_t> beg_idx  = {start, 0u};
        const ttnn::SmallVector<uint32_t> end_idx  = {end_r, 1024u};
        const ttnn::SmallVector<uint32_t> step_idx = {1u, 1u};
        auto arc = ttnn::slice(ar_f, beg_idx, end_idx, step_idx, mc);
        auto aic = ttnn::slice(ai_f, beg_idx, end_idx, step_idx, mc);
        auto brc = ttnn::slice(br_f, beg_idx, end_idx, step_idx, mc);
        auto bic = ttnn::slice(bi_f, beg_idx, end_idx, step_idx, mc);
        auto [crc, cic] = complex_mul(arc, aic, brc, bic);
        cr_vec.push_back(std::move(crc));
        ci_vec.push_back(std::move(cic));
    }

    // Concat along dim 0: (k × b_safe, 1024) → (total_rows, 1024).
    auto cr_f = ttnn::concat(cr_vec, /*dim=*/0);
    auto ci_f = ttnn::concat(ci_vec, /*dim=*/0);

    // Reassemble (total_rows, 1024) → (B, P_col).
    // For large P_col (output page > 64 KB), ttnn::reshape allocates
    // CB = 2 × P_col × elem_bytes which can overflow L1.  Use
    // rebank_rm_merge instead (CB = 2 × 1024 × elem_bytes = 8 KB).
    if ((uint64_t)P_col * elem_bytes > kBluesteinRebankThreshold) {
        return {ttnn::prim::rebank_rm_merge(cr_f, nchunks),
                ttnn::prim::rebank_rm_merge(ci_f, nchunks)};
    }
    const auto orig = ttnn::Shape{ttnn::SmallVector<uint32_t>{B, P_col}};
    return {ttnn::reshape(cr_f, orig), ttnn::reshape(ci_f, orig)};
}

static std::tuple<ttnn::Tensor, ttnn::Tensor> complex_mul_safe(
    const ttnn::Tensor& ar, const ttnn::Tensor& ai,
    const ttnn::Tensor& br, const ttnn::Tensor& bi)
{
    const auto& sh = ar.padded_shape();
    const uint32_t P = static_cast<uint32_t>(sh[-1]);
    if (P <= 1024u)
        return complex_mul(ar, ai, br, bi);

    uint32_t B = 1u;
    for (int d = 0; d < static_cast<int>(sh.size()) - 1; ++d)
        B *= static_cast<uint32_t>(sh[d]);

    const auto mc = ar.memory_config();

    if (P % 1024u == 0u) {
        return complex_mul_chunked(ar, ai, br, bi, B, P);
    }

    // Case B: P not divisible by 1024 — pad to P_pad, multiply, slice back.
    //
    // L1 capacity constraint: ttnn::pad allocates CB ≈ 17 × output_page_bytes.
    // For large P (P_pad × elem_bytes > kBluesteinRebankThreshold), this
    // overflows the 1.5 MB L1 limit.  The concat-based alternative also
    // overflows (CB = 2 × P_pad × elem_bytes).
    //
    // Callers must ensure P × elem_bytes ≤ kBluesteinRebankThreshold when
    // using this function.  In Bluestein, steps 1 and 7 multiply (B, N)
    // tensors; for large N the Bluestein planner must only schedule this path
    // for N < ~16 K (fp32) or ~32 K (bf16).  Larger non-1024-aligned N
    // requires a new streaming kernel (tracked as a future enhancement).
    const uint32_t P_pad = (P + 1023u) & ~1023u;
    const uint32_t pad_len = P_pad - P;

    const ttnn::SmallVector<std::array<uint32_t, 2>> padding = {
        {{0u, 0u}},
        {{0u, pad_len}},
    };
    auto ar_p = ttnn::pad(ar, padding, 0.0f, /*use_multicore=*/true, mc);
    auto ai_p = ttnn::pad(ai, padding, 0.0f, /*use_multicore=*/true, mc);
    auto br_p = ttnn::pad(br, padding, 0.0f, /*use_multicore=*/true, mc);
    auto bi_p = ttnn::pad(bi, padding, 0.0f, /*use_multicore=*/true, mc);

    auto [cr_p, ci_p] = complex_mul_chunked(ar_p, ai_p, br_p, bi_p, B, P_pad);

    // Slice result back to (B, P) — zero-padded positions are 0 × anything = 0.
    const ttnn::SmallVector<uint32_t> begins = {0u, 0u};
    const ttnn::SmallVector<uint32_t> ends   = {B,  P};
    const ttnn::SmallVector<uint32_t> step   = {1u, 1u};
    auto cr = ttnn::slice(cr_p, begins, ends, step, mc);
    auto ci = ttnn::slice(ci_p, begins, ends, step, mc);
    return {std::move(cr), std::move(ci)};
}

// zero_pad_to_m: zero-pad (B, N) → (B, M) without using ttnn::pad.
//
// For small output (M × elem_bytes ≤ kBluesteinRebankThreshold):
//   Creates (B, M-N) zeros and column-concatenates.  CB = 2 × M × elem_bytes.
//
// For large output (M × elem_bytes > kBluesteinRebankThreshold) and N%1024==0:
//   Uses a streaming rebank+merge approach (all steps CB ≤ 8 KB):
//     1. rebank_rm(t, 1024)         → (B*N/1024, 1024)
//     2. zeros(B*(M-N)/1024, 1024)  → tiny pages
//     3. concat dim=0               → (B*M/1024, 1024),  CB = 2×4 KB = 8 KB
//     4. rebank_rm_merge(., M/1024) → (B, M),            CB = 2×4 KB = 8 KB
//   This avoids creating any tensor with a page > 4 KB in L1.
//   Requires N%1024==0 and (M-N)%1024==0 (guaranteed when M is pow-2 and N%1024==0).
static ttnn::Tensor zero_pad_to_m(
    const ttnn::Tensor& t, uint32_t M)
{
    const auto& s    = t.padded_shape();
    const uint32_t B = static_cast<uint32_t>(s[0]);
    const uint32_t N = static_cast<uint32_t>(s[-1]);
    if (N == M) return t;

    auto* dev = t.device();
    TT_FATAL(dev != nullptr, "zero_pad_to_m: tensor has no device.");
    const auto mc = t.memory_config();
    const uint32_t elem_bytes =
        (t.dtype() == tt::tt_metal::DataType::BFLOAT16) ? 2u : 4u;

    // Fast path for large M and 1024-aligned N (avoids large-CB concat).
    // Uses a streaming rebank+zeros-concat+merge chain; all steps have tiny
    // (8 KB) CBs regardless of M or N.  The rebank_rm factory's grid-
    // validation loop correctly handles non-pow-2 num_units (e.g. N=64512 →
    // 63 units → 21 cores with grid {7,3}; N=525312 → 513 units → 1 core).
    // The concat fallback below only handles small M (CB = 2×M×elem ≤ 750 KB).
    if ((uint64_t)M * elem_bytes > kBluesteinRebankThreshold &&
        N % 1024u == 0u && (M - N) % 1024u == 0u) {
        const uint32_t pad_chunks = B * (M - N) / 1024u;

        auto rebankd = ttnn::prim::rebank_rm(t, 1024u);
        auto zeros_r = ttnn::zeros(
            ttnn::Shape{ttnn::SmallVector<uint32_t>{pad_chunks, 1024u}},
            t.dtype(), t.layout(), std::ref(*dev), mc);
        auto stacked = ttnn::concat({rebankd, zeros_r}, /*dim=*/0);
        return ttnn::prim::rebank_rm_merge(stacked, M / 1024u);
    }

    // Fallback: (B, M-N) zeros, column-concatenated.  CB = 2 × M × elem_bytes.
    // Only reached when M × elem_bytes ≤ kBluesteinRebankThreshold (≤ 64 KB),
    // so CB ≤ 128 KB — well within the 1.5 MB L1 limit.
    auto zeros_tail = ttnn::zeros(
        ttnn::Shape{ttnn::SmallVector<uint32_t>{B, M - N}},
        t.dtype(), t.layout(), std::ref(*dev), mc);
    return ttnn::concat({t, zeros_tail}, /*dim=*/1);
}

// trim_to_n: extract first N elements from (B, M) without ttnn::slice.
//
// ttnn::slice on a 2-D RM tensor allocates CB = 32 × 2 × output_row_bytes,
// which overflows L1 for N > ~22 K elements (fp32).  This helper rebands to
// (B·M/1024, 1024), takes the first B·N/1024 rows via a row-slice (page =
// 4 KB), then reassembles to (B, N):
//   - small N (N × elem ≤ 64 KB): ttnn::reshape — CB = 2 × N × elem ≤ 1 MB.
//   - large N (N × elem >  64 KB): rebank_rm_merge — CB = 2 × 1024 × elem = 8 KB.
//     n_chunks = N/1024 need not be a power of 2.
//
// PRECONDITION: N % 1024 == 0, M must be a power of 2 (Bluestein always
// satisfies this: M = next_pow2(2N-1) ≥ 2048), M >= N.
static ttnn::Tensor trim_to_n(
    const ttnn::Tensor& t, uint32_t N)
{
    const auto& s    = t.padded_shape();
    const uint32_t B = static_cast<uint32_t>(s[0]);
    const uint32_t M = static_cast<uint32_t>(s[-1]);
    if (M == N) return t;

    const uint32_t n_chunks = N / 1024u;
    const auto mc = t.memory_config();
    const uint32_t elem_bytes =
        (t.dtype() == tt::tt_metal::DataType::BFLOAT16) ? 2u : 4u;

    // (B, M) → (B·m_chunks, 1024).  CB = 8 KB via rebank_rm.
    auto flat_m = ttnn::prim::rebank_rm(t, 1024u);

    // Row-slice: keep first B·n_chunks rows.
    // page = 4 KB → CB = 32 × 2 × 4 KB = 256 KB.
    const ttnn::SmallVector<uint32_t> beg = {0u, 0u};
    const ttnn::SmallVector<uint32_t> end = {B * n_chunks, 1024u};
    const ttnn::SmallVector<uint32_t> stp = {1u, 1u};
    auto flat_n = ttnn::slice(flat_m, beg, end, stp, mc);

    // Reassemble (B·n_chunks, 1024) → (B, N).
    if ((uint64_t)N * elem_bytes > kBluesteinRebankThreshold) {
        // Large N: CB = 2 × 1024 × elem_bytes = 8 KB (n_chunks need not be pow-2).
        return ttnn::prim::rebank_rm_merge(flat_n, n_chunks);
    }
    // Small N: page-growing reshape, CB = 2 × N × elem_bytes ≤ 1 MB.
    return ttnn::reshape(
        flat_n,
        ttnn::Shape{ttnn::SmallVector<uint32_t>{B, N}});
}

// Build a (1, N) zeros tensor matching `like` for the implicit zero-imag
// case (Bluestein needs an explicit imag input because the pipeline does
// a complex_mul as its very first step).
ttnn::Tensor make_zeros_like(const ttnn::Tensor& like) {
    auto* dev = like.device();
    TT_FATAL(dev != nullptr, "bluestein_fft: input tensor has no device.");
    return ttnn::zeros(
        like.logical_shape(),
        like.dtype(),
        like.layout(),
        std::ref(*dev),
        like.memory_config());
}

}  // namespace

std::tuple<ttnn::Tensor, ttnn::Tensor> bluestein_fft(
    const ttnn::Tensor& input_real,
    std::optional<ttnn::Tensor> input_imag,
    uint32_t N,
    FFTPrecision precision,
    bool inverse)
{
    using namespace ttnn::experimental::prim::bluestein_host;

    // ── Validation ──────────────────────────────────────────────────────
    const auto& in_shape = input_real.padded_shape();
    TT_FATAL(in_shape.size() == 2u,
        "bluestein_fft: input must be 2-D (B, N).  Got rank {}.",
        in_shape.size());
    const uint32_t B = static_cast<uint32_t>(in_shape[0]);
    TT_FATAL(B >= 1u,
        "bluestein_fft: batch dim must be ≥ 1 (got {}).", B);
    TT_FATAL(static_cast<uint32_t>(in_shape[1]) == N,
        "bluestein_fft: input last-dim must equal N = {} (got {}).",
        N, static_cast<uint32_t>(in_shape[1]));
    TT_FATAL(input_real.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "bluestein_fft: input must be ROW_MAJOR.");
    TT_FATAL(input_real.dtype() == tt::tt_metal::DataType::FLOAT32 ||
             input_real.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "bluestein_fft: only Float32 / BFloat16 supported.");

    const uint32_t M = bluestein_M(N);
    // M is capped by fft_three_pass limit (2^30).  Inner fft() / ifft() calls
    // route automatically through fft_two_pass (M ≤ 2^20) or fft_three_pass
    // (2^20 < M ≤ 2^30) via the unified router in fft.cpp.
    TT_FATAL(M <= (1u << 30),
        "bluestein_fft: padded M = {} > 2^30 is not yet supported (N = {}).",
        M, N);

    if (input_imag.has_value()) {
        TT_FATAL(input_imag->padded_shape() == in_shape &&
                 input_imag->dtype()        == input_real.dtype() &&
                 input_imag->layout()       == input_real.layout(),
            "bluestein_fft: input_imag must match input_real in "
            "shape/dtype/layout.");
    }

    // ── Get plan (chirp_n, chirp_k, B_fft) — cached per (device, N, dtype, B, inverse).
    auto plan = get_or_create(
        input_real.device(),
        N,
        input_real.dtype(),
        B,
        precision,
        inverse);

    // ── Materialise an explicit zero imag input when omitted ───────────
    //   The first step (complex_mul with chirp_n) requires both halves;
    //   our complex_mul is shape-strict and doesn't broadcast a missing
    //   imag, so we synthesise one here.
    const ttnn::Tensor x_imag = input_imag.has_value()
        ? *input_imag
        : make_zeros_like(input_real);

    // ── Step 1: pre-multiply by chirp_n  (1, N) × (1, N) → (1, N).
    auto [a_re, a_im] = complex_mul_safe(
        input_real, x_imag, plan->chirp_n_re, plan->chirp_n_im);

    // ── Step 2: zero-pad last dim from N to M  (B, N) → (B, M).
    //   ttnn::pad's 16× async CB overflows L1 for M ≥ 131072 fp32 (CB = 17×
    //   M × elem_bytes ≈ 17 × 524 KB = 8.9 MB).  Use zero_pad_to_m (concat)
    //   which needs CB = 2 × M × elem_bytes ≤ 1 MB.  No N%1024 constraint.
    const uint32_t elem_bytes =
        (a_re.dtype() == tt::tt_metal::DataType::BFLOAT16) ? 2u : 4u;
    const bool large_m_pad = ((uint64_t)M * elem_bytes > kBluesteinRebankThreshold);

    ttnn::Tensor a_pad_re, a_pad_im;
    if (large_m_pad) {
        a_pad_re = zero_pad_to_m(a_re, M);
        a_pad_im = zero_pad_to_m(a_im, M);
    } else {
        ttnn::SmallVector<std::array<uint32_t, 2>> padding = {
            {{0u, 0u}},
            {{0u, M - N}},
        };
        a_pad_re = ttnn::pad(a_re, padding, /*value=*/0.0f,
                             /*use_multicore=*/true, a_re.memory_config());
        a_pad_im = ttnn::pad(a_im, padding, /*value=*/0.0f,
                             /*use_multicore=*/true, a_im.memory_config());
    }

    // ── Step 3: forward FFT_M  ─────────────────────────────────────────
    //   Routes through SingleTileStockham (M ≤ 1024) or fft_two_pass
    //   (1024 < M ≤ 1M).
    auto [A_re, A_im] = fft(a_pad_re, a_pad_im, precision);

    // ── Step 4: convolution multiply  A ⊙ B   (B, M) × (B, M).
    //   Uses complex_mul_safe because M may exceed the 1024-element kernel cap
    //   (e.g. N=997 → M=2048).
    auto [C_re, C_im] = complex_mul_safe(A_re, A_im, plan->B_re, plan->B_im);

    // ── Step 5: inverse FFT_M  ─────────────────────────────────────────
    auto [c_re, c_im] = ifft(C_re, C_im, precision);

    // ── Step 6: extract first N elements from (B, M).
    //   ttnn::slice on a 2-D RM tensor allocates CB = 32 × 2 × output_row_bytes,
    //   which overflows L1 for N > ~22 K (fp32).  Use the rebank_rm-based
    //   helper when N is large and a multiple of 1024.
    const bool large_n_slice = ((uint64_t)N * elem_bytes > kBluesteinRebankThreshold)
                               && (N % 1024u == 0u);

    ttnn::Tensor c_re_n, c_im_n;
    if (large_n_slice) {
        c_re_n = trim_to_n(c_re, N);
        c_im_n = trim_to_n(c_im, N);
    } else {
        ttnn::SmallVector<uint32_t> begins = {0u, 0u};
        ttnn::SmallVector<uint32_t> ends   = {B,  N};
        ttnn::SmallVector<uint32_t> step   = {1u, 1u};
        c_re_n = ttnn::slice(c_re, begins, ends, step, c_re.memory_config());
        c_im_n = ttnn::slice(c_im, begins, ends, step, c_im.memory_config());
    }

    // ── Step 7: post-multiply by chirp_k → final DFT output X[k].
    auto [X_re, X_im] = complex_mul_safe(
        c_re_n, c_im_n, plan->chirp_k_re, plan->chirp_k_im);

    return {std::move(X_re), std::move(X_im)};
}

}  // namespace ttnn::operations::experimental
