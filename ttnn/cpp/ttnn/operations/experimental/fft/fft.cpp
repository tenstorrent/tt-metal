// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/fft/fft.hpp"

#include <cstdlib>
#include <optional>
#include <tuple>
#include <utility>

#include "device/fft_device_operation.hpp"
#include "device/fft_radix_pass_device_operation.hpp"
#include "device/apply_twiddles_xl_device_operation.hpp"
#include "device/transpose_rm_device_operation.hpp"
#include "device/rebank_rm_device_operation.hpp"
#include "device/rebank_rm_merge_device_operation.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/types.hpp"  // ttnn::Shape, ttnn::SmallVector

// Bluestein: included after fft.hpp to avoid circular dependency
#include "ttnn/operations/experimental/fft/bluestein.hpp"

namespace ttnn::operations::experimental {

namespace {

// ───────────────────────────────────────────────────────────────────────
// Two-pass Cooley–Tukey composite (commit 3c, corrected commit 5c,
// extended for complex input commit 6a)
//
// For pow-2 N with 1024 < N ≤ 1M, factor N = N1 · N2 (both pow-2, both
// in [32, 1024]).  We use the standard mixed-radix DIT decomposition
// that — for natural-order input AND natural-order output — requires
// pre- and post-transposes so each pass FFTs along the LAST axis.
//
// COMPLEX INPUT (commit 6a, for Bluestein): the optional `input_imag`
// is shape-matched to `input_real` and gets the same pre-transpose
// before being threaded into Pass-1 of the radix kernel.  Pass-1
// already supports complex input natively (the underlying Stockham
// kernel computes a true complex FFT when both halves are provided),
// so the only extra cost is one transpose_rm dispatch on the imag
// tensor.
//
// Index packing (input n natural, output K natural):
//   n = n1·N2 + n2   (n1 OUTER, n2 INNER of the (B, N) row)
//   K = k2·N1 + k1   (k1 INNER, k2 OUTER  of the (B, N) row)
//
// With this packing every (n_i, k_j) cross-term in n·K/N is either
// integer (vanishes) or matches a clean FFT/twiddle factor:
//
//   X[k2·N1 + k1] = Σ_{n2} W_{N2}^{n2·k2} · ( exp(-2πi·n2·k1/N) ·
//                       Σ_{n1} W_{N1}^{n1·k1} · x[n1, n2] )
//                     ╰── Pass-2 ──╯ ╰─ twiddle ─╯  ╰── Pass-1 ──╯
//
// Implementation chain (3 transposes + 2 fft_radix_pass dispatches):
//   1. reshape (B, N) → (B, N1, N2)              [view, free]
//   2. transpose_rm   → (B, N2, N1)              [data movement]
//   3. view as (B·N2, N1)                        [view]
//   4. fft_radix_pass(P=N1, twiddle_N2=N2)       [Pass-1 FFT_N1 fused with
//                                                 twiddle exp(-2πi·n2·k1/N)]
//   5. view + transpose_rm + view → (B·N1, N2)   [data movement]
//   6. fft_radix_pass(P=N2, twiddle_N2=0)        [Pass-2 pure FFT_N2]
//   7. view + transpose_rm → (B, N2, N1)         [data movement]
//   8. reshape → (B, N)                          [view, free]
//
// NOTE: the EARLIER version of fft_two_pass (commit 4) did the inner
// FFT first WITHOUT the initial transpose and applied the twiddle on
// pass 2 with arguments (P=N2, twiddle_N2=N1).  That doesn't correspond
// to any valid Cooley–Tukey decomposition with natural I/O; on N=4 it
// produced [10, -1+i, -4, -1-i] instead of the correct
// [10, -2+2i, -2, -2-2i].  The diagnostic at N=2048/4096 showed
// rel_err ≈ √2 (output uncorrelated with reference) — symptom of an
// algorithm that sums elements correctly (DC bin matched) but applies
// the wrong twiddles everywhere else.
//
// Gated by TT_FFT_NATIVE=1.  Falls back to legacy CachedProgram path
// (prim::fft) when not enabled.
// ───────────────────────────────────────────────────────────────────────

bool native_path_enabled() {
    const char* v = std::getenv("TT_FFT_NATIVE");
    if (v == nullptr) return true;               // default ON
    return !(v[0] == '0' && v[1] == '\0');       // OFF only if explicitly "0"
}

constexpr bool is_pow2(uint32_t n) { return n != 0u && (n & (n - 1u)) == 0u; }

// Balanced pow-2 factorization: N2 = 2^(log2N/2), N1 = N/N2.
// For our gated range (1024 < N ≤ 1M pow-2) both factors land in [32, 1024].
std::pair<uint32_t, uint32_t> pick_factorization(uint32_t N) {
    uint32_t log2N = 0u;
    while ((1u << log2N) < N) ++log2N;
    const uint32_t log2N2 = log2N / 2u;
    const uint32_t log2N1 = log2N - log2N2;
    return {1u << log2N1, 1u << log2N2};
}

ttnn::Shape make_shape(std::initializer_list<uint32_t> dims) {
    ttnn::SmallVector<uint32_t> v;
    v.reserve(dims.size());
    for (auto d : dims) v.push_back(d);
    return ttnn::Shape{v};
}

// ── Large-page reshape guard ──────────────────────────────────────────────
// On-device reshape that reduces the last dimension is a page-size-changing
// copy.  The Metal copy kernel buffers the OLD page in an L1 CB alongside
// other static allocations (transpose_rm tiles, radix-pass tiles).
// Wormhole L1 = ~1.5 MB total, but fft_two_pass/three_pass already use
// several hundred KB for their own CBs.  Empirically, a source page ≥ 128 KB
// pushes the combined static allocation over the L1 limit.  Anything above
// this threshold routes through rebank_rm (pure DRAM-to-DRAM, CB ≤ 4 KB).
//
// Threshold derivation (applies to BOTH fp32 and bf16):
//   N=32768, fp32 → page = 128 KB > 64 KB → rebanked
//   N=32768, bf16 → page =  64 KB = 64 KB → NOT rebanked (at limit)
//   N=65536, fp32 → page = 256 KB > 64 KB → rebanked  ← cascade fix
//   N=65536, bf16 → page = 128 KB > 64 KB → rebanked  ← cascade fix
//   N=2^20,  fp32 → page =   4 MB > 64 KB → rebanked (would overflow L1)
// Keeping the threshold at 64 KB ensures both fp32 and bf16 large-N cases
// avoid the multi-hundred-KB reshape CB that destabilises subsequent tests.
constexpr uint32_t kRebankThresholdBytes = 64u * 1024u;  // 64 KB

// Reshape or rebank a (B_total, N) tensor to (B_total * N/chunk, chunk).
// Uses rebank_rm when the source page exceeds the L1 safe limit.
static ttnn::Tensor reshape_or_rebank(
    const ttnn::Tensor& t,
    uint32_t rows,
    uint32_t chunk)
{
    const auto& s = t.padded_shape();
    const uint32_t N = static_cast<uint32_t>(s[-1]);
    const uint32_t elem_bytes =
        (t.dtype() == tt::tt_metal::DataType::BFLOAT16) ? 2u : 4u;
    const uint32_t src_page_bytes = N * elem_bytes;

    // Compute B_total = product of all dims except the last.
    uint32_t B_total = 1u;
    for (int d = 0; d < static_cast<int>(s.size()) - 1; ++d)
        B_total *= static_cast<uint32_t>(s[d]);

    // Large-page path: always use rebank_rm to avoid L1 overflow in the
    // Metal copy CB (CB ≥ src_page_bytes; 64+ KB overflows the twiddle/tile
    // allocations already present in fft_two_pass).
    if (src_page_bytes > kRebankThresholdBytes) {
        return ttnn::prim::rebank_rm(t, chunk);
    }

    // Small-page, single-row (B_total == 1): use rebank_rm.
    //
    // A 2D→3D metadata reshape of a single-DRAM-page tensor leaves the
    // physical page_size unchanged.  transpose_rm then reads incorrect row
    // boundaries (it uses page_size/elem_bytes as row width) → zeros / garbage.
    // rebank_rm is a true DRAM-to-DRAM copy that sets page_size = chunk*elem_bytes.
    //
    // Precondition satisfied: fft_two_pass is only called by two_pass_eligible(),
    // which asserts is_pow2(N), so rebank_rm's pow-2 last-dim requirement holds.
    if (B_total == 1u) {
        return ttnn::prim::rebank_rm(t, chunk);
    }

    // Small-page, multi-row (B_total > 1): ttnn::reshape is safe.
    //
    // Each batch row is a separate DRAM page (page_size = N*elem_bytes).
    // The 2D→2D page-shrinking reshape physically splits each row's page
    // into N/chunk smaller pages of chunk elements — a proper copy that
    // sets page_size correctly for all B_total rows.
    // rebank_rm does NOT correctly handle B_total > 1 (it assumes a
    // single contiguous DRAM page of N elements).
    return ttnn::reshape(t, make_shape({rows, chunk}));
}

std::tuple<ttnn::Tensor, ttnn::Tensor> fft_two_pass(
    const ttnn::Tensor& input_real,
    std::optional<ttnn::Tensor> input_imag,
    FFTPrecision precision,
    bool inverse) {
    // precision is currently unused — both passes route through
    // prim::fft_radix_pass which keeps its compute precision implicit
    // (fp32 input → fp32 compute, bf16 input → packed bf16).  Kept in
    // the signature for API symmetry with fft_three_pass and for the
    // future tile-quant lowering knob (commit 6+).
    (void)precision;
    const auto& in_shape = input_real.padded_shape();
    const uint32_t N = static_cast<uint32_t>(in_shape[-1]);
    uint32_t B = 1u;
    for (int d = 0; d < static_cast<int>(in_shape.size()) - 1; ++d) {
        B *= static_cast<uint32_t>(in_shape[d]);
    }

    const auto [N1, N2] = pick_factorization(N);

    // ── INVERSE PATH (commit 6c): zero-extra-dispatch IFFT via the
    //   "swap-trick".  Standard conjugate trick is
    //
    //       IFFT(X) = (1/N) · conj( FFT( conj(X) ) ).
    //
    //   Both conjugations are NEGATE-IMAG, which is an antilinear op
    //   and CANNOT be folded into a complex multiply (complex_mul has
    //   no representable element that maps z ↦ z̄).  Materialising a
    //   (-1 on imag) constant tensor would cost 8 GB at N=1G in
    //   three-pass — unacceptable.
    //
    //   The swap-trick rewrites IFFT using only LINEAR operations:
    //
    //       Let X̃ = X_im + i·X_re = i · conj(X), so
    //       FFT(X̃) = i · FFT(conj(X))
    //              ⇒ FFT(conj(X)) = -i · FFT(X̃)
    //              ⇒ IFFT(X) = (1/N) · conj(-i · FFT(X̃))
    //                        = (1/N) · ( W_im,  W_re )      where W = FFT(X̃)
    //
    //   So we (1) swap input halves → pass (X_im, X_re) to forward FFT,
    //   (2) fold the 1/N scale into the LAST radix_pass writer via
    //   output_scale (zero extra dispatch — see commit 6c kernel patch),
    //   and (3) swap output halves before returning.  Both swaps are
    //   pure C++ relabels, FREE.
    TT_FATAL(!inverse || input_imag.has_value(),
        "fft_two_pass: inverse=true requires both input_real and input_imag.");

    // ── INPUT swap (free, just relabel which tensor is real / imag).
    const ttnn::Tensor& src_real = inverse ? *input_imag : input_real;
    const std::optional<ttnn::Tensor> src_imag =
        inverse ? std::make_optional(input_real) : input_imag;

    const bool has_imag = src_imag.has_value();
    const float final_scale =
        inverse ? (1.0f / static_cast<float>(N)) : 1.0f;

    // ── Step 1: reshape input (B, N) → (B*N1, N2) then view as (B, N1, N2).
    //
    //   ALWAYS go through reshape_or_rebank (not a direct 2D→3D reshape).
    //
    //   Rationale: ttnn::reshape((B, N) → (B, N1, N2)) is a 2D→3D
    //   page-shrinking operation.  For small N this may be treated as a
    //   metadata-only view, leaving the DRAM buffer with the original
    //   large page_size.  transpose_rm then reads the wrong row boundaries
    //   (it uses page_size / elem_bytes as the row width), producing
    //   garbage output and ultimately zeros after the FFT.
    //
    //   reshape_or_rebank always produces a proper 2D (B*N1, N2) tensor
    //   with the physically-correct page_size = N2 * elem_bytes (either
    //   via rebank_rm for large pages or via a 2D ttnn::reshape for small
    //   pages — a 2D→2D reshape is well-tested and always copies correctly).
    //   The subsequent 2D→3D ttnn::reshape((B*N1, N2) → (B, N1, N2)) is
    //   then truly metadata-only (page_size unchanged), safe to do.
    std::optional<ttnn::Tensor> x_3d_i;
    ttnn::Tensor x_3d_r;
    // reshape_or_rebank produces (B*N1, N2) with correct page_size.
    x_3d_r = ttnn::reshape(reshape_or_rebank(src_real, B * N1, N2),
                           make_shape({B, N1, N2}));
    if (has_imag)
        x_3d_i = ttnn::reshape(reshape_or_rebank(*src_imag, B * N1, N2),
                               make_shape({B, N1, N2}));

    // ── Step 2: initial transpose (B, N1, N2) → (B, N2, N1).
    //   So that Pass-1 (FFT_N1) sees stride-N1 sub-samples as
    //   contiguous rows.  This is the bit-reversal-equivalent step
    //   that the earlier (commit 4) version was missing.
    auto x_t_r  = ttnn::prim::transpose_rm(x_3d_r);
    auto x_p1_r = ttnn::reshape(x_t_r, make_shape({B * N2, N1}));
    std::optional<ttnn::Tensor> x_p1_i;
    if (has_imag) {
        auto x_t_i = ttnn::prim::transpose_rm(*x_3d_i);
        x_p1_i = ttnn::reshape(x_t_i, make_shape({B * N2, N1}));
    }

    // ── Step 3: Pass-1 batched length-N1 (complex if has_imag else real)
    //   FFT + between-pass twiddle.  Row r = b·N2 + n2, so (r % twiddle_N2=N2) = n2:
    //       post-twiddle = exp(-2πi · n2 · k1 / (N1·N2))
    //                    = exp(-2πi · n2 · k1 / N)        ← Cooley–Tukey twiddle
    auto [r1, i1] = ttnn::prim::fft_radix_pass(
        x_p1_r, /*input_imag=*/x_p1_i,
        /*P=*/N1, /*twiddle_N2=*/N2);

    // ── Step 4: transpose (B, N2, N1) → (B, N1, N2) to bring n2 to
    //   the last axis ready for Pass-2.
    auto r1_3d = ttnn::reshape(r1, make_shape({B, N2, N1}));
    auto i1_3d = ttnn::reshape(i1, make_shape({B, N2, N1}));
    auto r2t = ttnn::prim::transpose_rm(r1_3d);
    auto i2t = ttnn::prim::transpose_rm(i1_3d);
    auto r2 = ttnn::reshape(r2t, make_shape({B * N1, N2}));
    auto i2 = ttnn::reshape(i2t, make_shape({B * N1, N2}));

    // ── Step 5: Pass-2 batched length-N2 complex FFT, NO twiddle
    //   (all Cooley–Tukey twiddles were absorbed into Pass-1 above).
    //   For IFFT: the 1/N scale is folded INTO this writer via
    //   output_scale (single compile-time kernel branch, runtime float
    //   arg).  Zero extra dispatch vs forward FFT.
    auto [r3, i3] = ttnn::prim::fft_radix_pass(
        r2, /*input_imag=*/i2,
        /*P=*/N2, /*twiddle_N2=*/0u, /*stride=*/1u,
        /*output_scale=*/final_scale);

    // ── Step 6: final transpose (B, N1, N2) → (B, N2, N1) to put
    //   the output in natural-K order under flat reshape.
    //   Recall: algorithm produces X[K = k2·N1 + k1] at position
    //   (b, k1, k2) of the (B, N1, N2) post-Pass-2 tensor.  After this
    //   transpose, element (b, k2, k1) lives at flat (b·N + k2·N1 + k1)
    //   = (b·N + K), i.e. natural K-ordered output.
    auto r3_3d = ttnn::reshape(r3, make_shape({B, N1, N2}));
    auto i3_3d = ttnn::reshape(i3, make_shape({B, N1, N2}));
    auto r4t = ttnn::prim::transpose_rm(r3_3d);
    auto i4t = ttnn::prim::transpose_rm(i3_3d);

    // Final reshape (B, N2, N1) → (B, N).  For large N the destination
    // page = N*elem_bytes can exceed L1 and crash reshape_rm.  Use
    // rebank_rm_merge (CB = 2*N1*elem_bytes, tiny) in that case.
    const uint32_t elem_bytes_fp =
        (input_real.dtype() == tt::tt_metal::DataType::BFLOAT16) ? 2u : 4u;
    const bool need_merge = ((uint64_t)N * elem_bytes_fp > kRebankThresholdBytes);

    auto merge_or_reshape = [&](const ttnn::Tensor& t3d) -> ttnn::Tensor {
        if (need_merge) {
            auto t2d = ttnn::reshape(t3d, make_shape({B * N2, N1}));
            return ttnn::prim::rebank_rm_merge(t2d, N2);
        }
        return ttnn::reshape(t3d, in_shape);
    };
    auto out_r = merge_or_reshape(r4t);
    auto out_i = merge_or_reshape(i4t);

    // ── OUTPUT swap (free) — completes the swap-trick.  After the
    //   forward FFT chain with scale=1/N we have (W_re/N, W_im/N);
    //   the IFFT result is (W_im/N, W_re/N), i.e. swap halves.
    if (inverse) {
        return {std::move(out_i), std::move(out_r)};
    }
    return {std::move(out_r), std::move(out_i)};
}

// Compute next power-of-2 ≥ v without including bluestein_host.hpp
// (which would create a circular dependency through fft.hpp).
constexpr uint32_t next_pow2_local(uint32_t v) {
    if (v <= 1u) return 1u;
    --v;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return ++v;
}

// Bluestein padded length M = next_pow2(2N - 1).
constexpr uint32_t bluestein_M_local(uint32_t N) {
    return next_pow2_local(2u * N - 1u);
}

// ── L1 twiddle-table hard limit (WH B0) ──────────────────────────────────────
// Both fft_radix_pass (two-pass) and apply_twiddles_xl (three-pass) store a
// complex twiddle table of total size N × 2 × sizeof(dtype) bytes in L1.
// Wormhole L1 per core = 1,499,136 B (~1.46 MB).
//
//   fp32: N × 8 ≤ 1,499,136  →  N ≤ 187,392  →  max pow-2 = 2^17 = 131,072
//   bf16: N × 4 ≤ 1,499,136  →  N ≤ 374,784  →  max pow-2 = 2^18 = 262,144
//
// Measured on WH B0 with find_n_limit.py:
//   N = 2^17 fp32  PASS  (twiddle = 1.0 MB < 1.46 MB)
//   N = 2^18 fp32  FAIL  (twiddle = 2.1 MB > 1.46 MB)  — two-pass AND three-pass
//   N = 2^18 bf16  PASS  (twiddle = 1.0 MB < 1.46 MB)
//   N = 2^19 bf16  FAIL  (twiddle = 2.1 MB > 1.46 MB)
//
// Gap: N ∈ [2^18, 2^20] (fp32) and N ∈ [2^19, 2^20] (bf16) cannot be
// handled by two-pass OR three-pass on WH B0.  The apply_twiddles_xl kernel
// switches to on-the-fly (streaming) twiddle computation only for N > 2^20,
// so three-pass works for AGGRESSIVE large-N but not the gap range.
//
// The routing below preserves the original two-pass range (≤ 2^20) and
// three-pass range (> 2^20).  Calls within the gap throw TT_THROW from the
// kernel; this is the correct behaviour given the hardware constraint.

bool two_pass_eligible(const ttnn::Tensor& input_real) {
    if (!native_path_enabled()) return false;
    const auto& shape = input_real.padded_shape();
    if (shape.size() < 1) return false;
    const uint32_t N = static_cast<uint32_t>(shape[-1]);
    uint32_t B = 1u;
    for (int d = 0; d < static_cast<int>(shape.size()) - 1; ++d) {
        B *= static_cast<uint32_t>(shape[d]);
    }
    const auto dt = input_real.dtype();
    const bool dtype_ok =
        dt == tt::tt_metal::DataType::FLOAT32 ||
        dt == tt::tt_metal::DataType::BFLOAT16;
    const bool layout_ok =
        input_real.layout() == tt::tt_metal::Layout::ROW_MAJOR;
    return dtype_ok && layout_ok &&
           is_pow2(N) && N > 1024u && N <= (1u << 20) &&
           is_pow2(B) && B >= 1u;
}

// Three-pass eligible: pow-2 N in (1M, 1G].
// apply_twiddles_xl uses streaming twiddle computation for N > 2^20,
// so no L1 twiddle table overflow occurs for AGGRESSIVE large-N cases.
bool three_pass_eligible(const ttnn::Tensor& t) {
    if (!native_path_enabled()) return false;
    const auto& shape = t.padded_shape();
    if (shape.size() < 1) return false;
    const uint32_t N = static_cast<uint32_t>(shape[-1]);
    const auto dt = t.dtype();
    return (dt == tt::tt_metal::DataType::FLOAT32 ||
            dt == tt::tt_metal::DataType::BFLOAT16) &&
           t.layout() == tt::tt_metal::Layout::ROW_MAJOR &&
           is_pow2(N) && N > (1u << 20) && N <= (1u << 30);
}

// Bluestein eligible: non-pow-2 N where M = next_pow2(2N-1) ≤ 2^30.
// Covers N up to ~715M for the XL range once three-pass inner FFTs are
// available.
bool bluestein_eligible(const ttnn::Tensor& t) {
    if (!native_path_enabled()) return false;
    const auto& shape = t.padded_shape();
    if (shape.size() < 1) return false;
    const uint32_t N = static_cast<uint32_t>(shape[-1]);
    if (is_pow2(N)) return false;      // pow-2 handled by stockham / pass paths
    const uint32_t M = bluestein_M_local(N);
    const auto dt = t.dtype();
    return (dt == tt::tt_metal::DataType::FLOAT32 ||
            dt == tt::tt_metal::DataType::BFLOAT16) &&
           t.layout() == tt::tt_metal::Layout::ROW_MAJOR &&
           M <= (1u << 30);
}

// ───────────────────────────────────────────────────────────────────────
// Small-N IFFT (pow-2 N ≤ 1024, complex input): swap trick via fft_radix_pass.
//
// The SingleTileStockhamFactory / BatchedStockhamFactory kernels implement
// only the forward DFT.  For IFFT at small N we apply the same swap trick
// that the two-pass / three-pass composites use, but delegate to
// fft_radix_pass (which supports output_scale to fold in the 1/N factor):
//
//   IFFT(X)[n] = (1/N) · (W_im[n] + i · W_re[n])
//   where W = FFT(X̃)  and  X̃ = X_im + i · X_re.
//
// Steps:
//   (1) Swap inputs: pass (X_im, X_re) as (real, imag) to fft_radix_pass.
//   (2) fft_radix_pass computes FFT_N × (1/N) via output_scale — zero
//       extra dispatch vs a forward FFT call.
//   (3) Swap outputs: return (W_im, W_re).  No negation required.
//
// Precondition: spectrum_imag must be provided (complex spectrum).
static std::tuple<ttnn::Tensor, ttnn::Tensor> small_pow2_ifft(
    const ttnn::Tensor&  spectrum_real,
    const ttnn::Tensor&  spectrum_imag,
    uint32_t             N,
    FFTPrecision         /*precision*/)
{
    const auto& sh  = spectrum_real.padded_shape();
    uint32_t    B   = 1u;
    for (int d = 0; d < static_cast<int>(sh.size()) - 1; ++d)
        B *= static_cast<uint32_t>(sh[d]);

    const float inv_n = 1.0f / static_cast<float>(N);

    // Flatten to (B, N) for fft_radix_pass.
    auto re_2d = (sh.size() == 2u)
        ? spectrum_real
        : ttnn::reshape(spectrum_real,  make_shape({B, N}));
    auto im_2d = (sh.size() == 2u)
        ? spectrum_imag
        : ttnn::reshape(spectrum_imag,  make_shape({B, N}));

    // Swap inputs: X̃ = X_im + i·X_re → W = FFT(X̃) × (1/N)
    auto [W_re, W_im] = ttnn::prim::fft_radix_pass(
        im_2d,                            // real part  = X_im
        std::make_optional(re_2d),        // imag part  = X_re
        /*P=*/N, /*twiddle_N2=*/0u, /*stride=*/1u,
        /*output_scale=*/inv_n);

    // Swap outputs: IFFT(X) = (W_im, W_re); 1/N already applied above.
    auto out_re = (sh.size() == 2u) ? std::move(W_im) : ttnn::reshape(W_im, sh);
    auto out_im = (sh.size() == 2u) ? std::move(W_re) : ttnn::reshape(W_re, sh);
    return {std::move(out_re), std::move(out_im)};
}

// ───────────────────────────────────────────────────────────────────────
// Three-pass Cooley–Tukey composite (commit 5, corrected commit 5c)
//
// ⚠ API NOTES:
//   (1) INPUT pre-shape: fft_three_pass takes its input ALREADY PRE-SHAPED
//       as (B·N1·N2, N3) [last dim = N3 ≤ 1024], NOT as (B, N).  This is
//       because the (B, N) → (B·N1·N2, N3) reshape requires moving an
//       N-element row through a CB per core, blowing L1 for N > ~256K.
//       Caller does `torch.view(B·N1·N2, N3)` on host (metadata-only)
//       BEFORE `ttnn.from_torch` so the device buffer is allocated with
//       small page_size from the start.
//   (2) OUTPUT shape change (commit 5c): output is now (B·N3, N2, N1)
//       instead of (B·N1, N2, N3).  Caller's `to_torch().reshape(B, N)`
//       on host still gives natural-order X[k] because the (N3, N2, N1)
//       dim layout encodes K = k3·N1·N2 + k2·N1 + k1 = natural flat K.
//       The OLD shape was returning index-permuted (wrong-order) data
//       due to the underlying algorithmic bug (see ALGORITHM section).
//
//   TODO (commit 7): write an L1-friendly DRAM→DRAM rebank kernel that
//   handles the page-size change in chunks, so the public `fft()` API
//   can transparently route (B, N) inputs into the three-pass composite.
//
// ── ALGORITHM ───────────────────────────────────────────────────────────
// For pow-2 N with 2^20 < N ≤ 2^30, factor N = N1 · N2 · N3 (each pow-2
// in [32, 1024]).  We use the standard mixed-radix DIT decomposition.
//
//   Input packing  : n = n1·N2·N3 + n2·N3 + n3   (n1 OUTER, n3 INNER)
//   Output packing : K = k3·N1·N2 + k2·N1 + k1   (k1 INNER, k3 OUTER)
//
// Crucially, K is the REVERSED-digit packing (k_i factor positions
// swapped relative to n_i).  Empirically (and provably) the natural-K
// packing K = k1·N2·N3 + k2·N3 + k3 has a non-integer phase term
// (n2·k2·N3/(N1·N2) is fractional when N3/(N1·N2) is not integer), so
// it does NOT admit a clean Cooley-Tukey decomposition for asymmetric
// (N1, N2, N3).  The reversed packing makes every cross-term integer-
// vanishing or assignable to a clean FFT/twiddle factor:
//
//   n·K/N  ≡  n1·k1/N1
//          + n2·k2/N2  +  n2·k1/(N1·N2)
//          + n3·k3/N3  +  n3·k2/(N2·N3)  +  n3·k1/N        (mod 1)
//
// Three FFT stages with assignable twiddles:
//
//   Stage 1: FFT_N1 over n1 (→ k1).
//   Twiddle-1 (post Stage 1): exp(-2πi · (n2·N3 + n3) · k1 / N)
//     fuses the n2·k1/(N1·N2) and n3·k1/N cross-terms.
//   Stage 2: FFT_N2 over n2 (→ k2).
//   Twiddle-2 (post Stage 2, fused into Stage 2's post-twiddle):
//     exp(-2πi · n3 · k2 / (N2·N3))
//   Stage 3: FFT_N3 over n3 (→ k3).
//
// Output naturally lands at (B, k1, k2, k3) after Stage 3.  A final
// transpose chain reverses the last 3 dims → (B, N3, N2, N1) so that
// `.reshape(B, N)` on host yields X[K] at flat K.
//
// ── DISPATCH CHAIN (8 device ops) ──────────────────────────────────────
//
//   Initial rearrangement (input is (B·N1·N2, N3) with n1 OUTER):
//     1. reshape (B·N1·N2, N3) → (B, N1, N2·N3)        [page: N3·elem
//                                                        → N2·N3·elem]
//     2. transpose_rm        → (B, N2·N3, N1)          [×2 r,i]
//     3. reshape             → (B·N2·N3, N1)           [free]
//
//   Stage 1 + Twiddle-1:
//     4. fft_radix_pass(P=N1, twiddle_N2=0)            [pure FFT_N1]
//     5. apply_twiddles_xl(P=N1, big_mod=N2·N3,
//                          full_N=N)                   [twiddle-1]
//
//   Bring n2 to inner (was at position 1 of (B, N2, N3, k1)):
//     6. reshape → (B, N2, N3·N1)                      [free]
//     7. transpose_rm → (B, N3·N1, N2)                 [×2 r,i]
//     8. reshape → (B·N3·N1, N2)                       [free]
//
//   Stage 2 + Twiddle-2 (FUSED in fft_radix_pass post-twiddle):
//     9. fft_radix_pass(P=N2, twiddle_N2=N3,
//                       stride=N1)                     [FFT_N2 + tw-2]
//
//   Bring n3 to inner (was at position 1 of (B, N3, k1, k2)):
//    10. reshape → (B, N3, N1·N2)                      [free]
//    11. transpose_rm → (B, N1·N2, N3)                 [×2 r,i]
//    12. reshape → (B·N1·N2, N3)                       [free]
//
//   Stage 3 (no twiddle):
//    13. fft_radix_pass(P=N3, twiddle_N2=0)            [pure FFT_N3]
//
//   Final dim-reverse to natural-K order:
//    14. reshape → (B, N1·N2, N3)                      [free]
//    15. transpose_rm → (B, N3, N1·N2)                 [×2 r,i]
//    16. reshape → (B, N3, N1, N2)                     [page change]
//    17. transpose_rm → (B, N3, N2, N1)                [×2 r,i, small]
//
// Above 2^30, we'd need a 4-pass or Bluestein composite (commit 6).
// ───────────────────────────────────────────────────────────────────────

// Max-N3 then balanced N1/N2 split.  Both N1, N2, N3 ∈ [32, 1024], pow-2.
//   N3 = min(1024, N / 32^2)   ← cap by tile-size on innermost
//   then split remaining log2 between N1 and N2 (N1 gets ceil-half).
std::tuple<uint32_t, uint32_t, uint32_t> pick_three_factorization(uint32_t N) {
    uint32_t log2N = 0u;
    while ((1u << log2N) < N) ++log2N;
    TT_FATAL((1u << log2N) == N,
        "fft_three_pass: N must be a power of two (got {}).", N);
    TT_FATAL(log2N >= 15u && log2N <= 30u,
        "fft_three_pass: N must be in [2^15, 2^30] (got 2^{}).", log2N);

    // Cap log2(N3) at 10 (= 1024, the per-row FFT length limit); also
    // leave ≥ 10 bits for N1+N2 split (= 32 · 32 minimum).
    uint32_t log2_N3 = 10u;
    if (log2N - log2_N3 < 10u) {
        // Pathological tiny case (log2N < 20).  Shouldn't happen since
        // routing kicks in at log2N > 20, but be safe.
        log2_N3 = (log2N >= 10u) ? (log2N - 10u) : 5u;
    }
    const uint32_t log2_rest = log2N - log2_N3;
    const uint32_t log2_N1 = (log2_rest + 1u) / 2u;   // ceil half → N1
    const uint32_t log2_N2 = log2_rest - log2_N1;
    TT_FATAL(log2_N1 >= 5u && log2_N1 <= 10u &&
             log2_N2 >= 5u && log2_N2 <= 10u &&
             log2_N3 >= 5u && log2_N3 <= 10u,
        "fft_three_pass: N=2^{} factorization N1=2^{} N2=2^{} N3=2^{} "
        "out of supported [32, 1024] range.",
        log2N, log2_N1, log2_N2, log2_N3);
    return {1u << log2_N1, 1u << log2_N2, 1u << log2_N3};
}

}  // namespace

// ── Three-pass auto-reshape wrapper (P2) ─────────────────────────────────
// Makes large pow-2 FFTs transparent: caller passes (B, N) and gets (B, N)
// back, matching the cuFFT API contract.  Internally we pre-shape the input
// to (B·N1·N2, N3) (a free ROW_MAJOR view) before calling fft_three_pass,
// then flatten the (B·N3, N2, N1) factored output back to (B, N).
//
// Both reshapes are zero-copy for ROW_MAJOR contiguous tensors.
static std::tuple<ttnn::Tensor, ttnn::Tensor> fft_three_pass_auto(
    const ttnn::Tensor& input_real,
    std::optional<ttnn::Tensor> input_imag,
    FFTPrecision precision,
    bool inverse)
{
    const auto& shape = input_real.padded_shape();
    const uint32_t N  = static_cast<uint32_t>(shape[-1]);
    uint32_t B = 1u;
    for (int d = 0; d < static_cast<int>(shape.size()) - 1; ++d)
        B *= static_cast<uint32_t>(shape[d]);

    const auto [N1, N2, N3] = pick_three_factorization(N);

    // Pre-shape: (B, N) → (B·N1·N2, N3).
    // When the source page (N × elem_bytes) exceeds the L1 safe limit, the
    // on-device reshape copies through an L1 CB that overflows.  Use
    // reshape_or_rebank which transparently substitutes a DRAM-to-DRAM
    // rebank_rm (CB = N3 × elem_bytes ≤ 4 KB, never overflows L1).
    const uint32_t rows = B * N1 * N2;
    auto re_pre = reshape_or_rebank(input_real, rows, N3);
    std::optional<ttnn::Tensor> im_pre;
    if (input_imag.has_value())
        im_pre = reshape_or_rebank(*input_imag, rows, N3);

    auto [out_re, out_im] = fft_three_pass(re_pre, im_pre, N, precision, inverse);

    // fft_three_pass output is (B·N3, N2, N1) — flatten to (B, N).
    // For large N the destination page = N*elem_bytes overflows L1; use
    // rebank_rm_merge (CB = 2*N1*elem_bytes, tiny) in that case.
    const uint32_t elem_bytes_tp =
        (input_real.dtype() == tt::tt_metal::DataType::BFLOAT16) ? 2u : 4u;
    const bool need_merge_3p = ((uint64_t)N * elem_bytes_tp > kRebankThresholdBytes);

    if (need_merge_3p) {
        // CPM = N2 * N3 (both pow-2 → product is pow-2).
        // 3D→2D metadata flatten (page = N1*elem unchanged) then merge-rebank.
        const uint32_t cpm = N2 * N3;
        auto out_re_2d = ttnn::reshape(out_re, make_shape({B * N3 * N2, N1}));
        auto out_im_2d = ttnn::reshape(out_im, make_shape({B * N3 * N2, N1}));
        return {ttnn::prim::rebank_rm_merge(out_re_2d, cpm),
                ttnn::prim::rebank_rm_merge(out_im_2d, cpm)};
    }
    return {ttnn::reshape(out_re, make_shape({B, N})),
            ttnn::reshape(out_im, make_shape({B, N}))};
}

// ── Bluestein batch-dim flatten helper ───────────────────────────────────
// bluestein_fft expects 2-D (B, N).  Flatten multi-dim inputs before call.
static std::tuple<ttnn::Tensor, ttnn::Tensor> bluestein_dispatch(
    const ttnn::Tensor& input_real,
    std::optional<ttnn::Tensor> input_imag,
    FFTPrecision precision,
    bool inverse)
{
    const auto& shape = input_real.padded_shape();
    const uint32_t N  = static_cast<uint32_t>(shape[-1]);
    uint32_t B = 1u;
    for (int d = 0; d < static_cast<int>(shape.size()) - 1; ++d)
        B *= static_cast<uint32_t>(shape[d]);

    // Flatten to (B, N) if needed
    auto re_2d = (shape.size() == 2u)
        ? input_real
        : ttnn::reshape(input_real, make_shape({B, N}));
    std::optional<ttnn::Tensor> im_2d;
    if (input_imag.has_value())
        im_2d = (shape.size() == 2u)
            ? *input_imag
            : ttnn::reshape(*input_imag, make_shape({B, N}));

    auto [out_re, out_im] = bluestein_fft(re_2d, im_2d, N, precision, inverse);

    // Restore original leading dims if they were multi-dim
    if (shape.size() != 2u) {
        ttnn::SmallVector<uint32_t> orig_dims;
        for (int d = 0; d < static_cast<int>(shape.size()) - 1; ++d)
            orig_dims.push_back(static_cast<uint32_t>(shape[d]));
        orig_dims.push_back(N);
        ttnn::Shape orig_shape{orig_dims};
        out_re = ttnn::reshape(out_re, orig_shape);
        out_im = ttnn::reshape(out_im, orig_shape);
    }
    return {std::move(out_re), std::move(out_im)};
}

// ────────────────────────────────────────────────────────────────────
// Public entrypoint — caller-visible.  Input is REQUIRED to be pre-
// shaped as (B·N1·N2, N3) [last dim = N3 ≤ 1024]; the (B, N) → factored
// reshape would otherwise blow L1 (commit 7 will add a rebank kernel
// to lift this restriction).  Output is returned in the factored shape
// (B·N3, N2, N1) — caller does `to_torch().reshape(B, N)` on host to
// recover natural-order X[k] (the (N3, N2, N1) dim order encodes
// K = k3·N1·N2 + k2·N1 + k1, which IS the natural flat K).
// ────────────────────────────────────────────────────────────────────
std::tuple<ttnn::Tensor, ttnn::Tensor> fft_three_pass(
    const ttnn::Tensor& input_real,
    std::optional<ttnn::Tensor> input_imag,
    uint32_t full_N,
    FFTPrecision precision,
    bool inverse) {
    (void)precision;  // see fft_two_pass note.
    // Same swap-trick as fft_two_pass.  See its long comment for
    // derivation.  The 1/N scale is folded into the Stage-3 (last)
    // fft_radix_pass writer via output_scale.
    TT_FATAL(!inverse || input_imag.has_value(),
        "fft_three_pass: inverse=true requires both input_real and input_imag.");
    const ttnn::Tensor& src_real = inverse ? *input_imag : input_real;
    const std::optional<ttnn::Tensor> src_imag =
        inverse ? std::make_optional(input_real) : input_imag;
    const float final_scale =
        inverse ? (1.0f / static_cast<float>(full_N)) : 1.0f;

    const auto& in_shape = src_real.padded_shape();
    TT_FATAL(in_shape.size() >= 2,
        "fft_three_pass: pre-shaped input must be ≥2-D, e.g. (M, N3). Got {}-D.",
        in_shape.size());
    const uint32_t P_in = static_cast<uint32_t>(in_shape[-1]);
    // Sum total rows = product of all dims except last.  Caller may pass
    // either flat 2-D (B·N1·N2, N3) or N-D (..., N1·N2, N3) — we treat
    // everything except the last dim as a contiguous row-stream of length
    // B·N1·N2 and derive B from there.
    uint32_t M_in = 1u;
    for (int d = 0; d < static_cast<int>(in_shape.size()) - 1; ++d) {
        M_in *= static_cast<uint32_t>(in_shape[d]);
    }

    const auto [N1, N2, N3] = pick_three_factorization(full_N);
    TT_FATAL(P_in == N3,
        "fft_three_pass: pre-shaped input last dim must be N3={} for full_N={} "
        "(N1={}, N2={}, N3={}); got {}.",
        N3, full_N, N1, N2, N3, P_in);
    TT_FATAL(M_in % (N1 * N2) == 0u,
        "fft_three_pass: total rows {} must be a multiple of N1·N2={} for "
        "full_N={} (N1={}, N2={}, N3={}).",
        M_in, N1 * N2, full_N, N1, N2, N3);
    const uint32_t B = M_in / (N1 * N2);

    // Complex-input mode (commit 6a, for Bluestein): when input_imag is
    // supplied it gets the SAME pre-rearrangement chain as input_real
    // (reshape → transpose_rm → reshape), adds 1 transpose_rm dispatch
    // on the input.  Pass-1 then sees a true complex input.  The rest
    // of the pipeline already handles complex (Pass-2 and Pass-3 both
    // do) so no further changes are required.
    //
    // commit 6c: for inverse=true, src_real/src_imag are the post-swap
    // labels (X_im, X_re) — has_imag is therefore always true and the
    // imag path is unconditionally taken.
    const bool has_imag = src_imag.has_value();
    if (has_imag) {
        const auto& im_shape = src_imag->padded_shape();
        TT_FATAL(im_shape == in_shape,
            "fft_three_pass: input_imag shape must match input_real shape.");
    }

    // ── Dtype helpers (used for large-N path selection below).
    const uint32_t elem_bytes_3p =
        (src_real.dtype() == tt::tt_metal::DataType::BFLOAT16) ? 2u : 4u;
    // L1 safe limit: reshape CB = 2 × page; overflow when page > 750 KB.
    constexpr uint64_t kPageOverflowBytes = 750u * 1024u;

    // ── Initial rearrangement (input n1 OUTER → n1 to LAST axis).
    //   Input (B·N1·N2, N3) is row-major with n = n1·N2·N3 + n2·N3 + n3.
    //   Normal path: view as (B, N1, N2·N3) [page = N2·N3·elem], then
    //     transpose_rm → (B, N2·N3, N1).
    //   Large-N path (N ≥ 2^26, page = N2·N3·elem > 750 KB → CB > 1.5 MB):
    //     Use rebank_rm_merge(src, N2) to merge N2 rows at CB = 2·N3·elem (tiny),
    //     producing (B·N1, N2·N3).  A metadata-only 3-D view gives (B, N1, N2·N3).
    //     The subsequent transpose_rm and flatten are unchanged.
    const bool large_page_in = ((uint64_t)N2 * N3 * elem_bytes_3p > kPageOverflowBytes);
    auto rearrange_3d = [&](const ttnn::Tensor& src) -> ttnn::Tensor {
        if (large_page_in) {
            auto merged = ttnn::prim::rebank_rm_merge(src, N2);         // (B·N1, N2·N3)
            return ttnn::reshape(merged, make_shape({B, N1, N2 * N3})); // metadata-only
        }
        return ttnn::reshape(src, make_shape({B, N1, N2 * N3}));
    };
    auto x_3d_r = rearrange_3d(src_real);
    auto x_t_r  = ttnn::prim::transpose_rm(x_3d_r);                   // (B, N2·N3, N1)
    auto x_p1_r = ttnn::reshape(x_t_r, make_shape({B * N2 * N3, N1}));
    std::optional<ttnn::Tensor> x_p1_i;
    if (has_imag) {
        auto x_3d_i = rearrange_3d(*src_imag);
        auto x_t_i  = ttnn::prim::transpose_rm(x_3d_i);               // (B, N2·N3, N1)
        x_p1_i = ttnn::reshape(x_t_i, make_shape({B * N2 * N3, N1}));
    }

    // ── Stage 1: pure FFT_N1 over the (now-inner) n1 axis.
    auto [r1, i1] = ttnn::prim::fft_radix_pass(
        x_p1_r, /*input_imag=*/x_p1_i,
        /*P=*/N1, /*twiddle_N2=*/0u);

    // ── Twiddle-1: exp(-2πi · (n2·N3 + n3) · k1 / N).
    //   Row r = b·N2·N3 + n2·N3 + n3, so (r % (N2·N3)) = n2·N3 + n3.
    //   apply_twiddles_xl with big_modulus=N2·N3 picks exactly that.
    //   Combines the n2·k1/(N1·N2) and n3·k1/N cross-terms of the
    //   Cooley-Tukey decomposition into a single dispatch.
    auto [r1t, i1t] = ttnn::prim::apply_twiddles_xl(
        r1, i1, /*P=*/N1, /*big_modulus=*/N2 * N3, /*full_N=*/full_N);

    // ── Bring n2 to inner for Stage 2.
    //   Logical (B, N2, N3, k1) → (B, N3, k1, N2).
    //   Input  (B·N2·N3, N1).  Output (B·N3·N1, N2).
    //
    //   Normal path (N3·N1·elem ≤ 750 KB):
    //     reshape (B·N2·N3, N1) → (B, N2, N3·N1)  [dest page = N3·N1·e, small]
    //     transpose_rm → (B, N3·N1, N2)             [page = N2·e, tiny]
    //     reshape → (B·N3·N1, N2)                   [metadata]
    //
    //   Large-N path (N3·N1·elem > 750 KB → page-growing reshape CB > 1.5 MB):
    //     Use rebank_rm_merge to merge N3 consecutive rows (CB = 2·N1·e = 8 KB),
    //     avoiding any large-page reshape:
    //       (1) rebank_rm_merge(t, N3) → (B·N2, N3·N1)   [CB = 8 KB]
    //       (2) reshape → (B, N2, N3·N1)                  [metadata]
    //       (3) transpose_rm → (B, N3·N1, N2)             [CB = 8 KB, page=N2·e]
    //       (4) reshape → (B·N3·N1, N2)                   [metadata]
    //     Step (1) is correct because input rows are ordered n2-major, n3-minor:
    //     N3 consecutive rows [b·N2·N3+n2·N3, …, b·N2·N3+n2·N3+N3-1] all share
    //     the same (b, n2) and together form the (n3=0..N3-1) slice, which is
    //     exactly what rebank_rm_merge concatenates into one output row. ✓
    const bool large_intermed = ((uint64_t)N3 * N1 * elem_bytes_3p > kPageOverflowBytes);
    auto bring_n2_inner = [&](const ttnn::Tensor& t) -> ttnn::Tensor {
        if (large_intermed) {
            // Step (1): merge N3 consecutive rows.  Input (B·N2·N3, N1).
            // CB = 2·N1·elem = 8 KB.  N3 is always pow-2 (=1024).
            auto merged = ttnn::prim::rebank_rm_merge(t, N3);           // (B·N2, N3·N1)
            auto t3d    = ttnn::reshape(merged, make_shape({B, N2, N3 * N1}));  // metadata
            auto tt     = ttnn::prim::transpose_rm(t3d);                // (B, N3·N1, N2)
            return ttnn::reshape(tt, make_shape({B * N3 * N1, N2}));   // metadata
        }
        auto t3d = ttnn::reshape(t, make_shape({B, N2, N3 * N1}));
        auto tt  = ttnn::prim::transpose_rm(t3d);                       // (B, N3·N1, N2)
        return ttnn::reshape(tt, make_shape({B * N3 * N1, N2}));
    };
    auto r2p = bring_n2_inner(r1t);
    auto i2p = bring_n2_inner(i1t);

    // ── Stage 2 + Twiddle-2 fused: FFT_N2 + post-twiddle
    //       exp(-2πi · n3 · k2 / (N2·N3)).
    //   Row r' = b·N3·N1 + n3·N1 + k1.  (r' / stride=N1) % twiddle_N2=N3
    //   = (b·N3 + n3) % N3 = n3.  P·twiddle_N2 = N2·N3.  ✓
    auto [r2, i2] = ttnn::prim::fft_radix_pass(
        r2p, /*input_imag=*/i2p,
        /*P=*/N2, /*twiddle_N2=*/N3, /*stride=*/N1);

    // ── Bring n3 to inner for Stage 3.
    //   Logical (B, N3, k1, k2) → (B, k1, k2, N3).
    //   View as (B, N3, N1·N2) [merge last two — page change], then
    //   transpose_rm → (B, N1·N2, N3).
    auto r3_3d = ttnn::reshape(r2, make_shape({B, N3, N1 * N2}));
    auto i3_3d = ttnn::reshape(i2, make_shape({B, N3, N1 * N2}));
    auto r3t   = ttnn::prim::transpose_rm(r3_3d);                     // (B, N1·N2, N3)
    auto i3t   = ttnn::prim::transpose_rm(i3_3d);
    auto r3p   = ttnn::reshape(r3t, make_shape({B * N1 * N2, N3}));
    auto i3p   = ttnn::reshape(i3t, make_shape({B * N1 * N2, N3}));

    // ── Stage 3: pure FFT_N3.
    //   For IFFT (commit 6c): fold the 1/full_N scale into THIS writer
    //   via output_scale.  Zero extra dispatch — the writer's element
    //   loop multiplies every STATE element by 1/full_N in-place.
    auto [r3, i3] = ttnn::prim::fft_radix_pass(
        r3p, /*input_imag=*/i3p,
        /*P=*/N3, /*twiddle_N2=*/0u, /*stride=*/1u,
        /*output_scale=*/final_scale);

    // ── FINAL rearrangement (k1, k2, k3) → (k3, k2, k1) so that
    //   `.reshape(B, N)` on host gives natural-order X[K] at flat K.
    //
    //   After Stage 3 we have (B·N1·N2, N3) ≡ (B, k1, k2, k3).
    //   Target: (B, N3, N2, N1) ≡ (B, k3, k2, k1).
    //
    //   Normal path (N1·N2·elem ≤ 375 KB):
    //     (a) view → (B, N1·N2, N3)         [page=N3·e, tiny]
    //     (b) transpose_rm → (B, N3, N1·N2) [CB=8 KB; page_out=N1·N2·e]
    //     (c) view → (B, N3, N1, N2)        [page-shrink CB=4×N1·N2·e ≤ 1 MB]
    //     (d) transpose_rm → (B, N3, N2, N1)[CB=8 KB]
    //
    //   Large-N path (N1·N2·elem > 375 KB, e.g. N=2^27 → 512 KB → CB≈2 MB):
    //     The page-shrink reshape in step (c) would overflow L1.
    //     Use rebank_rm to split (B·N3, N1·N2) into (B·N3·N1, N2) first:
    //       (a) view → (B, N1·N2, N3)         [tiny]
    //       (b) transpose_rm → (B, N3, N1·N2) [CB=8 KB]
    //       (c) view → (B·N3, N1·N2)          [metadata, same large page]
    //       (d) rebank_rm(., N2) → (B·N3·N1, N2)  [CB=2·N2·e ≤ 2 KB]
    //       (e) view → (B·N3, N1, N2)             [metadata]
    //       (f) transpose_rm → (B·N3, N2, N1)     [CB=8 KB]
    //       (g) view → (B, N3, N2, N1)            [metadata]
    //     N2 is always pow-2 (from pick_three_factorization).  ✓
    constexpr uint64_t kSrcPageOverflowBytes = 375u * 1024u;
    const bool large_n1n2 = ((uint64_t)N1 * N2 * elem_bytes_3p > kSrcPageOverflowBytes);
    auto final_rearrange = [&](const ttnn::Tensor& src) -> ttnn::Tensor {
        auto t3d = ttnn::reshape(src, make_shape({B, N1 * N2, N3}));   // (B, N1·N2, N3) tiny page
        auto tt1  = ttnn::prim::transpose_rm(t3d);                      // (B, N3, N1·N2)
        if (large_n1n2) {
            auto tt1_2d = ttnn::reshape(tt1, make_shape({B * N3, N1 * N2}));  // metadata
            auto rb     = ttnn::prim::rebank_rm(tt1_2d, N2);                  // (B·N3·N1, N2)
            auto t4d    = ttnn::reshape(rb, make_shape({B * N3, N1, N2}));    // metadata
            auto tt2    = ttnn::prim::transpose_rm(t4d);                       // (B·N3, N2, N1)
            return ttnn::reshape(tt2, make_shape({B, N3, N2, N1}));           // metadata
        }
        auto t4d  = ttnn::reshape(tt1, make_shape({B, N3, N1, N2}));   // page-shrink CB≤1 MB
        auto tout = ttnn::prim::transpose_rm(t4d);                      // (B, N3, N2, N1)
        return tout;
    };
    auto r_out = final_rearrange(r3);
    auto i_out = final_rearrange(i3);

    // ── OUTPUT swap (free) — completes the swap-trick for IFFT.
    if (inverse) {
        return {std::move(i_out), std::move(r_out)};
    }
    return {std::move(r_out), std::move(i_out)};
}

// Public real-input wrapper (preserves the original commit-5 signature).
// The complex-input form is the lower-level overload above and is
// what Bluestein (commit 6d) drives.  Inverse mode requires complex
// input (swap-trick), so we don't expose it here.
std::tuple<ttnn::Tensor, ttnn::Tensor> fft_three_pass(
    const ttnn::Tensor& input_real,
    uint32_t full_N,
    FFTPrecision precision) {
    return fft_three_pass(input_real, /*input_imag=*/std::nullopt,
                          full_N, precision, /*inverse=*/false);
}

std::tuple<ttnn::Tensor, ttnn::Tensor> fft(
    const ttnn::Tensor& input_real, FFTPrecision precision) {
    // Unified cuFFT-style router — any N, any dtype, no env var needed.
    //
    //   N ≤ 1024,  pow-2           → prim::fft  → SingleTile/BatchedStockham
    //   1024 < N ≤ 2^20, pow-2     → fft_two_pass   (3 transposes + 2 passes)
    //   2^20 < N ≤ 2^30, pow-2     → fft_three_pass_auto (auto-reshape)
    //   non-pow-2, M ≤ 2^30        → bluestein_dispatch (7-op device chain)

    // Accept 1D (N,) inputs by promoting to (1, N) so all downstream ops
    // (rebank_rm, transpose_rm, …) see rank ≥ 2.  Squeeze back on return.
    const bool was_1d = (input_real.padded_shape().size() == 1u);
    const uint32_t N_last = static_cast<uint32_t>(input_real.padded_shape()[-1]);
    std::optional<ttnn::Tensor> real_2d_buf;
    if (was_1d) real_2d_buf = ttnn::reshape(input_real, make_shape({1u, N_last}));
    const ttnn::Tensor& real_in = was_1d ? *real_2d_buf : input_real;

    auto squeeze = [&](std::tuple<ttnn::Tensor, ttnn::Tensor> out)
        -> std::tuple<ttnn::Tensor, ttnn::Tensor> {
        if (!was_1d) return out;
        auto& [r, i] = out;
        return {ttnn::reshape(r, make_shape({N_last})),
                ttnn::reshape(i, make_shape({N_last}))};
    };
    if (two_pass_eligible(real_in))
        return squeeze(fft_two_pass(real_in, /*input_imag=*/std::nullopt, precision, false));
    if (three_pass_eligible(real_in))
        return squeeze(fft_three_pass_auto(real_in, /*input_imag=*/std::nullopt, precision, false));
    if (bluestein_eligible(real_in))
        return squeeze(bluestein_dispatch(real_in, /*input_imag=*/std::nullopt, precision, false));
    return squeeze(ttnn::prim::fft(real_in, /*inverse=*/false, /*input_imag=*/std::nullopt, precision));
}

std::tuple<ttnn::Tensor, ttnn::Tensor> fft(
    const ttnn::Tensor& input_real,
    const ttnn::Tensor& input_imag,
    FFTPrecision precision) {
    // Complex-input variant — same routing table as the real-input overload.
    const bool was_1d = (input_real.padded_shape().size() == 1u);
    const uint32_t N_last = static_cast<uint32_t>(input_real.padded_shape()[-1]);
    std::optional<ttnn::Tensor> real_2d_buf, imag_2d_buf;
    if (was_1d) {
        real_2d_buf = ttnn::reshape(input_real, make_shape({1u, N_last}));
        imag_2d_buf = ttnn::reshape(input_imag, make_shape({1u, N_last}));
    }
    const ttnn::Tensor& real_in = was_1d ? *real_2d_buf : input_real;
    const ttnn::Tensor& imag_in = was_1d ? *imag_2d_buf : input_imag;

    auto squeeze = [&](std::tuple<ttnn::Tensor, ttnn::Tensor> out)
        -> std::tuple<ttnn::Tensor, ttnn::Tensor> {
        if (!was_1d) return out;
        auto& [r, i] = out;
        return {ttnn::reshape(r, make_shape({N_last})),
                ttnn::reshape(i, make_shape({N_last}))};
    };
    if (two_pass_eligible(real_in))
        return squeeze(fft_two_pass(real_in, imag_in, precision, false));
    if (three_pass_eligible(real_in))
        return squeeze(fft_three_pass_auto(real_in, imag_in, precision, false));
    if (bluestein_eligible(real_in))
        return squeeze(bluestein_dispatch(real_in, imag_in, precision, false));
    return squeeze(ttnn::prim::fft(real_in, /*inverse=*/false, imag_in, precision));
}

std::tuple<ttnn::Tensor, ttnn::Tensor> ifft(
    const ttnn::Tensor& spectrum_real,
    const ttnn::Tensor& spectrum_imag,
    FFTPrecision precision) {
    // Inverse routing mirrors fft() with the inverse flag set.
    //
    // Two-pass / three-pass IFFT: swap-trick (see fft_two_pass comment)
    //   — forward FFT on conjugated spectrum with 1/N folded into the
    //   last radix_pass writer.  Zero extra dispatch vs forward FFT.
    //
    // Bluestein IFFT: sign-flipped chirps + 1/N folded into chirp_k;
    //   see bluestein_host.hpp get_or_create(inverse=true).

    // Promote 1D (N,) inputs to (1, N) — same contract as fft().
    const bool was_1d = (spectrum_real.padded_shape().size() == 1u);
    const uint32_t N_last = static_cast<uint32_t>(spectrum_real.padded_shape()[-1]);
    std::optional<ttnn::Tensor> real_2d_buf, imag_2d_buf;
    if (was_1d) {
        real_2d_buf = ttnn::reshape(spectrum_real, make_shape({1u, N_last}));
        imag_2d_buf = ttnn::reshape(spectrum_imag, make_shape({1u, N_last}));
    }
    const ttnn::Tensor& real_in = was_1d ? *real_2d_buf : spectrum_real;
    const ttnn::Tensor& imag_in = was_1d ? *imag_2d_buf : spectrum_imag;

    auto squeeze = [&](std::tuple<ttnn::Tensor, ttnn::Tensor> out)
        -> std::tuple<ttnn::Tensor, ttnn::Tensor> {
        if (!was_1d) return out;
        auto& [r, i] = out;
        return {ttnn::reshape(r, make_shape({N_last})),
                ttnn::reshape(i, make_shape({N_last}))};
    };
    if (two_pass_eligible(real_in))
        return squeeze(fft_two_pass(real_in, imag_in, precision, true));
    if (three_pass_eligible(real_in))
        return squeeze(fft_three_pass_auto(real_in, imag_in, precision, true));
    if (bluestein_eligible(real_in))
        return squeeze(bluestein_dispatch(real_in, imag_in, precision, true));
    // Small pow-2 N ≤ 1024: SingleTile/BatchedStockhamFactory only implement
    // the forward DFT, so prim::fft(inverse=true) would fall through to the
    // legacy FFTProgramFactory → TT_THROW.  Use the same swap trick as the
    // two-pass IFFT but via fft_radix_pass (which has output_scale for 1/N).
    {
        const uint32_t N = static_cast<uint32_t>(real_in.padded_shape()[-1]);
        if (native_path_enabled() && is_pow2(N) && N >= 2u && N <= 1024u)
            return squeeze(small_pow2_ifft(real_in, imag_in, N, precision));
    }
    // Unreachable with TT_FFT_NATIVE ON for any supported N.  Kept as a
    // safety valve for the TT_FFT_NATIVE=0 debug mode.
    return squeeze(ttnn::prim::fft(real_in, /*inverse=*/true,
                                   std::make_optional(imag_in), precision));
}

}  // namespace ttnn::operations::experimental
