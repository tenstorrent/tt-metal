// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fast host conversion of row-major bfloat16 data into Bfp8_b / Bfp4_b tiles (32x32, 16x16 faces), byte-identical to
// pack_as_bfp_tiles (tt_metal/impl/data_format/blockfloat_common.cpp) with is_exp_a = false: same shared exponent
// (max fp32 exponent field per 16-value face row), same truncating shift, same round-to-nearest-even and saturation.
// Differences are only in how the work is laid out: one pass straight from row-major bf16 (no fp32 copy, no separate
// tilize), every tile written at its fixed offset in a preallocated buffer, threads over 32-row bands, GIL released.

#include "fast_bfp.hpp"

#include <sys/mman.h>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <thread>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include <tt-metalium/bfp_fast_pack.hpp>
#include <tt-metalium/host_buffer.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::fast_bfp {
namespace {

using tt::tt_metal::bfp_fast_pack::Geometry;
using tt::tt_metal::bfp_fast_pack::pack_bf16;
constexpr auto packed_dwords = tt::tt_metal::bfp_fast_pack::packed_words;

// Large host staging buffers are written once right after allocation; with 4 KB pages the first-touch faults (~30k per
// 117 MB weight, serialized in the kernel) dominated the copy. Ask for transparent huge pages (THP is in madvise mode
// on typical hosts). Contents are unaffected; a no-op for small buffers or when THP is off.
inline void advise_huge_pages(void* p, size_t bytes) {
    constexpr uintptr_t kHuge = uintptr_t{2} << 20;
    if (p == nullptr || bytes < (size_t{8} << 20)) {
        return;
    }
    const uintptr_t begin = (reinterpret_cast<uintptr_t>(p) + kHuge - 1) & ~(kHuge - 1);
    const uintptr_t end = (reinterpret_cast<uintptr_t>(p) + bytes) & ~(kHuge - 1);
    if (end > begin) {
        madvise(reinterpret_cast<void*>(begin), end - begin, MADV_HUGEPAGE);
    }
}

#if defined(__SSE2__)
// 8x8 transpose of 16-bit elements: in rows at `in + k * is`, out rows at `out + k * os`.
inline void transpose_8x8_u16(const uint16_t* in, size_t is, uint16_t* out, size_t os) {
    const __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 0 * is));
    const __m128i a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 1 * is));
    const __m128i a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 2 * is));
    const __m128i a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 3 * is));
    const __m128i a4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 4 * is));
    const __m128i a5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 5 * is));
    const __m128i a6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 6 * is));
    const __m128i a7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 7 * is));
    const __m128i b0 = _mm_unpacklo_epi16(a0, a1), b1 = _mm_unpackhi_epi16(a0, a1);
    const __m128i b2 = _mm_unpacklo_epi16(a2, a3), b3 = _mm_unpackhi_epi16(a2, a3);
    const __m128i b4 = _mm_unpacklo_epi16(a4, a5), b5 = _mm_unpackhi_epi16(a4, a5);
    const __m128i b6 = _mm_unpacklo_epi16(a6, a7), b7 = _mm_unpackhi_epi16(a6, a7);
    const __m128i c0 = _mm_unpacklo_epi32(b0, b2), c1 = _mm_unpackhi_epi32(b0, b2);
    const __m128i c2 = _mm_unpacklo_epi32(b1, b3), c3 = _mm_unpackhi_epi32(b1, b3);
    const __m128i c4 = _mm_unpacklo_epi32(b4, b6), c5 = _mm_unpackhi_epi32(b4, b6);
    const __m128i c6 = _mm_unpacklo_epi32(b5, b7), c7 = _mm_unpackhi_epi32(b5, b7);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 0 * os), _mm_unpacklo_epi64(c0, c4));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 1 * os), _mm_unpackhi_epi64(c0, c4));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 2 * os), _mm_unpacklo_epi64(c1, c5));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 3 * os), _mm_unpackhi_epi64(c1, c5));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 4 * os), _mm_unpacklo_epi64(c2, c6));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 5 * os), _mm_unpackhi_epi64(c2, c6));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 6 * os), _mm_unpacklo_epi64(c3, c7));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 7 * os), _mm_unpackhi_epi64(c3, c7));
}
#endif

// One 64x64-ish block: out rows [r0, r1), cols [c0, c1); source is cols x rows.
template <typename E>
inline void transpose_block(const E* in, E* o, size_t rows, size_t cols, size_t r0, size_t r1, size_t c0, size_t c1) {
#if defined(__SSE2__)
    if constexpr (sizeof(E) == 2) {
        const size_t c8 = c0 + (c1 - c0) / 8 * 8;
        const size_t r8 = r0 + (r1 - r0) / 8 * 8;
        for (size_t c = c0; c < c8; c += 8) {
            for (size_t r = r0; r < r8; r += 8) {
                transpose_8x8_u16(in + c * rows + r, rows, o + r * cols + c, cols);
            }
        }
        // ragged edges
        for (size_t r = r0; r < r1; ++r) {
            for (size_t c = (r < r8 ? c8 : c0); c < c1; ++c) {
                o[r * cols + c] = in[c * rows + r];
            }
        }
        return;
    }
#endif
    for (size_t r = r0; r < r1; ++r) {
        for (size_t c = c0; c < c1; ++c) {
            o[r * cols + c] = in[c * rows + r];
        }
    }
}

// out[b][r][c] = src[b][c][r] for `batch` matrices; output is rows x cols, source is cols x rows, both row-major.
// Blocked (64x64) and split over threads by row blocks; output in aligned huge-page backed memory.
template <typename E>
nb::ndarray<nb::array_api> transpose_core(
    const E* src, size_t batch, size_t rows, size_t cols, const std::vector<size_t>& out_shape, nb::dlpack::dtype dt) {
    const size_t total = batch * rows * cols;
    const size_t bytes = total * sizeof(E);
    constexpr size_t kHuge = size_t{2} << 20;
    const size_t alloc_bytes = std::max<size_t>(kHuge, (bytes + kHuge - 1) / kHuge * kHuge);
    auto* out = static_cast<E*>(std::aligned_alloc(kHuge, alloc_bytes));
    TT_FATAL(out != nullptr, "fast_bfp: transpose allocation of {} bytes failed", alloc_bytes);
    advise_huge_pages(out, alloc_bytes);
    constexpr size_t B = 64;
    const size_t row_blocks = (rows + B - 1) / B;
    const size_t units = batch * row_blocks;
    auto work = [&](size_t u0, size_t u1) {
        for (size_t u = u0; u < u1; ++u) {
            const size_t b = u / row_blocks;
            const size_t r0 = (u % row_blocks) * B;
            const size_t r1 = std::min(rows, r0 + B);
            const E* in = src + b * rows * cols;
            E* o = out + b * rows * cols;
            for (size_t c0 = 0; c0 < cols; c0 += B) {
                transpose_block(in, o, rows, cols, r0, r1, c0, std::min(cols, c0 + B));
            }
        }
    };
    const unsigned nthreads = static_cast<unsigned>(std::min<size_t>(
        {units,
         std::max<size_t>(1, bytes / (4u << 20)),
         std::max(1u, std::min(32u, std::thread::hardware_concurrency()))}));
    {
        nb::gil_scoped_release release;
        if (nthreads <= 1) {
            work(0, units);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(nthreads);
            for (unsigned i = 0; i < nthreads; ++i) {
                pool.emplace_back(work, units * i / nthreads, units * (i + 1) / nthreads);
            }
            for (auto& th : pool) {
                th.join();
            }
        }
    }
    nb::capsule owner(out, [](void* p) noexcept { std::free(p); });
    return nb::ndarray<nb::array_api>(
        out, out_shape.size(), out_shape.data(), owner, nullptr, dt, nb::device::cpu::value, 0);
}

nb::ndarray<nb::array_api> transpose_dispatch(
    const void* src,
    size_t batch,
    size_t rows,
    size_t cols,
    const std::vector<size_t>& out_shape,
    nb::dlpack::dtype dt) {
    TT_FATAL(dt.lanes == 1 && (dt.bits == 16 || dt.bits == 32), "fast_bfp: transpose supports 16/32-bit dtypes");
    return dt.bits == 16 ? transpose_core(static_cast<const uint16_t*>(src), batch, rows, cols, out_shape, dt)
                         : transpose_core(static_cast<const uint32_t*>(src), batch, rows, cols, out_shape, dt);
}

// Leading dims, input rows/cols from an ndarray shape; logical rows/cols swap when transposing.
template <typename Array>
Geometry geometry_of(const Array& input, bool transpose) {
    TT_FATAL(input.ndim() >= 2, "fast_bfp: need a >= 2-D tensor");
    Geometry g;
    const size_t in_rows = input.shape(input.ndim() - 2);
    const size_t in_cols = input.shape(input.ndim() - 1);
    g.batch = 1;
    for (size_t d = 0; d + 2 < input.ndim(); ++d) {
        g.batch *= input.shape(d);
    }
    g.transpose = transpose;
    g.rows = transpose ? in_cols : in_rows;
    g.cols = transpose ? in_rows : in_cols;
    return g;
}

}  // namespace

nb::ndarray<nb::array_api> transposed_copy_of_view(const nb::ndarray<nb::array_api>& view) {
    const size_t nd = view.ndim();
    std::vector<size_t> shape(nd);
    size_t batch = 1;
    for (size_t d = 0; d < nd; ++d) {
        shape[d] = view.shape(d);
        if (d + 2 < nd) {
            batch *= shape[d];
        }
    }
    return transpose_dispatch(view.data(), batch, shape[nd - 2], shape[nd - 1], shape, view.dtype());
}

void py_module(nb::module_& mod) {
    mod.def(
        "transpose",
        [](const nb::ndarray<nb::array_api, nb::c_contig, nb::device::cpu>& input) {
            TT_FATAL(input.ndim() >= 2, "fast_bfp.transpose: need a >= 2-D tensor");
            const size_t nd = input.ndim();
            std::vector<size_t> out_shape(nd);
            size_t batch = 1;
            for (size_t d = 0; d < nd; ++d) {
                out_shape[d] = input.shape(d);
                if (d + 2 < nd) {
                    batch *= out_shape[d];
                }
            }
            std::swap(out_shape[nd - 2], out_shape[nd - 1]);
            return transpose_dispatch(
                input.data(), batch, out_shape[nd - 2], out_shape[nd - 1], out_shape, input.dtype());
        },
        nb::arg("input"),
        R"doc(Contiguous copy of `input` with its last two dims swapped (like input.mT.contiguous()), for row-major
contiguous CPU tensors with 16- or 32-bit elements. Blocked multithreaded transpose into huge-page memory; wrap the
result with torch.from_dlpack. Much faster than torch's own copy for bf16 weights.)doc");

    using In = nb::ndarray<nb::array_api, const int16_t, nb::c_contig, nb::device::cpu>;
    mod.def(
        "packed_words",
        [](const In& input, int bits, bool transpose) { return packed_dwords(geometry_of(input, transpose), bits); },
        nb::arg("input"),
        nb::arg("bits") = 8,
        nb::arg("transpose") = false,
        "Number of uint32 words pack_bf16 produces for this input.");
    mod.def(
        "pack_bf16",
        [](const In& input, int bits, unsigned nthreads, bool transpose) {
            const Geometry g = geometry_of(input, transpose);
            const size_t n = packed_dwords(g, bits);
            auto* out = new uint32_t[n];
            const auto* src = reinterpret_cast<const uint16_t*>(input.data());
            {
                nb::gil_scoped_release release;
                pack_bf16(g, src, bits, out, nthreads);
            }
            nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<uint32_t*>(p); });
            size_t shape[] = {n};
            return nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>>(out, 1, shape, std::move(owner));
        },
        nb::arg("input"),
        nb::arg("bits") = 8,
        nb::arg("nthreads") = 0,
        nb::arg("transpose") = false,
        R"doc(Row-major bfloat16 (pass tensor.view(torch.int16)) -> Bfp8_b (bits=8) or Bfp4_b (bits=4) tiles in ttnn tile
order, byte-identical to ttnn's host packer. Last two dims are zero-padded to multiples of 32; leading dims are batch.
transpose=True packs the input with its last two dims swapped (e.g. an HF [out, in] weight as [in, out]) without a
separate torch transpose. Returns raw uint32 data.)doc");
    mod.def(
        "pack_bf16_into",
        [](const In& input,
           nb::ndarray<uint32_t, nb::c_contig, nb::device::cpu> out,
           int bits,
           unsigned nthreads,
           bool transpose) {
            const Geometry g = geometry_of(input, transpose);
            const size_t n = packed_dwords(g, bits);
            TT_FATAL(out.size() >= n, "fast_bfp.pack_bf16_into: out holds {} words, need {}", out.size(), n);
            const auto* src = reinterpret_cast<const uint16_t*>(input.data());
            auto* dst = out.data();
            nb::gil_scoped_release release;
            pack_bf16(g, src, bits, dst, nthreads);
            return n;
        },
        nb::arg("input"),
        nb::arg("out"),
        nb::arg("bits") = 8,
        nb::arg("nthreads") = 0,
        nb::arg("transpose") = false,
        R"doc(Same as pack_bf16 but writes into a caller-owned uint32 buffer (e.g. pinned host memory). Returns the
number of words written.)doc");
    mod.def(
        "to_ttnn",
        [](const In& input, tt::tt_metal::DataType dtype, bool transpose, unsigned nthreads) {
            TT_FATAL(
                dtype == tt::tt_metal::DataType::BFLOAT8_B || dtype == tt::tt_metal::DataType::BFLOAT4_B,
                "fast_bfp.to_ttnn: dtype must be bfloat8_b or bfloat4_b");
            const int bits = dtype == tt::tt_metal::DataType::BFLOAT8_B ? 8 : 4;
            const Geometry g = geometry_of(input, transpose);
            ttsl::SmallVector<uint32_t> dims;
            for (size_t d = 0; d + 2 < input.ndim(); ++d) {
                dims.push_back(static_cast<uint32_t>(input.shape(d)));
            }
            dims.push_back(static_cast<uint32_t>(g.rows));
            dims.push_back(static_cast<uint32_t>(g.cols));
            const auto* src = reinterpret_cast<const uint16_t*>(input.data());
            std::vector<uint32_t> words(packed_dwords(g, bits));
            {
                nb::gil_scoped_release release;
                pack_bf16(g, src, bits, words.data(), nthreads);
            }
            tt::tt_metal::TensorSpec spec(
                ttnn::Shape(dims),
                tt::tt_metal::TensorLayout(
                    dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), tt::tt_metal::MemoryConfig{}));
            return ttnn::Tensor(tt::tt_metal::HostBuffer(std::move(words)), spec);
        },
        nb::arg("input"),
        nb::arg("dtype"),
        nb::arg("transpose") = false,
        nb::arg("nthreads") = 0,
        R"doc(bf16 tensor (pass tensor.view(torch.int16)) -> host ttnn.Tensor in TILE layout with dtype bfloat8_b or
bfloat4_b, same as ttnn.from_torch(tensor[.mT], dtype=dtype, layout=ttnn.TILE_LAYOUT) but packed in one multithreaded
pass. transpose=True swaps the last two dims.)doc");
}

}  // namespace ttnn::fast_bfp
