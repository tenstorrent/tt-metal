// SPDX-License-Identifier: Apache-2.0
//
// RMSNorm over each row:
//
//     out[h, w] = x[h, w] / sqrt(mean(x[h, :]^2) + eps) * weight[w]
//
// No new library work: the square is an ordinary SFPU multiply of a block by itself, the
// mean is a reduction along Cols, and the two rescalings are broadcasts along DIFFERENT
// axes -- Cols for the per-row reciprocal RMS, Rows for the per-feature weight. That pair
// is the point of this kernel: attention only ever broadcast along Cols.
//
// The epsilon is added with a scalar broadcast onto the Ht x 1 mean vector, which is the
// case the axis has to be DECLARED for: against a Shape<Ht, 1> block, both Axis::Both and
// Axis::Rows want a Shape<1, 1> vector, so no shape could tell them apart.
//
// Compile-time args, all named, plus a dfb_<name> per buffer:
//   rows in tiles
//   cols in tiles
//
// Runtime args, named and identical on all three kernels:
//   eps as a packed bfloat16 pair

#include <tt/unified/core>
#include "experimental/kernel_args.h"

namespace u = tt::unified;

void kernel_main() {
    constexpr uint32_t kDfbX = get_arg(args::dfb_x);
    constexpr uint32_t kDfbW = get_arg(args::dfb_w);
    constexpr uint32_t kDfbEps = get_arg(args::dfb_eps);
    constexpr uint32_t kDfbInvN = get_arg(args::dfb_inv_n);
    constexpr uint32_t kDfbSq = get_arg(args::dfb_sq);
    constexpr uint32_t kDfbMean = get_arg(args::dfb_mean);
    constexpr uint32_t kDfbRsqrt = get_arg(args::dfb_rsqrt);
    constexpr uint32_t kDfbNormed = get_arg(args::dfb_normed);
    constexpr uint32_t kDfbOut = get_arg(args::dfb_out);
    // ht is the ROW-CHUNK height, not the whole tensor's: rows are normalised
    // independently -- each one's RMS depends on that row alone -- so the tensor is walked
    // in chunks of ht rows and only ht*wt tiles are ever resident. Which chunks this core
    // owns comes from runtime args. Without this the whole [S, d_model] block had to fit
    // L1 at once, which is 1024 tiles at S=512 by d_model 2048.
    constexpr uint32_t ht = get_arg(args::ht);
    constexpr uint32_t wt = get_arg(args::wt);
    const uint32_t eps_bits = get_arg(args::eps_bits);
    const uint32_t chunk_begin = get_arg(args::chunk_begin);
    const uint32_t chunk_count = get_arg(args::chunk_count);

    constexpr auto kAxis = u::Axis::Cols;  // each row is normalised independently

    using X = u::Shape<ht, wt>;
    using Vec = u::reduce_shape<X, kAxis>;  // Ht x 1, one RMS per row
    using W = u::Shape<1, wt>;              // one weight per feature column
    using One = u::Shape<1, 1>;

    u::compute_init(kDfbX, kDfbOut);

    // Only the buffers a DATAFLOW call names still need a Storage of their own: the two
    // noc_loads, the two scaler fills, and the noc_store. Every pure-compute intermediate
    // names its buffer in its own declaration -- u::ComputeBlock<Shape, dfb> -- so the
    // initialiser is nothing but the expression.
    u::Storage<X> x_storage(kDfbX);
    u::Storage<W> w_storage(kDfbW);
    u::Storage<One> eps_storage(kDfbEps);
    u::Storage<One> inv_n_storage(kDfbInvN);
    u::Storage<X> out_storage(kDfbOut);

    const auto x_acc = TensorAccessor(tensor::x);
    const auto w_acc = TensorAccessor(tensor::w);
    const auto out = TensorAccessor(tensor::out);

    // A mean divides by the number of ELEMENTS folded into one output, which for a Cols
    // collapse is wt * 32. Feeding a reduce_mean a scaler of 1 makes it a sum, silently.
    const uint32_t inv_n_bits = u::bf16_pair(1.0f / static_cast<float>(u::ReduceGeometry<X>::elements(kAxis)));

    // Both are kernel-scope residents: every reduce_tile re-reads the scaler, and the
    // epsilon is read by every tile of the broadcast below.
    u::ComputeBlock inv_n = u::fill_reduce_scaler<1>(inv_n_storage, inv_n_bits);
    u::ComputeBlock eps = u::fill_reduce_scaler<1>(eps_storage, eps_bits);

    // The weights are one row of wt tiles, the same for every chunk, so this is loaded ONCE
    // and stays resident -- a ComputeBlock declared inside the loop would be popped after a
    // single use and the next chunk would wait for a refill that never comes.
    u::ComputeBlock w = u::noc_load<0>(w_storage, w_acc, 0).wait();

    for (uint32_t c = 0; c < chunk_count; ++c) {
        const uint32_t i = chunk_begin + c;

        // Chunk i is rows [i*ht, +ht), which in a row-major tile layout is contiguous pages
        // -- so an ordinary block load, no gather. That is the difference between chunking
        // the ROWS of a row-major tensor and slicing its columns.
        u::ComputeBlock x = u::noc_load<0>(x_storage, x_acc, i).wait();

        // x^2, as an ordinary two-leaf SFPU tree whose leaves happen to be the same buffer.
        u::ComputeBlock<X, kDfbSq> sq = x * x;

        u::ComputeBlock<Vec, kDfbMean> mean = u::reduce_mean<kAxis>(sq, inv_n);

        // (mean + eps)^-1/2, the reciprocal square root riding the broadcast's epilogue so
        // the epsilon and the rsqrt are one pass.
        u::ComputeBlock<Vec, kDfbRsqrt> inv_rms = (mean + u::bcast<u::Axis::Both>(eps)).rsqrt();

        // Cols: one value per ROW, replicated across that row's columns.
        u::ComputeBlock<X, kDfbNormed> normed = x * u::bcast<kAxis>(inv_rms);

        // Rows: one value per COLUMN, replicated down the rows. The other axis, in the same
        // kernel, which is what makes this more than a second reduction test.
        u::noc_store<1>(out_storage, normed * u::bcast<u::Axis::Rows>(w), out, i);
    }
}
