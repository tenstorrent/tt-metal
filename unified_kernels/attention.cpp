// SPDX-License-Identifier: Apache-2.0
//
// One attention head, one core, non-flash:
//
//     out = softmax(Q @ Kt * scale + mask) @ V
//
// Q is Sq x D tiles, K is Sk x D, V is Sk x D, out is Sq x D. The score block is
// Sq x Sk, materialised in full -- no chunking, which is what "non-flash" means and
// what bounds this kernel to a score block that fits L1 and the DST budget.
//
// TWO HALVES MAKE THE TRANSPOSE. Metal's matmul flag transposes each 32x32 tile of B and
// leaves the tile GRID alone, so the host hands K in grid-transposed order -- tile (s, d)
// at slot (d, s) -- and this kernel supplies the per-tile half with TransposeB::Yes. K's
// shape here is therefore D x Sk, which is what the geometry infers from and why nothing
// needs to be told the inner dimension.
//
// The mask is ADDITIVE and host-supplied: 0 where a position is visible and a large
// negative value where it is not, so exp() takes it to zero. Causal masking guarantees
// every row keeps at least its diagonal, so no row sums to zero and the reciprocal is
// finite.
//
// Compile-time args, all named, plus a cb_<name> per buffer:
//   Sq in tiles
//   Sk in tiles
//   D  in tiles
//
// Runtime args, named and identical on all three kernels:
//   1/sqrt(head dim) as a packed bfloat16 pair. Computed on the host: the
//            kernel toolchain has no sqrtf, and the host knows the head dim exactly.

#include <tt/unified/core>
#include "experimental/kernel_args.h"

namespace u = tt::unified;

void kernel_main() {
    constexpr uint32_t sq = get_arg(args::sq);
    constexpr uint32_t sk = get_arg(args::sk);
    constexpr uint32_t dt = get_arg(args::dt);

    constexpr uint32_t kCbQ = get_arg(args::cb_q);
    constexpr uint32_t kCbK = get_arg(args::cb_k);
    constexpr uint32_t kCbV = get_arg(args::cb_v);
    constexpr uint32_t kCbMask = get_arg(args::cb_mask);
    constexpr uint32_t kCbOne = get_arg(args::cb_one);
    constexpr uint32_t kCbScale = get_arg(args::cb_scale);
    constexpr uint32_t kCbScores = get_arg(args::cb_scores);
    constexpr uint32_t kCbScaled = get_arg(args::cb_scaled);
    constexpr uint32_t kCbMasked = get_arg(args::cb_masked);
    constexpr uint32_t kCbRowMax = get_arg(args::cb_row_max);
    constexpr uint32_t kCbExp = get_arg(args::cb_exp);
    constexpr uint32_t kCbRecip = get_arg(args::cb_recip);
    constexpr uint32_t kCbProb = get_arg(args::cb_prob);
    constexpr uint32_t kCbOut = get_arg(args::cb_out);
    const uint32_t scale_bits = get_arg(args::scale_bits);

    using Q = u::Shape<sq, dt>;                          // Sq x D
    using Kt = u::Shape<dt, sk>;                         // D x Sk -- K, grid-transposed by the host
    using V = u::Shape<sk, dt>;                          // Sk x D
    using Scores = u::Shape<sq, sk>;                     // Sq x Sk
    using Vec = u::reduce_shape<Scores, u::Axis::Cols>;  // Sq x 1, one entry per query row
    using Out = u::Shape<sq, dt>;

    // matmul_init and not compute_init: compute_kernel_hw_startup programs BOTH source
    // registers and the packer, which is a superset of what the SFPU path needs, and the
    // broadcast, reduce and SFPU passes each re-init their own state per use. Its
    // SrcOrder::Reverse is what matmul requires.
    u::matmul_init<Q, Kt>(kCbQ, kCbK, kCbOut);

    u::Storage<Q> q_storage(kCbQ);
    u::Storage<Kt> k_storage(kCbK);
    u::Storage<V> v_storage(kCbV);
    u::Storage<Scores> mask_storage(kCbMask);
    u::Storage<u::Shape<1, 1>> one_storage(kCbOne);
    u::Storage<u::Shape<1, 1>> scale_storage(kCbScale);
    u::Storage<Scores> scores_storage(kCbScores);
    u::Storage<Scores> scaled_storage(kCbScaled);
    u::Storage<Scores> masked_storage(kCbMasked);
    u::Storage<Vec> rowmax_storage(kCbRowMax);
    u::Storage<Scores> exp_storage(kCbExp);
    u::Storage<Vec> recip_storage(kCbRecip);
    u::Storage<Scores> prob_storage(kCbProb);
    u::Storage<Out> out_storage(kCbOut);

    const auto q_acc = TensorAccessor(tensor::q);
    const auto k_acc = TensorAccessor(tensor::k);
    const auto v_acc = TensorAccessor(tensor::v);
    const auto mask_acc = TensorAccessor(tensor::mask);
    const auto out = TensorAccessor(tensor::out);

    // 1.0 for max and sum alike -- metal folds the scaler into every reduce_tile, and
    // neither of these reductions is an average.
    u::ComputeBlock one = u::fill_reduce_scaler<1>(one_storage, u::kReduceScalerOne);

    // 1/sqrt(head dim), packed by the host.
    u::ComputeBlock scale = u::fill_reduce_scaler<1>(scale_storage, scale_bits);

    u::ComputeBlock q = u::noc_load<0>(q_storage, q_acc, 0).wait();
    u::ComputeBlock k = u::noc_load<0>(k_storage, k_acc, 0).wait();
    u::ComputeBlock v = u::noc_load<0>(v_storage, v_acc, 0).wait();
    u::ComputeBlock mask = u::noc_load<0>(mask_storage, mask_acc, 0).wait();

    // Q @ Kt. Single-shot: one k-block, so no Accumulator.
    u::ComputeBlock scores = scores_storage.store(u::matmul<u::TransposeB::Yes>(q, k));

    // The scale is its own pass: a broadcast is an FPU fusion and cannot be an operand of
    // the elementwise add below, which the model rejects rather than mis-emits.
    u::ComputeBlock scaled = scaled_storage.store(scores * u::bcast<u::Axis::Both>(scale));

    // The mask is a full block, so this is an ordinary two-buffer elementwise add.
    u::ComputeBlock masked = masked_storage.store(scaled + mask);

    // Softmax over each query row: subtract the row max for stability, exponentiate,
    // then divide by the row sum. `masked` is read twice and `exp` twice, so both are
    // held for the whole sequence rather than re-loaded.
    u::ComputeBlock rowmax = rowmax_storage.store(u::reduce_max<u::Axis::Cols>(masked, one));
    u::ComputeBlock e = exp_storage.store((masked - u::bcast<u::Axis::Cols>(rowmax)).exp());

    // The reciprocal rides the reduction's own epilogue chain, so the sum and its inverse
    // are one pass rather than two.
    u::ComputeBlock recip = recip_storage.store(u::reduce_sum<u::Axis::Cols>(e, one).recip());

    u::ComputeBlock prob = prob_storage.store(e * u::bcast<u::Axis::Cols>(recip));

    u::noc_store<1>(out_storage.store(u::matmul(prob, v)), out, 0);
}
