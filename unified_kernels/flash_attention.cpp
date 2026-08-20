// SPDX-License-Identifier: Apache-2.0
//
// Flash attention: the same answer as unified_kernels/attention.cpp, with K and V streamed in
// chunks so the score block never exists in full. That is what lets a sequence longer than L1
// be attended over, and it is the one thing on the llama-prefill path that needed a new IDIOM
// rather than a new op.
//
// Per query row, three values are carried across the chunk loop:
//
//     m   the running maximum score
//     l   the running sum of exp(score - m)
//     o   the running UNNORMALISED output
//
// and when a chunk raises the maximum, everything accumulated under the old one is corrected.
// Accumulator cannot express that: it carries a running total, and nothing in it can rescale
// the total between steps.
//
// THE FORMULATION MATTERS. The textbook writing computes p = exp(s - m'), which needs the NEW
// maximum in the iteration that produces it -- and a retained value cannot be read without
// releasing it. So each chunk is normalised by its OWN row max and the difference is folded
// into two corrections:
//
//     rm    = rowmax(s)
//     p     = exp(s - rm)              bounded by 1 by construction
//     m'    = max(m, rm)               written as state, never read here
//     c_old = exp(m  - m')             rescales everything accumulated so far
//     c_new = exp(rm - m')             rescales THIS chunk's contribution
//     l'    = l * c_old + rowsum(p * c_new)
//     o'    = o * c_old + (p * c_new) @ V
//
// Every exponent is non-positive, so nothing can overflow -- which matters, because this
// SFPU's exp has a finite input domain and returns garbage outside it.
//
// c_new is folded into p ONCE and both the sum and the matmul read the scaled version, which
// is cheaper than scaling their two results separately.
//
// STATE BUFFERS. One circular buffer each, sized 2x the block: release() waits on the live
// value, store() reserves the free half, and the pop happens at the end of the iteration --
// so the next iteration finds the new value at the front. No parity bookkeeping.
//
// HOST LAYOUT. K arrives grid-transposed per chunk (see TransposeB), and K, V and the mask
// arrive chunk-major so each chunk's slice is contiguous.
//
// Compile-time args:
//   0        Sq in tiles
//   1        Sk per CHUNK, in tiles
//   2        D in tiles
//   3        number of chunks
//   4..      TensorAccessorArgs for q, k, v, mask, then out
//
// Runtime args (identical on all three kernels):
//   0..4     q, k, v, mask, out base addresses
//   5        1/sqrt(head dim) as a packed bfloat16 pair

#include <tt/unified/core>

namespace u = tt::unified;

constexpr uint32_t kCbQ = 0;
constexpr uint32_t kCbK = 1;
constexpr uint32_t kCbV = 2;
constexpr uint32_t kCbMask = 3;
constexpr uint32_t kCbOne = 4;
constexpr uint32_t kCbScale = 5;
constexpr uint32_t kCbScores = 6;
constexpr uint32_t kCbScaled = 7;
constexpr uint32_t kCbMasked = 8;
constexpr uint32_t kCbRowMax = 9;
constexpr uint32_t kCbProb = 10;
constexpr uint32_t kCbProbScaled = 11;
constexpr uint32_t kCbRowSum = 12;
constexpr uint32_t kCbPV = 13;
constexpr uint32_t kCbOScaled = 14;
constexpr uint32_t kCbCorrOld = 15;
constexpr uint32_t kCbOut = 16;
constexpr uint32_t kCbCorrNew = 17;
constexpr uint32_t kCbM = 18;  // state, 2x pages
constexpr uint32_t kCbL = 19;  // state, 2x pages
constexpr uint32_t kCbO = 20;  // state, 2x pages
constexpr uint32_t kCbRecipL = 21;
constexpr uint32_t kCbMNow = 22;  // this chunk's new maximum, before it becomes state

void kernel_main() {
    constexpr uint32_t sq = get_compile_time_arg_val(0);
    constexpr uint32_t sk = get_compile_time_arg_val(1);
    constexpr uint32_t dt = get_compile_time_arg_val(2);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(3);

    constexpr auto q_args = TensorAccessorArgs<4>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t mask_addr = get_arg_val<uint32_t>(3);
    const uint32_t out_addr = get_arg_val<uint32_t>(4);
    const uint32_t scale_bits = get_arg_val<uint32_t>(5);

    using Q = u::Shape<sq, dt>;
    using Kt = u::Shape<dt, sk>;
    using V = u::Shape<sk, dt>;
    using Scores = u::Shape<sq, sk>;
    using Vec = u::reduce_shape<Scores, u::Axis::Cols>;  // Sq x 1
    using Out = u::Shape<sq, dt>;
    using One = u::Shape<1, 1>;

    u::matmul_init<Q, Kt>(kCbQ, kCbK, kCbOut);

    u::Storage<Q> q_storage(kCbQ);
    u::Storage<Kt> k_storage(kCbK);
    u::Storage<V> v_storage(kCbV);
    u::Storage<Scores> mask_storage(kCbMask);
    u::Storage<One> one_storage(kCbOne);
    u::Storage<One> scale_storage(kCbScale);
    u::Storage<Scores> scores_storage(kCbScores);
    u::Storage<Scores> scaled_storage(kCbScaled);
    u::Storage<Scores> masked_storage(kCbMasked);
    u::Storage<Vec> rowmax_storage(kCbRowMax);
    u::Storage<Scores> prob_storage(kCbProb);
    u::Storage<Scores> probscaled_storage(kCbProbScaled);
    u::Storage<Vec> rowsum_storage(kCbRowSum);
    u::Storage<Out> pv_storage(kCbPV);
    u::Storage<Out> oscaled_storage(kCbOScaled);
    u::Storage<Vec> corrold_storage(kCbCorrOld);
    u::Storage<Vec> corrnew_storage(kCbCorrNew);
    u::Storage<Vec> m_storage(kCbM);
    u::Storage<Vec> l_storage(kCbL);
    u::Storage<Out> o_storage(kCbO);
    u::Storage<Vec> recipl_storage(kCbRecipL);
    u::Storage<Vec> mnow_storage(kCbMNow);
    u::Storage<Out> out_storage(kCbOut);

    const auto q_acc = TensorAccessor(q_args, q_addr);
    const auto k_acc = TensorAccessor(k_args, k_addr);
    const auto v_acc = TensorAccessor(v_args, v_addr);
    const auto mask_acc = TensorAccessor(mask_args, mask_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    // Kernel scope: Q feeds every chunk's first matmul, and both constants are re-read by
    // every reduction and broadcast that folds them in.
    u::ComputeBlock one = u::fill_reduce_scaler<1>(one_storage, u::kReduceScalerOne);
    u::ComputeBlock scale = u::fill_reduce_scaler<1>(scale_storage, scale_bits);
    u::ComputeBlock q = u::noc_load<1>(q_storage, q_acc, 0).wait();

    // The running state. Declared here because here is how long it lives.
    u::RetainedBlock<Vec> m_slot;
    u::RetainedBlock<Vec> l_slot;
    u::RetainedBlock<Out> o_slot;

    for (uint32_t j = 0; j < num_chunks; ++j) {
        u::ComputeBlock k = u::noc_load<1>(k_storage, k_acc, j).wait();
        u::ComputeBlock v = u::noc_load<1>(v_storage, v_acc, j).wait();
        u::ComputeBlock mask = u::noc_load<1>(mask_storage, mask_acc, j).wait();

        u::ComputeBlock s = scores_storage.store(u::matmul<u::TransposeB::Yes>(q, k));
        u::ComputeBlock sc = scaled_storage.store(s * u::bcast<u::Axis::Both>(scale));
        u::ComputeBlock sm = masked_storage.store(sc + mask);

        u::ComputeBlock rm = rowmax_storage.store(u::reduce_max<u::Axis::Cols>(sm, one));
        u::ComputeBlock p = prob_storage.store((sm - u::bcast<u::Axis::Cols>(rm)).exp());

        if (j == 0) {
            // Nothing accumulated yet, so no correction and c_new is 1. The state IS this
            // chunk. `m` has to be COPIED out of rowmax_storage into its own buffer, and the
            // model has no bare copy op -- the max of a value with itself is the identity and
            // is a real op rather than a trick.
            m_slot = m_storage.store(u::copy(rm));
            l_slot = l_storage.store(u::reduce_sum<u::Axis::Cols>(p, one));
            o_slot = o_storage.store(u::matmul(p, v));
        } else {
            u::ComputeBlock<Vec> m_prev = m_slot.release();

            // The new maximum goes to a SCRATCH buffer first, because both corrections read it
            // and the state buffer cannot serve both roles: with the old value not yet popped
            // it holds two blocks, and a read takes the front -- the old one.
            u::ComputeBlock m_now = mnow_storage.store(u::max_(m_prev, rm));

            u::ComputeBlock c_old = corrold_storage.store((m_prev - m_now).exp());
            u::ComputeBlock c_new = corrnew_storage.store((rm - m_now).exp());

            // Now it can become the state: copied rather than recomputed, so the two cannot
            // drift apart.
            m_slot = m_storage.store(u::copy(m_now));

            // c_new folded into p once; the sum and the matmul both read the scaled version.
            u::ComputeBlock p2 = probscaled_storage.store(p * u::bcast<u::Axis::Cols>(c_new));
            u::ComputeBlock rs = rowsum_storage.store(u::reduce_sum<u::Axis::Cols>(p2, one));

            u::ComputeBlock<Vec> l_prev = l_slot.release();
            l_slot = l_storage.store(l_prev * c_old + rs);

            u::ComputeBlock<Out> o_prev = o_slot.release();
            u::ComputeBlock os = oscaled_storage.store(o_prev * u::bcast<u::Axis::Cols>(c_old));
            u::ComputeBlock pv = pv_storage.store(u::matmul(p2, v));
            o_slot = o_storage.store(os + pv);
        }
    }

    // out = o / l. Releasing every slot is not optional: ~RetainedBlock asserts on a state
    // that was pushed and never waited on, which is exactly what a forgotten drain is.
    u::ComputeBlock<Vec> m_done = m_slot.release();
    u::ComputeBlock<Vec> l_done = l_slot.release();
    u::ComputeBlock<Out> o_done = o_slot.release();
    (void)sizeof(m_done);

    u::ComputeBlock rl = recipl_storage.store(u::recip(l_done));
    u::noc_store<0>(out_storage.store(o_done * u::bcast<u::Axis::Cols>(rl)), out, 0);
}
