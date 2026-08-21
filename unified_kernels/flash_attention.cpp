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
// m' is needed in the iteration that produces it, and a RETAINED value cannot be read without
// releasing it -- so it goes to a scratch buffer, and a copy carries it into the state slot.
// An earlier version avoided that by normalising each chunk to its own row max and folding the
// difference into a second correction, which cost two extra passes and bought nothing, since
// the scratch buffer was needed either way.
//
//     m'    = max(m, rowmax(s))
//     c_old = exp(m - m')              rescales everything accumulated so far
//     p     = exp(s - m')              bounded by 1 by construction
//     l'    = l * c_old + rowsum(p)
//     o'    = o * c_old + p @ V
//
// Every exponent is non-positive, so nothing can overflow -- which matters, because this
// SFPU's exp has a finite input domain and returns garbage outside it.
//
// Every exponent is non-positive, so nothing can overflow -- which matters, because this SFPU's
// exp has a finite input domain and returns garbage outside it.
//
// The scale is pre-applied to Q on the host, which removes a broadcast pass per chunk. Twelve
// passes per chunk, where the first working version had fifteen.
//
// Passes are the right thing to count: measured, an extra chunk costs ~29us whatever its SIZE,
// so per-chunk cost is dominated by fixed per-pass overhead -- a circular-buffer round trip
// and an acquire/pack cycle each -- rather than by arithmetic.
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
//
// Q arrives ALREADY scaled by 1/sqrt(head dim): folding it into the operand costs the host one
// multiply and saves a broadcast pass per chunk on device.

#include <tt/unified/core>

namespace u = tt::unified;

constexpr uint32_t kCbQ = 0;
constexpr uint32_t kCbK = 1;
constexpr uint32_t kCbV = 2;
constexpr uint32_t kCbMask = 3;
constexpr uint32_t kCbOne = 4;
constexpr uint32_t kCbColOnes = 23;
constexpr uint32_t kCbMasked = 8;
constexpr uint32_t kCbRowMax = 9;
constexpr uint32_t kCbProb = 10;
constexpr uint32_t kCbRowSum = 12;
constexpr uint32_t kCbPV = 13;
constexpr uint32_t kCbOScaled = 14;
constexpr uint32_t kCbCorrOld = 15;
constexpr uint32_t kCbOut = 16;
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
    constexpr auto colones_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<colones_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t mask_addr = get_arg_val<uint32_t>(3);
    const uint32_t colones_addr = get_arg_val<uint32_t>(4);
    const uint32_t out_addr = get_arg_val<uint32_t>(5);

    using Q = u::Shape<sq, dt>;
    using Kt = u::Shape<dt, sk>;
    using V = u::Shape<sk, dt>;
    using Scores = u::Shape<sq, sk>;
    using Vec = u::reduce_shape<Scores, u::Axis::Cols>;  // Sq x 1
    using Out = u::Shape<sq, dt>;
    using One = u::Shape<1, 1>;
    // A column of ones, sk tiles tall: matmul(p, this) IS the row sum, since summing a
    // row is a matvec. Cheaper than reduce_sum along the same axis -- a reduction folds
    // sk INPUT tiles per output tile through reduce_tile, where the matmul does sk MACs
    // and the FPU is what that unit is for. ttnn's SDPA does the same thing and calls it
    // matmul_reduce against a cb_col_identity.
    //
    // The ones sit in COLUMN 0 only, which keeps the result identical to what the
    // reduction produced: column 0 carries the sum and the rest of the tile is zero.
    // All-ones would put the sum in every column instead, and while nothing downstream
    // reads those, `bcast<Cols>` taking column 0 is a contract worth not quietly changing.
    using Kt1 = u::Shape<sk, 1>;

    u::matmul_init<Q, Kt>(kCbQ, kCbK, kCbOut);

    u::Storage<Q> q_storage(kCbQ);
    u::Storage<Kt> k_storage(kCbK);
    u::Storage<V> v_storage(kCbV);
    u::Storage<Scores> mask_storage(kCbMask);
    u::Storage<One> one_storage(kCbOne);
    u::Storage<Kt1> colones_storage(kCbColOnes);
    u::Storage<Scores> masked_storage(kCbMasked);
    u::Storage<Vec> rowmax_storage(kCbRowMax);
    u::Storage<Scores> prob_storage(kCbProb);
    u::Storage<Vec> rowsum_storage(kCbRowSum);
    u::Storage<Out> pv_storage(kCbPV);
    u::Storage<Out> oscaled_storage(kCbOScaled);
    u::Storage<Vec> corrold_storage(kCbCorrOld);
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
    const auto colones_acc = TensorAccessor(colones_args, colones_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    // Kernel scope: Q feeds every chunk's first matmul, and both constants are re-read by
    // every reduction and broadcast that folds them in.
    u::ComputeBlock one = u::fill_reduce_scaler<1>(one_storage, u::kReduceScalerOne);
    // Read once and used by every chunk, like q. reduce_max keeps the scaler above:
    // a maximum has no matmul form.
    u::ComputeBlock col_ones = u::noc_load<1>(colones_storage, colones_acc, 0).wait();
    u::ComputeBlock q = u::noc_load<1>(q_storage, q_acc, 0).wait();

    // The running state. Declared here because here is how long it lives.
    u::RetainedBlock<Vec> m_slot;
    u::RetainedBlock<Vec> l_slot;
    u::RetainedBlock<Out> o_slot;

    for (uint32_t j = 0; j < num_chunks; ++j) {
        u::ComputeBlock k = u::noc_load<1>(k_storage, k_acc, j).wait();
        u::ComputeBlock v = u::noc_load<1>(v_storage, v_acc, j).wait();
        u::ComputeBlock mask = u::noc_load<1>(mask_storage, mask_acc, j).wait();

        // The mask rides along in the matmul: `add` puts it into the product while that is
        // still in DST, so what used to be a separate 8x8-tile pass over the scores -- read
        // s, read mask, add, write sm -- is now one FPU instruction per output tile with no
        // L1 round trip. The scores buffer is gone with it.
        u::ComputeBlock sm = masked_storage.store(u::matmul<u::TransposeB::Yes>(q, k).add(mask));
        u::ComputeBlock rm = rowmax_storage.store(u::reduce_max<u::Axis::Cols>(sm, one));

        if (j == 0) {
            // Nothing accumulated yet, so this chunk IS the state and there is nothing to
            // correct. `rm` still has to be copied out of the reduction's buffer into the one
            // that will carry it.
            m_slot = m_storage.store(u::copy(rm));
            u::ComputeBlock p = prob_storage.store((sm - u::bcast<u::Axis::Cols>(rm)).exp());
            l_slot = l_storage.store(u::matmul(p, col_ones));
            o_slot = o_storage.store(u::matmul(p, v));
        } else {
            u::ComputeBlock<Vec> m_prev = m_slot.release();

            // The new maximum, to a scratch buffer: the state buffer cannot serve, because with
            // the old value not yet popped it holds two blocks and a read takes the front.
            u::ComputeBlock m_now = mnow_storage.store(u::max_(m_prev, rm));
            u::ComputeBlock c_old = corrold_storage.store((m_prev - m_now).exp());

            // Normalised to the NEW maximum in one pass, which is what removes the separate
            // rescale and c_new entirely.
            u::ComputeBlock p = prob_storage.store((sm - u::bcast<u::Axis::Cols>(m_now)).exp());

            m_slot = m_storage.store(u::copy(m_now));

            u::ComputeBlock rs = rowsum_storage.store(u::matmul(p, col_ones));

            u::ComputeBlock<Vec> l_prev = l_slot.release();
            l_slot = l_storage.store(l_prev * c_old + rs);

            u::ComputeBlock<Out> o_prev = o_slot.release();
            u::ComputeBlock os = oscaled_storage.store(o_prev * u::bcast<u::Axis::Cols>(c_old));
            u::ComputeBlock pv = pv_storage.store(u::matmul(p, v));
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
