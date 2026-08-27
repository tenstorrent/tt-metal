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
// One LAUNCH is one attention head: a loop over query chunks, each with its own loop over
// key chunks and its own online-softmax state. Previously one launch was one query chunk
// and the host ran the outer loop, which cost 13.1us of fixed setup per chunk -- program
// launch, matmul_init's hardware startup, the scaler fill, the column of ones. Those now
// happen once for the head.
//
// The causal walk lives here rather than only in the mask: query chunk i sees k_offset +
// (i+1)*sq key tiles and no more, so chunks entirely above the diagonal are never visited.
// A rectangular score block cannot express the half-masked diagonal chunk, but it can skip
// the wholly-masked ones, and that is what the loop bound does.
//
// Compile-time args, all named, plus a cb_<name> per buffer that TT_U_CB reads:
//   Sq per query CHUNK, in tiles
//   Sk per key CHUNK, in tiles
//   D in tiles
//   number of query chunks
//   key tiles already behind the first query chunk (history; 0 for a fresh prefill)
//   total key tiles -- read only under FLASH_NONCAUSAL
//
// Define FLASH_NONCAUSAL to make every query chunk sweep the whole key range.
//
// No runtime args: the tensors are bound, so their addresses ride with the accessors.
//
// Q arrives ALREADY scaled by 1/sqrt(head dim): folding it into the operand costs the host one
// multiply and saves a broadcast pass per chunk on device.

#include <tt/unified/core>
#include "experimental/kernel_args.h"

namespace u = tt::unified;

void kernel_main() {
    constexpr uint32_t sq = get_named_compile_time_arg_val("sq");
    constexpr uint32_t sk = get_named_compile_time_arg_val("sk");
    constexpr uint32_t dt = get_named_compile_time_arg_val("dt");
    constexpr uint32_t num_q_chunks = get_named_compile_time_arg_val("num_q_chunks");

    constexpr uint32_t kCbQ = TT_U_CB(q);
    constexpr uint32_t kCbK = TT_U_CB(k);
    constexpr uint32_t kCbV = TT_U_CB(v);
    constexpr uint32_t kCbMask = TT_U_CB(mask);
    constexpr uint32_t kCbOne = TT_U_CB(one);
    constexpr uint32_t kCbColOnes = TT_U_CB(col_ones);
    constexpr uint32_t kCbMasked = TT_U_CB(masked);
    constexpr uint32_t kCbRowMax = TT_U_CB(row_max);
    constexpr uint32_t kCbProb = TT_U_CB(prob);
    constexpr uint32_t kCbRowSum = TT_U_CB(row_sum);
    constexpr uint32_t kCbPV = TT_U_CB(p_v);
    constexpr uint32_t kCbOScaled = TT_U_CB(o_scaled);
    constexpr uint32_t kCbCorrOld = TT_U_CB(corr_old);
    constexpr uint32_t kCbOut = TT_U_CB(out);
    constexpr uint32_t kCbM = TT_U_CB(m);
    constexpr uint32_t kCbL = TT_U_CB(l);
    constexpr uint32_t kCbO = TT_U_CB(o);
    constexpr uint32_t kCbRecipL = TT_U_CB(recip_l);
    constexpr uint32_t kCbMNow = TT_U_CB(m_now);
    // Key tiles already behind the first query chunk. Zero is a fresh prefill; a positive
    // value is prefill-with-history, where the queries see context they did not produce.
    constexpr uint32_t k_offset = get_named_compile_time_arg_val("k_offset");
    // The head's whole key range in tiles. The causal walk derives its per-chunk bound
    // from k_offset instead, but this is still what the head stride is measured in.
    constexpr uint32_t k_tiles = get_named_compile_time_arg_val("k_tiles");
    // GQA: n_heads query heads share n_kv_heads key/value heads, n_heads/n_kv_heads of
    // them per KV head. n_kv_heads == n_heads is ordinary multi-head attention and
    // n_kv_heads == 1 is multi-query; both fall out of the same mapping.
    //
    // These are the counts for the WHOLE tensor, not for this core. They fix the group
    // size and so the mapping; which heads this core walks is head_begin/head_count
    // below. A core holding four of thirty-two heads still needs to know there are
    // thirty-two, or it would map its heads onto the wrong KV heads entirely.
    constexpr uint32_t n_heads = get_named_compile_time_arg_val("n_heads");
    constexpr uint32_t n_kv_heads = get_named_compile_time_arg_val("n_kv_heads");
    // This core's slice of the heads. Runtime rather than compile-time because it is the
    // one thing that differs per core, and RUNTIME ARGS rather than a coordinate because
    // every projection reads the same values from them: a head range derived from
    // PhysicalCoord::this_core() would be right on the two data-movement threads and the
    // ORIGIN's range on compute, so the loads and the compute would disagree about how
    // many blocks exist and the circular buffers would deadlock. See the warning on
    // PhysicalCoord::this_core() in api.h. LogicalCoord::this_core() would be safe, but
    // the partition is a host policy and this keeps it there.
    const uint32_t head_begin = get_arg(args::head_begin);
    const uint32_t head_count = get_arg(args::head_count);

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

    // d_model in tiles. The output is ONE [S_q, d_model] activation tensor, with head h
    // occupying columns [h*dt, (h+1)*dt) -- the heads concatenated as they are written,
    // rather than stacked head-major for someone else to gather later. See the store at
    // the bottom of the head loop for why that costs nothing.
    constexpr uint32_t dm = n_heads * dt;

    static_assert(
        n_kv_heads > 0 && n_heads % n_kv_heads == 0,
        "GQA needs n_kv_heads to divide n_heads: every KV head serves the same number of "
        "query heads, and a remainder has no meaning");
    constexpr uint32_t kv_group = n_heads / n_kv_heads;
    // Blocks of sk tiles per head, which is the stride between one head's keys and the
    // next's. Derived rather than passed so it cannot disagree with the loop bound below.
    constexpr uint32_t k_blocks_per_head = k_tiles / sk;
    static_assert(k_blocks_per_head * sk == k_tiles, "sk must divide the head's key range");

    // Every q-chunk's key range has to divide into whole chunks, which for the causal
    // walk means sk must divide both the offset and sq. Checked here because getting it
    // wrong would silently run a short final chunk over the wrong tiles.
#if defined(FLASH_NONCAUSAL)
    static_assert(k_tiles % sk == 0, "sk must divide the key range");
#else
    // Query chunk i's key range is k_offset + (i+1)*sq tiles, and each has to be a whole
    // number of chunks. Consecutive ranges differ by sq, so it is enough that sk divides
    // the FIRST range, plus sq itself once there is more than one chunk to step by.
    static_assert(
        (k_offset + sq) % sk == 0 && (num_q_chunks == 1 || sq % sk == 0),
        "every query chunk's key range must divide into whole key chunks: sk has to divide "
        "k_offset + sq, and sq as well when there is more than one query chunk");
#endif

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

    const auto q_acc = TensorAccessor(tensor::q);
    const auto k_acc = TensorAccessor(tensor::k);
    const auto v_acc = TensorAccessor(tensor::v);
    const auto mask_acc = TensorAccessor(tensor::mask);
    const auto colones_acc = TensorAccessor(tensor::colones);
    const auto out = TensorAccessor(tensor::out);

    // Kernel scope, and this is the whole point of the q-loop: the scaler fill and the
    // column of ones happen ONCE for the head rather than once per query chunk, as does
    // matmul_init's hardware startup above. Measured, a separate launch per q-chunk cost
    // 13.1us of fixed work each; what remains per chunk below is only what genuinely
    // belongs to it -- its queries, its state, its tail.
    u::ComputeBlock one = u::fill_reduce_scaler<1>(one_storage, u::kReduceScalerOne);
    // reduce_max keeps the scaler above: a maximum has no matmul form.
    u::ComputeBlock col_ones = u::noc_load<0>(colones_storage, colones_acc, 0).wait();

    // One launch covers this core's whole slice of the heads, so everything above --
    // matmul_init's hardware startup, the reduce scaler, the column of ones -- is paid once
    // for the slice rather than once per head. Heads share nothing: no core reads another's
    // queries or writes another's output, so partitioning them across cores needs no
    // communication at all, only this range.
    //
    // Measured at sq=2 sk=4 dt=2 with two query chunks: a single head is 29.24us, and eight
    // heads fused are 201.20us against 233.94us as eight launches. That is 4.1us saved per
    // head, 14% at eight, and the per-head cost falls 29.24 -> 25.15us as the group grows.
    // (Not the 13.1us the q-loop saved per query chunk -- that was a different measurement
    // and it does not transfer; this one is the head loop's own.)
    for (uint32_t h = head_begin; h < head_begin + head_count; ++h) {
        // The GQA mapping, and the only place the two head counts meet: consecutive query
        // heads share a KV head, so query head h reads KV head h / kv_group.
        const uint32_t kv_head = h / kv_group;
        const uint32_t q_base = h * num_q_chunks;
        const uint32_t kv_base = kv_head * k_blocks_per_head;

        // Mask blocks are consumed in one flat sequence across the query nest, so the host
        // can lay them out in exactly that order and the kernel needs no two-dimensional
        // indexing. The sequence does not depend on the head -- a causal mask is the same
        // for every one -- so this resets here and the same blocks are re-read per head,
        // rather than the host storing n_heads identical copies.
        uint32_t mask_idx = 0;

        for (uint32_t i = 0; i < num_q_chunks; ++i) {
            // Causal: this chunk's queries see the history plus their own rows, and nothing
            // after. Chunks beyond that are entirely -inf, and skipping them is what the
            // rectangular block cannot express any other way.
#if defined(FLASH_NONCAUSAL)
        const uint32_t chunks = k_tiles / sk;
#else
        const uint32_t chunks = (k_offset + (i + 1) * sq) / sk;
#endif

        u::ComputeBlock q = u::noc_load<0>(q_storage, q_acc, q_base + i).wait();

        // The running state, per query chunk: fresh here, drained by the tail below.
        // ~RetainedBlock asserts on a slot that was pushed and never waited on, so a
        // forgotten drain on any iteration is caught rather than leaked.
        u::RetainedBlock<Vec> m_slot;
        u::RetainedBlock<Vec> l_slot;
        u::RetainedBlock<Out> o_slot;

        for (uint32_t j = 0; j < chunks; ++j) {
            u::ComputeBlock k = u::noc_load<0>(k_storage, k_acc, kv_base + j).wait();
            u::ComputeBlock v = u::noc_load<0>(v_storage, v_acc, kv_base + j).wait();
            u::ComputeBlock mask = u::noc_load<0>(mask_storage, mask_acc, mask_idx++).wait();

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

        // out = o / l, for THIS query chunk. Releasing every slot is not optional:
        // ~RetainedBlock asserts on a state that was pushed and never waited on, which is
        // exactly what a forgotten drain is -- and with the loop, a drain forgotten on one
        // iteration would otherwise corrupt the next.
        u::ComputeBlock<Vec> m_done = m_slot.release();
        u::ComputeBlock<Vec> l_done = l_slot.release();
        u::ComputeBlock<Out> o_done = o_slot.release();
        (void)sizeof(m_done);

        u::ComputeBlock rl = recipl_storage.store(u::recip(l_done));

        // The concat, done by the writer for free. This head's chunk is an [sq, dt]
        // rectangle inside an [S_q, d_model] tensor: rows [i*sq, +sq), columns
        // [h*dt, +dt). The built-in store would put it at one contiguous block index,
        // which is the head-major layout the projection then had to gather from -- and
        // gathering it there cost 30% of the projection, because for a fixed query
        // chunk the heads sit num_q_chunks blocks apart and cannot be read as one
        // operand.
        //
        // This costs NO extra NOC traffic. The built-in store already issues one write
        // per page, since consecutive pages of an interleaved tensor sit on different
        // banks, so all that changes is the destination page index each write is given.
        u::noc_store<1>(out_storage.store(o_done * u::bcast<u::Axis::Cols>(rl)), [&](u::L1Pages pages) {
            for (uint32_t p = 0; p < pages.count; ++p) {
                // The block is row-major in L1: page p is its tile (p / dt, p % dt).
                const uint32_t row = i * sq + p / dt;
                const uint32_t col = h * dt + p % dt;
                noc_async_write(pages.addr(p), out.get_noc_addr(row * dm + col), pages.page_bytes);
            }
        });
        }
    }
}
