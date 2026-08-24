// SPDX-License-Identifier: Apache-2.0
//
// The attention block's tail: the output projection, [S_q, d_model] @ Wo.
//
// There is no head concat here, and no k-loop over heads. Both are gone because the attention
// kernel WRITES the concatenated layout: it stores head h's query chunk as an [sq, dt]
// rectangle at columns [h*dt, +dt) of one [S_q, d_model] tensor, which costs it nothing -- the
// built-in store already issues one write per page, since consecutive pages of an interleaved
// tensor sit on different banks, so only the destination index changes. By the time the
// projection runs the activation is already one contiguous operand.
//
// An earlier version recovered the concat arithmetically instead, as an accumulating matmul
// whose k-blocks were the HEADS. That needed no strided store and cost 30%: for a fixed query
// chunk the heads sit num_q_chunks blocks apart, so it was n_heads accumulate CALLS where one
// would do, 19.67us against 13.80us a chunk. Per-call pass overhead, not the k-blocking itself.
//
// K-BLOCKING, which is what lets this reach a real d_model. Wo is dm*dm tiles: 64 tiles at
// d_model 256, but 4096 tiles -- 8MB -- at d_model 2048, far past L1. So K is split into kb
// blocks of kt tiles and the matmul accumulates over them. The two operands split differently,
// and that asymmetry is the whole content of this kernel:
//
//   Wo   block b is ROWS [b*kt, +kt) of a row-major [dm, dm], which is contiguous pages.
//        An ordinary block load, index b.
//   A    block b is COLUMNS [b*kt, +kt) of an [sq, dm] activation, which is NOT contiguous:
//        row r's dm tiles are contiguous, so a k-slice of it is strided. A custom load, at
//        the same page count -- one read per page either way, exactly as with the store.
//
// kt == dm means kb == 1, and that takes the single-shot path instead: with one k-block there
// is no partial to carry, and going through the accumulator would pay a pack and a reload for
// nothing.
//
// What each thread ends up executing, per query chunk:
//
//   NCRISC   per k-block: cb_in (sq x kt, strided) and cb_wo (kt x dm, contiguous)
//   TRISC    matmul per k-block into the accumulator, then the sq x dm block in subblocks
//   BRISC    drain cb_out (sq * dm tiles) per chunk
//
// Compile-time args:
//   0        sq          query tiles per chunk
//   1        dm          d_model in tiles
//   2        num_q_chunks
//   3        kt          tiles per k-block; kt == dm is the single-shot case
//   4..      TensorAccessorArgs for attn_out, then wo, then out
//
// Runtime args (identical on all three kernels):
//   0        attn_out base address -- the attention kernel's [S_q, d_model] output
//   1        wo base address
//   2        out base address
//   3        first query chunk this core owns
//   4        how many query chunks this core owns

#include <tt/unified/core>

namespace u = tt::unified;

constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbWo = 1;
constexpr uint32_t kCbOut = 16;
constexpr uint32_t kCbAcc = 24;  // running total; a separate CB from kCbOut

void kernel_main() {
    constexpr uint32_t sq = get_compile_time_arg_val(0);
    constexpr uint32_t dm = get_compile_time_arg_val(1);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    (void)num_q_chunks;  // the chunk range comes from runtime args; this documents the whole

    static_assert(kt > 0 && dm % kt == 0, "the k-block width must divide d_model");
    constexpr uint32_t kb = dm / kt;

    constexpr auto attn_args = TensorAccessorArgs<4>();
    constexpr auto wo_args = TensorAccessorArgs<attn_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<wo_args.next_compile_time_args_offset()>();

    const uint32_t attn_addr = get_arg_val<uint32_t>(0);
    const uint32_t wo_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    // Query chunks are the unit of work: they are rows of the output and stay independent,
    // so a core's share needs no reduction with anyone else's.
    const uint32_t chunk_begin = get_arg_val<uint32_t>(3);
    const uint32_t chunk_count = get_arg_val<uint32_t>(4);

    using A = u::Shape<sq, kt>;  // one k-slice of one query chunk
    using W = u::Shape<kt, dm>;  // one row block of Wo
    using Out = u::Shape<sq, dm>;

    u::matmul_init<A, W>(kCbIn, kCbWo, kCbOut);

    u::Storage<A> a_storage(kCbIn);
    u::Storage<W> w_storage(kCbWo);
    u::Storage<Out> acc_storage(kCbAcc);
    u::Storage<Out> out_storage(kCbOut);

    const auto attn = TensorAccessor(attn_args, attn_addr);
    const auto wo = TensorAccessor(wo_args, wo_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    // Dst mode reloads the running total into DST before every k-block and packs it back
    // after, which costs O(output block) per k-block -- and this output block is sq*dm
    // tiles, 128 at sq=2 and d_model 2048. L1 mode lets the PACKER add into the partial
    // instead, so the total never enters DST at all: one pack per k-block rather than a
    // copy-in and a pack. See the numbers in unified_llama_prefill.md.
#if defined(PROJ_ACC_DST)
    u::Accumulator<Out, u::AccumulatorMode::Dst> acc(acc_storage, out_storage);
#else
    u::Accumulator<Out, u::AccumulatorMode::L1> acc(acc_storage, out_storage);
#endif

    for (uint32_t c = 0; c < chunk_count; ++c) {
        const uint32_t i = chunk_begin + c;
        acc.clear();

        for (uint32_t b = 0; b < kb; ++b) {
            const bool finish = (b == kb - 1);

            // A's k-slice: columns [b*kt, +kt) of rows [i*sq, +sq). Strided, because the
            // activation's rows are what is contiguous. Same number of reads as a
            // contiguous block -- only the addresses differ.
            u::ComputeBlock a =
                u::noc_load<1>(a_storage, [&](u::L1Pages pages) {
                    for (uint32_t p = 0; p < pages.count; ++p) {
                        // The block is row-major in L1: page p is its tile
                        // (p / kt, p % kt).
                        const uint32_t row = i * sq + p / kt;
                        const uint32_t col = b * kt + p % kt;
                        noc_async_read(attn.get_noc_addr(row * dm + col), pages.addr(p), pages.page_bytes);
                    }
                }).wait();
            // Wo's row block, which IS contiguous.
            u::ComputeBlock w = u::noc_load<1>(w_storage, wo, b).wait();

            if constexpr (kb == 1) {
                // One k-block: nothing to carry, so skip the accumulator's pack and reload.
                (void)finish;
                u::noc_store<0>(out_storage.store(u::matmul(a, w)), out, i);
            } else {
                u::Block result = acc.accumulate(u::matmul(a, w), finish);
                if (finish) {
                    u::noc_store<0>(std::move(result), out, i);
                }
            }
        }
    }
}
