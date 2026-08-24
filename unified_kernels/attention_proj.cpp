// SPDX-License-Identifier: Apache-2.0
//
// The attention block's tail: head concat and the output projection, as ONE matmul.
//
// There is no concat pass, and that is the point. Concatenating heads and projecting is
//
//     concat(O_0 .. O_{H-1}) @ Wo   ==   sum over h of  O_h @ Wo_h
//
// where Wo_h is Wo's h-th row block of dt tiles. The concat only ever existed to put the
// heads side by side so a single matmul could see them; splitting the SAME matmul along
// its k dimension instead reaches the same sum with no data movement at all. So the
// projection is an ACCUMULATING matmul whose k-blocks are the heads, and "head concat"
// stops being an operation and becomes a choice of k-blocking.
//
// That also means the attention kernel's output layout needs no changing. It writes head
// h's query chunk i at block h * num_q_chunks + i -- head-major -- and this reads exactly
// those blocks, in the order its k-loop wants them. Writing a column-concatenated
// [S_q, n_heads * dt] instead would have made each head's chunk a STRIDED rectangle rather
// than contiguous pages, which is a strided store on one side and no benefit on the other.
//
// Wo needs no rearranging either: row block h of a row-major [d_model, d_model] is
// contiguous pages, so k-block h is one ordinary load.
//
// What each thread ends up executing, per query chunk:
//
//   NCRISC   fill cb_in0 (sq x dt, one head's chunk) and cb_in1 (dt x dm, one Wo block)
//   TRISC    matmul into the accumulator, n_heads times, then the finished sq x dm block
//   BRISC    drain cb_out (sq * dm tiles)
//
// The output block is sq * dm tiles, which for any real d_model is far past the 8-tile DST
// budget -- 2 x 8 is 16 tiles at four 64-wide heads. It only works because the accumulating
// path walks the output in subblocks; before that it could not have been expressed at all.
//
// Compile-time args:
//   0        sq          query tiles per chunk
//   1        dt          head dim in tiles
//   2        dm          d_model in tiles (n_heads * dt)
//   3        num_q_chunks
//   4        n_heads
//   5..      TensorAccessorArgs for attn_out, then wo, then out
//
// Runtime args (identical on all three kernels):
//   0        attn_out base address (the attention kernel's output, head-major)
//   1        wo base address
//   2        out base address
//   3        first query chunk this core owns
//   4        how many query chunks this core owns

#include <tt/unified/core>

namespace u = tt::unified;

constexpr uint32_t kCbIn0 = 0;
constexpr uint32_t kCbIn1 = 1;
constexpr uint32_t kCbOut = 16;
constexpr uint32_t kCbAcc = 24;  // running total; a separate CB from kCbOut

void kernel_main() {
    constexpr uint32_t sq = get_compile_time_arg_val(0);
    constexpr uint32_t dt = get_compile_time_arg_val(1);
    constexpr uint32_t dm = get_compile_time_arg_val(2);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t n_heads = get_compile_time_arg_val(4);

    constexpr auto attn_args = TensorAccessorArgs<5>();
    constexpr auto wo_args = TensorAccessorArgs<attn_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<wo_args.next_compile_time_args_offset()>();

    const uint32_t attn_addr = get_arg_val<uint32_t>(0);
    const uint32_t wo_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    // Query chunks are what gets partitioned here, not heads: a head is a k-block of this
    // matmul, so a core cannot own part of one without owning a partial sum that someone
    // then has to reduce. Chunks are rows of the output and stay independent.
    const uint32_t chunk_begin = get_arg_val<uint32_t>(3);
    const uint32_t chunk_count = get_arg_val<uint32_t>(4);

    static_assert(dm == n_heads * dt, "d_model must be exactly n_heads head-dims wide");

    using A = u::Shape<sq, dt>;  // one head's query chunk
    using B = u::Shape<dt, dm>;  // one head's row block of Wo
    using Out = u::Shape<sq, dm>;

    u::matmul_init<A, B>(kCbIn0, kCbIn1, kCbOut);

    u::Storage<A> a_storage(kCbIn0);
    u::Storage<B> b_storage(kCbIn1);
    u::Storage<Out> acc_storage(kCbAcc);
    u::Storage<Out> out_storage(kCbOut);

    const auto attn = TensorAccessor(attn_args, attn_addr);
    const auto wo = TensorAccessor(wo_args, wo_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    u::Accumulator<Out, u::AccumulatorMode::Dst> acc(acc_storage, out_storage);

    for (uint32_t c = 0; c < chunk_count; ++c) {
        const uint32_t i = chunk_begin + c;
        acc.clear();

        for (uint32_t h = 0; h < n_heads; ++h) {
            const bool finish = (h == n_heads - 1);

            // The concat, such as it is: head h's chunk i, at the block index the attention
            // kernel wrote it to. Nothing is gathered or copied -- the k-loop just visits
            // the heads.
            u::ComputeBlock a = u::noc_load<1>(a_storage, attn, h * num_q_chunks + i).wait();
            u::ComputeBlock b = u::noc_load<1>(b_storage, wo, h).wait();

            u::Block result = acc.accumulate(u::matmul(a, b), finish);
            if (finish) {
                u::noc_store<0>(std::move(result), out, i);
            }
        }
    }
}
