// SPDX-License-Identifier: Apache-2.0
//
// The attention block's tail: the output projection, [S_q, d_model] @ Wo.
//
// There is no head concat here, and no k-loop over heads either. Both are gone because the
// attention kernel now WRITES the concatenated layout: it stores head h's query chunk as an
// [sq, dt] rectangle at columns [h*dt, +dt) of one [S_q, d_model] tensor, which costs it
// nothing -- the built-in store already issues one write per page, since consecutive pages
// of an interleaved tensor sit on different banks, so only the destination index changes.
// By the time the projection runs the activation is already one contiguous operand and this
// is an ordinary matmul.
//
// It did not start that way. The first version left the attention output head-major and
// recovered the concat as a k-loop over heads -- sum over h of O_h @ Wo_h, arithmetically
// the same and needing no strided store. That cost 30% of the projection: for a fixed query
// chunk the heads sit num_q_chunks blocks apart, so it was four accumulate CALLS where one
// would do, 19.67us against 13.80us a chunk. The k-blocking was not expensive in itself --
// the per-call pass overhead was.
//
// Wo is loaded ONCE and stays resident for every chunk, like the reduce scaler and the
// column of ones in the attention kernel.
//
// LIMIT, and it is why this is not yet a general projection: the whole of Wo lives in L1,
// dm*dm tiles -- 64 tiles at d_model 256, but 4096 tiles (8MB) at d_model 2048, far past L1.
// A real d_model needs the matmul k-blocked over slices of dm, and the activation operand
// then becomes strided (row r's dm tiles are contiguous, so a k-slice of it is not), which
// noc_load's Fn form can express exactly as the attention store does. The per-call cost
// measured above is what that would pay, and it is what any implementation pays for a matmul
// too large to hold.
//
// What each thread ends up executing:
//
//   NCRISC   fill cb_wo once (dm x dm), then cb_in (sq x dm) per chunk
//   TRISC    one matmul per chunk, packing the sq x dm output block in subblocks
//   BRISC    drain cb_out (sq * dm tiles) per chunk
//
// Compile-time args:
//   0        sq          query tiles per chunk
//   1        dm          d_model in tiles
//   2        num_q_chunks
//   3..      TensorAccessorArgs for attn_out, then wo, then out
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

void kernel_main() {
    constexpr uint32_t sq = get_compile_time_arg_val(0);
    constexpr uint32_t dm = get_compile_time_arg_val(1);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(2);
    (void)num_q_chunks;  // the chunk range comes from runtime args; this documents the whole

    constexpr auto attn_args = TensorAccessorArgs<3>();
    constexpr auto wo_args = TensorAccessorArgs<attn_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<wo_args.next_compile_time_args_offset()>();

    const uint32_t attn_addr = get_arg_val<uint32_t>(0);
    const uint32_t wo_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    // Query chunks are the unit of work: they are rows of the output and stay independent,
    // so a core's share needs no reduction with anyone else's.
    const uint32_t chunk_begin = get_arg_val<uint32_t>(3);
    const uint32_t chunk_count = get_arg_val<uint32_t>(4);

    using A = u::Shape<sq, dm>;  // one query chunk of the concatenated activation
    using W = u::Shape<dm, dm>;  // the whole projection matrix
    using Out = u::Shape<sq, dm>;

    u::matmul_init<A, W>(kCbIn, kCbWo, kCbOut);

    u::Storage<A> a_storage(kCbIn);
    u::Storage<W> w_storage(kCbWo);
    u::Storage<Out> out_storage(kCbOut);

    const auto attn = TensorAccessor(attn_args, attn_addr);
    const auto wo = TensorAccessor(wo_args, wo_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    // Once, for every chunk this core owns. Declared outside the loop deliberately: a
    // ComputeBlock pops in its destructor, so one declared inside would be popped after a
    // single use and the next chunk would wait for a refill that never comes.
    u::ComputeBlock w = u::noc_load<1>(w_storage, wo, 0).wait();

    for (uint32_t c = 0; c < chunk_count; ++c) {
        const uint32_t i = chunk_begin + c;
        u::ComputeBlock a = u::noc_load<1>(a_storage, attn, i).wait();
        // Single-shot: one k-block, so no accumulation buffer and no reload. The output
        // block is sq * dm tiles -- 16 at sq=2 and d_model 256 -- which the single-shot
        // path walks in subblocks.
        u::noc_store<0>(out_storage.store(u::matmul(a, w)), out, i);
    }
}
