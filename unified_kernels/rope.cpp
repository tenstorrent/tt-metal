// SPDX-License-Identifier: Apache-2.0
//
// Rotary position embedding, llama flavour:
//
//     out = x * cos + (x @ M) * sin
//
// M is a single 32x32 tile with M[2i][2i+1] = +1 and M[2i+1][2i] = -1, so `x @ M` maps each
// adjacent pair (x[2i], x[2i+1]) to (-x[2i+1], x[2i]) -- the rotate-half, as an ordinary
// matmul. cos and sin are full blocks with each value duplicated across its pair.
//
// TWO THINGS MAKE THIS FIT THE MODEL UNCHANGED.
//
// The rotation is PER TILE: every output tile depends only on the matching input tile,
// because the pairing never crosses a 32-element boundary. A block matmul expresses that
// exactly when kt_dim is 1 -- out(rt x 1) = A(rt x 1) @ B(1 x 1) has no sum over k, so each
// output tile is one input tile times the single M tile. ttnn spells the same thing as a
// matmul_tiles loop.
//
// And because the op is per-tile, the block's 2-D shape is irrelevant: a chunk of N tiles
// is declared Shape<N, 1> whatever the sequence and head dimensions actually are. N is
// capped at 8 by the matmul's DST budget, so the kernel walks the tensor in chunks.
//
// The whole rotation then lands in one SFPU pass: `x * cos + rot * sin` is a four-leaf
// tree, the deepest in this model so far.
//
// Compile-time args:
//   0        tiles per chunk (at most 8)
//   1        chunks
//   2..      TensorAccessorArgs for x, cos, sin, trans_mat, then out
//
// Runtime args (identical on all three kernels):
//   0..4     x, cos, sin, trans_mat, out base addresses

#include <tt/unified/core>

namespace u = tt::unified;

constexpr uint32_t kCbX = 0;
constexpr uint32_t kCbCos = 1;
constexpr uint32_t kCbSin = 2;
constexpr uint32_t kCbM = 3;
constexpr uint32_t kCbRot = 4;
constexpr uint32_t kCbOut = 16;

void kernel_main() {
    constexpr uint32_t chunk = get_compile_time_arg_val(0);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(1);

    constexpr auto x_args = TensorAccessorArgs<2>();
    constexpr auto cos_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    const uint32_t m_addr = get_arg_val<uint32_t>(3);
    const uint32_t out_addr = get_arg_val<uint32_t>(4);

    using Blk = u::Shape<chunk, 1>;  // N tiles, shape irrelevant to a per-tile op
    using M = u::Shape<1, 1>;

    u::matmul_init<Blk, M>(kCbX, kCbM, kCbOut);

    u::Storage<Blk> x_storage(kCbX);
    u::Storage<Blk> cos_storage(kCbCos);
    u::Storage<Blk> sin_storage(kCbSin);
    u::Storage<M> m_storage(kCbM);
    u::Storage<Blk> rot_storage(kCbRot);
    u::Storage<Blk> out_storage(kCbOut);

    const auto x_acc = TensorAccessor(x_args, x_addr);
    const auto cos_acc = TensorAccessor(cos_args, cos_addr);
    const auto sin_acc = TensorAccessor(sin_args, sin_addr);
    const auto m_acc = TensorAccessor(m_args, m_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    // KERNEL SCOPE: every chunk's matmul re-reads the same rotation tile, so it must not be
    // popped until the kernel ends -- the same rule the reduce scaler and a fused bias obey.
    u::ComputeBlock m = u::noc_load<0>(m_storage, m_acc, 0).wait();

    for (uint32_t c = 0; c < num_chunks; ++c) {
        u::ComputeBlock x = u::noc_load<0>(x_storage, x_acc, c).wait();
        u::ComputeBlock cos = u::noc_load<0>(cos_storage, cos_acc, c).wait();
        u::ComputeBlock sin = u::noc_load<0>(sin_storage, sin_acc, c).wait();

        // The rotation. kt_dim is 1, so this is per-tile rather than a sum over k.
        u::ComputeBlock rot = rot_storage.store(u::matmul(x, m));

        // x * cos + rot * sin, in ONE pass: four leaves, three DST slots.
        u::noc_store<1>(out_storage.store(x * cos + rot * sin), out, c);
    }
}
