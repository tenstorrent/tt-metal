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
// Compile-time args, all named, plus a cb_<name> per buffer:
//   tiles per chunk (at most 8)
//   chunks
//
// Runtime args, named and identical on all three kernels:
//   chunk_begin    first chunk this core owns
//   chunk_count    how many it owns
//
// Chunks are the unit of partitioning and split with no coordination: the rotation is
// per-tile, so chunk c depends on nothing outside its own tiles of x, cos and sin and
// writes only its own tiles of the output. num_chunks stays compile-time because it is what
// the host divides, not what a core walks. The rotation tile is read by every core.

#include <tt/unified/core>
#include "experimental/kernel_args.h"

namespace u = tt::unified;

void kernel_main() {
    constexpr uint32_t chunk = get_named_compile_time_arg_val("chunk");

    constexpr uint32_t kCbX = get_named_compile_time_arg_val("cb_x");
    constexpr uint32_t kCbCos = get_named_compile_time_arg_val("cb_cos");
    constexpr uint32_t kCbSin = get_named_compile_time_arg_val("cb_sin");
    constexpr uint32_t kCbM = get_named_compile_time_arg_val("cb_m");
    constexpr uint32_t kCbRot = get_named_compile_time_arg_val("cb_rot");
    constexpr uint32_t kCbOut = get_named_compile_time_arg_val("cb_out");
    [[maybe_unused]] constexpr uint32_t num_chunks = get_named_compile_time_arg_val("num_chunks");
    const uint32_t chunk_begin = get_arg(args::chunk_begin);
    const uint32_t chunk_count = get_arg(args::chunk_count);

    using Blk = u::Shape<chunk, 1>;  // N tiles, shape irrelevant to a per-tile op
    using M = u::Shape<1, 1>;

    u::matmul_init<Blk, M>(kCbX, kCbM, kCbOut);

    u::Storage<Blk> x_storage(kCbX);
    u::Storage<Blk> cos_storage(kCbCos);
    u::Storage<Blk> sin_storage(kCbSin);
    u::Storage<M> m_storage(kCbM);
    u::Storage<Blk> rot_storage(kCbRot);
    u::Storage<Blk> out_storage(kCbOut);

    const auto x_acc = TensorAccessor(tensor::x);
    const auto cos_acc = TensorAccessor(tensor::cos);
    const auto sin_acc = TensorAccessor(tensor::sin);
    const auto m_acc = TensorAccessor(tensor::m);
    const auto out = TensorAccessor(tensor::out);

    // KERNEL SCOPE: every chunk's matmul re-reads the same rotation tile, so it must not be
    // popped until the kernel ends -- the same rule the reduce scaler and a fused bias obey.
    u::ComputeBlock m = u::noc_load<0>(m_storage, m_acc, 0).wait();

    for (uint32_t n = 0; n < chunk_count; ++n) {
        const uint32_t c = chunk_begin + n;
        u::ComputeBlock x = u::noc_load<0>(x_storage, x_acc, c).wait();
        u::ComputeBlock cos = u::noc_load<0>(cos_storage, cos_acc, c).wait();
        u::ComputeBlock sin = u::noc_load<0>(sin_storage, sin_acc, c).wait();

        // The rotation. kt_dim is 1, so this is per-tile rather than a sum over k.
        u::ComputeBlock rot = rot_storage.store(u::matmul(x, m));

        // x * cos + rot * sin, in ONE pass: four leaves, three DST slots.
        u::noc_store<1>(out_storage.store(x * cos + rot * sin), out, c);
    }
}
