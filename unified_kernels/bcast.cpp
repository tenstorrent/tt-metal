// SPDX-License-Identifier: Apache-2.0
//
// A broadcast: one block, one vector, expanded along a declared axis.
//
//   out = block <op> bcast<axis>(vec)
//
// The axis is a define, so one source covers all nine (op, axis) pairs. Which metal
// call each pair lowers to is what test_unified_bcast.py MEASURES -- metal's own
// documentation contradicts itself on the direction (its COL paragraph says both "a
// filled 0-column" and "C[h,w] = A[h,w] + B[w]"), so the mapping is established by
// numbers rather than by reading.
//
// Compile-time args, all named, plus a cb_<name> per buffer:
//   block height in tiles
//   block width in tiles
//
// No runtime args: the tensors are bound, so their addresses ride with the accessors.
//
// Defines: one of BC_AXIS_ROWS / BC_AXIS_COLS / BC_AXIS_BOTH, and one of
//          BC_OP_ADD / BC_OP_SUB / BC_OP_MUL.

#include <tt/unified/core>
#include "experimental/kernel_args.h"

namespace u = tt::unified;

#if defined(BC_AXIS_ROWS)
constexpr auto kAxis = u::Axis::Rows;
#elif defined(BC_AXIS_COLS)
constexpr auto kAxis = u::Axis::Cols;
#else
constexpr auto kAxis = u::Axis::Both;
#endif

#if defined(BC_OP_SUB)
#define BC_APPLY(b, v) ((b) - u::bcast<kAxis>(v))
#elif defined(BC_OP_MUL)
#define BC_APPLY(b, v) ((b) * u::bcast<kAxis>(v))
#else
#define BC_APPLY(b, v) ((b) + u::bcast<kAxis>(v))
#endif

void kernel_main() {
    constexpr uint32_t ht = get_arg(args::ht);
    constexpr uint32_t wt = get_arg(args::wt);

    constexpr uint32_t kCbBlock = get_arg(args::cb_block);
    constexpr uint32_t kCbVec = get_arg(args::cb_vec);
    constexpr uint32_t kCbOut = get_arg(args::cb_out);
    constexpr uint32_t kCbTmp = get_arg(args::cb_tmp);

    u::compute_init(kCbBlock, kCbOut);

    // The vector's shape is not stated: it is whatever the axis requires of the block,
    // which is the same shape a reduction along that axis produces.
    using In = u::Shape<ht, wt>;
    using Vec = u::reduce_shape<In, kAxis>;

    u::Storage<In> block_storage(kCbBlock);
    u::Storage<Vec> vec_storage(kCbVec);
    u::Storage<In> out_storage(kCbOut);

    const auto block_acc = TensorAccessor(tensor::block);
    const auto vec_acc = TensorAccessor(tensor::vec);
    const auto out = TensorAccessor(tensor::out);

    u::ComputeBlock b = u::noc_load<0>(block_storage, block_acc, 0).wait();
    u::ComputeBlock v = u::noc_load<0>(vec_storage, vec_acc, 0).wait();

#if defined(BC_THEN_SFPU)
    // A broadcast leaves the unpacker in a BROADCAST mode. An SFPU op afterwards has to
    // put it back or it reads the broadcast operand's replication instead of whole tiles.
    // Phase 4 gave every SFPU leaf its own copy_tile_to_dst_init_short; this path is what
    // proves that covers it.
    u::Storage<In> tmp_storage(kCbTmp);
    u::ComputeBlock t = tmp_storage.store(BC_APPLY(b, v));
    u::noc_store<1>(out_storage.store(u::relu(t + t)), out, 0);
#else
    u::noc_store<1>(out_storage.store(BC_APPLY(b, v)), out, 0);
#endif
}
