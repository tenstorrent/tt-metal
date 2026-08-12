// SPDX-License-Identifier: Apache-2.0
//
// A unified/single-threaded programming model built on top of the existing metal
// programming model. Metal typically has 2 DM threads and 1 compute thread
// (which is then split into 3, but this header does not concern itself with
// compute thread splitting and treats compute as the abstraction metal provides
// -- this is just an extension).
//
// One kernel source describes the whole pipeline. It is compiled once per baby
// RISC-V thread, and each statement lowers to that thread's half of the
// circular-buffer protocol:
//
//   INPUT                    OUTPUT                   INTERMED
//        DM    Compute            DM    Compute            DM    Compute
//   reserve <- *               * -> reserve                   reserve
//     write                          write                      write
//      push ->    wait         wait <-  push                     push
//                read          read                              wait
//         * <-     pop          pop -> *                         read
//                                                                 pop
//
// Layering:
//   unified_expr.hpp  -- domain-free expression tree + DST register allocator
//   fusion.hpp        -- leaves, ops, fusion kinds, driver strategies
//   unified.hpp       -- this file: the core API
//
// Core API:
//   Storage       a circular buffer. `store()` evaluates a compute fusion into it.
//   Block         move-only evidence that something was produced into a Storage.
//   ComputeBlock  compute-side consumption of a Block; a leaf in an expression.
//   noc_*         the data-movement operations, each pinned to a DM thread.

#pragma once

#include <type_traits>
#include <utility>

#include "fusion.hpp"

namespace tt {
namespace unified {

class Tensor;
struct Block;
class ComputeBlock;

struct Coord {
    int y;
    int x;
};

struct Shape {
    int h;
    int w;
};

struct Mcast {
    Coord coord;
    Shape shape;
};

// ---------------------------------------------------------------------------
// Storage -- a circular buffer
// ---------------------------------------------------------------------------

struct Storage {
    Storage(int cb_id, int num_tiles) : cb_id(cb_id), num_tiles(num_tiles) {}

    Storage(Storage&&) = delete;
    Storage(const Storage&) = delete;
    Storage& operator=(Storage&&) = delete;
    Storage& operator=(const Storage&) = delete;

    // Evaluate a compute fusion into this buffer. The loop shape is chosen by
    // the fusion's kind; see Strategy in fusion.hpp. Defined out-of-line below,
    // once Block is complete.
    template <typename Node>
    Block store(const Node& node);

    int cb_id;
    int num_tiles;  // This could eventually be N dimensional (maybe via template params?)
};

// ---------------------------------------------------------------------------
// Block -- move-only evidence that a Storage was produced into
//
// Every Block comes from an operation that has already pushed, which is what
// makes it safe to hand one to a DM thread to drain. Move-only so it reaches
// exactly one consumer; consumers take it by value.
// ---------------------------------------------------------------------------

struct Block {
    explicit Block(const Storage& storage) : cb_id(storage.cb_id), num_tiles(storage.num_tiles) {}
    Block(int cb_id, int num_tiles) : cb_id(cb_id), num_tiles(num_tiles) {}

    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;

    // TODO: does not disengage the source, so a moved-from Block is
    // indistinguishable from a live one and a second consumer silently issues a
    // duplicate cb_pop. See the debug-guard note in the design discussion.
    Block(Block&& o) {
        cb_id = o.cb_id;
        num_tiles = o.num_tiles;
    }

    Block& operator=(Block&& o) {
        cb_id = o.cb_id;
        num_tiles = o.num_tiles;
        return *this;
    }

    int cb_id;
    int num_tiles;  // This could eventually be N dimensional (maybe via template params?)
};

template <typename Node>
Block Storage::store(const Node& node) {
    Strategy<expr::kind_of_t<Node>>::run(node, cb_id, num_tiles);
    return Block(cb_id, num_tiles);
}

// ---------------------------------------------------------------------------
// ComputeBlock -- compute-side consumption of a Block
// ---------------------------------------------------------------------------

class ComputeBlock {
public:
    ComputeBlock(Block block) : cb_id(block.cb_id), num_tiles(block.num_tiles) {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        cb_wait(cb_id);
#endif
    }

    ~ComputeBlock() {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        cb_pop(cb_id);
#endif
    }

    ComputeBlock(const ComputeBlock&) = delete;
    ComputeBlock& operator=(const ComputeBlock&) = delete;
    ComputeBlock(ComputeBlock&&) = delete;
    ComputeBlock& operator=(ComputeBlock&&) = delete;

    int get_cb_id() const { return cb_id; }

    int get_num_tiles() const { return num_tiles; }

    // TODO: reduces within a tile, not across them. A cross-tile reduction wants
    // its own Strategy that accumulates over the tile loop and packs once.
    expr::Un<SumOp, TileSource> sum() const { return {ld()}; }

private:
    int cb_id;
    int num_tiles;  // This could eventually be N dimensional (maybe via template params?)
};

// ---------------------------------------------------------------------------
// Adaptors that let a ComputeBlock stand in for an expression leaf.
// These are the hooks fusion.hpp declares; they live here because they are the
// only place the fusion layer needs to know about a core type.
// ---------------------------------------------------------------------------

inline TileSource as_node(const ComputeBlock& b) { return TileSource{b.get_num_tiles()}; }

inline auto relu(const ComputeBlock& b) { return expr::Un<ReluOp, TileSource>{as_node(b)}; }

template <typename Geometry>
auto matmul(const ComputeBlock& a, const ComputeBlock& b) {
    return matmul<Geometry>(as_node(a), as_node(b));
}

// ---------------------------------------------------------------------------
// Data movement. Each is pinned to a DM thread by its `thread` argument, and
// compiles away entirely on every other thread.
// ---------------------------------------------------------------------------

template <int thread>
Block noc_load(const Storage& storage, const Tensor& t, int idx) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_reserve(storage.cb_id);
        noc_async_read();
        cb_push(storage.cb_id);
    }
#endif
    return Block(storage);
}

template <int thread>
Block noc_load_mcast(const Storage& storage, Mcast mcast, const Tensor& t, int idx);

template <int thread>
void noc_store(Block block, const Tensor& t, int idx) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait(block.cb_id);
        noc_async_write();
        cb_pop(block.cb_id);
    }
#endif
}

// TODO: core-to-core movement. `cb_wait(block.cb_id)` is correct (the source is
// local) but cb_reserve/cb_push on the *destination* id are not -- CB ids are
// per-core, so the peer's pointers have to be updated over the NOC. See
// api/remote_circular_buffer.h (remote_cb_reserve_back /
// remote_cb_push_back_and_write_pages), which is asymmetric between sender and
// receiver, or the explicit semaphore handshake the matmul mcast kernels use.
template <int thread>
Block noc_read(const Storage& storage, Block block, Coord coord, int offset) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait(block.cb_id);
        cb_reserve(storage.cb_id);
        noc_async_read();
        cb_push(storage.cb_id);
        cb_pop(block.cb_id);
    }
#endif
    return Block(storage);
}

template <int thread>
Block noc_write(const Storage& storage, Block block, Coord coord, int offset) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait(block.cb_id);
        cb_reserve(storage.cb_id);
        noc_async_write();
        cb_push(storage.cb_id);
        cb_pop(block.cb_id);
    }
#endif
    return Block(storage);
}

// ===========================================================================
// Examples
// ===========================================================================

void test() {
    Storage lhs_storage(0, 2);
    Storage rhs_storage(1, 2);
    Storage tmp_storage(2, 2);
    Storage out_storage(3, 2);

    for (int i = 0; i < 1; ++i) {
        ComputeBlock lhs = noc_load<0>(lhs_storage, t0, i);
        ComputeBlock rhs = noc_load<1>(rhs_storage, t1, i);

        ComputeBlock tmp = tmp_storage.store(lhs + rhs);

        Block result = out_storage.store(tmp + lhs);
        noc_store<0>(std::move(result), t2, i);
    }
}

void reduce() {
    Storage stage0_storage(0, 8);
    Storage stage1_storage(1, 8);
    Storage tmp_storage(2, 2);
    Storage out1_storage(3, 8);

    for (int i = 0; i < 1; ++i) {
        ComputeBlock s0 = noc_load<0>(stage0_storage, t0, i);

        Block tmp = tmp_storage.store(s0.sum());

        ComputeBlock s1 = noc_write<0>(stage1_storage, std::move(tmp), coord_0x0, offset);

        noc_store<0>(out1_storage.store(s1.sum()), t2, i);
    }
}

// The FPU path: matmul with a fused relu epilogue, then the SFPU path consuming
// its result out of an intermediate Storage.
void matmul_relu() {
    Storage a_storage(0, 1);
    Storage b_storage(1, 1);
    Storage mm_storage(2, 1);
    Storage out_storage(3, 1);

    // out_subblock 2x2 = 4 DST tiles, k-dim 2, 2 inner blocks
    using Geom = MatmulGeometry</*h=*/2, /*w=*/2, /*in0_block_w=*/2, /*num_blocks=*/2>;

    ComputeBlock a = noc_load<0>(a_storage, t0, 0);
    ComputeBlock b = noc_load<1>(b_storage, t1, 0);

    // relu folds into the matmul's pack-side epilogue rather than wrapping it
    ComputeBlock mm = mm_storage.store(relu(matmul<Geom>(a, b)));

    // ... and the SFPU path picks it up from there
    noc_store<0>(out_storage.store(mm + a), t2, 0);
}

}  // namespace unified
}  // namespace tt
