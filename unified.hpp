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

// Binds the model's intrinsics to a backend. Define TT_UNIFIED_CUSTOM_BINDING to
// supply your own (the host trace harness does) before including this header.
#ifndef TT_UNIFIED_CUSTOM_BINDING
#include "unified_metal.hpp"
#endif

#include "fusion.hpp"

namespace tt {
namespace unified {

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
        cb_wait(cb_id, num_tiles);
#endif
    }

    ~ComputeBlock() {
#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
        cb_pop(cb_id, num_tiles);
#endif
    }

    ComputeBlock(const ComputeBlock&) = delete;
    ComputeBlock& operator=(const ComputeBlock&) = delete;
    ComputeBlock(ComputeBlock&&) = delete;
    ComputeBlock& operator=(ComputeBlock&&) = delete;

    int get_cb_id() const { return cb_id; }

    int get_num_tiles() const { return num_tiles; }

    expr::Un<ExpOp, TileSource> exp() const { return {TileSource{cb_id}}; }

private:
    int cb_id;
    int num_tiles;  // This could eventually be N dimensional (maybe via template params?)
};

// ---------------------------------------------------------------------------
// Adaptors that let a ComputeBlock stand in for an expression leaf.
// These are the hooks fusion.hpp declares; they live here because they are the
// only place the fusion layer needs to know about a core type.
// ---------------------------------------------------------------------------

// Declares ComputeBlock usable as an expression operand; without this the
// operator+ in fusion.hpp is SFINAE'd out and `lhs + rhs` does not resolve.
template <>
struct is_operand<ComputeBlock> : std::true_type {};

// TileSource identifies a circular buffer, so this must be the cb id.
inline TileSource as_node(const ComputeBlock& b) { return TileSource{b.get_cb_id()}; }

inline auto relu(const ComputeBlock& b) { return expr::Un<ReluOp, TileSource>{as_node(b)}; }

inline auto exp_(const ComputeBlock& b) { return expr::Un<ExpOp, TileSource>{as_node(b)}; }

template <typename Geometry>
auto matmul(const ComputeBlock& a, const ComputeBlock& b) {
    return matmul<Geometry>(as_node(a), as_node(b));
}

// ---------------------------------------------------------------------------
// Data movement. Each is pinned to a DM thread by its `thread` argument, and
// compiles away entirely on every other thread.
// ---------------------------------------------------------------------------

// Reads `storage.num_tiles` pages into the buffer, starting at page
// `block_idx * storage.num_tiles`, then pushes.
//
// The accessor comes from make_accessor(); on the compute projection that is a
// NullAccessor, since a real one cannot be built on a TRISC.
template <int thread, typename Accessor>
Block noc_load(const Storage& storage, const Accessor& acc, int block_idx) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_reserve(storage.cb_id, storage.num_tiles);
        uint32_t l1 = cb_write_addr(storage.cb_id);
        const uint32_t bytes = cb_page_bytes(storage.cb_id);
        const uint32_t first = static_cast<uint32_t>(block_idx * storage.num_tiles);
        for (int p = 0; p < storage.num_tiles; ++p) {
            noc_read_page(acc, first + static_cast<uint32_t>(p), l1, bytes);
            l1 += bytes;
        }
        noc_read_barrier();
        cb_push(storage.cb_id, storage.num_tiles);
    }
#else
    (void)acc;
    (void)block_idx;
#endif
    return Block(storage);
}

// Drains a Block to a tensor. Takes the Block by value: this call consumes it.
template <int thread, typename Accessor>
void noc_store(Block block, const Accessor& acc, int block_idx) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait(block.cb_id, block.num_tiles);
        uint32_t l1 = cb_read_addr(block.cb_id);
        const uint32_t bytes = cb_page_bytes(block.cb_id);
        const uint32_t first = static_cast<uint32_t>(block_idx * block.num_tiles);
        for (int p = 0; p < block.num_tiles; ++p) {
            noc_write_page(acc, first + static_cast<uint32_t>(p), l1, bytes);
            l1 += bytes;
        }
        noc_write_barrier();
        cb_pop(block.cb_id, block.num_tiles);
    }
#else
    (void)block;
    (void)acc;
    (void)block_idx;
#endif
}

template <int thread, typename Accessor>
Block noc_load_mcast(const Storage& storage, Mcast mcast, const Accessor& acc, int block_idx);

// ---------------------------------------------------------------------------
// Core-to-core movement.
//
// Pulls a peer core's block into this core's Storage (noc_read), or pushes this
// core's block into a peer's Storage (noc_write). Both consume the incoming
// Block and yield one for the destination Storage.
//
// NOTE: the reserve/push below act on the *local* view of the destination CB.
// For a genuine peer buffer the far side's pointers have to be advanced too --
// see api/remote_circular_buffer.h (remote_cb_reserve_back /
// remote_cb_push_back_and_write_pages, which are asymmetric between sender and
// receiver) or the explicit semaphore handshake the matmul mcast kernels use.
// ---------------------------------------------------------------------------

template <int thread>
Block noc_read(const Storage& storage, Block block, Coord coord, int offset) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait(block.cb_id, block.num_tiles);
        cb_reserve(storage.cb_id, storage.num_tiles);
        const uint32_t bytes = cb_page_bytes(storage.cb_id);
        const uint64_t src =
            noc_addr_on_core(coord.x, coord.y, cb_read_addr(block.cb_id) + static_cast<uint32_t>(offset));
        noc_read_from(src, cb_write_addr(storage.cb_id), bytes * static_cast<uint32_t>(storage.num_tiles));
        noc_read_barrier();
        cb_push(storage.cb_id, storage.num_tiles);
        cb_pop(block.cb_id, block.num_tiles);
    }
#else
    (void)block;
    (void)coord;
    (void)offset;
#endif
    return Block(storage);
}

template <int thread>
Block noc_write(const Storage& storage, Block block, Coord coord, int offset) {
#if defined(IS_DM_THREAD) && IS_DM_THREAD
    if constexpr (thread == TT_DM_THREAD_ID) {
        cb_wait(block.cb_id, block.num_tiles);
        cb_reserve(storage.cb_id, storage.num_tiles);
        const uint32_t bytes = cb_page_bytes(storage.cb_id);
        const uint64_t dst =
            noc_addr_on_core(coord.x, coord.y, cb_write_addr(storage.cb_id) + static_cast<uint32_t>(offset));
        noc_write_to(cb_read_addr(block.cb_id), dst, bytes * static_cast<uint32_t>(block.num_tiles));
        noc_write_barrier();
        cb_push(storage.cb_id, storage.num_tiles);
        cb_pop(block.cb_id, block.num_tiles);
    }
#else
    (void)block;
    (void)coord;
    (void)offset;
#endif
    return Block(storage);
}

}  // namespace unified
}  // namespace tt
