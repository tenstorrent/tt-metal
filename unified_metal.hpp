// SPDX-License-Identifier: Apache-2.0
//
// Metal binding for the unified programming model.
//
// unified.hpp includes this automatically. To bind the model to something else
// -- the host trace harness, say -- define TT_UNIFIED_CUSTOM_BINDING before
// including unified.hpp and provide the same names yourself.
//
// Two jobs:
//
//   1. Derive the model's thread identity from the defines metal already emits
//      for every kernel build (tt_metal/llrt/hal/tt-1xx/hal_1xx_common.cpp).
//      Nothing is passed from the host -- pointing three KernelDescriptors at
//      one source is enough, because each build already knows what it is.
//
//        COMPILE_FOR_BRISC   -> DM thread 0   (writer, by metal convention)
//        COMPILE_FOR_NCRISC  -> DM thread 1   (reader)
//        UCK_CHLKC_{UNPACK,MATH,PACK} -> compute
//
//   2. Map the model's intrinsics onto real metal APIs. CB ops go through
//      CircularBuffer, which is already thread-polymorphic: one call lowers to
//      cb_* on a DM core and PACK(llk_*)/UNPACK(llk_*) on a TRISC.
//
// Note on what is NOT here: there are almost no cross-projection no-ops. Every
// compute intrinsic is only ever called from inside a `#if IS_COMPUTE_THREAD`
// region (in fusion.hpp's Strategy and op guards), and every data-movement
// intrinsic only from inside a `#if IS_DM_THREAD` region (in unified.hpp's
// noc_*). So each projection declares only its own half. Two exceptions, both
// because kernels name them on a shared path: `compute_init` (data movement
// needs a no-op) and `make_accessor` (compute needs one returning NullAccessor).

#pragma once

// ---------------------------------------------------------------------------
// Thread identity
// ---------------------------------------------------------------------------

#if defined(COMPILE_FOR_BRISC)
#define IS_DM_THREAD 1
#define TT_DM_THREAD_ID 0
#elif defined(COMPILE_FOR_NCRISC)
#define IS_DM_THREAD 1
#define TT_DM_THREAD_ID 1
#elif defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_PACK)
#define IS_COMPUTE_THREAD 1
#else
#error "unified_metal.hpp: no metal thread-identity define present"
#endif

// ---------------------------------------------------------------------------
// Metal headers, per projection
//
// tensor_accessor_args.h works on both (it only needs get_compile_time_arg_val),
// which is what lets a shared source chain CT-arg offsets. The full
// tensor_accessor.h does NOT compile on a TRISC -- it wants NOC_INDEX and
// redeclares get_common_arg_addr -- so accessors are constructed only inside
// data-movement regions.
// ---------------------------------------------------------------------------

#include <cstdint>

#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD
#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/tensor/tensor_accessor_args.h"
#else
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/tensor_accessor_args.h"
#endif

#include "api/dataflow/circular_buffer.h"

namespace tt {
namespace unified {

// ---------------------------------------------------------------------------
// Circular-buffer protocol -- available on every projection
// ---------------------------------------------------------------------------

inline void cb_reserve(int cb, int pages) { CircularBuffer(cb).reserve_back(pages); }
inline void cb_push(int cb, int pages) { CircularBuffer(cb).push_back(pages); }
inline void cb_wait(int cb, int pages) { CircularBuffer(cb).wait_front(pages); }
inline void cb_pop(int cb, int pages) { CircularBuffer(cb).pop_front(pages); }

#if defined(IS_COMPUTE_THREAD) && IS_COMPUTE_THREAD

// ---------------------------------------------------------------------------
// Compute
// ---------------------------------------------------------------------------

using ckernel::tile_regs_acquire;
using ckernel::tile_regs_commit;
using ckernel::tile_regs_release;
using ckernel::tile_regs_wait;

inline void copy_tile_to_dst(int cb, int tile, int dst) {
    ckernel::copy_tile(static_cast<uint32_t>(cb), static_cast<uint32_t>(tile), static_cast<uint32_t>(dst));
}

inline void pack_dst_tile(int dst, int cb) {
    ckernel::pack_tile(static_cast<uint32_t>(dst), static_cast<uint32_t>(cb));
}

// Configures unpack/pack for a CB pair. Kernels call this unconditionally, so it
// is the one intrinsic that also needs a data-movement definition.
inline void compute_init(int in_cb, int out_cb) {
    ckernel::init_sfpu(static_cast<uint32_t>(in_cb), static_cast<uint32_t>(out_cb));
}

// Each `*_init` is cheap and metal kernels routinely re-init per use (see
// SFPU_OP_CHAIN_0 in tests/.../compute/eltwise_sfpu.cpp), so it is folded in
// here rather than hoisted -- revisit if it shows up in a profile.
inline void sfpu_add_dst(int a, int b, int out) {
    ckernel::add_binary_tile_init();
    ckernel::add_binary_tile(static_cast<uint32_t>(a), static_cast<uint32_t>(b), static_cast<uint32_t>(out));
}

// Unary ops are in-place by construction: Emit<Base, Un<Op, C>> evaluates the
// child into Base then applies the op with src == out == Base, which is exactly
// what the SFPU tile ops do.
inline void sfpu_exp_dst(int src, int out) {
    (void)src;  // == out
    ckernel::exp_tile_init();
    ckernel::exp_tile(static_cast<uint32_t>(out));
}

inline void sfpu_relu_dst(int src, int out) {
    (void)src;  // == out
    ckernel::relu_tile_init();
    ckernel::relu_tile(static_cast<uint32_t>(out));
}

// The full TensorAccessor does not compile on a TRISC: it wants NOC_INDEX and
// redeclares get_common_arg_addr against api/compute/common.h. Compute never
// dereferences an accessor -- it only carries one through a statement it shares
// with the data-movement projections -- so a stub is enough.
struct NullAccessor {};

template <typename Args>
inline NullAccessor make_accessor(Args, uint32_t) {
    return NullAccessor{};
}

// TODO: the FPU pack-side epilogue is not bound to metal yet. Declared without a
// definition so the (uninstantiated) template body type-checks -- a program that
// actually reaches it fails to LINK with this name, rather than silently doing
// nothing.
void relu_from_pack(int base, int count);

#else

// ---------------------------------------------------------------------------
// Data movement
// ---------------------------------------------------------------------------

// The only compute intrinsic a data-movement build needs: kernels call it
// unconditionally at entry.
inline void compute_init(int, int) {}

inline uint32_t cb_write_addr(int cb) { return get_write_ptr(static_cast<uint32_t>(cb)); }
inline uint32_t cb_read_addr(int cb) { return get_read_ptr(static_cast<uint32_t>(cb)); }
inline uint32_t cb_page_bytes(int cb) { return get_tile_size(static_cast<uint32_t>(cb)); }

// Accessors are constructed here, inside a data-movement region, from the
// TensorAccessorArgs the shared source names on every projection.
template <typename Args>
inline auto make_accessor(Args args, uint32_t base_addr) {
    return TensorAccessor(args, base_addr);
}

template <typename Accessor>
inline void noc_read_page(const Accessor& acc, uint32_t page_id, uint32_t l1_addr, uint32_t bytes) {
    noc_async_read(acc.get_noc_addr(page_id), l1_addr, bytes);
}

template <typename Accessor>
inline void noc_write_page(const Accessor& acc, uint32_t page_id, uint32_t l1_addr, uint32_t bytes) {
    noc_async_write(l1_addr, acc.get_noc_addr(page_id), bytes);
}

// Core-to-core: form a NOC address for `local_addr` as seen on core (x, y).
inline uint64_t noc_addr_on_core(int x, int y, uint32_t local_addr) {
    return get_noc_addr(static_cast<uint32_t>(x), static_cast<uint32_t>(y), local_addr);
}

inline void noc_read_from(uint64_t src_noc_addr, uint32_t dst_l1_addr, uint32_t bytes) {
    noc_async_read(src_noc_addr, dst_l1_addr, bytes);
}

inline void noc_write_to(uint32_t src_l1_addr, uint64_t dst_noc_addr, uint32_t bytes) {
    noc_async_write(src_l1_addr, dst_noc_addr, bytes);
}

inline void noc_read_barrier() { noc_async_read_barrier(); }

// Writes have DEPARTED the local L1 (not landed at the destination). This is the
// release condition for a source buffer -- see the note on NocAsyncWriteTx.
inline void noc_writes_flushed() { noc_async_writes_flushed(); }

// Writes have LANDED at the destination.
inline void noc_write_barrier() { noc_async_write_barrier(); }

#endif

}  // namespace unified
}  // namespace tt
