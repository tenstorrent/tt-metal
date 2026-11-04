// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Gives the union kernel ONE out-of-line copy of each circular-buffer call, shared by both halves.
//
// The CB API is always_inline, which is the right trade for a single kernel and the wrong one
// here: this binary carries both halves' call sites, and the whole program config is written into
// the kernel-config ring on every dispatch, so text is the resource the merged op is short of.
//
// The redirection costs the bodies nothing: a using-declaration at the enclosing namespace's
// scope wins unqualified lookup over the global name, so every call site written in either body
// resolves here without being touched. It reaches only calls written in the bodies themselves --
// a helper header included at file scope is outside the namespace and keeps the inline version.
//
// Included TWICE per union kernel: once early for the declarations, which is all the call sites
// need, and once after both halves with HYB_CB_SHIMS_DEFINE set, where the real API is finally in
// scope. Deliberately without an include guard for that reason.

#include <cstdint>

// Which side of a circular buffer this processor actually drives. On the TRISC triple the calls
// are role-gated -- cb_api.h wraps them in UNPACK()/PACK(), which expand to nothing off-role -- so
// out-of-lining an off-role call would replace nothing at all with a call to an empty function.
#if !defined(HYB_CB_FRONT) && !defined(HYB_CB_BACK)
#if !defined(COMPILE_FOR_TRISC)
#define HYB_CB_FRONT 1
#define HYB_CB_BACK 1
#elif COMPILE_FOR_TRISC == 0
#define HYB_CB_FRONT 1
#elif COMPILE_FOR_TRISC == 2
#define HYB_CB_BACK 1
#endif
#endif

// noclone as well as noinline: under -flto, IPA constant propagation will otherwise clone a
// specialization per distinct argument pair and put the copies straight back.
#define HYB_CB_SHIM __attribute__((noinline, noclone))

#ifdef HYB_CB_SHIMS_DEFINE
#if defined(COMPILE_FOR_TRISC)
#define HYB_CB_CALL(fn, cb, n) ::ckernel::fn((cb), (n))
#else
#define HYB_CB_CALL(fn, cb, n) ::fn((cb), (n))
#endif
#endif

namespace hyb_cb {

#ifdef HYB_CB_FRONT
#ifdef HYB_CB_SHIMS_DEFINE
HYB_CB_SHIM void cb_wait_front(uint32_t cb, uint32_t n) { HYB_CB_CALL(cb_wait_front, cb, n); }
HYB_CB_SHIM void cb_pop_front(uint32_t cb, uint32_t n) { HYB_CB_CALL(cb_pop_front, cb, n); }
#else
HYB_CB_SHIM void cb_wait_front(uint32_t cb, uint32_t n);
HYB_CB_SHIM void cb_pop_front(uint32_t cb, uint32_t n);
#endif
#endif

#ifdef HYB_CB_BACK
#ifdef HYB_CB_SHIMS_DEFINE
HYB_CB_SHIM void cb_reserve_back(uint32_t cb, uint32_t n) { HYB_CB_CALL(cb_reserve_back, cb, n); }
HYB_CB_SHIM void cb_push_back(uint32_t cb, uint32_t n) { HYB_CB_CALL(cb_push_back, cb, n); }
#else
HYB_CB_SHIM void cb_reserve_back(uint32_t cb, uint32_t n);
HYB_CB_SHIM void cb_push_back(uint32_t cb, uint32_t n);
#endif
#endif

}  // namespace hyb_cb
