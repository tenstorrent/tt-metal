// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Rebases one half's compile-time and runtime argument indices inside the union kernel.
//
// The union kernel puts both implementations in ONE binary per RISC-V, because a program may hold
// only one kernel per processor per core (tt_metal/impl/program/program.cpp, "Core Overlap
// Between") and both halves run on all 88. One binary means one compile-time arg array and one
// runtime arg block per core, so the two halves' lists are concatenated and the second half's
// indices all shift.
//
// Nothing in either body is renumbered to achieve that. `get_compile_time_arg_val(i)` is a macro
// over `get_ct_arg<i>()` and `get_arg_val<T>(i)` is a plain template, both resolved by ordinary
// unqualified lookup -- so a declaration of the same name at the enclosing namespace's scope wins
// over the global one for every call site in the body, and the bodies stay what they are upstream.
//
// Deliberately WITHOUT an include guard: included once inside each half's namespace, with
// HYB_CT_BASE / HYB_RT_BASE redefined in between.
//
// The rule this imposes on the bodies: only a LITERAL index may go through
// get_compile_time_arg_val. Anything derived from a TensorAccessorArgs<> offset is already
// absolute -- that template indexes kernel_compile_time_args directly, so its anchors are rebased
// by hand and next_compile_time_args_offset() propagates the base -- and must read the array with
// ::get_ct_arg, or the shim adds the base twice. Both escapes are marked at their sites.

#ifndef HYB_CT_BASE
#error "HYB_CT_BASE must be defined before including hybrid_arg_shims.hpp"
#endif
#ifndef HYB_RT_BASE
#error "HYB_RT_BASE must be defined before including hybrid_arg_shims.hpp"
#endif

template <uint32_t Idx>
constexpr uint32_t get_ct_arg() {
    return ::get_ct_arg<HYB_CT_BASE + Idx>();
}

template <typename T>
FORCE_INLINE T get_arg_val(int arg_idx) {
    return ::get_arg_val<T>(HYB_RT_BASE + arg_idx);
}
