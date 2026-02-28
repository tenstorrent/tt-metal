// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shim for experimental/matmul_custom.h that avoids redefinition conflicts with
// llk_math_matmul.h (already included via compute_kernel_api.h).
//
// The experimental llk_math_matmul_custom_no_mop.h redefines matmul_configure_addrmod,
// matmul_configure_mop, and matmul_configure_mop_throttled — identical to the versions
// in llk_math_matmul.h. This shim suppresses those redefinitions by extracting only
// the unique no-mop functions we need.
//
// TODO: Remove this shim once the LLK submodule is updated to avoid the conflict
// (LLK commit 74bf675+).

#pragma once

#ifdef ARCH_BLACKHOLE

// Pull in the unique no-mop LLK functions directly, skipping the conflicting header.
// We only need: _llk_math_matmul_init_no_mop_, _llk_math_matmul_no_mop_,
//               matmul_configure_addrmod_reinit, llk_math_matmul_reinit_no_mop
// These live in llk_math_matmul_custom_no_mop.h which also redefines shared functions.
// Solution: provide forward declarations that reference the already-included versions,
// then include only the API header with the conflicting lower-level header pre-empted.

// The unique functions we need are defined in:
//   experimental/llk_math_matmul_custom_no_mop.h  (has conflicts)
//   experimental/llk_math_matmul_custom_api.h     (includes the above)
//   api/compute/experimental/matmul_custom.h       (includes the API header)
//
// Strategy: manually forward-declare the no-mop functions from the API header.
// This avoids pulling in the conflicting lower-level header entirely.

#include "api/compute/common.h"

// --- Extracted from experimental/llk_math_matmul_custom_api.h ---
// These call into _llk_math_matmul_init_no_mop_ etc. which are in the
// conflicting header. We need those low-level functions too.
// Since we can't include the header without conflicts, we include
// the complete file but suppress the redefinitions via wrapper macros.

// Temporarily rename the conflicting functions before including
#define matmul_configure_addrmod matmul_configure_addrmod_experimental_
#define matmul_configure_mop matmul_configure_mop_experimental_
#define matmul_configure_mop_throttled matmul_configure_mop_throttled_experimental_

#ifdef TRISC_MATH
#include "experimental/llk_math_matmul_custom_no_mop.h"
#include "experimental/llk_math_matmul_custom_api.h"
#endif

#undef matmul_configure_addrmod
#undef matmul_configure_mop
#undef matmul_configure_mop_throttled

// Now the standard versions from llk_math_matmul.h are still the canonical ones,
// and the no-mop unique functions (_llk_math_matmul_init_no_mop_ etc.) are available.
// The renamed experimental_ versions are unused but harmless.

// Re-export the public API from matmul_custom.h (minus the conflicting include)
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_matmul_api.h"
#endif
#ifdef TRISC_PACK
#include "llk_pack_api.h"
#endif

#ifndef MM_THROTTLE
#define MM_THROTTLE 0
#endif

namespace ckernel {

ALWI void mm_no_mop_init_short(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    const bool transpose = false,
    uint32_t ct_dim = 1,
    uint32_t rt_dim = 1,
    uint32_t kt_dim = 1) {
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_init_no_mop<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
}

ALWI void matmul_block_no_mop(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t in0_tile_index,
    uint32_t in1_tile_index,
    uint32_t idst,
    const bool transpose,
    uint32_t ct_dim,
    uint32_t rt_dim,
    uint32_t kt_dim) {
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_no_mop<MATH_FIDELITY, MM_THROTTLE>(idst, ct_dim, rt_dim)));
}

ALWI void mm_no_mop_reinit_short(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    const bool transpose = false,
    uint32_t ct_dim = 1,
    uint32_t rt_dim = 1,
    uint32_t kt_dim = 1) {
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim, kt_dim)));
    MATH((llk_math_matmul_reinit_no_mop<MATH_FIDELITY, MM_THROTTLE>(transpose)));
}

}  // namespace ckernel

#endif  // ARCH_BLACKHOLE
