// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Gives the union compute kernel ONE out-of-line copy of the LLK helpers it expands most.
//
// The whole program config is written into the kernel-config ring on every dispatch, and the
// union program carries both halves' binaries, so text is the resource this op is short of. The
// helpers below are `inline` in LLK, which is the right trade for a kernel with a handful of call
// sites and the wrong one here: each is expanded at roughly 20-100 sites across the two halves,
// and together they are about a quarter of the TRISC triple's text.
//
// Interception is by MACRO, not by unqualified lookup: these are called from inside LLK's own
// headers, not from either half's body, so a using-declaration at the halves' namespace scope
// cannot see them. That makes include order the mechanism, and this header has to come FIRST in
// the union compute kernel:
//
//   1. it pulls in each helper's DEFINING header under the helper's own name, so the real
//      definition is parsed unrenamed and keeps calling its neighbours directly;
//   2. it defines forwarding wrappers next to each original, named `hyb_<original>`;
//   3. it renames the helper, so every header parsed after -- the LLK pack/unpack lib, the
//      compute API, both halves -- lands on the wrapper.
//
// A header already parsed when the rename lands keeps the inline version, because an unqualified
// call to a non-dependent name is bound where it is written. That is what fixes the order of the
// blocks below: each helper is pulled in and renamed before the header that drags in the layer
// calling it, which is why the pack side stages cpack_common -> pack lib -> metal API -> CB.
//
// Each wrapper lives in its original's namespace and differs only in name, so both spellings at
// the call sites keep working -- the qualified `ckernel::packer::f(x)` and the unqualified `f(x)`
// found from inside that namespace.
//
// Three rules on what may be added here, each of which cost text or time when violated:
//   * nothing on the MATH core. The op runs at its FPU rate, so trisc1 is the critical path and a
//     call there is paid per tile -- out-of-lining its dst-address helper measured 1.27-1.40x.
//   * nothing whose Tensix instruction needs a literal operand (the `TTI_` forms), because only
//     inlining constant-folds those.
//   * nothing whose body is comparable to its own argument setup. A helper taking several
//     arguments to do a handful of instructions gets BIGGER out-of-line, not smaller.

#pragma once

// Compiles to nothing off the TRISC triple. The dataflow processors reach their circular buffers
// through the API that hybrid_dataflow_cb_shims.hpp intercepts instead.
#if defined(COMPILE_FOR_TRISC)

// noclone as well as noinline: under -flto, IPA constant propagation will otherwise clone a
// specialization per distinct argument set and put the copies straight back.
#define HYB_LLK_SHIM __attribute__((noinline, noclone))

// ---------------------------------------------------------------- UNPACK (TRISC0)
#if COMPILE_FOR_TRISC == 0

// Before the CB header below, which drags in the metal unpack API that calls this one.
#include "llk_unpack_common.h"

template <bool is_fp32_dest_acc_en, p_dim_stride_target dim_stride_target, bool skip_int8 = false>
HYB_LLK_SHIM void hyb_llk_unpack_reconfig_data_format_srca_impl_(
    const std::uint32_t unpack_src_format,
    const std::uint32_t unpack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t unpack_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpack_num_faces = 4) {
    _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, dim_stride_target, skip_int8>(
        unpack_src_format, unpack_dst_format, tile_size, unpack_face_r_dim, unpack_num_faces);
}

#define _llk_unpack_reconfig_data_format_srca_impl_ hyb_llk_unpack_reconfig_data_format_srca_impl_

#include "llk_io_unpack.h"

HYB_LLK_SHIM inline void hyb_llk_wait_tiles(int operand, std::int32_t num_tiles) { llk_wait_tiles(operand, num_tiles); }

HYB_LLK_SHIM inline void hyb_llk_pop_tiles(
    const std::int32_t operand, const std::int32_t num_tiles, const std::int32_t block_c_dim = 0) {
    llk_pop_tiles(operand, num_tiles, block_c_dim);
}

#define llk_wait_tiles hyb_llk_wait_tiles
#define llk_pop_tiles hyb_llk_pop_tiles

#endif  // COMPILE_FOR_TRISC == 0

// ---------------------------------------------------------------- PACK (TRISC2)
#if COMPILE_FOR_TRISC == 2

// Ahead of every header below, each of which drags in a layer of the pack lib that calls into
// this one.
#include "cpack_common.h"

namespace ckernel::packer {
HYB_LLK_SHIM inline void hyb_program_packer_destination(std::uint32_t addr) { program_packer_destination(addr); }

template <DstSync Dst>
HYB_LLK_SHIM inline void hyb_select_packer_dest_registers() {
    select_packer_dest_registers<Dst>();
}
}  // namespace ckernel::packer

#define program_packer_destination hyb_program_packer_destination
#define select_packer_dest_registers hyb_select_packer_dest_registers

#include "llk_pack_common.h"

template <DstSync Dst, bool is_fp32_dest_acc_en>
HYB_LLK_SHIM inline void hyb_llk_pack_dest_section_done_() {
    _llk_pack_dest_section_done_<Dst, is_fp32_dest_acc_en>();
}

#define _llk_pack_dest_section_done_ hyb_llk_pack_dest_section_done_

#include "llk_pack_common_api.h"

template <bool out_of_order_output, PackMode pack_addr_mode = PackMode::Default>
HYB_LLK_SHIM inline std::uint32_t hyb_get_output_tile_address(std::uint8_t output_id, std::uint32_t output_tile_index) {
    return get_output_tile_address<out_of_order_output, pack_addr_mode>(output_id, output_tile_index);
}

#define get_output_tile_address hyb_get_output_tile_address

#include "llk_io_pack.h"

template <bool skip_sync = false, bool wait_for_blocks = false, bool brisc_pack = false>
HYB_LLK_SHIM inline void hyb_llk_wait_for_free_tiles(const std::int32_t operand, const std::int32_t num_tiles) {
    llk_wait_for_free_tiles<skip_sync, wait_for_blocks, brisc_pack>(operand, num_tiles);
}

HYB_LLK_SHIM inline void hyb_llk_push_to_brisc(
    const std::int32_t operand, const std::int32_t num_tiles, const std::int32_t num_words) {
    llk_push_to_brisc(operand, num_tiles, num_words);
}

template <bool push_blocks = false, bool brisc_pack = false>
HYB_LLK_SHIM inline void hyb_llk_push_tiles(const std::int32_t operand, const std::int32_t num_tiles) {
    llk_push_tiles<push_blocks, brisc_pack>(operand, num_tiles);
}

#define llk_wait_for_free_tiles hyb_llk_wait_for_free_tiles
#define llk_push_to_brisc hyb_llk_push_to_brisc
#define llk_push_tiles hyb_llk_push_tiles

#endif  // COMPILE_FOR_TRISC == 2

#endif  // COMPILE_FOR_TRISC
