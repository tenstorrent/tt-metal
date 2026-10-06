// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/addcdiv.h"
#include "api/dataflow/dataflow_buffer.h"

#if defined(ARCH_BLACKHOLE)
// The copy init of c_0 serves an operand that shares its formats and tile geometry.
template <uint32_t cb>
constexpr bool copy_init_shared_with_c0() {
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    constexpr uint32_t c0 = tt::CBIndex::c_0;
    return unpack_src_format[cb] == unpack_src_format[c0] && unpack_dst_format[cb] == unpack_dst_format[c0] &&
           unpack_tile_num_faces[cb] == unpack_tile_num_faces[c0] &&
           unpack_tile_face_r_dim[cb] == unpack_tile_face_r_dim[c0] &&
           unpack_partial_face[cb] == unpack_partial_face[c0] && unpack_narrow_tile[cb] == unpack_narrow_tile[c0] &&
           unpack_tile_r_dim[cb] == unpack_tile_r_dim[c0] && unpack_tile_c_dim[cb] == unpack_tile_c_dim[c0];
#else
    return true;
#endif
}
#endif

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1

    DataflowBuffer dfb_in0(tt::CBIndex::c_0);  // input_a
    DataflowBuffer dfb_in1(tt::CBIndex::c_1);  // input_b
    DataflowBuffer dfb_in2(tt::CBIndex::c_2);  // input_c
    DataflowBuffer dfb_out(tt::CBIndex::c_3);

    compute_kernel_hw_startup(dfb_in0.get_id(), dfb_out.get_id());
    copy_init(dfb_in0.get_id());
#if defined(ARCH_BLACKHOLE)
    TERNARY_SFPU_OP_INIT();
    constexpr bool shared_copy_init =
        copy_init_shared_with_c0<tt::CBIndex::c_1>() && copy_init_shared_with_c0<tt::CBIndex::c_2>();
#else
    constexpr bool shared_copy_init = false;
#endif

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        dfb_in0.wait_front(num_tiles_per_cycle);
        dfb_in1.wait_front(num_tiles_per_cycle);
        dfb_in2.wait_front(num_tiles_per_cycle);

        dfb_out.reserve_back(num_tiles_per_cycle);

        tile_regs_acquire();

        if constexpr (!shared_copy_init) {
            copy_init(dfb_in0.get_id());
        }
        copy_tile(dfb_in0.get_id(), 0 /*in_tile_index*/, 0 /*dst_tile_index*/);

        if constexpr (!shared_copy_init) {
            copy_init(dfb_in1.get_id());
        }
        copy_tile(dfb_in1.get_id(), 0 /*in_tile_index*/, 1 /*dst_tile_index*/);

        if constexpr (!shared_copy_init) {
            copy_init(dfb_in2.get_id());
        }
        copy_tile(dfb_in2.get_id(), 0 /*in_tile_index*/, 2 /*dst_tile_index*/);

#if !defined(ARCH_BLACKHOLE)
        TERNARY_SFPU_OP_INIT();
#endif
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0, scalar_arg);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, dfb_out.get_id());

        tile_regs_release();

        dfb_out.push_back(num_tiles_per_cycle);
        dfb_in0.pop_front(num_tiles_per_cycle);
        dfb_in1.pop_front(num_tiles_per_cycle);
        dfb_in2.pop_front(num_tiles_per_cycle);
    }
}
