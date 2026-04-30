// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PARTIAL MIGRATION: PREPROCESS pre-passes + caller-side cb wait/pop stay raw
// LLK. Inner DEST acquire/commit/wait/pack/release migrates to V2 helper via
// BinaryFpuMacroOp wrapping the BINARY_OP macro.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_binary.h"
#include "eltwise_utils_common.hpp"
#include "eltwise_utils.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace {
template <tt::CBIndex CbA, tt::CBIndex CbB>
struct BinaryFpuMacroOp : compute_kernel_lib::UnaryOp<BinaryFpuMacroOp<CbA, CbB>, compute_kernel_lib::Dst::D0> {
    static constexpr bool clobbers_sfpu_lut = false;
    static constexpr bool clashes_with_fpu = true;

    ALWI static void init() {
#if HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST)
        binary_tiles_init<true, BINARY_OP_TYPE>(CbA, CbB);
#endif
    }
    ALWI static void call(uint32_t dst) {
        BINARY_OP(CbA, CbB, 0, 0, dst);
        PROCESS_POST_ACTIVATIONS(dst);
    }
};
}  // namespace

template <
    tt::CBIndex cb_pre_lhs,
    tt::CBIndex cb_post_lhs,
    tt::CBIndex cb_pre_rhs,
    tt::CBIndex cb_post_rhs,
    tt::CBIndex cb_out>
ALWI void process_tiles(uint32_t n) {
    using namespace ckernel;
    using compute_kernel_lib::eltwise_chain;
    using compute_kernel_lib::eltwise_pipeline;

    PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, n);
    cb_wait_front(cb_post_lhs, n);

    PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, n);
    cb_wait_front(cb_post_rhs, n);

#if HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST)
    binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs, cb_post_rhs);
#endif

    eltwise_pipeline<static_cast<uint32_t>(cb_out)>(n, eltwise_chain(BinaryFpuMacroOp<cb_post_lhs, cb_post_rhs>{}));

    cb_pop_front(cb_post_lhs, n);
    cb_pop_front(cb_post_rhs, n);
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);
    static_assert(num_tiles_per_cycle == 1, "binary_fpu_no_bcast chain path runs one tile per chain invocation");

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    ckernel::compute_kernel_hw_startup(cb_post_lhs, cb_out);
    binary_op_init_common(cb_post_lhs, cb_post_rhs, cb_out);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST))
    binary_tiles_init<true, BINARY_OP_TYPE>(cb_post_lhs, cb_post_rhs);
#endif

    uint32_t num_full_chunks = num_tiles / num_tiles_per_cycle;
    for (uint32_t chunk = 0; chunk < num_full_chunks; ++chunk) {
        process_tiles<cb_pre_lhs, cb_post_lhs, cb_pre_rhs, cb_post_rhs, cb_out>(num_tiles_per_cycle);
    }

    uint32_t remainder = num_tiles % num_tiles_per_cycle;
    if (remainder > 0) {
        process_tiles<cb_pre_lhs, cb_post_lhs, cb_pre_rhs, cb_post_rhs, cb_out>(remainder);
    }
}
