// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — matmul-core compute (TRISCs). fp32 DEST + full sync (8-tile subblocks).
//
// Per entry: matmul_block_entry — core_m x core_n x Kt through matmul_block, K-blocks of kb, packer L1 accumulation
// into ONE caller-reserved fp32 slot of cb_partial_hold (TileRowMajor + Interm + caller_owns_pack_target).
// Then finish_block for the lagged entry (or this one): copy (chain start), hold + arrival (relay / fin with one
// upstream: SFPU fp32 add), or hold + fwd + bwd (fin with two upstreams: FPU fwd+bwd into D1, SFPU D0 += D1).
// Every cross-device add runs in fp32 DEST; the result is packed to bf16 into cb_finished.

#include <cstdint>
#ifdef MRS_ZONES  // perf instrumentation only
#include <tools/profiler/kernel_profiler.hpp>
#define MRS_ZONE(name) DeviceZoneScopedN(name)
#else
#define MRS_ZONE(name)
#endif
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"

namespace ckl = compute_kernel_lib;

constexpr uint32_t ROLE_COPY = 0, ROLE_ADD_FWD = 1, ROLE_ADD_BWD = 2, ROLE_ADD_BOTH = 3;

template <uint32_t cb_hold, uint32_t cb_finished, uint32_t cb_arr_fwd, uint32_t cb_arr_bwd, uint32_t P>
ALWI void finish_block(uint32_t role) {
    MRS_ZONE("FIN");
    using namespace ckl;
    constexpr auto shape = IterationShape::tiles(P);
    if (role == ROLE_COPY) {
        copy<input(cb_hold), output(cb_finished)>(shape);
    } else if (role == ROLE_ADD_FWD) {
        binary_sfpu<AddBinary<>, input(cb_hold), input(cb_arr_fwd), output(cb_finished)>(shape);
    } else if (role == ROLE_ADD_BWD) {
        binary_sfpu<AddBinary<>, input(cb_hold), input(cb_arr_bwd), output(cb_finished)>(shape);
    } else {
        eltwise_chain(
            shape,
            CopyTile<input(cb_hold)>{},
            BinaryFpu<BinaryFpuOp::Add, input(cb_arr_fwd), input(cb_arr_bwd), Dst::D1>{},
            AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
            PackTile<output(cb_finished)>{});
    }
}

void kernel_main() {
    constexpr uint32_t cb_act_k_window = get_compile_time_arg_val(0);
    constexpr uint32_t cb_weight_k_window = get_compile_time_arg_val(1);
    constexpr uint32_t cb_partial_hold = get_compile_time_arg_val(2);
    constexpr uint32_t cb_finished = get_compile_time_arg_val(3);
    constexpr uint32_t cb_arrivals_fwd = get_compile_time_arg_val(4);
    constexpr uint32_t cb_arrivals_bwd = get_compile_time_arg_val(5);
    constexpr uint32_t core_m = get_compile_time_arg_val(6);
    constexpr uint32_t core_n = get_compile_time_arg_val(7);
    constexpr uint32_t kb = get_compile_time_arg_val(8);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(9);
    constexpr uint32_t sb_h = get_compile_time_arg_val(10);
    constexpr uint32_t sb_w = get_compile_time_arg_val(11);
    constexpr uint32_t P = get_compile_time_arg_val(12);
    constexpr uint32_t num_entries = get_compile_time_arg_val(13);
    static_assert(P == core_m * core_n, "P is the per-core sub-block");

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act_k_window, cb_weight_k_window, cb_partial_hold);

    CircularBuffer act_buf(cb_act_k_window), weight_buf(cb_weight_k_window), hold_buf(cb_partial_hold);
    constexpr auto mm_shape = ckl::MatmulBlockShape::of(core_m / sb_h, core_n / sb_w, sb_h, sb_w, kb, num_k_blocks);

    auto matmul_block_entry = [&]() {
        MRS_ZONE("MM");
        hold_buf.reserve_back(P);  // caller-owned pack target: one fp32 slot per entry
        ckl::matmul_block<
            /*transpose=*/false,
            /*packer_l1_acc=*/true,
            ckl::LastBlockTarget::Interm,
            ckl::OutputCBLayout::TileRowMajor,
            ckl::matmul_config::InitMode::Short,
            ckl::InputPolicy::WaitAndPopPerKBlock,
            ckl::InputPolicy::WaitAndPopPerKBlock,
            ckl::NoPostCompute,
            ckl::NoPreKBlock,
            ckl::NoPostKBlock,
            0,
            ckl::NoKBlockInnerDimFn,
            ckl::NoIn0Source,
            ckl::NoIn1BaseOffset,
            /*caller_owns_pack_target=*/true>(
            act_buf,
            weight_buf,
            hold_buf,
            hold_buf,
            mm_shape,
            {},
            {},
            /*in1_per_core_w=*/core_n,
            /*out_row_width=*/core_n);
        hold_buf.push_back(P);
        // matmul_block leaves packer L1 accumulation on after its last K-block; the finish chains pack plainly.
        PACK((llk_pack_reconfig_l1_acc(0)));
    };

    constexpr uint32_t NONE = 0xFFFFFFFF;
    uint32_t pending = NONE;
    for (uint32_t e = 0; e < num_entries; ++e) {
        matmul_block_entry();
        if (pending != NONE) {
            finish_block<cb_partial_hold, cb_finished, cb_arrivals_fwd, cb_arrivals_bwd, P>(
                get_arg_val<uint32_t>(pending));
            pending = NONE;
        }
        const uint32_t role = get_arg_val<uint32_t>(e);
        if (role != ROLE_COPY && e + 1 < num_entries) {
            pending = e;  // the add waits for its arrival behind the next entry's matmul
        } else {
            finish_block<cb_partial_hold, cb_finished, cb_arrivals_fwd, cb_arrivals_bwd, P>(role);
        }
    }
}
