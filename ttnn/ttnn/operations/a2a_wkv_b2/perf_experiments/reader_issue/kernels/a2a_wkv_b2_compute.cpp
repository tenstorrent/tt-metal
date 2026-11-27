// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// a2a_wkv_b2 worker compute.
//
// Per head, per mt block:
//   tilize_attn_block : cb_attn_rm (mt_rows x Kt bf16 RM tile-pages) -> cb_attn_tiled
//   project_block     : [mt_rows x Kt] @ W[h] [Kt x Nt] -> cb_out (mt_rows x Nt bf8_b), K in one
//                       block (accumulated in DEST); W retained (never popped by the helper)
// After the head's last block: release_weight_block (raw cb_pop_front: the matmul helper's
// WaitAndPopPerKBlock would pop W after every block, see op_design.md).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_attn_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_attn_tiled = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weight = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t Kt = get_compile_time_arg_val(4);
    constexpr uint32_t Nt = get_compile_time_arg_val(5);
    constexpr uint32_t Mt = get_compile_time_arg_val(6);
    constexpr uint32_t mt_block = get_compile_time_arg_val(7);

    constexpr uint32_t num_mt_blocks = (Mt + mt_block - 1) / mt_block;

    const uint32_t num_heads = get_arg_val<uint32_t>(0);

    using namespace compute_kernel_lib;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_attn_tiled, cb_weight, cb_out);

    CircularBuffer in0(cb_attn_tiled), in1(cb_weight), out(cb_out);

    for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
        for (uint32_t mb = 0; mb < num_mt_blocks; ++mb) {
            const uint32_t mt_rows = (Mt - mb * mt_block) < mt_block ? (Mt - mb * mt_block) : mt_block;

#ifdef A2A_ABL_NO_COMPUTE
            {
                MaybeDeviceZoneScope("compute_tilize");
                cb_wait_front(cb_attn_rm, mt_rows * Kt);
                cb_pop_front(cb_attn_rm, mt_rows * Kt);
            }
            {
                MaybeDeviceZoneScope("compute_matmul");
                if (mb == 0) {
                    cb_wait_front(cb_weight, Kt * Nt);
                }
                cb_reserve_back(cb_out, mt_rows * Nt);
                cb_push_back(cb_out, mt_rows * Nt);
            }
#else
            {
                // unpack: waits on the reader (cb_attn_rm); pack: on cb_attn_tiled space
                MaybeDeviceZoneScope("compute_tilize");
                tilize<Kt, cb_attn_rm, cb_attn_tiled>(mt_rows);
            }
            {
                // unpack: waits on W (first block of a head) + the tiled block; pack: cb_out space
                MaybeDeviceZoneScope("compute_matmul");

                // One output subblock per tile row (1 x Nt <= DEST, checked on the host); with a single
                // in1 subblock SubblockMajor is row-major, which the writer / fabric_tx rely on.
                matmul_block<
                    /*transpose=*/false,
                    /*packer_l1_acc=*/false,
                    LastBlockTarget::Out,
                    OutputCBLayout::SubblockMajor,
                    matmul_config::InitMode::Short,
                    InputPolicy::WaitAndPopPerKBlock,
                    InputPolicy::WaitAndRetainOnLastBlock>(
                    in0,
                    in1,
                    out,
                    out,
                    MatmulBlockShape::of(
                        /*in0_num_subblocks=*/mt_rows,
                        /*in1_num_subblocks=*/1,
                        /*out_subblock_h=*/1,
                        /*out_subblock_w=*/Nt,
                        /*in0_block_k=*/Kt,
                        /*num_k_blocks=*/1));
            }
#endif
        }
        // release_weight_block
        cb_pop_front(cb_weight, Kt * Nt);
    }
}
