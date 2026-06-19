// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0
//
// Flash-Attention SDPA compute kernel (UNPACK/MATH/PACK).
//
// Online-softmax recurrence per work unit (b,h,q), q_chunk_t = 1 (one 32-row
// query tile-row). All running statistics / accumulators are fp32
// (fp32_dest_acc_en). Score block is kv_chunk_t tiles — never Sq_t × Skv_t.
//
// Phase 0 (per unit): scale Q in-place by `scale`.
// Per KV block j:
//   A  cb_scores = (scaled Q)·Kⱼᵀ                          (matmul_block, transpose)
//   B  cb_scores += maskⱼ                                  (if use_mask)
//   C  m_blk = rowmax(cb_scores)                           (reduce MAX REDUCE_ROW)
//   D1 m_new = max(m_prev, m_blk)         (j>0)
//   D2 α = exp(m_prev − m_new)            (j>0)
//   D3 m_prev ← m_new                     (j>0)
//   E  cb_p = exp(cb_scores − m)                           (bcast col)
//   F  l_blk = rowsum(cb_p)                                (reduce SUM REDUCE_ROW)
//   G  l_new = α·l_prev + l_blk           (j>0)
//   H  PV = cb_p·Vⱼ                                        (matmul_block)
//   I  O_new = α·O_prev + PV              (j>0, bcast col)
// After KV loop:
//   J  recip = 1/l ; K  cb_out = O_accum·recip (bcast col → bf16)
//   L  release retained Q ; drain leftover running-max m.
//
// HELPER NOTE: raw cb_pop_front is used only to release inputs that helpers
// intentionally left fronted — cb_q (matmul in0=WaitAndRetainOnLastBlock),
// cb_alpha / cb_recip (eltwise B=WaitNoPop), and the running-max cb_max (read
// no-pop by exp; the running m is dead after the final normalize). These are
// the documented caller-owned tail of retain/no-pop policies, not a hand-rolled
// replacement of any helper.
//
// BOOT: mm_init does the full hw_configure; matmul_block uses InitMode::Short.
// (Advisory deviation from op_design.md's mm_block_init: mm_init is the
// documented partner of compute_kernel_hw_startup and is exactly the boot the
// production SDPA kernel uses for this matmul+reduce+eltwise mix.)

#include "api/debug/device_print.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"

using namespace compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr uint32_t kv_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t num_kv_chunks = get_compile_time_arg_val(2);
    constexpr uint32_t use_mask = get_compile_time_arg_val(3);
    constexpr uint32_t osw_qk = get_compile_time_arg_val(4);
    constexpr uint32_t in1sb_qk = get_compile_time_arg_val(5);
    constexpr uint32_t osw_pv = get_compile_time_arg_val(6);
    constexpr uint32_t in1sb_pv = get_compile_time_arg_val(7);

    constexpr uint32_t cb_q = get_compile_time_arg_val(8);
    constexpr uint32_t cb_k = get_compile_time_arg_val(9);
    constexpr uint32_t cb_v = get_compile_time_arg_val(10);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(11);
    constexpr uint32_t cb_scaler_max = get_compile_time_arg_val(12);
    constexpr uint32_t cb_scaler_sum = get_compile_time_arg_val(13);
    constexpr uint32_t cb_mblock = get_compile_time_arg_val(14);
    constexpr uint32_t cb_mnew = get_compile_time_arg_val(15);
    constexpr uint32_t cb_lblock = get_compile_time_arg_val(16);
    constexpr uint32_t cb_out = get_compile_time_arg_val(17);
    constexpr uint32_t cb_scores = get_compile_time_arg_val(18);
    constexpr uint32_t cb_p = get_compile_time_arg_val(19);
    constexpr uint32_t cb_pv = get_compile_time_arg_val(20);
    constexpr uint32_t cb_out_accum = get_compile_time_arg_val(21);
    constexpr uint32_t cb_max = get_compile_time_arg_val(22);
    constexpr uint32_t cb_sum = get_compile_time_arg_val(23);
    constexpr uint32_t cb_alpha = get_compile_time_arg_val(24);
    constexpr uint32_t cb_recip = get_compile_time_arg_val(25);

    const uint32_t num_units = get_arg_val<uint32_t>(0);
    const uint32_t scale_u32 = get_arg_val<uint32_t>(1);

    // ---- Boot ----
    compute_kernel_hw_startup(cb_q, cb_k, cb_scores);
    mm_init(cb_q, cb_k, cb_scores);

    CircularBuffer q_buf(cb_q), k_buf(cb_k), v_buf(cb_v);
    CircularBuffer scores_buf(cb_scores), p_buf(cb_p);
    CircularBuffer out_accum_buf(cb_out_accum), pv_buf(cb_pv);

    DEVICE_PRINT("C:boot num_units={}\n", num_units);
    for (uint32_t u = 0; u < num_units; ++u) {
        // Phase 0: scale Q in-place by `scale`.
        // DEBUG EXPERIMENT 4: chain with NO SFPU op (copy+pack only) to test whether the
        // SFPU MulUnary is what clobbers matmul state vs the chain's tile_regs/sync.
        pack_reconfig_data_format(cb_q);
        eltwise_chain(
            Dt,
            CopyTile<cb_q, Dst::D0, CopyTilePolicy::WaitAndPop>{},
            PackTile<cb_q, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
        pack_reconfig_data_format(cb_scores);  // DEBUG: restore for phase A
        DEVICE_PRINT_UNPACK("C:scaleQ done UNPACK\n");
        DEVICE_PRINT_MATH("C:scaleQ done MATH\n");
        DEVICE_PRINT_PACK("C:scaleQ done PACK\n");

        for (uint32_t j = 0; j < num_kv_chunks; ++j) {
            DEVICE_PRINT_UNPACK("C:preA UNPACK j={}\n", j);
            DEVICE_PRINT_MATH("C:preA MATH j={}\n", j);
            DEVICE_PRINT_PACK("C:preA PACK j={}\n", j);
            // A: cb_scores = (scaled Q)·Kⱼᵀ. transpose=true (within-tile) + reader
            // block-transpose => Q·Kᵀ. Q retained across KV blocks (popped in L).
            matmul_block<
                /*transpose=*/true,
                /*packer_l1_acc=*/false,
                LastBlockTarget::Out,
                OutputCBLayout::TileRowMajor,
                matmul_config::InitMode::Short,
                InputPolicy::WaitAndRetainOnLastBlock,
                InputPolicy::WaitAndPopPerKBlock>(
                q_buf,
                k_buf,
                scores_buf,
                scores_buf,
                MatmulBlockShape::of(/*in0_sb=*/1, in1sb_qk, /*sbh=*/1, osw_qk, /*in0_block_k=*/Dt, /*num_k=*/1));
            DEVICE_PRINT_UNPACK("C:A(QK) UNPACK j={} done\n", j);
            DEVICE_PRINT_MATH("C:A(QK) MATH j={} done\n", j);
            DEVICE_PRINT_PACK("C:A(QK) PACK j={} done\n", j);

            // B: cb_scores += maskⱼ (element-wise).
            if constexpr (use_mask) {
                binary_add<cb_scores, cb_mask, cb_scores, BroadcastDim::None>(kv_chunk_t);
            }

            // C: row-max over keys. j=0 -> running m (cb_max); j>0 -> block max (cb_mblock).
            if (j == 0) {
                reduce<
                    ckernel::PoolType::MAX,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_scores,
                    cb_scaler_max,
                    cb_max,
                    ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(1, kv_chunk_t));
            } else {
                reduce<
                    ckernel::PoolType::MAX,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_scores,
                    cb_scaler_max,
                    cb_mblock,
                    ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(1, kv_chunk_t));

                // D1: m_new = max(m_prev, m_blk).
                eltwise_chain(
                    1,
                    CopyTile<cb_max, Dst::D0, CopyTilePolicy::WaitNoPop>{},
                    CopyTile<cb_mblock, Dst::D1, CopyTilePolicy::WaitAndPop>{},
                    BinaryMax<Dst::D0, Dst::D1, Dst::D0>{},
                    PackTile<cb_mnew, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});

                // D2: α = exp(m_prev − m_new). Consumes m_prev (cb_max pop), keeps cb_mnew.
                eltwise_chain(
                    1,
                    BinaryFpu<
                        cb_max,
                        cb_mnew,
                        BinaryFpuOp::Sub,
                        BroadcastDim::None,
                        BinaryDataFormatReconfig::Input,
                        CopyTilePolicy::WaitAndPop,
                        CopyTilePolicy::WaitNoPop>{},
                    Exp<>{},
                    PackTile<cb_alpha, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});

                // D3: m_prev ← m_new.
                copy<cb_mnew, cb_max>(1);
            }

            DEVICE_PRINT("C:C(rowmax) j={} done\n", j);
            // E: cb_p = exp(cb_scores − m), m broadcast across columns.
            eltwise_chain(
                kv_chunk_t,
                BinaryFpu<
                    cb_scores,
                    cb_max,
                    BinaryFpuOp::Sub,
                    BroadcastDim::Col,
                    BinaryDataFormatReconfig::Input,
                    CopyTilePolicy::WaitAndPop,
                    CopyTilePolicy::WaitNoPop,
                    CbIndexMode::FirstTile,
                    Dst::D0,
                    CbIndexMode::FirstTile>{},
                Exp<>{},
                PackTile<cb_p, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});

            DEVICE_PRINT("C:E(exp P) j={} done\n", j);
            // F: row-sum over keys. j=0 -> running l (cb_sum); j>0 -> block sum (cb_lblock).
            if (j == 0) {
                reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_p,
                    cb_scaler_sum,
                    cb_sum,
                    ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(1, kv_chunk_t));
            } else {
                reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_p,
                    cb_scaler_sum,
                    cb_lblock,
                    ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(1, kv_chunk_t));

                // G: l_new = α·l_prev + l_blk.
                eltwise_chain(
                    1,
                    BinaryFpu<
                        cb_sum,
                        cb_alpha,
                        BinaryFpuOp::Mul,
                        BroadcastDim::None,
                        BinaryDataFormatReconfig::Input,
                        CopyTilePolicy::WaitAndPop,
                        CopyTilePolicy::WaitNoPop>{},
                    DestReuseBinary<
                        cb_lblock,
                        BinaryFpuOp::Add,
                        DestReuseType::DEST_TO_SRCA,
                        Dst::D0,
                        Dst::D0,
                        DestReuseReconfig::Input,
                        CopyTilePolicy::WaitAndPop>{},
                    PackTile<cb_sum, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});
            }

            DEVICE_PRINT("C:F(rowsum) j={} done\n", j);
            // H: PV = cb_p·Vⱼ. j=0 -> running O (cb_out_accum); j>0 -> block PV (cb_pv).
            if (j == 0) {
                matmul_block<
                    /*transpose=*/false,
                    /*packer_l1_acc=*/false,
                    LastBlockTarget::Out,
                    OutputCBLayout::TileRowMajor,
                    matmul_config::InitMode::Short,
                    InputPolicy::WaitAndPopPerKBlock,
                    InputPolicy::WaitAndPopPerKBlock>(
                    p_buf,
                    v_buf,
                    out_accum_buf,
                    out_accum_buf,
                    MatmulBlockShape::of(
                        /*in0_sb=*/1,
                        in1sb_pv,
                        /*sbh=*/1,
                        osw_pv,
                        /*in0_block_k=*/kv_chunk_t,
                        /*num_k=*/1));
            } else {
                matmul_block<
                    /*transpose=*/false,
                    /*packer_l1_acc=*/false,
                    LastBlockTarget::Out,
                    OutputCBLayout::TileRowMajor,
                    matmul_config::InitMode::Short,
                    InputPolicy::WaitAndPopPerKBlock,
                    InputPolicy::WaitAndPopPerKBlock>(
                    p_buf,
                    v_buf,
                    pv_buf,
                    pv_buf,
                    MatmulBlockShape::of(
                        /*in0_sb=*/1,
                        in1sb_pv,
                        /*sbh=*/1,
                        osw_pv,
                        /*in0_block_k=*/kv_chunk_t,
                        /*num_k=*/1));

                // I: O_new = α·O_prev + PV, α broadcast across columns.
                eltwise_chain(
                    Dt,
                    BinaryFpu<
                        cb_out_accum,
                        cb_alpha,
                        BinaryFpuOp::Mul,
                        BroadcastDim::Col,
                        BinaryDataFormatReconfig::Input,
                        CopyTilePolicy::WaitAndPop,
                        CopyTilePolicy::WaitNoPop,
                        CbIndexMode::FirstTile,
                        Dst::D0,
                        CbIndexMode::FirstTile>{},
                    DestReuseBinary<
                        cb_pv,
                        BinaryFpuOp::Add,
                        DestReuseType::DEST_TO_SRCA,
                        Dst::D0,
                        Dst::D0,
                        DestReuseReconfig::Input,
                        CopyTilePolicy::WaitAndPop>{},
                    PackTile<cb_out_accum, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});

                cb_pop_front(cb_alpha, 1);  // release α (kept WaitNoPop by G and I)
            }
            DEVICE_PRINT("C:H(PV) j={} done\n", j);
        }

        // J: recip = 1/l.
        unary<Recip<>, cb_sum, cb_recip>(1);
        DEVICE_PRINT("C:J(recip) done\n");

        // K: cb_out = O_accum·recip (bcast col → bf16).
        eltwise_chain(
            Dt,
            BinaryFpu<
                cb_out_accum,
                cb_recip,
                BinaryFpuOp::Mul,
                BroadcastDim::Col,
                BinaryDataFormatReconfig::Input,
                CopyTilePolicy::WaitAndPop,
                CopyTilePolicy::WaitNoPop,
                CbIndexMode::FirstTile,
                Dst::D0,
                CbIndexMode::FirstTile>{},
            PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{});

        cb_pop_front(cb_recip, 1);  // release recip (kept WaitNoPop by K)
        cb_pop_front(cb_q, Dt);     // L: release retained Q
        cb_pop_front(cb_max, 1);    // drain dead running-max m
        DEVICE_PRINT("C:K(normalize) unit done\n");
    }
    DEVICE_PRINT("C:ALL done\n");
}
