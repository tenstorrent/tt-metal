// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// Compute for fused_experts_prefill. One instance per worker core; the 8 cores of a group run the
// same experts in lock-step, each owning 1/8 of the I dim (phase 1) and 1/8 of the H dim (phase 2).
//
// Per owned expert (count tokens -> m = ceil(count / 32) tile rows), per chunk of m_chunk tile rows,
// the rows are processed in M-blocks of m_block tile rows. M is the INNERMOST loop of every matmul:
// each weight tile is unpacked once and multiplied into m_block DST accumulators.
//
// PHASE 1 -- per M-block:
//   * tilize: the reader delivers each tile row as row-major segments (cb_rm); they are tilized into
//     cb_x, tile (m, k) at m * kt + k. A short last block pads cb_x with an uninitialised push so that
//     every block occupies exactly m_block * kt tiles (constant ring stride).
//   * per I-tile jj owned by this core:
//       gate[m] = x[m] @ gate_up_w[:, jj, gate], up[m] = x[m] @ gate_up_w[:, jj, up]   (kt matmul_tiles
//       each, k outer / m inner), staged through the fp32 CB cb_mm (gate at m, up at m_block + m), then
//       act[m, jj] = silu(min(gate, limit)) * clamp(up, -limit, limit) * w_token   -> cb_act_local (bf8)
//     where w_token is the tile row's per-token routing weight (a scalar tile per row from the reader:
//     row r of the tile holds the weight of token r in all columns). The down matmul is linear, so
//     applying the weight to act is the same as weighting the expert output.
//
// (the writer all-gathers the eight cores' act blocks into cb_act_full: [core][r][jj]; the K order of
//  the down matmul is core-major since core c owns I-tiles c*it_pc + jj)
//
// PHASE 2 -- per M-block: for every output tile n owned by this core (nt_pc of them)
//     y[m, n] = sum_k act_full[k, m] @ down_w[k, n]     (k outer / m inner)
//   -> cb_outt (bf16 tiles [m][n]), then untilized row by row into cb_outrm (row-major, 32 rows of this
//   core's nt_pc*32 columns) for the writer, which scatters each valid row to its (slot, token) row.
//
// All CB push/pop sizes are constants so the rings' pointers wrap on exact equality regardless of each
// expert's m (short blocks pad their cb_x / cb_outt with uninitialised push/pop).
//
// Compile-time args:
//   0 cb_meta 1 cb_x 2 cb_w 3 cb_mm 4 cb_act_local 5 cb_act_full 6 cb_outt 7 cb_rm 8 cb_outrm 9 cb_rscal
//   10 m_chunk 11 m_block 12 kt 13 it 14 it_pc 15 nt_pc 16 slot_tiles 17 limit_bits
//   18 act_local_tiles 19 act_full_tiles 20 rm_chunk_tiles
// Runtime args:
//   0 n (experts owned by this group)
void kernel_main() {
    constexpr uint32_t cb_meta_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_mm_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_act_local_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_act_full_id = get_compile_time_arg_val(5);
    constexpr uint32_t cb_outt_id = get_compile_time_arg_val(6);
    constexpr uint32_t cb_rm_id = get_compile_time_arg_val(7);
    constexpr uint32_t cb_outrm_id = get_compile_time_arg_val(8);
    constexpr uint32_t cb_rscal_id = get_compile_time_arg_val(9);
    constexpr uint32_t m_chunk = get_compile_time_arg_val(10);
    constexpr uint32_t m_block = get_compile_time_arg_val(11);
    constexpr uint32_t kt = get_compile_time_arg_val(12);
    constexpr uint32_t it = get_compile_time_arg_val(13);
    constexpr uint32_t it_pc = get_compile_time_arg_val(14);
    constexpr uint32_t nt_pc = get_compile_time_arg_val(15);
    constexpr uint32_t slot_tiles = get_compile_time_arg_val(16);
    constexpr uint32_t limit_bits = get_compile_time_arg_val(17);
    constexpr uint32_t act_local_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t act_full_tiles = get_compile_time_arg_val(19);
    constexpr uint32_t rm_chunk_tiles = get_compile_time_arg_val(20);

    const uint32_t n_owned = get_arg_val<uint32_t>(0);

    // gate: clamp(min = -inf, max = limit); up: clamp(min = -limit, max = limit).
    constexpr uint32_t kNegInfBits = 0xFF800000u;
    constexpr uint32_t neg_limit_bits = limit_bits ^ 0x80000000u;

    constexpr uint32_t rm_chunks = kt / rm_chunk_tiles;
    // act tiles produced per core = m_chunk * it_pc; a peer core's block starts at c' * that.
    constexpr uint32_t act_block_tiles = act_local_tiles;

    CircularBuffer meta_cb(cb_meta_id);
    CircularBuffer x_cb(cb_x_id);
    CircularBuffer w_cb(cb_w_id);
    CircularBuffer mm_cb(cb_mm_id);
    CircularBuffer act_local_cb(cb_act_local_id);
    CircularBuffer act_full_cb(cb_act_full_id);
    CircularBuffer outt_cb(cb_outt_id);
    CircularBuffer rscal_cb(cb_rscal_id);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_id, cb_w_id, cb_mm_id);
    matmul_init(cb_x_id, cb_w_id);

    // The routing lists are pushed once by the reader and stay resident.
    meta_cb.wait_front(1);

    for (uint32_t jl = 0; jl < n_owned; ++jl) {
        // This expert's token count via the UNPACK -> {MATH, PACK} mailbox (MATH / PACK cannot read
        // the counts L1 through the CB interface).
        uint32_t count_value = 0;
        UNPACK(({
            const uint32_t meta_l1_addr = get_local_cb_interface(cb_meta_id).fifo_rd_ptr << 4;
            const volatile tt_l1_ptr uint32_t* counts_ptr =
                reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(meta_l1_addr);
            count_value = counts_ptr[jl];
            ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, count_value);
            ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, count_value);
        }));
        MATH(count_value = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        PACK(count_value = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);)
        if (count_value == 0) {
            continue;
        }
        const uint32_t m = (count_value + 31) / 32;

        for (uint32_t r0 = 0; r0 < m; r0 += m_chunk) {
            const uint32_t mc = (m - r0) < m_chunk ? (m - r0) : m_chunk;

            // ======================= PHASE 1: gate_up + SwiGLU =======================
            {
                DeviceZoneScopedN("C_WAIT_GU");
                w_cb.wait_front(slot_tiles);
            }
            act_local_cb.reserve_back(act_local_tiles);

            for (uint32_t b0 = 0; b0 < mc; b0 += m_block) {
                const uint32_t mb = (mc - b0) < m_block ? (mc - b0) : m_block;

                // ---- tilize this block's rows into cb_x ----
                // One 16-tile block per row-major segment, in the order the reader pushes them.
                {
                    DeviceZoneScopedN("C_TILIZE");  // includes waiting for the reader's row segments / scalars
                    compute_kernel_lib::tilize<rm_chunk_tiles, cb_rm_id, cb_x_id>(mb * rm_chunks);
                    if (mb < m_block) {
                        x_cb.reserve_back((m_block - mb) * kt);
                        x_cb.push_back((m_block - mb) * kt);
                    }
                    x_cb.wait_front(m_block * kt);
                    rscal_cb.wait_front(m_block);
                }

                for (uint32_t jj = 0; jj < it_pc; ++jj) {
                    const uint32_t w_base = jj * kt * 2;

                    {
                        DeviceZoneScopedN("C_MM_GATEUP");
                        matmul_init(cb_x_id, cb_w_id);
                        reconfig_full_operand(cb_w_id, cb_x_id);
                        pack_reconfig_data_format(cb_mm_id);
                        mm_cb.reserve_back(2 * m_block);

                        // gate: K outer, M inner (each weight tile feeds mb accumulators)
                        tile_regs_acquire();
                        for (uint32_t k = 0; k < kt; ++k) {
                            for (uint32_t mi = 0; mi < mb; ++mi) {
                                matmul_tiles(cb_x_id, cb_w_id, mi * kt + k, w_base + k * 2, mi);
                            }
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t mi = 0; mi < mb; ++mi) {
                            pack_tile<true>(mi, cb_mm_id, mi);
                        }
                        tile_regs_release();

                        // up
                        tile_regs_acquire();
                        for (uint32_t k = 0; k < kt; ++k) {
                            for (uint32_t mi = 0; mi < mb; ++mi) {
                                matmul_tiles(cb_x_id, cb_w_id, mi * kt + k, w_base + k * 2 + 1, mi);
                            }
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t mi = 0; mi < mb; ++mi) {
                            pack_tile<true>(mi, cb_mm_id, m_block + mi);
                        }
                        tile_regs_release();

                        mm_cb.push_back(2 * m_block);
                    }

                    {
                        DeviceZoneScopedN("C_SWIGLU");
                        mm_cb.wait_front(2 * m_block);
                        pack_reconfig_data_format(cb_act_local_id);
                        for (uint32_t mi = 0; mi < mb; ++mi) {
                            copy_tile_to_dst_init_short(cb_mm_id);
                            reconfig_full_operand_srca(cb_mm_id);

                            tile_regs_acquire();
                            copy_tile(cb_mm_id, mi, 0);
                            copy_tile(cb_mm_id, m_block + mi, 1);
                            copy_tile_to_dst_init_short(cb_rscal_id);
                            reconfig_full_operand_srca(cb_rscal_id);
                            copy_tile(cb_rscal_id, mi, 2);
                            clamp_tile_init();
                            clamp_tile(0, kNegInfBits, limit_bits);
                            silu_tile_init();
                            silu_tile(0);
                            clamp_tile_init();
                            clamp_tile(1, neg_limit_bits, limit_bits);
                            mul_binary_tile_init();
                            mul_binary_tile(0, 1, 0);
                            mul_binary_tile(0, 2, 0);
                            tile_regs_commit();
                            tile_regs_wait();
                            pack_tile<true>(0, cb_act_local_id, (b0 + mi) * it_pc + jj);
                            tile_regs_release();
                        }
                        mm_cb.pop_front(2 * m_block);
                    }
                }
                rscal_cb.pop_front(m_block);
                x_cb.pop_front(m_block * kt);
            }

            act_local_cb.push_back(act_local_tiles);
            w_cb.pop_front(slot_tiles);

            // ======================= PHASE 2: down =======================
            // Published by the writer once all 8 cores' act blocks have landed.
            {
                DeviceZoneScopedN("C_WAIT_ACT");  // gathered activations (writer) + down weight slot (reader)
                act_full_cb.wait_front(act_full_tiles);
                w_cb.wait_front(slot_tiles);
            }

            for (uint32_t b0 = 0; b0 < mc; b0 += m_block) {
                const uint32_t mb = (mc - b0) < m_block ? (mc - b0) : m_block;

                {
                    DeviceZoneScopedN("C_MM_DOWN");
                    outt_cb.reserve_back(m_block * nt_pc);

                    matmul_init(cb_act_full_id, cb_w_id);
                    reconfig_full_operand(cb_w_id, cb_act_full_id);
                    pack_reconfig_data_format(cb_outt_id);

                    for (uint32_t n = 0; n < nt_pc; ++n) {
                        const uint32_t w_base = (n / 2) * (it * 2) + (n % 2);
                        tile_regs_acquire();
                        for (uint32_t k = 0; k < it; ++k) {
                            for (uint32_t mi = 0; mi < mb; ++mi) {
                                // act_full layout: [core = k / it_pc][r][jj = k % it_pc].
                                const uint32_t a_idx = (k / it_pc) * act_block_tiles + (b0 + mi) * it_pc + (k % it_pc);
                                matmul_tiles(cb_act_full_id, cb_w_id, a_idx, w_base + k * 2, mi);
                            }
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t mi = 0; mi < mb; ++mi) {
                            pack_tile<true>(mi, cb_outt_id, mi * nt_pc + n);
                        }
                        tile_regs_release();
                    }
                    outt_cb.push_back(m_block * nt_pc);
                }

                {
                    DeviceZoneScopedN("C_UNTILIZE");  // includes waiting for the writer to drain cb_outrm
                    for (uint32_t mi = 0; mi < mb; ++mi) {
                        compute_kernel_lib::untilize<nt_pc, cb_outt_id, cb_outrm_id>(1);
                    }
                    if (mb < m_block) {
                        outt_cb.pop_front((m_block - mb) * nt_pc);
                    }
                }
            }

            w_cb.pop_front(slot_tiles);
            act_full_cb.pop_front(act_full_tiles);
        }
    }
}
