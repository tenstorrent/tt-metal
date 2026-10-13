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

// Compute for fused_experts_prefill. One instance per worker core; the group_cores cores of a group run the
// same experts in lock-step, each owning a run of the I dim (phase 1: n_c I tiles, it_base or it_base + 1 of them)
// and a run of the H dim (phase 2: nt_pc tile columns, at most nt_max).
//
// Per owned expert (count tokens -> m = ceil(count / 32) tile rows), per chunk of m_chunk tile rows,
// the rows are processed in M-blocks of m_block tile rows. M is the INNERMOST loop of every matmul:
// each weight tile is unpacked once and multiplied into m_block DST accumulators.
//
// PHASE 1 -- per M-block:
//   * tilize: the reader delivers each tile row as row-major segments (cb_rm); they are tilized into
//     cb_x, tile (m, k) at m * kt + k. A short last block pads cb_x with an uninitialised push so that
//     every block occupies exactly m_block * kt tiles (constant ring stride).
//   * the weights arrive in slots of up to spp gate_up shards (one shard = one I tile): RESIDENT, one slot
//     serves every M-block of the chunk (waited before the blocks, popped after); STREAMED, the core's
//     ceil(n_c / spp) slots pass through the ring once per M-block, each waited and popped around its shards.
//   * per I-tile jj owned by this core:
//       gate[m] = x[m] @ gate_up_w[:, jj, gate], up[m] = x[m] @ gate_up_w[:, jj, up]   (kt matmul_tiles
//       each, k outer / m inner), staged through the fp32 CB cb_mm (gate at m, up at m_block + m), then
//       act[m, jj] = silu(min(gate, limit)) * clamp(up, -limit, limit) * w_token   -> cb_act_local (bf8)
//     where w_token is the tile row's per-token routing weight (a scalar tile per row from the reader:
//     row r of the tile holds the weight of token r in all columns). The down matmul is linear, so
//     applying the weight to act is the same as weighting the expert output.
//
// (the writer all-gathers the group's act blocks into cb_act_full: [core][r][jj]; the K order of
//  the down matmul is core-major since core c owns the consecutive I-tiles [first_c, first_c + n_c), so
//  walking the cores in order and jj within each is the global I order)
//
// PHASE 2 -- per M-block: for every output tile n owned by this core (nt_pc of them)
//     y[m, n] = sum_k act_full[k, m] @ down_w[k, n]     (k outer / m inner)
//   -> cb_outt (bf16 tiles [m][n], rows at a stride of nt_max), then untilized row by row into cb_outrm
//   (row-major, 32 rows of this core's nt_pc*32 columns, padded to nt_max pages) for the writer, which
//   scatters each valid row to its (slot, token) row. nt_max is the most columns any core of the group
//   owns, so every core moves the same pages through cb_outt / cb_outrm.
//
// All CB push/pop sizes are constants so the rings' pointers wrap on exact equality regardless of each
// expert's m (short blocks pad their cb_x / cb_outt with uninitialised push/pop).
//
// Compile-time args:
//   0 cb_meta 1 cb_x 2 cb_w 3 cb_mm 4 cb_act_local 5 cb_act_full 6 cb_outt 7 cb_rm 8 cb_outrm 9 cb_rscal
//   10 m_chunk 11 m_block 12 kt 13 it 14 it_max (act block stride) 15 nt_pc 16 slot_tiles 17 limit_bits
//   18 act_local_tiles 19 act_full_tiles 20 rm_chunk_tiles 21 spp 22 resident 23 it_base 24 it_rem
//   25 dpp (down shards per slot) 26 dn_resident 27 nt_max 28 group_cores
// Runtime args:
//   0 n (experts owned by this group)  1 n_c (this core's gate_up shards)
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
    constexpr uint32_t it_max = get_compile_time_arg_val(14);
    constexpr uint32_t nt_pc = get_compile_time_arg_val(15);
    constexpr uint32_t slot_tiles = get_compile_time_arg_val(16);
    constexpr uint32_t limit_bits = get_compile_time_arg_val(17);
    constexpr uint32_t act_local_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t act_full_tiles = get_compile_time_arg_val(19);
    constexpr uint32_t rm_chunk_tiles = get_compile_time_arg_val(20);
    constexpr uint32_t spp = get_compile_time_arg_val(21);
    constexpr bool resident = get_compile_time_arg_val(22) == 1;
    constexpr uint32_t it_base = get_compile_time_arg_val(23);
    constexpr uint32_t it_rem = get_compile_time_arg_val(24);
    constexpr uint32_t dpp = get_compile_time_arg_val(25);
    constexpr bool dn_resident = get_compile_time_arg_val(26) == 1;
    constexpr uint32_t nt_max = get_compile_time_arg_val(27);
    constexpr uint32_t kGroupCores = get_compile_time_arg_val(28);
    constexpr uint32_t dn_shards = nt_pc / 2;

    const uint32_t n_owned = get_arg_val<uint32_t>(0);
    const uint32_t n_c = get_arg_val<uint32_t>(1);

    // gate: clamp(min = -inf, max = limit); up: clamp(min = -limit, max = limit).
    constexpr uint32_t kNegInfBits = 0xFF800000u;
    constexpr uint32_t neg_limit_bits = limit_bits ^ 0x80000000u;

    constexpr uint32_t rm_chunks = kt / rm_chunk_tiles;
    // act tiles produced per core = m_chunk * it_max; a peer core's block starts at c' * that.
    constexpr uint32_t act_block_tiles = act_local_tiles;

    CircularBuffer meta_cb(cb_meta_id);
    CircularBuffer x_cb(cb_x_id);
    CircularBuffer w_cb(cb_w_id);
    CircularBuffer mm_cb(cb_mm_id);
    CircularBuffer act_local_cb(cb_act_local_id);
    CircularBuffer act_full_cb(cb_act_full_id);
    CircularBuffer outt_cb(cb_outt_id);
    CircularBuffer outrm_cb(cb_outrm_id);
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
            if constexpr (resident) {
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

                for (uint32_t slot0 = 0; slot0 < n_c; slot0 += spp) {
                    if constexpr (!resident) {
                        DeviceZoneScopedN("C_WAIT_GU");
                        w_cb.wait_front(slot_tiles);
                    }
                    const uint32_t in_slot = (n_c - slot0) < spp ? (n_c - slot0) : spp;

                    for (uint32_t js = 0; js < in_slot; ++js) {
                        const uint32_t jj = slot0 + js;
                        const uint32_t w_base = js * kt * 2;
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
                                pack_tile<true>(0, cb_act_local_id, (b0 + mi) * it_max + jj);
                                tile_regs_release();
                            }
                            mm_cb.pop_front(2 * m_block);
                        }
                    }
                    if constexpr (!resident) {
                        w_cb.pop_front(slot_tiles);
                    }
                }
                rscal_cb.pop_front(m_block);
                x_cb.pop_front(m_block * kt);
            }

            act_local_cb.push_back(act_local_tiles);
            if constexpr (resident) {
                w_cb.pop_front(slot_tiles);
            }

            // ======================= PHASE 2: down =======================
            // Published by the writer once all the group's act blocks have landed.
            {
                DeviceZoneScopedN("C_WAIT_ACT");  // gathered activations (writer) + down weight slot (reader)
                act_full_cb.wait_front(act_full_tiles);
                if constexpr (dn_resident) {
                    w_cb.wait_front(slot_tiles);
                }
            }

            for (uint32_t b0 = 0; b0 < mc; b0 += m_block) {
                const uint32_t mb = (mc - b0) < m_block ? (mc - b0) : m_block;

                {
                    DeviceZoneScopedN("C_MM_DOWN");
                    outt_cb.reserve_back(m_block * nt_max);

                    matmul_init(cb_act_full_id, cb_w_id);
                    reconfig_full_operand(cb_w_id, cb_act_full_id);
                    pack_reconfig_data_format(cb_outt_id);

                    // The down shards ([I, 64] = output tile columns 2 * shard, 2 * shard + 1) come in slots of up to
                    // dpp: one slot kept across the M-blocks, or streamed through the ring once per M-block.
                    for (uint32_t slot0 = 0; slot0 < dn_shards; slot0 += dpp) {
                        if constexpr (!dn_resident) {
                            w_cb.wait_front(slot_tiles);
                        }
                        const uint32_t in_slot = (dn_shards - slot0) < dpp ? (dn_shards - slot0) : dpp;
                        for (uint32_t n_local = 0; n_local < 2 * in_slot; ++n_local) {
                            const uint32_t n = 2 * slot0 + n_local;
                            const uint32_t w_base = (n_local / 2) * (it * 2) + (n_local % 2);
                            tile_regs_acquire();
                            // act_full layout: [core][r][jj] with core cp's it_base (+1 for the first it_rem cores)
                            // tiles per row at a stride of it_max; k walks the cores in order = the global I order.
                            uint32_t k = 0;
                            for (uint32_t cp = 0; cp < kGroupCores; ++cp) {
                                const uint32_t n_cp = it_base + (cp < it_rem ? 1 : 0);
                                for (uint32_t jj = 0; jj < n_cp; ++jj, ++k) {
                                    for (uint32_t mi = 0; mi < mb; ++mi) {
                                        const uint32_t a_idx = cp * act_block_tiles + (b0 + mi) * it_max + jj;
                                        matmul_tiles(cb_act_full_id, cb_w_id, a_idx, w_base + k * 2, mi);
                                    }
                                }
                            }
                            tile_regs_commit();
                            tile_regs_wait();
                            for (uint32_t mi = 0; mi < mb; ++mi) {
                                pack_tile<true>(mi, cb_outt_id, mi * nt_max + n);
                            }
                            tile_regs_release();
                        }
                        if constexpr (!dn_resident) {
                            w_cb.pop_front(slot_tiles);
                        }
                    }
                    outt_cb.push_back(m_block * nt_max);
                }

                {
                    DeviceZoneScopedN("C_UNTILIZE");  // includes waiting for the writer to drain cb_outrm
                    for (uint32_t mi = 0; mi < mb; ++mi) {
                        compute_kernel_lib::untilize<nt_pc, cb_outt_id, cb_outrm_id>(1);
                        if constexpr (nt_max > nt_pc) {
                            outt_cb.pop_front(nt_max - nt_pc);
                            outrm_cb.reserve_back(nt_max - nt_pc);
                            outrm_cb.push_back(nt_max - nt_pc);
                        }
                    }
                    if (mb < m_block) {
                        outt_cb.pop_front((m_block - mb) * nt_max);
                    }
                }
            }

            if constexpr (dn_resident) {
                w_cb.pop_front(slot_tiles);
            }
            act_full_cb.pop_front(act_full_tiles);
        }
    }
}
