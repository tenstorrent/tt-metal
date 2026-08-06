// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// [#48552 DFB-ONLY STRIP] Blocker B isolation build of conv_bmm_tilize_metal2.cpp.
//
// ALL COMPUTATION has been removed, leaving ONLY the DFB (DataflowBuffer / circular-buffer) credit
// handshake: every wait_front / reserve_back / push_back / pop_front is preserved verbatim (same CB,
// same tile counts, same order, same loop/branch structure), plus the matmul-partials ring snapshot/
// restore (PARTIALS_* macros, which are DFB ring-state ops) and the TEN-4746 TTI_NOP interpose guards.
//
// Removed: compute_kernel_hw_startup, matmul_block_init / matmul_block, tilize / untilize compute,
// pack_tile / pack_block / copy_tile / copy_block, tile_regs_*, all reconfig_* / pack_reconfig_* /
// llk_* init, bias add, SFPU, and every DPRINT diagnostic. The three helpers that hid CB ops inside
// compute (tilize_in, reblock_and_untilize, compute_kernel_lib::untilize) are replaced by DFB-only
// loops with the IDENTICAL per-call CB accounting.
//
// PURPOSE: determine whether the BLOCK_SHARDED 0x0119 hang reproduces from the DFB credit handshake
// alone (no matmul / tilize / pack). If it still hangs -> the fault is in the DFB/credit machinery,
// not the compute LLK ops. If it runs clean -> the compute path is required to trip it.
//
// To restore the real kernel: `git checkout <pre-strip> -- <this file>`.

#include <cstdint>

#include "internal/mod_div_lib.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// In-place matmul-partials accumulate ring snapshot/restore (DFB ring-state ops, KEPT). See the original
// file history for the full rationale; on Quasar these poke g_dfb_interface's wr_entry_idx / rd_entry_idx.
#ifdef ARCH_QUASAR
struct QsrDfbRingPos {
    uint16_t entry_idx[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];
    uint16_t offset[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];
    uint16_t entry_ptr;
    uint8_t tc_idx;
};
using PartialsRingPos = QsrDfbRingPos;
#define QSR_SNAPSHOT_WR(pos, dfb)                                         \
    do {                                                                  \
        LocalDFBInterface& _qd = get_local_dfb_interface((dfb).get_id()); \
        for (uint8_t _qi = 0; _qi < _qd.num_tcs_to_rr; ++_qi) {           \
            (pos).entry_idx[_qi] = _qd.tc_slots[_qi].wr_entry_idx;        \
        }                                                                 \
        (pos).entry_ptr = _qd.wr_entry_ptr;                               \
        (pos).tc_idx = _qd.tc_idx;                                        \
    } while (0)
#define QSR_RESTORE_WR(pos, dfb)                                          \
    do {                                                                  \
        LocalDFBInterface& _qd = get_local_dfb_interface((dfb).get_id()); \
        for (uint8_t _qi = 0; _qi < _qd.num_tcs_to_rr; ++_qi) {           \
            _qd.tc_slots[_qi].wr_entry_idx = (pos).entry_idx[_qi];        \
        }                                                                 \
        _qd.wr_entry_ptr = (pos).entry_ptr;                               \
        _qd.tc_idx = (pos).tc_idx;                                        \
    } while (0)
#define QSR_SNAPSHOT_RD(pos, dfb)                                         \
    do {                                                                  \
        LocalDFBInterface& _qd = get_local_dfb_interface((dfb).get_id()); \
        for (uint8_t _qi = 0; _qi < _qd.num_tcs_to_rr; ++_qi) {           \
            (pos).entry_idx[_qi] = _qd.tc_slots[_qi].rd_entry_idx;        \
        }                                                                 \
        (pos).tc_idx = _qd.tc_idx;                                        \
    } while (0)
#define QSR_RESTORE_RD(pos, dfb)                                          \
    do {                                                                  \
        LocalDFBInterface& _qd = get_local_dfb_interface((dfb).get_id()); \
        for (uint8_t _qi = 0; _qi < _qd.num_tcs_to_rr; ++_qi) {           \
            _qd.tc_slots[_qi].rd_entry_idx = (pos).entry_idx[_qi];        \
        }                                                                 \
        _qd.tc_idx = (pos).tc_idx;                                        \
    } while (0)
#define SAVE_PARTIALS_WR(var, dfb) \
    PartialsRingPos var;           \
    QSR_SNAPSHOT_WR(var, dfb)
#define SAVE_PARTIALS_RD(var, dfb) \
    PartialsRingPos var;           \
    QSR_SNAPSHOT_RD(var, dfb)
#define RESAVE_PARTIALS_WR(var, dfb) QSR_SNAPSHOT_WR(var, dfb)
#define RESAVE_PARTIALS_RD(var, dfb) QSR_SNAPSHOT_RD(var, dfb)
#define RESTORE_PARTIALS_WR(var, dfb) QSR_RESTORE_WR(var, dfb)
#define RESTORE_PARTIALS_RD(var, dfb) QSR_RESTORE_RD(var, dfb)
#else
using PartialsRingPos = uint32_t;
#define SAVE_PARTIALS_WR(var, dfb) uint32_t var = (dfb).get_write_ptr()
#define SAVE_PARTIALS_RD(var, dfb) uint32_t var = (dfb).get_read_ptr()
#define RESAVE_PARTIALS_WR(var, dfb) var = (dfb).get_write_ptr()
#define RESAVE_PARTIALS_RD(var, dfb) var = (dfb).get_read_ptr()
#define RESTORE_PARTIALS_WR(var, dfb) (dfb).evil_set_write_ptr(var)
#define RESTORE_PARTIALS_RD(var, dfb) (dfb).evil_set_read_ptr(var)
#endif

// [#48552 DFB-ONLY] tilize compute removed. Reproduces compute_kernel_lib::tilize's WaitBlock CB flow:
// per subblock -> in.wait_front(in_block_w); out.reserve_back(in_block_w); out.push_back(in_block_w);
// in.pop_front(in_block_w).
template <
    uint32_t in_block_w,
    uint32_t in_cb_id,
    uint32_t out_cb_id,
    bool init_tilize = true,
    bool uninit_tilize = true,
    compute_kernel_lib::tilize_config::RemapMode remap_mode = compute_kernel_lib::tilize_config::RemapMode::Configure>
void tilize_in(uint32_t in_num_subblocks) {
    DataflowBuffer in_dfb(in_cb_id);
    DataflowBuffer out_dfb(out_cb_id);
    for (uint32_t b = 0; b < in_num_subblocks; ++b) {
        in_dfb.wait_front(in_block_w);
        out_dfb.reserve_back(in_block_w);
        out_dfb.push_back(in_block_w);
        in_dfb.pop_front(in_block_w);
    }
}

// [#48552 DFB-ONLY] untilize (packer_untilize path) compute removed. Reproduces reblock_and_untilize's
// CB flow: interm.wait_front(row); per out_subblock_h -> out.reserve_back(out_block_w) / push_back;
// interm.pop_front(row).
template <uint32_t out_subblock_w, uint32_t out_block_w>
inline void reblock_and_untilize(
    DataflowBuffer& interm_cb,
    DataflowBuffer& out_cb,
    uint32_t num_out_subblocks_in_col,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_h) {
    uint32_t num_tiles_in_row_of_subblocks = mulsi3(out_subblock_num_tiles, num_out_subblocks_in_col);
    interm_cb.wait_front(num_tiles_in_row_of_subblocks);
    for (uint32_t h = 0; h < out_subblock_h; h++) {
        out_cb.reserve_back(out_block_w);
        out_cb.push_back(out_block_w);
    }
    interm_cb.pop_front(num_tiles_in_row_of_subblocks);
}

void kernel_main() {
    [[maybe_unused]] constexpr uint32_t in0_block_w = get_arg(args::in0_block_w);
    [[maybe_unused]] constexpr uint32_t in0_num_subblocks = get_arg(args::in0_num_subblocks);
    [[maybe_unused]] constexpr uint32_t in0_block_num_tiles = get_arg(args::in0_block_num_tiles);
    [[maybe_unused]] constexpr uint32_t in0_subblock_num_tiles = get_arg(args::in0_subblock_num_tiles);
    [[maybe_unused]] constexpr uint32_t reader_num_h_subblocks = get_arg(args::reader_num_h_subblocks);
    [[maybe_unused]] constexpr uint32_t in1_num_subblocks = get_arg(args::in1_num_subblocks);
    [[maybe_unused]] constexpr uint32_t in1_block_num_tiles = get_arg(args::in1_block_num_tiles);
    [[maybe_unused]] constexpr uint32_t in1_block_w = get_arg(args::in1_block_w);
    [[maybe_unused]] constexpr uint32_t in0_num_blocks_h = get_arg(args::in0_num_blocks_h);
    [[maybe_unused]] constexpr uint32_t in0_num_blocks_w = get_arg(args::in0_num_blocks_w);
    [[maybe_unused]] constexpr uint32_t in1_num_blocks_w = get_arg(args::in1_num_blocks_w);
    [[maybe_unused]] constexpr uint32_t out_subblock_h = get_arg(args::out_subblock_h);
    [[maybe_unused]] constexpr uint32_t out_subblock_w = get_arg(args::out_subblock_w);
    [[maybe_unused]] constexpr uint32_t out_subblock_num_tiles = get_arg(args::out_subblock_num_tiles);
    [[maybe_unused]] constexpr bool height_sharded = get_arg(args::height_sharded);
    [[maybe_unused]] constexpr bool untilize_out = get_arg(args::untilize_out);
    [[maybe_unused]] constexpr uint32_t in0_cb_id = dfb::act;
    [[maybe_unused]] constexpr uint32_t in1_cb_id = dfb::weights;
#ifdef HAS_ACT_ROW_MAJOR
    [[maybe_unused]] constexpr uint32_t in0_pretilize_cb_id = dfb::act_row_major;
#else
    [[maybe_unused]] constexpr uint32_t in0_pretilize_cb_id = dfb::act;
#endif
#ifdef SPLIT_READER
    [[maybe_unused]] constexpr uint32_t in0_cb_second_reader_id = dfb::act_second_reader;
#endif
    [[maybe_unused]] constexpr uint32_t matmul_partials_cb = dfb::matmul_partials;
    [[maybe_unused]] constexpr uint32_t tilized_in0_cb_id = dfb::act_tilized;
    [[maybe_unused]] constexpr uint32_t out_cb_id = dfb::out;
    [[maybe_unused]] constexpr bool partials_cb_uses_output = get_arg(args::partials_cb_uses_output);
    [[maybe_unused]] constexpr uint32_t in0_nblocks_w_tilize = get_arg(args::in0_nblocks_w_tilize);
    [[maybe_unused]] constexpr bool pack_relu = get_arg(args::pack_relu);
    [[maybe_unused]] constexpr bool packer_untilize = get_arg(args::packer_untilize);
    [[maybe_unused]] constexpr bool packer_l1_acc = get_arg(args::packer_l1_acc);
    [[maybe_unused]] constexpr bool fuse_bias = get_arg(args::fuse_bias);
    [[maybe_unused]] constexpr bool split_reader = get_arg(args::split_reader);
    [[maybe_unused]] constexpr bool activation_reuse = get_arg(args::activation_reuse);
    [[maybe_unused]] constexpr uint32_t image_width_in_tiles = get_arg(args::image_width_in_tiles);
    [[maybe_unused]] constexpr uint32_t window_reuse_offset = get_arg(args::window_reuse_offset);
    [[maybe_unused]] constexpr uint32_t tilized_cb_row_offset = get_arg(args::tilized_cb_row_offset);
    [[maybe_unused]] constexpr uint32_t tilized_cb_second_reader_offset =
        get_arg(args::tilized_cb_second_reader_offset);
    [[maybe_unused]] constexpr bool split_reader_cb_shared = get_arg(args::split_reader_cb_shared) == 1;

    [[maybe_unused]] constexpr uint32_t out_block_num_tiles =
        in0_num_subblocks * in1_num_subblocks * out_subblock_num_tiles;
    [[maybe_unused]] constexpr uint32_t out_block_w = in1_block_w;
    [[maybe_unused]] constexpr bool spill = in0_num_blocks_w > 1;

    [[maybe_unused]] constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? matmul_partials_cb : out_cb_id;
    [[maybe_unused]] constexpr uint32_t bias_ntiles_w = get_arg(args::bias_ntiles_w);
#ifdef FUSE_BIAS
    [[maybe_unused]] constexpr uint32_t bias_cb_id = dfb::bias;
#endif
    [[maybe_unused]] constexpr uint32_t mm_out_cb_id = fuse_bias ? matmul_partials_cb : untilize_mode_out_cb_id;
    [[maybe_unused]] constexpr uint32_t mm_in0_cb_id = height_sharded ? tilized_in0_cb_id : in0_cb_id;

    [[maybe_unused]] constexpr uint32_t in0_num_subblocks_read_last =
        (split_reader && !split_reader_cb_shared) ? reader_num_h_subblocks / 2 : 0;
    [[maybe_unused]] constexpr uint32_t in0_num_subblocks_read = reader_num_h_subblocks - in0_num_subblocks_read_last;

    [[maybe_unused]] DataflowBuffer cb_in0(in0_cb_id);
#ifdef SPLIT_READER
    [[maybe_unused]] DataflowBuffer cb_in0_second_reader(in0_cb_second_reader_id);
#endif
    [[maybe_unused]] DataflowBuffer cb_tilized_in0(tilized_in0_cb_id);
    [[maybe_unused]] DataflowBuffer cb_mm_in0(mm_in0_cb_id);
    DataflowBuffer cb_in1(in1_cb_id);
    DataflowBuffer cb_matmul_partials(matmul_partials_cb);
    [[maybe_unused]] DataflowBuffer cb_mm_out(mm_out_cb_id);
    DataflowBuffer cb_out(out_cb_id);
#ifdef FUSE_BIAS
    DataflowBuffer cb_bias(bias_cb_id);
#endif
    [[maybe_unused]] DataflowBuffer cb_untilize_mode_out(untilize_mode_out_cb_id);

#ifdef CHECK_SKIP_COMPUTE
    bool skip_compute = (bool)get_arg(args::skip_compute);
#endif

    // [#48552 DFB-ONLY] compute_kernel_hw_startup + matmul_block_init + SFPU init removed.
    UNPACK(SAVE_PARTIALS_RD(partials_cb_read_ptr, cb_matmul_partials);)
    PACK(SAVE_PARTIALS_WR(partials_cb_write_ptr, cb_matmul_partials);)
    for (uint32_t in1_block_w_i = 0; in1_block_w_i < in1_num_blocks_w; ++in1_block_w_i) {
        for (uint32_t in0_block_h_i = 0; in0_block_h_i < in0_num_blocks_h; ++in0_block_h_i) {
            bool enable_reload = false;

            if constexpr (partials_cb_uses_output) {
                UNPACK(RESAVE_PARTIALS_RD(partials_cb_read_ptr, cb_matmul_partials);)
                PACK(RESAVE_PARTIALS_WR(partials_cb_write_ptr, cb_matmul_partials);)
            }
            uint32_t curr_matmul_out_cb = matmul_partials_cb;
            for (uint32_t in0_block_w_i = 0; in0_block_w_i < in0_num_blocks_w; ++in0_block_w_i) {
                bool last_inner_dim_block = (in0_block_w_i == in0_num_blocks_w - 1);
                if constexpr (!height_sharded) {
                    if (in0_block_w_i % in0_nblocks_w_tilize == 0) {
                        // [#48552 DFB-ONLY] pack_relu / pack_reconfig / MATH+PACK re-seed inits removed.
                        tilize_in<
                            in0_block_w,
                            in0_pretilize_cb_id,
                            tilized_in0_cb_id,
                            true,
                            !split_reader || split_reader_cb_shared,
                            compute_kernel_lib::tilize_config::RemapMode::Configure>(in0_num_subblocks_read);
#ifdef SPLIT_READER
                        if constexpr (split_reader && !split_reader_cb_shared) {
                            tilize_in<in0_block_w, in0_cb_second_reader_id, tilized_in0_cb_id, false, true>(
                                in0_num_subblocks_read_last);
                        }
#endif
                        // [#48552 DFB-ONLY] reconfig_data_format + matmul_block_init removed.
                    }
                } else {
                    // [#48552 DFB-ONLY] pack_relu / pack_reconfig / MATH+PACK re-seed inits removed.
                    if constexpr (!activation_reuse) {
                        tilize_in<in0_block_w, in0_cb_id, tilized_in0_cb_id, true, !split_reader>(
                            in0_num_subblocks_read);
                    }
#ifdef SPLIT_READER
                    if constexpr (split_reader) {
                        if constexpr (!activation_reuse) {
                            tilize_in<in0_block_w, in0_cb_second_reader_id, tilized_in0_cb_id, false, true>(
                                in0_num_subblocks_read_last);
                        }
                    }
#endif
                    // [#48552 DFB-ONLY] reconfig_data_format + matmul_block_init removed.
                }

                cb_mm_in0.wait_front(in0_block_num_tiles);

#ifdef CHECK_SKIP_COMPUTE
                if (skip_compute) {
                    cb_mm_in0.pop_front(in0_block_num_tiles);
                    continue;
                }
#endif

                cb_in1.wait_front(in1_block_num_tiles);

                if (last_inner_dim_block) {
                    if constexpr (!fuse_bias) {
                        // [#48552 DFB-ONLY] pack_relu config removed; keep CB-target control flow.
                        curr_matmul_out_cb = mm_out_cb_id;
                    }
                }

                // [#48552 DFB-ONLY] pack_reconfig / llk_pack_init / MATH+PACK re-seed removed.
                for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                    for (uint32_t in1_subblock_i = 0; in1_subblock_i < in1_num_subblocks; ++in1_subblock_i) {
                        if (enable_reload) {
                            // [#48552 DFB-ONLY] copy_tile_to_dst_init / copy_block / matmul_block_init removed;
                            // keep the partials read handshake.
                            cb_matmul_partials.wait_front(out_subblock_num_tiles);
                            cb_matmul_partials.pop_front(out_subblock_num_tiles);
                        }

                        // [#48552 DFB-ONLY] matmul_block MVMUL loop removed.

                        {
                            DataflowBuffer curr_out_cb =
                                curr_matmul_out_cb == matmul_partials_cb ? cb_matmul_partials : cb_mm_out;
                            curr_out_cb.reserve_back(out_subblock_num_tiles);
                            // [#48552 DFB-ONLY] tile_regs_wait / pack_reconfig_l1_acc / pack_tile removed.
                            curr_out_cb.push_back(out_subblock_num_tiles);
                        }
                    }  // for in1_num_subblocks
                }
                if (curr_matmul_out_cb == matmul_partials_cb) {
                    if constexpr (!partials_cb_uses_output) {
                        UNPACK(RESTORE_PARTIALS_RD(partials_cb_read_ptr, cb_matmul_partials);)
                        PACK(RESTORE_PARTIALS_WR(partials_cb_write_ptr, cb_matmul_partials);)
                    }
                }
                if constexpr (packer_l1_acc) {
                    if constexpr (fuse_bias) {
                        if (in0_block_w_i < in0_num_blocks_w - 1) {
                            cb_matmul_partials.wait_front(out_block_num_tiles);
#ifdef ARCH_QUASAR
                            // TEN-4746: TDMA interpose between bare wait_front/pop_front (KEPT).
                            UNPACK(TTI_NOP);
#endif
                            cb_matmul_partials.pop_front(out_block_num_tiles);
                            if constexpr (spill) {
                                UNPACK(RESTORE_PARTIALS_RD(partials_cb_read_ptr, cb_matmul_partials));
                                PACK(RESTORE_PARTIALS_WR(partials_cb_write_ptr, cb_matmul_partials));
                            }
                        }
                        enable_reload = false;
                    } else {
                        if (in0_block_w_i < in0_num_blocks_w - 2) {
                            cb_matmul_partials.wait_front(out_block_num_tiles);
#ifdef ARCH_QUASAR
                            UNPACK(TTI_NOP);
#endif
                            cb_matmul_partials.pop_front(out_block_num_tiles);
                            if constexpr (spill) {
                                UNPACK(RESTORE_PARTIALS_RD(partials_cb_read_ptr, cb_matmul_partials));
                                PACK(RESTORE_PARTIALS_WR(partials_cb_write_ptr, cb_matmul_partials));
                            }
                        }
                        if (in0_block_w_i == in0_num_blocks_w - 2) {
                            enable_reload = true;
                        }
                    }
                } else {
                    if constexpr (spill) {
                        enable_reload = true;
                        if constexpr (fuse_bias) {
                            if (!last_inner_dim_block) {
                                UNPACK(RESTORE_PARTIALS_RD(partials_cb_read_ptr, cb_matmul_partials));
                                PACK(RESTORE_PARTIALS_WR(partials_cb_write_ptr, cb_matmul_partials));
                            }
                        } else {
                            if (!last_inner_dim_block) {
                                UNPACK(RESTORE_PARTIALS_RD(partials_cb_read_ptr, cb_matmul_partials));
                            }
                            if (in0_block_w_i < in0_num_blocks_w - 2) {
                                PACK(RESTORE_PARTIALS_WR(partials_cb_write_ptr, cb_matmul_partials));
                            }
                        }
                    }
                }

                cb_mm_in0.pop_front(in0_block_num_tiles);
                cb_in1.pop_front(in1_block_num_tiles);
            }  // for in0_num_blocks_w
            if constexpr (matmul_partials_cb == mm_out_cb_id && partials_cb_uses_output) {
                UNPACK(RESTORE_PARTIALS_RD(partials_cb_read_ptr, cb_matmul_partials));
            }
#ifdef CHECK_SKIP_COMPUTE
            if (skip_compute) {
                continue;
            }
#endif
#ifdef FUSE_BIAS
            if constexpr (fuse_bias) {
                // [#48552 DFB-ONLY] bias pack inits / add_tiles_bcast / pack_tile removed; keep CB flow.
                cb_bias.wait_front(bias_ntiles_w);
                cb_matmul_partials.wait_front(out_block_num_tiles);
                for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                    for (uint32_t in1_subblock_i = 0; in1_subblock_i < in1_num_subblocks; ++in1_subblock_i) {
                        cb_matmul_partials.pop_front(out_subblock_num_tiles);
                        cb_untilize_mode_out.reserve_back(out_subblock_num_tiles);
                        cb_untilize_mode_out.push_back(out_subblock_num_tiles);
                    }  // for in1_num_subblocks
                }  // in0_num_subblocks
                if constexpr (untilize_out) {
                    UNPACK(RESTORE_PARTIALS_RD(partials_cb_read_ptr, cb_matmul_partials));
                    PACK(RESTORE_PARTIALS_WR(partials_cb_write_ptr, cb_matmul_partials));
                }
            }
#endif  // FUSE_BIAS
            if constexpr (untilize_out) {
                // [#48552 DFB-ONLY] pack_reconfig / pack_relu / reconfig_data_format_srca removed.
                if constexpr (packer_untilize) {
                    // [#48552 DFB-ONLY] pack_untilize_dest_init / copy_tile_to_dst_init / pack_untilize_uninit removed.
                    for (uint32_t in0_subblock_i = 0; in0_subblock_i < in0_num_subblocks; ++in0_subblock_i) {
                        reblock_and_untilize<out_subblock_w, out_block_w>(
                            cb_matmul_partials, cb_out, in1_num_subblocks, out_subblock_num_tiles, out_subblock_h);
                    }
                } else {
                    // [#48552 DFB-ONLY] compute_kernel_lib::untilize compute removed; reproduce its single-pass
                    // WaitBlock CB flow: per block -> partials.wait_front(out_block_w); out.reserve_back(out_block_w);
                    // out.push_back(out_block_w); partials.pop_front(out_block_w).
                    const uint32_t untilize_num_blocks = in0_num_subblocks * out_subblock_h;
                    for (uint32_t r = 0; r < untilize_num_blocks; ++r) {
                        cb_matmul_partials.wait_front(out_block_w);
                        cb_out.reserve_back(out_block_w);
                        cb_out.push_back(out_block_w);
                        cb_matmul_partials.pop_front(out_block_w);
                    }
                }
            }
            // [#48552 DFB-ONLY] trailing reconfig_data_format removed.
        }  // for in0_num_blocks_h
    }  // for in1_num_blocks_w
}  // void kernel_main()
