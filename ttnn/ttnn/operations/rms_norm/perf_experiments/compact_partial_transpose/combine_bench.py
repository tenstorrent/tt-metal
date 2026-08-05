# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""ISOLATED bake-off: rms_norm's cross-core width COMBINE at two partial layouts.

Not the op.  A single-core, compute-only reconstruction of just the stage that the whole-op
profile showed to be the critical path on (1,1,8192,1024) BLOCK_SHARDED / 64 cores:

    compute_root_sum  26773 ns  +  compute_root_finalize 22501 ns  +  compute_stat_handoff 8177 ns
    = 57.5 us of an 84.8 us wall, ALL of it serial on the 8 group-root cores.

Every one of those costs is per TILE-ROW, because a core's partial sum(x^2) for one tile-row is a
REDUCE_ROW result: 32 useful fp32 numbers living in COLUMN 0 of a 1024-position tile.  With
GROUP_SIZE = 8 and rows = 10 the root pays 80 tile-copies + 10 SFPU finalizes + 10 tile-copies per
combine round to move 320 useful floats.

The idea under test: PACK the partial to its 32 useful floats -- tile-row r's column vector into
COLUMN r of one COMPACT tile -- so the root sums GROUP_SIZE compact tiles instead of
GROUP_SIZE * rows full ones, finalizes ONE tile instead of `rows`, and hands off ONE tile.  The
pack and the un-pack are ONE FPU matmul against a one-hot tile each (see the kernel head).

Variants (MODE):
  0 base_phase0   the op's Phase-0 root fold: copy slot 0 + (GROUP_SIZE-1) in-place adds, `rows`
                  tiles per call, then `rows` transform_in_place finalizes, then `rows` copies out.
  1 base_l1acc    the op's CURRENT root fold (descriptor D16): one CopyTile+PackTile chain per
                  tile-row over the row's GROUP_SIZE contiguous partials with L1Accumulation::
                  SeedFirst, then `rows` finalizes, then `rows` copies out.
  2 cand_root     compact: one chain over GROUP_SIZE compact tiles, ONE finalize, then the
                  un-pack (`rows` matmuls) ON THE ROOT before the handoff.
  3 cand_recv     compact: one chain over GROUP_SIZE compact tiles, ONE finalize, ONE copy out --
                  the un-pack is moved to the RECEIVERS (mode 6 measures what they pay).
  4 member_pack   what a member pays to produce a compact partial: `rows` matmuls into one DEST
                  tile + one pack.  Replaces...
  5 member_copy   ...the member's CURRENT partial handoff: `rows` tile copies.
  6 recv_unpack   what every receiver pays to un-pack the multicast compact stat back into `rows`
                  column-shaped tiles that pass B's mul<BroadcastDim::Col> can consume.

Precision contract (FIXED, identical for every variant): fp32 partials, math_fidelity = HiFi2,
fp32_dest_acc_en = False, math_approx_mode = False.  Nothing here tunes those.

Everything is resident in one core's L1 (no DRAM, no NoC, no gather) so the measured delta is the
combine's COMPUTE alone.  The resident input is re-exposed per call, so the physical page count is
bounded by max(GROUP_SIZE, rows) tiles even where the modelled logical stream is 1024 tiles (the
baseline's real cb_partials_gathered is GROUP_SIZE * rows pages -- the L1 cap that pins BLOCK_ROWS
at 10 -- which is itself part of what the compact layout buys back).

MEASURED (blackhole p150b, 1350 MHz, ns per combine round = (t_9 - t_1) / 8, DEST_BATCH = 4;
two independent profiled runs agreed to < 0.5%).  GROUP_SIZE = 8:

    rows        1      2      4     10     32     marginal ns per tile-row
    base_phase0      2359   3471   5678  12937  38975      1213
    base_l1acc       1542   2618   5045  12321  38969      1211
    cand_recv (root) 1541   1543   1536   1544   1914      ~0   (+373 at rows=32: RC rsqrt)
    cand_root (root) 1619   1668   1774   2060   3306        56
    member_copy       230    293    389    690   1837        52
    member_pack       240    262    310    475   1035        26
    recv_unpack       237    286    396    689   1548        43

  Per-stage zones at (G=8, rows=10), busiest TRISC: base_l1acc root_sum 5173 / root_finalize 6443 /
  stat_handoff 2253; cand_recv 636 / 676 / 752 -- i.e. the fold is 8.1x, the finalize 9.5x and the
  handoff 3.0x cheaper, and NONE of them scales with rows any more.
"""

import ttnn

TILE = 32
FP32_TILE_BYTES = TILE * TILE * 4

CB_PART = 0  # resident partials (fp32) -- the cb_partials_gathered / cb_row_stat analogue
CB_BANK = 1  # resident one-hot bank (fp32): [E_0..E_{rows-1}, F_0..F_{rows-1}, ZERO]
CB_ACC = 2  # local accumulator (fp32) -- the cb_row_stat analogue
CB_OUT = 3  # result (fp32) -- the cb_stat_handoff / cb_row_final analogue

MODES = {
    "base_phase0": 0,
    "base_l1acc": 1,
    "cand_root": 2,
    "cand_recv": 3,
    "member_pack": 4,
    "member_copy": 5,
    "recv_unpack": 6,
}

# DEST seeding for the matmul modes (matmul_tiles ACCUMULATES: DST += A*B).
SEED_NONE = 0  # rely on the packer's ZEROACC in llk_pack_dest_section_done
SEED_COPY = 1  # copy a zero tile into each DEST slot first (portable, +1 op per tile)

_KERNEL = r"""
// ISOLATED perf experiment for rms_norm's cross-core combine -- NOT the op's kernel.
//
// RAW-LLK / RAW-API JUSTIFICATION (matmul against a one-hot tile).
//   A core's partial sum(x^2) for one tile-row is a REDUCE_ROW result: 32 useful fp32 numbers in
//   COLUMN 0 of a tile.  Packing tile-row r's column into COLUMN r of one compact tile is a
//   HORIZONTAL data move, and the FPU's only horizontal-mixing primitive is the matmul:
//       pack   : C = partial_r x E_r ,  E_r[0][r] = 1  ->  C[i][r] = partial_r[i][0]
//       un-pack: C = compact  x F_r ,  F_r[r][0] = 1  ->  C[i][0] = compact[i][r]
//   Each is ONE matmul_tiles.  No kernel_lib helper expresses a column permutation (the eltwise /
//   bcast / reduce families all preserve or collapse the column axis), so these two lines are the
//   raw compute API by necessity, not by preference.
//   SAFETY NOTE that any later refactor must preserve: the matmul sums 32 products, so EVERY
//   column of the operand must be FINITE -- an inf/NaN in an unused column would become inf*0 =
//   NaN and poison the result.  The op's reduce leaves finite partial sums in columns 1..15 and
//   the writer's boot-zeroing leaves 0 in faces 1/3, so both operands are finite by construction.
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"

namespace ckl = compute_kernel_lib;

// Verbatim from the op (rms_norm_compute.cpp, Lamp L6b): rsqrt scoped to the tile's COLUMN faces.
#ifdef TRISC_MATH
template <bool legacy_compat = false, bool FAST_APPROX = false>
ALWI void rsqrt_tile_col(uint32_t idst) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_rsqrt,
        (APPROX, 8 /* ITERATIONS */, DST_ACCUM_MODE, FAST_APPROX, legacy_compat),
        idst,
        VectorMode::C);
}
#endif

namespace {
constexpr uint32_t cb_part = 0;
constexpr uint32_t cb_bank = 1;
constexpr uint32_t cb_acc = 2;
constexpr uint32_t cb_out = 3;
}  // namespace

void kernel_main() {
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(1);
    constexpr uint32_t ROWS = get_compile_time_arg_val(2);
    constexpr uint32_t INV_W_BITS = get_compile_time_arg_val(3);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(4);
    constexpr uint32_t RSQRT_RC = get_compile_time_arg_val(5);  // 0 = VectorMode::C, 1 = RC
    constexpr uint32_t SEED = get_compile_time_arg_val(6);  // 0 = none, 1 = zero-tile copy
    constexpr uint32_t DEST_BATCH = get_compile_time_arg_val(7);
    // Runtime (not compile-time) so the ITERS=1 and ITERS=N launches share one compiled kernel:
    // the amortisation pair (t_N - t_1) / (N - 1) removes the fixed launch floor exactly.
    const uint32_t ITERS = get_arg_val<uint32_t>(0);

    constexpr bool COMPACT = (MODE == 2 || MODE == 3);
    constexpr bool USES_MM = (MODE == 2 || MODE == 4 || MODE == 6);
    // Pages the resident input CB holds == the pages one call consumes (see the module docstring).
    constexpr uint32_t CHUNK = (MODE == 0 || MODE == 4 || MODE == 5) ? ROWS : (COMPACT || MODE == 1 ? GROUP_SIZE : 1);
    constexpr uint32_t CHUNKS = (MODE == 0) ? GROUP_SIZE : ((MODE == 1) ? ROWS : 1);
    constexpr uint32_t BANK_PAGES = 2 * ROWS + 1;
    constexpr uint32_t ZERO_IDX = 2 * ROWS;
    constexpr uint32_t NFIN = COMPACT ? 1 : ROWS;                                    // finalize calls
    constexpr uint32_t NOUT = (MODE == 3 || MODE == 4) ? 1 : ROWS;                    // output pages

    if constexpr (USES_MM) {
        compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(cb_part, cb_bank, cb_out);
    } else {
        compute_kernel_hw_startup(cb_part, cb_part, cb_out);
    }

    // The one-hot bank is resident for the whole kernel; expose it once.
    cb_reserve_back(cb_bank, BANK_PAGES);
    cb_push_back(cb_bank, BANK_PAGES);
    cb_wait_front(cb_bank, BANK_PAGES);

    // 1/rms = rsqrt(sum/W + eps) -- verbatim from the op's `finalize` lambda.  RSQRT_RC picks the
    // vector mode: the baseline's per-tile-row stat is column-shaped so only faces 0/2 are read
    // (VectorMode::C, the op's L6b); a COMPACT tile with more than 16 packed tile-rows has live
    // data in faces 1/3 too and needs the full RC.
    auto finalize = [](uint32_t dst) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, INV_W_BITS);
        add_unary_tile(dst, EPS_BITS);
        rsqrt_tile_init();
        if constexpr (RSQRT_RC == 0) {
            MATH((rsqrt_tile_col(dst)));
        } else {
            rsqrt_tile(dst);
        }
    };

    // The op's D16 fold target: one output tile pinned for the whole call, packer-accumulated.
    constexpr auto FOLD_OUT = ckl::output(
        cb_acc,
        ckl::ReservePolicy::OneUpfront,
        ckl::PushPolicy::OneAtEnd,
        ckl::DataFormatReconfig::Enabled,
        ckl::PackRelu::Disabled,
        ckl::L1Accumulation::SeedFirst);

    for (uint32_t iter = 0; iter < ITERS; ++iter) {
        if constexpr (MODE == 0) {
            // ---- the op's Phase-0 root fold: a seeding copy + (GROUP_SIZE-1) in-place adds ----
            MaybeDeviceZoneScope("root_sum");
            cb_reserve_back(cb_part, CHUNK);
            cb_push_back(cb_part, CHUNK);
            ckl::copy<ckl::input(cb_part), ckl::output(cb_acc)>(ckl::EltwiseShape::tiles(ROWS));
            for (uint32_t g = 1; g < GROUP_SIZE; ++g) {
                cb_reserve_back(cb_part, CHUNK);
                cb_push_back(cb_part, CHUNK);
                ckl::add<ckl::input(cb_acc), ckl::input(cb_part), ckl::output(cb_acc)>(
                    ckl::EltwiseShape::tiles(ROWS));
            }
        } else if constexpr (MODE == 1 || COMPACT) {
            // ---- the op's CURRENT (D16) fold: one pack-accumulating chain per output tile ----
            MaybeDeviceZoneScope("root_sum");
            for (uint32_t c = 0; c < CHUNKS; ++c) {
                cb_reserve_back(cb_part, CHUNK);
                cb_push_back(cb_part, CHUNK);
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(GROUP_SIZE),
                    ckl::CopyTile<ckl::input(cb_part)>{},
                    ckl::PackTile<FOLD_OUT>{});
            }
        } else if constexpr (MODE == 4) {
            // ---- member: PACK `rows` column-shaped partials into ONE compact tile ----
            // `rows` matmuls accumulate into a single DEST tile (matmul_tiles is DST += A*B), so
            // the whole compact tile costs one pack.  Column j >= rows stays exactly 0.
            MaybeDeviceZoneScope("member_pack");
            cb_reserve_back(cb_part, CHUNK);
            cb_push_back(cb_part, CHUNK);
            cb_wait_front(cb_part, ROWS);
            cb_reserve_back(cb_out, 1);
            tile_regs_acquire();
            if constexpr (SEED == 1) {
                reconfig_data_format_srca(cb_bank);
                copy_tile_to_dst_init_short(cb_bank);
                copy_tile(cb_bank, ZERO_IDX, 0);
            }
            matmul_init(cb_part, cb_bank, 0);
            for (uint32_t r = 0; r < ROWS; ++r) {
                matmul_tiles(cb_part, cb_bank, r, r, 0);
            }
            tile_regs_commit();
            pack_reconfig_data_format(cb_out);
            tile_regs_wait();
            pack_tile(0, cb_out, 0);
            tile_regs_release();
            cb_push_back(cb_out, 1);
            cb_pop_front(cb_part, ROWS);
        } else if constexpr (MODE == 5) {
            // ---- member: the CURRENT partial handoff, `rows` tile copies ----
            MaybeDeviceZoneScope("member_copy");
            cb_reserve_back(cb_part, CHUNK);
            cb_push_back(cb_part, CHUNK);
            ckl::copy<ckl::input(cb_part), ckl::output(cb_out)>(ckl::EltwiseShape::tiles(ROWS));
        } else {  // MODE == 6
            // ---- receiver: UN-PACK the compact stat into `rows` column-shaped tiles ----
            MaybeDeviceZoneScope("recv_unpack");
            cb_reserve_back(cb_part, 1);
            cb_push_back(cb_part, 1);
            cb_wait_front(cb_part, 1);
            cb_reserve_back(cb_out, ROWS);
            matmul_init(cb_part, cb_bank, 0);
            for (uint32_t r0 = 0; r0 < ROWS; r0 += DEST_BATCH) {
                const uint32_t n = (ROWS - r0 < DEST_BATCH) ? (ROWS - r0) : DEST_BATCH;
                tile_regs_acquire();
                for (uint32_t d = 0; d < n; ++d) {
                    if constexpr (SEED == 1) {
                        reconfig_data_format_srca(cb_bank);
                        copy_tile_to_dst_init_short(cb_bank);
                        copy_tile(cb_bank, ZERO_IDX, d);
                        matmul_init(cb_part, cb_bank, 0);
                    }
                    matmul_tiles(cb_part, cb_bank, 0, ROWS + r0 + d, d);
                }
                tile_regs_commit();
                pack_reconfig_data_format(cb_out);
                tile_regs_wait();
                for (uint32_t d = 0; d < n; ++d) {
                    pack_tile(d, cb_out, r0 + d);
                }
                tile_regs_release();
            }
            cb_push_back(cb_out, ROWS);
            cb_pop_front(cb_part, 1);
        }

        if constexpr (MODE <= 3) {
            {
                MaybeDeviceZoneScope("root_finalize");
                for (uint32_t i = 0; i < NFIN; ++i) {
                    ckl::transform_in_place(cb_acc, finalize);
                }
            }
            if constexpr (MODE == 2) {
                // ---- un-pack ON THE ROOT: `rows` matmuls, then the handoff is `rows` pages ----
                MaybeDeviceZoneScope("root_unpack");
                cb_wait_front(cb_acc, 1);
                cb_reserve_back(cb_out, ROWS);
                matmul_init(cb_acc, cb_bank, 0);
                for (uint32_t r0 = 0; r0 < ROWS; r0 += DEST_BATCH) {
                    const uint32_t n = (ROWS - r0 < DEST_BATCH) ? (ROWS - r0) : DEST_BATCH;
                    tile_regs_acquire();
                    for (uint32_t d = 0; d < n; ++d) {
                        if constexpr (SEED == 1) {
                            reconfig_data_format_srca(cb_bank);
                            copy_tile_to_dst_init_short(cb_bank);
                            copy_tile(cb_bank, ZERO_IDX, d);
                            matmul_init(cb_acc, cb_bank, 0);
                        }
                        matmul_tiles(cb_acc, cb_bank, 0, ROWS + r0 + d, d);
                    }
                    tile_regs_commit();
                    pack_reconfig_data_format(cb_out);
                    tile_regs_wait();
                    for (uint32_t d = 0; d < n; ++d) {
                        pack_tile(d, cb_out, r0 + d);
                    }
                    tile_regs_release();
                }
                cb_push_back(cb_out, ROWS);
                cb_pop_front(cb_acc, 1);
            } else {
                MaybeDeviceZoneScope("stat_handoff");
                ckl::copy<ckl::input(cb_acc), ckl::output(cb_out)>(ckl::EltwiseShape::tiles(NFIN));
            }
        }

        // Drain the result between in-kernel iterations so the next pass starts clean; the LAST
        // iteration's output stays in L1 for the host to read back.
        if (iter + 1 < ITERS) {
            cb_wait_front(cb_out, NOUT);
            cb_pop_front(cb_out, NOUT);
        }
    }
}
"""


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def single_core_shard(n_tiles):
    """The whole [n_tiles*32 x 32] tile column as ONE shard on core (0,0)."""
    return ttnn.create_sharded_memory_config(
        shape=(n_tiles * TILE, TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def geometry(variant, group_size, rows):
    """(resident input pages, bank pages, output pages) for a variant."""
    mode = MODES[variant]
    chunk = rows if mode in (0, 4, 5) else (group_size if mode in (1, 2, 3) else 1)
    out_pages = 1 if mode in (3, 4) else rows
    return chunk, 2 * rows + 1, out_pages


def create_program_descriptor(
    part, bank, out, *, variant, group_size, rows, inv_w_bits, eps_bits, iters, seed, dest_batch
):
    mode = MODES[variant]
    # A compact tile with more than 16 packed tile-rows has live data in faces 1/3, which
    # VectorMode::C does not walk -- those variants must run the full-tile rsqrt.
    rsqrt_rc = 1 if (mode in (2, 3) and rows > 16) else 0
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [iters]
    compute = ttnn.KernelDescriptor(
        kernel_source=_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[mode, group_size, rows, inv_w_bits, eps_bits, rsqrt_rc, seed, dest_batch],
        runtime_args=rt,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )
    _, _, out_pages = geometry(variant, group_size, rows)
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_PART, part),
        ttnn.cb_descriptor_from_sharded_tensor(CB_BANK, bank),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out),
        # cb_acc mirrors the op's cb_row_stat: CB_ROW_STAT_DEPTH (= 2) * pages (descriptor D6).
        ttnn.CBDescriptor(
            total_size=2 * max(out_pages, 1) * FP32_TILE_BYTES,
            core_ranges=_single_core(),
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_ACC, data_format=ttnn.float32, page_size=FP32_TILE_BYTES)
            ],
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def run(part, bank, out, **kw):
    return ttnn.generic_op([part, bank, out], create_program_descriptor(part, bank, out, **kw))
