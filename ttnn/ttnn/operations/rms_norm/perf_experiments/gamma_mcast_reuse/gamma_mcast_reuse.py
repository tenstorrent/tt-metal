# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# ISOLATED PERF BENCH — rms_norm's GAMMA STAGING STAGE only.
#
# The stage under test: N cores each need the SAME `wt_chunk` gamma tiles resident in
# `cb_gamma`.  rms_norm splits the independent ROW axis across cores and gamma is
# indexed by WIDTH only, so every core in a width group reads BYTE-IDENTICAL gamma.
# On the interleaved row split that means all N cores read the same few DRAM pages.
#
# Variants (all deliver the identical bytes to the identical L1 address on every core):
#   baseline        : the op's current `stage_gamma_chunk` read loop, verbatim, on every core
#   mcast_1inj_*    : ONE injector reads the chunk once and Mcast2D-broadcasts it to the rect
#   mcast_perrow_*  : one injector per grid ROW (Mcast1D PerRow) -> h concurrent injectors
#   mcast_percol_*  : one injector per grid COLUMN (Mcast1D PerColumn) -> w concurrent injectors
# each in a `_noc0` (the reader's NoC, NCRISC/NOC_0 — where the op's reader lives) and a
# `_noc1` (the writer's otherwise-idle NoC) flavour.
#
# ISOLATION (perf-lab concept-isolation table, DRAM-read / NoC-contention row):
#   * gamma lives in DRAM interleaved, exactly as the op sees it;
#   * the per-core job is trivial and IDENTICAL in every variant (a tile-sum checksum),
#     so the output is one tile per core and the kernel stays read/mcast-bound;
#   * the core geometry is a fixed rectangle shared by every variant, so placement is
#     not a hidden variable.
#
# PRECISION: this bench moves BYTES.  Every variant must deliver bit-identical L1
# content, which `mode="dump"` proves directly (the writer dumps each core's whole
# staged region).  The checksum compute is a *measurement fixture*, identical in every
# variant; it is not rms_norm's math and does not touch the op's precision contract.
#
# ======================================================================================
# MEASURED — blackhole p150b, 110-core grid, CHIP_FREQ 1350 MHz, 2026-08-05.
# `DEVICE KERNEL DURATION [ns]`, one fresh profiled launch per cell (rep0/rep1 agree
# within 1.5%).  Zones from MaybeDeviceZoneScope, max-ns-per-core.
# ======================================================================================
# (1) THE STAGE IN ISOLATION (no x), TILE gamma, whole-program ns:
#         cores  Wt=32 baseline   mcast_1inj_noc0
#             8          3289          3944  0.83x   <- LOSES below ~12 cores
#            12          4464          3860  1.16x
#           110         20747          5298  3.92x   (gamma zone 19269 -> 3272 = 5.9x)
#     Wt=224, 110 cores: 128438 -> 21170 = 6.07x.
#
# (2) THE STAGE IN THE OP'S CONTEXT (110 cores, Wt=32, + 3 tile-rows of DISTINCT x per
#     core = the primary target (1,1,8192,1024)).  This is the number that matters:
#         variant             reader wall   gamma zone   x zone
#         baseline                  70518        57355    19776     <- op measures 61099 / 24415
#         mcast_1inj_noc0           56390         3254    54006     1.24x on the reader
#         mcast_percol_noc0         56856         3299    53557     1.24x
#         mcast_perrow_noc0         59370         7328    52163     1.19x
#         mcast_1inj_noc1           90490         3265    87494     0.78x  (x moved to NOC_1)
#
# (3) WHY THE STAGE ZONE LIES.  Sweeping how much x is read AFTER the gamma stage:
#         x_rows   baseline gamma zone   baseline reader wall
#              0                 20809                  20853
#              1                 32353                  36149
#              3                 57078                  70982
#     The gamma zone STRETCHES with x read after it, because reader DRAM traffic is ONE
#     shared queue: early cores start flooding x while late cores' gamma is still in
#     flight, so the last core's gamma barrier absorbs the wait.  gamma's real marginal
#     cost is its DRAM BYTES, not its zone.  The reader is bandwidth-bound at ~405 GB/s
#     in both variants (28.8 MB / 70.5 us; 21.7 MB / 56.4 us), and
#         new reader wall = x_bytes / roofline + injector prefix
#                         = 53.3 us + 3.25 us = 56.5 us  (measured 56.4)
#     predicts the win exactly.  So the lever is worth gamma's SHARE OF READER DRAM
#     BYTES = num_sharing_cores / (Rt + num_sharing_cores), not its zone share.

import ttnn

TILE = 32

CB_GAMMA = 6  # mirrors the op's cb_gamma_tiles / cb_gamma_sticks slot
CB_ZERO = 2  # a single all-zero tile (built on device) — the +0 operand of the acc add
CB_OUT = 16  # per-core checksum tile
CB_X = 1  # contention probe: this core's OWN distinct x tiles (never consumed)

VARIANTS = (
    "baseline",
    "mcast_1inj_noc0",
    "mcast_1inj_noc1",
    "mcast_perrow_noc0",
    "mcast_perrow_noc1",
    "mcast_percol_noc0",
)

# --------------------------------------------------------------------------------------
# BASELINE reader — rms_norm_reader.cpp `stage_gamma_chunk`, reconstructed verbatim.
# --------------------------------------------------------------------------------------
_BASELINE_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace { constexpr uint32_t cb_gamma = 6; constexpr uint32_t cb_x = 1; constexpr uint32_t TILE_DIM = 32; }

void kernel_main() {
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(0);
    constexpr uint32_t WT = get_compile_time_arg_val(1);
    constexpr uint32_t GAMMA_IS_RM = get_compile_time_arg_val(2);
    constexpr uint32_t W_ELEMS = get_compile_time_arg_val(3);
    constexpr uint32_t GAMMA_ELEM_BYTES = get_compile_time_arg_val(4);
    constexpr uint32_t X_ROWS = get_compile_time_arg_val(5);   // contention probe (0 = off)
    constexpr auto g_args = TensorAccessorArgs<6>();
    constexpr auto x_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

    const uint32_t gamma_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t x_row_start = get_arg_val<uint32_t>(2);

    constexpr uint32_t G_TILE_COL_BYTES = TILE_DIM * GAMMA_ELEM_BYTES;
    constexpr uint32_t ROW_BYTES = W_ELEMS * GAMMA_ELEM_BYTES;

    // ROW_MAJOR gamma stages one stick into a tile-paged ring, so the ring's pad lanes are
    // never written by the read.  Zero them ONCE so the staged region is a DEFINED function
    // of the delivered bytes in every variant (the op does the same for its x staging ring).
    if constexpr (GAMMA_IS_RM != 0) {
        Noc noc;
        DataflowBuffer g_dfb(cb_gamma);
        noc.async_write_zeros(g_dfb, g_dfb.get_total_size_bytes());
        noc.write_zeros_l1_barrier();
    }

    {
        MaybeDeviceZoneScope("reader_read_gamma");
        const auto g_acc = TensorAccessor(g_args, gamma_addr);
        if constexpr (GAMMA_IS_RM != 0) {
            // The op calls dataflow_kernel_lib::read_sticks_for_tilize<cb>(g_acc, 1, row_bytes, 0, off),
            // which for total_num_rows == 1 is exactly: reserve width_in_tiles pages, ONE
            // noc_async_read of row_bytes into the write ptr, barrier, push.
            cb_reserve_back(cb_gamma, WT_CHUNK);
            noc_async_read(g_acc.get_noc_addr(0, 0), get_write_ptr(cb_gamma), ROW_BYTES);
            noc_async_read_barrier();
            cb_push_back(cb_gamma, WT_CHUNK);
        } else {
            const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
            cb_reserve_back(cb_gamma, WT_CHUNK);
            uint32_t l1_addr = get_write_ptr(cb_gamma);
            for (uint32_t w = 0; w < WT_CHUNK; ++w) {
                const uint32_t wt = w;
                noc_async_read_tile((wt < WT) ? wt : (WT - 1), g_acc, l1_addr);
                l1_addr += gamma_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma, WT_CHUNK);
        }
    }

    // Contention probe: this core's OWN distinct x tiles, read AFTER gamma exactly as the
    // op's reader orders the two stages.  Nothing consumes cb_x; it is sized to hold the lot.
    if constexpr (X_ROWS != 0) {
        MaybeDeviceZoneScope("reader_read_x");
        const auto x_acc = TensorAccessor(x_args, x_addr);
        const uint32_t x_tile_bytes = get_tile_size(cb_x);
        for (uint32_t r = 0; r < X_ROWS; ++r) {
            const uint32_t base = (x_row_start + r) * WT_CHUNK;
            cb_reserve_back(cb_x, WT_CHUNK);
            uint32_t l1_addr = get_write_ptr(cb_x);
            for (uint32_t w = 0; w < WT_CHUNK; ++w) {
                noc_async_read_tile(base + w, x_acc, l1_addr);
                l1_addr += x_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_x, WT_CHUNK);
        }
    }
}
"""

# --------------------------------------------------------------------------------------
# CANDIDATE — injector reads the chunk ONCE, mcast_pipe-broadcasts it to its group.
#
# One source serves every injector topology and both NoCs: the host decides who is a
# sender (an `is_sender` runtime word) and what the receiver rect / sender coords are
# (the McastArgs wire that ttnn.Mcast1D / ttnn.Mcast2D emit).  The injector broadcasts
# IN PLACE from the same cb_gamma slot it read into (src == dst), which is the op's own
# hard-won convention: an in-place send takes the EXCLUDE-source path under BOTH host
# emitters (Mcast1D's per-row rect excludes the sender, Mcast2D's contains it), so the
# injector is never served twice and never skipped.
# --------------------------------------------------------------------------------------
_MCAST_KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

using namespace dataflow_kernel_lib;

namespace { constexpr uint32_t cb_gamma = 6; constexpr uint32_t cb_x = 1; constexpr uint32_t TILE_DIM = 32; }

void kernel_main() {
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(0);
    constexpr uint32_t WT = get_compile_time_arg_val(1);
    constexpr uint32_t GAMMA_IS_RM = get_compile_time_arg_val(2);
    constexpr uint32_t W_ELEMS = get_compile_time_arg_val(3);
    constexpr uint32_t GAMMA_ELEM_BYTES = get_compile_time_arg_val(4);
    constexpr uint32_t X_ROWS = get_compile_time_arg_val(5);
    constexpr auto mc = McastArgs</*CT=*/6, /*RT=*/3>();
    constexpr uint32_t AFTER_MC = mc.next_compile_time_args_offset();
    constexpr auto g_args = TensorAccessorArgs<AFTER_MC>();
    constexpr auto x_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

    const uint32_t gamma_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t x_row_start = get_arg_val<uint32_t>(2);
    // RT 3..6 = the McastArgs block (dest rect for a sender, sender coords for a receiver).
    const uint32_t is_sender = get_arg_val<uint32_t>(7);

    constexpr uint32_t ROW_BYTES = W_ELEMS * GAMMA_ELEM_BYTES;

    Noc noc;

    if constexpr (GAMMA_IS_RM != 0) {
        DataflowBuffer g_dfb(cb_gamma);
        noc.async_write_zeros(g_dfb, g_dfb.get_total_size_bytes());
        noc.write_zeros_l1_barrier();
    }

    {
        MaybeDeviceZoneScope("reader_read_gamma");
        if (is_sender != 0) {
            const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
            const auto g_acc = TensorAccessor(g_args, gamma_addr);
            cb_reserve_back(cb_gamma, WT_CHUNK);
            const uint32_t dst = get_write_ptr(cb_gamma);
            uint32_t bcast_bytes;
            if constexpr (GAMMA_IS_RM != 0) {
                noc_async_read(g_acc.get_noc_addr(0, 0), dst, ROW_BYTES);
                bcast_bytes = ROW_BYTES;
            } else {
                uint32_t l1_addr = dst;
                for (uint32_t w = 0; w < WT_CHUNK; ++w) {
                    const uint32_t wt = w;
                    noc_async_read_tile((wt < WT) ? wt : (WT - 1), g_acc, l1_addr);
                    l1_addr += gamma_tile_bytes;
                }
                bcast_bytes = WT_CHUNK * gamma_tile_bytes;
            }
            noc_async_read_barrier();
            if constexpr (mc.active) {
                auto pipe = mc.sender(noc);
                pipe.send(dst, dst, bcast_bytes);  // src == dst -> EXCLUDE-source in both emitters
            }
            cb_push_back(cb_gamma, WT_CHUNK);
        } else {
            cb_reserve_back(cb_gamma, WT_CHUNK);  // reserve FIRST: receive()'s ack means "slot free"
            auto pipe = mc.receiver(noc);
            pipe.receive();
            cb_push_back(cb_gamma, WT_CHUNK);
        }
    }

    if constexpr (X_ROWS != 0) {
        MaybeDeviceZoneScope("reader_read_x");
        const auto x_acc = TensorAccessor(x_args, x_addr);
        const uint32_t x_tile_bytes = get_tile_size(cb_x);
        for (uint32_t r = 0; r < X_ROWS; ++r) {
            const uint32_t base = (x_row_start + r) * WT_CHUNK;
            cb_reserve_back(cb_x, WT_CHUNK);
            uint32_t l1_addr = get_write_ptr(cb_x);
            for (uint32_t w = 0; w < WT_CHUNK; ++w) {
                noc_async_read_tile(base + w, x_acc, l1_addr);
                l1_addr += x_tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_x, WT_CHUNK);
        }
    }
}
"""

# --------------------------------------------------------------------------------------
# Checksum compute (measurement fixture; IDENTICAL in every variant).
# Folds the whole staged region into one tile in the fp32 DEST adder — a cheap stream of
# FPU adds, negligible next to the read, so it cannot mask the read strategy.  It reads
# every byte of every staged page, so a corrupted delivery shows up in the checksum.
# --------------------------------------------------------------------------------------
_COMPUTE_KERNEL = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma = 6, cb_zero = 2, cb_out = 16;

    compute_kernel_hw_startup(cb_gamma, cb_out);

    cb_wait_front(cb_gamma, WT_CHUNK);
    sub_tiles_init(cb_gamma, cb_gamma);
    tile_regs_acquire();
    sub_tiles(cb_gamma, cb_gamma, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_zero, 1);
    pack_tile(0, cb_zero);
    cb_push_back(cb_zero, 1);
    tile_regs_release();
    cb_wait_front(cb_zero, 1);

    add_tiles_init(cb_gamma, cb_zero, /*acc_to_dest=*/true);
    tile_regs_acquire();
    for (uint32_t t = 0; t < WT_CHUNK; ++t) {
        add_tiles(cb_gamma, cb_zero, t, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
    cb_pop_front(cb_gamma, WT_CHUNK);
    cb_pop_front(cb_zero, 1);
}
"""

# Writer, perf mode: this core's single checksum tile.
_WRITER_SUM_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t my_index = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = 16;
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out_acc = TensorAccessor(out_args, out_addr, get_tile_size(cb_out));
    cb_wait_front(cb_out, 1);
    noc_async_write_tile(my_index, out_acc, get_read_ptr(cb_out));
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
"""

# Writer, dump mode (the delivery gate): this core's WHOLE staged gamma region, byte for byte.
_WRITER_DUMP_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t my_index = get_arg_val<uint32_t>(1);
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(0);
    constexpr uint32_t GAMMA_IS_RM = get_compile_time_arg_val(1);
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();
    constexpr uint32_t cb_gamma = 6;

    cb_wait_front(cb_gamma, WT_CHUNK);
    const uint32_t src = get_read_ptr(cb_gamma);
    if constexpr (GAMMA_IS_RM != 0) {
        // out is ROW_MAJOR [n_cores, W]: one stick per core, straight out of the staged ring.
        const auto out_acc = TensorAccessor(out_args, out_addr, ROW_BYTES);
        noc_async_write(src, out_acc.get_noc_addr(my_index, 0), ROW_BYTES);
    } else {
        const uint32_t tile_bytes = get_tile_size(cb_gamma);
        const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);
        for (uint32_t w = 0; w < WT_CHUNK; ++w) {
            noc_async_write_tile(my_index * WT_CHUNK + w, out_acc, src + w * tile_bytes);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_gamma, WT_CHUNK);
}
"""


# ======================================================================================
# host side
# ======================================================================================
def rect_for(device, n_cores):
    """A rectangle anchored at (0,0) holding exactly `n_cores` cores.

    Anchored + rectangular because that is what BOTH host emitters need (Mcast1D
    TT_FATALs on a grid not anchored at (0,0); Mcast2D takes the bounding box as THE
    rect).  The SAME rectangle is used by every variant at a given core count, so
    placement is never a hidden variable.
    """
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    if n_cores >= gx * gy:
        return gx, gy
    for w in range(min(gx, n_cores), 0, -1):
        if n_cores % w == 0 and n_cores // w <= gy:
            return w, n_cores // w
    raise ValueError(f"gamma_mcast_reuse: no rectangle for {n_cores} cores on a {gx}x{gy} grid")


def _crs_rect(w, h):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(w - 1, h - 1))])


def _cores_rowmajor(w, h):
    return [(x, y) for y in range(h) for x in range(w)]


# Two RISC/NoC placements for the gamma stage, matching the op's two real homes.
#
#   noc0: the gamma stage on NCRISC/NOC_0 — where rms_norm's READER lives (and where its
#         x reads ride).  The output write then takes BRISC/NOC_1 (the op's writer).
#   noc1: the gamma stage on BRISC/NOC_1 — the op's WRITER RISC, whose NoC is idle during
#         gamma staging.  The output write moves to NCRISC/NOC_0.
#
# The ROLES are swapped rather than both RISCs put on NOC_1: under DM_DEDICATED_NOC a NoC
# has ONE owning RISC per core, and running NCRISC and BRISC both on NOC_1 hangs the core
# (MEASURED: BRISC stuck in noc_async_write_barrier, NCRISC already done — triage
# 2026-08-05).  So "which NoC does the broadcast ride" is necessarily also "which RISC
# issues it", exactly as it would be in the op.
_NCRISC_NOC0 = dict(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0)
_BRISC_NOC1 = dict(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1)


def _stage_config(noc_id):
    return ttnn.DataMovementConfigDescriptor(**(_NCRISC_NOC0 if noc_id == 0 else _BRISC_NOC1))


def _out_config(noc_id):
    return ttnn.DataMovementConfigDescriptor(**(_BRISC_NOC1 if noc_id == 0 else _NCRISC_NOC0))


def create_program_descriptor(gamma, out, *, variant, n_cores, mode="sum", x=None, x_rows=0):
    """`mode="sum"` = perf (tiny per-core checksum output); `mode="dump"` = the delivery gate
    (every core's whole staged region written out)."""
    if variant not in VARIANTS:
        raise ValueError(f"gamma_mcast_reuse: variant must be one of {VARIANTS}, got {variant!r}")
    device = gamma.device()
    rw, rh = rect_for(device, n_cores)
    cores = _cores_rowmajor(rw, rh)
    assert len(cores) == n_cores
    all_crs = _crs_rect(rw, rh)

    is_rm = gamma.layout == ttnn.ROW_MAJOR_LAYOUT
    W = int(gamma.shape[-1])
    wt_chunk = W // TILE
    if gamma.dtype == ttnn.bfloat16:
        g_elem_bytes = 2
    elif gamma.dtype == ttnn.float32:
        g_elem_bytes = 4
    else:
        raise ValueError(f"gamma_mcast_reuse: unsupported gamma dtype {gamma.dtype}")
    row_bytes = W * g_elem_bytes
    # cb_gamma pages are TILE-sized in both layouts, exactly as the op's cb_gamma_tiles /
    # cb_gamma_sticks are (the RM ring is tile-paged so the compute tilize can consume it).
    gamma_tile_bytes = TILE * TILE * g_elem_bytes

    cbs = []
    cbs.append(
        ttnn.CBDescriptor(
            total_size=wt_chunk * gamma_tile_bytes,  # single-buffered — the op's resident staging
            core_ranges=all_crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_GAMMA, data_format=gamma.dtype, page_size=gamma_tile_bytes)
            ],
        )
    )
    if x_rows:
        x_tile_bytes = x.buffer_aligned_page_size()
        cbs.append(
            ttnn.CBDescriptor(
                total_size=x_rows * wt_chunk * x_tile_bytes,  # holds the lot; nothing consumes it
                core_ranges=all_crs,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB_X, data_format=x.dtype, page_size=x_tile_bytes)
                ],
            )
        )

    g_ct = ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
    x_ct = ttnn.TensorAccessorArgs(x).get_compile_time_args() if x is not None else list(g_ct)
    out_ct = ttnn.TensorAccessorArgs(out).get_compile_time_args()
    g_addr = gamma.buffer_address()
    x_addr = x.buffer_address() if x is not None else 0
    out_addr = out.buffer_address()

    scalar_ct = [wt_chunk, wt_chunk, 1 if is_rm else 0, W, g_elem_bytes, x_rows]

    kernels = []
    semaphores = []

    # ---- the stage under test -------------------------------------------------------
    noc_id = 1 if variant.endswith("_noc1") else 0
    if variant == "baseline":
        rt = ttnn.RuntimeArgs()
        for i, (cx, cy) in enumerate(cores):
            rt[cx][cy] = [g_addr, x_addr, i * x_rows]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=_BASELINE_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=all_crs,
                compile_time_args=[*scalar_ct, *g_ct, *x_ct],
                runtime_args=rt,
                config=_stage_config(0),
            )
        )
    else:
        mc_cfg = ttnn.McastConfig(
            noc=(ttnn.NOC.NOC_0 if noc_id == 0 else ttnn.NOC.NOC_1), handshake=True, base_sem_id=0
        )
        if variant.startswith("mcast_1inj"):
            mc = ttnn.Mcast2D(device, all_crs, ttnn.CoreCoord(0, 0), mc_cfg)
        elif variant.startswith("mcast_perrow"):
            mc = ttnn.Mcast1D(device, all_crs, ttnn.Mcast1DShape.PerRow, 0, mc_cfg)
        else:
            mc = ttnn.Mcast1D(device, all_crs, ttnn.Mcast1DShape.PerColumn, 0, mc_cfg)
        semaphores = list(mc.owned_semaphores())
        rt = ttnn.RuntimeArgs()
        for i, (cx, cy) in enumerate(cores):
            core = ttnn.CoreCoord(cx, cy)
            rt[cx][cy] = [
                g_addr,
                x_addr,
                i * x_rows,
                *mc.runtime_args(core),
                1 if mc.is_sender(core) else 0,
            ]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=_MCAST_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=all_crs,
                compile_time_args=[*scalar_ct, *mc.compile_time_args(), *g_ct, *x_ct],
                runtime_args=rt,
                config=_stage_config(noc_id),
            )
        )

    # ---- consumer + output ----------------------------------------------------------
    if mode == "sum":
        cbs.append(
            ttnn.CBDescriptor(
                total_size=gamma_tile_bytes,
                core_ranges=all_crs,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB_ZERO, data_format=gamma.dtype, page_size=gamma_tile_bytes)
                ],
            )
        )
        out_bytes = out.buffer_aligned_page_size()
        cbs.append(
            ttnn.CBDescriptor(
                total_size=out_bytes,
                core_ranges=all_crs,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB_OUT, data_format=out.dtype, page_size=out_bytes)
                ],
            )
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=_COMPUTE_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=all_crs,
                compile_time_args=[wt_chunk],
                config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True, math_fidelity=ttnn.MathFidelity.HiFi2),
            )
        )
        wrt = ttnn.RuntimeArgs()
        for i, (cx, cy) in enumerate(cores):
            wrt[cx][cy] = [out_addr, i]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=_WRITER_SUM_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=all_crs,
                compile_time_args=[*out_ct],
                runtime_args=wrt,
                config=_out_config(noc_id),
            )
        )
    else:
        wrt = ttnn.RuntimeArgs()
        for i, (cx, cy) in enumerate(cores):
            wrt[cx][cy] = [out_addr, i]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=_WRITER_DUMP_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=all_crs,
                compile_time_args=[wt_chunk, 1 if is_rm else 0, row_bytes, *out_ct],
                runtime_args=wrt,
                config=_out_config(noc_id),
            )
        )

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)


def gamma_stage(gamma, out, *, variant, n_cores, mode="sum", x=None, x_rows=0):
    desc = create_program_descriptor(gamma, out, variant=variant, n_cores=n_cores, mode=mode, x=x, x_rows=x_rows)
    tensors = [gamma, out] if x is None else [gamma, x, out]
    return ttnn.generic_op(tensors, desc)
