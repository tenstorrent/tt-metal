# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# ISOLATED BAKE-OFF: `bank_coalesced_txn` -- collapse rms_norm_ttnn's per-tile DRAM
# transactions into BANK-CONTIGUOUS multi-page transactions on the read AND the write
# side, accepting a permuted (bank-major) tile order in L1.
#
# THE MECHANISM.  An interleaved DRAM tensor maps page `i` to bank `i % NB` at bank
# offset `(i / NB) * aligned_page_size` (TensorAccessor::get_bank_and_offset ->
# round_robin_mapping, then get_noc_addr adds bank_page_offset * aligned_page_size).
# So the pages `p, p+NB, p+2NB, ...` are CONSECUTIVE BYTES inside one DRAM bank and one
# `noc_async_read` of `cnt * page_bytes` fetches them all, landing them contiguously in
# L1.  A run of WT consecutive tile ids therefore costs NB transactions instead of WT.
#
# WHAT IS ISOLATED (per /perf-lab's concept-isolation table).  Everything that is NOT the
# transaction shape is held identical across variants:
#   * NO compute kernel at all -- the bench is a pure DRAM -> L1 -> DRAM move with the
#     op's exact staging structure (reader on NCRISC/NOC_0 reserving WT_CHUNK pages per
#     tile-row, one barrier per tile-row, push; writer on BRISC/NOC_1 waiting WT_CHUNK,
#     one barrier per tile-row, pop).  TXN_ROWS == 1, as the op ships (DM_TXN_ROWS_MAX=1).
#   * the same 110-core grid, the same per-core row split, the same CB depth,
#     the same dtype (bf16) / layout (TILE) / memory config (DRAM interleaved).
#   * the precision contract is irrelevant here (no math) and is never a variant knob.
# The ONLY thing a variant changes is HOW the WT_CHUNK pages of a tile-row are moved.
#
# CORRECTNESS is the only pass/fail: the coalesced reader's permutation and the coalesced
# writer's inverse permutation compose to the identity, so out == in BITWISE for every
# payload-carrying variant.  A variant with an ablated payload is perf-only (it computes
# the wrong answer by construction and is never a candidate).

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import ttnn

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
CB_IN = 0
TILE = 32

# MODE 0 = one transaction per tile (THE OP'S CURRENT APPROACH -- the honest baseline)
# MODE 1 = NB bank-contiguous multi-page transactions (THE IDEA)
#
# ABLATE: 1 = keep every CB handshake / barrier / trip count, delete the NoC payload.
#         Used only to split the read half from the write half; never a candidate.
#
# PPT = pages per transaction inside a bank run (0 = the whole run, the maximal coalesce).
#       PPT == 1 is the ORDER-ONLY CONTROL: the same page count as the baseline, issued
#       bank-major -- it separates "transaction size" from "issue order".
# ROT = 1 rotates each core's starting bank by (core index % NB), so the 110 cores do not
#       all issue their bank-0 transaction at t=0.
#
# variant -> (mode, ablate_read, ablate_write, ppt, rot)
VARIANTS = {
    "base_rw": (0, 0, 0, 0, 0),  # BASELINE (op's current approach), both halves
    "base_r": (0, 0, 1, 0, 0),  # baseline read half only
    "base_w": (0, 1, 0, 0, 0),  # baseline write half only
    "floor": (0, 1, 1, 0, 0),  # scaffolding floor: handshakes + barriers, zero payload
}
# BANK-STAGGER family (MODE 2): baseline transactions, baseline L1 layout, rotated
# ISSUE ORDER only.  This is the control that separates "fewer/bigger transactions"
# from "spread the 110 cores' first request across the NB banks".
for _tag, _m in (("stag", 2),):
    VARIANTS[_tag] = (_m, 0, 0, 0, 1)
    VARIANTS[_tag + "_r"] = (_m, 0, 1, 0, 1)
    VARIANTS[_tag + "_w"] = (_m, 1, 0, 0, 1)

# candidate family: coal{PPT}[_rot]{,_r,_w}
for _ppt in (0, 1, 2, 3, 4, 6, 8):
    for _rot in (0, 1):
        _tag = f"coal{_ppt}" + ("_rot" if _rot else "")
        VARIANTS[_tag] = (1, 0, 0, _ppt, _rot)
        VARIANTS[_tag + "_r"] = (1, 0, 1, _ppt, _rot)
        VARIANTS[_tag + "_w"] = (1, 1, 0, _ppt, _rot)

# The only variants whose reader/writer permutations compose to the identity, i.e. the
# only ones whose output can be correctness-gated.  Payload-ablated variants are perf-only.
EXACT = {v for v in VARIANTS if not (VARIANTS[v][1] or VARIANTS[v][2])}

_DM_KERNEL = r"""
// Isolated `bank_coalesced_txn` bench for rms_norm_ttnn.
//
// MEASURED RESULT (Blackhole p150b, 11x10 = 110-core grid, AICLK 1350 MHz, NB = 8 DRAM
// banks, bf16 TILE, DRAM interleaved, 3-rep medians, spread < 0.4%):
//   FOCUS (1,1,8192,2304) WT_CHUNK=72, 110 cores, read+write:
//     MODE 0  base            72 txn/tile-row   173,344 ns   (the op's current approach)
//     MODE 1  PPT=1 (perm)    72 txn/tile-row   172,050 ns   1.008x  FLAT
//     MODE 1  PPT=2           40 txn/tile-row   180,835 ns   0.959x  REGRESSION
//     MODE 1  PPT=9 (max)     16 txn/tile-row   199,899 ns   0.867x  REGRESSION
//     MODE 2  bank stagger    72 txn/tile-row   171,482 ns   1.011x
//   Halves (FOCUS): read 91,755 -> 91,826 (PPT=2, FLAT) -> 99,495 (PPT=9);
//                   write 122,750 -> 123,172 (PPT=2, FLAT) -> 141,641 (PPT=9).
// So HALVING the transaction count buys EXACTLY NOTHING on either half in isolation --
// per-transaction issue cost is fully hidden behind DRAM bandwidth -- and the 6% loss at
// PPT=2 appears ONLY when read and write run together: a multi-page burst holds one DRAM
// bank for longer and the opposite-direction traffic to that bank can no longer interleave
// with it.  The op's fine-grained page-per-transaction round-robin IS the thing that keeps
// the two directions overlapped.  The idea only wins where DRAM is NOT the constraint:
// (1,1,32,1024), one active core, 2,821 -> 2,350 ns (1.200x at PPT=2).
//
// RAW-API JUSTIFICATION (/perf-lab suspends the prefer-helpers rule inside a bench):
// MODE 1 bypasses `noc_async_read_tile` / `noc_async_write_tile` (the page-shaped
// dataflow helpers the shipped reader/writer use) for the raw `noc_async_read` /
// `noc_async_write` with an explicit byte extent, because the WHOLE MEASUREMENT is the
// byte extent and transaction count of the transfer -- a page-shaped helper fixes both
// at one page by definition.  The NoC address of the run's first page still comes from
// the TensorAccessor (`get_noc_addr`), so the bank mapping is the accessor's, not a
// re-derivation.  MODE 0 uses the helpers verbatim: it IS the shipped code path.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/tensor_accessor_args.h"

void kernel_main() {
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(0);
    constexpr uint32_t NUM_W_CHUNKS = get_compile_time_arg_val(1);
    constexpr uint32_t WT = get_compile_time_arg_val(2);
    constexpr uint32_t TILE_BYTES = get_compile_time_arg_val(3);
    constexpr uint32_t NB = get_compile_time_arg_val(4);
    constexpr uint32_t MODE = get_compile_time_arg_val(5);
    constexpr uint32_t ABLATE = get_compile_time_arg_val(6);
    constexpr uint32_t IS_WRITER = get_compile_time_arg_val(7);
    constexpr uint32_t PPT_CT = get_compile_time_arg_val(8);   // pages per transaction (0 = whole run)
    constexpr uint32_t USE_ROT = get_compile_time_arg_val(9);  // 1 = rotate the start bank per core
    constexpr uint32_t base_cta = 10;

    constexpr auto args = TensorAccessorArgs<base_cta>();
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t row0 = get_arg_val<uint32_t>(1);
    const uint32_t nrows = get_arg_val<uint32_t>(2);
    const uint32_t rot = (USE_ROT != 0) ? get_arg_val<uint32_t>(3) : 0;
    (void)rot;
    const auto acc = TensorAccessor(args, addr);
    // The multi-page run relies on page p and p+NB being ALIGNED_PAGE_SIZE apart inside
    // one bank, and on the L1 landing stride (the CB page) matching it.  For a bf16 tile
    // page (2048 B) on a 64 B-aligned DRAM both are 2048; assert rather than assume.
    ASSERT(acc.get_aligned_page_size() == TILE_BYTES);

    // Bank-major run lengths for a WT_CHUNK-long run of consecutive tile ids.  The run
    // starts at an arbitrary tile_base, so residue class j of the run (tile_base + j) has
    // `Q + (j < REM)` pages -- NEVER assume equal lengths, a truncated or over-long read
    // is a silent correctness bug.
    constexpr uint32_t Q = WT_CHUNK / NB;
    constexpr uint32_t REM = WT_CHUNK % NB;
    constexpr uint32_t RUNS = (WT_CHUNK < NB) ? WT_CHUNK : NB;
    constexpr uint32_t QMAX = Q + ((REM > 0) ? 1u : 0u);
    constexpr uint32_t PPT = (PPT_CT == 0 || PPT_CT > QMAX) ? QMAX : PPT_CT;
    constexpr uint32_t KBLOCKS = (QMAX + PPT - 1) / PPT;

    for (uint32_t r = 0; r < nrows; ++r) {
        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            const uint32_t tile_base = (row0 + r) * WT + c * WT_CHUNK;
            uint32_t l1;
            if constexpr (IS_WRITER) {
                cb_wait_front(CB_IN_IDX, WT_CHUNK);
                l1 = get_read_ptr(CB_IN_IDX);
            } else {
                cb_reserve_back(CB_IN_IDX, WT_CHUNK);
                l1 = get_write_ptr(CB_IN_IDX);
            }
            if constexpr (MODE == 2) {
                // BANK-STAGGER CONTROL.  Same per-tile transactions as the baseline and
                // the SAME (unpermuted) L1 slot for every tile -- the ONLY change is the
                // ORDER in which the WT_CHUNK pages are issued: core c starts at width
                // tile (c % NB) and wraps.  Because tile w lands in bank (tile_base+w)%NB,
                // this makes the 110 cores' first request hit NB different banks instead
                // of all hitting the same one.  Nothing downstream sees any difference,
                // so it is legal on EVERY path (partial W, ragged chunk, RM, sharded).
                for (uint32_t i = 0; i < WT_CHUNK; ++i) {
                    uint32_t w = i + rot;
                    if (w >= WT_CHUNK) {
                        w -= WT_CHUNK;
                    }
                    if constexpr (ABLATE == 0) {
                        if constexpr (IS_WRITER) {
                            noc_async_write_tile(tile_base + w, acc, l1 + w * TILE_BYTES);
                        } else {
                            noc_async_read_tile(tile_base + w, acc, l1 + w * TILE_BYTES);
                        }
                    }
                }
            } else if constexpr (MODE == 0) {
                for (uint32_t w = 0; w < WT_CHUNK; ++w) {
                    if constexpr (ABLATE == 0) {
                        if constexpr (IS_WRITER) {
                            noc_async_write_tile(tile_base + w, acc, l1);
                        } else {
                            noc_async_read_tile(tile_base + w, acc, l1);
                        }
                    }
                    l1 += TILE_BYTES;
                }
            } else {
                // Bank-major issue.  Outer loop = sub-block index within a bank run, inner
                // loop = bank -- so with PPT < Q the issue stream still visits all NB banks
                // before it comes back to any one of them (max bank-level parallelism at
                // every instant).  `rot` rotates each core's starting bank so 110 cores do
                // not all hammer bank 0 at t=0 (the convoy this bench measured at rot=0).
                for (uint32_t kb = 0; kb < KBLOCKS; ++kb) {
                    const uint32_t k0 = kb * PPT;
                    for (uint32_t jj = 0; jj < RUNS; ++jj) {
                        const uint32_t j = (jj + rot) % RUNS;
                        const uint32_t cnt = Q + ((j < REM) ? 1u : 0u);
                        if (k0 >= cnt) {
                            continue;
                        }
                        const uint32_t n = ((cnt - k0) < PPT) ? (cnt - k0) : PPT;
                        // prefix[j] = j*Q + min(j, REM): run j's first L1 slot.
                        const uint32_t prefix = j * Q + ((j < REM) ? j : REM);
                        const uint32_t off = l1 + (prefix + k0) * TILE_BYTES;
                        if constexpr (ABLATE == 0) {
                            const uint64_t noc = acc.get_noc_addr(tile_base + j + k0 * NB);
                            if constexpr (IS_WRITER) {
                                noc_async_write(off, noc, n * TILE_BYTES);
                            } else {
                                noc_async_read(noc, off, n * TILE_BYTES);
                            }
                        }
                    }
                }
            }
            if constexpr (IS_WRITER) {
                noc_async_write_barrier();
                cb_pop_front(CB_IN_IDX, WT_CHUNK);
            } else {
                noc_async_read_barrier();
                cb_push_back(CB_IN_IDX, WT_CHUNK);
            }
        }
    }
}
"""


def _core_list(grid):
    w, h = grid
    return [(x, y) for y in range(h) for x in range(w)]


def _core_ranges(grid):
    w, h = grid
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(w - 1, h - 1))])


def _split(total, n):
    """Same shape as split_work_to_cores: the first `rem` cores get one extra row."""
    base, rem = divmod(total, n)
    out, acc = [], 0
    for i in range(n):
        k = base + (1 if i < rem else 0)
        out.append((acc, k))
        acc += k
    return out


def build(device, shape, wt_chunk, variant, grid, nb, depth=2):
    import torch

    mode, abl_r, abl_w, ppt, rot = VARIANTS[variant]
    H, W = shape[-2], shape[-1]
    WT = W // TILE
    assert W % TILE == 0 and H % TILE == 0, "bench models the tile-aligned path only"
    assert WT % wt_chunk == 0
    num_chunks = WT // wt_chunk
    rows = H // TILE

    torch.manual_seed(0)
    tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    x = ttnn.from_torch(
        tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    cores = _core_list(grid)
    ranges = _core_ranges(grid)
    tile_bytes = ttnn.tile_size(ttnn.bfloat16)

    rd_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, ((cx, cy), (r0, k)) in enumerate(zip(cores, _split(rows, len(cores)))):
        rd_rt[cx][cy] = [x.buffer_address(), r0, k, i % nb]
        wr_rt[cx][cy] = [out.buffer_address(), r0, k, i % nb]

    cbs = [
        ttnn.CBDescriptor(
            total_size=wt_chunk * depth * tile_bytes,
            core_ranges=ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_IN, data_format=ttnn.bfloat16, page_size=tile_bytes)
            ],
        )
    ]

    def _cta(mode, ablate, is_writer, tensor):
        return [
            wt_chunk,
            num_chunks,
            WT,
            tile_bytes,
            nb,
            mode,
            ablate,
            is_writer,
            ppt,
            rot,
        ] + list(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())

    src = _DM_KERNEL.replace("CB_IN_IDX", str(CB_IN))
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=src,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=_cta(mode, abl_r, 0, x),
            runtime_args=rd_rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=src,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=_cta(mode, abl_w, 1, out),
            runtime_args=wr_rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
        ),
    ]

    descriptor = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    def run():
        return ttnn.generic_op([x, out], descriptor)

    return run, tx, [x, out]


def read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def measure(device, shape, wt_chunk, variant, grid, nb):
    """ONE fresh-cache run per variant -- device kernel time has no warm-up transient."""
    import torch

    run, tx, live = build(device, shape, wt_chunk, variant, grid, nb)
    o = run()
    ttnn.synchronize_device(device)
    read_kernel_ns(device)  # drain
    ok = None
    if variant in EXACT:
        got = ttnn.to_torch(o)
        ok = bool(torch.equal(got.float(), tx.float()))
        del got
    del o
    run()
    ttnn.synchronize_device(device)
    ns = read_kernel_ns(device)
    for t in live:
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    return ns, ok
