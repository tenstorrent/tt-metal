# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""q_stem x-supply bench (single chip): the stage-1 x-round multicast pipeline in isolation.

One injector (row 0) reads the latent x (L1 interleaved, bf16 TILE) one round at a time (one x row tile =
round_tiles tiles) and multicasts it into slot r % depth of every stage-1 core's cb_x_round (rect cols 0-8,
rows 1-8 = 72 cores). Receivers credit freed slots back; a consumer (BRISC) checksums each round, optionally
holds it for `spin` ticks (stand-in for the matmul), and pops it.

Variants (all deliver the same bytes; the consumer checksum is checked against the host):
  baseline        the op today: Mcast helper (Flag, handshake off) + one credit ahead (helper's single flag word)
  raw_ahead1      raw multicast, per-slot flags, but still one credit ahead (checks the raw path == the helper)
  slotflags       per-slot flags, credits `depth` rounds ahead (removes the per-round credit round trip)
  noreserve       slotflags + multicast without path reservation
  posted          slotflags + posted (un-acked) multicast data writes
  posted_nr       posted + no path reservation
  dualnoc         one injector core, both RISCs multicast half of every round (NoC0 + NoC1), self-read
  twoinj          two injector cores, half of every round each (NoC0 / NoC1)
  twoinj_rot      two injector cores, alternate rounds (the op's X_INJECTORS=2), credits depth ahead
  fourlane        two injector cores x two RISCs, a quarter of every round each
  rectsplit       two injector cores, each the whole round to half the rect (rows 1-4 NoC0 / rows 5-8 NoC1)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import ttnn

KDIR = Path(__file__).resolve().parent / "kernels"
CB_X, CB_INJ, CB_OUT = 0, 1, 2
OUT_PAGE = 64  # bytes, one uint32 row of 16
RECT = (0, 1, 8, 8)  # logical x0, y0, x1, y1 (the op's stage-1 rect on the 11x10 grid)
INJ = (5, 0)


@dataclass
class Lane:
    core: tuple
    risc: str  # "RISCV_0" | "RISCV_1"
    noc: int
    t_frac: tuple = (0, 1)  # (part, parts) of each round's tiles
    t_range: tuple | None = None  # explicit (first tile, tiles) of each round, overrides t_frac
    r0: int = 0
    rstep: int = 1
    rect: tuple = RECT
    mode: int = 0  # 0 = reader on the other RISC feeds cb_x_round, 1 = self-read ring


@dataclass
class Variant:
    kind: str = "raw"  # "helper" | "raw"
    lanes: list = field(default_factory=list)
    ahead: int | None = None  # None = depth
    path_reserve: bool = True
    posted: bool = False
    chunk: int = 16384
    setcount: bool = False  # one monotone data-ready counter per lane (set_multicast of the round count)
    noread: bool = False  # diagnostic: the injector reader skips the x reads (data is garbage)


def _l(core=INJ, risc="RISCV_1", noc=0, **kw):
    return Lane(core=core, risc=risc, noc=noc, **kw)


VARIANTS = {
    "baseline": Variant(kind="helper", lanes=[_l()], ahead=1),
    "raw_ahead1": Variant(lanes=[_l()], ahead=1),
    "slotflags": Variant(lanes=[_l()]),
    "noreserve": Variant(lanes=[_l()], path_reserve=False),
    "posted": Variant(lanes=[_l()], posted=True),
    "posted_nr": Variant(lanes=[_l()], posted=True, path_reserve=False),
    "dualnoc": Variant(
        lanes=[_l(t_frac=(0, 2), mode=1), _l(risc="RISCV_0", noc=1, t_frac=(1, 2), mode=1)],
    ),
    "twoinj": Variant(lanes=[_l(core=(4, 0), t_frac=(0, 2)), _l(core=(6, 0), noc=1, t_frac=(1, 2))]),
    "twoinj_rot": Variant(lanes=[_l(core=(4, 0), r0=0, rstep=2), _l(core=(6, 0), noc=1, r0=1, rstep=2)]),
    "twoinj_rot_a1": Variant(lanes=[_l(core=(4, 0), r0=0, rstep=2), _l(core=(6, 0), noc=1, r0=1, rstep=2)], ahead=2),
    "fourlane": Variant(
        lanes=[
            _l(core=(4, 0), t_frac=(0, 4), mode=1),
            _l(core=(4, 0), risc="RISCV_0", noc=1, t_frac=(1, 4), mode=1),
            _l(core=(6, 0), t_frac=(2, 4), mode=1),
            _l(core=(6, 0), risc="RISCV_0", noc=1, t_frac=(3, 4), mode=1),
        ],
    ),
    "slotflags_noread": Variant(lanes=[_l()], noread=True),
    "baseline_noread": Variant(kind="helper", lanes=[_l()], ahead=1, noread=True),
    "twoinj_noread": Variant(
        lanes=[_l(core=(4, 0), t_frac=(0, 2)), _l(core=(6, 0), noc=1, t_frac=(1, 2))], noread=True
    ),
    "twoinj_n00": Variant(lanes=[_l(core=(4, 0), t_frac=(0, 2)), _l(core=(6, 0), noc=0, t_frac=(1, 2))]),
    "twoinj_n11": Variant(lanes=[_l(core=(4, 0), noc=1, t_frac=(0, 2)), _l(core=(6, 0), noc=1, t_frac=(1, 2))]),
    "threeinj": Variant(
        lanes=[_l(core=(4, 0), t_frac=(0, 3)), _l(core=(5, 0), noc=1, t_frac=(1, 3)), _l(core=(6, 0), t_frac=(2, 3))]
    ),
    "fourinj": Variant(
        lanes=[
            _l(core=(3, 0), t_frac=(0, 4)),
            _l(core=(4, 0), noc=1, t_frac=(1, 4)),
            _l(core=(6, 0), t_frac=(2, 4)),
            _l(core=(7, 0), noc=1, t_frac=(3, 4)),
        ]
    ),
    "slotflags_n1": Variant(lanes=[_l(noc=1)]),
    "twoinj_rot_n00": Variant(lanes=[_l(core=(4, 0), r0=0, rstep=2), _l(core=(6, 0), noc=0, r0=1, rstep=2)]),
    "twoinj_28_20": Variant(lanes=[_l(core=(4, 0), t_range=(0, 28)), _l(core=(6, 0), noc=1, t_range=(28, 20))]),
    "twoinj_20_28": Variant(lanes=[_l(core=(4, 0), t_range=(0, 20)), _l(core=(6, 0), noc=1, t_range=(20, 28))]),
    "twoinj_swap": Variant(lanes=[_l(core=(4, 0), noc=1, t_frac=(0, 2)), _l(core=(6, 0), noc=0, t_frac=(1, 2))]),
    "setcount": Variant(lanes=[_l()], setcount=True),
    "twoinj_sc": Variant(lanes=[_l(core=(4, 0), t_frac=(0, 2)), _l(core=(6, 0), noc=1, t_frac=(1, 2))], setcount=True),
    "twoinj_sc_d2": Variant(
        lanes=[_l(core=(4, 0), t_frac=(0, 2)), _l(core=(6, 0), noc=1, t_frac=(1, 2))], setcount=True
    ),
    "twoinj_sc_a1": Variant(
        lanes=[_l(core=(4, 0), t_frac=(0, 2)), _l(core=(6, 0), noc=1, t_frac=(1, 2))], setcount=True, ahead=1
    ),
    "twoinj_sc_a2": Variant(
        lanes=[_l(core=(4, 0), t_frac=(0, 2)), _l(core=(6, 0), noc=1, t_frac=(1, 2))], setcount=True, ahead=2
    ),
    "setcount_a2": Variant(lanes=[_l()], setcount=True, ahead=2),
    "rectsplit": Variant(
        lanes=[_l(core=(4, 0), rect=(0, 1, 8, 4)), _l(core=(6, 0), noc=1, rect=(0, 5, 8, 8))],
    ),
}


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sorted(cores)])


def _rect_cores(rect):
    x0, y0, x1, y1 = rect
    return [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


def _dm(risc, noc):
    return ttnn.DataMovementConfigDescriptor(
        processor=getattr(ttnn.DataMovementProcessor, risc), noc=ttnn.NOC.NOC_0 if noc == 0 else ttnn.NOC.NOC_1
    )


def _cb(index, dtype, page, pages, crs):
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


def _src(name):
    return str(KDIR / name)


def create_program_descriptor(x, out, *, variant, depth=3, spin=0, full=False):
    v = VARIANTS[variant] if isinstance(variant, str) else variant
    dev = x.device()
    tile_bytes = x.buffer_aligned_page_size()
    rows_t, round_tiles = x.shape[-2] // 32, x.shape[-1] // 32
    n_rounds = rows_t
    recv = _rect_cores(RECT)
    inj_cores = sorted({ln.core for ln in v.lanes})
    all_cores = recv + inj_cores
    all_crs, recv_crs, inj_crs = _crs(all_cores), _crs(recv), _crs(inj_cores)
    ahead = depth if v.ahead is None else v.ahead
    virt = lambda c: dev.worker_core_from_logical_core(ttnn.CoreCoord(*c))
    x_ct = ttnn.TensorAccessorArgs(x).get_compile_time_args()
    out_ct = ttnn.TensorAccessorArgs(out).get_compile_time_args()
    xaddr = x.buffer_address()

    cbs = [_cb(CB_X, x.dtype, tile_bytes, depth * round_tiles, all_crs)]
    if any(ln.mode == 1 for ln in v.lanes):
        cbs.append(_cb(CB_INJ, x.dtype, tile_bytes, 2 * round_tiles, _crs({ln.core for ln in v.lanes if ln.mode == 1})))
    cbs.append(_cb(CB_OUT, ttnn.uint32, OUT_PAGE, 1, recv_crs))

    # consumer (shared by all variants): BRISC on the receivers
    cons_rt = ttnn.RuntimeArgs()
    for i, (cx, cy) in enumerate(recv):
        cons_rt[cx][cy] = [out.buffer_address(), i]
    consumer = ttnn.KernelDescriptor(
        kernel_source=_src("xs_consumer.cpp"),
        core_ranges=recv_crs,
        compile_time_args=[CB_X, CB_OUT, tile_bytes, round_tiles, n_rounds, spin, int(full), OUT_PAGE, *out_ct],
        runtime_args=cons_rt,
        config=_dm("RISCV_0", 1),
    )

    def reader(ln, t_lo, n_t):
        rt = ttnn.RuntimeArgs()
        rt[ln.core[0]][ln.core[1]] = [xaddr]
        other = "RISCV_0" if ln.risc == "RISCV_1" else "RISCV_1"
        return ttnn.KernelDescriptor(
            kernel_source=_src("xs_reader.cpp"),
            core_ranges=_crs([ln.core]),
            compile_time_args=[CB_X, tile_bytes, round_tiles, n_rounds, ln.r0, ln.rstep, t_lo, n_t, *x_ct],
            runtime_args=rt,
            defines=[("XS_NOREAD", "1")] if v.noread else [],
            config=_dm(other, 1 - ln.noc),
        )

    if v.kind == "helper":
        ln = v.lanes[0]
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=all_crs, initial_value=0)]
        send = ttnn.KernelDescriptor(
            kernel_source=_src("xs_helper_sender.cpp"),
            core_ranges=inj_crs,
            compile_time_args=[CB_X, tile_bytes, round_tiles, n_rounds, depth, len(recv), 0],
            runtime_args=ttnn.RuntimeArgs(),
            config=_dm(ln.risc, ln.noc),
        )
        iv = virt(ln.core)
        rcv = ttnn.KernelDescriptor(
            kernel_source=_src("xs_helper_receiver.cpp"),
            core_ranges=recv_crs,
            compile_time_args=[CB_X, round_tiles, n_rounds, ahead, 0, iv.x, iv.y],
            runtime_args=ttnn.RuntimeArgs(),
            config=_dm("RISCV_1", 0),
        )
        desc = ttnn.ProgramDescriptor(cbs=cbs, semaphores=sems)
        cfg = ttnn.McastConfig()
        cfg.noc = ttnn.NOC.NOC_0 if ln.noc == 0 else ttnn.NOC.NOC_1
        cfg.handshake = False
        x0, y0, x1, y1 = RECT
        rect_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])
        ttnn.Mcast2D(dev, rect_crs, ttnn.Mcast2DFixedSenderConfig(ttnn.CoreCoord(*ln.core)), cfg).attach(
            desc, "mcast0", [send, rcv]
        )
        desc.kernels = [reader(ln, 0, round_tiles), send, rcv, consumer]
        return desc

    # raw lanes: lane l owns credit semaphore l (on its core) and flags 4 + l * depth + slot (on the receivers)
    assert len(v.lanes) <= 4 and 4 + 4 * depth <= 16
    nsem = 4 + len(v.lanes) * depth
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_crs, initial_value=0) for i in range(nsem)]
    kernels = []
    lane_rt = {c: [] for c in recv}
    for l, ln in enumerate(v.lanes):
        part, parts = ln.t_frac
        assert round_tiles % parts == 0
        n_t = round_tiles // parts
        t_lo = part * n_t
        if ln.t_range is not None:
            t_lo, n_t = ln.t_range
        x0, y0, x1, y1 = ln.rect
        lo, hi = virt((x0, y0)), virt((x1, y1))
        mx = (lo.x, lo.y, hi.x, hi.y) if ln.noc == 0 else (hi.x, hi.y, lo.x, lo.y)
        lrecv = _rect_cores(ln.rect)
        rt = ttnn.RuntimeArgs()
        rt[ln.core[0]][ln.core[1]] = [xaddr]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=_src("xs_lane_sender.cpp"),
                core_ranges=_crs([ln.core]),
                compile_time_args=[
                    CB_X,
                    CB_INJ,
                    tile_bytes,
                    round_tiles,
                    n_rounds,
                    ln.r0,
                    ln.rstep,
                    t_lo,
                    n_t,
                    depth,
                    len(lrecv),
                    l,
                    4 + l * depth,
                    int(v.path_reserve),
                    int(v.posted),
                    *mx,
                    len(lrecv),
                    ln.mode,
                    v.chunk,
                    *x_ct,
                ],  # fmt: skip
                runtime_args=rt,
                defines=[("XS_SETCOUNT", "1")] if v.setcount else [],
                config=_dm(ln.risc, ln.noc),
            )
        )
        if ln.mode == 0:
            kernels.append(reader(ln, t_lo, n_t))
        cv = virt(ln.core)
        for c in lrecv:
            lane_rt[c] += [cv.x, cv.y, l, 4 + l * depth, ln.r0, ln.rstep]
    rrt = ttnn.RuntimeArgs()
    for (cx, cy), a in lane_rt.items():
        rrt[cx][cy] = [len(a) // 6, *a]
    kernels.append(
        ttnn.KernelDescriptor(
            kernel_source=_src("xs_lane_receiver.cpp"),
            core_ranges=recv_crs,
            compile_time_args=[CB_X, round_tiles, n_rounds, depth, ahead],
            runtime_args=rrt,
            defines=[("XS_SETCOUNT", "1")] if v.setcount else [],
            config=_dm("RISCV_1", 0),
        )
    )
    kernels.append(consumer)
    return ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems)


def x_supply(x, out, **kw):
    return ttnn.generic_op([x, out], create_program_descriptor(x, out, **kw))
