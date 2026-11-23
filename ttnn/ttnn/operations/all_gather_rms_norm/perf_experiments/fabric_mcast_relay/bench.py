# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""fabric_mcast_relay bench: the all_gather_rms_norm bulk gather (TRANSPORT only, no norm), relay vs chip-level mcast.

Only the 2 * num_links port cores of every chip run. Each chip's output (per-device (R*32, G*W_local) bf16 TILE,
interleaved DRAM, slot p at tile-columns p*Wlt..) is prefilled with its own slice; after one call every chip holds
every slice (checked bit-exactly).

Variants
  relay        the op's port reader + sender, UNCHANGED (kernel files read from the op dir), workers stubbed
               (n_active = 0: the own slice is already in the destination pages, the own entry reads it back as
               OWN_SRC_DST = 1 does). Own entry + relay entry (balanced far slot when R * Wlt >= 64).
  relay_u      the same with uniform port placement (every chip's (d, l) port on the same logical core).
  mcast        the op's port reader (own entry only, expect_in 0) + kernels/mcast_port_sender.cpp: each port sends its
               own lane once as chip-level multicast segments whose routes are host-computed from the physical link
               directions (ttnn.get_eth_forwarding_direction): straight range or N/S-then-E/W L; on the QB2 square only
               one direction per chip is expressible, so that port covers both the neighbour and the far chip. The
               op's port placement; arrival counters on a uniform "counter core" per lane (extra global semaphore).
  mcast_u      the same with uniform port placement.
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm_program_descriptor as agd

HERE = Path(__file__).parent
MY_KERNELS = HERE / "kernels"
OP_KERNELS = agd.KERNEL_DIR
TILE = 32
OPP = {"fwd": "bwd", "bwd": "fwd"}


def plan_for(mesh, num_links, uniform):
    """The op's port plan (cluster_axis 1, Ring); uniform: every chip takes chip 0's port cores."""
    plan = agd.build_plan(mesh, 1, ttnn.Topology.Ring, num_links)
    if not uniform:
        return plan
    chips = {}
    c0 = sorted(plan["chips"])[0]
    for coord, ch in plan["chips"].items():
        chips[coord] = dict(ch)
        chips[coord]["ports"] = dict(plan["chips"][c0]["ports"])
    out = dict(plan)
    out["chips"] = chips
    return out


def _union_plan(mesh, num_links):
    """Every port core of either placement: the global semaphores exist on all of them (same address)."""
    a, b = plan_for(mesh, num_links, False), plan_for(mesh, num_links, True)
    chips = {
        c: {
            "ports": {
                **{("a",) + k: v for k, v in a["chips"][c]["ports"].items()},
                **{("b",) + k: v for k, v in b["chips"][c]["ports"].items()},
            }
        }
        for c in a["chips"]
    }
    return {"chips": chips}


def make_output(mesh, R, Wl, G, seed=0):
    import torch  # local: no global torch import under ttnn/

    """Per-chip output with only the own slot filled; returns (device tensor, expected full gather [R*32, G*Wl])."""
    g = torch.Generator().manual_seed(seed)
    slices = [torch.randn(R * TILE, Wl, generator=g).to(torch.bfloat16) for _ in range(G)]
    full = torch.cat(slices, dim=1)
    per = torch.zeros(G, 1, R * TILE, G * Wl, dtype=torch.bfloat16)
    for p in range(G):
        per[p, 0, :, p * Wl : (p + 1) * Wl] = slices[p]
    t = ttnn.from_torch(
        per,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    return t, full


def check_output(mesh, t, full):
    import torch  # local: no global torch import under ttnn/

    got = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    bad = []
    G = got.shape[0]
    Wl = full.shape[1] // G
    for p in range(G):
        if not torch.equal(got[p, 0], full):
            slots = [
                int((got[p, 0][:, q * Wl : (q + 1) * Wl] != full[:, q * Wl : (q + 1) * Wl]).sum()) for q in range(G)
            ]
            bad.append((p, slots))
    if bad:
        print("FMR got shape", tuple(got.shape), "sample", got[0, 0, 0, :4], full[0, :4])
    return bad


def _sems(mesh, plan):
    port_cores = {(c.x, c.y) for ch in plan["chips"].values() for c in ch["ports"].values()}
    key = ("bench_sems", tuple(sorted(port_cores)))
    if key not in _SEMS:
        cores = agd._single_cores(port_cores)
        _SEMS[key] = tuple(ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(3))
        ttnn.synchronize_device(mesh)
    return [int(ttnn.get_global_semaphore_address(s)) for s in _SEMS[key]]


_SEMS = {}


def build(mesh, variant, out, R, Wl, num_links=2, tag=0):
    uniform = variant in ("relay_u", "mcast_u")
    is_mc = variant.startswith("mcast")
    plan = plan_for(mesh, num_links, uniform)
    chips, G, ring, L = plan["chips"], plan["G"], plan["ring"], plan["num_links"]
    assert ring and G == 4
    Wlt = Wl // TILE
    Wt = G * Wlt
    out_page = 2 * TILE * TILE
    nb = mesh.dram_grid_size().x * mesh.dram_grid_size().y
    run_pages = max(1, ttnn.get_tt_fabric_max_payload_size_bytes() // out_page)
    batch, inc_every, scatter_runs, read_depth = agd.BATCH_CHUNKS, agd.INC_EVERY, agd.SCATTER_RUNS, agd.READ_DEPTH
    split_runs = agd.SPLIT_RUNS
    walk = agd.Walk(nb, R, Wlt, Wt, Wlt, run_pages, scatter_runs, 0, split_runs)
    balance = G % 2 == 0 and G >= 4 and R * Wlt >= agd.BALANCE_MIN_TILES
    ready_addr, bulk_addr, mc_addr = _sems(mesh, _union_plan(mesh, num_links))
    dst_addr = int(out.buffer_address())
    port_ct = [
        agd.CB_PORT_CHUNKS,
        out_page,
        run_pages,
        batch,
        inc_every,
        nb,
        walk.rows_eff,
        walk.cols_eff,
        walk.row_stride,
        walk.slot_stride,
        0,
        Wlt,
        1,
        1,
        1,
        1,
        1,
        ttnn.get_tt_fabric_max_payload_size_bytes(),
        scatter_runs,
        0,
        1,
        read_depth,
        1,
        agd.CB_PORT_DESC,
        split_runs,
        1,
        agd.EAGER_PUSH,
    ] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    # bulk schedules of every chip (the op's)
    sched = {}
    for cc, ch in chips.items():
        f, b = agd._schedule(ch["p"], G, ring, balance)
        sched[cc] = {"fwd": f, "bwd": b}

    # multicast segments per (chip, d, l): [(packets, range)] in walk order (part 1 = prefix, part 2 = suffix)
    def step(coord, d, h):
        rg = agd._fag.build_groups(tuple(mesh.shape), 1)[0]
        i = rg.index(coord)
        return rg[(i + h) % G] if d == "fwd" else rg[(i - h) % G]

    def ldir(a, b):  # physical eth direction of the first hop a -> b (E 0, W 1, N 2, S 3)
        return ttnn.get_eth_forwarding_direction(agd._node(mesh, a), agd._node(mesh, b))

    def spec_for(X, d, r):
        """MeshMcastRange spec writing the ring's next r chips of X along d, or None (not one 2D mcast route)."""
        c1 = step(X, d, 1)
        d1 = ldir(X, c1)
        hops = [0, 0, 0, 0]
        if r == 1:
            hops[d1] = 1
        else:
            d2 = ldir(c1, step(X, d, 2))
            if d1 == d2:
                hops[d1] = 2
            elif d1 in (2, 3) and d2 in (0, 1):  # N/S spine hop, then an E/W branch hop
                hops[d1] = 1
                hops[d2] = 1
            else:
                return None
        return hops[0] | hops[1] << 8 | hops[2] << 16 | hops[3] << 24

    seg_cache = {}

    def segments(X, d, l):
        """(n_all, seg_split, spec_a, spec_b, range_a, range_b) of port (d, l) of chip X: its own lane, sent once."""
        key = (X, d, l)
        if key in seg_cache:
            return seg_cache[key]
        p = chips[X]["p"]
        n_all = walk.chunks(p, 0, l, L)
        exp = {dd: spec_for(X, dd, 2) is not None for dd in ("fwd", "bwd")}
        if exp["fwd"] and exp["bwd"] and balance:
            n1 = walk.chunks(p, 1, l, L)
            ra, rb = (2, 1) if d == "fwd" else (1, 2)
        else:
            if not balance and exp["fwd"]:
                far = "fwd"  # the op's unbalanced ring: the far slot goes forward
            elif exp["bwd"]:
                far = "bwd"
            elif exp["fwd"]:
                far = "fwd"
            else:
                raise RuntimeError(f"chip {X}: no 2-chip multicast route in either direction")
            n1 = n_all
            ra = rb = 2 if d == far else 1
        out = (n_all, n1, spec_for(X, d, ra), spec_for(X, d, rb), ra, rb)
        seg_cache[key] = out
        return out

    # readies (mcast variant): every chip a port writes into sends that port one ready, on its own port whose link is
    # the fabric table's first hop back (a 2-hop ready is routed by the table)
    readies = {}
    if is_mc:
        for X in chips:
            for d in ("fwd", "bwd"):
                for l in range(L):
                    _, _, _, _, ra, rb = segments(X, d, l)
                    for h in range(1, max(ra, rb) + 1):
                        Y = step(X, d, h)
                        dy = ldir(Y, X)
                        dd = "fwd" if ldir(Y, step(Y, "fwd", 1)) == dy else "bwd"
                        assert ldir(Y, step(Y, dd, 1)) == dy, (Y, X, dy)
                        xc = chips[X]["ports"][(d, l)]
                        xy = mesh.worker_core_from_logical_core(ttnn.CoreCoord(xc.x, xc.y))
                        readies.setdefault((Y, dd, l), []).append(
                            (int(agd._node(mesh, X).chip_id), (xy.x & 0xFFFF) | (xy.y << 16))
                        )
    if is_mc:
        for X in sorted(chips):
            print(
                f"FMR route {X}: "
                + " ".join(f"{d}{l}:{segments(X, d, l)}" for d in ("fwd", "bwd") for l in range(L))
                + f" readies {[(k, v) for k, v in readies.items() if k[0] == X]}"
            )
    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ch in chips.items():
        p = ch["p"]
        ports = ch["ports"]
        port_list = [ports[(d, l)] for d in ("fwd", "bwd") for l in range(L)]
        port_set = {(c.x, c.y) for c in port_list}
        pcores = agd._single_cores(port_set)
        virt = lambda c: mesh.worker_core_from_logical_core(ttnn.CoreCoord(*c))
        pack = lambda v: (v.x & 0xFFFF) | (v.y << 16)
        program = ttnn.ProgramDescriptor()
        program.semaphores = [ttnn.SemaphoreDescriptor(id=sid, core_ranges=pcores, initial_value=0) for sid in range(8)]
        program.cbs = [
            agd._cb(agd.CB_PORT_CHUNKS, 2 * batch * run_pages, out_page, ttnn.bfloat16, pcores),
            agd._cb(agd.CB_PORT_DESC, 2 * batch, agd.PORT_DESC_BYTES, ttnn.float32, pcores),
        ]
        reader_rt, sender_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        sig = []
        up_of = {"fwd": ch["prev"], "bwd": ch["next"]}
        for d in ("fwd", "bwd"):
            peer = ch["next"] if d == "fwd" else ch["prev"]
            up = up_of[d]
            for l in range(L):
                core = ports[(d, l)]
                pn = agd._node(mesh, peer)
                if variant in ("relay", "relay_u"):
                    sends = sched[coord][d]
                    up_sends = sched[up][d]
                    up_counts = [walk.chunks(q, part, l, L) for q, part in up_sends]
                    S0, I0, expect_in = agd._inc_sim(up_counts, inc_every)
                    entries = []
                    for kk, (q, part) in enumerate(sends):
                        if kk == 0:
                            entries += [q | (part << 16), 0, 0]
                            continue
                        uq, upart = up_sends[kk - 1]
                        off = walk.chunks(q, 1, l, L) if (part == 2 and upart == 0) else 0
                        e = kk - 1
                        entries += [q | (part << 16), I0[e] - S0[e] // inc_every + 1, S0[e] + off]
                else:
                    sends = [(p, 0)]
                    entries = [p, 0, 0]
                    # arrivals into my (d, l) counter: one per non-empty segment of every upstream sender reaching me
                    expect_in = 0
                    for h in (1, 2):
                        n_all, n1, _, _, ra, rb = segments(step(coord, OPP[d], h), d, l)
                        expect_in += int(n1 > 0 and ra >= h) + int(n_all - n1 > 0 and rb >= h)
                mc_expect = 0
                if is_mc:
                    mc_expect, expect_in = expect_in, 0  # the candidate sender waits (counter on the lane's core)
                r_args = [dst_addr, l, L, bulk_addr, expect_in, 0, 0, len(sends)] + entries
                r_args += [0, 0, 0, 0]  # hub_sem, mc_start, mc_end, mc_dests
                r_args += [bulk_addr, 0, 0xFFFF, 0, 0]  # stats: addr, slot bytes, own slot, writers, n_recv
                r_args += [0, 0, bulk_addr]  # own_bands = 1: need, row end, counter (unused)
                reader_rt[core.x][core.y] = r_args
                pc = virt((chips[peer]["ports"][(d, l)].x, chips[peer]["ports"][(d, l)].y))
                rc = virt((chips[peer]["ports"][(OPP[d], l)].x, chips[peer]["ports"][(OPP[d], l)].y))
                if variant in ("relay", "relay_u"):
                    s_args = [
                        dst_addr,
                        l,
                        L,
                        bulk_addr,
                        ready_addr,
                        1,
                        pack(rc),
                        pack(pc),
                        int(pn.mesh_id),
                        int(pn.chip_id),
                        1,
                        0,
                        0,
                        bulk_addr,
                        0,
                        0,
                        0,
                    ]
                    s_args += [len(sends)]
                    for q, part in sends:
                        s_args += [q | (part << 16), walk.chunks(q, part, l, L)]
                else:
                    n_all, n1, sa, sb, ra, rb = segments(coord, d, l)
                    c0 = chips[sorted(chips)[0]]["ports"][(d, l)]  # lane (d, l) counter core: chip 0's port core
                    s_args = [
                        dst_addr,
                        mc_addr,
                        ready_addr,
                        max(ra, rb),
                        pack(virt((c0.x, c0.y))),
                        int(pn.mesh_id),
                        int(pn.chip_id),
                        1,
                        n_all,
                        n1,
                        sa,
                        sb,
                    ]
                    rd = readies.get((coord, d, l), [])
                    s_args += [len(rd)] + [v for r_ in rd for v in r_] + [mc_expect]
                sig.append((d, l, core.x, core.y, tuple(r_args[1:]), tuple(s_args[1:])))
                s_args += list(
                    ttnn.setup_fabric_connection(agd._node(mesh, coord), pn, ch[f"{d}_links"][l], program, core)
                )
                sender_rt[core.x][core.y] = s_args
        sender_src = (
            (OP_KERNELS / "all_gather_rms_norm_port_sender.cpp")
            if not is_mc
            else (MY_KERNELS / "mcast_port_sender.cpp")
        )
        program.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(OP_KERNELS / "all_gather_rms_norm_port_reader.cpp"),
                core_ranges=pcores,
                compile_time_args=port_ct,
                runtime_args=reader_rt,
                config=agd._dm(ttnn.DataMovementProcessor.RISCV_1, agd.NOC_READER),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(sender_src),
                core_ranges=pcores,
                defines=[("FMR_DPRINT", "1")] if os.environ.get("FMR_DPRINT") else [],
                compile_time_args=port_ct,
                runtime_args=sender_rt,
                config=agd._dm(ttnn.DataMovementProcessor.RISCV_0, agd.NOC_WRITER),
            ),
        ]
        h = hash((variant, coord, R, Wl, L, tuple(sig), tag)) & ((1 << 63) - 1)
        program.custom_program_hash = h
        r, c = coord
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return mesh_desc, plan


def make_call(mesh, variant, R, Wl, num_links=2, seed=0):
    out, full = make_output(mesh, R, Wl, mesh.shape[1], seed)
    desc, plan = build(mesh, variant, out, R, Wl, num_links)
    call = lambda: ttnn.generic_op([out], desc)
    return call, out, full, plan
