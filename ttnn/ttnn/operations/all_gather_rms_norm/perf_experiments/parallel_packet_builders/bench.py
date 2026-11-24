# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""parallel_packet_builders bench: the all_gather_rms_norm bulk gather (TRANSPORT only, no norm).

Each chip's output (per-device (R*32, G*W_local) bf16 TILE, interleaved DRAM, slot p at tile-columns p*Wlt..) is
prefilled with its own slice; after one call every chip holds every slice (checked bit-exactly, first call and
after the perf runs). Same packets, destination pages, arrival protocol and port placement as the op.

Variants
  base       the op's port reader + sender, UNCHANGED (kernel files read from the op dir); workers stubbed
             (n_active = 0: the own slice already is in the destination pages, the own entry reads it back as the
             op's DRAM read-back path does).
  ppbH[_dD]  H helper cores per port build every packet (kernels/ppb_helper.cpp: walk + run reads + header), the
             port BRISC (kernels/ppb_port_sender.cpp) only forwards finished packets from its ring of H*D slots, the
             port NCRISC (kernels/ppb_port_reader.cpp) forwards the arrival counter / own readiness to the helpers.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import ttnn

from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm_program_descriptor as agd

HERE = Path(__file__).parent
MY_KERNELS = HERE / "kernels"
OP_KERNELS = agd.KERNEL_DIR
TILE = 32
HDR_REGION = 128  # >= meta (16) + packet header, rounded to 64 (ppb_helper.cpp PAY_OFF)
CB_HELPER = 22
CB_RING = 35  # ppbo: the helpers' packet ring on the port (cb_chunks stays the relay path's)


def parse_variant(v):
    m = re.fullmatch(r"ppbo?(\d+)(?:_d(\d+))?(?:_p(\d+))?", v)
    if not m:
        return None
    H = int(m.group(1))
    D = int(m.group(2) or 2)
    P = int(m.group(3) or min(D, 2))
    assert 1 <= H <= 6 and P <= D
    return H, D, P


def make_output(mesh, R, Wl, G, seed=0):
    import torch

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
    import torch

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
    return bad


_SEMS = {}


def _sems(mesh, plan):
    port_cores = {(c.x, c.y) for ch in plan["chips"].values() for c in ch["ports"].values()}
    key = tuple(sorted(port_cores))
    if key not in _SEMS:
        cores = agd._single_cores(port_cores)
        _SEMS[key] = tuple(ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2))
        ttnn.synchronize_device(mesh)
    return [int(ttnn.get_global_semaphore_address(s)) for s in _SEMS[key]]


def pick_helpers(grid, ports, H):
    """H free cores per port, nearest first (round-robin over the ports so each gets close ones)."""
    taken = {(c.x, c.y) for c in ports.values()}
    free = [(x, y) for x in range(grid[0]) for y in range(grid[1]) if (x, y) not in taken]
    out = {k: [] for k in ports}
    for _ in range(H):
        for k, c in sorted(ports.items()):
            best = min(free, key=lambda f: (abs(f[0] - c.x) + abs(f[1] - c.y), f))
            free.remove(best)
            out[k].append(best)
    return out


def build(mesh, variant, out, R, Wl, num_links=2, tag=0):
    split_runs = 1 if variant.endswith("_s") else agd.SPLIT_RUNS  # _s: fuller packets (runs split across packets)
    vname = variant[:-2] if variant.endswith("_s") else variant
    ppb = parse_variant(vname)
    own_only = vname.startswith("ppbo")  # helpers build the own entry only; the port keeps the relay entries
    plan = agd.build_plan(mesh, 1, ttnn.Topology.Ring, num_links)
    chips, G, ring, L = plan["chips"], plan["G"], plan["ring"], plan["num_links"]
    assert ring and G == 4
    Wlt = Wl // TILE
    Wt = G * Wlt
    out_page = 2 * TILE * TILE
    nb = mesh.dram_grid_size().x * mesh.dram_grid_size().y
    run_pages = max(1, ttnn.get_tt_fabric_max_payload_size_bytes() // out_page)
    batch, inc_every, scatter_runs, read_depth = agd.BATCH_CHUNKS, agd.INC_EVERY, agd.SCATTER_RUNS, agd.READ_DEPTH
    walk = agd.Walk(nb, R, Wlt, Wt, Wlt, run_pages, scatter_runs, 0, split_runs)
    balance = G % 2 == 0 and G >= 4 and R * Wlt >= agd.BALANCE_MIN_TILES
    ready_addr, bulk_addr = _sems(mesh, plan)
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
        agd.CB_STAGE,
        0,
    ] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    chunk_bytes = run_pages * out_page
    pitch = ((16 + HDR_REGION + chunk_bytes + 63) // 64) * 64
    local_pitch = HDR_REGION + chunk_bytes
    grid = tuple(plan["grid"])
    sched = {}
    for cc, ch in chips.items():
        f, b = agd._schedule(ch["p"], G, ring, balance)
        sched[cc] = {"fwd": f, "bwd": b}

    virt = lambda c: mesh.worker_core_from_logical_core(ttnn.CoreCoord(*c))
    pack = lambda v: (v.x & 0xFFFF) | (v.y << 16)
    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ch in chips.items():
        ports = ch["ports"]
        port_set = {(c.x, c.y) for c in ports.values()}
        pcores = agd._single_cores(port_set)
        helpers = pick_helpers(grid, ports, ppb[0]) if ppb else {}
        h_set = {c for v in helpers.values() for c in v}
        hcores = agd._single_cores(h_set) if h_set else None
        all_cores = agd._single_cores(port_set | h_set)
        program = ttnn.ProgramDescriptor()
        n_ready = ppb[0] if ppb else 0
        program.semaphores = [
            ttnn.SemaphoreDescriptor(id=sid, core_ranges=pcores, initial_value=0) for sid in range(8 + n_ready)
        ]
        if ppb:
            program.semaphores += [
                ttnn.SemaphoreDescriptor(id=sid, core_ranges=hcores, initial_value=0) for sid in (8, 9, 10)
            ]
        if ppb and own_only:
            H, D, P = ppb
            program.cbs = [
                agd._cb(CB_RING, H * D, pitch, ttnn.bfloat16, all_cores),  # first: same address on every core
                agd._cb(agd.CB_PORT_CHUNKS, 2 * batch * run_pages, out_page, ttnn.bfloat16, pcores),
                agd._cb(agd.CB_PORT_DESC, 2 * batch, agd.PORT_DESC_BYTES, ttnn.float32, pcores),
                agd._cb(CB_HELPER, P + 1, local_pitch, ttnn.bfloat16, hcores),
            ]
        elif ppb:
            H, D, P = ppb
            program.cbs = [
                agd._cb(agd.CB_PORT_CHUNKS, H * D, pitch, ttnn.bfloat16, all_cores),  # same address on helpers
                agd._cb(agd.CB_PORT_DESC, 2 * batch, agd.PORT_DESC_BYTES, ttnn.float32, pcores),
                agd._cb(CB_HELPER, P + 1, local_pitch, ttnn.bfloat16, hcores),
            ]
        else:
            program.cbs = [
                agd._cb(agd.CB_PORT_CHUNKS, 2 * batch * run_pages, out_page, ttnn.bfloat16, pcores),
                agd._cb(agd.CB_PORT_DESC, 2 * batch, agd.PORT_DESC_BYTES, ttnn.float32, pcores),
            ]
        reader_rt, sender_rt, helper_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        sig = []
        up_of = {"fwd": ch["prev"], "bwd": ch["next"]}
        for d in ("fwd", "bwd"):
            peer = ch["next"] if d == "fwd" else ch["prev"]
            up = up_of[d]
            for l in range(L):
                core = ports[(d, l)]
                pn = agd._node(mesh, peer)
                sends = sched[coord][d]
                up_sends = sched[up][d]
                up_counts = [walk.chunks(q, part, l, L) for q, part in up_sends]
                S0, I0, expect_in = agd._inc_sim(up_counts, inc_every)
                entries, entries4 = [], []
                for kk, (q, part) in enumerate(sends):
                    tot = walk.chunks(q, part, l, L)
                    if kk == 0:
                        A_, B_ = 0, 0
                    else:
                        uq, upart = up_sends[kk - 1]
                        off = walk.chunks(q, 1, l, L) if (part == 2 and upart == 0) else 0
                        e = kk - 1
                        A_, B_ = I0[e] - S0[e] // inc_every + 1, S0[e] + off
                    entries += [q | (part << 16), A_, B_]
                    entries4 += [q | (part << 16), A_, B_, tot]
                hl = [pack(virt(c)) for c in helpers.get((d, l), [])]
                r_args = [dst_addr, l, L, bulk_addr, expect_in, 0, 0, len(sends)] + entries
                r_args += [0, 0, 0, 0]  # hub_sem, mc_start, mc_end, mc_dests
                r_args += [bulk_addr, 0, 0xFFFF, 0, 0]  # stats: addr, slot bytes, own slot, writers, n_recv
                r_args += [0, 0, bulk_addr]  # own_bands = 1: need, row end, counter (unused)
                r_args += [0]  # st_on
                if ppb:
                    r_args += [len(hl)] + hl
                reader_rt[core.x][core.y] = r_args
                pc = virt((chips[peer]["ports"][(d, l)].x, chips[peer]["ports"][(d, l)].y))
                rc = virt(
                    (
                        chips[peer]["ports"][("bwd" if d == "fwd" else "fwd", l)].x,
                        chips[peer]["ports"][("bwd" if d == "fwd" else "fwd", l)].y,
                    )
                )
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
                if ppb:
                    H, D, P = ppb
                    s_args += [H, D, pitch] + ([CB_RING] if own_only else []) + hl
                    my = pack(virt((core.x, core.y)))
                    for hi, hc in enumerate(helpers[(d, l)]):
                        helper_rt[hc[0]][hc[1]] = (
                            [
                                dst_addr,
                                l,
                                L,
                                H,
                                hi,
                                D,
                                P,
                                my,
                                CB_RING if own_only else agd.CB_PORT_CHUNKS,
                                pitch,
                                local_pitch,
                                bulk_addr,
                                pack(pc),
                                int(pn.mesh_id),
                                int(pn.chip_id),
                                1,
                                1 if own_only else len(sends),
                            ]
                            + (entries4[:4] if own_only else entries4)
                            + [0]
                        )
                sig.append((d, l, core.x, core.y, tuple(r_args[1:]), tuple(s_args[1:])))
                s_args += list(
                    ttnn.setup_fabric_connection(agd._node(mesh, coord), pn, ch[f"{d}_links"][l], program, core)
                )
                sender_rt[core.x][core.y] = s_args
        kdir = MY_KERNELS if ppb else OP_KERNELS
        dbg = [("PPB_DEBUG", "1")] if os.environ.get("PPB_DEBUG") else []
        dbg += [("AGRN_PROFILE_ZONES", "1")] if os.environ.get("PPB_ZONES") else []
        pre = "ppbo" if own_only else "ppb"
        rname = f"{pre}_port_reader.cpp" if ppb else "all_gather_rms_norm_port_reader.cpp"
        sname = f"{pre}_port_sender.cpp" if ppb else "all_gather_rms_norm_port_sender.cpp"
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(kdir / rname),
                core_ranges=pcores,
                compile_time_args=port_ct,
                defines=dbg,
                runtime_args=reader_rt,
                config=agd._dm(ttnn.DataMovementProcessor.RISCV_1, agd.NOC_READER),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(kdir / sname),
                core_ranges=pcores,
                compile_time_args=port_ct,
                defines=dbg,
                runtime_args=sender_rt,
                config=agd._dm(ttnn.DataMovementProcessor.RISCV_0, agd.NOC_WRITER),
            ),
        ]
        if ppb:
            hct = list(port_ct)
            hct[0] = CB_HELPER
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(MY_KERNELS / "ppb_helper.cpp"),
                    core_ranges=hcores,
                    compile_time_args=hct,
                    defines=dbg,
                    runtime_args=helper_rt,
                    config=agd._dm(ttnn.DataMovementProcessor.RISCV_1, agd.NOC_READER),
                )
            )
        program.kernels = kernels
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
