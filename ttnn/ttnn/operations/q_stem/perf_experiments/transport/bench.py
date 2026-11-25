# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""q_stem TRANSPORT bench: the op's all-to-all of q on the 4-chip ring, by itself (no stage 1 / stage 3).

Per chip p, the op's exchange plan (q_stem_program_descriptor.exchange_plan / head_placement / landing_slot, imported
read-only): 2 * L port cores in row 0 (direction x link) send every remote quarter's head-rows (segments of hw tiles)
to the destination chip's compute core that owns the head, into its landing slot, with the op's per-row fused
arrival increments and the op's ready fence. Destination cores wait for every arrival counter to reach its total and
re-arm it (the program ends on a chip when its last remote row has landed).

Variants (knobs; see VARIANTS):
  split      table     every destination by the routing table's direction (the op: the antipodal quarter whole one way)
             balanced  the antipodal quarter split per link by destination-core group: first half the table way, the
                       rest the other way on an explicit 2-hop east/west route (one stream per (core, source) kept)
             neighbor  only the two 1-hop quarters (no antipodal): the raw per-link rate reference
  batch      segments per sender flush / reader barrier (CB = 2 batches)            op: 4
  hpd_cap    packet headers per destination (rotation)                               op: 4
  blocking   1 = flush-blocking header per packet                                    op: 0
  direct_src 1 = no reader / CB: send from the source shard                          op: 0
  inc_every  rows per arrival increment of a (core, source) (99 = once at the end)  op: 1
  posted     1 = posted NoC writes into the router slot                              op: 0
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import ttnn
from ttnn.operations.q_stem import q_stem_program_descriptor as qpd

KERNEL_DIR = Path(__file__).parent / "kernels"
FWD, BWD = qpd.FWD, qpd.BWD
MAX_HEADERS = 12
NSRC = 8  # source ring segments per port
CB_STAGE = 0

CASES = {
    "kimi": dict(B=1, S=640, Q=1536, Hl=24, Dn=128, Dr=64, N=512),
    "dsv32": dict(B=1, S=640, Q=1536, Hl=32, Dn=128, Dr=64, N=512),
    "glm": dict(B=1, S=640, Q=2048, Hl=16, Dn=192, Dr=64, N=512),
}

BASE = dict(split="table", batch=4, hpd_cap=4, blocking=0, direct_src=0, inc_every=1, posted=0)
VARIANTS = {
    "baseline": dict(BASE),
    "balanced": dict(BASE, split="balanced"),
    "neighbor": dict(BASE, split="neighbor"),
    "batch1": dict(BASE, batch=1),
    "batch2": dict(BASE, batch=2),
    "batch8": dict(BASE, batch=8),
    "hpd1": dict(BASE, hpd_cap=1),
    "blocking": dict(BASE, blocking=1),
    "direct_src": dict(BASE, direct_src=1),
    "inc_end": dict(BASE, inc_every=99),
    "inc2": dict(BASE, inc_every=2),
    "bal_inc2": dict(BASE, split="balanced", inc_every=2),
    "posted": dict(BASE, posted=1),
    "bal_batch8": dict(BASE, split="balanced", batch=8),
    "bal_direct_src": dict(BASE, split="balanced", direct_src=1),
    "bal_inc_end": dict(BASE, split="balanced", inc_every=99),
    "bal_hpd1": dict(BASE, split="balanced", hpd_cap=1),
    "nb_direct_src": dict(BASE, split="neighbor", direct_src=1),
    "nb_inc_end": dict(BASE, split="neighbor", inc_every=99),
}


class _T:
    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, dtype


def geometry(mesh, case):
    c = CASES[case]
    G = int(tuple(mesh.shape)[1])
    x = _T((c["B"], 1, c["S"], c["Q"]), ttnn.bfloat16)
    q_b = _T((1, 1, c["Q"], c["Hl"] * (c["Dn"] + c["Dr"])), ttnn.bfloat8_b)
    wkv = _T((1, G * c["Hl"], c["Dn"], c["N"]), ttnn.bfloat16)
    grid = mesh.compute_with_storage_grid_size()
    return qpd.Geometry(x, q_b, wkv, G, grid, False, ttnn.ROW_MAJOR_LAYOUT)


def _dir(mesh, a, b):
    """eth_chan_directions of the one-hop a -> b."""
    e = ttnn.get_eth_forwarding_direction(qpd._node(mesh, a), qpd._node(mesh, b))
    assert e in (0, 1, 2, 3), f"no cardinal direction {a} -> {b}: {e}"
    return int(e)


def eth_placement(mesh, L):
    """{ring position p: {(d, l): row-0 core}}: each port in row 0 at the fewest NoC1 hops to its connection's
    Ethernet core (fabric_all_gather's probe + _worker_nearest_noc1, used read-only), instead of the op's centred
    row-0 block."""
    from ttnn.operations.examples.fabric_all_gather import program_descriptor_with_inline_kernels as ag

    chips, G = qpd.exchange_plan(mesh, 1, L)
    gx = int(mesh.compute_with_storage_grid_size().x)
    allowed = {(x, 0) for x in range(gx)}
    connections = {}
    for coord, ch in chips.items():
        conns = []
        for d, peer in ((FWD, ch["next"]), (BWD, ch["prev"])):
            links = ttnn.get_forwarding_link_indices(qpd._node(mesh, coord), qpd._node(mesh, peer))
            conns += [(peer, links[l]) for l in range(L)]
        connections[coord] = conns
    eth = ag.probe_ethernet_cores(mesh, connections, allowed)
    out = {}
    for coord, ch in chips.items():
        taken, ports = set(), {}
        for d, peer in ((FWD, ch["next"]), (BWD, ch["prev"])):
            links = ttnn.get_forwarding_link_indices(qpd._node(mesh, coord), qpd._node(mesh, peer))
            for l in range(L):
                exy = ag.eth_noc_coord(mesh, coord, eth[(coord, peer, links[l])])
                core = ag._worker_nearest_noc1(mesh, coord, exy, taken, allowed)
                ports[(d, l)] = core
                taken.add((core.x, core.y))
        out[ch["p"]] = ports
    return out


class Resources:
    """Global semaphores (ready on row 0, arrive[G] on the compute rect), the landing (compute rect, uint32 words,
    one shard per core) and the per-port source rings (row 0; pattern unique per chip / port (2l+d) / slot).
    placement: "centre" (the op: qpd.port_cores on every chip) or "eth" (eth_placement, per chip)."""

    def __init__(self, mesh, geo, L, placement="centre"):
        import torch

        self.mesh, self.geo, self.L = mesh, geo, L
        self.G = geo.G
        if placement == "centre":
            self.ports_of = {p: qpd.port_cores(geo.gx, L) for p in range(self.G)}
        else:
            self.ports_of = eth_placement(mesh, L)
        self.row0 = [ttnn.CoreCoord(x, 0) for x in range(geo.gx)]
        self.port_crs = qpd._rect(0, 0, geo.gx - 1, 0)
        self.compute_crs = qpd._rect(0, 1, geo.gx - 1, geo.gy - 1)
        self.sems_ready = ttnn.create_global_semaphore(mesh, self.port_crs, 0)
        self.sems_arrive = [ttnn.create_global_semaphore(mesh, self.compute_crs, 0) for _ in range(self.G)]
        self.seg_bytes = geo.hw * geo.tile_bytes
        self.seg_words = self.seg_bytes // 4
        self.land_words = geo.landing_tiles * geo.tile_bytes // 4
        C = geo.C
        self.zeros = torch.zeros((self.G * C, self.land_words), dtype=torch.int32)
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.compute_crs, (1, self.land_words), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.landing = ttnn.from_torch(
            self.zeros,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=mc,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        src = torch.zeros((self.G, geo.gx, NSRC * self.seg_words), dtype=torch.int32)
        for p in range(self.G):
            for (d, l), core in self.ports_of[p].items():
                for s in range(NSRC):
                    src[p, core.x, s * self.seg_words : (s + 1) * self.seg_words] = self.pattern(p, 2 * l + d, s)
        smc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.port_crs, (1, NSRC * self.seg_words), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.src = ttnn.from_torch(
            src.reshape(self.G * geo.gx, -1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=smc,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        ttnn.synchronize_device(mesh)

    def pattern(self, p, k, s):
        import torch

        return torch.arange(self.seg_words, dtype=torch.int32) + (p << 28) + (k << 24) + (s << 20) + 1

    def zero_landing(self):
        host = ttnn.from_torch(
            self.zeros,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
        )
        ttnn.copy_host_to_device_tensor(host, self.landing)
        ttnn.synchronize_device(self.mesh)

    @property
    def ready_addr(self):
        return int(ttnn.get_global_semaphore_address(self.sems_ready))

    @property
    def arrive_addrs(self):
        return [int(ttnn.get_global_semaphore_address(s)) for s in self.sems_arrive]


def plan(res, split):
    """Per chip p, per port (d, l): ordered entries [(t, ew or None, [head records])]; head record (h, k, inc).
    None ew = table route."""
    mesh, geo, L, G = res.mesh, res.geo, res.L, res.G
    chips, _ = qpd.exchange_plan(mesh, 1, L)
    coords = sorted(chips, key=lambda c: chips[c]["p"])
    route = chips[coords[0]]["route"]
    core_of_head = {h: k for k, hs in enumerate(geo.heads_of) for h in hs}
    sends = {}  # (p, d, l) -> list of (t, ew, heads)
    for p in range(G):
        ch = chips[coords[p]]
        for l in range(L):
            my = sorted(
                (hl for hl in range(geo.Hl) if core_of_head[p * geo.Hl + hl] % L == l),
                key=lambda hl: (core_of_head[p * geo.Hl + hl], hl),
            )
            hs = [p * geo.Hl + hl for hl in my]
            groups = []
            for h in hs:
                if groups and core_of_head[groups[-1][-1]] == core_of_head[h]:
                    groups[-1].append(h)
                else:
                    groups.append([h])
            for d in (FWD, BWD):
                sends[(p, d, l)] = []
            if split == "none":
                continue
            for t in qpd.remote_order(p, G):
                d0 = route[(p, t)]
                anti = G % 2 == 0 and (t - p) % G == G // 2
                if anti and split == "neighbor":
                    continue
                if anti and split == "balanced":
                    half = (len(groups) + 1) // 2
                    a = [h for g in groups[:half] for h in g]
                    b = [h for g in groups[half:] for h in g]
                    if a:
                        sends[(p, d0, l)].append((t, None, a))
                    if b:
                        d1 = 1 - d0
                        nb = ch["next"] if d1 == FWD else ch["prev"]
                        sends[(p, d1, l)].append((t, (_dir(mesh, nb, coords[t]), 2), b))
                else:
                    sends[(p, d0, l)].append((t, None, hs))
    return chips, coords, route, core_of_head, sends


def _records(geo, core_of_head, heads, t):
    """[(h, k, inc, base, stride, jump)] for heads sent as one entry to chip t (inc on the last head of a core
    group, by the group's tiles per row)."""
    out = []
    for i, h in enumerate(heads):
        k = core_of_head[h]
        last = i + 1 == len(heads) or core_of_head[heads[i + 1]] != k
        group = sum(1 for hh in heads if core_of_head[hh] == k)
        base, stride, jump = qpd.landing_slot(geo.heads_of[k], h, t, geo)
        out.append((h, k, group * geo.hw if last else 0, base, stride, jump))
    return out


def link_load(res, sends, coords, chips):
    """Bytes per directed (chip hop, direction, link): max over the ring and the totals."""
    geo, G = res.geo, res.G
    rows = geo.rows_per_head
    load = {}
    for (p, d, l), entries in sends.items():
        for t, ew, heads in entries:
            hops = (t - p) % G if d == FWD else (p - t) % G
            nbytes = len(heads) * rows * res.seg_bytes
            for i in range(hops):
                a = (p + i) % G if d == FWD else (p - i) % G
                load[(a, d, l)] = load.get((a, d, l), 0) + nbytes
    return load


def create(res, variant, payload):
    v = VARIANTS[variant]
    mesh, geo, L, G = res.mesh, res.geo, res.L, res.G
    chips, coords, route, core_of_head, sends = plan(res, v["split"])
    virt = lambda c: mesh.worker_core_from_logical_core(c)
    pkt_tiles = max(1, payload // geo.tile_bytes)
    rows = geo.rows_per_head
    compute = geo.compute_core_list
    arrive = res.arrive_addrs
    # readies: for each sending port (s, dd, l) and destination t: t's port (route(t, s), l) -> s's port (dd, l)
    ready_out = {key: [] for key in sends}
    ready_wait = {key: 0 for key in sends}
    for (s, dd, l), entries in sends.items():
        ts_ = sorted({t for t, _, _ in entries})
        ready_wait[(s, dd, l)] = len(ts_)
        for t in ts_:
            ready_out[(t, route[(t, s)], l)].append((s, dd))
    # expected per (chip t, compute core k, source s)
    expect = {}
    for (s, dd, l), entries in sends.items():
        for t, _, heads in entries:
            for h in heads:
                key = (t, core_of_head[h], s)
                expect[key] = expect.get(key, 0) + rows * geo.hw
    mesh_desc = ttnn.MeshProgramDescriptor()
    for p in range(G):
        coord = coords[p]
        ch = chips[coord]
        program = ttnn.ProgramDescriptor()
        sender_rt, reader_rt, recv_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        sig = [variant, payload, geo.Hl, geo.hw, L, str(coord)]
        for l in range(L):
            for d in (FWD, BWD):
                core = res.ports_of[p][(d, l)]
                entries = sends[(p, d, l)]
                rdy = ready_out[(p, d, l)]
                dest_keys = []
                for t, ew, _ in entries:
                    key = (t, ew)
                    if key not in dest_keys:
                        dest_keys.append(key)
                for s, _ in rdy:
                    if (s, None) not in dest_keys:
                        dest_keys.append((s, None))
                hpd = max(1, min(v["hpd_cap"], MAX_HEADERS // max(1, len(dest_keys))))
                has_conn = bool(entries) or bool(rdy)
                src_addr = int(res.src.buffer_address())
                args = [int(res.landing.buffer_address()), arrive[p], res.ready_addr, rows, geo.ts, hpd, src_addr, NSRC]
                args += [len(dest_keys)]
                for t, ew in dest_keys:
                    n = qpd._node(mesh, coords[t])
                    args += [int(n.chip_id), int(n.mesh_id), 0xFF if ew is None else ew[0], 0 if ew is None else ew[1]]
                args += [len(entries)]
                nseg = 0
                for t, ew, heads in entries:
                    recs = _records(geo, core_of_head, heads, t)
                    args += [dest_keys.index((t, ew)), len(recs)]
                    for h, k, inc, base, stride, jump in recs:
                        args += [qpd._pack(virt(compute[k])), inc, base, stride, jump]
                    nseg += len(recs) * rows
                args += [len(rdy)]
                for s, sdd in rdy:
                    rc = virt(res.ports_of[s][(sdd, l)])
                    args += [dest_keys.index((s, None)), rc.x, rc.y]
                args += [ready_wait[(p, d, l)], int(has_conn)]
                if has_conn:
                    peer = ch["next"] if d == FWD else ch["prev"]
                    link_ids = ttnn.get_forwarding_link_indices(qpd._node(mesh, coord), qpd._node(mesh, peer))
                    args += list(
                        ttnn.setup_fabric_connection(
                            qpd._node(mesh, coord), qpd._node(mesh, peer), link_ids[l], program, core
                        )
                    )
                sender_rt[core.x][core.y] = args
                reader_rt[core.x][core.y] = [src_addr, NSRC, 0 if v["direct_src"] else nseg]
                sig += [len(args), nseg, core.x]
        used = {(c.x, c.y) for c in res.ports_of[p].values()}
        for c in res.row0:
            if (c.x, c.y) not in used:
                sender_rt[c.x][c.y] = [0] * 8 + [0, 0, 0, 0, 0]  # no dests / entries / readies, no connection
                reader_rt[c.x][c.y] = [0, NSRC, 0]
        for k, c in enumerate(compute):
            rec = []
            for s in range(G):
                e = expect.get((p, k, s), 0)
                if e:
                    rec += [arrive[s], e]
            recv_rt[c.x][c.y] = [len(rec) // 2] + rec
            sig.append(len(rec))
        key = "q_stem_transport|" + "|".join(str(x) for x in sig) + "|" + repr(sorted(v.items()))
        program.custom_program_hash = int.from_bytes(hashlib.sha1(key.encode()).digest()[:8], "little") >> 1
        tile = geo.tile_bytes
        program.cbs = [
            ttnn.CBDescriptor(
                total_size=2 * v["batch"] * geo.hw * tile,
                core_ranges=res.port_crs,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB_STAGE, data_format=ttnn.bfloat16, page_size=tile)
                ],
            )
        ]
        program.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader.cpp"),
                core_ranges=res.port_crs,
                compile_time_args=[CB_STAGE, res.seg_bytes, geo.hw, v["batch"]],
                runtime_args=reader_rt,
                config=qpd._dm("RISCV_1", ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "sender.cpp"),
                core_ranges=res.port_crs,
                compile_time_args=[
                    CB_STAGE,
                    tile,
                    pkt_tiles,
                    v["batch"],
                    geo.hw,
                    MAX_HEADERS,
                    v["blocking"],
                    v["direct_src"],
                    v["inc_every"],
                    v["posted"],
                ],
                runtime_args=sender_rt,
                config=qpd._dm("RISCV_0", ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "receiver.cpp"),
                core_ranges=res.compute_crs,
                compile_time_args=[],
                runtime_args=recv_rt,
                config=qpd._dm("RISCV_0", ttnn.NOC.NOC_1),
            ),
        ]
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = program
    return mesh_desc, sends, coords, chips


def expected_landing(res, variant):
    """[chip][compute core][word] after one call (unwritten words None-masked by `mask`)."""
    import torch

    v = VARIANTS[variant]
    geo, L, G = res.geo, res.L, res.G
    chips, coords, route, core_of_head, sends = plan(res, v["split"])
    exp = torch.zeros((G, geo.C, res.land_words), dtype=torch.int32)
    mask = torch.zeros_like(exp, dtype=torch.bool)
    rows = geo.rows_per_head
    tw = geo.tile_bytes // 4
    for (p, d, l), entries in sends.items():
        kport = 2 * l + d
        c = 0
        recs = [(t, _records(geo, core_of_head, heads, t)) for t, _, heads in entries]
        for row in range(rows):
            for t, rr in recs:
                for h, k, inc, base, stride, jump in rr:
                    slot = base + row * stride + (jump if row >= geo.ts else 0)
                    w0 = slot * tw
                    exp[t, k, w0 : w0 + res.seg_words] = res.pattern(p, kport, c % NSRC)
                    mask[t, k, w0 : w0 + res.seg_words] = True
                    c += 1
    return exp, mask


def run_once(res, variant, payload):
    desc, *_ = create(res, variant, payload)
    return ttnn.generic_op([res.landing, res.src], desc)


def check(res, variant, payload):
    import torch

    res.zero_landing()
    run_once(res, variant, payload)
    ttnn.synchronize_device(res.mesh)
    got = ttnn.to_torch(res.landing, mesh_composer=ttnn.ConcatMeshToTensor(res.mesh, dim=0)).to(torch.int32)
    got = got.reshape(res.G, res.geo.C, res.land_words)
    exp, mask = expected_landing(res, variant)
    bad = (got != exp) & mask
    n = int(mask.sum())
    assert not bad.any(), f"{variant}: {int(bad.sum())} / {n} landing words wrong, first {bad.nonzero()[:6].tolist()}"
    return n * 4
