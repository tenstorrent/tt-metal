# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""fabric_ring_exchange: the stats exchange of a fused CCL op on a G-chip ring, by itself.

Each chip p owns one slot of `seg_bytes` per link and wave; after one call every chip holds every peer's slot.
Variants (one program per chip, one generic_op dispatch for the mesh):

  empty    idle kernels on `idle_cores` compute cores plus the port cores: the fixed cost of a program of that
           size (launch, runtime args, teardown) with no fabric traffic.
  fence    the ports open their fabric connections, run the ready fence and send header-only items (seg_bytes is
           ignored), on the relay schedule: the minimal synchronization a correct exchange needs.
  fence_direct  the same on the direct schedule: its fence waits for a ready from every destination, including
           the 2-hop one, before the first send.
  relay    one-hop store-and-forward: the forward port sends its own slot to p+1, then relays p-1's on to p+2; the
           backward port sends its own slot to p-1.
  direct   direct multi-hop unicast: the forward port sends its own slot to p+1 and to p+2 (the second routed by the
           fabric through p+1's router, no kernel in between); the backward port sends its own slot to p-1.

The host plan, port placement, persistent counters and landing layout are the distributed_rms_norm op's
(malimpic/2026_10_22_1342_run1_distributed_rms_norm, distributed_rms_norm_program_descriptor.py), reduced to one
ring and no roots.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
VARIANTS = ("empty", "fence", "fence_direct", "relay", "direct")
FWD, BWD = 0, 1
_OPP = {FWD: BWD, BWD: FWD}
MAX_HEADERS = 8  # packet headers per port (one per destination chip)
SEG_ALIGN = 64  # landing slot alignment (Blackhole DRAM 64 B covers L1 16 B)
# program-local semaphores on the port cores; the kernel addresses them by id even when unused (n_roots = 0)
SEM_OWN_STATS, SEM_LANDED, SEM_OWN_COUNTED = 0, 1, 2


def _align_up(v, a):
    return (v + a - 1) // a * a


def _node(mesh, coord):
    return mesh.get_fabric_node_id(ttnn.MeshCoordinate(*coord))


def ring_coords(mesh):
    """The ring's chips in order: the mesh's single row (1 x G) or column (G x 1)."""
    rows, cols = tuple(mesh.shape)
    assert rows == 1 or cols == 1, f"fabric_ring_exchange needs a 1 x G or G x 1 mesh, got {rows} x {cols}"
    return [(0, c) for c in range(cols)] if rows == 1 else [(r, 0) for r in range(rows)]


def _first_hop_dir(mesh, src, dst, nxt, prv):
    """FWD / BWD: the port whose connection carries the first hop of the fabric's route src -> dst."""
    d = ttnn.get_eth_forwarding_direction(_node(mesh, src), _node(mesh, dst))
    if d is None:
        return None
    for port, nb in ((FWD, nxt), (BWD, prv)):
        if d == ttnn.get_eth_forwarding_direction(_node(mesh, src), _node(mesh, nb)):
            return port
    return None


def exchange_plan(mesh, direct):
    """Per chip: {p, next, prev, sends, entries, ready_out} (the op's _exchange_plan for one ring).
    sends[d]     items of port d in send order: (slot q, destination coord, destination entry, hops, wait); wait 0 =
                 own segment, j + 1 = relay of this port's entry j
    entries[d]   slots landing on port d, in entry order (entry i has arrival counter i)
    ready_out[d] ready packets port d sends: (target coord, target port direction, hops)"""
    grp = ring_coords(mesh)
    G = len(grp)
    assert G >= 3, "a ring of at least 3 chips"
    nxt = [grp[(p + 1) % G] for p in range(G)]
    prv = [grp[(p - 1) % G] for p in range(G)]
    info = [
        dict(
            p=p,
            next=nxt[p],
            prev=prv[p],
            sends={FWD: [], BWD: []},
            entries={FWD: [], BWD: []},
            ready_out={FWD: [], BWD: []},
        )
        for p in range(G)
    ]
    hops_dir = lambda s, t, d: (t - s) % G if d == FWD else (s - t) % G
    route = None
    if direct:
        route = {
            (s, t): _first_hop_dir(mesh, grp[s], grp[t], nxt[s], prv[s]) for s in range(G) for t in range(G) if s != t
        }
        assert None not in route.values(), "direct sends: some route leaves through neither ring neighbour"
        for t in range(G):
            for d in (FWD, BWD):
                srcs = [s for s in range(G) if s != t and route[(s, t)] == d]
                info[t]["entries"][d] = sorted(srcs, key=lambda s: hops_dir(s, t, d))
        for s in range(G):
            for d in (FWD, BWD):
                dsts = sorted((t for t in range(G) if t != s and route[(s, t)] == d), key=lambda t: hops_dir(s, t, d))
                info[s]["sends"][d] = [(s, t, info[t]["entries"][d].index(s), hops_dir(s, t, d), 0) for t in dsts]
    else:
        kf = G // 2
        kb = G - 1 - kf
        for s in range(G):
            fwd = [(s - i) % G for i in range(kf)]
            bwd = [(s + i) % G for i in range(kb)]
            for d, lst, peer in ((FWD, fwd, nxt[s]), (BWD, bwd, prv[s])):
                t = grp.index(peer)
                info[s]["sends"][d] = [(q, t, i, 1, 0 if i == 0 else i) for i, q in enumerate(lst)]
                info[t]["entries"][d] = list(lst)
    for s in range(G):
        for d in (FWD, BWD):
            for t in sorted({it[1] for it in info[s]["sends"][d]}):
                back = route[(t, s)] if route is not None else _OPP[d]
                info[t]["ready_out"][back].append((s, d, hops_dir(t, s, back)))
    chips = {}
    for s in range(G):
        for d in (FWD, BWD):
            info[s]["sends"][d] = [(q, grp[t], i, h, wt) for q, t, i, h, wt in info[s]["sends"][d]]
            info[s]["ready_out"][d] = [(grp[t], dd, h) for t, dd, h in info[s]["ready_out"][d]]
        chips[grp[s]] = info[s]
    return chips


def port_cores(grid, links):
    """2 * links port cores on logical row 0, centred (the op's placement): {(direction, link): core}."""
    n = 2 * links
    x0 = max(0, grid.x // 2 - links)
    cores = [ttnn.CoreCoord(x0 + i, 0) for i in range(n)]
    return {(d, l): cores[2 * l + d] for l in range(links) for d in (FWD, BWD)}


def idle_cores(grid, ports, n):
    """The first n grid cores in row-major order that are not port cores."""
    busy = {(c.x, c.y) for c in ports.values()}
    free = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in busy]
    assert n <= len(free), f"{n} idle cores requested, {len(free)} free"
    return free[:n]


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


class Resources:
    """Per-test persistent state: global semaphores (ready, arrival counters) on the port cores and the landing
    scratch, filled with a pattern so the exchange can be checked. Created once and reused by every call, like the
    op's persistent buffers (trace-safe: every call re-arms the counters)."""

    def __init__(self, mesh, ports, seg_bytes, num_waves):
        import torch  # host-side only (ttnn forbids a module-level torch import)

        self.mesh, self.ports = mesh, ports
        self.G = len(ring_coords(mesh))
        self.links = len(ports) // 2
        self.num_waves = num_waves
        self.seg_bytes = seg_bytes
        self.seg_stride = _align_up(max(seg_bytes, 4), SEG_ALIGN)
        crs = _crs(list(ports.values()))
        self.sems = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(1 + self.G - 1)]
        words = num_waves * self.G * self.seg_stride // 4
        self.words = words
        n_ports = len(ports)
        # landing[device][port shard][word]: chip p's own slot q = p on every one of its ports holds a pattern unique
        # to (chip, wave, port); every other slot starts at zero and must be filled by the exchange.
        host = torch.zeros((self.G, n_ports, words), dtype=torch.int32)
        for p in range(self.G):
            for k in range(n_ports):
                for w in range(num_waves):
                    base = ((w * self.G) + p) * self.seg_stride // 4
                    host[p, k, base : base + seg_bytes // 4] = self.pattern(p, w, k, seg_bytes // 4)
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs, (1, words), ttnn.ShardOrientation.ROW_MAJOR),
        )
        rows, cols = tuple(mesh.shape)
        self.scratch = ttnn.from_torch(
            host.reshape(self.G * n_ports, words),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=mc,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        ttnn.synchronize_device(mesh)  # every chip's counters are zero before any chip's first call

    @staticmethod
    def pattern(p, w, k, n):
        import torch  # host-side only (ttnn forbids a module-level torch import)

        return torch.arange(n, dtype=torch.int32) + (p << 24) + (w << 20) + (k << 16) + 1

    @property
    def ready_addr(self):
        return int(ttnn.get_global_semaphore_address(self.sems[0]))

    @property
    def arrival_addrs(self):
        return [int(ttnn.get_global_semaphore_address(s)) for s in self.sems[1:]]

    @property
    def landing_addr(self):
        return int(self.scratch.buffer_address())


def create_mesh_program_descriptor(res: Resources, variant: str, idle: list, payload: int):
    """One program per ring chip for `variant` (see the module docstring)."""
    assert variant in VARIANTS, variant
    mesh, ports = res.mesh, res.ports
    virt = lambda c: mesh.worker_core_from_logical_core(c)
    port_crs = _crs(list(ports.values()))
    send = variant != "empty"
    chips = exchange_plan(mesh, direct=variant in ("direct", "fence_direct"))
    seg = 0 if variant in ("empty", "fence", "fence_direct") else res.seg_bytes
    port_ct = [res.G, res.num_waves, payload, MAX_HEADERS, SEM_OWN_STATS, SEM_LANDED, SEM_OWN_COUNTED]
    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ch in chips.items():
        program = ttnn.ProgramDescriptor()
        # generic_op's program cache keys a kernel on its source, cores and compile-time args, and only on the
        # OUTER size of its runtime-arg grid, not on how many args each core gets: relay and direct (same kernel,
        # different per-core arg layout) would share a cached program and the second would read the first's
        # layout. Key each program on everything that changes its layout.
        key = f"fabric_ring_exchange|{variant}|{res.links}|{res.seg_bytes}|{res.num_waves}|{len(idle)}|{coord}"
        program.custom_program_hash = int.from_bytes(hashlib.sha1(key.encode()).digest()[:8], "little") >> 1
        program.semaphores = [
            ttnn.SemaphoreDescriptor(id=s, core_ranges=port_crs, initial_value=0)
            for s in (SEM_OWN_STATS, SEM_LANDED, SEM_OWN_COUNTED)
        ]
        kernels = []
        if idle:
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "idle.cpp"),
                    core_ranges=_crs(idle),
                    compile_time_args=[],
                    runtime_args=ttnn.RuntimeArgs(),
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
                    ),
                )
            )
        if not send:
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "idle.cpp"),
                    core_ranges=port_crs,
                    compile_time_args=[],
                    runtime_args=ttnn.RuntimeArgs(),
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
                    ),
                )
            )
        else:
            port_rt = ttnn.RuntimeArgs()
            peer_of = {FWD: ch["next"], BWD: ch["prev"]}
            for l in range(res.links):
                for d in (FWD, BWD):
                    core = ports[(d, l)]
                    sends, ready_out, entries = ch["sends"][d], ch["ready_out"][d], ch["entries"][d]
                    has_conn = len(sends) > 0 or len(ready_out) > 0
                    pc = virt(core)
                    dests = sorted({(dst, hops) for _, dst, _, hops, _ in sends} | {(t, h) for t, _, h in ready_out})
                    assert len(dests) <= MAX_HEADERS
                    args = [
                        res.landing_addr,
                        res.seg_stride,
                        res.ready_addr,
                        len(dests),
                        len(sends),
                        len(entries),
                        0,  # n_roots: the own segment is in the landing already
                        len(ready_out),
                        len({it[1] for it in sends}),  # readies to wait for: one per destination chip
                        pc.x,
                        pc.y,
                        0,  # n_notify
                        int(has_conn),
                        0,  # n_counted
                        1,  # notify quantum
                        SEM_LANDED,
                    ]
                    args += res.arrival_addrs[: len(entries)]
                    for dst, hops in dests:
                        dn = _node(mesh, dst)
                        args += [int(dn.chip_id), int(dn.mesh_id), hops]
                    for q, dst, idx, hops, wait in sends:
                        args += [q, dests.index((dst, hops)), res.arrival_addrs[idx], wait]
                    for tgt, td, hops in ready_out:
                        rc = virt(ports[(td, l)])
                        args += [dests.index((tgt, hops)), rc.x, rc.y]
                    args += [seg] * res.num_waves
                    if has_conn:
                        pn = _node(mesh, peer_of[d])
                        link_ids = ttnn.get_forwarding_link_indices(_node(mesh, coord), pn)
                        args += list(ttnn.setup_fabric_connection(_node(mesh, coord), pn, link_ids[l], program, core))
                    port_rt[core.x][core.y] = args
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "port.cpp"),
                    core_ranges=port_crs,
                    compile_time_args=port_ct,
                    runtime_args=port_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
                    ),
                )
            )
        program.kernels = kernels
        r, c = coord
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return mesh_desc


def ring_exchange(res: Resources, variant: str, idle: list, payload: int | None = None):
    """One call: dispatch `variant` on every ring chip (one generic_op)."""
    payload = int(payload or ttnn.get_tt_fabric_max_payload_size_bytes())
    return ttnn.generic_op([res.scratch], create_mesh_program_descriptor(res, variant, idle, payload))


def expected_landing(res: Resources, variant: str):
    """[device][port][word] the scratch must hold after a relay / direct call: own slots unchanged, every peer slot
    filled with the peer's pattern on the port it lands on (per link: the same direction for every link)."""
    import torch  # host-side only (ttnn forbids a module-level torch import)

    chips = exchange_plan(res.mesh, direct=(variant == "direct"))
    n_ports = 2 * res.links
    exp = torch.zeros((res.G, n_ports, res.words), dtype=torch.int32)
    order = ring_coords(res.mesh)
    for coord, ch in chips.items():
        p = ch["p"]
        for l in range(res.links):
            for d in (FWD, BWD):
                k = 2 * l + d
                for w in range(res.num_waves):
                    n = res.seg_bytes // 4
                    own = ((w * res.G) + p) * res.seg_stride // 4
                    exp[p, k, own : own + n] = Resources.pattern(p, w, k, n)
                    for q in ch["entries"][d]:
                        base = ((w * res.G) + q) * res.seg_stride // 4
                        # the slot was sent by chip q's port of the same direction and link (same core index k)
                        exp[p, k, base : base + n] = Resources.pattern(q, w, k, n)
    assert [c for c in chips] == order
    return exp
