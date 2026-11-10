# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host side of the CCL blocks (kernels/rs/*, kernels/ag/*, kernels/fabric/*) on a 1D ring: worker roles, fabric
connections or program-local fabric muxes, route info, chunking and the global semaphores, as the
reduce_scatter_minimal_async / all_gather_async ring factories compute them.

Workers are laid out per (link, direction): `workers[(link, dir)]` is the list of that slot's worker cores. With one
worker per slot each worker connects to the fabric directly; with more, `muxes[(link, dir)]` hosts a fabric mux that
the slot's workers send through (the factories' num_workers_per_link > 1 configuration)."""

from __future__ import annotations

from dataclasses import dataclass

import ttnn

from .compose import Block, LegoProgram
from .mm import KDIR, div_up, tile_bytes

TOPOLOGY_RING = 2  # tt::tt_fabric::Topology::Ring


def route_words(ring_size: int) -> list[int]:
    """ccl_routing_utils route info CT words for a 1D ring: fwd unicast {mesh 0, 1 hop}, fwd multicast
    {start 1 hop, range ring-1, e w n s}, then the same backward."""
    uni = [0, 1]
    mc = [1, ring_size - 1, 0, 0, 0, 0]
    return uni + mc + uni + mc


def _xy(device, core):
    c = device.worker_core_from_logical_core(ttnn.CoreCoord(*core))
    return [c.x, c.y]


def _node(mesh, coord):
    return mesh.get_fabric_node_id(ttnn.MeshCoordinate(*coord))


def fabric_block(mesh, prog: LegoProgram, core, coord, dst_coord, link: int, forward: bool) -> list[int]:
    """FabricConnectionManager::build_from_args block for one direction: [1, eth, teardown, bufidx, 0] (forward) or
    [0, 1, eth, teardown, bufidx] (backward). The two connection semaphores are program semaphores on `core`."""
    scratch = ttnn.ProgramDescriptor()
    words = list(
        ttnn.setup_fabric_connection(
            mesh, _node(mesh, coord), _node(mesh, dst_coord), link, scratch, ttnn.CoreCoord(*core)
        )
    )
    assert len(words) == 3, words
    conn = [words[0], prog.sem([core], 0), prog.sem([core], 0)]
    return [1, *conn, 0] if forward else [0, 1, *conn]


class GlobalSems:
    """Global semaphores of one CCL instance (same L1 address on every core and device), kept across calls."""

    def __init__(self, mesh, n: int, cores=None):
        g = mesh.compute_with_storage_grid_size()
        crs = cores or ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
        self.sems = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(n)]
        ttnn.synchronize_device(mesh)

    def addr(self, i: int) -> int:
        return ttnn.get_global_semaphore_address(self.sems[i])


@dataclass
class RingInfo:
    ring_size: int
    index: int
    coord: tuple
    fwd: tuple  # coord of index+1
    bwd: tuple  # coord of index-1


def ring_of(mesh, coord) -> RingInfo:
    rows, cols = tuple(mesh.shape)
    assert rows == 1, "1D ring along the mesh columns"
    j = coord[1]
    return RingInfo(cols, j, coord, (0, (j + 1) % cols), (0, (j - 1) % cols))


@dataclass
class Mux:
    """A program-local fabric mux placed on one core (kernels/fabric/mux.hpp) and how its clients connect."""

    core: tuple
    cfg: object
    client_ct: dict
    term_sem: int  # termination-sync semaphore id, the same on every client of this mux
    clients: list = None  # every client core, in channel order (clients[0] is the termination master)

    def client_rt(self, mesh, prog: LegoProgram, client, index: int, master) -> list[int]:
        fm = ttnn.experimental.fabric_mux
        scratch = ttnn.ProgramDescriptor()
        w = list(
            fm.client_runtime_args(
                connection_valid=True,
                is_termination_master=index == 0,
                channel_type=fm.ChannelType.FULL_SIZE,
                mux_virtual_core=ttnn.CoreCoord(*_xy(mesh, self.core)),
                client_index=index,
                client_logical_core=ttnn.CoreCoord(*client),
                config=self.cfg,
                program_descriptor=scratch,
                termination_master_virtual_core=ttnn.CoreCoord(*_xy(mesh, master)),
                termination_master_semaphore_id=0,
            )
        )
        assert len(w) == 17, w
        w[10] = self.term_sem
        for k in range(11, 15):
            w[k] = prog.sem([client], 0)
        return w


def place_mux(
    prog: LegoProgram, mesh, coord, dst_coord, link: int, core, clients: list, *, num_buffers=2, label="mux"
) -> Mux:
    """FabricMuxConfig in program-local L1 (CB + l1-base semaphore on the mux core), the mux block on RISCV_0 / NOC_0
    (as the CCL factories place it), and the clients' CT values. num_buffers: per client channel."""
    fm = ttnn.experimental.fabric_mux
    cfg = fm.Config(
        mesh_device=mesh,
        num_full_size_channels=len(clients),
        num_header_only_channels=0,
        num_buffers_per_full_size_channel=num_buffers,
        num_buffers_per_header_only_channel=0,
        full_size_channel_buffer_size_bytes=fm.channel_buffer_size_bytes(mesh),
        base_l1_address=0,
    )
    map_size = cfg.memory_map_end_address()
    l1_sem = prog.sem([core], 0)
    cb = prog.cb([core], 1, map_size, ttnn.uint32)
    ct = list(cfg.kernel_compile_time_args())
    ct[18], ct[19], ct[20] = 1, cb, l1_sem  # in_program_local_l1, its CB, its l1-base semaphore
    scratch = ttnn.ProgramDescriptor()
    rt = list(
        cfg.kernel_runtime_args(
            mesh_device=mesh,
            source_node_id=_node(mesh, coord),
            destination_node_id=_node(mesh, dst_coord),
            link_index=link,
            program_descriptor=scratch,
            mux_logical_core=ttnn.CoreCoord(*core),
        )
    )
    rt[-2], rt[-1] = prog.sem([core], 0), prog.sem([core], 0)  # fabric connection teardown / buffer-index sems
    prog.set_noc([core], "dm0", ttnn.NOC.NOC_0)
    prog.place(
        [core],
        Block(
            cls="lego::fabric::Mux",
            header=f"{KDIR}/fabric/mux.hpp",
            risc="dm0",
            rt={core: rt},
            tensors={"mux": ct},
            label=label,
        ),
    )
    cct = list(fm.client_compile_time_args(num_clients=len(clients), channel_type=fm.ChannelType.FULL_SIZE, config=cfg))
    client_ct = dict(
        use_mux=True,
        mux_num_buffers=cct[0],
        mux_buffer_size=cct[1],
        mux_l1_base_sem=l1_sem,
        mux_termination_offset=cct[3],
        num_mux_clients=cct[4],
    )
    term = prog.sem(clients, 0)
    return Mux(core, cfg, client_ct, term, list(clients))


def _slots(workers: dict):
    """[(link, dir, w, core)] in (link, dir, w) order."""
    return [(l, d, w, c) for (l, d) in sorted(workers) for w, c in enumerate(workers[(l, d)])]


def _connections(prog, mesh, ring, workers, muxes, dst_of, forward_of, label):
    """Per worker: its connection RT words and its connection CT values. muxes[(link, dir)] is either a core (a mux
    is placed there for that slot's workers) or (Mux, first client index) to join an existing mux as extra clients."""
    conn, wct = {}, {}
    for (l, d), ws in sorted(workers.items()):
        dst = dst_of(d)
        if muxes:
            m = muxes[(l, d)]
            if isinstance(m, tuple) and isinstance(m[0], Mux):
                mux, first = m
            else:
                mux, first = place_mux(prog, mesh, ring.coord, dst, l, m, ws, label=f"{label}_mux"), 0
            for w, c in enumerate(ws):
                conn[c] = mux.client_rt(mesh, prog, c, first + w, mux.clients[0])
                wct[c] = mux.client_ct
        else:
            assert len(ws) == 1, "more than one worker per (link, direction) needs a mux"
            conn[ws[0]] = fabric_block(mesh, prog, ws[0], ring.coord, dst, l, forward_of(d))
            wct[ws[0]] = dict(use_mux=False)
    return conn, wct


def place_reduce_scatter(
    prog: LegoProgram,
    mesh,
    ring: RingInfo,
    workers: dict,
    *,
    input: ttnn.Tensor,
    interm: ttnn.Tensor,
    output: ttnn.Tensor,
    sems: GlobalSems,
    muxes: dict | None = None,
    residual: ttnn.Tensor | None = None,
    reader_risc: str = "dm1",
    writer_risc: str = "dm0",
    hw_startup: bool = True,
    chunks_per_sync: int | None = None,
    label: str = "rs",
    nocs=(ttnn.NOC.NOC_0, ttnn.NOC.NOC_1),
    fp32_dest_acc_en: bool = False,
    dst_full_sync_en: bool = False,
    dual: tuple | None = None,
    signal_own: tuple | None = None,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    batches: int = 1,
    batch_wait: tuple | None = None,
):
    """Ring reduce-scatter along dim 3 of [1,1,S,W] (W split ring_size ways), optionally adding `residual`
    ([1,1,S,W/ring_size]) in the last step. workers[(link, dir)]: dir 1 sends to index+1, dir 0 to index-1.
    `sems` = 4 global semaphores (sem0, sem1, ack, barrier). Returns the worker cores.
    dual = (tensor [1,1,S,ring*W/ring], column tile offset): also write the result there (e.g. this device's slot of a
    following all-gather's output). signal_own = (counters, targets, slot): once written, each worker increments
    counter word 16*slot on every target (batched: targets is a list per batch).
    batches > 1: the rows are that many batches, reduced one ring traversal each (so a producer's next batch overlaps
    this one); batch_wait = (counters on the worker cores, expected): a traversal starts once counter word 16*b
    reaches expected."""
    links = sorted({l for l, _ in workers})
    nw = len(workers[(links[0], 0)])
    slots = _slots(workers)
    cores = [c for *_, c in slots]
    S, W = input.padded_shape[-2], input.padded_shape[-1]
    page = tile_bytes(input.dtype)
    payload = ttnn.get_tt_fabric_max_payload_size_bytes()
    pages_per_packet = payload // page
    tiles_per_packet = min(4, pages_per_packet)
    # DST tiles available per acquire (the factory's max_dst_size): 8 in half-sync bf16; fp32 halves it, full sync doubles
    max_dst = (4 if fp32_dest_acc_en else 8) * (2 if dst_full_sync_en else 1)
    max_dst = min(max_dst, 8)
    tile_granularity = min(4 * pages_per_packet, max_dst)
    Ht, Wt = S // 32 // batches, W // 32  # per batch
    slice_Wt = Wt // ring.ring_size
    in_pages = Ht * Wt
    out_pages = in_pages // ring.ring_size
    num_workers = len(links) * nw

    cbn = 2 * tile_granularity
    df = input.dtype
    cb = {name: prog.cb(cores, cbn, page, df) for name in ("input", "interm", "interm2", "reader_out", "compute_out")}
    fuse_residual = residual is not None
    cb_resid = prog.cb(cores, cbn, page, df) if fuse_residual else cb["interm2"]

    shape_ct = dict(
        my_chip_id=ring.index,
        ring_size=ring.ring_size,
        tile_granularity=tile_granularity,
        page_size=page,
        input_batch_num_pages=in_pages,
        output_batch_num_pages=out_pages,
        input_channel_num_pages=in_pages,
        output_channel_num_pages=out_pages,
        input_tensor_Wt=Wt,
        slice_C=1,
        slice_Ht=Ht,
        slice_Wt=slice_Wt,
        dim=3,
        traversals=batches,
    )
    common = [
        input.buffer_address(),
        interm.buffer_address(),
        output.buffer_address(),
        0,
        sems.addr(3),
        sems.addr(0),
        sems.addr(1),
        sems.addr(2),
    ]
    info = {c: (l, d, w) for l, d, w, c in slots}

    def split(core):
        l, d, w = info[core]
        wid = l * nw + w
        start = wid * out_pages // num_workers
        end = (wid + 1) * out_pages // num_workers
        cps_step = div_up(end - start, tile_granularity)
        if chunks_per_sync is not None:
            cps = chunks_per_sync
        else:
            default = min(80, max(cps_step // 2, 1))
            cps = 1 if cps_step <= 4 else min(default, 4)
        return start, end, start % slice_Wt, (start // slice_Wt) * Wt, cps

    def opposite(core):
        l, d, w = info[core]
        return workers[(l, 1 - d)][w]

    prog.set_noc(cores, reader_risc, nocs[0])
    prog.set_noc(cores, writer_risc, nocs[1])
    prog.place(
        cores,
        Block(
            cls="lego::rs::Reader",
            header=f"{KDIR}/rs/reader.hpp",
            risc=reader_risc,
            ct=dict(
                shape_ct,
                cb_input_id=cb["input"],
                cb_interm_id=cb["interm"],
                cb_interm2_id=cb["interm2"],
                cb_reader_output_id=cb["reader_out"],
                fuse_residual=fuse_residual,
                cb_resid=cb_resid,
                batch_wait=batch_wait is not None,
            ),
            rt=lambda c: (lambda s: [info[c][1], s[4], s[0], s[1], s[2], s[3], 0, batches])(split(c))
            + ([batch_wait[0].buffer_address(), batch_wait[1]] if batch_wait else []),
            crt=common + ([residual.buffer_address()] if fuse_residual else []),
            tensors={
                "input": input,
                "interm": interm,
                "output": output,
                "resid": residual if fuse_residual else output,
            },
            label=f"{label}_reader",
        ),
    )
    conn, wct = _connections(
        prog, mesh, ring, workers, muxes, lambda d: ring.fwd if d == 1 else ring.bwd, lambda d: d == 1, label
    )
    hook_ct = dict(
        dual_output=dual is not None,
        signal_own=signal_own is not None,
        out_Wt=slice_Wt,
        dual_Wt=(dual[0].padded_shape[-1] // 32) if dual else 0,
        dual_col_offset=dual[1] if dual else 0,
        own_slot=signal_own[2] if signal_own else 0,
    )
    own_rt = []
    if signal_own:
        per_batch = signal_own[1] if batches > 1 else [signal_own[1]]
        own_rt = [signal_own[0].buffer_address()]
        for targets in per_batch:
            own_rt += [len(targets)] + [v for t in targets for v in _xy(mesh, t)]
    hook_rt = ([dual[0].buffer_address()] if dual else []) + own_rt
    for c in cores:
        prog.place(
            [c],
            Block(
                cls="lego::rs::Writer",
                header=f"{KDIR}/rs/writer.hpp",
                risc=writer_risc,
                ct=dict(
                    shape_ct,
                    cb_compute_output_id=cb["compute_out"],
                    cb_reader_output_id=cb["reader_out"],
                    num_tiles_to_write_per_packet=tiles_per_packet,
                    **wct[c],
                    **hook_ct,
                ),
                rt=lambda c, s=split(c): _xy(mesh, c)
                + _xy(mesh, opposite(c))
                + [1, info[c][1], s[4], s[2], s[3], s[0], s[1], 0, batches]
                + conn[c]
                + hook_rt,
                crt=common,
                tensors={
                    "route": route_words(ring.ring_size),
                    "interm": interm,
                    "output": output,
                    "dual": dual[0] if dual else output,
                },
                label=f"{label}_writer",
            ),
        )
    prog.place(
        cores,
        Block(
            cls="lego::rs::Reduction",
            header=f"{KDIR}/rs/reduction.hpp",
            risc="compute",
            ct=dict(
                cb_input_id=cb["input"],
                cb_interm_id=cb["interm"],
                cb_interm2_id=cb["interm2"],
                cb_compute_output_id=cb["compute_out"],
                tile_granularity=tile_granularity,
                ring_size=ring.ring_size,
                fuse_residual=fuse_residual,
                cb_resid=cb_resid,
                hw_startup=hw_startup,
                traversals=batches,
                slice_C=1,
            ),
            rt=lambda c: (lambda s: [s[0], s[1], info[c][1], 0, batches])(split(c)),
            compute=dict(
                math_fidelity=math_fidelity,
                fp32_dest_acc_en=fp32_dest_acc_en,
                math_approx_mode=False,
                dst_full_sync_en=dst_full_sync_en,
            ),
            label=f"{label}_reduce",
        ),
    )
    return cores


def place_all_gather(
    prog: LegoProgram,
    mesh,
    ring: RingInfo,
    workers: dict,
    *,
    input: ttnn.Tensor,
    output: ttnn.Tensor,
    sems: GlobalSems,
    muxes: dict | None = None,
    chunks_per_sync: int = 10,
    reader_risc: str = "dm1",
    writer_risc: str = "dm0",
    label: str = "ag",
    nocs=(ttnn.NOC.NOC_0, ttnn.NOC.NOC_1),
    signal: tuple | None = None,
    write_local: bool = True,
    row_group_tiles: int = 0,
):
    """Ring all-gather along dim 3: [1,1,S,W] per device -> [1,1,S,ring_size*W]. workers[(link, dir)]: dir 0
    sends to index+1, dir 1 to index-1. `sems` = 3 global semaphores (sem0, sem1, barrier). Returns the workers.
    signal = (counters tensor, target cores): each worker increments counter word 16*slice on every target as its
    part of a slice lands (the dir-1 workers also for the local slice they write). Counts per slice: local
    links*nw, each ring neighbour links*nw, the others 2*links*nw. With row_group_tiles, target i is signalled per
    slice once this worker's tiles in [i*row_group_tiles, (i+1)*row_group_tiles) have landed; ag_row_group_waits()
    gives the resulting counts."""
    links = sorted({l for l, _ in workers})
    nw = len(workers[(links[0], 0)])
    slots = _slots(workers)
    cores = [c for *_, c in slots]
    S, W = input.padded_shape[-2], input.padded_shape[-1]
    page = tile_bytes(input.dtype)
    payload = ttnn.get_tt_fabric_max_payload_size_bytes()
    tiles_per_packet = min(4, payload // page)
    cb_out = prog.cb(cores, 3 * tiles_per_packet, page, input.dtype)
    rs = ring.ring_size
    n_fwd = div_up(rs - 1, 2)
    n_bwd = rs - 1 - n_fwd
    common = [input.buffer_address(), output.buffer_address(), sems.addr(2), sems.addr(0), sems.addr(1), S // 32]
    ct = dict(
        ring_size=rs,
        my_chip_id=ring.index,
        cb_output_id=cb_out,
        num_tiles_to_write_per_packet=tiles_per_packet,
        page_size=page,
        num_targets_forward_direction=n_fwd,
        num_targets_backward_direction=n_bwd,
        topology=TOPOLOGY_RING,
        gather_dim=3,
        input_batch_head_count=1,
        input_tensor_Wt=W // 32,
        input_tensor_C=1,
        output_tensor_Wt=output.padded_shape[-1] // 32,
        output_tensor_C=1,
        reverse=False,
    )
    info = {c: (l, d, w) for l, d, w, c in slots}
    count = len(links) * nw

    def wid(c):
        l, d, w = info[c]
        return l * nw + w

    def opposite(core):
        l, d, w = info[core]
        return workers[(l, 1 - d)][w]

    if signal is not None:
        counters, targets = signal
        # targets[i] may be a list of cores: every one of them is signalled for row group i
        groups = [t if isinstance(t, list) else [t] for t in targets]
        copies = len(groups[0])
        assert all(len(g) == copies for g in groups)
        sig_rt = [counters.buffer_address(), len(groups)] + [v for g in groups for t in g for v in _xy(mesh, t)]
    else:
        sig_rt, copies = [], 1
    ct["signal_copies"] = copies
    ct["signal_slices"] = signal is not None
    ct["write_local"] = write_local
    ct["row_group_tiles"] = row_group_tiles
    prog.set_noc(cores, reader_risc, nocs[0])
    prog.set_noc(cores, writer_risc, nocs[1])
    prog.place(
        cores,
        Block(
            cls="lego::ag::Reader",
            header=f"{KDIR}/ag/reader.hpp",
            risc=reader_risc,
            ct=dict(ct),
            rt=lambda c: [info[c][1], wid(c), count, chunks_per_sync] + sig_rt,
            crt=common,
            tensors={"input": input, "output": output},
            label=f"{label}_reader",
        ),
    )
    conn, wct = _connections(
        prog, mesh, ring, workers, muxes, lambda d: ring.fwd if d == 0 else ring.bwd, lambda d: d == 0, label
    )
    for c in cores:
        prog.place(
            [c],
            Block(
                cls="lego::ag::Writer",
                header=f"{KDIR}/ag/writer.hpp",
                risc=writer_risc,
                ct=dict(ct, barrier_target_count=rs - 1, **wct[c]),
                rt=lambda c: _xy(mesh, c)
                + [1]
                + _xy(mesh, opposite(c))
                + [info[c][1], wid(c), count, chunks_per_sync]
                + conn[c]
                + sig_rt,
                crt=common,
                tensors={"route": route_words(rs), "output": output},
                label=f"{label}_writer",
            ),
        )
    return cores


def ccl_layout(origin=(0, 0), num_links=2, nw=1, grid_x=12, mux=None):
    """Worker (and mux) cores for one CCL, laid out like choose_worker_cores: per link, per direction, [mux,]
    workers, filling rows from `origin`. Returns (workers, muxes or None)."""
    use_mux = nw > 1 if mux is None else mux
    x, y = origin

    def nxt():
        nonlocal x, y
        c = (x, y)
        x += 1
        if x >= grid_x:
            x, y = 0, y + 1
        return c

    workers, muxes = {}, {}
    for l in range(num_links):
        for d in range(2):
            if use_mux:
                muxes[(l, d)] = nxt()
            workers[(l, d)] = [nxt() for _ in range(nw)]
    return workers, (muxes or None)


def ag_row_group_waits(S: int, slice_W: int, num_links: int, nw: int, group_tiles: int, ring_size: int = 4):
    """For place_all_gather(row_group_tiles=group_tiles): per row group, the number of signals it gets for a ring
    neighbour's slice and for the farther slices (split forwarding on even rings: halves of each worker range)."""
    pages = (S // 32) * (slice_W // 32)
    n = num_links * nw
    ranges = [(i * (pages // n) + min(i, pages % n), (i + 1) * (pages // n) + min(i + 1, pages % n)) for i in range(n)]
    groups = -(-pages // group_tiles)

    def count(portions):
        out = []
        for g in range(groups):
            a, b = g * group_tiles, min((g + 1) * group_tiles, pages)
            out.append(sum(1 for (p0, p1) in portions if p0 < b and p1 > a and p1 > p0))
        return out

    near = count(ranges)
    split = ring_size % 2 == 0 and ring_size > 2
    far = (
        count([(s, s + (e - s) // 2) for s, e in ranges] + [(s + (e - s) // 2, e) for s, e in ranges])
        if split
        else [2 * v for v in near]
    )
    return near, far
