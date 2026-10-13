# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_all_gather — an all-gather of rows that moves each row only as far as the chips that need it (prototype).

The output is the dense all-gather's: chip q's row t at slot q * S + t, on every chip. A row lands in a chip's slot
when that chip needs it, when it only passes through on the way to a chip further on, or both; slots no row reaches
keep whatever they held (a consumer reads only the rows it needs). This is dispatch with deduplication: a row goes
once to a chip however many of its experts are there, and once along the ring however many chips beyond need it.

Routing, per row: the chips that need it are a set on the ring; the row goes `a` hops forward and `b` back, the
smallest a + b that covers the set (ties: the less loaded direction), on one link (rows alternate links per
direction, by load). Every port core (one per direction and link, as fabric_all_gather) sends a sequence of rows:
first its chip's own rows that travel its way on its link, then, origin by origin outward, the rows its upstream sent
it that have hops left. A relayed row is read from this chip's output once the upstream port's arrival counter says it
has landed (the counter rises once per INC_EVERY packets and on the last); the sender writes each row one fabric hop
into the same slot of the next chip's output. One row per packet (the row size is chosen to be one packet).

Prototype scope: the plan (each port's sequence and, per relayed row, how far upstream's counter must have risen) is
computed on the host from every row's need mask and uploaded as a table per chip; the kernels read their slice of it.
The production form computes the same table on every chip from the gathered top-k ids. Ring groups only (one mesh axis
or a snake), placement / fence / fabric connections from fabric_all_gather_py.plan.
"""

import math

import torch

import ttnn

from .fabric_all_gather_py import _OPP, _needs_ready, NOC0, NOC1, plan as ag_plan

CB_ROWS = 0
CB_LIST = 1
INC_EVERY = 8

_READER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Port reader: loads this port's row list, then reads each row (own: from the input; relayed: from my output once my
// arrival counter covers it) into the row CB, `group` rows per barrier.
void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t list_cb = get_compile_time_arg_val(1);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t group = get_compile_time_arg_val(3);
    constexpr auto in_args = TensorAccessorArgs<4>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto plan_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    size_t a = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t plan_addr = get_arg_val<uint32_t>(a++);
    const uint32_t plan_page = get_arg_val<uint32_t>(a++);   // this port's list: [rows n][page x n][wait x n]
    const uint32_t list_bytes = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t own_base = get_arg_val<uint32_t>(a++);    // my rows' first output slot
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);
    const uint32_t plan_page_bytes = get_arg_val<uint32_t>(a++);
    const auto in = TensorAccessor(in_args, in_addr, row_bytes);
    const auto out = TensorAccessor(out_args, out_addr, row_bytes);
    const auto plan = TensorAccessor(plan_args, plan_addr, plan_page_bytes);
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);

    cb_reserve_back(list_cb, 1);
    const uint32_t list_addr = get_write_ptr(list_cb);
    if (list_bytes > 0) {
        noc_async_read(plan.get_noc_addr(plan_page), list_addr, list_bytes);
        noc_async_read_barrier();
    }
    cb_push_back(list_cb, 1);  // the sender reads the same list
    volatile tt_l1_ptr uint32_t* pages = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(list_addr);
    volatile tt_l1_ptr uint32_t* waits = pages + n;

    uint32_t batch = 0, wptr = 0;
    for (uint32_t j = 0; j < n; ++j) {
        if (batch == 0) {
            cb_reserve_back(cb, group);
            wptr = get_write_ptr(cb);
        }
        const uint32_t page = pages[j];
        const uint32_t w = waits[j];
        uint64_t src;
        if (w == 0) {
            src = in.get_noc_addr(page - own_base);
        } else {
            noc_semaphore_wait_min(arrived, w);
            src = out.get_noc_addr(page);
        }
        noc_async_read(src, wptr + batch * row_bytes, row_bytes);
        if (++batch == group || j + 1 == n) {
            noc_async_read_barrier();
            cb_push_back(cb, batch);
            batch = 0;
        }
    }
}
"""

_SENDER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

using namespace tt::tt_fabric;

inline void route_one_hop(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)) || defined(EMULE_FABRIC_2D)
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
    (void)chip_id;
    (void)mesh_id;
    (void)fabric_set_unicast_route<false>(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(hdr), static_cast<uint16_t>(1));
#endif
}

// Port sender: the fence (as fabric_all_gather), then every row of the list one hop into the same slot of the next
// chip's output, every INC_EVERY-th packet and the last a fused write + increment of the downstream port's counter;
// then wait for everything upstream sends me and re-arm my counter.
void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t list_cb = get_compile_time_arg_val(1);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t group = get_compile_time_arg_val(3);
    constexpr uint32_t inc_every = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();

    size_t a = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);
    const uint32_t expect_in = get_arg_val<uint32_t>(a++);  // increments upstream sends me
    const uint32_t n = get_arg_val<uint32_t>(a++);          // rows I send
    const uint32_t ready_addr = get_arg_val<uint32_t>(a++);
    const uint32_t send_ready = get_arg_val<uint32_t>(a++);
    const uint32_t ready_x = get_arg_val<uint32_t>(a++);
    const uint32_t ready_y = get_arg_val<uint32_t>(a++);
    const uint32_t peer_x = get_arg_val<uint32_t>(a++);
    const uint32_t peer_y = get_arg_val<uint32_t>(a++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t connected = get_arg_val<uint32_t>(a++);
    const auto out = TensorAccessor(out_args, out_addr, row_bytes);

    cb_wait_front(list_cb, 1);
    volatile tt_l1_ptr uint32_t* pages = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(list_cb));

    if (connected) {
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[group];
        for (uint32_t h = 0; h < group; ++h) {
            hdrs[h] = PacketHeaderPool::allocate_header();
            route_one_hop(hdrs[h], dst_chip_id, dst_mesh_id);
        }
        conn.open();
        if (send_ready) {
            hdrs[0]->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader{get_noc_addr(ready_x, ready_y, ready_addr), 1, true});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
        }
        {
            volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
            noc_semaphore_wait_min(ready, 1);
            noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
            noc_async_atomic_barrier();
        }
        const uint64_t arrival_noc = get_noc_addr(peer_x, peer_y, arrival_addr);
        uint32_t unflushed = 0;
        for (uint32_t j = 0; j < n; ++j) {
            cb_wait_front(cb, unflushed + 1);
            const uint32_t src = get_read_ptr(cb) + unflushed * row_bytes;
            const uint64_t dst = out.get_noc_addr(pages[j], 0, 0);
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[unflushed];
            if ((j + 1) % inc_every == 0 || j + 1 == n) {
                hdr->to_noc_fused_unicast_write_atomic_inc(
                    NocUnicastAtomicIncFusedCommandHeader{dst, arrival_noc, 1, true}, row_bytes);
            } else {
                hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, row_bytes);
            }
            conn.wait_for_empty_write_slot();
            conn.send_current_slot_non_blocking(src, row_bytes, reinterpret_cast<uint32_t>(hdr));
            if (++unflushed == group || j + 1 == n) {
                noc_async_writes_flushed();
                cb_pop_front(cb, unflushed);
                unflushed = 0;
            }
        }
        conn.close();
    }
    cb_pop_front(list_cb, 1);
    if (expect_in > 0) {
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
        noc_semaphore_wait_min(arrived, expect_in);
        noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - expect_in);
        noc_async_atomic_barrier();
    }
}
"""

_COPY = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Copy core: rows [first, first + count) of my own shard into my own output slots (no fabric), `group` per barrier.
void kernel_main() {
    constexpr uint32_t row_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t group = get_compile_time_arg_val(1);
    constexpr uint32_t cb = get_compile_time_arg_val(2);
    constexpr auto in_args = TensorAccessorArgs<3>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    size_t a = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t first = get_arg_val<uint32_t>(a++);
    const uint32_t count = get_arg_val<uint32_t>(a++);
    const uint32_t stride = get_arg_val<uint32_t>(a++);
    const uint32_t own_base = get_arg_val<uint32_t>(a++);
    const auto in = TensorAccessor(in_args, in_addr, row_bytes);
    const auto out = TensorAccessor(out_args, out_addr, row_bytes);
    cb_reserve_back(cb, group);
    const uint32_t buf = get_write_ptr(cb);
    for (uint32_t i = 0; i < count; i += group) {
        const uint32_t k = count - i < group ? count - i : group;
        for (uint32_t j = 0; j < k; ++j) {
            noc_async_read(in.get_noc_addr(first + (i + j) * stride), buf + j * row_bytes, row_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t j = 0; j < k; ++j) {
            noc_async_write(buf + j * row_bytes, out.get_noc_addr(own_base + first + (i + j) * stride), row_bytes);
        }
        noc_async_write_barrier();
    }
}
"""


def route_rows(need, *, num_links, ring=True):
    """Each row's path. need: bool [G, S, G] (origin position, row, receiving position: positions along the ring).

    Returns (fwd_hops, bwd_hops, fwd_link, bwd_link, load): each of the first four [G, S] int, the row going fwd_hops
    toward p+1 and bwd_hops toward p-1 (0 = not that way) on the given link; load [2, L, G] = rows per (direction,
    link, hop leaving position e). Minimal a + b covering every needing chip; among the minima and the link pairs, the
    one whose busiest hop ends least loaded, then the least total load on its hops (greedy over rows, origins
    interleaved): 1121 rows on the busiest link for a dense 8-ring at 2 links, against the 1120 of a balanced one."""
    G, S, _ = need.shape
    nd = need.tolist()
    fh = [[0] * S for _ in range(G)]
    bh = [[0] * S for _ in range(G)]
    fl = [[0] * S for _ in range(G)]
    bl = [[0] * S for _ in range(G)]
    load = [[[0] * G for _ in range(num_links)] for _ in range(2)]

    for t in range(S):
        for o in range(G):
            dist = [(q - o) % G for q in range(G) if q != o and nd[o][t][q]]
            if not dist:
                continue
            if ring:
                options = [(a + max([G - d for d in dist if d > a], default=0), a) for a in range(G)]
                cbest = min(c for c, _ in options)
                cands = [(a, cbest - a) for c, a in options if c == cbest]
            else:
                cands = [
                    (
                        max([d for d in dist if o + d < G], default=0),
                        max([G - d for d in dist if o + d >= G], default=0),
                    )
                ]
            # every minimal (a, b) and every link pair: the busiest hop it would make, then the total load on its hops
            best = None
            for a, b in cands:
                fe = [(o + i) % G for i in range(a)]
                be = [(o - i) % G for i in range(b)]
                for lf in range(num_links) if a else (0,):
                    for lb in range(num_links) if b else (0,):
                        used = [load[0][lf][e] for e in fe] + [load[1][lb][e] for e in be]
                        score = (max(used) + 1, sum(used))
                        if best is None or score < best[0]:
                            best = (score, a, b, lf, lb, fe, be)
            _, a, b, lf, lb, fe, be = best
            fh[o][t], bh[o][t], fl[o][t], bl[o][t] = a, b, lf, lb
            for e in fe:
                load[0][lf][e] += 1
            for e in be:
                load[1][lb][e] += 1
    T = lambda v: torch.tensor(v, dtype=torch.int64)
    return T(fh), T(bh), T(fl), T(bl), torch.tensor(load, dtype=torch.float64)


def build_sequences(fh, bh, fl, bl, num_links):
    """Per port (position p, direction d, link l): the rows it sends, as (origin, row), own first, then origin by
    origin outward; and for each, its index in the upstream port's sequence (-1 for own rows)."""
    G, S = fh.shape
    seq = {}
    for p in range(G):
        for d, hops, lnk, step in (("fwd", fh, fl, 1), ("bwd", bh, bl, -1)):
            for l in range(num_links):
                rows = []
                for k in range(G):  # k = hops already travelled to reach p
                    o = (p - step * k) % G
                    for t in range(S):
                        if lnk[o, t] == l and hops[o, t] > k:
                            rows.append((o, t))
                seq[(p, d, l)] = rows
    pos = {key: {r: i for i, r in enumerate(rows)} for key, rows in seq.items()}
    up_index = {}
    for (p, d, l), rows in seq.items():
        up = (p - 1) % G if d == "fwd" else (p + 1) % G
        idx = []
        for o, t in rows:
            idx.append(-1 if o == p else pos[(up, d, l)][(o, t)])
        up_index[(p, d, l)] = idx
    return seq, up_index


def sparse_all_gather(
    input_tensor,
    need,
    *,
    cluster_axis=0,
    num_links=2,
    output=None,
    group=4,
    inc_every=INC_EVERY,
    routes=None,
):
    """input_tensor: [1, 1, S, W] ROW_MAJOR, DRAM interleaved, one shard per chip of each ring group (a mesh axis).
    need: bool [G, S, G], need[o, t, r] = the chip with output rank r needs origin rank o's row t (ring = rank order).
    Returns the output [1, 1, G * S, W]: every needed row at its slot; relayed-through rows land too."""
    mesh_device = input_tensor.device()
    chips, ring = ag_plan(mesh_device, cluster_axis=cluster_axis, topology=ttnn.Topology.Ring, num_links=num_links)
    G = next(iter(chips.values()))["G"]
    S = int(input_tensor.shape[-2])
    W = int(input_tensor.shape[-1])
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT, "rows are pages: ROW_MAJOR only"
    row_bytes = int(input_tensor.buffer_aligned_page_size())
    assert row_bytes <= ttnn.get_tt_fabric_max_payload_size_bytes(), "one row per packet"
    # positions along the ring = output ranks (one ring per group, in group order)
    for ch in chips.values():
        assert ch["rings"][0]["p"] == ch["out"], "prototype: ring order must be the output order"
    if routes is None:
        routes = route_rows(need, num_links=num_links, ring=ring)
    fh, bh, fl, bl, _ = routes
    seq, up_index = build_sequences(fh, bh, fl, bl, num_links)

    if output is None:
        output = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, G * S, W]),
            input_tensor.dtype,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
    # the plan table: per chip, one 64-byte aligned page per port: [page x n][wait x n]
    ports = [(d, l) for d in ("fwd", "bwd") for l in range(num_links)]
    max_words = max(2 * len(rows) for rows in seq.values())
    page_words = max(16, -(-max_words // 16) * 16)
    rows_, cols_ = tuple(mesh_device.shape)
    table = torch.zeros(rows_, cols_, len(ports), page_words, dtype=torch.int32)
    meta = {}
    for coord, ch in chips.items():
        p = ch["out"]
        for i, (d, l) in enumerate(ports):
            rows = seq[(p, d, l)]
            n = len(rows)
            pages = [o * S + t for o, t in rows]
            waits = [0 if u < 0 else u // inc_every + 1 for u in up_index[(p, d, l)]]
            table[coord[0], coord[1], i, :n] = torch.tensor(pages, dtype=torch.int32) if n else table[0, 0, 0, :0]
            table[coord[0], coord[1], i, n : 2 * n] = (
                torch.tensor(waits, dtype=torch.int32) if n else table[0, 0, 0, :0]
            )
            up = (p - 1) % G if d == "fwd" else (p + 1) % G
            meta[(coord, d, l)] = dict(n=n, i=i, expect=-(-len(seq[(up, d, l)]) // inc_every))
    plan_t = ttnn.from_torch(
        table.reshape(rows_, cols_, len(ports), page_words),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows_, cols_), dims=(0, 1)),
    )

    from .fabric_all_gather_py import _SEM_CACHE

    port_cores = {(c.x, c.y) for ch in chips.values() for c in ch["ports"].values()}
    key = (id(mesh_device), tuple(sorted(port_cores)), "sparse")
    if key not in _SEM_CACHE:
        cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sorted(port_cores)]
        )
        sems = tuple(ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(2))
        ttnn.synchronize_device(mesh_device)
        _SEM_CACHE[key] = (mesh_device, sems)
    arrival_addr, ready_addr = (int(ttnn.get_global_semaphore_address(x)) for x in _SEM_CACHE[key][1])

    in_ct = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    out_ct = list(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    plan_ct = list(ttnn.TensorAccessorArgs(plan_t).get_compile_time_args())
    in_addr, out_addr, plan_addr = (int(t.buffer_address()) for t in (input_tensor, output, plan_t))
    virt = lambda c: mesh_device.worker_core_from_logical_core(c)
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
        processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
    )
    src = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    list_bytes_max = page_words * 4

    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ch in chips.items():
        p = ch["out"]
        program = ttnn.ProgramDescriptor()
        port_list = list(ch["ports"].items())
        port_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for _, c in port_list])
        copy_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in ch["copy"]])
        program.cbs = [
            ttnn.CBDescriptor(
                total_size=2 * group * row_bytes,
                core_ranges=port_set.merge(copy_set),
                format_descriptors=[ttnn.CBFormatDescriptor(CB_ROWS, input_tensor.dtype, row_bytes)],
            ),
            ttnn.CBDescriptor(
                total_size=list_bytes_max,
                core_ranges=port_set,
                format_descriptors=[ttnn.CBFormatDescriptor(CB_LIST, ttnn.uint32, list_bytes_max)],
            ),
        ]
        reader_rt, sender_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        rg = ch["rings"][0]
        for (j, d, l), core in port_list:
            m = meta[(coord, d, l)]
            n = m["n"]
            reader_rt[core.x][core.y] = [
                in_addr,
                out_addr,
                plan_addr,
                m["i"],
                min(list_bytes_max, -(-(2 * n * 4) // 64) * 64),
                n,
                p * S,
                arrival_addr,
                list_bytes_max,
            ]
            peer = rg["next"] if d == "fwd" else rg["prev"]
            send_ready = _needs_ready(chips, coord, j, d)
            rc = virt(chips[peer]["ports"][(j, _OPP[d], l)]) if send_ready else None
            args = [out_addr, arrival_addr, m["expect"], n, ready_addr]
            args += [1, rc.x, rc.y] if send_ready else [0, 0, 0]
            if rg.get(f"{d}_links"):
                pc = virt(chips[peer]["ports"][(j, d, l)])
                pn = node(peer)
                args += [pc.x, pc.y, int(pn.mesh_id), int(pn.chip_id), 1]
                args += list(ttnn.setup_fabric_connection(node(coord), pn, rg[f"{d}_links"][l], program, core))
            else:
                assert n == 0
                args += [0, 0, 0, 0, 0]
            sender_rt[core.x][core.y] = args
        copy_rt = ttnn.RuntimeArgs()
        J = len(ch["copy"])
        for jj, core in enumerate(ch["copy"]):
            copy_rt[core.x][core.y] = [in_addr, out_addr, jj, len(range(jj, S, J)), J, p * S]
        program.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_READER,
                source_type=src,
                core_ranges=port_set,
                compile_time_args=[CB_ROWS, CB_LIST, row_bytes, group] + in_ct + out_ct + plan_ct,
                runtime_args=reader_rt,
                config=dm("RISCV_1", NOC0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_SENDER,
                source_type=src,
                core_ranges=port_set,
                compile_time_args=[CB_ROWS, CB_LIST, row_bytes, group, inc_every] + out_ct,
                runtime_args=sender_rt,
                config=dm("RISCV_0", NOC1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_COPY,
                source_type=src,
                core_ranges=copy_set,
                compile_time_args=[row_bytes, group, CB_ROWS] + in_ct + out_ct,
                runtime_args=copy_rt,
                config=dm("RISCV_1", NOC0),
            ),
        ]
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = program
    out = ttnn.generic_op([input_tensor, plan_t, output], mesh_desc)
    return out, plan_t


def landed(fh, bh):
    """bool [G, S, G]: which output slots a routing writes (the chips each row reaches, its origin included)."""
    G, S = fh.shape
    reach = torch.zeros(G, S, G, dtype=torch.bool)
    for o in range(G):
        reach[o, :, o] = True
        for i in range(1, G):
            reach[o, :, (o + i) % G] |= fh[o] >= i
            reach[o, :, (o - i) % G] |= bh[o] >= i
    return reach


def link_rows(load):
    """Rows on the busiest link direction of a routing (route_rows' load)."""
    return float(load.max())
