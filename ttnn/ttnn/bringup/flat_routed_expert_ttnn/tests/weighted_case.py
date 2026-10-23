# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The routed experts' fused call of deepseek_v41_flash_d_p, for the fork's model case of
ttnn.bringup.flat_routed_expert_with_program: a port of models/demos/deepseek_v41_flash_d_p/tt/weighted_combine.py
(WeightedCombine, P.8-P.11; copied, not imported: the case must survive the model changing) on the fork builders it
uses (combine_rf: combine_planner_prepare, sparse_combine_rf_prepare, l1_zero_program; the column barrier's kernel).

Per chip (r, c): the column's tokens (gathered over axis 0) and their top-k ids / bf16 weights -> the planners (counts,
regions, token index, the combine's tables) -> per call an L1 arena on every core, the combine's state zeroed, a
cross-chip barrier over the column -> ONE flat_routed_expert_with_program per pass: the flat expert over the chip's
experts plus sparse_combine_rf (axis 0, the L1 hand-off of each expert row to its token's home chip, multiplied by the
routing weight and summed there, deterministic order). Returns the combine's output [P S_pass, H] ROW_MAJOR (the home
tokens' sums over this column's experts)."""

from __future__ import annotations

from loguru import logger

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
SP_AXIS = 0
TILE = 32
LINKS = 2  # fabric links of the column's port (one port core per direction drives both: ports_per_direction)
FABRIC_PAYLOAD = 10880  # B: one 10240 B expert row + its 32 B trailer per packet
ARENA_MAX = 1356 * 1024  # the largest arena the flat op itself uses on this grid
PLANNER_CORES = 32


def _ceil(a: int, b: int) -> int:
    return -(-a // b) * b


def _l1_sharded(mesh, cores, rows, width=16, dtype=ttnn.uint32):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])
    return (
        ttnn.Shape([len(cores) * rows, width]),
        dtype,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs, [rows, width], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )


_BARRIER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
void kernel_main() {
    size_t a = 0;
    const uint32_t sem = get_arg_val<uint32_t>(a++), n_peer = get_arg_val<uint32_t>(a++);
    const uint16_t mesh = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t mx = get_arg_val<uint32_t>(a++), my = get_arg_val<uint32_t>(a++);
    uint32_t pchip[8], pconn[8];
    for (uint32_t i = 0; i < n_peer; ++i) {
        pchip[i] = get_arg_val<uint32_t>(a++);
        pconn[i] = get_arg_val<uint32_t>(a++);
    }
    cb_reserve_back(0, 1);
    const uint32_t hdr_addr = get_write_ptr(0);
    auto c0 = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
#if __NCONN__ > 1
    auto c1 = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    c1.open();
#endif
    c0.open();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_addr);
    for (uint32_t i = 0; i < n_peer; ++i) {
        hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(mx, my, sem), 1, true});
        (void)fabric_set_unicast_route(hdr, static_cast<uint16_t>(pchip[i]), mesh);
#if __NCONN__ > 1
        auto& c = pconn[i] ? c1 : c0;
#else
        auto& c = c0;
        (void)pconn;
#endif
        c.wait_for_empty_write_slot();
        c.send_payload_flush_blocking_from_address(hdr_addr, sizeof(PACKET_HEADER_TYPE));
    }
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
    while (*s < n_peer) {
        invalidate_l1_cache();
    }
    noc_semaphore_inc(get_noc_addr(mx, my, sem), static_cast<uint32_t>(0u - n_peer));
    noc_async_atomic_barrier();
    c0.close();
#if __NCONN__ > 1
    c1.close();
#endif
}
"""


def _column_barrier(mesh, core) -> tuple:
    """(MeshProgramDescriptor, semaphore) of the column barrier on logical ``core`` of every chip (built once)."""
    from ttnn.bringup.combine_rf.combine_plan import axis_ports
    from ttnn.bringup.combine_rf.sparse_all_gather_2d import mesh_plan
    from ttnn.bringup.combine_rf.sparse_combine_rf import first_hop

    R, C = tuple(mesh.shape)
    cc = ttnn.CoreCoord(*core)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc)])
    sem = ttnn.create_global_semaphore(mesh, crs, 0, ttnn.BufferType.L1_SMALL)
    sem_addr = int(ttnn.get_global_semaphore_address(sem))
    chips, _, (wrap_x, wrap_y) = mesh_plan(mesh, 1)
    chips = axis_ports(chips, 0)
    node = lambda ch: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*ch))  # noqa: E731
    v = mesh.worker_core_from_logical_core(cc)
    desc = ttnn.MeshProgramDescriptor()
    nconn = max(len(p) for p in chips.values())
    for (r, c), ports in chips.items():
        dirs = sorted(ports)  # (the chip's Y directions: a connection each)
        prog = ttnn.ProgramDescriptor()
        prog.cbs = [
            ttnn.CBDescriptor(
                total_size=128, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, 128)]
            )
        ]
        peers = [(i, c) for i in range(R) if i != r]
        args = [sem_addr, len(peers), int(node((0, 0)).mesh_id), v.x, v.y]
        for pr in peers:
            d = first_hop((r, c), pr, R, C, wrap_x, wrap_y)
            args += [int(node(pr).chip_id), dirs.index(d)]
        for d in dirs:
            args += list(
                ttnn.setup_fabric_connection(node((r, c)), node(ports[d]["peer"]), ports[d]["links"][0], prog, cc)
            )
        for _ in range(nconn - len(dirs)):  # (a chip with one direction where others have two: unused, repeated)
            d = dirs[0]
            args += list(
                ttnn.setup_fabric_connection(node((r, c)), node(ports[d]["peer"]), ports[d]["links"][0], prog, cc)
            )
        rt = ttnn.RuntimeArgs()
        rt[cc.x][cc.y] = args
        prog.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_BARRIER.replace("__NCONN__", str(nconn)),
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=crs,
                compile_time_args=[],
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
        ]
        desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = prog
    return desc, sem


class WeightedCase:
    """The model's WeightedCombine (one mesh, one chunk size: tables, planners, buffers, the combine's programs).
    ``plan``: the flat expert's plan (FlatRoutedExpert.plan)."""

    def __init__(
        self,
        mesh,
        plan,
        *,
        H: int,
        I: int,
        epc: int,
        K: int,
        S: int,
        out_dtype,
        x_bf16: bool,
        h_bf16: bool,
        arena_bytes: int | None = None,
        passes: int | None = None,
        acc_dtype=None,
        planner: str = "one",
        planner_cores: int = PLANNER_CORES,
    ):
        """arena_bytes: the per-core arena (None: the flat expert's, grown for the accumulators up to ARENA_MAX; a
        test override). passes: the pass count (None: the fewest that fit; a test override, >= that). acc_dtype: the
        combine's running sums per home token in L1 (setting COMBINE_ACC, P.9): ttnn.float32 (None: 20 KB a token,
        2x2: 2 passes of 512) or ttnn.bfloat16 (10 KB a token: the 1024 home tokens in one pass). planner (setting
        COMBINE_PLANNER, P.11): "one" = each pass's planner on one core; "multi" = the column multi-core planner on
        up to planner_cores cores per pass (disjoint per pass: the passes' planners run in one program)."""
        from ttnn.bringup.combine_rf.combine_plan import axis_ports, combine_planner_prepare, port_luts
        from ttnn.bringup.combine_rf.sparse_all_gather_2d import mesh_plan, place_ports
        from ttnn.bringup.combine_rf.sparse_combine_rf import accumulator_bytes

        R, C = tuple(mesh.shape)
        self.mesh, self.R, self.C, self.H, self.I, self.epc, self.K, self.S = mesh, R, C, H, I, epc, K, S
        self.Eg = R * epc  # the column's experts (ExpertMapping column-major: column c holds [c Eg, (c + 1) Eg))
        self.out_dtype, self.x_bf16, self.h_bf16 = out_dtype, x_bf16, h_bf16
        self.acc_dtype = ttnn.float32 if acc_dtype is None else acc_dtype
        acc_kw = dict(out_dtype=out_dtype, acc_dtype=self.acc_dtype)
        acc_name = "fp32" if self.acc_dtype == ttnn.float32 else "bf16"
        self.plan = plan
        grid = mesh.compute_with_storage_grid_size()
        self.grid = grid
        rd_key = "tensix_readers" if "tensix_readers" in plan else "readers"
        taken = {tuple(c) for k in ("arena_cores", "gu", "down", rd_key, "relays", "coords") for c in plan[k]}
        self.cells = [(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in taken]
        self.wcores = [tuple(c) for c in plan["down"]] + ([tuple(c) for c in plan["rdn"]] if plan["rdown"] else [])
        self.writers = len(self.wcores)
        assert self.writers <= 32, f"{self.writers} y writers > the L1 hand-off's 32"
        # the column's ports (Y, one core per direction driving both links) and the accumulators the cells leave
        links = LINKS if R > 1 else 1
        chips, _, _ = mesh_plan(mesh, links)
        pc = place_ports(mesh, axis_ports(chips, 0), self.cells)
        port_cells = {(v.x, v.y) for k, v in pc.items() if k[2] == 0}
        self.A = min(16, len([c for c in self.cells if c not in port_cells]))
        assert self.A >= 1, f"no accumulator cell left ({len(self.cells)} free cells, ports {sorted(port_cells)})"
        self.links = links
        # passes: the fewest whose home tokens' sums (fp32 or bf16 rows, acc_dtype) fit the accumulators in an arena
        # of at most ARENA_MAX (or the flat expert's own, if larger)
        flat_bytes = int(plan["arena_tiles"]) * 2048
        cap_bytes = max(flat_bytes, ARENA_MAX)
        P = 1
        while accumulator_bytes(R, self.A, H, -(-S // P), depth=1, **acc_kw) > cap_bytes:
            P += 1
            assert P <= S, "no pass size fits the arena"
        if P > 1:  # (L1 capacity: the accumulators one pass would need)
            a1 = next(
                (a for a in range(self.A, 65) if accumulator_bytes(R, a, H, S, depth=1, **acc_kw) <= cap_bytes),
                None,
            )
            logger.info(
                f"EXPERTS_WEIGHTED S={S}: {P} passes ({acc_name} sums: one pass needs >= {a1} accumulators of <= "
                f"{cap_bytes >> 10} KB, this grid leaves {self.A})"
            )
        if passes is not None:
            assert passes >= P, f"passes={passes}: {-(-S // passes)} home tokens a pass do not fit (fewest {P})"
            P = passes
        self.P = P
        self.S_pass = -(-S // P)
        # (rings: 4 slots per source; the local ring 8 slots, else what the arena leaves: >= 2)
        need = accumulator_bytes(R, self.A, H, self.S_pass, depth=4, local_slots=8, **acc_kw)
        arena_bytes = arena_bytes or max(flat_bytes, min(cap_bytes, _ceil(need, 2048)))
        # the arena: every role's buffers (~all of L1) on every core; per call, at one address
        self.a_rows = arena_bytes // 64
        all_cores = [(x, y) for y in range(grid.y) for x in range(grid.x)]
        self._arena_spec = _l1_sharded(mesh, all_cores, self.a_rows)
        # rows a chip's flat expert may get in a pass: every column token on this chip's experts (<= min(K, epc) each)
        self.rows_cap = _ceil(R * self.S_pass * min(K, epc) + epc * (TILE - 1), TILE)
        self.cap = max(256, _ceil(R * self.S_pass, TILE))  # rows one expert may get in a pass
        logger.info(
            f"EXPERTS_WEIGHTED S={S} passes={P} x {self.S_pass} acc={acc_name} accumulators={self.A} "
            f"port cells={sorted(port_cells)} "
            f"free cells={len(self.cells)} rows_cap={self.rows_cap} arena={arena_bytes >> 10} KB "
            f"flat arena={flat_bytes >> 10} KB acc={need >> 10} KB (ring depth 4, local ring 8) planner={planner}"
        )
        # per-pass planners (one core, column axis), outputs allocated once
        self.luts, self.nps = port_luts(mesh, links, per_direction=True, axis=0)
        Eg = self.Eg
        assert planner in ("one", "multi"), planner
        self.planner = planner
        ncore = grid.x * grid.y
        per = min(int(planner_cores), ncore // P)  # (multi: each pass's planner on its own cores)
        assert planner == "one" or per >= R, f"{P} passes x {R} planner cores do not fit the {ncore}-core grid"
        all_xy = [(i % grid.x, i // grid.x) for i in range(ncore)]
        self.planners = [
            combine_planner_prepare(
                mesh,
                S=self.S_pass,
                K=K,
                EPC=epc,
                A=self.A,
                L=links,
                port_luts=self.luts,
                n_ports=self.nps,
                rows_cap=self.rows_cap,
                axis=0,
                id_offset=lambda r, c: c * Eg,
                tokens=S,
                token_base=p * self.S_pass,
                ids_uint32=True,
                tile_in=True,  # (the gathered top-k tiles as they are: no untilize / reshape to one page)
                core=(p % grid.x, p // grid.x),  # (every pass's planner on its own core: one program, in parallel)
                cores=all_xy[p * per : (p + 1) * per] if planner == "multi" else None,
                # the deterministic combine's order table: every source chip's flat schedule (the plan's schedule
                # parameters, pin 1 and the op's default pin ratio 8), a token's rows summed in a fixed order
                det=dict(
                    rps=plan["sched_rps"], max_e=plan["sched_max_e"], gu_nreg=plan["sched_gu_nreg"], pin=1, pin_ratio=8
                ),
            )
            for p in range(P)
        ]
        if planner == "multi":
            M, CT = self.planners[0].mcc
            self.planner_desc = f"multi: {M} cores x {CT} tokens a pass"
        else:
            self.planner_desc = "one core a pass"
        self._plan_desc = {}  # (ids, wts) addresses -> the passes' planners merged into one program per chip
        # y (the flat expert's row buffer: unused with the L1 hand-off, the op checks its shape) and the combine output
        self.y = ttnn.allocate_tensor_on_device(
            ttnn.Shape([self.rows_cap, H]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, DRAM
        )
        self.out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([P * self.S_pass, H]), out_dtype, ttnn.ROW_MAJOR_LAYOUT, mesh, DRAM
        )
        self.combines = {}  # (arena, words) addresses -> per pass (run, desc, l1_args), and the zero program
        self.seen = set()  # flat_routed_expert_with_program program-cache keys seen (a hit needs no descriptor)
        self.empty = ttnn.MeshProgramDescriptor()
        self.key0 = 9_100_000 + (id(self) % 100_000) * 8
        # the column barrier after the zeroing (built once; its semaphore persistent in L1_SMALL) and its io tensor
        self.tick = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, TILE, TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, DRAM
        )
        self.barrier = _column_barrier(mesh, self.cells[-1]) if R > 1 else None

    # ------------------------------------------------------------------ per arena address
    def _combine_for(self, arena, words):
        from ttnn.bringup.combine_rf.l1_zero import l1_zero_program
        from ttnn.bringup.combine_rf.sparse_combine_rf import sparse_combine_rf_prepare

        key = (int(arena.buffer_address()), int(words.buffer_address()))
        hit = self.combines.get(key)
        if hit is not None:
            return hit
        chips = [(r, c) for r in range(self.R) for c in range(self.C)]
        runs = []
        for p, pl in enumerate(self.planners):
            run = sparse_combine_rf_prepare(
                self.mesh,
                self.S_pass,
                self.H,
                {ch: [] for ch in chips},  # (the device plan decides the rows; this only places the cores)
                num_links=self.links,
                n_acc=self.A,
                depth=4,
                cells=self.cells,
                arena=arena,
                writers=self.writers,
                l1=dict(writers=self.wcores, words=key[1]),
                devplan=dict(plan=pl.outs["plan"], table=pl.outs["table"], dord=pl.outs["dord"], top_k=self.K),
                deterministic=True,  # (fp32 sums in a fixed order per token: bit-identical calls)
                ports_per_direction=True,
                piece_max=2 * self.H,  # (one piece a row: the L1 hand-off)
                axis=0,
                out_dtype=self.out_dtype,
                out_row0=p * self.S_pass,
                zero_unseen=True,
                acc_dtype=self.acc_dtype,
            )
            assert len(run.cells["accs"]) == self.A, (run.cells, self.A)
            desc, l1_args = run.build_l1(self.y, self.out, arena)
            runs.append([run, desc, l1_args, arena])
        # zero: the combine's state (same cells and layout in every pass's program) and the writers' words
        zr = {c: [(key[0] + o, b, st, n) for o, b, st, n in rs] for c, rs in runs[0][0].zero_runs.items()}
        for c in self.wcores:
            zr.setdefault(c, []).append((key[1], 192 * 16 * 4, 0, 1))
        hit = self.combines[key] = (runs, l1_zero_program(self.mesh, zr), runs[0][0].collector)
        return hit

    # ------------------------------------------------------------------ per call
    def routing(self, idx: ttnn.Tensor, wts: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """idx [1, 1, s, K] (TILE, any integer dtype), wts [1, 1, s, K] fp32 (TILE) of the chip's mesh row -> the
        column's uint32 ids / bf16 weights [1, 1, R s, K] TILE (gathered over axis 0, home row-major; the planners
        read the tiles, tile_in). wts may already be bf16 (ROUTER_IMPL=fused, P.22: no typecast)."""
        w16 = wts if wts.dtype == ttnn.bfloat16 else ttnn.typecast(wts, ttnn.bfloat16, memory_config=DRAM)
        i32 = idx if idx.dtype == ttnn.uint32 else ttnn.typecast(idx, ttnn.uint32, memory_config=DRAM)
        out = []
        for t in (i32, w16):
            out.append(ttnn.all_gather(t, dim=2, cluster_axis=SP_AXIS, memory_config=DRAM) if self.R > 1 else t)
        for t, o in zip((i32, w16), out):
            if t is not o and t is not idx and t is not wts:  # (idx / wts: the caller's)
                ttnn.deallocate(t)
        return out[0], out[1]

    def run_planners(self, ids: ttnn.Tensor, wts: ttnn.Tensor) -> list[dict]:
        """Every pass's planner in one program per chip (each on its own core); returns each pass's tables."""
        key = (int(ids.buffer_address()), int(wts.buffer_address()))
        desc = self._plan_desc.get(key)
        if desc is None:
            progs = [pl.programs(ids, wts) for pl in self.planners]
            desc = ttnn.MeshProgramDescriptor()
            for ch in progs[0]:
                pd = ttnn.ProgramDescriptor()
                pd.cbs = [cb for pg in progs for cb in pg[ch].cbs]
                pd.kernels = [k for pg in progs for k in pg[ch].kernels]
                pd.semaphores = [sm for pg in progs for sm in pg[ch].semaphores]  # (multi: per pass, disjoint cores)
                desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*ch), ttnn.MeshCoordinate(*ch))] = pd
            self._plan_desc = {key: desc}  # (the latest addresses only)
        ttnn.generic_op([ids, wts] + [t for pl in self.planners for t in pl.tensors], desc)
        return [pl.outs for pl in self.planners]

    def gather_x(self, qx: ttnn.Tensor) -> ttnn.Tensor:
        """q(x) [1, 1, s, H] bf16 TILE of the mesh row -> the column's rows [R s, H] bf16 ROW_MAJOR."""
        xr = ttnn.to_layout(qx, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        g = ttnn.all_gather(xr, dim=2, cluster_axis=SP_AXIS, memory_config=DRAM) if self.R > 1 else xr
        if g is not xr:
            ttnn.deallocate(xr)
        return ttnn.reshape(g, [g.shape[-2], self.H])

    def __call__(self, fe, xg: ttnn.Tensor, ids: ttnn.Tensor, wts: ttnn.Tensor, fidelity: int) -> ttnn.Tensor:
        """One layer: fe = its FlatRoutedExpert (column-local expert ids), xg / ids / wts from gather_x / routing.
        Returns the combine output self.out [P S_pass, H] (out_dtype, ROW_MAJOR): the home tokens' sums."""
        mesh = self.mesh
        pls = self.run_planners(ids, wts)  # (before the arena: the planners' CBs need L1)
        # (the arena first, at the top of L1 as the flat op's own; the writers' and relays' words below it on their
        # cores: allocated first they push the arena down by their size on every core, into the fused program's static
        # circular buffers when the model holds other L1 tensors - seen on layer 14: CBs end 142464, arena at 140736)
        arena = ttnn.allocate_tensor_on_device(*self._arena_spec)
        words = ttnn.allocate_tensor_on_device(*_l1_sharded(mesh, self.wcores, 192))
        rl = [ttnn.CoreCoord(*c) for c in self.plan["relays"]]
        from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import crs_rects

        relay_words = ttnn.allocate_tensor_on_device(
            ttnn.Shape([len(rl) * TILE, TILE]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            mesh,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs_rects(rl), (TILE, TILE), ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
        runs, zero_desc, coll = self._combine_for(arena, words)
        ttnn.generic_op([arena, words], zero_desc)
        if self.R > 1:  # (the barrier: every chip of the column zeroed its arena before any combine row lands)
            ttnn.generic_op([self.tick], self.barrier[0])
        a = lambda t: 0 if t is None else int(t.buffer_address())  # noqa: E731
        for p, (run, desc, l1_args, desc_arena) in enumerate(runs):
            pl = pls[p]
            key = (
                id(fe),
                p,
                a(xg),
                a(arena),
                a(words),
                a(relay_words),
                a(pl["counts"]),
                a(pl["token_index"]),
                fidelity,
            )
            if key in self.seen:
                extra = self.empty
            else:
                if desc_arena is not arena:  # (the CBs bind the arena's buffer: rebind to this call's)
                    desc = run.rebind(arena)
                    runs[p][1], runs[p][3] = desc, arena
                extra = desc
                self.seen.add(key)
            ttnn.bringup.flat_routed_expert_with_program(
                xg,
                pl["counts"],
                pl["regions"],
                fe.gidx,
                fe.w_gu,
                fe.w_d,
                reader_down_weights=fe.w_rd,
                done_words=fe.done,
                intermediate=self.I,
                max_tokens_per_expert=fe.m,
                arena=arena,
                y=self.y,
                extra=extra,
                extra_key=self.key0 + p,
                collector=coll["core"],
                counts_addr=coll["counts_addr"],
                writers=self.writers,
                activation=fe.act,
                pin=fe.pin,
                token_index=pl["token_index"],
                down_fp32=True,
                x_bf16=self.x_bf16,
                h_bf16=self.h_bf16,
                l1_args=l1_args,
                relay_words=relay_words,
                math_fidelity=fidelity,
                weights_nk=fe.weights_nk,
            )
        for t in (arena, relay_words, words):
            ttnn.deallocate(t)
        return self.out
