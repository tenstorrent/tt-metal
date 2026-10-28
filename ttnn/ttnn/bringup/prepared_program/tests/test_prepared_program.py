# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.prepare_program / run_prepared on every program a model runs through them (cases.py, F201): the
program built as the model builds it at the case's shapes on random inputs, then run (1) by the plain ttnn.generic_op
on the io tensors it was built for and (2) prepared, on copies of the "move" tensors at other addresses: every output
bit-identical (the prepared contract: only the io tensors' addresses change). Then the determinism check (section 4b of
the skill): 5 prepared runs on A with one on B (other inputs: the model's next layer / call) between, bit-identical.

The programs come from Marko Staletovic's MoE block (models/demos/moe_block, as deepseek_v41_flash_d_p's
tt/moe_block_adapter.py builds it: fused head, fused tail, the exchange) on random weights and inputs, two forwards
(two layers: layers mode, as the model's): the test records each prepare_program / run_prepared call and replays the
case's program. Routed experts: uninitialised weights ("fake": no program here reads them).

Then the outputs of A (the generic_op run) and of B (a prepared run) against a CPU reference of the call's inputs:
the fused head (_check_head) vs float64 - shared rows, router picks and weights, xbuf own rows, route masks, the
planner's tables; the dispatch (_check_dispatch) vs its host model, bit for bit."""

import importlib.util
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.common.bringup.testing import determinism
from models.demos.common.bringup.testing.fork_capture import program_summary

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_prepared_program_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CASES = _load("cases").CASES

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="the MoE block's programs are Blackhole-only")


def _device_params(c):
    p = dict(c["device_params"])
    p["fabric_config"] = getattr(ttnn.FabricConfig, p["fabric_config"])
    if "fabric_payload_bytes" in p:
        rc = ttnn._ttnn.fabric.FabricRouterConfig()
        rc.max_packet_payload_size_bytes = int(p.pop("fabric_payload_bytes"))
        p["fabric_router_config"] = rc
    p["require_exact_physical_num_devices"] = True
    return p


def _block(mesh, c, rec_wrap):
    """The MoE block as tt/moe_block_adapter.py builds it, random weights, two layers sharing the routed op."""
    from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

    from models.demos.moe_block.moe_block import (
        ExpertActivation,
        Geometry,
        MoEBlock,
        MoEDtypes,
        MoELayer,
        MoEModel,
        MoEOptions,
        MoEWeights,
        RoutedExperts,
        RouterConfig,
        WeightCache,
        fuse_gate_up,
    )

    E, H, I = c["E"], c["H"], c["I"]
    m = MoEModel(
        num_experts=E,
        top_k=c["K"],
        emb_dim=H,
        routed_expert_hidden_dim=I,
        routed_expert_activation=ExpertActivation("clamped_silu", limit=c["limit"]),
        router=RouterConfig(score_func="sqrtsoftplus", route_scale=c["route_scale"]),
        shared_expert_hidden_dim=I,
        shared_expert_activation=ExpertActivation("clamped_silu", limit=c["limit"]),
    )
    g = Geometry.of(mesh, m, c["chunk"], c["links"], zigzag=False)
    op = FlatRoutedExpert(
        mesh,
        "fake",
        m=RoutedExperts.cap_of(g),
        H=H,
        I=I,
        gids=RoutedExperts.gids_of(g),
        n_global=E,
        wdtype="bf4",
        act="clamped_silu",
        pin=1,
        weights_nk=True,
        layout=dict(c["layout"]),
    )
    dtypes = MoEDtypes(shared_fidelity=getattr(ttnn.MathFidelity, c["shared_fidelity"]))

    def host(seed):
        def make():
            gw = torch.Generator().manual_seed(seed)
            r = lambda *s, sc=c["w_std"]: torch.randn(*s, generator=gw) * sc  # noqa: E731
            b = r(E, sc=0.01)
            return MoEWeights(
                router={"weight": r(E, H).to(torch.bfloat16), "bias": (b - b.mean()).float()},
                routed=None,
                shared={"gate_up": fuse_gate_up(r(I, H), r(I, H)), "down": r(H, I)},
                latent=None,
            )

        return make

    hosts = [host(c["seed"] * 10 + k) for k in range(2)]
    layers = [MoELayer.build(mesh, g, m, h, dtypes, routed_op=op, layout=dict(c["layout"])) for h in hosts]
    rec_wrap()  # (from here on: the block's own warm-ups too)
    blk = MoEBlock(
        mesh,
        m,
        c["chunk"],
        layers[0],
        WeightCache(path=None, layer=0),
        num_links=c["links"],
        dtypes=dtypes,
        options=MoEOptions(head="fused", tail="fused", combine_snap=c["combine_snap"]),
    )
    return blk, layers, hosts


def _copy_to(t, dst):
    ttnn.copy_host_to_device_tensor(ttnn.from_device(t), dst)


def _moved(t):
    """t's data in a new tensor at another address (allocated while t lives)."""
    n = ttnn.allocate_tensor_on_device(ttnn.Shape(list(t.shape)), t.dtype, t.layout, t.device(), t.memory_config())
    _copy_to(t, n)
    assert int(n.buffer_address()) != int(t.buffer_address())
    return n


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in CASES],
    ids=[c["id"] for c in CASES],
    indirect=["mesh_device", "device_params"],
)
def test_prepared_program(mesh_device, device_params, case, monkeypatch):
    c = case
    R, C = c["mesh"]
    preps, runs, digest_of = [], [], {}
    pre = []  # (the case's program: host copies of its "restore" tensors before each of the model's runs)
    orig_prep, orig_run = ttnn.bringup.prepare_program, ttnn.bringup.run_prepared

    def prep(desc, io):
        p = orig_prep(desc, io)
        d = program_summary(desc)["digest"]
        digest_of[id(p)] = d
        preps.append((d, desc, list(io), p))
        return p

    def run(p, io):
        d = digest_of.get(id(p))
        if d == c["digest"]:
            pre.append(
                (
                    list(io),
                    [ttnn.from_device(io[i]) for i in c["restore"]],
                    [[ttnn.to_torch(t) for t in ttnn.get_device_tensors(io[i])] for i in c["restore"]],
                )
            )
        runs.append((d, p, list(io)))
        return orig_run(p, io)

    def rec_wrap():
        monkeypatch.setattr(ttnn.bringup, "prepare_program", prep)
        monkeypatch.setattr(ttnn.bringup, "run_prepared", run)

    t0 = time.time()
    blk, layers, hosts = _block(mesh_device, c, rec_wrap)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(2, 3))
    gx = torch.Generator().manual_seed(c["seed"] + 100)
    xs = [
        ttnn.from_torch(
            torch.randn(1, 1, c["chunk"], c["H"], generator=gx).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=mapper,
        )
        for _ in range(3)
    ]
    for k in range(3):  # (layers 0, 1, 0: a cached program's first call is the plain generic_op, then prepared)
        ttnn.deallocate(blk.forward(xs[k], layer=layers[k % 2]))
    ttnn.synchronize_device(mesh_device)
    monkeypatch.setattr(ttnn.bringup, "prepare_program", orig_prep)
    monkeypatch.setattr(ttnn.bringup, "run_prepared", orig_run)
    logger.info(
        f"{c['id']}: block built and run 3 times in {time.time() - t0:.0f} s; programs prepared "
        f"{sorted({d for d, *_ in preps})}, prepared runs {len(runs)}"
    )
    mine = [x for x in preps if x[0] == c["digest"]]
    assert mine, f"the model's program {c['digest']} was not built (prepared: {sorted({d for d, *_ in preps})})"
    _, desc, io_p, _ = mine[-1]
    assert len(io_p) == c["n_io"], len(io_p)
    my_runs = [io for d, _, io in runs if d == c["digest"]]
    alive = [i for i in range(len(io_p)) if all(io[i].is_allocated() for io in my_runs)]
    model_moves = sorted(i for i in alive if len({int(io[i].buffer_address()) for io in my_runs}) > 1)
    logger.info(f"{c['id']}: {len(my_runs)} model runs; io positions the model moved between them: {model_moves}")
    head = blk.fused.head if c["program"] == "fused_head" else None
    if head is not None:  # (the head gives its arena back after each call: take it again, at its address)
        addr = int(io_p[3].buffer_address()) if io_p[3].is_allocated() else None
        head.eng.realloc_arena()
        if head.router is not None:
            head.router.arena = head.eng.arena
        assert addr is None or addr == int(head.eng.arena.buffer_address())
        io_p[3] = head.eng.arena
        head._rebind()  # (its CBs onto the arena taken again: the descriptor the head runs now)
        desc = head.state["desc"]
        assert program_summary(desc)["digest"] == c["digest"]
    assert all(t.is_allocated() for t in io_p), [i for i, t in enumerate(io_p) if not t.is_allocated()]
    # A: the model's last run of this prepared program on the tensors it was prepared with; B: another of its runs (the
    # other layer's weights, or other data in the same tensors); each with its "restore" tensors as that run found them
    same = lambda io: all(io[i] is io_p[i] for i in c["move"] if i != 3)  # noqa: E731
    a_i = max(i for i, (io, *_) in enumerate(pre) if same(io))
    b_i = max((i for i, (io, *_) in enumerate(pre) if not same(io)), default=None)
    b_i = b_i if b_i is not None else max(i for i in range(len(pre)) if i != a_i)
    st_a, st_b = pre[a_i][1], pre[b_i][1]
    io_b = [pre[b_i][0][i] if i in c["move"] else io_p[i] for i in range(len(io_p))]
    assert all(t.is_allocated() for t in io_b)
    logger.info(f"{c['id']}: A = the model's prepared run {a_i}, B = run {b_i} (of {len(pre)})")

    def restore(io, st):
        for i, h in zip(c["restore"], st):
            ttnn.copy_host_to_device_tensor(h, io[i])

    outs = lambda io: [io[i] for i in c["compare"]]  # noqa: E731

    # (1) the plain generic op on the tensors the program was built for, after one prepared run (the same state before
    # both: whatever the program leaves outside its outputs)
    p = ttnn.bringup.prepare_program(desc, io_p)
    restore(io_p, st_a)
    ttnn.bringup.run_prepared(p, io_p)
    restore(io_p, st_a)
    ttnn.generic_op(io_p, desc)
    want = determinism.snapshot(outs(io_p))
    assert any(bool((t != 0).any()) for _, t in want), f"{c['id']}: every output is zero"

    def reference(io, run_i, label):
        """The outputs vs the torch reference of the call's inputs (the run's x / route / xbuf as it found them)."""
        if head is not None:
            k = next(k for k in range(2) if layers[k].heads[id(blk.fused)][0] is io[1])
            _check_head(c, head, io, pre[run_i][2][0], hosts[k](), label)
        else:
            _check_dispatch(c, io, pre[run_i][2], label)

    reference(io_p, a_i, "A (generic_op)")
    # (2) prepared, the "move" tensors at other addresses
    io_m = [_moved(t) if i in c["move"] else t for i, t in enumerate(io_p)]

    def run_a():
        restore(io_m, st_a)
        ttnn.bringup.run_prepared(p, io_m)
        return outs(io_m)

    got = run_a()
    err = determinism.diff(want, determinism.snapshot(got))
    logger.info(f"{c['id']}: prepared with io {c['move']} moved vs generic_op: {err or 'bit-identical'}")
    assert err is None, f"prepared run (io {c['move']} moved) != generic_op: {err}"

    def run_b():
        restore(io_b, st_b)
        ttnn.bringup.run_prepared(p, io_b)
        return outs(io_b)

    determinism.assert_deterministic(run_a, run_b, first=got, label=c["id"])
    run_b()
    reference(io_b, b_i, "B (prepared)")
    if head is not None:
        head.free_arena()


def _rows(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (a.norm() * b.norm()))


def _check_head(c, head, io, x_dev, w, label):
    """The fused head's outputs vs float64 on the inputs the device held (x bf16, the router's bf16 W_r and fp32 bias,
    the shared expert's fp32 weights - stored bfp8), what Marko's test_router_sqrtsoftplus / test_head_se3
    (models/demos/moe_block/tests/test_fused_head_parts.py) compare, with F139's limits (reference.py; no number read
    off the device's own error):
    - y, the shared expert's owner rows (chip (r, c): rows [c n, (c + 1) n) of mesh row r's shared expert, its TP slices
      summed): rel L2 <= noise_limit's (calib x the formats run - bfp8 weights, the fidelity's operand bits, h bf16,
      fp32 sums, bf16 out - vs float64, over the mesh row); PCC a diagnostic;
    - the joint routing rows (chip (r, c): its ring position p's n tokens): the picks equal float64's top 6 of score +
      bias on every token whose 6th / 7th keys are apart by more than calib x twice the formats run's largest key error
      (fp32 scores, the key's 15 mantissa bits); the share of all tokens equal is a diagnostic; the weights (the picks'
      scores / their sum x 1.5, bf16) of the tokens with float64's picks: rel L2 <= noise_limit's;
    - xbuf's own slots (chip d: rows [d n, (d + 1) n)) = x's rows of p, bit for bit;
    - the route words' destination masks (bits 0-15) = the chips holding the device's picks (expert e on chip e // epc),
      the own chip excluded (the order / link bits are the dispatch tree's choice: not modelled);
    - the planner's tables (counts, regions, token index, table, plan) = combine_plan.plan_reference on the device's own
      routing rows, bit for bit."""
    from ttnn.bringup.combine_rf.combine_plan import plan_reference

    from models.demos.common.bringup.testing.fork_cases import CALIB, noise_limit
    from models.demos.moe_block.moe_block import split_gate_up

    ref = _load("reference")
    R, C = c["mesh"]
    N, H, E, K, lim, scale = R * C, c["H"], c["E"], c["K"], c["limit"], c["route_scale"]
    n = c["chunk"] // N  # (tokens a chip owns; mesh row r's c["chunk"] / R rows, column-split into C slices)
    epc = E // N
    ns, nr = c["noise"]["shared"], c["noise"]["router"]
    gate, up = split_gate_up(w.shared["gate_up"].float())
    se_w = [gate.T.contiguous(), up.T.contiguous(), w.shared["down"].float().T.contiguous()]  # (the engine's [K, N])
    wr, br = w.router["weight"].float(), w.router["bias"].float()
    ys = [t.double().reshape(n, H) for t in _rows(io[2])]
    jr = [t.to(torch.int64).reshape(n, 2 * K) & 0xFFFF for t in _rows(io[4])]
    xb = [t.reshape(N * n, H) for t in _rows(io[5])]
    rw = [t.to(torch.int64).reshape(-1) & 0xFFFFFFFF for t in _rows(io[6])]
    fn_se = lambda x, g, u, d: ref.shared_expert(x, g, u, d, lim, ns["fidelity"])  # noqa: E731
    out = {}
    for r in range(R):
        xf = torch.cat([x_dev[r * C + q].reshape(-1, H // C) for q in range(C)], dim=1)  # [c["chunk"] / R, H] bf16
        se_l = noise_limit(fn_se, [xf.float()] + se_w, ns["inputs"], ns["out"])
        want = fn_se(xf.double(), *(t.double() for t in se_w))
        for cc in range(C):
            d = r * C + cc
            p = head.xg.chips[(r, cc)]["rings"][0]["p"]
            wd = want[cc * n : (cc + 1) * n]
            o = dict(se_rel=float((ys[d] - wd).norm() / wd.norm()), se_lim=se_l["rel"], se_pcc=_pcc(ys[d], wd))
            # the router on my ring position's tokens
            xo = xf[p * n : (p + 1) * n]
            key64 = ref.topk_keys(ref.router_scores(xo.double(), wr.double()), br.double(), nr["key_bits"])
            keyf = ref.topk_keys(ref.router_scores(xo.float(), wr), br, nr["key_bits"])
            srt, ids = torch.sort(key64, dim=-1, descending=True)
            gap = srt[:, K - 1] - srt[:, K]
            o["gap"] = CALIB * 2 * float((keyf.double() - key64).abs().max())
            dids, wb = jr[d][:, :K], jr[d][:, K:]
            dw = torch.where(wb >= 32768, wb - 65536, wb).to(torch.int16).view(torch.bfloat16).double()
            same = (torch.sort(dids, 1)[0] == torch.sort(ids[:, :K], 1)[0]).all(1)
            clear = gap > o["gap"]
            o.update(
                clear=int(clear.sum()), clear_differ=int((~same & clear).sum()), picks_all=same.float().mean().item()
            )
            sid = torch.sort(ids[same, :K], 1)[0]  # (float64's picks, by id: the device's of the same tokens, by id)
            fn_w = lambda x, w_, i=sid: ref.route_weights(x, w_, i, scale)  # noqa: E731
            w_l = noise_limit(fn_w, [xo[same].float(), wr], nr["inputs"], nr["out"])
            ww = fn_w(xo[same].double(), wr.double())
            gw = torch.gather(dw[same], 1, torch.sort(dids[same], 1)[1])
            o.update(w_rel=float((gw - ww).norm() / ww.norm()), w_lim=w_l["rel"])
            out[d] = o
            bad = (xb[d][d * n : (d + 1) * n] != xo).any(1).nonzero().flatten()
            assert bad.numel() == 0, f"{label} chip {d} xbuf own rows: {bad.numel()} differ, first {bad[:8].tolist()}"
            mask = torch.zeros(n, dtype=torch.int64)
            for k in range(K):
                mask |= 1 << (dids[:, k] // epc)
            mask &= ~(1 << d)
            bad = ((rw[d][:n] & 0xFFFF) != mask).nonzero().flatten()
            assert bad.numel() == 0, f"{label} chip {d} route masks: {bad.numel()} differ, first {bad[:8].tolist()}"
    rt = head.routing
    ids_all = torch.stack([j[:, :K] for j in jr])
    wb_all = torch.stack([j[:, K:] for j in jr])
    tabs = {k: _rows(v) for k, v in rt.outs.items()}
    for d in range(N):
        ch = (d // C, d % C)
        ref = plan_reference(
            ids_all,
            wb_all,
            d,
            R,
            C,
            n,
            K,
            rt.EPC,
            N * rt.EPC,
            rt.A,
            rt.L,
            rt.nps[ch],
            rt.luts[ch],
            rt.rows_cap,
            wrap=(rt.wrap_x, rt.wrap_y),
            token_major=True,
        )
        for k in ("counts", "regions", "token_index", "table", "plan"):
            got = tabs[k][d].to(torch.int64).reshape(-1) & 0xFFFFFFFF
            exp = torch.tensor(ref[k], dtype=torch.int64)
            if k in ("token_index", "table"):  # (the valid words: the plan's row count)
                nw = min(ref["plan"][600], rt.rows_cap) * (2 if k == "table" else 1)
                got, exp = got[:nw], exp[:nw]
            bad = (got[: exp.numel()] != exp).nonzero().flatten()
            assert (
                bad.numel() == 0
            ), f"{label} chip {d} planner {k}: {bad.numel()} words differ, first {bad[:8].tolist()}"
    for d, o in out.items():
        logger.info(
            f"{c['id']} {label} chip {d}: shared rows rel {o['se_rel']:.3e} (limit {o['se_lim']:.3e}) pcc "
            f"{o['se_pcc']:.6f}; picks: {o['clear']} tokens clear of the {o['gap']:.2e} gap, {o['clear_differ']} differ, "
            f"{100 * o['picks_all']:.2f}% of all equal (a diagnostic); weights rel {o['w_rel']:.3e} (limit {o['w_lim']:.3e})"
        )
    logger.info(f"{c['id']} {label}: xbuf own rows, route masks, planner tables bit-exact")
    for d, o in out.items():
        assert o["clear_differ"] == 0, f"{label} chip {d}: picks differ on {o['clear_differ']} tokens clear of noise"
        assert o["se_rel"] <= o["se_lim"], f"{label} chip {d}: shared rows rel {o['se_rel']:.3e} > {o['se_lim']:.3e}"
        assert o["w_rel"] <= o["w_lim"], f"{label} chip {d}: weights rel {o['w_rel']:.3e} > {o['w_lim']:.3e}"


def _touched(R, C, s_, mask, order):
    """The chips a row of chip s_ lands on in its dimension-order tree on lines (sparse_all_gather_2d): order 0 (X first)
    along its mesh row over the columns from its own to every destination column, then down each such column over the
    rows from its own to that column's destinations; order 1 (Y first) the same with rows and columns swapped. A chip
    on the way keeps the row in its slot too (it relays from there)."""
    r0, c0 = divmod(s_, C)
    dest = [(i // C, i % C) for i in range(R * C) if (mask >> i) & 1]
    out = set()
    if order == 0:
        cols = {c for _, c in dest}
        for c in range(min(cols | {c0}), max(cols | {c0}) + 1):
            out.add((r0, c))
            rows = {r for r, cc in dest if cc == c}
            for r in range(min(rows | {r0}), max(rows | {r0}) + 1):
                out.add((r, c))
    else:
        rows = {r for r, _ in dest}
        for r in range(min(rows | {r0}), max(rows | {r0}) + 1):
            out.add((r, c0))
            cols = {c for rr, c in dest if rr == r}
            for c in range(min(cols | {c0}), max(cols | {c0}) + 1):
                out.add((r, c))
    return {r * C + c for r, c in out} - {s_}


def _check_dispatch(c, io, pre, label):
    """xbuf after the dispatch vs its host model: each source chip s's own row t (slot s n + t) on every chip of its
    route word's tree (_touched: the destination chips, bits 0-15, and the chips it passes on the way), every other row
    as before the call; bit for bit. Every destination is in the tree (asserted)."""
    R, C = c["mesh"]
    N = R * C
    route0, xbuf0 = pre
    n = xbuf0[0].numel() // (N * c["H"])
    x0 = [t.reshape(N * n, -1) for t in xbuf0]
    rw = [t.to(torch.int64).reshape(-1)[:n] & 0xFFFFFFFF for t in route0]
    got = [t.reshape(N * n, -1) for t in _rows(io[1])]
    exp = [x.clone() for x in x0]
    sent = relayed = 0
    for s_ in range(N):
        for t in range(n):
            w = int(rw[s_][t])
            mask, order = w & 0xFFFF, (w >> 16) & 1
            if mask == 0:
                continue
            hit = _touched(R, C, s_, mask, order)
            assert all(d in hit for d in range(N) if (mask >> d) & 1)
            for d in hit:
                exp[d][s_ * n + t] = x0[s_][s_ * n + t]
            sent += bin(mask).count("1")
            relayed += len(hit) - bin(mask).count("1")
    for d in range(N):
        bad = (got[d].view(torch.int16) != exp[d].view(torch.int16)).any(1).nonzero().flatten()
        assert (
            bad.numel() == 0
        ), f"{label} chip {d}: {bad.numel()} rows differ from the host model, first {bad[:8].tolist()}"
    assert sent > 0, f"{label}: no row routed anywhere"
    logger.info(
        f"{c['id']} {label}: xbuf bit-exact vs the host model on every chip ({sent} rows to destinations, "
        f"{relayed} kept by a chip on the way)"
    )
