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
case's program. Routed experts: uninitialised weights ("fake": no program here reads them). The block's numerics are
not checked here (models/demos/moe_block/tests, the model's ladder): only the program runner's."""

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

    layers = [
        MoELayer.build(mesh, g, m, host(c["seed"] * 10 + k), dtypes, routed_op=op, layout=dict(c["layout"]))
        for k in range(2)
    ]
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
    return blk, layers, g


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
            pre.append((list(io), [ttnn.from_device(io[i]) for i in c["restore"]]))
        runs.append((d, p, list(io)))
        return orig_run(p, io)

    def rec_wrap():
        monkeypatch.setattr(ttnn.bringup, "prepare_program", prep)
        monkeypatch.setattr(ttnn.bringup, "run_prepared", run)

    t0 = time.time()
    blk, layers, g = _block(mesh_device, c, rec_wrap)
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
    a_i = max(i for i, (io, _) in enumerate(pre) if same(io))
    b_i = max((i for i, (io, _) in enumerate(pre) if not same(io)), default=None)
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
    if head is not None:
        head.free_arena()
