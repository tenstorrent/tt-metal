# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.flat_routed_expert / flat_routed_expert_with_program against their torch semantics (reference.py), one
random-input case per captured model call (cases.py). The op's plan, weight layout and done words come from the fork's
FlatRoutedExpert (flat_expert.py) with the case's literal configuration; the weights are random values the case's
block-float dtype holds exactly (check_exact), so the reference multiplies the same weights as the device.

flat_routed_expert (the shared expert's call): every chip's y rows (flat row i = its expert's token_index[i]) vs the
float64 FFN of those rows. flat_routed_expert_with_program (the routed experts with the weighted combine fused in, built
as the model wires it: weighted_case.py): every chip's combine output (its mesh row's home tokens, summed over the
column's experts each selected, x its routing weight) vs float64; exactly zero rows for the home tokens with no expert
in the column, nonzero ones for the rest.

Limits (F139): rel L2 per chip within calib x the reference at the op's real formats vs float64
(fork_cases.noise_limit; reference.ffn(formats=): bfp8 x / h tiles, the FPU's operand bits at the call's fidelity,
fp32 sums, bf16 y; the combine: bf16 weights, bf16 running sums in a fixed order). PCC printed (a diagnostic).
Determinism: 5 runs on A with one on B between, bit-identical (section 4b of the skill).

The fork's own unit suite is tests/unit/; tests/test_flat_bench.py is its perf bench."""

import importlib.util
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.common.bringup.testing import determinism
from models.demos.common.bringup.testing.fork_cases import CALIB, noise_limit, round_to

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_flat_routed_expert_ttnn_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASES = _load("cases").CASES
_MANT = {"bf8": 7, "bf4": 3}

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="the flat routed expert is Blackhole-only")


def _device_params(c):
    p = dict(c["device_params"])
    p["fabric_config"] = getattr(ttnn.FabricConfig, p["fabric_config"])
    if "fabric_payload_bytes" in p:  # the model's fabric packet payload (harness.device_params)
        rc = ttnn._ttnn.fabric.FabricRouterConfig()
        rc.max_packet_payload_size_bytes = int(p.pop("fabric_payload_bytes"))
        p["fabric_router_config"] = rc
    p["require_exact_physical_num_devices"] = True
    return p


def _expert(c, g):
    """One expert's (W1 [I, H], W3 [I, H], W2 [H, I]) as stored [N, K]: N(0, w_std) rounded to values the case's
    block-float dtype holds exactly (16-value groups along K, the groups the op packs with weights_nk)."""
    H, I, m = c["H"], c["I"], _MANT[c["wdtype"]]
    w = lambda *shape: ref.block_round(  # noqa: E731
        torch.randn(*shape, generator=g) * c["w_std"], m, ties="even", clamp=True
    )
    return w(I, H), w(I, H), w(H, I)


def _rm(mesh_device, t, mapper, dtype):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _formats(c):
    return dict(x_bf16=c["x_bf16"], h_bf16=c["h_bf16"], fidelity=ref.FIDELITY[c["math_fidelity"]])


def _rel_pcc(got, want):
    rel = float((got - want).norm() / want.norm())
    a, b = (got - got.mean()).flatten(), (want - want.mean()).flatten()
    return rel, float(a @ b / (a.norm() * b.norm()))


_SHARED = [c for c in CASES if c["op"] == "flat_routed_expert"]
_ROUTED = [c for c in CASES if c["op"] == "flat_routed_expert_with_program"]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in _SHARED],
    ids=[c["id"] for c in _SHARED],
    indirect=["mesh_device", "device_params"],
)
def test_flat_routed_expert(mesh_device, device_params, case):
    from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

    c = case
    R, C = c["mesh"]
    H, S = c["H"], c["S"]
    assert c["epc"] == 1 and c["n_global"] == 1, "the case's one always-on expert"
    g = torch.Generator().manual_seed(c["seed"])
    cols = [_expert(c, g) for _ in range(C)]  # mesh column cc's slice of the shared expert (device r C + cc)
    fe = FlatRoutedExpert(
        mesh_device,
        [[cols[d % C]] for d in range(R * C)],
        m=c["m"],
        H=H,
        I=c["I"],
        gids=[[0] for _ in range(R * C)],
        n_global=1,
        wdtype=c["wdtype"],
        act=c["act"],
        pin=c["pin"],
        x_bf16=c["x_bf16"],
        h_bf16=c["h_bf16"],
        weights_nk=c["weights_nk"],
        check_exact=True,
    )
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    chips = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(0, 1))
    counts = _rm(mesh_device, torch.tensor([[S]], dtype=torch.int32), rep, ttnn.uint32)
    regions = _rm(mesh_device, torch.tensor([[0]], dtype=torch.int32), rep, ttnn.uint32)
    # chip (r, cc) reads its mesh row's S rows of the column's gathered [R S, H] rows: token index r S + t
    tok = torch.stack([r * S + torch.arange(S) for r in range(R)]).to(torch.int32)
    tidx = _rm(mesh_device, tok.reshape(R, 1, 1, S).expand(R, C, 1, S), chips, ttnn.uint32)

    def inputs(seed):
        gx = torch.Generator().manual_seed(seed)
        x = torch.randn(C, R * S, H, generator=gx).to(torch.bfloat16)  # mesh column cc's gathered rows
        return x, _rm(mesh_device, x.unsqueeze(0).expand(R, C, R * S, H), chips, ttnn.bfloat16)

    def call(xd):
        return fe(
            xd,
            counts,
            regions,
            token_index=tidx,
            y_bf16=c["y_bf16"],
            down_fp32=c["down_fp32"],
            math_fidelity=c["math_fidelity"],
            pack_stochastic_rounding=False,
        )

    x, xd = inputs(c["seed"] + 100)
    out = call(xd)
    assert out.dtype == ttnn.bfloat16 and list(out.shape)[-1] == H, (out.dtype, out.shape)
    fmt = _formats(c)
    for d, t in enumerate(ttnn.get_device_tensors(out)):
        r, cc = divmod(d, C)
        got = ttnn.to_torch(t).double().reshape(-1, H)[:S]
        xs = x[cc, r * S : (r + 1) * S]
        w = cols[cc]

        def fn(xx):
            return ref.ffn(xx, *(t_.to(xx.dtype) for t_ in w), c["act"], formats=fmt)

        want = fn(xs.double())
        lim = noise_limit(fn, [xs], c["noise"]["inputs"], c["noise"]["out"])
        rel, pcc = _rel_pcc(got, want)
        logger.info(
            f"{c['id']} chip {d}: rel {rel:.3e} (limit {lim['rel']:.3e}, noise {lim['noise_rel']:.3e}) "
            f"pcc {pcc:.6f} (a diagnostic)"
        )
        assert rel <= lim["rel"], f"chip {d}: rel L2 {rel:.3e} > limit {lim['rel']:.3e}"
    _, xd_b = inputs(c["seed"] + 101)
    determinism.assert_deterministic(lambda: call(xd), lambda: call(xd_b), first=out, label=c["id"])


def _combine_ref(xr, ids, wts, col, Eg, experts, act, formats=None):
    """Mesh row r's home-token sums over column ``col``'s experts: out[t] = sum_k w[t, k] ffn_{ids[t, k]}(x[t]) over
    the k whose expert is in [col Eg, (col + 1) Eg), in x's dtype. ``formats`` (float32): the op's arithmetic (the
    experts as reference.ffn(formats=), y rows bf16; the combine: w (bf16) x y in fp32, each add rounded to bf16, in
    slot order)."""
    T_, K = ids.shape
    dt = xr.dtype
    ys = torch.zeros(T_, K, xr.shape[-1], dtype=dt)
    sel = (ids >= col * Eg) & (ids < (col + 1) * Eg)
    for e in sorted(set(ids[sel].tolist())):
        tk = (ids == e).nonzero()
        w1, w3, w2 = (t_.to(dt) for t_ in experts[e])
        ys[tk[:, 0], tk[:, 1]] = ref.ffn(xr[tk[:, 0]], w1, w3, w2, act, formats=formats)
    w = wts.to(dt)
    if formats is None or dt != torch.float32:
        return (ys * (w * sel).unsqueeze(-1)).sum(1)
    acc = torch.zeros(T_, xr.shape[-1], dtype=dt)
    for k in range(K):
        m = sel[:, k]
        acc[m] = round_to(acc[m] + w[m, k : k + 1] * ys[m, k], "bf16")
    return acc


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in _ROUTED],
    ids=[c["id"] for c in _ROUTED],
    indirect=["mesh_device", "device_params"],
)
def test_flat_routed_expert_with_program(mesh_device, device_params, case):
    from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

    wcase = _load("weighted_case")
    c = case
    R, C = c["mesh"]
    H, I, S, K, E, epc = c["H"], c["I"], c["S"], c["K"], c["E"], c["epc"]
    Eg = R * epc
    assert E == C * Eg, "experts column-major over the mesh: column cc holds [cc Eg, (cc + 1) Eg)"
    g = torch.Generator().manual_seed(c["seed"])
    t0 = time.time()
    experts = [_expert(c, g) for _ in range(E)]
    # chip (r, cc): column cc's experts [r epc, (r + 1) epc) (column-local ids: the planner's id_offset cc Eg)
    per_dev = [[experts[cc * Eg + r * epc + le] for le in range(epc)] for r in range(R) for cc in range(C)]
    fe = FlatRoutedExpert(
        mesh_device,
        per_dev,
        m=max(256, R * S),
        H=H,
        I=I,
        gids=[[r * epc + le for le in range(epc)] for r in range(R) for cc in range(C)],
        n_global=Eg,
        wdtype=c["wdtype"],
        act=c["act"],
        pin=c["pin"],
        x_bf16=c["x_bf16"],
        h_bf16=c["h_bf16"],
        weights_nk=c["weights_nk"],
        check_exact=True,
    )
    del per_dev
    logger.info(f"{c['id']}: {E} experts built and laid out in {time.time() - t0:.0f} s")
    wc = wcase.WeightedCase(
        mesh_device,
        dict(fe.plan),
        H=H,
        I=I,
        epc=epc,
        K=K,
        S=S,
        out_dtype=getattr(ttnn.DataType, c["out_dtype"]),
        x_bf16=c["x_bf16"],
        h_bf16=c["h_bf16"],
        acc_dtype=getattr(ttnn.DataType, c["acc_dtype"]),
        planner=c["planner"],
    )
    assert wc.P == 1, f"{S} home tokens in {wc.P} passes (the captured call: one pass)"
    rows = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(2, None))

    def inputs(seed):
        """Per mesh row r: its S tokens' x [S, H], top-K distinct global ids and bf16 weights, and the column's
        gathered device inputs (wc.gather_x / wc.routing, as the model)."""
        gi = torch.Generator().manual_seed(seed)
        x = (torch.randn(R, S, H, generator=gi) * 0.3).to(torch.bfloat16)
        ids = torch.stack([torch.stack([torch.randperm(E, generator=gi)[:K] for _ in range(S)]) for _ in range(R)])
        wts = (torch.rand(R, S, K, generator=gi) + 0.5).to(torch.bfloat16)
        put = lambda t, dt: ttnn.from_torch(  # noqa: E731
            t.reshape(1, 1, R * S, -1).contiguous(),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rows,
        )
        xg = wc.gather_x(put(x, ttnn.bfloat16))
        ids_g, wts_g = wc.routing(put(ids.to(torch.int32), ttnn.uint32), put(wts, ttnn.bfloat16))
        return (x, ids, wts), (xg, ids_g, wts_g)

    host, dev = inputs(c["seed"] + 100)
    out = wc(fe, *dev, c["math_fidelity"])
    assert out.dtype == getattr(ttnn.DataType, c["out_dtype"]) and list(out.shape) == [S, H], (out.dtype, out.shape)
    x, ids, wts = host
    fmt = _formats(c)
    for d, t in enumerate(ttnn.get_device_tensors(out)):
        r, cc = divmod(d, C)
        got = ttnn.to_torch(t).double().reshape(S, H)
        # exact: a home token none of whose experts is in this column gets a zero row (the combine's zero_unseen),
        # every other row is finite and nonzero
        seen = ((ids[r] >= cc * Eg) & (ids[r] < (cc + 1) * Eg)).any(-1)
        assert bool((got[~seen] == 0).all()), f"chip {d}: {int((got[~seen] != 0).any(-1).sum())} unseen rows not zero"
        assert bool(torch.isfinite(got).all()) and bool((got[seen] != 0).any(-1).all()), f"chip {d}: a seen row is 0"
        # (fork_cases.noise_limit's arithmetic with the float64 reference computed once: calib x rel of the
        # formats run vs float64, every output rounded to the case's out dtype)
        want = _combine_ref(x[r].double(), ids[r], wts[r].double(), cc, Eg, experts, c["act"])
        real = round_to(
            _combine_ref(x[r].float(), ids[r], wts[r].float(), cc, Eg, experts, c["act"], formats=fmt),
            c["noise"]["out"],
        ).double()
        noise = float((real - want).norm() / want.norm())
        rel, pcc = _rel_pcc(got, want)
        rel_m = float((got - real).norm() / want.norm())
        logger.info(
            f"{c['id']} chip {d}: rel {rel:.3e} (limit {CALIB * noise:.3e}, noise {noise:.3e}; vs the formats run "
            f"{rel_m:.3e}) pcc {pcc:.6f} (a diagnostic)"
        )
        assert rel <= CALIB * noise, f"chip {d}: rel L2 {rel:.3e} > limit {CALIB * noise:.3e}"
    _, dev_b = inputs(c["seed"] + 101)
    determinism.assert_deterministic(
        lambda: wc(fe, *dev, c["math_fidelity"]), lambda: wc(fe, *dev_b, c["math_fidelity"]), first=out, label=c["id"]
    )
