# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.moe_block_main.fused_head's bf8 shared-expert options (DeepSeek-V4.1 P.37), on the machine's mesh
(a Galaxy 8x4, FABRIC_2D_TORUS_XY as DeepSeek-V4.1 runs it; any R x C with C >= 2), DeepSeek-V4.1-Flash's dims, a
chunk of R x 128 tokens:

* MoEDtypes.shared_activations = bfloat8_b (FIX.9's x / h in bf8 inside the head; the caller's xq):
  test_fused_head_bf8_accuracy - the shared expert's output against torch with x and h on the bf8 grid, closer to it
  than the bf16-activation head is.
* MoEDtypes.shared_pack_precise (SE_PACK_PRECISE): test_fused_head_bf8_pack_rounding - bit for bit, how the head
  packs h. Crafted inputs make h exactly predictable: gate = 16 (one x column, weight 1.0) is clamped to 10, so
  h = c u with c = the SFPU's silu(10) (a constant) and u = one x column (up's weight 1.0), u in [1, 1.6) at 1/64
  steps or an anchor 3.0 that fixes each 16-value group's shared exponent at 2^4; the down projection is the identity
  on the first Is output columns (weights 1.0, one chip's term a column), so the head's output there IS the packed h.
  Probes sit just below a bf8 half-step at the shared exponent, where one rounding of the fp32 value (rounds down)
  and the default two (7 bits at the value's own exponent first: lands on the half-step, then ties away: up) differ.
  On: every value equals the one-rounding rule (half away from zero at the shared exponent - ttnn.typecast's pack, the
  reference's single cast of h); off: the two-rounding rule (DS41-137). Only values whose result does not depend on
  c within [C_LO, C_HI] (the SFPU's silu(10) is not bit-known) are compared, at least MIN_DIFF of them where the
  rules differ."""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

mb = ttnn.bringup.moe_block_main

E, K, H, I, IS = 384, 6, 5120, 2304, 2304  # DeepSeek-V4.1-Flash
ROUTE_SCALE, LIMIT = 1.5, 10.0
ROW_TOKENS = 128  # tokens a mesh row (32 x TP on a 4-wide mesh row: the head's rows rule)
C_LO, C_HI = 9.98, 10.0  # the SFPU's silu(10) (torch fp32: 9.99954605)
MIN_DIFF = 64


def _device_params():
    rc = ttnn._ttnn.fabric.FabricRouterConfig()
    rc.max_packet_payload_size_bytes = (2 * H + 32 + 63) // 64 * 64  # (moe_impl.moe_block_fabric_payload)
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        "fabric_router_config": rc,
        "l1_small_size": 24576,
    }


MESH = (8, 4)  # (a Galaxy as DeepSeek-V4.1 runs it; the head runs on any R x C, C >= 2, R x 128 tokens)
PARAMS = pytest.mark.parametrize(
    "mesh_device, device_params", [pytest.param(MESH, _device_params(), id="x".join(map(str, MESH)))], indirect=True
)


def _dims():
    act = mb.ExpertActivation(kind=mb.ExpertActivationKind.ClampedSilu, limit=LIMIT)
    return mb.MoEModelDims(
        num_experts=E,
        top_k=K,
        emb_dim=H,
        routed_expert_hidden_dim=I,
        routed_expert_activation=act,
        router=mb.RouterSpec(score_func=mb.RouterScoreFunc.SqrtSoftplus, route_scale=ROUTE_SCALE),
        shared_expert_hidden_dim=IS,
        shared_expert_activation=act,
    )


def _cap(mesh):
    """The Galaxy's 110-core rule (agents' rule 18): the ops' grid capped where the chip has more cores."""
    g = mesh.compute_with_storage_grid_size()
    if g.x * g.y > 110 and mb.max_grid() == (0, 0):
        mb.set_max_grid(110 // g.y, g.y)


class Head:
    """The head on a chunk of R x ROW_TOKENS tokens, its weights and x from the host (torch Linear layouts)."""

    def __init__(
        self, mesh, gate, up, down, router, bias, x, activations=None, pack_precise=False, dtypes=None, dims=None
    ):
        """dtypes / dims: a model's MoEDtypes / MoEModelDims as they are (e.g. DeepSeek-V4.1's adapter's, from its
        settings); default DeepSeek-V4.1-Flash's dims, LoFi, the given activations and pack_precise."""
        _cap(mesh)
        self.mesh = mesh
        self.R, self.C = tuple(mesh.shape)
        assert self.C >= 2, "the head's TP axis"
        self.chunk = self.R * ROW_TOKENS
        if dtypes is None:
            dtypes = mb.MoEDtypes()
            dtypes.shared_fidelity = ttnn.MathFidelity.LoFi  # (DeepSeek-V4.1's, D5)
            dtypes.shared_activations = activations
            dtypes.shared_pack_precise = pack_precise
        self.dtypes = dtypes
        activations = dtypes.shared_activations
        cfg = mb.MoEBlockConfig(chunk_tokens=self.chunk, num_links=2, persist_l1_small=True)
        self.res = mb.MoEBlockResources(mesh, dims if dims is not None else _dims(), cfg, self.dtypes)
        self.A = 11  # (the combine's accumulators: the head only plans the routing for them)
        fit = mb.plan_fused_head(self.res, self.dtypes, self.A, None)
        assert fit.ok, fit.why
        lay = fit.planned[0].lay
        logger.info(f"head plan GY {lay.GY} MT {lay.MT} NCH {lay.NCH} KB {lay.KB} xdram {lay.xdram}")
        assert not lay.xdram, "the DRAM fallback (x_full) is not wired in this test"
        f32 = lambda t: ttnn.from_torch(t.float().contiguous(), dtype=ttnn.float32)  # noqa: E731
        self.w = mb.prepare_fused_head_weights(
            f32(gate), f32(up), f32(down), None, f32(router), f32(bias),
            resources=self.res, dtypes=self.dtypes, accumulators=self.A,
        )  # fmt: skip
        self.x = ttnn.from_torch(
            x.reshape(1, 1, *x.shape).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(self.R, self.C), dims=(2, 3)),
        )
        self.xq = None
        if activations == ttnn.bfloat8_b:
            self.xq = ttnn.typecast(self.x, ttnn.bfloat8_b)

    def __call__(self) -> torch.Tensor:
        """The shared expert's output [chunk, H] in token order (each chip's owner rows, chip order)."""
        kw = {"xq": self.xq} if self.xq is not None else {}
        out = mb.fused_head(self.x, self.w, resources=self.res, dtypes=self.dtypes, accumulators=self.A, **kw)
        S = self.chunk // (self.R * self.C)
        return torch.cat(
            [ttnn.to_torch(d).reshape(-1, d.shape[-1])[:S].float() for d in ttnn.get_device_tensors(out.se)], dim=0
        )

    def close(self):
        self.res.release()


def _q(t, dtype):
    """t on a device dtype's grid (ttnn's host packer)."""
    return ttnn.to_torch(ttnn.from_torch(t.float(), dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()


def _clamped_swiglu(g, u):
    return torch.nn.functional.silu(g.clamp(max=LIMIT)) * u.clamp(-LIMIT, LIMIT)


def _skip_unless_blackhole():
    if ttnn.get_arch_name() != "blackhole":
        pytest.skip("Blackhole only")


# ------------------------------------------------------------------------------------------------ accuracy (bf8 x / h)
@pytest.mark.timeout(1800)
@PARAMS
def test_fused_head_bf8_accuracy(mesh_device, device_params):
    """bf8 x / h: the head's shared expert against torch with x and h on the bf8 grid (as the reference fp8-quantizes
    both), on the bf8 weights; closer to that reference than the bf16-activation head, and as close to the
    full-precision one as a bf8 rounding of x and h allows."""
    _skip_unless_blackhole()
    g = torch.Generator().manual_seed(7)
    r = lambda *s, sc=1.0: torch.randn(*s, generator=g) * sc  # noqa: E731
    gate, up, down = r(IS, H, sc=H**-0.5), r(IS, H, sc=H**-0.5), r(H, IS, sc=IS**-0.5)
    router, bias = r(E, H, sc=H**-0.5), r(E, sc=0.01)
    chunk = mesh_device.shape[0] * ROW_TOKENS
    x = r(chunk, H).to(torch.bfloat16).float()
    b8 = ttnn.bfloat8_b
    wg, wu, wd = _q(gate, b8), _q(up, b8), _q(down, b8)
    ref_full = _clamped_swiglu(x @ wg.T, x @ wu.T) @ wd.T
    xq = _q(x, b8)
    ref_q = _q(_clamped_swiglu(xq @ wg.T, xq @ wu.T), b8) @ wd.T
    outs = {}
    for name, act in (("bf16", ttnn.bfloat16), ("bf8", b8)):
        head = Head(mesh_device, gate, up, down, router, bias, x, act, pack_precise=True)
        try:
            outs[name] = head()
            again = head()
            assert torch.equal(outs[name], again), f"{name}: two calls on the same inputs differ"
        finally:
            head.close()
    rel = lambda a, b: float((a - b).norm() / b.norm())  # noqa: E731
    for name, y in outs.items():
        ok, pcc = comp_pcc(ref_q, y, 0.999)
        logger.info(
            f"{name}: vs bf8-x/h reference rel {rel(y, ref_q):.5f} pcc {pcc}; vs full precision rel {rel(y, ref_full):.5f}"
        )
        assert ok, f"{name}: pcc {pcc} vs the bf8-x/h reference"
    assert rel(outs["bf8"], ref_q) < rel(outs["bf16"], ref_q), "bf8 x / h is not closer to the bf8-x/h reference"
    # (bf8 x / h: one more rounding of each than the bf16 head - its distance to full precision grows by at most the
    # reference's own bf8 drift, twice)
    drift = rel(ref_q, ref_full)
    assert rel(outs["bf8"], ref_full) < rel(outs["bf16"], ref_full) + 2 * drift, (
        rel(outs["bf8"], ref_full),
        rel(outs["bf16"], ref_full),
        drift,
    )


# ------------------------------------------------------------------------------------------------ the pack's rounding
def _single(v: torch.Tensor, e_shared: int) -> torch.Tensor:
    """|v| packed to bf8 at the shared exponent in one rounding (half away from zero; 7 magnitude bits)."""
    step = 2.0 ** (e_shared - 6)
    return (torch.floor(v.abs() / step + 0.5).clamp(max=127) * step).copysign(v)


def _double(v: torch.Tensor, e_shared: int) -> torch.Tensor:
    """The packer's default: first 7 significant bits at the value's own exponent (half away), then as _single."""
    a = v.abs()
    e = torch.floor(torch.log2(a.clamp(min=2.0**-100)))
    ulp = torch.pow(2.0, e - 6)
    return _single(torch.floor(a / ulp + 0.5) * ulp, e_shared).copysign(v)


def _crafted(chunk: int):
    """(x [chunk, H], gate [IS, H], up [IS, H], down [H, IS], u [chunk, IS]): h[t, j] = c u[t, j] (module doc)."""
    x = torch.zeros(chunk, H)
    t = torch.arange(chunk).unsqueeze(1)
    i = torch.arange(16).unsqueeze(0)
    x[:, 0:16] = 3.0  # anchors (exponent 1 in x; c x 3 = 29.99: every h group's shared exponent 2^4)
    x[:, 16:32] = 1.0 + ((7 * t + 5 * i) % 38).float() / 64  # probes: [1, 1.6), 1/64 steps (bf8-exact in their group)
    x[:, 32:48] = 16.0  # the gate's source (clamped to 10)
    gate = torch.zeros(IS, H)
    gate[:, 32] = 1.0
    up = torch.zeros(IS, H)
    src = torch.zeros(IS, dtype=torch.long)
    for j in range(IS):
        src[j] = 0 if j % 16 == 0 else 16 + ((j // 16) * 3 + j % 16) % 16
    up[torch.arange(IS), src] = 1.0
    down = torch.zeros(H, IS)
    down[torch.arange(IS), torch.arange(IS)] = 1.0
    return x, gate, up, down, x[:, src]


def check_pack_rounding(mesh_device, pack_precise: bool, dtypes=None, dims=None):
    """Run the crafted head (module doc) and check its packed h against the option's rule, bit for bit. dtypes /
    dims: a model's (its activations bf8 and pack_precise as given), default this module's."""
    chunk = mesh_device.shape[0] * ROW_TOKENS
    x, gate, up, down, u = _crafted(chunk)
    g = torch.Generator().manual_seed(11)
    router, bias = torch.randn(E, H, generator=g) * H**-0.5, torch.randn(E, generator=g) * 0.01
    if dtypes is not None:
        assert dtypes.shared_activations == ttnn.bfloat8_b and bool(dtypes.shared_pack_precise) == pack_precise
    head = Head(
        mesh_device,
        gate,
        up,
        down,
        router,
        bias,
        x,
        ttnn.bfloat8_b,
        pack_precise=pack_precise,
        dtypes=dtypes,
        dims=dims,
    )
    try:
        y = head()
    finally:
        head.close()
    h_dev = y[:, :IS].double()
    assert torch.count_nonzero(y[:, IS:]) == 0, "the down projection's zero columns are not zero"
    u = u.double()
    rule = _single if pack_precise else _double
    outs = {name: [] for name in ("single", "double")}
    for c in torch.linspace(C_LO, C_HI, 41, dtype=torch.float64):
        v = (c.float() * u.float()).double()  # (fp32 c x u, as the SFPU multiplies)
        outs["single"].append(_single(v, 4))
        outs["double"].append(_double(v, 4))
    single, double = torch.stack(outs["single"]), torch.stack(outs["double"])
    robust = (single == single[0]).all(0) & (double == double[0]).all(0)
    want = (single if rule is _single else double)[0]
    differ = robust & (single[0] != double[0])
    n_robust, n_differ = int(robust.sum()), int(differ.sum())
    bad = robust & (h_dev != want)
    logger.info(
        f"pack_precise={pack_precise}: {n_robust} of {robust.numel()} values decided for every c in [{C_LO}, {C_HI}], "
        f"{n_differ} where one and two roundings differ; {int(bad.sum())} not as the "
        f"{'one-rounding' if pack_precise else 'two-rounding'} rule"
    )
    assert n_differ >= MIN_DIFF, f"only {n_differ} probes tell the rules apart"
    if bad.any():
        idx = bad.nonzero()[:8].tolist()
        detail = [
            (t, j, float(u[t, j]), float(h_dev[t, j]), float(single[0][t, j]), float(double[0][t, j])) for t, j in idx
        ]
        pytest.fail(f"{int(bad.sum())} values not as the rule (t, j, u, device, one rounding, two): {detail}")
    # (the rule that is not the option's must be refuted on the probes: the test tells them apart)
    other = double[0] if pack_precise else single[0]
    assert int((differ & (h_dev != other)).sum()) == n_differ
    return n_robust, n_differ


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("pack_precise", [True, False], ids=["precise", "default"])
@PARAMS
def test_fused_head_bf8_pack_rounding(mesh_device, device_params, pack_precise):
    """The head's bf8 h, bit for bit: shared_pack_precise on - one rounding of the fp32 value at the group's shared
    exponent (the reference's single cast); off - the packer's two roundings."""
    _skip_unless_blackhole()
    check_pack_rounding(mesh_device, pack_precise)
