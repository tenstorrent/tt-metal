# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Python side of ttnn.experimental.moe_block.flat_routed_expert: the weights in the op's bank layout and a small
wrapper that owns what a set of local experts needs across calls.

    op = FlatRoutedExpert(mesh_device, weights, m=capacity, H=H, I=I, gids=per_device_global_ids, n_global=E)
    y = op(dispatch_buffer_rm_bf16, counts, regions)   # y: [rows, H] bfp8 TILE, expert e's rows at its region
    op.close()                                         # stops the DRISC senders once no FlatRoutedExpert needs them

The plan comes from C++ (ttnn.experimental.moe_block.flat_routed_expert_plan); flat_weight_regions packs each
device's experts into the plan's per-core regions and bank_sharded puts them in DRAM, one region list per bank. The
packing is torch work on the host, done once per weight set, so it stays in Python.
"""

from __future__ import annotations

import torch

import ttnn

W_DTYPES = {"bf8": ttnn.bfloat8_b, "bf4": ttnn.bfloat4_b}
ACTS = {"silu": 0, "swigluoai": 1, "situ": 2, "clamped_silu": 3, "gelu_tanh": 4}
KBLK = 8  # K tiles per gate/up weight block (the plan's kblk)


def _moe_block():
    return ttnn._ttnn.operations.experimental.moe_block


def act_ref(g, u, act="silu"):
    """Reference gate/up activation (the compute kernel's variants)."""
    F = torch.nn.functional
    if act == "silu":
        return F.silu(g) * u
    if act == "swigluoai":  # GPT-OSS / MiniMax-M3: alpha 1.702, limit 7
        g = g.clamp(max=7.0)
        return (u.clamp(-7.0, 7.0) + 1) * g * torch.sigmoid(1.702 * g)
    if act == "situ":  # Kimi K3: beta 4 (gate), 25 (up)
        return 4 * torch.tanh(g / 4) * torch.sigmoid(g) * 25 * torch.tanh(u / 25)
    if act == "clamped_silu":  # DeepSeek V4: limit 10
        return F.silu(g.clamp(max=10.0)) * u.clamp(-10.0, 10.0)
    if act == "gelu_tanh":  # Gemma 4
        return F.gelu(g, approximate="tanh") * u
    raise ValueError(act)


def kd_of(pcd, it):
    """Down K-block tiles for a down core of pcd tile columns (the plan's FlatRoutedExpertPlan::kd_of)."""
    return max(k for k in (8, 4, 2, 1) if k * pcd <= 16 and it % k == 0)


def _hg_units(Wg, Wu):
    """(Wg, Wu) [H, I] -> [H, 2 I]: unit u's 32 columns = gate 16 u..16 u + 15 then up 16 u..16 u + 15."""
    H, I = Wg.shape
    return torch.cat([Wg.reshape(H, I // 16, 16), Wu.reshape(H, I // 16, 16)], dim=2).reshape(H, 2 * I)


def flat_weight_regions(W_l, lay):
    """One device's weights in the plan's per-core bank regions. W_l: per local expert (Wg [H, I], Wu [H, I],
    Wd [I, H]); lay: the plan's as_dict(). Returns (gate/up regions per weight slot, down regions per bank and down
    core), each a list of equal-size [tiles, 32, 32]:
    * slot r: per expert, per K-block of KBLK tiles, its RG column groups' blocks (group (r % n_rd_sg) * RG + pl):
      [KBLK x (gate, up) x NP] tile pairs, or with 16-column units (nu) NU unit tiles [gate | up] of 16 columns each;
    * down core d: per expert, per K-block of kd_of(pcds[d]) tiles, its [kd x pcds[d]] columns from col0s[d], rotated
      over the banks (block k in bank (d + k) % banks) and zero-padded to the widest core's region."""
    tiles = lambda w: w.view(w.shape[0] // 32, 32, w.shape[1] // 32, 32).permute(0, 2, 1, 3)  # noqa: E731
    nu = lay.get("nu") or 0
    if nu:  # 16-column units: unit u's tile = [gate | up] of columns 16 u..; a column group's NU units in order
        U_l = [tiles(_hg_units(a, b)) for a, b, _ in W_l]
    T_l = [(tiles(a), tiles(b), tiles(c)) for a, b, c in W_l]
    it = W_l[0][0].shape[1] // 32
    RG, NP = lay["rg"], lay["np"]
    regions = []
    for r in range(lay["n_rd"]):
        p0 = (r % lay["n_rd_sg"]) * RG
        blocks = []
        for e_, (Wg_t, Wu_t, _) in enumerate(T_l):
            for c in range(lay["nk_gu"]):
                ks = slice(c * KBLK, (c + 1) * KBLK)
                for pl in range(RG):
                    if nu:
                        Ut = U_l[e_]
                        units = range((p0 + pl) * nu, (p0 + pl + 1) * nu)
                        blocks.append(torch.stack([Ut[ks, u_] for u_ in units], dim=1).reshape(-1, 32, 32))
                        continue
                    cols = [(p0 + pl) * NP + p_ for p_ in range(NP)]
                    blocks.append(
                        torch.stack([w_[ks, c_] for c_ in cols for w_ in (Wg_t, Wu_t)], dim=1).reshape(-1, 32, 32)
                    )
        regions.append(torch.cat(blocks))
    pcds, col0s = lay["pcds"], lay["col0s"]

    def dreg(d):
        p_ = pcds[d]
        k_ = kd_of(p_, it)
        return torch.cat(
            [
                Wd_t[c * k_ : (c + 1) * k_, col0s[d] : col0s[d] + p_].reshape(-1, 32, 32)
                for _, _, Wd_t in T_l
                for c in range(it // k_)
            ]
        )

    # down core d's block k in bank (d + k) % banks, at d * wd_rot_tiles + (k // banks) blocks: region d * banks + j
    # (which bank_sharded puts in bank j at offset d * region) holds the blocks k = j - d (mod banks) in order
    banks, sub = lay["banks"], lay["wd_rot_tiles"]
    dregs = []
    for d in range(lay["nd"]):
        k_ = kd_of(pcds[d], it)
        blocks = dreg(d).reshape(-1, k_ * pcds[d], 32, 32)
        for j in range(banks):
            mine = blocks[[k for k in range(blocks.shape[0]) if (d + k) % banks == j]].reshape(-1, 32, 32)
            pad = torch.zeros(sub - mine.shape[0], 32, 32, dtype=mine.dtype, device=mine.device)
            dregs.append(torch.cat([mine, pad]))
    n_max = max(r_.shape[0] for r_ in dregs)
    dregs = [
        torch.cat([r_, torch.zeros(n_max - r_.shape[0], 32, 32, dtype=r_.dtype, device=r_.device)]) for r_ in dregs
    ]
    return regions, dregs


def bank_host_tensor(regions, banks):
    """Equal-size regions [tiles, 32, 32] -> the [rows, banks * 32] host matrix whose column block b is bank b: region
    i contiguous in bank i % banks at row offset (i // banks) * region rows (zero regions pad the list to whole
    banks)."""
    per = -(-len(regions) // banks)
    regions = list(regions) + [torch.zeros_like(regions[0])] * (per * banks - len(regions))
    return torch.cat(
        [torch.cat([regions[b + h * banks] for h in range(per)]).reshape(-1, 32) for b in range(banks)], dim=1
    )


def _bank_memory_config(rows, banks):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (rows, 32), ttnn.ShardOrientation.ROW_MAJOR),
    )


def bank_sharded(regions_per_dev, banks, dtype, mesh_device, replicate=False):
    """Per-core weight regions -> one width-sharded DRAM tensor (bank_host_tensor's layout). One region list per
    device (row-major device order; lockstep allocation gives every device the same address); ``replicate``: one
    list, the same on every device."""
    hosts = [bank_host_tensor(regions, banks) for regions in regions_per_dev]
    mc = _bank_memory_config(hosts[0].shape[0], banks)
    one = len(hosts) == 1 or replicate
    host = ttnn.from_torch(
        hosts[0] if one else torch.stack(hosts),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mc,
        mesh_mapper=None if one else ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    return host.to(mesh_device, mc)


def uninitialized_bank_sharded(n_regions, region_tiles, banks, dtype, mesh_device):
    """An uninitialised device tensor with bank_sharded's spec (perf runs: the bfp matmuls take the same time whatever
    the bytes are, so the host packing and upload are skipped)."""
    rows = -(-n_regions // banks) * region_tiles * 32
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows, banks * 32]), dtype, ttnn.TILE_LAYOUT, mesh_device, _bank_memory_config(rows, banks)
    )


def flat_weight_shapes(H, I, E, lay):
    """(regions, tiles per region) of each laid-out weight tensor, from flat_weight_regions on shape-only tensors."""
    meta = lambda r, c: torch.empty(r, c, device="meta")  # noqa: E731
    regions, dregs = flat_weight_regions([(meta(H, I), meta(H, I), meta(I, H))] * E, lay)
    return {"w_gu": (len(regions), regions[0].shape[0]), "w_d": (len(dregs), dregs[0].shape[0])}


FLAT_LAYOUT_FIELDS = (
    "x_helpers",
    "pin_ratio",
    "rows_hint",
    "x_ready",
    "drisc_tiles",
    "drisc_ub",
    "drisc_dma_cap",
    "drisc_mpfe",
    "drisc_early_free",
    "drisc_probe",
)


def flat_layout(layout=None, **kw):
    """A FlatLayoutOptions from a dict of fields / an existing FlatLayoutOptions / None, plus field overrides."""
    opts = _moe_block().FlatLayoutOptions
    if isinstance(layout, opts):
        layout = {f: getattr(layout, f) for f in FLAT_LAYOUT_FIELDS}
    return opts(**{**(layout or {}), **kw})


class FlatRoutedExpert:
    """ttnn.experimental.moe_block.flat_routed_expert for one set of local experts per device: the plan, the weights in
    its bank layout, the expert id table, the done words and the DRISC senders, kept across calls.

    ``weights``: per device (row-major device order), per local expert (Wg [H, I], Wu [H, I], Wd [I, H]); a callable
    returning that; or ``"uninitialized"`` (perf runs: weight tensors of the right spec, no packing / upload, garbage
    outputs). ``gids``: per device, the local experts' global ids (rows of the routing's counts / regions).
    ``layout``: FlatLayoutOptions fields (dict) or a FlatLayoutOptions. ``done_buffer_type``: L1 or L1_SMALL."""

    def __init__(
        self,
        mesh_device,
        weights,
        *,
        m,
        H,
        I,
        gids,
        n_global,
        wdtype="bf4",
        act="silu",
        pin=1,
        layout=None,
        done_buffer_type=None,
    ):
        mb = _moe_block()
        E, n_dev = len(gids[0]), len(gids)
        self.mesh_device, self.H, self.I, self.m, self.pin = mesh_device, H, I, m, pin
        self.act = ACTS[act]
        self.layout = flat_layout(layout)
        self.plan = mb.flat_routed_expert_plan(
            mesh_device, H, I, E, n_global, m, weights_bf8=wdtype == "bf8", pin=pin, layout=self.layout
        )
        lay = self.plan.as_dict()
        self.lay = lay
        dtype, banks = W_DTYPES[wdtype], lay["banks"]
        if isinstance(weights, str):
            assert weights == "uninitialized", weights
            shapes = flat_weight_shapes(H, I, E, lay)
            self.w_gu, self.w_d = (
                uninitialized_bank_sharded(*shapes[n], banks, dtype, mesh_device) for n in ("w_gu", "w_d")
            )
        else:
            weights = weights() if callable(weights) else weights
            assert len(weights) == n_dev and all(len(w_) == E for w_ in weights)
            # every device the same experts (e.g. a test's one expert set everywhere): pack once, replicate
            same = n_dev > 1 and all(all(a is b for a, b in zip(w_, weights[0])) for w_ in weights[1:])
            packed = [flat_weight_regions(W_l, lay) for W_l in (weights[:1] if same else weights)]
            self.w_gu = bank_sharded([p[0] for p in packed], banks, dtype, mesh_device, replicate=same)
            self.w_d = bank_sharded([p[1] for p in packed], banks, dtype, mesh_device, replicate=same)
        gid_host = torch.tensor(gids, dtype=torch.int32).reshape(n_dev, 1, E)
        self.gidx = ttnn.from_torch(
            gid_host if n_dev > 1 else gid_host[0],
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0) if n_dev > 1 else None,
        )
        self.done = mb.create_flat_routed_expert_done_words(mesh_device, self.plan, done_buffer_type)
        self.drisc = mb.FlatDriscSenders.get(
            mesh_device, self.plan.drisc_n, self.plan.drisc_tiles, self.plan.drisc_probe
        )

    def __call__(
        self,
        x,
        counts,
        regions,
        token_index=None,
        x_pages_per_row=1,
        y_row_major=False,
        down_fp32=False,
        pack_stochastic_rounding=False,
        y_bf16=False,
        math_fidelity=0,
        output=None,
    ):
        """x: the row-major bf16 dispatch buffer [rows, H]; counts / regions: the routing's [1, n_global] uint32
        rows. See ttnn.experimental.moe_block.flat_routed_expert for the options."""
        return ttnn.experimental.moe_block.flat_routed_expert(
            x,
            counts,
            regions,
            self.gidx,
            self.w_gu,
            self.w_d,
            self.done,
            self.drisc,
            self.I,
            self.m,
            activation=self.act,
            pin=self.pin,
            token_index=token_index,
            x_pages_per_row=x_pages_per_row,
            y_row_major=y_row_major,
            down_fp32=down_fp32,
            pack_stochastic_rounding=pack_stochastic_rounding,
            y_bf16=y_bf16,
            math_fidelity=math_fidelity,
            layout=self.layout,
            output=output,
        )

    def close(self):
        """Drops this op's hold on the DRISC senders (they stop once nothing holds them)."""
        self.drisc = None
