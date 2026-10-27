# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.update_padded_kv_cache against its torch semantics (reference.py), one random-input case per captured
model call (cases.py). Pure data movement, in place: every chip's whole cache is compared exactly after the call, the
rows the call writes against the input rows the reference maps there, every other element against the cache's prior
(random, distinct per chip) content, so a row written where it should not be fails too. Nothing is don't-care.

The cache is built as the case's placement (ND-sharded DRAM, one record per shard round-robin over the DRAM banks, its
topology) and the input as mesh row r's rows replicated over the columns; both are checked against the captured
placement (fork_capture's description of a tensor). Metadata-path cases pass slot / start / valid as 1-element uint32
device tensors.

Determinism (section 4b of the skill): the op writes into its cache input, so the cache is restored from the same host
copy before every call; 5 runs on A with one on B (other input rows, same arguments) after the first, the whole cache
compared bit for bit."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import determinism
from models.demos.common.bringup.testing.fork_capture import _tensor as describe

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_update_padded_kv_cache_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASES = _load("cases").CASES
_PLACEMENT = ("shape", "dtype", "layout", "buffer", "memory_layout")


def _device_params(c):
    p = dict(c["device_params"])
    p["fabric_config"] = getattr(ttnn.FabricConfig, p["fabric_config"])
    if "fabric_payload_bytes" in p:  # the model's fabric packet payload (harness.device_params)
        rc = ttnn._ttnn.fabric.FabricRouterConfig()
        rc.max_packet_payload_size_bytes = int(p.pop("fabric_payload_bytes"))
        p["fabric_router_config"] = rc
    # a smaller mesh opened on a bigger box fails the 2D fabric handshake: the case runs on its own mesh size only
    p["require_exact_physical_num_devices"] = True
    return p


def _check_placement(t, want, mesh, what):
    got = describe(t)
    for k in _PLACEMENT:
        assert got[k] == want[k], f"{what}: {k} {got[k]} != captured {want[k]}"
    assert got["mesh"] == list(mesh) and got["devices"] == mesh[0] * mesh[1], (what, got)


def _per_device(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


class _Cache:
    """The case's cache: random prior content (distinct per chip, kept on the host) and a device tensor in the case's
    placement; ``restore`` writes the prior content back into the same device buffers."""

    def __init__(self, mesh, c):
        spec = c["cache"]
        rows, cols = c["mesh"]
        b, heads, n, width = spec["shape"]
        banks = mesh.dram_grid_size().x
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(i, 0), ttnn.CoreCoord(i, 0)) for i in range(banks)])
        nd = ttnn.NdShardSpec(
            shard_shape=[1, 1, spec["shard_rows"], width],
            grid=grid,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        )
        mc = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd)
        g = torch.Generator().manual_seed(c["seed"] + 500)
        prior = torch.randn(b, heads, n * rows * cols, width, generator=g).to(torch.bfloat16)
        kw = dict(dtype=getattr(ttnn.DataType, spec["dtype"]), layout=getattr(ttnn.Layout, spec["layout"]))
        mapper = ttnn.ShardTensorToMesh(mesh, dim=2)
        self.host = ttnn.from_torch(prior, mesh_mapper=mapper, memory_config=mc, **kw)
        self.t = ttnn.from_torch(prior, device=mesh, mesh_mapper=mapper, memory_config=mc, **kw)
        # the model's topology (tt/served.py alloc_served): TP-dedup Shard(2) x Shard(2) over the mesh, else replicated
        phys = ttnn.MeshShape(rows, cols)
        coords = [ttnn.MeshCoordinate([x[i] for i in range(x.dims())]) for x in ttnn.MeshCoordinateRange(phys)]
        if spec["dedup"]:
            topo = ttnn.TensorTopology(phys, [ttnn.PlacementShard(2), ttnn.PlacementShard(2)], coords)
        else:
            topo = ttnn.TensorTopology(ttnn.MeshShape([rows * cols]), [ttnn.PlacementReplicate()], coords)
        self.t.update_tensor_topology(topo)
        _check_placement(self.t, spec, c["mesh"], "cache")
        self.before = _per_device(self.t)
        assert len(self.before) == rows * cols

    def restore(self):
        ttnn.copy_host_to_device_tensor(self.host, self.t)


def _u32(mesh, v):
    return ttnn.from_torch(
        torch.tensor([[v]], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _input(mesh, c, seed):
    """x [SP, heads, W, width] random bf16 (mesh row r's input rows) and its device tensor, replicated over the
    columns."""
    spec = c["input"]
    rows, cols = c["mesh"]
    _, heads, w, width = spec["shape"]
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, heads, w, width, generator=g).to(torch.bfloat16)
    t = ttnn.from_torch(
        x,
        dtype=getattr(ttnn.DataType, spec["dtype"]),
        layout=getattr(ttnn.Layout, spec["layout"]),
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(0, None)),
    )
    _check_placement(t, spec, c["mesh"], "input")
    # the device order the test assumes (row-major, d = r cols + col): device d holds mesh row d / cols's rows
    for d, got in enumerate(_per_device(t)):
        assert torch.equal(got[0], x[d // cols]), f"device order: dev {d}"
    return x, t


def _call(c, cache, x_t, meta):
    args = dict(
        layer_idx=c["layer_idx"], num_layers=c["num_layers"], cluster_axis=c["cluster_axis"], tp_axis=c["tp_axis"]
    )
    if c["path"] == "scalar":
        return ttnn.bringup.update_padded_kv_cache(
            cache,
            x_t,
            slot_idx=c["slot_idx"],
            kv_actual_global=c["kv_actual_global"],
            valid_global=c["valid_global"],
            **args,
        )
    slot, kv, valid = meta
    return ttnn.bringup.update_padded_kv_cache(cache, x_t, slot, kv, valid_global=valid, **args)


@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in CASES],
    ids=[c["id"] for c in CASES],
    indirect=["mesh_device", "device_params"],
)
def test_update_padded_kv_cache(mesh_device, device_params, case):
    c = case
    cache = _Cache(mesh_device, c)
    x, x_t = _input(mesh_device, c, c["seed"])
    meta = None
    if c["path"] == "metadata":
        meta = tuple(_u32(mesh_device, c[k]) for k in ("slot_idx", "kv_actual_global", "valid_global"))
        for t in meta:
            _check_placement(t, c["meta"], c["mesh"], "metadata")

    out = _call(c, cache.t, x_t, meta)
    want = ref.update(
        cache.before,
        x,
        tuple(c["mesh"]),
        start=c["kv_actual_global"],
        valid=c["valid_global"],
        cluster_axis=c["cluster_axis"],
        tp_axis=c["tp_axis"],
        slot=c["slot_idx"],
        layer=c["layer_idx"],
        num_layers=c["num_layers"],
    )
    got = _per_device(out)
    n_written = 0
    for d, (a, b, w) in enumerate(zip(got, cache.before, want)):
        changed = (w != b).any(dim=-1)
        n_written += int(changed.sum())
        bad = (a != w).any(dim=-1).nonzero()
        assert not bad.numel(), (
            f"chip {d}: {bad.shape[0]} cache rows differ (first [batch, head, row] {bad[0].tolist()}); "
            f"{int(changed.sum())} rows should have been written"
        )
    assert n_written > 0, "the case writes no row"
    print(f"{c['id']}: {n_written} written rows over {len(got)} chips exact, every other row unchanged")

    _, x_b = _input(mesh_device, c, c["seed"] + 1)

    def run(xt):
        cache.restore()
        return _call(c, cache.t, xt, meta)

    # (``out`` is the cache itself: A run 1 is snapshotted from it before any restore)
    determinism.assert_deterministic(lambda: run(x_t), lambda: run(x_b), first=out, label=c["id"])
    cache.restore()  # the restore the repeats relied on really puts the prior content back
    assert all(torch.equal(a, b) for a, b in zip(_per_device(cache.t), cache.before)), "restore missed the prior cache"
