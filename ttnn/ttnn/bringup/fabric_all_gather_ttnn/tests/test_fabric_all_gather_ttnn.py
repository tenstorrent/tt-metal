# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.fabric_all_gather against its torch semantics (reference.py), one random-input case per captured call
(cases.py). Pure data movement: every element of every chip's output is compared exactly (bits), including, for a
partial gather (gathered_dim_size), the bytes outside each rank's prefix, which must keep the preallocated output's
random fill. Every chip's input differs, so a chip gathered into the wrong group or slot fails. Don't-care: nothing.

The op's own unit suite (tests/unit/) checks it against ttnn.experimental.high_bw_all_gather."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import determinism

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_fabric_all_gather_ttnn_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASES = _load("cases").CASES


def _device_params(c):
    p = dict(c["device_params"])
    p["fabric_config"] = getattr(ttnn.FabricConfig, p["fabric_config"])
    if "fabric_payload_bytes" in p:  # the model's fabric packet payload (harness.device_params)
        rc = ttnn._ttnn.fabric.FabricRouterConfig()
        rc.max_packet_payload_size_bytes = int(p.pop("fabric_payload_bytes"))
        p["fabric_router_config"] = rc
    # A case runs on a box of its own mesh size only: a smaller mesh opened on a bigger box fails the FABRIC_2D router
    # handshake, and the groups depend on the mesh.
    p["require_exact_physical_num_devices"] = True
    return p


def _put(mesh_device, host, spec, mapper):
    return ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=getattr(ttnn.DataType, spec["dtype"]),
        layout=getattr(ttnn.Layout, spec["layout"]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


_BITS = {"BFLOAT16": torch.int16, "FLOAT32": torch.int32, "UINT32": torch.int32}


def _random(shape, dtype, g):
    """Random values in the captured dtype: bf16 / fp32 normal, uint32 random 31-bit words (as torch int32)."""
    if dtype == "UINT32":
        return torch.randint(0, 2**31 - 1, shape, generator=g, dtype=torch.int32)
    return torch.randn(shape, generator=g).to(torch.float32 if dtype == "FLOAT32" else torch.bfloat16)


def _semaphores(mesh_device, c):
    """The caller-owned pair the model passes (ready, data_valid): zero, the whole worker grid, L1_SMALL."""
    if not c.get("semaphores"):
        return {}
    g = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
    s0, s1 = [ttnn.create_global_semaphore(mesh_device, crs, 0, ttnn.BufferType.L1_SMALL) for _ in range(2)]
    return {"ready_semaphore": s0, "data_valid_semaphore": s1}


def _inputs(mesh_device, c, seed):
    """Host per-device inputs (row-major list) and the output's prior fill; the device input, chip (r, c) holding
    xs[r * cols + c]. Distribution as the model declares it: along an axis, 2D (chip (r, c) = host[r, c]); over the
    whole mesh, `dim` sharded over every chip in row-major order (what the op validates)."""
    rows, cols = c["mesh"]
    g = torch.Generator().manual_seed(seed)
    _, _, n, w = c["input"]["shape"]
    host = _random((rows, cols, n, w), c["input"]["dtype"], g)
    fill = _random(c["output"]["shape"], c["input"]["dtype"], g)
    xs = [host[r, cc].reshape(c["input"]["shape"]) for r in range(rows) for cc in range(cols)]
    if c["cluster_axis"] is None:
        tt_in = _put(
            mesh_device, torch.cat(xs, dim=c["dim"]), c["input"], ttnn.ShardTensorToMesh(mesh_device, dim=c["dim"])
        )
    else:
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1))
        tt_in = _put(mesh_device, host, c["input"], mapper)
    assert list(tt_in.shape) == c["input"]["shape"]
    return xs, fill, tt_in


def _call(mesh_device, c, tt_in, fill, sems=None):
    """One call into a fresh output holding `fill` on every chip."""
    out = _put(mesh_device, fill, c["input"], ttnn.ReplicateTensorToMesh(mesh_device))
    kw = {"gathered_dim_size": c["gathered_dim_size"]} if c.get("gathered_dim_size") is not None else {}
    kw.update(sems or {})
    res = ttnn.bringup.fabric_all_gather(
        tt_in, dim=c["dim"], output_tensor=out, cluster_axis=c["cluster_axis"], num_links=c["num_links"], **kw
    )
    return res if res is not None else out


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in CASES],
    ids=[c["id"] for c in CASES],
    indirect=["mesh_device", "device_params"],
)
def test_fabric_all_gather_ttnn(mesh_device, device_params, case):
    c = case
    assert c["exact"]
    rows, cols = c["mesh"]
    sems = _semaphores(mesh_device, c)
    xs, fill, tt_in = _inputs(mesh_device, c, c["seed"])
    out = _call(mesh_device, c, tt_in, fill, sems)
    outs = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
    assert len(outs) == rows * cols
    for d in range(rows * cols):
        want = ref.all_gather(
            xs,
            (rows, cols),
            d,
            dim=c["dim"],
            cluster_axis=c["cluster_axis"],
            fill=fill,
            gathered_dim_size=c.get("gathered_dim_size"),
        )
        got = outs[d].reshape(want.shape).to(want.dtype)
        bits = _BITS[c["input"]["dtype"]]
        bad = got.view(bits) != want.view(bits)
        n_bad = int(bad.sum())
        assert n_bad == 0, f"dev {d}: {n_bad}/{want.numel()} elements differ, first at {bad.nonzero()[0].tolist()}"
    _, fill_b, tt_in_b = _inputs(mesh_device, c, c["seed"] + 1)
    determinism.assert_deterministic(
        lambda: _call(mesh_device, c, tt_in, fill, sems),
        lambda: _call(mesh_device, c, tt_in_b, fill_b, sems),
        first=out,
        label=c["id"],
    )
