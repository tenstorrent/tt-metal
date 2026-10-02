# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.offset_cumsum against its torch semantics (reference.py), one random-input case per captured call
(cases.py). Integer prefix sums: all three outputs (offsets, totals, regions) are compared exactly, every element."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import determinism

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_offset_cumsum_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASES = _load("cases").CASES


def _device_params(c):
    p = dict(c["device_params"])
    p["fabric_config"] = getattr(ttnn.FabricConfig, p["fabric_config"])
    # A case runs on a box of its own mesh size only (conftest skips it elsewhere): a smaller mesh opened on a bigger
    # box fails the FABRIC_2D router handshake (e.g. a 2x2 case on a 4x2 box), and the case's math depends on its mesh.
    p["require_exact_physical_num_devices"] = True
    return p


def _mem(name):
    return {"DRAM": ttnn.DRAM_MEMORY_CONFIG, "L1": ttnn.L1_MEMORY_CONFIG}[name]


@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in CASES],
    ids=[c["id"] for c in CASES],
    indirect=["mesh_device", "device_params"],
)
def test_offset_cumsum(mesh_device, device_params, case):
    c = case
    rows, cols = c["mesh"]
    (E,) = c["hist_shape"]
    epc = c["experts_per_chip"]
    hists, tt_h = _inputs(mesh_device, c, c["seed"])
    outs = _call(c, tt_h)
    assert len(outs) == 4, len(outs)
    per_dev = [[ttnn.to_torch(t).reshape(-1).to(torch.int64) for t in ttnn.get_device_tensors(o)] for o in outs]
    for dev in range(rows * cols):
        r, col = divmod(dev, cols)
        # The dispatch group of device (r, col): the devices along cluster_axis.
        group = hists[:, col, :] if c["cluster_axis"] == 0 else hists[r, :, :]
        pos = r if c["cluster_axis"] == 0 else col
        offsets, totals, regions = ref.offset_cumsum(group, epc)
        for name, want, got in zip(("offsets", "totals", "regions"), (offsets[pos], totals, regions), per_dev):
            got = got[dev]
            assert got.shape == want.shape, (name, dev, got.shape)
            bad = got != want
            assert not bad.any(), f"dev {dev} {name}: {int(bad.sum())}/{E} differ"
        # all_global_dispatch_offsets (#57859): row k is what device k of the group receives in `offsets`.
        got = per_dev[3][dev]
        assert got.numel() == offsets.numel(), ("all_offsets", dev, got.shape)
        bad = got != offsets.reshape(-1)
        assert not bad.any(), f"dev {dev} all_offsets: {int(bad.sum())}/{offsets.numel()} differ"
    tt_h_b = _inputs(mesh_device, c, c["seed"] + 1)[1]
    determinism.assert_deterministic(lambda: _call(c, tt_h), lambda: _call(c, tt_h_b), first=outs, label=c["id"])


def _call(c, tt_h):
    return ttnn.bringup.offset_cumsum(
        tt_h,
        cluster_axis=c["cluster_axis"],
        num_links=c["num_links"],
        experts_per_chip=c["experts_per_chip"],
        memory_config=_mem(c["memory_config"]),
    )


def _inputs(mesh_device, c, seed):
    """The host histograms [rows, cols, E] for `seed` and the device histogram tensor."""
    rows, cols = c["mesh"]
    (E,) = c["hist_shape"]
    epc = c["experts_per_chip"]
    g = torch.Generator().manual_seed(seed)

    # hists[r, col, :] is the histogram of device (r, col); random counts, about 1 in 8 experts empty.
    hists = torch.randint(0, c["max_count"] + 1, (rows, cols, E), generator=g, dtype=torch.int64)
    hists[torch.rand(rows, cols, E, generator=g) < 0.125] = 0
    if c["local_experts_only"]:  # a device's masked_bincount counts only the experts of its dispatch group
        assert c["cluster_axis"] == 0, "local_experts_only assumes the dispatch groups are the mesh columns"
        grp = torch.arange(E) // (epc * rows)  # column col holds experts col*rows*epc .. (ExpertMapping col-major)
        for col in range(cols):
            hists[:, col, grp != col] = 0

    tt_h = ttnn.from_torch(
        hists.reshape(rows, cols * E).to(torch.int32),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, 1)),
        device=mesh_device,
        dtype=getattr(ttnn.DataType, c["hist"]["dtype"]),
        layout=getattr(ttnn.Layout, c["hist"]["layout"]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_h = ttnn.reshape(tt_h, (E,))  # per device [1, E] -> [E], as captured
    assert list(tt_h.shape) == c["hist_shape"]
    return hists, tt_h
