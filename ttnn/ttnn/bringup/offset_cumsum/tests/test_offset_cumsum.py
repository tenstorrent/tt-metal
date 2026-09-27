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
    g = torch.Generator().manual_seed(c["seed"])

    # hists[r, col, :] is the histogram of device (r, col); random counts, about 1 in 8 experts empty.
    hists = torch.randint(0, c["max_count"] + 1, (rows, cols, E), generator=g, dtype=torch.int64)
    hists[torch.rand(rows, cols, E, generator=g) < 0.125] = 0
    if c["local_experts_only"]:  # a device's masked_bincount counts only the experts of its dispatch group
        assert c["cluster_axis"] == 0, "local_experts_only assumes the dispatch groups are the mesh columns"
        grp = torch.arange(E) // epc
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

    outs = ttnn.bringup.offset_cumsum(
        tt_h,
        cluster_axis=c["cluster_axis"],
        num_links=c["num_links"],
        experts_per_chip=epc,
        memory_config=_mem(c["memory_config"]),
    )
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
