# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dump the per-chip output of the tx_send_rate variant in env TXV (seed 0) to results/out_<TXV>.pt, plus its
PCC vs the fp64 reference, so variants can be checked for bit-identity offline (compare_outs.py)."""
import os, sys
import pytest, ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tx_op import a2a_wkv_b2
from test_tx_accept import MESH_SHAPE, _make_inputs, _expected


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X}], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_probe_tx_dump(mesh_device):
    import torch

    chips, attn, w = _make_inputs(mesh_device, 0)
    exp = _expected(mesh_device, chips, w)
    outs = []
    for _ in range(3):  # 3 calls: every one must match
        out = a2a_wkv_b2(attn, w)
        outs.append(torch.stack([ttnn.to_torch(t).to(torch.float32) for t in ttnn.get_device_tensors(out)]))
        ttnn.deallocate(out)
    e = torch.stack(exp).flatten().double()
    a = outs[0].flatten().double()
    pcc = torch.corrcoef(torch.stack([a, e]))[0, 1].item()
    name = os.environ.get("TXV", "base")
    torch.save(outs, os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"out_{name}.pt"))
    print(f"\nTX_DUMP {name}: pcc={pcc:.6f} calls_identical={all(torch.equal(outs[0], o) for o in outs)}")
