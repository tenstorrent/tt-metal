# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.prepare_chunk_recurrence against its torch semantics (reference.py), one random-input case per captured
call (cases.py). Math: for each of the seven outputs, per device, every value finite, PCC and the relative L2 error
against the float64 reference on the same bf16 inputs; t_inv also a max abs error over its strictly-lower triangle
(the source op's test does the same). Every chip's inputs differ. Don't-care: nothing (no actual_end, so no padded
chunks).

Inputs: q, k, v ~ N(0, 1); the gate g = sigmoid(N(mu_h, 1.5^2)) in (0, 1) with mu_h from -4 (head 0) to 3 (the last
head): slow heads keep final_decay = exp(G_last) well inside the float range (G_last ~ -6 per chunk), fast heads put
gate_scale * g near the -5 bound (G_last ~ -150, GLM's fast-decay heads) where the fork's precise_gate_factors matters
and exp(G) underflows to 0 (in the reference's float64 it is merely tiny: an absolute error below 1e-38). The beta
source holds N(0, 1) logits (every column random; the op reads only its H columns)."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import determinism

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"bringup_prepare_chunk_recurrence_tests_{name}", _HERE / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASES = _load("cases").CASES


def _device_params(c):
    p = dict(c["device_params"])
    p["fabric_config"] = getattr(ttnn.FabricConfig, p["fabric_config"])
    # A case runs on a box of its own mesh size only: a smaller mesh opened on a bigger box fails the FABRIC_2D router
    # handshake, and the op's sequence-parallel rank comes from the mesh.
    p["require_exact_physical_num_devices"] = True
    return p


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _inputs(mesh_device, c, seed):
    """Host (q, k, v, g, beta source) with one leading entry per device, and their device tensors (device d holds
    entry d, row-major)."""
    rows, cols = c["mesh"]
    n_dev = rows * cols
    T, H, K, V = c["rows"], c["num_heads"], c["key_dim"], c["value_dim"]
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(n_dev, T, H * K, generator=gen).to(torch.bfloat16)
    k = torch.randn(n_dev, T, H * K, generator=gen).to(torch.bfloat16)
    v = torch.randn(n_dev, T, H * V, generator=gen).to(torch.bfloat16)
    mu = torch.linspace(-4.0, 3.0, H).repeat_interleave(K)  # per head: slow (G_last ~ -6) .. fast (~ -150) decay
    g = torch.sigmoid(mu + 1.5 * torch.randn(n_dev, T, H * K, generator=gen)).to(torch.bfloat16)
    beta = torch.randn(n_dev, T, c["beta_columns"], generator=gen).to(torch.bfloat16)
    host = (q, k, v, g, beta)
    dev = tuple(
        ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        for t in host
    )
    assert list(dev[0].shape) == [1, T, H * K] and list(dev[4].shape) == [1, T, c["beta_columns"]]
    return host, dev


def _call(mesh_device, c, dev, start):
    ck = c["compute_kernel_config"]
    return ttnn.bringup.prepare_chunk_recurrence(
        *dev,
        c["num_heads"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, ck["math_fidelity"]),
            math_approx_mode=ck["math_approx_mode"],
            fp32_dest_acc_en=ck["fp32_dest_acc_en"],
            packer_l1_acc=ck["packer_l1_acc"],
            dst_full_sync_en=ck["dst_full_sync_en"],
        ),
        output_bf16_mask=c["output_bf16_mask"],
        actual_start=start,
        actual_end=None,
        sequence_parallel_axis=c["sequence_parallel_axis"],
        gate_scale=c["gate_scale"],
        beta_logits_column_offset=c["beta_logits_column_offset"],
        precise_gate_factors=c["precise_gate_factors"],
    )


@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in CASES],
    ids=[c["id"] for c in CASES],
    indirect=["mesh_device", "device_params"],
)
def test_prepare_chunk_recurrence(mesh_device, device_params, case):
    c = case
    rows, cols = c["mesh"]
    n_dev = rows * cols
    H, K, V, N = c["num_heads"], c["key_dim"], c["value_dim"], c["rows"] // ref.CHUNK
    # actual_start: a replicated [1, 1] uint32 ROW_MAJOR scalar, as captured
    start = ttnn.from_torch(
        torch.tensor([[c["actual_start"]]], dtype=torch.int64),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    host, dev = _inputs(mesh_device, c, c["seed"])
    outs = _call(mesh_device, c, dev, start)
    assert len(outs) == 7
    shapes = {
        "v_beta": [H, N, 32, V],
        "kd": [H, N, 32, K],
        "q_decay": [H, N, 32, K],
        "intra": [H, N, 32, 32],
        "k_dec_t": [H, N, K, 32],
        "final_decay": [H, N, K, 1],
        "t_inv": [H, N, 32, 32],
    }
    got_all = []
    for i, (name, o) in enumerate(zip(ref.OUTPUT_NAMES, outs)):
        want_dtype = ttnn.bfloat16 if c["output_bf16_mask"] & (1 << i) else ttnn.float32
        assert o.dtype == want_dtype and o.layout == ttnn.TILE_LAYOUT, f"{name}: {o.dtype} {o.layout}"
        assert list(o.shape) == shapes[name], f"{name}: shape {list(o.shape)}"
        got_all.append([ttnn.to_torch(t).double() for t in ttnn.get_device_tensors(o)])
    lower = torch.tril_indices(32, 32, offset=-1)
    for d in range(n_dev):
        want_all = ref.prepare_chunk_recurrence(
            *(t[d : d + 1] for t in host),
            num_heads=H,
            gate_scale=c["gate_scale"],
            beta_logits_column_offset=c["beta_logits_column_offset"],
        )
        for i, name in enumerate(ref.OUTPUT_NAMES):
            got, want = got_all[i][d].reshape(shapes[name]), want_all[i]
            tol = c["tol"][name]
            assert bool(torch.isfinite(got).all()), f"dev {d} {name}: non-finite values"
            pcc = _pcc(got, want)
            rel = float((got - want).norm() / want.norm())
            msg = f"dev {d} {name}: pcc {pcc:.7f} rel {rel:.3e}"
            if name == "t_inv":
                lo = float((got[..., lower[0], lower[1]] - want[..., lower[0], lower[1]]).abs().max())
                msg += f" strict-lower max abs {lo:.3e}"
            print(msg)
            assert pcc >= tol["pcc"] and rel <= tol["rel"], f"{msg} (limits pcc {tol['pcc']}, rel {tol['rel']})"
            if name == "t_inv":
                assert lo <= tol["lower_max_abs"], f"{msg} (limit {tol['lower_max_abs']})"
    _, dev_b = _inputs(mesh_device, c, c["seed"] + 1)
    determinism.assert_deterministic(
        lambda: _call(mesh_device, c, dev, start),
        lambda: _call(mesh_device, c, dev_b, start),
        first=outs,
        label=c["id"],
    )
