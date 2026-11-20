# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 16 guard configs off the FLUX.2 focus set: device ns per config (signposts `cmp|<id>|conv2d_3x3|<i>`), PCC vs
torch and an output sha1 (GUARD_HASH_OUT jsonl), for A/B runs of two trees. Run with `scripts/run_safe_pytest.sh --profile
--run-all`, parse with perf_experiments/asym_padding/cmp_windows.py."""

import hashlib
import json
import os

import pytest
import ttnn
from tracy import signpost

from ttnn.operations.conv2d_3x3 import conv2d_3x3, prepare_conv2d_bias, prepare_conv2d_weights

# (id, H, W, C_in, C_out, layout, weight dtype, bias, relu, padding)
CONFIGS = [
    ("rm258x512_64to256_bf8w_nobias_relu", 258, 512, 64, 256, "RM", "bf8", False, True, (0, 0, 1, 1)),
    ("rm258x512_64to256_bf8w_nobias_relu_p1", 256, 512, 64, 256, "RM", "bf8", False, True, 1),
    ("rm258x512_64to256_bf16_bias", 258, 512, 64, 256, "RM", "bf16", True, False, (0, 0, 1, 1)),
    ("rm514x1024_32to128_nobias_relu", 514, 1024, 32, 128, "RM", "bf16", False, True, (0, 0, 1, 1)),
    ("rm130x256_64to64_bias", 130, 256, 64, 64, "RM", "bf16", True, False, 1),
    ("tile1026x2048_32to128_bias", 1026, 2048, 32, 128, "TILE", "bf16", True, False, (0, 0, 1, 1)),
    ("tile514x1024_64to256_bias", 514, 1024, 64, 256, "TILE", "bf16", True, False, (0, 0, 1, 1)),
    ("tile258x512_64to256_bf8w_nobias_relu", 258, 512, 64, 256, "TILE", "bf8", False, True, (0, 0, 1, 1)),
]
_ONLY = {s for s in os.environ.get("GUARD_CASES", "").split(",") if s}
ITERS = 3


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("cfg", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_guard_cfg(device, cfg):
    import torch

    cid, H, W, c_in, c_out, layout, wdt, has_bias, relu, padding = cfg
    if _ONLY and cid not in _ONLY:
        pytest.skip("not selected")
    torch.manual_seed(0)
    x = torch.randn(H, W, c_in, dtype=torch.bfloat16)
    w = (torch.randn(c_out, c_in, 3, 3) / (9 * c_in) ** 0.5).to(torch.bfloat16)
    bvec = (0.1 * torch.randn(c_out)).to(torch.bfloat16)
    in_layout = ttnn.ROW_MAJOR_LAYOUT if layout == "RM" else ttnn.TILE_LAYOUT
    wdtype = ttnn.bfloat8_b if wdt == "bf8" else ttnn.bfloat16
    tt_x = ttnn.from_torch(
        x.reshape(1, 1, H * W, c_in),
        dtype=ttnn.bfloat16,
        layout=in_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    kw = dict(
        input_dtype=ttnn.bfloat16,
        input_layout=in_layout,
        weight_dtype=wdtype,
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_w = prepare_conv2d_weights(w, device=device, padding=padding, **kw)
    tt_b = prepare_conv2d_bias(bvec, device=device, **kw) if has_bias else None

    def run():
        return conv2d_3x3(
            tt_x,
            tt_w,
            input_height=H,
            input_width=W,
            bias=tt_b,
            padding=padding,
            activation="relu" if relu else None,
        )

    out = run()
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    for i in range(ITERS):
        ttnn.deallocate(out)
        signpost(header=f"cmp|{cid}|conv2d_3x3|{i}")
        out = run()
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
    signpost(header=f"cmpend|{cid}")
    got = ttnn.to_torch(out)
    sha = hashlib.sha1(got.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
    top, bottom, left, right = padding if isinstance(padding, tuple) else (padding,) * 4
    xin = torch.nn.functional.pad(x.float().permute(2, 0, 1).unsqueeze(0), (left, right, top, bottom))
    ref = torch.nn.functional.conv2d(
        xin, ttnn.to_torch(ttnn.from_torch(w, dtype=wdtype)).float(), bvec.float() if has_bias else None
    )
    if relu:
        ref = torch.relu(ref)
    ref = ref[0].permute(1, 2, 0).reshape(-1, c_out)
    got = got.reshape(-1, got.shape[-1])[: ref.shape[0], :c_out].float()
    pcc = torch.corrcoef(torch.stack([got.flatten().double(), ref.flatten().double()]))[0, 1].item()
    path = os.environ.get("GUARD_HASH_OUT")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"id": cid, "sha1": sha, "pcc": pcc}) + "\n")
    assert pcc > 0.99, pcc
