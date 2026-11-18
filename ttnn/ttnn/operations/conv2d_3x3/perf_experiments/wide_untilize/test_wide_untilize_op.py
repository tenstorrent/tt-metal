# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""wide_untilize end to end: a private copy of conv2d_3x3 (op_copy/) whose compute kernel takes the untilize of
>= 2-tile-wide TILE slabs from a compile-time knob (op_copy/conv2d_3x3_program_descriptor.py WIDE_UNTILIZE):
  base    0: compute_kernel_lib::untilize (the committed path)
  group   1: wide_untilize, CB sync per section group
  hybrid  2: wide_untilize, input synced per batch, output reserved per batch and pushed per group
  batch   3: wide_untilize, CB sync per batch
  lazy    4: wide_untilize, input waited per group and popped per batch, output reserved per batch, pushed per group

Each case runs every selected variant on the same inputs in one process: the outputs must be bit-identical to base
(torch.equal) and meet the case's PCC vs torch. DEVICE KERNEL DURATION of the op's one program is measured in process
(one dispatch per variant; WU_ORDER=rev runs the variants in reverse order) and appended to results_op.jsonl.

    WU_PROFILE=1 WU_CASES=semantic,aspp_convs_1 WU_VARIANTS=base,hybrid \\
      scripts/run_safe_pytest.sh --run-all <this file> -p no:cacheprovider --import-mode=prepend

Never run this copy next to the real op in one profiled process (zone-id hash collisions abort the profile).
"""

import os

if os.environ.get("WU_PROFILE") == "1":
    os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
    os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
    os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
    os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).parent
RESULTS = HERE / "results_op.jsonl"
OPC = "wide_untilize_op_copy"
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _load():
    if OPC not in sys.modules:
        root = HERE / "op_copy"
        spec = importlib.util.spec_from_file_location(OPC, root / "__init__.py", submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[OPC] = module
        spec.loader.exec_module(module)
    return sys.modules[OPC], sys.modules[OPC + ".conv2d_3x3_program_descriptor"]


opc, pd = _load()

VARIANTS = {"base": 0, "group": 1, "hybrid": 2, "batch": 3, "lazy": 4}
HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def _row_major_cores(num_cores, grid_x=11):
    full_rows, rem = divmod(num_cores, grid_x)
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full_rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


# HEIGHT harness TILE layers (tests/.../test_conv2d_3x3_perf.py HEIGHT_SHAPES): bf8_b act + weights, bias + ReLU,
# the op's default compute config. (layer, ((H, W, C_in), C_out, padding, stride), shard_shape, num_cores)
HEIGHT_TILE = {
    "stem_conv3": (((256, 512, 64), 128, 1, 1), [1216, 64], 110),
    "sem_res2_fuse_conv0": (((128, 256, 288), 256, 1, 1), [320, 288], 103),
    "inst_res2_fuse_conv0": (((128, 256, 160), 128, 1, 1), [320, 160], 103),
    "res3_conv2": (((64, 128, 128), 128, 1, 1), [96, 128], 86),
    "sem_res3_fuse_conv0": (((64, 128, 320), 256, 1, 1), [96, 320], 86),
}
# DRAM-interleaved TILE (test_conv2d_3x3_perf PERF_SHAPES): bf16 weights, bias, no activation, default config.
INTERLEAVED = {
    "tile_bf8b_H64_W128_Ci128": ((64, 128, 128), 128, ttnn.bfloat8_b),
    "tile_bf16_H64_W128_Ci128": ((64, 128, 128), 128, ttnn.bfloat16),
    "tile_bf8b_H8_W16_Ci512": ((8, 16, 512), 256, ttnn.bfloat8_b),
}


def _focus_cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    return {
        c["extras"]["layer"].split(" ")[0].replace(".", "_"): c for c in LOOSE_CASES if "[perf]" in c["extras"]["layer"]
    }


def _build(case_id, device):
    """-> (run(), reference, pcc threshold): run() dispatches the op copy once and returns the torch output."""
    import torch
    from eval.golden_tests.conv2d_3x3 import helpers

    torch.manual_seed(0)
    focus = _focus_cases()
    if case_id in focus:
        case = focus[case_id]
        (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
        extras = case["extras"]
        dtype, weight_dtype, layout = case["dtype"], case["weight_dtype"], case["layout"]
        mem = helpers.build_input_memory_config(
            (1, 1, H * W, c_in), case["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
        )
        if mem.shard_spec is None or list(mem.shard_spec.shape) != list(extras["shard_shape"]):
            pytest.skip("pinned shard does not fit")
        has_bias = case["bias_mode"] == "with_bias"
        activation = None if case["activation"] == "none" else case["activation"]
        torch_weight = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
        torch_bias = helpers.make_bias(c_out, dtype=torch.bfloat16) if has_bias else None
        torch_input = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
        config = helpers.make_compute_kernel_config(
            case["fp32_dest_acc_en"], extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
        )
        pcc = extras.get("tolerance", (0.99, None))[0]
    elif case_id in HEIGHT_TILE:
        ((H, W, c_in), c_out, padding, stride), shard, cores = HEIGHT_TILE[case_id]
        dilation, dtype, weight_dtype, layout = 1, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT
        mem = ttnn.MemoryConfig(
            HS, ttnn.BufferType.L1, ttnn.ShardSpec(_row_major_cores(cores), shard, ttnn.ShardOrientation.ROW_MAJOR)
        )
        activation, config, pcc = "relu", None, 0.99
        torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
        torch_bias = torch.randn(c_out).to(torch.bfloat16)
        torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    else:
        (H, W, c_in), c_out, dtype = INTERLEAVED[case_id]
        padding, stride, dilation, weight_dtype, layout = 1, 1, 1, ttnn.bfloat16, ttnn.TILE_LAYOUT
        mem, activation, config = ttnn.DRAM_MEMORY_CONFIG, None, None
        pcc = 0.995 if dtype == ttnn.bfloat16 else 0.99
        torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
        torch_bias = torch.randn(c_out).to(torch.bfloat16)
        torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)

    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=weight_dtype,
        input_memory_config=mem,
        device=device,
    )
    tt_weight = opc.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = opc.prepare_conv2d_bias(torch_bias, **kwargs) if torch_bias is not None else None
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=mem)

    def run():
        out = opc.conv2d_3x3(
            tt_input,
            tt_weight,
            input_height=H,
            input_width=W,
            bias=tt_bias,
            padding=padding,
            stride=stride,
            dilation=dilation,
            activation=activation,
            compute_kernel_config=config,
        )
        return out

    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(
        x,
        torch_weight.float(),
        None if torch_bias is None else torch_bias.float(),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if activation == "relu":
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    return run, ref, pcc


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total, found = 0.0, False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _selected(env, default):
    v = os.environ.get(env)
    return [x for x in v.split(",") if x] if v else default


CASES = _selected(
    "WU_CASES",
    ["semantic", "aspp_convs_1", "aspp_convs_2", "res5_1_conv2", "res5_2_conv2", "res4_0_conv2"],
)
SELECTED = _selected("WU_VARIANTS", ["base", "hybrid"])


@pytest.mark.parametrize("case_id", CASES)
def test_wide_untilize_op(case_id, device, monkeypatch):
    import torch

    profile = os.environ.get("WU_PROFILE") == "1"
    run, ref, pcc_min = _build(case_id, device)
    order = list(SELECTED)
    if os.environ.get("WU_ORDER") == "rev":
        order = order[::-1]
    outs, ns = {}, {}
    for v in order:
        monkeypatch.setattr(pd, "WIDE_UNTILIZE", VARIANTS[v])
        out = run()  # compile + correctness dispatch
        outs[v] = ttnn.to_torch(out).float().reshape(ref.shape)
        if profile:
            _read_kernel_ns(device)  # flush
            run()
            ns[v] = _read_kernel_ns(device)
    lines = [f"{case_id}:"]
    failures = []
    base_out = outs.get("base")
    for v in SELECTED:
        pcc = _pcc(outs[v], ref)
        same = base_out is None or bool(torch.equal(outs[v], base_out))
        t = ns.get(v)
        speed = f"{ns['base'] / t:6.3f}x" if (t and ns.get("base")) else ""
        lines.append(
            f"  {v:<8} {t if t else float('nan'):>10.0f} ns {speed}  pcc {pcc:.6f}  "
            f"{'bit-identical to base' if same else 'DIFFERS from base'}"
        )
        if not same:
            failures.append(f"{case_id}/{v}: differs from base (max {float((outs[v] - base_out).abs().max())})")
        if pcc < pcc_min:
            failures.append(f"{case_id}/{v}: pcc {pcc} < {pcc_min}")
        with RESULTS.open("a") as f:
            f.write(
                json.dumps(
                    dict(
                        case=case_id,
                        variant=v,
                        ns=t,
                        pcc=pcc,
                        identical=same,
                        order=os.environ.get("WU_ORDER", "fwd"),
                        tag=os.environ.get("WU_TAG", ""),
                        box=socket.gethostname(),
                    )
                )
                + "\n"
            )
    print("\n" + "\n".join(lines))
    assert not failures, failures
