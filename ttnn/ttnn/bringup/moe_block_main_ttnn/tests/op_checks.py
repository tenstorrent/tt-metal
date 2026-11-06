# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks the MoE block op tests share: determinism folded on the device, and device time per op from the real-time
program profiler (no Tracy build, no CSVs).

* DeterminismMarker: the first call's outputs kept on the device; every later call's outputs compared to them with
  ttnn.ne (as bf16: max has no uint32 path) and folded into one running max - one scalar a chip goes to the host at
  the end (tests/ttnn/nightly/unit_tests/operations/sdpa/test_sparse_sdpa.py test_sparse_sdpa_determinism's pattern).
* op_device_times: the device time of each op of a run, the op found by a path in its kernels' sources (the programs a
  run dispatches, merged over the chips: the slowest chip's time).
* assert_not_slower: a perf gate that fails on a slowdown past the margin and only reports a speedup."""

import os
import statistics
import sys
from pathlib import Path

from loguru import logger

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # (the repo root: tests.ttnn.profiling)
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged  # noqa: E402

# the MoE block's ops, each by a path its program's kernel sources contain
BLOCK_OPS = {
    "fused_head": "moe_block/fused_head/device/kernels",
    "sparse_dispatch_2d": "moe_block/sparse_dispatch_2d/device/kernels",
    "routed_experts_combine": "moe_block/routed_experts_combine/device/kernels",
    "fused_tail": "moe_block/fused_tail/device/kernels",
}


class DeterminismMarker:
    """Bit-identity of named outputs across calls, kept on the device."""

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.ref = None
        self.marker = None
        self.calls = 0

    def add(self, outputs: dict):
        """One call's outputs (name -> device tensor); the first call's are cloned as the reference."""
        if self.ref is None:
            self.ref = {name: ttnn.clone(t) for name, t in outputs.items()}
            self.calls = 1
            return
        for name, t in outputs.items():
            m = ttnn.max(ttnn.ne(self.ref[name], t, dtype=ttnn.bfloat16))  # (0 iff bit-identical)
            self.marker = m if self.marker is None else ttnn.maximum(self.marker, m)
        self.calls += 1

    def assert_identical(self, what: str):
        assert self.marker is not None, f"{what}: determinism needs at least two calls"
        # (max returns a 0-d tensor a chip: read each chip's scalar - a mesh composer cannot concat 0-d tensors)
        multi = hasattr(self.mesh_device, "get_num_devices") and self.mesh_device.get_num_devices() > 1
        shards = ttnn.get_device_tensors(self.marker) if multi else [self.marker]
        mismatch = max(float(ttnn.to_torch(t).float().max()) for t in shards)
        assert mismatch == 0.0, f"{what}: outputs differ between calls (over {self.calls} calls)"
        logger.info(f"{what}: bit-identical over {self.calls} calls")
        for t in self.ref.values():
            ttnn.deallocate(t)


# A lead of device work ahead of a measured run (MOE_PERF_LEAD_MATMULS matmuls, 0: none): the host has enqueued
# everything the run dispatches while the device is still busy on the lead, so no measured program waits on host
# dispatch (a program whose cores, or a chip of a mesh op, start late shows that wait as device time). The lead's
# programs land in "other".
LEAD_MATMULS = int(os.environ.get("MOE_PERF_LEAD_MATMULS", "0"))


def device_lead(mesh_device, n: int = None):
    """Enqueue n long matmuls (4096^3, bf16) on every chip without waiting on them; returns the operand (the caller
    deallocates it after its run) or None."""
    n = LEAD_MATMULS if n is None else n
    if n <= 0:
        return None
    import torch

    multi = hasattr(mesh_device, "get_num_devices") and mesh_device.get_num_devices() > 1
    a = ttnn.from_torch(
        torch.randn(1, 1, 4096, 4096) * 0.01,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if multi else None,
    )
    for _ in range(n):
        ttnn.deallocate(ttnn.matmul(a, a))
    return a


class MeshBarrier:
    """An all-gather over each mesh axis, put ahead of the measured ops in a trace: every chip starts them together.
    Without it the trace's first cross-chip op takes the chips' start skew as its slowest chip's time (the MoE block's
    head: 230 us traced vs 202 us, 877 us after 56 ms of matmuls on each chip - GLM 8x4 5120, Nov 5); with it the skew
    lands in the barrier ("other"). Needs a bank of L1_SMALL for its semaphores."""

    def __init__(self, mesh_device):
        import torch

        self.tok = ttnn.from_torch(
            torch.zeros(1, 1, 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def __call__(self):
        for axis in (0, 1):
            ttnn.deallocate(ttnn.all_gather(self.tok, dim=3, cluster_axis=axis))

    def close(self):
        ttnn.deallocate(self.tok)


def op_device_times(mesh_device, run_fn, ops: dict = BLOCK_OPS, lead: int = None):
    """run_fn's programs on the real-time profiler: {op: [device ns of each of its programs, in dispatch order]}
    (each program's slowest chip), plus the programs that matched no op under "other"."""

    def run():
        a = device_lead(mesh_device, lead)
        run_fn()
        if a is not None:
            ttnn.deallocate(a)

    _, per_program = profile_realtime_program_merged(mesh_device, run)
    out = {op: [] for op in ops}
    out["other"] = []
    for entry in per_program.values():
        srcs = [s.replace("\\", "/") for s in entry["kernel_sources"]]
        op = next((op for op, path in ops.items() if any(path in s for s in srcs)), "other")
        out[op].append(entry["duration_ns"])
    return out


def traced_op_device_times(
    mesh_device, trace_id, replays: int, ops: dict = BLOCK_OPS, after_replay=None, lead: int = None
):
    """Each of `replays` executions of a captured trace on its own profiler window (a trace's programs keep one runtime
    id, so replays in one window merge into one record): {op: [device ns a replay]}. after_replay(): e.g. fold the
    replay's outputs into a DeterminismMarker (outside the timed program set: its own programs land in "other")."""
    out = {op: [] for op in ops}
    out["other"] = []
    for _ in range(replays):

        def one():
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            if after_replay is not None:
                after_replay()

        t = op_device_times(mesh_device, one, ops, lead)
        for op, v in t.items():
            out[op].extend(v)
    return out


def median_us(ns_list):
    return statistics.median(ns_list) / 1e3 if ns_list else float("nan")


def assert_not_slower(label: str, measured_us: float, expected_us: float, margin: float):
    """Fails past expected * (1 + margin); a run faster than expected * (1 - margin) passes with a note to re-baseline."""
    upper, lower = expected_us * (1 + margin), expected_us * (1 - margin)
    logger.info(f"PERF {label}: {measured_us:.1f} us (expected {expected_us:.1f} +{margin:.0%})")
    assert measured_us <= upper, f"{label}: {measured_us:.1f} us is slower than {upper:.1f} us"
    if measured_us < lower:
        logger.warning(f"PERF {label}: {measured_us:.1f} us is faster than {lower:.1f} us - lower the baseline")
