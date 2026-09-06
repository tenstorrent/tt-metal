# Shared helpers for the overnight MoE FFN benchmarks (scratch, not for commit).
import json
import math
import os
import statistics
import time

import torch
from loguru import logger

import ttnn

SCR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(SCR, "results")
os.makedirs(RESULTS, exist_ok=True)

TILE = 32
TILE_BYTES = {"bf4": 576, "bf8": 1088, "bf16": 2048}
DTYPE = {"bf4": ttnn.bfloat4_b, "bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}
MODELS = {  # emb, hidden, activation name
    "kimi": dict(emb=7168, hidden=2048, act="silu"),
    "m3": dict(emb=6144, hidden=3072, act="swigluoai"),
    # 7-DRAM-bank-compatible stand-ins for band-mode validation on the local P100
    # (tile counts divisible by 7): Kimi-like 224x70 tiles, M3-like 196x98 tiles.
    "kimi7": dict(emb=7168, hidden=2240, act="silu"),
    "m3_7": dict(emb=6272, hidden=3136, act="swigluoai"),
}
OP_KERNEL_DIR = "/unified_routed_expert_ffn/"


def open_dev(trace_region_size=0):
    dev = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
        l1_small_size=24576,
        trace_region_size=trace_region_size,
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )
    return dev


def env_info(dev):
    g = dev.compute_with_storage_grid_size()
    info = {
        "grid_x": g.x,
        "grid_y": g.y,
        "dram_channels": dev.dram_grid_size().x,
        "rt_profiler": bool(ttnn.device.IsProgramRealtimeProfilerActive()),
    }
    try:
        cores = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(dev, ttnn.NOC.NOC_0)
        info["bank_cores_noc0"] = [(c.x, c.y) for c in cores]
        cores1 = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(dev, ttnn.NOC.NOC_1)
        info["bank_cores_noc1"] = [(c.x, c.y) for c in cores1]
    except Exception as e:  # noqa: BLE001
        info["bank_cores_err"] = repr(e)
    return info


class RtProfile:
    """Collect realtime-profiler records (duration_ns + frequency GHz) for a window of device work."""

    def __init__(self, dev, run_fn, timeout_s=10.0, settle_s=0.2):
        self.records = []
        self.dropped = 0

        def cb(batch):
            self.dropped += int(batch.dropped)
            for r in batch.records:
                s, e, f = int(r.start_timestamp), int(r.end_timestamp), float(r.frequency)
                if f <= 0 or e <= s:
                    continue
                self.records.append(
                    dict(
                        runtime_id=int(r.runtime_id),
                        chip=int(r.chip_id),
                        ns=(e - s) / f,
                        ghz=f,
                        kernels=tuple(str(k) for k in r.kernel_sources),
                    )
                )

        handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(cb)
        try:
            self.result = run_fn()
            ttnn.synchronize_device(dev)
            deadline = time.monotonic() + timeout_s
            last_n, last_t = 0, time.monotonic()
            while time.monotonic() < deadline:
                n = len(self.records)
                if n != last_n:
                    last_n, last_t = n, time.monotonic()
                elif n and time.monotonic() - last_t >= settle_s:
                    break
                time.sleep(0.01)
        finally:
            ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)
        if self.dropped:
            raise RuntimeError(f"RT profiler dropped {self.dropped} records")

    def durations(self, kernel_substr):
        per = {}
        for r in self.records:
            if any(kernel_substr in k.replace("\\", "/") for k in r["kernels"]):
                d = per.setdefault(r["runtime_id"], dict(ns=0.0, ghz=r["ghz"]))
                d["ns"] = max(d["ns"], r["ns"])
        # Records are delivered asynchronously: a late record from a program that ran BEFORE the
        # window (e.g. the warm-up) can land inside it. Order by runtime_id so callers can keep the
        # most recent `iters` programs.
        ids = sorted(per)
        return [per[i]["ns"] for i in ids], [per[i]["ghz"] for i in ids]


def timed(dev, fn, kernel_substr, iters=3, warmup=1):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(dev)
    prof = RtProfile(dev, lambda: [fn() for _ in range(iters)])
    ns, ghz = prof.durations(kernel_substr)
    if len(ns) < iters:
        seen = sorted({k.rsplit("/", 1)[-1] for r in prof.records for k in r["kernels"]})
        raise RuntimeError(f"expected {iters} programs matching {kernel_substr}, got {len(ns)}; kernels seen: {seen}")
    ns, ghz = ns[-iters:], ghz[-iters:]
    return dict(ns=statistics.median(ns), ns_all=ns, ghz=statistics.median(ghz) if ghz else None)


def append_jsonl(name, rec):
    rec = dict(rec)
    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(RESULTS, name), "a") as f:
        f.write(json.dumps(rec) + "\n")


# ----------------------------------------------------------------------------- expert FFN harness
def ceil32(n):
    return (n + TILE - 1) // TILE * TILE


def expert_weight_bytes(emb, hidden, dtype_key):
    tiles = 3 * (emb // TILE) * (hidden // TILE)
    return tiles * TILE_BYTES[dtype_key]


def torch_expert_ref(x, w, act):
    import torch.nn.functional as F

    gate = F.linear(x, w["gate_proj"])
    up = F.linear(x, w["up_proj"])
    if act == "silu":
        a = F.silu(gate) * up
    elif act == "swigluoai":
        gate_c = gate.clamp(max=7.0)
        up_c = up.clamp(min=-7.0, max=7.0)
        a = (up_c + 1.0) * (gate_c * torch.sigmoid(1.702 * gate_c))
    else:
        raise ValueError(act)
    return F.linear(a, w["down_proj"])


ACT_ENUM = {"silu": ttnn.RoutedExpertActivation.Silu, "swigluoai": ttnn.RoutedExpertActivation.SwiGluOai}


class MultiExpertCase:
    """N experts on one chip with per-expert token counts, production-style region packing.

    Builds weights + shared dispatch buffer once; `run()` dispatches the op; `check()` PCCs per expert.
    """

    def __init__(
        self,
        dev,
        counts,
        model,
        dtype_key="bf4",
        x_row_major=True,
        max_tokens=5120,
        seed=123,
        slack_rows=32,
        weight_scale=0.02,
        **ffn_kwargs,
    ):
        from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert

        torch.manual_seed(seed)
        m = MODELS[model]
        self.emb, self.hidden, self.act = m["emb"], m["hidden"], m["act"]
        self.counts = list(counts)
        self.E = len(counts)
        self.dtype_key = dtype_key
        self.x_row_major = x_row_major
        self.max_tokens = max_tokens
        # production-style packing: regions 32-row aligned, back to back, plus slack after each region
        offs, cur = [], 0
        for c in self.counts:
            offs.append(cur)
            cur += ceil32(c) + slack_rows
        self.offsets = offs
        # The op requires the buffer to hold >= max_tokens rows (m_tiles <= rows/32), as the
        # production dispatch buffer always does.
        rows = max(cur, max_tokens, TILE)
        self.rows = rows
        self.weights = [
            {
                "gate_proj": torch.randn(self.hidden, self.emb) * weight_scale,
                "up_proj": torch.randn(self.hidden, self.emb) * weight_scale,
                "down_proj": torch.randn(self.emb, self.hidden) * weight_scale,
            }
            for _ in range(self.E)
        ]
        buf = torch.randn(rows, self.emb)  # random slack: must never be read
        self.inputs = []
        for e, c in enumerate(self.counts):
            x = torch.randn(c, self.emb) if c > 0 else torch.zeros(0, self.emb)
            self.inputs.append(x)
            if c > 0:
                buf[offs[e] : offs[e] + c] = x
        self.buf_torch = buf
        if x_row_major:
            self.tt_buf = ttnn.from_torch(
                buf,
                mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=dev,
                dtype=ttnn.bfloat16,
            )
        else:
            self.tt_buf = ttnn.from_torch(
                buf,
                mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                dtype=ttnn.bfloat8_b,
            )

        def idx(vals):
            return ttnn.from_torch(
                torch.tensor(vals, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32
            )

        self.tt_counts = idx(self.counts)
        self.tt_offsets = idx(self.offsets)
        self.expert = TtRoutedExpert(
            mesh_device=dev,
            experts_per_chip=self.E,
            global_expert_idx_table=idx(list(range(self.E))),
            emb_dim=self.emb,
            hidden_dim=self.hidden,
            max_tokens=max_tokens,
            torch_weights=self.weights,
            activations_dtype=ttnn.bfloat8_b,
            weights_dtype=DTYPE[dtype_key],
            activation=ACT_ENUM[self.act],
            **ffn_kwargs,
        )
        self.out = None

    def run(self):
        self.out = self.expert(self.tt_buf, self.tt_counts, self.tt_offsets)
        return self.out

    def bytes_and_flops(self, chunk_tokens):
        """Weight bytes actually streamed (one full read per M-chunk of `chunk_tokens`) and FLOPs."""
        wb = expert_weight_bytes(self.emb, self.hidden, self.dtype_key)
        total_b, total_f = 0, 0
        for c in self.counts:
            if c <= 0:
                continue
            chunks = math.ceil(c / chunk_tokens)
            total_b += chunks * wb
            total_f += 6 * c * self.emb * self.hidden
        return total_b, total_f

    def check(self, pcc_threshold=0.97):
        from tests.ttnn.utils_for_testing import comp_pcc

        out = ttnn.to_torch(self.out, mesh_composer=ttnn.ConcatMeshToTensor(self.out.device(), dim=0))
        res = []
        for e, c in enumerate(self.counts):
            if c <= 0:
                continue
            ref = torch_expert_ref(self.inputs[e], self.weights[e], self.act)
            got = out[self.offsets[e] : self.offsets[e] + c].float()
            ok, pcc = comp_pcc(ref, got, pcc_threshold)
            nan = bool(torch.isnan(got).any() or torch.isinf(got).any())
            res.append(dict(expert=e, count=c, pcc=float(pcc), ok=bool(ok) and not nan, nan=nan))
        return res
