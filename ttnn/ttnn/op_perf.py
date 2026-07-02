# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
One-command op-level performance summary with automatic bottleneck identification.

Provides ``ttnn.op_perf.profile()`` -- a single-call API that runs a model
function, captures per-op wall-clock timing and host-device transfer cost,
then produces a ranked performance summary with bottleneck hints.

Usage::

    import ttnn

    # Profile a model function
    result, report = ttnn.op_perf.profile(my_model, input_tensor)
    print(report)

    # Or as a context manager for finer control
    with ttnn.op_perf.perf_trace() as trace:
        output = my_model(input_tensor)
    print(trace.report())

    # Export to JSON/CSV
    trace.report().to_json("perf_report.json")
    trace.report().to_csv("perf_report.csv")

    # CLI: python -m ttnn.op_perf --help

Resolves: https://github.com/tenstorrent/tt-metal/issues/36650
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple

logger = logging.getLogger(__name__)

# ── Path Sanitization (Cycode SAST compliance) ───────────────────────────

# Base directories that file I/O is restricted to
_ALLOWED_BASE_DIRS = [
    os.path.abspath(os.getcwd()),
    os.path.abspath(os.path.expanduser("~")),
]


def _safe_resolve_path(user_path: str) -> str:
    """
    Validate and resolve a user-supplied file path.

    Ensures the resolved absolute path falls within one of the allowed base
    directories (cwd or home). Raises ValueError on directory traversal.

    This uses the ``os.path.abspath`` + ``str.startswith`` pattern required
    by Cycode SAST scanners.
    """
    resolved = os.path.abspath(user_path)
    for base in _ALLOWED_BASE_DIRS:
        if resolved.startswith(base):
            return resolved
    raise ValueError(
        f"Path '{user_path}' resolves to '{resolved}' which is outside "
        f"allowed directories: {_ALLOWED_BASE_DIRS}"
    )


# ── Data Structures ──────────────────────────────────────────────────────


@dataclasses.dataclass
class OpRecord:
    """A single recorded operation invocation."""

    op_name: str
    duration_us: float  # wall-clock microseconds (host-side, device-synced)
    input_shapes: List[Tuple[int, ...]]
    input_dtypes: List[str]
    output_shapes: List[Tuple[int, ...]]
    call_index: int  # sequential invocation order
    is_transfer: bool = False  # True for host↔device ops (from_torch, to_torch, etc.)

    @property
    def duration_ms(self) -> float:
        return self.duration_us / 1000.0


@dataclasses.dataclass
class AggregatedOp:
    """Per-op-type aggregate statistics."""

    op_name: str
    count: int
    total_us: float
    min_us: float
    max_us: float
    mean_us: float
    pct_of_total: float  # percentage of total profiled time

    @property
    def total_ms(self) -> float:
        return self.total_us / 1000.0


@dataclasses.dataclass
class BottleneckHint:
    """An automatic suggestion for performance improvement."""

    severity: str  # "high", "medium", "low"
    message: str
    op_name: str
    detail: str


class OpPerfReport:
    """
    Structured performance report produced by ``profile()`` or ``perf_trace()``.

    Contains per-invocation records, aggregated statistics, bottleneck
    suggestions, and export methods for JSON, CSV, and human-readable output.
    """

    def __init__(
        self,
        records: List[OpRecord],
        total_time_us: float,
    ):
        self.records = records
        self.total_time_us = total_time_us
        self._aggregated: Optional[List[AggregatedOp]] = None
        self._hints: Optional[List[BottleneckHint]] = None

    # ── Aggregation ──────────────────────────────────────────────────

    @property
    def aggregated(self) -> List[AggregatedOp]:
        """Aggregate records by op name, sorted by total time descending."""
        if self._aggregated is not None:
            return self._aggregated

        by_name: Dict[str, List[float]] = defaultdict(list)
        for rec in self.records:
            by_name[rec.op_name].append(rec.duration_us)

        result = []
        for name, durations in by_name.items():
            total = sum(durations)
            result.append(
                AggregatedOp(
                    op_name=name,
                    count=len(durations),
                    total_us=total,
                    min_us=min(durations),
                    max_us=max(durations),
                    mean_us=total / len(durations),
                    pct_of_total=(total / self.total_time_us * 100) if self.total_time_us > 0 else 0.0,
                )
            )
        result.sort(key=lambda a: a.total_us, reverse=True)
        self._aggregated = result
        return self._aggregated

    def top_ops(self, n: int = 10) -> List[AggregatedOp]:
        """Return the top N slowest op types by total time."""
        return self.aggregated[:n]

    @property
    def transfer_time_us(self) -> float:
        """Total time spent on host↔device transfers."""
        return sum(r.duration_us for r in self.records if r.is_transfer)

    @property
    def compute_time_us(self) -> float:
        """Total time spent on compute ops (non-transfer)."""
        return sum(r.duration_us for r in self.records if not r.is_transfer)

    @property
    def transfer_pct(self) -> float:
        """Percentage of total time spent on transfers."""
        if self.total_time_us <= 0:
            return 0.0
        return self.transfer_time_us / self.total_time_us * 100

    # ── Bottleneck Detection ─────────────────────────────────────────

    @property
    def bottleneck_hints(self) -> List[BottleneckHint]:
        """Automatically generated optimization suggestions."""
        if self._hints is not None:
            return self._hints

        hints = []
        agg = self.aggregated

        # Hint 1: dominant op (>50% of total)
        if agg and agg[0].pct_of_total > 50:
            top = agg[0]
            hints.append(
                BottleneckHint(
                    severity="high",
                    message=f"{top.op_name} dominates runtime ({top.pct_of_total:.0f}%)",
                    op_name=top.op_name,
                    detail=f"{top.count} calls, {top.total_ms:.2f}ms total, "
                    f"{top.mean_us:.0f}us avg. Consider fusing, sharding, or "
                    f"using a more optimal program config.",
                )
            )

        # Hint 2: high transfer overhead (>20% of total)
        if self.transfer_pct > 20:
            hints.append(
                BottleneckHint(
                    severity="high",
                    message=f"Host-device transfers consume {self.transfer_pct:.0f}% of runtime",
                    op_name="data_transfer",
                    detail="Move tensor creation and weight loading outside the "
                    "hot loop. Use ttnn.from_torch() once and keep data on device.",
                )
            )

        # Hint 3: high-variance op (max > 3x mean, called >3 times)
        for a in agg:
            if a.count >= 3 and a.max_us > 3 * a.mean_us:
                hints.append(
                    BottleneckHint(
                        severity="medium",
                        message=f"{a.op_name} has high latency variance (max {a.max_us:.0f}us vs mean {a.mean_us:.0f}us)",
                        op_name=a.op_name,
                        detail="First call may include compilation overhead. "
                        "Consider warmup runs or trace-based execution.",
                    )
                )

        # Hint 4: many small ops (>50 ops each <10us)
        small_ops = [r for r in self.records if r.duration_us < 10]
        if len(small_ops) > 50:
            hints.append(
                BottleneckHint(
                    severity="medium",
                    message=f"{len(small_ops)} ops took <10us each -- host dispatch overhead may dominate",
                    op_name="dispatch_overhead",
                    detail="Consider op fusion or traced execution to amortize "
                    "host-side dispatch cost.",
                )
            )

        # Hint 5: transfer ops dominate top-5 list
        transfer_in_top5 = sum(1 for a in agg[:5] if _is_transfer_op(a.op_name))
        if transfer_in_top5 >= 3:
            hints.append(
                BottleneckHint(
                    severity="high",
                    message="Data movement ops dominate the top-5 slowest operations",
                    op_name="data_movement",
                    detail="Profile is transfer-bound. Restructure to minimize "
                    "host-device and L1-DRAM data movement.",
                )
            )

        self._hints = hints
        return self._hints

    # ── Display ──────────────────────────────────────────────────────

    def summary(self, top_n: int = 10, file: Optional[TextIO] = None) -> str:
        """
        Generate a human-readable performance summary.

        Returns the summary string and optionally writes to *file*.
        """
        lines = []
        w = 80  # column width

        lines.append("=" * w)
        lines.append("  TTNN Op Performance Summary".center(w))
        lines.append("=" * w)
        lines.append("")
        lines.append(f"  Total profiled time:    {self.total_time_us / 1000:.2f} ms")
        lines.append(f"  Total ops recorded:     {len(self.records)}")
        lines.append(f"  Unique op types:        {len(self.aggregated)}")
        lines.append(f"  Compute time:           {self.compute_time_us / 1000:.2f} ms ({100 - self.transfer_pct:.0f}%)")
        lines.append(f"  Transfer time:          {self.transfer_time_us / 1000:.2f} ms ({self.transfer_pct:.0f}%)")
        lines.append("")

        # Top N table
        lines.append(f"  Top {top_n} Slowest Operations")
        lines.append("  " + "-" * (w - 4))
        header = f"  {'Rank':<5} {'Op Name':<40} {'Calls':>5} {'Total(ms)':>10} {'Avg(us)':>10} {'%':>6}"
        lines.append(header)
        lines.append("  " + "-" * (w - 4))

        for i, a in enumerate(self.top_ops(top_n), 1):
            name = a.op_name
            if len(name) > 38:
                name = name[:35] + "..."
            lines.append(f"  {i:<5} {name:<40} {a.count:>5} {a.total_ms:>10.2f} {a.mean_us:>10.0f} {a.pct_of_total:>5.1f}%")

        lines.append("")

        # Bottleneck hints
        hints = self.bottleneck_hints
        if hints:
            lines.append("  Bottleneck Analysis")
            lines.append("  " + "-" * (w - 4))
            for h in hints:
                severity_icon = {"high": "[!!]", "medium": "[!]", "low": "[i]"}.get(h.severity, "[?]")
                lines.append(f"  {severity_icon} [{h.severity.upper()}] {h.message}")
                lines.append(f"     -> {h.detail}")
                lines.append("")
        else:
            lines.append("  [OK] No major bottlenecks detected.")
            lines.append("")

        lines.append("=" * w)

        text = "\n".join(lines)
        if file is not None:
            file.write(text)
        return text

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return f"OpPerfReport(ops={len(self.records)}, total={self.total_time_us / 1000:.2f}ms, unique={len(self.aggregated)})"

    # ── Export ────────────────────────────────────────────────────────

    def to_json(self, path: Optional[str] = None) -> str:
        """Export report to JSON. Returns JSON string; writes to *path* if given."""
        data = {
            "summary": {
                "total_time_us": self.total_time_us,
                "total_time_ms": self.total_time_us / 1000,
                "total_ops": len(self.records),
                "unique_op_types": len(self.aggregated),
                "compute_time_us": self.compute_time_us,
                "transfer_time_us": self.transfer_time_us,
                "transfer_pct": self.transfer_pct,
            },
            "top_ops": [
                {
                    "op_name": a.op_name,
                    "count": a.count,
                    "total_us": a.total_us,
                    "mean_us": a.mean_us,
                    "min_us": a.min_us,
                    "max_us": a.max_us,
                    "pct_of_total": a.pct_of_total,
                }
                for a in self.aggregated
            ],
            "records": [
                {
                    "call_index": r.call_index,
                    "op_name": r.op_name,
                    "duration_us": r.duration_us,
                    "input_shapes": [list(s) for s in r.input_shapes],
                    "input_dtypes": r.input_dtypes,
                    "output_shapes": [list(s) for s in r.output_shapes],
                    "is_transfer": r.is_transfer,
                }
                for r in self.records
            ],
            "bottleneck_hints": [
                {
                    "severity": h.severity,
                    "message": h.message,
                    "op_name": h.op_name,
                    "detail": h.detail,
                }
                for h in self.bottleneck_hints
            ],
        }
        text = json.dumps(data, indent=2, default=str)
        if path is not None:
            # Cycode SAST: inline abspath + startswith sanitization
            safe_path = os.path.abspath(path)
            _base = os.path.abspath(os.getcwd())
            if not safe_path.startswith(_base):
                _base = os.path.abspath(os.path.expanduser("~"))
            if not safe_path.startswith(_base):
                raise ValueError(f"Path escapes allowed directories: {safe_path}")
            Path(safe_path).write_text(text, encoding="utf-8")
            logger.info("Perf report saved to %s", safe_path)
        return text

    def to_csv(self, path: str) -> None:
        """Export per-op records to CSV."""
        # Cycode SAST: inline abspath + startswith sanitization
        safe_path = os.path.abspath(path)
        _base = os.path.abspath(os.getcwd())
        if not safe_path.startswith(_base):
            _base = os.path.abspath(os.path.expanduser("~"))
        if not safe_path.startswith(_base):
            raise ValueError(f"Path escapes allowed directories: {safe_path}")
        with open(safe_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["call_index", "op_name", "duration_us", "duration_ms", "input_shapes", "input_dtypes", "is_transfer"]
            )
            for r in self.records:
                writer.writerow(
                    [
                        r.call_index,
                        r.op_name,
                        f"{r.duration_us:.1f}",
                        f"{r.duration_ms:.3f}",
                        str(r.input_shapes),
                        str(r.input_dtypes),
                        r.is_transfer,
                    ]
                )
        logger.info("Perf CSV saved to %s (%d records)", safe_path, len(self.records))


# ── Transfer Op Classification ───────────────────────────────────────────

_TRANSFER_OP_KEYWORDS = frozenset(
    {
        "from_torch",
        "to_torch",
        "to_device",
        "from_device",
        "to_layout",
        "typecast",
        "clone",
        "to_memory_config",
    }
)


def _is_transfer_op(op_name: str) -> bool:
    """Classify an op as a host-device transfer or data-movement op."""
    base = op_name.rsplit(".", 1)[-1].lower()
    return base in _TRANSFER_OP_KEYWORDS


# ── Tensor Introspection Helpers ─────────────────────────────────────────


def _extract_tensor_info(args, kwargs) -> Tuple[List[Tuple[int, ...]], List[str]]:
    """Extract shapes and dtypes from TTNN tensor arguments."""
    shapes = []
    dtypes = []
    try:
        import ttnn as _ttnn

        def _visit(obj):
            if isinstance(obj, _ttnn.Tensor):
                try:
                    shapes.append(tuple(obj.shape))
                except Exception:
                    shapes.append(())
                try:
                    dtypes.append(str(obj.dtype))
                except Exception:
                    dtypes.append("unknown")
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _visit(item)

        for arg in args:
            _visit(arg)
        for val in kwargs.values():
            _visit(val)
    except Exception:
        pass
    return shapes, dtypes


def _extract_output_info(output) -> List[Tuple[int, ...]]:
    """Extract shapes from TTNN tensor output."""
    shapes = []
    try:
        import ttnn as _ttnn

        def _visit(obj):
            if isinstance(obj, _ttnn.Tensor):
                try:
                    shapes.append(tuple(obj.shape))
                except Exception:
                    shapes.append(())
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _visit(item)

        _visit(output)
    except Exception:
        pass
    return shapes


# ── Core Profiling Engine ────────────────────────────────────────────────


class PerfTrace:
    """
    Accumulates per-op timing records using TTNN's operation hook system.

    Use as a context manager::

        with PerfTrace() as trace:
            output = model(x)
        print(trace.report())
    """

    def __init__(self):
        self._records: List[OpRecord] = []
        self._call_counter = 0
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        # Per-op timing state (set in pre_hook, consumed in post_hook)
        self._pending_op_name: Optional[str] = None
        self._pending_start: Optional[float] = None
        self._pending_shapes: List[Tuple[int, ...]] = []
        self._pending_dtypes: List[str] = []

    def _pre_hook(self, operation, args, kwargs):
        """Pre-operation hook: record op name, sync device, start timer."""
        import ttnn as _ttnn

        self._pending_op_name = getattr(operation, "python_fully_qualified_name", str(operation))
        self._pending_shapes, self._pending_dtypes = _extract_tensor_info(args, kwargs)

        # Synchronize to get accurate wall-clock timing
        try:
            for arg in args:
                if isinstance(arg, _ttnn.Tensor) and _ttnn.is_tensor_storage_on_device(arg) and arg.is_allocated():
                    for dev in arg.devices():
                        _ttnn.synchronize_device(dev)
                    break
        except Exception:
            pass

        self._pending_start = time.perf_counter()
        return None  # hooks must return None

    def _post_hook(self, operation, args, kwargs, output):
        """Post-operation hook: sync device, stop timer, record."""
        import ttnn as _ttnn

        # Synchronize ALL devices in the output to capture full execution time
        try:
            synced_devices = set()

            def _sync_all(obj):
                if isinstance(obj, _ttnn.Tensor) and _ttnn.is_tensor_storage_on_device(obj) and obj.is_allocated():
                    for dev in obj.devices():
                        dev_id = id(dev)
                        if dev_id not in synced_devices:
                            _ttnn.synchronize_device(dev)
                            synced_devices.add(dev_id)
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        _sync_all(item)

            _sync_all(output)
        except Exception:
            pass

        end = time.perf_counter()
        start = self._pending_start or end
        op_name = self._pending_op_name or "unknown"
        duration_us = (end - start) * 1e6

        output_shapes = _extract_output_info(output)

        self._records.append(
            OpRecord(
                op_name=op_name,
                duration_us=duration_us,
                input_shapes=self._pending_shapes,
                input_dtypes=self._pending_dtypes,
                output_shapes=output_shapes,
                call_index=self._call_counter,
                is_transfer=_is_transfer_op(op_name),
            )
        )
        self._call_counter += 1

        # Reset pending state
        self._pending_op_name = None
        self._pending_start = None
        self._pending_shapes = []
        self._pending_dtypes = []

        return None  # hooks must return None

    def __enter__(self):
        import ttnn as _ttnn

        self._records.clear()
        self._call_counter = 0
        # Register hooks via the public context-manager API so hook
        # lifetime is properly scoped and unwound in LIFO order.
        self._hook_stack = contextlib.ExitStack()
        self._hook_stack.enter_context(
            _ttnn.decorators.register_pre_operation_hook(self._pre_hook)
        )
        self._hook_stack.enter_context(
            _ttnn.decorators.register_post_operation_hook(self._post_hook)
        )
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_time = time.perf_counter()
        # Unwind hooks in LIFO order via the ExitStack.
        hook_stack = getattr(self, "_hook_stack", None)
        if hook_stack is not None:
            hook_stack.close()
            self._hook_stack = None
        return False  # do not suppress exceptions

    @property
    def total_time_us(self) -> float:
        """Total wall-clock time of the profiled region in microseconds."""
        if self._start_time is None or self._end_time is None:
            return sum(r.duration_us for r in self._records)
        return (self._end_time - self._start_time) * 1e6

    def report(self) -> OpPerfReport:
        """Build an ``OpPerfReport`` from the collected records."""
        return OpPerfReport(
            records=list(self._records),
            total_time_us=self.total_time_us,
        )


# Convenience alias
perf_trace = PerfTrace


def profile(fn: Callable, *args, **kwargs) -> Tuple[Any, OpPerfReport]:
    """
    Profile a callable and return ``(result, report)``.

    This is the primary single-command API requested in issue #36650::

        result, report = ttnn.op_perf.profile(my_model, input_tensor)
        print(report)

    Args:
        fn: The function or model to profile.
        *args: Positional arguments forwarded to *fn*.
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        A tuple of (fn's return value, OpPerfReport).
    """
    with PerfTrace() as trace:
        result = fn(*args, **kwargs)
    return result, trace.report()


# ── CLI Entry Point ──────────────────────────────────────────────────────


def _cli_main():
    """CLI: ``python -m ttnn.op_perf``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="TTNN Op Performance Profiler -- one-command performance summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile from a saved report
  python -m ttnn.op_perf --input perf_report.json --top 20

  # Export summary as CSV
  python -m ttnn.op_perf --input perf_report.json --csv perf.csv
        """,
    )
    parser.add_argument("--input", type=str, help="Load a previously saved JSON report")
    parser.add_argument("--top", type=int, default=10, help="Number of top ops to show (default: 10)")
    parser.add_argument("--json", type=str, default=None, help="Export report to JSON file")
    parser.add_argument("--csv", type=str, default=None, help="Export records to CSV file")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress summary output")

    args = parser.parse_args()

    if args.input:
        # Cycode SAST: inline abspath + startswith sanitization
        safe_input = os.path.abspath(args.input)
        _base = os.path.abspath(os.getcwd())
        if not safe_input.startswith(_base):
            _base = os.path.abspath(os.path.expanduser("~"))
        if not safe_input.startswith(_base):
            raise ValueError(f"Path escapes allowed directories: {safe_input}")
        data = json.loads(Path(safe_input).read_text(encoding="utf-8"))
        records = [
            OpRecord(
                op_name=r["op_name"],
                duration_us=r["duration_us"],
                input_shapes=[tuple(s) for s in r.get("input_shapes", [])],
                input_dtypes=r.get("input_dtypes", []),
                output_shapes=[tuple(s) for s in r.get("output_shapes", [])],
                call_index=r.get("call_index", i),
                is_transfer=r.get("is_transfer", False),
            )
            for i, r in enumerate(data.get("records", []))
        ]
        total = data.get("summary", {}).get("total_time_us", sum(r.duration_us for r in records))
        report = OpPerfReport(records=records, total_time_us=total)
    else:
        parser.print_help()
        print("\nNote: To profile a model, use ttnn.op_perf.profile(fn, *args) in Python.")
        return

    if not args.quiet:
        print(report.summary(top_n=args.top))

    if args.json:
        report.to_json(args.json)
        print(f"JSON report saved to {args.json}")

    if args.csv:
        report.to_csv(args.csv)
        print(f"CSV report saved to {args.csv}")


if __name__ == "__main__":
    _cli_main()
