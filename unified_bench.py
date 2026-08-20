# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-time benchmarking for unified kernels, and for any ttnn op to compare against.

Uses metal's real-time profiler, which streams a ProgramRealtimeRecord per completed program
over the existing dispatch path: start and end timestamps plus the device frequency, so the
number is DEVICE time for the program and excludes host dispatch entirely. Each record also
carries kernel_sources, which is how a measurement is attributed to the op that produced it.

That matters here more than usual. A unified kernel at these sizes runs in microseconds, and
host dispatch is tens of microseconds -- wall-clock timing would measure the dispatcher.

    from unified_bench import bench
    stats = bench(device, lambda: ttnn.generic_op(tensors, program), iters=50, match="flash")
    print(stats)
"""

import statistics
import threading

import ttnn


class _Collector:
    """Accumulates ProgramRealtimeRecords. Callbacks run concurrently, so it locks."""

    def __init__(self):
        self.lock = threading.Lock()
        self.rows = []
        self.dropped = 0

    def __call__(self, batch):
        with self.lock:
            self.dropped += batch.dropped
            for r in batch.records:
                self.rows.append((r.start_timestamp, r.end_timestamp, r.frequency, tuple(r.kernel_sources)))

    def take(self):
        with self.lock:
            rows, self.rows = self.rows, []
            return rows


def _us(start, end, freq):
    """frequency is cycles per ns, so cycles / freq is ns."""
    return (end - start) / freq / 1000.0


class Bench:
    """Collects device timings for everything run inside the `with` block."""

    def __init__(self):
        self.collector = _Collector()
        self.handle = None

    def __enter__(self):
        if not ttnn.device.IsProgramRealtimeProfilerActive():
            raise RuntimeError(
                "the real-time profiler is inactive on this dispatch setup, so device timings "
                "are unavailable -- see tech_reports/real_time_profiler/getting-started.md"
            )
        self.handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(self.collector)
        return self

    def __exit__(self, *exc):
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(self.handle)
        return False

    def records(self, match=None):
        """Rows as microseconds, optionally keeping only those whose kernel_sources mention
        `match` -- the op's kernel filename, which is what separates our program from ttnn's."""
        out = []
        for start, end, freq, sources in self.collector.rows:
            if match is not None and not any(match in s for s in sources):
                continue
            out.append(_us(start, end, freq))
        return out


def bench(device, call, iters=50, warmup=5, match=None):
    """Run `call` iters times and report device microseconds per program.

    The first calls are discarded: they build and cache the program. generic_op keys its cache
    on the program descriptor's hash, so later calls reuse it and measure execution alone.
    """
    for _ in range(warmup):
        call()
    ttnn.synchronize_device(device)

    with Bench() as b:
        for _ in range(iters):
            call()
        ttnn.synchronize_device(device)
        ttnn.device.ReadDeviceProfiler(device)
        samples = b.records(match)
        dropped = b.collector.dropped

    if not samples:
        raise RuntimeError(
            f"no device records matched {match!r}. Records seen: "
            f"{len(b.collector.rows)}; check the match string against a kernel path."
        )
    samples.sort()
    return {
        "n": len(samples),
        "min_us": samples[0],
        "median_us": statistics.median(samples),
        "mean_us": statistics.fmean(samples),
        "max_us": samples[-1],
        "dropped": dropped,
    }


def show(label, stats, extra=""):
    print(
        f"{label:34s} median={stats['median_us']:8.2f}us  min={stats['min_us']:8.2f}us  "
        f"n={stats['n']:3d}{'  dropped=' + str(stats['dropped']) if stats['dropped'] else ''}"
        f"{('  ' + extra) if extra else ''}"
    )
