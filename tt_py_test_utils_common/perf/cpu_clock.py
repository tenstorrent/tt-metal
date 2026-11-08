# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run host-timed perf measurements at full CPU clock.

Untraced (eager) device time is mostly host dispatch, and a forward/sync loop blocks for part of each iteration, so a
utilization-driven governor (schedutil) can leave the dispatching core at a low frequency for a long time. On an EPYC
host with acpi-cpufreq (3.25 GHz base with boost to 3.8 GHz, lower states 2.3 and 1.5 GHz) that shows up as discrete
slow levels in host-observed time (same work, main-thread CPU time == wall, 1.5-2x slower). Warm-up does not reliably
remove it.

A minimum utilization clamp (uclamp_min, Linux >= 5.3 with CONFIG_UCLAMP_TASK) makes schedutil request the policy
maximum whenever a clamped thread runs. It needs no privilege and changes no affinity. Other governors ignore it:
`performance` already runs at the maximum; active intel_pstate/amd-pstate (EPP) do not use it.

Usage (pytest):

    clock = hold_full_cpu_clock()
    request.addfinalizer(clock.restore)
    print(f"host: {clock.status}")

or as a context manager: `with hold_full_cpu_clock() as clock: ...`. restore() writes back each thread's previous
uclamp_min value, so a clamp set before (e.g. by uclampset) is kept and holds nested in LIFO order are safe. It restores
the value, not the kernel's "never set" state: a thread that had no clamp ends with its current value set explicitly
(0 for normal threads, the RT default for RT threads), so it no longer follows later changes of the RT default.
Resetting to "never set" needs Linux >= 5.11, so it isn't used. Threads created while restore() runs may miss it and
keep the clamp: hold and restore when the process is not starting threads (e.g. not mid device open/close).
"""

import ctypes
import errno
import os
import threading


class _SchedAttr(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint32),
        ("sched_policy", ctypes.c_uint32),
        ("sched_flags", ctypes.c_uint64),
        ("sched_nice", ctypes.c_int32),
        ("sched_priority", ctypes.c_uint32),
        ("sched_runtime", ctypes.c_uint64),
        ("sched_deadline", ctypes.c_uint64),
        ("sched_period", ctypes.c_uint64),
        ("sched_util_min", ctypes.c_uint32),
        ("sched_util_max", ctypes.c_uint32),
    ]


# (sched_setattr, sched_getattr) syscall numbers
_SYSCALLS = {"x86_64": (314, 315), "aarch64": (274, 275)}.get(os.uname().machine)
_SCHED_FLAG_KEEP_ALL, _SCHED_FLAG_UTIL_CLAMP_MIN = 0x08 | 0x10, 0x20
UCLAMP_MAX = 1024


def current_cpu():
    """The CPU this thread last ran on, or -1."""
    try:
        with open("/proc/thread-self/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[36])  # field 39: processor
    except (OSError, ValueError, IndexError):
        return -1


def cpufreq_governor(cpu=None):
    """scaling_governor of `cpu` (default: the CPU this thread last ran on), or "unknown"."""
    cpu = current_cpu() if cpu is None else cpu
    try:
        with open(f"/sys/devices/system/cpu/cpu{max(cpu, 0)}/cpufreq/scaling_governor") as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def _libc():
    return ctypes.CDLL(None, use_errno=True)


def _set_util_min(libc, tid, value):
    """sched_setattr(uclamp_min) on one thread, keeping its policy and parameters; returns 0 or an errno."""
    attr = _SchedAttr(
        size=ctypes.sizeof(_SchedAttr),
        sched_flags=_SCHED_FLAG_KEEP_ALL | _SCHED_FLAG_UTIL_CLAMP_MIN,
        sched_util_min=value,
    )
    return 0 if libc.syscall(_SYSCALLS[0], tid, ctypes.byref(attr), 0) == 0 else ctypes.get_errno()


def util_min(tid=0):
    """uclamp_min of thread `tid` (0: the calling thread), or None if sched_getattr fails or is unsupported here.

    Kernels without uclamp (CONFIG_UCLAMP_TASK off, or < 5.3) report 0."""
    return _get_util_min(_libc(), tid) if _SYSCALLS else None


def _get_util_min(libc, tid):
    attr = _SchedAttr()
    if libc.syscall(_SYSCALLS[1], tid, ctypes.byref(attr), ctypes.sizeof(attr), 0) != 0:
        return None
    return attr.sched_util_min


def _tids():
    """This process's thread ids, or None if /proc/self/task can't be listed."""
    try:
        return [int(t) for t in os.listdir("/proc/self/task")]
    except (OSError, ValueError):
        return None


def _describe(errors):
    return "; ".join(f"{e} ({', '.join(map(str, t[:4]))})" for e, t in errors.items())


class FullCpuClock:
    """uclamp_min = max on every thread of this process until restore(). See hold_full_cpu_clock().

    Attributes: governor; status (one log line); clamped / total (threads); errors ({reason: [tids]} from clamping);
    restore_errors (same, from the last restore()). "main" in those lists is the thread that created the hold."""

    def __init__(self):
        self.governor = cpufreq_governor()
        self.clamped, self.total, self.errors, self.restore_errors = 0, 0, {}, {}
        self._libc, self._saved, self._seen = None, {}, set()
        self._main = threading.get_native_id()
        if not _SYSCALLS:
            self.status = f"governor {self.governor}; clock clamp unsupported on {os.uname().machine}"
            return
        libc = _libc()
        # Threads created after this point inherit the clamp; restore() gives them this thread's previous value, so
        # nothing is clamped unless it can be read.
        self._default = _get_util_min(libc, self._main)
        tids = _tids() or [self._main]  # without /proc only this thread is known
        self.total = len(tids)
        if self._default is None:
            self._error(self.errors, f"sched_getattr: {os.strerror(ctypes.get_errno())}", self._main)
        else:
            self._libc = libc
            self._seen = set(tids)
            for tid in tids:
                saved = _get_util_min(libc, tid)
                if saved is None:  # previous value unknown: leave the thread unclamped rather than unrestorable
                    if ctypes.get_errno() != errno.ESRCH:
                        self._error(self.errors, f"sched_getattr: {os.strerror(ctypes.get_errno())}", tid)
                    continue
                err = _set_util_min(libc, tid, UCLAMP_MAX)
                if err == 0:
                    self.clamped += 1
                    self._saved[tid] = saved
                elif err != errno.ESRCH:  # ESRCH: the thread exited
                    self._error(self.errors, os.strerror(err), tid)
            if not self.clamped:  # nothing holds the clamp, so restore() has nothing to do
                self._libc = None
        effect = "" if self.governor == "schedutil" else f" (no effect under governor {self.governor})"
        self.status = (
            f"governor {self.governor}; uclamp_min {UCLAMP_MAX} on {self.clamped}/{self.total} threads{effect}"
        )
        if self.errors:
            self.status += "; failed: " + _describe(self.errors)

    def _error(self, errors, reason, tid):
        errors.setdefault(reason, []).append("main" if tid == self._main else tid)

    def restore(self):
        """Write back each clamped thread's previous uclamp_min; threads created since (which inherited the clamp) get
        the creating thread's previous value, and threads that were not clamped are left alone. Idempotent. Child
        processes started meanwhile keep the clamp until they exit. Returns True if every write succeeded (see
        restore_errors)."""
        if self._libc is None:
            return not self.restore_errors
        libc, self._libc = self._libc, None
        self.restore_errors = {}
        tids = _tids()
        # Without /proc, fall back to the threads clamped earlier; with it, a saved tid that is gone may have been
        # reused by another process, so only listed threads are touched.
        for tid in self._saved if tids is None else tids:
            if tid in self._saved:
                value = self._saved[tid]
            elif tid in self._seen:  # existed at clamp time but was not clamped
                continue
            else:
                value = self._default
            err = _set_util_min(libc, tid, value)
            if err not in (0, errno.ESRCH):
                self._error(self.restore_errors, os.strerror(err), tid)
        if self.restore_errors:
            print(f"uclamp_min restore failed: {_describe(self.restore_errors)}")
        return not self.restore_errors

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.restore()


def hold_full_cpu_clock():
    """Ask the cpufreq governor for full clock on every thread of this process while host time is measured.

    Returns a FullCpuClock: `.status` is a one-line description for the log (governor, threads clamped, failures),
    `.restore()` undoes it (idempotent; also on context-manager exit). Never raises: on unsupported hosts it only
    reports (`.clamped == 0`, `.errors`). Call it after device open so the runtime's threads exist; threads created
    later inherit the clamp.
    """
    return FullCpuClock()
