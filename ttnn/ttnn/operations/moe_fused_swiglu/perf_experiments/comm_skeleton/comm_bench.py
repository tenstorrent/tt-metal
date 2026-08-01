# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""comm_skeleton — a COST MODEL for pure communication bookkeeping, measured directly.

THE QUESTION. Ablation arithmetic on moe_fused_swiglu attributes a large slice of its runtime to
work with no payload: circular-buffer reserve/wait/push/pop, semaphore waits and increments, NoC
command issue. Every ablation is still a composite, so that attribution is an inference, not a
measurement. This module measures the primitives THEMSELVES — kernels that do only the
communication pattern, with zero or near-zero payload — so the inference can be checked against
numbers.

WHAT IS MEASURED (one probe per primitive, each in its own kernel, none of them moving real data):

  launch   an EMPTY kernel on the same core set. The fixed per-dispatch intercept that must come
           off every other number before a slope means anything.
  cb       `cb_reserve_back`/`cb_push_back` (producer) and `cb_wait_front`/`cb_pop_front`
           (consumer), swept over iteration count, pages-per-call and CB depth.
  noc      `noc_async_read`/`noc_async_write` swept over COMMAND COUNT at a 32 B payload (issue
           cost) and over PAYLOAD SIZE at fixed count (bandwidth), on NCRISC and BRISC separately,
           with and without a per-command barrier, and with vs. without TensorAccessor addressing.
  sem      `noc_semaphore_inc` point-to-point, the N-way incast, `noc_semaphore_wait_min` on an
           already-satisfied semaphore, and `noc_semaphore_inc_multicast`.
  mcast    the full grid-wide rendezvous — ack incast + data multicast + ready signal — repeated R
           rounds with a near-zero payload, peeled STAGE by STAGE.
  dest     `tile_regs_acquire`/`commit`/`wait`/`release` with no math between them.

HOW TO READ IT. Every probe is swept over a repeat count and reported as a SLOPE (ns per
operation) from an ordinary least-squares fit, with the INTERCEPT reported separately — because
these kernels do almost nothing, a single absolute number at a small repeat count is mostly launch
overhead and says nothing about the primitive.

This module is a plain library (NOT collected by pytest — pytest's --import-mode=importlib cannot
safely collect a test_*.py file living inside the `ttnn/ttnn/operations/...` tree; it derives a
dotted module path starting with "ttnn" and re-executes ttnn/ttnn/__init__.py under a second
qualified name, crashing on duplicate C++ op registration). The pytest entry point that imports it
lives at tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_perfexp_comm_skeleton.py.
"""

import os
from pathlib import Path

# Enable the on-device profiler IN-PROCESS (must be set before the device opens). The pytest entry
# point imports this module before opening the device, so module import time is early enough;
# scoped via setdefault so it never clobbers an outer profiler run.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

# PRIVATE PROFILER ARTIFACTS DIR. `generated/profiler/` (both `.logs/` and `reports/`) is SHARED
# across every process using this clone. When two agents run concurrently the device flock
# serialises execution correctly, but whichever run reaches teardown first CONSUMES
# `profile_log_device.csv` — the loser silently gets no logs, and anything that picks a report
# directory by mtime can read the OTHER run's numbers and never know.
#
# This bench does not read either of those (it takes its numbers from
# `ttnn.get_latest_programs_perf_data()`, which is in-process C++ state on
# `MetalContext::profiler_state_manager()` — see tt_metal/impl/profiler/tt_metal_profiler.cpp
# GetLatestProgramsPerfData — so a sibling's files cannot reach it). The isolation is set anyway,
# for two reasons: it removes the question entirely, and MID_RUN_DUMP means this process WRITES to
# that shared dir, so without it we are a hazard to the siblings even though they are not one to us.
#
# `TT_METAL_PROFILER_DIR` is honoured by both the C++ side
# (tt_metal/impl/profiler/profiler_paths.hpp::get_profiler_artifacts_dir) and the Python side
# (tools/tracy/common.py::PROFILER_ARTIFACTS_DIR), so one variable covers both.
_PRIVATE_PROFILER_DIR = Path(__file__).parent / "profiler_artifacts"
_PRIVATE_PROFILER_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TT_METAL_PROFILER_DIR", str(_PRIVATE_PROFILER_DIR))

# NOTE: `torch` is imported LAZILY. `scripts/validate_no_global_torch_imports.py` forbids a
# module-level torch import anywhere under `ttnn/ttnn/`, and these perf experiments live under the
# op directory, so they obey the same rule.
import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# --- circular-buffer indices (must match the kernels' compile-time constants) ---
CB_PROBE = 0  # cb probe's producer/consumer buffer
CB_SCRATCH = 1  # noc probe's L1 landing/source scratch
CB_MCAST = 2  # mcast rendezvous payload slot
CB_DEST_IN = 3  # dest probe's (unused but required) startup operands
CB_DEST_OUT = 4

# --- semaphore ids ---
SEM_TARGET = 0  # incremented / waited on by the sem probe
SEM_SAT = 1  # pre-satisfied (initial value 1) — the WAIT_SAT role
SEM_FREE = 2  # mcast rendezvous: receiver -> root ack ("my slot is free")
SEM_RDY = 3  # mcast rendezvous: root -> grid ready signal

# --- cb probe roles (kernels/cb_probe.cpp) ---
CB_ROLE_PRODUCER = 0
CB_ROLE_CONSUMER = 1
CB_ROLE_BULK_PRODUCER = 2

# --- cb probe MODES (host-level compositions of those roles) ---
CB_MODE_PROD_DEEP = 0  # producer alone; CB holds the whole run, so it never blocks
CB_MODE_CONS_PREFILL = 1  # producer pushes everything in ONE call; consumer's loop is measured alone
CB_MODE_PAIR_DEEP = 2  # both loop; CB holds the whole run (no back-pressure)
CB_MODE_PAIR_SHALLOW = 3  # both loop; CB is DEPTH deep (the realistic ping-pong)
CB_MODES = {
    CB_MODE_PROD_DEEP: "prod_deep",
    CB_MODE_CONS_PREFILL: "cons_prefill",
    CB_MODE_PAIR_DEEP: "pair_deep",
    CB_MODE_PAIR_SHALLOW: "pair_shallow",
}

# --- noc probe modes (kernels/noc_probe.cpp) ---
NOC_DRAM_READ_ACC = 0
NOC_DRAM_READ_FIXED = 1
NOC_DRAM_WRITE_ACC = 2
NOC_DRAM_WRITE_FIXED = 3
NOC_L1_READ_REMOTE = 4
NOC_L1_WRITE_REMOTE = 5
NOC_MODES = {
    NOC_DRAM_READ_ACC: "dram_read_accessor",
    NOC_DRAM_READ_FIXED: "dram_read_fixed",
    NOC_DRAM_WRITE_ACC: "dram_write_accessor",
    NOC_DRAM_WRITE_FIXED: "dram_write_fixed",
    NOC_L1_READ_REMOTE: "l1_read_remote",
    NOC_L1_WRITE_REMOTE: "l1_write_remote",
}

# --- sem probe roles (kernels/sem_probe.cpp) ---
SEM_ROLE_INC_P2P = 0
SEM_ROLE_INCAST_SENDER = 1
SEM_ROLE_INCAST_ROOT = 2
SEM_ROLE_WAIT_SAT = 3
SEM_ROLE_MCAST_SEM = 4

# The DRAM scratch buffer every NoC probe reads from / writes to. 128 pages x 8192 B = 1 MiB, which
# is enough pages that a command-count sweep never revisits a page inside one run and enough bytes
# per page for the payload-size sweep to reach 8 KiB.
SCRATCH_PAGES = 128
SCRATCH_PAGE_BYTES = 8192


# ---------------------------------------------------------------------------
# Precision contract — mirrors moe_fused_swiglu.default_compute_kernel_config() VERBATIM. Fixed
# input, never a lever. Only the `dest` probe uses it, and it uses the op's own settings so the
# DEST handshake is measured on the machine the op actually configures.
# ---------------------------------------------------------------------------
def compute_kernel_config():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.LoFi
    cfg.math_approx_mode = True
    cfg.fp32_dest_acc_en = False
    cfg.dst_full_sync_en = False
    cfg.bfp8_pack_precise = True
    return cfg


# ---------------------------------------------------------------------------
# grid / core helpers
# ---------------------------------------------------------------------------
def device_grid(device):
    g = device.compute_with_storage_grid_size()
    return int(g.x), int(g.y)


def clamp_grid(device, hgroups, kgroups):
    gx, gy = device_grid(device)
    return min(hgroups, gx), min(kgroups, gy)


def rect(hgroups, kgroups):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(hgroups - 1, kgroups - 1))])


def single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def row_range(hgroups, row):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, row), ttnn.CoreCoord(hgroups - 1, row))])


def virt(device, x, y):
    c = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
    return int(c.x), int(c.y)


def cb(index, core_ranges, num_pages, page_size, data_format=None):
    fmt = data_format if data_format is not None else ttnn.uint32
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page_size)],
    )


# ---------------------------------------------------------------------------
# measurement (/perf-measure discipline: one fresh run per point, no averaging loop)
# ---------------------------------------------------------------------------
def read_kernel_ns(device):
    """DEVICE KERNEL DURATION summed over programs dispatched since the last read."""
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def measure_once(device, run_fn):
    ttnn.synchronize_device(device)
    read_kernel_ns(device)  # drain any pending window
    run_fn()
    ns = read_kernel_ns(device)
    assert ns is not None, "profiler produced no data (profiler-enabled build?)"
    return ns


def fit(xs, ys):
    """Ordinary least squares -> (slope ns/op, intercept ns, r2).

    The intercept is the fixed launch/teardown cost; the slope is the primitive. Reporting only an
    absolute at one repeat count would be reporting mostly the intercept.
    """
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return slope, intercept, r2


def _dummy_io(device):
    """`generic_op` requires at least one input AND one output tensor (io_tensors.size() >= 2).
    Probes that touch no tensor still need a pair."""
    a = ttnn.allocate_tensor_on_device(
        ttnn.Shape([32, 32]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    b = ttnn.allocate_tensor_on_device(
        ttnn.Shape([32, 32]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    return a, b


# ---------------------------------------------------------------------------
# PROBE: launch intercept — an empty kernel on the same core set
# ---------------------------------------------------------------------------
def run_launch(device, hgroups, kgroups, *, both_riscs=True):
    cores = rect(hgroups, kgroups)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "noop.cpp"),
            core_ranges=cores,
            compile_time_args=[],
            runtime_args=ttnn.RuntimeArgs(),
            config=ttnn.ReaderConfigDescriptor(),
        )
    ]
    if both_riscs:
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "noop.cpp"),
                core_ranges=cores,
                compile_time_args=[],
                runtime_args=ttnn.RuntimeArgs(),
                config=ttnn.WriterConfigDescriptor(),
            )
        )
    a, b = _dummy_io(device)
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])
    return ttnn.generic_op([a, b], program)


# ---------------------------------------------------------------------------
# PROBE 1: CB bookkeeping with no data
# ---------------------------------------------------------------------------
CB_PROBE_PAGE_BYTES = 32  # smallest useful L1-aligned page: the payload is irrelevant, the cycle is not


def run_cb(device, mode, *, n_iters, pages_per_call, hgroups, kgroups, depth_pages=None):
    """One CB cycle per iteration, no payload. `depth_pages` only applies to PAIR_SHALLOW."""
    cores = rect(hgroups, kgroups)
    total_pages = n_iters * pages_per_call

    if mode == CB_MODE_PAIR_SHALLOW:
        cap = depth_pages if depth_pages is not None else 2 * pages_per_call
        assert cap % pages_per_call == 0, "CB capacity must be a whole number of chunks (pointer wrap)"
        assert total_pages % cap == 0, "the run must be a whole number of CB cycles (pointer wrap)"
    else:
        cap = total_pages

    cbs = [cb(CB_PROBE, cores, cap, CB_PROBE_PAGE_BYTES)]

    rt = ttnn.RuntimeArgs()
    for x in range(hgroups):
        for y in range(kgroups):
            rt[x][y] = [n_iters]

    kernels = []
    prod_role = CB_ROLE_BULK_PRODUCER if mode == CB_MODE_CONS_PREFILL else CB_ROLE_PRODUCER
    kernels.append(
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "cb_probe.cpp"),
            core_ranges=cores,
            compile_time_args=[prod_role, pages_per_call, CB_PROBE],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
    )
    if mode != CB_MODE_PROD_DEEP:
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "cb_probe.cpp"),
                core_ranges=cores,
                compile_time_args=[CB_ROLE_CONSUMER, pages_per_call, CB_PROBE],
                runtime_args=rt,
                config=ttnn.WriterConfigDescriptor(),
            )
        )

    a, b = _dummy_io(device)
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    return ttnn.generic_op([a, b], program)


# ---------------------------------------------------------------------------
# PROBE 2: NoC command issue cost
# ---------------------------------------------------------------------------
def make_scratch(device):
    import torch

    return ttnn.from_torch(
        torch.zeros((SCRATCH_PAGES, SCRATCH_PAGE_BYTES // 4), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def run_noc(device, mode, *, n_cmds, xfer_bytes, barrier_each, on_writer, hgroups, kgroups, scratch):
    cores = rect(hgroups, kgroups)
    cbs = [cb(CB_SCRATCH, cores, 1, SCRATCH_PAGE_BYTES)]

    ct = [mode, 1 if barrier_each else 0, CB_SCRATCH, SCRATCH_PAGES]
    ct.extend(ttnn.TensorAccessorArgs(scratch).get_compile_time_args())

    # The remote-L1 modes aim at a peer one column over (wrapping), so every core has a distinct,
    # non-self target and the pattern is a genuine core-to-core transfer rather than a loopback.
    rt = ttnn.RuntimeArgs()
    for x in range(hgroups):
        for y in range(kgroups):
            pvx, pvy = virt(device, (x + 1) % hgroups, y)
            rt[x][y] = [scratch.buffer_address(), scratch.buffer_page_size(), pvx, pvy, 0, n_cmds, xfer_bytes]

    config = ttnn.WriterConfigDescriptor() if on_writer else ttnn.ReaderConfigDescriptor()
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "noc_probe.cpp"),
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )
    ]
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([32, 32]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    return ttnn.generic_op([scratch, out], program)


# ---------------------------------------------------------------------------
# PROBE 3: semaphores
# ---------------------------------------------------------------------------
def _sem_descriptors(cores):
    return [
        ttnn.SemaphoreDescriptor(id=SEM_TARGET, core_ranges=cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_SAT, core_ranges=cores, initial_value=1),
    ]


def _sem_rt(device, hgroups, kgroups, *, peer_of, noc1_rect, n_ops, n_senders=1):
    """peer_of(x, y) -> logical (px, py) of this core's unicast target."""
    far, near = virt(device, hgroups - 1, kgroups - 1), virt(device, 0, 0)
    r = (far[0], far[1], near[0], near[1]) if noc1_rect else (near[0], near[1], far[0], far[1])
    rt = ttnn.RuntimeArgs()
    for x in range(hgroups):
        for y in range(kgroups):
            pvx, pvy = virt(device, *peer_of(x, y))
            rt[x][y] = [pvx, pvy, r[0], r[1], r[2], r[3], n_ops, n_senders]
    return rt


def run_sem_inc_p2p(device, *, n_ops, barrier_each, on_writer, hgroups, kgroups):
    """N unicast `noc_semaphore_inc` to a peer one column over. Every core both sends and is a
    target, so the fan-in per target is 1 — this is the POINT-TO-POINT cost, not the incast."""
    cores = rect(hgroups, kgroups)
    ct = [SEM_ROLE_INC_P2P, 1 if barrier_each else 0, SEM_TARGET, SEM_SAT, 0]
    rt = _sem_rt(
        device,
        hgroups,
        kgroups,
        peer_of=lambda x, y: ((x + 1) % hgroups, y),
        noc1_rect=on_writer,
        n_ops=n_ops,
    )
    config = ttnn.WriterConfigDescriptor() if on_writer else ttnn.ReaderConfigDescriptor()
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "sem_probe.cpp"),
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )
    ]
    a, b = _dummy_io(device)
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=_sem_descriptors(cores), cbs=[])
    return ttnn.generic_op([a, b], program)


def run_sem_incast(device, *, n_rounds, n_senders, hgroups, kgroups):
    """N_SENDERS cores each increment ONE root's semaphore n_rounds times; the root performs
    n_rounds successive wait_min, each raised by n_senders. Measures how fast a single semaphore
    absorbs a grid-wide incast — the op's per-round ack pattern."""
    cores = rect(hgroups, kgroups)
    # Senders: the first n_senders cores in row-major order over the rect.
    sender_coords = [(x, y) for y in range(kgroups) for x in range(hgroups)][:n_senders]
    assert len(sender_coords) == n_senders, f"grid {hgroups}x{kgroups} has fewer than {n_senders} cores"
    sender_ranges = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for (x, y) in sender_coords]
    )
    root = single_core()
    rvx, rvy = virt(device, 0, 0)

    sender_rt = ttnn.RuntimeArgs()
    for x, y in sender_coords:
        sender_rt[x][y] = [rvx, rvy, 0, 0, 0, 0, n_rounds, n_senders]
    root_rt = ttnn.RuntimeArgs()
    root_rt[0][0] = [rvx, rvy, 0, 0, 0, 0, n_rounds, n_senders]

    kernels = [
        # Senders on the WRITER RISC-V (where the op's acks are issued), root's wait on the READER,
        # so the root's own send does not serialise behind its own wait.
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "sem_probe.cpp"),
            core_ranges=sender_ranges,
            compile_time_args=[SEM_ROLE_INCAST_SENDER, 1, SEM_TARGET, SEM_SAT, 0],
            runtime_args=sender_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "sem_probe.cpp"),
            core_ranges=root,
            compile_time_args=[SEM_ROLE_INCAST_ROOT, 0, SEM_TARGET, SEM_SAT, 0],
            runtime_args=root_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
    ]
    a, b = _dummy_io(device)
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=_sem_descriptors(cores), cbs=[])
    return ttnn.generic_op([a, b], program)


def run_sem_wait_sat(device, *, n_ops, hgroups, kgroups):
    """N `noc_semaphore_wait_min` on a semaphore whose initial value already satisfies them: the
    pure call overhead under every wait in the op."""
    cores = rect(hgroups, kgroups)
    rt = _sem_rt(device, hgroups, kgroups, peer_of=lambda x, y: (x, y), noc1_rect=False, n_ops=n_ops)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "sem_probe.cpp"),
            core_ranges=cores,
            compile_time_args=[SEM_ROLE_WAIT_SAT, 0, SEM_TARGET, SEM_SAT, 0],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
    ]
    a, b = _dummy_io(device)
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=_sem_descriptors(cores), cbs=[])
    return ttnn.generic_op([a, b], program)


def run_sem_mcast(device, *, n_ops, barrier_each, hgroups, kgroups):
    """N `noc_semaphore_inc_multicast` over the whole worker rect from ONE core, on the writer
    (NOC1) — the op's ready-signal, priced alone. Fan-out = hgroups*kgroups - 1."""
    cores = rect(hgroups, kgroups)
    n_dests = hgroups * kgroups - 1
    far, near = virt(device, hgroups - 1, kgroups - 1), virt(device, 0, 0)
    rt = ttnn.RuntimeArgs()
    # NOC1 routing order: the rect starts at the FAR corner.
    rt[0][0] = [near[0], near[1], far[0], far[1], near[0], near[1], n_ops, 1]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "sem_probe.cpp"),
            core_ranges=single_core(),
            compile_time_args=[SEM_ROLE_MCAST_SEM, 1 if barrier_each else 0, SEM_TARGET, SEM_SAT, n_dests],
            runtime_args=rt,
            config=ttnn.WriterConfigDescriptor(),
        )
    ]
    a, b = _dummy_io(device)
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=_sem_descriptors(cores), cbs=[])
    return ttnn.generic_op([a, b], program)


# ---------------------------------------------------------------------------
# PROBE 4: the full grid-wide multicast rendezvous
# ---------------------------------------------------------------------------
MCAST_STAGE_ACK = 0  # ack incast only (nothing serialises the rounds)
MCAST_STAGE_ACK_DATA = 1  # + data multicast (still unserialised — a race, kept only as a diagnostic)
MCAST_STAGE_FULL = 2  # + ready signal: the real round
MCAST_STAGE_ACK_READY = 3  # the real round MINUS the payload write — subtract from FULL for the data cost
MCAST_STAGES = {
    MCAST_STAGE_ACK: "ack_only",
    MCAST_STAGE_ACK_DATA: "ack+data(race)",
    MCAST_STAGE_FULL: "ack+data+ready",
    MCAST_STAGE_ACK_READY: "ack+ready(no data)",
}

ROOT_ROW = 0


def make_mcast_io(device, payload_bytes, num_cores):
    import torch

    torch.manual_seed(0)
    src = torch.randint(1, 2**30, (1, payload_bytes // 4), dtype=torch.int32)
    tt_in = ttnn.from_torch(
        src, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.from_torch(
        torch.zeros((num_cores, payload_bytes // 4), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return src, tt_in, tt_out


def run_mcast(device, *, rounds, stage, depth, payload_bytes, hgroups, kgroups, verify=False, io=None):
    num_cores = hgroups * kgroups
    cores = rect(hgroups, kgroups)
    roots = row_range(hgroups, ROOT_ROW)
    verify = verify and stage in (MCAST_STAGE_ACK_DATA, MCAST_STAGE_FULL)  # only these move the payload

    if io is None:
        io = make_mcast_io(device, payload_bytes, num_cores)
    _, tt_in, tt_out = io

    cbs = [cb(CB_MCAST, cores, 1, payload_bytes)]
    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_FREE, core_ranges=cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_RDY, core_ranges=cores, initial_value=0),
    ]

    far, near = virt(device, hgroups - 1, kgroups - 1), virt(device, 0, 0)

    sender_ct = [hgroups, num_cores, stage, SEM_FREE, SEM_RDY, CB_MCAST]
    sender_ct.extend(ttnn.TensorAccessorArgs(tt_in).get_compile_time_args())
    sender_rt = ttnn.RuntimeArgs()
    for x in range(hgroups):
        # NOC1 routing order: the multicast hardware walks from `start` in the NoC's own direction,
        # which is the reverse of NOC0's — so the writer's rect starts at the FAR corner.
        sender_rt[x][ROOT_ROW] = [
            x,
            far[0],
            far[1],
            near[0],
            near[1],
            tt_in.buffer_address(),
            tt_in.buffer_page_size(),
            rounds,
            payload_bytes,
        ]

    recv_ct = [hgroups, depth, stage, SEM_FREE, SEM_RDY, 1 if verify else 0, CB_MCAST]
    recv_ct.extend(ttnn.TensorAccessorArgs(tt_out).get_compile_time_args())
    recv_rt = ttnn.RuntimeArgs()
    root_coords = []
    for c in range(hgroups):
        root_coords.extend(virt(device, c, ROOT_ROW))
    for x in range(hgroups):
        for y in range(kgroups):
            core_idx = y * hgroups + x
            recv_rt[x][y] = [
                x,
                1 if y == ROOT_ROW else 0,
                tt_out.buffer_address(),
                tt_out.buffer_page_size(),
                core_idx,
                rounds,
                payload_bytes,
            ] + root_coords

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "mcast_receiver.cpp"),
            core_ranges=cores,
            compile_time_args=recv_ct,
            runtime_args=recv_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "mcast_sender.cpp"),
            core_ranges=roots,
            compile_time_args=sender_ct,
            runtime_args=sender_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
    return ttnn.generic_op([tt_in, tt_out], program)


# ---------------------------------------------------------------------------
# PROBE 5: DEST sync with no math
# ---------------------------------------------------------------------------
def run_dest(device, *, n_iters, hgroups, kgroups):
    cores = rect(hgroups, kgroups)
    tile = ttnn.tile_size(ttnn.bfloat16)
    cbs = [
        cb(CB_DEST_IN, cores, 2, tile, ttnn.bfloat16),
        cb(CB_DEST_OUT, cores, 2, tile, ttnn.bfloat16),
    ]
    rt = ttnn.RuntimeArgs()
    for x in range(hgroups):
        for y in range(kgroups):
            rt[x][y] = [n_iters]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "dest_probe.cpp"),
            core_ranges=cores,
            compile_time_args=[CB_DEST_IN, CB_DEST_OUT],
            runtime_args=rt,
            config=compute_kernel_config(),
        )
    ]
    a, b = _dummy_io(device)
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    return ttnn.generic_op([a, b], program)
