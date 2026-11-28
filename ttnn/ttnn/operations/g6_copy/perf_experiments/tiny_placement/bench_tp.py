"""bench_cases.py - the case bench of a meta campaign (template, plain ASCII).

Why: a golden test launches the op once, and launch 1 of a program is never a timing. This file
runs chosen LOOSE_CASES entries of the op's golden suite through the suite's own runner, with
the op dispatched BENCH_LAUNCHES times on the same inputs in one process. The runner checks one
result against the suite's reference; launches 2..N are what the profiler readers take the
median of.

Where: copy this template to generated/meta/<op>/bench/bench_cases.py under the tt-metal root
of the pipeline checkout. That folder is git-ignored and lies outside the op's test folder and
the suite, so no pipeline gate collects it and no pipeline agent commits it.

Adapt: the block between the two ADAPT lines, nothing else.

Run, from the tt-metal root, never as a bare pytest (chip_run.sh pins one chip and holds the
locks; the procedure is references/measure.md of the meta skill):
    T=.claude/skills/tooling; B=generated/meta/<op>/bench/bench_cases.py
    BENCH_CASES=<case> BENCH_LAUNCHES=2 $T/scripts/chip_run.sh 0 <run> -t 600 -- pytest $B
    BENCH_CASES=<case> $T/scripts/chip_run.sh 0 <run> -t 600 -- \
        $T/scripts/profile_op.sh --direct --op GenericOp --skip-first 1 $B

Environment:
    BENCH_CASES     comma list of case ids to run. Default: all of them (more than 8 must be
                    named). One case per process for anything that goes on the page: cases
                    can share a program hash
    BENCH_LAUNCHES  dispatches of the op per runner call. Default 12: launch 1 and 11 measured
    BENCH_KEEP      first (default): the result of launch 1 goes to the reference check and
                    stays allocated while the others run. last: every result is freed before
                    the next launch and the last one is checked (two outputs do not fit in L1)
    BENCH_TIMEOUT   seconds allowed per case. Default 3000; pytest.ini alone allows 300

Output, one line each, grep '^BENCH_':
    BENCH_DEVICE    arch and worker grid of the chip that ran
    BENCH_CASE      case id, index in LOOSE_CASES, inputs, axes
    BENCH_NOTE      the suite marks this case xfail (the op does not claim it yet) or skip (an
                    INVALID cell; the bench runs it all the same: take another case)
    BENCH_LAUNCHES  launches done, programs per launch, op id of the first and of the last
                    program (--run-id of zone_table.py and grid_check.py), last trace ID
                    (None: this ttnn does not tell the ids)
    BENCH_OK        the suite's reference check passed

Checked against eval/golden_harness.py, the suites' helpers.py and test_golden.py and
skills/golden-tests/SKILL.md of tt_ops_code_gen b93830b, and against conftest.py and pytest.ini
of tt-metal 204116418f. NOT RUN with ttnn and not on a device: only the control flow ran,
against stand-in modules for ttnn and the suite (pytest 7.2.2 and 9.2.0). The first use on a
checkout proves or refutes: that the op can be dispatched again on the same input tensors;
ttnn.deallocate of a repeated result; ttnn.synchronize_device between launches; that
ttnn._ttnn.get_device_operation_id() moves by one per program; the timeout marker. So begin
with the BENCH_LAUNCHES=2 line above (pytest without a profiler, still through chip_run.sh) and
read the BENCH_ lines before any capture.
"""

from __future__ import annotations

import importlib
import os
import random
import zlib
from pathlib import Path

import pytest

torch = __import__("torch")  # experiment script; ttnn bans a global torch import
import ttnn

# ================================ ADAPT: start ================================
# The op. In generated/meta/<op>/bench/ the folder names it; write the name when the file lives elsewhere.
OP = "g6_copy"
# The suite's runner in eval/golden_tests/<op>/helpers.py.
RUNNER = "run_" + OP
# The name under which helpers.py holds the op: "from ...axes import observed as <name>" or "import <name>".
OP_CALLABLE = OP
# Campaign case id -> index into the suite's LOOSE_CASES, for example {"s4096": 0, "s1024": 2}.
# Empty: every loose case, named L0, L1, ... in file order.
# Extension (g6_copy r01): a value ("prod", i) names PRODUCTION_CASES[i] of feature_spec.py, run
# through the same runner with no extras (the suite's production-row call). Losing production
# scenarios of the lead's A/B (/localdev/pmilojevic/campaign/group6/ab/copy/ab.json) per group:
CASES = {
    "t_bf16_1x1": ("prod", 14),  # FLUX 33   (1, 1) bf16 TILE            tiny, 1 page
    "t_u32_32": ("prod", 23),  # Qwen 182  (32,) uint32 ROW_MAJOR       tiny, 1 page
    "t_bf8b": ("prod", 15),  # Qwen 163  (5, 1, 32, 256) bf8b TILE    tiny, 40 pages
    "rm_1": ("prod", 16),  # Qwen 181/188/190 (1, 3, 2560) bf16 RM  3 pages of 5120 B
    "rm_16": ("prod", 19),  # Qwen 165/169/171 (16, 3, 2560) bf16 RM 48 pages
    "f32_2": ("prod", 2),  # Qwen 172  (2, 12, 128, 128) fp32 TILE  768 pages
    "g_f32_16": 0,  # guard: the flagged loose case (16, 12, 128, 128) fp32 TILE
    "g_bf16_8192": ("prod", 10),  # guard: FLUX 39 (1, 1, 8192, 128) bf16 TILE
}
# Perf 1 coordinator: every production row, p<i> = PRODUCTION_CASES[i].
CASES.update({f"p{i}": ("prod", i) for i in range(29)})
# Extension: BENCH_IMPL=stock dispatches stock ttnn.copy(src, dst) (a black box, as the lead's A/B
# harness calls it) in place of the op, on the same tensors, with the same reference check.
IMPL = os.environ.get("BENCH_IMPL", "new")
# Extension: BENCH_SYNC=0 drops the device synchronize between launches (launches go back to back,
# closer to the lead's A/B harness, which issues them without a synchronize). Default 1.
SYNC = os.environ.get("BENCH_SYNC", "1") != "0"


def _stock_copy(src, dst, *args, **kwargs):
    ttnn.copy(src, dst)
    return dst


def launch_again(real_op, args, kwargs):
    """One more dispatch after launch 1. Rewrite it when the op cannot run twice on the same
    tensors (it works in place, frees its inputs or advances a state): rebuild what it needs here."""
    return real_op(*args, **kwargs)


# ================================ ADAPT: end ==================================


class BenchSetupError(RuntimeError):
    pass


def _load(name):
    try:
        return importlib.import_module(name)
    except Exception as err:
        raise BenchSetupError(
            f"bench_cases.py: cannot import {name} ({type(err).__name__}: {err}). Check OP in the ADAPT block, "
            "run from the tt-metal root and keep that root on PYTHONPATH (chip_run.sh sets it)."
        ) from err


def _number(name, default, lowest):
    text = os.environ.get(name, str(default))
    if not text.isdigit() or int(text) < lowest:
        raise BenchSetupError(f"bench_cases.py: {name}={text!r} must be a whole number of at least {lowest}")
    return int(text)


LAUNCHES = _number("BENCH_LAUNCHES", 12, 1)
TIMEOUT_S = _number("BENCH_TIMEOUT", 3000, 1)
KEEP = os.environ.get("BENCH_KEEP", "first")
if KEEP not in ("first", "last"):
    raise BenchSetupError(f"bench_cases.py: BENCH_KEEP={KEEP!r} must be first or last")
# More loose cases than this run only when BENCH_CASES names them: a suite can hold hundreds.
MOST_UNNAMED_CASES = 8

SPEC = _load(f"eval.golden_tests.{OP}.feature_spec")
HELPERS = _load(f"eval.golden_tests.{OP}.helpers")
OP_PACKAGE = _load(f"ttnn.operations.{OP}")
HARNESS = _load("eval.golden_harness")

for _name in (RUNNER, OP_CALLABLE):
    if not callable(getattr(HELPERS, _name, None)):
        _public = sorted(key for key, value in vars(HELPERS).items() if callable(value) and not key.startswith("_"))
        raise BenchSetupError(
            f"bench_cases.py: eval/golden_tests/{OP}/helpers.py has no callable {_name!r}. "
            f"Set RUNNER and OP_CALLABLE in the ADAPT block. Callables there: {', '.join(_public)}"
        )


def _selected_cases():
    loose = list(getattr(SPEC, "LOOSE_CASES", []))
    if not loose:
        raise BenchSetupError(f"bench_cases.py: eval/golden_tests/{OP}/feature_spec.py has no LOOSE_CASES entry")
    # The suite's own entry point computes the axes; its ids and marks are not used as they are.
    rows = HARNESS.parametrize_loose_cases(
        loose,
        OP_PACKAGE.INPUT_TAGGERS,
        OP_PACKAGE.SUPPORTED,
        OP_PACKAGE.EXCLUSIONS,
        getattr(SPEC, "INVALID", []),
        shape_id=lambda _inputs: "bench",
    )
    prod = [dict(case, extras={}) for case in getattr(SPEC, "PRODUCTION_CASES", [])]
    prod_rows = HARNESS.parametrize_loose_cases(
        prod,
        OP_PACKAGE.INPUT_TAGGERS,
        OP_PACKAGE.SUPPORTED,
        OP_PACKAGE.EXCLUSIONS,
        getattr(SPEC, "INVALID", []),
        shape_id=lambda _inputs: "bench",
    )
    index_of = dict(CASES) or {f"L{index}": index for index in range(len(rows))}
    for case_id, index in index_of.items():
        if isinstance(index, tuple) and len(index) == 2 and index[0] == "prod" and 0 <= index[1] < len(prod_rows):
            continue
        if not isinstance(index, int) or not 0 <= index < len(rows):
            raise BenchSetupError(
                f"bench_cases.py: CASES[{case_id!r}] = {index!r}, but LOOSE_CASES has entries 0..{len(rows) - 1}"
            )
    wanted = [item.strip() for item in os.environ.get("BENCH_CASES", "").split(",") if item.strip()]
    unknown = [item for item in wanted if item not in index_of]
    if unknown:
        raise BenchSetupError(f"bench_cases.py: BENCH_CASES names {unknown}; the case ids are {sorted(index_of)}")
    if not wanted and len(index_of) > MOST_UNNAMED_CASES:
        raise BenchSetupError(
            f"bench_cases.py: {len(index_of)} cases and no BENCH_CASES: name the cases to run "
            "(BENCH_CASES=<id>[,<id>]) or fill CASES in the ADAPT block"
        )
    selected = []
    for case_id in wanted or list(index_of):
        index = index_of[case_id]
        row = prod_rows[index[1]] if isinstance(index, tuple) else rows[index]
        inputs, axes, extras = row.values
        notes = [
            f"{mark.name}: {mark.kwargs.get('reason', '')}" for mark in row.marks if mark.name in ("skip", "xfail")
        ]
        selected.append(pytest.param(case_id, index_of[case_id], inputs, axes, extras, notes, id=case_id))
    return selected


def _op_id():
    """The id the next device program gets, or None when this ttnn does not tell."""
    try:
        return int(ttnn._ttnn.get_device_operation_id())
    except Exception:
        return None


def _tensors(value):
    if isinstance(value, ttnn.Tensor):
        return [value]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _tensors(item)]
    return []


def _release(result, protected):
    """Free the device memory of one launch's result. Never an input and never the kept result:
    an op may hand one of its inputs back."""
    for tensor in _tensors(result):
        if not any(tensor is other for other in protected):
            ttnn.deallocate(tensor)


def _launch(dispatch, device):
    before = _op_id()
    result = dispatch()
    # Back-to-back launches start their cores further apart, and that spread is part of the kernel span.
    if SYNC:
        ttnn.synchronize_device(device)
    return result, before, _op_id()


pytestmark = [pytest.mark.use_module_device, pytest.mark.timeout(TIMEOUT_S)]


@pytest.mark.parametrize("case_id,loose_index,inputs,axes,extras,notes", _selected_cases())
def test_bench(case_id, loose_index, inputs, axes, extras, notes, device, monkeypatch):
    # The suite's conftest seeds its tests; this file is outside it. Same data on every chip and in every run.
    seed = zlib.crc32(case_id.encode()) & 0x7FFFFFFF
    torch.manual_seed(seed)
    random.seed(seed)
    # The leading newline: pytest -v leaves the test name on the open line, and the greps anchor on ^BENCH_.
    try:
        grid = device.compute_with_storage_grid_size()
        print(f"\nBENCH_DEVICE arch={device.arch()} worker_grid={grid.x}x{grid.y}", flush=True)
    except Exception as err:
        print(f"\nBENCH_DEVICE unknown ({type(err).__name__}: {err})", flush=True)
    shown_axes = " ".join(f"{key}={axes[key]}" for key in sorted(axes))
    print(
        f"BENCH_CASE case={case_id} impl={IMPL} loose_index={loose_index} inputs={inputs} axes: {shown_axes}",
        flush=True,
    )
    for note in notes:
        print(f"BENCH_NOTE case={case_id} the suite marks this case {note}", flush=True)

    if IMPL not in ("new", "stock"):
        raise BenchSetupError(f"bench_cases.py: BENCH_IMPL={IMPL!r} must be new or stock")
    real_op = getattr(HELPERS, OP_CALLABLE) if IMPL == "new" else _stock_copy
    calls = []

    def repeated(*args, **kwargs):
        given = _tensors(args) + _tensors(kwargs)
        kept, first_op, after = _launch(lambda: real_op(*args, **kwargs), device)
        per_launch = None if None in (first_op, after) else after - first_op
        for number in range(2, LAUNCHES + 1):
            try:
                if KEEP == "last":
                    _release(kept, given)
                result, _, after = _launch(lambda: launch_again(real_op, args, kwargs), device)
                if KEEP == "last":
                    kept = result
                else:
                    _release(result, given + _tensors(kept))
            except Exception as err:
                # pytest.fail and not a re-raise: the runner's oom_guard would turn an allocation error into a skip.
                pytest.fail(
                    f"launch {number} of {LAUNCHES} failed after launch 1 ran: {type(err).__name__}: "
                    f"{' '.join(str(err).split())[:400]}. If memory ran out, two outputs do not fit: BENCH_KEEP=last. "
                    "If the op cannot run again on the same tensors, rewrite launch_again in the ADAPT block.",
                    pytrace=False,
                )
        calls.append(1)
        last_op = None if after is None else after - 1
        trace_id = None if last_op is None else last_op << 10
        print(
            f"BENCH_LAUNCHES case={case_id} impl={IMPL} call={len(calls)} launches={LAUNCHES} keep={KEEP} sync={int(SYNC)} "
            f"programs_per_launch={per_launch} first_op={first_op} last_op={last_op} last_trace_id={trace_id}",
            flush=True,
        )
        return kept

    monkeypatch.setattr(HELPERS, OP_CALLABLE, repeated)
    getattr(HELPERS, RUNNER)(inputs, device=device, extras=extras, **axes)
    if not calls:
        pytest.fail(
            f"{RUNNER} never called helpers.{OP_CALLABLE}: nothing was repeated. Set OP_CALLABLE in the ADAPT "
            "block to the name helpers.py calls the op by.",
            pytrace=False,
        )
    checked = 1 if KEEP == "first" else LAUNCHES
    print(f"BENCH_OK case={case_id} impl={IMPL} the suite's check passed on the result of launch {checked}", flush=True)


# Perf 1 variant: the op's descriptor with KERNEL_DIR taken from env G6C_KDIR (ablation / experiments).
import ttnn.operations.g6_copy.g6_copy_program_descriptor as _PD  # noqa: E402

if os.environ.get("G6C_KDIR"):
    _PD.KERNEL_DIR = Path(os.environ["G6C_KDIR"]).resolve()
print(f"BENCH_VARIANT kdir={_PD.KERNEL_DIR}", flush=True)

# tiny_placement: TP_VARIANT=cand swaps the op's descriptor builder for the candidate placement.
if os.environ.get("TP_VARIANT", "base") == "cand":
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).parent))
    import tp_candidate as _TPC  # noqa: E402

    _OPM = _sys.modules["ttnn.operations.g6_copy.g6_copy"]  # the module (the package attr is the function)
    assert hasattr(_OPM, "get_program_descriptor")
    _OPM.get_program_descriptor = _TPC.get_program_descriptor
print(f"BENCH_TP variant={os.environ.get('TP_VARIANT', 'base')}", flush=True)
if os.environ.get("TP_VARIANT", "base") == "cand":
    _orig_build = _TPC.build

    def _logged_build(inp, out, p):
        print(
            f"BENCH_TP plan cores={len(p.work)} kind={p.plan} block={p.block_pages} "
            f"first={[(w.core.x, w.core.y, w.first_page, w.num_pages) for w in p.work[:4]]}",
            flush=True,
        )
        return _orig_build(inp, out, p)

    _TPC.build = _logged_build

# TP_VARIANT=grad: the graduation copy of the descriptor module (graduate/), op kernels unchanged.
if os.environ.get("TP_VARIANT", "base") == "grad":
    import importlib.util as _ilu
    import sys as _sys

    _spec = _ilu.spec_from_file_location(
        "g6c_grad", str(Path(__file__).parent / "graduate/g6_copy_program_descriptor.py")
    )
    _GRAD = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_GRAD)
    _GRAD.KERNEL_DIR = _PD.KERNEL_DIR
    _sys.modules["ttnn.operations.g6_copy.g6_copy"].get_program_descriptor = _GRAD.get_program_descriptor
    _orig_plan = _GRAD._plan

    def _logged_plan(*a, **k):
        p = _orig_plan(*a, **k)
        print(
            f"BENCH_TP grad plan cores={len(p.work)} kind={p.plan} bank_major={p.bank_major} "
            f"block={p.block_pages} depth={p.cb_depth_blocks}",
            flush=True,
        )
        return p

    _GRAD._plan = _logged_plan
