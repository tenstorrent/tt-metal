#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Local qualification driver for experimental-quasar binary ops.

This is the tt-metal-side companion to the (upstream-owned) Quasar foundation CI. It does NOT
re-qualify the simulator or the LLK: craq-sim's `quasar-llk.yml` already runs the whole
`test_*_quasar.py` LLK suite on the QSR sim, sharded, with structured reporting. Instead it ties
the two layers we build on into a single, repeatable, fast check:

  Foundation layer : the LLK binary micro-tests (run on the sim => transitively cover craq-sim).
  Op layer         : our ttnn binary op tests (test_binary_ng_no_bcast.py).

For each capability cell (op x dtype x compute-path) it runs our op probe on the QSR sim and -- when
that op probe FAILS (or with --force-foundation) -- a single representative LLK variant to localize
the failure. It records a PASS/FAIL/SKIP grid, diffs against a saved baseline (to surface regressions
and progressions as tt-llk / craq-sim move), and prints a triage verdict that localizes any op failure
to the op layer vs the foundation. The foundation probe is expensive (a full compile-producer), so by
default it is fired only on demand; a green op run does not pay for it.

Why a single LLK variant: the LLK suites are huge (the FPU file alone is ~55k variants) and the
full sweep is exactly what craq-sim's CI runs. A local *probe* only needs one representative
variant per cell to answer "is the primitive alive on the sim?". The driver collects the LLK
test ids through the harness venv (tests/.venv), filters them by exact id-substrings (pytest -k
cannot express the ':' in format tokens, and substring-matches the output token), and runs the
first match as a single --test-id through run_test.sh (flock-serialised sim access).

The op-cell <-> LLK-test mapping below is the authoritative source; the runbook
(QUASAR_BINARY_QUALIFICATION.md) documents the method and points here.

Usage:
    python qualify_quasar_binary.py --show-mapping
    python qualify_quasar_binary.py --validate                       # resolve every cell (no sim)
    python qualify_quasar_binary.py --coverage                       # LLK supports but our op doesn't (no sim)
    python qualify_quasar_binary.py --supports pow                   # does LLK already have op X? (no sim)
    python qualify_quasar_binary.py --run --out baseline.json        # op grid + foundation on op-fail
    python qualify_quasar_binary.py --run --force-foundation --out base.json  # exhaustive op+LLK grid (sim bump)
    python qualify_quasar_binary.py --run --baseline baseline.json --out new.json
    python qualify_quasar_binary.py --run --layer op --cells '*.bf16.*'
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------------
def find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "tt_metal" / "tt-llk").is_dir():
            return parent
    raise SystemExit("could not locate repo root (no tt_metal/tt-llk above this script)")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
TT_LLK_DIR = REPO_ROOT / "tt_metal" / "tt-llk"
TT_LLK_TESTS = TT_LLK_DIR / "tests"
RUN_TEST_SH = TT_LLK_DIR / ".claude" / "scripts" / "run_test.sh"
LLK_VENV_PY = TT_LLK_TESTS / ".venv" / "bin" / "python"
LLK_TEST_SUBDIR = "python_tests/quasar"  # relative to TT_LLK_TESTS, used for collection
OP_TEST = (
    REPO_ROOT
    / "tests"
    / "ttnn"
    / "nightly"
    / "unit_tests"
    / "operations"
    / "experimental"
    / "quasar"
    / "test_binary_ng_no_bcast.py"
)
SOC_DESC_SRC = REPO_ROOT / "tt_metal" / "soc_descriptors" / "quasar_32_arch.yaml"
DEFAULT_SIM_DIR = Path("/workspaces/craq-sim/src/_out/release_qsr")

LLK_FPU = "test_eltwise_binary_quasar.py"
LLK_SFPU = "test_eltwise_binary_sfpu_quasar.py"


# --------------------------------------------------------------------------------------------
# The capability matrix: op-cell <-> foundation (LLK) probe + op probe.
#
# llk_tokens is a list of substrings that must ALL appear in an LLK test id (Python substring-AND,
# because pytest -k cannot express the ':' in the format tokens and would substring-match the
# wrong half of a mixed in/out format). A None foundation (llk_test) means there is no LLK
# primitive for the cell -- a genuine foundation gap, not a missing test. A None op_k means our
# focused op suite does not yet exercise the cell (an op-test COVERAGE gap, worth surfacing).
# `expected` is the current known-good status so a deviation stands out even without a baseline.
# --------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Cell:
    id: str
    llk_test: Optional[str]
    llk_tokens: Optional[list]
    op_k: Optional[str]
    expected: str  # "PASS" | "KNOWN_GAP" | "OP_COVERAGE_GAP"
    note: str = ""


_BF16 = "formats:Float16_b->Float16_b"  # FPU id format token (pure bf16 in/out)
_SF_BF16 = "A:Float16_b,B:Float16_b,out:Float16_b"  # SFPU id format token (pure bf16 in/out)
_SF_FP32 = "A:Float32,B:Float32,out:Float32"  # SFPU id format token (pure fp32 in/out)

MAPPING: list = [
    # add / sub: FPU; bf16 covered both layers.
    Cell("add.bf16.fpu", LLK_FPU, ["mathop:Elwadd", _BF16], "test_no_bcast_interleaved and add and bfloat16", "PASS"),
    Cell(
        "sub.bf16.fpu", LLK_FPU, ["mathop:Elwsub", _BF16], "test_no_bcast_interleaved and subtract and bfloat16", "PASS"
    ),
    # mul / div: SFPU float family; bf16 covered both layers.
    Cell(
        "mul.bf16.sfpu",
        LLK_SFPU,
        ["test_eltwise_binary_sfpu_float_quasar", _SF_BF16, "-MUL]"],
        "test_no_bcast_interleaved and multiply and bfloat16",
        "PASS",
    ),
    Cell(
        "div.bf16.sfpu",
        LLK_SFPU,
        ["test_eltwise_binary_sfpu_float_quasar", _SF_BF16, "-DIV]"],
        "test_no_bcast_sfpu_divide and interleaved and bfloat16",
        "PASS",
    ),
    # fp32 mul / div: SFPU float, fp32 dest-acc. Foundation covers it.
    Cell(
        "mul.fp32.sfpu",
        LLK_SFPU,
        ["test_eltwise_binary_sfpu_float_quasar", _SF_FP32, "-MUL]"],
        "test_no_bcast_interleaved and multiply and float32",
        "PASS",
    ),
    Cell(
        "div.fp32.sfpu",
        LLK_SFPU,
        ["test_eltwise_binary_sfpu_float_quasar", _SF_FP32, "-DIV]"],
        "test_no_bcast_sfpu_divide and interleaved and float32",
        "PASS",
    ),
    # fp32 add / sub: route SFPU, but the SFPU float family is MUL/DIV only -> no float-add
    # primitive on Quasar. The op test skips these on Quasar; documented foundation gap.
    Cell(
        "add.fp32.sfpu",
        None,
        None,
        "test_no_bcast_interleaved and add and float32",
        "KNOWN_GAP",
        note="no SFPU float-add primitive on Quasar (LLK float family = MUL/DIV; int family = Int32)",
    ),
    Cell(
        "sub.fp32.sfpu",
        None,
        None,
        "test_no_bcast_interleaved and subtract and float32",
        "KNOWN_GAP",
        note="no SFPU float-sub primitive on Quasar (see add.fp32.sfpu)",
    ),
]


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------
def git_sha(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=30
        )
        return out.stdout.strip() if out.returncode == 0 else "?"
    except Exception:
        return "?"


def tt_llk_sha() -> str:
    """tt-llk is a gitlink in this checkout (no nested .git), so read the pinned submodule commit."""
    if (TT_LLK_DIR / ".git").exists():
        return git_sha(TT_LLK_DIR)
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD:tt_metal/tt-llk"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return out.stdout.strip() if out.returncode == 0 else "?"
    except Exception:
        return "?"


def select_cells(pattern: Optional[str]) -> list:
    return list(MAPPING) if not pattern else [c for c in MAPPING if fnmatch.fnmatch(c.id, pattern)]


# --- LLK id collection (through the harness venv; sim-free) -------------------------------
_llk_id_cache: dict = {}


def llk_collect_ids(test_file: str, timeout: int) -> list:
    if test_file in _llk_id_cache:
        return _llk_id_cache[test_file]
    if not LLK_VENV_PY.is_file():
        raise SystemExit(
            f"LLK harness venv not found: {LLK_VENV_PY}\n"
            f"Create it:  cd {TT_LLK_TESTS} && CHIP_ARCH=quasar ./setup_testing_env.sh"
        )
    env = dict(os.environ, CHIP_ARCH="quasar")
    out = subprocess.run(
        [str(LLK_VENV_PY), "-m", "pytest", "--compile-producer", "--co", "-q", f"{LLK_TEST_SUBDIR}/{test_file}"],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(TT_LLK_TESTS),
        env=env,
    )
    ids = [ln.strip() for ln in out.stdout.splitlines() if "::" in ln]
    _llk_id_cache[test_file] = ids
    return ids


def matching_llk_ids(cell: Cell, timeout: int) -> list:
    if not cell.llk_test:
        return []
    ids = llk_collect_ids(cell.llk_test, timeout)
    return [i for i in ids if all(tok in i for tok in cell.llk_tokens)]


def pick_llk_id(cell: Cell, timeout: int) -> Optional[str]:
    """First matching id, normalised to the `file.py::func[params]` form run_test.sh expects."""
    matches = matching_llk_ids(cell, timeout)
    if not matches:
        return None
    func_and_params = matches[0].split("::", 1)[1]
    return f"{cell.llk_test}::{func_and_params}"


# --- op test collection / env ------------------------------------------------------------
def op_count(cell: Cell, timeout: int) -> Optional[int]:
    if not cell.op_k:
        return None
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(OP_TEST),
            "-k",
            cell.op_k,
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )
    # `--collect-only -q` prints a summary, not per-test node-ids; parse the selected count
    # (and never count "::" lines -- the module docstring contains "SubtileBroadcastType::NONE").
    text = out.stdout
    # Order matters: the "X/Y tests collected" (deselection) form must be tried before the bare
    # "N tests collected" form, else the bare pattern grabs Y (the total) instead of X (selected).
    for pat in (
        r"(\d+)\s+selected",
        r"(\d+)/\d+\s+tests collected",
        r"(\d+)\s+tests?\s+collected",
        r"collected\s+(\d+)\s+items?",
    ):
        m = re.search(pat, text)
        if m:
            return int(m.group(1))
    return 0


def op_sim_env(sim_dir: Path) -> dict:
    # The QSR sim auto-discovers soc_descriptor.yaml beside the .so (release_qsr symlinks it).
    # Do NOT set TT_METAL_MOCK_CLUSTER_DESC_PATH: that expects a *cluster* descriptor, and pointing
    # it at the soc descriptor makes UMD fail with "Invalid YAML". Recipe mirrors debug/TTSIM_TUTORIAL.md
    # (TT_SIMULATOR_LOCALHOST is not in its required set, so it's intentionally omitted).
    env = dict(os.environ)
    env.update(
        TT_METAL_HOME=str(REPO_ROOT),
        TT_METAL_SIMULATOR=str(sim_dir / "libttsim.so"),
        TT_METAL_SIMULATOR_HOME=str(sim_dir),
        TT_METAL_SLOW_DISPATCH_MODE="1",
        ARCH_NAME="quasar",
        CHIP_ARCH="quasar",
    )
    return env


def ensure_sim(sim_dir: Path) -> None:
    if not (sim_dir / "libttsim.so").is_file():
        raise SystemExit(
            f"QSR simulator not found at {sim_dir / 'libttsim.so'}\n"
            f"Build it:  cd /workspaces/craq-sim && ./make.py src/_out/release_qsr/libttsim.so"
        )
    soc_desc = sim_dir / "soc_descriptor.yaml"
    if not soc_desc.is_file():
        if not SOC_DESC_SRC.is_file():
            raise SystemExit(f"soc descriptor source missing: {SOC_DESC_SRC}")
        shutil.copyfile(SOC_DESC_SRC, soc_desc)


_PYTEST_COUNTS = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped)")


def classify_pytest(returncode: int, stdout: str) -> str:
    counts = {kind.rstrip("s"): int(n) for n, kind in _PYTEST_COUNTS.findall(stdout)}
    if counts.get("failed", 0) or counts.get("error", 0):
        return "FAIL"
    if counts.get("passed", 0):
        return "PASS"
    if counts.get("skipped", 0):
        return "SKIP"
    if returncode == 5:  # pytest: no tests collected
        return "NOCOLLECT"
    return "FAIL" if returncode else "PASS"


# --------------------------------------------------------------------------------------------
# Running on the sim
# --------------------------------------------------------------------------------------------
def _run_llk_once(cell: Cell, test_id: str, sim_dir: Path, timeout: int) -> int:
    out = subprocess.run(
        [
            "bash",
            str(RUN_TEST_SH),
            "run",
            "--worktree",
            str(TT_LLK_DIR),
            "--arch",
            "quasar",
            "--test",
            cell.llk_test,
            "--test-id",
            test_id,
            "--timeout",
            str(timeout),
            "--sim-path",
            str(sim_dir / "libttsim.so"),
        ],
        capture_output=True,
        text=True,
        timeout=timeout + 300,
    )
    return out.returncode


def run_llk(cell: Cell, sim_dir: Path, timeout: int) -> str:
    if not cell.llk_test:
        return "NO_LLK"
    try:
        test_id = pick_llk_id(cell, timeout)
        if not test_id:
            return "NO_MATCH"
        rc = _run_llk_once(cell, test_id, sim_dir, timeout)
        # A foundation probe can fail transiently in a grid run: a compile-step failure (exit 2) on a
        # cold build cache / parallel-compile contention, OR a test-run failure (exit 1) when the probe
        # races the sim the op probe just used (same localhost port) -- both were observed passing on an
        # isolated re-run. So disambiguate any exit-1/2 rather than trust one attempt: re-run the single
        # --test-id in isolation. A clean isolated pass => transient (FLAKY / FLAKY_COMPILE, not a real
        # gap); a repeat failure => a reproducible tt-llk gap. Hangs (exit 5) / env errors are NOT retried.
        # (The isolated retry can double a cell's foundation cost, but only for a cell that first failed.)
        if rc in (1, 2):
            rc2 = _run_llk_once(cell, test_id, sim_dir, timeout)
            if rc2 == 0:
                return "FLAKY_COMPILE" if rc == 2 else "FLAKY"
            rc = rc2  # reproducible -> report the retry's code (FAIL(1) / FAIL(2) / ...)
    except subprocess.TimeoutExpired:
        return "HANG"  # record HANG and let the grid finish (baseline still gets written)
    return "PASS" if rc == 0 else f"FAIL({rc})"


def run_op(cell: Cell, sim_dir: Path, timeout: int) -> str:
    if not cell.op_k:
        return "NO_OP_TEST"
    try:
        out = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(OP_TEST),
                "-k",
                cell.op_k,
                f"--timeout={timeout}",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 300,
            env=op_sim_env(sim_dir),
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired:
        return "HANG"  # record HANG and let the grid finish (baseline still gets written)
    return classify_pytest(out.returncode, out.stdout)


def triage(cell: Cell, foundation: str, op: str) -> str:
    op_failed = op in ("FAIL", "NOCOLLECT", "HANG")
    found_failed = foundation.startswith("FAIL") or foundation == "HANG"
    if op_failed:
        if foundation == "PASS":
            return "OP BUG (foundation passes => fix in the op layer)"
        if found_failed:
            return "FOUNDATION GAP (LLK fails too => file tt-llk / craq-sim; not an op fix)"
        if foundation in ("FLAKY_COMPILE", "FLAKY"):
            return "op fails; LLK probe was only flaky (transient) => re-run the probe in isolation to localize"
        if foundation in ("NO_LLK", "NO_MATCH"):
            return "KNOWN GAP (no foundation primitive for this cell)"
        return "op failed; foundation NOT probed => re-run with --force-foundation (or --layer foundation) to localize"
    # Op is fine. Only flag genuinely-actionable foundation states (a reproducible LLK fail on a path the
    # op doesn't exercise); a not-probed or flaky-compile foundation is not actionable and stays quiet.
    if found_failed:
        return "FOUNDATION-ONLY FAIL: LLK probe failed but op does not => reproducible LLK path the op doesn't exercise; inspect the probe"
    if foundation in ("FLAKY_COMPILE", "FLAKY"):
        return "foundation probe was flaky (transient; isolated re-run passed) -- no action"
    return ""


# --------------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------------
def cmd_validate(args) -> int:
    print(f"Validating {len(select_cells(args.cells))} cell(s) (collection-only; no sim)\n")
    print(f"{'cell':<18} {'llk':>5} {'op':>5}  status")
    print("-" * 50)
    bad = 0
    for cell in select_cells(args.cells):
        lc = None if not cell.llk_test else len(matching_llk_ids(cell, args.timeout))
        oc = op_count(cell, args.timeout)
        flags = []
        if cell.llk_test and not lc:
            flags.append("LLK tokens MATCH 0 ids")
        if cell.op_k and not oc:
            flags.append("op -k MATCHES 0")
        if flags:
            bad += 1  # count unresolved cells, not individual flags
        print(
            f"{cell.id:<18} {('-' if lc is None else lc):>5} {('-' if oc is None else oc):>5}  "
            f"{', '.join(flags) if flags else 'ok'}"
        )
    print("\n" + ("OK: every mapped probe resolves." if not bad else f"PROBLEM: {bad} unresolved probe(s)."))
    return 1 if bad else 0


def cmd_run(args) -> int:
    sim_dir = Path(args.sim_dir)
    ensure_sim(sim_dir)
    cells = select_cells(args.cells)
    do_op = args.layer in ("both", "op")
    # Foundation probing is expensive (a full compile-producer per cell). In the default `both` mode we
    # run the OP probe first and fire the foundation (LLK) probe only for cells whose op FAILED -- the
    # only time we need to localize op-vs-foundation (the runbook's triage rule). `--layer foundation`
    # probes every selected cell (foundation-only); `--force-foundation` restores the exhaustive
    # op+foundation grid on every cell (use it on a sim/tt-llk bump, when checking the foundation is
    # exactly the point).
    # --force-foundation only widens the default `both` mode to probe every cell; it does NOT override an
    # explicit `--layer op` (which stays strictly op-only, per its contract and the flag's help text).
    force_found = args.layer == "foundation" or (args.force_foundation and args.layer == "both")

    results = {}
    record = {
        "timestamp": args.now or datetime.datetime.now().isoformat(timespec="seconds"),
        "tt_metal_sha": git_sha(REPO_ROOT),
        "tt_llk_sha": tt_llk_sha(),
        "craq_sim_sha": git_sha(Path("/workspaces/craq-sim")),
        "sim_dir": str(sim_dir),
        "layer": args.layer,
        "force_foundation": bool(args.force_foundation),
        "results": results,
    }
    mode = args.layer + ("+force-foundation" if args.force_foundation else "")
    print(f"Running {len(cells)} cell(s) on {sim_dir}  [layer={mode}]\n", flush=True)
    print(f"{'cell':<18} {'foundation':<12} {'op':<10} verdict", flush=True)
    print("-" * 78, flush=True)
    for cell in cells:
        o_status = run_op(cell, sim_dir, args.timeout) if do_op else "-"
        op_failed = o_status in ("FAIL", "NOCOLLECT", "HANG")
        if force_found:
            f_status = run_llk(cell, sim_dir, args.timeout)
        elif args.layer == "both" and op_failed:
            f_status = run_llk(cell, sim_dir, args.timeout)  # localize the failure on demand
        elif args.layer == "both":
            f_status = "NO_LLK" if cell.llk_test is None else "NOT_PROBED"  # op ok -> foundation not needed
        else:  # --layer op
            f_status = "-"
        verdict = ""
        if do_op:
            verdict = "; ".join(v for v in (triage(cell, f_status, o_status), expected_note(cell, o_status)) if v)
        results[cell.id] = {"foundation": f_status, "op": o_status, "expected": cell.expected}
        print(f"{cell.id:<18} {f_status:<12} {o_status:<10} {verdict}", flush=True)
        if args.out:  # write incrementally so a later hang/timeout still leaves a partial grid
            Path(args.out).write_text(json.dumps(record, indent=2) + "\n")

    if args.out:
        print(f"\nWrote {args.out}", flush=True)
    if args.baseline:
        diff_baseline(args.baseline, record)
    return 0


def diff_baseline(baseline_path: str, current: dict) -> None:
    base = json.loads(Path(baseline_path).read_text())
    print(
        f"\n=== diff vs {baseline_path} "
        f"(llk {base.get('tt_llk_sha')}->{current['tt_llk_sha']}, "
        f"sim {base.get('craq_sim_sha')}->{current['craq_sim_sha']}) ==="
    )
    regressions, progressions = [], []
    # States that carry no comparable signal: not-run ("-"), foundation skipped on an op-pass
    # ("NOT_PROBED"), or an inconclusive transient ("FLAKY_COMPILE"). Never diff these either way
    # (else an on-demand run that didn't probe a foundation cell would look like a regression).
    INCONCLUSIVE = (None, "-", "NOT_PROBED", "FLAKY_COMPILE", "FLAKY")
    for cell_id, cur in current["results"].items():
        prev = base.get("results", {}).get(cell_id, {})
        for layer in ("foundation", "op"):
            b, c = prev.get(layer), cur.get(layer)
            if b in INCONCLUSIVE or c in INCONCLUSIVE or b == c:
                continue
            if b == "PASS" and c != "PASS":
                regressions.append(f"{cell_id}/{layer}: {b} -> {c}")
            elif c == "PASS" and b != "PASS":
                progressions.append(f"{cell_id}/{layer}: {b} -> {c}")
    if regressions:
        print("REGRESSIONS (investigate / file):")
        for r in regressions:
            print(f"  - {r}")
    if progressions:
        print("PROGRESSIONS (enable in routing, update skips + QUASAR_PARITY_GAPS.md):")
        for p in progressions:
            print(f"  - {p}")
    if not regressions and not progressions:
        print("no status changes.")


def cmd_show_mapping(args) -> int:
    for cell in MAPPING:
        llk = f"{cell.llk_test}  tokens={cell.llk_tokens}" if cell.llk_test else "(no foundation primitive)"
        print(f"{cell.id:<16} [{cell.expected}]")
        print(f"    foundation: {llk}")
        print(f"    op:         {cell.op_k or '(not in focused op suite)'}")
        if cell.note:
            print(f"    note:       {cell.note}")
    return 0


# --------------------------------------------------------------------------------------------
# Foundation capability discovery: what the LLK no-broadcast binary suite supports, derived from
# its test collection (independent of our hand-written cells). Answers:
#   (1) --coverage   : LLK (op, dtype) the suite tests that our op does NOT cover  -> headroom.
#   (2) --supports OP: does the LLK suite already have a primitive for OP (e.g. pow) -> can we
#                      broaden the op now, or is it blocked on an LLK/sim port.
# --------------------------------------------------------------------------------------------
_FMT_CANON = {"float16_b": "bf16", "float16": "fp16", "float32": "fp32", "int32": "int32", "int8": "int8"}
_LLK_BINARY_FILES = [LLK_FPU, LLK_SFPU]  # the no-broadcast binary capability surface
_CANON_OPS = {"add", "sub", "mul", "div", "gt", "lt", "le", "ge", "max", "min", "eq", "ne"}
_OP_ALIAS = {
    "subtract": "sub",
    "multiply": "mul",
    "divide": "div",
    "greater": "gt",
    "less": "lt",
    "greater_equal": "ge",
    "less_equal": "le",
    "power": "pow",
}


def _canon_op(token: str) -> str:
    t = token.lower()
    t = t[3:] if t.startswith("elw") else t  # FPU Elwadd -> add
    return _OP_ALIAS.get(t, t)  # ttnn names (subtract/divide/power...) -> LLK canon


def _canon_fmt(token: str) -> str:
    return _FMT_CANON.get(token.lower(), token.lower())  # Mx* / unknowns pass through


_DTYPE_RE = re.compile(r"(Float16_b|Float16|Float32|Int32|Int8|MxFp8R|MxFp8P|MxFp4|MxInt8|MxInt4|MxInt2)")


def _id_dtype(test_id: str) -> str:
    m = re.search(r"A:(\w+),", test_id) or re.search(r"formats:(\w+)->", test_id)
    if m:
        return _canon_fmt(m.group(1))
    m = _DTYPE_RE.search(test_id)  # int / max-min families encode the format as a bare token
    return _canon_fmt(m.group(1)) if m else "?"


_SFPU_OP_RE = re.compile(r"[-\[](ADD|SUB|MUL|DIV|GT|LT|LE|GE|MAX|MIN|EQ|NE)[-\]]")


def _id_ops(test_id: str) -> list:
    """Canonical op(s) an LLK binary id represents (function-name aware); [] if not a binary op.

    The op token's position varies by family (FPU uses mathop:; SFPU float puts the op label last,
    SFPU int / max-min put tile_indices last), so SFPU is matched by op label ANYWHERE in the id.
    """
    func = test_id.split("::", 1)[1].split("[", 1)[0] if "::" in test_id else ""
    if "max_min" in func:
        return ["max", "min"]  # one test exercises both; op is in the function name, not the id
    m = re.search(r"mathop:(\w+)", test_id)  # FPU: ...-mathop:Elwadd-...
    if m:
        op = _canon_op(m.group(1))
        return [op] if op in _CANON_OPS else []
    m = _SFPU_OP_RE.search(test_id)  # SFPU op label (MUL/DIV/GT/...) anywhere -> position-independent
    if m:
        op = _canon_op(m.group(1))
        return [op] if op in _CANON_OPS else []
    return []


def llk_inventory(timeout: int) -> dict:
    """{(op, dtype): set(paths)} that the LLK no-broadcast binary suite tests on Quasar."""
    inv: dict = {}
    for f in _LLK_BINARY_FILES:
        path = "FPU" if f == LLK_FPU else "SFPU"
        for test_id in llk_collect_ids(f, timeout):
            dtype = _id_dtype(test_id)
            for op in _id_ops(test_id):
                inv.setdefault((op, dtype), set()).add(path)
    return inv


def _op_covered() -> set:
    """(op, dtype) pairs our op layer actually tests (a cell with an op_k)."""
    return {(c.id.split(".")[0], c.id.split(".")[1]) for c in MAPPING if c.op_k}


def cmd_coverage(args) -> int:
    inv = llk_inventory(args.timeout)
    covered = _op_covered()
    print(f"LLK no-broadcast binary capability vs our op coverage (LLK: {', '.join(_LLK_BINARY_FILES)})\n")
    print(f"{'op':<8} {'dtype':<8} {'LLK path':<10} op-covered?")
    print("-" * 44)
    headroom = []
    for (op, dtype), paths in sorted(inv.items()):
        cov = (op, dtype) in covered
        if not cov:
            headroom.append((op, dtype, "/".join(sorted(paths))))
        print(f"{op:<8} {dtype:<8} {'/'.join(sorted(paths)):<10} {'yes' if cov else 'NO -- headroom'}")
    print()
    if headroom:
        print("LLK SUPPORTS, our op does NOT cover/test (candidates to broaden into):")
        for op, dtype, paths in headroom:
            print(f"  - {op} @ {dtype} ({paths})")
    else:
        print("Our op covers every (op,dtype) the LLK binary suite tests.")
    return 0


def cmd_supports(args) -> int:
    query = _canon_op(args.supports)
    inv = llk_inventory(args.timeout)
    hits = sorted({(op, dt, "/".join(sorted(p))) for (op, dt), p in inv.items() if op == query})
    print(f"Does the LLK no-broadcast binary suite support '{args.supports}' (normalised '{query}')?\n")
    if hits:
        print("SUPPORTED in the binary LLK suite:")
        for op, dt, paths in hits:
            print(f"  - {op} @ {dt} ({paths})")
        print("\n=> foundation is present; broadening the op is wiring (add a cell + op test), not an LLK ask.")
        return 0
    print(f"NOT found in the no-broadcast binary LLK suite ({', '.join(_LLK_BINARY_FILES)}).")
    hits_elsewhere = []
    for d in (TT_LLK_TESTS / "python_tests" / "quasar", TT_LLK_DIR / "tt_llk_quasar"):
        if d.is_dir():
            out = subprocess.run(["grep", "-rilE", args.supports, str(d)], capture_output=True, text=True, timeout=60)
            hits_elsewhere += [ln for ln in out.stdout.splitlines() if ln.strip()]
    if hits_elsewhere:
        print("\nName appears elsewhere in tt-llk (maybe a different / unary family -- inspect):")
        for p in hits_elsewhere[:10]:
            print(f"  - {p}")
        print("\n=> not in the binary no-broadcast suite; check whether it's a different op family or unported.")
    else:
        print("\nNo hits anywhere in the Quasar LLK tests/sources => likely unported. File/await an LLK primitive.")
    return 1


def expected_note(cell: Cell, op: str) -> str:
    """Flag a single-run result that diverges from the cell's recorded `expected`."""
    if cell.expected == "KNOWN_GAP" and op == "PASS":
        return "PROGRESSION: expected a gap but op PASSES -> enable it + update mapping/skips/parity-doc"
    if cell.expected == "PASS" and op == "SKIP":
        return "DEVIATION: expected PASS but op SKIPPED"
    if cell.expected == "OP_COVERAGE_GAP" and op not in ("NO_OP_TEST", "-"):
        return "NOTE: coverage-gap cell now returns a result -> give it an op_k / flip expected"
    return ""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--show-mapping", action="store_true", help="print the op-cell <-> LLK-test mapping and exit")
    p.add_argument("--validate", action="store_true", help="collection-only check that every cell resolves (no sim)")
    p.add_argument("--run", action="store_true", help="run the grid on the QSR sim")
    p.add_argument(
        "--coverage",
        action="store_true",
        help="report LLK binary (op,dtype) the suite tests that our op does NOT cover (headroom)",
    )
    p.add_argument(
        "--supports",
        metavar="OP",
        help="ask whether the LLK binary suite already supports OP (e.g. pow, gt) before broadening",
    )
    p.add_argument("--layer", choices=["both", "foundation", "op"], default="both")
    p.add_argument(
        "--force-foundation",
        action="store_true",
        help="in --layer both, probe the foundation (LLK) for EVERY cell, not just op-failed cells "
        "(exhaustive, slow). Default: foundation is probed on-demand only when a cell's op fails. "
        "Use on a sim/tt-llk bump, when checking the foundation is the point.",
    )
    p.add_argument("--cells", help="fnmatch glob over cell ids, e.g. '*.bf16.*'")
    p.add_argument("--sim-dir", default=str(DEFAULT_SIM_DIR), help="dir containing libttsim.so")
    p.add_argument("--timeout", type=int, default=420, help="per-probe pytest/sim timeout (s)")
    p.add_argument("--baseline", help="prior JSON grid to diff against")
    p.add_argument("--out", help="write the JSON grid here")
    p.add_argument("--now", help="ISO timestamp to stamp the record with (default: now)")
    args = p.parse_args()

    if args.show_mapping:
        return cmd_show_mapping(args)
    if args.coverage:
        return cmd_coverage(args)
    if args.supports:
        return cmd_supports(args)
    if args.validate:
        return cmd_validate(args)
    if args.run:
        return cmd_run(args)
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
