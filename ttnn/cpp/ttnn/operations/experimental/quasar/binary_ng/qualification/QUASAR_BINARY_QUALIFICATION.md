<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Quasar binary-op qualification — runbook

How we test and qualify experimental-quasar binary ops on a foundation (LLK + craq-sim) that is
itself under active development. The goal is to (a) **localize** any failure to its owner and
(b) **re-baseline** on every foundation bump so regressions and newly-unblocked capability both
surface immediately.

This is the tt-metal-side companion to the foundation's own CI. We do **not** re-qualify the
simulator: craq-sim's `quasar-llk.yml` already runs the whole `test_*_quasar.py` LLK suite on the
QSR sim (8 shards, structured pass/fail/skip + failure signatures), and tt-llk gates its own
Quasar paths. We consume that and add a thin local layer.

## The two-layer model

Every binary-op capability (a "cell" = `op × dtype × compute-path`) sits on two layers:

| Layer | Probe | Answers |
|---|---|---|
| **Foundation** | LLK binary micro-tests (`test_eltwise_binary_quasar.py` FPU, `test_eltwise_binary_sfpu_quasar.py` SFPU), run **on the sim** so they transitively exercise craq-sim | "Is the primitive alive on the sim for this cell?" |
| **Op (ours)** | `test_binary_ng_no_bcast.py` (Quasar-runnable), each test asserting against a torch golden | "Did we build on it correctly?" |

"Correct" = the op test's PCC-vs-torch assertion. There is no Quasar silicon yet, so the sim is
the Quasar oracle; the sim's own fidelity-vs-hardware is craq-sim CI's concern, cross-checkable
via craq-sim's WH/BH sim builds (out of scope here).

## Triage rule (localize a failure)

When an op cell fails on the sim, run the **mapped LLK probe** for that cell:

```
op FAILS on sim
├── foundation (LLK) PASSES  → OP BUG. Fix in ttnn/.../experimental/quasar/binary_ng/.
├── foundation (LLK) FAILS   → FOUNDATION GAP. Not an op fix. Cross-check craq-sim quasar-llk.yml
│                              failure signatures; file against tt-llk or craq-sim (see below).
└── no foundation primitive  → KNOWN GAP (e.g. fp32 add/sub: SFPU float family is MUL/DIV only).
                               Keep the op test skipped with a reason; record in QUASAR_PARITY_GAPS.md.
```

This formalizes the manual bf16-mul investigation (PCC ~0.02): run the LLK SFPU float test for the
same `(op, dtype)`; if it passes, the gap is in the op, not the LLK/sim. **In `--run` default mode the
driver does this automatically** — it fires the LLK probe for exactly the cells whose op failed, so a
green run never pays the foundation-compile cost.

**Reading foundation-probe failures.** A foundation probe can fail transiently in a grid run — a
`FAIL(2)` *compile-step* failure on a cold build cache / parallel-compile contention, or a `FAIL(1)`
*test-run* failure when the probe races the sim the op probe just used (same localhost port). The driver
disambiguates both automatically: on any exit-1/2 it re-runs the same `--test-id` in isolation and reports
`FLAKY` / `FLAKY_COMPILE` when that clean re-run passes (transient — no action) versus `FAIL(1)` / `FAIL(2)`
only when it reproduces (a real tt-llk gap). Hangs (exit 5) are recorded as `HANG`, not retried. A
reproducible foundation `FAIL` while the op `PASS`es (the "FOUNDATION-ONLY FAIL" verdict) means the failing
LLK path is one our op does not exercise — not a blocker for us, but worth confirming it is not a narrow LLK gap.

## Filing a foundation gap (which repo)

Use craq-sim's simulator error taxonomy (the sim is intentionally stricter than silicon):

- `UnimplementedFunctionality` / `Unsupported` / `MissingSpecification` → **craq-sim** (ISA/sim gap),
  branch `quasar`. craq-sim also has a `/quasar-fix` autofix flow.
- LLK fails to JIT-compile (`#ifndef ARCH_QUASAR` guard) or computes a wrong result on the sim →
  **tt-llk** (`tt_llk_quasar/`), op unported or numerically wrong.
- `UndefinedBehavior` / `AssertionFailure` → simulated SW illegal / simulator bug respectively.

## The driver: `qualify_quasar_binary.py`

The authoritative op-cell ↔ LLK-test mapping lives in the driver. It runs our op probe per cell and —
on op-failure, or with `--force-foundation` — a **single representative LLK variant** (not the full
sweep — that is craq-sim CI's job) by collecting LLK ids through the harness venv (`tt-llk/tests/.venv`),
filtering by exact id-substrings, and running the first match as one `--test-id` through `run_test.sh`
(flock-serialised sim access). Op probes run under the QSR-sim env (`TT_METAL_SIMULATOR` + slow dispatch;
**not** `TT_METAL_MOCK_CLUSTER_DESC_PATH`, which expects a cluster — not soc — descriptor).

```bash
# See the mapping (op cell -> LLK probe + op probe):
python qualify_quasar_binary.py --show-mapping

# Confirm every cell resolves (collection-only, no sim) — run after editing the mapping or when
# an LLK/op test is renamed or reparametrized:
python qualify_quasar_binary.py --validate

# Default grid on the sim (op probe per cell; the foundation LLK probe fires only for a cell whose op
# FAILS, to localize it -- a green run pays no foundation-compile cost). Save a baseline:
python qualify_quasar_binary.py --run --out baseline.json

# Exhaustive: probe the foundation for EVERY cell too (slow; use on a sim/tt-llk bump):
python qualify_quasar_binary.py --run --force-foundation --out baseline.json

# Just one layer / a subset:
python qualify_quasar_binary.py --run --layer op         --cells '*.bf16.*'
python qualify_quasar_binary.py --run --layer foundation --cells 'mul.bf16.sfpu'
```

Prerequisites: the QSR sim built at `/workspaces/craq-sim/src/_out/release_qsr/libttsim.so`
(`cd /workspaces/craq-sim && ./make.py src/_out/release_qsr/libttsim.so`) and the LLK harness venv
(`cd tt_metal/tt-llk/tests && CHIP_ARCH=quasar ./setup_testing_env.sh`). Bound every run with a
hard `timeout` — a sim hang ignores pytest's `--timeout`; recover a wedged device/sim before re-running.

## Discovering what the foundation supports (the two qualification questions)

Both answers are derived **live from the LLK test collection**, so they self-update as tt-llk /
craq-sim evolve — no hand-maintenance:

**"What does LLK support that our op does not cover/test?"** (headroom to broaden into):
```bash
python qualify_quasar_binary.py --coverage
```
Lists every `(op, dtype)` the LLK no-broadcast binary suite tests, marking which our op covers and
which is headroom — e.g. `gt/lt/le/ge @ int32`, `max/min` (all dtypes), `add/sub/mul @ fp16/int8/Mx*`,
int `add/mul`. A headroom row you want is the prompt to add a cell + an op test.

**"Before I build op X (e.g. `pow`), does LLK already support it?"**:
```bash
python qualify_quasar_binary.py --supports pow       # NOT in the binary suite -> needs an LLK primitive/composite
python qualify_quasar_binary.py --supports max       # SUPPORTED @ bf16/fp16/fp32/int32 -> wiring only
python qualify_quasar_binary.py --supports subtract  # ttnn names alias to LLK ops (sub/mul/div/...)
```
SUPPORTED ⇒ broadening the op is wiring (add a cell + op test). NOT FOUND ⇒ it is an LLK/sim ask —
the report points at any other place the name appears in tt-llk (a unary family, the sources) so you
can tell "different family" from "unported". Scope: the FPU (`test_eltwise_binary_quasar.py`) and SFPU
(`test_eltwise_binary_sfpu_quasar.py`) no-broadcast suites; broadcast is a separate LLK file.

## Verify-on-bump procedure

When bumping the tt-llk pin, updating craq-sim, or rebuilding the sim. **The op unit test is the gate;
the qualification harness is the localizer / discovery / recorder — run the test first.** (The harness's
op layer only re-runs a subset of the unit test, so running it *first* would tell you strictly less; its
foundation grid is expensive and overlaps craq-sim's own `quasar-llk.yml` sweep.)

0. **Prep by trigger.**
   - **craq-sim changed** → rebuild the sim: `cd /workspaces/craq-sim && ./make.py src/_out/release_qsr/libttsim.so`
     (otherwise you are testing the old sim).
   - **only tt-llk changed** → no sim rebuild needed: the op JIT-compiles kernels from the `tt_metal/tt-llk`
     headers at run time. Just ensure the updated tt-llk is checked out (don't trust a stale kernel cache).
1. **Run the op unit test first — this is the gate.** It is the fast, comprehensive, authoritative signal
   for "does our op still work on the new foundation" (real FPU/SFPU paths, every layout, the fused
   activations, PCC-vs-torch), under the QSR-sim env (see "The driver" above for the exact env):
   ```bash
   <QSR-sim env>  pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_binary_ng_no_bcast.py
   ```
   - **Green** → nothing regressed; the gate passes. Done (optionally do step 2).
   - **Fails** → localize with step 3.
2. **Cheap no-sim discovery (always worth it):** `qualify_quasar_binary.py --coverage` and
   `--supports <op>` — did the bump *unblock* an LLK primitive you can now broaden the op into? Costs no
   sim; a new headroom row is the prompt to add a cell + an op test + drop a skip (a "progression").
3. **Localize a failure (only if step 1 failed):** `qualify_quasar_binary.py --run` (op-first) re-runs the
   failing cell and fires the LLK probe on failure → **OP BUG** (foundation passes → fix in the op) vs
   **FOUNDATION GAP** (LLK fails too → file tt-llk / craq-sim, not an op fix). See the triage rule above.
4. **Optional — record a foundation baseline for the bump:**
   `qualify_quasar_binary.py --run --force-foundation --baseline <prev>.json --out <new>.json` probes
   *every* foundation cell (not just op-failed ones) and diffs:
   - **REGRESSIONS** (was PASS, now not): if foundation regressed, file upstream; if only the op regressed, it is ours.
   - **PROGRESSIONS** (was not PASS, now PASS): enable the cell in op routing, drop the op-test skip, update `QUASAR_PARITY_GAPS.md`.
   Keep the new JSON as the reference. This is for a durable local record, not a required gate (craq-sim's
   `quasar-llk.yml` already sweeps the whole LLK suite). The `expected` field
   (`PASS` / `KNOWN_GAP` / `OP_COVERAGE_GAP`) flags a deviation even on a first run with no baseline.

## Pointers

- **LLK-facing gap inventory** — the binary ops + activations Wormhole has that Quasar lacks, written for
  the LLK team (classes, per-primitive file:line, fix pattern, priorities): `QUASAR_LLK_GAPS.md` (this dir).
  It is the curated snapshot of the two discovery folds above; keep it in sync on foundation bumps.
- Op parity inventory & known gaps: `QUASAR_PARITY_GAPS.md` (in the op dir,
  `ttnn/cpp/ttnn/operations/experimental/quasar/binary_ng/`).
- Foundation CI: craq-sim `.github/workflows/quasar-llk.yml` (LLK on sim), `quasar-ops.yml` (op lane,
  currently a matmul/resnet scaffold — our binary tests are not wired in upstream).
- LLK test runner: tt-llk's `run-test` skill / `.claude/scripts/run_test.sh`.
- craq-sim build/run recipe & error taxonomy: see the team's craq-sim notes.
