# Agent Pipeline Architecture Review

## Based on: layer_norm operation (branch `dstoiljkovic/layer_norm_agents_1.1`)

---

## Pipeline Overview

```
                          ┌─────────────────────────────────┐
                          │         create-op SKILL          │
                          │      (orchestrator / router)     │
                          └──────────────┬──────────────────┘
                                         │
          ┌──────────────────────────────────────────────────────────┐
          │                    PHASE 0: Discovery                    │
          │              (runs inside create-op itself)              │
          │   Detects: input/output layouts → reference selection    │
          └──────────────────────────┬───────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
 ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
 │   ANALYZER    │          │   ANALYZER    │          │   ANALYZER    │
 │  (input_stage)│          │(compute_core) │          │(output_stage) │
 │   tilize      │          │   softmax     │          │   untilize    │
 └───────┬───────┘          └───────┬───────┘          └───────┬───────┘
         │                          │                          │
         │  tilize_analysis.md      │ softmax_analysis.md      │ untilize_analysis.md
         └───────────────────┬──────┴──────────────────────────┘
                             ▼
                   ┌───────────────────┐
                   │    ARCHITECT       │
                   │  Reads all 3 refs │
                   │  Produces:        │
                   │  • op_design.md   │
                   │  • .tdd_state.json│
                   │  • test files     │
                   └────────┬──────────┘
                            ▼
                   ┌───────────────────┐
                   │     BUILDER        │
                   │  Reads:           │
                   │  • op_design P1   │
                   │  • .tdd_state     │
                   │  Produces:        │
                   │  • entry point.py │
                   │  • descriptor.py  │
                   │  • stub kernels   │
                   │  • test suite     │
                   └────────┬──────────┘
                            ▼
                   ┌───────────────────┐
                   │  KERNEL WRITER    │
                   │  (TDD, single     │
                   │   persistent      │
                   │   agent)          │
                   │                   │
                   │  Reads:           │
                   │  • op_design P2   │
                   │  • .tdd_state     │
                   │                   │
                   │  Iterates:        │
                   │  stage0 → test →  │
                   │  stage1 → test →  │
                   │  stage2 → test →  │
                   │  stage3 → test →  │
                   │  DONE             │
                   └────────┬──────────┘
                            ▼
                   ┌───────────────────┐
                   │   REPORT (manual) │
                   │   REPORT.md       │
                   └───────────────────┘
```

---

## Parallelism Map

```
TIME ──────────────────────────────────────────────────────────────────────────────────►

Phase 0   │ Discovery │
          └───────────┘

Phase 1   │ analyzer(tilize) ─────────│
          │ analyzer(softmax) ────────│   ← TRUE PARALLEL (3 agents)
          │ analyzer(untilize) ───────│
          └───────────────────────────┘

Phase 2                               │ architect ────────│
                                      └──────────────────┘

Phase 3                                                    │ builder ──────────────────│
                                                           └──────────────────────────┘

Phase 4                                                                                │ TDD writer ─────────────────────────────────────│
                                                                                       │  ┌stage0┐┌stage1┐┌stage2┐┌stage3┐              │
                                                                                       └──────────────────────────────────────────────────┘

Phase 5                                                                                                                                    │report│
```

**Key observation**: The pipeline is almost entirely **sequential after Phase 1**. Only the 3 analyzers run in parallel. Everything else is a strict waterfall.

---

## Timing (layer_norm run)

| Phase | Agent | Duration | Verdict |
|-------|-------|----------|---------|
| 0: Discovery | (manual) | — | Clean |
| 1: Analysis | ttnn-operation-analyzer ×3 | ~2.5 min (parallel) | Clean |
| 2: Design | ttnn-operation-architect | ~4 min | Clean |
| 3: Build | ttnn-generic-op-builder | **~17 min** | Slow |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | **~29 min** | 4 failures across stages |
| **Total** | | **~53 min** | |

---

## Responsibility Matrix

| Agent | Reads | Writes | Can Modify Others' Output | Model |
|-------|-------|--------|---------------------------|-------|
| **Analyzer** | C++ program factory + kernels | `*_analysis.md` | No | Opus |
| **Architect** | analysis files + helper headers | `op_design.md`, `.tdd_state.json`, test files | No | Opus |
| **Builder** | `op_design.md` Part 1, `.tdd_state.json` | Python files, stub kernels, test suite | No | **Sonnet** |
| **TDD Writer** | `op_design.md` Part 2, `.tdd_state.json` | Kernel .cpp files | **Yes** (descriptor, entry point) | Opus |

---

## Responsibility Overlap

```
                    Python     Program     Stub        Real       Integration
                    Entry      Descriptor  Kernels     Kernels    Tests
                    ─────      ──────────  ───────     ───────    ─────────
  ARCHITECT         designs    designs     ·           designs    designs
                    ▼          ▼                       ▼          ▼
  BUILDER           CREATES    CREATES     CREATES     ·          CREATES
                    │          │           │                      │
  TDD WRITER        modifies   MODIFIES    replaces    CREATES    reads
                    (fixes)    (fixes)     (replaces)             (never touches)
```

The overlap between Builder and TDD Writer on the program descriptor is the main friction point. Either:
- The builder should produce a descriptor so correct that the TDD writer never needs to touch it, OR
- The TDD writer should own the descriptor from the start, with the builder only producing the Python entry point

---

## Strong Points

### 1. Clean separation of concerns via file contracts

Each agent communicates through well-defined files (`*_analysis.md` → `op_design.md` → `.tdd_state.json`). No agent needs to "understand" another agent's internal state — they just read the output file. This makes the pipeline debuggable: you can inspect any intermediate artifact.

### 2. The TDD staging concept is excellent

Progressive kernel implementation (identity → mean → normalize → affine) is the single best design decision. It caught the `prepare_reduce_scaler` issue at stage 1 instead of having to debug a full layer_norm with 8 compute phases. Each stage adds ~1-2 compute phases, keeping the debugging surface small.

### 3. Role-based analysis prompting

Telling the tilize analyzer to focus on "input_stage" and the untilize analyzer on "output_stage" means each analysis is targeted rather than exhaustive. The architect gets precisely the information it needs from each reference.

### 4. The architect's two-pass design (Part 1 + Part 2)

Part 1 (architecture) is hardware-agnostic: CB layout, data flow, math. Part 2 (implementation) maps to specific helper APIs. This separation means the builder reads Part 1 (it only needs shapes and CB indices) while the kernel writer reads Part 2 (it needs exact helper calls). Good division.

### 5. Budget-managed retry system

The FREE/HARD failure classification prevents the TDD writer from burning expensive attempts on compilation errors while still limiting retries on real bugs (hangs, numerical failures). This is a practical cost-control mechanism.

---

## Weak Points

### 1. The builder is a bottleneck and does too much (~17 min)

The builder (Sonnet model) creates Python files, stub kernels, test infrastructure, AND sometimes modifies framework files. In the layer_norm run, it touched 16 files including `chlkc_list.h` and `ttnn/__init__.py`. This scope creep makes it slow and error-prone.

```
        ARCHITECT                    BUILDER                     TDD WRITER
     ┌─────────────┐           ┌──────────────┐            ┌──────────────┐
     │ op_design.md │──────────▶│ 16 files     │───────────▶│ Fix builder  │
     │ .tdd_state   │           │ including    │            │ mistakes in  │
     │ test files   │           │ framework    │            │ descriptor   │
     └─────────────┘           │ globals      │            └──────────────┘
                                └──────────────┘
                                 Problem: too broad,
                                 uses weaker model
```

The builder uses **Sonnet** while every other agent uses **Opus**. For the most complex single-agent phase (creating the program descriptor with 13 CBs, 3 kernels, and all arguments), this is the wrong model. Descriptor bugs surface as cryptic hang/compile errors in Phase 4, wasting TDD writer budget.

### 2. No parallelism after Phase 1

The architect → builder → TDD writer chain is purely sequential. There's an unexploited opportunity:

```
CURRENT:
  architect ──► builder ──► TDD writer

POSSIBLE:
  architect ──┬──► builder (Python infra)  ──┐
              └──► TDD writer (kernels)  ────┘ merge
```

The builder and TDD writer could theoretically run in partial overlap if the builder committed its Python infrastructure first and the TDD writer started on the data_pipeline stage while the builder finishes tests. But the current design doesn't support this because the TDD writer needs the builder's stubs to exist first.

### 3. The architect → TDD writer information chain is lossy

The architect writes `op_design.md` with helper call signatures. But the **builder** sits between them and generates stub kernels with different include paths and argument structures. The TDD writer then reads both the design doc AND the builder's stubs, and must reconcile discrepancies.

```
  ARCHITECT says:                    BUILDER generates:             TDD WRITER sees:
  "use prepare_reduce_scaler         stub with basic includes      conflicting guidance →
   from dataflow helpers"            and no scaler setup           tries architect's API →
                                                                   compilation fails ×2
```

This is exactly what happened with `prepare_reduce_scaler`. The architect recommended it, the builder didn't include it in stubs, and the TDD writer followed the design doc blindly.

### 4. No validation gate between architect and builder

The architect outputs a design doc, commits it, and the builder runs. There's no check that the design doc is self-consistent or that its recommended APIs actually exist. A "design validator" step (even a lightweight Haiku agent) could catch:
- Helpers referenced in Part 2 that don't exist in the codebase
- Helpers used in the wrong kernel context (compute vs dataflow)
- CB indices that collide
- Argument count mismatches

### 5. The orchestrator (create-op skill) runs in the main conversation context

The create-op skill is not an agent — it runs in the outer Claude session. This means:
- It competes for context window with user conversation
- It can't be parallelized or backgrounded as a unit
- If the user interrupts or the session drops, pipeline state is partially lost (though `.tdd_state.json` provides recovery)

### 6. The TDD writer has too much authority with too little guardrails

The TDD writer can modify the program descriptor, entry point, and `__init__.py`. While this "integration authority" is necessary to fix builder mistakes, it means the TDD writer is effectively a second builder that also writes kernels. There's no diff review between what the builder produced and what the TDD writer changed.

In the layer_norm run, the TDD writer modified the descriptor to fix compile-time arg indexing — a bug that should have been caught in Phase 3.

### 7. No smoke test between builder and TDD writer

The builder runs tests on stub kernels, but stubs are trivially simple (passthrough or zeros). The first real hardware execution happens when the TDD writer runs stage 0 (data_pipeline). This is where the first hang occurred.

A minimal smoke test — "do the stubs run without hanging on a single tile?" — should be mandatory before Phase 4 starts.

---

## Failures Observed in layer_norm Run

| Stage | Failure | Classification | Retries | Root Cause |
|-------|---------|---------------|---------|------------|
| data_pipeline | Device hang | `hang_unknown` (HARD) | 1 hard | CB push/pop imbalance in initial stubs |
| subtract_mean | `get_dataformat` compile error | `compilation_error` (FREE) | 2 free | `prepare_reduce_scaler` not available in dataflow kernel context |
| affine_transform | Compile-time arg index OOB | `compilation_error` (FREE) | 1 free | TensorAccessor offset mismatch between descriptor and kernel |

**Total budget spent**: 1 hard attempt + 3 free retries across 4 stages.

---

## Recommendations

| Priority | Change | Impact |
|----------|--------|--------|
| **High** | Use Opus for the builder (or split builder into Opus-descriptor + Sonnet-boilerplate) | Fewer descriptor bugs reaching Phase 4 |
| **High** | Add a smoke test gate: builder must run `test_stage_data_pipeline` on stubs before committing | Catches hangs before TDD writer starts |
| **High** | Architect must validate helper availability by grepping the actual header files, not just referencing them from memory | Eliminates `prepare_reduce_scaler` class of errors |
| **Medium** | Add a lightweight design validator between architect and builder | Catches self-inconsistencies in op_design.md |
| **Medium** | Reduce builder scope: don't touch framework files, only operation directory | Faster, fewer side effects |
| **Medium** | TDD writer's descriptor modifications should be logged as "design deviations" in `.tdd_state.json` | Traceability for what the builder got wrong |
| **Low** | Explore partial overlap: builder commits Python first, TDD writer starts stage 0 while builder finishes tests | Could save ~10 min |
| **Low** | Move orchestration into a dedicated agent rather than the main conversation | Cleaner separation, resumable pipeline |
