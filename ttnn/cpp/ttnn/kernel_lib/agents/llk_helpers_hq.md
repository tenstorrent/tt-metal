# Kernel Helper Library HQ

Entry point for creating and maintaining `compute_kernel_lib` helpers — unified APIs that hide LLK init/compute/pack complexity in tt-metal compute kernels.

## Quick Start

| What you need | Document |
|---|---|
| **Writing helper code** (file structure, op structs, policies, CRTP bases, perf testing) | [llk_helpers_conventions.md](llk_helpers_conventions.md) |
| **Running the agent pipeline** (catalog, investigate, propose, validate, implement) | [llk_helpers_pipeline.md](llk_helpers_pipeline.md) |

## When to Use What

| Situation | Action |
|---|---|
| Adding ops to an existing helper | Read [conventions](llk_helpers_conventions.md) section 5 (op structs). Add CRTP struct + init/call. |
| Creating a new helper (known ops, known LLK calls) | Read [conventions](llk_helpers_conventions.md), write .hpp/.inl directly, add perf tests. |
| Creating a new helper (unknown territory) | Run the [pipeline](llk_helpers_pipeline.md) — new helper mode, starts at Phase 0. |
| Updating/improving an existing helper | Run the [pipeline](llk_helpers_pipeline.md) — update mode, starts at Phase 0 (reads existing files). |
| **Migrating an existing kernel to a helper** | See "Kernel Migration Steps" below. Always the last step — helper features must exist first. |

## Agent Files

| Agent | Pipeline Phase | Purpose |
|-------|---------------|---------|
| `llk_catalog_agent.md` | 0: Understand (new mode, step 1) | Catalog ops via bidirectional grep; produces group→ops + locator |
| `llk_investigation_agent.md` | 0: Understand (new mode, step 2) | Deep analysis per group (parallel); inline CONFIRMED/UNCERTAIN flags |
| (orchestrator) | 0: Understand (update mode) | Read existing `.hpp`/`.inl`, scope the change |
| `llk_helper_proposal_agent.md` | 1: Design | Full proposal (new) or delta proposal (update); handles L1 re-entry |
| `llk_validation_agent.md` | 2: Validate | Raw LLK → params → integration → perf; emits L1 trigger on failure |
| (orchestrator) + `llk_review_fix_agent.md` | 3: Implement | Write/edit files, L2 post-write validation, L3 scope gap detection, report |

**Deprecated**: `llk_verification_agent.md` (inline flags in investigation output replace it). `llk_device_validation_agent.md` is reference material for sub-stage 2a, not an agent.

## Feedback Loops

| Loop | Trigger | Path |
|------|---------|------|
| **L1** | Validation sub-stage 2a/2b fails, or 2c/2d fix needs API change | Phase 2 → Phase 1 (amend proposal) |
| **L2** | Files written or edited (always) | Phase 3 → Phase 2 (re-run 2c + 2d only) |
| **L3** | Scope gap found during implementation | Phase 3 → Phase 1 (amend design) → Phase 2 (validate new scope) → Phase 3 (resume) |

## Helpers Location

```
ttnn/cpp/ttnn/kernel_lib/
  {name}_helpers.hpp      <- declarations, enums, structs, examples
  {name}_helpers.inl      <- implementation
  agents/                 <- this directory (pipeline docs + agent prompts)
```

## Kernel Migration Steps

Migration is the FINAL step — it consumes helpers that already exist and are validated. If a missing op struct, missing enum value, or missing API surface is discovered mid-migration, stop and close that gap first (via "Adding ops" / "Updating helper" rows above), then resume.

### Step 1 — Audit the target kernel

For the kernel being migrated, enumerate:

- **Raw LLK calls it makes** (`*_tile_init`, `*_tile`, `reconfig_data_format*`, `cb_*`, `tile_regs_*`, `pack_tile*`). Classify each as: covered by an existing helper op struct, covered by a pipeline/chain primitive, or not yet covered.
- **CB lifecycle per operand**: waited-once-popped-once (persistent), waited+popped each iteration (streaming), or caller-managed (preloaded). This decides `LoadPolicy` for SFPU chains or `BinaryInputPolicy` for binary ops.
- **Dtype assumptions**: presence of `copy_tile_init_with_dt` / `pack_tile_with_dt` from `moreh_common.hpp` or explicit `reconfig_data_format_srca/_srcb` — indicates FP32_DEST_ACC paths that the helper must also handle. If the helper only does one entry-time reconfig and the kernel has multiple operand dtypes, flag this as a correctness risk and test with `fp32_dest_acc_en=True`.
- **Control-flow shape of the replaceable block**:
  - `if constexpr (compile_time_flag)` → trivially migratable; body is one chain.
  - `if (runtime_bool)` → write one `sfpu_pipeline` / `binary_op` call per branch. Each branch constructs its own compile-time chain; the runtime `if` just selects which to invoke. Do NOT try to lift the runtime condition into the chain DSL.
  - Interleaved SFPU+FPU in one DEST window (e.g. `add_tiles(...) ; rsqrt_tile(dst)`, `sub_tiles_bcast(...) ; exp_tile(dst)`) → use `binary_op_helpers` with a **chain-PostOp**: the final `PostOp` argument accepts `compute_kernel_lib::sfpu_chain(Op1{}, Op2{}, ...)` and runs it inside the same acquire/commit/wait/release window. Valid ops: any chain element. `Load<>` is permitted (the binary helper handles the unpacker init + per-tile reinit after the load). Ops on slots other than `D0` are permitted provided the caller pre-loads the secondary DEST slots.
  - Interleaved SFPU+FPU that needs to load a *third* tile mid-window (e.g. post-binary mask load) → cannot fuse; split into `binary_op(...)` (pack to intermediate CB) + `sfpu_pipeline(...)` that loads the extra tile. The intermediate CB is the cost of the split; the helper still removes all the boilerplate around each phase.

### Step 2 — Gate-check against the helper API

If the audit turns up ANY of these, return to a prior step before writing the migration:

- A needed op struct does not exist → add it (conventions §5), then resume.
- An enum value or policy is missing (e.g. partial reconfig mode, persistent load) → update the helper via the pipeline in update mode, then resume.
- The control-flow is Tier 3 — accept that this kernel stays on raw LLK and move on.

### Step 3 — Write the migration

Keep the scope surgical:

- **Only replace the LLK-call block**. Do NOT refactor surrounding code, rename CBs, or touch unrelated operations. If the kernel had reduce/binary helpers already, leave those calls alone.
- **Always fully qualify helper symbols** (`compute_kernel_lib::Load<...>`, `compute_kernel_lib::sfpu_pipeline(...)`). Do NOT introduce `using namespace compute_kernel_lib;` or namespace aliases such as `namespace ckl = compute_kernel_lib;`. Namespace pollution — even block-scoped — leaks helper names into later additions to that scope and hides the dependency on the helper library at the call site. The verbosity of full qualification is the point: it is searchable, auditable, and keeps unrelated identifiers from silently binding to helper symbols.
- **Delete dead locals** — `reduce_dst_idx`, `mask_dst_idx`, and similar DEST-index variables become unreachable once the chain manages DEST itself. Remove them.
- **Preserve surrounding CB lifecycle**. The pipeline honours the Load's `LoadPolicy` for wait/pop; if the kernel still does `cb_wait_front(cb_mask, N)` / `cb_pop_front(cb_mask, N)` at the top/bottom of the kernel scope (persistent CBs), those stay.
- **Swap `api/compute/*.h` includes** for `ttnn/cpp/ttnn/kernel_lib/*_helpers.hpp` only when the raw headers become unused. Do not drop headers still needed by unmigrated code paths in the same file.

### Step 4 — Verify on device

- Build (`./build_metal.sh`).
- Run the nightly pytest suite for this operation (e.g. `scripts/tt-test.sh --run-all tests/ttnn/nightly/unit_tests/operations/<op>/*.py`).
- Confirm both `fp32_dest_acc_en=False` and `fp32_dest_acc_en=True` cases pass when the operation tests FP32 accumulation.
- If any test fails: diagnose whether it's a helper gap (→ Step 2) or a migration mistake (→ Step 3).

### Step 5 — Record the migration

- Update `migration_blockers.md` or the relevant `*_analysis.md` to mark the kernel as migrated.
- Cross-reference the helper feature(s) it now consumes so later removals notice the dependency.

### Anti-patterns (do NOT do these during migration)

- Migrating across a helper gap by hand-coding the missing op inline. Fix the helper first.
- Restructuring control flow to fit the helper. If the kernel's runtime conditionals don't fit a single chain, use multiple `sfpu_pipeline` calls or leave that branch on raw LLK.
- Batching migrations of unrelated kernels in one commit. One kernel per commit; failures should bisect to a single change.
- Silently dropping FP32_DEST_ACC-guarded reconfig calls. Verify the helper emits an equivalent or flag the gap.
