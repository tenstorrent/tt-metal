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
  tests/{feature}/        <- validation kernels + pytest (see "Validation Tests")
```

## Validation Tests

Every helper change (new helper OR update) MUST land alongside at least one
pytest that runs a custom compute kernel on device and validates against a
torch golden. Host-side `./build_metal.sh` is NOT sufficient — kernels JIT
at runtime, so compilation success is not correctness.

Existing suites — run these AND add to them for any change that touches the
covered surface:

| Suite | What it covers | Location |
|---|---|---|
| `test_helpers_chain_and_binary.py` | `sfpu_chain` Load lifecycle (fan-out, `LoadPolicy`, `NoWaitPop`); `binary_op` same-CB dedup; `DestReuseOp` as chain element; Load inside a PostOp chain (FPU-clash reinit) | `tests/ttnn/unit_tests/kernel_lib/` + kernels in `ttnn/cpp/ttnn/kernel_lib/tests/chain_and_binary/` |

When a new surface is added (new enum value, new policy, new op struct, new
reconfig path) the update MUST:

1. Add or extend a test kernel that exercises the exact new path.
2. Parameterize over at least `num_tiles ∈ {1, 8, 64}` — single tile, fits in
   DEST, and spans multiple DEST windows. These three cover most off-by-one
   and batching regressions.
3. Include a torch golden using `comp_pcc(...)` (>= 0.9999 for bf16-only
   paths; >= 0.999 when fp32 mixed dtypes are involved).
4. Skip Blackhole unless the feature is explicitly Blackhole-tested.

Run via:
```
scripts/tt-test.sh --run-all tests/ttnn/unit_tests/kernel_lib/*.py
```

Phase 2 (Validate) of the pipeline is responsible for adding these tests —
see [llk_validation_agent.md](llk_validation_agent.md) for the concrete
sub-stage 2c / 2d steps.

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

- Build (`./build_metal.sh`) — note this only compiles the host library.
  Kernels JIT at runtime. A passing build is necessary but insufficient.
- Run the kernel_lib validation suite to confirm the helpers themselves still
  work: `scripts/tt-test.sh --run-all tests/ttnn/unit_tests/kernel_lib/*.py`.
- Run the nightly pytest suite for this operation (e.g. `scripts/tt-test.sh --run-all tests/ttnn/nightly/unit_tests/operations/<op>/*.py`).
- Confirm both `fp32_dest_acc_en=False` and `fp32_dest_acc_en=True` cases pass when the operation tests FP32 accumulation.
- If any test fails: diagnose whether it's a helper gap (→ Step 2) or a migration mistake (→ Step 3).

### Step 5 — Record the migration

- Update `migration_blockers.md` or the relevant `*_analysis.md` to mark the kernel as migrated.
- Cross-reference the helper feature(s) it now consumes so later removals notice the dependency.

### Policy Mapping — Raw CB Lifecycle → Helper Policy

Use this table when reading a raw kernel to choose `BinaryInputPolicy` and `BinaryOutputPolicy`.

**Input policies**

| Raw pattern | Policy |
|-------------|--------|
| `cb_wait_front(A, 1); ... cb_pop_front(A, 1)` per tile | `WaitAndPopPerTile` |
| `cb_wait_front(A, N); ... cb_pop_front(A, N)` at end of block | `WaitUpfrontPopAtEnd` |
| `cb_wait_front(A, N)` once before loop, never popped in loop (persistent) | `NoWaitNoPop` (caller manages) |
| `cb_wait_front(A, N)` once before loop, popped once after loop | `WaitUpfrontPopAtEnd` if inside-helper block, `NoWaitPopAtEnd` if pre-waited by caller |
| Tiles already present (pushed by reader/dataflow, no wait in compute) | `NoWaitNoPop` |
| Cumulative wait (`cb_wait_front(A, base + i)` growing each iteration) | **Not supported** — leave on raw LLK |

**Output policies**

| Raw pattern | Policy |
|-------------|--------|
| `cb_reserve_back(1); pack_tile; cb_push_back(1)` per tile | `PerTile` |
| `cb_reserve_back(N)` upfront, `pack_tile` sequential, `cb_push_back(N)` at end | `Bulk` |
| `cb_reserve_back(chunk)` per chunk, `pack_tile` per tile, `cb_push_back(chunk)` per chunk | `PerChunk` |
| CB pre-reserved by caller before this block, `pack_tile` sequential, `cb_push_back(N)` at end | `Bulk` — a second `cb_reserve_back` on an already-reserved CB is a no-op (just checks free slots; write pointer hasn't advanced) |

### Blockers Checklist (Step 1 gate)

Before writing a migration, check each stage of the raw kernel for these patterns. **Any hit → that stage stays on raw LLK** (log as NOT-MIGRATED or PARTIAL):

1. **Non-zero absolute tile index on A or B** — `*_tiles(A, B, j, j + offset, j)` where `offset` is non-zero and runtime-varying or non-zero and compile-time. Helper tile indexing is sequential from 0.
2. **Cumulative wait** — `cb_wait_front(CB, base + iter * step)` where the count grows each iteration. No helper policy covers this.
3. **Non-sequential output pack** — `pack_tile(i, out_cb, absolute_idx)` where `absolute_idx` is not `base + i`. Helper `Bulk` packs sequentially; `PerChunk` packs at `wt` relative index.
4. **Asymmetric wait/process/pop** — `cb_wait_front(N); process(M); cb_pop_front(N)` with `M < N`. Helper's wait count == process count == pop count.
5. **In-place output with capacity=1** — output CB is the same as input A, AND the program factory allocates `num_tiles = 1` for that CB. Raw code does `cb_pop_front(A, 1); cb_reserve_back(A, 1)` (pop before reserve). Helper does `cb_reserve_back` before `cb_pop_front` — **deadlocks** on a capacity-1 CB. Check `{cb_name}_num_tiles` in the program factory before migrating. If capacity ≥ 2, migration is safe.
6. **Runtime op selection** — `if (runtime_flag) mul_tiles(...) else add_tiles(...)` dispatching at runtime. Write two separate helper calls in branches; cannot lift into a single helper call.
7. **L1 accumulation pack** — `pack_tile<true>(i, cb, 0)` with `llk_pack_reconfig_l1_acc(1)` that accumulates into the same output tile slot. No helper output policy covers this.
8. **Matmul interleaved** — `matmul_tiles(...)` inside the same loop as binary ops. Helpers are eltwise-only; split into separate matmul block + binary helper.

### Partial Migration

A kernel that has SOME migratable stages and SOME blocked stages is **PARTIAL** — not NOT-MIGRATED. Replace only the migratable stages; leave blocked stages on raw LLK. Document which stages were migrated and which blockers remain.

Good indicators that partial migration is worth doing:
- The migratable stage involves multiple CB lifecycle calls (`wait + acquire + init + loop + commit + pack + release + pop`) — the helper reduces all of these to one call.
- The blocker is in a different stage that has no dependency on the migratable stage's output CB.

### Verifying the Test Exercises the Changed Kernel

Kernel filenames are NOT unique across the repo (e.g. `rmsnorm_post_allgather.cpp` exists in multiple directories). A passing test only validates the kernel if the test's program factory compiles THAT file.

Before trusting a passing test:

1. Grep for the **exact relative path** string in program factories:
   ```
   grep -rn "path/to/kernel.cpp" ttnn/cpp --include="*.cpp"
   ```
2. Trace the factory back to the `ttnn.*` op it creates.
3. Confirm the test file calls that op (grep for `ttnn.op_name` in the test).
4. Optional paranoid check: introduce a `static_assert(false, "sentinel")` in the kernel, run the test, confirm it FAILS to compile, then revert.

### Anti-patterns (do NOT do these during migration)

- Migrating across a helper gap by hand-coding the missing op inline. Fix the helper first.
- Restructuring control flow to fit the helper. If the kernel's runtime conditionals don't fit a single chain, use multiple `sfpu_pipeline` calls or leave that branch on raw LLK.
- Batching migrations of unrelated kernels in one commit. One kernel per commit; failures should bisect to a single change.
- Silently dropping FP32_DEST_ACC-guarded reconfig calls. Verify the helper emits an equivalent or flag the gap.
- Marking a kernel NOT-MIGRATED when only some stages are blocked. Log as PARTIAL, migrate the clean stages, record the specific blocker per blocked stage.
- Assuming the first passing test validates your edit. Verify the test exercises your exact kernel file — same-named files in other directories can silently shadow the one you changed.
