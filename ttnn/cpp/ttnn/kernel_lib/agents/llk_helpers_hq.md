# Kernel Helper Library HQ

Guide for creating `compute_kernel_lib` helpers — unified APIs that hide LLK init/compute/pack complexity in tt-metal compute kernels.

## Library Overview

All helpers live in `ttnn/cpp/ttnn/kernel_lib/`, use the `compute_kernel_lib` namespace, and follow a header-only `.hpp` (declarations) + `.inl` (implementation) split.

### Existing Helpers

| Helper | What it unifies | Key patterns |
|---|---|---|
| `tilize_helpers.hpp` | tilize/fast_tilize init/block/uninit | Config enums, CB indices as template params, auto fast_tilize dispatch |
| `untilize_helpers.hpp` | untilize/pack_untilize | Auto-dispatches pack_untilize vs standard based on width/datatype |
| `reduce_helpers_compute.hpp` | reduce ROW/COL/SCALAR with SUM/AVG/MAX | InputPolicy enum, Accumulate type trait, PostReduceOp callback, auto DEST limit detection |
| `dest_helpers.hpp` | DEST register capacity detection | constexpr functions, JIT header integration, sync/accum mode auto-detect |
| `binary_op_helpers.hpp` | add/sub/mul with broadcast | BroadcastDim, input/output policies, DEST chunking, post-op callback |
| `common_types.hpp` | Shared types | `NoOp`, `NoAccumulation` |
| `cb_helpers.hpp` | CB query utilities | tile size, page count, validation |

## Architecture Conventions

Every helper MUST follow these patterns. Read existing helpers before creating new ones.

### 1. File structure

```
ttnn/cpp/ttnn/kernel_lib/
  {name}_helpers.hpp      ← declarations, enums, structs, doc comments, examples
  {name}_helpers.inl      ← implementation, #include'd at bottom of .hpp
```

The `.hpp` contains ALL documentation — usage examples in the file-level docblock, per-function doc comments, enum/struct descriptions. The `.inl` contains only implementation code.

### 2. Namespace and includes

```cpp
// .hpp
#pragma once
#include "ttnn/cpp/ttnn/kernel_lib/common_types.hpp"

namespace compute_kernel_lib {
// declarations
}
#include "{name}_helpers.inl"

// .inl
#pragma once
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
// ... operation-specific LLK includes
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace compute_kernel_lib {
// implementations
}
```

### 3. Policy enums

Each helper defines its own policy enums following the established naming pattern:

- **InputPolicy**: Controls when to wait for input tiles and whether to pop them
  - `WaitAndPopPerTile` — streaming, one at a time (default, safest)
  - `WaitUpfrontNoPop` — tiles persist for reuse
  - `WaitUpfrontPopAtEnd` — wait all, pop all at end
  - `NoWaitNoPop` — caller manages externally
  - `NoWaitPopAtEnd` — caller manages wait, pop at end

- **OutputPolicy**: Controls when to reserve/push output tiles
  - `PerTile` — reserve/push one at a time (default)
  - `Bulk` — reserve all upfront, push all at end
  - `PerChunk` — (binary_op only) reserve/push per DEST chunk

- **DataFormatReconfig**: Controls unpacker/packer reconfiguration
  - `NONE`, `INPUT`, `OUTPUT`, `INPUT_AND_OUTPUT` (default)

Not every helper needs all variants. Only add what the use cases require.

### 4. DEST register management

Use `DEST_AUTO_LIMIT` from `dest_helpers.hpp` for automatic chunking. The pattern:

```cpp
constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
for (uint32_t base = 0; base < total; base += dest_limit) {
    uint32_t chunk = min(dest_limit, total - base);
    tile_regs_acquire();
    // process chunk tiles into DEST[0..chunk-1]
    tile_regs_commit();
    tile_regs_wait();
    // pack chunk tiles from DEST
    tile_regs_release();
}
```

Per-tile streaming is the special case where chunk=1 and acquire/release wrap each tile individually.

### 5. Op-type structs (for polymorphic dispatch)

When a helper supports multiple operations through the same API, use lightweight structs:

```cpp
template </* op-specific template params */>
struct OpName {
    ALWI void init() const { /* call *_tile_init() */ }
    ALWI void apply(uint32_t dst_idx) const { /* call *_tile(dst_idx) */ }
};
```

Each struct carries its configuration as template parameters and exposes a uniform `init()`/`apply()` interface. The main function is templated on `typename Op` and calls `op.init()` / `op.apply(dst_idx)`. This lets callers select the operation by passing a struct instance:

```cpp
helper_func(cb_in, cb_out, n, Exp<>{});
helper_func(cb_in, cb_out, n, Recip<>{});
```

### 6. PostOp callbacks

For operations that need a follow-up computation (e.g., recip after reduce), accept a callable:

```cpp
template <typename PostOp = NoOp>
void main_func(..., PostOp post_op = {});

// Called as: post_op(dst_idx);
```

`NoOp` from `common_types.hpp` compiles away entirely.

### 7. Function signatures

- Use `ALWI` (always inline) on all functions
- Template params: operation config first, then policies, then PostOp/Accumulate
- Function params: input CBs, output CB, shape/count, then optional PostOp/Accumulate
- Provide convenience aliases where the operation type is the only difference (e.g., `add()`, `sub()`, `mul()` all delegate to `binary_op()`)

### 8. LLK Sequence Rules

**Critical**: Each helper internally calls LLK init and exec functions. These sequences MUST match patterns found in existing kernels. A helper that composes LLK calls in a novel sequence (not seen in any codebase kernel) risks silent data corruption or hangs.

Rules:
- Each `*_tile_init()` must immediately precede its corresponding `*_tile()` calls
- Some inits are mutually exclusive (e.g., `copy_tile_init` and `reduce_init` reconfigure the same hardware)
- After a disruptive init (like `copy_tile_to_dst_init_short_with_dt`), subsequent operations may need re-initialization
- When in doubt, verify the sequence against an existing kernel that uses the same call pattern

## Creating a New Helper

### Decision: Direct vs Investigation Path

| Situation | Path | What to do |
|---|---|---|
| Clear spec (ops, params, LLK calls all known) | **Direct** | Read existing helpers + LLK wrapper headers → implement |
| Unknown territory (new category, unclear init rules) | **Investigation** | Run the agent pipeline below to research first |
| Adding ops to existing helper | **Direct** | Read the helper + new op's wrapper header → add struct |

### Direct Path

1. **Read the LLK wrapper headers** for each operation you're wrapping:
   - Location: `tt_metal/hw/inc/api/compute/eltwise_unary/{op}.h` (or equivalent)
   - Note: `*_tile_init()` template params, `*_tile()` params, any enums
2. **Read an existing kernel** that uses these operations to verify the LLK call sequence:
   - Location: `ttnn/cpp/ttnn/operations/eltwise/*/device/kernels/compute/*.cpp`
   - Note: order of init/copy/compute/pack, DEST acquire/commit/wait/release placement
3. **Read the closest existing helper** as a template (see Existing Helpers table)
4. **Create `.hpp` + `.inl`** following the conventions above
5. **Write a test kernel** that uses the new helper and verify against the original

### Investigation Path (Agent Pipeline)

Use this when you need to systematically discover and analyze operations before designing a helper. The pipeline has 6 stages.

```
Stage 1 (Catalog)
    → Stage 2 (Investigation: 3 parallel agents per group — Device, Host, Usage+Params)
    → Stage 3 (Verification)
    → Stage 4 (Proposal + Structs)
    → Stage 5 (Review-Fix + Device Validation + Param Coverage Loop)
    → Stage 6 (Implementation)
```

See [Investigation Pipeline Details](#investigation-pipeline-details) for full agent specifications.

## Known Operation Categories

Parameters needed to run the investigation pipeline or locate LLK sources for each category.

| Category | LLK Prefix(es) | Compute API Dir | LLK Dir | Codegen / Enum File | Program Factory Dir | Notes |
|---|---|---|---|---|---|---|
| Elementwise unary | `llk_math_eltwise_unary_sfpu`, `llk_math_eltwise_unary_datacopy` | `hw/inc/api/compute/eltwise_unary/` | `hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/` | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | `ttnn/cpp/ttnn/operations/eltwise/unary/device` | Largest category. Many ops use SFPU macros directly. |
| Binary eltwise | `llk_math_eltwise_binary` | `hw/inc/api/compute/eltwise_binary/` | `hw/ckernels/wormhole_b0/metal/llk_api/` | `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` | `ttnn/cpp/ttnn/operations/eltwise/binary/device` | |
| Ternary SFPU | `llk_math_eltwise_ternary_sfpu` | `hw/inc/api/compute/eltwise_unary/` | `hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/` | (shared with unary) | (shared with unary) | Some ternary ops filed under unary directory. |
| Matmul | `llk_math_matmul` | `hw/inc/api/compute/matmul/` | `hw/ckernels/wormhole_b0/metal/llk_api/` | — | `ttnn/cpp/ttnn/operations/matmul/device` | |
| Reduce (FPU) | `llk_math_reduce` | `hw/inc/api/compute/reduce/` | `hw/ckernels/wormhole_b0/metal/llk_api/` | — | `ttnn/cpp/ttnn/operations/reduction/device` | |
| Pack | `llk_pack` | `hw/inc/api/compute/pack/` | `hw/ckernels/wormhole_b0/metal/llk_api/` | — | — | |
| Unpack | `llk_unpack` | `hw/inc/api/compute/unpack/` | `hw/ckernels/wormhole_b0/metal/llk_api/` | — | — | |
| Tilize | `llk_math_fast_tilize`, `llk_math_eltwise_unary_datacopy` | `hw/inc/api/compute/tilize/` | `hw/ckernels/wormhole_b0/metal/llk_api/` | — | — | |

All paths relative to repo root. `tt_metal/` prefix for hw paths, repo root for ttnn paths. `—` = no known file.

## Migration Strategy

When migrating existing kernels to use a new helper:

| Tier | Criteria | Action |
|------|----------|--------|
| **Tier 1** (Easy) | Direct 1-for-1 swap. Pattern matches helper exactly. | Migrate first. Proof-of-concept + test baseline. |
| **Tier 2** (Medium) | Minor restructuring needed: PostOp callback, policy change. | Migrate after Tier 1 verified. |
| **Tier 3** (Hard) | Conditional dispatch, multi-CB reread, interleaved FPU+SFPU. | Evaluate case-by-case. Often better left manual. |

Tier 3 kernels are **not failures of the helper** — they confirm the helper's scope is correctly bounded.

---

## Investigation Pipeline Details

Full specifications for the agent-driven research pipeline. Use when the Direct Path doesn't apply.

### Pipeline Diagram

Each box is one agent invocation. Parallel boxes run concurrently.
`[ORCH]` = orchestrator (main Claude) aggregation step. `[H#]` = human checkpoint.

```
                    ┌──────────────────────┐
                    │       Stage 1        │
                    │      Catalog         │   (one per category)
                    │       Agent          │
                    └──────────┬───────────┘
                               │
              ╔════════════════╧══════════════════╗
              ║  [H1] Validate LLK list and        ║  ← cheapest to fix
              ║  group assignments before dispatch ║
              ╚════════════════╤══════════════════╝
                               │
                    (for each group, launch 3 parallel)
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
  ┌──────▼───────┐     ┌───────▼──────┐     ┌───────▼──────┐
  │   Stage 2    │     │   Stage 2    │     │   Stage 2    │
  │    Device    │     │     Host     │     │  Usage       │
  │    Agent     │     │    Agent     │     │  + Params    │
  │ (per group)  │     │ (per group)  │     │ (per group)  │
  └──────┬───────┘     └───────┬──────┘     └───────┬──────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │  [ORCH] Consolidate  │
                    │  per-group outputs   │
                    │  → investigation.md  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │       Stage 3        │
                    │    Verification      │
                    │       Agent          │
                    └──────────┬───────────┘
                               │
              ╔════════════════╧══════════════════╗
              ║  [H2] Check INCORRECT verdicts:    ║
              ║  does evidence actually contradict ║
              ║  the claim? (high-value finds)     ║
              ╚════════════════╤══════════════════╝
                               │
                    ┌──────────▼───────────┐
                    │       Stage 4        │
                    │  Proposal + Structs  │
                    │       Agent          │
                    └──────────┬───────────┘
                               │
              ╔════════════════╧══════════════════╗
              ║  [H3] Review proposal:             ║
              ║  · LLK Sequence Validation table   ║
              ║    — any UNVALIDATED sequences?     ║
              ║    — any mutual-exclusion VIOLATIONs║
              ║  · Before/After vs. real code       ║
              ║  · Tier 3 call-site decisions       ║
              ║  · pytest commands exist?           ║
              ╚════════════════╤══════════════════╝
                               │
                    ┌──────────▼───────────┐
                    │       Stage 5        │
                    │  Review-Fix Loop     │
                    │  + Device Validation │
                    │       Agent          │
                    └──────────┬───────────┘
                               │
              ╔════════════════╧══════════════════╗
              ║  [H4] Route FEEDBACK_TARGET:       ║
              ║  accept upstream fix or re-enter   ║
              ║  earlier stage? (judgment call)    ║
              ╚════════════════╤══════════════════╝
                               │
                    ┌──────────▼───────────┐
                    │       Stage 6        │
                    │   Implementation     │
                    │       Agent          │
                    └──────────┬───────────┘
                               │
              ╔════════════════╧══════════════════╗
              ║  [H5] Final review:                ║
              ║  · Tests pass? Code correct?       ║
              ║  · Migration tier 1 sites updated? ║
              ╚════════════════╤══════════════════╝
                               │
                          [ done ]
```

Stage 5 (Review-Fix + Device Validation + Parameter Coverage + Helper Integration + Performance) runs autonomously — no human intervention between Phase A (document review), Phase B (device validation with raw LLK), Phase C (parameter coverage across dtypes/template args/runtime args), Phase D (helper integration with dtype/arg/policy variation), and Phase E (performance comparison of helper vs raw LLK). The agent generates test kernels, builds, runs on device, fixes proposal sequences on failure, exercises the full parameter space, validates the actual helper API end-to-end, and measures performance overhead. Human checkpoint [H4] only triggers after Stage 5 completes all five phases.

### Agent Descriptions

#### Stage 1: Catalog Agent
**File**: `llk_catalog_agent.md`
**When**: Starting investigation of a new operation category. Runs once per category.
**What it does**: Lightweight enumeration — greps for LLK prefix functions (bottom-up) and compute API wrapper names (top-down) without reading file bodies. Performs gap analysis, assigns ops to functional groups. Also locates all source files for each LLK (wrapper headers, LLK/ckernel files, codegen entries, program factories, custom kernels) and produces a **Locator Results table**.
**Output**: Full op list, gap analysis, **group→ops assignment table**, and **Locator Results table** (op → wrapper file, LLK file, codegen file, program factory, custom kernel paths).
**Skip if**: The op list, groups, and file locations are already known — go directly to Stage 2.

#### Stage 2: Investigation (3 parallel agents per group)
**Files**: `llk_investigation_device_agent.md`, `llk_investigation_host_agent.md`, `llk_investigation_usage_agent.md`
**When**: After Stage 1. Launch all 3 in parallel **for each functional group**.
**What they do**: Each agent receives the Locator Results for all LLKs in its group and investigates all of them.
- **Device agent** → wrapper signatures, LLK impl, init state, **DEST batching limits per op**, **FP32 accumulation requirements**, **disruptive inits list**
- **Host agent** → codegen, program factory, CB layout, **Parameter Encoding Reference table** (user API value → host transform → kernel receives)
- **Usage+Params agent** → ALL kernel call sites, boilerplate patterns, chaining, init mutual exclusion, **and full parameter collection** (see below)

**Output**: Structured tables. Orchestrator consolidates to `{category}_investigation.md`.

##### Data Flow Contracts

Each Stage 2 agent MUST produce specific outputs that downstream stages consume. Missing outputs block the pipeline.

**Device Agent → consumed by Stage 4 (Proposal)**:
- Wrapper Signatures table (init & exec signatures)
- Init State Compatibility table (which inits can coexist)
- **DEST Batching Limits table** (max tiles per DEST batch per op) — Proposal uses this for chunking logic
- **FP32 Accumulation Requirements** (boolean per op) — Proposal uses this for accumulation mode setup
- **Disruptive Inits list** (which inits require re-calling after copy_tile) — Proposal uses this for sequence validation
- Outliers table — Proposal MUST explicitly decide include/exclude per outlier

**Host Agent → consumed by Stage 4 (Proposal)**:
- Code Generation table
- Program Factory layout (CB layout, runtime args)
- **Parameter Encoding Reference table** (complete mapping: user API value → host transform → what kernel receives → could kernel compute it?) — Proposal uses this directly for op struct field design instead of re-deriving
- Factory Sharing status — Proposal uses this for grouping decisions

**Usage+Params Agent → consumed by Stage 3 (Verification) and Stage 4 (Proposal)**:
- Call Sites table
- Init/Exec Pairing Rules — Verification checks these; Proposal uses for sequence validation
- Init Mutual Exclusion table — Proposal MUST cross-reference when grouping ops in same helper
- Chaining Patterns table — Proposal MUST reference when designing multi-op helpers
- **Parameter Usage Matrix per LLK** (see below) — feeds Stage 5 Phase C directly

##### Parameter Collection (Usage Agent responsibility)

The Usage agent MUST collect **all arguments and parameters** observed at every call site of the LLK — not just data types. Each LLK function has a full signature (template params + function args), and we need to know which values are actually used in the codebase.

For each call site found, record **every parameter dimension**:

1. **Data formats / dtypes**: Input and output `DataFormat` / `DataType` from CB configs (e.g., `Float16_b`, `BFloat16`, `Float32`). Note any `unpack_reconfig_data_format` / `pack_reconfig_data_format` calls.
2. **Template parameters**: Values passed as template args to `*_tile_init()` and `*_tile()` (e.g., approximation mode `true`/`false`, `ELU` vs `RELU`, `EltwiseBinaryType::ELWADD`).
3. **Function arguments**: Runtime args to init and compute calls (e.g., `dst_index`, dimension enums like `ReduceDim::ROW`, scalar params, coefficient values).
4. **Init configuration**: Which init variant is called (`*_init` vs `*_init_short` vs `*_init_short_with_dt`), and what params it receives.
5. **Surrounding context**: Whether the call site uses DEST acquire/release per-tile or per-chunk, whether there's a math_fidelity define, whether `APPROX` is defined, etc.

**Output format** — append a **Parameter Usage Matrix** per LLK:

```markdown
### Parameter Usage Matrix: {llk_name}

#### Function Signature (from wrapper header)
- `template <bool approx_mode> void exp_tile_init()`
- `template <bool approx_mode> void exp_tile(uint32_t dst_idx)`

#### Observed Parameter Values

| Param | Type | Observed Values | Call Sites |
|---|---|---|---|
| approx_mode (init) | template bool | `true` (14 sites), `false` (2 sites) | eltwise_sfpu.cpp:42, softmax.cpp:18, ... |
| approx_mode (compute) | template bool | `true` (14 sites), `false` (2 sites) | (same as init) |
| dst_idx | runtime uint32_t | 0..7 (tile loop) | all sites |
| input dtype | CB DataFormat | Float16_b (12), BFloat16 (3), Float32 (2) | ... |
| output dtype | CB DataFormat | Float16_b (12), BFloat16 (3), Float32 (2) | ... |
| math_fidelity | define | HiFi4 (8), HiFi2 (6), LoFi (3) | ... |
| DEST mode | context | per-tile (10), per-chunk (7) | ... |

**Observed combos**: {approx=true, Float16_b, HiFi4} (8 sites), {approx=true, BFloat16, HiFi2} (3 sites), ...
**Never observed**: {approx=false, BFloat4_b, *}, {*, Int32, *} (candidates for unsupported assertion)
```

This matrix feeds directly into Stage 5 parameter coverage testing.

#### Stage 3: Verification
**File**: `llk_verification_agent.md`
**When**: After Stage 2. Mandatory.
**What it does**: Takes claims from investigation, checks each against actual code. Returns CONFIRMED / INCORRECT / UNVERIFIABLE per claim.
**Output**: `{category}_verification.md`.

**INCORRECT verdicts are high-value findings** — they directly change the helper implementation.

#### Stage 4: Proposal + Op Structs
**File**: `llk_helper_proposal_agent.md`
**When**: After Stages 1-3.
**What it does**: Proposes helper API (signatures, enums, dispatch, before/after examples, migration tiers) + designs op-type-trait structs. Uses upstream data directly:
- **DEST Batching Limits** (from Device) → chunking logic in helper
- **FP32 Accumulation Requirements** (from Device) → accumulation mode setup
- **Disruptive Inits** (from Device) → LLK Sequence Validation
- **Parameter Encoding Reference** (from Host) → op struct field design (no re-derivation)
- **Init Mutual Exclusion** (from Usage) → validates grouping decisions
- **Chaining Patterns** (from Usage) → multi-op helper design
- **Outliers** (from Device) → explicit include/exclude decision per outlier
- **Factory Sharing** (from Host) → grouping decisions
- **Open Questions** generated here MUST be resolved before Stage 5 exit

**Output**: `{category}_helper_proposal.md`.

#### Stage 5: Review-Fix + Device Validation + Parameter Coverage + Performance
**Files**: `llk_review_fix_agent.md`, `llk_device_validation_agent.md`
**subagent_type**: `general-purpose`
**When**: After Stage 4. Mandatory before implementation.
**What it does**: Five-phase loop.

- **Phase A (Document Review)**: Reviews target document, classifies issues (BLOCKER/CONFUSING/MINOR), fixes, re-reviews until 0 document blockers.

- **Phase B (Device Validation — Raw LLK)**: Takes the reviewed proposal's LLK Sequence Validation table, generates test kernels that exercise the EXACT proposed LLK call sequences using raw API calls (not the helper), runs them on device against golden references, and reports pass/fail/hang per sequence. If any device test fails or hangs, routes back to Phase A with the failure as a BLOCKER. These raw LLK test kernels are kept for Phase E performance comparison.

- **Phase C (Parameter Coverage Testing)**: Uses the Parameter Usage Matrix from Stage 2 to systematically test each LLK across its FULL parameter space — data types, template arguments, runtime arguments, and their cross-products. This is not sampling — every dimension must be exercised.

  **Three mandatory test dimensions:**

  1. **Data format coverage**: For EVERY op, test with at minimum these input/output data format combinations:
     - Float16_b → Float16_b (default)
     - BFloat16 → BFloat16
     - Float32 → Float32
     - Mixed: Float16_b → Float32 (if the helper supports mixed I/O)

     Create SEPARATE CB configurations for each DataFormat. Do not test only the default. If the investigation found the codebase only uses one format, still probe the others — the purpose is to discover what works, not just confirm what is already used.

  2. **Template argument coverage**: For EVERY op that has non-Dst template parameters (Approx, Legacy, RoundingMode, DataFormat, etc.), test EVERY value of each parameter:
     - Approx: both `Exact` and `Fast`
     - Legacy: both `Off` and `On`
     - RoundingMode: `None`, `Trunc`, `Floor`
     - DataFormat template args on ternary ops: at minimum Float16_b and Float32

     A helper that only tests default template args is NOT validated.

  3. **Runtime argument coverage**: For EVERY op that has runtime parameters (scalar, alpha, threshold, etc.), test with at minimum:
     - Typical value (e.g., alpha=1.0)
     - Edge value (e.g., alpha=0.0, threshold=±inf, scalar=very large)
     - Negative value where semantically meaningful

     A helper that only tests one runtime argument value is NOT validated.

  **Test matrix structure**: Each op produces a combinatorial matrix of (dtype × template_args × runtime_args). Full cross-product is not required — instead, use a covering-array strategy:
  - Test each dtype with default template/runtime args
  - Test each non-default template arg with the default dtype
  - Test each edge-case runtime arg with the default dtype
  - Test at least ONE non-default cross-product combo (e.g., Float32 + Approx::Fast + edge scalar)

  **Output per LLK**: A **Parameter Support Matrix** classifying each parameter combination as SUPPORTED, UNSUPPORTED (→ assert in helper), or UNTESTED, with the specific test evidence.

  **Phase C CAN loop back.** If parameter testing reveals that an observed combo (one actually used in the codebase) FAILS, this is a BLOCKER — it means the investigation or proposal has incorrect assumptions. Route back to Phase A with the failure diagnosis, fix the proposal, and re-run. Only unobserved combos that fail are simply recorded as UNSUPPORTED.

- **Phase D (Helper Integration Testing)**: After raw LLK validation passes, write SEPARATE test kernels that use the ACTUAL helper API from the `.hpp` — not raw LLK calls. This tests that the op structs, the main helper template function, and the CB/DEST management all compose correctly end-to-end.

  **Mandatory test coverage per op:**

  1. **Default path**: Use the helper with default template args and default dtype (Float16_b). Verify against the same golden references used in Phase B.

  2. **Dtype variation**: Run the helper with at least TWO different data formats (e.g., Float16_b and Float32). The helper's data format reconfiguration logic must work for non-default formats.

  3. **Template arg variation**: For ops with non-Dst template params, test the helper with at least ONE non-default value (e.g., `Exp<Approx::Fast>{}` in addition to `Exp<>{}`).

  4. **Runtime arg variation**: For parameterized ops, test with at least TWO different runtime argument values.

  5. **Policy variation**: Test at least TWO input policies (default `WaitAndPopPerTile` + one other, e.g., `NoWaitNoPop` or `WaitUpfrontNoPop`).

  6. **Chain composition**: Test at least ONE chain that combines the new op with another op (e.g., `sfpu_chain(Load, NewOp, Recip)`). This verifies that the init/exec sequence composes with other ops in the chain without interference.

  If the helper test fails but the raw LLK test passed, the bug is in the helper/struct integration — fix the `.hpp`/`.inl` and re-run.

  Phase D is NOT optional. A helper that was never tested through its own API with dtype, arg, and policy variation is not validated.

- **Phase E (Performance Testing)**: After Phase D passes, measure the performance overhead (if any) of the helper abstraction vs raw LLK calls. Phase B already produced raw LLK test kernels — reuse them as the baseline.

  **What to measure:**
  - Wall-clock time for processing N tiles (use a large enough N for stable measurement — at least 1024 tiles, ideally 4096+)
  - Measure BOTH the raw LLK kernel (from Phase B) and the helper kernel (from Phase D) using identical input data, CB configurations, and tile counts

  **How to measure:**
  - Use `ttnn.profiler` or device-side cycle counters (e.g., `RISCV_DEBUG_REG_READ_WALL_CLOCK`) to get per-kernel execution time
  - If profiler infrastructure is not available, use host-side timing around `ttnn.generic_op()` with enough iterations to amortize launch overhead (run the same program 10+ times, discard first 2 as warmup, average the rest)
  - Measure the **compute kernel only** — exclude reader/writer time by using identical reader/writer kernels for both raw and helper tests

  **How to report:**
  For each op tested, produce a row in the Performance Report:

  ```markdown
  ### Performance Report: {category}

  | Op | Dtype | N tiles | Raw LLK (µs) | Helper (µs) | Overhead | Status |
  |---|---|---|---|---|---|---|
  | Exp | Float16_b | 4096 | 123.4 | 124.1 | +0.6% | OK |
  | Exp | Float32 | 4096 | 198.2 | 199.0 | +0.4% | OK |
  | Sin | Float16_b | 4096 | 156.7 | 157.3 | +0.4% | OK |
  | FillTile | Float16_b | 4096 | 45.2 | 46.8 | +3.5% | REVIEW |
  ```

  **Overhead thresholds:**
  - **< 2%**: OK — expected from ALWI inlining, no action needed
  - **2-5%**: REVIEW — investigate whether the overhead is from the abstraction or measurement noise. Re-run with larger N. If persistent, document the cause.
  - **> 5%**: BLOCKER — the helper is adding unacceptable overhead. Investigate and fix before proceeding. Common causes: non-inlined functions, unnecessary re-initialization, extra CB operations.

  **Minimum coverage:**
  - Test at least 3 representative ops per category (e.g., one simple parameterless op, one parameterized op, one chain)
  - Test with at least 2 data formats per op
  - Report must include the raw numbers, not just pass/fail

  Phase E is NOT optional. A helper without performance characterization cannot be merged.

Loop exits only when 0 document blockers AND all LLK sequences pass on device AND parameter coverage is complete (dtypes + template args + runtime args) AND helper integration tests pass (with dtype/arg/policy variation) AND performance overhead is characterized AND all Open Questions from Stage 4 are resolved.

**Output**: Updated target document in-place + `{category}_device_validation.md` with per-op verdicts + `{category}_param_support.md` with per-LLK parameter support matrix + `{category}_perf_report.md` with performance comparison + helper integration test results + generated test files at `tests/tt_metal/tt_metal/test_kernels/compute/` and `tests/tt_metal/tt_metal/llk/`.

#### Stage 6: Implementation
**subagent_type**: `general-purpose`
**When**: After Stage 5 passes AND human validates the proposal at [H4].
**What it does**: Writes the actual `.hpp` and `.inl` helper files following the validated proposal. Migrates Tier 1 call sites.

Steps:
1. **Read the validated proposal** (`{category}_helper_proposal.md`) — signatures, op structs, dispatch logic, LLK sequences, parameter support matrix
2. **Read the Parameter Encoding Reference** from `{category}_investigation.md` — use directly for struct field design
3. **Read the Parameter Support Matrix** from `{category}_param_support.md` — add `static_assert` / runtime assert for UNSUPPORTED combos
4. **Create `{name}_helpers.hpp`** — declarations, enums, op structs, doc comments with usage examples, `#include "{name}_helpers.inl"` at bottom
5. **Create `{name}_helpers.inl`** — implementation following validated LLK sequences exactly as tested in Phase B
6. **Migrate Tier 1 call sites** — replace raw LLK code with helper calls at sites marked Tier 1 in the proposal
7. **Run existing tests** — `scripts/tt-test.sh` on affected test files to verify no regressions
8. **Run Stage 5 Phase D tests** — the helper integration tests generated during Stage 5 must still pass with the final implementation
9. **Commit** — one commit per logical unit (helper files, then migrations)

**Output**: `ttnn/cpp/ttnn/kernel_lib/{name}_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/{name}_helpers.inl`, migrated Tier 1 kernel files.

### Discovery Strategy: Bidirectional Search

Discovery MUST search from two directions:

1. **Bottom-up (LLK prefix)**: Search for functions matching `llk_math_eltwise_unary_sfpu_*`, etc. Catches ops with named LLK functions.
2. **Top-down (compute API directory)**: List ALL headers in the compute API dir. Catches ops that use SFPU macros directly.
3. **Cross-reference**: Any op found in only one direction is a gap that must be investigated.

### Logging and Commits

Uses the tt-agents logging infrastructure (`tt_metal/third_party/tt-agents/scripts/logging/`). Every agent MUST:

1. **Initialize breadcrumbs** at start:
   ```bash
   .claude/scripts/logging/init_breadcrumbs.sh \
     agent_logs/{category_slug} \
     {agent_name} \
     {category_name} \
     "{predecessor_agent}" \
     "{input_file_path}"
   ```

2. **Append breadcrumb events** during execution:
   ```bash
   .claude/scripts/logging/append_breadcrumb.sh \
     agent_logs/{category_slug} \
     {agent_name} \
     '{"event":"action","detail":"reading wrapper headers","status":"start"}'
   ```

3. **Write execution log** at completion following `tt_metal/third_party/tt-agents/references/agent-log-template.md`

4. **Git commit** at stage boundaries with structured messages:
   ```
   [llk-{agent_name}] stage {N}: {description}

   - {bullet list of key outputs}

   category: {category_name}
   stage: {N}
   status: {COMPLETE|PARTIAL|FAILED}
   ```

Standard breadcrumb event types: `start`, `input_parse`, `action`, `result`, `hypothesis`, `recovery`, `test`, `deviation`, `upstream_feedback`, `complete`.

**File layout**:
```
agent_logs/{category_slug}/
  catalog_breadcrumbs.jsonl
  catalog_execution_log.md
  {group_slug}_device_breadcrumbs.jsonl
  {group_slug}_device_execution_log.md
  {group_slug}_host_breadcrumbs.jsonl
  {group_slug}_host_execution_log.md
  {group_slug}_usage_breadcrumbs.jsonl
  {group_slug}_usage_execution_log.md
  verification_breadcrumbs.jsonl
  verification_execution_log.md
  proposal_breadcrumbs.jsonl
  proposal_execution_log.md
  review_fix_breadcrumbs.jsonl
  review_fix_execution_log.md
  device_validation_breadcrumbs.jsonl
  device_validation_execution_log.md
  param_coverage_breadcrumbs.jsonl
  param_coverage_execution_log.md
  helper_integration_breadcrumbs.jsonl
  helper_integration_execution_log.md
  perf_comparison_breadcrumbs.jsonl
  perf_comparison_execution_log.md
  implementation_breadcrumbs.jsonl
  implementation_execution_log.md
```

### Feedback Loop Protocol

The pipeline can loop back when the review-fix agent finds issues that require upstream changes.

```
Stage 4 (Proposal+Structs) ──▶ Phase A (Doc Review) ──▶ Phase B (Raw LLK) ──▶ Phase C (Param Coverage) ──▶ Phase D (Helper Integration) ──▶ Phase E (Perf) ──[ALL PASS]──▶ Stage 6
        ▲                            │                         │                         │                              │                          │
        │            [FAIL: proposal wrong]    [FAIL/HANG: LLK invalid]  [FAIL: observed combo]     [FAIL: helper bug]         [>5% overhead]
        └────────────────────────────┘                         │                         │                              │                          │
        │                                                      │                         │                              │                          │
        │    ┌─────────────────────────────────────────────────┘                         │                              │                          │
        │    │  Device failure → BLOCKER → re-enter Phase A                              │                              │                          │
        └────┘                                                                           │                              │                          │
        │                                                                                │                              │                          │
        │    Phase C: observed combo fails → BLOCKER → re-enter Phase A                  │                              │                          │
        │    Phase C: unobserved combo fails → record as UNSUPPORTED                     │                              │                          │
        └────────────────────────────────────────────────────────────────────────────────┘                              │                          │
        │                                                                                                               │                          │
        │    Phase D failure (raw LLK passed but helper failed) → fix .hpp/.inl → re-run Phase D only  ◀───────────────┘                          │
        │                                                                                                                                          │
        │    Phase E: >5% overhead → BLOCKER → investigate and fix .hpp/.inl → re-run Phase D + Phase E  ◀────────────────────────────────────────┘
        │
        │    [FAIL: investigation incomplete]
        └──▶ Stage 2 (Investigation) ──▶ Stage 3 ──▶ Stage 4 ──▶ Stage 5
        │
        │    [FAIL: impl .hpp/.inl wrong]
        └──▶ fix .hpp/.inl ──▶ Stage 5
```

Feedback targets emitted by the review agent:
```
FEEDBACK_TARGET: stage4_proposal
FEEDBACK_TARGET: stage3_verification
FEEDBACK_TARGET: stage2_investigation
FEEDBACK_TARGET: implementation_hpp
DEVICE_VALIDATION: ALL_PASS | N_FAILED | NOT_RUN
UNVALIDATED_SEQUENCES: <count>
PARAM_COVERAGE: COMPLETE | PARTIAL | NOT_RUN
PARAM_SUPPORTED: <count>
PARAM_UNSUPPORTED: <count>  (→ becomes assert in helper)
DATAFORMAT_COVERAGE: COMPLETE | PARTIAL | NOT_RUN
TEMPLATE_ARG_COVERAGE: COMPLETE | PARTIAL | NOT_RUN
RUNTIME_ARG_COVERAGE: COMPLETE | PARTIAL | NOT_RUN
HELPER_INTEGRATION: ALL_PASS | N_FAILED | NOT_RUN
HELPER_DTYPE_VARIATION: TESTED | NOT_TESTED
HELPER_ARG_VARIATION: TESTED | NOT_TESTED
HELPER_POLICY_VARIATION: TESTED | NOT_TESTED
HELPER_CHAIN_COMPOSITION: TESTED | NOT_TESTED
PERF_COMPARISON: ALL_OK | N_REVIEW | N_BLOCKER | NOT_RUN
PERF_MAX_OVERHEAD: <percentage>
```

The Phase A ↔ Phase B loop within Stage 5 is autonomous — the agent fixes proposal
sequences and re-runs device tests without human intervention. Up to 5 fix attempts
per op before marking as UNVALIDATED.

Phase C (Parameter Coverage) runs after Phase B passes. It tests dtypes, template args,
and runtime args — not just the default combo. If an **observed combo** (one used in the
codebase) fails, this is a BLOCKER — route back to Phase A with the failure diagnosis,
fix the proposal, and re-run from Phase B. Unobserved combos that fail are recorded as
UNSUPPORTED (→ assert in helper). Phase D (Helper Integration) runs after Phase C — it
tests the actual helper API end-to-end with dtype variation, arg variation, policy
variation, and chain composition. If Phase D fails, fix the .hpp/.inl and re-run Phase D
only (raw LLK already passed in Phase B, so the issue is in the helper composition).
Phase E (Performance) runs after Phase D — it compares helper vs raw LLK execution time
using the kernels from Phase B and Phase D. If overhead exceeds 5%, it's a BLOCKER —
investigate and fix the .hpp/.inl, then re-run Phase D + Phase E.

The output is `{category}_param_support.md`:

```markdown
## Parameter Support Matrix: {category}

### {llk_name}

#### Data Format Support

| Input dtype | Output dtype | Status | Evidence |
|---|---|---|---|
| BFloat8_b | BFloat8_b | SUPPORTED | Observed in 5 call sites, device test PASS |
| BFloat16 | BFloat16 | SUPPORTED | Observed in 12 call sites, device test PASS |
| Float32 | Float32 | SUPPORTED | Observed in 2 call sites, device test PASS |
| BFloat16 | Float32 | SUPPORTED | Probed, device test PASS |
| BFloat4_b | BFloat4_b | UNSUPPORTED | Never observed, device test FAIL |

#### Template Argument Support

| Parameter | Value | Status | Evidence |
|---|---|---|---|
| approx_mode | Exact (false) | SUPPORTED | Observed in 2 call sites, device test PASS |
| approx_mode | Fast (true) | SUPPORTED | Observed in 14 call sites, device test PASS |
| legacy | On (true) | SUPPORTED | Default for recip, 16 call sites |
| legacy | Off (false) | SUPPORTED | Probed, device test PASS |

#### Runtime Argument Support

| Parameter | Test Value | Status | Evidence |
|---|---|---|---|
| alpha | 1.0 (typical) | SUPPORTED | Device test PASS, golden match PCC>0.99 |
| alpha | 0.0 (edge) | SUPPORTED | Device test PASS, golden match PCC>0.99 |
| alpha | -1.0 (negative) | SUPPORTED | Device test PASS, golden match PCC>0.98 |
| threshold | +inf (edge) | UNSUPPORTED | Device test produces NaN output |

#### Cross-Dimension Combos (notable)

| dtype | template_args | runtime_args | Status | Evidence |
|---|---|---|---|---|
| Float16_b | approx=Fast | default | SUPPORTED | Most common combo (8 sites) |
| Float32 | approx=Exact | default | SUPPORTED | Observed in 2 sites |
| Float32 | approx=Fast | alpha=0.0 | SUPPORTED | Probed, device PASS |
| BFloat4_b | approx=Fast | default | UNSUPPORTED | Probed, device FAIL |

**Helper implementation guidance**:
- SUPPORTED params: no guard needed
- UNSUPPORTED params: add `static_assert` or runtime assert in helper with clear error message
- Cross-dimension failures: document which specific combos fail, not just individual values
- Runtime arg edge cases that produce NaN/Inf: document in helper doc comments as known limitations
```

This matrix directly informs which parameter combinations the helper should accept vs reject.

### Trust-the-Output Protocol

When an agent completes and returns structured output, the orchestrator MUST:
1. Trust the agent's data — use tables directly without re-reading source files
2. NOT duplicate research
3. Ask for clarification via SendMessage, not re-research
4. Make judgment calls — orchestrator's role is decisions, not research

Break this rule only when agent output is clearly wrong, self-contradictory, or obviously missing files.

### How to Invoke Each Stage

#### Stage 1 — Catalog

```python
catalog = Agent(
    subagent_type="Explore",
    prompt=catalog_template
        .replace("{{LLK_CATEGORY}}", category_name)
        .replace("{{LLK_PREFIX}}", primary_llk_prefix)
        .replace("{{ADDITIONAL_LLK_PREFIXES}}", extra_prefixes)
        .replace("{{COMPUTE_API_DIR}}", compute_api_dir)
        .replace("{{LLK_DIR}}", llk_dir)
        .replace("{{SECONDARY_SOURCE}}", codegen_or_enum_file)
        .replace("{{PROGRAM_FACTORY_DIR}}", program_factory_dir)
        .replace("{{KNOWN_GROUPS}}", known_groups_from_prior_run)
)
# Output: group_assignments + locator_results (op → file paths)
```

#### Stage 2 — Investigation (3 parallel agents per group)

```python
# For each group, launch 3 parallel investigation agents.
# Each agent receives ALL LLKs in its group + their locator results.
for group_name, group_ops in groups_from_catalog.items():
    group_locator = locator_results_for_group(group_name)
    group_ops_list = ", ".join(group_ops)

    Agent(subagent_type="Explore", run_in_background=True,
        prompt=device_template
            .replace("{{LOCATOR_RESULTS}}", group_locator)
            .replace("{{GROUP_NAME}}", group_name)
            .replace("{{LLK_CATEGORY}}", category_name)
            .replace("{{OPS_LIST}}", group_ops_list))

    Agent(subagent_type="Explore", run_in_background=True,
        prompt=host_template
            .replace("{{LOCATOR_RESULTS}}", group_locator)
            .replace("{{CODEGEN_FILE}}", codegen_file)
            .replace("{{GROUP_NAME}}", group_name)
            .replace("{{LLK_CATEGORY}}", category_name)
            .replace("{{OPS_LIST}}", group_ops_list))

    # Usage agent also collects all LLK parameter values from all call sites.
    Agent(subagent_type="Explore", run_in_background=True,
        prompt=usage_param_template
            .replace("{{GROUP_NAME}}", group_name)
            .replace("{{LLK_CATEGORY}}", category_name)
            .replace("{{OPS_LIST}}", group_ops_list))
```

#### Stage 3 — Verification

```python
Agent(
    subagent_type="Explore",
    prompt=verification_template
        .replace("{{CLAIMS}}", claims_from_investigation)
        .replace("{{COMPUTE_API_DIR}}", compute_api_dir)
)
```

#### Stage 4 — Proposal + Op Structs

```python
Agent(
    subagent_type="Explore",
    prompt=proposal_template
        .replace("{{LLK_CATEGORY}}", category_name)
        .replace("{{INVESTIGATION_FILE}}", f"{category_slug}_investigation.md")
        .replace("{{VERIFICATION_FILE}}", f"{category_slug}_verification.md")
        .replace("{{LOCATOR_RESULTS}}", locator_results)
        .replace("{{DEST_BATCHING_LIMITS}}", dest_limits_from_device)
        .replace("{{FP32_ACCUM_REQUIREMENTS}}", fp32_accum_from_device)
        .replace("{{DISRUPTIVE_INITS}}", disruptive_inits_from_device)
        .replace("{{PARAM_ENCODING_REFERENCE}}", param_encoding_from_host)
        .replace("{{INIT_MUTUAL_EXCLUSION}}", mutual_exclusion_from_usage)
        .replace("{{CHAINING_PATTERNS}}", chaining_from_usage)
        .replace("{{OUTLIERS}}", outliers_from_device)
        .replace("{{FACTORY_SHARING}}", factory_sharing_from_host)
)
```

#### Stage 5 — Review-Fix + Device Validation Loop

```python
Agent(
    subagent_type="general-purpose",
    prompt=review_fix_template
        .replace("{{TARGET_FILE}}", "{category}_helper_proposal.md")
        .replace("{{REFERENCE_FILES}}", """
- ttnn/cpp/ttnn/kernel_lib/{closest_existing}_helpers.hpp
- ttnn/cpp/ttnn/kernel_lib/{closest_existing}_helpers.inl
- {category}_investigation.md
- {category}_verification.md
""")
        .replace("{{REVIEW_CRITERIA}}", """
- Does every helper have a signature, Before/After, and what-it-absorbs?
- Does every op covered by the helpers have a corresponding op struct definition?
- Are struct fields named (no opaque uint32_t args), and are derived params absorbed internally?
- Are design decisions (e.g. per-tile init, cb_pop_front timing) explained?
- Do Before blocks show original kernel code, not already-migrated code?
- Does the proposal include an LLK Sequence Validation section?
- Is every helper's internal init→exec sequence validated against a codebase exemplar?
- Does each init immediately precede its own exec in the helper's internal flow?
- Are any init mutual exclusions violated?
- Are UNVALIDATED sequences flagged for human review in Open Questions?
- Are all Open Questions from Stage 4 resolved?
- Does the proposal reference DEST Batching Limits, FP32 Accum Requirements, and Disruptive Inits from investigation?
- Does the proposal use the Parameter Encoding Reference for struct field design (not re-derived)?
""")
        .replace("{{DONE_CONDITION}}", "0 blockers and 0 confusing issues, AND all LLK sequences pass on device (or marked UNVALIDATED), AND parameter coverage testing complete (dtypes: at minimum Float16_b + BFloat16 + Float32; template args: every non-Dst param value; runtime args: typical + edge values), AND helper integration tests pass (with dtype variation, template arg variation, runtime arg variation, policy variation, and chain composition), AND performance comparison complete (helper vs raw LLK, all ops < 5% overhead), AND all Open Questions resolved")
        .replace("{{LLK_CATEGORY}}", category_name)
        .replace("{{CATEGORY_SLUG}}", category_slug)
        .replace("{{EXISTING_TEST_REFERENCE}}", existing_test_path)
        .replace("{{EXISTING_KERNEL_REFERENCE}}", existing_kernel_path)
        .replace("{{PARAM_USAGE_MATRIX}}", param_usage_from_investigation)
        .replace("{{PARAM_PROBE_DIMENSIONS}}", """
dtypes: Float16_b,BFloat16,Float32,BFloat8_b,BFloat4_b,Int32
approx_mode: true,false
init_variants: *_init,*_init_short,*_init_short_with_dt
math_fidelity: HiFi4,HiFi3,HiFi2,LoFi
""")
)
```

#### Stage 6 — Implementation

```python
Agent(
    subagent_type="general-purpose",
    prompt=implementation_template
        .replace("{{LLK_CATEGORY}}", category_name)
        .replace("{{CATEGORY_SLUG}}", category_slug)
        .replace("{{PROPOSAL_FILE}}", f"{category_slug}_helper_proposal.md")
        .replace("{{INVESTIGATION_FILE}}", f"{category_slug}_investigation.md")
        .replace("{{PARAM_SUPPORT_FILE}}", f"{category_slug}_param_support.md")
        .replace("{{DEVICE_VALIDATION_FILE}}", f"{category_slug}_device_validation.md")
        .replace("{{CLOSEST_EXISTING_HELPER}}", closest_helper_path)
        .replace("{{HELPER_NAME}}", helper_name)
        .replace("{{PHASE_D_TEST_FILES}}", phase_d_test_paths)
)
```

### Output File Conventions

| Stage | Output File | Contents |
|---|---|---|
| 1: Catalog | `{category}_catalog.md` | Op list, gap analysis, group→ops assignments, Locator Results |
| 2: Investigation | `{category}_investigation.md` | Consolidated tables from 3 sub-agents per group: Device (signatures, DEST limits, FP32 accum, disruptive inits, outliers), Host (codegen, factory, param encoding ref, factory sharing), Usage (call sites, mutual exclusion, chaining, **Parameter Usage Matrix per LLK**) |
| 3: Verification | `{category}_verification.md` | Per-claim verdicts |
| 4: Proposal | `{category}_helper_proposal.md` | Design document + op struct definitions |
| 5: Review-Fix + Validation + Perf | (updates target in-place) + `{category}_device_validation.md` + `{category}_param_support.md` + `{category}_perf_report.md` | Logs in `agent_logs/`, test files in `tests/tt_metal/`, parameter support matrix per LLK (dtypes + template args + runtime args), helper integration test kernels (with variation), performance report (helper vs raw LLK) |
| 6: Implementation | `ttnn/cpp/ttnn/kernel_lib/{name}_helpers.hpp` + `.inl` | Helper files + migrated Tier 1 call sites |

All intermediate files live in the repo root. Final helper files live in `ttnn/cpp/ttnn/kernel_lib/`.
