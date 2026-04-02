---
name: LLK Investigation Agent
description: "Phase 1 agent. Analyzes a group of ops across device, host, and usage dimensions. One instance per group, all run in parallel. Replaces the previous 3-agent split (device/host/usage)."
type: reference
---

## Usage

Invoke with `subagent_type: Explore`. One instance per functional group, all in parallel.

Replace placeholders:
- `{{GROUP_NAME}}` — functional sub-group (e.g. Activations, Trigonometry)
- `{{LLK_CATEGORY}}` — operation category (e.g. elementwise unary)
- `{{OPS_LIST}}` — comma-separated operation names assigned to this group
- `{{LOCATOR_RESULTS}}` — locator table from Phase 0 (op -> file paths)
- `{{CODEGEN_FILE}}` — path to op_utils (e.g. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`)
- `{{FOCUS}}` — optional role-based focus directive to scope the analysis

## Prompt Template

```
Investigate the {{GROUP_NAME}} group of {{LLK_CATEGORY}} operations: {{OPS_LIST}}.

{{FOCUS}}

Use the locator results below to find files without searching:
{{LOCATOR_RESULTS}}

Log breadcrumbs to agent_logs/. See tt_metal/third_party/tt-agents/scripts/logging/ for format.

For EACH operation in the group, analyze three dimensions:

═══ DIMENSION 1: DEVICE-SIDE ═══

Read the compute API wrapper header and LLK/ckernel implementation.

Produce these tables:

### Wrapper Signatures
| Op | Init Signature | Exec Signature | Template Params | Runtime Params |
|---|---|---|---|---|

### Init State Compatibility
| Op | Configures HW Resource | Disruptive? | Can Coexist With |
|---|---|---|---|

### DEST Batching Limits
| Op | Max Tiles Per DEST Batch | FP32 Accumulation Required? |
|---|---|---|

═══ DIMENSION 2: HOST-SIDE ═══

Read the codegen/op_utils file ({{CODEGEN_FILE}}) and program factory.

Produce these tables:

### Code Generation
| Op | Generated Init Call | Generated Exec Call | In Section |
|---|---|---|---|

### Parameter Encoding Reference
| Op | User API Param | Host Transform | Kernel Receives | Could Kernel Compute? |
|---|---|---|---|---|

### Program Factory Layout
| Op | CB Layout | Runtime Args Order | Factory Sharing |
|---|---|---|---|

═══ DIMENSION 3: USAGE PATTERNS ═══

Search ALL kernel call sites across the codebase.

Search directories:
- ttnn/cpp/ttnn/operations/**/kernels/compute/*.cpp
- tt_metal/kernels/compute/*.cpp
- tests/**/test_kernels/compute/*.cpp

Produce these tables:

### Call Sites
| Op | File:Line | Pattern | Init Placement | Batching |
|---|---|---|---|---|

### Init/Exec Pairing Rules
| Op | Rule | Evidence |
|---|---|---|

### Init Mutual Exclusion
| Op A Init | Op B Init | Compatible? | Evidence |
|---|---|---|---|

### Chaining Patterns
| Pattern | Ops Involved | File:Line | Description |
|---|---|---|---|

### Parameter Usage Matrix
For each op with non-trivial params, record observed parameter values:

| Param | Type | Observed Values | Call Sites |
|---|---|---|---|

Include: template args, runtime args, input/output dtypes, math_fidelity, DEST mode.

═══ DIMENSION 4: ENCAPSULATION ANALYSIS ═══

This dimension determines what the helper MUST encapsulate. Complexity in target
kernels is not a reason to exclude — it is the specification of what the helper hides.

### Compile-Time Feature Matrix

Examine ALL `#ifdef` / `#if defined` blocks in the target kernels. For each:

| Feature Flag | Affects Core Loop? | Classification | Should Become |
|---|---|---|---|
(example: PACKER_L1_ACC | Yes — changes pack path and reload logic | Loop-internal | Template bool param)
(example: FUSE_BIAS | No — post-loop phase | Adjacent operation | Separate helper or callback)

Classifications:
- **Loop-internal**: Flag changes behavior INSIDE the main compute/pack loop → must be a template param on the helper
- **Adjacent operation**: Flag enables a separate phase before/after the loop → callback hook or separate helper
- **Orthogonal**: Flag controls unrelated behavior (e.g., early return, reader mode) → out of scope

### Cross-Iteration State Analysis

Examine all variables in the target loop. For each that carries state across iterations:

| Variable | Loop Level | Mutated When | Read When | Implication |
|---|---|---|---|---|
(example: enable_reload | K-block loop | Set true after block 0 | Checked at subblock start | Helper must own K-loop)

**Rule**: If ANY cross-iteration state exists in a loop, the helper must own that loop.
Splitting it into caller-managed pieces leaks the state management to the caller.

### Side-Effect Operations

Any CB operation targeting a buffer OTHER than the one being packed to, or any
hardware configuration call that serves a non-obvious purpose:

| Operation | Target | Purpose | Correctness Requirement? |
|---|---|---|---|
(example: cb_reserve_back(out_cb, ...) on block 0 when packing to interm_cb | out_cb | Shared memory guard — prevents interm from overwriting output space | Yes — data corruption without it)

These are correctness requirements the helper must preserve, not optional complexity.

### Parameter Independence Analysis

For each parameter used in the target kernel body:

| Param | Used In | Derivable From | Independent? |
|---|---|---|---|
(example: in0_block_num_tiles | cb_wait_front, cb_pop_front | out_subblock_h × block_w × in0_num_subblocks | No — derive internally)

**Rule**: The helper API should expose ONLY independent parameters.
Everything derivable must be computed internally.

### CB Compile-Time Analysis

For each CB index used in target kernels:

| CB | Declared As | Varies at Runtime? | Recommendation |
|---|---|---|---|
(example: in0_cb | constexpr uint32_t | No — always compile-time | Template param — enables static_assert)

**Rule**: If a CB is constexpr in ALL call sites → template param.
If it varies at runtime → runtime param.

═══ OUTPUT ═══

Save to: agent_logs/{{CATEGORY_SLUG}}_{{GROUP_SLUG}}_investigation.md

The orchestrator will consolidate per-group outputs into {category}_investigation.md.
```

## Focus Directives

When the category or situation calls for emphasizing one dimension over others, include a focus directive. The Encapsulation dimension is ALWAYS required — never skip it.

**Compute-heavy category** (SFPU, matmul):
```
Focus on: Device dimension (wrapper signatures, init state, batching limits),
Usage dimension (chaining patterns, init mutual exclusion), and
Encapsulation dimension (feature flags, cross-iteration state, parameter independence).
De-emphasize Host dimension — program factory details are less critical when the
helper wraps only the compute kernel.
```

**Data-movement category** (tilize, untilize):
```
Focus on: Host dimension (CB layout, parameter encoding, factory sharing),
Usage dimension (call sites, boilerplate patterns), and
Encapsulation dimension (feature flags, cross-iteration state, parameter independence).
De-emphasize Device dimension — the LLK implementations are simpler for data movement ops.
```

**New/unknown category**:
```
Equal emphasis on all four dimensions. Flag any surprising findings.
```
