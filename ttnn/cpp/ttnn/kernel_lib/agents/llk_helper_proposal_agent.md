---
name: LLK Helper Proposal + Op Struct Agent Prompt
description: Stage 4 combined agent. Takes verified investigation results, proposes a compute_kernel_lib helper API design (Part 1), then designs the op-type-trait structs that implement it (Part 2). Produces both a design document and compilable C++ struct definitions.
type: reference
---

## Usage

Invoke with `subagent_type: Explore`. Replace placeholders:
- `{{LLK_CATEGORY}}` — the operation category (e.g. elementwise unary, binary eltwise)
- `{{INVESTIGATION_FILE}}` — path to the investigation report from Stage 2
- `{{VERIFICATION_FILE}}` — path to the verified claims from Stage 3
- `{{LOCATOR_RESULTS}}` — the Locator Results table from Stage 1 (Phase 4 output)

## Prompt Template

```
Propose a compute_kernel_lib helper API for {{LLK_CATEGORY}} operations.

BREADCRUMB LOGGING — do this first:
Derive CATEGORY_SLUG from {{LLK_CATEGORY}} (lowercase, spaces → underscores).
BCRUMB="agent_logs/${CATEGORY_SLUG}_proposal_breadcrumbs.jsonl"
Run at start:
  mkdir -p agent_logs
  echo '{"ts":"'"$(date -Iseconds)"'","event":"start","agent":"proposal","category":"{{LLK_CATEGORY}}"}' >> $BCRUMB
After prerequisite check:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","check":"prerequisites","investigation_exists":true/false,"verification_exists":true/false}' >> $BCRUMB
After reading investigation file (log what pain points you extracted):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"read","file":"{{INVESTIGATION_FILE}}","boilerplate_pattern":"cb_wait/copy/compute/pack","hw_constraints_found":N,"derived_param_opportunities":M}' >> $BCRUMB
After reading verification file (log what changed your understanding):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"read","file":"{{VERIFICATION_FILE}}","corrections_found":N,"confirmed_claims":M}' >> $BCRUMB
For each grouping decision (WHY these ops belong together):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"decision","what":"group","ops":["OP1","OP2"],"helper":"eltwise_unary","shared_pattern":"same CB/DEST batching/copy pattern","key_differentiator":"only tile() call differs"}' >> $BCRUMB
For each op excluded from a helper:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"decision","what":"exclude","op":"OP","reason":"unique CB layout","alternative":"separate helper or no helper"}' >> $BCRUMB
For each before/after design iteration (when redesigning because AFTER was not simpler):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"decision","what":"redesign","helper":"NAME","issue":"AFTER was not simpler than BEFORE","change":"what was adjusted"}' >> $BCRUMB
For each anti-pattern deliberately avoided:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"decision","what":"anti_pattern_avoided","pattern":"mega_enum/caller_init/opaque_params","alternative":"op_type_trait_structs"}' >> $BCRUMB
At completion:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"complete","helpers_proposed":N,"ops_covered":M,"ops_excluded":K,"open_questions":Q}' >> $BCRUMB
Write agent_logs/${CATEGORY_SLUG}_proposal_execution_log.md: helpers proposed with op counts, grouping rationale, exclusions with reasons, design iterations that needed rework, open questions.

PREREQUISITE CHECK — verify these files exist before proceeding:
- Investigation results: {{INVESTIGATION_FILE}}
- Verification results: {{VERIFICATION_FILE}}

If EITHER file does not exist or is empty:
  OUTPUT EXACTLY: "STAGE_INCOMPLETE: Required input file(s) missing. {{INVESTIGATION_FILE}} exists: {yes/no}. {{VERIFICATION_FILE}} exists: {yes/no}. Cannot produce proposal without completed investigation and verification stages."
  Then STOP. Do NOT attempt to investigate or propose without the prerequisite data.

If both files exist, read them:
- Read the investigation results: {{INVESTIGATION_FILE}}
- Read the verification results: {{VERIFICATION_FILE}}

Read these existing helpers to understand what GOOD helpers look like:
- `/localdev/astancov/tt-metal/ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`
- `/localdev/astancov/tt-metal/ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`
- `/localdev/astancov/tt-metal/ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
- `/localdev/astancov/tt-metal/ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`

CRITICAL CONSTRAINT: The helper must be SIMPLER to use than the raw API it replaces. If your proposed signature is not obviously easier to read, understand, and use than calling the raw *_tile_init() / *_tile() functions directly, your design has FAILED. Every parameter the caller passes must have clear, domain-meaningful semantics — no "param0", no "ParamMode", no opaque uint32_t values.

CRITICAL CONSTRAINT: Helpers MUST be FREE FUNCTIONS. The main entry point is always a free function like `eltwise_unary<in_cb, out_cb>(num_tiles, OpStruct{...})`. This is bare-metal RISC-V firmware — no RAII, no virtual dispatch, no dynamic allocation.

For multi-op helpers, see the op-type-trait struct pattern in the Reference section below — it's the preferred approach when many ops share the same boilerplate but differ only in which *_tile() function they call. Use it when it fits; if the operation set has a genuinely different structure, propose what makes sense.

## What a Helper Does

A helper exists to:
1. ABSORB BOILERPLATE — the CB management (wait_front, pop_front, reserve_back, push_back), tile register lifecycle (acquire, copy, compute, commit, wait, pack, release), and init/uninit sequencing that every kernel repeats identically.
2. DETECT AND WORK AROUND HARDWARE LIMITATIONS — DEST register capacity, block sizes that exceed hardware limits, FP32 accumulation mode, tile count batching. The caller should NEVER need to know about these constraints. Example: untilize_helpers auto-splits blocks when width > DEST capacity. Your helper must do the same kind of thing for its domain.
3. COMPUTE DERIVED PARAMETERS INTERNALLY — if the investigation shows that callers always pass (alpha, 1/alpha) or (beta, 1/beta, threshold), the helper should accept just (alpha) or (beta, threshold) and compute the reciprocal internally. The investigation's "parameter semantics" section tells you exactly which derived values to absorb.
4. GROUP GENUINELY SIMILAR OPS — ops that share the SAME calling pattern, SAME CB management, SAME hardware constraints, and differ ONLY in which *_tile() function they call, belong in one helper. Ops that differ in any of these dimensions are SEPARATE helpers.

## What a Helper Does NOT Do

- Does NOT use an enum to dispatch 80+ ops through a giant switch statement — each op is its own type
- Does NOT use raw callbacks/lambdas where the caller writes init+compute calls — the op struct encapsulates those
- Does NOT add parameters the caller must set but has no reason to care about (ParamMode, InitMode, DataType variants)
- Does NOT make the caller specify things the helper can figure out itself
- Does NOT expose internal implementation details (SfpuType, APPROX mode, callback functions)
- Does NOT require the caller to bit-cast floats to uint32_t or compute reciprocals — the op struct handles encoding
- Does NOT invent LLK call sequences that have no precedent in the codebase — every init→exec sequence the helper generates internally must match a pattern observed in real kernels (see LLK Sequence Validation below)

## LLK Sequence Validation

**This section is mandatory. Skip it and the proposal is rejected.**

Before proposing any helper, validate that the LLK call sequence it will generate internally is proven to work by existing codebase usage. Helpers are convenience wrappers — they must not compose LLK primitives in novel ways that no kernel has ever tested.

### Rule 1: Every init must immediately precede its own exec

Each LLK function has a paired init (e.g. `copy_tile_to_dst_init_short` → `copy_tile`, `exp_tile_init` → `exp_tile`). The helper's internal sequence must respect this pairing: the init for function F must be the last init called before F is executed. If the helper calls init_A then init_B then exec_A, that is invalid — init_B may have overwritten hardware state that exec_A depends on.

Check the investigation's "Init/Exec Pairing Rules" table. For every LLK function the helper will call internally, verify the init→exec adjacency is preserved.

### Rule 2: Respect init mutual exclusion

The investigation's "Init Mutual Exclusion" table lists init functions that never coexist in the same kernel. If two inits are never observed together across the entire codebase, the helper MUST NOT combine them. This is a hard constraint — the hardware paths may be fundamentally incompatible.

If the helper needs functionality from two mutually-exclusive init categories, it must be split into separate helpers or the caller must invoke them in separate phases.

### Rule 3: Match observed init ordering from real kernels

The investigation's "Init Ordering Sequences" table shows exact sequences from kernels that chain multiple LLK operations. The helper's internal call order must match one of these observed sequences. If no existing kernel chains the proposed LLK functions in the proposed order, the sequence is unproven and must not be used.

### Rule 4: Validate against the codebase, not against documentation

LLK documentation may be incomplete or misleading. The source of truth is what compiles and runs correctly in existing kernels. The investigation and verification stages exist to extract these ground-truth patterns. If the investigation data does not cover a proposed sequence, flag it as an open question — do not assume it works.

### How to apply

For each proposed helper, write out the EXACT sequence of LLK init and exec calls it will generate internally (accounting for all template parameter combinations). Then cross-reference each sequence against the investigation tables. If any sequence has no codebase precedent, either:
1. Find a codebase example that proves the sequence works (cite file:line)
2. Restructure the helper to use a proven sequence
3. Flag the sequence as UNVALIDATED in the Open Questions section

## Design Process

### Step 1: Identify Caller Pain Points

From the investigation's "boilerplate pattern" and "hardware constraints" sections, list:
- What repeated code does every kernel author write around these ops?
- What hardware limitations do callers currently work around manually?
- What derived parameters do callers compute that could be absorbed?

### Step 2: Group by Caller-Visible Similarity

Two ops belong in the SAME helper function if and only if:
- Same CB management pattern (same wait/pop/reserve/push sequence)
- Same DEST constraints and batching behavior
- Same boilerplate structure (copy → compute → pack)
- Parameters with compatible semantics (or no params)

They differ ONLY in which internal *_tile() is called → each op is a separate struct type, but they all flow through the SAME helper function via `typename Op` template parameter.

Two ops belong in DIFFERENT helper functions if:
- Different CB patterns (e.g., one reads from 2 CBs, another from 1)
- Different hardware constraint handling
- Fundamentally different compute patterns (e.g., multi-tile vs single-tile, multi-DST-slot vs single)
- Different boilerplate structure (e.g., one needs intermediate CB round-trips)

**Auto-exclude signals** — if any of these are present, the op does NOT belong in the standard helper:
- Requires an intermediate CB round-trip between phases (e.g., logit: clamp → pack → reload → div → log)
- Mixes SFPU and FPU ops on the same DST in the same acquire block (e.g., hardswish: SFPU hardsigmoid + FPU mul_tiles)
- Has conditional operation dispatch at runtime (e.g., groupnorm: copy_tile vs add_tiles based on flag)
- Requires multi-DST-slot allocation for intermediate results (e.g., lgamma: log + sin + floor + where simultaneously)
- Has a multi-phase init/uninit cycle within a single kernel invocation

Note: Different param counts do NOT require different helper functions. The op struct carries its own params and the helper just calls `op.compute(dst)` regardless of how many fields the struct has.

### Step 3: Design Each Helper

For each helper function:
a. Write the CALLER'S code FIRST (how the kernel author uses it)
b. Compare against the raw API it replaces
c. Verify the helper is simpler
d. Then design the internals

### Step 4: Name everything deliberately

Names communicate contracts. Test each name: would a new contributor understand it without reading the implementation?

- Prefer names that describe the *grid of tiles* (`TileGrid`) not the *shape of one tile* (`TileShape`)
- Prefer factory methods over positional constructors: `TileGrid::of(Ht, Wt, NC)` not `TileGrid{Ht, Wt, NC}`
- Group related CBs into a named struct rather than three positional args: `ReduceCBs::of(in, scaler, out)` prevents argument-swap bugs
- For optional sentinel types (`NoOp`, `NoAccumulation`), use `explicit` default constructors — this prevents silent `{}` brace-initialization at call sites and forces the caller to write `NoAccumulation{}` which is self-documenting
- Rename when the current name misleads; add a backward-compat alias to avoid breaking existing call sites

### Step 5: Document state contracts

For every helper that interacts with persistent state (CB contents, SFPU register state, reduce init), document the contract explicitly in the function comment:
- What state must be true at entry (preconditions the caller must guarantee)
- What state is true at exit (postconditions the caller can rely on)

Example: reduce scaler tile stays in its CB after `reduce()` completes — the helper never pops it. This is a postcondition the caller relies on for multi-phase kernels. If it's not documented, it becomes a silent landmine.

### Step 6: Validate LLK Call Sequences

For each proposed helper, write out the internal LLK call sequence it will emit (init and exec calls in order). Cross-reference against the investigation's Init/Exec Pairing Rules, Init Mutual Exclusion, and Init Ordering Sequences tables. Every sequence must have a codebase precedent. See the "LLK Sequence Validation" section above.

### Step 7: Validate with Before/After Examples

For EVERY proposed helper, show:

BEFORE (current raw API):
```cpp
// Show the actual kernel code a developer writes today
// Include ALL boilerplate: CB management, tile regs, init, loop, pack
```

AFTER (with helper):
```cpp
// Show what the same kernel looks like using the helper
// Must be obviously simpler
```

If the AFTER is not clearly better, the helper is wrong. Redesign it.

## Output Format

For EACH proposed helper function:

### Helper N: {descriptive_name}

#### Covers
- [list of ops this helper wraps]
- [why these ops belong together — shared pattern]

#### Before / After

BEFORE (raw API, typical kernel):
```cpp
// actual code
```

AFTER (with helper):
```cpp
// simplified code
```

#### Proposed Signature
```cpp
// Show the actual proposed function signature(s).
// Must be a free function. Named parameters only — no opaque uint32_t args.
// Include compile-time template params (CB indices, modes) and runtime params.
```

#### What the Helper Absorbs
- [boilerplate it handles internally]
- [hardware limitations it detects and works around]
- [derived parameters it computes from caller-provided values]

#### State Contracts
- **Entry preconditions**: [what the caller must guarantee before calling — e.g., "scaler CB must already contain one tile"]
- **Exit postconditions**: [what state the helper leaves — e.g., "scaler tile remains in CB (not popped)", "reduce_uninit called"]

#### Internal Dispatch
- [how it routes to different *_tile() calls — brief, this is implementation detail]

#### Excluded from this Helper
- [ops that look similar but don't fit, with reason — reference auto-exclude signals from Step 2]

---

### LLK Sequence Validation

For EACH helper, list the internal LLK call sequence and its codebase evidence:

| Helper | Internal LLK Sequence | Codebase Precedent (file:line) | Status |
|--------|-----------------------|-------------------------------|--------|
| (name) | init_A → exec_A → init_B → exec_B | path/to/kernel.cpp:42 | VALIDATED |
| (name) | init_A → exec_A → exec_B (no init_B) | (none found) | UNVALIDATED — see Open Questions |

Init mutual exclusion check:

| Init A | Init B | Helper Uses Both? | Codebase Co-occurrence | Status |
|--------|--------|-------------------|------------------------|--------|
| (init_fn) | (init_fn) | yes/no | "0 of N kernels" / "file:line" | OK / VIOLATION |

If ANY sequence is UNVALIDATED or any mutual exclusion is VIOLATED:
1. The helper MUST NOT be implemented as-is.
2. Move the unvalidated sequence to Open Questions.
3. Flag it for human review — the orchestrator must ask the user to confirm or reject the proposed sequence before proceeding. The user may provide a codebase example the agent missed, confirm the sequence is safe based on hardware knowledge, or reject it outright.
4. Do not attempt to "prove" an unvalidated sequence by writing a test kernel autonomously — the risk of silent data corruption means a human must sign off.

---

### Excluded Operations (No Helper)

For each excluded op:
- [op name]: [why no helper helps — it's too unique, already has a helper, dead code, etc.]

### Design Decisions
- [decision]: [rationale grounded in investigation data]

### Migration Tier Table

Categorize every identified call site. This is mandatory — it tells the implementer where to start.

| File | Pattern | Tier | Notes |
|------|---------|------|-------|
| `example_kernel.cpp` | Standard SFPU_OP_CHAIN | 1 | Direct swap |
| `complex_kernel.cpp` | FPU+SFPU mixed | 3 | Skip — manual |

- **Tier 1**: Direct swap, no surrounding logic changes
- **Tier 2**: Minor restructuring needed (PostOp callback, grid() wrapper)
- **Tier 3**: Complex — conditional dispatch, multi-CB reread, FPU+SFPU mixed. Leave manual.

### Test Plan

Specific pytest commands to run after migration. Not optional.

```bash
# [describe what each command tests and which tier/pattern it covers]
pytest path/to/tests/ -v -k "relevant_filter"
```

Include: device reset command (`tt-smi -r`) for use if tests hang.

### Open Questions
- [things that need resolution before writing code]

---

## PART 2: Op Struct Design

After the proposal above is complete, design the op-type-trait structs for every operation that will be covered by the proposed helpers. Use the locator results to find each op's source files:

{{LOCATOR_RESULTS}}

For EACH operation, read ONLY these two things:

1. COMPUTE API WRAPPER (from locator): Read the `*_tile_init()` and `*_tile()` signatures. Extract:
   - Template parameters (name, type, default)
   - Runtime parameters (name, type — typically uint32_t for bit-cast floats)
   - What the wrapper calls internally (for documentation only)

2. HOST CODEGEN (from locator): Find the operation in the codegen file. Extract:
   - The generated init string (e.g., `"exp_tile_init<{}u>();"`)
   - The generated compute string (e.g., `"exp_tile<{1}u>({0});"`)
   - Any host-side parameter transforms (e.g., `std::bit_cast<uint32_t>(param0)`, `1.0f / param0`)
   - Which parameters are caller-provided vs host-derived

Then design a struct for each op following these rules:

STRUCT DESIGN RULES:
- Template params on the struct for compile-time config (e.g., approx mode): `template <bool approx = false> struct Exp`
- Named float fields for runtime params the CALLER provides: `float alpha;`
- `std::bit_cast<uint32_t>(...)` happens INSIDE `compute()`, never exposed to caller
- Derived params (reciprocals, etc.) computed INSIDE `compute()` or a constexpr constructor
- `init()` is `static` if it takes no runtime params (most ops)
- `compute(uint32_t dst)` is `static` for zero-param ops, `const` member for parameterized ops
- All methods are `ALWI` (always inline)
- Zero-param structs must be empty types (0 bytes)
- For sentinel/optional types, use `explicit` default constructors to prevent silent `{}` brace-initialization

CLASSIFY each op into one of these patterns:
- **ZERO_PARAM**: No runtime params. Empty struct with optional template params. E.g., `Exp<true>{}`
- **NAMED_FIELDS**: 1+ runtime params as named float fields. E.g., `Elu{.alpha = 1.0f}`
- **DERIVED_PARAM**: Has params the host currently pre-computes (reciprocals, etc.) that the struct should compute internally.

BREADCRUMB EVENTS for Part 2:
After reading wrapper signature for each op:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"read","phase":"p2_op_struct","op":"OP","wrapper_file":"path","init_sig":"...","compute_sig":"..."}' >> $BCRUMB
After reading codegen entry for each op:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"read","phase":"p2_op_struct","op":"OP","codegen_file":"path:line","host_transforms":["..."],"caller_provided_params":["..."]}' >> $BCRUMB
After classifying each op:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","phase":"p2_op_struct","op":"OP","pattern":"ZERO_PARAM/NAMED_FIELDS/DERIVED_PARAM","reason":"WHY"}' >> $BCRUMB

OUTPUT for Part 2 — append to the same output document:

### Op Struct Designs

For each operation:

#### {OpName}

**Pattern**: ZERO_PARAM | NAMED_FIELDS | DERIVED_PARAM
**Wrapper**: `{init_signature}` / `{compute_signature}`
**Host transforms**: {what the host currently computes, or "none"}

```cpp
// {one-line description}
{struct definition — complete, compilable}
```

**Caller usage**: `eltwise_unary<cb_in, cb_out>(n, {example instantiation});`

---

Then output a summary table:

| Op | Pattern | Template Params | Runtime Fields | Derived Params | Struct Size |
|----|---------|-----------------|----------------|----------------|-------------|
```

## Anti-Patterns to Avoid

These are FAILURES. If your proposal contains any of these, start over:

1. **Caller-written init/compute** — do NOT make the caller write `[](uint32_t dst) { exp_tile(dst); }` or call `init()` / `compute()` themselves. The helper must encapsulate that — the caller specifies WHAT to run, not HOW.
2. **Mega-enum dispatch** — do NOT create `enum UnaryOp { Exp, Relu, ... }` with 80+ values and a switch statement. Each op is its own struct type; the helper dispatches via `typename Op` template parameter.
3. **RAII / virtual dispatch / dynamic allocation** — no `class UnaryOp` with virtual methods. Op structs are trivial value types with static or inline methods. This is bare-metal RISC-V firmware.
4. **Opaque parameters** — no `param0`, `param1`, `param2`. Every parameter is a named field on the op struct (`float alpha`, `float min_val`).
5. **Caller-visible encoding** — the caller must NEVER bit-cast floats to uint32_t or pre-compute reciprocals. The op struct handles all encoding in its `compute()` method or constructor.
6. **Requiring the caller to know hardware details** — no SfpuType, APPROX booleans exposed as raw template params. Approx mode should be a template param on the op struct (e.g., `Exp<true>`) or a separate struct name (e.g., `ExpApprox`).
7. **Grouping ops by internal init patterns** instead of caller-visible similarity
8. **Not showing before/after code comparisons** for every helper
9. **Hidden mutable state** — if a function needs state across calls (like cumsum's "first tile" flag), make it a caller-visible parameter, not hidden object state
10. **Bool template params without demonstrated use cases** — `bool init = true, bool uninit = true` sounds flexible, but if every call site uses the defaults, the configurability is pure complexity. Remove it. If a use case appears later, add it then. The reduce helper removed `init`/`uninit` template params after finding zero call sites needed them.
11. **One enum encoding multiple orthogonal behaviors** — an enum like `ReduceInputMode { STREAMING, PRELOADED, PERSISTENT }` hides that "when to wait" and "whether to pop" are independent choices. Prefer policy structs with named members: `struct StreamingPolicy { static constexpr WaitMode wait = PER_TILE; static constexpr PopMode pop = POP; }`. These are self-documenting, extensible, and still resolve via `if constexpr` with zero overhead.
12. **Runtime dispatch where `if constexpr` works** — for pattern selection based on template parameters (REDUCE_SCALAR vs REDUCE_ROW loop structure, op with/without approx mode), always use `if constexpr`. Never select via runtime branching on a template parameter value. The generated code for the unused branch must be zero bytes.
13. **Novel LLK call sequences with no codebase precedent** — if the helper's internal init→exec ordering does not match any existing kernel in the codebase, the sequence is unproven. LLK init functions configure shared hardware state (unpack pipelines, SFPU registers, ADDR_MOD). Calling them in an untested order can silently produce wrong results, hangs, or data corruption. Every sequence the helper emits must have a codebase exemplar. See LLK Sequence Validation.

## Reference: How Existing Helpers Work

ALL existing helpers are **free functions** with template parameters for compile-time configuration. Study these patterns:

### tilize_helpers.hpp — `tilize<block_width, in_cb, out_cb>(...)`
- **Free function**, NOT a class
- Template params: block_width_tiles, input_cb, output_cb, InitUninitMode, WaitMode, ReconfigMode
- Runtime params: num_blocks, total_input_pages
- Auto-detects fast_tilize availability
- Config enums (InitUninitMode, WaitMode, ReconfigMode) control lifecycle without hiding it

### untilize_helpers.hpp — `untilize<out_cb, in_cb>(...)`
- **Free function**, NOT a class
- Detects DEST capacity and auto-splits blocks when width exceeds it
- Handles pack_untilize vs standard untilize dispatch based on datatype

### reduce_helpers_compute.hpp — `reduce<PoolType, ReduceDim>(...)`
- **Free function**, NOT a class
- Template params: pool_type, reduce_dim
- Runtime params: cb_in, cb_scaler, cb_out, shape, input_policy, post_reduce_op
- InputPolicy enum controls how tiles are fed (meaningful to caller: do I wait or not?)
- PostReduceOp callback for fused operations (e.g., recip_tile after sum for softmax)
- Auto-detects DEST limits, handles accumulation

### Common patterns across ALL existing helpers:
- Free functions in `compute_kernel_lib` namespace
- Template parameters for compile-time config (CB indices, modes, enums)
- Runtime parameters for data-dependent values (block counts, shapes)
- No mutable state — every call is self-contained
- Lightweight value types for configuration (e.g., `ReduceInputBlockShape`, `Accumulate`, `AccumulationConfig`)
- `if constexpr` for all pattern selection — no runtime branching on template parameter values

### Named Aggregate Structs for Configuration (PREFERRED over positional args)

When a helper takes 3+ related values (CBs, dimensions), group them into a named struct with static factory methods. This prevents argument-swap bugs and makes call sites self-documenting.

```cpp
// BAD: positional args — easy to swap Ht and Wt, easy to forget NC
reduce(cb_in, cb_scaler, cb_out, Ht, Wt, NC);

// GOOD: named factory method — intent is clear, order doesn't matter semantically
reduce(ReduceCBs::of(cb_in, cb_scaler, cb_out), TileGrid::of(Ht, Wt, NC));
```

Factory method pattern:
```cpp
struct TileGrid {
    uint32_t rows, cols, batches;
    static constexpr TileGrid of(uint32_t r, uint32_t c, uint32_t b = 1) { return {r, c, b}; }
    static constexpr TileGrid single() { return {1, 1, 1}; }
    static constexpr TileGrid row(uint32_t c, uint32_t b = 1) { return {1, c, b}; }
    static constexpr TileGrid col(uint32_t r, uint32_t b = 1) { return {r, 1, b}; }
};
```

For sentinel/optional types, use `explicit` default constructors:
```cpp
struct NoOp {
    explicit NoOp() = default;
    ALWI void operator()(uint32_t = 0) const {}
};
// Now: reduce(..., NoOp{}) compiles; reduce(..., {}) does NOT — prevents silent defaults
```

### The Op-Type-Trait Struct Pattern (PREFERRED for multi-op helpers)

When a helper needs to support MANY operations that share the same boilerplate but differ in init/compute calls, use the **op-type-trait struct** pattern:

1. Each operation is a **struct type** that defines `init()` and `compute(uint32_t dst)` methods
2. Compile-time config (like approx mode) is a **template parameter on the struct**
3. Runtime parameters are **named fields** on the struct (e.g., `float alpha`, not `uint32_t param0`)
4. Derived parameters (reciprocals, etc.) are computed by the struct, not the caller
5. The helper is a **single free function** templated on `typename Op` — it calls `op.init()` + `op.compute(dst)` and handles all boilerplate

This pattern gives:
- **Caller simplicity**: `eltwise_unary<cb_in, cb_out>(n, Exp<true>{})` — one line, says WHAT not HOW
- **Zero dispatch overhead**: the compiler resolves `Op` at compile time and inlines everything
- **Extensibility**: adding a new op = one new struct, helper is untouched
- **Named params**: `Elu{.alpha = 1.0f}` not `helper(n, 1.0f)` with opaque positional args
- **Type safety**: `Elu{.alpha = 1.0f}` won't accidentally compile where `Hardtanh{.min_val, .max_val}` is expected

Note: This pattern is for the op-specific dispatch only. The helper function itself follows the same free-function convention as tilize/reduce. The op struct is a lightweight value type passed as a function argument, not an OOP class.

Example of how the helper function and op structs relate:
```cpp
// Op struct — one per operation
template <bool approx = false>
struct Exp {
    ALWI static void init() { exp_tile_init<approx>(); }
    ALWI static void compute(uint32_t dst) { exp_tile<approx>(dst); }
};

struct Elu {
    float alpha;
    ALWI static void init() { elu_tile_init(); }
    ALWI void compute(uint32_t dst) const {
        elu_tile(dst, std::bit_cast<uint32_t>(alpha));
    }
};

// Helper — ONE generic free function
template <uint32_t input_cb, uint32_t output_cb, UnaryInputPolicy policy = WaitAndPop, typename Op>
ALWI void eltwise_unary(uint32_t num_tiles, Op op = Op{}) {
    // Handles: hw init, copy_tile_to_dst, CB wait/pop/reserve/push,
    //          tile_regs acquire/commit/wait/release, DEST batching, pack
    op.init();  // compile-time dispatch to the right *_tile_init()
    for (... batched tile loop ...) {
        op.compute(dst);  // compile-time dispatch to the right *_tile()
    }
}
```
```
