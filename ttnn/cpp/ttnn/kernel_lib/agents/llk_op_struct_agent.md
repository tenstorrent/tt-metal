---
name: LLK Op Struct Design Agent Prompt
description: Takes raw compute API signatures and host parameter flow for a set of operations and designs op-type-trait structs that encapsulate init/compute/params following the compute_kernel_lib pattern.
type: reference
---

## Usage

Invoke with `subagent_type: Explore`. Replace placeholders:
- `{{OPS_LIST}}` — comma-separated operation names
- `{{LOCATOR_RESULTS}}` — the locator table from Stage 0.5 (file paths for each op)
- `{{CODEGEN_FILE}}` — path to the op_utils file (for parameter flow)

## Prompt Template

```
Design op-type-trait structs for these operations: {{OPS_LIST}}.

BREADCRUMB LOGGING — do this first:
Derive CATEGORY_SLUG from the category implied by the ops (or ask orchestrator).
BCRUMB="agent_logs/${CATEGORY_SLUG}_op_struct_breadcrumbs.jsonl"
Run at start:
  mkdir -p agent_logs
  echo '{"ts":"'"$(date -Iseconds)"'","event":"start","agent":"op_struct","ops":"{{OPS_LIST}}"}' >> $BCRUMB
After reading wrapper signature for each op:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"read","op":"OP","wrapper_file":"path","init_sig":"OP_tile_init(...)","compute_sig":"OP_tile(uint32_t dst, ...)"}' >> $BCRUMB
After reading codegen entry for each op:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"read","op":"OP","codegen_file":"path:line","host_transforms":["std::bit_cast<uint32_t>(alpha)","1.0f/alpha"],"caller_provided_params":["alpha"]}' >> $BCRUMB
After classifying each op:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","op":"OP","pattern":"ZERO_PARAM/NAMED_FIELDS/DERIVED_PARAM","template_params":N,"runtime_fields":M,"reason":"WHY this classification"}' >> $BCRUMB
When absorbing a host-computed derived param:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"decision","op":"OP","absorbed":"1/alpha computed from alpha","reason":"caller should not manually bit_cast or pre-compute reciprocal"}' >> $BCRUMB
When a non-obvious struct design choice is made:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"decision","op":"OP","what":"init_static_vs_member","choice":"static because init takes no runtime params","alternative_considered":"member function"}' >> $BCRUMB
At completion:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"complete","zero_param":N,"named_fields":M,"derived_param":K}' >> $BCRUMB
Write agent_logs/${CATEGORY_SLUG}_op_struct_execution_log.md: classification summary table, each non-obvious struct decision, all host-transform absorptions.



Use the locator results to find each op's source files:
{{LOCATOR_RESULTS}}

For EACH operation, read ONLY these two things:

1. COMPUTE API WRAPPER (from locator): Read the `*_tile_init()` and `*_tile()` signatures. Extract:
   - Template parameters (name, type, default)
   - Runtime parameters (name, type — typically uint32_t for bit-cast floats)
   - What the wrapper calls internally (for documentation only)

2. HOST CODEGEN (from locator): Find the operation in {{CODEGEN_FILE}}. Extract:
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
- For sentinel/optional types used as helper parameters (e.g., `NoOp`, `NoAccumulation`), use `explicit` default constructors. This prevents silent `{}` brace-initialization at call sites — callers must write `NoOp{}` not `{}`, making intent visible. Example: `struct NoOp { explicit NoOp() = default; ALWI void operator()(uint32_t = 0) const {} };`

CLASSIFY each op into one of these patterns:
- **ZERO_PARAM**: No runtime params. Empty struct with optional template params. E.g., `Exp<true>{}`
- **NAMED_FIELDS**: 1+ runtime params as named float fields. E.g., `Elu{.alpha = 1.0f}`
- **DERIVED_PARAM**: Has params the host currently pre-computes (reciprocals, etc.) that the struct should compute internally. E.g., `Softplus{.beta = 1.0f, .threshold = 20.0f}` where struct computes `1/beta`

OUTPUT FORMAT (strict):

## Op Struct Designs

For each operation, output exactly this block:

### {OpName}

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
