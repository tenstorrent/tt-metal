---
name: LLK Investigation Device Agent Prompt
description: "Investigation sub-agent: device-side analysis. Reads compute API wrappers, LLK/ckernel implementations, init state, and parameter semantics. Produces structured tables."
type: reference
---

## Usage

Invoke with `subagent_type: Explore`. Replace placeholders:
- `{{GROUP_NAME}}` — functional sub-group (e.g. Activations)
- `{{LLK_CATEGORY}}` — operation category (e.g. elementwise unary)
- `{{OPS_LIST}}` — comma-separated operation names
- `{{LOCATOR_RESULTS}}` — the locator table from Stage 0.5

## Prompt Template

```
Analyze the DEVICE-SIDE implementation of these {{GROUP_NAME}} {{LLK_CATEGORY}} operations: {{OPS_LIST}}.

BREADCRUMB LOGGING — do this first:
Derive CATEGORY_SLUG from {{LLK_CATEGORY}} (lowercase, spaces → underscores).
BCRUMB="agent_logs/${CATEGORY_SLUG}_device_breadcrumbs.jsonl"
Run at start:
  mkdir -p agent_logs
  echo '{"ts":"'"$(date -Iseconds)"'","event":"start","agent":"device","group":"{{GROUP_NAME}}","ops":"{{OPS_LIST}}"}' >> $BCRUMB

Log granularly as you read each file. Capture reasoning, not just facts:

After opening each wrapper file:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"read","op":"OP","file":"wrapper_path","key_observation":"e.g. init calls LLK function with specific type tag or configures specific HW resource"}' >> $BCRUMB

After reading LLK/ckernel (or determining there is none):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"read","op":"OP","file":"llk_path_or_NONE","key_observation":"e.g. no named LLK fn — wrapper calls HW macro directly"}' >> $BCRUMB

After determining init HW config status (explain why):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","op":"OP","fact":"init_configures_hw","value":true/false,"hw_resources":"e.g. SFPU pipeline / FPU pipeline / unpack mode","evidence":"file:line — init body is {} / calls specific HW init"}' >> $BCRUMB

After counting params (explain each param's role):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","op":"OP","fact":"params","count":N,"params":["alpha:float_bitcast","dst_index:uint32"],"note":"alpha is user value bit-cast to uint32 before calling"}' >> $BCRUMB

When you spot an outlier or something unexpected:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","op":"OP","fact":"outlier","observation":"e.g. only op in group with 3 runtime params","implication":"cannot share struct template with rest of group"}' >> $BCRUMB

For each init compatibility pair (explain the reasoning chain):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","op_pair":"A+B","fact":"init_compat","compatible":true/false,"type_tag_a":"X","type_tag_b":"Y","conclusion":"same type tag and HW config → shared init / different → must re-init between ops"}' >> $BCRUMB

At completion:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"complete","ops_analyzed":N,"trivial_init_count":M,"outliers":K}' >> $BCRUMB
Write agent_logs/${CATEGORY_SLUG}_device_execution_log.md: per-op summary of what was read and concluded, init HW config breakdown with evidence, compatibility pairs with reasoning, outliers.



Use the locator results to find each op's files — do NOT search for files yourself:
{{LOCATOR_RESULTS}}

IMPORTANT: Not all operations have a named LLK function. Some wire directly to hardware implementations via macros. For these ops, the compute API wrapper header IS the primary definition.

For EACH operation, READ the wrapper file and LLK/ckernel file from the locator results. Extract:

1. WRAPPER SIGNATURE:
   - `*_tile_init()` (or equivalent init): full signature including template params and defaults
   - `*_tile()` (or equivalent exec): full signature including template params, runtime params, and defaults
   - What macro, LLK function, or hardware call the wrapper invokes internally

2. INIT STATE:
   - What enum value or type tag is used to identify this op (e.g., SfpuType for SFPU ops, BinaryType for binary ops, ReduceFunc for reductions — the specific enum depends on the category)
   - What ckernel/LLK init function is called
   - Does the init function configure hardware state, or is it empty/trivial?
   - Does init take any parameters or is it purely config?
   - What hardware resources does init configure? (e.g., SFPU pipeline, FPU pipeline, unpack mode, math pipeline)

3. PARAMETER SEMANTICS — for each runtime parameter of the exec function:
   - Semantic name (not "param0" — what does it mean?)
   - Type at the wrapper boundary (uint32_t for bit-cast floats, bool, enum, etc.)
   - Is it a raw value, a bit-cast float, or derived from another param?
   - Where does the value end up in hardware? (register load, function arg, template param)

4. CKERNEL/LLK COMPUTE FUNCTION:
   - Function name and signature
   - Key template parameters that affect behavior
   - Does it use DST_ACCUM_MODE?

OUTPUT FORMAT (strict tables — minimal narrative):

## Device-Side Analysis: {{GROUP_NAME}}

### Wrapper Signatures

| Operation | Init Signature | Exec Signature | Internal Call |
|-----------|---------------|----------------|---------------|

### Init State

| Operation | Op Type Tag | Init Function | Init Configures HW? | HW Resources Configured | Init Params |
|-----------|------------|---------------|---------------------|------------------------|-------------|

(Op Type Tag = whatever enum/tag identifies this op in the LLK layer — e.g. SfpuType for SFPU, BinaryType for binary, ReduceFunc for reduce. Write "none" if the op has no type tag.)

### Parameter Semantics

| Operation | Param Name | Meaning | Type at Wrapper | Encoding | Derived From | HW Destination |
|-----------|-----------|---------|-----------------|----------|-------------|----------------|

### LLK/Ckernel Functions

| Operation | Ckernel Function | Key Template Params | Uses DST_ACCUM_MODE |
|-----------|-----------------|--------------------|--------------------|

### Init State Compatibility

Can any ops in this group share init state? Compare type tags and init functions.
Two ops are compatible if calling init_A followed by exec_B produces correct results (i.e., init_A sets up the same hardware state that exec_B requires).
Output ONE line per pair: `{op1} + {op2}: COMPATIBLE | INCOMPATIBLE | UNKNOWN (reason)`

### Outliers

List any operation that breaks the common pattern in this group, with explanation.
```
