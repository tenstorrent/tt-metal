---
name: LLK Locator Agent Prompt
description: Lightweight pre-stage agent that finds where each operation is defined (wrapper file, LLK file, codegen line). Output feeds into Investigation sub-agents so they don't waste time searching.
type: reference
---

## Usage

Invoke with `subagent_type: Explore`. Replace placeholders:
- `{{OPS_LIST}}` — comma-separated operation names to locate
- `{{COMPUTE_API_DIR}}` — e.g. `tt_metal/hw/inc/api/compute/eltwise_unary`
- `{{LLK_DIR}}` — LLK/ckernel directory, e.g. `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu` (unary SFPU) or `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api` (broader)
- `{{CODEGEN_FILE}}` — e.g. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `{{PROGRAM_FACTORY_DIR}}` — e.g. `ttnn/cpp/ttnn/operations/eltwise/unary/device`

## Prompt Template

```
Locate the source files for these operations: {{OPS_LIST}}.

BREADCRUMB LOGGING — do this first:
Derive CATEGORY_SLUG from the compute API dir (e.g. "eltwise_unary" from "eltwise_unary/").
BCRUMB="agent_logs/${CATEGORY_SLUG}_locator_breadcrumbs.jsonl"
Run at start:
  mkdir -p agent_logs
  echo '{"ts":"'"$(date -Iseconds)"'","event":"start","agent":"locator","ops":"{{OPS_LIST}}"}' >> $BCRUMB

Log each search and its result. When a search yields no matches, that is important to record.

After each grep for each op (log even zero-match searches):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"search","op":"OP","looking_for":"wrapper","pattern":"exp_tile_init","dir":"{{COMPUTE_API_DIR}}","matches":N,"matched_file":"exp.h:12_or_NONE"}' >> $BCRUMB
  echo '{"ts":"'"$(date -Iseconds)"'","event":"search","op":"OP","looking_for":"llk","pattern":"exp","dir":"{{LLK_DIR}}/llk_sfpu/","matches":N,"matched_file":"ckernel_sfpu_exp.h_or_NONE"}' >> $BCRUMB

When NONE is returned for any field, explain why:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","op":"OP","field":"llk","result":"NONE","reason":"no ckernel_sfpu_exp.h found — op uses SFPU macro directly in wrapper header"}' >> $BCRUMB

When multiple matches are found and you must pick one:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","op":"OP","field":"codegen","matches":["file1:34","file2:89"],"chosen":"file1:34","reason":"file2 is for a different arch/variant"}' >> $BCRUMB

After locating each op (summary event):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","op":"OP","wrapper":"file:line","llk":"file_or_NONE","codegen":"file:line","factory":"path","custom_kernel":"path_or_NONE"}' >> $BCRUMB
At completion:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"complete","ops_located":N,"ops_missing_wrapper":M,"ops_missing_llk":K}' >> $BCRUMB
Then write agent_logs/${CATEGORY_SLUG}_locator_execution_log.md listing: per-op search results, NONE fields with reasons, any ambiguous matches and resolution.



For EACH operation, find these files (use grep/glob, do NOT read file contents beyond confirming the match):

1. COMPUTE API WRAPPER: grep for `{op}_tile_init` or `{op}_tile(` in `/localdev/astancov/tt-metal/{{COMPUTE_API_DIR}}/`
2. LLK / CKERNEL: grep for `{op}` in `/localdev/astancov/tt-metal/{{LLK_DIR}}/` and its subdirectories
3. CODE GENERATION: grep for the operation name (or its enum value) in `/localdev/astancov/tt-metal/{{CODEGEN_FILE}}`
4. PROGRAM FACTORY: grep for the operation name in `/localdev/astancov/tt-metal/{{PROGRAM_FACTORY_DIR}}/` (*.cpp files)
5. CUSTOM KERNEL: check if a dedicated kernel exists at `/localdev/astancov/tt-metal/{{PROGRAM_FACTORY_DIR}}/kernels/compute/` matching the op name

IMPORTANT:
- Do NOT read file contents. Only find file paths and line numbers.
- Do NOT analyze, interpret, or summarize what the code does.
- If an operation has no match for a category, output "NONE".
OUTPUT FORMAT (strict — no other text):

## Locator Results

| Operation | Wrapper File | Wrapper Line | LLK/Ckernel File | Codegen File:Line | Program Factory | Custom Kernel |
|-----------|-------------|-------------|-------------------|-------------------|-----------------|---------------|
| {op1}     | {path}      | {line}      | {path}            | {path}:{line}     | {path}          | {path or NONE}|
| {op2}     | ...         | ...         | ...               | ...               | ...             | ...           |

One row per operation. Absolute paths. No narrative.
```
