---
name: LLK Catalog Agent Prompt
description: "Phase 0: Understand — catalog step (new helper mode). Enumerates all ops via bidirectional grep, assigns to groups, and discovers the call-site kernel-file set by content (not path globs). Runs before the investigation step within the same phase."
type: reference
---

## Usage

Invoke with `subagent_type: Explore`. Replace placeholders:
- `{{LLK_CATEGORY}}` — e.g. "elementwise unary"
- `{{LLK_PREFIX}}` — primary LLK function name prefix, e.g. `llk_math_eltwise_unary_sfpu`
- `{{ADDITIONAL_LLK_PREFIXES}}` — other prefixes for this category (may be empty)
- `{{COMPUTE_API_DIR}}` — e.g. `tt_metal/hw/inc/api/compute/eltwise_unary`
- `{{SECONDARY_SOURCE}}` — enum or op_utils file for cross-check, e.g. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
- `{{KNOWN_GROUPS}}` — optional: pre-seeded group→ops mapping from the HQ table. If provided, use it for group assignment; only assign "Ungrouped" to ops that don't match any known group.

## Prompt Template

```
Enumerate all {{LLK_CATEGORY}} operations using lightweight bidirectional grep. Do NOT read full file bodies — only grep for function names and list file paths.

BREADCRUMB LOGGING:
Derive CATEGORY_SLUG = "{{LLK_CATEGORY}}" lowercased, spaces/slashes → underscores.
LOG_DIR="agent_logs/${CATEGORY_SLUG}"
BCRUMB="${LOG_DIR}/catalog_breadcrumbs.jsonl"
Run at start:
  mkdir -p "${LOG_DIR}"
  echo '{"ts":"'"$(date -Iseconds)"'","event":"start","agent":"catalog","category":"{{LLK_CATEGORY}}","prefix":"{{LLK_PREFIX}}"}' >> $BCRUMB

PHASE 1A: BOTTOM-UP — grep LLK prefix

Grep for `{{LLK_PREFIX}}` (and `{{ADDITIONAL_LLK_PREFIXES}}` if provided) in:
- `/localdev/astancov/tt-metal/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/`
- `/localdev/astancov/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_api/`

For each match: extract base op name (strip prefix and `_init` suffix). Record the file.
Produce LLK_OPS = deduplicated base names.

  echo '{"ts":"'"$(date -Iseconds)"'","event":"search","phase":"1A","file":"FILE","ops_found":["OP1","OP2"]}' >> $BCRUMB

PHASE 1B: TOP-DOWN — grep compute API directory

List all headers in `/localdev/astancov/tt-metal/{{COMPUTE_API_DIR}}/` (skip README.md, sfpu_split_includes.h).
For each header, grep for `_tile_init\|_tile(` — extract base op name from matched function names.
Do NOT read full file contents — use grep only.
Produce API_OPS = deduplicated base names.

  echo '{"ts":"'"$(date -Iseconds)"'","event":"search","phase":"1B","file":"FILE","ops_found":["OP1"],"note":"grep-only, no read"}' >> $BCRUMB

Also grep `/localdev/astancov/tt-metal/tt_metal/include/compute_kernel_api/` for any additional `*_tile(` not yet found.

PHASE 1C: GAP ANALYSIS

Compare LLK_OPS and API_OPS:
- In API_OPS but not LLK_OPS: top-down-only (op uses SFPU macro directly)
- In LLK_OPS but not API_OPS: LLK-only (no public compute API wrapper)

  echo '{"ts":"'"$(date -Iseconds)"'","event":"gap","op":"OP","found_via":"API","missing_from":"LLK"}' >> $BCRUMB
  echo '{"ts":"'"$(date -Iseconds)"'","event":"phase_done","phase":"1C","both":N,"top_down_only":A,"llk_only":B}' >> $BCRUMB

PHASE 1D: KERNEL FILE DISCOVERY (content-based)

Discover the set of compute kernel files that downstream agents (investigation Dimension 3,
fleet-migration audits) must search for call sites. Do NOT use path globs — they have
historically missed real kernels:
- `kernels_ng/compute/` (binary_ng production path) — `kernels_ng` not `kernels`.
- `kernels/attention/compute/` (softmax) — extra subdir between `kernels/` and `compute/`.
- `ttnn/cpp/ttnn/kernel/compute/` — singular `kernel`, not `kernels`.
- `tt-train/sources/ttml/metal/.../kernels/compute/` — sister tree, not under `ttnn/` or `tests/`.
- `tt_metal/programming_examples/.../kernels/compute/` — examples dir.

Discover by content (file imports the compute kernel API):

  grep -rlE '#include[[:space:]]+["<](compute_kernel_api|compute_kernel_lib|llk_math|llk_unpack|llk_pack)' \
    --include='*.cpp' \
    ttnn tt_metal tt-train tools \
    2>/dev/null \
    | grep -vE '\.cpmcache|/build|/third_party|/3rd_party' \
    | sort -u

Produce KERNEL_FILES = the resulting list.

Do NOT read these files in the catalog phase — produce the list only. The investigation
agent (Dimension 3) consumes KERNEL_FILES to grep for op call sites.

Notes for downstream consumers:
- `tt-train/` kernels build against ttnn helpers (same monorepo, shared build). Include
  unless the helper being designed is explicitly scoped out of tt-train.
- `tt_metal/programming_examples/` kernels are reference / illustrative. Treat their
  patterns as weak signal vs. production `ttnn/cpp/ttnn/operations/` kernels.
- `tools/tests/triage/hang_apps/` kernels are deliberately broken fixtures — exclude
  from pattern mining (they model hangs, not idiomatic usage).

  echo '{"ts":"'"$(date -Iseconds)"'","event":"discovery","phase":"1D","kernel_files":N,"method":"content"}' >> $BCRUMB

PHASE 2: GROUP ASSIGNMENT

If {{KNOWN_GROUPS}} is provided: assign each op in the union (LLK_OPS ∪ API_OPS) to its group.
Ops not matching any known group → "Ungrouped".

If no known groups provided: cluster by naming patterns only (no code reading).
Use the operation names themselves to infer functional groupings — ops that share a common prefix,
suffix, or domain concept belong together. Examples of the kind of clustering to apply:
- Ops that share a mathematical domain (trig, rounding, bitwise, etc.)
- Ops that share a computational role (activations, comparisons, reductions)
- Ops that share a naming prefix or suffix pattern

The specific groups will vary by category — do NOT apply a fixed group list.
Assign "Ungrouped" to any op that does not clearly fit a cluster.
Exclude infrastructure functions (_params_, generic dispatchers) and ops that belong to a different category (note separately).

  echo '{"ts":"'"$(date -Iseconds)"'","event":"decision","phase":"2","group":"GROUP","ops":["OP1","OP2"],"method":"known_groups/name_pattern"}' >> $BCRUMB

PHASE 3: CROSS-CHECK

Grep `{{SECONDARY_SOURCE}}` for op names / enum values.
Flag any op found there but not already in LLK_OPS ∪ API_OPS.

  echo '{"ts":"'"$(date -Iseconds)"'","event":"gap","phase":"3","op":"OP","found_in":"secondary","missing_from":"1A+1B"}' >> $BCRUMB

At completion:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"complete","total_ops":N,"groups":M,"ungrouped":K,"secondary_gaps":J}' >> $BCRUMB
Write ${LOG_DIR}/catalog_execution_log.md: counts per phase, ungrouped ops with reason, secondary gaps found.

OUTPUT FORMAT — structured tables only:

## 1. Full Op List

| Op | Found Via | Header File | LLK File |
|----|-----------|-------------|----------|

(Found via = "LLK prefix", "compute API", or "both")

## 2. Gap Analysis

### Top-down only (no named LLK function)
| Op | Header File |
|----|------------|

### LLK-only (no compute API wrapper)
| Op | LLK File |
|----|----------|

## 3. Group Assignments

| Group | Ops | Count |
|-------|-----|-------|

**This table is the primary output consumed by Stage 1b agents.**
One row per group. Ops is a comma-separated list.

## 4. Excluded / Ungrouped

| Op | Reason |
|----|--------|

## 5. Cross-Check

| Op | Found In | Missing From |
|----|----------|-------------|

Only ops found in secondary source but NOT in Phases 1A+1B.

## 6. Kernel Files (call-site search set)

Plain list — one repo-relative path per line, sorted, deduped. Result of Phase 1D.

```
<path/to/kernel1.cpp>
<path/to/kernel2.cpp>
...
```

Also write the same list to `${LOG_DIR}/kernel_files.txt` so downstream agents can
read it directly without re-deriving. Investigation Dimension 3 grep call sites
within this list — not within path globs.
```
