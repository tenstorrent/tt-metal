---
name: LLK Investigation Usage Agent Prompt
description: "Investigation sub-agent: usage patterns. Searches ALL kernel call sites, extracts boilerplate patterns, hardware constraints, and CB management. Must be thorough — finding all usage patterns is critical."
type: reference
---

## Usage

Invoke with `subagent_type: Explore`. Replace placeholders:
- `{{GROUP_NAME}}` — functional sub-group (e.g. Activations)
- `{{LLK_CATEGORY}}` — operation category (e.g. elementwise unary)
- `{{OPS_LIST}}` — comma-separated operation names

## Prompt Template

```
Find ALL usage patterns for these {{GROUP_NAME}} {{LLK_CATEGORY}} operations: {{OPS_LIST}}.

BREADCRUMB LOGGING — do this first:
Derive CATEGORY_SLUG from {{LLK_CATEGORY}} (lowercase, spaces → underscores).
BCRUMB="agent_logs/${CATEGORY_SLUG}_usage_breadcrumbs.jsonl"
Run at start:
  mkdir -p agent_logs
  echo '{"ts":"'"$(date -Iseconds)"'","event":"start","agent":"usage","group":"{{GROUP_NAME}}","ops":"{{OPS_LIST}}"}' >> $BCRUMB

Log granularly as you search and read. Every search and every insight needs a breadcrumb.

After each grep in each directory (even zero-match ones):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"search","op":"OP","dir":"ttnn/cpp/ttnn/operations/","matches":N,"note":"e.g. all 5 hits are in auto-generated SFPU_OP_CHAIN kernels"}' >> $BCRUMB

After reading each call site (capture what you saw in the surrounding code):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"read","op":"OP","file":"path","line":N,"surrounding_pattern":"cb_wait_front→copy_tile→op_tile→pack_tile→cb_pop","init_placement":"before_loop","batching":"ndst=4"}' >> $BCRUMB

When you identify a new pattern variant (explain WHY it's different):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","fact":"pattern_variant","variant":"fused_after_binary","ops":["OP"],"differentiator":"no copy_tile call — op applied directly to DST after binary result","example_file":"path:line"}' >> $BCRUMB

When an op has zero call sites (this is significant — log explicitly):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","op":"OP","fact":"zero_call_sites","searched_dirs":["ttnn/...","tt_metal/..."],"interpretation":"dead code / only used via SFPU_OP_CHAIN auto-generation"}' >> $BCRUMB

When an init/exec pairing rule is identified:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","fact":"init_exec_pairing","llk_fn":"copy_tile","init":"copy_tile_to_dst_init_short","exec_follows_init":true,"placement":"before_loop","evidence":"file:line"}' >> $BCRUMB

When an init mutual exclusion is identified:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","fact":"init_mutual_exclusion","init_a":"unary_op_init_common","init_b":"binary_op_init_common","coexist":false,"kernels_checked":N,"evidence":"0 of N kernels use both"}' >> $BCRUMB

When an init ordering sequence is found in a chained kernel:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","fact":"init_ordering","file":"path","sequence":["copy_init","copy_exec","sfpu_init","sfpu_exec"],"note":"init always immediately precedes its own exec block"}' >> $BCRUMB

When a hardware constraint is identified (explain how you recognized it):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","fact":"hw_constraint","op":"OP","constraint":"DEST_limit","evidence":"caller manually loops ndst=min(4,remaining)","implication":"helper must absorb DEST batching logic","file":"path:line"}' >> $BCRUMB

When you see inconsistent boilerplate across call sites (flag it):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","fact":"boilerplate_inconsistency","op":"OP","inconsistency":"3 sites use init before loop, 1 site re-inits per tile","implication":"current callers are buggy OR init_per_tile is intentional for some reason"}' >> $BCRUMB

At completion:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"complete","total_call_sites":N,"dirs_searched":4,"pattern_variants":M,"hw_constraints":K,"zero_site_ops":["OP1","OP2"]}' >> $BCRUMB
Write agent_logs/${CATEGORY_SLUG}_usage_execution_log.md: per-dir search counts per op, each pattern variant with file:line example and what makes it distinct, each hw constraint with evidence file, ops with zero call sites flagged.



IMPORTANT: Thoroughness is critical. Missing a usage pattern means the helper design may not cover real-world use cases. Search ALL directories below.

PHASE 1: FIND ALL CALL SITES

For EACH operation, search for `{op}_tile_init` and `{op}_tile(` in ALL of these directories:
- `/localdev/astancov/tt-metal/ttnn/cpp/ttnn/operations/` (TTNN operation kernels — most important)
- `/localdev/astancov/tt-metal/tt_metal/kernels/compute/` (core compute kernels)
- `/localdev/astancov/tt-metal/models/` (demo/model kernels)
- `/localdev/astancov/tt-metal/tests/` (test kernels)

For each call site found, READ the surrounding code (20-30 lines before and after) to capture the FULL pattern.

PHASE 2: EXTRACT BOILERPLATE PATTERN

From the call sites, identify the common boilerplate that surrounds EVERY call to ops in this group:

a. CB MANAGEMENT: What `cb_wait_front`, `cb_pop_front`, `cb_reserve_back`, `cb_push_back` calls surround the op?
b. TILE REGISTERS: What `tile_regs_acquire`, `tile_regs_commit`, `tile_regs_wait`, `tile_regs_release` pattern is used?
c. DATA INGESTION: How does data reach DEST before the op executes? Common patterns:
   - Unary SFPU: `copy_tile` from one CB into DEST, then SFPU op on DEST
   - Binary FPU: `unpack AB` from two CBs, FPU op produces result in DEST
   - Reduce: `unpack A` with accumulation across tiles in DEST
   - Matmul: `unpack AB` with matrix multiply accumulation
   Record the EXACT ingestion sequence observed, do not assume a pattern.
d. PACK: How is `pack_tile` called after the op?
e. INIT PLACEMENT: Is the init function called once before the loop, or inside the loop per tile?
f. BATCHING: How many tiles per acquire/release cycle? Is DEST batching used (multiple tiles in DST simultaneously)?

PHASE 3: IDENTIFY PATTERN VARIANTS

Are there different usage patterns for the same op? For example:
- Standalone unary (copy → op → pack) vs fused with binary (op applied after binary result in DST)
- Streaming (wait/pop per tile) vs bulk (wait all upfront)
- Single-tile DST usage vs multi-tile batching
- Used via SFPU_OP_CHAIN (auto-generated) vs used in custom kernel code

PHASE 4: INIT/EXEC PAIRING RULES

From the call sites, extract the EXACT init-then-execute sequences that appear in real kernels. This is the most critical phase — helpers must only compose LLK calls in sequences that actually exist in the codebase.

For each operation, record:
a. What init function is called, and WHERE relative to the tile loop (before loop / inside loop / both)?
b. What execute function is called, and does it always immediately follow its own init (possibly with other code in between, but no conflicting init)?
c. When multiple LLK functions appear in the same kernel, what is the ordering of their inits and execs? Do any inits overwrite state needed by a previously-inited function?
d. Are there ANY kernels where two different init functions (e.g. from different LLK categories) coexist? If not, record that as a mutual-exclusion constraint.

Build the INIT PAIRING TABLE: for every pair of LLK functions that appear in the same kernel, record whether their inits are called together, sequentially, or never in the same file.

PHASE 5: HARDWARE CONSTRAINTS

From the call sites, identify what hardware constraints callers currently handle manually:
- DEST register limits (batching by ndst or DEST capacity)
- FP32 accumulation mode detection
- Block sizing
- Data format reconfiguration between ops

OUTPUT FORMAT:

## Usage Analysis: {{GROUP_NAME}}

### Call Sites

| Operation | File | Line | Context (standalone/fused/chain) | Batching | Init Placement |
|-----------|------|------|----------------------------------|----------|----------------|

### Common Boilerplate Pattern

Show the TYPICAL code pattern that surrounds these ops (pseudocode is fine):
```
{the common pattern with placeholders for the specific op call}
```

### Pattern Variants

| Variant | Description | Operations Using It | Example File:Line |
|---------|------------|--------------------|--------------------|

### Init/Exec Pairing Rules

For each LLK function used by ops in this group, document the observed init-then-exec contract from real kernels:

| LLK Function | Init Call | Init Placement | Exec Call | Exec Always After Own Init? | Example File:Line |
|--------------|-----------|----------------|-----------|----------------------------|-------------------|

### Init Mutual Exclusion

Record which init functions are NEVER observed in the same kernel. If no kernel in the codebase combines two categories of init, that is a hard constraint the helper must respect.

| Init A | Init B | Co-occur in Same Kernel? | Evidence (file or "0 of N kernels") |
|--------|--------|--------------------------|--------------------------------------|

### Init Ordering Sequences

For kernels that chain multiple LLK operations on the same data in DEST, record the exact init→exec ordering:

| Kernel File | Sequence (init/exec order) | Notes |
|-------------|---------------------------|-------|

### Hardware Constraints Callers Handle

| Constraint | How Callers Handle It | Could Helper Absorb? | Example |
|-----------|----------------------|---------------------|---------|

### Auto-Generated vs Custom Kernel

| Operation | Used via SFPU_OP_CHAIN? | Has Custom Kernel? | Custom Kernel Path |
|-----------|------------------------|-------------------|-------------------|

### Chaining Patterns

List any cases where ops from this group are chained with other ops (in SFPU_OP_CHAIN or custom kernels):
| Chain | Ops in Chain | Kernel File | Pattern (sequential DST[0] / multi-register / intermediate CB) |
|-------|-------------|-------------|----------------------------------------------------------------|
```
