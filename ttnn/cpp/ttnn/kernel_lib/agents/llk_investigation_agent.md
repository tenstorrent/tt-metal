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

═══ OUTPUT ═══

Save to: agent_logs/{{CATEGORY_SLUG}}_{{GROUP_SLUG}}_investigation.md

The orchestrator will consolidate per-group outputs into {category}_investigation.md.
```

## Focus Directives

When the category or situation calls for emphasizing one dimension over others, include a focus directive:

**Compute-heavy category** (SFPU, matmul):
```
Focus on: Device dimension (wrapper signatures, init state, batching limits) and
Usage dimension (chaining patterns, init mutual exclusion). De-emphasize Host
dimension — program factory details are less critical when the helper wraps
only the compute kernel.
```

**Data-movement category** (tilize, untilize):
```
Focus on: Host dimension (CB layout, parameter encoding, factory sharing) and
Usage dimension (call sites, boilerplate patterns). De-emphasize Device
dimension — the LLK implementations are simpler for data movement ops.
```

**New/unknown category**:
```
Equal emphasis on all three dimensions. Flag any surprising findings.
```
