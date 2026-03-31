---
name: LLK Verification Agent Prompt
description: Claim-specific verification agent. Takes individual claims from the orchestrator and verifies each against actual code. Returns structured verdicts. Does NOT verify entire groups at once.
type: reference
---

## Usage

Invoke with `subagent_type: Explore`. Replace placeholders:
- `{{CLAIMS}}` — structured list of claims to verify (see format below)

## Claim Format

Each claim must be a structured entry:

```
CLAIM_ID: {unique id}
CLAIM: {the specific assertion to verify}
EVIDENCE_NEEDED: {what code to check}
OPS: {which operations this claim is about}
PRIORITY: high | medium | low
```

Example claims:
```
CLAIM_ID: init_softplus_empty
CLAIM: softplus_tile_init() has an empty init function (no SFPU state setup)
EVIDENCE_NEEDED: Read ckernel_sfpu_softplus.h, find softplus_init()
OPS: softplus
PRIORITY: high

CLAIM_ID: exp_elu_init_incompatible
CLAIM: exp and elu use different SfpuType values and cannot share SFPU init
EVIDENCE_NEEDED: Compare SfpuType enum values in their respective init functions
OPS: exp, elu
PRIORITY: medium
```

## Prompt Template

```
Verify these specific claims about LLK operations by checking actual source code.

BREADCRUMB LOGGING — do this first:
Derive CATEGORY_SLUG from the ops in the claims (or ask orchestrator if unclear).
BCRUMB="agent_logs/${CATEGORY_SLUG}_verification_breadcrumbs.jsonl"
Run at start:
  mkdir -p agent_logs
  echo '{"ts":"'"$(date -Iseconds)"'","event":"start","agent":"verification","claim_count":N}' >> $BCRUMB

Log the verification process in detail — one breadcrumb per file read, one per intermediate finding, one for the final verdict.

Before starting each claim (what you plan to check and why):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"claim","id":"CLAIM_ID","ops":"OP_LIST","plan":"read softplus_init() in ckernel_sfpu_softplus.h and check if body is empty","expected_location":"llk_sfpu/ckernel_sfpu_softplus.h:~line_40"}' >> $BCRUMB

After opening each file for a claim:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"read","for_claim":"CLAIM_ID","file":"ckernel_sfpu_softplus.h","line":42,"what_was_seen":"softplus_init() { } — body is empty, no SFPU state setup"}' >> $BCRUMB

When intermediate evidence is found but more context is needed:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","for_claim":"CLAIM_ID","intermediate":"found SfpuType::EXP at exp_tile_init:12 — now need to check elu_tile_init to compare"}' >> $BCRUMB

After determining the verdict (explain the full reasoning chain):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"verdict","id":"CLAIM_ID","verdict":"CONFIRMED/INCORRECT/UNVERIFIABLE","evidence":"file:line — exact quote of what the code shows","reasoning":"claim said X, code at file:line shows Y, therefore CONFIRMED/INCORRECT because Z"}' >> $BCRUMB

If INCORRECT, log what the correct fact is:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"verdict","id":"CLAIM_ID","verdict":"INCORRECT","evidence":"file:line","correction":"claim said init is empty but softplus_init() calls sfpu_init(SfpuType::SOFTPLUS) at line 44"}' >> $BCRUMB

At completion:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"complete","confirmed":N,"incorrect":M,"unverifiable":K}' >> $BCRUMB
Write agent_logs/${CATEGORY_SLUG}_verification_execution_log.md: per-claim file trail (what was read, what was seen), reasoning chain from evidence to verdict, corrections for INCORRECT claims.

Claims to verify:
{{CLAIMS}}

RULES:
- Verify ONLY the claims listed. Do not explore beyond what each claim requires.
- For each claim, read the minimum code needed to confirm or refute it.
- Cite exact file paths and line numbers for every verdict.
- If a claim is ambiguous or the code doesn't clearly confirm/refute, say UNVERIFIABLE.

For each claim, check the relevant code:

INIT CLAIMS — Read the init wrapper and trace into the LLK/ckernel init:
- Check the compute API directory for the wrapper (path depends on category — use `{{COMPUTE_API_DIR}}` if provided, or search `/localdev/astancov/tt-metal/tt_metal/hw/inc/api/compute/`)
- Trace into the LLK implementation (path depends on category — search under `/localdev/astancov/tt-metal/tt_metal/hw/ckernels/`)
- Compare type tags (SfpuType, BinaryType, ReduceFunc, etc.), init callbacks, and hardware configuration

PARAMETER CLAIMS — Read the compute API wrapper and count actual runtime parameters (excluding dst_index and vector_mode).

WRAPPER CLAIMS — Search the appropriate compute API directory for the op's tile functions.

DEAD CODE CLAIMS — Search for function calls (not declarations) across the codebase.

SHARED STATE CLAIMS — Compare type tags and init callback functions between ops. Two ops share state if they configure the same hardware resource to the same mode.

INIT PAIRING CLAIMS — Verify that a proposed init→exec sequence actually appears in real kernels:
- Search the codebase for kernels that use the claimed LLK functions together
- Verify the init placement (before loop vs per-tile) matches the claim
- Verify mutual-exclusion claims by searching for counterexamples (any kernel that uses both inits)
- For ordering claims, read the kernel and confirm the exact init→exec sequence

OUTPUT FORMAT (strict — one block per claim, then summary):

## Verification Results

### CLAIM_ID: {id}
**Claim**: {the claim text}
**Verdict**: CONFIRMED | INCORRECT | UNVERIFIABLE
**Evidence**: {file:line — what the code shows}
**Correction** (if INCORRECT): {what the correct fact is}

### CLAIM_ID: {id}
...

## Summary

| Claim ID | Verdict | Brief Reason |
|----------|---------|-------------|
```

## When the Orchestrator Should Use This Agent

Use this agent ONLY for claims that require reading implementation details the orchestrator hasn't already seen. Do NOT use it for:
- Wrapper existence checks (orchestrator can grep directly)
- Simple parameter counts (orchestrator can read the wrapper)
- Dead code checks (orchestrator can grep directly)

DO use it for:
- Init state compatibility between ops (requires reading SFPU internals)
- ADDR_MOD configuration details
- Hardware state interactions between chained ops
- Whether two ops can share SFPU init without re-initializing
