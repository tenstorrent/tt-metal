# Operation Requirements: g6_sharded_to_interleaved

## Definition
- **Formula**: `out[i] = input_tensor[i]` for every logical index `i`, bit for bit. Page `p = a*C + b` of the interleaved output holds the input tile at tile row `a`, tile column `b`.
- **PyTorch Reference**: `def ref(x, memory_config=None): return torch.clone(x)`
- **Import Path**: `from ttnn.operations.g6_sharded_to_interleaved import g6_sharded_to_interleaved`
- **Function Signature**: `g6_sharded_to_interleaved(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor` (both positional; input TILE, height- or width-sharded in L1, rank 2 or 4; `memory_config` interleaved DRAM or L1; output has the input's shape and dtype, TILE, interleaved as `memory_config` names)

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED op-specific axes**: in_scheme ∈ {width_sharded}, buffer ∈ {dram}
- **Regime**: `owner_writes`. Each shard core writes its resident shard (zero-copy `cb_input_shard`) straight to the interleaved pages: one NoC crossing per tile, one barrier per kernel.
- **Cores**: the input's shard grid (32 cores for p2, 103 for p1). Placement fixes the assignment; this is the minimum-traffic split.
- **Knobs**: `Knobs.blocks_per_shard = 1`, `Knobs.writer_lanes = 1` (single source, part of the cache key).
- **Compute config**: none (no compute kernel, no arithmetic)
- **Golden baseline**: 13 / 17 cells passing (13 supported_pass, 4 xfail_expected, 0 loud), per `verifier_report.json`

### [x] Refinement 1 — Claim the Panoptic production contract: bfloat8_b, height_sharded, L1 output

**Goal**: add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`, `"height_sharded"` to `SUPPORTED["in_scheme"]` and `"l1"` to `SUPPORTED["buffer"]`, so that SUPPORTED == TARGET. This lands the full contract of the perf-flagged `LOOSE_CASES[0]`: (1,128,256,32) bf8b, height (320,32) on two ranges, written to L1 interleaved. All three are **knob-turns** of the one owner-writes block operation:
- the dtype only changes the page size (1088 B bf8b pages, shared exponents moved verbatim as raw bytes, never through unpack/pack);
- the scheme only changes host-computed RT args (shard rectangle origin);
- the buffer only changes `TensorAccessorArgs(output)`.

EXCLUSIONS stays `[]`. The prompt forbids gating any production cell, and no size/shape tagger may be added. No `compute_kernel_config` is exposed: the op has no compute kernel, and the contract fixes the two-argument signature.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- The verifier already ran p1 production and p1 other spec through the real entry point with SUPPORTED widened, and both are bit-exact (verification_report.md § Precision Baseline). This refinement should therefore need zero kernel or descriptor changes; any kernel change here is a red flag.
- `in_scheme` / `buffer` are memory-placement values with no dedicated skill. The native path is already the zero-copy sharded-CB placement (`cb_descriptor_from_sharded_tensor`) and must stay so: never re-read the local shard through a TensorAccessor.
- Remove the `monkeypatch` SUPPORTED widening from `test_..._precision_baseline.py` and `test_..._perf.py` once claimed (it becomes a no-op; delete it for clarity).
- Run only the cells this change can affect (BB-22): the four xfail_expected cells plus the acceptance file.

**Done when**:
- the four current `xfail_expected` cells pass bit-exact: golden 17/17 supported_pass, 0 loud categories;
- the acceptance file shows 8 passed, 0 xfailed;
- the precision baseline stays bit-exact on all four cases.

### [x] Refinement 2 — Speed up the perf-flagged profile: kernel-side write issue

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` flags p1 as the mandatory perf target: (1,128,256,32) bfloat8_b, height sharded (320,32) on (0,0)-(10,8)+(0,9)-(3,9), row-major, written to L1 interleaved. Its perf goal is a kernel time below the 2.05 µs stock incumbent, and its gate is exact.

Phase 0 measures 1939 ns kernel and 5954–5960 ns/op trace replay. Speed up the kernel's write-issue phase on this exact config using the relevant `ttnn/ttnn/operations/examples/master.md` levers:
- `split_reader`: the existing `Knobs.writer_lanes=2`, measured at 1564.5 ns kernel but +210 ns replay.
- `bank_stagger`: rotate each core's start row so neighbouring cores do not hit the same L1 banks at once. Use a host-derived per-core start index, not a per-row modulo.
- `noc_placement`: choose the NoC per lane or per core.

These are T2 knob-turns on the surface the planner exposed, so several can be tried in one phase. No SUPPORTED change.

**Verifier notes**:
- Depends on Refinement 1: the flagged config must be claimed, not measured through a widened SUPPORTED.
- The model replays traces, so a lever counts only if it does not regress replay ns/op. The second-lane result above (kernel −19%, replay +3.5%) shows the two measures can disagree. If a lever wins kernel but loses replay, it may be gated by a host predicate, but the plan must stay reportable via `last_plan()` (BB-08) and in the cache key (BB-07).
- Use `test_g6_sharded_to_interleaved_perf.py` (kernel ns under `--profile` + `probes/perf_csv_summary.py`; replay via `test_trace_replay_ns_per_op`).

**Done when**:
- measured DEVICE KERNEL DURATION on the flagged p1 config improves over 1939 ns with replay ns/op not above ~5960;
- the gate stays exact and the golden suite is green;
- there is no regression across the guard set: one representative per kernel path × placement, i.e. p2 width bf16 → DRAM (1429.5 ns kernel / 2820 ns replay) and p2 → L1.
**Outcome**: p1 kernel 1939 → **1299 ns** (−33%), replay 5952–5960 → **5883–5887 ns/op**; gate exact, golden 17/17. Bottleneck was NoC congestion, not RISC issue cost: on one lane the 10-write issue zone took 1198 cycles p50 (2215 max) against 292 uncontended, and at every step the 103 cores hit only 11 of the 110 L1 banks. Winning plan (gated by `select_knobs`: L1 output with ≥ `L1_OUTPUT_MIN_PAGES`=512 pages) = one BRISC kernel in DM_DYNAMIC_NOC that alternates writes over NoC0/NoC1 (`dual_noc`), plus a host-derived start slot `(first_page // num_banks) % lane_slots` (`bank_stagger`). Neither wins alone (stagger 2067, dual 1710); together they reach 1297, beating the second-lane variants (lanes2 1570, lanes2+stagger 1390) without the second kernel's replay cost. NoC0 for the single lane loses (2397). Guards unchanged: p2 → DRAM keeps the Phase 0 plan (dual/NoC0 hurt DRAM writes), and p2 → L1 (160 pages) keeps it too (dual was a flat kernel and +1.5% replay). Now: per-core end spread is tight (max 1821 cycles vs 2812), and issue is ~98 cycles/write evenly across the grid (aggregate L1-bank/NoC throughput) plus ~230 cycles of launch skew. Next would be per-call dispatch cost (Refinement 3, which dominates replay at ~3× kernel). Not tried here because it is out of this heading's scope.

### [ ] Refinement 3 — Speed up the perf-flagged profile: per-call dispatch / launch cost

**Type**: perf

**Goal**: same flagged p1 config as Refinement 2. On it, the traced per-op cost (≈ 5960 ns/op) is about 3× the kernel (≈ 1939 ns) and ≈ 2× the FW duration (≈ 2874 ns), while p2 on 32 cores replays at ≈ 2820 ns/op. The gap is per-call dispatch/launch work, which scales with the 103 cores.

Reduce it on this exact config:
- shrink the per-core **unique** runtime args. Today each core gets 6 (`num_blocks, block_rows, block_cols, total_rows, first_page, src_first_slot`), most of which are uniform across a scheme and can move to common RT args / CT args, leaving only the shard index or first page per core;
- keep kernel core ranges to the shard grid's few rectangles (BB-17);
- keep the kernel binary small (BB-18).

Cross-check against the catalog in `ttnn/ttnn/operations/examples/master.md` for any applicable entry. No SUPPORTED change.

**Verifier notes**:
- Run after Refinement 2: its lane/NoC choice decides how many kernel instances (and RT-arg sets) exist, which is what this refinement trims.
- The program-cache contract must hold: only the output address may be patched per call, and orientation / range order must stay in `custom_program_hash` (the `cache_order` regression).

**Done when**:
- trace-replay ns/op on the flagged p1 config improves measurably over the Refinement 2 result, with no kernel-ns regression;
- the gate stays exact and the golden suite is green;
- there is no regression on the guard set (p2 → DRAM, p2 → L1) in kernel ns or replay ns/op.
