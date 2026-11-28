# Operation Requirements: g6_zeros

## Definition
- **Formula**: `out[i0, i1, i2, i3] = +0.0` for every logical index, stored as the all-zero bit pattern of `dtype` (never -0.0; for bfloat8_b every datum's sign and magnitude bits are zero). `out` is new storage on every call, on every device of `device`.
- **PyTorch Reference**: `torch.zeros(tuple(shape), dtype=torch_dtype)`. bfloat8_b uses `torch.bfloat16`, whose zeros render in bfloat8_b as exact zeros.
- **Import Path**: `from ttnn.operations.g6_zeros import g6_zeros`
- **Function Signature**:
  ```python
  g6_zeros(
      shape,                                # (d_0, d_1, d_2, d_3): 4 ints, list or tuple; d_2, d_3 multiples of 32
      *,
      dtype: ttnn.DataType,                 # ttnn.float32 | ttnn.bfloat16 | ttnn.bfloat8_b
      layout: ttnn.Layout,                  # ttnn.TILE_LAYOUT
      device: ttnn.MeshDevice,              # device or mesh device
      memory_config: ttnn.MemoryConfig,     # DRAM interleaved
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: none. The spec forbids size, shape and rank axes. Rank 4 with `d2, d3 % 32 == 0` is an argument check (ValueError), not a support axis.
- **SUPPORTED op-specific axes**: none. Memory config is DRAM interleaved only, an argument check.
- **Cores**: 104 writer cores (13 per DRAM bank × 8 banks), in 2 row-wise rectangles of the 11×10 grid. Regime `bank_run`: each core writes one contiguous slot run in one bank, from a 16 KiB core-local zero source.
- **Compute config**: none. The op has no compute kernel: one writer kernel on BRISC/NoC1.
- **Golden baseline**: 13 / 19 cells pass. The other 6 are `xfail_expected`: every bfloat16 and bfloat8_b cell. 0 `supported_fail`, 0 `xpass_drift`, 0 `xfail_wrong_mode`.
- **Device time** (Blackhole, device kernel duration): p1 bf8b 1,102,674 ns (259 GB/s), p2 fp32 48,693 ns, p3 fp32 24,827 ns, p4 bf16 1,603 ns.

### [x] Refinement 1 — Numerical formats: bfloat16 and bfloat8_b

**Goal**: add `ttnn.bfloat16` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`. This moves the six `xfail_expected` golden cells (p4 bf16 ×3, p1 bf8b ×3: plain, model profile, dirty destination) to passing. No `compute_kernel_config` is exposed: the op has no compute kernel, and the spec says no compute configuration exists.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- This refinement is small by design. It is the whole TARGET − SUPPORTED gap: `layout` is already complete (TARGET = [TILE]), and there is nothing else to bundle with it.
- It alone lands the full contract of the perf-flagged case that Refinement 2 optimizes: p1 `(32831, 1, 32, 256)` bfloat8_b, TILE, DRAM interleaved, `math_fidelity=None`, exact gate.
- Kernel reach is already verified. Probe `probes/probe_029.py` ran under `--dev` (watcher, NoC sanitizer, kernel `ASSERT`s), dispatching the current program past `validate()` onto a poisoned, freed range. p4 bf16 and p1 bf8b both came out exact: 0 wrong bits out of 131,072 and 0 out of 268,951,552 elements. The skill's "zero kernel changes" pass condition holds already:
  - `aligned_page_size` comes from `ttnn.tile_size(dtype)` and is cross-checked against the buffer.
  - The CB format tag follows `output.dtype`.
  - 1088 B bfp8 pages tile the 16 KiB chunks correctly.
- The spec's `## Rules` forbid an EXCLUSIONS entry, a size tagger or a shape axis that refuses a production case. bfloat8_b has no non-aligned cell in this suite, so the skill's usual bfloat8_b EXCLUSIONS route does not arise. Do not add one.

**Done when**: the golden run shows 19/19 `supported_pass`, 0 `xfail_expected`, and the loud categories at 0. The acceptance suite's p4/p1 cases pass, including both dirty-destination cases. The precision baseline's p4/p1 cases pass exactly: max ULP 0, no -0.0.

### [ ] Refinement 2 — Speed up the perf-flagged p1 profile

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` flags p1 `(32831, 1, 32, 256)` bfloat8_b, TILE, DRAM interleaved (262,648 tiles, 285,761,024 B written) as the mandatory perf target. Its goal is device kernel time < 1,374.51 µs and ≤ 1.35 × 620.14 µs = **837.19 µs**, exact gate. Phase 0 runs it at **1,102,674 ns (259 GB/s, 51 % of the 512 GB/s DRAM peak)** and misses the target by 1.32×.

The region is the write stream's core↔bank geometry. Phase 0 puts 13 writers on each bank, spread row-wise over the whole grid, so 13 NoC1 paths from every row converge on each bank endpoint. Use the relevant levers in `ttnn/ttnn/operations/examples/master.md`:
- `noc_placement`: core placement against the DRAM-facing axis.
- `read_coalescing`: its bank-affinity half, the write-side mirror.
- the design's own grid-synchronization and core↔bank placement lamps.

Both levers are knob-turns on the surface the planner exposed: the host plan's `writers_per_bank`, plus a core-placement choice in `plan_bank_runs`. The block schedule, CB and kernel need no change. Measured with the Phase 0 kernel, dispatched past `validate()`, last of 2 repeats:

| plan | p1 bf8b | p2 fp32 | p3 fp32 | p4 bf16 |
|------|---------|---------|---------|---------|
| Phase 0: 13 per bank, row-wise (104 cores) | 1,102,674 | 48,693 | 24,827 | 1,603 |
| 1 per bank, row-wise (8) | 902,602 | 40,736 | 20,935 | 1,890 |
| 2 per bank, row-wise (16) | 960,713 | 43,375 | 22,421 | 2,184 |
| 4 per bank, row-wise (32) | 948,897 | 42,195 | 21,971 | 1,796 |
| 8 per bank, row-wise (64) | 1,013,425 | 43,982 | 22,633 | 1,644 |
| **1 per bank, on its NoC1-optimal worker** (8 single-core ranges) | **559,640 (510.6 GB/s)** | **25,807** | **13,471** | 1,775 |

(ns; `test_g6_zeros_perf.py` and `probes/perf_bank_affine_probe.py`.) No SUPPORTED change.

**Verifier notes**:
- Depends on Refinement 1. p1 is bfloat8_b; measure and optimize exactly that config, never a float32 stand-in.
- The naive turn regresses p4 by 13 % (1,573 → 1,775 ns, Phase 0 vs bank-affine within one probe run): one writer per bank serializes its 16 slots, where Phase 0 spreads them 1–2 per core. p4 is 128 tiles, so `perf_rules.txt` measures it and it is in the guard set. The plan may choose `writers_per_bank` and placement by shape (one descriptor per regime, one dispatch per call). If it does, `describe_plan` must report the placement and any regime chosen (BB-08), and the knobs stay single-sourced in the plan-cache key.
- BB-17 asks for few rectangular core ranges per kernel group. The bank-affine set is 8 single-core ranges. The alternative is one rectangle with idle cores returning early, which launches more cores. Measure the launch cost on p4 rather than assume it.
- On a mesh, one program runs on every chip, so the placement anchors come from one chip's bank→worker map. That stays correct on a differently harvested chip, only possibly farther from its banks. Keep allocation and dispatch SPMD.
- After this lands, p1 at about 510 GB/s is at the DRAM bound (below the lead's 620.14 µs roofline). No further perf refinement is filed; see `verification_report.md` → Recommendations.

**Done when**:
- Measured device kernel time on p1 bfloat8_b is ≤ 837,190 ns and the exact gate holds.
- The golden suite is green.
- No regression beyond noise (~2–3 %, `perf_rules.txt`) across the guard set: p2 fp32, p3 fp32, p4 bf16. That is one representative per page size of the single kernel path, TILE, DRAM interleaved.
