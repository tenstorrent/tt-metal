# Operation Requirements: g6_zeros

## Definition
- **Formula**: `out[i0, i1, i2, i3] = +0.0` for every logical index, stored as the all-zero bit pattern of `dtype` (never -0.0; for bfloat8_b every datum's sign and magnitude bits are zero). `out` is newly allocated on every call, on every device of `device`.
- **PyTorch Reference**: `torch.zeros(tuple(shape), dtype=torch_dtype)`. bfloat8_b uses `torch.bfloat16`, whose zeros read back from bfloat8_b as exact zeros.
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
- **SUPPORTED shape-derived axes**: none. The spec forbids size, shape and rank axes. Rank 4 with `d_2, d_3 % 32 == 0` is an argument check (`ValueError`), not a support axis.
- **SUPPORTED op-specific axes**: none. DRAM interleaved is an argument check.
- **Cores**: 104 = `B · cores_per_bank` = 8 DRAM banks × 13, the first 104 cores of the 11×10 grid in row-major order (2 rectangular ranges). Regime R1, bank-affine coalesced runs: core `k` writes one contiguous run of bank-rows in bank `k % 8`, in ≤16 KiB `noc_async_write`s from one core-local zero source.
- **Compute config**: none. There is no compute kernel, only one writer kernel on BRISC/NoC1.
- **Golden baseline**: 13 / 19 cells pass. The other 6 are `xfail_expected`: every bfloat16 and bfloat8_b cell (p4 ×3, p1 ×3). 0 supported_fail, 0 xpass_drift, 0 xfail_wrong_mode.
- **Device kernel time** (Blackhole, default plan): p1 1,092,860 ns, p2 47,114 ns, p3 24,271 ns, p4 1,807 ns.

### [x] Refinement 1 — Numerical formats: bfloat16 and bfloat8_b

**Goal**: add `ttnn.bfloat16` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`. This moves the six `xfail_expected` golden cells to passing: p4 `(1,4,256,128)` bf16 and p1 `(32831,1,32,256)` bf8b, each as the plain cell, the model-profile loose case and the dirty-destination loose case. The acceptance and precision-baseline xfails for these dtypes must also pass. Do not expose a `compute_kernel_config`: the op has no compute kernel, and the spec says no compute configuration exists.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- This one refinement is the entire `TARGET − SUPPORTED` gap. `layout` is already complete (TARGET = [TILE]), and TARGET has no other axis, so there is nothing to bundle with it. It is small because the op is small, not because work was split off.
- **Anchor**: it lands the full contract of the perf-flagged case that Refinement 2 optimizes: p1 bf8b, TILE, DRAM interleaved, `math_fidelity=None`, exact gate. After this refinement p1 must be in SUPPORTED, not xfail.
- Kernel readiness is already verified. Probe `probes/probe_031.py` dispatched the current descriptor past `validate()` onto a poisoned, freed range: 0 non-zero bits out of 131,072 for p4 bf16, and 0 out of 268,951,552 for p1 bf8b. The dtype only enters through `aps` (2048 / 1088 B) and the `burst_pages` derived from it (8 / 15). The expected diff is the SUPPORTED list alone, so the skill's "zero kernel changes" pass condition holds.
- The spec's `## Rules` forbid any EXCLUSIONS entry, size tagger or shape axis that refuses a production case. The skill's usual `bfloat8_b + non_tile_aligned` EXCLUSIONS route does not arise here: every INPUTS shape is tile-aligned. Do not add one.

**Done when**: `SUPPORTED["dtype"] == [float32, bfloat16, bfloat8_b]`, and the golden suite passes 19 / 19 with 0 `xfail_expected` and every loud category at 0. The acceptance suite passes with no xfail, and the precision baseline reports max ULP 0 and no -0.0 on all four production cases.

### [ ] Refinement 2 — Speed up the perf-flagged p1 profile (writer count and bank-near placement)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` flags p1, `(32831, 1, 32, 256)` bfloat8_b TILE DRAM-interleaved, as the mandatory target. That is 262,648 pages, 285,761,024 B written, exact gate. Its goal: device kernel time < 1374.51 µs (stock ttnn) **and ≤ 837.19 µs** (1.35 × the 620.14 µs DRAM roofline); the second limit is the one that binds. Phase 0 takes **1,092,860 ns** (261 GB/s), 1.31× too slow. Optimize this exact config, in bf8b once Refinement 1 lands, never a proxy dtype. No SUPPORTED change.

The op writes to DRAM only, so the levers are write-stream geometry, all knob-turns on the Blocking Model's surface (catalog `ttnn/ttnn/operations/examples/master.md`: `noc_placement`, `bank_stagger`, `read_coalescing`'s bank-affine assignment):
- **`cores_per_bank`** (grid-sync lamp; already a host knob in `make_plan`). The verifier measured a sweep on the current row-major placement. p1 device time by `cores_per_bank`:

  | `cores_per_bank` | 13 (default, 104 cores) | 8 | 4 | 2 | 1 (8 cores) |
  |---|---|---|---|---|---|
  | p1 (ns) | 1,092,860 | 1,047,264 | 942,677 | 971,191 | 906,971 |

  Fewer writers per bank is faster: concurrent writers into one bank contend. Even 1 writer per bank (907 µs) misses the target on placement alone.
- **Core placement** (Blocking Model lamp "core placement"; today `G6ZerosPlan.core_coord` is a fixed row-major `k → (k % gx, k // gx)`). Pick each bank's writer core(s) near that bank's DRAM endpoint, so the NoC1 write path is short and the 8 streams do not share links. Expose placement as a plan knob, with one source of truth for which core serves which bank, and keep the core groups as few rectangular ranges (BB-17).
- Co-tune with `burst_pages` (16 KiB cap) only if the two levers above leave headroom.

**Verifier notes**: after Refinement 1, because p1 is bf8b. The other three cases also got faster at `cores_per_bank = 1` in the same sweep (p2 47.1→40.6 µs, p3 24.3→20.8 µs, p4 1.81→1.76 µs). So if the per-shape choice differs, it must be a plan the op can report (`g6_zeros_plan`, BB-08), keyed in the descriptor cache (BB-07). It must not be a silent fall to a slower plan. Fold one ledger-level change in here: the ledger's "block-size default: full grid" row must record the measured departure. `tests/ttnn/unit_tests/operations/g6_zeros/test_g6_zeros_perf.py` is the sweep harness; extend it with the placement knob.

**Done when**: measured device kernel time on p1 is ≤ 837.19 µs, the exact gate still holds, and the golden suite is green (19 / 19). There is no device-time regression on the guard set: p2 and p3 fp32 (same kernel, 4096 B pages), and p4 bf16 (128 pages, per-call overhead).
