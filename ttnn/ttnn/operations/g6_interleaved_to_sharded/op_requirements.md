# Operation Requirements: g6_interleaved_to_sharded

## Definition
- **Formula**: `y[i] = x[i]` for every logical index `i`. `shape(y) = shape(x)`, `dtype(y) = dtype(x)`, `layout(y) = TILE`. `placement(y) = memory_config`: L1, height or width sharded, with the given grid, shard shape and orientation.
  - Height sharded: folded tile row `r` lives on core `G[r div sh]`.
  - Width sharded: tile column `c` lives on core `G[c div sw]`.
  - `G` = shard-grid cores, row by row (ROW_MAJOR) or column by column (COL_MAJOR).
  - Bit-exact, NaN payloads included.
- **PyTorch Reference**:
  ```python
  def pytorch_g6_interleaved_to_sharded(input_tensor, memory_config=None):
      return input_tensor.clone()  # memory_config chooses where y lives, never what it holds
  ```
- **Import Path**: `from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded`
- **Function Signature**: `g6_interleaved_to_sharded(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor`.
  - `input_tensor`: `(..., H, W)`, rank 2 or 4, bfloat16, TILE, DRAM interleaved.
  - `memory_config`: L1 HEIGHT_SHARDED or WIDTH_SHARDED, with a shard spec.
  - Both are positional. There is no dtype, layout or compute-config parameter.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: rank ∈ {2}, alignment = h_non_aligned only
- **SUPPORTED op-specific axes**: out_scheme ∈ {width_sharded}
- **Cores**: owner-fetch on the shard grid, one owner core per shard (p1: 32 cores of 8x4), with an NCRISC/NOC0 issuer.
- **Compute config**: none. The op has no compute kernel and does no arithmetic.
- **Golden baseline**: 4 supported_pass, 2 xfail_expected, 0 loud categories. Regression tests: 5/7 pass, 1 is xfail-strict (Refinement 1), and 1 fails on an L1-profiling harness defect (see `verification_report.md`).
- **Perf (FLUX #58 p1)**: 1101 ns profiled device kernel duration (5-call median); 1106 ns in the golden harness. Stock ttnn: 1.43 µs.

Queue shape: TARGET − SUPPORTED is `rank: 4` and `out_scheme: height_sharded`. Both are knob-turns with the same production witness (p2), so they form one generality bundle. The perf-1 target (`flux58_p1`) is fully inside Phase 0 SUPPORTED, so no unlock precedes it. With a single generality bundle, the 2:1 cadence gives R1 generality, then R2 perf. No generality candidates remain after R1.

### [x] Refinement 1 — Height-sharded placement and rank-4 inputs

**Goal**: add `"height_sharded"` to `SUPPORTED["out_scheme"]` and `4` to `SUPPORTED["rank"]`. These cells move from `xfail_expected` to `supported_pass`:
- golden `test_op[1x1x1x128-height_sharded…]` (FLUX case 57, p2: `(1, 1, 1, 128)` into one `(32, 128)` height shard on core (0,0));
- loose `flux57_p2`;
- regression `test_height_multi_shard[2x1x33x128-height-4cores]`;
- acceptance `flux57_height_1core` and `height_multi_shard_2x1x33x128`.

The kernel and the per-tile-row span predicate stay the same:
- Height shard `s` is folded tile rows `[s*shard_tile_rows, min((s+1)*shard_tile_rows, tensor_tile_rows))` on `G[s]`, all tile columns.
- Rank 4 folds the leading dims into tile rows with per-image padding: `image_tile_rows = ceil(H/32)`, and the partial row is found by `r mod image_tile_rows`.

**Verifier notes**:
- **Knob-turn, not a scheme change** (op_design.md Blocking Model, "TARGET axes seen as placements"). The host rectangle in `owner_block()` and the rank folding in `compute_geometry()` are already built.
- **Pre-check.** The gate-lifted probe (`probes/probe_010.py`, re-run by the verifier) is bit-exact with NaN-poisoned padding on:
  - p2;
  - `(2,1,33,128)` on 4 cores;
  - `(2,3,33,128)` on a 4x2 COL_MAJOR grid;
  - a partial last height shard, `(3,1,1,256)`;
  - rank-4 width `(1,1,1,5120)` and `(2,1,33,256)`.

  The expected change is SUPPORTED plus confirming those cases through the real gate. Do not invent kernel work, and do not narrow anything to make the cells pass.
- **MUST rule that re-arms here.** Tile row `r` → `G[r div sh]` for any number of shards and images. A map that never leaves core 0 passes p2, so keep the multi-shard, multi-image and col-major checks above in the acceptance run, not just p2.
- **Malformed vs support order is already right.** A height shard whose width ≠ the tensor's tile width (narrower or wider) is a `ValueError` in `validate()` before the support check, pinned by `test_g6_interleaved_to_sharded_validation.py`.
- **Performance bar.** Hold the new path to the same bar as Phase 0: one block = the whole shard per issuer, every read in flight, one barrier. p2 is 4 tiles, below the perf rules' 32-tile measurement floor, so it only has to pass golden.
- **Knobs.** The `issuers_per_core` and `partial_tile_read` knobs must stay bit-exact on the height path. Extend `test_g6_interleaved_to_sharded_knobs.py` with one height scenario (the 4-core `(2,1,33,128)` case).
- **No implementation-skill pointer.** Sharding placement has no dedicated skill: `/memory-layouts` covers RM↔TILE, and `/interleaved-parallel` excludes sharded outputs.

**Done when**:
- `SUPPORTED` holds `rank: [2, 4]` and `out_scheme: ["width_sharded", "height_sharded"]`.
- The golden suite shows 6 supported_pass and 0 xfail_expected / xpass_drift / xfail_wrong_mode / supported_fail.
- Regression `test_height_multi_shard` passes.
- The acceptance file passes with no xfails.
- The knob tests pass on the added height scenario.

### [ ] Refinement 2 — Speed up the perf-flagged FLUX #58 profile

**Type**: perf

**Goal**: speed up `flux58_p1`, which `feature_spec.LOOSE_CASES` flags as the mandatory perf target.
- **Target cell.** `(1, 5120)` bfloat16 TILE, DRAM interleaved → L1 width shards `(32, 160)` on (0,0)-(7,3), ROW_MAJOR. That is 160 tiles over 32 owner cores. Its precision gate is `"exact"`.
- **Its `extras` goals.**
  - Below stock 1.43 µs: met at 1101 ns.
  - ≤ 1.35 × 0.02 µs: a pure DRAM-bandwidth bound, unreachable for any one-program implementation (the read-ablated floor is 548 ns).

The verifier's per-core zone breakdown of the default (`verification_report.md` § Performance):

| Component | Time | Controllable? |
|---|---|---|
| Kernel-start skew across cores | ~450 ns | No: go-signal arrival, identical with reads ablated |
| `I2S-FETCH`: NCRISC issuing 5 one-packet reads | ~233 ns (~44 ns/read; 13 ns with reads ablated) | **Yes: this region** |
| `I2S-RETIRE`: DRAM round trip after the last issue | ~326 ns | No: irreducible |

Shrink the issue region using the relevant entries of `ttnn/ttnn/operations/examples/master.md`:
- `split_reader`: the knob already exists. `issuers_per_core = 2` measured 1044 ns against 1101 ns (−5.2%), yet the default stayed at 1 per RB-053. Decide the default from the measurements in the Done-when.
- **Per-read issue cost**: about 44 ns per read is far above a one-packet command's register writes. Candidates:
  - program the size/VC once with `noc_async_read_one_packet_set_state` and issue each tile with `noc_async_read_one_packet_with_state`;
  - walk the bank/offset incrementally instead of calling `TensorAccessor::get_noc_addr` per page;
  - specialize the span size per tile row at compile time.
- `noc_placement`: the second issuer's NoC/RISC assignment and the split weights. The implementer measured NOC1 reads draining slower than NOC0 on these cores; a single reader on NOC1 ran at 1309 ns.
- `bank_stagger` (design lamp L3), as a cheap check. Likely a no-op on p1 (5 tiles per core, coprime with the 8 banks). It matters on the `(32, 640)` guard placement, where starts alternate banks 0/4.

No SUPPORTED change.

**Verifier notes**:
- These are T2 levers on the issue loop the planner exposed (core-internal issuer split + transfer issue path): one phase. The knob table and bench are in `l1_ledger.md` and `test_g6_interleaved_to_sharded_perf.py`. The per-zone breakdown comes from the raw profile log (`generated/profiler/.logs/profile_log_device.csv`, zones `I2S-FETCH` / `I2S-RETIRE`).
- **Realistic ceiling**: ~150–200 ns of the 1101, i.e. the fetch region above its ablated cost. Do not chase the go-signal skew or the DRAM round trip.
- **RB-053**: a lever that changes which RISCs run (2 issuers) can look better profiled than it is. Check it on unprofiled trace replay as well. `probes/probe_011.py` shows trace capture/replay working bit-exact; reuse it.
- **Independent of Refinement 1.** If Refinement 1 has landed, the height path joins the correctness side of the guard set. Its production cell (4 tiles) is below the 32-tile measurement floor and is not timed.
- **Last perf phase.** No further perf phase is queued. After this one the remainder is go-signal skew + firmware launch + one DRAM round trip, which no owner-fetch kernel change removes (see `verification_report.md` § Recommendations).

**Done when**:
- The profiled device kernel duration of `flux58_p1` (5-call median, `test_g6_interleaved_to_sharded_perf.py` default variant) improves beyond the 2–3% noise band.
- Unprofiled trace-replay time per call is not worse than Phase 0 within noise.
- The bit-exact gate holds, the golden suite is green, and the knob tests are bit-exact.
- No regression across the config-spanning guard set:
  - p1 ROW_MAJOR;
  - p1 COL_MAJOR;
  - the `(32, 640)` 8-core placement;
  - the `(33, 5120)` regime pin (full + partial tile rows in one block);
  - the height path for correctness, once Refinement 1 is in.
