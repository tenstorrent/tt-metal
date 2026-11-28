# Operation Requirements: rms_norm_head_rope

## Definition
- **Formula**: for each row `r`, head `h` (head_dim 128): `ms = (1/128)·Σ_d x[r,h·128+d]²`;
  `y = x·rsqrt(ms + ε)`; `y *= w[h·128+d]` (if weight); if rope: `out = y·cos[r,d] + rot(y)[d]·sin[r,d]`,
  `rot(y)[2i] = −y[2i+1]`, `rot(y)[2i+1] = y[2i]`; else `out = y`.
- **PyTorch Reference**: `eval/golden_tests/rms_norm_head_rope/helpers.py::pytorch_rms_norm_head_rope`
- **Import Path**: `from ttnn.operations.rms_norm_head_rope import rms_norm_head_rope`
- **Function Signature**:
  `rms_norm_head_rope(input_tensor, weight=None, *, rope_cos=None, rope_sin=None, transformation_mat=None,
  epsilon=1e-6, compute_kernel_config=None) -> ttnn.Tensor`
  (input (1,1,R,W) bf16 TILE DRAM-interleaved, R%32==0, W%128==0; weight (1,1,1,W); cos/sin (1,1,R,128);
  transformation_mat (1,1,32,32) validated and ignored; default config HiFi4 + fp32 DEST, no approx).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **Queue shape**: `TARGET − SUPPORTED` is empty (SUPPORTED ⊇ TARGET on every axis), so there are no generality
> refinements and the 2:1 cadence degenerates to an all-perf queue. No `LOOSE_CASES` entry is perf-flagged, so
> regions are verifier-selected among cells > 32 tiles (`eval/prompts/perf_rules.txt`). Ordered by step-time
> impact: 2304×3072 runs 96×/step (~11.4 ms of ~13.5 ms), 2048×3072 16× (~1.8 ms), 256×3072 16× (~0.3 ms).
>
> **Shared guard set (every perf refinement)**: one representative per distinct kernel path —
> {plain, weight-only, rope-only, weight+rope} × {fp32 DEST, bf16 DEST} at 2304×3072 (single layout/placement:
> TILE, DRAM interleaved), plus the 256×3072 and 2048×3072 weight+rope cells. Precision gate: the precision
> baseline must still pass **and** its weight+rope fp32-DEST ratio median must stay within 0.999 ± 0.002 (the
> srcA = bf16 / srcB = DEST-width operand placement rule — see the compute kernel header).
>
> **Perf catalog**: `ttnn/ttnn/operations/examples/master.md` is absent on this branch; levers below are named by
> the catalog entries `op_design.md` cites. The implementer owns the exact lever.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment = tile_aligned (R%32, W%128), rank = 4
- **SUPPORTED op-specific axes**: affine ∈ {no_affine, weight} (bf16 TILE), rope ∈ {no_rope, rope_bcast} (bf16),
  memory_layout = INTERLEAVED (DRAM), fp32_dest_acc_en ∈ {True, False}
- **Cores**: full grid — 2-D (row-group × head-group) split from `select_grid_split` (120/120 cores at R ≥ 2048)
- **Compute config**: exposed (`compute_kernel_config`, default HiFi4 + fp32 DEST)
- **Golden baseline**: 32 / 32 supported cells passing; 224 INVALID skipped; 0 loud (per verifier CLI)
- **Perf (weight+rope, default config)**: 256×3072 20.0 µs · 2048×3072 ~115 µs · 2304×3072 ~119 µs
  (roofline 8.3 / 58.1 / 65.2 µs)

### [ ] Refinement 1 — DRAM streaming efficiency on the FLUX joint/img shapes

**Type**: perf

**Goal**: speed up 2304×3072 (primary) and 2048×3072, weight+rope, default config (HiFi4, fp32 DEST, bf16 TILE,
DRAM interleaved). Phase 0 attribution at 2304: DM-only 89 µs with reads ~290 GB/s and writes ~250 GB/s isolated
(vs ~460 GB/s roofline), compute-only 76 µs, full 120 µs. The dataflow is the binding resource and overlaps poorly
with compute. Each core runs only 1–2 blocks of 32 tiles, so the first load and the last store are exposed, and
every block drains its reads at a barrier before the next block's reads issue. Candidate levers, all knob-turns on
the surface the planner exposed (catalog: `noc_placement`, `double_buffer`, `compute_block_size`):
- Keep reads in flight across block boundaries: push per sub-block or chunk, with no full drain per block.
- Co-tune the `INPUT_DEPTH`/`OUTPUT_DEPTH` × `block_heads` knobs.
- Core enumeration / NoC assignment for DRAM-bank locality.
- Sweep the grid-split objective (`G ∈ {1,3,5,8}` at R=2304, per the design's "Grid-split objective" lamp).
- **Re-expose the collapsed `BLOCK_ROW_TILES` knob.** It is a host constant asserted `== 1` that the kernels never
  receive. Coarsening along `r` needs cos/sin addressed per row inside the block (Strided, or a per-row-tile chain
  call). It is a candidate lever here because every core owns ≤ 2 row-tiles at these shapes, and it stops the knob
  being a constant.

No SUPPORTED change.

**Verifier notes**: this is first because the dominant shape is DM-bound before it is compute-bound (89 vs 76 µs).
Refinement 2's compute savings only show once the dataflow stops being the floor. The `BLOCK_ROW_TILES`
re-exposure is a design-conformance item from the verifier review (a collapsed knob). Carry it out even if it
measures flat, as a CT arg with a single source of truth. If it measures flat, leave the default at 1.
Ledger: any depth or block change must update `l1_ledger.md` (the worst case is 996 KB today).

**Done when**: measured device-ns improves on 2304×3072 (and does not regress 2048×3072) beyond the 2–3% noise
floor, the golden suite is green, acceptance and tail-block tests pass, the precision gate above holds, there is no
regression across the shared guard set, and `BLOCK_ROW_TILES` is a kernel CT arg rather than a host-only assert.

### [ ] Refinement 2 — Cut the per-tile compute cost of the weight + RoPE path

**Type**: perf

**Goal**: speed up 2304×3072 weight+rope (default config) by reducing the FPU/SFPU work per output tile. Phase 0
compute-only ablation at 2304:
- plain 24 µs
- +weight 40 µs
- +rope 59 µs
- weight+rope 76 µs

So the weight DEST-reuse pass costs ~16 µs, and the rope path (separate matmul pass into `cb_rotated`, two FPU
multiplies, SFPU add) costs ~35 µs. Candidate levers from the design's perf lamps (catalog: `compute_fusion`):
- **RoPE reformulation:** `D0 = cos·y`, then matmul-accumulate `(sin_p∘y) @ T` into D0, with `sin_p` computed once
  per row-tile. This removes the SFPU add and the `cb_rotated` pack/unpack round-trip.
- Restructuring the weight multiply, e.g. the "normalize fusion" lamp alternative, measured both ways.
- Fewer element inits per DEST block.

No SUPPORTED change.

**Verifier notes**: it must follow Refinement 1, because at Phase 0 the dataflow hides compute wins (the
implementer's chain-blocking compute win of 2.6 µs did not show end-to-end). Precision constraints:
- Any reformulation must keep bf16 operands on srcA and DEST-width operands on srcB. Without this the weight+rope
  output had a −0.36% bias; the verifier fixed it.
- bf16 pages for `cb_normed`/`cb_rotated` under fp32 DEST are **not** an allowed lever (L1 ledger audit 3,
  under-width page).
- Changing the rstd-first application order is allowed only if the precision baseline holds.

**Done when**: measured device-ns improves on 2304×3072 weight+rope beyond the noise floor, the precision gate
holds, the golden suite is green, acceptance and tail-block tests pass, and there is no regression across the
shared guard set (the plain / weight-only / rope-only paths must not slow down).

### [ ] Refinement 3 — Read cos/sin once per row-group and multicast (regime R2)

**Type**: perf

**Goal**: build the design's deferred **R2** regime. cos/sin are reuse-shared across the `G` head-group cores of a
row-group, yet each of those cores re-reads them from DRAM. Instead, the `g = 0` core reads the 8 cos/sin tiles per
row-tile and multicasts them into the same CB slots on the other `G − 1` cores. Those cores are already enumerated
adjacently (`k = rg·G + g`), so the receiver set is a contiguous run. Use the multicast helper: host `ttnn.Mcast` +
`attach()`, kernel `McastArgs` `sender(noc).send()` / `receiver(noc).receive()` (`mcast/kernel/mcast_args.hpp`).
Compute and writer are untouched.

Primary target: **256×3072** weight+rope (G=12). There, cos/sin re-reads are ~1.6 MB of ~4.7 MB total DRAM traffic,
and the cell runs 20 µs vs an 8.3 µs roofline. Secondary: 2304×3072 (G=3, 2.4 MB of 32.4 MB). Catalog:
`shared_input_reuse`, `mcast_topology`. No SUPPORTED change.

**Verifier notes**: this is a scheme-change (new injector + broadcast topology, semaphore per row-tile), so it
stands alone. The design deferred it on catalog thresholds: a 0.25× window, and receiver lines of ≤ 4 measured as
losses. The verifier rules require a deferred broadcast of a reuse-shared operand to be **built and measured on
device**, not ruled out by inference. It may lose at G=3. In that case keep it predicate-gated on `G` (e.g.
enabled only where it measures a win) as an explicit regime row in `op_design.md`/`l1_ledger.md`, or record the
measured loss and drop it. Also evaluate the design's "launch floor" lamp at 256×3072 here: capping occupancy
through the selection objective interacts with `G`.

**Done when**: R2 is built. Device-ns is measured on 256×3072 and 2304×3072 against the R1 path. The variant
ships only where it is a device-ns win beyond noise, with no regression on the shared guard set. The golden suite
is green, acceptance and tail-block tests pass, and the precision baseline is unchanged (data movement only).
The regime table in `op_design.md` and the traffic budget in `l1_ledger.md` record the measured outcome.
