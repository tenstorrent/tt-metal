# Operation Requirements: concat

## Definition
- **Formula**: `out[..., o_k + i, ...] = tensors[k][..., i, ...]` along `dim`, with `o_k = Σ_{j<k} tensors[j].shape[dim]`. Values are moved, never computed (spec R1).
- **PyTorch Reference**: `torch.cat(tensors, dim=dim)` (spec `reference`: `torch.cat([t.double() for t in tensors], dim=dim)`)
- **Import Path**: `from ttnn.operations.concat import concat`
- **Function Signature**: `concat(tensors: list, dim: int = 0) -> ttnn.Tensor` — output always DRAM interleaved (R5), the tensors' dtype and layout.
- **Specification**: `eval/specs/concat/spec.py` (the only source of requirements).
- **Target / backend**: Quasar on ttsim. No device-ns can be measured here: every perf refinement below is gated on a **structural count** (backend page, *Performance measurement*), and its gain in time is **unconfirmed** until a timed backend or hardware measures it.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Session rule (Quasar)**: every refinement that adds or changes a kernel path keeps `test_programs_share_a_device_session` covering it — the new path once after another program and once before `ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)` in one device session — and every kernel still hands its transaction id back as its last statement.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED memory_config / sharding**: DRAM, L1 / interleaved
- **SUPPORTED shape-derived axes**: dim ∈ {-1, -2}, count ∈ {2, 3}, rank ∈ {3, 4}, alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}
- **Cores**: full grid; flat output-tile axis split in contiguous ranges (`split_work_to_cores(row_wise=True)`)
- **Thread plan**: (R, C, W) = (1, 0, 1), data-movement bound, no compute kernel
- **Regimes**: `page_copy` (implicit-sync whole-page reads) and `run_assembly` (explicit-sync sub-page runs, one barrier per block of `BLOCK_TILES = 8`)
- **Compute config**: none (pure data movement; CONVENTIONS §2.2 exemption)
- **Golden baseline**: 56 / 56 in-SUPPORTED cells passing, 0 supported_fail / xpass_drift / xfail_wrong_mode (`verifier_report.json`)

### [ ] Refinement 1 — ROW_MAJOR layout (native stick kernels)

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]` for every tensor of the list (they share one layout) with a native row-major path in the kernels — the design's deferred `row_major_runs` regime: page = stick. `dim=-1`: each output stick is one sub-page run per piece (byte runs at element offsets; Quasar's NoC has 1-B transfer alignment). `dim=-2`: each output stick is one whole input stick (the `page_copy` shape in the stick domain). Output is ROW_MAJOR, DRAM interleaved (R5). Covers the PHASE0 rectangle in RM: bfloat16, dim ∈ {-1, -2}, count ∈ {2, 3}, rank ∈ {3, 4}, every alignment, DRAM and L1 inputs.

**Implementation skill**: /memory-layouts, /interleaved-parallel

**Verifier notes**:
- First because it is the hardest axis left (difficulty tier 3, layouts) and Refinements 2 and 4 extend both layouts: landing the stick path first means each later axis is built once against the final kernel structure.
- Keep Phase 0's structure: one ring, flat output-page axis split over the **whole grid** (now output sticks; the `out_tile` block knob becomes "sticks per block", still the single `BLOCK_TILES`-style host constant feeding ring capacity and loop bounds), implicit sync wherever an entry is one whole transfer, explicit + tagged where an entry is assembled from runs. Do not fall back to one core or to one stick per barrier.
- Size ring entries from the output stick bytes (or a block of sticks per entry), never from a whole-op dimension; a wide stick (`[1,32,4096]×2` → 16 KiB) must stay under the 1,048,560 B ring cap through the existing clamp.
- No `ttnn.to_layout` / `ttnn.tilize` / `ttnn.untilize` wrapper: one native dispatch per call (task integrity rule). There is no partial-tick escape hatch for this refinement.
- The INVALID entry `bfloat8_b × ROW_MAJOR` already covers the impossible cell.

**Done when**: every golden cell with `layout=ROW_MAJOR` inside the new SUPPORTED passes exactly; regime-pinned RM tests (dim -1 with a non-16-aligned seam, dim -2) assert the host's regime choice; the session test covers the RM path; all TILE cells still pass.

### [ ] Refinement 2 — Leading dimensions, one and five tensors, rank 2

**Goal**: add `-3` and `-4` to `SUPPORTED["dim"]`, `1` and `5` to `SUPPORTED["count"]`, and `2` to `SUPPORTED["rank"]`, for both layouts.
- `SUPPORTED["dim"]` -3, -4: the design's deferred `leading_dim_copy` regime — a join along N or C moves whole pages (TILE: whole padded images, regardless of H/W alignment; RM: whole sticks). Generalise the `page_copy` source map to `(outer, chunk)`: `outer = Π shape[:dim]`, piece `k` contributes a contiguous run of `D_k · pages_per_slice` output pages per outer index. No new kernel structure.
- `SUPPORTED["count"]` 1, 5 and `SUPPORTED["rank"]` 2: the verifier probed them through the Phase 0 kernels with `validate()` bypassed (`probes/probe_002.py`): count 1 (`(1,1,32,64)`, `(4,128,47)`), count 5 (aligned and element-granular seams) and rank 2 (`(32,64)×2`, `(32,17)+(32,47)`, `(17,64)+(17,64)+(30,64)`) are all bit-exact today. They need the SUPPORTED change, tests, and the RM path from Refinement 1 checked on them.

**Verifier notes**:
- Knob-turns on the existing scheme (count is already the CT arg and the array size, `MAX_INPUTS = 5` already bound in the reader) — bundled, no skill covers a source-map change. Depends on Refinement 1 only in that the leading-dim map must serve both layouts.
- rank 2 has no dim -3/-4 (X3 rejects them before the support check); keep the canonicalisation before the gate (R2).

**Done when**: every golden cell with dim ∈ {-3, -4}, count ∈ {1, 5} or rank 2 inside the new SUPPORTED passes exactly in both layouts; a regime-pinned test for `leading_dim_copy` asserts the host's choice; prior cells still pass.

### [ ] Refinement 3 — Fewer, larger reads on element-granular `dim=-1` seams

**Type**: perf

**Goal**: the spec has no perf-flagged `LOOSE_CASES`, so the region is free-selected: `run_assembly` with an element-granular `dim=-1` seam (some `off_k % 16 ≠ 0`). Target shape `(-1, (4,8,32,47), (4,8,32,100), (4,8,32,50))`, bfloat16, TILE, DRAM interleaved (in SUPPORTED since Phase 0). The reader issues one sub-page NoC read per output face-row per segment: **22,560 reads for 224 output tiles = 100.7 reads per tile, 17.9 B per read** (verifier's model of the reader's emit/merge loop; `(1,1,64,17)+(…,47)`: 96 per tile at 21 B). Cut the DRAM-side transactions using the relevant patterns in `ttnn/ttnn/operations/examples/master.md` — `tile_reorder` (whole pages are the fastest transfer shape) points at the design's deferred `staged_shift` regime (whole source pages into an L1 staging ring, ≤ 1 halo page per piece per segment, then the element shift done locally); `split_reader` is the complementary lever. The implementer owns the lever. No SUPPORTED change.

**Done when**:
- the **structural count** "NoC reads that leave the cluster per output tile" on the target shape drops from 100.7 to **≤ 4** (whole or near-whole source pages), with DRAM bytes read ≤ 2× the minimum (`staged_shift`'s halo), read off the op by a test or probe (e.g. a test-only CT flag that has the reader count its remote reads into a host-visible location, checked against a host-side model) and recorded before/after in the changelog;
- the golden suite is green and exact;
- no regression of the same counts on the config-spanning guard set — one representative per regime × layout × placement: `page_copy` TILE DRAM `(-1,(1,1,64,64),(1,1,64,80))` (1.2 reads/tile), `run_assembly` row-granular TILE L1 `(-2,(1,64,144),(1,16,144),(1,4,144))` (1.7 reads/tile), and the RM `dim=-1` path from Refinement 1;
- the entry and the changelog state that the gain in time is unconfirmed on ttsim.

### [ ] Refinement 4 — Numeric dtypes (float32 and integers)

**Goal**: add `ttnn.float32`, `ttnn.int32`, `ttnn.uint32`, `ttnn.uint16`, `ttnn.uint8` to `SUPPORTED["dtype"]`, in both layouts, every value bit-exact (R1: no element passes through another dtype; the R1 check is the int32 cells). The ring and every byte offset derive from the tensor's page bytes: TILE `elem_bytes = tile_bytes / 1024` (1, 2 or 4) in the run generator; RM stick bytes `W · elem_bytes`. uint16 / uint32 are not DFB formats (`validation/resource/dfb.cpp:148`): the ring is DM-only, so omit `data_format_metadata` for them (legal with no compute binding, `dfb.cpp:272-281`) rather than relabel the bytes.

Omitted: SUPPORTED["dtype"] bfloat8_b — Quasar has no bfloat8_b path (target page, *Family facts*): its golden cells skip as `unbuildable on quasar`, and the target page says to queue no refinement for it. Keep refusing it in `validate()`.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Use only the dtype half of the skill: there is no `compute_kernel_config` to expose and no intermediate CB to retag — the op is pure data movement (spec DEFINITION; CONVENTIONS §2.2). The pass condition is the skill's own: no change to the run generator's logic beyond the element width it already derives.
- Last of the generality refinements because it is the cheapest tier (precision) and extends both the TILE and RM paths built by Refinements 1–2.
- 1-byte elements (uint8) make element-granular `dim=-1` runs 1-B granular; Refinement 3's path must carry them too.
- A dtype that cannot be built at all on this part by the golden runner is a skip, not an EXCLUSION; a structural kernel gap on one (dtype, layout) cell may go to `EXCLUSIONS` with the reason.

**Done when**: every golden cell of the five dtypes inside the new SUPPORTED passes with the `exact` metric (NUMERIC_RESOLVED), int32 cells included; the session test runs one 32-bit and one 8-bit call; prior cells still pass.

### [ ] Refinement 5 — Two reader harts and overlapped barriers on `run_assembly`

**Type**: perf

**Goal**: free-selected region: `run_assembly` row-granular joins at YOLOv8's full sizes (spec DECISIONS: the same call sites as the traced ones), target `concat([a, b], dim=1)` with `a: [1, 4, 8400]`, `b: [1, 80, 8400]`, bfloat16, TILE, DRAM and L1 interleaved (in SUPPORTED since Phase 0). 789 output tiles, 7.3 reads per tile (5,775 reads), one explicit-sync reader hart per cluster and one barrier per block of 8 tiles that drains every read in flight before the next block is issued. Levers from `ttnn/ttnn/operations/examples/master.md`: the thread plan (`THREAD_PLAN` reader threads 1 → 2 on the `run_assembly` path, tiles dealt round-robin on the STRIDED ring, R + W ≤ 6) and `trid_double_issue` (two alternating user trids so block i+1's reads stay on the wire across block i's barrier). No SUPPORTED change.

**Done when**:
- the **structural counts** on the target shape move from (reader harts issuing `run_assembly` reads = 1, blocks of reads outstanding when the reader waits at a barrier = 1) to (2, ≥ 2), read off the op by a test — the thread count shown by a test in which a missing thread leaves work undone (target page, *Thread plan*), not by a correct result alone — and recorded before/after in the changelog;
- the golden suite is green and exact, and the session test runs the new plan between tt-metal programs;
- no regression of the counts on the guard set of Refinement 3 (one representative per regime × layout × placement);
- the entry and the changelog state that the gain in time is unconfirmed on ttsim.
