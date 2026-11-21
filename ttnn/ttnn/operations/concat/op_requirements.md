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

### [x] Refinement 20 — `memory_config`, `sub_core_grids`, four and nine tensors (Llama, Qwen)

**Goal**: what Llama's and Qwen's calls need, at the dtype, layouts and placements supported by now.
1. The signature gained two keyword-only parameters (the stub in `eval/specs/concat/spec.py`): `memory_config=None` (`CALL["canonical"]`, axis `output_memory`: `None` for no argument or the DRAM interleaved config, `"L1"` for the L1 interleaved config; rule R5) and `sub_core_grids=None` (None is its one accepted value). Add `output_memory` to SUPPORTED with `None` and `"L1"`: the result is allocated in the buffer the config names.
2. Add `4` and `9` to SUPPORTED["count"].

**Origin**: the model-level tests of the Llama and Qwen jobs. Llama joins nine tile-layout pieces along W (eight `[1, 1, 32, 512]` and one `[1, 1, 32, 256]`, DRAM) with `memory_config=ttnn.L1_MEMORY_CONFIG` and gets `[1, 1, 32, 4352]` in L1. Qwen joins four row-major `[1, 1, 32, 37984]` pieces along `dim=3` with `sub_core_grids=None` and gets `[1, 1, 32, 151936]` in DRAM. Both joins are INPUTS entries now (the block "Llama and Qwen"). The op as it is refuses both calls: a TypeError for the keyword.

**Operator notes** (queued by the operator of the run, not by the verifier):
- Refinement 2 has `count` 1 and 5 and the leading dimensions; this entry takes 4 and 9 along the last two dimensions only.
- Qwen's join is large: 4.9 M elements, 9.7 MB of bfloat16 out. Check the L1 ledger's bounds on a join of that size, in row-major and in tile layout, and in the L1 and height-sharded cells the suite makes of it (the runner skips a sharded cell whose input shards do not fit one core's L1).
- A sharded `memory_config` reads as a value no cell has (`"L1/height"`) and must be refused by name.

**Done when**: the entry point's signature equals the stub; every golden cell with `count` 4 or 9 or `output_memory=L1` inside the new SUPPORTED passes exactly, the two model entries among them; earlier cells unchanged; `verify_supported` loud categories are 0; a test in the op's own test file makes the two models' calls and asserts where each result is (L1 interleaved; DRAM interleaved).

**Result** (implementer): signature = the stub (`*, memory_config=None, sub_core_grids=None`); SUPPORTED gains `output_memory` [None, "L1"] (read through the specification's `canonical_memory`; a sharded config is refused by name as `output_memory='L1/height'`, a non-None `sub_core_grids` by name) and `count` 4, 9. The output is allocated in the config's buffer (DRAM interleaved without one); the reader binds `in_0 … in_8` (`MAX_INPUTS` 5 → 9, the host constant asserted equal to the kernel's list). No buffer, regime or split change. Golden slice: 196/198 in-SUPPORTED cells with `count` 4/9 or `output_memory=L1` exact, 2 skipped by the runner (Qwen TILE height-sharded: four 2.4 MB shards on one core do not fit), both model entries green; 8/8 prior representatives; unit dir 182/182.

### [x] Refinement 1 — ROW_MAJOR layout (native stick kernels)

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]` for every tensor of the list (they share one layout) with a native row-major path in the kernels — the design's deferred `row_major_runs` regime: page = stick. `dim=-1`: each output stick is one sub-page run per piece (byte runs at element offsets; Quasar's NoC has 1-B transfer alignment). `dim=-2`: each output stick is one whole input stick (the `page_copy` shape in the stick domain). Output is ROW_MAJOR, DRAM interleaved (R5). Covers the PHASE0 rectangle in RM: bfloat16, dim ∈ {-1, -2}, count ∈ {2, 3}, rank ∈ {3, 4}, every alignment, DRAM and L1 inputs.

**Implementation skill**: /memory-layouts, /interleaved-parallel

**Verifier notes**:
- First because it is the hardest axis left (difficulty tier 3, layouts) and Refinements 2 and 4 extend both layouts: landing the stick path first means each later axis is built once against the final kernel structure.
- Keep Phase 0's structure: one ring, flat output-page axis split over the **whole grid** (now output sticks; the `out_tile` block knob becomes "sticks per block", still the single `BLOCK_TILES`-style host constant feeding ring capacity and loop bounds), implicit sync wherever an entry is one whole transfer, explicit + tagged where an entry is assembled from runs. Do not fall back to one core or to one stick per barrier.
- Size ring entries from the output stick bytes (or a block of sticks per entry), never from a whole-op dimension; a wide stick (`[1,32,4096]×2` → 16 KiB) must stay under the 1,048,560 B ring cap through the existing clamp.
- No `ttnn.to_layout` / `ttnn.tilize` / `ttnn.untilize` wrapper: one native dispatch per call (task integrity rule). There is no partial-tick escape hatch for this refinement.
- The INVALID entry `bfloat8_b × ROW_MAJOR` already covers the impossible cell.

**Done when**: every golden cell with `layout=ROW_MAJOR` inside the new SUPPORTED passes exactly; regime-pinned RM tests (dim -1 with a non-16-aligned seam, dim -2) assert the host's regime choice; the session test covers the RM path; all TILE cells still pass.

**Result** (implementer): SUPPORTED["layout"] = [TILE, ROW_MAJOR]. Regimes `row_major_copy` (dim -2, implicit whole-stick reads) and `row_major_runs` (dim -1, `count` tagged sub-page reads per stick, one barrier per block of sticks) in the existing reader; writer and ring unchanged. Golden slice: 58/58 in-SUPPORTED ROW_MAJOR cells exact, TILE representatives green; unit dir 147/147.

### [x] Refinement 10 — Height-sharded tensors in L1 (the YOLOv8 neck join)

**Goal**: add `"height"` to SUPPORTED["sharding"]: every tensor of the list height-sharded in L1, each with a shard shape and cores of its own, in both layouts the op supports by now (TILE and, after Refinement 1, ROW_MAJOR), at the supported values of every other axis. The specification's TARGET gained this value after Phase 0 (rules R3, R5 and R6, the INVALID entry DRAM x height, the INPUTS entry `(-1, [1, 1, 64, 64], [1, 1, 64, 32])`, the DECISIONS entry of 2026-11-21). The result stays in DRAM, interleaved (R5).

**Origin**: the model-level test. YOLOv8's neck (`models/experimental/ops/quasar/tests/yolo_ops/test_concat_small_grid.py::test_concat_neck_sharded_small_grid`) calls `concat([a, b], dim=-1)` on a `[1, 1, 64, 64]` and a `[1, 1, 64, 32]` bfloat16 ROW_MAJOR tensor, height-sharded in L1 over two cores (shard shapes [32, 64] and [32, 32]) and over one (shard shapes [64, 64] and [64, 32]), and gets a `[1, 1, 64, 96]` tensor in DRAM, interleaved. The Phase 0 op refuses both calls by name, and tt-metal's op serves them. This refinement comes after Refinement 1 because those calls are row-major.

**Operator notes** (queued by the operator of the run, not by the verifier):
- A sibling op gives a hint, not a proof: the generated `add`, which reads and writes pages through `TensorAccessor`, served height-sharded operands once its gate let them through, with no change to its kernels. Check whether the same holds for this op's readers in each layout before changing anything, and say in the changelog what you found.
- Not looked at by anyone yet: tensors of one call whose shard grids differ (the runner gives each tensor the cores its own shape divides over, `eval/CONVENTIONS.md` §2.3, `eval/memory.py`; a join along `dim=-2` has tensors with different numbers of rows), a shard grid of one core, and the op in one device session with tt-metal programs before and after it on sharded tensors.
- The op must not depend on how many cores a shard grid spans. A cell whose input shards do not fit one core's L1 is skipped by the runner.
- Keep the data path: reading each shard on the core that holds it is a performance refinement of its own and is not asked for here.

**Done when**: every golden cell with `sharding=height` inside the new SUPPORTED passes exactly; the interleaved cells are unchanged; `verify_supported` loud categories are 0; a test in the op's own test file makes the model's call (row-major, two cores and one) and asserts the values and that the result is in DRAM, interleaved.

**Result** (implementer): SUPPORTED["sharding"] = [interleaved, height]. No kernel or host-program change: the operator's hint held — every input page is reached through its `TensorAccessor`, which maps a logical page to the cluster holding its shard, so all four regimes (`page_copy`, `run_assembly`, `row_major_copy`, `row_major_runs`) serve height-sharded inputs unchanged, bit-exact, with tensors of one call on different shard grids (32/16/4, 8/24/8, 2/1/1 cores), one-core grids and grids of 25 clusters. Golden slice: 58/58 in-SUPPORTED `sharding=height` cells (29 TILE, 29 ROW_MAJOR) exact + 4 interleaved representatives; `verify_supported --queue` gaps 0. The data path is kept (accessor reads, output split over the whole grid); native per-shard reading is left as its own perf refinement, as the entry says.

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
