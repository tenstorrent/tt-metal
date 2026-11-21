# Operation Requirements: embedding_gen

## Definition
- **Formula**: `out[b, s, i] = weight[input_tensor[b, ..., s], i]` for `b < B, s < S, i < D` — output `[B, S, D]` for indices `[B, S]` or `[B, 1, 1, S]`, `[S, D]` for indices `[S]`; the table is `[V, D]` or `[1, 1, V, D]`. Pure data movement: every output element is a table element, unchanged (R1). An index ≥ V is not checked (R2); the kernels clamp it to `V-1` before forming an address.
- **PyTorch Reference**: `torch.nn.functional.embedding(input.reshape(B, S), weight.reshape(V, D))` (`reference()` in `eval/specs/embedding_gen/spec.py`, float64).
- **Import Path**: `from ttnn.operations.embedding_gen import embedding_gen`
- **Function Signature**:
  ```python
  def embedding_gen(
      input_tensor: ttnn.Tensor,     # uint32 row numbers: [B, S], [B, 1, 1, S] or [S]
      weight: ttnn.Tensor,           # the table: [V, D] or [1, 1, V, D]
      *,
      padding_idx: int = None,                            # changes no result (R6)
      layout: ttnn.Layout = None,                         # output layout; None: the table's (R9)
      embeddings_type: "ttnn.EmbeddingsType" = GENERIC,   # GENERIC / PADDED / BINARY (R7)
      memory_config: "ttnn.MemoryConfig" = None,          # output buffer; None: the indices' buffer (R12)
  ) -> ttnn.Tensor: ...
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Backend (ttsim, Quasar)**: the simulator has no clock. Every perf refinement below is gated on a **structural count** read off the op by a test or probe (backend page, *Performance measurement*); its gain in time is unconfirmed until `ttsim-perf` or hardware measures it, and no simulator timing is a speedup.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout** (the table's): [TILE, ROW_MAJOR]
- **SUPPORTED placement**: indices `memory_config` ∈ {DRAM}, `sharding` ∈ {interleaved}; table always DRAM interleaved
- **SUPPORTED output**: `output_layout` ∈ {TILE, ROW_MAJOR}; `output_memory` ∈ {None, DRAM}
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, h_non_aligned}; rank ∈ {2, 4}; table_rank ∈ {4}
- **SUPPORTED op-specific axes**: embeddings_type ∈ {generic, padded, binary}
- **Scheme**: gather-direct, one Metal 2.0 `ProgramSpec` dispatch, reader + writer only (no compute kernel). Token groups split over the whole 8×4 grid (`split_work_to_cores(row_wise=True)`), column chunks fill the cores the tokens leave idle; every knob from `plan_blocks()`.
- **Thread plan**: (R, C, W) = (1, 0, 1) — data-movement bound, stated in op_design.md.
- **Compute config**: none (pure data movement; the specification has no compute config).
- **Golden baseline**: 96 / 96 PHASE0 cells pass + 5 / 5 rejections (per verifier CLI); 800 xfail_expected, 448 skipped (bfloat8_b: INVALID / unbuildable on quasar), 0 loud.

### [ ] Refinement 1 — Shape surface: non-aligned width, rank-1 indices, `[V, D]` tables

**Goal**: unlock the remaining shape-derived axes:
- add `"w_non_aligned"` to SUPPORTED["alignment"] (D not a multiple of 32), in-kernel;
- add `1` to SUPPORTED["rank"] (indices `[S]`, output `[S, D]`, R10);
- add `2` to SUPPORTED["table_rank"] (table `[V, D]`, R11).

`rank` and `table_rank` are canonicalization only: `create_program_artifacts` already takes `B = 1` for rank-1 indices and `V, D` from the last two dims, and `output_shape` already returns `[S, D]`; the work is validating the golden cells. `w_non_aligned` is kernel work at the data-access boundary: the last column unit holds `D % 32` valid elements, so a face-row half can be partial or empty. **RM table**: a row page is `D·esize` B, so every read must clip at column `D`. An unclipped read of the last row runs past the end of the table buffer, which R2 forbids. **RM output**: an output page is `D·esize` B, so slab writes must clip at `D`; otherwise they overwrite the next row. **TILE table / TILE output**: the tile padding columns exist, so whole face-row reads and whole-tile writes stay inside the tensors, and the output padding holds anything (R8). Pass the clip (`D` or the last unit's valid bytes) as one host-derived common runtime arg; every other quantity keeps coming from `plan_blocks`.

**Implementation skill**: /memory-layouts

**Verifier notes**: hardest of the generality work (it changes the reader's and writer's transfer shapes), so it goes first; Refinement 2's float32 cells include the `w_non_aligned` entries and Refinement 3 changes the TILE-output block geometry on top of these transfer shapes. INPUTS that exercise it: `(8,16)×(1,1,1000,50)`, `(1,1,1,32)×(1,1,1024,17)` pad 7, `(8,16)×(1000,50)`, `(1,32)×(1024,17)`, `(7,)×(100,50)`, plus every table `[V, D]` and every rank-1 entry. Keep the scheme gather-direct and multi-core; do not add a `ttnn.to_layout` / padding wrapper (the native kernel support policy). The column split may put a partial last unit on its own core, so each core must clip from the global column index, not from `n_cols`. R2's review check needs an op-carried test: an RM table with `D % 32 ≠ 0`, index `V-1` (the last row) in both output layouts, exact, with no fault.

**Done when**: every golden cell with `alignment=w_non_aligned`, `rank=1` or `table_rank=2` at bfloat16 / indices in DRAM passes; `supported_fail`, `xpass_drift` and `xfail_wrong_mode` all 0; the last-row clip test above passes; prior op-carried tests still pass.

### [ ] Refinement 2 — float32 tables and indices in L1

**Goal**:
- add `ttnn.float32` to SUPPORTED["dtype"] (R1: every float32 element moved unchanged);
- add `"L1"` to SUPPORTED["memory_config"] (indices in L1, interleaved, R3; with `memory_config=None` the output then goes to L1, interleaved, R12).
- Omitted: SUPPORTED["dtype"] bfloat8_b — unbuildable on quasar (target page family facts: no bfloat8_b path on this part; the golden suite skips those cells as `unbuildable on quasar`; refuse it in validate() as today, no refinement).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: a probe during verification (validate() bypassed via `run()`) showed **both values already run bit-exact with the Phase 0 kernels**: a float32 `(2,16)×(1,1,1000,64)` table in all four (table layout × output layout) pairs, and `(1,32)` indices in L1 with the output landing in L1. So the expected kernel delta is zero. `element_size()` already maps float32 → 4, and `face_row_bytes` / `tile_bytes` / `col_unit_bytes` are derived from it. Only the dtype part of the skill applies. This op has no `compute_kernel_config`, because the specification exempts pure data movement, and its DM-only rings carry no `data_format_metadata`. Do not add either. Placement has no skill: the indices are read through their TensorAccessor wherever they live, and `run()` already allocates the output in the indices' buffer type when `memory_config` is None. HOOKS["check"] verifies placement in every cell. `output_memory` keeps {None, DRAM}: an L1 `memory_config` argument is not in TARGET and stays refused by name (R13). Extend `test_embedding_gen_precision_baseline.py` with a float32 case (R1's check is the float32 cells, graded `exact`). Ordered after Refinement 1 so its float32 × `w_non_aligned` cells run against the clipped kernels.

**Done when**: every golden cell with `dtype=float32` and/or `memory_config=L1` (inside the rest of SUPPORTED) passes; the three loud categories are 0; the float32 precision-baseline case is bit-exact; bfloat8_b is still refused by validate().

### [ ] Refinement 3 — Fill the grid on the traced small-S TILE-output calls

**Type**: perf

**Goal**: no `LOOSE_CASES` entry is perf-flagged, so this region is free-selected. It is the traced model calls with a TILE output, which leave most of the 32-cluster grid idle. `plan_blocks` step 1 pins a TILE-output block at `block_tokens = 32`. The design lists this as a mechanism cap, but the real constraint is only that a block stays inside one output tile-row. So a call with one tile-row of tokens gets only the column split to spread over the grid:
- Llama `(1,32)×(1,1,8192,64)` TILE→TILE (also RM→TILE): **2 cores**; the busiest reader issues **64** face-row table reads alone.
- Llama `(1,1,1,32)×(1,1,64,128)` {TILE, RM}→TILE, incl. `padding_idx=5`, PADDED 63 and BINARY `(1,1,2,128)`: **4 cores**, **64** reads each.

Lever: let a TILE-output block cover a sub-tile-row token range (a divisor of the tile height, chosen with the column split so token groups × column chunks fill the grid), and size the ring entry to the block's live rows, not to a whole tile per column unit. A block that does not cross a face band (`block_tokens` dividing the 16-row face height) keeps each (column unit, half) of the block one contiguous run in the output tile, so the writer issues `faces_per_row` run-writes per tile column instead of one whole-tile write. That trade is bounded by master.md `tile_reorder` (whole pages beat sub-tile scatter on a DRAM-bound move: do not go finer than the grid needs). The grid-fill pattern is master.md `width_split`. Qwen `(1,1)×(1,1,4096,128)` (one token) is launch-bound and is not a target. No SUPPORTED change.

**Verifier notes**: this also discharges the L1-ledger finding that the interleaved block-size default (spread the work units over the full grid, then take the coarsest block that fits) is not met for TILE output when `B·ceil(S/32)·Dt < num_cores`. Update `op_design.md`'s Mechanism-caps row and `l1_ledger.md` with the new entry-size expression. Every knob stays in `plan_blocks` (single source); the writer's page and offset math must come from the same `Geometry` (`kernels/embedding_gen_common.hpp`).

**Done when** (structural gate, read off the built program by an op-carried test: per-core `num_units` runtime args × reads per block from the plan): on `(1,32)×(1,1,8192,64)` TILE→TILE and `(1,1,1,32)×(1,1,64,128)` RM→TILE, cores engaged go from 2 / 4 to **≥ 16**, and table reads issued by the busiest reader thread go from 64 to **≤ 8**. Results stay exact; the golden suite is green. No regression across the config-spanning guard set: one representative per (table layout × output layout), plus the multi-block (`(2,1,1,1024)×(1,1,64,128)`) and ring-cap paths. On each, cores engaged must not fall and busiest-reader reads must not rise. The changelog records before/after counts and states that the time gain is unconfirmed on ttsim.

### [ ] Refinement 4 — Issue parallelism on the face-row paths: blocks per core and thread plan

**Type**: perf

**Goal**: every INPUTS entry runs exactly one block per busy core. So within a core the writer starts only after the whole block has landed (no reader/writer overlap), and a second reader thread would have nothing to do. The face-row paths (TILE table, or TILE output) issue `2·n_cols` 16-element reads per token, all from one DM hart:
- `(2,32)×(1,1,64,4096)` TILE→TILE / RM→TILE: **512** reads on one reader thread;
- `(4,1,1,128)×(1,1,2048,512)` TILE→TILE / TILE→RM / RM→TILE: **512**.

Levers (design perf lamps *Overlap: one block per core* and *Thread plan*; master.md `split_reader`, `double_buffer`):
- split a core's work into ≥ 2 blocks when it has one, so the depth-2 ring overlaps gather and store;
- raise `THREAD_PLAN`'s reader (and writer) counts so each reader thread issues a fraction of the core's reads.

The kernels already deal blocks `t, t+N` and the op-carried test runs (2,0,2), (2,0,1) and (1,0,2), so the plan is one constant (R + W ≤ 6, target page). The scratchpad and the ring already scale with R. No SUPPORTED change.

**Verifier notes**: after Refinement 3, because its block-geometry change sets the blocks-per-core baseline this one splits further. The target page's default for a data-movement-bound op is one thread of each kind. The case for more here is the design's issue-bound argument for 32 B transactions, so on a timed backend measure the plan against (1,0,1) before keeping it. Keep the plan test that reads `num_threads` off the built `ProgramSpec`, and the one-device-session test.

**Done when** (structural gate, read off the built program): on `(2,32)×(1,1,64,4096)` TILE→TILE and `(4,1,1,128)×(1,1,2048,512)` TILE→RM, blocks per busy core go from 1 to **≥ 2**, and face-row reads issued per reader thread on the busiest core go from 512 to **≤ 128**. Results stay exact; the golden suite is green; every thread is shown to have run (missing work if one did not). No regression on the Refinement 3 guard set: cores engaged must not fall and busiest-thread reads must not rise. The changelog records before/after counts and states that the time gain is unconfirmed on ttsim.

### [ ] Refinement 5 — Coalesce RM-table reads for a TILE output (row staging)

**Type**: perf

**Goal**: in an RM table, a row's column chunk is `n_cols·32·esize` contiguous bytes of one DRAM page. Yet RM→TILE issues `2·n_cols` 16-element DRAM reads per token, because the two halves of each column unit land in different faces of the output tile. The traced Llama calls with `padding_idx=5` and BINARY are RM→TILE. Lever (design perf lamp *RM-row staging*; master.md `read_coalescing`): one DRAM read per token of the row chunk into a reader-private scratchpad row (depth ≥ 2 so the next row's read lands under the current scatter), then place its `faces_per_row·n_cols` pieces in the ring entry with local copies. RM→RM is already one read per token, and a TILE table cannot coalesce (a row there is `2·Dt` non-contiguous 32 B runs). No SUPPORTED change.

**Verifier notes**: last, because it reshapes the reader's RM→TILE inner loop, which Refinements 3 and 4 also touch. The staging scratchpad is a new L1 buffer: add its row to `l1_ledger.md` (size per reader thread, live set, why it cannot share with the index slots) and keep it under the ring-cap / L1-budget solve in `plan_blocks`.

**Done when** (structural gate, read off the op by a probe or op-carried test): DRAM read transactions per token on RM→TILE go from `2·n_cols` to **1**, on `(4,1,1,128)×(1,1,2048,512)` RM→TILE (16 → 1 at the current plan) and `(1,1,1,32)×(1,1,64,128)` RM→TILE. Local L1 copies are counted and reported separately. Results stay exact; the golden suite is green. No regression on the guard set: DRAM transactions per token on every other layout pair unchanged, cores engaged and busiest-thread reads not worse. The changelog states that the time gain is unconfirmed on ttsim.
