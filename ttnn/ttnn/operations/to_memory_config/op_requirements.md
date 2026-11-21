# Operation Requirements: to_memory_config

Target: `quasar` on the ttsim simulator (untimed). Every perf gate below is a **structural count**
(backend page, *Performance measurement*): the gain in time is unconfirmed until a timed backend or
hardware measures it, and no simulator timing is a speedup.

## Definition
- **Formula**: `output.page[p] = input.page[p]` for every page id `p` (a tile in TILE layout, a row in
  ROW_MAJOR); the output has the input's shape, dtype, layout and bytes and lives exactly where
  `memory_config` says (spec rules R1, R2).
- **PyTorch Reference**: `torch.clone(tensor)` (spec `reference`: `tensor.double()`)
- **Import Path**: `from ttnn.operations.to_memory_config import to_memory_config`
- **Function Signature**: `to_memory_config(tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor`
  (no other parameters, no compute config; `memory_config` is required)

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE, ROW_MAJOR]
- **SUPPORTED placement axes**: memory_config ∈ {DRAM, L1}, sharding ∈ {interleaved, height}, to ∈ {DRAM, L1, L1/height}
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}, rank ∈ {2, 3, 4}
- **Cores**: R-bal spreads pages over the full 8×4 grid (`split_work_to_cores(row_wise=True)`); R-in / R-out use the shard grid
- **Thread plan**: (R, C, W) = (1, 0, 1): bound by data movement, no compute kernel
- **Compute config**: none (the op computes nothing)
- **Golden baseline**: 558 supported_pass, 558 xfail_expected (all float32), 372 invalid_skipped, 0 loud; X1 / X2 rejections pass

### [ ] Refinement 1 — Native height-shard side (zero-copy owner side)

**Goal**: implement `SUPPORTED["sharding"]` height and `SUPPORTED["to"]` L1/height natively. Today the
owner core of a height shard moves it through the staging ring: in R-in it reads its **own** input shard
back through the input `TensorAccessor` (a NoC loopback into `dfb_pages`), and in R-out it lands pages in
`dfb_pages` and then writes them again into its **own** output shard. Back the owner side's ring on the
shard itself (regime R-borrow of `op_design.md`): `ttnn.experimental.dfb_spec_from_sharded_tensor(...,
borrowed_from=<input|output>)`. In R-in the reader only publishes the shard's entries and the writer
sends them out; in R-out the reader lands every page directly in the output shard and the writer only
retires the entries. R-bal (both sides interleaved) is unchanged. No new SUPPORTED value: the values
already claimed become natively served.

**Verifier notes**:
- Hardest first: this reshapes the shard-side dataflow that Refinement 2 (float32 widens the entry
  size), Refinement 3 (ring slicing; after this refinement it only concerns R-bal) and Refinement 5
  (coalesced shard runs) build on.
- No skill covers `memory_layout` / sharding placement. The pattern is the Metal 2.0 counterpart of
  `cb_descriptor_from_sharded_tensor`: one named DFB whose spec swaps by regime. Keep **one kernel
  pair**; only the ring spec and each side's sync protocol change (spec GUIDANCE: one data path, no
  path per pair of placements).
- Protocol: the side that does no NoC transfer (R-in reader, R-out writer) uses explicit sync
  (`reserve_back`/`push_back` or `wait_front`/`pop_front`, listed in `disable_dfb_implicit_sync_for`).
  The transferring side may stay implicit **only when** the runtime finds a txn-id count for
  `num_entries = pages_per_shard` with ≤ 255 entries per id (`compute_optimal_txn_id_count`; e.g. 400
  pages → 2 × 200 is fine, a prime > 255 is not). Otherwise it goes explicit as well, with every
  transfer tagged `NocOptions::TXN_ID {.trid = 0}` and a barrier on the same id (target page, *Ring
  protocols*). Both kernels still hand the tag back at exit.
- Equal-config sharded → sharded stays one core-local copy (R7: copy, one dispatch).
- This also closes the L1-ledger finding: Phase 0 deferred R-borrow without a sanctioned reason
  (`l1_ledger.md`, data-movement budget). Update the ledger row (borrowed ring = shard, no ring L1 on
  the owner side) and the regime table.

**Done when**: core-local staging copies per page on the shard owner side go from 1 to 0 in R-in and
R-out (no input-accessor read of a page the core itself holds, no ring between a landed page and its
own output shard), shown by a test that reads the program's DFB specs and their `borrowed_from`; the
golden suite stays at 0 loud with every Phase 0 pass kept; `test_to_memory_config_regimes.py` (R6
reshard 2→1 and 1→2 in both layouts, COL_MAJOR + partial shard, the one-session test with
`ttnn.to_layout` afterwards) stays green. The time gain is unconfirmed on this backend.

### [ ] Refinement 2 — float32

**Goal**: add `ttnn.float32` to `SUPPORTED["dtype"]` (the only TARGET − SUPPORTED value). Each element must
arrive bit-exact (spec R1: no element passes through bfloat16 on the way), in every placement pair,
both layouts, every alignment and rank. The spec gives the op no compute config, so none is exposed.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- The verifier probed it in-process (validate gate relaxed, no kernel change): 108 / 108 bit-exact
  float32 copies over 6 shapes × 2 layouts × 3 sources × 3 destinations, including ragged row-major
  slots. Expect a SUPPORTED widening + the dtype-aware ring format (`data_format_metadata` already
  follows `tensor.dtype`) + tests, and zero kernel changes; the skill's compute-config and
  intermediate-CB parts do not apply (no compute kernel, no DEST).
- `entry_bytes` doubles: a float32 RM row of W > 16,384 hits the 64 KiB packet cap that `validate()`
  already refuses; nothing in INPUTS comes close. A 1-core float32 shard of `[1,1,32,4096]` TILE is
  128 pages × 4 KiB = 512 KiB, above `RING_BYTES_CAP`: it takes the depth-2 path today (after
  Refinement 1 the borrowed ring has no ring cap on the owner side).
- Ordered after Refinement 1 so float32 lands once on the final shard-side dataflow.

**Done when**: every golden float32 cell (558 xfail_expected today) passes with relative RMS 0 and no
cell regresses; the precision baseline gains float32 cases at 0 ULP.

### [ ] Refinement 3 — Overlap reads and writes inside each R-bal core's ring

**Type**: perf

**Goal**: when a core's whole assignment fits one block, `ring_geometry` sizes the ring to exactly
`core_pages_max` entries at depth 1. The runtime slices a ring into 2–4 txn ids only when the entry count
has a factor in {2, 3, 4} (`compute_optimal_txn_id_count`, `dataflow_buffer.cpp:1452`); otherwise it uses
**one** id, and the writer cannot start until every read of the core has landed: reads and writes
serialize. Among INPUTS × layouts the R-bal rings of `[1,1,1600,256]` TILE (13 pages per core),
`[1,1,400,512]` TILE (7) and RM (13), `[1,400,144]` RM (13), `[1,144,400]` RM (5) and `[2,1,100,47]` RM
(7) get one id. Co-tune the page-axis block knob and the buffer depth (catalog `double_buffer` in
`ttnn/ttnn/operations/examples/master.md`: keep several entries in flight, depth 2), e.g. `block_pages =
ceil(core_pages_max / 2)` at depth 2, or ring entries padded to a multiple of 4. The design's "Overlap"
perf lamp names exactly this. No SUPPORTED change.

**Verifier notes**: a departure from the "coarsest block that fits" default is justified here by the
structural count, as this backend has no clock; record it in `op_design.md` / `l1_ledger.md`. After
Refinement 1 the borrowed shard-side rings need no slicing (their explicit side issues no transfer to
overlap), so this concerns R-bal rings only. Keep `block_pages` single-sourced in `ring_geometry`.

**Done when**: on every engaged core with ≥ 2 pages, the ring gets ≥ 2 txn-id slices, preferably as many
as fit (today 1 on the six configurations above), shown by a test that reads the ring geometry and
applies the runtime's slicing rule (or reads its `implicit sync: producer txn_ids=[...]` debug log);
golden green; no regression across the guard set (one representative per regime R-bal / R-in / R-out ×
layout × placement). The time gain is unconfirmed on this backend.

### [ ] Refinement 4 — L1-interleaved bank-owner split (R-bank)

**Type**: perf

**Goal**: in R-bal a page of an L1-interleaved tensor takes a mesh hop from its bank's core to the
staging core and, if the other side is L1 too, another one onward. Assign each page to the core that
holds its L1 bank (regime R-bank of `op_design.md`): `start_page = own bank index`, `page_stride =
num_l1_banks` (the kernels already take `page_stride`). Both L1-interleaved tensors map page `p` to bank
`p mod num_banks`, so an L1 → L1 copy becomes core-local, and DRAM ↔ L1 saves one hop per page. No
SUPPORTED change. Catalog pointer: `read_coalescing`'s "bank affinity" arm in
`ttnn/ttnn/operations/examples/master.md`.

**Verifier notes**: the design deferred this as unreachable for want of an L1 bank → core map. It is
reachable: `MemoryView.num_banks` gives the bank count, and a kernel can find its own bank by comparing
the accessor's NoC coordinates for pages `0..num_banks-1` with its own (`my_x` / `my_y`), or a host
query can be added. Check first that every bank-owning core is a worker core of the grid. DRAM → DRAM
stays R-bal.

**Done when**: pages moved through a core that owns neither the page's source bank nor its destination
bank go from all of them (R-bal today) to 0 on L1-interleaved cells, shown by a test that reads the core
table off the host program; golden green; no regression across the guard set (one representative per
regime × layout × placement). The time gain is unconfirmed on this backend.

### [ ] Refinement 5 — Coalesced transfers over contiguous shard runs

**Type**: perf

**Goal**: inside one height shard consecutive pages are contiguous in L1, but every page is one NoC
transaction. On the shard → shard paths (the model's reshard two cores → one and back, R6, and the
equal-config copy), move each contiguous run (the intersection of a source shard and a destination
shard) in one wide transfer of up to 64 KiB (catalog `read_coalescing`, ⭐⭐ T2, measured 1.11–3.66×
on another part). No SUPPORTED change.

**Verifier notes**: depends on Refinement 1 (the owner side then reads/writes the shard in place, so a
run is a plain address range). It needs explicit sync on the coalescing side, or a BLOCKED ring with
`block × entry_bytes ≤ 64 KiB`, with every transfer tagged. Interleaved sides cannot coalesce (adjacent
pages sit in different banks).

**Done when**: NoC transactions per shard on the shard → shard paths go from `pages_per_shard` to
`ceil(run_bytes / 65536)` per contiguous run (e.g. `[1,1,64,512]` TILE, 2 cores → 1: 32 → 2), shown by
a test or probe that counts the transfers; golden and the R6 reshard tests green; no regression across
the guard set. The time gain is unconfirmed on this backend.
