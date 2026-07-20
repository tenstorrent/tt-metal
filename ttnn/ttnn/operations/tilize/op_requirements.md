# Operation Requirements: tilize

## Definition
- **Formula**: `output[i] = input[i]` — identity of element VALUES; only the byte
  positions change (ROW_MAJOR → TILE layout). Optional value-preserving cast when
  `dtype=` narrows/widens the storage format (bf16↔fp32↔bf8b).
- **PyTorch Reference**: none — the oracle is identity. `to_torch(tilize(from_torch(x,
  ROW_MAJOR))) == x` (PCC for bf8b / value-preserving-cast tolerance when `dtype=`
  narrows).
- **Import Path**: `from ttnn.operations.tilize import tilize`
- **Function Signature**:
  ```python
  tilize(
      input_tensor: ttnn.Tensor,                       # ROW_MAJOR_LAYOUT, on device, last two dims % 32 == 0
      memory_config: ttnn.MemoryConfig | None = None,  # output mem config (default: input's)
      *,
      dtype: ttnn.DataType | None = None,              # output dtype (default: input's); value-preserving cast only
      use_multicore: bool = True,                      # distribute tile-rows across the grid
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses `Refinement \d+[a-z]?`)**: primary refinements are `Refinement N`; a partial's follow-up appends a lowercase letter (`Refinement 1b`), ordered immediately after its parent.

### [x] Phase 0 — Core Implementation (already delivered, verified this pass)

- **SUPPORTED dtype**: [bfloat16, float32]
- **SUPPORTED output_dtype**: [bfloat16, float32, bfloat8_b]
- **SUPPORTED use_multicore**: [False, True]
- **SUPPORTED shard_api**: [none]
- **SUPPORTED out_scheme**: [interleaved]
- **SUPPORTED buffer**: [dram_to_dram, dram_to_l1, l1_to_dram, l1_to_l1]
- **SUPPORTED rank**: [2, 3, 4]
- **Cores**: single + multi (tile-row split, no inter-core sync)
- **Compute config**: bf16 = default `Fast`; fp32 = `Lossless` + `UnpackToDestFp32` +
  `fp32_dest_acc_en` (bit-exact terminal fp32); wide-W CB bounded by a constant chunk.
- **Golden baseline**: 36 / 36 in-scope cells passing (verifier CLI); bf16/fp32 identity
  exact, bf16→bf8b PCC ≥ 0.99.

### [x] Refinement 1 — uint32 integer passthrough

**Goal**: add `ttnn.uint32` to `SUPPORTED["dtype"]` and `SUPPORTED["output_dtype"]` so the
integer-passthrough family works (only `uint32 → uint32` is valid — int↔float crosses are
pruned by `INVALID` in `feature_spec.py`). Moves the 6 pure-interleaved `uint32→uint32`
cells from `xfail_expected` to `supported_pass`. The tilize LLK reorders integer bytes
with no arithmetic and no cast, so the work is: derive the output CB `data_format` and
`tile_size` from the uint32 dtype, and ensure the compute path does **not** take the
fp32 branch (no `Fp32Mode::Lossless` / `UnpackToDestFp32` / `fp32_dest_acc_en` for
integers — those apply only to float precision).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: this is integer passthrough, **not** float precision configuration —
the skill's ComputeKernelConfig float knobs (`math_fidelity`, `fp32_dest_acc_en`,
`UnpackToDestFp32`) do **not** apply; the only levers are the CB `data_format`/`tile_size`
(already dtype-derived in `tilize_program_descriptor.py`) and keeping the integer path off
the fp32 branch in `tilize_compute.cpp` (`is_fp32_in` is already 0 for uint32, so verify
the default `Fast` tilize handles integer formats — the helper falls back to standard
tilize for non-Float32/Float16_b formats). Independent of Refinement 2 and can land in
either order; the 4 `uint32 + sharded` crossed cells need **both** refinements. Land this
first — it is interleaved-only and the smaller change.

**Done when**: `SUPPORTED["dtype"]` and `SUPPORTED["output_dtype"]` include `uint32`; the
6 pure `uint32→uint32` interleaved golden cells pass (`comp_equal` exact); no regression
in the bf16/fp32/bf8b cells.



### [x] Refinement 1b — uint32 integer passthrough (debug: fix gate violations)

**Goal**: fix the hard violation from Refinement 1 so the completion gate's three bullets hold.

**Verifier notes** (mechanical, from the harness completion gate):

```
Bullet 3 FAIL: golden responsible cells 42/72 below majority threshold.
```

**Done when**: the gate passes — zero hangs in SUPPORTED, acceptance + refinement tests pass, golden majority with no regression.
### [~] Refinement 2 — Sharded I/O (legacy_2d HEIGHT/WIDTH/BLOCK + nd)

**Goal**: add `"legacy_2d"` and `"nd"` to `SUPPORTED["shard_api"]`, and
`HEIGHT_SHARDED`, `WIDTH_SHARDED`, `BLOCK_SHARDED`, `"nd"` to `SUPPORTED["out_scheme"]`.
Support RM-sharded L1 input → TILE-sharded L1 output on the same grid/shard spec, via
in-kernel sharded data access: the output CB is aliased directly onto the local L1 shard
buffer (zero-copy — `tilize_block` packs tiles straight into the shard, **no DRAM
writes**), and the reader consumes the local RM shard. Moves the ~30 sharded golden cells
(12 nd + 18 legacy_2d HEIGHT/WIDTH/BLOCK) from `xfail_expected` to `supported_pass`.

**Implementation skill**: /memory-layouts

**Verifier notes**: no dedicated sharding skill exists; `/memory-layouts` is the closest —
its native-support policy covers the in-kernel "sharded reader for sharded I/O" data-access
change, and `/interleaved-parallel` explicitly defers sharded tensors to `/memory-layouts`
territory. Two op-specific fixes this refinement must also make (the skill can't see them):
(1) `validate()._shard_api_of` currently returns `"legacy_2d"` for *any* sharded config,
including nd — fix it to return `"nd"` for the NdShardSpec path so the validated axis
matches the tagger; (2) add the design's `EXCLUSIONS` for single-core + sharded
(`{"use_multicore": False, "shard_api": "legacy_2d"}` and `... "nd"`) — sharding is
inherently multi-core. This carries the design's zero-copy write perf lever (R3): when it
lands, re-target the roofline — the write side becomes an L1 loopback, not a DRAM transfer
(`/perf-roofline-dm`, `/perf-measure` for the perf gate). Land after Refinement 1 (memory
config stress goes last; the 4 `uint32 + sharded` cells depend on both). If the full
legacy + nd surface is more than one focused pass, ship legacy_2d first as `[~]` and file
the nd remainder as `Refinement 2b` — do **not** partial-tick by wrapping the op in
`ttnn.to_memory_config` / host round-trips (the prompt forbids extra DRAM passes; the
whole point is the zero-copy shard write).

**Done when**: `SUPPORTED["shard_api"]` includes `legacy_2d` + `nd` and
`SUPPORTED["out_scheme"]` includes the three legacy schemes + `nd`; the sharded golden
cells pass identity per scheme; `test_golden_main_tests.py` sharded tests
(`test_tilize_row_major_to_width_sharded`, `test_tilize_nd_sharded`) pass; no DRAM write
on the sharded output path (tt-npe shows zero output-side DRAM traffic).

**Landed [~] (2026-07-17)**: same-spec, one-shard-per-core, zero-copy path for ALL
schemes (HEIGHT/WIDTH/BLOCK legacy + nd, ROW & COL orientation, rank 2/3/4). Both CBs
aliased onto the local L1 shard buffers; the compute kernel tilizes each core's resident
RM shard straight into its resident TILE shard — no reader, no writer, no DRAM/NoC at all
(strictly stronger than the design's "write becomes L1 loopback": there is no transfer on
either side). `test_golden.py` sharded cells 100% (77/77 responsible, was 42; the 35
sharded xfails now pass), `test_regression.py` 9/9, `test_tilize_row_major_to_width_sharded`
PASS. All named axis values are in SUPPORTED. Deferred to **Refinement 2b** (refused cleanly
in `validate()`, never a hang / wrong output): multi-shard-per-core, interleaved↔sharded
crossover, and cross-spec resharding — these are what keep `test_tilize_nd_sharded` /
`test_tilize_nd_sharded_to_legacy_sharded` from fully passing.

### [~] Refinement 2b — Sharded I/O remainder (multi-shard-per-core, crossover, cross-spec reshard)

**Goal**: extend the sharded path beyond same-spec one-shard-per-core to the cases
Refinement 2 refuses (`test_tilize_nd_sharded` / `test_tilize_nd_sharded_to_legacy_sharded`
crossover + cross-spec + cliff-core cases). Three independent sub-levers, in ascending
difficulty — land them in this order:

1. **Multi-shard-per-core (same-spec, even)** — when `num_shards % num_cores == 0` and
   `num_shards > num_cores`, a core owns `k = num_shards/num_cores` contiguous shards. Each
   shard is a contiguous RM block whose folded height is a multiple of 32, so the whole bank
   tilizes as `k * (shard_h/32)` blocks of `Wt` tiles straight into the concatenated output
   bank (shard-index order matches on both sides because in/out use the same distribution).
   **Exact lever**: in `_create_sharded_program_descriptor`, set
   `num_blocks = (num_shards // num_cores) * (shard_h_folded // 32)` (uniform CT arg); in
   `validate()`, relax the `num_shards != num_cores` refusal to `num_shards % num_cores != 0`
   for the even case. Cliff cores (uneven `num_shards % num_cores != 0`, e.g. `[23,96,160]`
   over 4 cores) need per-core RT `num_blocks` derived from the shard→core distribution — a
   second step. Probe first with `[4,128,128]`/`[2,64,64]` nd (8 shards / 4 cores).

2. **Interleaved↔sharded crossover** — DRAM-interleaved RM input → sharded TILE output
   (reader reads the rows for each core's output shard from DRAM, aliased output CB), and
   the reverse (aliased input shard → compute → DRAM-interleaved TILE writer). HEIGHT output
   sharding maps to per-core contiguous tile-row ranges (reuse the interleaved reader with a
   per-core `start_row` = output-shard row offset); WIDTH/BLOCK need column-chunk reads.

3. **Cross-spec resharding** — input shard spec ≠ output shard spec (different grid / shard
   shape / scheme, incl. nd→legacy). Requires cross-core NoC data movement (a shard's data
   redistributes across cores) — the largest lift; keep zero-DRAM by moving through L1.

**Done when**: `test_tilize_nd_sharded` and `test_tilize_nd_sharded_to_legacy_sharded` pass;
no regression on the Refinement 2 same-spec cells; no hangs.

**Landed [~] (2026-07-20)**: two of the three sub-levers, for the tractable
(contiguous-mapping) cases:
- **Sub-lever 1 — multi-shard-per-core, same-spec, even, no-padding (nd)**: a core
  owns k = n_shards/n_cores contiguous FULL shards, tilized as one k*shard_h x
  shard_w bank straight into the concatenated output bank (identity preserved
  because both sides use the IDENTICAL nd spec). `num_blocks` is now derived from
  the physical per-core bank (`buffer_num_pages/num_cores/32`). Legacy
  HEIGHT/WIDTH/BLOCK cannot be multi-shard-per-core (`from_torch` rejects
  `n_shards > n_cores`), so this is nd-only. Cliff/padded shards stay refused.
- **Sub-lever 2 — interleaved↔sharded crossover, BOTH directions, split
  reader/writer**, for legacy HEIGHT_SHARDED, ROW_MAJOR, tile-aligned,
  one-shard-per-core (each shard = a CONTIGUOUS global tile-row range, so the
  split reader/writer REUSE the interleaved reader/compute/writer kernels with a
  per-core row/tile offset — the exact mapping the native optimized factory uses):
  * DRAM-interleaved RM in → HEIGHT-sharded TILE out (split reader; write is an L1
    loopback into the aliased output shard — no DRAM write).
  * HEIGHT-sharded RM in → DRAM-interleaved TILE out (split writer; read is the
    aliased input shard — no DRAM read).
  Verified identity (max_diff=0) bf16/fp32/uint32, rank 2/3/4, on-device.

Golden target movement: `test_tilize_nd_sharded` + `_to_legacy` 8→9 passing (+1:
nd `[4,128,128]/[2,64,64]` same-spec multi-shard). The rest of the golden crossover
cases are **nd + DRAM interleaved** and the `_to_legacy` cases are **cross-spec**
(nd→legacy) — both the general cross-core NoC path (sub-lever 3), deferred to 2c.
No regression (golden responsible 86/86, sharded unit 25/25, acceptance 35/35), no
hangs. Deferred to **Refinement 2c** (all refused cleanly in `validate()`):
WIDTH/BLOCK crossover, nd crossover, cross-spec resharding, cliff/padded multi-shard.

### [ ] Refinement 2c — Sharded I/O general cross-core NoC path (nd/WIDTH/BLOCK crossover, cross-spec reshard, cliff cores)

**Goal**: the sharded cases Refinement 2b still refuses — all requiring a shard's
data to redistribute across cores via NoC (no contiguous per-core mapping), which
is the native's general `TilizeMultiCoreDefaultProgramFactory` path. Concretely:
`test_tilize_nd_sharded` crossover cases (nd sharded ↔ DRAM interleaved),
`test_tilize_nd_sharded_to_legacy_sharded` (nd→legacy cross-spec), cross-spec
reshard (input spec ≠ output spec), WIDTH/BLOCK↔interleaved crossover
(column-chunked, non-contiguous tile↔page mapping), and cliff/padded multi-shard
(`[3,160,160]`, `[23,96,160]`, `[5,4,160,160]` — uneven split and/or partial shards).

**Verifier notes / concrete levers for the next implementer**:
1. **The tile↔page mapping is no longer contiguous** for nd/WIDTH/BLOCK, so the
   split reader/writer cannot reuse the interleaved kernels with a single row
   offset. Compute per-core, per-tile the global interleaved page index (writer)
   or the per-shard RM (start_row, col_byte_offset) (reader) on the HOST and pass
   as runtime args — the kernel then needs no sharding knowledge. The host must
   match ttnn's shard→core→region enumeration exactly (ROW_MAJOR shard-grid order;
   probe the physical bank via `buffer_num_pages`/`buffer_page_size` as
   `probes/probe_014.py` did — genuine-nd multi-dim shards padded per bank).
2. **Cross-spec (nd→legacy, or different in/out spec)** needs cross-core NoC: a
   shard's data lands on a different core's output shard. Keep zero-DRAM by moving
   through L1 (semaphore handshake). This is the largest lift.
3. **cb_descriptor_from_sharded_tensor** on genuinely-nd tensors requires
   `core_ranges=grid` passed explicitly (no legacy `shard_spec()` to deref) — see
   Refinement 2b's fix; reuse it.
4. **Wide-W HEIGHT crossover perf**: the 2b crossover CBs are sized `2*Wt*tile`
   with `Wt = full W/32` (not chunked), so a wide HEIGHT shard is L1-unbounded.
   Add the interleaved path's `Wt_chunk` chunking to the crossover reader/writer
   (they already accept `num_chunks`/`Wt_chunk` CT args).
