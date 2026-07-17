# Operation Design: tilize

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (movement-dominated layout op with a small compute stage) |
| Goal | Convert a ROW_MAJOR tensor into TILE layout (32×32 tiles of four 16×16 faces) without changing element values. Native FPU-consumable layout. |
| Math | `output[i] = input[i]` (identity of values; only byte positions change). Optional value-preserving cast when `dtype=` narrows/widens the storage format. |
| Mode | Hybrid — dataflow readers/writer + a `tilize_block` compute stage. |
| References | `.claude/references/ttnn-cb-memory-fundamentals.md` (tilize dataflow pattern), `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp`, `ttnn/ttnn/operations/toy_tilize_untilize/` (reference kernels), `tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md`, `ttnn/ttnn/operations/examples/data_movement_perf_optimizations.md` |

Tilize does **no arithmetic**. Read-back must be bit-identical to the input values (PCC tolerance only when `dtype=` narrows to a lossy format such as bfloat8_b, or fp32→bf16). The `tilize_block` LLK reorders one tile-row of row-major faces into `Wt` output tiles per call.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | ROW_MAJOR, on device, last two dims % 32 == 0, rank ≥ 2 | — | tensor |
| `memory_config` | `ttnn.MemoryConfig \| None` | no | interleaved (DRAM/L1) or sharded (L1) | input's mem config | host (drives output alloc) |
| `dtype` | `ttnn.DataType \| None` | no (kw-only) | same-family cast only (bf16↔fp32↔bf8b; uint32→uint32) | input's dtype | host → output CB `data_format` |
| `use_multicore` | `bool` | no (kw-only) | True / False | True | host (drives grid) |

`dtype` is the OUTPUT dtype. int↔float casts are out of contract (see `INVALID` in `feature_spec.py`). The op declares `output_dtype` in `SUPPORTED` and threads it into `validate()`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank ≥ 2; last two dims (H, W) both divisible by 32 (op does NOT pad) |
| Dtype | bfloat16 (primary), float32, uint32/uint16/int32 (integer passthrough) |
| Layout | ROW_MAJOR_LAYOUT (always — the op's whole job is to produce TILE) |
| Memory | INTERLEAVED (DRAM or L1) or SHARDED (L1, ROW_MAJOR-sharded) |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input's logical shape |
| Dtype | input's dtype, unless `dtype=` provided (value-preserving cast) |
| Layout | TILE_LAYOUT |
| Memory | input's mem config, unless `memory_config=` provided |

## Dataflow Strategy

High-level data path (interleaved DRAM→DRAM, the Phase-0/perf-target path):

```
DRAM (RM sticks)
   │  reader (NCRISC, NoC0): read 32 sticks = 1 tile-row → cb_rm_in
   ▼
cb_rm_in  (row-major, Wt tile-pages per tile-row)
   │  compute (TRISC unpack/math/pack): tilize_block reorders faces → tiles
   ▼
cb_tiled_out  (TILE, Wt tile-pages per tile-row; pack casts to output dtype)
   │  writer (BRISC, NoC1): drain Wt tile pages per tile-row → DRAM
   ▼
DRAM (TILE pages)
```

- **Reader on NoC0, writer on NoC1** so the DRAM-read stream and DRAM-write stream overlap instead of contending (`data_movement_perf_optimizations.md` §9).
- The reader groups sticks into tile-height blocks (32 sticks = 1 tile-row) and pushes `Wt` tile-sized pages per block (TILE granularity of `read_sticks_for_tilize`).
- The compute stage consumes one tile-row (`Wt` pages) per `tilize_block` call and pushes `Wt` tiled pages.
- The writer drains `Wt` tile pages per tile-row to the output tensor.
- **No inter-Tensix communication** for the interleaved and legacy/nd sharded paths — each tile is independent, so cores work disjoint tile-row ranges with no multicast/semaphores.

**Sharded output (Refinement 3):** when both input and output are L1-sharded with identical grid/shard specs, the output CB is aliased directly onto the local L1 shard buffer (zero-copy: `cb_backing_buffer = shard_buffer`). The writer performs **no DRAM writes** — `tilize_block` packs tiles straight into the shard. Re-target the roofline: the write side becomes an L1 loopback, not a DRAM transfer.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one **tile-row** = 32 sticks of input = `Wt` output tiles (`Wt = W / 32`) |
| Grid | `device.compute_with_storage_grid_size()`; single-core when `use_multicore=False` |
| Per-core work | `ttnn.split_work_to_cores(grid, Nt_h)` where `Nt_h = (volume / W) / 32` (all leading dims fold into height). Returns `core_group_1` (more rows) + `core_group_2` (fewer). |
| Per-core args | `rows_per_core` tile-rows → reader `start_page = tile_row_start * 32` (stick index), `total_num_rows = rows_per_core * 32`; writer `start_tile = tile_row_start * Wt`, `num_tiles = rows_per_core * Wt` |
| Remainder | handled by `split_work_to_cores` two-group split (cliff cores in `core_group_2`); H is always tile-aligned so no partial tile-rows |

Distribute along the tile-row (height) axis — spread cores across the DRAM-bank-facing axis via `row_wise` ordering (`data_movement_perf_optimizations.md` §A1), not stacked on one column line.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_rm_in` | 0 | `ttnn.tile_size(in_dtype)` (TILE granularity) | `2 * Wt_chunk` (double-buffer) | input dtype | reader | compute (`tilize`) | streaming |
| `cb_tiled_out` | 16 | `ttnn.tile_size(out_dtype)` | `2 * Wt_chunk` (double-buffer) | output dtype | compute (`tilize`) | writer | streaming |

Rationale:
- Both CBs are **streaming reader↔compute↔writer** buffers on independent processors, so depth 2 (double-buffer) is correct — the reader fills tile-row *n+1* while compute/writer drain *n* (`ttnn-cb-memory-fundamentals.md` "Reader → Compute (independent processors)"; `data_movement_perf_optimizations.md` §16). They are NOT sequential-helper intermediates, so they need not hold a full block.
- `cb_rm_in` uses TILE granularity (page = tile_size): the reader batches 32 sticks and pushes `Wt_chunk` tile-sized pages per tile-row; compute waits `Wt_chunk` per block. Push count = wait count. ✓
- `Wt_chunk` is the width-chunk in tiles. For narrow/moderate W, `Wt_chunk = Wt` (whole row). For wide W (perf bench W=2048 ⇒ Wt=64), chunk the width to a **constant** (e.g. `Wt_chunk = 8`) and loop over `W/Wt_chunk` chunks using `read_sticks_for_tilize`'s `byte_offset_within_page = chunk_id * chunk_bytes` — this bounds CB L1 footprint by a constant, not by `Wt` (`data_movement_perf_optimizations.md` §CB-bounded; the perf gate requires CBs bounded by a constant, not `Wt`).
- `cb_tiled_out` format = **output** dtype. When `dtype=` differs from input, the format mismatch across `cb_rm_in`→`cb_tiled_out` drives the pack-stage cast; the `tilize` helper's `UnpackAndPackReconfigure` mode reconfigures the pack register to the output format (real value-preserving cast at pack time, NOT a byte copy).

**Sharded output (R3):** `cb_tiled_out` `total_size` = full local shard (`shard_num_tiles * tile_size`), `core_ranges` = the shard grid, and the CB is created with the shard buffer as its backing buffer (zero-copy). No separate writer DRAM path.

## API Mapping

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------------------------|----------|-----------|--------------|
| Read RM sticks | helper | `dataflow_kernel_lib::read_sticks_for_tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:87-93` | `<cb_rm_in, TilizeGranularity::TILE>(accessor, total_num_rows, row_bytes, start_page, byte_offset_within_page)` | — (DRAM) | `cb_rm_in` | Owns its own CB reserve/push; page_size must = tile_size for TILE granularity; `start_page` = per-core stick offset; `byte_offset_within_page` selects the W-chunk for wide W |
| Tilize | helper | `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:187-197` | `<Wt_chunk, cb_rm_in, cb_tiled_out, InitAndUninit, WaitBlock, UnpackAndPackReconfigure, Fp32Mode::Fast>(num_blocks)` (symmetric TILE granularity) | `cb_rm_in` | `cb_tiled_out` | Owns CB wait/pop (in) and reserve/push (out). Requires `compute_kernel_hw_startup(cb_rm_in, cb_tiled_out)` first. `Fast` fp32 mode (correct default even for max-precision — see header L47-71). Pack-stage dtype reconfig performs the `dtype=` cast. |
| Boot | raw_api | `compute_kernel_hw_startup` | `tilize_helpers.hpp:89-93` (prerequisite doc) | `(cb_rm_in, cb_tiled_out)` | — | — | Must be called once before `tilize`; no helper wraps it. |
| Write TILE pages | raw_api | `noc_async_write` + `TensorAccessor::get_noc_addr` (block-batched, one barrier per block) | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:566` (`noc_async_write`); pattern per `ttnn/ttnn/operations/examples/double_buffer/kernels/db_writer.cpp:31-43` | `<cb_tiled_out>` | — (DRAM/L1) | Loop: `cb_wait_front(cb_tiled_out, Wt_chunk)` → issue `Wt_chunk` `noc_async_write`s → one `noc_async_write_barrier()` → `cb_pop_front(cb_tiled_out, Wt_chunk)` |

### Write TILE pages — Helpers considered and rejected

- **`dataflow_kernel_lib::write_sticks_after_untilize`** (`tilize_helpers_dataflow.hpp:129-135`): rejected. This is the *untilize* output helper — it writes row-major **sticks** (`noc_async_write(l1_addr, noc_addr, row_bytes)` per stick, per its doxygen L96-107 and `.inl` implementation, which explicitly de-tilizes / skips L1 padding between rows). Tilize's output CB holds **TILE pages**, not RM sticks; feeding tiled data through the stick writer would emit row-major bytes and destroy the tile layout. There is no tile-page writer helper in `ttnn/cpp/ttnn/kernel_lib/` (verified: the only CB-draining DRAM writer in the dataflow kernel lib is `write_sticks_after_untilize`). The raw batched `noc_async_write` loop is the canonical example-op pattern (`double_buffer/kernels/db_writer.cpp:31-43`, `toy_reduce_partial/kernels/writer.cpp:11-29`) and is required here.

## Compute Phases

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|-------------------------|-------------------|----------------|
| 0 | HW startup | no (`compute_kernel_hw_startup`) | — | — | registers configured for `cb_rm_in`→`cb_tiled_out` |
| 1 | Tilize tile-row (× `num_blocks`, optionally × `W/Wt_chunk` chunks) | yes (`compute_kernel_lib::tilize`) | `cb_rm_in`, `Wt_chunk` RM tile-pages, streaming | `cb_tiled_out`, `Wt_chunk` tiles | `cb_rm_in` popped, `cb_tiled_out` pushed for the writer; both double-buffered so next block overlaps |

The op is a single-phase compute pipeline; there are no sequential-helper intermediates, so no CB needs to hold a full block.

## Registry Declarations (for the op file)

The implementer declares these in `ttnn/ttnn/operations/tilize/tilize.py`.

**INPUT_TAGGERS** — project axes off the golden scenario dict (`inputs[0]`), per `feature_spec.py`:

| Tagger | Returns | Source |
|--------|---------|--------|
| `tag_use_multicore(inputs, axes)` | `bool` | `inputs[0]["use_multicore"]` |
| `tag_shard_api(inputs, axes)` | `"none" \| "legacy_2d" \| "nd"` | `inputs[0]["shard_api"]` |
| `tag_out_scheme(inputs, axes)` | `"interleaved" \| TensorMemoryLayout \| "nd"` | `inputs[0]["out"]` kind/scheme |
| `tag_buffer(inputs, axes)` | `"{in}_to_{out}"` over `{dram,l1}` | `inputs[0]["in"]/["out"]` buffer |
| `tag_rank(inputs, axes)` | `int` | `len(inputs[0]["input_shape"])` |

`dtype` and `output_dtype` are free cartesian axes (not tagged).

**SUPPORTED (Phase 0 baseline; grows per refinement toward `feature_spec.py`'s TARGET):**

```python
SUPPORTED = {
    "dtype":         [ttnn.bfloat16],
    "output_dtype":  [ttnn.bfloat16],
    "use_multicore": [False],
    "shard_api":     ["none"],
    "out_scheme":    ["interleaved"],
    "buffer":        ["dram_to_dram"],
    "rank":          [4],
}
```

**EXCLUSIONS** — cells inside the SUPPORTED rectangle refused *for now*. Empty at Phase 0 (everything else is caught by a SUPPORTED axis). Grows as SUPPORTED widens; e.g. once multicore + sharded are supported, single-core + sharded is inherently impossible and is excluded:

```python
EXCLUSIONS = [
    # sharded is inherently multi-core — refuse single-core sharded once both axes widen
    {"use_multicore": False, "shard_api": "legacy_2d"},
    {"use_multicore": False, "shard_api": "nd"},
]
```

**validate()** builds the axes dict from the **real tensor + kwargs** (NOT the scenario dict — the taggers' scenario form is for the harness only): `dtype = input_tensor.dtype`, `output_dtype = dtype_kwarg or input_tensor.dtype`, `use_multicore` from kwarg, `shard_api`/`out_scheme`/`buffer` from the input/output `memory_config`, `rank = len(input_tensor.shape)`. Then gate per-axis against SUPPORTED (raise `UnsupportedAxisValue`) and per-cell against EXCLUSIONS (raise `ExcludedCell`), both from `ttnn.operations._op_contract`.

Hard input validation (raise `ValueError`/`RuntimeError`, independent of the registry): input not ROW_MAJOR, input not on device, last two dims not divisible by 32.

**Index-axis canonicalization:** N/A — tilize has no `dim`/`axis` parameter.

## Key Risks and Gotchas

- **CB sync:** reader pushes `Wt_chunk` pages per tile-row (TILE granularity), compute waits/pops `Wt_chunk`, compute pushes `Wt_chunk`, writer waits/pops `Wt_chunk`. Push count = wait count on both CBs. Do NOT push per-stick (32 sticks ≠ `Wt_chunk` pages) — the classic tilize deadlock.
- **CB page size must be `tile_size`** for TILE granularity (`read_sticks_for_tilize` requirement). Mismatched page size hangs on `cb_reserve_back`.
- **Cast at pack, not copy:** when `output_dtype != dtype`, the element width changes (bf16 2B, fp32 4B, bf8b 1.0625B). The cast MUST come from the pack-stage reconfig (`UnpackAndPackReconfigure`) driven by the differing output CB format — never a raw byte copy in the writer.
- **fp32:** use `Fp32Mode::Fast` (default). It truncates fp32→tf32 into DEST, which is what every downstream FPU consumer does anyway; `Lossless` is slower and only correct under SFPU-from-DEST chains that tilize does not have (header L47-71). Set `fp32_dest_acc_en=true` in `ComputeConfigDescriptor` for fp32 input.
- **Integer passthrough (uint32/uint16/int32):** `tilize_block` reorders bytes without arithmetic; no cast is legal for integers (int↔float is out of contract — see `INVALID`). Output dtype must equal input dtype for the integer family.
- **Wide W ⇒ bound CB by a constant:** for W=2048 (`Wt=64`), a whole-row CB is `2*64` tiles ≈ 256 KB — do NOT let CB depth scale with `Wt`. Chunk W to a constant `Wt_chunk` and loop with `byte_offset_within_page`; keeps L1 bounded and satisfies the perf gate.
- **Height folding:** all leading dims fold into the stick/tile-row count (`Nt_h`). rank 2/3/4 differ only in how `Nt_h` is computed; the kernel logic is rank-agnostic.
- **Sharded zero-copy (R3):** the output CB is aliased onto the L1 shard buffer — there must be NO DRAM write. Adding one would show up as extra DRAM traffic in tt-npe and fail the re-targeted roofline.
- **Program cache:** pass only buffer base addresses as runtime args; bake `TensorAccessorArgs`, `Wt_chunk`, `num_blocks` as compile-time args so the program caches and the 2nd call with the same shape/dtype/mem_config is a cache hit (`data_movement_perf_optimizations.md` §18-19).

## Structural impossibilities (informational — for `/golden-tests`, not authored here)

`feature_spec.py` already declares `INVALID` (int↔float casts: uint32 paired with any float output and vice-versa). No additional op-specific structural impossibilities beyond those. The op file does NOT declare `INVALID`.

## Performance Methodology

Tilize is **movement-dominated with a small compute stage**. The NoC roofline is a *partial* bound — it must be confirmed with an ablation before DM levers are chased.

**Classification (Refinement 0, mandatory):** with `/perf-measure`, stub the `tilize_block` math (keep the CB reserve/push/pop + barriers and the read/write) and diff the device duration.
- If duration barely changes → **DM-bound** → the DRAM roofline is the target; walk the DM levers.
- If duration drops a lot → **compute-bound** → the DM roofline does not apply; the `tilize_block` throughput is the concern.
Record the classification verdict in `changelog.md` for R0.

**Per-case target (use the skills — do NOT eyeball):**
1. `/perf-roofline-dm` — characterize **both** DRAM sides for the interleaved path: read = ROW_MAJOR sticks, write = TILE pages. Bracket each transfer `ONE_FROM_ALL`/`ONE_TO_ALL` (few-core, no contention) … `ALL_FROM_ALL`/`ALL_TO_ALL` (full-grid, full contention); the true value lands near the full-contention end for round-robin interleaved. Cap each bound at `dram_peak` (WH 288 GB/s). `op_target = MAX(read_bound, write_bound, compute_bound)`, with read/write overlapping per the depth-2 CB. Floor for the reference shape: `≈ (read_bytes + write_bytes) / 288 GB/s`.
2. tt-npe (`tt_npe.sh <trace> --noc-trace`, Step 6) — PIN estimated cycles (validated vs golden), DRAM BW util, congestion %, binding resource.
3. `/perf-measure` — median device Tracy kernel duration over the trial loop (not one untrialed number). `achieved = measured / target`.

**Perf bench (separate, NOT in `feature_spec.py` INPUTS):** an underscore-prefixed in-tree bench with a grid-filling, DRAM-bound shape — bf16 `[1,1,2048,2048]` RM→TILE interleaved across the full 8×8 grid — swept single/multi-core and interleaved/sharded output. Run under `--collect-noc-traces` + Tracy; does NOT assert PCC. The tiny golden cells (largest 512×512) are the correctness gate and are too small to be bandwidth-bound — never measure perf on them, never add perf shapes to the golden suite.

**Per-refinement gate (each landed refinement records target / tt-npe pin / measured / achieved in `changelog.md`):**
- R0: baseline correct + roofline bracket + tt-npe pin + duration + DM-vs-compute ablation verdict.
- R1: multicore cycles ↓ vs R0 and scale with core count; congestion characterized.
- R2: per-core ≈ `max(read, compute, write)` not sum; achieved BW ↑; congestion ~0; CBs bounded by a constant.
- R3: re-targeted roofline (write becomes L1 loopback); tt-npe DRAM traffic ↓ (zero output-side DRAM).
- R4: roofline re-run per dtype (page size changes the target).
- DM cases should sit at/near their binding bound (tt-npe DRAM util toward ~92% peak for interleaved; congestion low).
