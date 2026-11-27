# Operation Design: g6_reshard

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel, no CB, no semaphore) |
| Goal | Return a new BLOCK_SHARDED L1 tensor in the placement `memory_config` names whose every tile is the tile of `input_tensor` at the same tile index, moved as stored, in exactly one device program. |
| Math | `out[t] = in0[t]` for every tile index `t = (tile_row, tile_col)` of the 2-D view; bytes copied verbatim (bfloat8_b shared exponents included). |
| Regime | `PULL`: every core that holds an out block reads that block's tiles straight from the L1 of the cores that hold them, landing each run in its final slot of the out shard. |
| References | `eval/prompts/g6_reshard.txt` (contract, via the task), `eval/golden_tests/g6_reshard/feature_spec.py` (TARGET, SCENARIOS, LOOSE_CASES), `eval/golden_tests/g6_reshard/helpers.py` (exact gate), `docs/explorations/g6_reshard/{DOSSIER,math,tensix,risks}.md`, `tt_metal/impl/buffers/buffer_distribution_spec.cpp:162-200` (GRID_2D block→core map), `tt_metal/impl/buffers/buffer.cpp:459-473` (in-shard page order), `tech_reports/tensor_accessor/tensor_accessor.md` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` (positional) | yes | on device, rank 4, BLOCK_SHARDED L1, TILE, bfloat8_b (Phase 0), 2-D view H and W multiples of 32 | — | placement → CT (walk geometry + in0 accessor args); address → common RT |
| `memory_config` | `ttnn.MemoryConfig` (positional) | yes | BLOCK_SHARDED, L1, shard spec present, one rectangle inside the worker grid, shard dims multiples of 32, grid ≥ block rows/cols in its orientation | — | placement → CT (walk geometry) + kernel core range; per-core block coords → per-core RT |

No dtype, no output tensor, no compute config. `out` takes `input_tensor`'s dtype and layout.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d_0, d_1, d_2, d_3)`, rank 4; production `(1, 1, 2048, 1024)` |
| Dtype | `bfloat8_b` (Phase 0 SUPPORTED = TARGET) |
| Layout | TILE |
| Memory | BLOCK_SHARDED, L1, any legal block placement (production: grid (0,0)-(10,9), shard (224, 96) = 7×3 tiles, ROW_MAJOR; last block row 1 tile high, last block column 2 tiles wide) |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` |
| Dtype | `input_tensor.dtype` |
| Layout | `input_tensor.layout` |
| Memory | exactly `memory_config` (BLOCK_SHARDED L1, its grid / shard shape / orientation); a newly allocated tensor, never `input_tensor` or a view of it |

**Tile geometry (single source, host):** the 2-D view is taken from the **padded** shape, alignment-aware:
`tensor_row_tiles = prod(padded_shape[:-1]) / 32` (= `d_0·d_1·ceil(d_2/32)`), `tensor_col_tiles = padded_shape[-1] / 32` (= `ceil(d_3/32)`). Production: 64 × 32.

---

## Blocking Model

The op is a parameterized gather. The **block** is one out shard; its extents are pinned by the caller's
`memory_config` (a shard hands you the block size — honor it). What stays this design's choice is (a)
which core issues each move (`PULL`: the out-block holder), (b) how a block is cut among the issuing RISCs
(`num_issuers` row bands), and (c) the transfer unit inside a band (the maximal run, never a tile).

### Axes

Both axes live in the 2-D view; `d_0..d_2` fold into `tile_row` because the placement itself is defined on
the 2-D view (blocks, shards and tile indices all are), so there is no separate batch axis to block.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `tile_row` | **independent** — `out[t]` is `in0[t]` alone, no result spans the axis | block: `block_height_tiles = out_shard_row_tiles` (clipped to `tensor_row_tiles` on the last block row); per issuer: `band_height_tiles = ceil(actual_block_height_tiles / num_issuers)` | block = whole out shard height (8 tiles, production); band = 4 tiles (`num_issuers = 2`) | `out_shard_row_tiles = memory_config.shard_spec.shape[0] / 32` (host, CT arg); `num_issuers` = one host constant `NUM_ISSUERS` | out block row `block_row` → the core `memory_config` places it on (`(x0+block_col, y0+block_row)` ROW_MAJOR, `(x0+block_row, y0+block_col)` COL_MAJOR); band `issuer_idx` → data-movement RISC `issuer_idx` of that core | knob-turn (`num_issuers` 1↔2; band axis) |
| `tile_col` | **independent** — same reason | block: `block_width_tiles = out_shard_col_tiles` (clipped to `tensor_col_tiles` on the last block column); per issuer: whole block width (not split) | whole out shard width (4 tiles, production) | `out_shard_col_tiles = memory_config.shard_spec.shape[1] / 32` (host, CT arg) | out block column `block_col` → as above | knob-turn (band split along columns instead of rows, perf lamp) |

**Reuse check (every operand × the chosen split).** The split is by out block. `in0` varies along it: the
blocks of `in0` that overlap one out block partition the tile grid together with the out blocks
(math.md F2), so **no input tile has two destinations** and no operand is reuse-shared. Nothing to broadcast.

**Stall-shadow check.** The only wait in the scheme is each issuer's final read barrier. Every run of the
band is issued before it, so no independent work remains to place in the stall; nothing else in the op waits
on anything.

**Intermediate data.** The scheme introduces no stage that produces data other than `out` itself (no CB, no
staging, no partial). There is no second table to re-run. No core collects or coordinates: every
out-block core pulls its own block, and no core computes on behalf of another.

**Phase 0 is multi-core:** all `out_block_rows × out_block_cols` cores pull concurrently (64 cores for
production, 56 for O1), each with `num_issuers = 2` RISCs on separate NoCs → 128 concurrent issuers for
production. The core count follows the out blocking (the block is the out shard, the holder issues), not
the tensor shape.

**Every knob is a parameter.** `out_shard_row_tiles`, `out_shard_col_tiles`, `in_shard_row_tiles`,
`in_shard_col_tiles`, `tensor_row_tiles`, `tensor_col_tiles`, `tile_stride_bytes` and `num_issuers` are
host-derived once per cache entry and passed as CT args; `block_row`, `block_col` per core as RT args;
`band_height_tiles`, band bounds, run counts and every address are derived from them in the kernel. No
literal tile count, shard size, grid origin or orientation appears in a kernel.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| — (no CB exists) | `transactions_in_flight` = runs issued per barrier | all runs of the band (one barrier per band; ≤ 8 runs per issuer in production) | maximal NoC overlap: the read latency is paid once per band, not per run. There is no ring buffer to size — every run lands at its final, distinct address in the out shard, so depth costs no L1. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| L1 NoC alignment (16 B, `ttnn.get_l1_alignment()`) | run start offsets = tile index × `tile_stride_bytes` | host asserts `tile_stride_bytes % l1_alignment == 0` (1088 = 68 × 16 for bfloat8_b); `tile_stride_bytes = input_tensor.buffer_aligned_page_size()` and must equal out's | misaligned NoC read: corrupted bytes or a hang |
| NoC packet size `NOC_MAX_BURST_SIZE` = 16384 B (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:287-290`) | run length | none needed: runs are issued with `noc_async_read_with_state` (any length; splits bursts internally, `dataflow_api.h:678-705`). **If** a realization switches to the `*_one_packet_*` calls, clamp every run to `floor(16384 / tile_stride_bytes)` tiles (15 for bfloat8_b) | a >16 KB one-packet read truncates the run (wrong tiles, silently) |
| Source-shard addressing (in0 `TensorAccessor`, CT args) | `in_block_rows × in_block_cols` shard ids ≤ worker cores | bounded by the device grid (≤ 1 shard per core) | — |
| Kernel-config region (BB-04) | RT/CT arg count | constant in the placement: 2 common RT (addresses) + 2 per-core RT (`block_row`, `block_col`) + fixed CT list + accessor CT args (bounded by the core count). The run list is **derived on device**, never shipped as args (a host-built run list grows with runs per block, up to every tile of the block) | launch refused in some build flavour |
| In-shard page order (`buffer.cpp:464-469`) | src/dst tile offsets | in-shard offset of tile `(h, w)` = `(h − shard_row0) · shard_col_tiles + (w − shard_col0)`, stride = the **allocated** shard width even on a tail shard | reading a tail shard's padding slots or landing tiles in the wrong slots |
| Shard geometry / tails | band and intersection extents | every extent clipped to `tensor_row_tiles` / `tensor_col_tiles`; tail slots of in0 shards are never read, tail slots of out shards never written (contents unspecified per spec) | stale or out-of-range tiles |
| Empty band | `band_height_tiles` when `actual_block_height_tiles < num_issuers` | an issuer whose band is empty issues nothing, runs its (zero-outstanding) barrier and returns | a hang or an out-of-range walk |

### Regimes

Named memory boundary: **core L1 ↔ NoC**. Minimum: every tile of `in0` is read once from its resident
shard and every tile of `out` written once into its shard; tiles whose source and target core coincide
move by NoC loopback (no inter-core crossing); **0 DRAM bytes**.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `PULL` — out-block holder reads its block's runs from the source shards, each run landing in its final slot | **built** | always: every legal pair of block placements (one regime for the whole domain; `in0` shards are resident before the program starts, so no handshake is needed) | out shard `[block_height_tiles × block_width_tiles]`, cut into `num_issuers` row bands of `band_height_tiles`; inside a band, one maximal run per (band × source shard) intersection when it spans full rows of both shards, else one run per tile row | **minimum.** Production: 1880 tiles = 2,045,440 B cross between cores in 1024 runs over 256 core pairs, 168 tiles = 182,784 B by loopback, 0 DRAM. I0: 2048 tiles loopback (128 runs); I3: 256 loopback; I5: 0 loopback (2,228,224 B cross); O1: 960 runs, 43 loopback | the fixed costs per band — kernel launch and arg fetch, accessor construction, one NoC source-state programming per source shard, one barrier — paid once per out shard instead of per piece; a full-width intersection becomes one transfer (I0/I3/I5: one 16-tile run per band instead of 8 per-row runs). The block is the caller's out shard, so it is already at its bound. |
| `PUSH` — in-block holder writes its block's runs into the out shards that need them | **deferred** — `PULL` already serves every placement pair `PUSH` would, with identical bytes. The only delta is which end issues: production critical core 16 runs (pull, 8/RISC) vs 14 (push, 7/RISC), 64 vs 110 issuing cores | always (same domain) | in shard `[in_shard_row_tiles × in_shard_col_tiles]` in row bands | minimum (same bytes, same runs: 1024 in production) | same terms, plus a write-ack drain per band. **Reachable:** the same host CT geometry and the same intersection walk with the roles of the two placements swapped; `noc_async_read*` → `noc_async_write`, kernel core range = in-block cores, one `noc_async_write_barrier` per band (BB-03). No semaphores: nothing reads `out` inside the program. Settled by one measurement on production: pull vs push kernel duration. |
| `HYBRID` — each (source, target) pair's runs assigned to whichever end has the lighter issue load | **deferred** — `PULL` covers every shape; this only rebalances issue work across the union of both grids (production: 110 cores instead of 64) | always | as `PULL` and `PUSH` | minimum (same bytes) | same as `PULL`, plus a host-side assignment rule. **Reachable:** both kernels walk the same intersections and filter by a CT assignment predicate; same structure as `PUSH` + `PULL` side by side. |
| `TILE_AT_A_TIME` — one NoC transaction per tile | **rejected** — superseded by `PULL`'s maximal runs (2048 transactions instead of 64–1024; per-transfer cost dominates the short-run production regime, `risks.md` R3) | — | 1 tile | minimum bytes, but 2048 transactions | — |
| `SINGLE_ISSUER` — one core moves every tile | **rejected** — dead end (X3/X7): no better regime is built on top of it; `PULL` spreads the same work over every out-block core | — | whole tensor | minimum bytes, one issuer | — |
| `STAGED` — through DRAM, an interleaved intermediate, a third core's buffer, or another library device op (e.g. a sharded↔interleaved round trip) | **rejected** — forbidden (rule 1, BB-01/02); dead end: +2 DRAM crossings of 2,228,224 B and a second program | — | — | in0 crosses DRAM twice (write + read) | — |
| `CONVERT` — unpack/repack through compute (tilize/untilize, typecast, copy_tile) | **rejected** — forbidden (rule 1: no unpack/convert/repack); a compute pass the golden values cannot see; may regroup bfloat8_b exponents | — | — | minimum bytes plus a full unpack/pack pass | — |
| `HOST_ROUNDTRIP` — read values on host, upload in the new placement | **rejected** — forbidden (BB-01) | — | — | PCIe both ways | — |
| `SPLIT_DEPENDENT + COMBINE` | **rejected** — not applicable: no dependent axis exists (no result spans any axis) | — | — | — | — |
| `BROADCAST` (multicast a reuse-shared operand) | **rejected** — not applicable: no tile has two destinations; a multicast would deliver bytes every receiver but one discards | — | — | — | — |

Single built regime ⇒ no selection function and no regime-pinned tests are needed. BB-08 still applies: the
host reports the plan (`PULL`, `num_issuers`, core roles) for any call (Host Contract below).

### Traffic ranking

Production pair (in (0,0)-(10,9) 7×3-tile shards → out (0,0)-(7,7) 8×4-tile shards), bfloat8_b, 1088 B/tile.

| Candidate split / regime | DRAM crossings (in0 / out) | Core-to-core NoC bytes | Loopback bytes | Issuers / max runs per issuer | Verdict |
|--------------------------|----------------------------|------------------------|----------------|-------------------------------|---------|
| by out block (`PULL`), 2 issuers/core | 0 / 0 | 2,045,440 | 182,784 | 128 / 8 | **chosen** |
| by in block (`PUSH`), 2 issuers/core | 0 / 0 | 2,045,440 | 182,784 | 220 / 7 | tie on bytes; deferred |
| union of both ends (`HYBRID`) | 0 / 0 | 2,045,440 | 182,784 | ≤ 220 / ≥ 5 (1024 runs over 220 RISCs; the actual max depends on the assignment rule) | tie on bytes; deferred |
| through a third core / DRAM / interleaved (`STAGED`) | 1+1 extra (2 × 2,228,224 B) | ≥ 2× | — | — | rejected (forbidden) |
| dependent-axis split + combine | n/a | n/a | n/a | n/a | no dependent axis |

The three legal splits move identical bytes on every tier, so the ranking ties and the tiebreaker decides:
`PULL` makes every out tile the responsibility of the core that owns it, so completion is one local read
barrier per issuer (no cross-core completion, no write acks), and the per-core work is bounded by the out
shard the caller sized. Occupancy is not used to break the tie.

### Block schedule

Logical schedule of one issuer (`issuer_idx ∈ [0, num_issuers)`) on the core holding out block
`(block_row, block_col)`. `num_blocks_this_core == 1`: the block is the whole out shard (caller-pinned).

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // == 1
    pull_band(block_idx, issuer_idx);   // issue every run of this issuer's row band of the out block
    drain_band();                        // one read barrier: every run of the band has landed (BB-03)
}
```

| Block operation | Block shape it acts on | Resident across it | Intended frequency of fixed costs |
|-----------------|------------------------|--------------------|-----------------------------------|
| `pull_band` | rows `[band_row_begin, band_row_end)` × cols `[block_col_begin, block_col_end)` of the out block, where `band_row_begin = block_row_begin + issuer_idx · band_height_tiles`, `band_row_end = min(band_row_begin + band_height_tiles, block_row_end)`, `block_row_end = min((block_row+1) · out_shard_row_tiles, tensor_row_tiles)`, `block_col_end = min((block_col+1) · out_shard_col_tiles, tensor_col_tiles)` | in0 shards (model's, read-only, on their own cores); the out shard (written in place) | args + accessor: once per kernel. NoC source state (`noc_async_read_set_state`): once per (band × source shard) intersection — ≤ 4 per issuer in production. Read issue (`noc_async_read_with_state`): once per maximal run — ≤ 8 per issuer in production. No CB handshake, no semaphore, no multicast, no barrier inside. |
| `drain_band` | the band | same | `noc_async_read_barrier`: exactly once per issuer. |

`pull_band` contract (the run walk), all in tiles of the padded 2-D view:

1. Source shards overlapping the band: `src_block_row ∈ [band_row_begin / in_shard_row_tiles, (band_row_end−1) / in_shard_row_tiles]`, `src_block_col ∈ [block_col_begin / in_shard_col_tiles, (block_col_end−1) / in_shard_col_tiles]`, walked row-major. All divisors are CT args (constant-divisor arithmetic).
2. Per source shard: intersection rows `[h0, h1) = band ∩ [src_block_row · in_shard_row_tiles, …)`, cols `[w0, w1) = [block_col_begin, block_col_end) ∩ [src_block_col · in_shard_col_tiles, …) ∩ [0, tensor_col_tiles)`. `shard_id = src_block_row · in_block_cols + src_block_col`; its NoC address comes from the in0 `TensorAccessor` (`get_shard_noc_addr(shard_id, 0, noc)` with **this issuer's NoC id**) → `noc_async_read_set_state` once.
3. Offsets: `src_off(h, w) = (h − src_block_row·in_shard_row_tiles) · in_shard_col_tiles + (w − src_block_col·in_shard_col_tiles)`; `dst_off(h, w) = (h − block_row·out_shard_row_tiles) · out_shard_col_tiles + (w − block_col·out_shard_col_tiles)`; byte address = base + off · `tile_stride_bytes`, where the source base is the low 32 bits of the shard NoC address and the destination base is out's buffer address.
4. Runs: if `w1 − w0 == in_shard_col_tiles` **and** `w1 − w0 == out_shard_col_tiles` (full rows of both shards ⇒ contiguous in both), one run of `(h1 − h0) · (w1 − w0)` tiles from `src_off(h0, w0)` to `dst_off(h0, w0)`; otherwise one run of `w1 − w0` tiles per row `h ∈ [h0, h1)`.
5. Run bytes = run tiles × `tile_stride_bytes`. Same-core intersections use the same call (NoC loopback); no special path.

Production per issuer: 4 tile rows, 2–4 source shards, 8 runs of 1–3 tiles; I0/I3/I5: 1 source shard, 1 run of 16 tiles; O1: ≤ 6 source shards, ≤ 12 runs.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Split reader (`num_issuers = 2`, catalog `split_reader`) | Production is 16 short reads per out core, likely RISC-issue-bound — the situation the catalog's split pays in; but if the NoC, not issue, binds, the second kernel only adds launch/arg cost | `num_issuers = 1` (one kernel per core on NOC_0) vs 2, production case |
| Band axis | Row bands split production evenly (4 + 4 rows, 8 + 8 runs), but a tail block of odd height or a shard with few rows and many columns balances poorly; row bands also cut I0/I3/I5's single 32-tile run in two | column bands, or round-robin assignment of runs to issuers in walk order |
| NoC per issuer (catalog `noc_placement`) | Default issuer 0 → RISCV_1/NOC_0, issuer 1 → RISCV_0/NOC_1 spreads link load over both NoCs; NoC choice measured 2.5–4.8× on DRAM lines, untested for L1→L1 gathers | both issuers on NOC_0; swapped assignment |
| Source-core stagger (catalog `bank_stagger` analog) | Every puller walks its source shards row-major from its block's top-left; each production in0 shard feeds up to 4 pullers that may hit it at the same moment | rotate each core's walk start (e.g. start at source shard `(block_row + block_col) mod n_intersections`) |
| Stateful vs plain issue | `set_state` per source shard saves the coordinate write on later runs, but with 2 runs per source shard (production) the saving is small and the state call has its own cost | plain `noc_async_read` per run |

Overlap lamp: not attached — there is no compute to overlap and every run of a band is in flight before its
single barrier. Grid-synchronization lamp: not attached — no collective, barrier or semaphore spans cores.

---

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| in0 shard on its core | stored tiles (Bfp8_b, 1088 B, row-major tile order inside the shard) | resident L1 (the model's tensor) | read in place by remote NoC reads; never copied to a CB; never written |
| source L1 → target L1 | raw bytes of whole-tile runs | NoC read issued by the target core: `noc_async_read_set_state` per source shard + `noc_async_read_with_state` per run | issuer 0 on RISCV_1 / NOC_0, issuer 1 on RISCV_0 / NOC_1; same-core runs are loopback reads |
| out shard on target core | stored tiles, byte-identical | written at the final slot by the read itself | no CB, no staging; tail slots untouched |
| completion | — | `noc_async_read_barrier` once per issuer before return | nothing else in flight (no writes, no atomics) — BB-03 |

**Kernels.** One dataflow kernel source, instantiated twice per core (two `KernelDescriptor`s on the same
core range): issuer 0 with `ReaderConfigDescriptor` (RISCV_1, NOC_0), issuer 1 with
`WriterConfigDescriptor`-style placement (RISCV_0, NOC_1), differing only in the CT arg `issuer_idx`. No
compute kernel. Kernel core range = the out-block cores: the rectangle from out's grid start, `out_block_cols`
wide × `out_block_rows` tall for ROW_MAJOR (`out_block_rows` wide × `out_block_cols` tall for COL_MAJOR) —
one rectangle (BB-17); grid cores beyond the block count run nothing.

**Placement axis vs TARGET.** TARGET has no `memory_layout` axis: both placements are arguments, and every
legal block placement on either side is in Phase 0 (rule 2). There is no placement unlock to classify.

**Deferred regimes' contracts.** `PUSH`: source core writes `noc_async_write(src_l1, get_noc_addr(target_x,
target_y, out_addr + dst_off·stride), bytes)` — out's address is lockstep (same on every core), target core
coordinates from a second accessor over `out`; one write barrier per issuer; no semaphore. `HYBRID`: both
kernel sets as above, each run assigned by a host CT predicate.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one out block = the whole out shard (`block_height_tiles × block_width_tiles`), cut into `num_issuers` row bands |
| Grid | the out-block cores (one rectangle, see Dataflow Strategy); production 8×8 = 64 cores, O1 8×7 = 56 cores |
| Per-core work | core of out block `(block_row, block_col)`: rows `[block_row·out_shard_row_tiles, min(…+out_shard_row_tiles, tensor_row_tiles))`, cols likewise; issuer `i` takes band `i` |
| Remainder | `out_block_rows = ceil(tensor_row_tiles / out_shard_row_tiles)`, `out_block_cols = ceil(tensor_col_tiles / out_shard_col_tiles)`, `in_block_rows/cols` likewise; last block row/col clipped (`min`), tile counts from the padded shape (`ceil` per image) |
| Assignment API | none: `memory_config` pins block → core (`split_work_to_cores` does not apply). Per-core RT args `(block_row, block_col)` are the inverse of the placement formula over the trimmed rectangle. |

Single regime: no selection predicate.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| — none — | — | — | 0 | Every run lands at its final address in out's shard (pattern 1, pack into the destination), and in0 is read in place on its own core. A CB backed on out's shard (`cb_descriptor_from_sharded_tensor`) is not needed: out's base address is a common RT arg (it is the same on every core), so nothing would use the CB's FIFO. | — | — | — | — |

The op's only L1 is out's own shard; see `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input (semantic name, pages, state) | Output (semantic name, pages) | State after |
|---|-----------------|-------------|---------|-------------------------------------|-------------------------------|-------------|
| 1 | `pull_band` | `band_height_tiles × block_width_tiles` (clipped) | TensorAccessor (source shard address) + raw NoC read API | `in0` shards on their cores (tensor buffer, not a CB; all pages resident, read-only) | `out` shard of this core (tensor buffer, not a CB; the band's tiles) | all runs of the band in flight |
| 2 | `drain_band` | the band | raw NoC barrier | — | — | band landed; kernel returns with nothing outstanding |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input | Output | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|-------|--------|------------------------------|
| `pull_band` (source address) | helper | `TensorAccessor::get_shard_noc_addr(shard_id, offset, noc)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:229-233` | `TensorAccessorArgs<…>` from `ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()` appended last in the CT list; base address = common RT arg; `noc` = this issuer's NoC id | in0 | — | none (placement read from in0's own distribution spec) |
| `pull_band` (state) | raw_api | `noc_async_read_set_state(src_noc_addr, noc)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:644` | shard NoC address from the accessor | in0 shard | — | — |
| `pull_band` (run) | raw_api | `noc_async_read_with_state(src_local_l1_addr, dst_local_l1_addr, size, noc)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:677-707` | `size = run_tiles · tile_stride_bytes` (any length) | in0 shard | out shard | run length follows `band_height_tiles` (full-width merge) and the shard geometry |
| `pull_band` (alt., plain issue lamp) | raw_api | `noc_async_read(src_noc_addr, dst, size, noc)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547-555` | — | in0 shard | out shard | — |
| `drain_band` | raw_api | `noc_async_read_barrier(noc)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1921` | this issuer's NoC | — | — | — |
| args | raw_api | `get_arg_val`, `get_common_arg_val` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:148`, `:168` | per-core `(block_row, block_col)`; common `(in0_addr, out_addr)` | — | — | — |
| zones (BB-21) | raw_api | `DeviceZoneScopedN(name)` | `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012-1047` | `"G6R-PULL-ISSUE"` around `pull_band`, `"G6R-PULL-DRAIN"` around `drain_band` | — | — | — |

**Helpers considered and rejected (for every `raw_api` row above).**

| Candidate | File:Line of the mismatch | Why it cannot express the shape |
|-----------|---------------------------|---------------------------------|
| `dataflow_kernel_lib::set_read_state<transfer_size>` / `read_with_state` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57`, `:66-75` | self-aimed only (source = `my_x/my_y` via `local_addr`), and single-packet with the size a template parameter; runs here are remote and runtime-length (1 tile … a whole shard). |
| `dataflow_kernel_lib::set_read_trid` / `async_read_barrier_with_trid` | `local_copy_helpers_dataflow.hpp:147`, `:158` | for retiring batches while later ones stay in flight; this op has one batch per issuer and no buffer to recycle. |
| `McastArgs` / `ttnn.Mcast` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp` | no tile has two destinations. |
| `ccl_helpers_dataflow` | `ttnn/cpp/ttnn/kernel_lib/ccl/ccl_helpers_dataflow.hpp:8-20` | multi-device fabric egress; this op is one chip, L1→L1. |
| `tilize` / `untilize` / `eltwise` copy | `tilize_helpers.hpp`, `untilize_helpers.hpp`, `eltwise_convenience.hpp` | they unpack and repack — forbidden by rule 1. |
| `noc_async_read_shard(shard_id, accessor, dst)` | `tech_reports/tensor_accessor/tensor_accessor.md:139-142` | copies a whole source shard; an out block takes only the pieces of a source shard that overlap it. |

No compute phase exists, so no compute helper applies and `compute_kernel_hw_startup()` is not called.

---

## Host Contract

### Registry (op file)

| Item | Value |
|------|-------|
| `INPUT_TAGGERS` | `{}` — the spec names no shape-derived axis; placements are arguments, never axes (axes.py imports this dict) |
| `SUPPORTED` | `{"dtype": [ttnn.bfloat8_b], "layout": [ttnn.TILE_LAYOUT]}` (implementer writes it; TARGET = Phase 0) |
| `EXCLUSIONS` | `[]` |
| `validate(input_tensor, memory_config)` | builds axes `{dtype, layout}` from `input_tensor`; raises `UnsupportedAxisValue` / `ExcludedCell` only; first line of `g6_reshard` |

### Validation (after `validate()`; raises `ValueError` / `RuntimeError`, never `NotImplementedError`)

| Check | Error |
|-------|-------|
| `input_tensor` not on device, not BLOCK_SHARDED, not L1, or without a shard spec | ValueError |
| `memory_config` not BLOCK_SHARDED, not L1, or `shard_spec is None` | ValueError |
| `memory_config` shard dims not multiples of 32 | ValueError |
| `memory_config` grid not exactly one rectangle; fewer cores along a direction than its block count needs (`out_block_cols` along x for ROW_MAJOR, `out_block_rows` along x for COL_MAJOR, and the other on y) | ValueError |
| `memory_config` grid outside `device.compute_with_storage_grid_size()` | ValueError |
| rank ≠ 4, or the 2-D view's H or W not a multiple of 32 | ValueError |
| out shard bytes `out_shard_row_tiles · out_shard_col_tiles · tile_stride_bytes > 1347776` (BB-05: the L1 the model leaves free; out's shard is the op's only L1) | RuntimeError |

A legal placement is never refused (rule 2): no orientation, origin, tail or shard-shape check beyond the list.
Every message starts with `g6_reshard:` — the acceptance test matches that text through the repo's
`expect_error` fixture. All checks run in the op's host code before any allocation or dispatch.

### Dispatch, cache, plan (BB-01, BB-07, BB-08, BB-21)

| Item | Decision |
|------|----------|
| Program count | exactly one `ttnn.generic_op([input_tensor, out], descriptor)` per call; `out = ttnn.allocate_tensor_on_device(shape, dtype, layout, device, memory_config)` (positional) — always a new tensor, even when the placements are equal (I0) |
| Cache key | `(device id, padded shape, dtype, layout, tile_stride_bytes, in grid start/end, in shard shape, in orientation, out grid start/end, out shard shape, out orientation, num_issuers)` |
| Repeat call (BB-07) | module-level dict `key → ProgramDescriptor`; on a hit the host only rewrites the two common-RT address words (`in0_addr`, `out_addr`) of each kernel and dispatches — no descriptor rebuild. generic_op hashes CT args and core ranges (`generic_op_device_operation.cpp:78-147`), so every placement value lives in CT args and two different placements never share a compiled program; on a hit generic_op re-applies RT values from the passed descriptor (`mesh_device_operation_adapter.hpp:345-348`). |
| Plan report (BB-08) | `g6_reshard_plan(input_tensor, memory_config) -> dict`: `{"regime": "PULL", "num_issuers": …, "cores": {(x, y): role}}`; production always gets `PULL` |
| Roles (BB-21) | role per core: `"pull:issuer0/NOC_0+issuer1/NOC_1"` for out-block cores, `"idle"` otherwise; dumped by the same function. Zones as in API Mapping; they compile out of the plain build. |
| Integrity | no `ttnn.reshard`, `to_memory_config`, `interleaved_to_sharded`, `sharded_to_interleaved`, typecast or `to_torch` on the path; no reads of tensor values on host. |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| NoC coordinate space | issuer 1 runs on NOC_1; a shard address computed for NOC_0 reads a different core | `get_shard_noc_addr(shard_id, 0, noc)` and every read/barrier take the issuer's own NoC id (a CT arg), never the default |
| Stale NoC state | `noc_async_read_with_state` reuses the coordinates of the last `set_state` | call `set_state` at the start of every source-shard intersection; keep `skip_cmdbuf_chk` at its default `false` |
| Tail shards (production in0 has both a 1-tile-high last block row and a 2-tile-wide last block column; O1's out has a 2-tile-wide last block column) | the allocated shard width, not the actual one, is the row stride; full-width merging must not fire on a tail | offsets use `in_shard_col_tiles` / `out_shard_col_tiles` strides; merge condition compares the intersection width with both **allocated** widths, which a tail intersection never equals |
| Orientation / origin (I3, I5, O1) | block→core is the placement formula; a ROW_MAJOR assumption sends blocks off the diagonal to the wrong cores | in0's core comes from its own `TensorAccessor` (its distribution spec); out's per-core `(block_row, block_col)` come from the inverse of `memory_config`'s formula with its orientation |
| Cache replay (I0 vs I3 differ only in input orientation; P vs O1 only in out placement) | a key missing any placement field replays a wrong program | key covers both placements; geometry in CT args so generic_op's hash differs too |
| Address patching | `out` is new every call and `in0` may be another tensor at the same placement | both address words rewritten on every call, hit or miss |
| Aliasing (I0) | same placement on both sides tempts returning `input_tensor` | always allocate `out`; the walk then copies every tile by loopback |
| Padded geometry | a shape like `(1, 2, 16, 64)` passes the 2-D check yet is tile-padded per image | tile counts from `padded_shape`, never from the logical shape |
| Empty band | an out block shorter than `num_issuers` tile rows | empty band issues nothing and still returns cleanly |
| Accessor shard indexing | the walk's `shard_id = src_block_row · in_block_cols + src_block_col` assumes in0's distribution spec is the 2-D view's block grid; for `d_0 = d_1 = 1` (production) any rank-4 form coincides with it, but a rank-4 spec over `d_0·d_1 > 1` images could index shards per image | host asserts, at cache-miss time, that in0's shard count equals `in_block_rows · in_block_cols`; kernel `ASSERT(shard_id < num_shards)` under `--dev` |

## Catalog entries consulted

| Entry | Informed |
|-------|----------|
| `read_coalescing` | transfer unit = maximal contiguous run (one call per full-width intersection) |
| `split_reader` | `num_issuers = 2` default + its lamp (pays only when issue-bound) |
| `noc_placement` | issuer→NoC assignment default + lamp |
| `bank_stagger` | source-core stagger lamp |
| `trid_double_issue`, `double_buffer` | not used: no buffer to recycle; one barrier per issuer already keeps the band in flight |

## Structural impossibilities

None beyond `feature_spec.INVALID = []`.

## Acceptance test

`tests/ttnn/unit_tests/operations/g6_reshard/test_g6_reshard.py`: the production shape alone (BB-22) on the
five placement pairs the rules require (P, I0, I3, I5, O1), graded by bit equality (`helpers.TOLERANCES`
is `("exact", None)`), plus out's placement, no aliasing, input unchanged, all placements in one process,
repeat determinism, and the spec's validation errors. The ≥4-shape sweep of the generic template is replaced
by the placement sweep: the spec restricts the suite to the production shape.
