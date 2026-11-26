# L1 Ledger: g6_reshape

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes: `op_design.md` → Blocking Model
(`t_out` → `block_tiles`, `r` → rows of a tile, `fh` → `segments_per_strip`, `e` → `SEGMENT_ELEMS`, source axes `g_in`/`c_in`).
No compute kernel → no DEST, no `fp32_dest_acc_en`; every page holds raw bytes that never pass an unpacker or packer.

Inventory (Rule 1): two buffers. A pure move needs at least the landing buffer that DRAM reads target and DRAM writes drain
(the NoC cannot move DRAM → DRAM); the staging ring exists only because a 32-B face row cannot always be read straight into
its destination under the 64-B DRAM read congruence (mechanism cap). Pattern "pack into the destination" is applied: every
congruent segment is read straight into `cb_output_tiles`; there is no input-tile buffer.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_output_tiles` (page = `tile_bytes`) | `OUTPUT_DEPTH · block_tiles` = `2 · min(max_core_tiles, BLOCK_SLOT_BYTES_CAP // tile_bytes)` | `2 · block_tiles` when a core has ≥ 2 blocks (reader fills slot b+1 while writer drains slot b); `block_tiles` on every golden shape (1 block per core) — the second slot is then unused capacity, kept because the depth knob is shape-independent and costs ≤ 72 KiB on the production shapes | `{t_out: spans → block_tiles, r: spans → 32 (tile image; valid rows ≤ 32), fh: spans → 2, e: spans → 16, g_in: streams (not resident), c_in: streams (not resident)}` | input dtype, raw bytes (bfloat16 in Phase 0) | reader | writer | whole kernel; ring of `OUTPUT_DEPTH` block slots | cannot share with `cb_granule_staging`: both are live during `assemble_block` (staging is the source, the slot the destination of the same copies) |
| `cb_granule_staging` (page = `granule_bytes` = `max(NOC_DRAM_READ_ALIGNMENT_BYTES, face_row_bytes)`) | `STAGING_GRANULES + 1` = 513 (the +1 page lets the base be aligned up to 64 B) | ≤ staged pieces between two flushes ≤ `STAGING_GRANULES`; on golden shapes ≤ `2 · ceil(U / n_cores)` ≤ ~64 granules — the ring is sized for R2's worst case (up to 16 pieces per segment) so a flush stays rare, not for the golden live set; the gap is the explicit transfer-batching decision of the depth knob | `{t_out: streams (flushed per block or when full), r: streams, fh: streams, e: streams, g_in: streams → one granule per distinct source face row, c_in: streams}` | raw bytes | reader (private scratch; written by NoC reads it issues) | reader (copies out after its own barrier) | whole kernel | cannot alias `cb_output_tiles` (concurrent, above); cannot live in output padding rows (a full tile has none). Not a CB pipe: no push/pop, so it creates no cross-kernel credit |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `tile_bytes` | 2048 (bf16), 4096 (fp32, R7) | dtype ∈ TARGET |
| `BLOCK_SLOT_BYTES_CAP` | 262,144 B (host constant) | — |
| `block_tiles` | ≤ `BLOCK_SLOT_BYTES_CAP // tile_bytes` = 128 (bf16) / 64 (fp32); ≥ 1 | `min(max_core_tiles, cap)`; `max_core_tiles ≥ 1` because `n_cores ≤ U` |
| `max_core_tiles` | ≤ `ceil(U / n_cores) + 1` (a run of strips touches at most one more tile than its rows fill) | strip runs are contiguous |
| `OUTPUT_DEPTH` | 2 | host constant |
| `STAGING_GRANULES` | 512 | host constant |
| `granule_bytes` | 64 (BH bf16/fp32), 32 (WH bf16), 64 (WH fp32) | `NOC_DRAM_READ_ALIGNMENT_BYTES` of the arch |
| `U`, `N`, `W`, `H` | unbounded (no upper limit in the contract) | appear in **no** capacity expression — only in loop trip counts and RT args |

## Total per-core footprint

`L1 = OUTPUT_DEPTH · block_tiles · tile_bytes + (STAGING_GRANULES + 1) · granule_bytes`

- scales with `block_tiles` (the `t_out` knob) and `OUTPUT_DEPTH` only; staging is constant.
- worst case, any shape: `2 · 262,144 + 513 · 64 = 557,120 B` ≤ 1,347,776 B (BB-05) — no shape can OOM, so no host refusal by size exists.
- production worst case (`[1,1,16,4608]→[16,1,4608]` or `[16,1,5120]→[1,1,16,5120]` at a ~110-core grid: ≤ 25 tiles per core): `2 · 25 · 2048 + 32,832 ≈ 135 KiB`.

## Data-movement budget

Minimum (boundary DRAM): every logical input byte read once, every logical output byte written once. Bytes below for bf16,
Blackhole (64-B read congruence: a 32-B segment whose source and destination face-row parities differ is read as a 64-B granule).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | 1 × its valid segments (padding rows never read); staged segments cost one 64-B granule instead of 32 B; R2 reads each distinct granule once per block (last-granule memo) | gather on the output owner reads each source segment exactly once: the index map is a bijection and strip runs partition the output | none (only a granule straddling a core boundary in R2 is read by two cores: ≤ 1 granule per boundary) |
| `output` | 1 × its valid face rows (padding rows never written; padding columns of a partial segment ride along inside the 32-B face row) | the writer emits one write per (tile, face, row run) of the slot | none |

Per production shape (valid bytes in → out; reads / writes issued):

| Case | x valid bytes (whole tiles) | Reads (32 B direct / 64 B staged) | out valid bytes (whole tiles) | Writes |
|------|-----------------------------|-----------------------------------|-------------------------------|--------|
| `[16,1,5120]→[1,1,16,5120]` | 160 KiB (5.24 MB) | 5120 segments, ~half staged → ~240 KiB | 160 KiB (320 KiB) | 320 face runs of 512 B |
| `[1,1,16,4608]→[16,1,4608]` | 144 KiB (288 KiB) | 4608 segments, ~half staged → ~216 KiB | 144 KiB (4.7 MB) | 4608 face rows of 32 B |
| `[1,16,6,256]→[1,1,16,1536]` | 48 KiB (256 KiB) | 1536 segments → 48–72 KiB | 48 KiB (96 KiB) | 96 face runs of 512 B |
| `[16,1,64]→[1,1,16,64]` | 2 KiB (64 KiB) | 64 segments | 2 KiB (4 KiB) | 4 face runs |

Total per tier (chosen split): DRAM ≈ 1.0–1.5 × logical bytes read + 1.0 × logical bytes written; cross-core 0; core-local:
one L1→L1 copy of each staged segment.

> Cheapest-traffic split considered: contiguous output-strip runs with gather on the owner (R1/R2) — it IS the minimum
> (up to granule rounding) on both tensors; the input-driven scatter (R4) ties on bytes and differs only in transaction
> counts. Implemented: output-strip gather (R1/R2). The implemented split is also the cheapest. R4 is deferred because the
> built regime already reaches minimum bytes on every shape R4 would serve; its gain is a read-transaction count bounded by
> ≤ 2·ceil(U/n_cores) per core, the perf pass's first experiment.
