# Operation Requirements: tilize

## Definition
- **Formula**: `out[i, j] = x[i, j]` over the data region (`cast(x[i, j], out_dtype)` when `dtype=` is given); `out[p] = pad_value` over the pad region. Output layout is TILE at the requested tile height; the logical shape never changes.
- **PyTorch Reference**: identity — `def ref(x): return x.clone()`; padded: `torch.nn.functional.pad(x, pads, value=pad_value)` compared against `out.cpu().to_torch_with_padded_shape()`.
- **Import Path**: `from ttnn.operations.tilize import tilize`
- **Function Signature**:
  ```python
  tilize(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig | None = None, *,
         dtype: ttnn.DataType | None = None, low_l1: bool = False,
         output_padded_shape: list[int] | ttnn.Shape | None = None,
         pad_value: float | int | None = None, tile: ttnn.Tile | None = None) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype / output_dtype**: [bfloat16] / [bfloat16]
- **SUPPORTED layout**: ROW_MAJOR input (`in_tile_height = none`) → TILE output
- **SUPPORTED shape-derived axes**: alignment = tile_aligned; rank ∈ {2, 3, 4, 5, 6}; tile_grid ∈ {single_tile, small, tall_narrow}
- **SUPPORTED op-specific axes**: low_l1 ∈ {False, True}; buffer ∈ {dram_to_dram, dram_to_l1, l1_to_l1, l1_to_dram} (interleaved only); tile_height ∈ {32, 16, 8, 4, 2, 1}; shard_api / orientation / pad_mode / pad_value = none
- **Cores**: full grid, `interleaved_row_split` (`split_work_to_cores(grid, R, row_wise=True)`), bank-coalesced reader with bank stagger
- **Compute config**: default (`fp32_dest_acc_en=False`), `compute_kernel_lib::tilize` (fast tilize at 32×32, `tilize_block` fallback for tiny tiles)
- **Golden baseline**: 238 supported_pass / 0 supported_fail / 0 xpass_drift; 56 xfail_wrong_mode are harness-side fp8 input-creation failures (see verification_report.md)
- **Perf (BH p150b, 110 cores)**: `[1,1,16384,64]` 11.35 µs (reference 11.85 µs)

### [x] Refinement 1 — Full work geometry: 2-D split + padding

**Goal**: add `short_wide` and `square_large` to `SUPPORTED["tile_grid"]` by building the `interleaved_2d_split` regime (op_design.md Regimes), and add padding: `pad_mode` ∈ {auto, explicit}, `pad_value` ∈ {zero, positive, negative}, `alignment` ∈ {w_non_aligned, h_non_aligned, hw_non_aligned}, `rank` ∈ {0, 1} (rank 0/1 are only reachable with a pad; the pad synthesizes `[32,32]` / `[32, round_up(W,32)]`). Padding is filled in-kernel in `load_sticks_block` (op_design.md Dataflow, "Pad fill") — W tail, H tail, whole pad tile-rows/cols — with the fill encoded in the input format (negative integer fill bit_cast, never truncated). Output: logical shape = input's, padded shape = `output_padded_shape` or the tile-round. Interleaved placements only in this refinement (the sharded crosses land in Refinement 2).

**Implementation skill**: /interleaved-parallel

**Verifier notes**:
- The 2-D split is a **knob-turn** — the kernels already take the per-core rectangle `(row_start_tile, core_rows_tiles, col_start_tile, core_cols_tiles)`; the host selector (`R < num_cores and R·C_blocks ≥ num_cores` → choose `cores_r × cores_c`) is the work. Padding is a reader extension on the same scheme. Bundled because the REQUIRED_CROSSES `short_wide × w_non_aligned`, `short_wide × pad_mode=auto` and `square_large × pad_mode=auto` (with the leading-dim fold + H tail: `[8,1,249,2048]`) put the W/H-tail arithmetic on the split axis.
- Must land before Refinement 2: the `sharded_full_grid` fallback regime *is* the 2-D split with sharded accessors (true dependency).
- When columns are split, segment reads must stay DRAM/L1-aligned (`dram_align_unit_tiles` clamp — op_design.md Mechanism caps) and the bank-coalesced reader must keep its `num_col_blocks == 1` / whole-stick precondition (column-split cores read segments, not whole sticks). Keep both dataflow halves batched on the new path — no per-stick barrier regressions (performance-conformance bar).
- `short_wide_l1_forcing` / `low_l1_forcing_width` (`C = 256`) must not OOM at either `low_l1` setting; the A/B in `helpers.run_tilize` requires bit-identical output.
- `SUPPORTED["tile_grid"]` may list `short_wide` / `square_large` only once the split actually reaches the grid on them (report the core count reached on `[1,1,32,2048]` and `[1,1,2048,2048]`).

**Done when**: every interleaved golden cell with `tile_grid ∈ {short_wide, square_large}` or `pad_mode ≠ none` passes at bf16, the program-cache cases `short_wide_canonical`, `square_large`, `auto_hw_tails_negative_fill` pass, and the acceptance + knob tests stay green.

**Outcome**: landed. 26/26 targeted bf16 interleaved golden cells + 5 program-cache cases pass; translated padding/to_layout slice 421 pass. Cores reached (BH, 110): `[1,1,32,2048]` 64 cores (one tile each), `[1,1,2048,2048]` 110 cores (22×5). Device ns (row-only → 2-D): `[1,1,32,2048]` 5.6 → 4.1 µs, `[1,1,32,8192]` 12.2 → 6.1 µs (86 cores), `[1,1,2048,2048]` 43.2 → 39.6 µs (~420 GB/s, near DRAM roofline). short_wide stays latency-bound (32 × 64 B segment reads per core); next lever would be fewer, wider segment reads per core (e.g. cores_c capped so segments ≥ 128 B) — not tried, outside this refinement's scope.

### [x] Refinement 2 — Sharded placements (all schemes, both shard APIs, crossovers)

**Goal**: add `shard_api` ∈ {legacy_2d, nd}, `out_scheme` ∈ {HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED, ND_SHARDED}, `orientation` ∈ {ROW_MAJOR, COL_MAJOR} to SUPPORTED, covering same-spec, interleaved↔sharded crossovers, cross-spec (height in / width out), DRAM-sharded output, partial final shards and physical shard padding — the four sharded regimes in op_design.md (`sharded_same_spec_in_place`, `sharded_input_pinned`, `sharded_output_pinned`, `sharded_full_grid`).

**Verifier notes**:
- No implementation skill covers placement. **Local shards must be consumed natively**: a core whose input shard is resident in its own L1 backs `cb_rm_sticks` (and, for a same-spec sharded output, `cb_tiles_out`) on the shard via `ttnn.cb_descriptor_from_sharded_tensor` — never re-reads its own shard through a `TensorAccessor`. Same-spec = zero NoC/DRAM traffic, one block = the whole resident shard (clamped by the 255-tile fast-tilize cap). TensorAccessor stays for interleaved sides and genuinely non-local reads in crossovers/cross-spec (sharded accessor args). A shard needing reshaping (partial final shard, shard width ≠ block) is copied locally into a staging CB.
- No dependent axis exists, so there is no cross-core combine; the hard part is the regime selector and per-core rectangle mapping (COL_MAJOR re-linearizes shard→core; BLOCK has two shard axes).
- legacy_2d and nd must land **together**: a live tensor's `memory_config()` carries both `shard_spec` and `nd_shard_spec` and canonicalizes 2D-expressible ND specs, so validate() cannot tell them apart at runtime (the op's `_side_dict` prefers the legacy reading; the harness `axes.classify_call` has the mirror ambiguity — 88 declared/captured mismatches).
- Depends on Refinement 1 (2-D split for `sharded_full_grid`; padding for the `padded_to_height_sharded` cross).
- The sharded LOOSE_CASES (`[1,1,2048,512]` HEIGHT on 64 cores, `[1,1,512,512]` BLOCK COL_MAJOR) become guard-set members for later perf refinements; report their device-ns.

**Done when**: every golden cell with `shard_api ≠ none` passes at bf16 (DRAM-sharded output included), the program-cache cases `height_sharded_same_spec` and `interleaved_to_height_sharded` pass, and a dataflow check confirms same-spec cores issue no NoC reads of their own shard.

**Outcome**: all 12 bf16 `shard_api ≠ none` golden cells, the 4 sharded loose cases and both sharded program-cache cases pass; same-spec cores run the resident path (reader = one reserve/push, writer = one wait/pop, both CBs backed on the shards — pinned by `test_tilize_sharded.py`). Sharded loose cases on BH p150b (per-timer-cluster span): height-in→DRAM 8.7 µs (ref 8.3), DRAM→height-out 7.4–7.7 µs (ref 7.0), height same-spec 1.16 µs (ref 0.93), block COL same-spec 1.13 µs (ref 0.89). Same-spec is pure fixed overhead (no data movement); crossovers are bound by the non-resident side's NoC half.

### [x] Refinement 3 — Speed up the perf-flagged profile `[1,1,16384,64]`

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags `[1,1,16384,64]` bf16→bf16, interleaved DRAM→DRAM, 32×32 tiles, `low_l1=False` (`# attention: PERF FOCUS`) as the mandatory perf target; reference duration `measured_ns_blackhole = 11852` (WH 25998). Phase 0 measures 11.35 µs on 110 cores (4 MiB of DRAM traffic → ≈ 8.2 µs at the ~512 GB/s BH DRAM roofline, so ~25 % headroom). Optimize THIS config using the relevant patterns in `ttnn/ttnn/operations/examples/master.md`: knob co-tunes on the surface the planner exposed — `STAGING_SLOT_BUDGET_BYTES` (staging chunk size, ≥ whole tile-rows; coarser amortizes barriers) × staging slot count × `IN_DEPTH_TILE_ROWS` / `OUT_DEPTH_TILE_ROWS` (`double_buffer`), issue-ahead on the reader (`trid_double_issue`), and `split_reader` (the writer RISC has slack: it issues 2 tile writes per tile-row here). No SUPPORTED change.

**Done when**: measured device-ns on `[1,1,16384,64]` improves over 11.35 µs (report the core count reached alongside), output still bit-identical, golden suite green, and no regression across the config-spanning guard set (one representative per distinct kernel path × layout × placement: coalesced/non-coalesced reader, ragged column blocks, tiny tile, L1 placement, `[1,1,128,64]` few-core case, and each sharded regime from Refinement 2).

**Outcome**: measured null — no knob setting beats the existing default. 110 cores, 15–30-sample medians of the per-timer-cluster span: HEAD 10.94–10.96 µs, final default 10.96–11.13 µs (same schedule; the 11.35 µs Phase 0 number is the same config under a noisier single-sample method). Swept: staging chunk 4/8/12/16/32 KiB × 2–5 slots (trid ring) — 16 KiB × 2 is best, 512 B/bank reads (4 KiB chunk) are +25 %, a single 32 KiB chunk +4 %; issuing chunk 1 together with chunk 0 +4 % (slows the first chunk); short first chunk +2–4 %; IN/OUT depth 2–8 flat; writer-RISC share of the reorder copies 0–0.75 flat (also flat on 32768x64, 16384x32, 3520x64); 64/86/100 cores no better. Bottleneck: total NoC/DRAM throughput with a position-dependent tail, not per-core work. Reads only ≈ 7.0 µs (≈ 300 GB/s), writes only ≈ 8.3 µs (≈ 250 GB/s), both together ≈ 10.5 µs; with no DRAM traffic the reorder copies cost ≈ 2.8 µs per core (≈ 25 cycles per 128 B self-copy; NoC write, self-read, mixed, dual-NoC (dynamic mode) and RISC memcpy variants are all as slow or slower), but with the DRAM traffic on they only expose ≈ 0.9 µs. Per-core end times vary 3× by grid position: reads land last on low-y cores (NoC0 return path), writes last on x = 12–15 (NoC1). Moving reads to NoC1 is much worse (2 of 8 banks: +7 %, half: +46 %, full reader/writer NoC swap: 20 µs). Next to try: position-aware work split and per-core bank order (Refinement 6's `noc_placement` / `bank_stagger` levers). Not done here because they are outside this refinement's lever list and are queued for Refinement 6. Levers kept as live knobs at defaults that reproduce the old schedule: `STAGING_SLOTS` (trid ring), `FIRST_CHUNK_TILE_ROWS`, `WRITER_COPY_SHARE`.

### [x] Refinement 4 — Retile: TILE input at one height → TILE output at another

**Goal**: add `in_tile_height` ∈ {32, 16, 8, 4, 2, 1} to SUPPORTED: build the `retile_tile_input` regime (op_design.md) — the reader reads input TILE pages and de-faces them in L1 into the row layout `cb_rm_sticks` expects (local L1 copies, one dispatch, no DRAM round-trip, no ROW_MAJOR intermediate tensor); compute/CBs unchanged. Must also cover TILE inputs whose logical H is not a multiple of the output tile height (e.g. `[1,1,16,128]` at 16-high → 32-high: the input's own tile padding becomes the output's H tail).

**Implementation skill**: /memory-layouts

**Verifier notes**:
- Scheme-change (new reader de-face stage; unit = `lcm(in_tile_h, tile_h)` rows) — stands alone. Ordered before the dtype refinement so dtype widening extends a finished retile path once (`test_tilize_retile_dtype_conversion` needs both).
- Covers interleaved and sharded retile (`test_tilize_retile` WIDTH/HEIGHT/BLOCK cases) — reuse Refinement 2's placement plumbing.
- validate(): the verifier scoped the "H not a multiple of the tile height without padding → ValueError" rule to ROW_MAJOR inputs; a TILE input falls through to the SUPPORTED gate (`alignment` / `in_tile_height`). Keep that ordering.
- **Never** implement via `ttnn.untilize` + tilize or `ttnn.to_layout` — the spec forbids a second layout pass.

**Done when**: every golden and translated retile cell (`tile_geometry_retile` group, `test_tilize_retile`, program-cache `retile_32_to_16`) passes at bf16.

### [~] Refinement 5 — Numerical configurability: all input/output dtypes

**Goal**: add `dtype` ∈ {float32, fp8_e4m3, uint32, int32, uint16, uint8} and `output_dtype` ∈ {float32, bfloat8_b, bfloat4_b, uint32, int32, uint16, uint8} to SUPPORTED (all legal pairs per feature_spec INVALID), with in-kernel cast at pack time (output CB format = output dtype), bit-exact passthrough for fp32 and 32-bit ints (`fp32_dest_acc_en=True`, `UnpackToDestFp32` on `cb_rm_sticks`, `Fp32Mode::Lossless` — never fast tilize with UnpackToDestFp32), and correct per-width integer handling (uint8 segment alignment: `dram_align_unit_tiles = 2` on 64 B-aligned BH).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Easiest tier, deliberately last among generality refinements so it extends every finished path (2-D split, padding, sharded, retile) once. Integer padding with negative fill must bit_cast, not truncate (Refinement 1's reader fill gets its integer encodings here if not already).
- Expected EXCLUSIONS candidates (not their own refinement): blocked outputs (`bfloat8_b` / `bfloat4_b`) × `tile_height < 32` — the reference rejects tiny tiles for blocked formats.
- fp8_e4m3 cells on sharded placements and on the retile path fail inside the harness (`from_torch` cannot build those inputs on this board; fp8 has no TILE form) — they are not op work and are not part of Done-when; feature_spec fix requested in verification_report.md. fp8 interleaved ROW_MAJOR cells are buildable and in scope.
- Regression tests `test_integer_passthrough`, `test_extreme_magnitudes`, `test_pad_value_extremes` (currently failing on the dtype gate) must pass after this refinement. The fp32 LOOSE_CASE `[1,1,8192,32]` (BH ref 7000 ns) joins the perf guard set.

**Done when**: every buildable golden cell with a non-bf16 `dtype` or `output_dtype` passes (bit-exact on the diagonal, within `helpers._transition_tolerance` on casts), the three regression tests above pass, and no Phase 0–4 cell regresses.

### [ ] Refinement 5a — bfloat4_b round-to-nearest before pack (padded / tiny-sample bfp4 cells)

**Goal**: lift the bfloat4_b output PCC above the 0.98 floor on the remaining near-miss cells (`1x1x50x50` with a nonzero `pad_value`, rank-1 `[64]` auto-pad), all source dtypes.

**Finding (Refinement 5)**: the packer's bfp4 conversion truncates the mantissa; on the same data device bfp4 reaches PCC ≈ 0.984 unpadded / ≈ 0.980 padded vs ≈ 0.993 / ≈ 0.991 for the host (round-to-nearest) conversion. These cells measure 0.978–0.981 across seeds (flaky at the floor). `bfp8_pack_precise=True` (single-stage pack from Float16_b/Float32 instead of via Bfp8) was measured: no change.

**Next lever**: an SFPU pre-round in DEST between unpack and pack — per 16-element face row, find the max exponent and add half an ulp of the bfp4 grid (3-bit mantissa at that exponent) so the packer's truncation becomes round-to-nearest. Compute kernel: `eltwise_chain`-style dest-resident step after the tilize datacopy (requires the non-fast tilize path or a copy_tile pass). Gate it on `output_dtype == bfloat4_b` only.

**Done when**: the bfp4 golden cells pass reliably (≥ 0.98 over repeated runs) and no other cell regresses.

### [ ] Refinement 6 — Second perf pass on `[1,1,16384,64]`

**Type**: perf

**Goal**: the perf-flagged `[1,1,16384,64]` (bf16, interleaved DRAM, BH ref 11852 ns) is the only flagged loose case, so this pass targets it again with the levers Refinement 3 did not take — e.g. NoC placement / per-core start order (`noc_placement`, `bank_stagger` variants), reader/writer NoC assignment, or a T2 restructure of the local reorder copies (`read_coalescing`, `tile_reorder`) — after the full feature surface has landed, so the measurement includes any cost the generality refinements added to the shared path. Gate on headroom first (`/perf-ceiling-dm`): if Refinement 3 left the shape within noise of the DRAM roofline, record that in the changelog and close this refinement with no code change.

**Done when**: measured device-ns on `[1,1,16384,64]` improves over the post-Refinement-5 number (or the roofline check shows no headroom, recorded with numbers), output bit-identical, golden suite green, and no regression across the config-spanning guard set (including the short_wide, sharded and fp32 LOOSE_CASES now supported).
