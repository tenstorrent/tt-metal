# Metal 2.0 Port Report — `experimental/quasar/tilize_with_val_padding`

**Unit ported:** `TilizeWithValPaddingSingleCoreFactory` → `MetalV2FactoryConcept` (`create_program_artifacts`). Other 3 factories remain on `ProgramDescriptorFactoryConcept` (per-factory dispatch; op builds/runs). **Build/test: pending (user runs).**

## Concept realized
- Factory now returns `ttnn::device_operation::ProgramArtifacts` from `create_program_artifacts` (was `create_descriptor` → `ProgramDescriptor`). `.hpp` signature updated.
- Spec: 2 DFBs (`in`/`out`), 2 `TensorParameter`s (`input`/`output`, both Case 1), 3 KernelSpecs (reader/writer/compute), 1 WorkUnitSpec (single core). Pipeline `reader → in → compute → out → writer` (proper SPSC).

## Device-op-class edits
- None required: no custom `compute_program_hash` on `TilizeWithValPaddingDeviceOperation` (audit confirmed). The single-core factory stays in the `program_factory_t` variant; only its method changed (mixed-concept variant is valid — framework dispatches per factory).

## Shared-kernel forks (recipe: fork the source; `untilize_metal2.cpp` precedent)
- `compute/tilize.cpp` → **`compute/tilize_metal2.cpp`** (NEW). Shared with multi_core_default/sharded — legacy retained for them.
- `writer_unary_interleaved_start_id.cpp` → **`writer_unary_interleaved_start_id_metal2.cpp`** (NEW). Shared with multi_core_default — legacy retained.
- `reader_unary_pad_dims_split_rows.cpp` — single_core-only, **ported in place** (no fork).
- Forked kernels are JIT-compiled (path-referenced) — no `sources.cmake` change.

## Dropped plumbing (applied)
- Buffer-address RTAs (reader `src0_buffer`, writer `dst_buffer`) → `TensorBinding` (Case 1; kernels build `TensorAccessor(ta::input)` / `(ta::output)`).
- `TensorAccessorArgs` plumbing (reader `<2>`, writer `<1>`) → bindings.
- Magic CB indices (reader hardcoded `cb_id_in0=0`; writer `output_cb_index` CTA; compute `c_0`/`c_16`) → `DFBBinding` (`in`/`out`).
- Positional CTAs → named: reader `bytes_per_tile_row`; compute `per_core_block_cnt`/`per_core_block_tile_cnt`. **Dead reader CTA `unpadded_row_size_bytes` dropped** (kernel read CTA[0] then `TensorAccessorArgs<2>`, never CTA[1]).
- Per-core RTAs → named (15 reader, 2 writer).

## Open items for downstream / watch-fors
- **Risk point to watch at build:** the compute fork uses `is_fp32_input_format<dfb::in>()` and `compute_kernel_lib::tilize<…, dfb::in, dfb::out, …>` with `dfb::in`/`dfb::out` as template NTTP args — mirrors the `untilize_metal2.cpp` precedent (which uses `dfb::in`/`dfb::out` as `untilize<…>` NTTPs), so expected to work; flagged in case the `is_fp32_input_format<>` NTTP path behaves differently.
- Reader local var still named `cb_in0` (now a `DataflowBuffer`); left un-renamed to minimize diff (rule 1 rename is optional/readability).
- Remaining factories to port (later passes): `multi_core_default`, `multi_core_block_interleaved`, `multi_core_sharded`. When `multi_core_default` ports, it can either bind the existing `_metal2.cpp` forks or its own; reconcile the forks then (delete legacy twins once all factories on Metal 2.0).

## Friction (recipe note)
- The two shared kernels (compute + writer) forced forks for a single-factory port — expected per the shared-top-level-entry-point rule, but worth noting that `tilize_with_val_padding`'s factories heavily share kernels, so each factory pass will add/reconcile forks until the op is fully ported.

---
## Pass 2: multi_core_default factory (build pending)
- Ported `TilizeWithValPaddingMultiCoreDefaultFactory` → `create_program_artifacts`. Reuses the pass-1 forks (`tilize_metal2.cpp`, `writer_unary_interleaved_start_id_metal2.cpp`) — no new forks. Reader `reader_unary_pad_dims_split_rows_multicore.cpp` ported in place.
- Full+cliff compute (preserved multiplicity); reader/writer on all cores, referenced by both work units (mirrors the ported untilize interleaved factory).
- **RTA varargs**: reader's BlockRep RLE groups → `get_vararg` (rt_arg_idx starts at 0; named prefix separate). `num_runtime_varargs` = per-core max, zero-padded. Reported per recipe rule 4 (retained varargs — genuinely dynamic loop).
- Dropped: buffer-address RTAs (src/dst→bindings), TensorAccessorArgs, magic CB indices, positional CTAs; dead reader CTA `aligned_page_size`.
- No device-op-class edits (no custom hash).
- Remaining twvp factories: multi_core_block_interleaved (uses tilize_wh.cpp + reader_unary_pad_multicore_both_dims + writer_..._wh — distinct kernels), multi_core_sharded.

---
## Status after pass 2: 2/4 twvp factories ported & built (single_core, multi_core_default)
Remaining (each needs a dedicated pass — new structural patterns, deferred to avoid rushing a correctness-sensitive mapping):
- **multi_core_sharded** — borrowed-memory DFBs (c_1 input / c_16 output via `borrowed_from`) + a staging CB (c_2) whose endpoint multiplicity must be verified (sync-free fabricate-consumer vs single-ended STOP). Reuses `tilize_metal2.cpp`. Recommended next.
- **multi_core_block_interleaved** — per-group CB sizes ⇒ per-group DFB+reader/writer/compute multiplicity (the unified legacy reader/writer split per group); per-core→group RTA routing via `CoreRangeSet::contains`. Hardest twvp factory.
Full per-factory analysis in `METAL2_PORT_PLAN.md`.

---
## Pass 3 (attempt): multi_core_sharded + multi_core_block_interleaved — BLOCKED on scratchpad (DM scratch / single-ended producer)
Investigated both remaining factories' CB sync topology (read the kernels). Both use DM scratch/staging CBs that Metal 2.0 can't express yet — the same blocker class as `reshape_view` (DM single-ended producer; "scratchpad coming soon"). Per the patterns catalog (Sync-free/single-ended CBs) these **STOP**; do not improvise borrowed_from + fabricated consumers when a `reserve_back` is present (deadlock risk; "if not certain it's 100% pure scratchpad, STOP").

- **multi_core_sharded** — `reader_unary_pad_height_width_sharded.cpp`:
  - `c_0` (cb_in1, intermediate): reserve/get_write_ptr/push_back → CLEAN (reader producer → compute consumer).
  - `c_16` (output): compute producer → writer consumer (writer just wait_front+pop, sharded-in-place handshake) → CLEAN, borrowed_from OUTPUT.
  - `c_1` (cb_in0, borrowed input shard, `cb.buffer=a.buffer()`): `reserve_back` + `get_read_ptr` only (no push/wait/pop) → borrowed read with a vestigial `reserve_back` → not a clean synchronized borrowed-DFB. **STOP-ambiguous.**
  - `c_2` (cb_pad, scratch): `reserve_back` + `get_write_ptr` only → producer-side reserve, no consumer → sync-free-with-reserve scratch. **STOP** (not pure scratchpad).
- **multi_core_block_interleaved** — `reader_unary_pad_multicore_both_dims.cpp`:
  - `c_0` (input): reserve/get_write_ptr/push_back → CLEAN (reader → compute).
  - `c_1` (staging): `reserve_back` + `get_write_ptr` + `push_back`, **no consumer** → **DM single-ended producer → STOP** (under active design / scratchpad).
  - (also carries the per-group-CB-size multiplicity noted earlier.)

**twvp host-2.0 status: 2/4 factories ported, built & tested (single_core, multi_core_default). The other 2 are blocked on the upcoming scratchpad feature** (re-attempt once it lands). This is as far as twvp can go today.
