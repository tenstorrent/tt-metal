# Metal 2.0 Port Report — permute (`ttnn/cpp/ttnn/operations/data_movement/permute`)

## Outcome

**PORTED (all five factories)** — `permute` is fully on Metal 2.0 (`MetalV2FactoryConcept`, `create_program_artifacts`). Landed in two commits (see `METAL2_PORT_PLAN.md` for the three-pass structure):
- **Commit `8f7154fa2fb` — pass 1:** `MultiCoreRowInvariant`, `MultiCoreBlockedGeneric` (row-major, own kernels).
- **Commit `53be242ccab` — passes 2 + 3:** `MultiCoreTiledGeneric` (own kernels; self-loop tilize CB + conditional `cb_pad`), then `MultiCoreTileInvariant` and `MultiCoreTileRowInvariant` (use three forked donor kernels; branch on `swap_hw` and `needs_padding`).

The device-op no longer has any `create_descriptor` factory; every `program_factory_t` variant is `MetalV2FactoryConcept`.

### Verification (blackhole)
- **Build:** `./build_metal.sh --build-tests` — clean, no errors/warnings on permute (each pass).
- **Unit — `tests/ttnn/unit_tests/operations/data_movement/test_permute.py`:** all-five-factories run **1593 passed, 1 skipped, 0 failed** (`SAFE_PYTEST_RESULT: PASS`) — identical to the pass-1 (RM-only) result, so no regression from the tiled ports. Exercises every `select_program_factory` path.
- **Nightly — `test_universal_input_tm_permute.py`:** pre-port baseline 86 passed; post-port (all five factories) **86 passed** — no regression.
- **Anti-pattern self-audit:** clean across all factories — no `buffer()->address()`, no CB-index CTAs, no `TensorAccessorArgs<N>()`, no `.id` extraction, no `allow_instance_multi_binding`, all CTAs named, varargs only for genuine rank-length collections, `hw_config` reproduces legacy resolved values (including the two distinct `unpack_modes` translations — see Successes).

## Provenance

- **Recipe docs (this port):** `9ebb69d90cb 2026-07-24 docs(metal_2.0): fix dangling capitulation report target + add Outcome marker` — on branch `akertesz/op-porting-recipe`. The recipe docs are **not** present on the port branch's HEAD, so `git log -1 … -- docs/.../metal_2.0/` prints nothing here; the version is pinned to the `akertesz/op-porting-recipe` commit the port was run against (recorded above) instead.
- **Audit docs (inherited):** `2a53d817976 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` for all five factories — as the audit chose. No deviation.

### Device-op-class edits
- **Custom `compute_program_hash` deleted:** none — the op never had one (default reflection-based hash).
- **Pybind entry points removed:** none — the nanobind layer binds via `ttnn::bind_function<"permute">`; there was no `create_program_descriptor` pybind hook to remove.
- **Header change (forced, sanctioned):** `permute_device_operation.hpp` — all five factory structs' declarations changed from `static ProgramDescriptor create_descriptor(...)` to `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`; added `#include "ttnn/metal_v2_artifacts.hpp"`.

### Open items
- **Relaxation candidates:** none applied. `TensorParameter`s kept strict (default). Not investigated for relaxation — out of scope for the port.
- **Donor-kernel forks:** three `_metal2` forks created in the eltwise/unary and transpose op directories (see Handoff points + Open items for downstream). They carry a sunset/drift obligation until the sibling ops migrate.

## Handoff points

- **Cross-op donor-kernel forks (pass 3).** The two tiled-invariant factories instantiate three kernels owned by other ops. Because those ops are still on the legacy concept, the shared sources could not be modified in place — each was **forked** with a `_metal2` suffix alongside its legacy original (per the shared-dataflow-kernel Caution). The ported permute factories point at the forks; the legacy copies are untouched. Owning teams should fold these into the eventual migration of their own ops and delete the legacy copy at sunset:
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` — fork of the broadly-shared unary interleaved writer. Metal 2.0 changes: positional CTA 0 (`cb_id_out`) → `dfb::cb_out` binding; RTA0 dst address → `tensor::output`; `num_pages`/`start_id` named RTAs; `get_local_cb_interface(...).fifo_page_size` → `dfb.get_entry_size()`. `OUT_SHARDED`/`BACKWARDS` `#ifdef`s preserved (permute defines neither).
  - `ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_metal2.cpp` — fork of the transpose compute. Metal 2.0 changes: `c_0`/`c_16` → `dfb::cb_in`/`dfb::cb_out`; `NHtWt` named RTA.
  - `ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/reader_unary_transpose_hc_interleaved_tiled_padding_aware_metal2.cpp` — fork of the transpose padding-aware reader. Metal 2.0 changes: RTA0 src address → `tensor::input`; `num_tiles`/`start_id` named RTAs; `c_0` → `dfb::cb_in0`; conditional `c_1` padding CB → `dfb::cb_pad` gated by the `NEEDS_PADDING` define (promoted from the legacy `needs_padding` CTA); `swap_hw` kept as a CTA. `BACKWARDS` `#ifdef` preserved.

  No other handoff (no capitulation, no boundary-rule assumption violations, no kernel-lib gaps, no framework gaps, no removed pybind surface).

## Successes

- **Self-loop DFB (`port_patterns.md` — Self-loop DFB binding):** the tilize intermediate CB (`cb_tilize`, legacy c_1) in both `MultiCoreBlockedGeneric` and `MultiCoreTiledGeneric` is produced by the tilize helper and consumed by the transpose, both inside the one compute kernel. Bound PRODUCER + CONSUMER on that one kernel (shared accessor name) — the pattern applied directly, no multi-binding flag.
- **`unpack_modes` — the two translations, and why they differ (`migration_guide.md` / `metal2_port.md` — Hardware configuration).** The recipe's warning about the silent flip fired for real, in *both* directions across this op:
  - `MultiCoreBlockedGeneric` / `MultiCoreTiledGeneric` compute set **no** legacy `unpack_to_dest_mode` (default `Default`) → `UnpackMode::UnpackToSrc` for the consumed FP32 DFBs (`cb_in`, `cb_tilize`).
  - `MultiCoreTileInvariant` / `MultiCoreTileRowInvariant` compute (donor `transpose_wh`) explicitly set `unpack_to_dest_mode[c_0] = UnpackToDestFp32` → `UnpackMode::UnpackToDest` for the consumed FP32 DFB (`cb_src0`).
  Porting these to the same value would have been a silent perf/precision regression; carrying the legacy *value* (not the field name) kept them distinct. Both confined to the Float32 case (Int32/UInt32 deferred, #49936).
- **Conditional / optional DFB bindings (`port_patterns.md` — Conditional / optional DFB bindings):** the padding CBs are bound only on some configs — `cb_pad` (legacy c_3) in `MultiCoreTiledGeneric` gated by `NEEDS_Y_PADDING`; `cb_pad` (legacy c_1) in `MultiCoreTileRowInvariant` gated by `NEEDS_PADDING`. The legacy kernels gated the CB via an `if constexpr (needs_*_padding)` CTA, which still name-looks-up the absent `dfb::cb_pad` token in the discarded branch. Promoted each to a preprocessor define (emitted to every kernel that touches the CB — reader producer + writer consumer), `#ifdef`-gated the `dfb::cb_pad` construction *and* the conditionally-declared padding RTAs, and conditionally added the host binding + `runtime_arg_names`. Worked exactly as the pattern prescribes.
- **Runtime-selected kernel set (`metal2_port.md` — atomic unit / runtime kernel-source selection):** `MultiCoreTileInvariant` / `MultiCoreTileRowInvariant` select their kernel set by `swap_hw` — the `transpose_wh` compute and the `c_16` DFB exist only on the swap path, and the writer's output DFB switches between `c_0` and `c_16`. Handled by branching inside `create_program_artifacts` (conditional `KernelSpec`/DFB/`WorkUnitSpec` membership), with the kernel bodies config-agnostic (always `dfb::cb_out`; only the host binding's `dfb_spec_name` differs).
- **DFB metadata getters (`cb_dfb_api_whitelist.md` §A/§B):** the donor kernels' free-function metadata reads were rewritten to member getters — `get_tile_size(cb_id)` → `dfb.get_tile_size()` (TileInvariant reader), `get_local_cb_interface(...).fifo_page_size` → `dfb.get_entry_size()` (unary writer fork).
- **Runtime varargs rule (`port_patterns.md` — Avoid varargs):** every rank-length shape/perm/stride array (2·rank or 3·rank) stayed varargs (loop count = a CTA), while per-core scalars (`start_*`/`end_*`, `num_blocks`, `NHtWt`, `num_pages`/`start_id`, padding indices) became named RTAs. Unambiguous throughout.

## Friction

### Gaps
- *(none — no missing/stale doc answers surfaced across any of the five factories.)*

### Confusion
- **KernelSpec designated-initializer field order.** `advanced_options` sits *after* `hw_config` in the `KernelSpec` struct, so the varargs-carrying kernels must list `.hw_config` before `.advanced_options` (C++ designated initializers must follow declaration order). Easy to get backwards when mentally grouping "schema-ish" fields together. Minor; noting in case the recipe wants a field-order reminder near the varargs guidance.

## Open items for downstream

- **Cross-op kernel forks — sunset checklist.** Three donor kernels were **forked** (not modified in place) into their owning ops' directories. Each fork is short-lived: delete the legacy copy once every remaining consumer of it has migrated to Metal 2.0, and keep bug fixes in sync until then.
  - `eltwise/unary/.../writer_unary_interleaved_start_id_metal2.cpp` — legacy `writer_unary_interleaved_start_id.cpp` is broadly shared (many eltwise/data-movement ops). **Remaining consumers of the legacy copy: all except permute's `MultiCoreTileInvariant`.** Deleting the legacy copy requires all of them to migrate.
  - `data_movement/transpose/.../compute/transpose_wh_metal2.cpp` — legacy `transpose_wh.cpp` still used by the **transpose** op. Sunset when transpose migrates.
  - `data_movement/transpose/.../dataflow/reader_unary_transpose_hc_interleaved_tiled_padding_aware_metal2.cpp` — legacy still used by the **transpose** op. Sunset when transpose migrates.
- **Dead named CTAs dropped.** Legacy emitted several named CTAs that **no kernel reads** — dropped from the Metal 2.0 CTA tables (a named CTA should correspond to a kernel `args::` read). Zero functional change; flagging for the ops team in case the intent was to wire them somewhere: `input_tensor_page_size`/`output_tensor_page_size` (RM blocked generic reader/writer); `page_size` (tiled generic reader/writer — was already commented out in the kernel; tiled invariant reader — kept, still read but unused).
- **Dead locals / commented-out code removed where forced by the port.** Each of these lost its only input when the buffer-address RTA became a `TensorBinding`, so removal was forced, not discretionary: `curr_addr = src_addr/dst_addr` (RM row-invariant writer, tiled-invariant reader); `tile_bytes = get_tile_size(cb_id_out0)` (tiled row-invariant writer — dead local, `cb_id_out0` disappeared); two commented-out `get_compile_time_arg_val(0/1)` lines (`transpose_xw_rm_single_tile_size.cpp`, pre-named-CTA leftover).
- **Dead compute RTA slots dropped.** Legacy `MultiCoreBlockedGeneric` compute emitted `{num_blocks_per_core, 0u, 0u}` (kernel read slot 0 only); the two `0u` "historical layout" slots are gone (named `num_blocks` only). Zero functional change. (Audit "Misc anomalies" flagged this.)
