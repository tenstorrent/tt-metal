# QUASAR_UPLIFT_REPORT — ttnn.untilize (path exercised by llama32_1b_quasar graph-op test)

**Status: RED — Not Metal 2.0 on Gen1 yet (on the exercised code path).**

Per the RED-stop conditions in `docs/source/ttnn/ttnn/ai/quasar_porting.md` (§1 / "RED status"):
the program factories this test actually reaches are still on the legacy
`create_descriptor` → `ProgramDescriptor` host API, not
`create_program_artifacts` → `ProgramArtifacts`. The Quasar uplift starts from an
already-Metal-2.0 op; the Metal 2.0 port (`ai/port/metal2_port.md`, gated by
`ai/audit/metal2_audit.md`) must happen first. **No uplift was performed and no source
file was changed.** A RED here is the audit doing its job — it stops a bad port.

---

## 1. What the test exercises (routing analysis)

Test: `models/experimental/llama32_1b_quasar/tests/graph_ops/test_untilize.py` — two cases,
both `ttnn.untilize` on a `[1, 1, 32, 128256]` BFLOAT8_B TILE-layout **interleaved** tensor
(L1-in/DRAM-out and DRAM-in/DRAM-out), `use_multicore=True`, no `sub_core_grids`.
Geometry: Wt = 4008 tiles per row, exactly **one tile row** (Ht = 1), tile-aligned logical shape.

Dispatch chain (`ttnn/cpp/ttnn/operations/data_movement/untilize/untilize.cpp`, `ttnn::untilize`,
lines 142–170):

1. **Codegen route (the primary route).** `supported_execution_controls(use_multicore=true, sub_core_grids=nullopt)`
   → true, and `supported_by_codegen()` (`codegen/untilize_codegen_supported.cpp`) accepts the case:
   TILE, default 32×32 tile, bf8_b, interleaved in/out, tile-aligned, `total_tile_rows == 1`, and
   `column_parallel_plan_fits(4008)` holds on any normal WH/BH compute grid (8×8 grid → ≈63
   tiles/core → `2·63·2048 B ≈ 258 KB` ≤ usable L1). `is_demoted()` is unconditionally false.
   → **`ttnn::prim::untilize_codegen`** → `UntilizeCodegenProgramFactory::create_descriptor`
   (`codegen/untilize_codegen_program_factory.hpp:21`) — returns `tt::tt_metal::ProgramDescriptor`.

2. **Native fallback (small grids, e.g. a tiny Quasar-emulator grid where the column-parallel plan
   does not fit L1).** `untilize_native` computes `enough_space_height` via `is_enough_space`
   (`ttnn/cpp/ttnn/operations/data_movement/common/common.cpp:498`): estimated CBs =
   `2 · 4008 tiles · 1088 B/bf8-tile ≈ 8.7 MB` ≫ per-core L1 → **false**. In
   `UntilizeDeviceOperation::select_program_factory`
   (`device/untilize_device_operation.cpp:313`), `!enough_space_height && !sharded` short-circuits
   ahead of every other branch → **`UntilizeMultiCoreBlockProgramFactory::create_descriptor`**
   (`device/factories/untilize_multi_core_block_program_factory.cpp:56`) — also a
   `ProgramDescriptor` factory. (The Metal-2.0 `UntilizeMultiCoreParallelizeColumnProgramFactory`
   that `get_pf_type()==0` names for this shape is unreachable: the block-factory branch precedes it.)

So **every factory this test can land on is legacy**:

| Reachable factory | Host API | Kernel API |
|---|---|---|
| `codegen/untilize_codegen_program_factory.cpp` (`UntilizeCodegenProgramFactory`) | `create_descriptor` → `ProgramDescriptor` — **legacy** | device-2.0 headers (`api/compute/*`), but CB-id compile-time args, positional `get_arg_val<uint32_t>(i)`, `cb_wait_front`/`cb_reserve_back` explicit CB protocol, and the `api/dataflow/circular_buffer.h` include (`codegen/kernels/{compute_untilize,reader_tile_interleaved_unified,writer_untilize_col_parallel,writer_untilize_interleaved}.cpp`) — **no `dfb::`/`args::`/`tensor::` bindings** |
| `device/factories/untilize_multi_core_block_program_factory.cpp` (`UntilizeMultiCoreBlockProgramFactory`) | `create_descriptor` → `ProgramDescriptor` — **legacy** | legacy kernels (`untilize.cpp` / `untilize_wh.cpp` compute, `reader_unary_start_id.cpp`, `writer_unary_stick_layout_split_rows_multi_core.cpp` legacy paths) |

Per quasar_porting.md §1, an op qualifies for the uplift only when the factory is
`create_program_artifacts` → `ProgramArtifacts` with `dfb::`/`args::`/`tensor::`/`scratch::`
bindings **and** the kernels are on the Metal 2.0 named-binding device APIs. Neither reachable
factory qualifies → **RED, stop before the uplift**.

### Context: parts of this op ARE already Metal 2.0 (but the test never reaches them)

The untilize op directory is mid-migration. These factories are already
`create_program_artifacts`/`ProgramArtifacts` with `_metal2` kernels
(`untilize_metal2.cpp`, `untilize_variable_num_blocks_metal2.cpp`, `reader_unary_start_id_metal2.cpp`
use `dfb::src`/`dfb::out`, `get_arg(args::…)`):

- `untilize_multi_core_program_factory.cpp`
- `untilize_single_core_program_factory.cpp`
- `untilize_multi_core_parallelize_column_program_factory.cpp`
- `untilize_multi_core_sub_core_grids_program_factory.cpp`
- `untilize_multi_core_nd_shard_input_program_factory.cpp`
- `untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.cpp`

Still legacy `ProgramDescriptor`:

- `codegen/untilize_codegen_program_factory.cpp` ← **the test's primary path**
- `device/factories/untilize_multi_core_block_program_factory.cpp` ← **the test's fallback path**
- `device/factories/untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.cpp`

A Quasar uplift scoped to the already-M2 factories would not cover this test: for the
`[1,1,32,128256]` shape those factories are unreachable by construction (routing shown above).

## 2. Files changed

**None.** No source, kernel, or factory file was modified. This report is the only artifact
(uncommitted; delete before merge per the recipe).

## 3. §7–§8 gotchas applied / considered

None applied — the RED gate stops the uplift before §7–§12. Recorded for the eventual
post-Metal-2.0-port re-audit of this path:

- **§8.1 "`REDUCE_OP` was not declared" (plain tilize TU)** — fix already in `main`; tilize-family
  compute TUs on this op would inherit it; nothing to do in-op.
- **§5 / §8.3 `fifo_page_size` staleness** — the codegen writer kernels compute stick addressing
  from CT args, not `get_local_cb_interface().fifo_page_size`; after an M2 port the
  `get_entry_size()` rule applies to the rewritten kernels.
- **§7 Int32-only / no uint16-uint32** — untilize forwards dtype (bf8_b→bf16 conversion is a
  format decision, no uint branch in the exercised kernels); nothing to guard at the op level.
- **§7 non-zero-init semaphores** — none found in the exercised factories (no `SemaphoreSpec` /
  `CreateSemaphore` at all); clean under `quasar_audit.md` check 2.
- **§6 DM self-loop DFBs** — not assessable pre-port (the exercised factories still declare CBs);
  the codegen compute↔writer CBs are ordinary two-endpoint FIFOs, so no self-loop debt is visible
  today. Re-run `cb_dfb_quasar_audit_helper.md` after the M2 port.

## 4. Deferred / follow-up items

1. **Metal 2.0 port of `UntilizeCodegenProgramFactory` (+ its 5 kernels under `codegen/kernels/`)** —
   prerequisite for any Quasar uplift of this test's primary path. Run `metal2_audit.md` first
   (it is on the `ProgramDescriptor` concept, so the TTNN-factory-concept gate should pass).
2. **Metal 2.0 port of `UntilizeMultiCoreBlockProgramFactory`** — prerequisite for the wide-row
   native fallback this shape takes on small grids (`!enough_space_height` short-circuit).
3. (Out of this test's scope, same op) `untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.cpp`
   is the third still-legacy factory in the directory.
4. No LLK gaps were identified (none were assessable without a port + device run); the §8 tilize/untilize
   LLK items (`set_up_dest_dvalid_per_thread` in `tilize_init`, per-tile `llk_math_set_dvalid<FPU>`)
   live in the LLK layer and are not op edits in any case.

## 5. WH/BH parity claim

**Structural: the diff against the base branch is empty** (zero source changes; this report is the
only new file, and it is documentation outside any build). WH/BH behavior is therefore unchanged
by definition. No device runs were performed (recipe §9 — the user runs all builds/tests).

## 6. Test commands (for the user; not run here)

BH/WH parity baseline (existing behavior, should pass today):

```bash
# op-level coverage of the exercised shapes/paths
pytest tests/ttnn/unit_tests/operations/data_movement/test_untilize.py -q
# the graph-op test itself (on a Gen1 board it routes to the codegen path)
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_untilize.py -q
```

Quasar (emulator), once a Metal 2.0 port of the two legacy factories lands:

```bash
TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_untilize.py -q
```

Run Quasar both with `TT_METAL_LLK_ASSERTS` on and off (recipe §9).
