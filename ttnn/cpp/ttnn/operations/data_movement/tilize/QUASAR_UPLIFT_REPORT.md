# Quasar Uplift Report — `data_movement/tilize`

**Ran on:** post-#54805-merge state of branch `vsureshTT/llama_quasar_uplift`
(PR #54805 "[Metal 2.0] Port data_movement/tilize" already merged into this branch).
**Recipe:** `docs/source/ttnn/ttnn/ai/quasar_porting.md` + canonical
`.../metal_2.0/ai/audit/quasar_audit.md`, `ai/post_port/semantic/{dm_self_loop_dfbs,gen2_hardware_configs}.md`,
`ai/post_port/style/sync_free_dfbs.md`.
**No build, no device run** (per task). Parity argued structurally below.

## Status: GREEN (with per-factory RED carve-outs — see Deferred)

The three M2 factories that are the common/default path (default, single-core, sharded)
are uplifted and Quasar-ready pending device validation. Two other M2 factories (retile,
sharded-retile) and one legacy factory (block) are RED for reasons outside this op's control;
they are flagged, not forced.

## Gate — is the merged tilize really Metal 2.0? Which factories?

`TilizeDeviceOperation` is a Metal 2.0 decorated op (`ttnn::prim`, `select_program_factory`).
Per-factory gate (`create_program_artifacts`/`ProgramArtifacts` + `dfb::`/`args::`/`tensor::`
bindings + device-2.0 kernel APIs):

| Factory | Host API | Compute kernel | Verdict |
|---|---|---|---|
| `tilize_multi_core_default` | **M2** `create_program_artifacts` | shared `kernel/compute/tilize_metal2.cpp` (device-2.0) | **M2** |
| `tilize_single_core` | **M2** | shared `tilize_metal2.cpp` | **M2** |
| `tilize_multi_core_sharded` | **M2** | shared `tilize_metal2.cpp` | **M2** |
| `tilize_multi_core_retile` | **M2** | in-op `kernels/compute/retile.cpp` | M2 host, **Quasar-blocked kernel** (see Deferred) |
| `tilize_multi_core_sharded_retile` | **M2** | in-op `retile.cpp` | M2 host, **Quasar-blocked kernel** |
| `tilize_multi_core_block` | **LEGACY** `create_descriptor`/`ProgramDescriptor` | — | **not M2** (see Deferred) |

In-op reader kernels (`reader_unary_stick_layout_split_rows_{multi,single}core.cpp`) use the
device-2.0 API (`api/dataflow/*`, `Noc`, `DataflowBuffer`, `TensorAccessor(tensor::src)`,
`get_arg(args::…)`, `CoreLocalMem`, `get_write_ptr()` — not `fifo_page_size`).

**Which factory the graph_ops path exercises:** the `graph_case.build_tensor` on-device
row-major→tile conversion (and mainline `ttnn.tilize` on a plain interleaved row-major bf16
input, `use_multicore=true`, `enough_space_height=true`) routes to
**`TilizeMultiCoreDefaultProgramFactory`** — an M2 factory. Not the legacy block factory.

## Audit results (quasar_audit + §7–§12)

- **DM self-loop DFBs (dm_self_loop_dfbs.md):** none. The default/single/sharded factories are
  ordinary cross-kernel FIFOs (reader→compute `input`, compute→writer `output`, 1:1 per node).
  retile/sharded-retile have `MID`/`MID_VIEW` self-loops but they are **compute** self-loops
  (bound PRODUCER+CONSUMER by the single `retile.cpp` compute kernel) — legal on Gen2, leave as-is.
- **Sync-free DFBs (sync_free_dfbs.md):** none qualifying; input/output DFBs synchronize via the FIFO.
- **Gen1-only hardcoded `hw_config` (gen2_hardware_configs.md shape-4 compute):** FOUND in all 5 M2
  factories — each hardcodes `ComputeGen1Config`. **This is the applied fix** (below). DM configs
  use the arch-agnostic helper `ttnn::create_{reader,writer}_datamovement_config(arch)` (shape 1 —
  no work).
- **Non-zero-init semaphores:** none (`grep CreateSemaphore|SemaphoreSpec|initial_value` → nil).
- **`fifo_page_size` / `get_local_cb_interface`:** none.
- **`evil_set_*`:** `retile.cpp:117` `mid_view.evil_set_read_ptr(...)` — Gen1-only, RED (Deferred).
- **`disable_dfb_implicit_sync_*`:** none.
- **uint16/uint32 device-format branches:** none. The tilize kernels forward `DataType` via
  `data_format_metadata`; no kernel-level uint16/uint32 code path to guard (like typecast, the
  Int32/no-uint16 limitation lives at the format/LLK layer, not this op).
- **`MEM_ZEROS_BASE`, NoC/mcast tricks:** none.
- **§7–§8 runtime-symptom fixes:** none applied (no device run — reactive only).

## Files changed

Each change is arch-branched on `device->arch() == tt::ARCH::QUASAR`, matching the
`sharded_to_interleaved` idiom on this branch. The existing `ComputeGen1Config` initializer is
left **textually unchanged**; the new branch is only taken on Quasar. Fields copied verbatim per
the gen2_hardware_configs.md shape-4-compute table: `enable_32_bit_dest` (copy), `unpack_modes`
(copy + `TODO(#52269)` marker). `bfp_pack_precision_mode` not set by these ops → nothing to drop.
`enable_2x_src_register` left at default (never set).

1. `device/tilize_multi_core_default_program_factory.cpp` — Gen2 compute `hw_config` branch in the
   `make_compute` lambda; `.hw_config = std::move(compute_hw)`.
2. `device/tilize_single_core_program_factory.cpp` — Gen2 compute `hw_config` branch; `.hw_config = std::move(compute_hw)`.
3. `device/tilize_multi_core_sharded_program_factory.cpp` — Gen2 compute `hw_config` branch; `.hw_config = std::move(compute_hw)`.

(Also created: this report.)

## Gotchas considered but NOT applied

- **retile/sharded-retile Gen2 config:** deliberately NOT added. Those factories are already RED
  (evil_set_read_ptr, below); half-fixing a RED factory would misrepresent it as portable. Left
  their `ComputeGen1Config` untouched.
- All §7–§8 reactive fixes (implicit-sync double-count, `0x19`/`PACR0_TILE_INC`, DEST-dvalid,
  cache-flush): not applicable statically and no device run — apply reactively only.

## Deferred / RED items (flag, do not force)

1. **`tilize_multi_core_block_program_factory.cpp` — still legacy Metal 1.0.**
   Uses `create_descriptor`/`ProgramDescriptor` (legacy `CreateKernel` path). RED-stop condition
   "not Metal 2.0 on Gen1 yet". Needs the base Metal 2.0 port (`ai/port/metal2_port.md`) before any
   Quasar uplift — out of scope for this uplift pass. On Quasar it will still assert
   *"DataMovementKernel is not supported on Quasar"*. Reached when `enough_space_height == false`,
   or (Blackhole) non-sharded UINT8 input.

2. **`retile.cpp:117` uses `evil_set_read_ptr` (Gen1-only) → retile + sharded-retile factories are
   Quasar-blocked.** `evil_set_read_ptr`/`evil_set_write_ptr` are declared under
   `#ifndef ARCH_QUASAR` in `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h` — absent on Quasar, so
   `retile.cpp` will not compile for Quasar. Per §7/§8.3 this is a **missing-feature for the runtime
   team** (a sanctioned Gen2 DFB read-cursor/rewind API), **not** something to hand-roll in the op.
   The retile path is only reached by a tile-shape-changing tilize (`is_retile`), which is NOT the
   graph_ops input-tilize path.

3. **Stale comment in shared `ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp` (issue #52228).**
   Its header says the tilize factories "are still on the legacy host API" — no longer true after
   #54805 (5 of 6 are M2). This is a note for the shared-kernel owner / #52228 sunset plan; it is
   outside the op directory, so left unedited here.

## WH/BH structural parity claim

Every applied change is confined to a `if (device->arch() == tt::ARCH::QUASAR) { … }` block that
builds a `ComputeGen2Config`. On WH/BH the branch is not taken and `compute_hw` holds the original,
byte-for-byte-unchanged `ComputeGen1Config`. No Gen1 initializer field, value, or order was
touched; no kernel source, DFB spec, binding, or runtime-arg schema changed. Therefore WH/BH take
the original path unchanged. (Structural claim only — no device run; confirm with the parity tests below.)

## Harness-workaround verdict (graph_case host-tilize)

`graph_case.build_tensor` (line ~465) host-tilizes inputs and comments that an on-device
`from_torch(device=, layout=TILE, mesh_mapper=)` would run mainline `ttnn.tilize`, "unsupported on
Quasar (legacy CreateKernel path, no Gen2 branch → 'DataMovementKernel is not supported on Quasar')."

**That specific blocker is removed for the default/single-core/sharded factories.** Post-#54805
those factories are Metal 2.0 (no `CreateKernel`), and after this uplift they carry a Gen2 compute
config, so the from_torch on-device tilize of a plain interleaved row-major input routes through the
**M2 default factory** and no longer hits the legacy assertion. So the host-tilize workaround
(hazard #1 in the comment) is **very likely no longer needed for tilize** on that path.

**Do not remove it yet — confirm on the Quasar emulator first**, because:
- No device run was done here; M2-default correctness on the emulator (implicit sync, LLK behavior)
  is unverified.
- The **block factory remains legacy** and would still assert if any graph_ops shape routes there
  (`enough_space_height=false`, or BH UINT8). Removal is safe only for shapes provably on default/
  single/sharded.
- Hazard **#2** in the same comment (interleaved-DRAM→sharded-L1 via `to_memory_config`, routed to
  `ttnn.experimental.quasar.to_memory_config`) is a **separate** workaround, unrelated to tilize and
  unaffected by this uplift.

## Test commands (user runs; order BH → WH → Quasar)

Parity (must stay green, prove WH/BH unchanged):
```
# Blackhole / Wormhole
TT_METAL_FORCE_JIT_COMPILE=1 pytest tests/ttnn/unit_tests/operations/data_movement/test_tilize.py
TT_METAL_FORCE_JIT_COMPILE=1 pytest tests/ttnn/unit_tests/operations/data_movement/test_tilizer.py
TT_METAL_FORCE_JIT_COMPILE=1 pytest tests/ttnn/unit_tests/base_functionality/test_tilize_pad_cb.py
```

Quasar (emulator; purge kernel cache between baseline and post-port; run LLK asserts on first):
```
rm -rf ~/.cache/tt-metal-cache
TT_METAL_FORCE_JIT_COMPILE=1 pytest tests/ttnn/unit_tests/operations/data_movement/test_tilize.py
# graph_ops harness (confirm the host-tilize workaround can be relaxed for tilize):
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_untilize.py
```
