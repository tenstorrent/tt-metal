# QUASAR_UPLIFT_REPORT — data_movement/sharded (interleaved_to_sharded, sharded_to_interleaved, reshard, + to_memory_config dispatcher)

Scope: the sharding/memory-config ops exercised by the llama32_1b_quasar graph-trace tests
(`test_interleaved_to_sharded.py`, `test_sharded_to_interleaved.py`, `test_to_memory_config.py`).
`ttnn.to_memory_config` is a host-side composite (`ttnn/cpp/ttnn/operations/core/to_memory_config/to_memory_config_op.cpp`)
that dispatches to `ttnn::prim::reshard`, `ttnn::interleaved_to_sharded`, `ttnn::prim::sharded_to_interleaved`,
or the `ttnn::prim::copy` fallback; it has no device code of its own and is covered here.

Audit executed per `docs/source/ttnn/ttnn/ai/quasar_porting.md` (+ `ai/audit/quasar_audit.md`,
`ai/post_port/semantic/gen2_hardware_configs.md`). No builds or device runs were performed in this
session (recipe §9: user runs all builds/tests); §7–§8 fixes were therefore applied only where
statically determinable, and everything runtime-conditional is recorded as considered/deferred.

---

## Per-op status

| Op | Factory | Status |
|---|---|---|
| `interleaved_to_sharded` | `InterleavedToShardedProgramFactory` | **RED — Not Metal 2.0 on Gen1 yet** |
| `sharded_to_interleaved` | `ShardedToInterleavedProgramFactory` | **GREEN** (1 Gen2 hw_config site fixed) |
| `reshard` | `NdReshardCopyLocalShardFactory<true/false>` | **GREEN** (1 Gen2 hw_config site fixed) |
| `reshard` | `ReshardSameWidthFactory<true/false>` | **GREEN** — no changes needed |
| `reshard` | `ReshardSameHeightFactory<true/false>` | **GREEN** — no changes needed |
| `reshard` | `ReshardGenericFactory` | **GREEN** — no changes needed |
| `reshard` | `NdReshardCopyPagesFactory` | **GREEN** — no changes needed |
| `to_memory_config` | host composite, no device code | **GREEN** — nothing to uplift here |

### interleaved_to_sharded — RED

`interleaved_to_sharded_program_factory.{hpp,cpp}` is still on the `ProgramDescriptor` API
(`create_descriptor`, `CBDescriptor`, `CBFormatDescriptor`, buffer-index CBs) and binds the legacy
kernels (`reader_unary_sharded_blocks_interleaved_start_id.cpp`,
`writer_unary_sharded.cpp`, `eltwise_copy.cpp`, …), which use the legacy device API
(`cb_*`, `noc_async_*`, and one `get_local_cb_interface(cb_id_in1).fifo_page_size` read at
`reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:63` — a known §8.3 Quasar hazard,
but moot until the M2 port). Per the RED-stop conditions, **no Metal 2.0 port was performed and no
Quasar edits were made to this op.** The base Metal 2.0 port (`ai/port/metal2_port.md`) must land
first; then re-run this uplift audit. (Curiosity noted: the legacy factory already carries
`is_quasar` alignment branches at lines 96/179/359/369 — these ride along into the future port but
do not make the op Metal 2.0.)

Graph-trace impact: both `test_interleaved_to_sharded.py` cases and the i2s-routed
`test_to_memory_config.py` cases (ids 00, 02, 03) depend on this op.

### sharded_to_interleaved — GREEN

Fully Metal 2.0: `create_program_artifacts` → `ProgramArtifacts`, named `dfb::`/`args::`/`tensor::`
bindings; kernels are the device-2.0 forks
(`writer_unary_sharded_blocks_interleaved_start_id_metal2.cpp`,
`writer_unary_stick_layout_sharded_blocks_interleaved_start_id_metal2.cpp`, shared
`copy/typecast/.../reader_unary_sharded_metal2.cpp` and `ttnn/kernel/compute/eltwise_copy_metal2.cpp`
— the latter two audited read-only, clean, no edits needed).

One statically-determinable Gen2 defect found and fixed: the optional format-conversion compute
kernel hardcoded `.hw_config = ComputeHardwareConfig{ComputeGen1Config{}}` (gen2_hardware_configs
shape 4, compute) — a kernel whose hw_config carries only a Gen1 config cannot run on Quasar at all.
Fixed with the doc's prescribed form: Gen1 initializer hoisted verbatim, `ComputeGen2Config{}` on
`arch == QUASAR` (the Gen1 config sets no fields, so none are copied), plus the unconditional
`TODO(#52269)` unpack_modes marker.

The graph-trace s2i cases (all TILE, no dtype conversion) never instantiate the compute kernel, but
the `to_memory_config` reshard-workaround path calls s2i with `dtype` and can.

### reshard — GREEN

All 8 factory variants are Metal 2.0; all kernels
(`reshard_reader.cpp`, `reshard_reader_diff_width.cpp`, `reshard_same_width_{reader,writer}.cpp`,
`reshard_same_height_{reader,writer}.cpp`, `nd_reshard_copy_local_shards.cpp`,
`nd_reshard_copy_pages_{reader,writer}.cpp`) are device-2.0 (`Noc`, `DataflowBuffer`,
`TensorAccessor`, `get_arg`/`get_vararg`, `CoreLocalMem`; no `cb_*`, no `fifo_page_size`, no
`evil_set_*`).

One statically-determinable Gen2 defect found and fixed: `NdReshardCopyLocalShardFactory` built a
custom `DataMovementGen1Config{.processor, .noc, .noc_mode}` by hand (shape 4, DM). Per the doc,
Gen2 has no placement concept, so on Quasar it now takes a default-constructed
`DataMovementGen2Config{}` (matching the arch-agnostic helpers); the Gen1 initializer was moved
verbatim, not retyped. No `disable_dfb_implicit_sync_*` was set (and none exists anywhere in the
directory). No unpack_modes marker for this site — shape-4 DM has no such field, so its absence is
not a signal.

The graph-trace `to_memory_config` reshard case (id 06: TILE WS L1 → WS L1, differing grids) routes
to `ReshardGenericFactory` — a zero-change factory.

---

## Files changed

1. `ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.cpp`
   — add the missing Gen2 alternative for the compute kernel's hand-written `ComputeGen1Config{}`
   (gen2_hardware_configs shape 4 compute; arch-branched, Gen1 initializer moved verbatim,
   `TODO(#52269)` marker added).
2. `ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/nd_reshard_program_factory_copy_local.cpp`
   — add the missing Gen2 alternative for the hand-written `DataMovementGen1Config` placement
   (shape 4 DM; `DataMovementGen2Config{}` on Quasar, Gen1 initializer moved verbatim).

No kernel was edited. No file outside this directory was edited. No op moved or was renamed; nothing
tempted a move.

## `hw_config` survey (gen2_hardware_configs.md step 2, whole directory)

`grep -rn "hw_config|to_compute_hardware_config|Gen1Config|std::get<|std::get_if<|holds_alternative"`
found 12 sites: 10 are shape 1 (the arch-agnostic `ttnn::create_reader/writer_datamovement_config(arch)`
helpers — already Gen2-aware, untouched) or pass-throughs of them; the 2 shape-4 sites above were
converted. No shape-3 (`std::get`/`std::get_if` on a helper-built config) sites exist.

---

## §7–§8 gotchas: applied vs. considered

**Applied (statically determinable):**
- Gen2 `hw_config` variants (recipe §2 "what the uplift may touch" / `gen2_hardware_configs.md`) —
  the two sites above.

**Considered, verified clean (no change needed):**
- **quasar_audit check 2 — non-zero-init semaphores:** no semaphores anywhere in the directory (grep
  for `Semaphore|CreateSemaphore` is empty). Clean.
- **quasar_audit check 1 — DM self-loop DFBs:** none. Every DFB is bound by two distinct kernels
  with one endpoint each (s2i `in`/`out`; reshard `shard`/`scratch`/`output_shard`/`page`). No
  kernel binds both PRODUCER and CONSUMER of the same DFB.
- **§7 `disable_dfb_implicit_sync_*`:** not set anywhere; the DM helper's optional parameter is left
  at its `false` default. Clean.
- **§5 / §8.3 `fifo_page_size`:** all Metal 2.0 kernels read sizes off the `DataflowBuffer` object
  (`get_tile_size()`); the one `fifo_page_size` read in the directory is in a legacy kernel behind
  the RED i2s factory.
- **§4 `data_format_metadata`:** every `DataflowBufferSpec` sets a valid format from the tensor
  dtype. `unpack_modes`: not required — the only compute config has `enable_32_bit_dest = false`
  and no Float32 DFBs on the graph-trace paths (bf16/bfp8).
- **§7 Int32/no-uint16/uint32:** these ops forward `DataType` with no device-format branch of their
  own — nothing to guard; the limitation lives at the format/LLK layer.
- **§7 `compute_kernel_hw_startup` exactly once:** the one compute kernel
  (`eltwise_copy_metal2.cpp`, shared, out-of-op) calls it once at `main()` start. Clean.
- **§8.2/§8.5 wait→pop / reserve→push TDMA hazard:** the compute kernel has `copy_tile` between
  `wait_front`→`pop_front` and `pack_tile` between `reserve_back`→`push_back`. Clean.
- **§8.2 "Not done phys cores" s2i hang (8-bit DFB capacity truncation):** verified FIXED in this
  tree rather than assumed — `tt_metal/impl/dataflow_buffer/dataflow_buffer_impl.hpp:79` holds
  `uint16_t capacity` and `dataflow_buffer.cpp:655` documents "capacity is uint16 at bytes 26-27
  (HW BUFFER_CAPACITY width)". Nothing to do at op level.
- **§11 multicast:** no multicast anywhere in these kernels (all unicast / bank-addressed). Clean.
- **§4 opt_level:** s2i's compute kernel states `O3` explicitly (matching the legacy resolution);
  DM kernels leave it absent (→ O2, the legacy DM default). Base-port concern; verified correct,
  untouched.
- **§8.2 implicit-sync double-count (historical):** the s2i writers pass a DFB straight to
  `Noc::async_write` *and* use explicit `wait_front`/`pop_front` — the pattern behind the historical
  double-count rows. Per §7 the underlying runtime bug should be fixed; nothing to change
  pre-emptively. If s2i hangs on Quasar with credits stalled, report it to the runtime team as a
  regression — do **not** disable implicit sync.

**Considered, runtime-conditional — recorded as watch items (reactive per §2; do not fix pre-emptively):**
- **Uncached DFB pointers fed to NoC (post-`a00dd45`):** every reshard kernel and both s2i writers
  address the borrowed shard through `dfb.get_read/write_ptr()` and hand derived addresses to
  `Noc::async_read/async_write` via `CoreLocalMem` (or via the DFB source directly). On Quasar DM
  the getters return UNCACHED addresses while "NOC APIs still require cached addresses"
  (quasar_porting §8.3/§12). Whether `CoreLocalMem`'s NOC traits translate this correctly is not
  statically determinable here. If Quasar runs show wrong data / hangs on these paths, this is the
  first suspect; the fix belongs in the DFB/NOC-trait API (runtime), not hand-rolled in the op.
- **NoC-loopback local L1→L1 copies (§6):** `reshard_same_width_reader.cpp` (UNALIGNED path only)
  re-strides scratch→local with a self-targeted unicast read (`my_x/my_y`), and
  `nd_reshard_copy_local_shards.cpp` can write a page whose destination is the same core. §6 warns
  the emulator can spin on `can_post` or drop such loopbacks; the fix (direct RISC L1→L1 copy) is
  behavioral and untestable here — apply only if the symptom fires. Note: on Quasar the UNALIGNED
  same-width path corresponds to RM shard widths that §7 says must be 16-byte aligned anyway.
- **Address-source ("sync-free") borrowed DFBs:** the reshard `shard`/`output_shard` DFBs are pure
  address sources (no push/pop, never passed to NoC as DFB objects — role-free 1P+1C binding). This
  is the proven-portable cross-kernel bridge pattern; `Scratchpad`/`LocalTensorAccessor` conversion
  is a post-port style pass, not required for the uplift.

## Deferred / follow-up items

1. **interleaved_to_sharded Metal 2.0 port** — prerequisite for its Quasar uplift; run
   `ai/audit/metal2_audit.md` → `ai/port/metal2_port.md` first. Blocks 2 of the 3 graph-trace test
   files (i2s directly; to_memory_config's to-sharded cases).
2. **`ttnn::prim::copy` fallback of `to_memory_config`** (`data_movement/copy/`) — out of this
   task's directory; graph-trace case id 10 (L1 interleaved → DRAM interleaved) routes there. Needs
   its own gate/audit by its owner.
3. **Shared kernels audited read-only, clean, unchanged** (out-of-op; would have been the only
   sanctioned out-of-op writes, none needed): `copy/typecast/.../reader_unary_sharded_metal2.cpp`,
   `ttnn/kernel/compute/eltwise_copy_metal2.cpp`.
4. **Other `_metal2` kernels in this directory** (`writer_unary_sharded_metal2.cpp`,
   `reader_unary_nd_sharded_blocks_metal2.cpp`, `device/kernels/compute/eltwise_copy_metal2.cpp`)
   are bound by *other* ops' Metal 2.0 factories (untilize, transpose, tilize_with_val_padding,
   reduce); audited here since they live in the directory — all device-2.0 clean, no edits.
5. **TODO(#52269)** — Quasar unpack_modes optimization for the s2i compute config (marker added).
6. The two runtime-conditional watch items above (uncached-pointer NoC feeds; L1→L1 NoC loopback),
   for whoever runs the first Quasar bring-up of these ops.

## WH/BH parity claim (argued structurally — no device run this session)

The entire diff is two host-factory hunks. In both, the existing Gen1 config initializer was
*moved verbatim* (same fields, same values, same order) into a hoisted local, and the only new code
is an `if (arch == tt::ARCH::QUASAR)` branch that is never taken on WH/BH. No kernel source, DFB
spec, runtime arg, work unit, or selection logic changed. Therefore WH/BH build the byte-identical
`ProgramSpec` they built before ⇒ no behavior change on Gen1. (The `#ifdef ARCH_QUASAR` guard form
applies to device code; for host code the recipe's equivalent is this runtime arch branch, the exact
form `gen2_hardware_configs.md` prescribes.)

## Test commands (user-run; recipe §9 order BH → WH → Quasar)

Parity on WH/BH (run unchanged on both archs; expect zero deltas vs. the pre-change tree):

```bash
# sharded_to_interleaved + interleaved_to_sharded + to_memory_config surface
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_sharded.py
pytest tests/ttnn/unit_tests/operations/data_movement/test_interleaved_to_sharded.py
pytest tests/ttnn/unit_tests/operations/data_movement/test_sharded_to_interleaved_oob.py
pytest tests/ttnn/unit_tests/operations/data_movement/test_core.py
# reshard (legacy + ND factories, incl. NdReshardCopyLocalShardFactory)
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_reshard.py
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_nd_reshard.py
```

Quasar (emulator env per the craqsim runbook; force JIT since factories changed the spec hash
inputs; run both with and without `TT_METAL_LLK_ASSERTS`):

```bash
TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_sharded_to_interleaved.py
TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_to_memory_config.py
# test_interleaved_to_sharded.py (and to_memory_config ids 00/02/03) will exercise the RED legacy
# i2s op — expected to be blocked until its Metal 2.0 port lands.
TT_METAL_FORCE_JIT_COMPILE=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_interleaved_to_sharded.py
```

*(Purge `~/.cache/tt-metal-cache` between pre-change baseline and post-change runs.)*

---

*This report is uncommitted by design; delete before merge.*
