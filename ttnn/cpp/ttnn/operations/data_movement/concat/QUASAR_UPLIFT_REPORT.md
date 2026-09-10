# Quasar Uplift Report — `data_movement/concat`

**Date:** 2026-09-10 (audit session) · **updated 2026-09-10 (apply session — deferred in-op items applied)**
**Branch:** `vsureshTT/quasar_uplift_round_2` (PR #54651 `akertesz/concat-port-test` cherry-picked; op source is that PR's Metal 2.0 state on current main)
**Op directory:** `ttnn/cpp/ttnn/operations/data_movement/concat/`
**Quasar model-level test:** `models/experimental/llama32_1b_quasar/tests/graph_ops/test_concat.py`
**Recipe provenance:** `git log -1 -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` prints nothing in this checkout — recipe docs were consumed from a read-only reference copy (`quasar_porting.md` + `metal_2.0/ai/{audit,port,post_port,shared}`), not a tracked doc branch. Working-tree HEAD: `5c482a80f26 2026-09-10`.

This file is a vetting artifact: leave it uncommitted, delete before merge.

---

## Status: GREEN for the captured test path (unchanged); two off-path Metal 2.0 factories uplifted in place

**Reached path (`ConcatProgramFactory`) — GREEN, still a no-op.** Nothing on the path the Quasar test reaches was edited; the routing proof and audit in the previous session stand (kept verbatim in "Why GREEN" below).

**Applied this session (off the captured path, all inside the op directory, all in already-Metal-2.0 factories):**
1. `ConcatS2SRMProgramFactory` — three sync-free **borrowed** DFBs converted to `LocalTensorAccessor` per `sync_free_dfbs.md` ("Borrowed → `LocalTensorAccessor`"). This removes the Gen2-hostile role-free PRODUCER/CONSUMER labelling the audit flagged (§3.3 of the previous report).
2. `ConcatS2STiledProgramFactory` — hand-written `ComputeGen1Config` given its `ComputeGen2Config` alternative per `gen2_hardware_configs.md` shape 4 (compute), arch-selected on the host. **Audit miss from the previous session** (its §2 table said the Gen2 hw_config pass had nothing to add because *the reached path* has no compute kernel; the S2S tiled factory does).

**Still deferred (with why, §3):** the S2S tiled writer's DM self-loop on a **borrowed** DFB (recipe STOP condition, owner/API decision); the S2S RM/tiled local-L1 NOC self-reads (§6 item, reactive, no prescribed code shape); the captured test's `BFLOAT8_B` dtype (model-level); Quasar LLK gaps in the S2S tiled compute kernel (32-bit transpose unsupported; pack-side re-init question); the four legacy `create_descriptor` factories (base Metal 2.0 port is a different recipe).

**One flagged, non-op blocker on the test as captured (unchanged):** the generated case is `BFLOAT8_B`, and `Bfp8_b` is not a Quasar data format. Details in §3.1. It is a model-level dtype decision, not an op edit; the bf16 twin of the same route (`tests/ops/test_concat.py`) is the Quasar command that can pass today.

### Why GREEN on the reached path (from the audit session, unchanged)

1. **The factory the Quasar test reaches is one of the three ported (Metal 2.0) factories.** Proof of routing (`device/concat_device_operation.cpp:20-76`, `device/concat_tiled_unaligned_program_factory.cpp:61-108`):
   - Captured case `00_32x8192_bf8_host`: 16 inputs `[1,1,32,8192]`×15 + `[1,1,32,5376]`, `TILE`, `BFLOAT8_B`, `dim=-1`, output `L1 INTERLEAVED`. List-of-tensors args carry no memory config, so `graph_case.build_tensor` uploads them **DRAM interleaved** (unsharded).
   - `select_program_factory`: `input_tensors[0].is_sharded() == false` → the only two candidates are `ConcatTiledUnalignedProgramFactory` (legacy `create_descriptor`) and `ConcatProgramFactory` (Metal 2.0).
   - `can_use_tiled_unaligned_concat` → `is_supported_tiled_unaligned_config` returns **false** on three independent grounds: (a) `first.device()->arch() == tt::ARCH::QUASAR` is an explicit `return false` (`:80-82`); (b) dtype is not `BFLOAT16`/`FLOAT32` (`:87-90`); (c) every input width is tile-aligned (8192 = 256×32, 5376 = 168×32), so `any_unaligned == false` (`:100-105`).
   - Therefore the test reaches **`ConcatProgramFactory::create_program_artifacts`** (`device/concat_program_factory.cpp`). Same routing on WH/BH (grounds b and c hold there too), so all three archs exercise the identical factory + kernels. **Neither the S2S RM nor the S2S tiled factory is selectable for unsharded inputs** (`concat_device_operation.cpp:28-36`), so this session's edits cannot reach the captured test.
   - The `emulator` marker applies (`graph_ops/conftest.py`): primary input rows 32 ≤ 128, total input footprint ≈ 4.5 MB ≤ 16 MB, no L1-sharded inputs.

2. **Gen1 Metal 2.0 port precondition (recipe §1 step 1) holds on the reached path.** `create_program_artifacts → ProgramArtifacts` with `dfb::`/`args::`/`tensor::` bindings; kernels use device-2.0 APIs (`Noc`, `DataflowBuffer`, `TensorAccessor` via `make_tensor_accessors(tensor::inputs)`, `get_tile_size()`/`get_entry_size()`, `get_arg(args::…)`, `get_vararg`). Zero hits in the op dir for `get_local_cb_interface`, `fifo_*`, `evil_*`, `read_tile_value`/`get_tile_address`, `get_cb_tiles_*`, legacy `noc_async_*`/`cb_*` free functions, `MEM_ZEROS_BASE`, multicast, semaphores, `disable_dfb_implicit_sync*`. The only positional `get_arg_val`/`get_compile_time_arg_val` hits are in `reader_s2s_tensor_concat.cpp` and `reader_writer_block_sharded_concat.cpp`, which belong to the **unported** S2SMulti / BlockSharded factories.

3. **Quasar-uplift audit (`quasar_audit.md`) on the reached path — clean.** One DFB, `src0` — reader `PRODUCER`, writer `CONSUMER` (`concat_program_factory.cpp:268-275, 296-303`). Class 1 canonical cross-kernel FIFO, explicit credits, 2 entries (1 when a page exceeds half of L1). Not a DM self-loop, not borrowed, not multi-bound, no pointer surgery. No `SemaphoreSpec`.

4. **Host-side settings (recipe §4) on the reached path:** `hw_config` from `ttnn::create_reader/writer_datamovement_config(arch)` (shape 1 — Gen2 alternative already supplied); `data_format_metadata` set; no compute kernel; custom `compute_program_hash` untouched.

---

## 1. Files changed (this session)

| File | Reason |
|---|---|
| `device/concat_s2s_rm_program_factory.cpp` | `sync_free_dfbs.md` "Borrowed → `LocalTensorAccessor`": deleted the three borrowed `DataflowBufferSpec`s (`s2s_rm_input_0`, `s2s_rm_input_1`, `s2s_rm_output`) and every `DFBBinding` on them; bound `INPUT_0`/`INPUT_1`/`OUTPUT` as `TensorBinding`s (`output`, `input_0`, `input_1`) on both kernel instances (`:151-154, :170, :177`). Deleted the locals that only fed the DFB specs (`dfb_data_format`, shard unit/page-size arithmetic, `round_up_to_mul32` calls). Repaired three comments and the `.dataflow_buffers` initializer the change falsified. |
| `device/kernels/dataflow/reader_height_sharded_width_concat_two_tensors.cpp` | Same pass, kernel side: `DataflowBuffer` ×3 + `get_write_ptr()`/`get_read_ptr()` → `LocalTensorAccessor<uint8_t>` ×3 + `get_bank_base_address()` (`:38-45, :70`); `dataflow_buffer.h` include swapped for `local_tensor_accessor.h`; the "borrowed-memory buffers" comment repaired. Byte arithmetic and every NOC call untouched. |
| `device/concat_s2s_tiled_program_factory.cpp` | `gen2_hardware_configs.md` shape 4 (compute): the hand-written `ComputeGen1Config` is copied into a `ComputeHardwareConfig` and replaced by a `ComputeGen2Config` when `device().arch() == tt::ARCH::QUASAR` (`:323-331`); `KernelSpec::hw_config` now takes that variant (`:375`). Gen1 initializer textually untouched; `#52269` `unpack_modes` marker added. |
| `QUASAR_UPLIFT_REPORT.md` | This file (updated). |

**Nothing outside the op directory was touched.** No CMake, docs, tests, shared kernels, or `_metal2` forks. No file moved or renamed; namespace `ttnn::prim` unchanged; nothing copied to/from `experimental/quasar/`.

### 1.1 Repaired because the change falsified it (per `pass_procedure.md` Step 3)
- `concat_s2s_rm_program_factory.cpp:89` — "they are DFB bindings now" → "they are tensor bindings now".
- `concat_s2s_rm_program_factory.cpp:148-150, :203-204, :217-218` — comments describing the borrowed DFBs / "no tensor is bound on a KernelSpec" rewritten for the tensor bindings.
- `reader_height_sharded_width_concat_two_tensors.cpp:35-37` — "borrowed-memory buffers … buffer's cursor" comment rewritten.
- `concat_s2s_rm_program_factory.cpp` — `.dataflow_buffers = std::move(dataflow_buffers)` removed (the `Group` no longer exists; the field defaults to empty). The unused-variable set (`dfb_data_format`, `input_dfb_names`, `input_param_names`, the per-input loop, `num_output_units`, `output_unit_size`, `output_page_size`) was removed because each existed solely to populate the deleted specs (would otherwise be `-Werror=unused-variable`).

### 1.2 Noticed, not done (deliberately left; report-only)
- `concat_s2s_rm_program_factory.cpp:16` `#include <tt-metalium/tilize_utils.hpp>` is now unused (it supplied `round_up_to_mul32`). Harmless dead include; left to keep the diff to the transformation.

---

## 2. Applied passes — survey + apply record

### 2.1 `sync_free_dfbs.md` on `ConcatS2SRMProgramFactory` — APPLIED (3 sites, one kernel source, two instances)
- **Survey.** Specs: `s2s_rm_input_0`, `s2s_rm_input_1`, `s2s_rm_output`, all `borrowed_from` set. Binders: `reader{,_first,_last}` (PRODUCER) and `writer{,_first,_last}` (CONSUMER), all instances of the single source `reader_height_sharded_width_concat_two_tensors.cpp`. `grep -nE "reserve_back|push_back|wait_front|pop_front|pages_reservable|pages_available"` on that source: **zero hits**. Handle-follow: the three `DataflowBuffer` locals were used only for `get_write_ptr()` (output, once) and `get_read_ptr()` (each input, once); the handles/ids were never passed to any function, constructor, RAII guard, or template — no opaque callee. `data_format_metadata` was consulted by nothing (pure base-address grabs). No `dfb_run_overrides`. Sync-free everywhere, in every configuration → site. Borrowed → **`LocalTensorAccessor`** (the shared-node scratchpad restriction is for regular-backed buffers only; two kernels each binding the same L1 tensor is ordinary).
- **Apply.** `T = uint8_t` (kernel hands the address to NOC calls, never indexes elements — recipe table row 4). `get_bank_base_address()` replaces both getters 1:1; byte arithmetic kept (recipe: "keep legacy byte arithmetic — do not rewrite into `operator[]`"). Each `TensorParameter` now has ≥1 `TensorBinding` (validator rule), and no DFB is declared at all on this factory.
- **Address parity proof.** Borrowed DFB: `AttachBorrowedDFBBuffers` sets `set_borrowed_memory_base_addr(buffer->address())` from `tensor.mesh_buffer().get_reference_buffer()` (`program_run_args.cpp:~520-555`). TensorBinding: `EmitBindingCrtaValues` writes `tensor.address()` into the address CRTA word (`program_run_args.cpp:429-435`), which `LocalTensorAccessor(TensorBindingToken)` reads via `get_common_arg_val` (`local_tensor_accessor.h:53-58`). Same buffer, same address, so `get_bank_base_address()` == the former `get_read/write_ptr()` (a borrowed, never-advanced DFB pointer sits at its base). Every NOC read/write therefore hits the same source and destination bytes as before.
- **Residency check moved, not changed.** Before: host `TT_FATAL` "DFB … borrows memory from TensorParameter … not L1-resident" (`program_spec.cpp:1599-1604`). After: `LocalTensorAccessor`'s `static_assert(!is_dram)` at kernel JIT. Both reject a DRAM-sharded input; the routing (`concat_device_operation.cpp:52-59`) plus the kernel's `my_x/my_y` self-reads mean this path was only ever meaningful for L1 shards.
- **Quasar effect.** No DFB remains on this factory, so the Gen2 "roles become real / CONSUMER cannot write" concern (§3.3 of the previous report) is moot; nothing here spends a DFB id or tile counter.

### 2.2 `gen2_hardware_configs.md` on `ConcatS2STiledProgramFactory` — APPLIED (1 site, shape 4 compute)
- **Survey.** `grep -rn "hw_config\|to_compute_hardware_config\|Gen1Config\|std::get<\|std::get_if<\|holds_alternative"` over the op dir: reader/writer configs in all three ported factories are shape 1 (arch-agnostic DM helpers — no work); `concat_s2s_tiled_program_factory.cpp:315-319` is a hand-written `ComputeGen1Config` (shape 4). No `std::get`/`get_if` anywhere (no shape 3). `ConcatProgramFactory` and `ConcatS2SRMProgramFactory` have no compute kernel.
- **Apply.** Field dispositions from the recipe table: `fpu_math_fidelity` (copy, `HiFi4`), `enable_32_bit_dest` (copy, `fp32_dest_acc_en`), `unpack_modes` (copy verbatim from the Gen1 struct + `TODO(#52269)` marker), `sfpu_precision_mode`/`double_buffer_dest` not set on Gen1 → not written, `bfp_pack_precision_mode` dropped (no Gen2 field), `enable_2x_src_register` never set. `arch` read once in the added code → no hoisted local (the factory already calls `inputs[0].get().device().arch()` twice for the DM configs; those are untouched). Designated-initializer order checked against `ComputeGen2Config` (`fpu_math_fidelity`, `sfpu_precision_mode`, `enable_32_bit_dest`, `double_buffer_dest`, `unpack_modes`) — in order, trailing comma. `KernelSpec::hw_config` is `std::variant<DataMovementHardwareConfig, ComputeHardwareConfig>`; it now receives the `ComputeHardwareConfig` variant by move (exact alternative).
- **Gen1 path unchanged.** The `ComputeGen1Config` initializer is byte-identical; on WH/BH `compute_hw` holds a copy of it and the `if` is not taken.
- **Caveat the recipe requires stated:** nothing on this bench checks the Gen2 *values*; the branch is correct by transcription only. And the S2S tiled path still cannot run on Quasar until §3.2 is resolved, so this is one blocker removed, not a green light.

### 2.3 Considered on the other ported factories, not applicable / not a site
| Pass / gotcha | Verdict |
|---|---|
| `dm_self_loop_dfbs.md` on S2S tiled `OUTPUT_DFB` | **STOP** (survey step 5: `borrowed_from = OUTPUT`). See §3.2. |
| `sync_free_dfbs.md` on S2S tiled | `INPUT0/1_DFB` (borrowed, reader `push_back` → compute `wait/pop`), transpose/concat/output_transpose DFBs, `OUTPUT_DFB` (`reserve_back`/`push_back`) — all carry FIFO calls → not sites. |
| `sync_free_dfbs.md` / `dm_self_loop_dfbs.md` on `ConcatProgramFactory` | `src0` is a real cross-kernel FIFO → not a site (unchanged verdict). |
| TEN-4746 bare `wait_front→pop_front` / `reserve_back→push_back` in compute (`kernels/compute/height_sharded_width_concat_two_tensors.cpp:13-32`) | `transpose()` orders wait→pop with a real `transpose_tile` (UNPACR) and reserve→push with a real `pack_tile` (PACR); no BARE / CONFIG-ONLY / GUARDED-PATH shape (the `if constexpr` branches all route through `transpose()`). Not a site. |
| Re-`*_init` on DFB-id change (§7) | `transpose()` calls `transpose_init(dfb_in_id)` before every `transpose_tile` → unpack/math side covered. **Pack side is an open question** — see §3.4. Reactive per §2; not edited. |
| `hw_startup` once | `compute_kernel_hw_startup(dfb::input0, dfb::input0_transpose)` at `main()` start only. OK. |
| uint16/uint32 code path (§7) | Only a host-side bool (`fp32_dest_acc_en` includes `UInt32`); the op forwards dtype and has no kernel format branch. Format-layer limitation, flag not edit (§3.1). |
| Local L1→L1 NOC self-reads (§6) | Present in S2S RM reader and S2S tiled reader/writer (`my_x/my_y` loopback). Reactive; no prescribed code shape. See §3.3. |
| `disable_dfb_implicit_sync_*`, multicast, semaphores, borrow-with-offset, `MEM_ZEROS_BASE`, `evil_*`, `fifo_page_size`, `-Werror=int-to-pointer-cast` | None anywhere in the op (unchanged). |
| `opt_level` | Compute kernel carries explicit `O3` (matches legacy); DM defaults coincide. Not an uplift edit. |

---

## 3. Deferred / follow-up items (exact symptoms)

### 3.1 Test-as-captured: `BFLOAT8_B` is not a Quasar data format (model-level dtype decision) — unchanged
- **Where it fails:** `ValidateProgramSpec`, `tt_metal/impl/metal2_host_api/program_spec.cpp:~1738-1746`: `TT_FATAL(tt::is_data_format_supported(dfb.data_format_metadata.value(), arch), …)` — `is_supported_quasar()` (`tt_metal/common/tt_backend_api_types.cpp:97-121`) has no `Bfp8_b` (Gen2 replaced BFP with MXFP). `datatype_to_dataformat_converter(BFLOAT8_B) == Bfp8_b`.
- **Expected symptom on Quasar:** host-side fatal before launch: `DFB 'src0' has data format 'Bfp8_b' which is not supported on architecture QUASAR` (or an earlier tensor-upload rejection; either way it is the format layer).
- **Why not fixed in the op:** concat forwards the input dtype and has no format branch (recipe §7: flag, do not edit). An `ARCH_QUASAR` dtype swap would be an unguarded functional change to the op's contract. Dropping `data_format_metadata` is not possible (`get_tile_size()` derives from it; `Invalid` throws).
- **Owner action:** llama32_1b_quasar model owner picks bf16 (or MXFP) for the lm_head concat on the emulator; regenerate the `GENERATED FILE` from a bf16 capture, or use the bf16 twin `models/experimental/llama32_1b_quasar/tests/ops/test_concat.py`.

### 3.2 S2S tiled writer: DM self-loop on a **borrowed** DFB — STOP per `dm_self_loop_dfbs.md`, owner/API decision
- **Site:** `device/concat_s2s_tiled_program_factory.cpp:268-290` binds `OUTPUT_DFB` (`borrowed_from = OUTPUT`, `:139-145`) as both PRODUCER and CONSUMER on the DM writer; `kernels/dataflow/writer_height_sharded_width_concat_two_tensors_tiled.cpp:33-57` does `get_write_ptr()` once, then per row `reserve_back(w)` … `push_back(w)` with no re-read of the pointer (pure bookkeeping; one trip round the buffer).
- **Quasar symptom:** `ValidateProgramSpec` rejects at program creation (`program_spec.cpp:1495-1498`): `… Self-loop DFBs are not supported for data-movement kernels on Gen2 architectures. Consider using a scratchpad or LocalTensorAccessor instead.`
- **Why not applied:** `dm_self_loop_dfbs.md` survey step 5 — "If `borrowed_from` is set, stop and report … the destination cannot be a scratchpad — it would have to be a `LocalTensorAccessor` over the borrowed tensor, and fake-FIFO bookkeeping over borrowed memory is a combination nothing in this suite has examined." The obvious conversion (bind `OUTPUT` as a `TensorBinding` on the writer, `LocalTensorAccessor<uint8_t> output(tensor::output)`, drop the two FIFO calls, keep `base_l1_write_addr = output.get_bank_base_address()` and the running `l1_write_addr`) is exactly what the recipe declines to prescribe; and `cb_dfb_quasar_audit_helper.md` says of a DM sole-toucher producer: **STOP — surface to API owner**. Do **not** silence the validator by binding the unused endpoint to the reader/compute (recipe: explicitly forbidden).
- **Owner action:** confirm the writer never needs FIFO semantics on the output (nothing downstream drains it; compute never touches `OUTPUT_DFB`), then either sanction the LTA + drop-FIFO shape for this pattern in `dm_self_loop_dfbs.md`, or give the runtime team a "borrowed DM self-loop" case. Until then the height-sharded tiled `dim=3` two-tensor concat cannot run on Quasar.

### 3.3 Local L1→L1 NOC self-reads (`my_x/my_y` loopback) — §6 reactive item, not applied
- **Sites:** `reader_height_sharded_width_concat_two_tensors.cpp:46-92` (S2S RM), `reader_height_sharded_width_concat_two_tensors_tiled.cpp:59-137`, `writer_height_sharded_width_concat_two_tensors_tiled.cpp:44-51` (S2S tiled). Each issues `noc.async_read(UnicastEndpoint{…my_x, my_y…}, CoreLocalMem dst …)` with source and destination both in this core's L1.
- **Recipe text (§6):** "Local self-read/self-copy (src==dst L1) on the emulator can spin on `can_post` or silently drop the read — use a direct L1→L1 RISC copy, not a NoC loopback."
- **Why not applied:** §2 says §7–§8-style fixes are reactive ("apply one only when its symptom actually fires"); no device run has happened; the recipe gives no code shape for the RISC copy (cached vs. uncached alias choice on Quasar DM, alignment, `memcpy` vs. word loop), so writing one would be inventing. The S2S tiled sites are also unreachable on Quasar until §3.2 clears.
- **Symptom to watch (S2S RM on the emulator):** reader/writer stuck in `noc.async_read_with_state` / `async_read_barrier` (waypoint in the `NAR*`/`NRB*` family) or a clean run whose output shard is zeros/stale in the concatenated columns. If it fires: `#ifdef ARCH_QUASAR` a direct L1→L1 copy of `group_stick_size_{0,1}` bytes per group in place of the loopback, WH/BH keep the NOC read.

### 3.4 S2S tiled compute kernel — Quasar LLK / compute-API gaps (flag, not edit)
- **32-bit transpose unsupported on Quasar:** `api/compute/transpose.h:83-84` (Quasar branch of `transpose_init`) — `LLK_ASSERT(!enable_unpack_to_dest, "32-bit (unpack-to-dest) transpose not supported on Quasar")  // TODO: tt-llk#1559`. Fires for `Float32`/`Int32` inputs (the `fp32_dest_acc_en` cases). LLK-team item.
- **Pack-side re-init on output-DFB switch (§7 "BFDs live in the init"):** the kernel switches its pack destination `input0_transpose → input1_transpose → output_transpose` every row with only `pack_reconfig_data_format(...)` between (`kernels/compute/height_sharded_width_concat_two_tensors.cpp:60, :77`); `transpose_init(icb)` re-programs unpack/math only and the transpose API has no `(icb, ocb)` init form. On Quasar the pack BFD is baked at pack init, so this may pack to the wrong DFB (symptom: all-zero or misplaced transpose outputs on Quasar, correct on WH/BH). Needs a compute-API/LLK answer (a pack init exposed for transpose, or confirmation `pack_reconfig_data_format` re-targets on Quasar). Unverifiable until §3.2 clears; not edited.

### 3.5 Style debt on the reached path (unchanged, not applied — GREEN path, no unforced changes)
- `reader_concat_interleaved_start_id.cpp:52-58` / `reader_concat_stick_layout_interleaved_start_id.cpp:49-75` peek `dfb_in.get_write_ptr()` into a `CoreLocalMem` NOC destination rather than passing the DFB (`{.offset_bytes}`) as the destination. Functionally identical on all archs (`Noc::get_dst_ptr` routes `LOCAL_L1` through `l1_cached_view()`; implicit sync attaches only with `NocOptions::TXN_ID`).
- No DM kernel calls `dfb.finish()`; balanced explicit credits make the Quasar drain trivially satisfied. **Watch item:** a hang at kernel exit / next launch's DFB constructor (`DFW`) → runtime team, do not sprinkle `finish()`.

### 3.6 Unported (legacy `create_descriptor`) factories — out of this recipe's scope
`S2SMulti`, `S2I`, `BlockSharded`, `TiledUnaligned`: RED "not Metal 2.0 yet" if any Quasar path selects them. `TiledUnaligned` self-gates on Quasar; the other three are sharded-input paths the captured model does not hit. Base Metal 2.0 port (`metal2_port.md`) first.

---

## 4. Parity claim (WH/BH)

**Reached path: zero diff** — `ConcatProgramFactory`, its two readers and the two donor writers are byte-identical to the cherry-picked PR #54651 state; WH/BH and Quasar select this factory for the captured geometry (§Why GREEN), so the model-level tests below exercise exactly the unchanged kernels.

**S2S RM (`ConcatS2SRMProgramFactory`): behaviour-preserving by construction** (`sync_free_dfbs.md` header: "Results, numerics and observable behaviour are identical"). Argument: (i) the three deleted DFBs had **no** FIFO calls anywhere, so no synchronization was removed; (ii) their only use was a base-address read, and §2.1 shows `LocalTensorAccessor::get_bank_base_address()` yields the very same `buffer->address()` the borrowed DFB was attached with; (iii) every NOC transfer (`set_async_read_state` / `async_read_with_state` / `async_read_barrier`), every stride and every byte offset is textually unchanged; (iv) no L1 is allocated differently — borrowed DFBs never allocated L1, and no scratchpad was introduced. The only observable difference is the error surface for a DRAM-sharded input (host `TT_FATAL` before, kernel-JIT `static_assert` now), a configuration the path never supported. No `ARCH_QUASAR` guard was needed because the change is arch-neutral.

**S2S tiled (`ConcatS2STiledProgramFactory`): Gen1 textually unchanged.** The `ComputeGen1Config` initializer is untouched; on WH/BH `compute_hw` is a copy of it and the `arch == QUASAR` branch is dead. The reader, writer and compute kernels were not edited.

Structural argument per recipe §9 ("auditing without a device run"): a zero-or-arch-neutral-equivalent diff on WH/BH ⇒ no behaviour change. Confirm with the commands below; the S2S RM path is exercised by `test_sharded_concat` (ROW_MAJOR HEIGHT cases, 2 inputs), `test_sharded_concat_with_groups` (ROW_MAJOR rows), `test_concat_sharded_pad`; the S2S tiled path by `test_sharded_concat` (TILE HEIGHT, 2 inputs — incl. the unet-decoder regression case) and `test_sharded_concat_with_groups` (TILE rows), all in `tests/ttnn/unit_tests/operations/data_movement/test_concat.py`.

---

## 5. Test commands (user runs; order BH → WH → Quasar)

Kernels changed (`reader_height_sharded_width_concat_two_tensors.cpp`), so **force JIT** and purge any stale kernel cache: `export TT_METAL_FORCE_JIT_COMPILE=1`. Watcher on for every run: `export TT_METAL_WATCHER=10` (`unset` to turn off; `=0` still enables). Build first (host factories changed): `./build_metal.sh` (plus the install step your setup needs so python loads the rebuilt `_ttnn.so`).

### Blackhole (parity)
```bash
cd /localdev/vsuresh/tt-metal && source python_env/bin/activate
export TT_METAL_WATCHER=10 TT_METAL_FORCE_JIT_COMPILE=1
# S2S RM + S2S tiled paths (the edited factories) — run these first:
pytest tests/ttnn/unit_tests/operations/data_movement/test_concat.py -v -k "test_sharded_concat or test_concat_sharded_pad"
# full op suites:
pytest tests/ttnn/unit_tests/operations/data_movement/test_concat.py -v
pytest tests/ttnn/unit_tests/operations/data_movement/test_concat_program_cache.py -v
pytest tests/ttnn/nightly/unit_tests/operations/data_movement/test_concat.py -v
# same geometry as the Quasar case (reached path, unchanged), bf16 + bf8, on real HW:
pytest models/experimental/llama32_1b_quasar/tests/ops/test_concat.py -v
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_concat.py -v
```

### Wormhole (parity)
```bash
cd /localdev/vsuresh/tt-metal && source python_env/bin/activate
export TT_METAL_WATCHER=10 TT_METAL_FORCE_JIT_COMPILE=1
pytest tests/ttnn/unit_tests/operations/data_movement/test_concat.py -v -k "test_sharded_concat or test_concat_sharded_pad"
pytest tests/ttnn/unit_tests/operations/data_movement/test_concat.py -v
pytest tests/ttnn/unit_tests/operations/data_movement/test_concat_program_cache.py -v
pytest tests/ttnn/nightly/unit_tests/operations/data_movement/test_concat.py -v
pytest models/experimental/llama32_1b_quasar/tests/ops/test_concat.py -v
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_concat.py -v
```
Baseline: PR #54651 reports the confirmed `test_concat.py` set at 672 passed / 20 skipped / 3 xfailed on WH+BH; expect the same. (`test_sharded_concat_with_groups` has a pre-existing watcher-on N300 skip, #29557.)

### Quasar emulator
Run twice: once with `TT_METAL_LLK_ASSERTS=1`, once unset (§9). No `tt-triage`/`tt-exalens` on the emulator — WATCHER + host gdb.
```bash
cd /localdev/vsuresh/tt-metal && source python_env/bin/activate
export TT_METAL_WATCHER=10 TT_METAL_FORCE_JIT_COMPILE=1
export TT_METAL_LLK_ASSERTS=1        # first pass; then `unset TT_METAL_LLK_ASSERTS` and repeat

# (1) bf16 twin of the captured route (reached path, unchanged) — the one that can pass today:
pytest models/experimental/llama32_1b_quasar/tests/ops/test_concat.py -v

# (2) the captured bfloat8_b case — EXPECTED to fail at ValidateProgramSpec with
#     "DFB 'src0' has data format 'Bfp8_b' which is not supported on architecture QUASAR"
#     until the model/test moves to bf16 (or MXFP). Run to confirm the symptom text:
pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_concat.py -m emulator -v
TTNN_GRAPH_OPS_NO_GOLDEN=1 pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_concat.py -m emulator -v

# (3) OPTIONAL — the uplifted S2S RM path (height-sharded RM, 2 inputs, dim=3). Run the
#     ROW_MAJOR + bfloat16 rows of test_sharded_concat_with_groups (the first five shape rows; the
#     CoreGrid(x=1,y=1) ones fit the emulator grid). Use `--collect-only -q` first to pick the exact
#     ids, then:
pytest tests/ttnn/unit_tests/operations/data_movement/test_concat.py -v -k "test_sharded_concat_with_groups"
#     If it hangs in the reader (NOC self-read) or produces zeros -> §3.3 symptom; report before editing.

# (4) NOT runnable yet — S2S tiled (height-sharded TILE, 2 inputs, dim=3) is expected to stop at
#     ValidateProgramSpec: "Self-loop DFBs are not supported for data-movement kernels on Gen2" (§3.2).
```
If (1) hangs: `WFW`/`RBW` waypoints → DFB credit stall, report as a runtime regression (do **not** re-disable implicit sync, §7/§8.2); `DFW`/`AAW` → see §3.5 `finish()` watch item.

---

## 6. Definition-of-done checklist (recipe §10)

- [x] In place, existing directory + namespace; nothing copied to `experimental/quasar/`, no `::qsr`. (Nothing from that tree was read or cited.)
- [x] `create_program_artifacts`/`ProgramArtifacts`; `dfb::`/`args::`/`tensor::`; no `CBIndex::c_`, positional `get_arg_val`, or `TensorAccessorArgs` on any ported factory.
- [x] `opt_level`: compute `O3` explicit (S2S tiled); DM defaults coincide. Untouched.
- [x] Every remaining DFB has `data_format_metadata`; kernels use `get_tile_size()`/`get_entry_size()`.
- [x] Sync-free DFBs converted: S2S RM's three borrowed sync-free DFBs → `LocalTensorAccessor` (§2.1). DM self-loop: S2S tiled `OUTPUT_DFB` is a recipe STOP (borrowed) — deferred with symptom (§3.2).
- [x] No `disable_dfb_implicit_sync_*`.
- [x] No borrow-with-offset.
- [x] No multicast.
- [x] Re-`*_init` on DFB-id change: unpack/math covered by `transpose()`; pack side flagged (§3.4).
- [x] No bare compute `wait_front→pop_front` / `reserve_back→push_back` (§2.3).
- [x] No semaphores.
- [x] Gen2 `hw_config` on every ported factory: DM via helpers (shape 1); S2S tiled compute via added `ComputeGen2Config` branch (§2.2).
- [ ] BH and WH pass — **user to run** (§5).
- [ ] Quasar builds and runs — **user to run** (§5); bf8 case blocked at format layer (§3.1); S2S tiled blocked at validator (§3.2).
- [x] Every Quasar-specific change is arch-guarded (host `arch == QUASAR` in S2S tiled) or arch-neutral-equivalent (S2S RM LTA conversion; no `ARCH_QUASAR` needed).
- [x] No DIAG/debug leftovers.
- [x] Missing-feature flags fed back: `Bfp8_b` on Quasar (format layer); borrowed DM self-loop (recipe/API gap, §3.2); 32-bit transpose on Quasar tt-llk#1559 and pack-side re-init for transpose (§3.4).
- [x] This report updated; RED-stop conditions checked — none fires for the reached path; the S2S tiled path carries an owner-decision STOP (§3.2), recorded, not forced.

---

## craq-sim run 2026-09-10

**Environment.** Quasar functional simulator (craq-sim, `TT_METAL_SIMULATOR=/localdev/vsuresh/qsr-sim/libttsim.so`, SoC descriptor `arch_name: QUASAR`, 8×4 = 32 functional workers, 4 MB L1/core, 2 DRAM channels), `TT_METAL_SLOW_DISPATCH_MODE=1`, `TT_METAL_FORCE_JIT_COMPILE=1`, private kernel cache `TT_METAL_CACHE=/localdev/vsuresh/tt-metal-cache-qsr-concat`. Host libs = the current working tree (branch `vsureshTT/quasar_uplift_round_2`, HEAD `5c482a80f26`, plus the uncommitted op edits listed in §1); **no host rebuild was needed** (no factory/host file was edited this session). Every passing case was run twice: `TT_METAL_LLK_ASSERTS=1 TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1` and both unset. The watcher/debug env was never needed — nothing hung or faulted. Tests were serialized through `qsr_test` (shared lock). Logs: `<scratchpad>/qsr_concat/*.log`.

### Results

| # | Test / case | Factory reached | Result | Detail |
|---|---|---|---|---|
| 1a | `tests/ops/test_concat.py` as written, `two_equal` | — (never reaches concat) | **FAIL (helper op, not concat)** | `TT_FATAL @ tt_metal/impl/kernels/kernel.hpp:450 … DataMovementKernel is not supported on Quasar. Use QuasarDataMovementKernel instead.` Raised from `ttnn.from_torch(..., layout=TILE, device=mesh)` → `convert_python_tensor_to_tt_tensor` → `ttnn::to_layout` → `ttnn::prim::tilize` (legacy `create_descriptor` op). `can_construct_on_device()` (`ttnn/core/tensor/py_to_tt_tensor.cpp:65-99`) says yes for a bf16 TILE upload, so the upload runs the on-device tilize op. Owner: `tilize` op / `py_to_tt_tensor.cpp` (an arch gate there, or a Metal 2.0 tilize). Worked around in a temp copy (1b). |
| 1b | temp copy `tests/ops/test_tmp_concat_bf16_qsr.py` (identical, but host-tilize + `ttnn.to_device` upload — the pattern `tests/ops/test_to_device.py` already uses), 3 cases `two_equal`, `four_chunks`, `uneven_tail` | `ConcatProgramFactory` (`reader_concat_interleaved_start_id` + `writer_unary_interleaved_start_id_metal2` in the JIT cache) | **PASS ×3, asserts on and off** | PCC = **1.0** on all three (floor 0.999). ~10 s per 3-case run. |
| 2a | `tests/graph_ops/test_concat.py[00_32x8192_bf8_host]` as captured (bf8, 16 inputs) | `ConcatProgramFactory` (fails at `ValidateProgramSpec`, before launch) | **FAIL — expected (§3.1, model-level dtype)** | `TT_FATAL @ tt_metal/impl/metal2_host_api/program_spec.cpp:1746: tt::is_data_format_supported(dfb.data_format_metadata.value(), arch)` — `DFB 'src0' has data format 'Bfp8_b' which is not supported on architecture quasar`. (The bf8 upload itself succeeded: BFP tensors are tilized on host.) |
| 2b | temp copy `tests/graph_ops/test_tmp_concat_bf16_qsr.py` (sed `BFLOAT8_B`→`BFLOAT16`, plus the same host-tilize `build_tensor` workaround as 1b because bf16 hits 1a's tilize path) | `ConcatProgramFactory` | **PASS, asserts on and off** | Golden `torch.cat` PCC floor 0.999 met; output spec check passed (shape `[1,1,32,128256]`, bf16, TILE, L1 INTERLEAVED). 16 inputs × 32×8192 + 32×5376 ≈ 8 MB bf16 → 8 MB L1-interleaved output over 32 cores. ~8.7 s. |
| 3a | `tests/ttnn/.../test_concat.py::test_sharded_concat_with_groups` RM bf16 CoreGrid(1,1): `((1,1,1,32),(1,1,1,32))` groups=1; `((1,1,1,32),(1,1,1,64))` groups=4 | `ConcatS2SRMProgramFactory` (uplifted this branch; `reader_height_sharded_width_concat_two_tensors` in the JIT cache) | **PASS ×2 (asserts on)** | Exact match (`assert_equal`). The §3.3 local L1→L1 NOC self-read (`my_x/my_y` loopback) did **not** hang or drop data on craq-sim — §3.3 stays a watch item, not a fix. |
| 3b | same test, RM bf16 CoreGrid(4,1) `((1,1,1024,64),(1,1,1024,32))` groups=2 (multi-core, 256 rows/core) | `ConcatS2SRMProgramFactory` | **PASS (asserts on)** | Exact match. |
| 3c | same test, TILE bf16 CoreGrid(1,1) `((1,1,32,32),(1,1,32,32))` groups=1 | `ConcatS2STiledProgramFactory` (fails at `ValidateProgramSpec`) | **FAIL — expected (§3.2 STOP)** | `TT_FATAL @ program_spec.cpp:1500: !(is_gen2_arch(hal) && self_loop_kernel->is_data_movement_kernel())` — `DataflowBuffer 's2s_tiled_output' is self-looped by data-movement kernel 'writer' (bound as both PRODUCER and CONSUMER). Self-loop DFBs are not supported for data-movement kernels on Gen2 architectures. Consider using a scratchpad or LocalTensorAccessor instead.` Exactly the symptom §3.2 predicted; the borrowed-DFB DM self-loop remains an owner/API decision (recipe STOP), not forced. |

No SKIPs fired: every case run fits the 8×4 sim grid (the CoreGrid(8,1) rows of test 3 were not run — the sim's 8-wide grid would fit them, but they were out of the 1–2-case budget). No HANGs.

### Fixes applied this session
**None to the op.** No symptom fired on any concat path that reached the device, so per the recipe's reactive rule (§2/§7) nothing in `ttnn/cpp/ttnn/operations/data_movement/concat/` was edited; the op-dir diff is unchanged from §1 (`concat_s2s_rm_program_factory.cpp`, `concat_s2s_tiled_program_factory.cpp`, `reader_height_sharded_width_concat_two_tensors.cpp`). The `ConcatS2SRMProgramFactory` LocalTensorAccessor conversion (§2.1) is now **device-validated on Quasar** (3a/3b), and the reached `ConcatProgramFactory` path is validated end-to-end at the captured geometry in bf16 (2b).

Test-side workarounds (temp copies only, both **deleted** after the run):
- `tests/ops/test_tmp_concat_bf16_qsr.py` — inputs built on host (`ttnn.from_torch(..., layout=TILE)` without `device=`) then `ttnn.to_device(..., DRAM_MEMORY_CONFIG)`; plus a `QSR_PCC[...]` print of the PCC value.
- `tests/graph_ops/test_tmp_concat_bf16_qsr.py` — `sed BFLOAT8_B→BFLOAT16` of the generated file, case id renamed `00_32x8192_bf16_host_tmp`, and `G.build_tensor` monkeypatched to the same host-build + `to_device` upload.

### Open blockers (symptom → owner)
1. **`ttnn.from_torch(bf16, TILE, device=…)` runs the legacy on-device `tilize` on Quasar** → `kernel.hpp:450 DataMovementKernel is not supported on Quasar`. Blocks the *unmodified* `tests/ops/test_concat.py` (and any Quasar test that uploads bf16 TILE via `from_torch(device=...)`). Fix belongs in `ttnn/core/tensor/py_to_tt_tensor.cpp` (`can_construct_on_device`, gate on arch / Metal 2.0 tilize availability) or in the `tilize` op's Metal 2.0 port. Owner: tensor-upload / tilize op owners. Not a concat item.
2. **`Bfp8_b` not a Quasar data format** (`program_spec.cpp:1746`) — blocks the captured bf8 case; model-level dtype decision (§3.1). Owner: llama32_1b_quasar model owner (regenerate the capture in bf16/MXFP). The bf16 twin of the exact captured geometry passes.
3. **S2S tiled writer DM self-loop on borrowed DFB** (`program_spec.cpp:1500`) — blocks height-sharded TILE 2-tensor `dim=3` concat on Quasar (§3.2). Owner: recipe/API owner (sanction LTA + drop-FIFO for borrowed DM self-loops, or a runtime case). §3.4 (32-bit transpose, pack re-init) remains unverifiable behind it.
4. Legacy `create_descriptor` factories (`S2SMulti`, `S2I`, `BlockSharded`, `TiledUnaligned`) — not exercised; RED on Quasar if selected (§3.6).

### Reproduce
```bash
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env.sh concat
# 1a (fails in from_torch tilize, not concat):
qsr_test timeout 2400 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/ops/test_concat.py -k two_equal -v -s
# 1b / 2b: recreate the temp copies described above (host tilize + ttnn.to_device), then run them with and without
#          TT_METAL_LLK_ASSERTS=1 TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1 ; delete afterwards.
# 2a (expected Bfp8_b format failure):
qsr_test timeout 2400 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_concat.py -v -s
# 3a/3b (S2S RM, pass) and 3c (S2S tiled, expected self-loop validator failure):
T=tests/ttnn/unit_tests/operations/data_movement/test_concat.py
TT_METAL_LLK_ASSERTS=1 qsr_test timeout 2400 ./python_env/bin/python -m pytest \
 "$T::test_sharded_concat_with_groups[input_shapes=((1, 1, 1, 32), (1, 1, 1, 32))-output_shape=(1, 1, 1, 64)-core_grid=ttnn.CoreGrid(x=1, y=1)-layout=Layout.ROW_MAJOR-dtype=DataType.BFLOAT16-groups=1-dim=3]" \
 "$T::test_sharded_concat_with_groups[input_shapes=((1, 1, 1, 32), (1, 1, 1, 64))-output_shape=(1, 1, 1, 96)-core_grid=ttnn.CoreGrid(x=1, y=1)-layout=Layout.ROW_MAJOR-dtype=DataType.BFLOAT16-groups=4-dim=3]" \
 "$T::test_sharded_concat_with_groups[input_shapes=((1, 1, 1024, 64), (1, 1, 1024, 32))-output_shape=(1, 1, 1024, 96)-core_grid=ttnn.CoreGrid(x=4, y=1)-layout=Layout.ROW_MAJOR-dtype=DataType.BFLOAT16-groups=2-dim=3]" \
 "$T::test_sharded_concat_with_groups[input_shapes=((1, 1, 32, 32), (1, 1, 32, 32))-output_shape=(1, 1, 32, 64)-core_grid=ttnn.CoreGrid(x=1, y=1)-layout=Layout.TILE-dtype=DataType.BFLOAT16-groups=1-dim=3]" -v -s
```
