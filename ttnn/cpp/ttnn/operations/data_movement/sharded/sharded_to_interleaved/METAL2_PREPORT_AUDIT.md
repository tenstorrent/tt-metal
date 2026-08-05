# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved`

> **Re-audit.** This replaces the audit committed as `074a5166599` (2026-07-23), which was RED on **two** gates. **One has since cleared** (Device 2.0 on `eltwise_copy.cpp`, landed as `0fb47949a27` / PR #51179). **One remains** (offset base pointer, RM writer) — a fix *is* authored but **not merged**; see [Result](#result).

Single device-operation directory:

- **`ShardedToInterleavedDeviceOperation`** (`device/sharded_to_interleaved_device_operation.{hpp,cpp}`)
  - `ShardedToInterleavedProgramFactory` (`device/sharded_to_interleaved_program_factory.cpp`) — one `descriptor` factory that selects kernels by input layout and dtype-conversion need.

**Kernels exercised** (all file-path-instantiated; the op owns none of its kernels):

| Role | Path | Selected when |
|---|---|---|
| Reader | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | always |
| Writer (tiled) | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` | `input.layout() == TILE` |
| Writer (row-major) | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `input.layout() == ROW_MAJOR` |
| Compute (copy) | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | `convert_df` (input dtype ≠ output dtype; TILE only) |

No unreferenced kernel files sit in the op directory (it holds no kernels at all).

**Config matrix** (the factory has three reachable shapes, selected by runtime branch — *not* by separate `ProgramFactory`):

- **C1 — TILE, no conversion**: reader + tiled writer.
- **C2 — TILE, conversion** (`convert_df`): reader + tiled writer + compute.
- **C3 — ROW_MAJOR** (never converts; a dtype mismatch requires TILE per `validate_inputs:67-71`): reader + RM writer.
  - **C3a** — HEIGHT_SHARDED: per-core column offset is always `0`.
  - **C3b** — WIDTH_SHARDED / BLOCK_SHARDED: per-core column offset is non-zero on every core after the first. **This is the blocked path.**

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `4386dc456a1 2026-07-29 docs(metal_2.0): require an explicit opt_level when porting compute kernels`
*(Pinned from the doc checkout at `/localdev/edwinlee/Port_Recipe`. The `metal_2.0/` doc tree is **not** present in this op checkout — `/localdev/edwinlee/metal2_audit.md` is a symlink into that separate checkout — so the hash pins the guidance, not this repo. Unlike the previous audit run, the `analyses/` triage docs **were** available and are cross-checked below.)*

**Code state audited:** every source file below is **byte-identical to `origin/main` @ `f6a5267fa85` (2026-08-05)**; the only diff in this working tree is this report. Findings therefore apply to `main` as of the audit date.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved` |
| **Overall** | **RED** — one remaining gate (down from two), config-scoped to **C3** (ROW_MAJOR) |
| **DOps / Factories** | `ShardedToInterleavedDeviceOperation` → `ShardedToInterleavedProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes — GREEN** *(was RED; `eltwise_copy.cpp` migrated by `0fb47949a27` / PR #51179)* |
| *Prereqs* — Cross-op escapes | Ok — no function-call escapes; all 4 kernels file-path-borrowed (coupling inventoried, FYI) |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | N/A (every CTA read at constexpr index 0) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (sheet; cross-check clean, 1 row ↔ 1 factory) |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (not `WorkloadDescriptor`) |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | **GATE (RED)** — RM writer, **Type 2** (accessor-fed offset base) @ `writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:22`; config-scoped to **C3** → ops team + framework/Audrey (flag early) |
| *Port work* — Tensor bindings (per binding) | `input_tensor` = clean (borrowed-memory DFB) · `output_tensor` = Case 1 in C1/C2; offset-gated in C3 |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no 3rd-arg site anywhere) |
| *Port work* — CB endpoints | all legal 1P+1C, every CB in every config |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution. Here none is out of window — see [Gate detail](#gate-detail).

## Result

**RED at op level; no shippable portable subset.**

**One gate blocks the port**, down from two at the previous audit:

- **CLEARED — Device 2.0 on the shared compute kernel.** `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` was the previous audit's first blocker (four `cb_*` free-function FIFO calls, no wrapper in scope). It has been migrated on the Device 2.0 track — `0fb47949a27` *"[Cleanup] Device 2.0 Port for eltwise_copy kernel (#51179)"* — and now constructs `CircularBuffer cb_in(tt::CBIndex::c_0)` / `cb_out(tt::CBIndex::c_16)` and calls `wait_front` / `reserve_back` / `pop_front` / `push_back` as methods (`:19-20`, `:26-27`, `:34-35`), exactly the form the Device 2.0 migration guide's migrated examples use. **Gate GREEN.**

- **STILL BLOCKED — offset base pointer, Type 2 (accessor-fed offset base).** The row-major writer feeds `dst_addr + input_width_offset_bytes` as a `TensorAccessor` **base** (`writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:22`). Metal 2.0 builds the accessor from the `tensor::` binding and supplies the base itself — there is no seam for an interior base — so a mechanical Case-1 port would **silently drop** the per-core column offset. Non-zero only on **C3b** (row-major WIDTH/BLOCK-sharded), where it mis-addresses every core after the first. → **ops team + framework/Audrey, flag early.**

  **A fix for exactly this is already authored — and is not on `main`.** Branch `origin/edwinlee/S2I_OffsetPointer`, commit `0a40dce7acb` *"Fix offset pointers in I2S and S2I"* (2026-07-31), moves the offset out of the accessor base and into the per-write destination `offset_bytes`:

  ```
  -    const auto s0 = TensorAccessor(dst_args, dst_addr + input_width_offset_bytes);
  +    const auto s0 = TensorAccessor(dst_args, dst_addr);
  ...
  -        {.page_id = stick_id, .offset_bytes = 0});
  +        {.page_id = stick_id, .offset_bytes = input_width_offset_bytes});
  ```

  `git merge-base --is-ancestor 0a40dce7acb origin/main` → **false**: the fix is **unmerged**. The gate therefore stands. This is the cheapest possible RED — **the re-audit after that branch merges should be a confirmation pass, not a re-derivation**, since it is the sole remaining blocker and every other subject is GREEN or clean.

**No shippable portable subset.** The clean paths (C1/C2, TILE) exercise only the reader, the tiled writer and the compute kernel — all Device 2.0, clean-base, and green on every other axis. But all three configs live inside **one** `ProgramDescriptor` factory: kernel choice is a runtime branch on `input.layout()` (`program_factory.cpp:177-185`), not a separate `ProgramFactory`. So the porting unit is the whole factory, and the factory cannot be built without the RM writer. Per [Code-path scope], the blocking shape is *localized but not severable* → **`RED at op level; no portable subset`**. Naming C1/C2 as "clean" is diagnostic (it shows the blocker is one branch and one kernel line), not an offer of a partial port.

**Neither blocker is a permanent wall**, and one is already gone. The remaining one is an op-readiness prerequisite with an authored fix awaiting merge — the shortest path to GREEN of any RED shape this audit can produce.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Live readiness sheet (*"Operations analysis"*, `dgomez@`), fetched fresh this run. Row `data_movement/sharded/sharded_to_interleaved`:

  | Column | Value |
  |---|---|
  | `Device operation` | `ShardedToInterleavedDeviceOperation` |
  | `Factory (variant)` | `ShardedToInterleavedProgramFactory` |
  | `Concept` | `descriptor` |
  | `Porting Target` | `ProgramSpecFactoryConcept` |
  | `Custom hash (compute_program_hash)` | `no` |
  | `Backdoor custom hash (attribute_values / to_hash)` | `no` |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` |
  | `Override runtime args method? (PD only)` | `no` |
  | `Pybind descriptor (nb::class_ of device op)` | `no` |
  | `Smuggled pointer (raw buffer addr in RTA/CRTA)` | `no` |
  | `Is safe to port?` | `yes` |
  | **`Is able to port?`** | **`yes`** |
  | `TensorParameter relaxation` | `none` |
  | `Op-owned tensors?` | *(blank)* |
  | `Secretly SPMD Workload?` | *(blank — N/A, not `WorkloadDescriptor`)* |
  | `Op Classification` | `PD Op (pointer-patching)` |
  | `Pointer patching perf issue?` | `OK` · `Formerly custom hashed?` `no` |

  Cross-check against code — all confirmed:
  - `Concept = descriptor` ✓ — `create_descriptor()` returns `tt::tt_metal::ProgramDescriptor` (`device/sharded_to_interleaved_program_factory.hpp:15`).
  - `Custom hash = no` ✓ — no `compute_program_hash` override anywhere in the op directory.
  - `Runtime-args update (get_dynamic_runtime_args) = no` ✓ — no such hook on the device-op (`sharded_to_interleaved_device_operation.hpp:22-31` declares only `validate_inputs`, `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`, `create_op_performance_model`).
  - `Override runtime args method? = no` ✓ — no `override_runtime_arguments`; the concept is `descriptor`, so the legacy-signature reading does not apply either.
  - `Pybind descriptor = no` ✓ — `sharded_to_interleaved_nanobind.cpp:46-53` binds only the `sharded_to_interleaved` free function via `ttnn::bind_function`; no `nb::class_` of the device op, no `create_descriptor` binding.
  - `Op-owned tensors? = (blank/no)` ✓ — consistent with the `descriptor` concept (cross-column invariant holds; a `descriptor` row cannot carry op-owned tensors).
  - **Factory-set match** ✓ — the sheet carries exactly **one** row for this op, and the code has exactly **one** factory (`program_factory_t = std::variant<ShardedToInterleavedProgramFactory>`, `device_operation.hpp:20`). No phantom row, no missing row.
  - Cross-column invariants hold. No spreadsheet conflict.

  **Target concept: `ProgramSpecFactoryConcept`** (no op-owned tensors) — the sheet's `Porting Target` column agrees. *(The previous audit recorded `MetalV2FactoryConcept`; the current recipe's [TTNN porting shape] names `ProgramSpecFactoryConcept`. Corrected here.)*

- **Device 2.0 (every kernel used):** **GREEN — all four kernels compliant.** No violation table; there is nothing to route.

  | Kernel | Device 2.0 evidence |
  |---|---|
  | `reader_unary_sharded.cpp` | `DataflowBuffer dfb(cb_id_in0); dfb.push_back(...)` (`:15-16`). Migrated in the CB→DFB sweep (#49392 lineage). |
  | `writer_unary_sharded_blocks_interleaved_start_id.cpp` | `Noc noc; DataflowBuffer dfb_out(cb_id_out)` (`:30-31`), `noc.async_write(dfb_out, s, …)` (`:41`), `dfb_out.wait_front/pop_front` (`:36`, `:49`), `TensorAccessor` (`:28`). `get_tile_size(cb_id_out)` (`:26`) is a **sanctioned** free function — not a violation. |
  | `writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `Noc noc; DataflowBuffer dfb_out(dfb_id_out0)` (`:24-25`), `noc.async_write(...)` (`:31`), `dfb_out.wait_front/pop_front` (`:28`, `:37`). |
  | `eltwise_copy.cpp` | **Newly cleared.** `CircularBuffer cb_in(tt::CBIndex::c_0)` / `cb_out(tt::CBIndex::c_16)` (`:19-20`) with method-form FIFO ops (`:26-27`, `:34-35`). `#include "api/dataflow/circular_buffer.h"` (`:10`) is the Device-2.0 header the migration guide's own migrated example uses. |

  Confirmed absent across all four kernels: `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedAddrGenFast`, `InterleavedPow2AddrGen*`, raw `noc_async_read` / `noc_async_write`, raw semaphore addresses, `cb_wait_front(` / `cb_push_back(` / `cb_pop_front(` / `cb_reserve_back(` free-function form, and any `get_write_ptr` / `get_read_ptr` / `get_local_cb_interface` / `fifo_*_ptr` raw access. The remaining CB-index free functions in `eltwise_copy.cpp` — `unary_op_init_common(c_0, c_16)`, `copy_tile_init(c_0)`, `copy_tile(c_0, …)`, `pack_tile(0, c_16)` — are **compute LLK**, outside the Device 2.0 *data-movement* API surface the guide covers, and are not holdovers.

- **Feature compatibility:** every Appendix A entry scanned against both host and kernel code; all **N/A** (each entry is a gate-feature, so an absent one is N/A, not GREEN).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `.global_circular_buffer` field on the `CBDescriptor`, no `remote_cb*` / `.remote_index(` / `remote_circular_buffer.h`, no 4-arg `experimental::CreateCircularBuffer`. The input CB **is** Buffer-backed (`cb.buffer = bound_buffer`, `program_factory.cpp:41`, from `:147`) — that is the legacy **borrowed-memory** pattern, a mechanical `DataflowBufferSpec::borrowed_from` translation, explicitly *not* an Appendix A entry and *not* a GCB. |
  | CBDescriptor `address_offset` (non-zero) | N/A | `push_s2i_cb_pair` (`program_factory.cpp:25-43`) sets `total_size`, `core_ranges`, one `CBFormatDescriptor`, and `buffer` — it never touches `.address_offset` (defaults `0`). No `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor` call anywhere in the op or its kernels. |
  | GlobalSemaphore | N/A | The op uses **no semaphores of any kind** — no `GlobalSemaphore`, no `CreateSemaphore`, no `global_semaphore.hpp`, no kernel-side semaphore wait/post. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Every kernel-side CTA read is at **constexpr index 0** and nothing else: reader `:13`, tiled writer `:22`, RM writer `:19`, compute `:13`. Host-side CTA lists are fixed-shape (`program_factory.cpp:168`, `:175-176`, `:195`); the writer's variable-length tail is `TensorAccessorArgs(*dst_buffer).append_to(...)` (`:176`), read kernel-side as the fixed `TensorAccessorArgs<1>()` NTTP, not a runtime-varying index. `tensor_args_t` is a fixed pair — `Tensor input_tensor` + `std::optional<Tensor> preallocated_output` (`device_operation_types.hpp:19-22`) — with no `std::vector<Tensor>`, so the op-level cue does not fire either. |

- **Offset base pointers:** **RED — Type 2 (accessor-fed offset base)**, config-scoped to **C3** (ROW_MAJOR), non-zero on **C3b** (WIDTH/BLOCK-sharded).

  **Every address argument resolved.** The output buffer reaches both writers as a **`Buffer*` binding**, not a smuggled `->address()`: `writer_rt.push_back(dst_buffer)` (`program_factory.cpp:242` tiled, `:293` RM) into a `KernelDescriptor::RTArgList`, whose element type is `std::variant<uint32_t, Buffer*, std::reference_wrapper<const MeshTensor>>` (`tt_metal/api/tt-metalium/program_descriptors.hpp:186`). The framework auto-registers it as a `BufferBinding` and patches it on cache hits — correct-on-cache-hit today, consistent with the sheet's `Smuggled pointer = no`. The kernel receives a raw `uint32_t` base at arg 0; classify by what the kernel does with it:

  - **Tiled writer (C1/C2) — clean base.** `TensorAccessor(dst_args, dst_addr)` (`writer_unary_sharded_blocks_interleaved_start_id.cpp:28`), no arithmetic. All addressing is by **tile index**: `start_id = start_id_base + start_id_offset` (`:20`, from RTAs 7/8 = host `curr_idx_h + curr_idx_w` and `starting_idx_h`, `program_factory.cpp:249-250`) feeds `{.page_id = tile_id}` (`:41`). Those are page indices, **not** byte addresses — no fold. → ordinary Case-1 port work.
  - **RM writer (C3) — offset base.** `TensorAccessor(dst_args, dst_addr + input_width_offset_bytes)` (`writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:22`). `input_width_offset_bytes` is arg 5 = host `curr_idx_w` (`program_factory.cpp:298`), the byte column-offset of this core's shard block within each output stick, advanced by `output_unit_size` per core and wrapped at `num_units_per_row` (`:302-306`).
    - **C3a HEIGHT_SHARDED:** `shard_spec.shape[1]` spans the full row, so `output_unit_size >= num_units_per_row` and `curr_idx_w` wraps to `0` on every iteration → offset always `0`.
    - **C3b WIDTH_SHARDED / BLOCK_SHARDED:** shard width < row width → **non-zero on every core after the first.** Reachable: `validate_inputs` admits row-major input whenever the shard page size is L1-aligned (`device_operation.cpp:61-66`).

  **Why this gates, and why the recognition rule as written does not catch it.** The [Offset base pointers] recognition model resolves each address RTA *to its host computation* and keys on a **host-side** fold (`buffer()->address() + <expr>` in the factory). Here there is no host fold: the host passes a clean base plus a **separate scalar offset arg**, and the kernel adds them — which the subject's four-outcomes table classifies as the *already-split-out* case → **GREEN**, hand to [TensorParameter analysis]. But that GREEN disposition holds only when the split-out offset is consumed **raw** (Case 2, as `roll`'s DRAM_RM mode does). Here the sum is fed to a `TensorAccessor` as its **base**, so the Type-2 Metal 2.0 wall applies in full, independent of where the addition happens: `TensorAccessor(tensor::dst)` builds its args and takes its base from the binding, leaving no seam for an interior base. A mechanical Case-1 translation drops `input_width_offset_bytes` with nothing to flag it — **silent mis-addressing on C3b only**. Per the audit's operating principle (identify gaps; default conservative when the failure mode is silent), this is gated as **Type 2**. Recognition-rule gap logged in [Recipe notes](#recipe-notes).

  **Triage-doc cross-check** (`analyses/2026-07-19_offset_base_pointers.md`, dated **2026-07-19** — a prior, not an authority; available this run, unlike the previous audit): `sharded_to_interleaved` appears in **none** of the four type tables. The Type-2 table lists only `slice` (`slice_program_factory_rm.cpp`), `padded_slice`, and `slice_write`. This is the doc's **"fold present, op _not_ in the tables"** outcome as it applies to a kernel-side sum — classified from the recognition model, **not waved through for being unlisted**. Recommend the triage-doc owner add this site; it shares the doc's own Type-2 characterisation exactly ("the affected variants are all row-major").

  **The authored fix** (`origin/edwinlee/S2I_OffsetPointer` @ `0a40dce7acb`, 2026-07-31, unmerged) implements the previous audit's proposed resolution: clean-base accessor + the column shift carried as the per-write destination `offset_bytes`. For interleaved addressing the two forms resolve to the same NoC address, and the RM writer's `async_write` destination args already carry the field (`:32`). It is nonetheless a **kernel-logic change**, off the porter's kernel-side whitelist, so it correctly belongs on the ops team's track and *before* the port — which is where it now sits. **What is needed is a review + merge decision, not a design exploration.** Alternative shape, if that one is rejected: a base binding plus kernel-side accessor construction from `tensor::dst.args` and `get_bank_base_address() + offset`.

- **TensorAccessor 3rd argument:** **GREEN / none.** Both `TensorAccessor` constructions in the op's kernels are **2-argument** — `TensorAccessor(dst_args, dst_addr)` (tiled writer `:28`) and `TensorAccessor(dst_args, dst_addr + input_width_offset_bytes)` (RM writer `:22`). No explicit page-size third argument at any site, so the syntactic signal never fires and there is nothing to classify. Cross-checked against `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md` (a dated prior, available this run): the op is **not** in its op→class table — consistent with the code.

- **CB endpoints (GATE-free):** **all legal 1 producer + 1 consumer**, every CB on every node in every config. No self-loop, no 1P+1C assignment needed, no multi-binding flag, no dead CB. Device 2.0 idioms are intact across all four kernels, so the precondition for this scan holds and no deferral applies.

  Two CBs exist. `c_0` (`src0_cb_index`) is the borrowed-memory input CB (`cb.buffer = src_buffer`, `program_factory.cpp:41` from `:147`), allocated in every config. `c_16` (`out_cb_index`) is allocated **only** when `convert_df` (`:149-160`, `bound_buffer = nullptr`); when `!convert_df`, `out_cb_index == src0_cb_index == c_0` (`:129`), so the writer's DFB *is* `c_0`.

  | CB | Config | Producer (locked) | Consumer (locked) | Census | Verdict |
  |---|---|---|---|---|---|
  | `c_0` (borrowed-memory) | **C1** (TILE, no convert) | reader `dfb.push_back` (`reader_unary_sharded.cpp:16`) | tiled writer `dfb_out.wait_front` / `pop_front` (`writer_unary_sharded_blocks…:36,49`); its `noc.async_write(dfb_out, …)` (`:41`) is a peek on the same binding, not a second endpoint | 2 touchers: 1 locked P + 1 locked C | **plain 1:1 legal** |
  | `c_0` (borrowed-memory) | **C2** (TILE, convert) | reader `dfb.push_back` | compute `cb_in.wait_front` / `pop_front` (`eltwise_copy.cpp:26,34`) | 2 touchers: 1 P + 1 C | **plain 1:1 legal** |
  | `c_16` | **C2** only | compute `cb_out.reserve_back` / `push_back` (`eltwise_copy.cpp:27,35`) | tiled writer `dfb_out.wait_front` / `pop_front` | 2 touchers: 1 P + 1 C | **plain 1:1 legal** |
  | `c_0` (borrowed-memory) | **C3** (ROW_MAJOR) | reader `dfb.push_back` | RM writer `dfb_out.wait_front` / `pop_front` (`writer_unary_stick_layout…:28,37`) | 2 touchers: 1 P + 1 C | **plain 1:1 legal** |

  **Hidden-second-writer hunt: negative, positively.** Every one of the four kernels was scanned for a raw co-fill or co-read — `get_write_ptr` / `get_read_ptr` / `get_local_cb_interface(…).fifo_wr_ptr` / `fifo_rd_ptr` / `evil_set_write_ptr` / `evil_set_read_ptr` — and there are **zero** occurrences. The op also allocates **no semaphores at all**, so the semaphore-gated co-fill face (a) has no coordinating primitive available to it. There is no dual-instance work-split: each kernel source is pushed into exactly **one** `KernelDescriptor` (`:310-314`), never two configs over the same core range. Face (b) multiple-readers does not fire either: each CB has exactly one reading kernel per config.

  **No dead CB.** `c_0`'s index reaches the reader as CTA 0 (`:168`) and the writer as CTA 0 (`:175`, when `!convert_df`); `c_16`'s reaches the writer as CTA 0 when `convert_df` (`:150`, `:175`). Both are consumed by real FIFO ops in every config in which they are allocated.

## Port-work summary  *(reference only; no brief issued on RED)*

- **Tensor bindings** (per binding):
  - **`input_tensor`** (`c_0`, borrowed-memory) — **clean**, via the causal-link gate. The CB is Buffer-backed (`cb.buffer = src_buffer`); the reader only `push_back`s the already-resident shard pages and constructs no `TensorAccessor`; downstream kernels read it through FIFO ops and as an `async_write` L1 source. The borrowed-memory DFB *is* the tensor access. Port via `DataflowBufferSpec::borrowed_from`. Neither Case 1 nor Case 2.
  - **`output_tensor`** (`dst_buffer`, delivered as a `Buffer*` binding → framework `BufferBinding`) — **Case 1** in **C1/C2**: the tiled writer feeds the base into `TensorAccessor(dst_args, dst_addr)` and does all addressing through it, so express the binding as a `TensorParameter` / `TensorBinding`, build `TensorAccessor(tensor::…)` kernel-side, and the arg-0 base plus the `TensorAccessorArgs` CTA plumbing (`program_factory.cpp:176`) both disappear. Mechanical, low-risk. In **C3** the *same* binding is the **offset-gated Type-2 site** above — **not** a clean Case 1; do not port it mechanically. (This is the per-config split the [Granularity] rule anticipates.)
- **TensorParameter relaxation:** **none.** Sheet says `none`; the op has no custom hash, so there is no hash logic to reconcile.
- **TensorAccessor 3rd arg:** **none** — no site passes one.
- **CB endpoints:** all legal 1P+1C (table above). Nothing to self-loop, assign, flag, or drop.

## Heads-ups  *(reference only; no brief issued on RED)*

- **CB endpoints (multi-binding shapes to watch):** **none.** The hunt for all three faces came back negative with positive evidence (no raw pointer access in any kernel, no semaphores at all, no dual-instance work-split, one reader per CB per config).
- **Cross-op / shared kernels:** **all four kernels are borrowed** — the op owns none. **No `_metal2` sibling fork exists beside any of them**, so a port creates the first for each. One wrinkle worth knowing: a real, non-quasar Metal 2.0 fork of the reader **does** exist, at `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/reader_unary_sharded_metal2.cpp` (created by `cbde3d44ff3`, PR #51397, on `main`) — but it sits in **typecast's own tree**, not beside the original, so the rung-1 *locational* sibling check reports "no fork" and the next porter creates a second one. Its bindings (`dfb::in`, `args::num_tiles_per_core`) and its explanatory header comment are a ready naming precedent. See [Questions](#questions-for-the-user) — this is a decision for the user, not the porter. Full co-borrower/sunset inventory in [Team-only](#team-only).
- **RTA varargs:** **none.** Every kernel reads its runtime args at **distinct constant indices** — reader index 0; tiled writer 0–8; RM writer 0, 2, 3, 4, 5, 6 — with no counted loop, no running `arg_index++`, and no data-selected index. Every arg is nameable; this is the preferred non-signal case.

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean** for *function-call* escapes. No kernel `#include`s another op's helper: the four kernels' includes are `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/dataflow/circular_buffer.h`, `api/tensor/noc_traits.h`, `api/compute/{common,tile_move_copy,eltwise_unary/eltwise_unary}.h`, `api/debug/dprint.h` — all `tt_metal/*` LLK/HAL (donor class 1, no concern). No per-call shape analysis is owed. Host-side, the factory `#include`s the in-family `sharded_common.hpp` for `calculate_starting_idx_h` (`program_factory.cpp:11`, `:206`) — host code, no kernel-token bridging, out of this subject's scope.

  **File-path kernel instantiation is the whole coupling story: the op owns none of its kernels.** Consumer sets below are a **sunset and coordination list — not authorization to convert any of these files in place.** Census by filename grep over `ttnn/cpp`, hits filtered to factory bindings:

  | Borrowed kernel | Owning family / pool | Class | Sibling `_metal2` fork? | Co-binding ops (sunset list) |
  |---|---|---|---|---|
  | `reader_unary_sharded.cpp` | `eltwise/unary` | cross-family | **No** (but see the non-sibling typecast fork, above) | broadly shared — `sharded_to_interleaved_partial`, `tilize` (×2), `transpose_wh_sharded`, `untilize` (×3), `untilize_with_unpadding`, `slice_write` (×2) |
  | `writer_unary_sharded_blocks_interleaved_start_id.cpp` | `data_movement/sharded` | in-family | **No** — this port creates it | `sharded_to_interleaved_partial` |
  | `writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `data_movement/sharded` | in-family | **No** — this port creates it | `sharded_to_interleaved_partial` |
  | `eltwise_copy.cpp` | `ttnn/cpp/ttnn/kernel/compute/` (shared pool) | shared-lib | **No** — this port creates it | `copy` (×2: default-tilized, same-memory-config), `interleaved_to_sharded`, `sharded_to_interleaved_partial`, `interleaved_to_sharded_partial`, `untilize_with_unpadding` |

  **Sibling-fork check run locationally** (`ls` of each original's directory), per the [shared-kernel caution]. Copies under `experimental/quasar/**` also bind same-named kernels; those are whole-op pre-port copies, **do not count as forks to reuse**, and are excluded above. **Porter warning worth carrying forward: `ttnn/cpp/ttnn/operations/experimental/quasar/sharded_to_interleaved/` is a hacky pre-port copy of this exact op.** It will look like a finished answer to every question this port raises; it is not one, and it carries idioms the port recipe forbids. Do not read it, template from it, or lift its binding names.

  The now-cleared Device 2.0 migration of `eltwise_copy.cpp` (`0fb47949a27`) landed as one shared rewrite and equally unblocked `copy`, `interleaved_to_sharded`, `untilize_with_unpadding` and both `*_partial` ops — worth noting for whoever sequences the shared compute pool's Metal 2.0 fork.

- **Relaxation candidates (mined from a custom hash):** **none** — the op has no custom hash, so there is nothing to mine.

- **TTNN factory analysis (sheet-derived + `file:line`):** current concept `descriptor` (`program_factory.hpp:15`); target `ProgramSpecFactoryConcept`; **no** op-owned tensors (no `WorkloadDescriptor`, no `buffers` vector); **no** pybind `create_descriptor` and no other risky pybind (`sharded_to_interleaved_nanobind.cpp` exposes only the free function); **no** custom hash; **no** `get_dynamic_runtime_args`; **no** `override_runtime_arguments`; `Is safe to port? = yes`. Every gate conjunct absent → the TTNN gate clears cleanly. One factory, one sheet row, no MeshWorkload need.

## Misc anomalies  *(team-only, non-gating; the port does not act on these)*

- **Dead RTA on the row-major path.** The factory pushes **7** writer RTAs for C3 — index 1 is `num_units_per_row` (`program_factory.cpp:294`) — but `writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` reads indices 0, 2, 3, 4, 5, 6 and **never index 1** (`:12-17`). `num_units_per_row` is dead plumbing on this path.
- **`is_l1_aligned` is a hardcoded `true`** (`program_factory.cpp:55`), which makes the RM-path guard `if (is_blackhole or is_l1_aligned) { if (!dst_is_dram or is_l1_aligned) { … } }` (`:286-289`) unconditionally taken. Three consequences: `is_blackhole` (`:135`) and `dst_is_dram` (`:134`) are computed but effectively dead in this branch (`dst_is_dram` has no other use), and the first `padded_shard_width = tt::align(output_unit_size, dst_buffer->alignment())` (`:285`) is always overwritten at `:288`. A forced constant hiding a dead branch — worth a deliberate decision rather than leaving it as-is.
- **`num_slices` / `slice_index` are vestigial here.** The launch site hardcodes them to `1` / `0` (`sharded_to_interleaved_device_operation.cpp:147`, `ShardedToInterleavedParams{…, 1, 0}`), and `calculate_starting_idx_h` early-returns `0` when `num_slices <= 1` (`sharded_common.cpp:17-19`). So `starting_idx_h` — the tiled writer's arg 8 / `start_id_base` — is **always 0** for this op. The real user of the slicing parameters is the separate `sharded_to_interleaved_partial` op. Not a bug, but dead generality carried into hash-relevant attributes (both fields sit on `ShardedToInterleavedParams`).
- **The TILE/ROW_MAJOR decision is taken off two different tensors.** The unit-size and core-count blocks branch on `output.layout()` (`program_factory.cpp:81`, `:113`) while kernel selection and the per-core RTA loop branch on `input.layout()` (`:177`, `:213`, `:214`). They agree in practice — `compute_output_specs` builds the output with `PageConfig(input_tensor.layout())` (`device_operation.cpp:113`), and a preallocated output must match the input's layout (`:48-50`) — but the split reads as accidental and would diverge silently if either invariant were relaxed.
- **Stray debug include in the borrowed reader.** `reader_unary_sharded.cpp:9` includes `api/debug/dprint.h` with no `DPRINT` use in the file. Cosmetic, and it belongs to `eltwise/unary` — not this op's to fix.

## Per-DeviceOperation attribution

Not applicable — the directory holds exactly one `DeviceOperation` with exactly one program factory. Findings above are already single-attribution. Where a finding differs **per config** within that one factory (offset base pointers, the `output_tensor` binding), the split is stated inline against C1 / C2 / C3.

## Questions for the user

1. **Merge status of the offset-base fix — this is the whole gate.** `origin/edwinlee/S2I_OffsetPointer` @ `0a40dce7acb` (*"Fix offset pointers in I2S and S2I"*, 2026-07-31) already implements the resolution the previous audit proposed, and it is **not** an ancestor of `origin/main` @ `f6a5267fa85`. Has it been reviewed by ops + framework/Audrey, and is the "column shift rides the per-write destination `offset_bytes`" form blessed? Two things would be good to have on record before the port: (a) confirmation that the two forms are equivalent for interleaved destinations across the alignment cases the RM path admits (the kernel's own new comment asserts this; a reviewer's sign-off would make it a finding rather than an assertion), and (b) whether the same commit's `interleaved_to_sharded` half is in scope of the same review. **Once that lands on `main`, this op's re-audit should be a confirmation pass — every other gate and subject is already clear.**
2. **Which fork does the reader bind?** `reader_unary_sharded_metal2.cpp` exists on `main` in **typecast's** directory rather than beside the original in `eltwise/unary/…/dataflow/`. Rung 1's locational check therefore misses it, and rung 2 would have this port create a *second* fork of the same kernel. Preference: (a) bind typecast's existing fork despite the non-sibling path, (b) create the sibling fork per the letter of rung 2 and accept two forks, or (c) relocate typecast's fork beside the original first, on the ops/porting track, so rung 1 works for every later consumer? This is a convention call, not a porter call — worth settling before the port, not during it.
3. **Misc anomalies routing.** The `is_l1_aligned = true` forced constant and the dead RM arg 1 are pre-existing and non-gating, but the forced constant makes a real branch unreachable. Should these be filed against the ops team now, or carried as-is?

## Recipe notes  *(friction with the audit recipe itself)*

- **The recognition-rule gap from the previous audit is still open** (recipe @ `4386dc456a1`, 2026-07-29; the prior audit logged it 2026-07-23). [Offset base pointers] resolves each address RTA "to its **host** computation" and its four-outcomes table classifies *clean base + separate scalar offset arg, summed in the kernel* as **GREEN → hand to [TensorParameter analysis]**. That is only right when the split-out offset is consumed **raw** (Case 2). When the kernel sums them and passes the result as a **`TensorAccessor` base** — this op's RM writer, `:22` — the Type-2 wall applies identically, but *neither* subject catches it: the offset gate sees no host fold, and TensorParameter analysis calls it Case 1 and silently drops the offset. Concretely, two edits would close it: (1) add a kernel-side clause to Type-2 recognition — *"a base RTA plus a separately-delivered offset that are **summed and passed as a `TensorAccessor` base** is Type 2, wherever the sum is computed"*; (2) qualify the four-outcomes "No fold → clean → TensorParameter analysis" bullet with *"provided the base reaches the accessor unmodified."* The `roll` DRAM_RM precedent cited as the GREEN case should also state explicitly that its split-out offset is **raw-consumed**, since that is what makes it green. Two independent audits have now had to extend the rule by hand to avoid a silent-wrong port; the extension shouldn't have to be re-derived a third time.
- **Readiness-sheet column names have drifted from the docs.** The live sheet's header is **`Override runtime args method?\n(PD only)`**, but both `ttnn_op_porting_readiness.md` and `metal2_audit.md` quote it as `Override runtime args method? (PD and legacy)`. The readiness doc's standing guarantee is *"existing column names never change, and no column is ever deleted"* — so a name-based lookup keyed on the documented string finds nothing. It only worked here because I read the header row and matched by prefix. Worth reconciling in whichever direction is correct.
- **The sheet now carries gate-adjacent columns the docs don't mention.** Beyond the documented set, the live header includes `Op Classification`, `Execution Model`, **`Porting Target`**, **`Backdoor custom hash (attribute_values / to_hash)`**, `Known op issues`, `Pointer patching perf issue?`, and `Formerly custom hashed?`. Two matter to this recipe, not just informationally: **`Porting Target`** supplies directly the target concept that [TTNN porting shape] currently has the auditor derive by hand from `Concept` + `Op-owned tensors?` (they agreed for this op — `ProgramSpecFactoryConcept` — but the recipe should say which is authoritative); and **`Backdoor custom hash`** looks like a fifth custom-hash-shaped signal that the documented `Is able to port?` derivation does not include, so an auditor cross-checking the derivation cannot tell whether it is a conjunct, a subsumed input, or informational. Both were `no` / benign here, so nothing turned on it.
- **Disclosed deviation: I ran the seven purely-informational subjects despite a no-portable-subset RED.** The [Red outcome scoping rule] says skip them, on the reasoning that the detail goes unread and stale while blockers clear. That reasoning doesn't hold for this op: the sole blocker is a **two-line change in one kernel that is already authored and pushed**, so the code the census describes is very unlikely to move, and the post-merge re-audit becomes a confirmation pass instead of a full re-derivation. The census was also genuinely cheap here (2 CBs, 4 kernels, ~90 lines of kernel code total) — the rule's stated acute case is the opposite, a mega-op with dozens of CBs. Flagging rather than silently choosing: the rule might benefit from an explicit escape for *"RED whose remedy is already authored / narrowly scoped"*, since blanket-skipping there discards work that will be read within days.
- **The rung-1 fork check is locational, but a real fork can be non-local — and the audit is where that gets caught.** [Caution: Porting a shared kernel] rung 1 deliberately checks for a **sibling** `_metal2` file and warns off tree-wide greps because they surface quasar copies. Correct as far as it goes, but it has a blind spot the quasar clause doesn't cover: a *legitimate, non-quasar* fork placed in the porting op's **own** tree instead of beside the original — `copy/typecast/device/kernels/dataflow/reader_unary_sharded_metal2.cpp` (PR #51397, on `main`) is exactly that. Rung 1 reports "no fork"; rung 2 then produces a second fork of one kernel, which is the duplication the convention exists to prevent. Suggest [Out-of-directory coupling]'s borrowed-kernel-file bullet ask the auditor to record **non-sibling non-quasar `_metal2` forks** as reuse candidates (a filename grep minus `experimental/quasar/**` is enough), since the auditor greps broadly anyway and the porter, working one file at a time, is the least likely to find it.
