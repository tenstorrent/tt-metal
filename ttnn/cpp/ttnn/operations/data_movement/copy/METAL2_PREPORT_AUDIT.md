# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/copy`

One device operation, three program factories (all `descriptor` concept):

- **`CopyDeviceOperation`** (`device/copy_device_operation.hpp` / `.cpp`)
  - `SameMemoryConfig` (`device/copy_same_memory_config_program_factory.cpp`)
  - `DefaultRowMajor` (`device/copy_default_row_major_program_factory.cpp`)
  - `DefaultTilized` (`device/copy_default_tilized_program_factory.cpp`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `b11662e579e 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/copy` |
| **Overall** | **RED** — blocked on **Device 2.0** (confined to `SameMemoryConfig` row-major kernels); subset `{DefaultRowMajor, DefaultTilized}` is clear |
| **DOps / Factories** | `CopyDeviceOperation` → `SameMemoryConfig`, `DefaultRowMajor`, `DefaultTilized` |
| *Prereqs* — Device 2.0 (every kernel used) | **No (RED)** — 3 Device-1.0 kernels on `SameMemoryConfig`'s RM paths (broad Device 1.0; full migration). `DefaultRowMajor` + `DefaultTilized` kernels all Device 2.0. → Device 2.0 track |
| *Prereqs* — Cross-op escapes | Ok (workable; `DefaultTilized` file-path-borrows 3 broadly-shared kernels, all Device 2.0) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — GlobalCircularBuffer / address_offset / GlobalSemaphore / Variadic-CTA | All N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** — all 3 factory rows `yes` |
| *TTNN Readiness* — Concept (current) | `descriptor` (all 3 factories) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (not a WorkloadDescriptor op) |
| *TTNN Readiness* — Is safe to port? | Yes (all 3 rows) |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | none (no host-folded offsets; op not in triage tables) |
| *Port work* — Tensor bindings (per binding) | all Case 1 (via `TensorAccessor`) — clean subset |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no site passes a 3rd argument) |
| *Port work* — CB endpoints | `DefaultRowMajor`: self-loop `c_0` + legal `c_1`; `DefaultTilized`: all legal 1:1 |

**CB endpoints** are dispositions, not gates. Recorded per `(CB, config)` below for the clean subset only; the blocked `SameMemoryConfig` factory's census is deferred to re-audit per the Red-outcome scoping rule.

## Result

**RED at op level; subset `{DefaultRowMajor, DefaultTilized}` is clear.**

The **only** blocker is the **Device 2.0 prerequisite**, and it is **confined to the `SameMemoryConfig` factory's row-major code paths**: three data-movement kernels are still on Device 1.0 idioms (`ShardedAddrGen`, free-function CB management, raw `noc_async_*`). Every other gate is GREEN for all three factories. `DefaultRowMajor` and `DefaultTilized` use exclusively Device-2.0 kernels and clear every gate — a **brief is issued for that subset** (`METAL2_PORT_BRIEF.md`).

Path forward: the three kernels need Device 2.0 data-movement migration (an **op-readiness prerequisite**, owned by the Device 2.0 team). Once they migrate, `SameMemoryConfig` re-audits and is expected to clear (its tilized path is already Device 2.0; the RM paths are the only debt). This is a prerequisite fix, not a permanent blocker.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN** — the readiness sheet ("Operations analysis", fetched fresh this run) carries all three factory rows for `data_movement/copy` / `CopyDeviceOperation` (`DefaultRowMajor`, `DefaultTilized`, `SameMemoryConfig`), each `Is able to port? == yes`. Cross-check clean:
  - `Concept == descriptor` — confirmed: all three factories define `create_descriptor()` returning `ProgramDescriptor` (`copy_device_operation.hpp:24-42`). No `create_workload_descriptor`, no MetalV2 factory.
  - `Custom hash == no` — confirmed: no `compute_program_hash` override anywhere in the op (grep clean).
  - `get_dynamic_runtime_args == no` — confirmed: hook absent from `CopyDeviceOperation`.
  - `override_runtime_arguments == no` — confirmed: method absent.
  - `Pybind descriptor == no` — confirmed: `copy_nanobind.cpp` binds only the `ttnn::copy` free function; no `nb::class_` of the device op, no `create_descriptor` binding.
  - `Op-owned tensors == (blank/no)` — confirmed: `descriptor` concept, no `WorkloadDescriptor::buffers`.
  - **Factory-set match** — the sheet's three factory rows map one-to-one onto the code's three factories. No phantom/missing rows.
  - Cross-column invariants hold (no `get_dynamic_runtime_args` on a legacy concept; no op-owned tensors on a `descriptor` row).

- **Device 2.0 (every kernel used):** **RED** — three kernels used by `SameMemoryConfig`'s **row-major** code paths are broadly Device 1.0 (full migration, not isolated CB-index holdovers). Routed to the **Device 2.0 team**. `SameMemoryConfig`'s *tilized* path and the entire `DefaultRowMajor` and `DefaultTilized` factories are Device-2.0 compliant.

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `device/kernels/reader_unary_stick_start_id.cpp` (copy-owned) | 34 | `experimental::ShardedAddrGen<...>` (`.bank_base_address = src_addr`) | none — pre-Device-2.0 addr-gen (Shape 4) |
  | `device/kernels/reader_unary_stick_start_id.cpp` | 49 | `noc_async_read(src_noc_addr, l1_write_addr, stick_size)` | none — raw NoC |
  | `device/kernels/reader_unary_stick_start_id.cpp` | 50 | `noc_async_read_barrier()` | none — raw NoC |
  | `device/kernels/writer_unary_stick_start_id.cpp` (copy-owned) | 34 | `experimental::ShardedAddrGen<...>` (`.bank_base_address = dst_addr`) | none — pre-Device-2.0 addr-gen (Shape 4) |
  | `device/kernels/writer_unary_stick_start_id.cpp` | 49 | `noc_async_write(l1_read_addr, dst_noc_addr, stick_size)` | none — raw NoC |
  | `device/kernels/writer_unary_stick_start_id.cpp` | 50 | `noc_async_write_barrier()` | none — raw NoC |
  | `ttnn/cpp/ttnn/kernel/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp` (shared pool) | 28 | `cb_reserve_back(cb_id_in0, 1)` | none — free-function CB mgmt |
  | `…reader_unary_stick_layout_interleaved_start_id.cpp` | 29 | `get_write_ptr(cb_id_in0)` | none — free-function CB ptr (not sanctioned; no wrapper in scope) |
  | `…reader_unary_stick_layout_interleaved_start_id.cpp` | 31 | `noc_async_read(src_noc_addr, l1_write_addr, stick_size)` | none — raw NoC |
  | `…reader_unary_stick_layout_interleaved_start_id.cpp` | 32 | `noc_async_read_barrier()` | none — raw NoC |
  | `…reader_unary_stick_layout_interleaved_start_id.cpp` | 33 | `cb_push_back(cb_id_in0, 1)` | none — free-function CB mgmt |

  **Scope for the Device 2.0 team:**
  - `reader_unary_stick_start_id.cpp` and `writer_unary_stick_start_id.cpp` are **copy-owned** (`ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/`). Both use `experimental::ShardedAddrGen` (from `ttnn/operations/ccl/...`) + raw `noc_async_read/write` + raw barriers. **Broad Device 1.0** on the DM/addr-gen surface (they do use the `DataflowBuffer` wrapper for CB ops, so it is not a *total* rewrite, but the addr-gen + NoC path is a full migration). Used only on `SameMemoryConfig`'s **row-major sharded** path (`use_sharded_addrgen = sharded && !tilized`, `copy_same_memory_config_program_factory.cpp:65`).
  - `reader_unary_stick_layout_interleaved_start_id.cpp` lives in the **shared kernel pool** `ttnn/cpp/ttnn/kernel/dataflow/` (shared-lib class). It uses free-function CB management (`cb_reserve_back` / `cb_push_back` / `get_write_ptr(cb_id)`) + raw NoC. **Broad Device 1.0.** Used on `SameMemoryConfig`'s **row-major interleaved** path. *A grep of the tree finds no other consumer of this reader today* — despite living in the shared pool, copy is its only referencer, so migrating it is low-blast-radius. (Its companion writer, `writer_unary_stick_layout_interleaved_start_id.cpp`, is **already Device 2.0** — `Noc` + `CircularBuffer` wrapper + `TensorAccessor` — and is broadly shared with `concat` / `embedding`.)

  **Not flagged (Device-2.0 clean, for the record):**
  - `SameMemoryConfig` tilized path: `device/kernels/reader_unary_start_id.cpp`, `device/kernels/writer_unary_start_id.cpp` (both `Noc` + `DataflowBuffer` + `TensorAccessor`; `get_tile_size(cb_id)` is sanctioned) and `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` (compute; `CircularBuffer` wrappers).
  - `DefaultRowMajor`: `redistribute_pages_row_major_reader.cpp`, `redistribute_pages_row_major_writer.cpp` (both `Noc` + `DataflowBuffer` + `TensorAccessor` + `CoreLocalMem`; `tt_memmove(Noc, …)` Device-2.0 form).
  - `DefaultTilized`: `eltwise/unary/.../reader_unary_interleaved_start_id.cpp`, `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` (both `Noc` + `DataflowBuffer` + `TensorAccessor`; `get_local_cb_interface(cb_id)` sanctioned) and `data_movement/sharded/.../compute/eltwise_copy.cpp` (compute; `DataflowBuffer` wrappers).

- **Feature compatibility:** every Appendix A entry scanned against host + kernel + factory code. No entry fires.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `.global_circular_buffer` field, no `remote_cb`/`.remote_index(`, no 4-arg `CreateCircularBuffer`. All CBs are plain `CBDescriptor`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set on any `CBDescriptor`; no `set_address_offset` / 4-arg `UpdateDynamicCircularBufferAddress` / `cb_descriptor_from_sharded_tensor`. |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` / `CreateGlobalSemaphore`; the op uses no semaphores at all. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is fixed (`CopyInputs{input, preallocated_output}` — no `std::vector<Tensor>`). Every kernel reads CTAs at **constexpr** offsets (stick kernels: `get_compile_time_arg_val(2..8)` for `ShardedInfo`; `redistribute` reader: `get_compile_time_arg_val(0..9)` + `TensorAccessorArgs<10>()`). No runtime-varying CTA index. |

- **CB endpoints (GATE-free):** run for the clean subset only (Device 2.0 GREEN there). `SameMemoryConfig` deferred (its RM kernels are Device-1.0 RED; the census would re-key after migration).
  - **`DefaultRowMajor`** (`copy_default_row_major_program_factory.cpp`):
    - `c_0` (input, `dfb_in0`) — **self-loop** (one toucher). Only the reader touches it: it `reserve_back`s (line 42), fills via `get_write_ptr` as a scratch staging buffer, then `push_back`/`wait_front`/`pop_front`s at the end (`redistribute_pages_row_major_reader.cpp:183-185`). The writer never binds `c_0`. Single-config (config-independent).
    - `c_1` (output, `dfb_in1`) — **legal 1:1**. Reader is locked producer (`reserve_back`/`push_back`), writer is locked consumer (`wait_front`/`pop_front`). No flag. Single-config.
  - **`DefaultTilized`** (`copy_default_tilized_program_factory.cpp`): all **legal 1:1** in both configs.
    - `convert_df == false`: `c_0` = reader (producer) + writer (consumer); no compute kernel; output CB index aliases `c_0`.
    - `convert_df == true`: `c_0` = reader (producer) + compute (consumer); `c_16` = compute (producer) + writer (consumer).
  - No dead CBs, no multi-binding, no hidden second writer in the clean subset.

- **Offset base pointers:** **GREEN** — no address RTA folds a host-side offset into its base. All three factories deliver the tensor base via the **`Buffer*` binding form** (`reader_desc.emplace_runtime_args(core, {input.buffer(), …})` — `copy_default_row_major_program_factory.cpp:172`, `copy_default_tilized_program_factory.cpp:144`, `copy_same_memory_config_program_factory.cpp:179/198`), i.e. a bare `Buffer*`, never `->address() + offset`. Kernels consume the raw base directly (`ShardedAddrGen{.bank_base_address = src_addr}` or `TensorAccessor(args, src_addr)`) — a clean base, no interior/offset pointer. Op is **not** in the offset-base-pointer triage tables (`analyses/2026-07-19_offset_base_pointers.md`), consistent with this scan. No Type 1/2/3/4.

- **TensorAccessor 3rd argument:** **GREEN / N/A** — every `TensorAccessor` construction across all eight referenced kernels passes exactly two arguments (`TensorAccessor(args, addr)`); no site passes an explicit page-size third argument. Nothing to classify. Op is not in the 3rd-arg triage table (`analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`), consistent with this scan.

## Port-work summary  *(mirrors the brief — clean subset `{DefaultRowMajor, DefaultTilized}`)*

- **Tensor bindings** (per binding):
  - `DefaultRowMajor` — `input` **Case 1** (fed to `TensorAccessor(src_args, src_addr)` in `redistribute_pages_row_major_reader.cpp:37-38`); `output` **Case 1** (`redistribute_pages_row_major_writer.cpp:30-31`).
  - `DefaultTilized` — `input` **Case 1** (`reader_unary_interleaved_start_id.cpp` — `TensorAccessor(src_args, src_addr)`); `output` **Case 1** (`writer_unary_interleaved_start_id.cpp` — `TensorAccessor(dst_args, dst_addr)`).
  - Delivery today is the `Buffer*` **BufferBinding** form (framework patches it on cache hits — *not* the silent-wrong RTA hazard); the port replaces it with a typed `TensorParameter`/`TensorBinding` and the kernel builds `TensorAccessor(tensor::name)`. Mechanical, low-risk.
- **TensorParameter relaxation:** none (sheet `none`; no custom hash).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** `DefaultRowMajor` → self-loop `c_0` (config-independent) + legal `c_1`; `DefaultTilized` → all legal 1:1.

## Heads-ups  *(mirrors the brief — clean subset)*

- **CB endpoints (multi-binding shapes to watch):** none in the clean subset. (`DefaultRowMajor` `c_0` is a plain one-toucher self-loop; no hidden co-fill, no split-reader, no semaphore-gated raw write anywhere in either factory.)
- **Cross-op / shared kernels:**
  - `DefaultRowMajor` owns both its kernels — no file-path borrow. One **function-call escape**: `redistribute_pages_row_major_reader.cpp:11` `#include`s in-family `data_movement/common/kernels/common.hpp` and calls `tt::data_movement::common::tt_memmove(noc, …)` — the `Noc`-taking (Device 2.0) overload (`common.hpp:143`). Bridges cleanly; **no `_metal2` fork needed** for a function-call escape.
  - `DefaultTilized` **file-path-instantiates three shared kernels it does not own**, all already Device 2.0, **no `_metal2` fork exists for any** (this port would create the first fork of each):
    - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` — cross-family. **Sunset list (not authorization to convert in place):** `untilize`, `untilize_with_unpadding`, `pad`, `reduction/prod`, `experimental/transformer/nlp_create_qkv_heads_falcon7b`, `examples/*` (~10 factories).
    - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — cross-family, **very broadly shared**. **Sunset list:** `bcast`, `concat`, `permute`, `reshape_on_device`, `slice`, `tilize` (×several), `tilize_with_val_padding`, `transpose` (×several), `embedding`, `eltwise/unary_backward/tanh_bw`, `experimental/matmul/attn_matmul`, `experimental/transformer/nlp_concat_heads(_boltz)`, `examples/*` (~20+ factories).
    - `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy.cpp` — in-family (`data_movement/sharded`). **Sunset list:** `interleaved_to_sharded`, `sharded_partial/interleaved_to_sharded_partial`.
- **RTA varargs:** none in the clean subset — every RTA is a fixed, nameable field. `DefaultRowMajor`: `{input.buffer(), start_row_id, num_rows_to_process}` (reader), `{output.buffer(), start_row_id, num_rows_to_process}` (writer). `DefaultTilized`: `{input.buffer(), num_tiles, start_tile_id}` (reader), `{output.buffer(), num_tiles, start_tile_id}` (writer), `{num_tiles}` (compute). All read at constexpr offsets as distinct fields.

## Team-only

- **Out-of-directory coupling & donor shape (clean subset):**
  - Op-level roll-up: **✓ / ⚠** — no sequence-blocking donor shape. Function-call escapes bridge cleanly (`Noc`-taking `tt_memmove`). File-path borrows carry a **cross-op coordination cost** (broadly-shared kernels, no fork yet) but do not gate.

  | Op kernel (factory) | Donor file | Class | Shape / status |
  |---|---|---|---|
  | `redistribute_pages_row_major_reader.cpp` (`DefaultRowMajor`) | `data_movement/common/kernels/common.hpp` → `tt_memmove` | in-family fn-call | `Noc noc` param (`common.hpp:143`) — ✓ Device 2.0 native |
  | *file-path instantiations* (`DefaultTilized`) | `eltwise/unary/.../reader_unary_interleaved_start_id.cpp` | cross-family | Device 2.0; no `_metal2` fork; sunset list above |
  | *file-path instantiations* (`DefaultTilized`) | `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` | cross-family | Device 2.0; no `_metal2` fork; large sunset list above |
  | *file-path instantiations* (`DefaultTilized`) | `data_movement/sharded/.../compute/eltwise_copy.cpp` | in-family | Device 2.0; no `_metal2` fork; sunset list above |

  (The `SameMemoryConfig` factory's own donor/borrow inventory — including its shared-pool interleaved reader/writer and the `ccl` `ShardedAddrGen`/`sharding_addrgen_helper.hpp` coupling — is deferred to re-audit per the Red-outcome scoping rule; the Device 2.0 gate above already names the blocking kernels.)
- **TTNN factory analysis:** all three factory rows on the readiness sheet are `descriptor` / safe / no-custom-hash / no-dynamic-RTA / no-override / no-pybind-descriptor / no-op-owned-tensors / relaxation `none` / `Smuggled pointer == no` / `Pointer patching perf issue == OK`. Target concept `ProgramSpecFactoryConcept` for all three. Nothing gate-relevant beyond the Device 2.0 kernel debt.

## Misc anomalies  *(team-only, non-gating)*

- `device/kernels/writer_unary_start_id.cpp:19` carries an `#ifdef OUT_SHARDED` early-return branch, but the `SameMemoryConfig` factory never defines `OUT_SHARDED` for this (tilized) writer — a dead compile-time branch in that kernel. Harmless; not porter work.
- `copy_default_tilized_program_factory.cpp:126-134` default-constructs `compute_desc` and only fills/pushes it when `use_compute` (`convert_df`). Correct as written (the empty descriptor is never pushed on the no-convert path); noted only because the default-constructed object lingers in scope.

## Recipe notes

- **Config-scoped Device 2.0 RED, sub-factory granularity.** The recipe's config-scoped-GATE machinery and the `METAL2_PREPORT_AUDIT` template phrase Code-path scope at **factory** granularity (`RED at op level; subset <factories> is clear`). Here the Device 2.0 block is *sub-factory*: `SameMemoryConfig` is a single `create_descriptor` that selects Device-2.0-clean kernels for tilized inputs and Device-1.0 kernels for row-major inputs. I reported the clean subset at factory granularity (`{DefaultRowMajor, DefaultTilized}`) and treated `SameMemoryConfig` as wholly blocked (a factory ports as one unit), while surfacing the tilized-path-clean detail to size the Device 2.0 work. A one-line note in the audit doc on how to scope a *within-factory* config split (port-as-unit vs. carve the branch) would remove the judgment call.
- **Shared-pool reader with a single consumer.** `reader_unary_stick_layout_interleaved_start_id.cpp` lives in the shared `ttnn/cpp/ttnn/kernel/` pool yet is referenced only by `copy`. The Device 2.0 gate's "name the owning family so the dependency is schedulable" is slightly awkward for a shared-pool file with a lone consumer — I recorded both (pool location + sole-consumer fact) so the team can decide whether to migrate in place or fork.
