# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/pool/grid_sample`

One device operation, two program factories:

- **`GridSampleOperation`** (`device/grid_sample_device_operation.hpp`, `device/grid_sample_device_operation.cpp`)
  - `GridSampleBilinearProgramFactory` (`device/grid_sample_bilinear_program_factory.cpp`)
  - `GridSampleNearestProgramFactory` (`device/grid_sample_nearest_program_factory.cpp`)

The factory is chosen by the `mode` attribute (`select_program_factory`, `grid_sample_device_operation.cpp:14-21`): `"bilinear"` → bilinear factory, anything else → nearest factory.

Kernel sources bound by the two factories (all in scope, including the out-of-directory compute kernel):

| Kernel source | Bound by | Owner |
|---|---|---|
| `device/kernels/dataflow/reader_grid_sample_sharded.cpp` | bilinear (sharded grid; 1 or 2 instances) | grid_sample |
| `device/kernels/dataflow/reader_grid_sample_interleaved_start_id.cpp` | bilinear (interleaved grid) | grid_sample |
| `device/kernels/dataflow/writer_grid_sample_interleaved.cpp` | bilinear (interleaved grid) | grid_sample (**also bound by `pool/rotate`**) |
| `device/kernels/dataflow/writer_grid_sample_nearest_sharded.cpp` | nearest (2 instances, always) | grid_sample |
| `ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_pool_2d.cpp` | bilinear | `pool/generic` (**borrowed**) |

Shared headers consumed by those kernels: `device/kernels/grid_sample_reader_common.hpp` (op-owned, **also consumed by `pool/rotate`**), `ttnn/cpp/ttnn/operations/pool/device/kernels/pool_kernels_common.hpp`, `ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp`.

No unreferenced kernel files sit in the op directory — every `.cpp` under `device/kernels/` is bound by a factory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `1b475de4782 2026-07-27 docs(metal_2.0): make the shared-kernel _metal2 fork a reusable checked-in artifact`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/pool/grid_sample` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `GridSampleOperation` → `GridSampleBilinearProgramFactory`, `GridSampleNearestProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all five bound kernels and all three shared headers are structurally Device 2.0; no holdovers |
| *Prereqs* — Cross-op escapes | Ok — all donor signatures are Device 2.0 native (`Noc`, `DataflowBuffer`, raw L1 addresses) |
| *Feature Support* — overall | **GREEN** — no Appendix A entry fires |
| *Feature Support* — Variadic-CTA | Ok — every `get_compile_time_arg_val` call uses a literal index |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (both factory rows) |
| *TTNN Readiness* — Concept (current) | `descriptor` (both factories) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes (both rows); `Smuggled pointer` = `no`, `Op Classification` = `PD (pointer-patching)` |
| *TTNN Readiness* — Custom hash | No — no `compute_program_hash` override anywhere in the op |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No — hook absent from the device operation |
| *TTNN Readiness* — `override_runtime_arguments` | No — method absent from both factories |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `grid_sample_nanobind.cpp` contains no `nb::class_` and no descriptor binding |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | none — the op contains **zero** `->address()` expressions; every pointer reaches a kernel via the `Buffer*`-binding form |
| *Port work* — Tensor bindings (per binding) | `input` Case 1 · `grid` Case 1 (interleaved configs) / clean (sharded configs) · `output` Case 1 (bilinear-interleaved) / clean (all others) |
| *Port work* — TensorParameter relaxation | none (sheet: `none`, both rows; no custom hash to reconcile) |
| *Port work* — TensorAccessor 3rd arg | none — all seven `TensorAccessor(...)` constructions in scope are two-argument |
| *Port work* — CB endpoints | legal 1:1 · self-loop · 1P+1C — no multi-binding flag, no dead CB |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution — a **self-loop** (one toucher), a **1P+1C assignment** (two touchers), the **multi-binding advanced-option flag** (a census that cannot fit 1P+1C), or a **dead-CB drop** (zero endpoints). Recorded per `(CB, config)` below; none of grid_sample's CBs needs the flag or a drop.

## Result

**GREEN → brief issued.** Both prerequisite gates and both conditional gates clear:

- **Device 2.0** — every kernel the op exercises, including the borrowed `compute_pool_2d.cpp`, is on `Noc` / `DataflowBuffer` idioms. The only CB-index free functions present are `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)`, both explicitly sanctioned.
- **Feature compatibility** — no `GlobalCircularBuffer`, no non-zero `address_offset`, no `GlobalSemaphore`, no CTA varargs.
- **TTNN factory concept** — the readiness sheet reads `Is able to port? = yes` for both factory rows, and the cheaply-checkable columns all confirm against the code.
- **Offset base pointers** — the op folds no host-side offset into any device pointer; there is no `->address()` call site at all.
- **TensorAccessor 3rd argument** — the subject does not fire; no accessor passes a page size.

The port work is ordinary: five tensor-binding conversions (all Case 1) plus CB-endpoint role assignments. The one item that needs care is **shared-kernel coordination** — the port touches two kernel sources that other ops bind, in opposite directions (one borrowed, one lent). See *Heads-ups*.

`GREEN at op level; no scoped subset needed.`

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The readiness sheet (*"Operations analysis"*, fetched fresh this run) carries exactly two rows for `pool/grid_sample`, one per factory, both `Is able to port? = yes`. Every conjunct reads `no`/`yes` as required: `Concept = descriptor`, `Custom hash = no`, `Runtime-args update (get_dynamic_runtime_args) = no`, `Override runtime args method? (PD and legacy) = no`, `Pybind descriptor = no`, `Is safe to port? = yes`.

  Cross-check against the code — all clean, no conflicts with the sheet:

  | Column | Sheet | Code evidence |
  |---|---|---|
  | `Concept` | `descriptor` | `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` on both factory structs, `device/grid_sample_device_operation.hpp:24` and `:29`; definitions at `device/grid_sample_bilinear_program_factory.cpp:26` and `device/grid_sample_nearest_program_factory.cpp:22` |
  | `Custom hash` | `no` | No `compute_program_hash` in the op directory; `GridSampleOperation` declares only `select_program_factory`, `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors` (`device/grid_sample_device_operation.hpp:40-43`) |
  | `get_dynamic_runtime_args` | `no` | Hook absent from the device operation (same declaration list) |
  | `override_runtime_arguments` | `no` | Method absent from both factory structs and from the device operation |
  | `Pybind descriptor` | `no` | `grid_sample_nanobind.cpp` has no `nb::class_` of the device op; it binds only the two user-facing functions `grid_sample` and `prepare_grid_sample_grid` |
  | `Op-owned tensors?` | blank | `descriptor` concept; both `create_descriptor` bodies return a plain `ProgramDescriptor` with no buffer vector |
  | Factory-set match | 2 rows | Exactly two factories exist in the code, one-to-one with the two sheet rows — no phantom, no missing row |

  Cross-column invariants hold: `get_dynamic_runtime_args = no` on a `descriptor` concept is consistent, and `Op-owned tensors?` is not `yes` on a `descriptor` row.

  The sheet's `Op Classification = PD (pointer-patching)` and `Smuggled pointer = no` are consistent with what the code does — buffer pointers reach kernels as `Buffer*` objects pushed into `emplace_runtime_args`, which the framework auto-registers as `BufferBinding`s and patches on cache hits, rather than as bare `->address()` values. See *TensorParameter analysis* below.

- **Device 2.0 (every kernel used):** **GREEN.** All five bound kernels plus the three shared headers use the Device 2.0 object API throughout:

  - `Noc noc;` with method-form calls (`noc.async_read`, `noc.async_write`, `noc.async_read_barrier`, `noc.async_write_zeros`, `noc.write_zeros_l1_barrier`).
  - `DataflowBuffer` objects for every CB, with method-form pointer access (`dfb.get_read_ptr()`, `dfb.get_write_ptr()`) and method-form FIFO ops (`reserve_back`, `push_back`, `wait_front`, `pop_front`).
  - `TensorAccessor` / `TensorAccessorArgs` for all tensor memory access; `UnicastEndpoint` + `experimental::local_addr` for local self-reads.

  No legacy idiom appears anywhere in scope: no `noc_async_read` / `noc_async_write` free functions, no `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`, no `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedAddrGenFast` / `InterleavedPow2AddrGen*`, no raw semaphore addresses, no `get_noc_addr_from_bank_id`, no `evil_set_*_ptr`.

  The only CB-index-keyed free functions in scope are the **sanctioned** pair, both in the shared `pool_kernels_common.hpp`:

  | File | Line | Call | Wrapper in scope | Disposition |
  |---|---|---|---|---|
  | `ttnn/cpp/ttnn/operations/pool/device/kernels/pool_kernels_common.hpp` | 46 | `get_tile_size(cb_id)` | yes (`dfb`) | sanctioned — not a holdover |
  | `ttnn/cpp/ttnn/operations/pool/device/kernels/pool_kernels_common.hpp` | 47-48 | `get_local_cb_interface(cb_id)` | yes (`dfb`) | sanctioned — not a holdover |
  | `ttnn/cpp/ttnn/operations/pool/device/kernels/pool_kernels_common.hpp` | 61 | `get_tile_size(clear_value_cb_id)` | yes (`clear_value_dfb`) | sanctioned — not a holdover |
  | `ttnn/cpp/ttnn/operations/pool/device/kernels/pool_kernels_common.hpp` | 75-76 | `get_tile_size(cb_id)`, `get_local_cb_interface(cb_id)` | yes (`dfb`) | sanctioned — not a holdover |
  | `ttnn/cpp/ttnn/operations/pool/device/kernels/pool_kernels_common.hpp` | 129 | `get_local_cb_interface(dfb.get_id())` | yes | already object-derived |

  Of these, only `zero_out_tiles` (lines 74-79) and `zero_out_page` (lines 128-132) are actually called by grid_sample's kernels. Both are on the Device 2.0 sanctioned list, so the gate is GREEN. (A Metal 2.0 *port* will move these lookups onto the `DataflowBuffer` object per the port recipe's kernel-side whitelist rule 7 — but that is port work, not a Device 2.0 gap.)

- **Feature compatibility:** every Appendix A entry, in order. No entry fires.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type reference, no `CreateGlobalCircularBuffer`, no `.global_circular_buffer` field on any `CBDescriptor`, no `remote_index` / `remote_cb_*` idiom, no `experimental::CreateCircularBuffer(..., global_cb)`. **One near-miss to note:** `device/grid_sample_device_operation.hpp:14` carries `#include <tt-metalium/global_circular_buffer.hpp>`. Per the entry's own recognition rule, header presence alone is suggestive but not definitive — and no other signal fires, so this is `N/A`. The include is unused; recorded under *Misc anomalies*. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset` field set on any of the eight `CBDescriptor` literals, no `set_address_offset`, no `UpdateDynamicCircularBufferAddress` in any form, no `cb_descriptor_from_sharded_tensor` call. Every `CBDescriptor` leaves `address_offset` at its default zero. |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` type, no `CreateGlobalSemaphore`, no `global_semaphore.hpp` include. The op declares **no semaphores at all** — `ProgramDescriptor::semaphores` is left empty in both factories. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Op-level cue is absent to begin with: `GridSampleInputs` (`device/grid_sample_device_operation_types.hpp:40-43`) carries exactly two named `Tensor` members, no variable-count container. Kernel-level decider is also clean — every `get_compile_time_arg_val(...)` call in every kernel in scope uses a **literal** index; no loop-indexed or count-derived CTA read exists. |

- **CB endpoints (GATE-free):** every CB is either a legal 1:1 FIFO, a one-toucher self-loop, or a two-toucher 1P+1C assignment. **No CB needs the multi-binding advanced option, and no CB is dead.** Full per-`(CB, config)` census below in *Port-work summary*.

- **Offset base pointers:** **GREEN.** The op contains **no `->address()` expression at all** — a grep for `address()` across the whole op directory returns zero hits. Every buffer reaches a kernel through the `Buffer*`-binding form: the factory pushes the `Buffer*` object itself into `emplace_runtime_args` (`grid_sample_bilinear_program_factory.cpp:409`, `:416`, `:435-436`, `:445`; `grid_sample_nearest_program_factory.cpp:213`, `:220`, `:237-238`, `:248-249`), and the framework resolves it to a base address. There is consequently no site where a host-side offset could be folded into a pointer.

  The two scalar arguments that *look* like offsets are stick indices, not addresses, and are consumed as such:
  - `i * grid_nsticks_per_core` (`grid_sample_bilinear_program_factory.cpp:410`, `grid_sample_nearest_program_factory.cpp:214`) → read as `global_grid_stick_start`, used for batch arithmetic and page-index computation, never as a NoC address.
  - `grid_processed` / `output_processed` (`grid_sample_bilinear_program_factory.cpp:438`, `:447`) → read as `start_page_id` / `start_stick_id`, fed to `{.page_id = ...}` on accessor reads and writes.

  Neither triage table lists `grid_sample` or any `pool/` op — but that silence is not what clears the gate; the scan above is. Type 3 (`address_offset`) is `N/A` per Appendix A; Type 4 (`narrow`) does not appear (`ttnn::narrow` and `MeshBuffer::create` are both absent from the op).

- **TensorAccessor 3rd argument:** **GREEN — the subject does not fire.** All seven `TensorAccessor(...)` constructions across the op and its donors pass exactly two arguments (args pack + base address), so there is no manual page-size override to classify:

  | Site | Construction |
  |---|---|
  | `device/kernels/dataflow/reader_grid_sample_interleaved_start_id.cpp:44` | `TensorAccessor(grid_args, grid_addr)` |
  | `device/kernels/dataflow/reader_grid_sample_interleaved_start_id.cpp:45` | `TensorAccessor(src_args, input_addr)` |
  | `device/kernels/dataflow/reader_grid_sample_sharded.cpp:90` | `TensorAccessor(input_tensor_args, input_addr)` |
  | `device/kernels/dataflow/writer_grid_sample_interleaved.cpp:21` | `TensorAccessor(dst_args, dst_addr)` |
  | `device/kernels/dataflow/writer_grid_sample_nearest_sharded.cpp:178` | `TensorAccessor(input_tensor_args, input_addr)` |
  | `device/kernels/dataflow/writer_grid_sample_nearest_sharded.cpp:181` | `TensorAccessor(grid_tensor_args, grid_addr)` |
  | `pool/device/kernels/pool_kernels_common.hpp:84` | `TensorAccessor(config_tensor_args, config_dram_addr)` — in `load_config_tensor_if_in_dram`, **not called** by any grid_sample kernel |

  `grid_sample` does not appear in the 3rd-arg triage table; consistent with the read above, which is what settles it.

## Port-work summary  *(mirrors the brief)*

### Tensor bindings (per binding)

Three tensors participate: `input_tensor`, `grid`, and the output. Classification varies by factory and by memory-config path, so it is recorded per configuration. The five configurations are:

| Tag | Factory | Grid memory layout | Split reader |
|---|---|---|---|
| **B-INT** | bilinear | interleaved | off (forced — `grid_sample_utils.cpp:19-21`) |
| **B-SH** | bilinear | height-sharded | off (precomputed grid + DRAM input, or Blackhole with C > 224) |
| **B-SH-SR** | bilinear | height-sharded | on |
| **N-SH** | nearest | height-sharded | on (always — `grid_sample_utils.cpp:15-17`) |
| **N-INT** | nearest | interleaved | on (always) |

| Binding | B-INT | B-SH | B-SH-SR | N-SH | N-INT |
|---|---|---|---|---|---|
| `input` | Case 1 | Case 1 | Case 1 | Case 1 | Case 1 |
| `grid` | Case 1 | clean | clean | clean | Case 1 |
| `output` | Case 1 | clean | clean | clean | clean |

- **`input` — Case 1 (via `TensorAccessor`), every configuration.** Delivered as a `Buffer*` in runtime-arg slot 0 of every dataflow kernel; the kernel reads `get_arg_val<uint32_t>(0)` into `input_addr` and immediately builds a `TensorAccessor` from it, doing all memory access through the accessor. Sites: `reader_grid_sample_interleaved_start_id.cpp:19` → `:45`; `reader_grid_sample_sharded.cpp:64` → `:90`; `writer_grid_sample_nearest_sharded.cpp:167`/`:170` → `:178`. **Port work:** express as a `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::input)` and both the `Buffer*` runtime arg and the `TensorAccessorArgs` compile-time plumbing disappear.
- **`grid` — Case 1 in the interleaved configurations.** `Buffer*` in runtime-arg slot 1; kernel reads it into `grid_addr` and builds a `TensorAccessor` (`reader_grid_sample_interleaved_start_id.cpp:20` → `:44`; `writer_grid_sample_nearest_sharded.cpp:171` → `:181`). Same mechanical conversion as `input`.
- **`grid` — clean in the sharded configurations.** The grid CB is a borrowed-memory CB (`.buffer = grid_tensor.buffer()`, `grid_sample_bilinear_program_factory.cpp:110`, `grid_sample_nearest_program_factory.cpp:96`), and the kernel reads the grid data straight out of it by pointer (`grid_dfb.get_read_ptr()` at `reader_grid_sample_sharded.cpp:104`; `grid_dfb.get_write_ptr()` at `writer_grid_sample_nearest_sharded.cpp:192`). The causal-link gate applies: the borrowed-memory DFB *is* the tensor access. **Port work:** `DataflowBufferSpec::borrowed_from` — no Case-1/Case-2 item.
- **`output` — Case 1 in B-INT only.** `Buffer*` in the writer's runtime-arg slot 0 (`grid_sample_bilinear_program_factory.cpp:445`), read into `dst_addr` and fed to `TensorAccessor(dst_args, dst_addr)` (`writer_grid_sample_interleaved.cpp:11` → `:21`), then written through with `noc.async_write(out_dfb, s0, ...)`.
- **`output` — clean in every other configuration.** The output CB is borrowed-memory: `grid_sample_bilinear_program_factory.cpp:215` (sharded bilinear) and `grid_sample_nearest_program_factory.cpp:139` (nearest — **unconditional**, because `compute_output_specs` always produces a sharded output for nearest mode, `grid_sample_device_operation.cpp:199-268`). In bilinear-sharded the compute kernel pushes into it and there is no writer kernel at all (`has_writer = !is_sharded`, `grid_sample_bilinear_program_factory.cpp:380`); in nearest both writer instances raw-write into it at disjoint offsets.

No Case-2 binding exists anywhere in the op — no kernel takes a base address and does hand-rolled NoC arithmetic with it. There is also no CTA-baked-address form: `TensorAccessorArgs(...).append_to(...)` appends only the accessor's descriptor args, never a buffer address.

**Op-level roll-up:** ⚠ port work (five Case-1 binding sites; no Case 2; no correctness hazard, since the `Buffer*` form is patched on cache hits today).

### TensorParameter relaxation

**None.** The readiness sheet lists `TensorParameter relaxation = none` for both factory rows, and the op has no custom hash to reconcile a relaxation against. Nothing for the porter to apply, and no relaxation candidate to mine.

### TensorAccessor 3rd arg

**None** — no site passes a page size. See *Gate detail*.

### CB endpoints

Census per `(CB, config)`, per node. An endpoint is any kernel that touches the CB — FIFO-produces, FIFO-consumes, or accesses it by raw pointer.

**Bilinear factory.**

| CB (index) | Config | Touchers on a node | Verdict | Disposition |
|---|---|---|---|---|
| `grid_cb` (`c_0`) | B-INT | reader0 only — `grid_dfb.get_write_ptr()` (`reader_grid_sample_interleaved_start_id.cpp:79`) + `noc.async_read` into it (`:74`); no FIFO ops | 1 toucher, role-free | **self-loop** |
| `grid_cb` (`c_0`, borrowed) | B-SH | reader0 only — `grid_dfb.get_read_ptr()` (`reader_grid_sample_sharded.cpp:104`) | 1 toucher, role-free | **self-loop** |
| `grid_cb` (`c_0`, borrowed) | B-SH-SR | reader0 **and** reader1 — both instances read the same CB index (reader1's overrides touch only CTAs 0, 2 and 17, `grid_sample_bilinear_program_factory.cpp:276-278`; CTA 1 = `grid_cb_index` is shared) | 2 touchers, both role-free | **1P+1C** — bind reader0 PRODUCER, reader1 CONSUMER (cosmetic on Gen1) |
| `input_cb_0` | all three | reader0 FIFO-produces (`grid_sample_reader_common.hpp:390`, `:408`) + zeroes it (`reader_*:100`/`:64`); compute FIFO-consumes (`compute_pool_2d.cpp:170`, `:179`) | 1 locked producer + 1 locked consumer | **legal 1:1** — no action |
| `input_cb_1` | B-SH-SR only | reader1 FIFO-produces; compute FIFO-consumes | 1 producer + 1 consumer | **legal 1:1** — no action |
| `scalar_cb_0` | all three | reader0 FIFO-produces (`grid_sample_reader_common.hpp:370`, `:373`); compute FIFO-consumes (`compute_pool_2d.cpp:143`, `:263`) | 1 producer + 1 consumer | **legal 1:1** — no action |
| `scalar_cb_1` | B-SH-SR only | reader1 FIFO-produces; compute FIFO-consumes | 1 producer + 1 consumer | **legal 1:1** — no action |
| `output_cb` | B-INT | compute FIFO-produces (`compute_pool_2d.cpp:160`, `:258`); writer FIFO-consumes (`writer_grid_sample_interleaved.cpp:33`, `:41`) | 1 producer + 1 consumer | **legal 1:1** — no action |
| `output_cb` (borrowed) | B-SH, B-SH-SR | compute only — FIFO-produces into the borrowed output shard; **no writer kernel exists** in these configs | 1 toucher (locked producer) | **self-loop** — bind compute PRODUCER *and* CONSUMER |

**Nearest factory.** Split reader is unconditional, so both writer instances always exist.

| CB (index) | Config | Touchers on a node | Verdict | Disposition |
|---|---|---|---|---|
| `grid_cb_0` (`c_0`, borrowed) | N-SH | writer0 **and** writer1 — both bind `grid_cb_index0` (`grid_sample_nearest_program_factory.cpp:190` selects `grid_cb_index0` when sharded) and read via `grid_dfb.get_write_ptr()` (`writer_grid_sample_nearest_sharded.cpp:192`) | 2 touchers, both role-free | **1P+1C** |
| `grid_cb_0` (`c_0`) | N-INT | writer0 only — `get_write_ptr()` + `noc.async_read` into it (`writer_grid_sample_nearest_sharded.cpp:227-228`) | 1 toucher, role-free | **self-loop** |
| `grid_cb_1` | N-INT only | writer1 only (same access shape) | 1 toucher, role-free | **self-loop** |
| `fill_cb` | N-SH, N-INT | writer0 **and** writer1 — both call `fill_dfb.get_write_ptr()` (`:198`) and `zero_out_page(noc, fill_dfb)` (`:199`); no FIFO ops | 2 touchers, both role-free | **1P+1C** |
| `output_cb` (borrowed) | N-SH, N-INT | writer0 **and** writer1 — both raw-write into it via `noc.async_read(..., output_dfb, ..., {.offset_bytes = output_write_offset})` at disjoint offsets (`:97-102`, `:104-111`, `:256-261`); no FIFO ops | 2 touchers, both role-free | **1P+1C** |

**Notes on the census.**

- Every two-toucher CB above is the **dual-instance work-split** shape: the factory pushes the same `kernel_source` into two `KernelDescriptor`s differing only by `DataMovementProcessor` / `NOC` and a `reader_id` compile-time arg, both over one `core_ranges` (`grid_sample_bilinear_program_factory.cpp:280-287`, `grid_sample_nearest_program_factory.cpp:193-201`). The two instances split work by alternating grid points, and their co-touches are sync-free by construction. Per the classification table this resolves to a plain **1P+1C** assignment, **not** the multi-binding flag.
- **No hidden second writer.** Every CB was scanned for a raw `get_write_ptr()` / `fifo_wr_ptr` write by a kernel that is not its FIFO producer, gated by a semaphore pair. None exists — the op declares no semaphores at all, so the semaphore-coordinated co-fill shape cannot occur here.
- **No CB reaches three touchers.** In the widest configuration (B-SH-SR: two readers plus one compute kernel on every node) the two readers use *disjoint* input and scalar CBs, so no CB is touched by all three.
- **No dead CB.** Every allocated `buffer_index` is referenced by at least one bound kernel, in every configuration. Verified by tracing each index through its compile-time arg to its `DataflowBuffer` construction and use.
- **Two compute `KernelDescriptor`s in B-INT are not a two-toucher case.** When `split_work_to_cores` yields a second core group, the bilinear factory emits a second compute kernel (`grid_sample_bilinear_program_factory.cpp:369-376`) — but over a **disjoint** core range. Each node sees exactly one compute instance, so the per-node census is unchanged.

**Op-level roll-up:** ✓ no gate; dispositions are self-loop (4 `(CB, config)` pairs), 1P+1C (5 pairs), and legal 1:1 (the remainder). No multi-binding flag, no dead-CB drop.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. Every multi-toucher CB in this op has exactly two touchers and resolves to a 1P+1C assignment. The porter should still *find* all of them — an unbound toucher is a kernel that cannot legally access the DFB in Metal 2.0 — but no CB needs the advanced option.

- **Cross-op / shared kernels — two sources, in opposite directions, neither with an existing `_metal2` fork.** This is the port's main coordination cost.

  1. **`compute_pool_2d.cpp` — borrowed** (lives in `pool/generic`, grid_sample binds it at `grid_sample_bilinear_program_factory.cpp:354`). Other binders: `pool/generic` (`pool_multi_core_program_factory.cpp:920`) and `pool/rotate` (`rotate_bilinear_program_factory.cpp:293`). No `_metal2` fork exists beside it — the directory holds only `compute_pool_2d.cpp` and `compute_mpwi.cpp`. The copy under `ttnn/cpp/ttnn/operations/experimental/quasar/pool_generic/device/kernels/compute/compute_pool_2d.cpp` is a whole-op pre-port copy and **does not count** as a fork to reuse.

  2. **`writer_grid_sample_interleaved.cpp` — lent** (lives in grid_sample's own `device/kernels/dataflow/`, so nothing about its path warns you, yet `pool/rotate` binds it at `rotate_bilinear_program_factory.cpp:327`). No `_metal2` fork exists. This is precisely the shape where converting in place feels safe and silently breaks the borrower.

  Additionally, `pool/rotate` consumes two functions from grid_sample's own **header** `device/kernels/grid_sample_reader_common.hpp` — `read_four_corner_inputs_with_fill` (`reader_rotate_bilinear_interleaved.cpp:123`) and `fill_four_val` (`:139`) — and its host factory includes grid_sample's `device/grid_sample_utils.hpp` (`rotate_bilinear_program_factory.cpp:8`). Note that `read_four_corner_inputs_with_fill` is defined in grid_sample's header but called *only* by rotate; grid_sample itself never calls it.

  The consumer sets above are a **sunset and coordination list**, not authorization to convert either file in place.

- **RTA varargs:** none. Every kernel reads its runtime args a fixed number of times at distinct literal indices — `reader_grid_sample_sharded.cpp:64-65` (indices 0, 1); `reader_grid_sample_interleaved_start_id.cpp:19-22` (0-3); `writer_grid_sample_interleaved.cpp:11-13` (0-2); `writer_grid_sample_nearest_sharded.cpp:167-172` (0, 1, 3, selected by a `constexpr` branch). No counted loop over runtime args, no data-selected index. All of these become named arguments; the porter needs no vararg mechanism.

- **A shared donor kernel constructs `DataflowBuffer` objects over CB indices grid_sample never allocates.** `compute_pool_2d.cpp:104-110` unconditionally constructs seven `DataflowBuffer`s, including `in_dfb_1(in_cb_id_1)`, `in_scalar_dfb_1(in_scalar_cb_id_1)`, `pre_tilize_dfb(pre_tilize_cb_id)` and `fast_tilize_dfb(fast_tilize_cb_id)`. Grid_sample passes the sentinel index `32` for several of these — `DUMMY_CB_ID` (`device/grid_sample_utils.hpp:19`) for `input_cb_index_1` / `scalar_cb_index_1` when split reader is off (`grid_sample_bilinear_program_factory.cpp:156`, `:184`) and for `fast_tilize_cb_id` (`:350`), and a bare literal `32` for `pre_tilize_cb_id` (`:301-302`). Those code paths are compile-time dead for grid_sample (`is_output_tiled = false`), so no access ever occurs — but in Metal 2.0 the kernel binds `dfb::` tokens, and there is no binding for an unallocated index. This is neither a dead CB (no `CBDescriptor` is allocated) nor a census entry (no toucher); it is a construction-site question the porter must resolve when forking the compute kernel. Flagged here because a census-only reading of the CB subject would not surface it. See *Recipe notes*.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.** Every function-call escape crosses into a donor whose signature is already Device 2.0 native, so the Metal 2.0 named tokens bridge without donor-side work. No ⚠, no ✗, no ⭐. The coupling cost of this op is entirely in *file-path* sharing (below), not in call shapes.

**Summary table** — one row per (op kernel, donor file):

| Op kernel | Donor file | Donor class | Status |
|---|---|---|---|
| `reader_grid_sample_sharded.cpp` | `api/dataflow/dataflow_api.h` | 1 — `tt_metal/*` LLK/HAL | ✓ |
| `reader_grid_sample_sharded.cpp` | `pool/device/kernels/pool_kernels_common.hpp` | 5 — in-family shared | ✓ |
| `reader_grid_sample_interleaved_start_id.cpp` | `api/compile_time_args.h`, `api/dataflow/dataflow_api.h` | 1 | ✓ |
| `reader_grid_sample_interleaved_start_id.cpp` | `pool/device/kernels/pool_kernels_common.hpp` | 5 | ✓ |
| `writer_grid_sample_interleaved.cpp` | `api/dataflow/dataflow_api.h`, `api/dataflow/dataflow_buffer.h` | 1 | ✓ |
| `writer_grid_sample_interleaved.cpp` | `pool/device/kernels/experimental_device_api.hpp` | 5 | ✓ |
| `writer_grid_sample_nearest_sharded.cpp` | `api/dataflow/dataflow_api.h`, `api/debug/dprint.h` | 1 | ✓ |
| `writer_grid_sample_nearest_sharded.cpp` | `pool/device/kernels/pool_kernels_common.hpp` | 5 | ✓ |
| `grid_sample_reader_common.hpp` (op-owned) | `api/dataflow/dataflow_api.h`, `api/dataflow/dataflow_buffer.h`, `api/numeric/bfloat16.h` | 1 | ✓ |
| `grid_sample_reader_common.hpp` (op-owned) | `pool/device/kernels/experimental_device_api.hpp` | 5 | ✓ |
| `compute_pool_2d.cpp` (donor, bound by file path) | `api/compute/*`, `api/dataflow/dataflow_buffer.h` | 1 | ✓ |
| `compute_pool_2d.cpp` (donor, bound by file path) | `pool/device/kernels/experimental_device_api.hpp` | 5 | ✓ |

**Per-call detail** — the functions grid_sample's kernels actually call across a directory boundary, and the shape of their resource handles:

| Donor function | Signature shape | Called from | Status |
|---|---|---|---|
| `zero_out_tiles<cb_id>(Noc, DataflowBuffer)` (`pool_kernels_common.hpp:74`) | `Noc` + `DataflowBuffer` by value; `uint32_t cb_id` as a **template parameter** | `reader_grid_sample_interleaved_start_id.cpp:64`, `reader_grid_sample_sharded.cpp:100` | ✓ — `dfb::name`'s constexpr `uint32_t` cast covers the template-parameter position; the `DataflowBuffer` parameter takes `DataflowBuffer(dfb::name)` directly |
| `zero_out_page(Noc, DataflowBuffer)` (`pool_kernels_common.hpp:128`) | `Noc` + `DataflowBuffer` by value | `reader_grid_sample_sharded.cpp:31`, `writer_grid_sample_nearest_sharded.cpp:199` | ✓ |
| `experimental::local_addr(uint32_t addr, uint8_t noc_id)` (`experimental_device_api.hpp:37`) | plain `uint32_t` L1 address — not a resource handle | `grid_sample_reader_common.hpp:221`, `writer_grid_sample_nearest_sharded.cpp:109`, `:260` | ✓ — no bridge needed; the address comes from `dfb.get_write_ptr()` |

No donor takes a `uint32_t sem_id`, a `uint32_t sem_addr`, a `TensorAccessorArgs<N>`, a tensor CTA offset as an NTTP, an old-style addr-gen object, or a `CircularBuffer&`. The pool family's shared headers were written against the Device 2.0 object API from the start.

**Borrowed kernel files (file-path kernel instantiation).**

| Kernel file | Owning family | Also instantiated by | `_metal2` fork beside it? |
|---|---|---|---|
| `ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_pool_2d.cpp` | `pool/generic` | `pool/generic` (`pool_multi_core_program_factory.cpp:920`), `pool/rotate` (`rotate_bilinear_program_factory.cpp:293`) | **No** — this port creates the first |

The op owns the other four kernel sources it binds. One of them is **lent out**: `writer_grid_sample_interleaved.cpp` is also instantiated by `pool/rotate` (`rotate_bilinear_program_factory.cpp:327`), with no `_metal2` fork beside it either. A repository-wide search for `*_metal2*` outside `experimental/quasar/**` returns nothing, so no reusable fork exists anywhere yet.

Census method: filename grep across `ttnn/cpp/ttnn/operations/`, then each hit checked for whether it is a factory binding the file as a kernel source. Discarded as non-binders: `pool/generic/device/pool_op.cpp:121`, `pool/generic/device/kernels/dataflow/reader_mpwi.cpp:313`, and `pool/generic/device/kernels/compute/compute_mpwi.cpp:55` all merely *mention* `compute_pool_2d.cpp` in comments. The `experimental/quasar/pool_generic/` hits belong to a separate whole-op pre-port copy of the pool op, which binds its own duplicate of the kernel.

### Relaxation candidates

None. The op has no custom hash to mine, and the sheet proposes `none` for both factories.

### TTNN factory analysis

Sheet-derived facts, each confirmed against the code:

- **Op-owned tensors:** none. Both `create_descriptor` bodies build a plain `ProgramDescriptor` (`desc.cbs`, `desc.kernels`) and return it; no buffer vector is populated.
- **MeshWorkload need:** none. Neither factory returns a `WorkloadDescriptor`; the concept is plain `descriptor`.
- **Pybind `create_descriptor`:** absent. `grid_sample_nanobind.cpp` binds only `grid_sample` (via `bind_grid_sample_op`) and `prepare_grid_sample_grid` (`:153`). No `nb::class_` of the device operation, no descriptor internals exposed.
- **Other risky pybind:** none. `Is safe to port? = yes` with no `warning`.
- **Custom hash:** absent — the device operation relies on the framework default hash over `GridSampleParams::attribute_values()` (`device/grid_sample_device_operation_types.hpp:28-37`).
- **`get_dynamic_runtime_args`:** absent.
- **`override_runtime_arguments`:** absent.
- **Target concept:** `MetalV2FactoryConcept` for both factories (derived from `Concept = descriptor` with no op-owned tensors).

## Misc anomalies  *(team-only, non-gating)*

Latent issues noticed while auditing. These route to the ops team; the port does not act on them.

1. **Unused `global_circular_buffer.hpp` include** — `device/grid_sample_device_operation.hpp:14` includes `<tt-metalium/global_circular_buffer.hpp>`, but the op references no `GlobalCircularBuffer` anywhere. Worth removing: it is a recognition signal for an Appendix A gate feature, so its presence costs a future auditor a false-alarm investigation.

2. **Duplicated `TensorAccessorArgs` append in the nearest factory's sharded path** — `device/grid_sample_nearest_program_factory.cpp:169` appends the *input* tensor's accessor args, and then `:173` appends the **input tensor's accessor args again** in the `is_sharded` branch (where the interleaved branch at `:171` would append the grid tensor's). The kernel reads that second block as `grid_tensor_args` and builds `grid_tensor_accessor` over `grid_addr = 0` (`writer_grid_sample_nearest_sharded.cpp:180-181`); the accessor is never used, because its only consumer is guarded by `if constexpr (!is_sharded)` (`:226-230`). Benign today — it exists to keep the compile-time-arg layout parseable across both branches — but it is a duplicated block whose intent is not stated anywhere, and a reader could easily take it for a copy-paste bug.

3. **Dead runtime arg in the nearest factory's interleaved path** — `device/grid_sample_nearest_program_factory.cpp:239` and `:250` pass `grid_sticks` at runtime-arg slot 2, but `writer_grid_sample_nearest_sharded.cpp:170-173` reads slots 0, 1 and 3, skipping 2. The kernel bounds its loop with the compile-time `grid_nsticks_per_core` instead.

4. **Dead compile-time arg in the bilinear interleaved reader** — the bilinear factory emits `input_batch` at compile-time-arg slot 5 (`grid_sample_bilinear_program_factory.cpp:230`), but `reader_grid_sample_interleaved_start_id.cpp` reads slots 4 and then 6, never 5. The sharded reader does read it (`reader_grid_sample_sharded.cpp:73`), so the slot exists to keep one shared arg layout across the two readers — but in the interleaved kernel it is dead.

5. **Dead template parameters on `process_grid_point`** — `device/kernels/grid_sample_reader_common.hpp:293-294` declares `uint32_t input_cb_index` and `uint32_t scalar_cb_index` as template parameters, but neither is referenced in the function body (the body works entirely through the `input_dfb` / `scalar_dfb` object parameters). Both call sites pass them (`reader_grid_sample_sharded.cpp:156-157`, `reader_grid_sample_interleaved_start_id.cpp:93-94`).

6. **Hardcoded `32` where `DUMMY_CB_ID` is meant** — `device/grid_sample_bilinear_program_factory.cpp:301-302` writes `const uint32_t pre_tilize_cb_id = 32;` with a comment explaining it is an unused CB, while every other sentinel in the same argument list uses the named constant `DUMMY_CB_ID` (`device/grid_sample_utils.hpp:19`, also `32`). The two are the same value; the inconsistency is a readability trap.

7. **A function defined in grid_sample's header is used only by `pool/rotate`** — `read_four_corner_inputs_with_fill` (`device/kernels/grid_sample_reader_common.hpp:201-274`) has no caller inside grid_sample; its sole consumer is `pool/rotate` (`reader_rotate_bilinear_interleaved.cpp:123`). Not dead code, but the ownership is inverted relative to where it lives, which makes the coupling easy to miss.

8. **Both nearest writer instances zero the same fill page** — `writer_grid_sample_nearest_sharded.cpp:198-199` has each of the two instances call `zero_out_page(noc, fill_dfb)` on the shared `fill_cb`, unsynchronized. Harmless in practice (both write the same zeros to the same region, and the values are only read after), but it is a genuine unsynchronized concurrent write that a future reader may flag.

## Per-DeviceOperation attribution

The directory holds a single `DeviceOperation`, so no per-DeviceOperation split is needed. Findings that differ **between the two factories** are attributed inline throughout (the tensor-binding table, the CB-endpoint census, and the shared-kernel list are all factory-scoped).

## Recipe notes

1. **The CB-endpoint census has no row for the inverse of a dead CB — a kernel that references a CB index the factory never allocates.** The *CB endpoints* subject of `metal2_audit.md` classifies by *touchers per allocated CB*, and its *Dead CB (0, 0)* sub-section covers an allocation with no toucher. Grid_sample hits the mirror case: the shared `compute_pool_2d.cpp` constructs `DataflowBuffer` objects over sentinel index `32` for four CBs the grid_sample factory does not allocate (`compute_pool_2d.cpp:104-110`, fed by `grid_sample_bilinear_program_factory.cpp:156`, `:184`, `:301`, `:350`). It produces zero touchers on zero CBs, so the census is silent on it — yet it is a real Metal 2.0 question, because a `dfb::` token needs a binding and there is none. I recorded it under *Heads-ups* rather than in the census. A short guard bullet in the CB subject — "also note any kernel that constructs a DFB over an index the factory does not allocate (a disabled-path sentinel); it is not a census entry but it is a binding the porter must resolve" — would give this a home. The shape is likely common wherever a shared kernel is driven by a superset of compile-time args, as the whole pool family is.

2. **The donor-shape table has no row for `DataflowBuffer`.** The per-call shape table in the *Out-of-directory coupling* subject of `metal2_audit.md` lists `CircularBuffer` / `CircularBuffer&` as ⭐ flag and `uint32_t cb_id` as ✓ OK, but every donor grid_sample calls takes a **`DataflowBuffer` by value** (`zero_out_tiles`, `zero_out_page`) — which is the Metal 2.0 kernel-side buffer object itself (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:61`), constructed from a `dfb::name` accessor. It is trivially ✓, but the table's silence made me pause over whether `DataflowBuffer` was meant to be read as the modern spelling of the ⭐-flagged `CircularBuffer` row. It is not — `CircularBuffer` is a distinct type in `api/dataflow/circular_buffer.h`. An explicit `DataflowBuffer` / `DataflowBuffer&` ✓ row would remove the ambiguity.

3. **Minor, and I resolved it without difficulty:** the recognition model in the *Offset base pointers* subject of `metal2_audit.md` is written entirely in terms of `->address()` expressions and their host arithmetic, but an op that delivers pointers exclusively via the `Buffer*`-binding form (as this one does — zero `->address()` call sites op-wide) has no such expression to inspect. The gate is trivially clear in that case, since the framework resolves the `Buffer*` to a base and there is no seam for a host fold. Stating that explicitly — "a `Buffer*`-binding-form op cannot carry a folded offset; the gate clears without further tracing" — would let a future auditor close this subject in one line instead of reasoning it out.
