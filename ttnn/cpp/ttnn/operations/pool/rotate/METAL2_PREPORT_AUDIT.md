# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/pool/rotate`

One DeviceOperation, two program factories:

- **`RotateDeviceOperation`** (`device/rotate_device_operation.hpp` / `.cpp`)
  - `NearestProgramFactory` (`device/rotate_nearest_program_factory.cpp`)
  - `BilinearProgramFactory` (`device/rotate_bilinear_program_factory.cpp`)

Kernels in scope (followed by `kernel_source`, not by directory):

| Kernel | Owner | Used by |
|---|---|---|
| `device/kernels/dataflow/reader_rotate_nearest_interleaved.cpp` | rotate | Nearest |
| `device/kernels/dataflow/writer_rotate_nearest_interleaved.cpp` | rotate | Nearest |
| `device/kernels/dataflow/reader_rotate_bilinear_interleaved.cpp` | rotate | Bilinear |
| `pool/generic/device/kernels/compute/compute_pool_2d.cpp` | pool/generic (borrowed) | Bilinear |
| `pool/grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp` | pool/grid_sample (borrowed) | Bilinear, interleaved config only |

No unreferenced kernel files in the op directory — all three rotate-owned kernels are instantiated.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `dc266b472bd 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports`

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/pool/rotate` |
| **Overall** | **GREEN** — all five gates cleared; `METAL2_PORT_BRIEF.md` issued |
| **DOps / Factories** | `RotateDeviceOperation` → `NearestProgramFactory`, `BilinearProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all five kernels and all four donor headers are structurally Device 2.0 (`Noc`, `DataflowBuffer`, `UnicastEndpoint`, `TensorAccessor`). No holdovers. |
| *Prereqs* — Cross-op escapes | **Ok** — all donor function signatures are Device 2.0 native shapes |
| *Feature Support* — overall | **GREEN** — no Appendix A entry fires |
| *Feature Support* — Variadic-CTA | Ok — every `get_compile_time_arg_val` index is a literal constant |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** — both factory rows; cross-check clean |
| *TTNN Readiness* — Concept (current) | `descriptor` (sheet and code agree — both factories return `ProgramDescriptor` from `create_descriptor`) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (blank in the sheet) |
| *TTNN Readiness* — Is safe to port? | **Yes** (both rows); `Smuggled pointer` = `no` |
| *TTNN Readiness* — Custom hash | No — sheet `no`; no `compute_program_hash` anywhere in the op |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No — sheet `no`; hook absent from the device-op |
| *TTNN Readiness* — `override_runtime_arguments` | No — sheet `no`; method absent |
| *TTNN Readiness* — Pybind `create_descriptor` | No — sheet `no`; `rotate_nanobind.cpp` binds only `ttnn::rotate` |
| *TTNN Readiness* — Op-owned tensors | No (blank in the sheet; not expressible on the `descriptor` concept) |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | **none** — no address RTA exists at all; every tensor pointer rides the `Buffer*`-binding form |
| *Port work* — Tensor bindings (per binding) | 4 bindings, all **Case 1** (`TensorAccessor`); plus 2 borrowed-memory CBs (clean) |
| *Port work* — TensorParameter relaxation | **none** (sheet `TensorParameter relaxation` = `none`; consistent with the absent custom hash) |
| *Port work* — TensorAccessor 3rd arg | **none** — every `TensorAccessor` construction is the 2-argument form |
| *Port work* — CB endpoints | self-loop ×3 · legal 1:1 ×5 · dead-CB drop ×1 |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution. Dispositions are recorded per `(CB, config)` below because rotate's census flips between the interleaved and sharded configs.

---

## Result

**GREEN — brief issued** at `METAL2_PORT_BRIEF.md`, alongside this file.

All five gates cleared. This is a clean, straightforward port: Device 2.0 is fully done across every kernel the op touches, no Appendix A feature fires, there are no offset-folded pointers, no 3-argument `TensorAccessor`, no RTA varargs, and all four tensor bindings are the mechanical Case 1 shape. The readiness sheet reads `Is able to port? == yes` on both factory rows, and every cheaply-checkable column matches the code.

**No factory subset scoping is in play** — nothing found is confined to one factory or one branch. The op clears whole, both factories together.

Port work is small and mechanical: four Case-1 tensor bindings, three self-loop CBs, one confirmed dead-CB drop. The two things that will take actual thought are the borrowed shared kernels (no `_metal2` fork exists for either yet, so this port creates the first of each) and the `DUMMY_CB_ID = 32` sentinel the borrowed compute kernel constructs `DataflowBuffer` objects on — both carried into the brief.

---

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The readiness sheet carries **two** rows for `pool/rotate` — `RotateDeviceOperation` × `BilinearProgramFactory` and × `NearestProgramFactory` — and both read `Is able to port? = yes`. Every conjunct of the derivation is satisfied, and every cheaply-checkable column agrees with the code:

  | Column | Sheet | Code evidence | Agree? |
  |---|---|---|---|
  | `Concept` | `descriptor` | `create_descriptor()` returning `ProgramDescriptor` on both factories — [rotate_device_operation.hpp:32-44](device/rotate_device_operation.hpp#L32-L44), [rotate_nearest_program_factory.cpp:35](device/rotate_nearest_program_factory.cpp#L35), [rotate_bilinear_program_factory.cpp:41](device/rotate_bilinear_program_factory.cpp#L41) | ✓ |
  | `Custom hash (compute_program_hash)` | `no` | no `compute_program_hash` in the op directory | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | hook absent from `RotateDeviceOperation` — [rotate_device_operation.hpp:46-52](device/rotate_device_operation.hpp#L46-L52) lists the complete static surface | ✓ |
  | `Override runtime args method? (PD and legacy)` | `no` | no `override_runtime_arguments` in the op directory | ✓ |
  | `Pybind descriptor (nb::class_ of device op)` | `no` | [rotate_nanobind.cpp:63-74](rotate_nanobind.cpp#L63-L74) binds only the user-facing `ttnn::rotate`; no `create_descriptor` binding | ✓ |
  | `Op-owned tensors?` | *(blank)* | N/A — the `descriptor` concept cannot carry them | ✓ |
  | `Secretly SPMD Workload?` | *(blank)* | N/A — not a `WorkloadDescriptor` op | ✓ |
  | `Is safe to port?` | `yes` | *not verified — expert-judgment axis, trusted per the recipe* | — |
  | `Smuggled pointer (raw buffer addr in RTA/CRTA)` | `no` | *feeds `Is safe to port?`; not re-derived* | — |

  **Factory-set match:** exactly two sheet rows, one per factory, and both factories exist in the code — no phantom row, no missing row. **Cross-column invariants** hold: `get_dynamic_runtime_args` is `no` (and the concept is `descriptor`, where it would be permitted anyway), and `Op-owned tensors?` is empty on a `descriptor` row as required.

  One informational column worth carrying forward: **`Op Classification` = `PD (pointer-patching)`** on both rows. That is the sheet's name for the `Buffer*`-binding form this op uses throughout — the framework registers each pushed `Buffer*` as a `BufferBinding` and patches it on cache hits. It corroborates the tensor-binding inventory below (routine port work, not the silent-wrong `->address()` hazard) and is *not* a gate conjunct.

- **Device 2.0 (every kernel used):** **GREEN.** All five in-scope kernels use Device 2.0 objects throughout — `Noc noc;` for every transfer, `DataflowBuffer` for every CB, `UnicastEndpoint` / `experimental::local_addr` for local L1 reads, `TensorAccessor` for tensor addressing. A targeted scan for Device 1.0 idioms (`noc_async_read` / `noc_async_write` / `get_noc_addr` free functions, `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`, `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedAddrGenFast` / `InterleavedPow2AddrGen*`, raw semaphore addresses, `evil_set_*_ptr`) returns **zero hits** across all five kernels and all four donor headers.

  The only CB-index-keyed free functions anywhere in the reachable code are in the shared header `pool/device/kernels/pool_kernels_common.hpp` — `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)`. Both are **sanctioned** by the Green bullet and are not holdovers. Of the functions in that header, rotate calls only `zero_out_page` ([pool_kernels_common.hpp:128-132](../device/kernels/pool_kernels_common.hpp#L128-L132)), whose single free-function use is `get_local_cb_interface(dfb.get_id())` — sanctioned, and already keyed off the `DataflowBuffer` object rather than a bare index.

  No violations table — there are no violations.

- **Feature compatibility:** every Appendix A entry, in order.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `global_circular_buffer` field on any `CBDescriptor`, no `remote_index` / `remote_cb_*` idiom, no `remote_circular_buffer.h`. All five `CBDescriptor` literals ([rotate_nearest_program_factory.cpp:151](device/rotate_nearest_program_factory.cpp#L151), [:164](device/rotate_nearest_program_factory.cpp#L164), [:179](device/rotate_nearest_program_factory.cpp#L179), [rotate_bilinear_program_factory.cpp:156](device/rotate_bilinear_program_factory.cpp#L156), [:170](device/rotate_bilinear_program_factory.cpp#L170), [:185](device/rotate_bilinear_program_factory.cpp#L185), [:204](device/rotate_bilinear_program_factory.cpp#L204)) are plain, with at most a `.buffer` set (the borrowed-memory pattern — a mechanical porting-recipe translation, not this entry). |
  | CBDescriptor `address_offset` (non-zero) | N/A | The field is never named in the op. No `set_address_offset`, no 4-argument `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor` call. Every `CBDescriptor` leaves `address_offset` at its default zero. |
  | GlobalSemaphore | N/A | No `GlobalSemaphore`, no `CreateGlobalSemaphore`, no `global_semaphore.hpp`. The op declares **no semaphores at all** — `desc.semaphores` is never touched in either factory, and no kernel references one. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Op-level cue absent: `tensor_args_t` carries exactly one `const Tensor& input` ([rotate_device_operation.hpp:25-27](device/rotate_device_operation.hpp#L25-L27)) — a fixed-count, single-tensor op, not a variable input list. Kernel-level decider also absent: every `get_compile_time_arg_val` call in all five kernels uses a **literal constant** index (readers 0-10, writer 0-3, compute 0-16 and 38). The `TensorAccessorArgs<N>()` expansions are constexpr-offset, fixed-shape. |

- **CB endpoints (GATE-free):** full per-`(CB, config)`, per-node census below. Device 2.0 is GREEN, so the census runs on intact idioms — no deferral.

  **`NearestProgramFactory`** — kernels: reader + writer, both over `all_cores` (one instance per node).

  | CB | Config | Touchers on a node | Verdict | Disposition |
  |---|---|---|---|---|
  | `fill_cb` (`c_0`) | both | reader only — raw peek `fill_dfb.get_write_ptr()` ([reader:41](device/kernels/dataflow/reader_rotate_nearest_interleaved.cpp#L41)), `zero_out_page(noc, fill_dfb)` ([reader:43](device/kernels/dataflow/reader_rotate_nearest_interleaved.cpp#L43)), and as a local NoC read source ([reader:97-102](device/kernels/dataflow/reader_rotate_nearest_interleaved.cpp#L97-L102)). No FIFO ops. | 1 toucher, role-free | **self-loop** |
  | `input_cb` (`c_1`) | **sharded only** | **none** | **Dead CB** | **drop** — see below |
  | `output_cb` (`c_1` interleaved / `c_2` sharded) | both | reader locked-producer (`reserve_back` [:60](device/kernels/dataflow/reader_rotate_nearest_interleaved.cpp#L60) / `push_back` [:108](device/kernels/dataflow/reader_rotate_nearest_interleaved.cpp#L108)); writer locked-consumer (`wait_front` [writer:30](device/kernels/dataflow/writer_rotate_nearest_interleaved.cpp#L30) / `pop_front` [writer:44](device/kernels/dataflow/writer_rotate_nearest_interleaved.cpp#L44)) | 1P + 1C | **legal 1:1** — no action |

  **`BilinearProgramFactory`** — kernels: reader over `all_cores`; compute over `core_group_1` and (interleaved, when non-empty) `core_group_2`; writer over `all_cores`, **interleaved only**. The two compute instances cover **disjoint** core sets, so every node sees exactly one compute instance — this is *not* a dual-instance work-split.

  | CB | Config | Touchers on a node | Verdict | Disposition |
  |---|---|---|---|---|
  | `fill_cb` (`c_0`) | both | reader only — raw peek [:54](device/kernels/dataflow/reader_rotate_bilinear_interleaved.cpp#L54), `zero_out_page` [:56](device/kernels/dataflow/reader_rotate_bilinear_interleaved.cpp#L56), local NoC read source via `read_four_corner_inputs_with_fill`'s `fill_src` ([grid_sample_reader_common.hpp:221](../grid_sample/device/kernels/grid_sample_reader_common.hpp#L221)). No FIFO ops. | 1 toucher, role-free | **self-loop** |
  | `input_cb` (`c_1`) | both | reader locked-producer (`reserve_back` [:121](device/kernels/dataflow/reader_rotate_bilinear_interleaved.cpp#L121) / `push_back` [:143](device/kernels/dataflow/reader_rotate_bilinear_interleaved.cpp#L143)); compute locked-consumer (`wait_front` [compute_pool_2d.cpp:170](../generic/device/kernels/compute/compute_pool_2d.cpp#L170) / `pop_front` [:179](../generic/device/kernels/compute/compute_pool_2d.cpp#L179)) | 1P + 1C | **legal 1:1** |
  | `scalar_cb` (`c_3`) | both | reader locked-producer ([:137-140](device/kernels/dataflow/reader_rotate_bilinear_interleaved.cpp#L137-L140)); compute locked-consumer ([compute_pool_2d.cpp:143](../generic/device/kernels/compute/compute_pool_2d.cpp#L143), [:263](../generic/device/kernels/compute/compute_pool_2d.cpp#L263)) | 1P + 1C | **legal 1:1** |
  | `output_cb` (`c_5`) | **interleaved** | compute locked-producer (`reserve_back` [compute_pool_2d.cpp:160](../generic/device/kernels/compute/compute_pool_2d.cpp#L160) / `push_back` [:258](../generic/device/kernels/compute/compute_pool_2d.cpp#L258)); writer locked-consumer ([writer_grid_sample_interleaved.cpp:33](../grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp#L33), [:41](../grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp#L41)) | 1P + 1C | **legal 1:1** |
  | `output_cb` (`c_5`) | **sharded** | compute only — no writer kernel is created ([rotate_bilinear_program_factory.cpp:318](device/rotate_bilinear_program_factory.cpp#L318)); the CB is borrowed from `output_tensor.buffer()` ([:214](device/rotate_bilinear_program_factory.cpp#L214)) and nothing drains it | 1 toucher | **self-loop** |

  **Hidden-second-writer hunt (face (a)):** run over every CB above. Negative — the op declares **no semaphores at all**, and no kernel performs a `get_write_ptr()` / `fifo_wr_ptr` write into a CB it is not already the FIFO producer of. The only raw `get_write_ptr()` calls are the reader's own writes into `fill_cb` (a CB with no other toucher) and into `scalar_cb` **between its own** `reserve_back` and `push_back` — a producer peeking at its own buffer, which is one toucher, not two.

  **Multiple-readers hunt (face (b)):** negative — no borrowed-memory CB is base-pointer-read by a second kernel. In both sharded configs the borrowed output CB has exactly one toucher.

  **Dual-instance work-split hunt (face (c)):** negative. The two compute `KernelDescriptor`s in the bilinear factory are the *disjoint-node-set* shape, not the dual-instance shape — `core_group_1` and `core_group_2` never overlap ([rotate_bilinear_program_factory.cpp:306-315](device/rotate_bilinear_program_factory.cpp#L306-L315)), so each node has one compute instance. No CB gains a second toucher from it.

  **Dead CB — `input_cb` (`c_1`), `NearestProgramFactory`, sharded config only.** [rotate_nearest_program_factory.cpp:161-174](device/rotate_nearest_program_factory.cpp#L161-L174) creates a CB borrowed from `input_tensor.buffer()`, but `input_cb_index` is **never handed to a kernel**: the reader's compile-time args ([:194-206](device/rotate_nearest_program_factory.cpp#L194-L206)) carry `output_cb_index` and `fill_cb_index` only, and the writer's ([:212-217](device/rotate_nearest_program_factory.cpp#L212-L217)) carry `output_cb_index` only. Confirmed against the recipe's distrust rule:
  - No indirect path — the index is not computed, offset, or aliased from another value, and it reaches no helper function (it is written once at [:163](device/rotate_nearest_program_factory.cpp#L163) and read once at [:168](device/rotate_nearest_program_factory.cpp#L168), inside its own descriptor).
  - Both nearest kernels reference CBs **only** through their named CTAs; neither hardcodes a CB index anywhere.
  - Checked across all instantiations: the CB does not exist at all in the interleaved config, so there is no config in which it is live.

  The sharded reader gets its input through the `TensorAccessor` instead ([reader:34](device/kernels/dataflow/reader_rotate_nearest_interleaved.cpp#L34), [:90](device/kernels/dataflow/reader_rotate_nearest_interleaved.cpp#L90)), which is why the borrowed CB was never wired up. A dead CB has no behavior, so removing its allocation changes none; a zero-endpoint DFB also cannot be expressed in Metal 2.0 at all, so the drop is required, not optional. **No dead CTA accompanies it** — the index was never threaded to a kernel, so there is nothing else to remove.

- **Offset base pointers:** **GREEN**, and vacuously so — rotate has **no `->address()` call anywhere**. A repo-wide grep for `address()` over the whole op directory returns zero hits. Every tensor pointer reaches a kernel via the `Buffer*`-binding form instead: the factory pushes the `Buffer*` object itself into `emplace_runtime_args` ([rotate_nearest_program_factory.cpp:258](device/rotate_nearest_program_factory.cpp#L258), [:271](device/rotate_nearest_program_factory.cpp#L271), [:286](device/rotate_nearest_program_factory.cpp#L286), [:299](device/rotate_nearest_program_factory.cpp#L299); [rotate_bilinear_program_factory.cpp:367](device/rotate_bilinear_program_factory.cpp#L367), [:381](device/rotate_bilinear_program_factory.cpp#L381)) and the framework registers a `BufferBinding`. With no address expression, there is nothing an offset could be folded into — no Type 1, no Type 2. Type 3 (`address_offset`) is N/A per Appendix A above; Type 4 (`narrow`) does not appear. Rotate is **not** listed in the `2026-07-19_offset_base_pointers.md` triage tables, which agrees with the scan (and the scan, not the doc, is what settles it).

- **TensorAccessor 3rd argument:** **GREEN.** Every `TensorAccessor` construction in the reachable code is the **2-argument** form `TensorAccessor(args, addr)` — [reader_rotate_nearest:34](device/kernels/dataflow/reader_rotate_nearest_interleaved.cpp#L34), [writer_rotate_nearest:22](device/kernels/dataflow/writer_rotate_nearest_interleaved.cpp#L22), [reader_rotate_bilinear:43](device/kernels/dataflow/reader_rotate_bilinear_interleaved.cpp#L43), [writer_grid_sample_interleaved:21](../grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp#L21). No explicit page size is passed at any site, so no classification is needed and none of Classes 1/2/3/4/Special applies. Rotate is absent from `2026-07-06_tensor_accessor_3rd_arg_triage.md`, consistent with the scan.

  (The one 3rd-arg-free accessor in a donor header, [pool_kernels_common.hpp:84](../device/kernels/pool_kernels_common.hpp#L84), sits inside `load_config_tensor_if_in_dram`, which rotate does not call.)

---

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory):

  | Binding | Factory | Delivery today | Kernel use | Case |
  |---|---|---|---|---|
  | `input` | Nearest | `Buffer*` at reader RTA 0 | `TensorAccessor(src_args, input_addr)` [reader:33-34](device/kernels/dataflow/reader_rotate_nearest_interleaved.cpp#L33-L34) | **Case 1** |
  | `output` | Nearest | `Buffer*` at writer RTA 0 | `TensorAccessor(dst_args, output_addr)` [writer:21-22](device/kernels/dataflow/writer_rotate_nearest_interleaved.cpp#L21-L22) | **Case 1** |
  | `input` | Bilinear | `Buffer*` at reader RTA 0 | `TensorAccessor(src_args, input_addr)` [reader:42-43](device/kernels/dataflow/reader_rotate_bilinear_interleaved.cpp#L42-L43) | **Case 1** |
  | `output` | Bilinear, interleaved only | `Buffer*` at writer RTA 0 | `TensorAccessor(dst_args, dst_addr)` [writer_grid_sample_interleaved:19-21](../grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp#L19-L21) | **Case 1** |
  | `output` | Bilinear, sharded | borrowed-memory CB, `.buffer = output_tensor.buffer()` [:214](device/rotate_bilinear_program_factory.cpp#L214) | no accessor — the DFB *is* the access | **clean** (causal-link gate; port via `DataflowBufferSpec::borrowed_from`) |
  | `output` | Nearest, sharded | *both* — borrowed CB [:187](device/rotate_nearest_program_factory.cpp#L187) **and** the writer's Case-1 accessor | see Misc anomaly 1 | **Case 1** + `borrowed_from` |

  Every one is the mechanical shape: no `->address()` RTA exists to remove, so the port swaps a `Buffer*` RTA for a `TensorParameter` / `TensorBinding`, the kernel builds `TensorAccessor(tensor::name)`, and the `TensorAccessorArgs` CTA plumbing disappears. **No Case 2** — no kernel does hand-rolled address arithmetic on a tensor base, so the `get_bank_base_address` bridge is not needed anywhere.

  Note the `Buffer*` form is *correct on cache hits today* (the framework patches `BufferBinding`s), so none of these is the silent-wrong hazard — they are routine port work.

- **TensorParameter relaxation:** **none** — the sheet's `TensorParameter relaxation` column reads `none` on both factory rows, consistent with the absent custom hash (a relaxation co-occurs with one). Nothing for the porter to apply.

- **TensorAccessor 3rd arg:** none — no site passes one.

- **CB endpoints:**
  - self-loop: `fill_cb` (Nearest, both configs) · `fill_cb` (Bilinear, both configs) · `output_cb` (Bilinear, sharded)
  - legal 1:1, no action: `output_cb` (Nearest, both) · `input_cb`, `scalar_cb` (Bilinear, both) · `output_cb` (Bilinear, interleaved)
  - dead-CB drop: `input_cb` `c_1` @ [rotate_nearest_program_factory.cpp:161-174](device/rotate_nearest_program_factory.cpp#L161-L174) (Nearest, sharded config) — no dead CTA accompanies it

---

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. All three hidden-toucher faces were hunted and came back negative; no CB anywhere needs the multi-binding advanced option.

- **`DUMMY_CB_ID = 32` in the borrowed compute kernel — the sharpest thing on this port.** [rotate_bilinear_program_factory.cpp:34](device/rotate_bilinear_program_factory.cpp#L34) defines `DUMMY_CB_ID = 32` and feeds it to **eleven** of `compute_pool_2d.cpp`'s CB-index CTAs ([:256-287](device/rotate_bilinear_program_factory.cpp#L256-L287)) for the pool features rotate does not use. The donor kernel then **unconditionally constructs `DataflowBuffer` objects on that index** — `in_dfb_1`, `in_scalar_dfb_1`, `pre_tilize_dfb`, `fast_tilize_dfb` at [compute_pool_2d.cpp:105-110](../generic/device/kernels/compute/compute_pool_2d.cpp#L105-L110) — even though index 32 is outside the `c_0`…`c_31` CB index space and no such CB is allocated.

  This is harmless on Gen1 today: with `split_reader == 0` ([:39](../generic/device/kernels/compute/compute_pool_2d.cpp#L39)) and `is_output_tiled == 0` ([:53](../generic/device/kernels/compute/compute_pool_2d.cpp#L53)), every *use* of those four objects is compile-time dead, so only the constructors survive and they touch nothing. But the constructions themselves are not guarded, and Metal 2.0 has no `dfb::name` token to bind a nonexistent buffer to. Expect this to be the design question of the port: either guard the constructions behind the same `if constexpr` conditions that already gate their uses, or give the `_metal2` fork a way to express "this operand is unused." Decide it before writing the fork, not after.

- **Cross-op / shared kernels — two borrowed kernel files, no `_metal2` fork exists for either.** A repo-wide search finds **no `_metal2` kernel files at all** outside `experimental/quasar/**` (which do not count), so this port creates the first fork of each, beside the original.

  | Borrowed kernel | Owner | Other instantiating ops (**sunset list**) | `_metal2` fork? |
  |---|---|---|---|
  | `pool/generic/device/kernels/compute/compute_pool_2d.cpp` | pool/generic | `pool/generic` (`pool_multi_core_program_factory.cpp`), `pool/grid_sample` (`grid_sample_bilinear_program_factory.cpp`) | none — this port creates it |
  | `pool/grid_sample/device/kernels/dataflow/writer_grid_sample_interleaved.cpp` | pool/grid_sample | `pool/grid_sample` (`grid_sample_bilinear_program_factory.cpp`) | none — this port creates it |

  These lists are **sunset lists, not authorization to convert either kernel in place**. Both files are live for their other consumers; the legacy copy goes away only when the last consumer migrates.

  **Do not look at `experimental/quasar/` for a precedent here** even if a copy of one of these kernels appears there — those are deliberately hacky pre-port copies, out of bounds as a naming source or as evidence that a construct ports.

- **A transitive `circular_buffer.h` include reaches all three rotate kernels.** `pool/device/kernels/experimental_device_api.hpp` opens with `#include "api/dataflow/circular_buffer.h"` ([experimental_device_api.hpp:11](../device/kernels/experimental_device_api.hpp#L11)) and aliases `using CB = CircularBuffer` ([:24](../device/kernels/experimental_device_api.hpp#L24)). It is pulled in directly by `writer_rotate_nearest_interleaved.cpp` ([:8](device/kernels/dataflow/writer_rotate_nearest_interleaved.cpp#L8)) and transitively by both readers via `pool_kernels_common.hpp` / `grid_sample_reader_common.hpp`. Rotate's own kernels use **none** of it — they are already on `DataflowBuffer` — but the include arrives through shared headers rotate does not own, so the porter cannot simply delete it. Flagged so it is recognized as inherited, not as a rotate-side holdover to clean up.

- **Two compute `KernelDescriptor`s over disjoint core sets — leave them as two `KernelSpec`s.** [rotate_bilinear_program_factory.cpp:306-315](device/rotate_bilinear_program_factory.cpp#L306-L315) instantiates `compute_pool_2d.cpp` twice, differing only in `core_ranges` (`core_group_1` vs `core_group_2`) and the `total_interpolations` CTA. Each node sees exactly one instance, so this is the ordinary per-group split — **not** a dual-instance work-split, and no CB gets a second toucher from it. Do not collapse the pair into one `KernelSpec` by demoting `total_interpolations` to a runtime arg; that is the demoting-per-group-CTA anti-pattern.

- **RTA varargs:** none. Every kernel reads each runtime arg exactly once at a distinct literal index (reader 0-7, writer 0-2), with no counted loop and no data-selected index. The compute kernel's lone `get_arg_val<uint32_t>(0)` ([compute_pool_2d.cpp:129](../generic/device/kernels/compute/compute_pool_2d.cpp#L129)) sits on the dead side of a constant-folded ternary — rotate always supplies a non-zero `max_out_sticks_per_core` CTA and sets **no** runtime args on the compute kernel. Every RTA is nameable; the porter should name them all.

---

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.** Every donor function rotate calls takes Device 2.0 native handles (`Noc`, `DataflowBuffer`, `TensorAccessor<DSpec>`) or plain scalars. No `uint32_t sem_id`, no `uint32_t sem_addr`, no `TensorAccessorArgs<N>` parameter, no CTA-offset NTTP, no old-style addr-gen, no `CircularBuffer&` parameter. Nothing here creates a scheduling blocker, and no per-call detail section is warranted.

**Summary table** — one row per (op kernel, donor file):

| Op kernel | Donor file | Donor class | Status |
|---|---|---|---|
| all three rotate kernels | `tt_metal` `api/dataflow/*` | 1 — LLK/HAL | ✓ no concern |
| `reader_rotate_nearest`, `reader_rotate_bilinear` | `pool/device/kernels/pool_kernels_common.hpp` | 5 — in-family shared | ✓ |
| `reader_rotate_nearest`, `reader_rotate_bilinear` | `pool/device/kernels/fixed_point_arithmetic.hpp` | 5 — in-family shared | ✓ |
| `writer_rotate_nearest` (and transitively the readers) | `pool/device/kernels/experimental_device_api.hpp` | 5 — in-family shared | ✓ |
| `reader_rotate_bilinear` | `pool/grid_sample/device/kernels/grid_sample_reader_common.hpp` | 5 — in-family shared | ✓ |

**Per-call shapes** (recorded rather than omitted, since two of the calls carry a raw L1 address that is worth being explicit about):

| Donor function | Signature shape | Status |
|---|---|---|
| `zero_out_page(Noc, DataflowBuffer)` — [pool_kernels_common.hpp:128](../device/kernels/pool_kernels_common.hpp#L128) | `Noc` + `DataflowBuffer` | ✓ excellent |
| `read_four_corner_inputs_with_fill(Noc, const TensorAccessorT&, …, DataflowBuffer, uint32_t fill_stick_addr)` — [grid_sample_reader_common.hpp:202](../grid_sample/device/kernels/grid_sample_reader_common.hpp#L202) | Shape 1 `TensorAccessor` ref + `DataflowBuffer` + a raw L1 address | ✓ — the `uint32_t` is an L1 page address the **caller** derives from its own bound DFB (`fill_dfb.get_write_ptr()`), not a resource handle needing a token bridge |
| `fill_four_val(uint32_t begin_addr, uint16_t×4)` — [grid_sample_reader_common.hpp:30](../grid_sample/device/kernels/grid_sample_reader_common.hpp#L30) | raw L1 address + scalars | ✓ — same, caller-derived |
| `is_coordinate_valid` — [grid_sample_reader_common.hpp:26](../grid_sample/device/kernels/grid_sample_reader_common.hpp#L26) | pure scalars | ✓ |
| `experimental::local_addr(uint32_t, uint8_t)` — [experimental_device_api.hpp:37](../device/kernels/experimental_device_api.hpp#L37) | scalars | ✓ |
| `fixed_point_arithmetic::*` — [fixed_point_arithmetic.hpp](../device/kernels/fixed_point_arithmetic.hpp) | pure integer arithmetic | ✓ |

**Borrowed kernel files** — see the *Heads-ups* table above (both entries, with sunset lists and fork status).

### Relaxation candidates

None. Rotate has no custom hash, so there is nothing to mine.

### TTNN factory analysis

Sheet and code agree on every checkable column (full table in *Gate detail*). The non-gating facts that feed the port's TTNN ProgramFactory wiring:

- **Op-owned tensors:** none — blank in the sheet. Neither factory returns a `WorkloadDescriptor`, and the `descriptor` concept cannot carry them.
- **MeshWorkload need:** none — single-program, both factories.
- **Pybind `create_descriptor`:** absent (sheet `no`). **Other risky pybind:** none — the nanobind file exposes only the user-facing `ttnn::rotate` free function, and `Is safe to port?` is `yes` with no `warning`.
- **Custom hash / `get_dynamic_runtime_args` / `override_runtime_arguments`:** all `no` in the sheet, all absent in the code.
- **Smuggled pointer:** `no` — the readiness-sheet owner's call, consistent with the fact that no `->address()` expression exists anywhere in the op.
- **`Op Classification`:** `PD (pointer-patching)` — informational; the `Buffer*`-binding form, which the Metal 2.0 typed binding supersedes.
- **Target concept:** `MetalV2FactoryConcept`, no op-owned tensors.

---

## Misc anomalies  *(team-only, non-gating; the port does not act on these)*

1. **The nearest sharded path writes the output tensor onto itself.** [rotate_nearest_program_factory.cpp:187](device/rotate_nearest_program_factory.cpp#L187) borrows the output CB from `output_tensor.buffer()`, so the reader's NoC reads land **directly in the output tensor's L1**. The writer then reads the same CB and NoC-writes each stick to output page `start_stick_id + local_stick_idx` ([writer:35-40](device/kernels/dataflow/writer_rotate_nearest_interleaved.cpp#L35-L40)) — which, for a height-sharded output, resolves to the same core and the same offset the data already occupies. Source and destination coincide, making the entire writer pass a redundant self-copy on the sharded path. It is not wrong, just wasted bandwidth. (Contrast the bilinear factory, which correctly **skips the writer entirely** when the output is sharded, [rotate_bilinear_program_factory.cpp:318](device/rotate_bilinear_program_factory.cpp#L318).) Worth a look from the ops team; the dead `input_cb` in the same block suggests the sharded path was left half-wired.

2. **Four dead compile-time args.** Each is declared `constexpr` and never referenced in the kernel body, yet the host still emits the value and every later CTA index depends on its position:
   - `input_batch` — [reader_rotate_nearest_interleaved.cpp:23](device/kernels/dataflow/reader_rotate_nearest_interleaved.cpp#L23) (CTA 2)
   - `num_cb_pages` — [reader_rotate_nearest_interleaved.cpp:27](device/kernels/dataflow/reader_rotate_nearest_interleaved.cpp#L27) (CTA 6)
   - `num_cb_pages` — [writer_rotate_nearest_interleaved.cpp:19](device/kernels/dataflow/writer_rotate_nearest_interleaved.cpp#L19) (CTA 2)
   - `input_batch` — [reader_rotate_bilinear_interleaved.cpp:29](device/kernels/dataflow/reader_rotate_bilinear_interleaved.cpp#L29) (CTA 3)

3. **Two CB indices are burned with no CB behind them.** The bilinear factory advances its index counter twice without pushing a `CBDescriptor` — bare `cb_idx++` statements at [rotate_bilinear_program_factory.cpp:181](device/rotate_bilinear_program_factory.cpp#L181) and [:196](device/rotate_bilinear_program_factory.cpp#L196) — leaving `c_2` and `c_4` permanently unallocated holes in the index space. No comment explains them; my guess is they mirror slots that exist in the grid_sample factory this one was derived from. Harmless, but they make the index assignment hard to follow.

4. **Both factories carry unreachable sharding branches.** `validate_inputs` admits only `HEIGHT_SHARDED` for a sharded input ([rotate_device_operation.cpp:59-65](device/rotate_device_operation.cpp#L59-L65)), so `is_block_sharded` and `is_width_sharded` are always false by the time either factory runs — yet both keep three-way `start_stick_id` branches on them ([rotate_nearest_program_factory.cpp:246-253](device/rotate_nearest_program_factory.cpp#L246-L253), [rotate_bilinear_program_factory.cpp:348-355](device/rotate_bilinear_program_factory.cpp#L348-L355)). The nearest factory additionally re-asserts the width-sharded case it just branched on, with a `TT_FATAL(!is_width_sharded, …)` at [:100](device/rotate_nearest_program_factory.cpp#L100) that the device-op validation has already made unreachable.

5. **A redundant NoC barrier in the bilinear reader.** [reader_rotate_bilinear_interleaved.cpp:76](device/kernels/dataflow/reader_rotate_bilinear_interleaved.cpp#L76) issues `noc.async_read_barrier()` after the fill-page setup, but neither fill branch issues a NoC *read*: the zero branch calls `zero_out_page`, which already ends in its own `write_zeros_l1_barrier()` ([pool_kernels_common.hpp:131](../device/kernels/pool_kernels_common.hpp#L131)), and the non-zero branch writes L1 directly. The nearest reader, doing the same setup, has no such barrier.

6. **The two factories disagree about which sharding shapes they accept.** The nearest factory handles ND-sharded inputs through a dedicated `nd_shard_spec` path ([rotate_nearest_program_factory.cpp:103-112](device/rotate_nearest_program_factory.cpp#L103-L112)) and hard-fails on width sharding; the bilinear factory has neither — no ND-shard path at all, and no width-shard guard ([rotate_bilinear_program_factory.cpp:98-134](device/rotate_bilinear_program_factory.cpp#L98-L134)). Since `select_program_factory` dispatches purely on `interpolation_mode` ([rotate_device_operation.cpp:14-20](device/rotate_device_operation.cpp#L14-L20)), an ND-sharded input with `interpolation_mode="bilinear"` reaches a factory that does not handle it.

---

## Questions for the user

1. **Nearest sharded path — is the self-copy intentional?** Misc anomaly 1: a dead borrowed input CB plus a writer pass that copies the output tensor onto itself. Not a port blocker either way (the port drops the dead CB and preserves the writer's behavior byte-for-byte), but if the ops team is going to rework that path it is cheaper to do it *before* the port than after.

---

## Recipe notes

1. **The sheet fetch can fail in a way the recipe's troubleshooting doesn't anticipate, and there is no disposition for it.** The first `download_file_content` call this run was **refused by the Claude Code auto-mode permission classifier** — not by Google, with the connector fully authorized and the sheet fully shared. A retry succeeded unchanged, so the refusal appears transient. Two gaps this exposed:
   - `ttnn_op_porting_readiness.md`'s *Troubleshooting* section covers "requires authorization" and "file not found," but not a **local harness refusal**. One line telling the auditor to simply retry would have saved a whole delivery cycle here — I stopped and reported instead, because the standing rule is not to retry a denied tool call verbatim, and a permission refusal is normally a decision to respect rather than probe.
   - The *TTNN factory concept prerequisite* subject of `metal2_audit.md` gives routings for `yes`, `no`, a cross-check conflict, a missing op row, and `MetalV2` — but not for **the fetch itself failing**. Both nearby options are wrong: recording it as *spreadsheet-broken* misroutes a possibly-healthy sheet to its owner, and treating a clean code-side cross-check as a pass would issue a brief asserting a gate nobody checked (silently skipping `Is safe to port?`, the one axis the recipe says cannot be re-derived). I used a third state — PENDING, no brief, gate detail complete — which turned out to be the right shape but is not sanctioned anywhere in the recipe. Worth naming it, since the fetch is a network call in the middle of an otherwise offline audit and will fail again for someone.

2. **The CB endpoint census has no category for a toucher of a CB that does not exist.** `compute_pool_2d.cpp` constructs `DataflowBuffer` objects on `DUMMY_CB_ID = 32` for its unused operands (see the Heads-ups entry). Index 32 is outside the CB index space, so it is neither a real CB with endpoints nor a *dead* CB (a dead CB is allocated and unreferenced; this is referenced and unallocated — the exact mirror image). I recorded it as a porter heads-up rather than forcing it into the census table, but it is a genuinely new shape: a donor kernel with optional operands, hard-wired off by a sentinel index. My guess is this is common in any op that borrows a feature-rich shared compute kernel and uses a subset of it, so the recipe may want a named category for it — and a stance on whether the `_metal2` fork should guard the constructions or express "unused operand" some other way.

3. **The dead-CB distrust rule was well-calibrated here, and the tell was structural.** The recipe's warning that a `(0, 0)` result is "more likely a gap in your own analysis" made me re-check three times before believing it. What finally settled it was not the grep count but the *reason* the CB is dead: the sharded reader gets its input through a `TensorAccessor` instead, so the borrowed CB is a wiring job someone started and did not finish — corroborated by the self-copy in the very same code path (Misc anomaly 1). A line in the recipe suggesting that a believable dead CB usually comes with a visible explanation for *why* nobody wired it up — and that a dead CB with no such story deserves more suspicion — would help the next auditor distinguish the two cases faster.

4. **`Buffer*`-binding-form ops make the offset-base-pointer scan vacuous, which is worth saying out loud.** Rotate has zero `->address()` calls: every pointer rides the `Buffer*` form. The *Offset base pointers* subject of `metal2_audit.md` is written around resolving "each address RTA" to its host computation, and the *TensorParameter analysis* subject is what tells you the `Buffer*` form even exists. Reading them in recipe order, the offset subject appears to have nothing to scan and it is briefly unclear whether that is a clean result or a missed signal. It is genuinely clean — a `Buffer*` cannot carry a fold, since the framework resolves the address itself — but stating that in the offset subject (one clause: "the `Buffer*`-binding form cannot carry a fold; it is GREEN by construction") would save the next auditor the round trip. The readiness sheet independently names this shape in its `Op Classification` column (`PD (pointer-patching)`), which the recipe's column legend does not mention — a cheap prior for exactly this case.
