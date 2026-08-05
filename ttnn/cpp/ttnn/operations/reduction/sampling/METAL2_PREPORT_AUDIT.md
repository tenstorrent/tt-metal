# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/reduction/sampling`

One device operation, one program factory:

- **`SamplingDeviceOperation`** (`device/sampling_device_operation.hpp`, `device/sampling_device_operation.cpp`)
  - `SamplingProgramFactory` (`device/sampling_program_factory.cpp`), the only factory. `program_factory_t = std::variant<SamplingProgramFactory>` at `device/sampling_device_operation.hpp:24`.

Kernels referenced by `KernelDescriptor::kernel_source` (all three owned by this op, none borrowed by file path):

- `device/kernels/dataflow/reader_values_indices_tensor.cpp` (reader, created once over the whole core grid)
- `device/kernels/dataflow/writer_interleaved.cpp` (writer, one instance per core)
- `device/kernels/compute/sampling.cpp` (compute, one instance per core)

No unreferenced kernel files in the directory. No semaphores are declared: `desc.semaphores` is never populated.

**Scope:** TTNN op, Gen1 (WH/BH) target, within scope of `audit/metal2_audit.md`.

**Recipe docs:** `ccf3df7c4ab 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/reduction/sampling` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `SamplingDeviceOperation` → `SamplingProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes.** The op's three kernels and every donor function they reach are on Device 2.0 idioms (`Noc`, `DataflowBuffer` / `CircularBuffer` wrappers, `CoreLocalMem`, `UnicastEndpoint`). No holdovers found. |
| *Prereqs* — Cross-op escapes | **issue (non-gating, ⭐ flag).** One donor function takes a `CircularBuffer` by value. See *Out-of-directory coupling*. |
| *Feature Support* — overall | **GREEN.** Every Appendix A entry is `N/A`. |
| *Feature Support* — Variadic-CTA | Ok. All compile-time args are read at constexpr indices. |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | 6 bindings, all **Case 1** (`TensorAccessor`) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none. Every accessor is constructed with the 2-argument form. |
| *Port work* — CB endpoints | 10 legal 1:1 · 8 self-loop · 0 multi-binding · 0 dead |

**CB endpoints** are dispositions, not gates. Every out-of-window CB here resolves with a **self-loop** (a single toucher). No CB needs the multi-binding advanced option, and none is dead. There is only one config path in this op (see *CB endpoints* below), so each CB carries one disposition rather than a per-config set.

## Result

**GREEN, brief issued.** All five gate-bearing subjects cleared: Device 2.0, Feature compatibility, TTNN factory concept, Offset base pointers, TensorAccessor 3rd argument. The porter brief is at `METAL2_PORT_BRIEF.md` in this directory.

The port work is modest: six `TensorParameter` bindings (all mechanical Case 1), eight self-loop CB bindings, and the ordinary CB-to-DFB plus named-argument rewrite. Two things the porter should read before starting. First, the writer kernel is the only one still written against `CircularBuffer` rather than `DataflowBuffer`. Second, one shared donor function takes a `CircularBuffer` by value, a flagged shape needing cross-team discussion rather than a blocker.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The readiness sheet's single row for this op (`reduction/sampling`, `SamplingDeviceOperation`, `SamplingProgramFactory`) reads `Is able to port? = yes`, with every conjunct clear: `Concept = descriptor`, `Custom hash = no`, `Runtime-args update (get_dynamic_runtime_args) = no`, `Override runtime args method? = no`, `Pybind descriptor = no`, `Is safe to port? = yes`. `TensorParameter relaxation = none`. `Op-owned tensors?` and `Secretly SPMD Workload?` are blank, both N/A on a `descriptor` concept. The informational columns read `Op Classification = PD Op (pointer-patching)`, `Execution Model = SPMD`, `Porting Target = ProgramSpecFactoryConcept`, `Backdoor custom hash = no`, `Smuggled pointer = no`, `Pointer patching perf issue? = OK`, `Formerly custom hashed? = no`.

  Cross-check against the code. **All six checks agree with the sheet:**

  | Column | Sheet | Code evidence |
  |---|---|---|
  | `Concept` | `descriptor` | `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` @ `device/sampling_program_factory.hpp:14-15` |
  | `Custom hash` | `no` | no `compute_program_hash` anywhere in the op directory |
  | `Backdoor custom hash` | `no` | no `attribute_values` / `to_hash` in the op directory |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | no such hook on `SamplingDeviceOperation`. `device/sampling_device_operation.hpp:19-28` declares only `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`. |
  | `Override runtime args method?` | `no` | no `override_runtime_arguments` in the op directory |
  | `Pybind descriptor` | `no` | `sampling_nanobind.cpp` contains no `create_descriptor` binding and no `nb::class_` of the device op |
  | `Smuggled pointer` | `no` | confirmed. See *Port-work summary*: the factory hands whole `MeshTensor`s to `emplace_runtime_args`, never a `buffer()->address()`. |

  **Factory-set match:** the sheet has exactly one factory row and the code has exactly one factory, so one-to-one, with no phantom or missing rows. **Cross-column invariants:** `get_dynamic_runtime_args = no` (no invariant to violate), and `Op-owned tensors?` is blank on a `descriptor` concept, which is the only consistent value. The sheet is not stale for this op.

- **Device 2.0 (every kernel used):** **GREEN.** No violations to table. Evidence, per kernel:

  - `device/kernels/dataflow/reader_values_indices_tensor.cpp`: `Noc noc` plus `noc.async_read(...)` (`:89-90`, `:102`), `DataflowBuffer` objects for every CB (`:23`, `:79-80`), `CoreLocalMem<volatile uint32_t>` for local writes (`:25`), `dfb.get_write_ptr()` and `dfb.get_entry_size()` as wrapper methods (`:25`, `:81`).
  - `device/kernels/dataflow/writer_interleaved.cpp`: `Noc noc` (`:83`), `CircularBuffer` objects for all eight CBs it touches (`:84-91`), `CoreLocalMem<...>` for every local access (`:100`, `:111`, `:123`, `:133`, `:140`, `:143`, `:145`, `:148`), `noc.async_read` and `noc.async_write` through the wrappers (`:97`, `:241-246`).
  - `device/kernels/compute/sampling.cpp`: `DataflowBuffer` objects throughout. Compute-side only, so no NoC or semaphore use.
  - Donor functions the kernels actually reach are equally compliant. `generate_mask` uses `CircularBuffer cb_mask(cb_mask_in)` plus `noc.async_write_zeros` (`transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp:262-304`). `fill_tile` and `fill_tile_partial` construct `CircularBuffer cb(cb_id)` and go through `noc.async_write_zeros` (same file, `:27-101`). `copy_tile` uses `Noc`, `UnicastEndpoint` and `CoreLocalMem` (`transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp:317-328`). `calculate_and_prepare_reduce_scaler` uses `DataflowBuffer dfb(dfb_id)` plus `noc.async_write_zeros` (`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl:161-203`). `generate_bcast_unary_scalar` uses `cb.reserve_back` / `cb.get_write_ptr` / `cb.push_back` (`ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp:44-50`). `compute_kernel_lib::reduce` is `DataflowBuffer`-based throughout (`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl`).

  Three things deliberately **not** flagged as violations, so the reasoning is on record:

  - `get_tile_size(dfb_id)` appears in the reached donor code (`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl:167`, `:199`; `transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp:263`). This is a **sanctioned** CB-index free function per the Device 2.0 Green bullet, since Device 2.0's own migrated examples use it, so it is not a holdover.
  - `fill_tile(uint32_t cb_id, ...)` and its siblings take a `uint32_t` CB index. These are op-library helpers, not framework free functions with a wrapper-method replacement, so they are not the CB-index-keyed holdover the gate describes. The donor-shape table also classifies a `uint32_t cb_id` donor parameter as ✓ OK.
  - Some reached donor code still reaches SRAM through a raw `reinterpret_cast<volatile tt_l1_ptr ...>` on a wrapper-supplied pointer rather than through `CoreLocalMem<T>` (`transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp:37-42`, `:59-62`; `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp:16`, `:31`, `:47`). The Device 2.0 guide does list `CoreLocalMem<T>` as the replacement for that idiom, but the CB access itself already goes through the wrapper, so the Metal 2.0 binding token has a Device 2.0 object to attach to, and none of the gate's enumerated violation classes applies (CB-index-keyed holdover, raw NoC, manual CB index management, legacy addr-gen, raw semaphore address). **Not gated.** Recorded here so a reader who greps the donors independently knows it was seen and judged. See *Recipe notes* item 3.

- **Feature compatibility:** every Appendix A entry is `N/A`. None of their recognition signals fire.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `global_circular_buffer.hpp` include, no `remote_index` or `remote_cb_*` idiom. All 18 `CBDescriptor` literals in `device/sampling_program_factory.cpp` set only `total_size`, `core_ranges` and `format_descriptors`, so `global_circular_buffer` stays at its `nullptr` default (`tt_metal/api/tt-metalium/program_descriptors.hpp:82`). |
  | CBDescriptor `address_offset` (non-zero) | N/A | Same evidence: no `CBDescriptor` literal sets `address_offset`, so every one keeps the `0` default (same file, `:81`). No `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. |
  | GlobalSemaphore | N/A | The op declares no semaphores at all, since `desc.semaphores` is never touched, and there is no `GlobalSemaphore` reference or `global_semaphore.hpp` include. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | **Op-level cue absent:** `SamplingInputs` (`device/sampling_device_operation_types.hpp:19-26`) is a fixed set of five named tensors plus one `std::optional<Tensor>`, with no variable-count container. **Kernel-level decider absent:** every `get_compile_time_arg_val` call in all three kernels uses a constant or constexpr-computed index (`reader_values_indices_tensor.cpp:55-66`; `writer_interleaved.cpp:51-72` via the constexpr `args_base`; `compute/sampling.cpp:420-444`), and the `TensorAccessorArgs<N>` offsets chain through `constexpr next_compile_time_args_offset()`. There is no runtime-varying CTA index. |

- **CB endpoints (GATE-free):** classified below in *Port-work summary*. 10 CBs are already legal 1:1, 8 are single-toucher and take a self-loop, none is dead, and none needs the multi-binding advanced option. Nothing here blocks the port.

- **Offset base pointers:** **GREEN, no fold exists to split.** The factory passes whole tensors, not addresses: `reader_desc.emplace_runtime_args(core, {input_values_tensor, input_indices_tensor})` (`device/sampling_program_factory.cpp:374-379`) and `writer_desc.emplace_runtime_args(core, {output_mesh, temp, k, p})` (same file, `:422`). There is **no `buffer()->address()` expression anywhere in the op**, hence no host arithmetic that could fold an offset into a base. Type 3 (`address_offset`) is `0` on every CB, per the Appendix A table above. Type 4 (`ttnn::narrow`) does not appear.

  Reconciled against the offset-base-pointer triage analysis (a dated prior): the analysis does not list this C++ op. Its two `sampling`-matching lines, `fused_ops/lm_head_sampling/op.py` and `micro_ops/sampling/op.py`, are DeepSeek Python ops outside this op's scope. So this is the *"no fold, op not in the tables"* outcome: clean, handed to *TensorParameter analysis* as six clean bases.

  One note on the kernel side, so it is not mistaken for a fold. The writer does add offsets to pointers: `cb_final_indices.get_read_ptr() + core_id * final_indices_stick_size` (`writer_interleaved.cpp:145`) and the `{.offset_bytes = core_id * 4}` on its output write (same file, `:241-246`). Both are offsets applied on the device side, to a CB pointer or to accessor offset fields, rather than a host-folded device pointer, so neither is a Type-1 or Type-2 site.

- **TensorAccessor 3rd argument:** **GREEN, the subject does not fire.** All six `TensorAccessor` constructions use the 2-argument `(args, addr)` form and pass no explicit page size: `reader_values_indices_tensor.cpp:74`, `:76`; `writer_interleaved.cpp:93`, `:104`, `:115`, `:239`. Consistent with the 3rd-arg triage analysis (a dated prior), which does not list this op. Since there is no 3rd-arg site to classify, the doc's silence and the code agree.

## Port-work summary  *(mirrors the brief)*

### Tensor bindings: six bindings, all Case 1

The delivery mechanism today is already the typed form. The factory pushes `MeshTensor` references into `emplace_runtime_args`, which the framework auto-registers as buffer bindings and patches on cache hits (`tt_metal/api/tt-metalium/program_descriptors.hpp:110-118`, `:164-203`). So none of these is the silent-wrong RTA-smuggled-address hazard: they are routine port work, and they agree with the sheet's `Smuggled pointer = no`. Each is classified by what the kernel does with the base it receives, which in every case is to feed it to a `TensorAccessor`.

| Binding | Delivered at | Kernel arg | Consumed by | Case |
|---|---|---|---|---|
| `input_values` | `device/sampling_program_factory.cpp:377` | reader RTA 0 → `values_addr` (`reader_values_indices_tensor.cpp:52`) | `TensorAccessor(s0_args, values_addr)` (`:74`), then `noc.async_read(s0, ...)` (`:89-90`) | **Case 1** |
| `input_indices` | same file, `:378` | reader RTA 1 → `indices_addr` (`:53`) | `TensorAccessor(s1_args, indices_addr)` (`:76`), then `noc.async_read(s1, ...)` (`:102`) | **Case 1** |
| `output` | same file, `:422` | writer RTA 0 → `dst_addr` (`writer_interleaved.cpp:39`) | `TensorAccessor(dst_args, dst_addr)` (`:239`), then `noc.async_write(..., s_out, ...)` (`:241-246`) | **Case 1** |
| `temp` | same file, `:422` | writer RTA 1 → `temp_addr` (`:40`) | `TensorAccessor(temp_args, temp_addr)` (`:115`), then `noc.async_read(addrg_temp, ...)` (`:119`) | **Case 1** |
| `k` | same file, `:422` | writer RTA 2 → `k_addr` (`:41`) | `TensorAccessor(k_args, k_addr)` (`:93`), then `noc.async_read(addrg_k, ...)` (`:97`) | **Case 1** |
| `p` | same file, `:422` | writer RTA 3 → `p_addr` (`:42`) | `TensorAccessor(p_args, p_addr)` (`:104`), then `noc.async_read(addrg_p, ...)` (`:108`) | **Case 1** |

No Case 2: no kernel does hand-rolled NoC arithmetic on a tensor base. No borrowed-memory DFB reads either, since every `CBDescriptor` leaves `buffer` and `tensor` at `nullptr`, so no CB is backed by a device buffer and the causal-link gate never applies.

The matching host-side `TensorAccessorArgs` plumbing that disappears with the port sits at `device/sampling_program_factory.cpp:362-363` (reader) and `:388-391` (writer).

**Op-level roll-up:** ⚠ port work. Six Case-1 bindings, all mechanical.

### TensorParameter relaxation

**none.** The sheet's `TensorParameter relaxation` column reads `none`, consistent with `Custom hash = no`: there is no hash to reconcile a relaxation against.

### TensorAccessor 3rd arg

**none.** No site to drop.

### CB endpoints

The op allocates 18 CBs, all over the same `core_grid`. Per node the census is the same everywhere: exactly one reader instance, one writer instance and one compute instance are co-resident. The reader is created once over `core_grid` at `device/sampling_program_factory.cpp:365-381`, and the writer and compute are created per single core at `:415-457`, so each node sees one of each.

There is a single config path. The factory has no sharding or layout branches, and `sub_core_grids`, `num_users` and `Wt` change which and how many cores run, never which kernel touches which CB. So each CB carries one disposition rather than a per-config set.

**Legal 1:1, 10 CBs, no action:**

| CB | Name | Producer | Consumer |
|---|---|---|---|
| `c_0` | `input_values` | reader `input_values_dfb.reserve_back` / `push_back` (`reader_values_indices_tensor.cpp:88`, `:94`) | compute `input_dfb.wait_front` / `pop_front` (`compute/sampling.cpp:244`, `:265`) |
| `c_1` | `cb_local_vals` | compute `out_dfb_obj.reserve_back` / `push_back` in `mul_block_bcast_cols` (`compute/sampling.cpp:141`, `:147`) | writer `cb_local_values.wait_front` / `get_read_ptr` / `pop_front` (`writer_interleaved.cpp:137`, `:140`, `:236`) |
| `c_2` | `index` | reader `generate_index_tile` `dfb.reserve_back` / `push_back` (`reader_values_indices_tensor.cpp:24`, `:48`) | compute `index_dfb.wait_front` / `pop_front` (`compute/sampling.cpp:245`, `:266`) |
| `c_3` | `scaler_max` | writer `calculate_and_prepare_reduce_scaler<scaler_max_cb_id, MAX, REDUCE_ROW>` (`writer_interleaved.cpp:77-78`, resolving to `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl:163`, `:203`) | compute `scaler_dfb.wait_front(1)` inside `reduce` (`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl:364`), via `reduce_c` (`compute/sampling.cpp:477`) |
| `c_17` | `scaler_sum` | writer, same helper with `SUM` (`writer_interleaved.cpp:79-80`) | compute, same path via `reduce_c` (`compute/sampling.cpp:480`) |
| `c_4` | `topk_mask` | writer `generate_mask<cb_id_mask, one>`, whose body does `cb_mask.reserve_back` / `push_back` (`writer_interleaved.cpp:130`, resolving to `transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp:262`, `:304`) | compute `topk_mask_dfb.wait_front(Kt)` (`compute/sampling.cpp:473`) and `add_block_inplace` `in1_dfb_obj.wait_front` (`:102`) |
| `c_8` | `output_ind` | compute `output_ind_dfb.reserve_back` / `push_back` (`compute/sampling.cpp:368`, `:374`) | writer `cb_local_indices.wait_front` / `get_read_ptr` / `pop_front` (`writer_interleaved.cpp:138`, `:143`, `:236`) |
| `c_11` | `rand_tile` | compute `generate_rand_tile` `dfb_obj.reserve_back` / `push_back` (`compute/sampling.cpp:40`, `:49`) | writer `cb_rand.wait_front` / `get_read_ptr` / `pop_front` (`writer_interleaved.cpp:132-133`, `:234`) |
| `c_12` | `final_indices_rm` | reader `input_indices_dfb.reserve_back` / `push_back` (`reader_values_indices_tensor.cpp:101`, `:104`) | writer `cb_final_indices.wait_front` / `get_read_ptr` / `pop_front` (`writer_interleaved.cpp:136`, `:145`, `:237`) |
| `c_16` | `temp` | writer `generate_bcast_unary_scalar`, whose body does `cb.reserve_back` / `push_back` (`writer_interleaved.cpp:127`, resolving to `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp:46`, `:49`) | compute `in1_scalar_dfb_obj.wait_front(1)` in `mul_block_bcast_scalar_inplace` (`compute/sampling.cpp:397`), via `:475` |

**Self-loop, 8 CBs, one toucher each.** Bind the single touching kernel PRODUCER and CONSUMER:

| CB | Name | Sole toucher | Access shape |
|---|---|---|---|
| `c_5` | `input_transposed` | compute | full FIFO cycle inside `top_k`: `reserve_back` (`compute/sampling.cpp:238`), `push_back` (`:281`), `wait_front` (`:290`), `pop_front` (`:329`) |
| `c_6` | `index_transposed` | compute | same cycle (`:239`, `:282`, `:291`, `:330`) |
| `c_7` | `values` | compute | produced in `top_k` (`:348`, `:354`), then consumed in place as `in0` by `add_block_inplace`, `mul_block_bcast_scalar_inplace`, `reduce_c`, `sub_exp_block_bcast_cols_inplace` and `mul_block_bcast_cols` (`:474-482`) |
| `c_9` | `cb_cur_max` | compute | produced by `reduce_c` (`:477`), consumed by `sub_exp_block_bcast_cols_inplace` `in1_dfb_obj.wait_front` (`:65`, via `:479`) |
| `c_10` | `cb_cur_sum` | compute | produced by `reduce_c` (`:480`), cycled by `recip_block_inplace` (`:161-175`), consumed by `mul_block_bcast_cols` `in1_dfb_obj.pop_front` (`:150`) |
| `c_13` | `output` | writer | **role-free raw peek only:** `cb_out.get_write_ptr()` (`writer_interleaved.cpp:147`) and `use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_out)` as the NoC write source (`:242`). No FIFO ops at all. |
| `c_14` | `k` | writer | locked producer with no consumer: `cb_k.reserve_back` / `get_write_ptr` / `push_back` used as NoC read staging, then read back via `k_ptr[core_id]` (`:94-102`) |
| `c_15` | `p` | writer | same shape (`:105-113`) |

**Dead CBs: none.** Every one of the 18 `buffer_index` values is referenced by at least one kernel, and the two tables above account for all of them.

**Multi-binding: none needed.** No CB has three or more distinct touchers, and no CB has two kernels locked to the same FIFO role. The three faces were checked explicitly:

- *(a) hidden second writer.* No kernel raw-writes a CB it is not the FIFO producer of, and there are no semaphores in this op to coordinate such a co-fill. The nearest shape is `c_16`: the writer NoC-reads into `cb_temp.get_write_ptr()` before `generate_bcast_unary_scalar` performs the `reserve_back` and `push_back` (`writer_interleaved.cpp:117-127`). Both touches belong to the same kernel, so `c_16` stays a 1-producer / 1-consumer CB.
- *(b) multiple readers.* No CB's read sites span two kernels. Each of `c_1`, `c_8`, `c_11` and `c_12` is read only by the writer. `c_13`, `c_14` and `c_15` are touched only by the writer. The rest are touched only by compute.
- *(c) dual-instance work-split.* Not present. The factory does instantiate the same `kernel_source` many times, one writer and one compute `KernelDescriptor` per core (`device/sampling_program_factory.cpp:383-458`), but each instance covers a **disjoint single-core** `core_ranges` (`CoreRangeSet single_core{CoreRange(core, core)}`, `:385`), so every node sees exactly one instance of each. That is the ordinary 1:1 case, not the co-resident split.

**Op-level roll-up:** 10 legal · 8 self-loop · 0 multi-binding · 0 dead.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. No CB in this op needs the multi-binding advanced option.
- **Cross-op / shared kernels:** the op instantiates **no** borrowed kernel files. All three `kernel_source` paths are its own, and no other op or test references them, so there is no shared-kernel fork or sunset question. The coupling is entirely function-call escape through headers, and the one entry that needs attention is the `CircularBuffer`-by-value donor parameter described below.
- **RTA varargs:** none. Both dataflow kernels read a fixed, small run of runtime args at constant indices: `get_arg_val<uint32_t>(0)` and `(1)` in the reader (`reader_values_indices_tensor.cpp:52-53`), and `(0)` through `(3)` in the writer (`writer_interleaved.cpp:39-42`). No counted loop over arg indices, and no data-selected element. Every one of the six is nameable.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ⭐ (one flagged shape, not a gate).** The escapes are all header / function-call escapes, with no file-path kernel instantiation. What was found:

- One donor function takes a **`CircularBuffer` by value**, the ⭐ shape from the donor table, which the recipe routes to cross-team discussion rather than to the porter.
- Every other donor entry point takes the CB index as a **template non-type parameter** (`uint32_t`), which `dfb::name`'s constexpr cast handles: ✓.
- No donor uses legacy addr-gen (`InterleavedAddrGen`, `ShardedAddrGen` and friends), no donor takes a raw semaphore address or a `uint32_t sem_id`, and no donor takes a `TensorAccessorArgs<N>` or a CTA-offset NTTP.

**Summary table**, one row per (op kernel, donor file):

| Op kernel | Donor file | Donor class | Status |
|---|---|---|---|
| `writer_interleaved.cpp` | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp` | 6, cross-family donor (`transformer`) | ✓ |
| `writer_interleaved.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` (plus `.inl`) | 2, official shared kernel library | ✓ |
| `writer_interleaved.cpp` | `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | 3, second shared-kernel pool | ⭐ |
| `compute/sampling.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` (plus `.inl`) | 2, official shared kernel library | ✓ |
| all three kernels | `tt_metal/hw/inc/api/**` (`dataflow_api.h`, `noc.h`, `dataflow_buffer.h`, `circular_buffer.h`, `core_local_mem.h`, `noc_traits.h`, `numeric/bfloat16.h`, `compute/*`, `ckernel_sfpu.h`) | 1, LLK / HAL / firmware | ✓ no concern |

Transitive note: the `sdpa_decode` donor includes `transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp`, which is where `copy_tile` (reached via `generate_mask`) lives. Same class-6 treatment, ✓.

**Per-call detail** for the non-✓ donor, plus the two ✓ entry points whose shape is worth recording:

- `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp`: **⭐ flag.**
  - `generate_bcast_unary_scalar(CircularBuffer cb, uint32_t scalar)` (`:44`), called at `writer_interleaved.cpp:127` as `generate_bcast_unary_scalar(CircularBuffer(cb_id_temp), temp_packed)`.
  - Shape: `CircularBuffer` by value, which is the donor table's ⭐ ⚠ entry ("op-by-op porting plus DFB-replaces-CB on the consumer side leaves no clean per-op story today; flag for cross-team discussion"). The header is **broadly shared**: nine other kernels across `normalization/softmax`, `transformer/sdpa` and `data_movement/bcast` call into it, so a donor-side signature change is a multi-op decision rather than this port's to make.
  - Mechanically, the call site constructs the `CircularBuffer` explicitly from a CB id, so a `dfb::` token would still satisfy the constructor through its constexpr `uint32_t` cast. That is recorded as an observation only: the recipe's disposition for this shape is cross-team discussion, and this audit does not propose a resolution.
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp`: ✓.
  - `generate_mask<uint32_t cb_mask_in, uint32_t PNHt>(uint32_t k_num_chunks, uint32_t Sk_chunk_t, uint32_t cur_pos)` (`:215`), called at `writer_interleaved.cpp:130`. The CB travels as a template `uint32_t` non-type parameter, which is the `uint32_t cb_id` row, ✓ OK in both runtime and template-parameter position.
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` and `reduce_helpers_compute.hpp`: ✓.
  - `calculate_and_prepare_reduce_scaler<uint32_t dfb_id, PoolType, ReduceDim>()` (`writer_interleaved.cpp:77-80`) and `compute_kernel_lib::reduce<PoolType, ReduceDim, in0_dfb, scale_dfb, out_dfb, ...>(...)` (`compute/sampling.cpp:190-198`). Both carry every CB as a `uint32_t` NTTP, so ✓, and the lib team owns the files internally.

**Borrowed kernel files (file-path kernel instantiation): none.** All three `KernelDescriptor::kernel_source` values point into this op's own `device/kernels/` tree (`device/sampling_program_factory.cpp:366-367`, `:416-417`, `:448`). A repo-wide grep for `sampling/device/kernels` finds no consumer outside this factory, and no `*_metal2.cpp` fork exists beside any of the three. So the porter neither reuses nor creates a shared-kernel fork, and there is no sunset list.

### Relaxation candidates

None. The op has no custom hash to mine.

### TTNN factory analysis

The sheet-derived facts, with `file:line` evidence, are in *Gate detail, TTNN factory concept* above. Summarized here for the record:

- **Op-owned tensors:** none. `create_descriptor` returns a plain `ProgramDescriptor` and never populates a `buffers` vector. The output tensor is a normal framework-allocated output (`device/sampling_device_operation.cpp:183-190`).
- **MeshWorkload need:** none. This is the vanilla `descriptor` concept rather than a `WorkloadDescriptor`, so the secretly-SPMD question does not arise. The sheet's informational `Execution Model = SPMD` agrees.
- **Pybind `create_descriptor` and other risky pybind:** absent. `sampling_nanobind.cpp` binds only the user-facing `ttnn::sampling` operation.
- **Custom hash, `get_dynamic_runtime_args`, `override_runtime_arguments`:** all absent. The greps in the table above return nothing.
- **Target concept:** `ProgramSpecFactoryConcept`, no op-owned tensors, which matches the sheet's own `Porting Target` column.

## Misc anomalies  *(team-only, non-gating)*

These route to the ops team. The port does not act on them.

1. **A `Wt == 1` input hangs the device.** Validation admits any power-of-two `Wt`, including 1 (`device/sampling_device_operation.cpp:87-93`, meaning `W == 32`). But the reader pushes only `Ht * Wt == 1` value tile (`reader_values_indices_tensor.cpp:85-95`) while the compute kernel's local-sort loop unconditionally waits for two: `for (wt = 0; wt < Wt; wt += 2) { input_dfb.wait_front(2); index_dfb.wait_front(2); }` (`compute/sampling.cpp:242-245`). The second tile never arrives. The existing validation comment at `device/sampling_device_operation.cpp:83-86` reasons about odd and non-power-of-two `Wt` but not about `Wt == 1`, so the minimum supported `W` appears to be 64 rather than 32.
2. **Dead compile-time arg.** The factory passes `aligned_out0_unit_size` as writer CTA `args_base + 8` (`device/sampling_program_factory.cpp:403`), and the kernel documents it as unused rather than reading it (`writer_interleaved.cpp:59`).
3. **Dead local variable.** `uint32_t arg_id = 0;` at `writer_interleaved.cpp:44` is assigned and never read.
4. **Commented-out FIFO calls around the `temp` staging read.** `cb_temp.reserve_back(1)` and `cb_temp.push_back(1)` are commented out at `writer_interleaved.cpp:116` and `:121`, so the NoC read lands at `cb_temp.get_write_ptr()` without a reservation, the value is read back via `temp_ptr[core_id]` (`:125`), and then `generate_bcast_unary_scalar` overwrites the same region with its own `reserve_back` and `push_back` (`:127`). It works because a single kernel owns both touches, but the dead lines make the intended protocol hard to read.
5. **The `temp` CB is sized as a byte staging buffer yet also serves as a compute tile operand.** `c_16` is allocated `num_cores * 2` bytes with `page_size` equal to that (`device/sampling_program_factory.cpp:341-349`), so at most 64 bytes for 32 users. That is correct for the writer's NoC read of the `temp` row, but compute then uses the same CB as the scalar operand of `mul_tiles_bcast_scalar` (`compute/sampling.cpp:402`, via `:475`), where a bf16 tile is 2048 bytes. My guess is that the unpacker reads past the 64-byte allocation into whatever SRAM follows, and that the sampled result is still correct because only datum 0 is used for a scalar broadcast, which would mask the overread. Worth a look from someone who can confirm the LLK's unpack footprint for a scalar-broadcast operand. The same pattern is not a concern for `c_14` and `c_15`, which are writer-only staging buffers never handed to compute.
6. **Float `log2` for an exact power of two.** `static_cast<uint32_t>(std::log2(Wt))` (`device/sampling_program_factory.cpp:439`) computes `logWt` through a floating-point path. Validation guarantees `Wt` is a power of two so the result is exact today, but an integer bit-width computation would remove the dependence on `std::log2`'s rounding.
7. **Donor observation, not this op's file.** `generate_mask` takes its fill base from `cb_mask.get_read_ptr()` after calling `cb_mask.reserve_back(...)` (`transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp:262-263`), where `get_write_ptr()` is the conventional pairing. The two coincide while nothing has been popped, which is the case for sampling's `c_4`, so the current behavior is correct. The pairing is fragile for any future consumer that pops. Routes to the `sdpa_decode` owners rather than to this port.

## Questions for the user

None blocking. Items 1 and 5 in *Misc anomalies* are the two worth someone's attention, but neither affects the port. Item 1 is an input-validation gap in a configuration the port neither adds nor removes, and item 5 is a pre-existing sizing question about an unchanged CB.

## Recipe notes

1. **A readiness-sheet column header does not match the name the recipe quotes.** The audit's *TTNN factory concept prerequisite* refers to the column as `Override runtime args method? (PD and legacy)`, and its derivation block says `col "Override runtime args method?"`, while the live sheet's header reads `Override runtime args method?\n(PD only)`. `ttnn_op_porting_readiness.md` also spells it `(PD and legacy)`. Since the recipe's standing rule is to reference every column by header name and never by position, a name-based lookup for the quoted string finds nothing. The value was unambiguous here (`no`), so nothing was at risk, but the recipe's quoted header should be reconciled with the sheet's.
2. **`emplace_runtime_args` with a `MeshTensor` is a delivery shape the *TensorParameter analysis* detection list does not name.** The subject enumerates the descriptor form, the imperative form, the helper-function form, the `Buffer*`-binding form and the CTA-baked-address form. This op uses none of them: it pushes `MeshTensor` references (`KernelDescriptor::RTArgList::push_back(const MeshTensor&)`, `tt_metal/api/tt-metalium/program_descriptors.hpp:176`, with matching `emplace_runtime_args` overloads at `:195-202`). It is clearly the `MeshTensor` sibling of the `Buffer*`-binding form, with the same auto-registered binding, the same cache-hit patching, and the same "classify by what the kernel does with the base" rule, and I treated it that way. Worth adding as a named bullet so the next auditor does not have to reason it out, especially since the shape carries no `->address()` call at all, which can make a grep-first auditor read "no address RTAs found" as "no tensor bindings to classify."
3. **The Device 2.0 gate has no explicit ruling on raw `volatile tt_l1_ptr` local access.** The migration guide's *Memory Access* section lists `reinterpret_cast<volatile uint32_t*>(address)` as the Legacy API and `CoreLocalMem<T>` as its replacement, but the gate's RED bullets enumerate only CB-index-keyed holdovers and broad Device 1.0 idioms (raw NoC, manual CB index management, legacy addr-gen, raw semaphore addresses). The raw local pointer appears in neither list. It shows up in reached donor code here (`ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp:16`, `:31`, `:47`; `transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp:37-42`, `:59-62`). I judged it non-gating because the CB access itself already goes through the Device 2.0 wrapper, so the Metal 2.0 binding token has an object to attach to, which is the gate's own stated rationale. A one-line ruling either way would remove the judgment call. The "if Device 2.0 allows the free function, so do we" sanctioning bullet currently covers only `get_tile_size` and `get_local_cb_interface`.
4. **Minor.** The *Feature compatibility* GlobalCircularBuffer and `address_offset` entries are both settled by the same observation: the factory's `CBDescriptor` designated initializers omit those fields, so they keep their struct defaults. That absence-by-default is cleaner and more complete evidence than grepping for each signal, but neither entry's recognition list mentions it. A pointer to "check whether the `CBDescriptor` literals set the field at all" would shorten both scans on any `ProgramDescriptor`-form op.
