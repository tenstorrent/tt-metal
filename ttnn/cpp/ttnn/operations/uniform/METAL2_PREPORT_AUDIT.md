# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/uniform`

Single device operation, single factory. The `create_descriptor` / `override_runtime_arguments` pair lives on the
device-op itself (there is no separate `ProgramFactory` class), split across two files:

- **`UniformDeviceOperation`**
  - `UniformDeviceOperation` (single-descriptor) — `device/uniform_program_factory.cpp` (`create_descriptor`,
    `override_runtime_arguments`), declared in `device/uniform_device_operation.hpp`

Kernels referenced by the factory (both live in the op's own directory, both are also bound by `rand` — see
*Out-of-directory coupling*):

- `device/kernels/writer_uniform.cpp` (writer / DM)
- `device/kernels/compute_uniform.cpp` (compute)

No unreferenced kernel files in the directory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `086a669ff5e 2026-08-15 docs(metal_2.0): two porter-facing gaps a blind cold read turned up`

> ### Readiness-sheet gate — held `no` deliberately; audited as `yes` on the maintainer's instruction
>
> The readiness sheet's `Is able to port?` cell for `uniform` currently reads **`no`**. Per the recipe that is the
> gate, and a literal reading REDs the op. It is **not** an op defect: `uniform` belongs to a family whose Metal 2.0
> support (the `CustomProgramSpecFactoryConcept` path, selected by `Override runtime args method? == yes`) has only
> just been added to the audit and port recipes and is **still under test**. The sheet rows are being **held red on
> purpose**, to stop the porting team starting these ops before that testing completes.
>
> This audit was run with the cell treated as **`yes`**, on the recipe maintainer's explicit instruction, as part of
> that testing. The verdict below is therefore **GREEN**, and a porter brief is issued.
>
> **Downstream readers: this op is not yet released for porting.** The GREEN here certifies that every audit gate
> clears on the code; it does not lift the family-wide hold. The sheet remains the authority on when the port may
> start. Fingerprint of the held row, so a later reader can tell whether the hold has since lifted: `Concept` =
> `descriptor`, `Override runtime args method?` = `yes`, `Porting Target` = `CustomProgramSpecFactoryConcept`,
> `Is able to port?` = `no`. The same hold covers `rand` and 20 other rows sharing that shape.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/uniform` |
| **Overall** | **GREEN** (gate audited as `yes` per the note above; sheet row held `no`) |
| **DOps / Factories** | `UniformDeviceOperation` → `UniformDeviceOperation (single-descriptor)` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — both kernels fully Device 2.0; no holdovers |
| *Prereqs* — Cross-op escapes | Ok (function-call escapes all `tt_metal/*`) — but see the **lent-kernel** coordination finding |
| *Feature Support* — overall | **GREEN** (all four Appendix A entries `N/A`) |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (as audited — see the note above; sheet cell held `no` pending family test-out) |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | `compute_program_hash`: **No**. Backdoor (`attribute_values`): **Yes** — `device/uniform_device_operation.hpp:28-29` (not a gate; port leaves it intact) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (removed by #50338, 2026-07-30) |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** (not a gate; selects `CustomProgramSpecFactoryConcept`): `device/uniform_program_factory.cpp:213` |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `CustomProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | **none** (GREEN — clean base; page offset travels as a separate scalar) |
| *Port work* — Tensor bindings (per binding) | `output` → **Case 1** (`TensorAccessor`) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | **none** — the op's single accessor is 2-arg |
| *Port work* — CB endpoints | `c_24` intermed: legal 1:1 · `c_0` dst: **self-loop** |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution. Recorded per
`(CB, config)` below.

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`, beside this file), subject to the family-wide hold described in the
note above.

Every gate clears on evidence: Device 2.0 is complete on both kernels, no Appendix A feature is in use, no offset base
pointer, no `TensorAccessor` 3rd argument, `TensorParameter relaxation == none`. The TTNN factory concept gate is
audited `yes` per the maintainer's instruction.

`uniform` is a small, low-risk port. One tensor binding (Case 1), one CB needing a self-loop, no varargs, no donor
kernels, no semaphores. **The one thing that is not routine is the shared-kernel coupling:** both kernels live in
`uniform`'s directory but are *lent* to `rand`, so they must be forked (`_metal2`), not converted in place — see
*Heads-ups*.

## Gate detail

- **TTNN factory concept (`Is able to port?`): GREEN — audited as `yes`.** See the note at the head of this document
  for why the sheet cell reads `no` and on whose instruction it was overridden. The cross-check of every
  cheaply-checkable factual column against the code is **clean** — the sheet's factual columns and the code agree in
  every particular, so nothing here is a staleness signal.

  Sheet row, verbatim (`Op` == `uniform`, one row, matching the code's one factory — factory-set match ✓):

  | Column | Value |
  |---|---|
  | `Device operation` | `UniformDeviceOperation` |
  | `Factory (variant)` | `UniformDeviceOperation (single-descriptor)` |
  | `Concept` | `descriptor` |
  | `Op Classification` | `PD Op (custom)` |
  | `Execution Model` | `SPMD` |
  | `Porting Target` | `CustomProgramSpecFactoryConcept` |
  | `Custom hash (compute_program_hash)` | `no` |
  | `Backdoor custom hash (attribute_values / to_hash)` | `yes` |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` |
  | `Override runtime args method? (PD only)` | `yes` |
  | `Pybind descriptor (nb::class_ of device op)` | `no` |
  | `Smuggled pointer (raw buffer addr in RTA/CRTA)` | `no` |
  | `Known op issues` | *(empty)* |
  | `Diego validation` | `yes` |
  | **`Is able to port?`** | **`no`** — *deliberate family-wide hold; audited as `yes`, see the note above* |
  | `TensorParameter relaxation` | `none` |
  | `Op-owned tensors?` | *(empty)* |
  | `Secretly SPMD Workload?` | *(empty)* |
  | `Factory definition path` | `ttnn/cpp/ttnn/operations/uniform/device/uniform_program_factory.cpp` |
  | `Declared in` | `ttnn/cpp/ttnn/operations/uniform/device/uniform_device_operation.hpp` |

  Cross-check against the code — every factual column verified, all agree:

  | Column | Sheet | Code evidence | Agrees |
  |---|---|---|---|
  | `Concept` | `descriptor` | `create_descriptor()` returning `ProgramDescriptor` @ `device/uniform_program_factory.cpp:107` | ✓ |
  | `Custom hash` | `no` | no `compute_program_hash` anywhere in the op dir | ✓ |
  | `Backdoor custom hash` | `yes` | hand-written `attribute_names` / `attribute_values` listing only `memory_config`, `compute_kernel_config` — `from`/`to`/`seed` deliberately excluded @ `device/uniform_device_operation.hpp:28-29` | ✓ |
  | `Runtime-args update` | `no` | no `get_dynamic_runtime_args` hook on the device-op; removed by `48cb0736920` (#50338, 2026-07-30) | ✓ |
  | `Override runtime args method?` | `yes` | `UniformDeviceOperation::override_runtime_arguments` @ `device/uniform_program_factory.cpp:213`, declared `device/uniform_device_operation.hpp:51` | ✓ |
  | `Pybind descriptor` | `no` | `uniform_nanobind.cpp` binds only the user-facing `ttnn::uniform`; no `create_descriptor`, no `nb::class_` | ✓ |
  | `Smuggled pointer` | `no` | the output address is passed as an **annotated** `Buffer*` via `emplace_runtime_args` @ `device/uniform_program_factory.cpp:204`, not a bare `->address()` | ✓ |
  | `Op-owned tensors?` | *(empty)* | `descriptor` concept, no `buffers` vector | ✓ |
  | Factory-set match | 1 row | 1 factory in code | ✓ |

  Cross-column invariants: `get_dynamic_runtime_args == no` on a `descriptor` concept ✓; `Op-owned tensors?` not `yes`
  on a `descriptor` concept ✓. No invariant violated.

  Target-concept corroboration: `CustomProgramSpecFactoryConcept` is fully implemented in the framework today —
  `ttnn/api/ttnn/operation_concepts.hpp:132`, dispatched at `ttnn/api/ttnn/device_operation.hpp:243`, adapter at
  `ttnn/api/ttnn/mesh_device_operation_adapter.hpp:955`, static-asserted in
  `tests/ttnn/unit_tests/gtests/test_launch_operation.cpp:164`.

- **Device 2.0 (every kernel used): GREEN.** Both kernels the factory instantiates are fully Device 2.0. No
  CB-index-keyed free-function holdovers, no Device 1.0 idioms. Corroborated by history: `3be2663ed18` ("[Cleanup]
  Update rand kernels to use the Device 2.0 API", #47649) and `c60d961ade9` ("[Bug fix] Fix non-compliant CB usage in
  kernels", #47674) migrated exactly these files.

  | Kernel | Device 2.0 evidence | Verdict |
  |---|---|---|
  | `device/kernels/writer_uniform.cpp` | `Noc noc` @ 28 · `CircularBuffer cb_intermed/cb_dst` @ 29-30 · `noc.async_write(CoreLocalMem<uint32_t>(…), output_addrg, …)` @ 42, 66 · wrapper **methods** `cb_dst.get_write_ptr()` @ 33, `cb_intermed.get_read_ptr()` @ 38 (not the free functions) · `TensorAccessor` @ 24 | ✓ |
  | `device/kernels/compute_uniform.cpp` | `CircularBuffer cb_intermed` @ 25 with `.reserve_back` / `.push_back` @ 31, 41. Remaining CB-index arguments (`init_sfpu` @ 27, `pack_tile` @ 38) are compute-side LLK entry points, outside the Device 2.0 *data-movement* surface | ✓ |

  One free function checked explicitly rather than assumed:
  `get_local_cb_interface(dst_cb_id).fifo_page_size` @ `device/kernels/writer_uniform.cpp:26`. It is **not** a holdover
  on two independent grounds: (a) the recipe lists `get_local_cb_interface(cb_id)` as sanctioned; (b) the holdover test
  requires a wrapper-method replacement to exist, and the Device 2.0 `CircularBuffer`
  (`tt_metal/hw/inc/api/dataflow/circular_buffer.h`) exposes **no** page-size accessor — its metadata set is
  `get_tile_size` / `get_tile_hw` / `get_dataformat`, and `get_write_ptr()` / `get_read_ptr()` are themselves
  implemented *by calling* `get_local_cb_interface`. Nothing to replace it with at this stage. (Port-time breadcrumb in
  *Heads-ups*, and carried into the brief.)

- **Feature compatibility: GREEN** — all four entries `N/A`. Scan covered host code, the device-op, the factory, and
  both kernels.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` / `CreateGlobalCircularBuffer` / `global_circular_buffer` field / `remote_index` / `remote_cb_*` / `experimental::CreateCircularBuffer(…, global_cb)` anywhere in the op. Both CBs are plain `CBDescriptor` literals @ `device/uniform_program_factory.cpp:133, 144`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | Neither `CBDescriptor` sets `address_offset` (both literals omit the field → default 0). No `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. |
  | GlobalSemaphore | N/A | The op uses **no** semaphores at all — `grep -i semaphore` over the whole op directory returns nothing. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Op-level cue absent: `tensor_args_t` carries exactly one named `const Tensor& input` (`device/uniform_device_operation.hpp:32-34`), not a variable-count container. Kernel-level decider absent: every `get_compile_time_arg_val` is at a **literal constexpr** index — `writer_uniform.cpp:15,16` (0, 1) plus `TensorAccessorArgs<2>()` @ 17; `compute_uniform.cpp:11` (0). No loop over CTAs. |

- **CB endpoints (GATE-free): all resolvable, nothing blocks.** Two CBs, both over the same `all_cores` range, both
  present in every config. The only config axis is the output dtype (`OUTPUT_DTYPE_BFLOAT16` vs `OUTPUT_DTYPE_FLOAT32`,
  set at `device/uniform_program_factory.cpp:156-160`); the census is identical under both.

  | CB | Config | Touchers on a node | Verdict | Port-time resolution |
  |---|---|---|---|---|
  | `c_24` intermed (Float32) @ `uniform_program_factory.cpp:133` | both | **2** — compute is a **locked producer** (`cb_intermed.reserve_back` @ `compute_uniform.cpp:31`, `.push_back` @ 41); writer is a **locked consumer** (`cb_intermed.wait_front` @ `writer_uniform.cpp:36`, `.get_read_ptr` @ 38, `.pop_front` @ 49/64) | **plain 1:1** (one locked P + one locked C) | none — legal as-is, no flag |
  | `c_0` dst @ `uniform_program_factory.cpp:144` | both | **1** — the writer only (`cb_dst.reserve_back` @ `writer_uniform.cpp:32`, `.get_write_ptr` @ 33, `.push_back` @ 78) | **single-ended / sync-free** | **self-loop** — bind the writer PRODUCER **and** CONSUMER (legal on Gen1 for DM) |

  Hidden-second-writer hunt run and negative: there are no semaphores in the op, so no semaphore-gated raw co-fill is
  possible; and each CB's full access set is accounted for above. No dead CB — both indices are referenced (`c_0`'s
  index reaches the writer as CTA 1 and is genuinely used, including in the FLOAT32 path via
  `get_local_cb_interface(dst_cb_id).fifo_page_size` @ `writer_uniform.cpp:26`).

- **Offset base pointers: GREEN.** One address-bearing argument in the whole op, and it carries a clean base.
  `uniform` appears in neither the `2026-07-19_offset_base_pointers.md` tables nor as a new fold — scan run
  independently, per the recipe's "never let *not in the tables* stand in for *scanned and clean*."

  | Site | Expression | Fold? |
  |---|---|---|
  | `device/uniform_program_factory.cpp:204` (writer RTA 0, cache miss) | `output.buffer()` — an annotated `Buffer*`, no arithmetic | no |
  | `device/uniform_program_factory.cpp:228, 243` (writer RTA 0, cache hit) | `output.buffer()->address()`, assigned straight to `writer_args[0]` — no `+` | no |

  The per-core page offset is deliberately **not** folded into the address: it travels as a separate scalar
  (`tile_offset` → writer RTA 1 → `start_id`) and is applied on-device as a page index, `{.page_id = i}` @
  `writer_uniform.cpp:47, 71`. That is the shape the Type-1 fix produces, already present by construction. Type 3
  (`address_offset`) N/A; Type 4 (`narrow`) N/A.

- **TensorAccessor 3rd argument: GREEN — N/A, the subject does not fire.** The op constructs exactly one
  `TensorAccessor`, with **two** arguments: `TensorAccessor(dst_args, dst_addr)` @ `device/kernels/writer_uniform.cpp:24`.
  No explicit page-size override anywhere, so there is no site to classify. (`uniform` is likewise absent from
  `2026-07-06_tensor_accessor_3rd_arg_triage.md`, consistent with the read.) The write size passed to
  `noc.async_write` is a separate `page_bytes` argument, not an accessor constructor argument — not this subject.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding): `output` — **Case 1** (via `TensorAccessor`). The op has exactly one tensor
  (`create_output_tensors` returns the input — `uniform` is in-place, `device/uniform_device_operation.cpp:31-35`), and
  exactly one kernel touches tensor memory (the writer; the compute kernel only produces into a CB → out of scope).

  Delivery detail the porter should know: on **cache miss** the base arrives as an annotated `Buffer*` —
  `writer_desc.emplace_runtime_args(core, {output.buffer(), tile_offset, units_per_core})` @
  `device/uniform_program_factory.cpp:204` — which the framework auto-registers as a `BufferBinding` and patches on
  cache hits. On **cache hit** `override_runtime_arguments` *also* writes it explicitly (`writer_args[0] = out_addr` @
  `:243`), because the override supersedes binding resolution. Both paths are correct today; neither is the
  silent-wrong `->address()`-on-an-RTA hazard. Under Metal 2.0 both disappear into one `TensorParameter` /
  `TensorBinding`, the kernel builds `TensorAccessor(tensor::<name>)`, and RTA slot 0 plus the
  `TensorAccessorArgs(output.buffer()).append_to(writer_ct_args)` CTA plumbing @ `:163` both go away.

- **TensorParameter relaxation:** `none`.

- **TensorAccessor 3rd arg:** none — the single accessor is already 2-arg.

- **CB endpoints:** self-loop `c_0` dst (both dtype configs) · `c_24` intermed already legal 1:1. No multi-binding, no
  dead CB.

- **TTNN factory wiring (target concept):** `CustomProgramSpecFactoryConcept`. `override_runtime_arguments` @
  `device/uniform_program_factory.cpp:213-247` is *translated* into a `ProgramRunArgs`-returning method, not deleted.
  The backdoor hash (`attribute_names` / `attribute_values` @ `device/uniform_device_operation.hpp:28-29`) is left
  exactly as it is — it is load-bearing here, and the `override_runtime_arguments` body is precisely the mechanism that
  makes excluding `from`/`to`/`seed` from the hash safe. The two must stay in sync through the port.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. No CB on any node has ≥3 touchers or two kernels locked to the
  same FIFO role.

- **Cross-op / shared kernels — this is the one real coordination cost on this op.** Both kernels are **lent**: they
  live in `uniform`'s own directory, but `rand` binds them by file path —
  `ttnn/cpp/ttnn/operations/rand/device/rand_program_factory.cpp:28-29`, used at `:165` and `:181`. Nothing about the
  path warns a porter, which is exactly the trap the shared-kernel caution names ("the file sits inside your writeable
  surface, so converting it in place feels safe, and it breaks every borrower the moment you do").

  | Kernel | Owner | Also bound by | `_metal2` fork beside it? |
  |---|---|---|---|
  | `device/kernels/writer_uniform.cpp` | `uniform` (lent) | `rand` | **No** — this port creates the first |
  | `device/kernels/compute_uniform.cpp` | `uniform` (lent) | `rand` | **No** — this port creates the first |

  Census run by filename per the caution's procedure (`grep -rl <filename> ttnn/cpp/ttnn/operations/`), and each hit
  checked to be a real kernel-source binding. Resolution is **rung 2** — create `writer_uniform_metal2.cpp` and
  `compute_uniform_metal2.cpp` beside the originals, in `uniform`'s own directory, and leave the originals serving
  `rand`. `{rand}` is a **sunset list, not authorization to convert in place** — and note `rand` carries the *identical*
  sheet profile (`descriptor`, backdoor hash `yes`, override `yes`), so it sits under the same family-wide hold and
  cannot co-migrate today. Name the fork's bindings for the *kernel's* role vocabulary, not `uniform`'s locals, since
  `rand` inherits them at sunset.

- **RTA varargs:** none — every RTA is a distinct field at a constant index, in both kernels
  (`writer_uniform.cpp:19-21`; `compute_uniform.cpp:13,18,19,21,22`). No counted loop, no data-selected index. All
  nameable: writer → `dst_addr` (becomes the tensor binding), `start_id`, `num_tiles`; compute → `seed`, `f2u_from`,
  `f2u_to`, `start_id`, `num_tiles`.

- **Port-time breadcrumb (Device 2.0 → Metal 2.0):** `get_local_cb_interface(dst_cb_id).fifo_page_size` @
  `device/kernels/writer_uniform.cpp:26` is sanctioned at the Device 2.0 stage (above), but the port's kernel-side
  whitelist rule 7 moves such metadata lookups onto the DFB object. Worth a look before assuming a mechanical swap:
  `DataflowBuffer` (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:167+`) exposes `get_tile_size` / `get_tile_r_dim` /
  `get_tile_c_dim` / `get_tile_hw` / `get_tile_num_faces` — a **tile**-metadata set, with no direct `fifo_page_size`
  analog. For this op the CB's page size *is* its tile size (`page_size = dtype_tile_size` @
  `device/uniform_program_factory.cpp:150`), so `dfb::dst.get_tile_size()` should be the equivalent — but that
  equivalence is an inference from the descriptor, not an API identity, and is worth confirming at port time.

## Team-only

- **Out-of-directory coupling & donor shape: ✓ clean.** No function-call escape leaves `tt_metal/*`.

  | Op kernel | Donor include | Donor class | Status |
  |---|---|---|---|
  | `writer_uniform.cpp` | `<tt-metalium/constants.hpp>`, `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h` | 1 — `tt_metal/*` (LLK/HAL) | ✓ |
  | `compute_uniform.cpp` | `api/compute/compute_kernel_api.h`, `api/compute/eltwise_unary/eltwise_unary.h`, `api/compute/eltwise_unary/rand.h`, `api/dataflow/circular_buffer.h` | 1 — `tt_metal/*` (LLK/HAL) | ✓ |

  No `ttnn/cpp/ttnn/kernel_lib/`, no `ttnn/cpp/ttnn/kernel/`, no `kernel_helper_functions/`, no in-family or
  cross-family op donors. Per-call detail omitted (all rolls ✓). The **file-path** coupling is independent and is *not*
  clean — see the lent-kernel finding under *Heads-ups*.

- **Relaxation candidates (FALLIBLE — candidates to verify; the ops team owns the real analysis):** the backdoor hash
  @ `device/uniform_device_operation.hpp:28-29` hashes only `memory_config` and `compute_kernel_config`, with tensor
  shape/dtype/device coming from `tensor_args`. It reveals no *tensor*-property independence — the exclusions are all
  scalar op attributes (`from`, `to`, `seed`) re-applied by `override_runtime_arguments`, which is the intended design,
  not a latent relaxation. **No candidate.** Consistent with the sheet's `TensorParameter relaxation == none`.

- **TTNN factory analysis:** current concept `descriptor` (`create_descriptor` @ `device/uniform_program_factory.cpp:107`)
  · no op-owned tensors · no MeshWorkload need (sheet `Execution Model` == `SPMD`; the op returns a plain
  `ProgramDescriptor`) · no pybound `create_descriptor` and no other risky pybind (`uniform_nanobind.cpp` binds only
  `ttnn::uniform` with scalar/optional args) · no `compute_program_hash`, backdoor hash present @
  `device/uniform_device_operation.hpp:28-29` · no `get_dynamic_runtime_args` · `override_runtime_arguments` @
  `device/uniform_program_factory.cpp:213` → target `CustomProgramSpecFactoryConcept`. Gate conjuncts all clear:
  relaxation `none` ✓, `get_dynamic_runtime_args` absent ✓, not multi-program ✓.

## Misc anomalies  *(team-only, non-gating; route to the ops team — the port does not act on these)*

- **`fp32_dest_acc_en` and `packer_l1_acc` are destructured from the compute-kernel config and then ignored.**
  `device/uniform_program_factory.cpp:127-128` unpacks all five fields from
  `get_compute_kernel_config_args(...)`, but `:182` hard-codes `.fp32_dest_acc_en = true` (with a comment explaining
  why) and `packer_l1_acc` is never used at all. Meanwhile the whole `compute_kernel_config` attribute *is* fed to the
  program hash (`device/uniform_device_operation.hpp:29`) — so two user-settable fields participate in cache-key
  distinctions while having **no effect on the generated program**. Two distinct programs are cached for configs that
  are behaviourally identical. (The forcing of `fp32_dest_acc_en` is intentional and documented; the hash consequence
  and the dead `packer_l1_acc` look unintended.)

- **The `1e-6f` endpoint epsilon does not scale with the range.**
  `device/uniform_program_factory.cpp:97-99` computes the exclusive upper bound as `to - 1e-6f` in `float`. For `to`
  above roughly `8.4` the subtraction is already at or below one ULP, and for larger `to` (say `to = 1e6`)
  `to - 1e-6f == to` exactly — so the documented half-open `[from, to)` contract silently becomes closed, and the
  op can return exactly `to`. A relative epsilon (`std::nextafter(to, -inf)`, or `to - eps*max(1, |to|)`) would hold the
  contract across the range. Nothing validates `to`'s magnitude (`device/uniform_device_operation.cpp:18` only checks
  `from < to`).

- **The `c_0` output CB is allocated but never used as storage in the FLOAT32 configuration.**
  `device/kernels/writer_uniform.cpp` reserves it, takes its write pointer @ `:32-33`, and pushes it back @ `:78` in
  both configs, but only the `OUTPUT_DTYPE_BFLOAT16` branch @ `:53-63` writes bytes into it — the FLOAT32 branch @
  `:41-50` NOC-writes straight from the intermediate CB. So under FP32 output the CB costs a full Float32 tile of L1
  (`device/uniform_program_factory.cpp:144-152`) purely to carry its `fifo_page_size` @ `:26`, which is a host-known
  constant that could be a CTA. `dst_cb_write_ptr` @ `:33` is likewise dead in that config. Not a correctness issue,
  and **not** porter work — the port must carry the CB across unchanged (see the brief's self-loop item).

- **Neither `#ifdef` branch is taken for an out-of-range output dtype.** `device/uniform_program_factory.cpp:156-160`
  has `default: break;`, so a dtype other than BFLOAT16/FLOAT32 compiles a writer whose loop body performs **no NOC
  write at all** — a silent no-op rather than a diagnostic. `validate_inputs`
  (`device/uniform_device_operation.cpp:15-17`) does currently constrain the dtype, so this is unreachable today; a
  `TT_THROW` in the `default` arm would keep it that way if the validation ever loosens.

## Recipe notes

1. **A deliberately-held `Is able to port?` row is indistinguishable, to the auditor, from a broken one — and the
   recipe has no routing for either.** `#ttnn-factory-concept-prerequisite` says *"**`no`** → GATE. Attribute it with
   the blocking table above… **Name the column that blocked**"*, but on this op every column in that table is clear, so
   the instruction cannot be followed. The recipe's only nearby escape is the *spreadsheet is broken* clause, which is
   scoped to a **cross-check conflict** ("Cross-check conflicts with the sheet, or the op has no row") — and my
   cross-check found no conflict at all; every factual column matches the code. On my first pass I therefore REDed the
   op and routed it to the sheet owner as broken, which was wrong: the row is held on purpose, pending test-out of the
   newly-added `CustomProgramSpecFactoryConcept` support. The recipe should name this state. Suggested addition to the
   routing list: *"`no` that no blocking-table column explains → the row may be a deliberate hold (support landed in
   the recipe but not yet released to the porting team) or a stale derivation. Either way it is a GATE; report the
   unattributability and route to the sheet owner rather than inferring a cause."* A `Known op issues` cell reading
   e.g. `held: CustomProgramSpec test-out` would resolve it at source and cost the auditor nothing.

2. **The recipe says `Override runtime args method? == yes` "does not block", while every such row on the sheet is
   currently blocked.** Stated three times in the recipe (the non-blocking-columns list, the two-runtime-args-columns
   section, `#ttnn-porting-shape`), yet the sheet's verdict tracks that column exactly: **22/22** otherwise-clean
   `descriptor` rows with it `yes` read `no`, and **211/211** with it `no` read `yes`. That is the hold, and it is
   correct policy — but nothing in the recipe told me so, and finding it required a cross-tabulation the recipe
   arguably forbids (*"Do not reproduce or recompute its derivation… would invite you to argue with the cell"*). An
   auditor obeying that literally reports "RED, gate says no" with no path forward. Worth a sentence where the column
   is introduced: something like *"support for this target concept is newly landed and the sheet rows are held pending
   test-out; expect `Is able to port? == no` on these ops until the hold lifts."* That single line would have saved
   this entire investigation and prevented my incorrect broken-sheet routing.

3. **The *Red* scoping rule assumes a RED implies the code will change.** Not exercised in the final verdict, but it
   bit on my first pass and will recur for any held row. The rule's rationale is that *"the op is re-audited against
   possibly-changed code once the blockers clear, so producing the detail now is unread and likely stale effort."*
   When the blocker is a **hold, a sheet fix, or a wait-for-framework-feature**, the code is untouched and the
   informational detail is fully reusable at re-audit — deferring costs a second full pass and saves nothing. Suggest
   a carve-out: *"when the RED's resolution does not entail a code change, run the informational subjects anyway."*

4. **`get_local_cb_interface` is on the sanctioned list, but the stated justification for it doesn't check out.** The
   Device 2.0 Green bullet says `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)` are *"both of which the
   Device 2.0 migration guide keeps as free functions in its migrated examples."* That is true of `get_tile_size`
   (`device_api_migration_guide.md:630`, inside the migrated example) but **`get_local_cb_interface` does not appear in
   that guide at all**. The sanction still holds on the recipe's own second test — the wrapper exposes no page-size
   accessor, so no replacement exists — but an auditor who follows the recipe's "check the current Device 2.0 surface
   rather than assuming" instruction will find the cited evidence missing and have to re-derive the sanction. Suggest
   re-grounding that bullet on the wrapper header (`tt_metal/hw/inc/api/dataflow/circular_buffer.h`, where
   `get_write_ptr()` / `get_read_ptr()` are *implemented via* `get_local_cb_interface`) rather than on the guide's
   examples.

5. **Minor: "TensorAccessor 3rd argument" has no defined status value for "the subject never fires."** The status-summary
   row offers `drop (Class 1/2)` or `flag → GATE (Class 3/4/Special)`, and the gate-detail template offers *"GREEN —
   every site Class 1/2"* or a RED. An op whose accessors are all 2-arg — which the recipe itself says is the common
   case (*"Most accessors omit it and only a handful of ops set it, so this subject fires rarely"*) — matches none of
   these. I wrote `none` / `N/A, the subject does not fire`, distinguishing it from "sites found and classified Class
   2," which is a materially different finding.

6. **Minor: the brief template has no home for a port-time breadcrumb the Device 2.0 gate raises.** The Device 2.0
   section's own *Breadcrumb* note (kernel-side whitelist rule 7 — move metadata lookups onto the DFB object) produces
   a finding that is squarely porter-facing, but the brief's *Watch for* section enumerates exactly three bullets (CB
   endpoints, cross-op/shared kernels, RTA varargs) and none fits. I added a fourth bullet rather than drop it. If that
   is intended, the template could say so; if not, the breadcrumb has no route to the porter.
