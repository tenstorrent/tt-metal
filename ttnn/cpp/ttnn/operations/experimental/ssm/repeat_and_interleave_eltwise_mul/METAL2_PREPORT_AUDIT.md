# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul`

One device operation shares the directory:

- **`RepeatAndInterleaveEltwiseMulDeviceOperation`** (`device/repeat_and_interleave_eltwise_mul_device_operation.{hpp,cpp}`)
  - `RepeatAndInterleaveEltwiseMulProgramFactory` (`device/repeat_and_interleave_eltwise_mul_program_factory.cpp`) — the op's only factory; interleaved / tiled only (`validate_on_program_cache_miss` rejects any non-`INTERLEAVED` memory layout, `device/repeat_and_interleave_eltwise_mul_device_operation.cpp:44-49`)

Kernels, all owned by this op and all referenced by the single factory (no unreferenced kernel files in the directory):

- `device/kernels/reader_ssm_eltwise_mul.cpp` (`ReaderConfigDescriptor`)
- `device/kernels/writer_ssm_eltwise_mul.cpp` (`WriterConfigDescriptor`)
- `device/kernels/ssm_eltwise_mul.cpp` (`ComputeConfigDescriptor`)

The one factory compiles into **two kernel-source configurations** via `defines` derived from input widths (`..._program_factory.cpp:100-106`): `REPEAT_IN0` (set when `ashape[-1] == TILE_WIDTH`) and `REPEAT_INTERLEAVE_IN1` (set when `bshape[-1] == HIDDEN_SIZE`, i.e. 5120). These are `#ifdef` variants of one factory, **not** separate factories — see *Result* for why they do not constitute a portable subset.

*Negative pointer for a future porter:* there is **no** copy of this op under `ttnn/cpp/ttnn/operations/experimental/quasar/` (checked). Nothing in that tree is a source for this port.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `4386dc456a1 2026-07-29 docs(metal_2.0): require an explicit opt_level when porting compute kernels` — pinned from the doc-branch checkout at `/localdev/edwinlee/Port_Recipe`. The op's own checkout (`/localdev/edwinlee/Port_Rpt_Eltwise_Mul`) carries no `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` tree, so the provenance command prints nothing there. The audit ran against `/localdev/edwinlee/metal2_audit.md`, which is **byte-identical** (`diff -q`) to `ai/audit/metal2_audit.md` at that pinned revision.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul` |
| **Overall** | **RED** — blocked on the Device 2.0 prerequisite |
| **DOps / Factories** | `RepeatAndInterleaveEltwiseMulDeviceOperation` → `RepeatAndInterleaveEltwiseMulProgramFactory` (only) |
| *Prereqs* — Device 2.0 (every kernel used) | **No** (**RED** → Device 2.0 track). 5 retained legacy NoC primitives in `reader_ssm_eltwise_mul.cpp`; **isolated**, not broad Device 1.0 — the kernel is otherwise structurally Device 2.0 (`Noc`, `CircularBuffer`, `TensorAccessor`) |
| *Prereqs* — Cross-op escapes | Ok (gate-scope check only: all 3 kernels are op-owned; every `#include` resolves to `tt_metal/hw/inc/api/*`, donor class 1 — no concern). Full inventory skipped, see *Subjects not run* |
| *Feature Support* — overall | **GREEN** (all Appendix A entries `N/A`) |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (single factory row; cross-check clean) |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No (sheet also records `Formerly custom hashed? = yes` — historical, no hook in the code today) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (sheet `Porting Target`; the *TTNN porting shape* subject itself was skipped — see *Subjects not run*) |
| *Port work* — Offset base pointer | none (**GREEN** — no `->address()` anywhere; all three pointer args are clean `Buffer*` bindings) |
| *Port work* — Tensor bindings (per binding) | not produced (informational subject skipped — see *Subjects not run*) |
| *Port work* — TensorParameter relaxation | sheet says `none`; subject skipped |
| *Port work* — TensorAccessor 3rd arg | none (**GREEN** — no accessor passes a 3rd argument) |
| *Port work* — CB endpoints | deferred (subject skipped — re-census after the Device 2.0 migration lands) |

**CB endpoints** are dispositions, not gates. This op's census was **not** produced: the op is a whole-op RED with no portable subset, and the Device 2.0 fix rewrites the very kernel region whose CB touchers would be counted (see *Subjects not run*).

## Result

**RED at op level; no portable subset** → blocked on the **Device 2.0 data-movement prerequisite**, routed to the **Device 2.0 migration team**. No brief is issued.

- **Device 2.0** ✗ — `device/kernels/reader_ssm_eltwise_mul.cpp` retains **5 legacy Device 1.0 NoC primitives**: one `get_noc_addr(<local L1 addr>)` producing a precomposed `uint64_t`, and four raw `noc_async_read(uint64_t, uint32_t, uint32_t)` calls fed from it. The Device 2.0 migration guide lists exactly this pair as its *"Legacy API"* async-read form, with `Noc::async_read` + `UnicastEndpoint` as the replacement. The kernel's own in-code comments (`// Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address`) confirm the migration (PR #47583) knowingly left them. Details and routing below.
- **Feature compatibility** ✓ — no `GlobalCircularBuffer`, no `GlobalSemaphore`, no non-zero `address_offset`, no CTA varargs.
- **TTNN factory concept** ✓ — the readiness sheet marks `Is able to port? = yes` for the op's single `descriptor`-concept factory; every cheaply-checkable column agrees with the code.
- **Offset base pointers** ✓ — no host-folded offset reaches a device pointer.
- **TensorAccessor 3rd argument** ✓ — no site passes a page-size 3rd argument.

**Why no portable subset.** The blocking sites all sit inside the reader kernel's `#ifdef REPEAT_INTERLEAVE_IN1` region (`reader_ssm_eltwise_mul.cpp:68-139`), so they compile out when `bshape[-1] == TILE_WIDTH * HIDDEN_SIZE`. That is **not** a portable subset: `REPEAT_INTERLEAVE_IN1` is a `defines` variant of the *same* single factory and the *same* kernel source file (`..._program_factory.cpp:100-113, 178-186`), selected per-invocation by input shape. There is no factory, no code path, and no configuration the port could take that does not carry this kernel file. A port would have to rewrite the file, and the port recipe's kernel-side whitelist forbids touching Device 2.0 idioms. The `#ifdef` confinement is therefore useful only for **sizing** the Device 2.0 fix (all 5 sites are in one region of one kernel), not for scoping a partial port.

**Path forward.** This is an op-readiness prerequisite, not a missing Metal 2.0 feature — the narrowest kind of RED. Everything else this audit gates on is already clear, so once the 5 sites are migrated on the Device 2.0 track the op should re-audit cheaply and, on the evidence here, come back GREEN.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** From the live *TTNN Operations analysis* sheet (Drive id `1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`, owner `dgomez@tenstorrent.com`, modified `2026-07-31T13:48:57Z`), fetched fresh this run as CSV. The op has exactly **one** row:

  | Column | Value |
  |---|---|
  | `Op` | `experimental/ssm/repeat_and_interleave_eltwise_mul` |
  | `Device operation` | `RepeatAndInterleaveEltwiseMulDeviceOperation` |
  | `Factory (variant)` | `RepeatAndInterleaveEltwiseMulProgramFactory` |
  | `Concept` | `descriptor` |
  | `Op Classification` | `PD Op (pointer-patching)` |
  | `Execution Model` | `SPMD` |
  | `Porting Target` | `ProgramSpecFactoryConcept` |
  | `Custom hash (compute_program_hash)` | `no` |
  | `Backdoor custom hash (attribute_values / to_hash)` | `no` |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` |
  | `Override runtime args method? (PD only)` | `no` |
  | `Pybind descriptor (nb::class_ of device op)` | `no` |
  | `Smuggled pointer (raw buffer addr in RTA/CRTA)` | `no` |
  | `Known op issues` | *(blank)* |
  | `Is safe to port?` | `yes` |
  | **`Is able to port?`** | **`yes`** |
  | `TensorParameter relaxation` | `none` |
  | `Op-owned tensors?` | *(blank)* |
  | `Secretly SPMD Workload?` | *(blank)* |
  | `Formerly custom hashed?` | `yes` |

  `Op Classification = PD Op (pointer-patching)` matches what the factory does: it pushes `Buffer*` objects into `emplace_runtime_args` (`..._program_factory.cpp:222-246`), which the framework auto-registers as `BufferBinding`s and patches on cache hits.

  Cross-check (code side, per the trust-but-verify rule):
  - `Concept == descriptor` — confirmed: `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` (`..._program_factory.hpp:15-16`, defined `..._program_factory.cpp:24-25`, returning `ProgramDescriptor desc` at `:118, :259`). No mesh-workload return; no `create()` + `override_runtime_arguments()` legacy pair.
  - `Custom hash == no` — confirmed: no `compute_program_hash` (nor any renamed variant) in `..._device_operation.{hpp,cpp}`; the device-op declares only `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors` (`..._device_operation.hpp:28-32`).
  - `Runtime-args update (get_dynamic_runtime_args) == no` — confirmed: no such hook anywhere in the op directory.
  - `Override runtime args method? == no` — confirmed: no `override_runtime_arguments` anywhere in the op directory.
  - `Pybind descriptor == no` — confirmed: `repeat_and_interleave_eltwise_mul_nanobind.cpp:23-32` binds the plain host function via `ttnn::bind_function<"repeat_and_interleave_eltwise_mul", "ttnn.experimental.">`; there is no `nb::class_` of the device op and no `create_descriptor` binding.
  - `Op-owned tensors` blank — consistent with the `descriptor` concept (a `create_descriptor` factory cannot carry them). No `CBDescriptor` in the factory sets `.buffer` (`..._program_factory.cpp:121-175`), so there are no borrowed-memory or op-owned buffers at all.
  - **Factory-set match** — 1:1. The sheet's single factory row ↔ the code's single `program_factory_t = std::variant<RepeatAndInterleaveEltwiseMulProgramFactory>` (`..._device_operation.hpp:26`). No phantom row, no missing row.
  - **Cross-column invariants** hold: `descriptor` + no `get_dynamic_runtime_args` + no op-owned tensors. No conflict → sheet trusted.

  (`Is safe to port?` was **not** re-derived — that is the sheet owner's expert-judgment axis.)

- **Device 2.0 (every kernel used):** **RED (GATE)** → **Device 2.0 migration team**.

  **Scope of the incompleteness: isolated holdovers, not a broad Device 1.0 migration.** All three kernels are structurally Device 2.0 — `Noc noc` object NoC calls (`noc.async_read(s0, cb_in0, …)`, `noc.async_write(cb_out, s, …)`, `noc.async_read_barrier()`), `CircularBuffer` wrapper objects with method-form FIFO ops, and `TensorAccessor` for all tensor traffic. The violations are **5 sites in one kernel**, all in the `#ifdef REPEAT_INTERLEAVE_IN1` region, all instances of one pattern: a local L1→L1 tile-row copy expressed with a precomposed `uint64_t` NoC address plus the free-function `noc_async_read`.

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `device/kernels/reader_ssm_eltwise_mul.cpp` | 71 | `uint64_t cb_in1_transposed_read_ptr = get_noc_addr(cb_in1_transposed_buf.get_read_ptr());` | `Noc noc` (`:12`), `CircularBuffer cb_in1_transposed_buf` (`:35`) |
  | `device/kernels/reader_ssm_eltwise_mul.cpp` | 90 | `noc_async_read(cb_in1_transposed_read_ptr, cb_in1_bcast_row_write_ptr, bfloat16_one_row_in_face_bytes);` | `Noc noc` (`:12`), `CircularBuffer cb_in1_bcast_row_buf` (`:36`) |
  | `device/kernels/reader_ssm_eltwise_mul.cpp` | 92-95 | `noc_async_read(cb_in1_transposed_read_ptr + bfloat16_one_face_bytes, cb_in1_bcast_row_write_ptr + bfloat16_one_face_bytes, bfloat16_one_row_in_face_bytes);` | same |
  | `device/kernels/reader_ssm_eltwise_mul.cpp` | 122 | `noc_async_read(cb_in1_transposed_read_ptr, cb_in1_bcast_row_write_ptr, bfloat16_one_row_in_face_bytes);` | same |
  | `device/kernels/reader_ssm_eltwise_mul.cpp` | 124-127 | `noc_async_read(cb_in1_transposed_read_ptr + bfloat16_one_face_bytes, cb_in1_bcast_row_write_ptr + bfloat16_one_face_bytes, bfloat16_one_row_in_face_bytes);` | same |

  **Basis for the call.** The Device 2.0 migration guide (`docs/source/tt-metalium/tt_metal/apis/kernel_apis/data_movement/device_api_migration_guide.md:105-130`) prints this exact pair as its *"Legacy API"* async-read form — `uint64_t src_noc_addr = get_noc_addr(...); noc_async_read(src_noc_addr, dst_l1_addr, size_bytes);` — against a *"New API"* of `Noc::async_read` with a `UnicastEndpoint` source. Neither `get_noc_addr` nor free-function `noc_async_read` is on the audit's sanctioned-free-function list (only `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)` are). The kernel additionally self-documents the gap at `:70`, `:89`, `:91`, `:121`, `:123`: `// Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address` — added by the Device 2.0 migration itself (`git blame` → `20ace6139ed` *"[Cleanup] Migrate eltwise/conv experimental ops to Device 2.0 API (#47583)"*; this op was also touched by `7ff84036118` *"[Cleanup] Migrate experimental SSM / reduction / CNN ops to Device 2.0 API (#49334)"*, which left them in place).

  **Sizing help for the Device 2.0 team.** A replacement exists today and is per-site local, but it is **not** a one-line wrapper-method swap (so it is not the recipe's "isolated CB-index holdover" shape either — see *Recipe notes*). The source here is local L1 read through a `CircularBuffer`, and a plain `CircularBuffer` cannot serve as an `async_read` **source**: `noc_traits_t<CircularBuffer>::src_addr` `static_assert`s `address_type == LOCAL_L1` (`tt_metal/hw/inc/api/dataflow/circular_buffer.h:186-190`) while `Noc::async_read` requests `AddressType::NOC` for its source (`tt_metal/hw/inc/api/dataflow/noc.h:171-176`). So the guide's `UnicastEndpoint` form is the fit — source args `{.noc_x = my_x[noc_index], .noc_y = my_y[noc_index], .addr = cb_in1_transposed_buf.get_read_ptr() + <running offset>}`, destination the `cb_in1_bcast_row_buf` object with `{.offset_bytes = …}` (`noc_traits_t<CircularBuffer>::dst_args_type` carries `offset_bytes`, same header `:175-177`) — which also lets the running `cb_in1_transposed_read_ptr += …` cursor become an offset expression rather than a raw address.

  *Note for the re-audit (not a finding):* the memory these 5 sites move is `cb_in1_transposed` (`c_25`) → `cb_in1_bcast_row` (`c_26`), and `c_25` is FIFO-**produced by the compute kernel** (`ssm_eltwise_mul.cpp:94-99`) and FIFO-**consumed by the reader** (`reader_ssm_eltwise_mul.cpp:69, 137`) — a compute→DM direction. Whatever shape the Device 2.0 fix takes will change the toucher/role picture for `c_25`/`c_26`, which is the concrete reason the CB-endpoint census is deferred rather than produced now.

- **Feature compatibility:** every Appendix A entry, in order. No entry's recognition signals fire.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` / `CreateGlobalCircularBuffer` / `global_circular_buffer.hpp` / `remote_index` / `remote_cb_*` anywhere in the op directory. The 7 `CBDescriptor`s (`..._program_factory.cpp:121-175`) set only `total_size`, `core_ranges`, `format_descriptors`; the `.global_circular_buffer` field is never named. No factory signature takes a `std::optional<const GlobalCircularBuffer>&`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | `address_offset` is never named in the op (default 0 on all 7 `CBDescriptor`s). No `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. No `CBDescriptor` sets `.buffer` at all, so the borrowed-memory shape this field usually accompanies is absent too. |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` / `CreateGlobalSemaphore` / `global_semaphore.hpp`. The op uses **no semaphores of any kind** (`desc.semaphores` is never populated; kernels contain no `Semaphore`). |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Op-level cue absent: `tensor_args_t` = `RepeatMulInputs` is a fixed set — `const Tensor& a`, `const Tensor& b`, `std::optional<Tensor> preallocated_output` (`..._device_operation_types.hpp:20-24`); no variable-count container. Kernel-level decider absent: every CTA read is at a literal constexpr index — reader `get_compile_time_arg_val(0..3)` (`:22-25`), writer `(0)` (`:19`), compute `(0..6)` (`:16-22`) — and the accessor arg blocks use constexpr offsets `TensorAccessorArgs<4>()` and `TensorAccessorArgs<src0_args.next_compile_time_args_offset()>()` (`reader:27-31`), `TensorAccessorArgs<1>()` (`writer:24`). No CTA is read at a runtime-varying index. |

- **CB endpoints (GATE-free):** **not produced** — deferred. Nothing here can block a Gen1 port; the subject is porter-only, and this op is a whole-op RED with no portable subset, so the census is deferred to the re-audit (see *Subjects not run*, and the re-audit note under the Device 2.0 gate for why the counts will move).

- **Offset base pointers:** **GREEN.** No address RTA folds a host-side offset into a base — in fact the op never calls `->address()` at all. The factory delivers all three tensor bases as **`Buffer*` bindings**: `reader_kernel_desc.emplace_runtime_args(cores[i], {src0_buffer, src1_buffer, …})` (`..._program_factory.cpp:222-230`) and `writer_kernel_desc.emplace_runtime_args(cores[i], {out_buffer, …})` (`:240-246`), where `src0_buffer`/`src1_buffer`/`out_buffer` are the raw `a.buffer()`/`b.buffer()`/`output.buffer()` pointers taken at `:33-36` with **no arithmetic** applied anywhere between. The framework registers them as `BufferBinding`s and supplies the clean base. Kernel side, each arrives as a plain `uint32_t` (`reader:14-15`, `writer:13`) and is used only as a `TensorAccessor` base (`reader:28, 31`, `writer:25`) — no `base + offset` reconstruction. This op appears in **neither** Type-1 nor Type-2 table of `2026-07-19_offset_base_pointers.md`; "no fold, op not in the tables" → clean. No Type 3 (`address_offset` never used), no Type 4 (`ttnn::narrow` / interior-base `MeshBuffer::create` absent).

- **TensorAccessor 3rd argument:** **GREEN.** The op has **three** `TensorAccessor` constructions and none passes a page size: `TensorAccessor(src0_args, src0_addr)` (`reader:28`), `TensorAccessor(src1_args, src1_addr)` (`reader:31`), `TensorAccessor(dst_args, dst_addr)` (`writer:25`) — two arguments each. Nothing to classify, so no Class 1/2 drop and no Class 3/4/Special gate. Consistent with the dated triage `2026-07-06_tensor_accessor_3rd_arg_triage.md`, which lists no `ssm` op; the repo-wide cleanup `69eb869a7bf` *"[Refactor]: Remove redundant page_size arg from TensorAccessor"* already touched this op's history, so any historical site is gone.

## Port-work summary  *(mirrors the brief)*

**Not produced** — RED, no brief. The four informational subjects that populate this section (tensor bindings, relaxations, 3rd-arg drops, CB endpoints) were skipped per the audit's Red-outcome scoping rule; see *Subjects not run*. The two gate-bearing subjects that can also yield port work both came back with **nothing to do**: no offset-base fix, and no 3rd-arg drop.

## Heads-ups  *(mirrors the brief)*

**Not produced** — RED, no brief. See *Subjects not run*.

## Subjects not run  *(Red-outcome scoping disclosure)*

Verdict is a **whole-op RED with no portable subset**, so per the audit's Red-outcome scoping rule the seven purely-informational subjects were skipped. None of these is a clean result:

| Subject | Disclosure |
|---|---|
| TTNN porting shape | skipped — whole-op RED, no portable subset; re-audit on unblock. *(Sheet's `Porting Target` is recorded above as raw sheet data; the subject's own derivation/confirmation was not run.)* |
| TensorParameter relaxations | skipped — whole-op RED, no portable subset; re-audit on unblock. *(Sheet says `none`; no custom hash exists to confirm against, and the sheet's `Formerly custom hashed? = yes` is historical.)* |
| TensorParameter analysis | skipped — whole-op RED, no portable subset; re-audit on unblock. *(Not needed for the two gates that share its scan: the [Offset base pointers](#gate-detail) result above establishes all three pointer args are clean `Buffer*` bases.)* |
| CB endpoints | skipped — whole-op RED, no portable subset; re-audit on unblock. Additionally motivated: the Device 2.0 fix rewrites the `c_25`/`c_26` access sites, so a census taken now would be re-done. |
| RTA varargs | skipped — whole-op RED, no portable subset; re-audit on unblock. |
| Out-of-directory coupling | skipped — whole-op RED, no portable subset; re-audit on unblock. Only the gate-scoping slice was done (all 3 kernels op-owned, includes all `tt_metal/hw/inc/api/*`), so the Device 2.0 gate above is known to have **no donor component** — every violation is in this op's own kernel. No `_metal2` fork exists beside any of the three kernels, and no other op or test references them. |
| Incidental anomalies | not scanned (standing opportunistic instruction). Two items noticed while working the gates are recorded below. |

## Team-only

- **Out-of-directory coupling & donor shape:** full inventory skipped (above). Gate-relevant slice: no borrowed kernel files, no cross-family or in-family donors, no shared-pool kernels — this op instantiates only its own three kernel files by path (`..._program_factory.cpp:179-203`), and no other op or test instantiates them.
- **Relaxation candidates** (mined from a custom hash on a gated op): none available — the op has no custom hash to mine. The sheet's `Formerly custom hashed? = yes` says one existed historically; if the relaxation roadmap wants a candidate for this op, the source would be that removed hash in git history, not the current code. **FALLIBLE if pursued — verify before relying on it.**
- **TTNN factory analysis:** sheet-derived facts with `file:line` evidence are in *Gate detail* above. Summary: op-owned tensors **none** (no `CBDescriptor.buffer`, no `WorkloadDescriptor`); MeshWorkload need **none** (plain `descriptor` concept, `Execution Model = SPMD`); pybind `create_descriptor` **none** (plain `bind_function`, `..._nanobind.cpp:23`); other risky pybind **none** (`Is safe to port? = yes`, no `warning`); custom hash **none**; `get_dynamic_runtime_args` **none**; `override_runtime_arguments` **none**. All five gate conjuncts clear.

## Misc anomalies  *(team-only, non-gating; noticed incidentally, not from a scan)*

- **Hardcoded `5120` where the sibling loop uses an RTA** — `device/kernels/reader_ssm_eltwise_mul.cpp:118`: `{.page_id = block_h_id * 5120 + (i * in0_blocks_per_in1_block + tile_row_id), …}`. The structurally identical first-face loop 33 lines earlier (`:85`) computes the same stride from the runtime arg: `block_h_id * in0_num_blocks_w + …`. Inert today — this code compiles only under `#ifndef REPEAT_IN0`, which per `validate_on_program_cache_miss` (`..._device_operation.cpp:73-74`) means `ashape[-1] == TILE_WIDTH * HIDDEN_SIZE`, hence `in0_num_blocks_w == ashape[-1] / TILE_WIDTH == HIDDEN_SIZE == 5120` (`..._program_factory.cpp:230`). It silently mis-addresses the moment `HIDDEN_SIZE` (`..._device_operation_types.hpp:12`) changes or a new width is admitted. Suggested fix for the ops team: use `in0_num_blocks_w`.
- **Config-dead reader RTA** — reader RTA index 6 (`in0_num_blocks_w`, `reader:20`; supplied unconditionally at `..._program_factory.cpp:230`) is read only inside the `#ifndef REPEAT_IN0` branch (`:85`). Under `REPEAT_IN0`, and under any config without `REPEAT_INTERLEAVE_IN1`, it is unused. Harmless, but it is a dead argument in the shipped `REPEAT_IN0` configuration.

## Per-DeviceOperation attribution

Single DeviceOperation, single factory — no attribution split needed. Every finding above applies to `RepeatAndInterleaveEltwiseMulDeviceOperation` / `RepeatAndInterleaveEltwiseMulProgramFactory`.

## Questions for the user

1. **Were the retained legacy NoC primitives a deliberate Device 2.0 carve-out?** The five sites carry migration-authored comments saying so verbatim (`device/kernels/reader_ssm_eltwise_mul.cpp:70, 89, 91, 121, 123`: *"Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address"*), and two separate Device 2.0 cleanup PRs (`20ace6139ed`, `7ff84036118`) passed over this file without removing them. The audit gates conservatively — the migration guide's Legacy/New table and the sanctioned-free-function list both say these are not compliant, so the gate is RED. But if the Device 2.0 team considers a precomposed-`uint64_t` local L1→L1 read a **permanently sanctioned** primitive rather than an unfinished holdover, this op's only blocker dissolves and it should re-audit straight to GREEN. Worth one confirmation from that team before the migration work is scheduled.

## Recipe notes

- **The Device 2.0 sizing taxonomy has no bucket for this shape.** The gate's Red bullet offers *isolated holdovers* — defined narrowly as "**CB-index-keyed** free-function holdovers … where the corresponding Device-2.0 wrapper object is already in scope … *and* a wrapper-method replacement exists — e.g. `get_read_ptr(cb_id)` → `cb_obj.get_read_ptr()`" — or *broad Device 1.0*. This op is neither: the kernel is structurally Device 2.0 (so "broad" overstates it by a lot), but the holdovers are **raw NoC primitives**, not CB-index free functions, and the replacement is a call-shape change (`UnicastEndpoint` + source args) rather than a 1-line method swap. `raw noc_async_read` is listed only as a *broad Device 1.0* cue, which pulled the other way. I reported it as "isolated, non-CB-index" and sized it explicitly; a third bullet in that list — *isolated non-CB-index primitive holdovers: few sites, idioms otherwise intact, replacement exists but is not a 1-line wrapper swap* — would remove the judgment call.
- **The recipe doesn't say what to do with an in-code retention comment.** The Green bullet's sanctioning test is "if Device 2.0 allows the free function, so do we," with a closed list of two. Here the *migration commits themselves* left comments asserting the primitives are deliberately retained. That is evidence about intent that the recipe gives no home to — it is neither the sanctioned list nor a plain oversight. I gated conservatively and raised it as a question, but a line on how to weigh an explicit in-repo retention marker (trust it? still gate? escalate to the Device 2.0 team as I did?) would settle it for the next auditor.
- **CB endpoints' local precondition and the global Red rule point opposite ways.** CB endpoints says to run when "the op's kernels are structurally Device 2.0 — the Device 2.0 gate is GREEN, **or** RED only on isolated CB-index holdovers," and to defer only on *broadly* Device-1.0 ops. Read alone, that says run it here (idioms are intact). The Red-outcome scoping rule at the top says skip all seven informational subjects on a whole-op RED with no portable subset. I followed the global rule (it is stated as governing) and disclosed the skip. Naming which rule wins in that overlap would avoid the next auditor re-litigating it.
- **Sheet column name drifted from the readiness doc.** `ttnn_op_porting_readiness.md` documents the column as **`Override runtime args method? (PD and legacy)`**; the live sheet's header today reads **`Override runtime args method? (PD only)`**. Both the audit recipe and the readiness doc quote the longer name, and the readiness doc states the guarantee that "existing column names never change." No impact here (the value is `no` either way), but the guarantee is technically broken and the two docs should be reconciled with the sheet.
- **The `Is able to port?` derivation omits a column the sheet carries.** The formula lists six conjuncts but says nothing about **`Backdoor custom hash (attribute_values / to_hash)`**, which sits right beside `Custom hash` in the sheet and is `yes` for several ops (e.g. `bernoulli`, `conv/conv2d`, `ccl/reduce_to_root`). Ours is `no`, so nothing turned on it. Still, an auditor cross-checking "custom hash" can't tell whether a `yes` there is meant to gate; one clause either way would help.
