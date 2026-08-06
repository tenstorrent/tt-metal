# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul`

> **Re-audit** (2026-08-05). The previous run of this document was RED on the Device 2.0 prerequisite. `f6a5267fa85` *"[Cleanup] Device 2.0 Migration for Experimental SSM Ops (#52078)"* has since landed and removed all five flagged sites. This is a **full re-audit against the new tree**, not a patch of the old one: every subject was re-run from the code, and the readiness sheet was re-fetched.

One device operation shares the directory:

- **`RepeatAndInterleaveEltwiseMulDeviceOperation`** (`device/repeat_and_interleave_eltwise_mul_device_operation.{hpp,cpp}`)
  - `RepeatAndInterleaveEltwiseMulProgramFactory` (`device/repeat_and_interleave_eltwise_mul_program_factory.cpp`) — the op's only factory; interleaved / tiled only (`validate_on_program_cache_miss` rejects any non-`INTERLEAVED` memory layout, `..._device_operation.cpp:44-49`)

Kernels, all owned by this op and all referenced by the single factory (no unreferenced kernel files in the directory):

- `device/kernels/reader_ssm_eltwise_mul.cpp` (`ReaderConfigDescriptor`)
- `device/kernels/writer_ssm_eltwise_mul.cpp` (`WriterConfigDescriptor`)
- `device/kernels/ssm_eltwise_mul.cpp` (`ComputeConfigDescriptor`)

*Negative pointer for the porter:* there is **no** copy of this op under `ttnn/cpp/ttnn/operations/experimental/quasar/` (checked). Nothing in that tree is a source for this port.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `4386dc456a1 2026-07-29 docs(metal_2.0): require an explicit opt_level when porting compute kernels` — pinned from the doc-branch checkout at `/localdev/edwinlee/Port_Recipe`. The op's own checkout carries no `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` tree, so the provenance command prints nothing there. The audit ran against `/localdev/edwinlee/metal2_audit.md`, verified **byte-identical** (`diff -q`) to `ai/audit/metal2_audit.md` at that pinned revision.

## The three kernel-source configurations

The single factory compiles into **three** reachable configurations, selected per cache-miss by input width through `defines` (`..._program_factory.cpp:100-106`). Several findings below are config-scoped, so the labels are fixed here:

| Label | Defines | Trigger (`a` width × `b` width) |
|---|---|---|
| **Config A** | `REPEAT_IN0` + `REPEAT_INTERLEAVE_IN1` | `a[-1] == 32`, `b[-1] == 5120` |
| **Config B** | `REPEAT_INTERLEAVE_IN1` only | `a[-1] == 32·5120`, `b[-1] == 5120` |
| **Config C** | `REPEAT_IN0` only | `a[-1] == 32`, `b[-1] == 32·5120` |

The fourth combination (neither define) is **unreachable**: it needs `a[-1] == b[-1] == 32·5120`, which `TT_FATAL((ashape[3] != bshape[3]), "Use eltwise mul for same size inputs!")` (`..._device_operation.cpp:72`) rejects. All three reachable configs are exercised in CI — `tests/ttnn/nightly/unit_tests/operations/ssm/test_ssm_repeat_and_interleave_eltwise_mul.py:81-87` parametrizes exactly these three `(in0_W, in1_W)` pairs, and `:95-107` asserts `num_program_cache_entries() == 3`. Treat all three as production paths.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul` |
| **Overall** | **GREEN** — brief issued |
| **DOps / Factories** | `RepeatAndInterleaveEltwiseMulDeviceOperation` → `RepeatAndInterleaveEltwiseMulProgramFactory` (only) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 3 kernels fully on Device 2.0 idioms (was RED last run; `f6a5267fa85` cleared it) |
| *Prereqs* — Cross-op escapes | Ok — no donors, no borrowed kernel files; every `#include` resolves to `tt_metal/hw/inc/api/*` |
| *Feature Support* — overall | **GREEN** (all Appendix A entries `N/A`) |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (single factory row; cross-check clean) |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No (sheet also records `Formerly custom hashed? = yes` — historical; no hook in the code today) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | none (no `->address()` anywhere; all three pointer args are clean `Buffer*` bindings) |
| *Port work* — Tensor bindings (per binding) | `a` Case 1 · `b` Case 1 · `output` Case 1 (all via `TensorAccessor`) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no site passes a 3rd argument) |
| *Port work* — CB endpoints | 3 legal 1:1 · `c_24`/`c_27` self-loop · **`c_25` multi-binding flag** · 4 zero-toucher `(CB, Config C)` entries needing a binding decision |

**CB endpoints** are dispositions, not gates. Every out-of-window CB here has a port-time resolution — see the per-`(CB, config)` census in *Gate detail*. Two entries need the porter's attention rather than a mechanical translation: `c_25`'s double-consumer census, and the four CBs that have zero endpoints under Config C.

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`). All five gate-bearing subjects pass:

- **Device 2.0** ✓ — all three kernels are on Device 2.0 idioms. The five legacy NoC primitives this audit REDed last run are gone: `f6a5267fa85` replaced them with `Noc::async_read` over a `UnicastEndpoint` loopback source and the destination `CircularBuffer` object (`reader_ssm_eltwise_mul.cpp:95-108, 133-146`), exactly the shape the previous audit sized. Verified by re-scan, not by trusting the commit message.
- **Feature compatibility** ✓ — no `GlobalCircularBuffer`, no `GlobalSemaphore`, no non-zero `address_offset`, no CTA varargs.
- **TTNN factory concept** ✓ — the readiness sheet marks `Is able to port? = yes` for the op's single `descriptor`-concept factory; every cheaply-checkable column agrees with the code.
- **Offset base pointers** ✓ — no host-folded offset reaches a device pointer.
- **TensorAccessor 3rd argument** ✓ — no site passes a page-size 3rd argument.

Port work is small and mostly mechanical: three Case-1 tensor bindings, and a CB-endpoint set that is legal except for one multi-binding CB and one config's worth of zero-toucher CBs. The two judgment calls the porter inherits are written up in *Gate detail → CB endpoints* and *Questions*.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Re-fetched this run from the live *TTNN Operations analysis* sheet (Drive id `1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`, owner `dgomez@tenstorrent.com`) as CSV. The op has exactly **one** row:

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
  | `Pointer patching perf issue?` | `suspect perf regression (+ fixed latent bug)` ← **new since the 2026-07-31 fetch** |
  | `Formerly custom hashed?` | `yes` |

  Cross-check (code side, per the trust-but-verify rule) — all confirmed unchanged by the Device 2.0 PR, which touched only the reader kernel:
  - `Concept == descriptor` — `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` (`..._program_factory.hpp:15-16`; definition `..._program_factory.cpp:24-25`, returning `ProgramDescriptor desc` at `:118, :259`). No mesh-workload return, no legacy `create()` + `override_runtime_arguments()` pair.
  - `Custom hash == no` — no `compute_program_hash` (nor a renamed variant) in `..._device_operation.{hpp,cpp}`; the device-op declares only `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors` (`..._device_operation.hpp:28-32`).
  - `Runtime-args update (get_dynamic_runtime_args) == no` — no such hook anywhere in the op directory.
  - `Override runtime args method? == no` — no `override_runtime_arguments` anywhere in the op directory.
  - `Pybind descriptor == no` — `..._nanobind.cpp:23-32` binds the plain host function via `ttnn::bind_function<"repeat_and_interleave_eltwise_mul", "ttnn.experimental.">`; no `nb::class_` of the device op, no `create_descriptor` binding.
  - `Op-owned tensors` blank — consistent with the `descriptor` concept. No `CBDescriptor` sets `.buffer` (`..._program_factory.cpp:121-175`), so there are no borrowed-memory or op-owned buffers at all.
  - **Factory-set match** — 1:1. The sheet's single row ↔ the code's single `program_factory_t = std::variant<RepeatAndInterleaveEltwiseMulProgramFactory>` (`..._device_operation.hpp:26`). No phantom row, no missing row.
  - **Cross-column invariants** hold: `descriptor` + no `get_dynamic_runtime_args` + no op-owned tensors. No conflict → sheet trusted.

  (`Is safe to port?` was **not** re-derived — that is the sheet owner's expert-judgment axis.)

  **`Pointer patching perf issue? = suspect perf regression (+ fixed latent bug)`** is not a gate conjunct and does not affect the verdict, but it is new since the last fetch and is directly relevant to this port: the op is classified `PD Op (pointer-patching)`, and the Metal 2.0 typed binding supersedes exactly that mechanism. Carried to the brief as a watch-for so the porter measures rather than assumes.

- **Device 2.0 (every kernel used):** **GREEN.** No violations. Full re-scan of all three kernels for legacy data-movement idioms (`noc_async_read` / `noc_async_write` / `get_noc_addr` / `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedPow2AddrGen*` / `get_semaphore` / `noc_semaphore_*` / free-function `get_read_ptr(` / `get_write_ptr(` / `get_local_cb_interface` / `cb_*` free functions) returns **zero** hits.

  | Kernel | Idioms | Notes |
  |---|---|---|
  | `device/kernels/reader_ssm_eltwise_mul.cpp` | `Noc noc` (`:13`), `noc.async_read(...)` (`:59, :66, :87, :95, :101, :126, :133, :139`), `noc.async_read_barrier()`, `UnicastEndpoint local_src` (`:41`), `CircularBuffer` objects (`:34-37`) with method FIFO ops, `TensorAccessor` (`:29, :32`) | The former legacy sites now read `noc.async_read(local_src, cb_in1_bcast_row_buf, …, {.noc_x = local_noc_x, .noc_y = local_noc_y, .addr = …}, {.offset_bytes = …})`. `my_x[]`/`my_y[]` at `:42-43` are firmware coordinate globals (Device 2.0's own `Noc::is_local_bank` reads the same arrays, `tt_metal/hw/inc/api/dataflow/noc.h:122`) — not a data-movement holdover. |
  | `device/kernels/writer_ssm_eltwise_mul.cpp` | `Noc noc` (`:11`), `noc.async_write(cb_out, s, …)` (`:33`), `CircularBuffer cb_out` (`:27`), `TensorAccessor` (`:25`) | unchanged by the PR; already compliant |
  | `device/kernels/ssm_eltwise_mul.cpp` | `CircularBuffer` objects (`:34-40`) with method FIFO ops | Compute-LLK calls that take a CB *index* (`transpose_init`, `pack_tile`, `mul_tiles`, `binary_op_init_common`, `reconfig_data_format*`) are compute APIs, not data-movement holdovers — not violations. |

  **Sanctioned free functions present (explicitly not flagged):** `get_tile_size(cb_id)` at `reader:46, :47` and `writer:23`. `cb_in1_transposed_buf.get_read_ptr()` (`reader:78`) is the wrapper **method**, not the CB-index free function.

- **Feature compatibility:** every Appendix A entry, in order. No entry's recognition signals fire.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` / `CreateGlobalCircularBuffer` / `global_circular_buffer.hpp` / `remote_index` / `remote_cb_*` anywhere in the op. The 7 `CBDescriptor`s (`..._program_factory.cpp:121-175`) set only `total_size`, `core_ranges`, `format_descriptors`; the `.global_circular_buffer` field is never named. No factory signature takes a `std::optional<const GlobalCircularBuffer>&`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | `address_offset` is never named (default 0 on all 7 `CBDescriptor`s). No `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. No `CBDescriptor` sets `.buffer`, so the borrowed-memory shape this field usually accompanies is absent too. |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` / `CreateGlobalSemaphore` / `global_semaphore.hpp`. The op uses **no semaphores of any kind** — `desc.semaphores` is never populated and no kernel contains a `Semaphore`. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Op-level cue absent: `tensor_args_t` = `RepeatMulInputs` is fixed — `const Tensor& a`, `const Tensor& b`, `std::optional<Tensor> preallocated_output` (`..._device_operation_types.hpp:20-24`). Kernel-level decider absent: every CTA read is at a literal constexpr index — reader `get_compile_time_arg_val(0..3)` (`:23-26`), writer `(0)` (`:19`), compute `(0..6)` (`:16-22`) — and the accessor arg blocks use constexpr offsets `TensorAccessorArgs<4>()` / `TensorAccessorArgs<src0_args.next_compile_time_args_offset()>()` (`reader:28-32`) and `TensorAccessorArgs<1>()` (`writer:24`). No CTA is read at a runtime-varying index. |

- **CB endpoints (GATE-free):** the op allocates **7 CBs**, all over the same `all_cores` range as all three kernels (`..._program_factory.cpp:121-175`), so every CB is one per-node instance touched by whichever kernels reference it. Census per `(CB, config)`, counting distinct touchers and their role locks:

  | CB | Index | Config A | Config B | Config C | Disposition |
  |---|---|---|---|---|---|
  | `src0` | `c_0` | reader P (`:58,61`) / compute C (`:45,62`) | reader P (`:86,112,125,150`) / compute C (`:106,121`) | reader P (`:58,61`) / compute C (`:45,172`) | **legal 1:1** in all configs |
  | `src1` | `c_1` | reader P (`:65,74`) / compute C (`:70,100`) | same | reader P (`:65,74`) / compute C (`:70,87`) | **legal 1:1** in all configs |
  | `output` | `c_16` | compute P (`:156,161`) / writer C (`:32,40`) | same | compute P (`:81,86`) / writer C (`:32,40`) | **legal 1:1** in all configs |
  | `in0_transposed` | `c_24` | compute only — P (`:56,61`) + C (`:64,170`) | compute only — P (`:115,120`) + C (`:123,142`) | **0 touchers** | **self-loop** (A, B) · see *Config C* below |
  | `in1_transposed` | `c_25` | compute P (`:94,99`) **+ compute C** (`:165`) **+ reader C** (`:77,156`) | same | **0 touchers** | **multi-binding flag** (A, B) · see *Config C* below |
  | `in1_bcast_row` | `c_26` | reader P (`:83,114,122,152`) / compute C (`:126,144`) | same | **0 touchers** | **legal 1:1** (A, B) · see *Config C* below |
  | `out_transposed` | `c_27` | compute only — P (`:135,140`) + C (`:147,162`) | same | **0 touchers** | **self-loop** (A, B) · see *Config C* below |

  (Line numbers are in the owning kernel: `c_0`/`c_1`/`c_26` producer sites in `reader_ssm_eltwise_mul.cpp`, `c_16` consumer sites in `writer_ssm_eltwise_mul.cpp`, everything else in `ssm_eltwise_mul.cpp`.)

  **`c_25` is the one genuine multi-binding.** Its census has **two** distinct touchers but **two locked consumers**, which no relabelling removes:
  - compute is a **locked producer** — `cb_in1_transposed_buf.reserve_back` (`ssm_eltwise_mul.cpp:94`) + `push_back` (`:99`);
  - compute is *also* a **locked consumer** — `cb_in1_transposed_buf.pop_front` (`ssm_eltwise_mul.cpp:165`);
  - the reader is a **locked consumer** — `wait_front` (`reader:77`) + `pop_front` (`reader:156`), with the tile read out through `get_read_ptr()` (`reader:78`).

  Two kernels locked to the consumer role ⇒ the census cannot fit 1P+1C ⇒ **set the DFB multi-binding advanced option** (Config A and B). This is port work, not a gate, and the flag self-documents the Quasar debt. Note the underlying oddity is the producer-side `pop_front` at `ssm_eltwise_mul.cpp:165` (see *Misc anomalies*): if the ops team removes it on their own track, `c_25` collapses to a plain 1:1 and the flag becomes unnecessary — but that is a behavior change, out of port scope.

  **Config C leaves four CBs with zero endpoints.** Under `REPEAT_IN0`-only, every access to `c_24`, `c_25`, `c_26`, `c_27` is compiled out (all sit inside `#ifdef REPEAT_INTERLEAVE_IN1` blocks: `reader:76-158`, `ssm_eltwise_mul.cpp:47-65` and `:88-166`), yet the factory allocates all seven CBs unconditionally. Confirmed positively, per the recipe's distrust-a-`(0,0)` rule: greps for each `buffer_index` and each named CTA across all three kernels, evaluated per config, and cross-checked against the config's own CI test case. What *does* survive in Config C is **not** an access:
  - `CircularBuffer` wrapper constructions — `reader:36-37`, `ssm_eltwise_mul.cpp:37-40` — unconditional, no memory touched;
  - one format-metadata reference — `pack_reconfig_data_format(cb_in0_transposed, cb_id_out)` (`ssm_eltwise_mul.cpp:78`), which reads `c_24`'s data format, not its memory.

  A DFB with no producer and no consumer binding is rejected by the spec validator, so the Config-C spec cannot be emitted as-is. Two resolutions, both zero-functional-change; the auditor's recommendation is (a):

  - **(a) Self-loop them from the constructing kernel — recommended.** Compute already constructs wrappers for all four (and names `c_24` at `:78`); the reader constructs `c_25`/`c_26`. Bind the constructing kernel PRODUCER **and** CONSUMER. On Gen1 the role labels are cosmetic for a kernel that runs no FIFO ops, so runtime behavior and L1 footprint are both identical to legacy, and **no kernel code changes** — it stays inside a pure spec-side decision.
  - **(b) Drop the four DFBs in the Config-C spec.** Matches the recipe's dead-CB rule literally (0 touchers → drop), and is legitimate — a CB with no behavior has no behavior to lose — but it forces kernel-side edits the port would otherwise not make: the four wrapper constructions and the `:78` metadata reference need `#ifdef` guards, since their `dfb::name` tokens would no longer exist. It also changes the Config-C L1 footprint.

  Recorded as PORT WORK with (a) as the default, and raised in *Questions* because the recipe's dead-CB rule assumes op-wide deadness and does not cover a config-scoped zero-toucher whose index is still *named*.

- **Offset base pointers:** **GREEN.** No address RTA folds a host-side offset into its base — the op never calls `->address()` at all. The factory delivers all three tensor bases as **`Buffer*` bindings**: `reader_kernel_desc.emplace_runtime_args(cores[i], {src0_buffer, src1_buffer, …})` (`..._program_factory.cpp:222-230`) and `writer_kernel_desc.emplace_runtime_args(cores[i], {out_buffer, …})` (`:240-246`), where `src0_buffer` / `src1_buffer` / `out_buffer` are the raw `a.buffer()` / `b.buffer()` / `output.buffer()` pointers taken at `:33-36` with **no arithmetic** applied anywhere in between. Kernel side each arrives as a plain `uint32_t` (`reader:15-16`, `writer:13`) used only as a `TensorAccessor` base (`reader:29, :32`, `writer:25`) — no `base + offset` reconstruction. The op appears in **neither** the Type-1 nor the Type-2 table of `2026-07-19_offset_base_pointers.md`; "no fold, op not in the tables" → clean. No Type 3 (`address_offset` unused), no Type 4 (no `ttnn::narrow` / interior-base `MeshBuffer::create`).

- **TensorAccessor 3rd argument:** **GREEN.** Three `TensorAccessor` constructions, none passing a page size: `TensorAccessor(src0_args, src0_addr)` (`reader:29`), `TensorAccessor(src1_args, src1_addr)` (`reader:32`), `TensorAccessor(dst_args, dst_addr)` (`writer:25`) — two arguments each. No Class 1/2 drop and no Class 3/4/Special gate. Consistent with the dated triage `2026-07-06_tensor_accessor_3rd_arg_triage.md`, which lists no `ssm` op; the repo-wide cleanup `69eb869a7bf` *"[Refactor]: Remove redundant page_size arg from TensorAccessor"* already passed over this file.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, all three fed to a `TensorAccessor`):
  - `a` — **Case 1**. Host: `src0_buffer` `Buffer*` RTA (`..._program_factory.cpp:224`); kernel: `src0_addr` (`reader:15`) → `TensorAccessor s0` (`reader:29`).
  - `b` — **Case 1**. Host: `src1_buffer` `Buffer*` RTA (`..._program_factory.cpp:224`); kernel: `src1_addr` (`reader:16`) → `TensorAccessor s1` (`reader:32`).
  - `output` — **Case 1**. Host: `out_buffer` `Buffer*` RTA (`..._program_factory.cpp:242`); kernel: `dst_addr` (`writer:13`) → `TensorAccessor s` (`writer:25`).

  No binding is `clean`/borrowed — the op has no borrowed-memory CBs. No Case 2: no kernel does raw NoC arithmetic on a tensor base. The `Buffer*` delivery form is the framework's interim pointer-patching hack (correct on cache hits today, not the silent-wrong hazard); the port replaces it with typed `TensorParameter` / `TensorBinding`s, and the `TensorAccessorArgs` CTA plumbing (`..._program_factory.cpp:84-85, :89`) disappears with it.
- **TensorParameter relaxation:** none (sheet `TensorParameter relaxation = none`; no custom hash exists to confirm against).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `c_24` and `c_27` (Configs A, B) · set the multi-binding advanced option on `c_25` (Configs A, B) · resolve the four zero-toucher `(c_24, c_25, c_26, c_27) × Config C` entries, recommended via self-loop binding (option (a) above) · no dead-CB drop at op level, and `c_0` / `c_1` / `c_16` are legal 1:1 everywhere.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** `c_25` in Configs A and B — the extra endpoint is *not* a hidden raw writer but a **producer-side `pop_front`** (`ssm_eltwise_mul.cpp:165`) alongside the reader's consumer pop (`reader:156`). Confirm both before setting the flag.
- **Config-scoped census:** the dispositions above flip with config. Config C (`REPEAT_IN0` only) is a live, CI-tested path in which four of the seven CBs are untouched — do not carry Config A's census into it.
- **Cross-op / shared kernels:** none. All three kernels are op-owned, no `_metal2` fork exists beside any of them, and no other op or test instantiates them — so this port creates no fork and carries no sunset list.
- **RTA varargs:** none. Every kernel reads its runtime args as distinct fields at fixed constant indices — reader 7 (`:15-21`), writer 5 (`:13-17`), compute 2 (`:13-14`). No counted loop over `get_arg_val`, no data-selected index. All are nameable; port them as named args.
- **Perf:** the readiness sheet's `Pointer patching perf issue?` cell now reads `suspect perf regression (+ fixed latent bug)` for this op. Measure before/after rather than assuming parity.

## Team-only

- **Out-of-directory coupling & donor shape:** **✓ clean** at op level — no function-call escapes into another op's headers, and no borrowed kernel files.

  | Op kernel | Donor file(s) | Bucket | Status |
  |---|---|---|---|
  | `reader_ssm_eltwise_mul.cpp` | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/dataflow/endpoints.h`, `api/tensor/noc_traits.h` | 1 — `tt_metal/*` | ✓ no concern |
  | `writer_ssm_eltwise_mul.cpp` | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/tensor/noc_traits.h` | 1 — `tt_metal/*` | ✓ no concern |
  | `ssm_eltwise_mul.cpp` | `api/compute/bcast.h`, `api/compute/eltwise_binary.h`, `api/compute/transpose.h`, `api/dataflow/circular_buffer.h` | 1 — `tt_metal/*` | ✓ no concern |

  Per-call detail omitted (all rolls ✓). `api/dataflow/endpoints.h` is new to the reader with `f6a5267fa85`. **Borrowed kernel files:** none — the factory instantiates only its own three files by path (`..._program_factory.cpp:179-203`), no `_metal2` fork exists beside any of them, and no other op or test references them.
- **Relaxation candidates** (mined from a custom hash): none available — the op has no custom hash to mine. The sheet's `Formerly custom hashed? = yes` says one existed historically; if the relaxation roadmap wants a candidate, the source is that removed hash in git history, not the current code. **FALLIBLE if pursued — verify before relying on it.**
- **TTNN factory analysis:** sheet-derived facts with `file:line` evidence are in *Gate detail*. Summary: op-owned tensors **none** (no `CBDescriptor.buffer`, no `WorkloadDescriptor`); MeshWorkload need **none** (plain `descriptor` concept, `Execution Model = SPMD`); pybind `create_descriptor` **none** (plain `bind_function`, `..._nanobind.cpp:23`); other risky pybind **none** (`Is safe to port? = yes`, no `warning`); custom hash **none**; `get_dynamic_runtime_args` **none**; `override_runtime_arguments` **none**. All gate conjuncts clear. Target concept `ProgramSpecFactoryConcept`, no op-owned tensors — matches the sheet's `Porting Target`.

## Misc anomalies  *(team-only, non-gating; route to the ops team, not into the port diff)*

- **`c_25` is popped twice per push — producer-side backpressure is disabled.** Per in1-block iteration the compute kernel pushes one tile (`ssm_eltwise_mul.cpp:94, 99`) and *two* kernels pop it: compute itself (`:165`) and the reader (`reader:156`). Two acks per push means the producer's free-space check can never block, so `c_25`'s two-tile double buffer provides no protection against compute overwriting a tile the reader is still reading. It is masked today by an implicit handshake — compute cannot reach the next push until the reader has produced the `c_26` rows it waits on (`ssm_eltwise_mul.cpp:126`), which requires the reader to have finished with the `c_25` tile. Latent, not live; but it is also what forces the multi-binding disposition above, so removing the redundant `:165` pop would both restore backpressure and simplify the ported spec.
- **Hardcoded `5120` where the sibling loop uses an RTA** — `reader_ssm_eltwise_mul.cpp:130`: `{.page_id = block_h_id * 5120 + (i * in0_blocks_per_in1_block + tile_row_id), …}`. The structurally identical first-face loop 39 lines earlier (`:91`) computes the same stride from the runtime arg: `block_h_id * in0_num_blocks_w + …`. Inert today — this code compiles only under `#ifndef REPEAT_IN0`, which per `..._device_operation.cpp:73-74` means `a[-1] == TILE_WIDTH * HIDDEN_SIZE`, hence `in0_num_blocks_w == 5120` (`..._program_factory.cpp:230`). It mis-addresses the moment `HIDDEN_SIZE` (`..._device_operation_types.hpp:12`) changes or a new width is admitted. Suggested fix: use `in0_num_blocks_w`.
- **Config-dead reader RTA** — reader RTA index 6 (`in0_num_blocks_w`, `reader:21`; supplied unconditionally at `..._program_factory.cpp:230`) is read only inside the `#ifndef REPEAT_IN0` branch (`:91`). It is dead in Config A and Config C — i.e. in two of the three shipped configurations.

## Per-DeviceOperation attribution

Single DeviceOperation, single factory — no attribution split needed. Every finding above applies to `RepeatAndInterleaveEltwiseMulDeviceOperation` / `RepeatAndInterleaveEltwiseMulProgramFactory`. Findings that vary do so by **kernel-source configuration** (A / B / C), not by factory; those are labelled inline.

## Questions for the user

1. **Config-C zero-toucher CBs — self-loop or drop?** Under Config C, `c_24`, `c_25`, `c_26`, `c_27` have zero endpoints while the factory still allocates them (`..._program_factory.cpp:145-175`), and the kernels still construct wrappers for them (`reader:36-37`, `ssm_eltwise_mul.cpp:37-40`) plus one format reference (`ssm_eltwise_mul.cpp:78`). The audit recommends **self-loop binding the constructing kernel** (no kernel edits, identical L1 footprint) over the recipe's literal dead-CB drop (which would force `#ifdef` guards on kernel-side constructions). Please confirm the porter should take the self-loop route — or, if the ops team would rather make the *host-side* allocations config-conditional on their own track, that also resolves it and is arguably the cleaner end state.
2. **Should the redundant `c_25` pop be fixed before the port?** Removing `ssm_eltwise_mul.cpp:165` would drop `c_25` from multi-binding to a plain 1:1 and restore producer backpressure. It is a behavior change, so it is explicitly *not* port work — but if the ops team wants it, sequencing it first makes the ported spec simpler and avoids booking Quasar debt that would immediately be paid off.

## Recipe notes

- **The dead-CB rule has no shape for a *config-scoped* zero-toucher whose index is still named.** [CB endpoints](#cb-endpoints) says to classify per `(CB, config)` **and** says a 0-toucher CB "must be dropped" because the validator rejects a bindingless DFB — but the drop guidance is written for an op-wide dead CB ("positively confirmed its index is unreferenced by every kernel in every config"). This op is the in-between case: four CBs are genuinely live in two configs and untouched in a third, and in that third the index is still *named* by a kernel — by a `CircularBuffer` wrapper construction and by `pack_reconfig_data_format(cb_in0_transposed, cb_id_out)`. The two rules then point at different actions (drop the DFB vs. keep it and bind it), and the cheaper, zero-kernel-change answer — self-loop the constructing kernel — isn't in the table, because the table's self-loop row is defined by "one *toucher*," and a wrapper construction is not an access. Suggest the census gain an explicit **named-but-untouched** category, with the self-loop as its resolution.
- **"Toucher" is defined by memory access, but the Metal 2.0 constraint is by *token*.** The census counts FIFO ops and raw-pointer access; the thing that actually has to exist at port time is a `dfb::name` binding for every DFB a kernel *names*. Format-metadata calls (`pack_reconfig_data_format`, `reconfig_data_format_srca`, `binary_op_init_common`) name CB indices without touching memory, and on Gen1 they are pervasive in compute kernels. Worth one line saying whether such a reference obliges a binding — it decides the previous bullet.
- **A producer that also pops is a multi-binding trigger the "faces" don't picture.** [Multi-binding](#multi-binding-2-of-one-kind-on-a-node) hunts three faces — hidden second *writer*, multiple *readers*, dual-instance work-split — all of which look for an extra toucher. Here the extra locked role comes from a kernel already counted: compute produces `c_25` *and* pops it, so a 2-toucher CB is nonetheless ≥2 locked consumers. The classification table handles it correctly (I applied it), but the prose framing ("most CBs with two touchers are 1P+1C") points the other way, and an auditor who reads the faces before the table could easily land on 1P+1C. A fourth face — *self-popping producer* — would close that.
- **Sheet column name still drifted from the readiness doc.** `ttnn_op_porting_readiness.md` documents `Override runtime args method? (PD and legacy)`; the live sheet's header reads `Override runtime args method? (PD only)` — same as at the 2026-07-31 fetch, so it is stable drift, not a transient edit. No impact here (value `no`), but the doc's "existing column names never change" guarantee is broken and the two should be reconciled.
- **The `Is able to port?` derivation omits `Backdoor custom hash`.** The formula lists six conjuncts and says nothing about `Backdoor custom hash (attribute_values / to_hash)`, which sits beside `Custom hash` and is `yes` for several ops. Ours is `no`, so nothing turned on it; one clause either way would tell the next auditor whether a `yes` there gates.
- **Re-audit after a prereq lands was cheap and worked exactly as designed.** The previous RED named five `file:line` sites and sized the fix (`UnicastEndpoint` + CB destination with `offset_bytes`); the landed PR matches that shape, and this re-audit confirmed it by re-scan. Worth noting in the recipe that on a re-audit the informational subjects genuinely do have to be produced fresh — the Device 2.0 fix here changed the reader from raw `get_write_ptr()` writes into `c_26` to CB-object NoC destinations, which is exactly the kind of change that would have invalidated a CB census taken during the RED run. The deferral rule paid off.
