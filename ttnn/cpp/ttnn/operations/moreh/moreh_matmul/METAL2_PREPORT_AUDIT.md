# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_matmul`

- **`MorehMatmulOperation`**
  - `MultiCoreProgramFactory` (`device/moreh_matmul_program_factory.cpp`) — the op's single program factory.

Kernels referenced by the factory (all op-owned, under `device/kernels/`):
- `reader_moreh_matmul.cpp` (data movement — reads `input`, `other`, optional `bias`)
- `writer_moreh_matmul.cpp` (data movement — writes `output`)
- `moreh_matmul.cpp` (compute — matmul with optional transpose / edge-mask / bias-add; consumes/produces CBs only)

Shared kernel headers pulled in: `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` (reader/writer) and `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` (compute) — the shared moreh kernel pool.

Note (out of device-op scope): the host wrapper `moreh_matmul.cpp` (`ttnn::moreh_matmul`) may route to a *different* op, `moreh_dot`, when both inputs are 1-D 4-D tensors (`is_dot_forward`). That dispatch happens above the device operation; it does not affect this factory's port and `moreh_dot` is a separate op with its own audit.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_matmul` |
| **Overall** | GREEN |
| **DOps / Factories** | `MorehMatmulOperation` → `MultiCoreProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes — all three kernels + shared helpers on Device 2.0 idioms (`Noc`, `DataflowBuffer`, `TensorAccessor`) |
| *Prereqs* — Cross-op escapes | Ok — only the shared moreh kernel pool (`ttnn/kernel/`), Device 2.0 clean |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok (all CTAs read at constexpr offsets; fixed-count `tensor_args_t`) |
| *TTNN Readiness* — `Is able to port?` (the gate) | Yes |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (cleared) |
| *Port work* — Tensor bindings (per binding) | `input` Case 1 · `other` Case 1 · `bias` Case 1 · `output` Case 1 |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no accessor passes a 3rd arg) |
| *Port work* — CB endpoints | legal 1:1 (6 CBs) + self-loop (4 compute intermediates) |

**Tensorless-dispatch check (orchestrator ask):** GREEN — not tensorless. `tensor_args_t` carries `const Tensor& input` and `const Tensor& other` as **required, non-optional** references (`device/moreh_matmul_device_operation.hpp:26-27`). The MetalV2 factory adapter can source the MeshDevice from `input` (and `other`); there is no optional-only / empty-`tensor_args` dispatch path. No framework block.

**CB endpoints** are dispositions, not gates. Every out-of-window CB here has a port-time resolution (self-loop for the single-toucher compute intermediates); nothing blocks the Gen1 port.

## Result

**GREEN → brief issued.** Every gate clears: Device 2.0 ✓ · Feature compatibility ✓ (all Appendix A entries N/A) · TTNN factory concept ✓ (`Is able to port? == yes`, cross-check clean) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓. Target concept `MetalV2FactoryConcept` (plain — no op-owned tensors). Port work is routine: four Case-1 tensor bindings, self-loops on four compute-internal intermediate CBs.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. Readiness sheet row (fetched live this run): `Concept=descriptor`, `Custom hash=no`, `Runtime-args update=no`, `Override runtime args method?=no`, `Pybind descriptor=no`, `Smuggled pointer=no`, `Is safe to port?=yes`, **`Is able to port?=yes`**. Cross-check against code — all consistent:
  - `Concept=descriptor` ✓ — `MultiCoreProgramFactory::create_descriptor(...)` returns `tt::tt_metal::ProgramDescriptor` (`device/moreh_matmul_device_operation.hpp:37-40`).
  - `Custom hash=no` ✓ — no `compute_program_hash` override anywhere in the op directory.
  - `Runtime-args update=no` ✓ — no `get_dynamic_runtime_args` / `override_runtime_arguments`.
  - `Pybind descriptor=no` ✓ — `moreh_matmul_nanobind.cpp` binds the plain free function `ttnn::moreh_matmul` via `ttnn::bind_function`; no `create_descriptor` / device-op `nb::class_`.
  - No cross-column invariant violated (no op-owned tensors on a `descriptor` row; no runtime-args-update on a non-legacy concept issue).
- **Device 2.0 (every kernel used):** GREEN. All three op-owned kernels and the shared helpers they call are structurally Device 2.0:
  - `reader_moreh_matmul.cpp` — `Noc noc; noc.async_read(s0, dfb_in0, ...)`, `DataflowBuffer dfb_in0(...)` with `reserve_back`/`push_back`, `TensorAccessor(input_args, input_addr)`. The only CB-index free functions are `get_tile_size(cb_id)` (lines 120-124) — **sanctioned** (Green bullet), not a holdover.
  - `writer_moreh_matmul.cpp` — `Noc`, `DataflowBuffer dfb_out` with `wait_front`/`pop_front`, `noc.async_write(dfb_out, s, ...)`, `TensorAccessor`. `get_tile_size(cb_id)` (line 25) — sanctioned.
  - `moreh_matmul.cpp` (compute) — `DataflowBuffer` objects throughout (`reserve_back`/`push_back`/`wait_front`/`pop_front`) + LLK compute APIs (`matmul_*`, `transpose_*`, `mul_tiles`, `add_tiles_bcast_*`). No legacy DM idioms.
  - Shared helpers actually called: `ArgFetcher` (wraps `get_arg_val<T>(idx++)` — routine RTA read, not a DM idiom) and `generate_mask_tiles(DataflowBuffer cb_mask, ...)` (`ttnn/kernel/dataflow/moreh_common.hpp:473`, takes a `DataflowBuffer`, uses `cb_mask.get_write_ptr()` / `reserve_back` — Device 2.0 method-based). No `noc_async_read`, `InterleavedAddrGen`, `ShardedAddrGen`, raw sem addresses in any exercised path.
- **Feature compatibility:** every Appendix A entry, in order — all absent.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` type, no `global_circular_buffer` field on any `CBDescriptor`, no remote-CB idiom. `add_cb` lambda builds plain `CBDescriptor`s only (`program_factory.cpp:289-310`). |
  | CBDescriptor `address_offset` (non-zero) | N/A | `add_cb` never sets `.address_offset` (default 0); all `noc.async_read/write` use `{.offset_bytes = 0}`. |
  | GlobalSemaphore | N/A | op uses no semaphores at all. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is a fixed 4-tensor tuple (no `std::vector<Tensor>`). All kernels read CTAs at **constexpr** offsets (`get_compile_time_arg_val(0..N)`, `TensorAccessorArgs<constexpr>`); no runtime-varying CTA index. |

- **CB endpoints (GATE-free):** all legal or self-loop; nothing blocks a Gen1 port. Census per CB, per node (reader on `all_cores`; writer on `all_cores`; compute split as `compute_desc_1` over `core_group_1` and `compute_desc_2` over `core_group_2` — **disjoint** node sets, so each node has exactly one compute instance):

  | CB | Producer | Consumer | Touchers/node | Disposition |
  |---|---|---|---|---|
  | `c_0` (in0) | reader | compute | 2 (1P+1C) | legal 1:1 |
  | `c_1` (in1) | reader | compute | 2 (1P+1C) | legal 1:1 |
  | `c_2` (in2, input mask) | reader (`generate_mask_tiles`) | compute | 2 (1P+1C) | legal 1:1 |
  | `c_3` (in3, other mask) | reader (`generate_mask_tiles`) | compute | 2 (1P+1C) | legal 1:1 |
  | `c_4` (in4, bias) | reader | compute | 2 (1P+1C) | legal 1:1 (FUSE_BIAS only) |
  | `c_24` (im0, matmul reload) | compute | compute | 1 | **self-loop** |
  | `c_25` (im1, input transpose) | compute | compute | 1 | **self-loop** |
  | `c_26` (im2, other transpose) | compute | compute | 1 | **self-loop** |
  | `c_27` (im3, bias-add temp) | compute | compute | 1 | **self-loop** |
  | `c_16` (out0) | compute | writer | 2 (1P+1C) | legal 1:1 |

  No dead CB, no multi-binding (no hidden second writer, no multi-reader, no dual-instance work-split — reader and writer are distinct kernel sources, and the two compute instances cover disjoint cores). The four `c_24`–`c_27` intermediates are touched only by the single compute instance on each node → self-loop (bind compute as both PRODUCER and CONSUMER; legal on Gen1 for compute).

- **Offset base pointers:** GREEN — cleared. Every device address the kernels use is a **clean base**. The factory delivers buffers via the `Buffer*`-binding form: `reader_rt_args.push_back(input_buf)` / `other_buf` / `bias_buf`, `writer_desc.emplace_runtime_args(core, {output_buf, ...})` (`program_factory.cpp:481-499`, `input_buf`/etc. are bare `Buffer*` from `input.buffer()`, no arithmetic). The kernel reads each as `input_addr = arg_fetcher.get_next_arg_val<uint32_t>()` and feeds it straight to `TensorAccessor(input_args, input_addr)` (`reader_moreh_matmul.cpp:52-94`; writer `:14-21`). No `base + offset` fold, no `ttnn::narrow`, no interior base. Not in the offset-base-pointer triage tables (checked `2026-07-19_offset_base_pointers.md`) — and no fold present, so cleanly GREEN.
- **TensorAccessor 3rd argument:** GREEN — N/A. Every `TensorAccessor` construction is 2-arg (`TensorAccessor(input_args, input_addr)`, `TensorAccessor(other_args, other_addr)`, `TensorAccessor(bias_args, bias_addr)`, `TensorAccessor(output_args, output_addr)`); none passes an explicit page-size 3rd argument. Not in `2026-07-06_tensor_accessor_3rd_arg_triage.md` (checked); no 3rd-arg site exists to classify.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding), all via the `Buffer*`-binding delivery form today, all consumed through a `TensorAccessor` → **Case 1**:
  - `input` — Case 1. Reader feeds base into `TensorAccessor(input_args, input_addr)` and does all reads through it (`noc.async_read(s0, dfb_in0, ...)`).
  - `other` — Case 1. `TensorAccessor(other_args, other_addr)`.
  - `bias` — Case 1 (FUSE_BIAS path only). `TensorAccessor(bias_args, bias_addr)`.
  - `output` — Case 1. Writer feeds base into `TensorAccessor(output_args, output_addr)`, `noc.async_write(dfb_out, s, ...)`.
  - Compute kernel touches CB memory only (no tensor memory) → out of scope.
  - Port action: express each as a `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(tensor::name)`; the `Buffer*` RTA + the `TensorAccessorArgs(...).append_to(...)` CTA plumbing (`program_factory.cpp:325-336`) both disappear.
- **TensorParameter relaxation:** none (sheet `TensorParameter relaxation = none`).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `c_24`, `c_25`, `c_26`, `c_27` (single-toucher compute intermediates); all other CBs legal 1:1 (no assignment/flag/drop needed).

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden second writer, no multi-reader, no dual-instance work-split.
- **Cross-op / shared kernels:** the op owns all three kernel `.cpp` files (no borrowed kernel files). The reader/writer/compute `#include` the shared moreh kernel pool headers `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` and `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`. Only `ArgFetcher` and `generate_mask_tiles(DataflowBuffer,...)` are used from them; both are Device 2.0 native. These headers are broadly shared across the moreh family — if a Metal 2.0 rewrite ever touches them, coordinate across moreh ports; the current port needs no change to them (they already take `DataflowBuffer` / `cb_id`).
- **RTA varargs:** none. The reader reads five 8-element arrays (`input_stride`, `other_stride`, `output_stride`, `input_not_bcast`, `other_not_bcast`) and the compute reads one 8-element `output_stride` via an `ArgFetcher` `arg_idx++` run — but the loop bound `MAX_NUM_DIMENSIONS = 8` is a **hardcoded literal**, identical across every instantiation. This is the non-signal "fixed run via a running counter" case → nameable, not a vararg. The porter names these as fixed fields/arrays; no kernel-side vararg mechanism needed. (The tail `bias_addr` read is `#ifdef FUSE_BIAS`-gated — a compile-time-fixed field, also nameable.)

## Team-only

- **Out-of-directory coupling & donor shape:**
  - *Op-level roll-up:* ✓ clean. Only escape is to the shared moreh kernel pool (`ttnn/cpp/ttnn/kernel/`), all Device 2.0.
  - *Summary table:*

    | Op kernel | Donor file | Class | Status |
    |---|---|---|---|
    | `reader_moreh_matmul.cpp`, `writer_moreh_matmul.cpp` | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | shared pool (`ttnn/kernel/`) | ✓ |
    | `moreh_matmul.cpp` (compute) | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | shared pool (`ttnn/kernel/`) | ✓ |

  - *Per-call detail:* `ArgFetcher` (class; `get_arg_val<T>(idx++)`) — routine RTA read. `generate_mask_tiles(DataflowBuffer cb_mask, uint32_t mask_h, uint32_t mask_w, uint32_t single_tile_size=2048)` — takes a `DataflowBuffer`, writes tiles via `cb_mask.get_write_ptr()` / `reserve_back` (Device 2.0 native, cb_id/DFB shape). No `Semaphore`, `CircularBuffer&`, `TensorAccessorArgs<N>`, or old-style addr-gen in any called signature.
  - *Borrowed kernel files:* none — all three kernel `.cpp` files are op-owned under `device/kernels/`.
- **Relaxation candidates (FYI-U):** none — no custom hash to mine.
- **TTNN factory analysis:** `descriptor` concept, no op-owned tensors, no MeshWorkload need, no pybind `create_descriptor`, no custom hash, no custom `override_runtime_arguments`. Target concept `MetalV2FactoryConcept` (plain). All gate conjuncts confirmed absent both on the sheet and in code.

## Misc anomalies  *(team-only, non-gating)*

- `program_factory.cpp:388` allocates `unpack_to_dest_mode` sized `NUM_CIRCULAR_BUFFERS` and only sets index `c_24` under `fp32_dest_acc_en`. Benign (default-filled), noted only for completeness.
- The mask CBs `c_2`/`c_3` and transpose intermediates `c_25`/`c_26` are always allocated even when the corresponding transpose/mask path is inactive (masking/transpose is decided per compile-time flag inside the kernel). This is a small, unconditional L1 allocation, not a correctness issue; the port carries the CBs as-is (the self-loop/1:1 dispositions above hold regardless).

## Questions for the user

None.

## Recipe notes

- **RTA-varargs "CTA-bounded loop is a vararg" vs. a hardcoded literal bound.** The RTA varargs subject says a CTA-bounded loop "still varies across instantiations, so it's a vararg." Here the loop bound is a *hardcoded literal* (`MAX_NUM_DIMENSIONS = 8`), not a CTA — it is identical for every instantiation, so it reads as the non-signal fixed-run case. The recipe's non-signal wording ("a fixed run of reads via a running `arg_index++`") covers it, but the two statements sit close enough that a hurried reader could over-classify a literal-bounded loop as a vararg. Classified as non-vararg (nameable) here.
