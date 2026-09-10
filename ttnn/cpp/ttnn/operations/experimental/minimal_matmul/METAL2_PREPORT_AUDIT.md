# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/minimal_matmul`

One device operation, one program factory:

- **`MinimalMatmulDeviceOperation`**
  - `MinimalMatmulDeviceOperation::ProgramFactory` — `create_descriptor` +
    `override_runtime_arguments`, defined in `device/minimal_matmul_program_descriptor.cpp`
    (declared `device/minimal_matmul_device_operation.hpp:28-42`; the sole alternative of
    `program_factory_t`, `:42`)

Three things in this directory are **not** part of that unit, and all three will mislead a reader
who goes by file listing alone:

- `device/minimal_matmul_program_factory.{cpp,hpp}` declares `MinimalMatmulProgramFactory` and
  `minimal_matmul_factory_helper_common`. Despite the name, this is **no longer this device op's
  program factory** — it is a legacy `Program&`-based emitter kept for the CCL composite
  `experimental/ccl/minimal_matmul_strided_reduce_scatter_async`, which builds one fused
  reduce-scatter + matmul program (`minimal_matmul_program_factory.hpp:13-19` says so explicitly).
  Out of scope as a *factory*; **in scope as the second binder of this op's kernels** — see
  [Shared kernels](#team-only).
- `device/minimal_matmul_fabric_bound_program_factory.{cpp,hpp}` is a free-standing factory
  consumed by `experimental/ccl/strided_all_gather_minimal_matmul_async` and audited with **that**
  op. Its four `fabric_bound_*` kernels are out of scope here.
- The dead `MinimalMatmulSplitDeviceOperation` files flagged by the previous audit are **gone**
  (removed by #55051). Nothing to report.

Three Python entry points share the one factory, so a single port covers all three:
`ttnn.experimental.minimal_matmul` (`minimal_matmul.cpp:39`),
`ttnn.experimental.minimal_matmul_split` (`minimal_matmul_split.cpp:49`), and
`ttnn.experimental.dit_minimal_matmul_addcmul_fused`
(`experimental/transformer/dit_minimal_matmul_addcmul_fused/dit_minimal_matmul_addcmul_fused.cpp:23`).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

**Code audited:** `main` @ `bc119c5338b`.

**Readiness sheet:** fetched 2026-09-10.

**Supersedes:** the 2026-09-01 audit of this op (commit `872ff847fcf`, against `main` @
`beb2ea8f08a`). Every finding below was re-derived from the current code; where a verdict changed,
the change and its cause are named. The three material changes since: the op's
`ProgramDescriptor` migration landed (#55738), the two Device 2.0 CB-index holdovers were fixed
(#55051 / #55072), and the dead split device-op files were deleted (#55051).

## Addendum — 2026-09-10, later the same day: the sheet was refreshed, and the gate clears

The readiness row was updated after this audit was written. **The single blocker is resolved and
the op's verdict is now GREEN.** The audit body below is left as written (it is the record of what
was found at the time); this addendum is the delta.

| column | at audit time | after refresh | effect |
|---|---|---|---|
| `Concept` | `legacy device-op` | **`descriptor`** | conflict resolved — matches the code |
| `Is able to port?` | `yes (with PD step)` | **`yes`** | **the gate clears** |
| `Override runtime args method?` | `n/a` | **`no`** | see below |
| `Porting Target` | `ProgramSpecFactoryConcept` | `ProgramSpecFactoryConcept` | unchanged |
| `TensorParameter relaxation` | `none` | `none` | unchanged |

**Revised result: GREEN.** Every gate-bearing subject now clears — Device 2.0, feature
compatibility, offset base pointers, TensorAccessor 3rd argument, and the TTNN factory concept.

**Two fields are still stale**, and they are the remaining half of the factory-set cross-check.
Both still name the *legacy* CCL-only helper rather than the device op's factory:

- `Factory (variant)` = `MinimalMatmulProgramFactory` — but this device op's `program_factory_t`
  alternative is the nested `MinimalMatmulDeviceOperation::ProgramFactory`
  (`device/minimal_matmul_device_operation.hpp:42`). `MinimalMatmulProgramFactory` still exists as
  a type, but it is the legacy emitter for
  `experimental/ccl/minimal_matmul_strided_reduce_scatter_async`, not this op's factory.
- `Factory definition path` = `device/minimal_matmul_program_factory.hpp` — same problem; the
  factory is declared in `device/minimal_matmul_device_operation.hpp` with its body in
  `device/minimal_matmul_program_descriptor.cpp`.

These are **cosmetic relative to the gate** and do not re-RED the op; recorded for the sheet owner
because reusing the legacy name for this op's row will recreate exactly the confusion this audit
had to untangle.

**On `Override runtime args method? = no`.** At audit time the code *did* declare an
`override_runtime_arguments` on the descriptor factory
(`device/minimal_matmul_program_descriptor.cpp:951`), which by the recipe's rule maps to
`CustomProgramSpecFactoryConcept`. The refreshed cell says `no`, which maps to the base concept —
and the base concept is what the port targeted, on the invoker's decision, after analysis showed
that override refreshed *only* tensor addresses (the set the base concept refreshes automatically).
Sheet and port now agree. The port deleted the override, so the cell is accurate for the ported
code. Rationale and what was traded is in `METAL2_PORT_REPORT.md` → *Concept deviation*.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/minimal_matmul` |
| **Overall** | **RED** — bookkeeping only: the readiness-sheet row is stale. Every code-side gate clears. |
| **DOps / Factories** | `MinimalMatmulDeviceOperation` → `MinimalMatmulDeviceOperation::ProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** for every configuration this factory emits. Residue exists in the shared kernel files but only inside CCL-only `#ifdef`s this factory never defines — see [Gate detail](#gate-detail) and Question 1 |
| *Prereqs* — Cross-op escapes | Ok — one cross-family donor include, unreachable in every in-scope configuration |
| *Feature Support* — overall | GREEN |
| *Feature Support* — GlobalCircularBuffer / `address_offset` / GlobalSemaphore | N/A (none in use, anywhere in the op directory) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **`yes (with PD step)`** — but the row is **spreadsheet-broken** (primary-column conflicts, below) → **GATE** → readiness-sheet owner |
| *TTNN Readiness* — Concept (current) | **`descriptor`** (code). Sheet says `legacy device-op` — **conflict** |
| *TTNN Readiness* — Secretly SPMD | N/A (not `WorkloadDescriptor`) |
| *TTNN Readiness* — Custom hash | No (sheet and code agree — no `compute_program_hash`, no `attribute_values` / `to_hash`) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (sheet and code agree) |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** — `minimal_matmul_program_descriptor.cpp:951`. Sheet says `n/a` — **conflict** |
| *TTNN Readiness* — Pybind `create_descriptor` | No (sheet and code agree) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | **`CustomProgramSpecFactoryConcept`** (from the code's `override_runtime_arguments`). Sheet's `Porting Target` says `ProgramSpecFactoryConcept` — stale, same root cause |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — Offset base pointer | none — GREEN |
| *Port work* — Tensor bindings (per binding) | **Case 1** for all — `in0`, `in1`, `in2`?, `in3`?, `ternary_a`?, `ternary_b`?, `out[0..N_chunks)` |
| *Port work* — TensorAccessor 3rd arg | **Class 2 (drop)** — one site, inert |
| *Port work* — CB endpoints | legal 1:1 ×6 · **self-loop** ×1 (`c_3` intermediate) · **conditional bindings** on `c_2`/`c_4`/`c_5`/`c_6` |

**CB endpoints** are dispositions, not gates. Every CB here is either a plain 1:1 FIFO or the
single-toucher intermediate accumulator; no multi-binding, no dead CB. The port work that *is*
here is the conditional-binding promotion described in [CB endpoints](#gate-detail).

## Result

**RED at op level; no code-side blocker. Blocked on the readiness sheet.**

The op's row in the *"Operations analysis"* sheet still describes minimal_matmul **before** its
`ProgramDescriptor` migration. Three primary columns — the ones the audit is told to cross-check
against the code — now conflict with what the code says, so per the audit's
*spreadsheet-is-broken* rule the gate fails and routes to the **readiness-sheet owner** to
reconcile. This is a **bookkeeping block, not a code block**:

- The sheet's own verdict already reads `Is able to port? = yes (with PD step)`, and the PD step
  **landed** in #55738 (`device/minimal_matmul_program_descriptor.cpp`, 2026-09-05).
- Every other gate — Device 2.0, feature compatibility, offset base pointers, TensorAccessor 3rd
  argument — **clears**.
- Because the block clears **without touching the op's code**, the Red-outcome scoping rule's
  exception applies: all seven purely-informational subjects were **run in full** below, and their
  detail survives intact to the re-audit. Nothing here is deferred.

No brief is issued (the recipe issues one only on an all-green audit). The expectation is that a
refreshed sheet row clears this immediately and the re-audit is a re-read of this same report plus
one cell — see Recipe note 3.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **RED — spreadsheet broken**, routed to the
  readiness-sheet owner. Sheet row: `Op = experimental/minimal_matmul`,
  `Device operation = MinimalMatmulDeviceOperation`,
  `Factory (variant) = MinimalMatmulProgramFactory`, `Diego validation = yes`.

  Lightweight cross-check against the code — **three conflicts**:

  | Column | Sheet | Code | Agrees |
  |---|---|---|---|
  | `Concept` | `legacy device-op` | `create_descriptor` returning `ProgramDescriptor` (`minimal_matmul_device_operation.hpp:29-32`; body `minimal_matmul_program_descriptor.cpp:170`) ⇒ **`descriptor`** | ✗ **conflict** |
  | `Override runtime args method?` | `n/a` | present, on the *descriptor* factory (`minimal_matmul_device_operation.hpp:34-39`; body `minimal_matmul_program_descriptor.cpp:951`) | ✗ **conflict** |
  | Factory-set match | 1 row, named `MinimalMatmulProgramFactory` | `program_factory_t = std::variant<ProgramFactory>` (`minimal_matmul_device_operation.hpp:42`). `MinimalMatmulProgramFactory` still exists but is a **legacy CCL-only helper**, not this device op's factory (`minimal_matmul_program_factory.hpp:13-19`) | ✗ **phantom / mismatched row** |
  | `Custom hash` / `Backdoor custom hash` | `no` / `no` | no `compute_program_hash`, `attribute_values` or `to_hash` anywhere in the op | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | hook absent from the device op | ✓ |
  | `Pybind descriptor` | `no` | no `create_descriptor` binding in either `*_nanobind.cpp` (both bind only the op and the config struct) | ✓ |
  | `Smuggled pointer` | `no` | the descriptor pushes **`Buffer*`** into `KernelDescriptor::RTArgList` (`minimal_matmul_program_descriptor.cpp:834-836, 850-851, 856, 867-868, 882-883, 888`) — the framework's annotated `BufferBinding` form, not the un-annotated PD-migration bug | ✓ |
  | `Op-owned tensors?` | `no` | no `WorkloadDescriptor`, no `buffers` vector | ✓ |
  | `TensorParameter relaxation` | `none` | — (read, not re-derived) | clears |

  Cross-column invariants hold. The `Factory definition path` cell still points at
  `minimal_matmul_program_factory.hpp`; the factory now lives in
  `minimal_matmul_device_operation.hpp` with its body in `minimal_matmul_program_descriptor.cpp`.

  **What to change.** One row refresh: `Concept` → `descriptor`, `Factory (variant)` → the
  device-op's nested `ProgramFactory`, `Override runtime args method?` → `yes`, `Porting Target` →
  `CustomProgramSpecFactoryConcept`, `Factory definition path` → the descriptor file,
  `Is able to port?` → re-evaluate now that the PD step is done.

- **Device 2.0 (every kernel used):** **GREEN for every configuration this factory emits.**

  In-scope kernels are the three the descriptor names —
  `device/kernels/dm_in0_sender.cpp` (instantiated twice: sender + receiver),
  `device/kernels/dm_in1_sender_out.cpp` (twice), `device/kernels/compute.cpp` — plus the
  transitive closure `device/kernels/matmul_dataflow_common.hpp` and the cross-family donor
  `experimental/ccl/strided_all_gather_async/device/kernels/fused_receiver_utils.hpp`.

  **The two isolated CB-index holdovers the previous audit reported are fixed:**

  | File | Was | Now |
  |---|---|---|
  | `device/kernels/dm_in0_sender.cpp` | `get_read_ptr(cb_out_id)` @ `:296` | `cb_out.get_read_ptr()` @ `:296, :321` |
  | `device/kernels/dm_in1_sender_out.cpp` | `get_read_ptr(cb_out_id)` @ `:258` | `cb_out.get_read_ptr()` @ `:258, :283` |
  | `device/kernels/compute.cpp` | `cb_push_back(out_cb, …)` @ `:98` | `CircularBuffer cb_out(out_cb)` @ `:62`, `cb_out.push_back(...)` @ `:102` |

  Every remaining CB access in all four files is the Device 2.0 method form (`CircularBuffer`
  wrapper), every NoC access goes through `Noc` + `TensorAccessor`, and every semaphore is a
  `Semaphore<>` object built from an id. `get_tile_size(cb_id)` appears at four sites
  (`dm_in0_sender.cpp:111-112`, `dm_in1_sender_out.cpp:92-93`) and is **sanctioned** — not a
  violation.

  **Residue exists in the shared kernel files, and it is unreachable here.** Two shapes survive,
  both from the previous audit and both **unchanged**:

  | File | Line | Call | Compiled only under |
  |---|---|---|---|
  | `device/kernels/dm_in0_sender.cpp` | 222, 262-266 | `volatile tt_l1_ptr uint32_t*` cast over `rs_credit_counters_base` + `noc_semaphore_wait_min(&credits[r], …)` | `MM_WINDOW_BLOCKS` |
  | `device/kernels/dm_in1_sender_out.cpp` | 189-190, 225-229 | same | `MM_WINDOW_BLOCKS` |
  | donor `fused_receiver_utils.hpp` (family: `experimental/ccl/strided_all_gather_async`) | 142 | `reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(<id>))` | `FUSE_AG` |
  | donor `fused_receiver_utils.hpp` | 67, 260, 264, 356, 361-362 | raw `tt_l1_ptr` sem pointers + `noc_semaphore_wait_min` on peer-supplied addresses | `FUSE_AG` |

  **`MM_WINDOW_BLOCKS`, `SRS_FUSE_OP_SIGNALER`, `FUSE_AG` and `READ_FROM_LOCAL_INPUT` are defined
  by exactly one emitter, and it is not this factory.** The in-scope
  `create_descriptor` emits only `FUSE_BIAS`, `FUSE_SWIGLU`, `FUSE_TERNARY`,
  `TERNARY_B_IS_FLOAT32`, `IN0_VIRTUAL_CONCAT` / `IN0_K_SPLIT_TILES`, the fused-activation defines
  and the throttle defines (`minimal_matmul_program_descriptor.cpp:496-524, 717-729`). The four
  CCL defines come from the **legacy** helper (`minimal_matmul_program_factory.cpp:442, 445, 451,
  456`) and the fabric-bound factory (`:464, 527, 579`). So in every program this device op
  builds, the offending regions are preprocessed out, the donor's code is never called, and the
  Metal 2.0 binding tokens have Device 2.0 objects to attach to throughout.

  **This is a changed verdict from 2026-09-01, deliberately.** That audit gated on the same sites,
  reasoning from the kernels' *source and include closure*; it recorded the same reachability facts
  in its own "Reachability" paragraph. Two things moved the call: the op's program construction
  moved into `minimal_matmul_program_descriptor.cpp`, which emits none of those defines, and — more
  decisively — routing a Device 2.0 ticket at *minimal_matmul* for code minimal_matmul never
  compiles would misroute the work. The regions belong to the fused-CCL path; they will gate
  `minimal_matmul_strided_reduce_scatter_async` when that op is audited, which is where the fix is
  schedulable. **The recipe does not say whether this gate is per-file or per-compiled-configuration
  — see Recipe note 1 and Question 1**; the two readings give opposite verdicts on this op, and the
  facts above are complete enough for a reviewer to overturn the call cheaply.

- **Feature compatibility:** **GREEN.** Every Appendix A entry scanned against the descriptor
  factory, the in-scope kernels and the donor closure; none fires.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` / `global_cb` / `remote_cb` / `.remote_index(` / `CBDescriptor::global_circular_buffer` anywhere in the op directory |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `address_offset`, `set_address_offset`, `UpdateDynamicCircularBufferAddress`, `cb_descriptor_from_sharded_tensor` or `set_globally_allocated_address`. The seven `CBDescriptor`s are built by one helper (`minimal_matmul_program_descriptor.cpp:414-423`) that sets only `total_size`, `core_ranges` and one `CBFormatDescriptor` |
  | GlobalSemaphore | N/A | six plain `SemaphoreDescriptor`s, ids 0-5 over the full core grid (`minimal_matmul_program_descriptor.cpp:388-408`); none in the donor closure either |

- **CB endpoints (GATE-free):** every CB is legal or self-loops. The census is
  **config-independent** — see below.

  Per node the program places exactly **three** kernel instances: one in0-DM (sender *or*
  receiver — the two core ranges partition the grid, `minimal_matmul_program_descriptor.cpp:377-380`),
  one in1-DM (likewise), and compute (whole grid).

  | CB | Producer | Consumer | Census | Disposition |
  |---|---|---|---|---|
  | `c_0` in0 | in0-DM (`reserve_back`/`push_back`/`get_write_ptr` @ `dm_in0_sender.cpp:357, 359, 403`) | compute (`cb_in0.wait_front`/`pop_front` @ `compute.cpp:491, 516`) | 1 locked P + 1 locked C | **legal 1:1** |
  | `c_1` in1 | in1-DM (`dm_in1_sender_out.cpp:314, 316, 341`) | compute (`compute.cpp:492, 518`) | 1 P + 1 C | **legal 1:1** |
  | `c_2` out | compute (`reserve_back` @ `compute.cpp:530, 542, 554`; `push_back` in `copy_and_pack_block`/`add_bias_block`/`swiglu_block`/`add_bias_and_addcmul_block`) | the **output-writer** DM kernel only (`is_output_writer` CTA) | 1 P + 1 C | **legal 1:1** + conditional binding (below) |
  | `c_3` intermediate | compute | compute | **1 toucher** | **self-loop** (compute self-loop, legal on Gen1) |
  | `c_4` in2/bias | the **non**-output-writer DM (`dm_in0_sender.cpp:436-454` / `dm_in1_sender_out.cpp:380-398`, both `if constexpr (!is_output_writer)`) | compute | 1 P + 1 C | **legal 1:1** + conditional binding |
  | `c_5` ternary_a | non-output-writer DM (`read_ternary_blocks_sync`) | compute (`add_bias_and_addcmul_block`) | 1 P + 1 C | **legal 1:1** + conditional binding |
  | `c_6` ternary_b | non-output-writer DM | compute | 1 P + 1 C | **legal 1:1** + conditional binding |

  **No multi-binding, no dead CB, no hidden second writer.** The hunt was run: there is no raw
  `get_write_ptr`/`fifo_wr_ptr` co-fill by a non-producer, and no CB is read by two co-resident
  kernels. Every `get_write_ptr` / `get_read_ptr` in these files is a *peek by the CB's own single
  producer or consumer*, not a second endpoint. The op is not built on the dual-instance
  work-split shape: the two same-source `KernelDescriptor`s per DM kernel cover **disjoint** node
  sets (sender vs receiver core ranges), so each node sees exactly one — the ordinary 1:1 case, not
  the two-toucher case.

  **Config-independence.** `transpose_core_grid` (`M > N`) swaps *which* DM kernel writes the
  output and which reads bias/ternary, but never changes any CB's census: it is still one producer
  and one consumer on every node. `N_chunks`, `fuse_swiglu` and `IN0_VIRTUAL_CONCAT` change tile
  counts and addressing, not endpoints. `FUSE_BIAS` / `FUSE_TERNARY` toggle whether `c_4` / `c_5` /
  `c_6` exist at all — the conditional-DFB case, handled below rather than as a drop.

  **PORT WORK — conditional bindings (the `Pattern: Conditional / optional resource bindings`
  case).** Four places reference a CB index on a kernel instance (or in a configuration) that never
  touches it. All are inert today — `CircularBuffer(cb_id)` only stores the index
  (`tt_metal/hw/inc/api/dataflow/circular_buffer.h:27`) — and all become **compile errors or
  unnecessary extra bindings** under Metal 2.0, where the `dfb::` token exists only if the host
  binds it:

  1. `dm_in0_sender.cpp:155` / `dm_in1_sender_out.cpp:136` — `CircularBuffer cb_out(cb_out_id)` is
     constructed unconditionally, but every access sits under `if constexpr (is_output_writer)`.
     On the non-writer instance the port must **not** bind `c_2`; promote the `is_output_writer`
     CTA gate to a preprocessor define (the recipe's *Promote a CTA gate to a define* rule) and
     `#ifdef`-gate the construction. Binding it anyway would give `c_2` a third binding on every
     node and force the multi-binding flag for nothing.
  2. `dm_in0_sender.cpp:157` / `dm_in1_sender_out.cpp:138` — same for `cb_in2` (`c_4`), which only
     the **non**-writer instance fills.
  3. `dm_in0_sender.cpp:111-112` / `dm_in1_sender_out.cpp:92-93` — `get_tile_size(cb_ternary_a_id)`
     / `(cb_ternary_b_id)` are evaluated on **both** instances, but the only consumer,
     `read_ternary_blocks_sync`, runs under `if constexpr (!is_output_writer)`
     (`dm_in0_sender.cpp:457`, `dm_in1_sender_out.cpp:401`). Under Metal 2.0 these become
     `dfb::ternary_a.get_tile_size()` (kernel-side whitelist rule 7), so leaving them where they
     are forces the writer instance to bind `c_5`/`c_6` it never touches — again a third binding.
     Sink them into the `!is_output_writer` scope.
  4. `compute.cpp:440` — `CircularBuffer cb_in2(in2_cb)` is constructed **unconditionally**, and
     `in2_cb` (= `c_4`) is passed to `swiglu_block` (`:535`) and `add_bias_and_addcmul_block`
     (`:557`) — and wrapped again at `:157` — regardless of `FUSE_BIAS`. Without bias the host
     allocates no `c_4` at all (`minimal_matmul_program_descriptor.cpp:437-440`), so `dfb::in2`
     will not exist and these references must be `#ifdef FUSE_BIAS`-gated at the preprocessor
     level. `ternary_a_cb` / `ternary_b_cb` at `compute.cpp:433-434` need the same treatment under
     `FUSE_TERNARY`.

- **Offset base pointers:** **GREEN.** Two scan surfaces, both clean.

  On the cache-miss path the factory passes **`Buffer*`, not addresses** — no arithmetic is
  possible (`minimal_matmul_program_descriptor.cpp:763-768, 834-836, 850-851, 856, 867-868,
  882-883, 888`). On the cache-hit path `override_runtime_arguments` resolves every address as a
  bare `buffer()->address()` with nothing folded in: `in0` `:977`, `in1` `:978`, `in2` `:979`,
  `in3` `:980`, `ternary_a`/`ternary_b` `:991-992`, and each of the N outputs `:1009`. No Type 1
  and no Type 2 site; no `ttnn::narrow` / interior-base `MeshBuffer::create` (Type 4) either.
  The op is not in the dated triage's tables and the scan agrees — the *"no fold, not in tables"*
  outcome. Every address reaches TensorParameter analysis as a clean base.

- **TensorAccessor 3rd argument:** **GREEN — Class 2 (redundant / inert), one site.** Port work,
  not a gate.

  `device/kernels/matmul_dataflow_common.hpp:33-34` constructs **every output** accessor with an
  explicit page size; it is the only 3-argument `TensorAccessor` in the op (the nine other
  construction sites across the two DM kernels all use the 2-argument form). The value is
  `out_tile_size`, fed from `dm_in0_sender.cpp:30` / `dm_in1_sender_out.cpp:30` as
  `get_compile_time_arg_val(12)`, whose host value is `tt::tile_size(output_data_format)`
  (`minimal_matmul_program_descriptor.cpp:215`, pushed at `:553, 599, 640, 681`).

  - **Sharded or interleaved?** Either — the output memory config defaults to the activation's
    (`minimal_matmul_device_operation.cpp:297`). The classification does not depend on it here.
  - **Magnitude:** `tt::tile_size(...)` is the true logical page for a TILE-layout buffer and is
    block-float-safe (bf8 → 1088 B, not 1024). Correct magnitude, so inert under *both*
    specializations — the interleaved realignment is not even needed to rescue it.
  - **Verdict:** Class 2. Dropping it in the collapse to `TensorAccessor(tensor::name)` is a pure
    no-op. Do **not** set `dynamic_tensor_shape` — the page size is compile-time-pinned by the
    output dtype, which is part of the program hash.

  The dated triage lists `minimal_matmul` as `2 — Redundant`, `no` (not a bug); the independent
  classification agrees.

## Port-work summary  *(no brief issued — recorded for the re-audit)*

- **Tensor bindings** — **every binding is Case 1** (fed to a `TensorAccessor`; no raw-pointer
  Case 2, no borrowed-memory DFB anywhere in the op):

  | Binding | Present when | Delivery today | Kernel use |
  |---|---|---|---|
  | `in0` (activation) | always | `Buffer*` RTA idx 0 (both in0 kernels) | `TensorAccessor(in0_args, in0_addr)` @ `dm_in0_sender.cpp:68` |
  | `in1` (weight) | always | `Buffer*` RTA idx 0 (both in1 kernels) | `dm_in1_sender_out.cpp:67` |
  | `in2` (bias) | `FUSE_BIAS` | `Buffer*` RTA idx 1 (all four DM kernels) | `dm_in0_sender.cpp:79`, `dm_in1_sender_out.cpp:78` |
  | `in3` (`optional_input_tensor`) | `IN0_VIRTUAL_CONCAT` | `Buffer*` RTA idx 2 (in0 kernels only) | `dm_in0_sender.cpp:195` |
  | `ternary_a` | `FUSE_TERNARY` | `Buffer*` RTA idx 14 / 13 | `dm_in0_sender.cpp:117`, `dm_in1_sender_out.cpp:97` |
  | `ternary_b` | `FUSE_TERNARY` | `Buffer*` RTA idx 15 / 14 | `dm_in0_sender.cpp:118`, `dm_in1_sender_out.cpp:98` |
  | `out[0..N_chunks)` | always (N ≥ 1) | `Buffer*` RTA tail | `make_tensor_accessor_tuple_uniform_page_size` @ `matmul_dataflow_common.hpp:43` |

  All arrive as the **`Buffer*`-binding form** (`KernelDescriptor::RTArgList` /
  `emplace_runtime_args`), which the framework auto-registers as `BufferBinding`s — routine port
  work, *not* the silently-stale-address hazard. An absent optional passes a null `Buffer*`, for
  which the framework emits `0` and registers no binding
  (`minimal_matmul_program_descriptor.cpp:759-768`).

- **TensorParameter relaxation:** `none` — nothing to apply.

- **TensorAccessor 3rd arg:** drop the page-size argument at `matmul_dataflow_common.hpp:33-34`;
  it disappears together with the host-side CTA (index 12) that feeds it. Class 2, no
  `dynamic_tensor_shape`.

- **CB endpoints:** self-loop `c_3` (intermediate, all configs); everything else legal 1:1. Plus
  the four **conditional-binding** promotions listed in [Gate detail](#gate-detail) — this is the
  bulk of the CB-side port work and the part most likely to bite as a compile error rather than a
  design question.

- **Semaphores:** six, all plain. The descriptor hard-codes ids 0-5 as literals and hands them to
  the kernels as CTAs 14-16 (`minimal_matmul_program_descriptor.cpp:388-408, 555-557`); the port
  replaces both halves with named `sem::` bindings. The in0 kernels use ids 0/1/2, the in1 kernels
  3/4/5, so each kernel binds three of the six.

## Heads-ups  *(no brief issued — recorded for the re-audit)*

- **Target concept is `CustomProgramSpecFactoryConcept`, and the override owns the bindings.**
  `override_runtime_arguments` (`minimal_matmul_program_descriptor.cpp:951-1030`) must be
  translated, not deleted, and its return type changed to `ProgramRunArgs` — a `void` return
  silently leaves the factory on the base concept. **What it refreshes today is the complete set
  of tensor bindings and nothing else**: `in0`/`in2`/`in3` on the in0 kernels, `in1`/`in2` on the
  in1 kernels, `ternary_a`/`ternary_b` and all N outputs on both (`:1019-1027`). It deliberately
  refreshes **no** scalar run-args and never touches the compute kernel — everything else derives
  from hashed inputs (`:946-948, 963-966, 906-907`). So the ported override's `tensor_args` carries
  every `TensorParameter` and its `kernel_run_args` stays empty. Reproduce that set exactly.

- **The override exists for dispatch cost, and the port should preserve that intent.** Its own
  comment (`:949-950`) records the measurement: five hoisted `GetRuntimeArgs` grid references vs.
  one lookup per (kernel, core) — 260 here — for **~7% cheaper** dispatch on this op. The bindings
  are declared for the cache miss and then ignored. A Metal 2.0 translation that reverts to
  per-binding patching gives that back; flag it if the declarative `ProgramRunArgs` form cannot
  express the hoist.

- **`ProgramFactory` already exists — the direct-descriptor exception does not apply.** The op
  declares its `create_descriptor` inside a nested `ProgramFactory` struct with
  `program_factory_t = std::variant<ProgramFactory>`
  (`minimal_matmul_device_operation.hpp:28-42`), so the port is a method swap inside the existing
  struct, not the *"give a direct-descriptor op a conventional program factory"* conversion.

- **Shared kernels — this port creates the first `_metal2` forks.** All three kernel sources are
  bound by **two** emitters that will not convert together: the in-scope descriptor factory and
  the legacy `minimal_matmul_factory_helper_common`
  (`minimal_matmul_program_factory.cpp:551, 596, 635, 674, 704` — the *same* file paths). This is
  the **Lent / Intra-op** shape. `ls` of `device/kernels/` shows **no `_metal2` sibling**, so
  rung 2 applies: fork `dm_in0_sender.cpp`, `dm_in1_sender_out.cpp`, `compute.cpp` and
  `matmul_dataflow_common.hpp` beside the originals, convert the copies, leave the originals plus
  the pointer comment. **Sunset list** (*not* authorization to convert in place):
  `experimental/ccl/minimal_matmul_strided_reduce_scatter_async`, via
  `minimal_matmul_factory_helper_common`. See Question 2 — this is the single largest structural
  decision in the port.

- **A variable number of tensor bindings on one KernelSpec.** `N_chunks` (the `chunks` attribute,
  1 for `minimal_matmul`, N for `minimal_matmul_split`) sets the number of output tensors, and
  every DM kernel builds a `TensorAccessor` tuple of exactly that size
  (`matmul_dataflow_common.hpp:42-47`, driven by `make_tensor_accessor_args_tuple<N_chunks, …>`).
  The count is a CTA, so it is fixed per instantiation but varies across them. Resolve early how
  the `KernelSpec` expresses a CTA-sized set of `TensorParameter`s; the recipe does not obviously
  cover it. (Carried forward from the 2026-09-01 audit — still the sharpest open port question.)

- **RTA varargs: none survive the port.** Both DM kernels read a fixed `argidx++` run of 14 (in0)
  / 13 (in1) distinct scalar fields, plus 3 more under `FUSE_TERNARY` — all nameable
  (`dm_in0_sender.cpp:41-62`, `dm_in1_sender_out.cpp:41-61`). Compute reads 4 (+2) fixed fields
  (`compute.cpp:412-425`). The **only** variable-count run is the N output addresses at the tail
  (`out_addr_rt_arg_idx`, read via the pack expansion at `matmul_dataflow_common.hpp:33-34`) — and
  those are buffer addresses, so under Metal 2.0 they leave the runtime-arg channel entirely and
  become the variable-count binding set above. Do **not** port them as an RTA vararg block. No
  CTA-vararg site either: the `TensorAccessorArgs` CTA blocks are read at constexpr offsets and
  disappear wholesale when the accessors are built from `tensor::name`.

- **Two `if constexpr` role gates drive almost every conditional in the DM kernels.**
  `is_output_writer` (CTA 17) and `is_injector_core` (CTA 18) select, respectively, which DM kernel
  drains `c_2` and which one actually reads from DRAM rather than waiting on the mcast relay. The
  port has to promote at least `is_output_writer` to a define (see the conditional-binding item);
  `is_injector_core` gates no CB reference and can stay a CTA.

## Team-only

- **Out-of-directory coupling — full inventory.**

  *Op-level roll-up:* **✓ clean** for every configuration this factory emits. One cross-family
  donor is in the include closure but is never called; no kernel file is borrowed.

  *Function-call escapes, by donor:*

  | Op kernel | Donor file | Class | Functions called | Reachable in-scope? | Shape |
  |---|---|---|---|---|---|
  | `dm_in0_sender.cpp`, `dm_in1_sender_out.cpp`, `compute.cpp` | `tt_metal/hw/inc/api/**` (`dataflow_api.h`, `noc.h`, `noc_semaphore.h`, `endpoints.h`, `circular_buffer.h`, `core_local_mem.h`, `tensor/noc_traits.h`, `compute/*`) | 1 — `tt_metal/*` | framework primitives | yes | no concern |
  | `dm_in0_sender.cpp`, `dm_in1_sender_out.cpp` | `device/kernels/matmul_dataflow_common.hpp` | in-op | `read_in0_block_sync`, `read_in1_block_sync`, `write_block_sync[_split]`, `write_block_sync_granular[_split]`, `read_ternary_blocks_sync` | yes | not an escape — same directory; forks with the kernels |
  | `dm_in0_sender.cpp:14`, `dm_in1_sender_out.cpp:14` | `experimental/ccl/strided_all_gather_async/device/kernels/fused_receiver_utils.hpp` | 6 — cross-family donor | `MinimalMatmulOpReceiver` ctor / `reset()` / `compute_actual_k_block_iter()`; `OpSignaler` ctor / `signal_op_per_core()` | **no** — `FUSE_AG` and `SRS_FUSE_OP_SIGNALER` are never defined by this factory | ⭐ raw `uint32_t` L1 sem addresses (`:67, 142, 260, 264, 356, 361-362`) — the *"`uint32_t sem_addr` (L1)"* row, ✗ not OK, no Metal 2.0 → donor bridge today |
  | (transitively, via the donor) | `ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp`, `ccl/shared_with_host/hetergeneous_data_structs.hpp`, `ccl/ccl_host_types.hpp` | shared CCL pool | `OpSignaler` (defined `worker_sync_utils.hpp:25`) | no | mostly Device 2.0 (`Semaphore<>`); one `noc_semaphore_inc(dst_noc_addr, …)` @ `:125` |

  The donor `#include` is **unconditional** at line 14 of both DM kernels while every call site is
  `#ifdef`-guarded, so the header is parsed but nothing from it is emitted in an in-scope build.
  The fork inherits the include; nothing needs to change about it.

  *Borrowed kernel files (file-path instantiation):* **none.** The op owns all three kernel sources
  it instantiates. The coupling runs the other way — see the *Lent* case in Heads-ups. A tree-wide
  check confirms no factory outside `minimal_matmul/device/` binds these paths;
  `all_gather_minimal_matmul_async` and the `fabric_bound_*` variants each carry their own copies.

- **Relaxation candidates:** none — the op declares no custom hash and no backdoor hash, so there
  is nothing to mine.

- **TTNN factory analysis (sheet-derived facts, with code evidence):** current concept
  `descriptor`; op-owned tensors **no**; MeshWorkload **not** needed; pybound `create_descriptor`
  **absent**; custom hash **absent**; `get_dynamic_runtime_args` **absent**;
  `override_runtime_arguments` **present** at `minimal_matmul_program_descriptor.cpp:951` ⇒ target
  concept `CustomProgramSpecFactoryConcept`. The sheet's stale `Concept` / `Override runtime args
  method?` / `Porting Target` cells are the gate finding above, not separate facts.

## Misc anomalies  *(team-only, non-gating)*

- **Dead CTA — `in3_tile_size`.** The in0 sender/receiver CTA lists are 22 entries long, with
  `in3_tile_size` at index 21 (`minimal_matmul_program_descriptor.cpp:562, 608`; the legacy
  emitter does the same at `minimal_matmul_program_factory.cpp:532, 583`). `dm_in0_sender.cpp`
  reads compile-time args 0-20 and then starts the accessor block at `TensorAccessorArgs<22>()`
  (`:67`) — **index 21 is never read.** The in3 tiles are read with `in0_tile_size` instead
  (`:367-371`), which is safe only because validation forces the two dtypes equal
  (`minimal_matmul_device_operation.cpp:96-100`). Either wire it up or drop it; leaving a silently
  skipped CTA slot between the named args and the accessor block is a trap for the next person who
  edits either end.

- **Dead RTA — `max_defer_write_k_block`.** Read unconditionally at `dm_in0_sender.cpp:55` and
  `dm_in1_sender_out.cpp:54`, and pushed by the factory for every core
  (`minimal_matmul_program_descriptor.cpp:847, 879` — and it costs a whole grid sweep at `:753-757`
  to compute), but its only use in either kernel is inside `#ifdef SRS_FUSE_OP_SIGNALER`
  (`dm_in0_sender.cpp:429`, `dm_in1_sender_out.cpp:373`). In every configuration this device op
  builds it is dead weight in the RTA layout — and it is one of the fourteen slots the layout
  constants at `minimal_matmul_program_descriptor.cpp:151-160` are hard-coded around.

- **Inaccurate rationale on the page-size override** *(carried over — unchanged since 2026-09-01)*.
  `device/kernels/matmul_dataflow_common.hpp:31-32`: the comment justifying the `TensorAccessor`
  3rd argument says the page size comes "from runtime args" and guards a value that "may be stale
  on program cache hits". Both halves are wrong: the value is `out_tile_size`, a **compile-time**
  arg (`get_compile_time_arg_val(12)`), so it is baked per program and cannot go stale any
  differently than `TensorAccessorArgs::AlignedPageSize` can. (The *address* on the neighbouring
  line does come from a runtime arg — likely where the confusion started.) Does not change the
  Class 2 verdict, but a reader could take the comment as a reason to preserve the argument
  through the port.

- **Two hand-synchronized emitters of the same program.**
  `minimal_matmul_program_descriptor.cpp` and `minimal_matmul_program_factory.cpp` now build the
  same matmul — same kernels, same CB set, same CTA layouts, same semaphore ids — from two
  independent code paths, and nothing enforces that they agree. The descriptor file's header
  comment names the relationship but the coupling is silent: the CTA orders at
  `minimal_matmul_program_descriptor.cpp:540-563` and `minimal_matmul_program_factory.cpp:510-532`
  must stay index-for-index identical or the shared kernels misread their arguments in one of the
  two callers. The descriptor factory does guard its **RTA** layout against drift
  (`:921-941`); there is no equivalent guard on the CTA layout, on the CB set, or between the two
  emitters. This lasts until the fused CCL ops migrate; it is worth an explicit cross-reference
  comment at both CTA-list sites at minimum.

- **`determine_default_block_sizes` ignores `K`** (`minimal_matmul_program_descriptor.cpp:41-43`,
  `(void)K;` with a `TODO`), and the `M`/`N` arguments only affect `subblock_h`/`subblock_w`, and
  only when `fp32_dest_acc_en` is false. The three block sizes are unconditional literal `8`s.
  Fine as a default, but the signature promises a computation it does not perform.

## Questions for the user

1. **Is the Device 2.0 gate per-file or per-compiled-configuration?** This audit answers
   *per-compiled-configuration* and clears the gate; the 2026-09-01 audit answered *per-file* and
   RED'd it. The facts are not in dispute — the raw-L1-semaphore sites in `dm_in0_sender.cpp` /
   `dm_in1_sender_out.cpp` and the donor `fused_receiver_utils.hpp` are real, unchanged, and
   compiled **only** under `MM_WINDOW_BLOCKS` / `FUSE_AG` / `SRS_FUSE_OP_SIGNALER`, which this
   device op's factory never defines. What is in dispute is whether an unreachable region in a
   *shared* kernel file blocks the op that shares it. If you want the conservative reading, flip
   this gate to RED and route the four sites to the Device 2.0 team as before; the rest of this
   report is unaffected either way, because the block would clear on the op-code side and the
   informational detail here would then be re-derived at the re-audit.

2. **Fork the three kernels, or port `minimal_matmul_strided_reduce_scatter_async` in the same
   change?** Rung 2 of the shared-kernel caution says fork, and that is the default. But it puts a
   **third** copy of this matmul's kernels in the tree — the legacy trio, the `fabric_bound_*`
   trio, and the new `_metal2` trio, roughly 1,600 lines each time — in an op that is already
   carrying a duplicated *host* emitter for the same reason (see Misc anomalies). The alternative,
   a bundled port that converts SRS at the same time, is blocked today: SRS is
   `Concept = legacy (MeshWorkload)` with `Is able to port? = no` on the sheet, so it cannot
   co-migrate. Worth deciding deliberately rather than by default, and worth pairing with a
   sunset plan, because the drift-discipline cost here is real: the fork and the legacy copy will
   both be actively edited.

3. **Does the readiness-sheet refresh need anything from this audit?** The cross-check evidence in
   [Gate detail](#gate-detail) is the complete list of cells that moved. If the sheet owner wants a
   `Factory (variant)` name for the new row, the code offers only the nested
   `MinimalMatmulDeviceOperation::ProgramFactory`; the old `MinimalMatmulProgramFactory` name is
   now taken by the CCL-only helper and reusing it will re-create this confusion in six months.

## Recipe notes

1. **The Device 2.0 gate does not say whether it is per-file or per-compiled-configuration, and
   this op turns on the distinction.** The gate reads *"Confirm **every kernel this op exercises**
   is Device 2.0 compliant … What matters is whether the op's program factory instantiates or calls
   into the kernel."* Both halves are satisfiable in opposite directions here: the factory
   *instantiates* a file whose `#ifdef`-guarded regions are not compliant, and *calls into* none of
   them. The recipe is explicitly per-instantiation elsewhere (*"Classify per instantiation, not
   once for the op"*, and the conditional-DFB rule for a CB dead in only some configs), which is
   what tipped this audit — but that discipline is stated in the CB-endpoints subject, not in the
   gate. A sentence in the Device 2.0 bullet would settle it. Suggested shape: the gate applies to
   the code the op's own factory *compiles*, with any residue in a shared file recorded as
   coupling and routed to the ops that do compile it.

2. **The recipe still contradicts itself on `TensorParameter relaxations`** *(raised 2026-09-01,
   note 1; unchanged)*. The Red-outcome scoping rule lists it among the *seven purely-informational*
   subjects to skip on a whole-op RED (Feasibility audit intro), while the subject's own section
   ends *"Finding role: **GATE** (routed to the ops team)"* and the finding-roles table treats the
   relaxation column as a gate conjunct. Moot here again (the cell is `none`), but a future auditor
   hitting a non-`none` value on a RED op still gets opposite instructions from the two places.

3. **A "spreadsheet-broken" RED and a code RED are not the same kind of block, and the brief rule
   treats them alike.** The four spreadsheet-broken triggers rest on evidence the auditor holds
   independently — which is exactly why such a RED is the *cheapest* kind to clear: one row edit by
   the sheet owner, with the op's code untouched. Yet *"On any **RED** there is no brief"* withholds
   the porter document until a full re-audit is run against identical code. That is the same
   reasoning the Red-outcome scoping rule already accepts for the informational subjects (*"clear
   with the op untouched, so re-audit reads the same code and today's detail survives intact —
   deferring costs a second full pass and saves nothing"*), and the brief has the same property.
   Suggest either allowing a brief marked *provisional — pending sheet reconciliation*, or saying
   explicitly that a sheet-only RED still withholds it and why.

4. **The recipe has no rung for a shared *host* emitter — and the resolution the ecosystem picked
   is duplication.** The 2026-09-01 audit raised this (note 2): `minimal_matmul_factory_helper_common`
   was a host-side analogue of a shared kernel, with a CCL consumer coupled to its C++ type and
   entry point, and the fork convention did not obviously transfer. What actually happened is worth
   feeding back: the `ProgramDescriptor` migration **duplicated** the host emitter rather than
   forking or sharing it, leaving two hand-synchronized copies (see Misc anomalies) and one
   `TT_FATAL` guard covering only the RTA layout. If that is the sanctioned answer, the *Caution:
   Porting a shared kernel* entry should say so for the host side too — the drift discipline and
   sunset paragraphs transfer almost verbatim, and a porter arriving at this op has no guidance
   today on whether to add a third emitter or restructure.

5. **The conditional-binding pattern's "promote a CTA gate to a define" rule earned its keep here,
   and might deserve a pointer from the audit's CB-endpoints subject.** Four of this op's five port-work
   items are that pattern, and none of them is visible from an endpoint census — the census says
   *legal 1:1* while the actual port hazard is a kernel instance that *names* a CB it never touches.
   The audit subject that finds them is CB endpoints, but nothing in it tells the auditor to look for
   a reference-without-access; it counts touchers. One line under *The endpoint census* — "also record
   any kernel that references a CB index without touching it: in Metal 2.0 the reference needs a
   binding" — would route this reliably.
