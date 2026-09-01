# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/minimal_matmul`

One device operation, one program factory:

- **`MinimalMatmulDeviceOperation`**
  - `MinimalMatmulProgramFactory` (`device/minimal_matmul_program_factory.cpp`)

Two things in this directory are **not** part of that unit, and both will mislead a reader who
goes by file listing alone:

- `device/minimal_matmul_fabric_bound_program_factory.cpp` is a free-standing factory, **not** an
  alternative of this device op's `program_factory_t` (a single-alternative variant,
  `minimal_matmul_device_operation.hpp:24`). It is consumed by
  `experimental/ccl/strided_all_gather_minimal_matmul_async` and is audited with **that** op. Its
  four `fabric_bound_*` kernels are therefore out of scope here.
- `device/minimal_matmul_split_device_operation.{cpp,hpp,_types.hpp}` declare a second type,
  `MinimalMatmulSplitDeviceOperation`, but it is **dead code, not a second device operation**:
  the files are absent from `sources.cmake` (never built), nothing outside them references them,
  and both the header and the source `#include "minimal_matmul_split_program_factory.hpp"` —
  a header that does not exist anywhere in the tree, along with the
  `MinimalMatmulSplitProgramFactory` type it would declare. They cannot compile. Treated as
  unreferenced and excluded from the audit; recorded under *Misc anomalies*.

Two Python bindings (`ttnn.experimental.minimal_matmul`, `ttnn.experimental.minimal_matmul_split`)
share the one real factory, so a single port covers both.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate multi-program ops, and bound what the port covers`

**Code audited:** `main` @ `beb2ea8f08a`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/minimal_matmul` |
| **Overall** | **RED** |
| **DOps / Factories** | `MinimalMatmulDeviceOperation` → `MinimalMatmulProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **No (RED)** — CB-index holdovers *and* raw-address semaphore waits, in the op's own kernels and in a cross-family donor; routed to the Device 2.0 track |
| *Prereqs* — Cross-op escapes | Issue — see Device 2.0 detail (donor) and Recipe notes (shared **host** factory helper) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — GlobalCircularBuffer / `address_offset` / GlobalSemaphore | N/A (none in use) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **No**: `Concept == legacy device-op` |
| *TTNN Readiness* — Concept (current) | `legacy device-op` |
| *TTNN Readiness* — Secretly SPMD | N/A (not `WorkloadDescriptor`) |
| *TTNN Readiness* — Custom hash | No (sheet and code agree — no `compute_program_hash`, no backdoor) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | `n/a` (PD-only column; the method exists as part of the **legacy** concept signature) |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (sheet `Porting Target`) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — Offset base pointer | none — GREEN |
| *Port work* — TensorAccessor 3rd arg | **Class 2 (drop)** — one site, inert |
| *Port work* — Tensor bindings (per binding) | *skipped — whole-op RED, no portable subset* |
| *Port work* — CB endpoints | *skipped — whole-op RED, no portable subset* |

## Result

**RED at op level; no portable subset.**

The op has exactly one program factory, and the blocking condition
(`Concept == legacy device-op`) is a property of that factory as a whole, not of a branch within
it — so there is no clean code path to carve out for a scoped-subset port.

Two independent gates fail:

1. **TTNN factory concept** → the **TTNN / ProgramDescriptor-migration team**. This is the
   *expected* outcome for an op still on the legacy imperative API, not an alarm; it unblocks when
   this op's `ProgramDescriptor` migration lands.
2. **Device 2.0** → the **Device 2.0 team**, in three parts of differing size, one of them owned
   by another family.

Both are on the op-code side, so a re-audit will read materially different code. Per the Red
outcome scoping rule the seven purely-informational subjects are deferred to that re-audit (each
recorded below).

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **RED.** The readiness sheet's `Is able to port?`
  cell reads `no`, attributed by the primary column **`Concept` = `legacy device-op`** — the op is
  not on the `ProgramDescriptor` API. Routed to the TTNN / PD-migration team; a separate ongoing
  effort, and the expected outcome for a legacy op. The gate lifts when that migration lands.

  Lightweight cross-check against the code — **clean, no conflicts**:

  | Column | Sheet | Code | Agrees |
  |---|---|---|---|
  | `Concept` | `legacy device-op` | `create()` + `override_runtime_arguments()`, no `create_descriptor` (`minimal_matmul_program_factory.hpp:30,35`) | ✓ |
  | `Custom hash` / `Backdoor custom hash` | `no` / `no` | no `compute_program_hash`, `attribute_values` or `to_hash` in the op | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | hook absent from the device op | ✓ |
  | `Override runtime args method?` | `n/a` | method exists, but as the *legacy* concept signature — the column is PD-only | ✓ (no conflict) |
  | `Pybind descriptor` | `no` | no `create_descriptor` binding in `minimal_matmul_nanobind.cpp` | ✓ |
  | `Smuggled pointer` | `no` | address-in-RTA is present but is ordinary *legacy* plumbing, not the un-annotated PD-migration `Buffer*` bug | ✓ |
  | Factory-set match | 1 row | 1 alternative in `program_factory_t` | ✓ |

  Cross-column invariants hold (`get_dynamic_runtime_args` is `no`; `Op-owned tensors?` is `no` on
  a non-`WorkloadDescriptor` row). Sheet row validated by its owner (`Diego validation = yes`).

- **Device 2.0 (every kernel used):** **RED.** In-scope kernels are the three the factory names —
  `compute.cpp`, `dm_in0_sender.cpp`, `dm_in1_sender_out.cpp` — plus the transitive closure
  `matmul_dataflow_common.hpp` and the cross-family donor
  `experimental/ccl/strided_all_gather_async/device/kernels/fused_receiver_utils.hpp`.

  Findings fall into **three shapes**, and the distinction matters for sizing:

  **(a) Isolated CB-index holdovers — mechanical, one line each.** The Device-2.0 wrapper object
  is already in scope at each site, and in every case the *sibling* code a few lines away already
  uses the method form, so the intended shape is unambiguous. All three arrived together with the
  fused-SwiGLU feature (#48742).

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `device/kernels/dm_in0_sender.cpp` | 296 | `get_read_ptr(cb_out_id)` | `cb_out` (`:155`) — `.get_read_ptr()`; the `#else` branch at `:321` already uses it |
  | `device/kernels/dm_in1_sender_out.cpp` | 258 | `get_read_ptr(cb_out_id)` | `cb_out` (`:136`) — `.get_read_ptr()`; the `#else` branch at `:283` already uses it |
  | `device/kernels/compute.cpp` | 98 | `cb_push_back(out_cb, …)` | none constructed in `swiglu_block`, but siblings `copy_and_pack_block` (`:22,42`) and `add_bias_block` (`:113,133`) both build a `CircularBuffer` and call `.push_back()` |

  ⚠ **`compute.cpp:98` is not a pure rename.** `compute.cpp` is a TRISC kernel, where
  `CircularBuffer::push_back` dispatches to `PACK((llk_push_tiles<false, false>(…)))`, whereas the
  free `cb_push_back()` performs raw FIFO arithmetic (`pages_received_ptr[0] += n;
  fifo_wr_ptr += num_words`). `PACK(...)` expands to nothing off the packer thread
  (`tt_metal/hw/inc/api/compute/common_globals.h:24,27`), so the present form runs on **all three**
  TRISC threads and the wrapper form runs once on the packer. Migrating it may therefore be a
  latent-defect fix rather than a neutral swap — worth landing as its own reviewable change, not
  swept in with the other two.

  **(b) Raw-address semaphore waits with no Device 2.0 expression — a framework/protocol item,
  not a migration chore.**

  | File | Line | Call |
  |---|---|---|
  | `device/kernels/dm_in0_sender.cpp` | 222, 262-266 | `volatile tt_l1_ptr uint32_t*` cast + `noc_semaphore_wait_min(&credits[r], …)` |
  | `device/kernels/dm_in1_sender_out.cpp` | 189, 225-229 | same |

  `Semaphore`'s only constructor takes a semaphore **id** resolved through the program's own table
  (`tt_metal/hw/inc/api/dataflow/noc_semaphore.h:44`); there is no constructor from an address.
  These sites are an *array* of per-RS-reader credit counters living in an L1 sharded buffer whose
  base arrives as a runtime arg, so no `Semaphore` object can represent them. Clearing this needs
  either a `Semaphore`-from-address constructor (or a first-class semaphore-array / `GlobalSemaphore`
  primitive — the latter is itself UNSUPPORTED in Metal 2.0, see Appendix A) **or** an MM↔RS
  protocol change owned by the CCL team. It is not work the Device 2.0 team can schedule against
  this op as-is.

  **(c) Cross-family donor — one mechanical site, the rest shape (b).** Owning family:
  `experimental/ccl/strided_all_gather_async`.

  | Line | Call | Fixable? |
  |---|---|---|
  | 142 | `reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(<id>))` | **Yes — mechanical.** A semaphore *id* is in hand, so `Semaphore<>(id)` replaces it directly |
  | 67, 260, 264, 356, 361-362 | raw `tt_l1_ptr` sem pointers + `noc_semaphore_wait_min` on peer-supplied addresses | No — shape (b) |

  Note `noc_semaphore_wait_min` itself is **not** a Device 1.0 relic — it lives in the Device 2.0
  header and is exactly what `Semaphore::wait_min` calls internally (`noc_semaphore.h:108`). For
  shapes (b) and (c) the finding is the *hand-computed address*, not the function.

  **Reachability.** The shape-(b) sites in this op are compiled only under `#ifdef MM_WINDOW_BLOCKS`,
  which only `experimental/ccl/minimal_matmul_strided_reduce_scatter_async` defines; the donor is
  used only under `#ifdef FUSE_AG`. Both are nonetheless in the audited kernels' source and include
  closure, so they gate this op. **If the schedule needs this op sooner than the protocol work can
  land, relocating the `MM_WINDOW_BLOCKS` path into SRS-owned kernels would clear shape (b) without
  solving the protocol** — at the cost of another fork of the DM kernels. Flagged as a question
  below rather than recommended.

  `get_tile_size(cb_id)` appears at four sites and is **sanctioned** by Device 2.0 — not a
  violation, and deliberately excluded from the tables above.

- **Feature compatibility:** **GREEN.** Every Appendix A entry scanned against the factory and the
  in-scope kernels; none fires.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` / `global_cb` / `remote_cb` construct |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `address_offset` / `set_address_offset` / `set_globally_allocated_address` |
  | GlobalSemaphore | N/A | all semaphores are ordinary `CreateSemaphore` (7 on the factory's core grid) |

- **Offset base pointers:** **GREEN.** Every address RTA in the factory resolves to a bare
  `buffer()->address()` with no host-side arithmetic folded in — checked across all of
  `in0/in1/in2/in3`, the ternary pair, and the N output addresses
  (`minimal_matmul_program_factory.cpp:482-494, 810-818, 872-879`). No Type 1 and no Type 2 site.
  The op is not in the dated triage's tables, and the scan agrees — the "no fold, not in tables"
  outcome. Every address therefore reaches TensorParameter analysis as a clean base.

- **TensorAccessor 3rd argument:** **GREEN — Class 2 (redundant / inert), one site.** Port work,
  not a gate.

  `device/kernels/matmul_dataflow_common.hpp:33-34` constructs every output accessor with an
  explicit page size, fed from `dm_in0_sender.cpp:73` / `dm_in1_sender_out.cpp:72` as
  `out_tile_size`. Classified independently of the dated triage (which also lists `minimal_matmul`
  as Class 2):

  - **Magnitude:** `out_tile_size` is **compile-time** (`get_compile_time_arg_val(12)`), whose host
    value is `tt::tile_size(output_data_format)` (`minimal_matmul_program_factory.cpp:166`) — the
    Class 2(a) shape exactly, and block-float-safe (bf8 → 1088 B, not 1024).
  - **Verdict:** correct magnitude, TILE layout, so inert whether the output is interleaved or
    sharded. Dropping it in the collapse to `TensorAccessor(tensor::name)` is a pure no-op. Do
    **not** set `dynamic_tensor_shape` — the page size is compile-time-pinned, not dynamic.

- **CB endpoints:** *skipped — whole-op RED, no portable subset; re-audit on unblock.* (Also the
  acute case here: a full per-`(CB, config)`-per-node census across 7–9 CBs and the
  bias/ternary/SwiGLU/chunked configuration matrix is expensive and would be redone against
  PD-migrated code.)

## Port-work summary  *(no brief issued — recorded for the re-audit)*

- **TensorAccessor 3rd arg:** drop the page-size argument at `matmul_dataflow_common.hpp:33-34`;
  it disappears with the host-side CTA that feeds it. Class 2, no `dynamic_tensor_shape`.
- **TensorParameter relaxation:** `none` — nothing to apply.
- Everything else deferred with the informational subjects.

## Heads-ups  *(no brief issued — recorded for the re-audit)*

- **Target concept** is `ProgramSpecFactoryConcept` (sheet `Porting Target`; `Op-owned tensors?` is
  `no`, and `Override runtime args method?` is `n/a` on this legacy row — re-read that column after
  the PD migration, since it selects between the base and Custom concepts).
- **One factory serves two Python ops.** `minimal_matmul` (chunks=1) and `minimal_matmul_split`
  (chunks=N) both route through `ttnn::prim::minimal_matmul`; N outputs means a variable number of
  tensor bindings on one `KernelSpec`, and the output addresses today ride a variable-length RTA
  tail. Worth resolving early at port time — the recipe does not obviously cover it.

## Team-only

- **Skipped informational subjects** (whole-op RED, no portable subset; blockers are all
  op-code-side, so re-audit will read different code): TTNN porting shape *(target concept captured
  above from the sheet — cheap, and needed for routing)*, TensorParameter analysis, CB endpoints,
  Out-of-directory coupling (full by-shape donor inventory), RTA varargs, Incidental anomalies.
  `TensorParameter relaxations` was read from the sheet regardless because it is a gate conjunct —
  see Recipe notes.
- **Out-of-directory coupling, partial** — the gate-relevant part only, since the Device 2.0 gate
  needs it specific: the op's kernels `#include` one cross-family donor,
  `experimental/ccl/strided_all_gather_async/device/kernels/fused_receiver_utils.hpp`. The full
  by-shape per-call inventory is deferred with the subject.

## Misc anomalies  *(team-only, non-gating)*

- **Dead `MinimalMatmulSplitDeviceOperation` files.**
  `device/minimal_matmul_split_device_operation.{cpp,hpp,_types.hpp}` are unbuildable and unbuilt
  (details in the identifying section above). They appear to have been left behind by #36502, which
  merged the split device operation into `MinimalMatmulDeviceOperation` and removed
  `minimal_matmul_split_program_factory` but not these. Harmless at runtime, but they make the
  directory read as though it holds two device operations and a factory that does not exist —
  which costs every future reader, and every future audit, the same disambiguation. Recommend
  deletion as an independent cleanup. Noticed while establishing audit scope.

- **Inaccurate rationale on the page-size override.**
  `device/kernels/matmul_dataflow_common.hpp:32` — the comment justifying the `TensorAccessor` 3rd
  argument says it comes "from runtime args" and guards a value that "may be stale on program cache
  hits". Both halves are inaccurate: the value is a **compile-time** arg
  (`get_compile_time_arg_val(12)`), so it is baked per program and cannot go stale on a cache hit
  any differently than `TensorAccessorArgs::AlignedPageSize` can. This does not change the Class 2
  verdict (the *value* is correct either way), but the stated rationale for the override does not
  hold, and a reader could take it as a reason to preserve the argument through the port. Noticed
  while working the 3rd-argument gate.

## Questions for the user

1. **Relocate the `MM_WINDOW_BLOCKS` path, or wait for the protocol fix?** The shape-(b) Device 2.0
   finding is dead code for every consumer of this op except
   `minimal_matmul_strided_reduce_scatter_async`, yet it gates the whole file. Moving that path into
   SRS-owned kernels would clear it now and put the problem with the team that owns it; the cost is
   a third fork of the DM kernels in an op that already carries a large duplicated fork. Recommend
   deciding this *with* the PD-migration scheduling, not before — the factory-concept gate blocks
   the port regardless, so there is no schedule pressure yet.

## Recipe notes

1. **The recipe contradicts itself on `TensorParameter relaxations`.** The Red-outcome scoping rule
   lists it among the *seven purely-informational* subjects to skip on a whole-op RED
   (`metal2_audit.md`, Feasibility audit intro), but the subject's own section ends *"Finding role:
   **GATE** (routed to the ops team)"* and the finding-roles table treats the relaxation column as a
   gate conjunct. I read the cell regardless (it is `none`, so moot here), but a future auditor
   hitting a non-`none` value on a RED op would get opposite instructions from the two places.

2. **No rung covers a shared *host* program-factory helper.** *Caution: Porting a shared kernel*
   handles a kernel `.cpp` bound by several factories via a checked-in `_metal2` fork. This op's
   `minimal_matmul_factory_helper_common` is the host-side analogue:
   `experimental/ccl/minimal_matmul_strided_reduce_scatter_async` calls it directly, and also calls
   `MinimalMatmulProgramFactory::override_runtime_arguments` and `::cached_program_t::proxy`,
   consuming its `shared_variables_t` type. Porting the helper to produce a `ProgramSpec` breaks
   that consumer at compile time, and the fork convention does not obviously transfer (the coupling
   is a C++ type and entry point, not a file path). The `fabric_bound` helper has the same
   relationship with `strided_all_gather_minimal_matmul_async`. Worth an explicit rung, or an
   explicit statement that this routes to the PD-migration owner as a bundling decision.

3. **The Device 2.0 gate's two tiers do not cover "no Device 2.0 expression exists."** The Red
   bullet offers *isolated CB-index holdovers* (mechanical, wrapper in scope) and *broad Device 1.0*
   (full migration). This op exhibits both, plus a third shape the tiers miss: the kernel is
   otherwise structurally Device 2.0, and the construct is not a holdover because there is nothing
   to migrate *to* — `Semaphore` cannot be built from an address. Sizing guidance for the Device 2.0
   team is materially different for that third shape (it is a framework/API request, not work they
   can schedule against the op), so a third tier would help.

4. **A "mechanical" CB-index holdover can still be behaviour-affecting.** The gate's isolated-holdover
   tier is described as "a 1-line mechanical replacement", which is true of the two `get_read_ptr`
   sites (identical function bodies) but *not* of `cb_push_back` → `CircularBuffer::push_back` on a
   **compute** kernel, where the wrapper adds a `PACK()` thread guard and routes through an LLK.
   Worth a caution in the recipe: on TRISC, the free-function and wrapper forms of the FIFO calls are
   not interchangeable, so a Device 2.0 migration of a compute kernel needs its own verification
   rather than being waved through as a rename.
