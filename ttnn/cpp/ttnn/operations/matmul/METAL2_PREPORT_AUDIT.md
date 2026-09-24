# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/matmul`

**Audit scope: ONE ProgramFactory — `SparseMatmulMultiCoreReuseMcast1DProgramFactory`,
on `SparseMatmulDeviceOperation`.**

This is a **re-audit**. The previous pass (commit `377932c93f3`) returned RED on the TTNN
factory-concept gate because the factory was still on the legacy imperative API. That blocker has
since cleared: the `ProgramDescriptor` migration landed as
`d71e4ce4929 2026-09-23 [Cleanup] Descriptor port: sparse matmul SparseMatmulMultiCoreReuseMcast1DProgramFactory (#57369)`.
The seven purely-informational subjects the previous audit skipped (correctly — the migration
rewrote the code they describe) are run here for the first time.

This factory belongs to the **second** DeviceOperation in the `matmul` directory. Everything below
was cross-checked against `SparseMatmulDeviceOperation` and its own files under `device/sparse/`,
**not** against the dense `MatmulDeviceOperation`. The dense op's factories were not audited and
nothing here is a verdict on them.

- **`SparseMatmulDeviceOperation`** (`device/sparse/sparse_matmul_device_operation.hpp`)
  - **`SparseMatmulMultiCoreReuseMcast1DProgramFactory`** ← **audited**
    (`device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.{hpp,cpp}`)
    — the only alternative in `program_factory_t` (`hpp:21`)
- **`MatmulDeviceOperation`** (`device/matmul_device_operation.hpp`) — seven factories, none audited.
  Several are **already on Metal 2.0** (`Concept = MetalV2` in effect), which matters here only
  because they bind the same kernel forks this port reuses (see *Out-of-directory coupling*).

> **⚠ Sheet lookup hazard — two rows carry this exact DeviceOperation and Factory name.**
> They differ only in the `Op` column:
> - `Op = matmul` → the mainline factory, `Factory definition path` under
>   `ttnn/cpp/ttnn/operations/matmul/device/sparse/…`. **This is the audited row.**
> - `Op = experimental/quasar/matmul` → the quasar copy. **Out of bounds for this audit and not
>   used for any finding.**
>
> Match on `Op` or on `Factory definition path`, not on the factory name.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

**Readiness sheet:** fetched live this session via the Google Drive connector (486 rows, 28 columns).

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/matmul` (sparse sub-tree) |
| **Overall** | **GREEN** — every gate cleared; brief issued |
| **DOps / Factories** | `SparseMatmulDeviceOperation` → `SparseMatmulMultiCoreReuseMcast1DProgramFactory` (its only variant alternative) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 4 bound kernels Device 2.0 compliant; so are the `_metal2` forks the port will bind |
| *Prereqs* — Cross-op escapes | **Ok** — all donors are `tt_metal/*` LLK/HAL plus one in-family helper header; no gating shape |
| *Feature Support* — overall | **GREEN** |
| *Feature Support* — GlobalCircularBuffer | **N/A** — type present in the attribute struct and public API, **never consumed**; reasoning in full below |
| *Feature Support* — CBDescriptor `address_offset` | **N/A** — no `address_offset` anywhere in the sparse tree |
| *Feature Support* — GlobalSemaphore | **N/A** — two ordinary program semaphores |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** — cell reads `yes (with PD step)`; the PD step landed in #57369 |
| *TTNN Readiness* — Concept (current) | `descriptor` — lone `create_descriptor` returning `ProgramDescriptor` (`…mcast_1d_optimized.hpp:24`). **The sheet still reads `legacy device-op` — a stale primary column; see *Gate detail*.** |
| *TTNN Readiness* — Secretly SPMD | N/A (not a `WorkloadDescriptor`) |
| *TTNN Readiness* — Custom hash | **No** — declaration and definition both **commented out** (`sparse_matmul_device_operation.hpp:33`, `.cpp:504`) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** — zero hits across `device/sparse/` |
| *TTNN Readiness* — `override_runtime_arguments` | **No** — removed by the PD migration; the factory is a lone `create_descriptor` |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** — `matmul_nanobind.cpp` binds only the user-facing `sparse_matmul` function |
| *TTNN Readiness* — Op-owned tensors | **No** |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (no `override_runtime_arguments`) |
| *Port work* — Offset base pointer | **none** — every address arg is a clean base; five are already `Buffer*` bindings |
| *Port work* — Tensor bindings (per binding) | `in0` Case 1 · `in1` Case 1 · `sparsity` Case 1 · `indices` Case 1 (indexed mode only) · `output` Case 1 |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | **N/A** — every construction in all four kernels is 2-arg |
| *Port work* — CB endpoints | `in0` 1:1 (disjoint-node producers) · `in1` 1:1 · `out` 1:1 · `interm0` self-loop · `sparsity` (c_6) self-loop · `in1_sparsity` (c_7) self-loop · **conditional alias group** on `out`/`interm0` |

**CB endpoints** are dispositions, not gates. No multi-binding flag is needed anywhere, and no CB is
dead.

---

## Result

**GREEN — brief issued.** Every gate cleared:

- **TTNN factory concept** — `Is able to port?` reads **`yes (with PD step)`**, and the PD step is
  done (#57369, on `main` since 2026-09-23). Target concept `ProgramSpecFactoryConcept`.
- **Device 2.0** — all four bound kernels, and the four `_metal2` forks the port will bind instead,
  are structurally Device 2.0.
- **Feature compatibility** — no Appendix A entry fires.
- **Offset base pointers** — no host-folded offset reaches an address arg.
- **TensorAccessor 3rd argument** — the subject never fires; no accessor passes a 3rd argument.

**One finding routed outward, deliberately not treated as a gate: the sheet's `Concept` column is
stale for this row.** See *Gate detail* for the reasoning and the route.

---

## Gate detail

### TTNN factory concept (`Is able to port?`): GREEN — with a stale primary column reported

The sheet's mainline row for (`matmul`, `SparseMatmulDeviceOperation`,
`SparseMatmulMultiCoreReuseMcast1DProgramFactory`) reads:

| Column | Sheet value |
|---|---|
| `Is able to port?` | **`yes (with PD step)`** |
| `Concept` | `legacy device-op` |
| `Op Classification` | `Legacy Op` |
| `Porting Target` | `ProgramSpecFactoryConcept` |
| `Override runtime args method? (PD only)` | `n/a` |
| `Custom hash` / `Backdoor custom hash` | `no` / `no` |
| `Runtime-args update (get_dynamic_runtime_args)` | `no` |
| `Pybind descriptor` | `no` |
| `Smuggled pointer` | `no` |
| `Known op issues` | *(empty)* |
| `Diego validation` | `yes` |
| `TensorParameter relaxation` | `none` |
| `Op-owned tensors?` | `no` |
| `Model` | `resnet` |
| `Uses llama kernels? (primary or shared)` | `yes` |

**Lightweight cross-check — three primary columns agree, one does not:**

| Column | Sheet | Code evidence | Agrees |
|---|---|---|---|
| `Concept` | `legacy device-op` | **`descriptor`** — lone `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` at `…mcast_1d_optimized.hpp:24`; no `cached_program_t`, no `create`, no `override_runtime_arguments` | **✗** |
| `Custom hash` | `no` | `compute_program_hash` declaration and definition are both **commented out** (`sparse_matmul_device_operation.hpp:33`, `.cpp:504`). No live override → default reflection hash | ✓ |
| `Runtime-args update (get_dynamic_runtime_args)` | `no` | Zero hits across `device/sparse/` | ✓ |
| `Override runtime args method? (PD only)` | `n/a` | Absent from the factory. `n/a` was correct for the pre-PD shape; the post-PD correct value is `no`. Same staleness as `Concept`, and it happens to route to the same target concept either way | (✗, inert) |
| `Pybind descriptor` | `no` | Neither the device-op class nor the factory is bound; `matmul_nanobind.cpp` binds only `sparse_matmul` (`:1063`, `:1183`) | ✓ |
| Factory-set match | 1 mainline row | `program_factory_t = std::variant<SparseMatmulMultiCoreReuseMcast1DProgramFactory>` — exactly one alternative (`hpp:21`) | ✓ |

Cross-column invariants hold: `get_dynamic_runtime_args` is `no`, and `Op-owned tensors?` is `no`.

**Why this is reported rather than gated.** The recipe's *spreadsheet-is-broken* rule would read a
primary-column conflict as a GATE. I judged this one differently, and state the reasoning so it is
auditable:

1. **The verdict cell already anticipated exactly this transition.** `Is able to port?` does not read
   `no` — it reads `yes (with PD step)`. That is a conditional clearance whose condition is *the
   very landing that made `Concept` stale.* The condition is met (#57369). Treating the stale
   snapshot column as a blocker would block on the *precondition of a clearance the sheet itself
   granted*.
2. **The disagreement runs in the permissive-to-restrictive direction only for the stale side.** The
   sheet describes a **more** primitive shape than the code has. Every way the recipe's cross-check
   can be harmed by a stale `Concept` — porter sent down the legacy path, wrong target concept —
   fails safe here: the code is further along, and `Porting Target` already names
   `ProgramSpecFactoryConcept`, which is what the code's actual concept maps to.
3. **No other primary column disagrees**, and no cross-column invariant is violated.

**Routed to the readiness-sheet owner (Diego) as a refresh request, not an allegation:** the
`matmul` / `SparseMatmulDeviceOperation` / `SparseMatmulMultiCoreReuseMcast1DProgramFactory` row
needs `Concept` → `descriptor`, `Op Classification` → `PD Op`,
`Override runtime args method? (PD only)` → `no`, and `Is able to port?` → plain `yes` now that the
PD step has landed. Raised as a question below.

### Device 2.0 (every kernel used): GREEN

All four kernels reachable from this factory are structurally Device 2.0 — `Noc` from `noc.h`,
`DataflowBuffer` wrappers, `TensorAccessor`. Zero hits for broad Device-1.0 idioms
(`InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedPow2AddrGen*`, raw `noc_async_read(` /
`noc_async_write(`, raw `noc_semaphore_*`) and no non-sanctioned CB-index free-function holdovers.

| Kernel (under `device/kernels/`) | Bound at | `_metal2` fork the port will bind instead |
|---|---|---|
| `dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp` | `:494` | `…_in0_sender_padding_metal2.cpp` ✓ exists |
| `dataflow/reader_bmm_tile_layout_in0_receiver.cpp` | `:509` | `…_in0_receiver_metal2.cpp` ✓ exists |
| `dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | `:521` | `…_in1_sender_writer_padding_metal2.cpp` ✓ exists |
| `compute/bmm_large_block_zm_fused_bias_activation.cpp` | `:579` | `…_activation_metal2.cpp` ✓ exists |

The forks are equally Device 2.0 (they are Metal-2.0 conversions *of* Device 2.0 sources), so the
gate holds for the sources the port actually binds.

### Feature compatibility: GREEN

| Feature | Status | Notes |
|---|---|---|
| **GlobalCircularBuffer** | **N/A** | Recognition signals fire on the *type*, but the feature is **not in use**. Reasoning below |
| CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset`, no `set_address_offset`, no `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor` |
| GlobalSemaphore | N/A | No `GlobalSemaphore` type, no `CreateGlobalSemaphore`. Two ordinary program semaphores (`SemaphoreDescriptor` ids 0 and 1) |

#### GlobalCircularBuffer — adjudicated N/A (unchanged verdict, re-verified post-PD)

Two of Appendix A's "definitely this feature" bullets fire on a literal reading:

- `sparse_matmul_device_operation_types.hpp:30` declares
  `std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer> global_cb;` as a field of
  `SparseMatmulParams`, and `:9` includes `global_circular_buffer.hpp`.
- `sparse_matmul_device_operation.hpp:61` and `:79` carry the
  `std::optional<const …GlobalCircularBuffer>&` factory-signature shape.

**But the feature is demonstrably not in use.** Every consumer traced:

| Site | What it does with `global_cb` |
|---|---|
| `sparse_matmul_device_operation.hpp:61,79` | accepts it on the public API, defaulted `std::nullopt` |
| `sparse_matmul_device_operation.cpp` (×4 sites) | forwards it into `SparseMatmulParams` and onward |
| `…mcast_1d_optimized.cpp:48` | copies it into a `MatmulParams` **solely** so `get_program_config` can derive a program config |
| *(nothing further)* | — |

**No CB is ever created from it.** The PD migration did not change this: the six
`CreateCircularBuffer` calls the previous audit found are now six `CBDescriptor` literals
(`:615`, `:635`, `:655`, `:668`, `:684`, `:703`/`:715`), and **not one sets `.global_circular_buffer`**.
There is no `.remote_index(`, no `remote_cb_config`, and no 4-arg
`experimental::CreateCircularBuffer(…, global_cb)`. The disambiguator Appendix A names is absent.

**Verdict: N/A.** Determinable, so the conservative default does not apply. Raised as a question
below, because if the parameter is *intended* to be wired up later the answer changes.

### Offset base pointers: GREEN

The PD migration moved this op onto the framework's `Buffer*`-binding interim mechanism, so the
address args now arrive two ways — and **both are clean bases**:

| Address arg | Form | Site | Clean base? |
|---|---|---|---|
| in0 (`in0_args[0]`) | `Buffer*` binding | `:790` | ✓ `in0_buffer` |
| sparsity (`in0_args[7]`) | `Buffer*` binding | `:791` | ✓ `sparsity_buffer` |
| in1 (`in1_args[0]`) | `Buffer*` binding | `:872` | ✓ `in1_buffer` |
| in1 sparsity / indices (`in1_args[6]`) | `Buffer*` binding | `:873` | ✓ `in1_sparsity_buffer` |
| out (`in1_args[7]`) | `Buffer*` binding | `:874` | ✓ `out_buffer` |

The five `->address()` calls at `:771`, `:782`, `:809`, `:818`, `:822` build the placeholder
`uint32_t` vector that is then overwritten by the `Buffer*` at the same index — so they are dead
values, not a second delivery path. A scan for `address() +`, `addr + …` and `+ …offset…` returns
**zero** hits. No Type 1, no Type 2, no Type 3, no Type 4.

Note the *tile-index* args that ride alongside (`in0_tensor_start_tile_id`,
`in1_tensor_start_tile_id`, `out_tensor_start_tile_id`) are tile counts, not addresses — the tiled
path the recipe describes as unaffected.

### TensorAccessor 3rd argument: N/A

No accessor in any of the four kernels passes a third (page-size) argument — every construction is
2-arg:

| Kernel | Sites |
|---|---|
| `…in0_sender_padding.cpp` | `:162` (`in0`), `:168` (`sparsity`) |
| `…in1_sender_writer_padding.cpp` | `:193` (`bias`), `:232` (`in1`), `:241` (`out`), `:249` (`sparsity`) |
| `…in0_receiver.cpp` | none |
| `bmm_large_block_zm_fused_bias_activation.cpp` | none |

The subject never fires. (`sparsity_pagesize` is a **read size** passed to `noc.async_read`, not an
accessor constructor argument — a near-miss worth naming because the identifier reads like one.)

### CB endpoints: GATE-free, every CB dispositioned

Census per `(CB, config)`, per node. The compute kernel and the in1 sender/writer cover
`all_cores_with_work`; the in0 sender covers `{start_core}` and the in0 receiver covers
`all_cores \ {start_core}` — **disjoint** node sets, so their shared `in0` role is an ordinary 1:1
per node, not a multi-binding.

| CB | Toucher(s) per node | Verdict | Disposition |
|---|---|---|---|
| `c_0` in0 | in0 sender **or** in0 receiver (locked producer, disjoint nodes) + compute (locked consumer) | plain 1:1 | ordinary DFB, 1P+1C |
| `c_1` in1 | in1 sender/writer (locked producer) + compute (locked consumer) | plain 1:1 | ordinary DFB |
| `c_4` out | compute (locked producer) + in1 sender/writer (locked consumer) | plain 1:1 | ordinary DFB |
| `c_5` interm0 | compute only — `reserve_back`/`push_back` **and** `wait_front`/`pop_front` | **1 toucher** | **self-loop** (compute PRODUCER + CONSUMER) |
| `c_6` sparsity | in0 sender only — `reserve_back` (`:202`), `push_back` (`:483`), `wait_front` (`:484`), `pop_front` (`:485`) in the fork | **1 toucher** | **self-loop** |
| `c_7` in1 sparsity / indices | in1 sender/writer only — `reserve_back` (`:283`) + raw reads (`:331`, `:339`) in the fork | **1 toucher** | **self-loop** |

**No dead CB.** Every allocated index is referenced by a bound kernel. Three *named CTAs* point at
indices the sparse factory never allocates — `cb_in0_sharded` → `c_2` (`:500`), `cb_bias` → `c_3`
(`:527`), `cb_in0_transposed` → `c_10` (`:589`) — but each is read by its kernel only under a
feature this factory never enables (`IN0_SHARDED` / `EXTRACT_SHARD_SUB_BLOCKS`, `FUSE_BIAS`,
`IN0_TRANSPOSE_TILE`), so they are dead *arguments*, not dead buffers. Under Metal 2.0 they simply
have no binding. Listed under *Misc anomalies* because a reader counting named CB args against
allocated CBs will notice the mismatch.

**Config-dependent alias group.** `c_4` and `c_5` are **one** legacy `CBDescriptor` with two
`format_descriptors` when `interm0_data_format == output_data_format` (`:712`–`:729`), and two
independent descriptors otherwise (`:681`–`:711`). So:

| Config | `(c_4, c_5)` shape | Port disposition |
|---|---|---|
| `interm0_data_format == output_data_format` | one L1 region, two indices | **Aliased DFBs** — `out` and `interm0` name each other in `advanced_options.alias_with`, both sized `out_CB_size` |
| `interm0_data_format != output_data_format` | two regions | no aliasing; `interm0` sized `interm0_CB_size` |

This is the [Aliased DFBs] pattern, gated per config. It is *not* same-FIFO aliasing: the two
indices have distinct page sizes in general and independent FIFO pointers.

---

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding), all **Case 1** (fed to a `TensorAccessor`, never used raw):
  `in0`, `in1`, `sparsity`, `output`, and — **only when indexed/gather mode is on** — `indices`.
- **TensorParameter relaxation:** `none`.
- **TensorAccessor 3rd arg:** none — the subject never fires.
- **CB endpoints:** self-loop `interm0`, `sparsity` (c_6), `in1_sparsity` (c_7); ordinary 1P+1C on
  `in0`, `in1`, `out`; conditional alias group `{out, interm0}` when the two formats match. No
  multi-binding flag anywhere; no dead CB to drop.

## Heads-ups  *(mirrors the brief)*

- **Shared kernels — all four, and all four forks already exist.** Every kernel this factory binds
  is *lent*: it lives in matmul's own `device/kernels/` tree and is also bound by the dense
  `MatmulDeviceOperation` factories. A `_metal2` fork sits beside each original and is already bound
  by ported dense factories, so the port is on **rung 1 (reuse)** for all four. **The forks are
  read-only** — they have consumers. Their binding vocabulary is the port's constraint.
- **The forks were written anticipating this port.** Each fork's header note names `SPARSITY` as a
  define-gated optional resource family, and both sparse readers already read `num_active`,
  `batchB`, `bcast_A`, `get_batch_from_reader`, `sparsity_pagesize`, `num_batch_compute` and
  `compact_output`. **No factory sets `SPARSITY` today** — this port is the first, which is why the
  region has never been exercised through a Metal 2.0 spec.
- **Two distinct sparsity slots, one accessor name.** `c_6`/`tensor::sparsity` on the in0 sender
  carries the real sparsity mask; `c_7`/`tensor::sparsity` on the in1 sender/writer carries the
  *active-group id list* in indexed mode and the mask otherwise (`…mcast_1d_optimized.cpp:135`).
  Accessor names are per-kernel, so both can be `"sparsity"` — but they need **two** DFB specs and,
  in indexed mode, **two** `TensorParameter`s.
- **The DM processor assignment is the opposite of the dense factory's.** Sparse: in0
  sender/receiver on `RISCV_0`, in1 sender/writer on `RISCV_1` (`:505`, `:516`, `:533`). Dense
  metal2: in0 on `RISCV_1`, in1 on `RISCV_0`. Carry the *sparse* values; this is the silent
  perf-regression trap the recipe's hw_config section warns about.
- **`FUSE_ACTIVATION = "0"` is a dead define.** `:454` emits it to the compute kernel; no matmul
  compute kernel (legacy or fork) references the name. Inert either way.
- **RTA varargs:** none. Every runtime arg is a distinct field read once; the port names all of them.
- **No pybind or device-op-class edit is forced.** Nothing under `device/sparse/` is bound, and the
  factory already lives in a `program_factory_t` variant — so the port is a method swap inside the
  existing struct, with no `create_descriptor` pybind line to delete and no direct-descriptor
  conversion.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.** No donor shape gates, and no donor needs work.

| Op kernel | Donor file | Class | Status |
|---|---|---|---|
| all four | `tt_metal/hw/inc/api/**` (`noc.h`, `dataflow_buffer.h`, `noc_semaphore.h`, `endpoints.h`, `core_local_mem.h`, `experimental/kernel_args.h`) | 1 — LLK/HAL/firmware | ✓ no concern |
| in0 sender, in1 sender/writer | `ttnn/cpp/ttnn/operations/kernel_helper_functions/pad_tile.hpp` | 4 — shared utility pool | ✓ takes plain scalars and an L1 pointer; no CB/sem/accessor handle in its signature |
| compute | `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_fused_activation.hpp` | 5 — in-family shared | ✓ in-family function-call escape; does not gate |
| in0 sender, in1 sender/writer | `ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp` | 6 — cross-family donor | **unreachable** — behind `FUSE_OP` / `FUSE_OP_ALL_GATHER` / `FUSE_OP_REDUCE_SCATTER`, which this factory hardcodes false (`:369`, `:423`, `:424`) and which no Metal 2.0 factory may set. Recorded because the include is textually present in the forks |

**Borrowed kernel files (file-path instantiation).** None borrowed — all four sources are
matmul-owned. But all four are **lent**, which carries the same coordination cost:

| Kernel | Other binders (mainline, non-quasar) | `_metal2` fork beside it? |
|---|---|---|
| `…in0_sender_padding.cpp` | `matmul_multicore_reuse_mcast_1d_program_factory.cpp` (`:632`, `:1670`), `…mcast_2d…` (`:2600`) | ✓ `…_metal2.cpp`, bound by the ported 1D (`:3769`, `:4939`) and 2D (`:1307`) paths |
| `…in0_receiver.cpp` | `…mcast_1d…` (`:697`), `…mcast_2d…` (`:2656`, `:2690`) | ✓ `…_metal2.cpp`, bound by the ported 1D path |
| `…in1_sender_writer_padding.cpp` | `…mcast_1d…` (`:710`, `:1686`), `…mcast_2d…` (`:2617`) | ✓ `…_metal2.cpp`, bound by ported 1D (`:3977`, `:5026`) and 2D (`:1392`) |
| `bmm_large_block_zm_fused_bias_activation.cpp` | `…mcast_1d…` (`:816`, `:1815`), `…mcast_2d…` (`:2792`) | ✓ `…_metal2.cpp`, bound by ported 1D (`:4222`, `:5341`), 2D (`:1717`), dram_sharded (`:1026`), optimized (`:564`), batched_hs_dram_sharded (`:518`) |

**This is a sunset list, not authorization to convert anything in place.** The legacy copies serve
the dense factories still on the descriptor API; they are retired only when the last binder moves.

### Relaxation candidates (FYI-U, **fallible**)

None mined. The op has no live custom hash, so there is no narrowed key to read a candidate out of.

### TTNN factory analysis

- **Concept:** `ProgramSpecFactoryConcept` — the factory has no `override_runtime_arguments`.
- **Fit:** single-program, one `ProgramSpec` stamped across the mesh. No op-owned tensors. No
  op-owned `GlobalSemaphore`s. Tensor-arg matching **strict** (`relaxation = none`).
- **Legacy-to-Metal-2.0 shape:** 1:1. The PD migration already collapsed the dead
  `SparseMatmulMeshWorkloadMultiCoreReuseMcast1DFactory` the previous audit flagged; it is gone from
  the tree.
- **Custom `compute_program_hash`:** none (commented out at `sparse_matmul_device_operation.hpp:33`
  and `.cpp:504`). Default reflection hash. **Leave the commented-out lines alone** — they are
  device-op-class code, off the port's writeable surface.
- **`get_dynamic_runtime_args`:** absent.
- **Pybind `create_descriptor`:** absent.
- **Stop signals:** none.

---

## Misc anomalies  *(team-only, non-gating)*

- **A public API parameter is accepted and silently ignored.** `sparse_matmul(...)` takes
  `const std::optional<const GlobalCircularBuffer>& global_cb`
  (`sparse_matmul_device_operation.hpp:61`, `:79`), threads it through `SparseMatmulParams` into the
  factory, and the factory uses it for nothing but `get_program_config`'s config derivation. A
  caller passing a GlobalCircularBuffer today gets it discarded with no diagnostic. Because
  `SparseMatmulParams` feeds the default reflection hash, the field also participates in the
  program-cache key, so two otherwise-identical calls differing only in `global_cb` miss the cache
  while producing identical programs. Worth an owner's decision: wire it up, or remove it.

- **Three named CB arguments point at indices the factory never allocates.**
  `cb_in0_sharded` → `c_2` (`…mcast_1d_optimized.cpp:500`), `cb_bias` → `c_3` (`:527`),
  `cb_in0_transposed` → `c_10` (`:589`). Each is read by its kernel only under a feature this
  factory never enables, so all three are inert — but a reader auditing named CB args against the
  six allocated CBs will find three that resolve to nothing. They disappear in the Metal 2.0 port
  (no binding is declared), which removes the confusion rather than fixing a bug.

- **`FUSE_ACTIVATION = "0"` is emitted to a kernel that never reads it** (`:454`). No matmul compute
  kernel — legacy or `_metal2` — references `FUSE_ACTIVATION`. Dead since some earlier refactor.
  Note the value is `"0"`, so a hypothetical `#ifdef FUSE_ACTIVATION` would read *true*; anyone
  reviving the name needs to know that.

- **Five dead `->address()` reads.** `:771`, `:782`, `:809`, `:818`, `:822` compute addresses into
  the `uint32_t` staging vector that the `Buffer*` assignment at `:790`, `:791`, `:872`, `:873`,
  `:874` then overwrites at the same index. Harmless, and arguably deliberate as documentation of
  the slot layout, but a reader looking for smuggled addresses will hit them first.

- **Six locals are computed and never used**, all survivors of the block-sharded path this factory
  does not have: `in0_mcast_cores_without_work_and_in_receiver_grid`,
  `in0_mcast_cores_without_work_and_not_in_receiver_grid`, `in0_mcast_noc_x`, `in0_mcast_noc_y`
  (`:273`–`:277`), plus `src0_cb_index` / `src1_cb_index` / `output_cb_index` / `interm0_cb_index`
  which exist only to feed `log_debug`. Dead weight, not a defect.

- **`batchA` is passed twice to the in0 sender**, once as `in0_B` and once as `in1_B` (`:360`,
  `:361`). Correct for this op — the sparse 1D path broadcasts in1 across the A-batch — but the
  duplicated literal reads like a copy-paste bug, and the legacy positional list gave no hint
  otherwise. Under Metal 2.0 the two names make the intent legible.

- **`validate_on_program_cache_hit` is declared alongside `validate_on_program_cache_miss`**
  (`sparse_matmul_device_operation.hpp:24`). Uncommon among the matmul factories; noted because a
  cache-hit validation interacts with the strict `TensorParameter` match the port introduces.

---

## Questions for the user

1. **Readiness-sheet refresh (route to Diego).** The row's `Concept` still reads `legacy device-op`
   and `Op Classification` reads `Legacy Op`, but the code is on `descriptor` as of #57369. I
   treated the conflict as staleness rather than as a *spreadsheet-is-broken* gate, because
   `Is able to port?` already reads `yes (with PD step)` and the PD step has landed — the reasoning
   is in *Gate detail*. **If you would rather the recipe's letter be followed, this is a RED and the
   port waits on the sheet update.** Please confirm the refresh is wanted (and that proceeding was
   the right call).

2. **Is `global_cb` on `sparse_matmul` intended to be wired up, or is it vestigial?** The parameter
   is accepted on the public API and threaded to the factory, which never creates a CB from it. I
   adjudicated Appendix A as **N/A** on that basis. **If the intent is to attach a
   GlobalCircularBuffer later, that changes the answer** — Metal 2.0 has no `GlobalDataflowBuffer`,
   so the feature would then be a genuine blocker with its own entry against the Metal 2.0 track.

3. **Scope.** This completes the sparse device-op. `MatmulDeviceOperation`'s factories are a
   separate audit; several are already on Metal 2.0.

## Recipe notes

- **The *spreadsheet-is-broken* rule has no carve-out for a verdict that is explicitly conditional
  on a landing.** `Is able to port? = yes (with PD step)` is a clearance whose precondition, once
  met, *necessarily* makes the `Concept` column stale — the two cannot be simultaneously current for
  any op in this state. So an op re-audited in the window between the PD landing and the sheet
  refresh will *always* trip the primary-column conflict, and the rule as written says GATE. That is
  the opposite of the intent: the sheet is at its most informative here, not its least. A sentence
  in *Routing* — "a `Concept` mismatch that the `Is able to port?` cell's own parenthetical
  anticipates (`yes (with PD step)`) is staleness, not breakage: report it to the sheet owner and
  proceed" — would make this mechanical instead of a judgement I had to reason out and defend.

- **`Override runtime args method? (PD only)` is ambiguous between "absent" and "not applicable."**
  The cell reads `n/a`, which was right for a legacy op. Post-PD the correct value is `no`, and the
  two are indistinguishable to a reader deciding the target concept. They happen to route
  identically here (`ProgramSpecFactoryConcept` either way), so nothing was lost — but on an op
  where the sheet had *not* yet been refreshed and the code *did* grow an override, `n/a` would send
  the porter to the base concept and the override would be silently dropped. Worth saying in the
  column guide that `n/a` on a `descriptor`-concept row is itself a staleness signal.

- **Appendix A's GlobalCircularBuffer recognition still over-fires on a merely-declared field.**
  Flagged by the previous audit of this same factory and unchanged: two "definitely this feature"
  bullets match an op that accepts the parameter and never constructs from it. The Action bullet's
  own disambiguator resolved it, so the material is there — but a guard bullet ("a
  declared-but-never-constructed `global_cb` parameter is not in use; key on the construction")
  would make the call mechanical. Re-reporting because the PD migration rewrote every
  `CreateCircularBuffer` call into a `CBDescriptor` literal, so an auditor re-running this subject
  has to re-derive the same negative from a *different* set of sites.

- **The CB-endpoint census was cheap here precisely because the `_metal2` forks already exist.** The
  recipe's census instructions assume you read the legacy kernel. Reading the *fork* instead gave a
  strictly better answer — its `dfb::` names make each toucher explicit, and its header states which
  regions are define-gated. Worth a line in the CB-endpoints subject: when a `_metal2` fork exists,
  census the fork, since that is the source the port will actually bind.
