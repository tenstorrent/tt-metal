# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/eltwise/unary`

**Device operations and factories in this directory** (one device-op, one factory — no bundling):

- **`UnaryDeviceOperation`** (`device/unary_device_operation.hpp:21`, `device/unary_device_operation.cpp`)
  - `ProgramFactory` (`device/unary_program_factory.cpp:337` `create_descriptor` · `:570` `override_runtime_arguments`)

`device/unary_composite_op.{hpp,cpp}` is host-side composition over other primitives — no `DeviceOperation`, not in scope.

**Kernels referenced by the factory** (all in scope, all owned by this op):

| Role | File | Selected by |
|---|---|---|
| Reader | `device/kernels/dataflow/reader_unary.cpp` | unconditional (`unary_program_factory.cpp:466`) |
| Writer | `device/kernels/dataflow/writer_unary.cpp` | unconditional (`unary_program_factory.cpp:485`) |
| Compute | `device/kernels/compute/{eltwise_sfpu,eltwise_identity_kernel,where_tss_kernel,mac_tss_kernel,logit_kernel,hardswish_kernel,logsigmoid_kernel,lgamma_kernel,lgamma_fast_kernel}.cpp` | `get_compute_kernel_path(op_chain[0].type(), input.dtype())` — `common/unary_op_utils.cpp:1190`; all nine reachable |

**Unreferenced kernel files in this directory — out of scope, and *not* this op's to touch.** Nine dataflow kernels in `device/kernels/dataflow/` are never instantiated by `UnaryDeviceOperation`; they are *lent* to other op families (`untilize`, `tilize`, `transpose`, `copy`, `concat`, `embedding`, `slice_write`, `topk`, `examples/*`, …). They are listed here only so a reader does not mistake them for this op's surface: `reader_unary_interleaved_start_id.cpp`, `reader_unary_interleaved_start_id_metal2.cpp`, `reader_unary_interleaved_col_multicore.cpp`, `reader_unary_interleaved_wh_multicore.cpp`, `reader_unary_sharded.cpp`, `reader_unary_sharded_metal2.cpp`, `writer_unary_interleaved_start_id.cpp`, `writer_unary_interleaved_start_id_metal2.cpp`, `writer_unary_interleaved_start_id_wh.cpp`.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** *not pinnable.* `git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` prints nothing — the `metal_2.0/` doc tree is staged-but-uncommitted in this checkout. Working tree for reference: branch `anasuya/metal2_port_unary`, HEAD `6e4a9a9b588 2026-09-17 Fix ND-sharded tensors reading an absent 2D shard spec`.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/eltwise/unary` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `UnaryDeviceOperation` → `ProgramFactory` (single) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 11 in-scope kernels structurally Device 2.0; the only CB-index free-function use is `get_local_cb_interface(cb_id)`, which is sanctioned |
| *Prereqs* — Cross-op escapes | Ok — donor includes are `tt_metal/*` and `ttnn/cpp/ttnn/kernel_lib/` only; all consumed signatures take `uint32_t cb_id` (✓) |
| *Feature Support* — overall | **GREEN** — every Appendix A entry `N/A` |
| *Feature Support* — Variadic-CTA | Ok (no Appendix A entry fires; CTA reads are at fixed indices) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** — value supplied by the launching user; the sheet could not be fetched in this session (see *Sheet provenance* below) |
| *TTNN Readiness* — Concept (current) | `descriptor` — verified: `create_descriptor` returns `tt::tt_metal::ProgramDescriptor` (`unary_device_operation.hpp:44`) |
| *TTNN Readiness* — Secretly SPMD | N/A (not `WorkloadDescriptor`) |
| *TTNN Readiness* — Custom hash | **Yes** (not a gate; port leaves it intact): `unary_device_operation.cpp:179` `compute_program_hash`, plus the backdoor `operation_attributes_t::to_hash()` at `:16` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** — no such hook on the device-op (grep clean) |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** (not a gate; selects `CustomProgramSpecFactoryConcept`): `ProgramFactory::override_runtime_arguments` @ `unary_program_factory.cpp:570` (decl `unary_device_operation.hpp:51`) |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** — `unary_nanobind.cpp` binds no `create_descriptor` |
| *TTNN Readiness* — Op-owned tensors | **No** (`descriptor` concept cannot carry them; no `buffers` vector) |
| *TTNN Readiness* — Target concept | **`CustomProgramSpecFactoryConcept`** |
| *Port work* — Offset base pointer | **none** — both `->address()` sites are bare bases, no host-folded offset |
| *Port work* — Tensor bindings (per binding) | `src` Case 1 (interleaved) / clean (sharded) · `dst` Case 1 (interleaved) / clean (sharded) |
| *TTNN Readiness* — TensorParameter relaxation | `dynamic (see analysis)` → **clears**. Analysis doc present; **all five validity checks re-run and now PASS** → verdict **CONFIRMED `dynamic`** |
| *Port work* — TensorAccessor 3rd arg | **none** — no accessor in the op passes a 3rd argument |
| *Port work* — CB endpoints | legal 1:1 for `c_0` and `c_2` in every config · **self-loop** for `c_1` (LOGIT only) · `borrowed_from` for `c_0`/`c_2` under sharding · no dead CBs · no multi-binding |

**CB endpoints** are dispositions, not gates. Recorded per `(CB, config)` below.

### Sheet provenance — read this before consuming the TTNN Readiness rows

The readiness *"Operations analysis"* sheet **could not be fetched**: this is a non-interactive session and the Google Drive connector is unauthenticated, so the OAuth flow cannot run here. Two cells were supplied directly by the launching user and are recorded as authoritative:

- `Is able to port?` = **`yes`**
- `TensorParameter relaxation` = **`dynamic`** (read as the `dynamic (see analysis)` vocabulary — the analysis doc exists and is cited below)

Every other TTNN Readiness row in the table above was derived from **this auditor's own read of the op's code**, which is exactly the cross-check the recipe prescribes; each carries its `file:line`. The cross-column invariants hold: `get_dynamic_runtime_args == no` is consistent with a `descriptor` concept, and `Op-owned tensors == no` is required on `descriptor`. The code has **one** factory, consistent with the single `Is able to port?` value supplied.

**What could not be checked, and why it matters:** the **`Known op issues`** free-text column was not read. The relaxation analysis explicitly warns that this cell is "a second, independent block" that its own document does not clear (`analyses/relaxations/eltwise_unary.md` §1). `Is able to port?` is a *derived* column, so a `yes` should already subsume `Known op issues` — that is the basis on which this audit reads GREEN. See *Questions for the user* #1.

---

## Result

**GREEN → brief issued.** All five gate-bearing subjects clear:

- **Device 2.0** — every kernel the op uses is structurally Device 2.0 (`Noc`, `DataflowBuffer`, `TensorAccessor`, `compute_kernel_lib`). No holdovers, isolated or broad.
- **Feature compatibility** — no Appendix A feature appears anywhere in the op.
- **TTNN factory concept** — `Is able to port?` = `yes`; the primary-column cross-check against the code is clean.
- **Offset base pointers** — the op's only two `->address()` expressions are bare bases with no arithmetic.
- **TensorAccessor 3rd argument** — the subject never fires; no accessor in the op passes a 3rd argument.

The relaxation conjunct also clears, and clears **confirmed**: the analysis doc's blocking validity check has been resolved in the op's code since the analysis was written (detail below).

Port work is modest and mechanical: two Case-1 tensor bindings, one relaxation declaration on both `TensorParameter`s, one compute-kernel self-loop, and `borrowed_from` on the two sharded CBs. The one genuine coordination cost is the shared compute kernel `eltwise_sfpu.cpp` (three external C++ consumers), which needs the `_metal2` fork treatment.

Written for: the Metal 2.0 porting team and the TTNN/eltwise owners.

---

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Cell value `yes`, supplied by the launching user (sheet not fetchable in this session — see *Sheet provenance*). Primary-column cross-check against the code came back clean on every column that has a code-visible counterpart:

  | Column | Sheet-expected shape | Code evidence | Agrees |
  |---|---|---|---|
  | `Concept` | `descriptor` | `create_descriptor` → `ProgramDescriptor`, `unary_device_operation.hpp:44` | ✓ |
  | `Custom hash` | yes | `compute_program_hash`, `unary_device_operation.cpp:179`; `to_hash()`, `:16` | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | no | no hook on the device-op | ✓ |
  | `Override runtime args method?` | yes | `unary_program_factory.cpp:570` | ✓ |
  | `Pybind descriptor` | no | no `create_descriptor` binding in `unary_nanobind.cpp` | ✓ |
  | `Op-owned tensors?` | no | `descriptor` concept; no `buffers` vector | ✓ |
  | Factory-set match | 1 row | `using program_factory_t = std::variant<ProgramFactory>;` (`unary_device_operation.hpp:59`), no `select_program_factory` | ✓ |

  Note the name-collision guard: `override_runtime_arguments` here sits on a **`descriptor`** op, so it is the *target-concept* signal (→ `CustomProgramSpecFactoryConcept`), **not** the legacy `create()` + `override_runtime_arguments()` signature. The factory defines `create_descriptor`, not `create()`.

- **Device 2.0 (every kernel used):** **GREEN.** All eleven in-scope kernels are structurally Device 2.0.

  - `reader_unary.cpp` / `writer_unary.cpp`: `Noc`, `DataflowBuffer`, `TensorAccessor`, `noc.async_read`/`async_write` with `NocOptVals`-style option structs. Includes are `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/tensor/noc_traits.h` — no `api/dataflow/circular_buffer.h`.
  - Compute kernels: `eltwise_sfpu.cpp` and `mac_tss_kernel.cpp` drive their FIFOs through `DataflowBuffer` objects; the other seven go through `compute_kernel_lib`'s chain surface. None uses a legacy `cb_wait_front` / `cb_reserve_back` / `cb_push_back` / `cb_pop_front` free function.
  - **No** `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedAddrGenFast`, `InterleavedPow2AddrGen*`, raw `noc_async_read`/`noc_async_write`, raw semaphore addresses, or manual CB index management anywhere in scope.

  **Two sanctioned free-function sites — recorded, not flagged:**

  | File | Line | Call | Wrapper in scope | Disposition |
  |---|---|---|---|---|
  | `device/kernels/dataflow/reader_unary.cpp` | 57 | `get_local_cb_interface(cb_id_src).fifo_page_size` | `DataflowBuffer dfb_src` | **Sanctioned** — on the Green-bullet list; not a holdover |
  | `device/kernels/dataflow/writer_unary.cpp` | 60 | `get_local_cb_interface(cb_id_dst).fifo_page_size` | `DataflowBuffer dfb_dst` | **Sanctioned** — same |

  These are exactly the case the recipe calls out as misfiring hardest: a `DataflowBuffer` *is* in scope and the DFB *does* expose its own accessors, but `get_local_cb_interface(cb_id)` stays sanctioned regardless of what object is at the call site. Not a Device 2.0 violation, and not a re-audit trigger. They are, however, a **port-stage** change — see the breadcrumb in *Heads-ups*.

- **Feature compatibility:** every Appendix A entry, in order. No entry's recognition signals fire anywhere in the op (host code, kernels, factory, descriptors, nanobind).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | **N/A** | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `global_circular_buffer` field on any `CBDescriptor`, no `remote_index(` / `remote_cb_*` / `remote_circular_buffer.h`, no 4-arg `experimental::CreateCircularBuffer(..., global_cb)`, neither include spelling. The three `CBDescriptor` literals (`unary_program_factory.cpp:420`, `:432`, `:444`) set only `total_size`, `core_ranges`, `format_descriptors`, `buffer`. |
  | CBDescriptor `address_offset` (non-zero) | **N/A** | The token `address_offset` does not appear in the op at all — no field set, no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. The two Buffer-backed CBs (`:428`, `:452`) set `.buffer` only, leaving `address_offset` at its default zero — the borrowed-memory pattern, which is a mechanical porting-recipe translation and not this entry. |
  | GlobalSemaphore | **N/A** | The op uses **no semaphores of any kind** — neither `GlobalSemaphore` nor plain `Semaphore`. No `global_semaphore.hpp` include, no `CreateGlobalSemaphore`. |

- **CB endpoints (GATE-free):** three CBs, censused per `(CB, config)` per node. **No dead CB, no multi-binding, no conditional-DFB retrofit.** Device 2.0 is GREEN and the idioms the scan keys on are intact, so this is a real census, not a deferral.

  The configuration axes are two compile-time defines plus one op-chain predicate, none of which can flip on a cache hit (`has_sharding` is pinned by the hashed shard-volume optionals — see the relaxation analysis validity check 3, re-confirmed below):

  - **INT-TILE** — `has_sharding = false`, `RM_INTERLEAVED = 0`
  - **INT-RM** — `has_sharding = false`, `RM_INTERLEAVED = 1`
  - **SHARDED** — `has_sharding = true` ⇒ `SRC_SHARDED = DST_SHARDED = 1` (TILE or ROW_MAJOR)
  - **+LOGIT** — orthogonal overlay: `op_chain[0].type() == LOGIT` adds `c_1` and selects `logit_kernel.cpp`

  | CB | Config | Touchers on a node | Roles | Verdict | Port-time disposition |
  |---|---|---|---|---|---|
  | `c_0` (src0) | INT-TILE, INT-RM | reader `reserve_back`/`push_back`; compute `wait_front`/`pop_front` | 1 locked P + 1 locked C | **plain 1:1** | none — bind reader PRODUCER, compute CONSUMER |
  | `c_0` (src0) | SHARDED | same two | 1 locked P + 1 locked C | **plain 1:1** | additionally `DataflowBufferSpec::borrowed_from` the input `TensorParameter` (legacy `.buffer = src_buffer`, `unary_program_factory.cpp:428`) |
  | `c_1` (tmp0) | +LOGIT only | `logit_kernel.cpp` **alone** — it both `PackTile`s into `dfb_tmp0_id` (`logit_kernel.cpp:41-45`) and `CopyTile`s out of it (`:49-56`) | 1 toucher, locked to *both* roles | **self-loop** | bind the compute kernel **PRODUCER and CONSUMER** (legal on Gen1 for compute) |
  | `c_2` (output) | INT-TILE, INT-RM | compute `reserve_back`/`push_back` (or `PackTile`); writer `wait_front`/`pop_front` | 1 locked P + 1 locked C | **plain 1:1** | none — bind compute PRODUCER, writer CONSUMER |
  | `c_2` (output) | SHARDED | same two | 1 locked P + 1 locked C | **plain 1:1** | additionally `borrowed_from` the output `TensorParameter` (legacy `.buffer = dst_buffer`, `unary_program_factory.cpp:452`) |

  **Hidden-second-writer hunt: run, came back empty.** No kernel in scope calls `get_write_ptr()`, `get_read_ptr()`, `fifo_wr_ptr`, `fifo_rd_ptr`, `evil_set_write_ptr` or `evil_set_read_ptr`. The op has no semaphores at all, so the semaphore-gated raw co-fill shape has nothing to hang on. No dual-instance work-split either: each of the three `KernelDescriptor`s carries a distinct `kernel_source` and there is no same-source pair differing only by `ReaderConfigDescriptor` / `WriterConfigDescriptor`.

  **`c_1` is not a "conditional DFB" retrofit.** The legacy factory already allocates it conditionally (`if (needs_tmp0_cb(op_chain[0].type()))`, `unary_program_factory.cpp:431`), gated by the same predicate that selects `logit_kernel.cpp`. The port mirrors the existing conditional; there is no unconditionally-allocated CB that is dead under some config.

- **Offset base pointers:** **GREEN.** `eltwise/unary` is **not** in the triage tables of `analyses/2026-07-19_offset_base_pointers.md` — and per the recipe that fact was not allowed to stand in for a scan. Every address-bearing site was resolved independently:

  | Site | Expression | Fold? | Consumed as | Verdict |
  |---|---|---|---|---|
  | `unary_program_factory.cpp:585` | `input.buffer()->address()` | **no** — bare, no arithmetic | written verbatim to reader RTA slot 0 on the cache-hit path | clean base → TensorParameter analysis |
  | `unary_program_factory.cpp:586` | `output.buffer()->address()` | **no** — bare, no arithmetic | written verbatim to writer RTA slot 0 | clean base → TensorParameter analysis |
  | `unary_program_factory.cpp:531,536,555` | `input.buffer()` (`Buffer*`, not `->address()`) | n/a — the pointer object, framework resolves the base | reader RTA slot 0 on the cache-miss path | clean base |
  | `unary_program_factory.cpp:532,546,557` | `output.buffer()` (`Buffer*`) | n/a | writer RTA slot 0 on the cache-miss path | clean base |

  The op's other `buffer()` uses are metadata reads, not addresses: `page_size()` (`:176-177`), `num_pages()` (`:186`), `aligned_page_size()` (`:190`). **Reconciliation outcome: "no fold, op not in the tables" → clean.** Type 3 (`address_offset`) is absent (Appendix A row above). Type 4 is absent: no `ttnn::narrow`, no `MeshBuffer::create(…, parent_base + offset)`.

- **TensorAccessor 3rd argument:** **N/A — the subject never fires.** No accessor in the op passes a 3rd argument. The op constructs exactly two `TensorAccessor`s, both two-argument:

  - `reader_unary.cpp:26` — `TensorAccessor(src_args, src_addr)`
  - `writer_unary.cpp:28` — `TensorAccessor(dst_args, dst_addr)`

  This is *no sites found*, not *sites found and classified redundant*. `eltwise/unary` is likewise absent from `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`, which is consistent, but the verdict rests on the syntactic scan above, not on the table's silence.

  Related but distinct: the host builds both accessors' args with `tensor_accessor::ArgConfig::RuntimeTensorShape` (`unary_program_factory.cpp:462`, `:481`). That is the *runtime-shape* arg config, not a page-size override, and it is precisely the legacy shape the `dynamic_tensor_shape` relaxation replaces — see *Port-work summary*.

---

## TensorParameter relaxation — verdict **CONFIRMED `dynamic`**

The sheet cell (user-supplied) is `dynamic`, reading onto the `dynamic (see analysis)` row. The analysis doc exists at `analyses/relaxations/eltwise_unary.md` and covers exactly this op/device-op/factory triple.

That doc is **authoritative but perishable**, and it carries five validity checks that must all pass before anything in it may be applied. **At the time it was written, validity check 1 FAILED, and the doc instructs the auditor to report UNCONFIRMED while that gap is open.** All five were re-run against the current code:

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | The cache key pins `tensor_layout` — no two tensors differing in dtype, page config (incl. `Tile`), memory config or `Alignment` may share a cache entry | **PASS** (was FAIL) | `compute_program_hash` now hashes `input_tensor.tensor_spec().tensor_layout()` **and** `output_spec.tensor_layout()` (`unary_device_operation.cpp:217-218`). `TensorLayout::attribute_values()` is `(dtype, page_config, memory_config, alignment)` (`tt_metal/api/tt-metalium/tensor/spec/layout/tensor_layout.hpp:74-75`) and `PageConfig` carries `TilePageConfig{Tile tile}` (`page_config.hpp:23-26,50-51`), so `Tile` and `Alignment` are both pinned, on both slots. |
| 2 | The dataflow kernels compile the accessor away when the slot is sharded | **PASS** | `TensorAccessorArgs<0,0>()` and `TensorAccessor` sit inside the `#else` of `#if SRC_SHARDED` (`reader_unary.cpp:20,25-26`); `writer_unary.cpp:86,93-94` mirrors it under `DST_SHARDED`. |
| 3 | `has_sharding` is itself pinned by the key | **PASS** | `src_shard_vol` / `dst_shard_vol` are both hashed (`unary_device_operation.cpp:223-224`), and both are set iff `get_shard_specs(...)` returned a value (`:188-195`). `get_shard_specs` returns a value only when `is_native_L1_sharding` holds, which requires **both** slots sharded (`common/unary_utils.cpp:33-39,66`), so `src_sharded == dst_sharded == has_sharding` and the compile-time defines cannot flip on a hit. |
| 4 | The op still has exactly one factory | **PASS** | `using program_factory_t = std::variant<ProgramFactory>;` (`unary_device_operation.hpp:59`); no `select_program_factory`. |
| 5 | The TILE-path key still omits shape, and the override still re-applies the whole split | **PASS** | `padded_shape` is hashed only on the `ROW_MAJOR` branch (`unary_device_operation.cpp:222`); `override_runtime_arguments` re-enumerates through the same `enumerate_core_rt_args` the miss path uses (`unary_program_factory.cpp:596`, shared with `:521`). |

**Check 1 was resolved by exactly the one-line route the doc names.** The doc prescribed "hashing `input_tensor.tensor_spec().tensor_layout()` in place of the present `dtype()` / `layout()` / `memory_config()` triple"; the current key does that for the input *and* adds an independent `output_spec.tensor_layout()` term, with an in-code comment at `unary_device_operation.cpp:197-213` explaining the Metal 2.0 relaxation requirement that motivated it. The op-code change landed after the analysis was written.

**Therefore the relaxation verdict is `dynamic`, CONFIRMED**, and the doc's §2 declaration applies as written. Its owner should be told so the "report UNCONFIRMED" paragraph in §1 can be retired — see *Recipe notes*.

The declaration the doc specifies, for **both** `TensorParameter`s (input and output), unconditionally:

```cpp
.relaxations = {.dynamic_tensor_shape = true, .relax_logical_rank = true},
```

with three explicit prohibitions carried forward from §2: **do not** set `match_page_size` (declined on precedent grounds — no shipped factory sets it), **do not** set `match_padded_shape_only` (strictly weaker), and treat a validation throw during the port as a plausible framework-side gap rather than automatically a mistake in the declaration (unary would be the first *shipped* factory to declare any relaxation, and `relax_logical_rank` has no shipped precedent at all).

**The doc's one stop condition, carried into the brief verbatim in substance:** if the porter is working a configuration where the input tensor is **sharded but the op took the interleaved code path** (doc §3 row 5), **stop and ask**. That is the one regime where the accessor is live over a sharded buffer, so the declaration's geometry term does real work instead of pinning dead code. It is plainly reachable: `get_shard_specs` returns `nullopt` on three documented fallbacks — `is_native_L1_sharding` failing (DRAM, mismatched grids, uneven input), an uneven output, and a ROW_MAJOR shard whose element count is not tile-aligned, that last one emitting a `log_warning` (`common/unary_utils.cpp:66,73-91`).

**Relaxation candidates mined from the custom hash (FYI-U, fallible):** none worth routing. The hash is, if anything, *over*-pinned rather than under-pinned now — see *Misc anomalies* #6. The one deliberate omission (shape on the TILE path) is the very thing `dynamic_tensor_shape` exists to cover, so it is already accounted for rather than a candidate.

---

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per config — the classification splits with config and both halves are real):

  | Binding | Config | Class | Detail |
  |---|---|---|---|
  | `src` (input) | INT-TILE, INT-RM | **Case 1** | Base delivered as `Buffer*` in reader RTA slot 0 on miss (`unary_program_factory.cpp:531,536,555`) and as a raw `uint32_t` on hit (`:613`); the kernel feeds it to `TensorAccessor(src_args, src_addr)` (`reader_unary.cpp:11,26`) and does all access through the accessor. → express as `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(tensor::src)`; the RTA slot and the `TensorAccessorArgs` plumbing at `:462-463` both disappear. |
  | `src` (input) | SHARDED | **clean** | Borrowed-memory DFB read — `c_0` is backed by `src_buffer` (`:428`) and the accessor is compiled out by `#if SRC_SHARDED`. The DFB *is* the tensor access. → `DataflowBufferSpec::borrowed_from`. Not Case 1, not Case 2. |
  | `dst` (output) | INT-TILE, INT-RM | **Case 1** | Symmetric: `Buffer*` on miss (`:532,546,557`), raw on hit (`:616`), consumed via `TensorAccessor(dst_args, dst_addr)` (`writer_unary.cpp:28`). |
  | `dst` (output) | SHARDED | **clean** | Borrowed-memory DFB — `c_2` backed by `dst_buffer` (`:452`). |

  **Roll-up: ⚠ port work** (two Case-1 bindings). Neither is the silent-wrong hazard: on the miss path the base arrives as a `Buffer*` (framework-resolved), and on the hit path this op defines `override_runtime_arguments`, which the adapter uses *instead of* `resolve_bindings` and `get_dynamic_runtime_args` (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:451-457,678-688`) — so the factory itself re-writes the address every hit. Both are routine port work; the typed binding supersedes both mechanisms.

- **TensorParameter relaxation:** `.relaxations = {.dynamic_tensor_shape = true, .relax_logical_rank = true}` on **both** `TensorParameter`s, unconditionally. Do **not** set `match_page_size` or `match_padded_shape_only`. Source: `analyses/relaxations/eltwise_unary.md` §2, verdict CONFIRMED above.

- **TensorAccessor 3rd arg:** none — no site.

- **CB endpoints:** self-loop `c_1` (LOGIT config only) · `borrowed_from` on `c_0` and `c_2` (SHARDED config only) · `c_0` and `c_2` plain 1:1 everywhere · no multi-binding flag anywhere · no dead-CB drop · no conditional-DFB retrofit (`c_1`'s conditional already exists in the legacy factory).

- **Runtime args → named args:** every RTA and CRTA is nameable; no varargs mechanism needed. Reader/writer slots: `0` = tensor base (**disappears** into the binding), `1` = `num_pages`, `2` = `start_id`, `3` = `chunks_per_row`, `4` = `chunk_size`, `5` = `last_chunk_size`, `6` = `rows_per_tile`, `7` = `total_rows`. Compute slots: `0` = `num_tiles`, `1` = `packed_scalar1`, `2` = `packed_scalar2`. The common runtime args are the `TensorAccessorArgs` payload only (`:472`, `:491`), which the framework auto-builds from the binding in Metal 2.0 and which therefore also disappears — along with its bespoke cache-hit refresh at `:638-653`.

- **Cache-hit CB-address patch disappears:** the `apply_descriptor_runtime_args(program, cb_addr_only)` block at `unary_program_factory.cpp:657-665` exists only to re-point the two tensor-backed CBs on a hit. Under `borrowed_from` the framework owns that, so the block — and the positional-CB-matching hazard its comment describes at `:655-656` — goes away.

---

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. The hidden-second-writer and multiple-reader hunts were both run and came back empty (no raw pointer access anywhere in scope, no semaphores, no dual-instance work-split).

- **Cross-op / shared kernels — the one real coordination cost.** The op **borrows no kernel files**; it instantiates only sources it owns. But it **lends** one:

  - `device/kernels/compute/eltwise_sfpu.cpp` is file-path-instantiated by **three external C++ factories** — `operations/examples/example/device/single_core_program_factory.cpp:91`, `operations/examples/example/device/multi_core_program_factory.cpp:89`, `operations/examples/example_multiple_return/device/single_core_program_factory.cpp:80` — plus **two test consumers**, `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:246` and `tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py:1436`.
  - **No `_metal2` fork exists beside it** — `device/kernels/compute/` contains no `_metal2` file. This port creates the first one.
  - The other eight compute kernels and both dataflow kernels (`reader_unary.cpp`, `writer_unary.cpp`) are **unary-exclusive** — no external consumer — so they convert in place with no fork.

  The consumer list above is a **sunset list**, not authorization to convert the kernel in place.

- **`_metal2` name-adjacency trap in this very directory.** `device/kernels/dataflow/` contains `reader_unary_interleaved_start_id_metal2.cpp`, `reader_unary_sharded_metal2.cpp` and `writer_unary_interleaved_start_id_metal2.cpp`. **None of these is a fork of `reader_unary.cpp` or `writer_unary.cpp`** — they are forks of the *other*, similarly-named kernels in the same directory, which this op does not use. The fork test is per-stem, not per-directory.

- **…but they are an excellent local precedent, and they are not quasar code.** These three are checked-in, shipped, non-experimental Metal 2.0 kernels sitting in the directory the porter will be editing. They demonstrate the target idioms directly: `dfb::in` / `dfb::out` and `tensor::src` / `tensor::dst` binding tokens, `get_arg(args::name)` named args via `experimental/kernel_args.h`, `TensorAccessor(tensor::src)` construction, and the fork-note comment convention. `writer_unary_interleaved_start_id_metal2.cpp:13-19` also records a live duplication issue (#52228) worth reading before creating a new fork.

- **Device 2.0 → Metal 2.0 breadcrumb — confirm, don't swap blind.** `get_local_cb_interface(cb_id_src).fifo_page_size` (`reader_unary.cpp:57`) and `get_local_cb_interface(cb_id_dst).fifo_page_size` (`writer_unary.cpp:60`) are sanctioned Device-2.0 idioms that the *port* moves onto the DFB object. The established in-tree equivalent is `dfb.get_entry_size()` — used for exactly this purpose by both neighbouring `_metal2` forks (`reader_unary_interleaved_start_id_metal2.cpp:53`, `writer_unary_interleaved_start_id_metal2.cpp:37`), each with a comment noting it works for both TILE and ROW_MAJOR. `DataflowBuffer::get_entry_size()` is declared at `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:113`.

- **`constexpr`-vs-`const` at the CB-id declarations.** Both dataflow kernels declare `constexpr auto cb_id_src = tt::CBIndex::c_0;` / `cb_id_dst = tt::CBIndex::c_2;` (`reader_unary.cpp:15`, `writer_unary.cpp:81`), and the compute kernels declare `constexpr auto dfb_input_id` / `dfb_output_id` / `dfb_tmp0_id` the same way. These are the sites that become `dfb::` tokens; the `constexpr` spelling is what keeps the `compute_kernel_lib` NTTP uses valid (below).

- **The compute kernels pass CB ids in NTTP position, through a struct.** Seven of the nine compute kernels build `compute_kernel_lib` chains where the CB id travels inside an `InputSpec` / `OutputSpec` used as a *non-type template parameter* — e.g. `ckl::CopyTile<ckl::input(dfb_input_id, ckl::WaitPolicy::PerTile, …), ckl::Dst::D0>{}`. `ckl::input` / `ckl::output` are `constexpr` and take `uint32_t cb_id` (`ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:356,383`), and `DFBBindingToken::operator uint32_t()` is `constexpr` (`tt_metal/hw/inc/api/dataflow/dfb_binding_token.h:36`), so the substitution is a constant expression and is expected to compile. It is nonetheless a deeper NTTP nesting than the plain `template<uint32_t cb>` case the shape table contemplates — worth a compile check early rather than at the end.

- **RTA varargs:** none — every arg is nameable.

- **Non-32×32 tiles: a live, family-wide bug that does *not* gate this port and must *not* be fixed in it.** `create_descriptor` sizes both CBs with `tile_size(DataFormat)` (`unary_program_factory.cpp:356,358`), which assumes 32×32, while `enumerate_core_rt_args` reads the tensor's *real* tile for the work split (`:167-169`). Confirmed on silicon in the relaxation analysis §4 (`ttnn.relu` on a 16×32-tile bf16 tensor returns wrong data in isolation). It does not gate because `copy/typecast` — **already ported** — ships the identical defect. Fixing it inside the port would violate the zero-functional-change invariant. Routed to the eltwise team below.

---

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.** No donor requires work, and no donor sits on pre-Device-2.0 idioms. Concretely: every escape lands in bucket 1 (`tt_metal/*`) or bucket 2 (`ttnn/cpp/ttnn/kernel_lib/`), and every consumed signature that carries a resource handle takes a plain `uint32_t cb_id` — the ✓ row of the shape table. No `Semaphore`, no `uint32_t sem_id`, no `sem_addr`, no `TensorAccessorArgs<N>` parameter, no NTTP CTA offset, no old-style addr-gen, no `CircularBuffer&`, no `DataflowBuffer&` donor parameter appears anywhere in scope.

**Summary table** — one row per (op kernel, donor file/bucket):

| Op kernel | Donor | Bucket | Status |
|---|---|---|---|
| `reader_unary.cpp` | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/tensor/noc_traits.h` | 1 — `tt_metal/*` | ✓ |
| `writer_unary.cpp` | same four | 1 — `tt_metal/*` | ✓ |
| `eltwise_sfpu.cpp` | `api/compute/*` (7 headers), `api/dataflow/dataflow_buffer.h` | 1 — `tt_metal/*` | ✓ |
| `mac_tss_kernel.cpp` | `api/compute/*` (6 headers), `api/dataflow/dataflow_buffer.h` | 1 — `tt_metal/*` | ✓ |
| `eltwise_identity_kernel.cpp` | `api/compute/compute_kernel_hw_startup.h`; `kernel_lib/eltwise/api/{chain,convenience}.hpp` | 1, 2 | ✓ |
| `hardswish_kernel.cpp` | `api/compute/compute_kernel_hw_startup.h`; `kernel_lib/eltwise/{api/chain,unary/activations,binary/sfpu/basic,core/optional}.hpp` | 1, 2 | ✓ |
| `logit_kernel.cpp` | `api/compute/compute_kernel_hw_startup.h`; `kernel_lib/eltwise/{api/chain,unary/math,unary/scalar,binary/sfpu/basic,core/optional}.hpp` | 1, 2 | ✓ |
| `logsigmoid_kernel.cpp` | `api/compute/compute_kernel_hw_startup.h`; `kernel_lib/eltwise/{api/chain,unary/math,unary/misc,unary/activations}.hpp` | 1, 2 | ✓ |
| `where_tss_kernel.cpp` | `api/compute/compute_kernel_hw_startup.h`; `kernel_lib/eltwise/{api/chain,unary/special,generators/fill,core/optional}.hpp` | 1, 2 | ✓ |
| `lgamma_kernel.cpp` | `api/compute/compute_kernel_hw_startup.h`; `kernel_lib/eltwise/{api/chain,generators/fill,binary/sfpu/basic,binary/sfpu/extended,unary/predicates,unary/special,unary/trig,unary/rounding,unary/misc,unary/math}.hpp` | 1, 2 | ✓ |
| `lgamma_fast_kernel.cpp` | same as `lgamma_kernel.cpp` minus `unary/predicates.hpp` | 1, 2 | ✓ |

**Per-call detail:** omitted — all rolls are ✓, per the report format. The one shape worth naming explicitly, because it recurs across seven kernels: `compute_kernel_lib`'s operand binders `constexpr InputSpec input(uint32_t cb_id, …)` and `constexpr OutputSpec output(uint32_t cb_id, …)` (`kernel_lib/eltwise/api/chain.hpp:356,383`) — the `uint32_t cb_id` ✓ row, in NTTP position (see the *Heads-ups* note). The bucket-1 Gen1 compute LLKs used directly (`compute_kernel_hw_startup`, `copy_init`, `copy_tile`, `pack_tile`, `fill_tile`, and whatever `SFPU_OP_CHAIN_0` expands to) likewise take raw CB ids and are covered by the same constexpr cast.

**Borrowed kernel files (file-path instantiation):** **none.** Every kernel the factory instantiates is owned by this op directory. The inverse coupling — `eltwise_sfpu.cpp` lent to three external factories and two tests, no `_metal2` fork beside it — is recorded in *Heads-ups*.

**No quasar entanglement.** There is no `unary` directory under `ttnn/cpp/ttnn/operations/experimental/quasar/`, so no quasar copy of this op exists to be mistaken for a precedent. The three `_metal2` kernels in this op's own directory are production, non-quasar forks and are safe to read.

### TTNN factory analysis

Sheet-derived facts with `file:line` evidence, for the port's TTNN ProgramFactory wiring:

- **Current concept:** `descriptor` — `ProgramFactory::create_descriptor` returns `tt::tt_metal::ProgramDescriptor` (`unary_device_operation.hpp:44`, `unary_program_factory.cpp:337`).
- **Target concept:** **`CustomProgramSpecFactoryConcept`**, selected by `Override runtime args method? == yes`. The porter translates `override_runtime_arguments` (`unary_program_factory.cpp:570-666`) into one returning a `ProgramRunArgs`; the method owns the entire cache-hit refresh today and must continue to.
- **Op-owned tensors:** none — and structurally impossible on a `descriptor` concept.
- **MeshWorkload need:** none. Single-program; `create_descriptor` takes and returns no mesh structure. The `mesh_dispatch_coordinate` parameter on `override_runtime_arguments` is accepted and unused (`unary_program_factory.cpp:575`).
- **Custom hash:** present and **stays as-is**. `compute_program_hash` (`unary_device_operation.cpp:179-225`) plus the backdoor `operation_attributes_t::to_hash()` (`:16-26`). The hash's own comments at `:197-213` were written with the Metal 2.0 relaxation contract in mind and should be read before touching anything nearby.
- **`get_dynamic_runtime_args`:** absent. (Consistent — the adapter `static_assert`s mutual exclusion with `override_runtime_arguments`, and this op has the latter.)
- **Pybind `create_descriptor`:** absent. No user-visible API deletion in this port.
- **Other risky pybind:** none found. `unary_nanobind.cpp` is 2177 lines of ordinary op bindings with no descriptor or internals exposure.
- **Gate conjuncts confirmed absent:** a `TensorParameter relaxation` value that neither clears nor points at an analysis; `get_dynamic_runtime_args`; genuine multi-program.

---

## Misc anomalies  *(team-only, non-gating; route to the eltwise / ops team — the port does not act on these)*

1. **Dead compute CTA on every op type.** `unary_program_factory.cpp:506` appends `static_cast<uint32_t>(cb_data_format)` to **every** compute kernel's compile-time args. No in-scope compute kernel reads it: only `logit_kernel.cpp:16` (CTA 0) and `hardswish_kernel.cpp:14-15` (CTA 0, 1) call `get_compile_time_arg_val` at all, and neither reaches the trailing slot. The `get_block_defines` expansions do not reference CTAs either (no `get_compile_time_arg_val` anywhere in `common/unary_op_utils.cpp`). Either the value is vestigial or a kernel that once consumed it has been rewritten.

2. **Dead reader/writer RTAs under `SHARDED`.** The factory passes three args (`unary_program_factory.cpp:531-532`) but the sharded branch of each kernel reads only `num_pages` — `src_addr` (slot 0, `reader_unary.cpp:11`) and `start_id` (slot 2, `:13`) are unpacked and never used under `#if SRC_SHARDED`; `writer_unary.cpp:77,79` likewise under `DST_SHARDED`. Harmless; the port dissolves slot 0 into the binding regardless.

3. **Dead reader/writer RTA tail under `INT-TILE`.** Slots 3–7 are written as literal zeros (`unary_program_factory.cpp:555-557`) and never read when `RM_INTERLEAVED == 0`. This one is *deliberate* — the uniform 8-slot layout is what lets `override_runtime_arguments` write every slot a flipped core might need (see the comment at `:590-591`) — so it is noted for completeness, not as a defect.

4. **Dead compute RTA scalars on most op types.** Compute slots 1–2 (`packed_scalar1`/`2`) are populated for every op (`unary_program_factory.cpp:560`) but read only by `logit_kernel.cpp`, `where_tss_kernel.cpp` and `mac_tss_kernel.cpp`. Same uniform-schema rationale as #3.

5. **`unpack_to_dest_mode` set for a CB that may not exist.** `unary_program_factory.cpp:403` sets `unpack_to_dest_mode[tmp0_cb_index]` whenever `preserve_fp32_precision` is on, but `c_1` is only allocated for LOGIT (`:431`). Benign — the entry is simply ignored — but it reads as though the tmp0 CB were unconditional.

6. **Over-pinned hash terms.** `to_hash()` hashes both `sub_core_grids` and `worker_grid` (`unary_device_operation.cpp:24-25`), but `worker_grid` is *derived* from `sub_core_grids` and the tensors by `get_worker_grid` (`unary_device_operation.cpp:253-258`). Separately, `output_dtype` and `memory_config` are hashed via `attributes` while `output_spec.tensor_layout()` (`:218`) already carries both. Over-pinning is the safe direction — it can only split cache entries that could have been shared, never merge distinct ones — but it is a latent program-cache-pressure cost worth a look now that the `tensor_layout` terms have been added.

7. **Silently-truncating accessor-refresh loops.** `unary_program_factory.cpp:643` and `:651` bound the copy by `i < common_args.size() && i < reader_common.size()`. If a fresh tensor ever needs *more* accessor words than the cached kernel's common-arg buffer holds, the extra words are dropped rather than caught. Under `dynamic_tensor_shape` the word count varies with rank on sharded slots. The relaxation analysis §3 flags this and records that reachability was **not** established (the distribution spec squeezes rank before that point, which may preclude it). The `&&` guard would mask it either way. Worth resolving independently of the port — though note the port deletes these loops, so the question becomes moot for unary specifically once ported.

8. **Non-32×32 tile mis-sizing — a confirmed live bug.** See the *Heads-ups* entry. `create_descriptor` sizes CBs from `tile_size(DataFormat)` (32×32 assumption) while the work split reads the real tile. Per the relaxation analysis §4 this spans at least `eltwise/unary`, `eltwise/binary_ng` and `copy/typecast`, and `data_movement/untilize` has the same split personality. It belongs to the eltwise team as one issue, and it explicitly does not gate any of these ports.

---

## Questions for the user  *(2)*

1. **`Known op issues` was not read.** The readiness sheet could not be fetched in this non-interactive session, so the `Known op issues` free-text cell for `eltwise/unary` is unverified. The relaxation analysis warns specifically that this cell is "a second, independent block" its own document does not clear (`analyses/relaxations/eltwise_unary.md` §1). This audit reads GREEN on the basis that `Is able to port?` is a **derived** column and the `yes` you supplied should already subsume it. **Please confirm the `Known op issues` cell is empty** (or tell me what it says) before the porter starts. If it is non-empty, this audit's Result flips to RED pending that item, with nothing else in the report changing.

2. **Relaxation-doc owner should be told check 1 now passes.** `analyses/relaxations/eltwise_unary.md` §1 currently instructs auditors to report the verdict UNCONFIRMED while validity check 1 is open. That check now passes — the op's `compute_program_hash` hashes `tensor_layout()` for both slots (`unary_device_operation.cpp:217-218`), which is exactly the resolution §1 prescribed. Until the doc is updated, the next auditor of this op will read a stale STOP instruction. Who owns that doc?

---

## Recipe notes  *(4)*

1. **Out-of-directory coupling is framed one-directionally, and the cost can point the other way.** The *Borrowed kernel files (file-path kernel instantiation)* subject asks for "every kernel `.cpp` file the op's program factory instantiates whose source it does **not** own." For `eltwise/unary` that list is empty — yet the op has a real, reportable shared-kernel coordination cost, because it *owns* a kernel (`eltwise_sfpu.cpp`) that three external factories instantiate. The `_metal2` fork machinery and the sunset-list framing apply identically in the lender direction, but nothing in the subject's wording asks the auditor to look that way. A strict reading would have produced "✓ no borrowed kernels" and silently dropped the finding. **Suggest:** add a sentence directing the auditor to also grep for external instantiations of the op's *own* kernel files.

2. **The `_metal2` fork check is described locationally in a way that invites a false positive.** The rule reads: "Whether a **`_metal2` fork already exists beside it** (same stem, `_metal2` suffix, same directory)." The stem condition is stated, but the directory framing dominates on a skim — and `eltwise/unary`'s `device/kernels/dataflow/` is a live trap: it contains three `*_metal2.cpp` files, none of which forks either kernel this op actually uses. **Suggest:** make the per-stem test the emphasised clause, e.g. "the fork of `X.cpp` is `X_metal2.cpp` — a `_metal2` file with a *different* stem in the same directory forks a different kernel and is not yours."

3. **The relaxation subject has no branch for "the analysis doc's validity check was failing and has since been fixed."** The subject says the doc "carries its own validity check: a short list of invariants. **Run it. If it fails, report the relaxation verdict as UNCONFIRMED**." It does not say what to do in the case actually encountered here: the doc *documents its own check as failing* and instructs the reader to report UNCONFIRMED, but the op's code has since changed and the check now passes. Reading the doc's §1 literally yields UNCONFIRMED; running the check yields CONFIRMED. I took the latter — the instruction is to *run* the check, and a doc that tells you its own finding is conditional on a check has delegated the outcome to the check. But the recipe's "the doc is authoritative, your own read is not the fallback" language points the other way and made this a real judgement call. **Suggest:** one sentence distinguishing *re-running the doc's stated invariants* (always do this; the fresh result wins) from *deriving a relaxation* (never do this).

4. **Minor factual drift in the relaxation analysis, noted for its owner rather than the recipe.** `analyses/relaxations/eltwise_unary.md` §4 lists as not-covered "The independent Quasar clone under `experimental/quasar/`." There is no `unary` directory under `ttnn/cpp/ttnn/operations/experimental/quasar/` in this checkout — the sibling clones that do exist are `binary`, `binary_ng`, `typecast` and others. Harmless, but it sent me looking for a clone that does not exist.
