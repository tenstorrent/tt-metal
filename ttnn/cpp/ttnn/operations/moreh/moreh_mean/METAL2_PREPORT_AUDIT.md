# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_mean`

One `DeviceOperation`, three program factories — all in this directory, all on the `ProgramDescriptor` API:

- **`MorehMeanOperation`** (`device/moreh_mean_device_operation.{hpp,cpp}`)
  - `MorehMeanHFactory` (`device/moreh_mean_h_program_factory.cpp`) — reduce over `dim == rank-2`
  - `MorehMeanNCFactory` (`device/moreh_mean_nc_program_factory.cpp`) — reduce over any outer dim
  - `MorehMeanWFactory` (`device/moreh_mean_w_program_factory.cpp`) — reduce over `dim == rank-1`

Factory selection is by reduced-dim position (`moreh_mean_device_operation.cpp:34-47`), so exactly one factory runs per invocation and the three are mutually exclusive.

**Kernels (8, all owned by this op — no borrowed kernel files):**

| Factory | Reader | Writer | Compute |
|---|---|---|---|
| H | `kernels/reader_moreh_mean_h.cpp` | `kernels/writer_moreh_mean_unary_interleaved_start_id.cpp` | `kernels/moreh_mean_h.cpp` |
| W | `kernels/reader_moreh_mean_w.cpp` | `kernels/writer_moreh_mean_unary_interleaved_start_id.cpp` | `kernels/moreh_mean_w.cpp` |
| NC | `kernels/reader_moreh_mean_nc.cpp` | `kernels/writer_moreh_mean_nc.cpp` | `kernels/moreh_mean_nc.cpp` |

No unreferenced kernel files in the directory; every kernel file is instantiated by a factory. Each factory instantiates its compute kernel **twice**, over the *disjoint* `core_group_1` / `core_group_2` core ranges (the per-group-CTA work split) — so each node sees exactly one compute instance.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `5fcf2963d45 2026-07-29 docs(metal_2.0): follow main's MetalV2FactoryConcept -> ProgramSpecFactoryConcept rename`

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_mean` |
| **Overall** | **GREEN — conditional.** Every gate clears on code evidence. One readiness-sheet-only conjunct (`Is safe to port?`) could not be read: the Google Drive connector cannot be authorized from a non-interactive session. See *Readiness-sheet availability* below. |
| **DOps / Factories** | `MorehMeanOperation` → `MorehMeanHFactory`, `MorehMeanNCFactory`, `MorehMeanWFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 8 kernels + all 5 donor headers. Kernels are on `DataflowBuffer` / `Noc` / `TensorAccessor`, i.e. *ahead of* the Device 2.0 `CircularBuffer` baseline. Zero Device 1.0 idioms, zero CB-index holdovers. |
| *Prereqs* — Cross-op escapes | **Ok** — 5 donor headers, every call shape ✓. No borrowed kernel `.cpp` files at all. |
| *Feature Support* — overall | **GREEN** — all four Appendix A entries `N/A` |
| *Feature Support* — Variadic-CTA | **Ok** — fixed `tensor_args_t`; all CTAs read at constexpr offsets |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes on all six code-checkable conjuncts**; `Is safe to port?` **unverified** (sheet unreachable) |
| *TTNN Readiness* — Concept (current) | `descriptor` (×3 factories) — verified: `create_descriptor()` → `tt::tt_metal::ProgramDescriptor`, `moreh_mean_device_operation.hpp:34-53` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — no `WorkloadDescriptor` factory |
| *TTNN Readiness* — Is safe to port? | **Unverified** (sheet-only, expert-judgment axis — not re-derivable by the auditor). Strong corroborating code evidence that the usual failure mode is absent — see *Gate detail*. |
| *TTNN Readiness* — Custom hash | **No** — no `compute_program_hash` override anywhere in the op |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** — hook absent from the device-op |
| *TTNN Readiness* — `override_runtime_arguments` | **No** — method absent from all three factories and the device-op |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** — `moreh_mean_nanobind.cpp:19-31` binds only `ttnn::moreh_mean`; no internals exposed |
| *TTNN Readiness* — Op-owned tensors | **No** — `descriptor` concept cannot carry them; no `buffers` vector |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | **none** — no `->address()` anywhere in the op; every offset rides a separate scalar tile-index arg |
| *Port work* — Tensor bindings (per binding) | **Case 1 ×6** (2 bindings × 3 factories), all via `TensorAccessor`. Delivered today by the `Buffer*`-binding form. |
| *Port work* — TensorParameter relaxation | Expected **none** (no custom hash ⇒ no relaxation); sheet cell unverified |
| *Port work* — TensorAccessor 3rd arg | **none** — all 5 accessor sites are 2-arg; the subject does not fire |
| *Port work* — CB endpoints | **legal + self-loop** — 10 plain 1:1, 6 self-loop (config-dependent for 2). No multi-binding, no dead CB. |

**CB endpoints** are dispositions, not gates: every out-of-window CB here resolves to a **self-loop** (one toucher). No CB needs the multi-binding advanced option, and none is dead.

---

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`), with one condition stated on its face.

No blocker was found in this op. All five gate-bearing subjects clear on direct code evidence:

- **Device 2.0** — clean, and notably *ahead* of the baseline: the kernels already use the Metal 2.0 `DataflowBuffer` object rather than the Device 2.0 `CircularBuffer` wrapper, so the port's kernel-side work reduces mostly to swapping the DFB constructor argument from a raw `tt::CBIndex` to a `dfb::name` binding token.
- **Feature compatibility** — all four Appendix A features absent.
- **TTNN factory concept** — three plain `descriptor` factories; all six shape/hook conjuncts confirmed clean in code.
- **Offset base pointers** — no host-folded offsets; the op never computes a device address on the host at all.
- **TensorAccessor 3rd argument** — no site passes a 3rd argument.

**The one condition.** The `Is able to port?` gate has a seventh conjunct, `Is safe to port?`, that lives only on the readiness sheet and that the recipe explicitly forbids the auditor from re-deriving ("*This is the readiness-sheet owner's expert judgment — trust it. Do not try to re-derive 'did the migration introduce a subtle bug.'*"). That cell could not be read — see below. **Confirm it before the port starts.**

### Readiness-sheet availability

The per-factory readiness sheet (`analyses/ttnn_op_porting_readiness.md`) could **not** be fetched this run. The fetch requires the claude.ai Google Drive MCP connector, which is not authorized in this environment, and — per that doc — *"You cannot authorize it from inside a session."* This session is additionally non-interactive, so the OAuth flow cannot run at all. No stale local CSV was used (none exists, and the doc forbids relying on one).

This is a **data-availability gap, not a "spreadsheet is broken" finding** — the sheet may well carry three clean rows for this op; it simply could not be reached. It is reported here rather than converted into a RED, because every conjunct the auditor is *permitted* to check is clean and no code evidence points at a blocker.

**Two cells are outstanding**, for each of the three factory rows:

1. **`Is safe to port?`** — the gate conjunct. Corroborating code evidence that its usual failure mode is absent is in *Gate detail* below, but the call is the sheet owner's.
2. **`TensorParameter relaxation`** — PORT WORK if non-`none`. Expected `none`: the recipe notes a relaxation-bearing op has a custom hash, and this op has none.

Also unconfirmed by the usual cross-check: the **factory-set match** (three sheet rows ↔ three code factories). The code side is settled — exactly three factories, named above.

---

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN on every code-checkable conjunct.** The sheet lookup that normally supplies this verdict was unavailable (above), so the conjuncts were verified directly instead:

  | Conjunct | Verdict | Evidence |
  |---|---|---|
  | `Concept == descriptor` | ✓ | `moreh_mean_device_operation.hpp:34-53` — three factories, each a `static ProgramDescriptor create_descriptor(...)`; no mesh-workload return, no `create()`/`override_runtime_arguments()` pair |
  | `Custom hash == no` | ✓ | no `compute_program_hash` in the op directory; the device-op declares only `validate_tensors` / `select_program_factory` / `validate_on_program_cache_miss` / `compute_output_specs` / `create_output_tensors` (`moreh_mean_device_operation.hpp:57-61`) |
  | `get_dynamic_runtime_args == no` | ✓ | hook absent from the device-op |
  | `override_runtime_args == no` | ✓ | method absent from device-op and all three factories |
  | `Pybind descriptor == no` | ✓ | `moreh_mean_nanobind.cpp:19-31` — `bind_function<"moreh_mean">` over `&ttnn::moreh_mean` only |
  | `Is safe to port?` | **unverified** | sheet-only; see corroboration below |

  **Corroboration on the `safe` axis (not a substitute for the sheet cell).** That axis most often fails on a *smuggled pointer* — a device pointer riding an RTA without being marked, which then silently mis-patches on cache hits. That failure mode is **structurally absent here**: no factory ever calls `->address()`. All six buffer arguments are passed as `Buffer*` objects into `KernelDescriptor::emplace_runtime_args` (`..._h_program_factory.cpp:212-216`, `..._w_program_factory.cpp:219-222`, `..._nc_program_factory.cpp:191-201`), which the framework auto-registers as `BufferBinding`s and patches on cache hit (`tt_metal/api/tt-metalium/program_descriptors.hpp:114-118`, `:170-203` — *"automatically registering any `Buffer*` entries"*). This is exactly the shape the recipe calls the framework's sanctioned interim hack: *"correct-on-cache-hit today — it is **not** the silent-wrong hazard."* The other `safe` failure mode, a pybound `create_descriptor`, is likewise absent.

  Cross-column invariants: consistent. No `get_dynamic_runtime_args` on a non-`descriptor` concept; no op-owned tensors on a `descriptor` concept.

- **Device 2.0 (every kernel used):** **GREEN.** No violations table — there are no violations.

  All 8 kernels use `Noc` for data movement, `DataflowBuffer` for buffer handling, and `TensorAccessor` for addressing. A targeted grep for Device 1.0 idioms across `device/kernels/` returned nothing: no `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedPow2AddrGen*`, no raw `noc_async_read(` / `noc_async_write(`, no `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`, no free-function `get_write_ptr(` / `get_read_ptr(`.

  The one free function in use is **`get_tile_size(cb_id)`** — `reader_moreh_mean_h.cpp:50`, `reader_moreh_mean_w.cpp:38`, `reader_moreh_mean_nc.cpp:43`, `writer_moreh_mean_nc.cpp:25`, `writer_moreh_mean_unary_interleaved_start_id.cpp:24`. This is **sanctioned** by the Device 2.0 Green bullet and is *not* a holdover. It is, separately, port work: `DataflowBuffer` exposes `get_tile_size()` as a method (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:167`), and kernel-side whitelist rule 7 moves the lookup onto the object.

  **Worth flagging as better-than-required:** the kernels are on `DataflowBuffer` (`api/dataflow/dataflow_buffer.h`), the Metal 2.0 successor to the Device 2.0 `CircularBuffer` wrapper (`api/dataflow/circular_buffer.h`) that the Device 2.0 migration guide documents. They currently use the low-level `DataflowBuffer(uint16_t)` constructor with a raw `tt::CBIndex`; the port swaps that argument for a `dfb::name` binding token via the `DataflowBuffer(DFBAccessor)` constructor (`dataflow_buffer.h:72`). The object, its methods, and every call site stay as they are.

  Donor headers were checked to the same standard — see *Out-of-directory coupling* under Team-only. All clean.

- **Feature compatibility:** all four Appendix A entries scanned against both host and kernel code.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | **N/A** | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `global_circular_buffer.hpp` include. All 16 `CBDescriptor` literals across the three factories leave `.global_circular_buffer` unset (default `nullptr`). No `remote_index(` / `remote_cb_*` / `remote_circular_buffer.h` idiom anywhere. |
  | CBDescriptor `address_offset` (non-zero) | **N/A** | No `.address_offset` assignment in any of the 16 `CBDescriptor` literals (default zero); no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. The runtime-team-consultation message does not apply. |
  | GlobalSemaphore | **N/A** | The op uses **no semaphores of any kind** — a grep for `[Ss]emaphore` across the whole op directory returns nothing. Neither `GlobalSemaphore` nor plain `Semaphore`. |
  | Variable-count compile-time arguments (CTA varargs) | **N/A** | Op-level cue absent: `tensor_args_t` is a fixed `{const Tensor& input, const std::optional<Tensor>& output}` (`moreh_mean_device_operation.hpp:26-29`) — no `std::vector<Tensor>`. Kernel-level decider absent: every `get_compile_time_arg_val` call uses a literal index (`0`–`3`) or the constexpr `src_args.next_compile_time_args_offset()` (`reader_moreh_mean_h.cpp:32`, `reader_moreh_mean_w.cpp:17`). No CTA read under a runtime-varying index. |

- **CB endpoints (GATE-free):** every CB is either plain 1:1 or a one-toucher self-loop. **No multi-binding, no dead CB.** Census is per node, per `(CB, config)`; each node hosts one reader, one writer, and one compute instance (the two compute `KernelDescriptor`s cover *disjoint* core groups, so they are not a dual-instance work-split).

  **`MorehMeanHFactory`** — CBs at `moreh_mean_h_program_factory.cpp:62-115`

  | CB | Reader | Writer | Compute | Census | Disposition |
  |---|---|---|---|---|---|
  | `c_0` in0 | P `reader_moreh_mean_h.cpp:57,60` | — | C via `reduce<cb_input,…>` + `:61,78` | 1P + 1C | plain 1:1 ✓ |
  | `c_2` scaler | P `reader_moreh_mean_h.cpp:33-37` → `reduce_helpers_dataflow.inl:163,203` | — | C `moreh_mean_h.cpp:35,100` | 1P + 1C | plain 1:1 ✓ |
  | `c_3` mask_h | P only under `DO_MASK_H` — `reader_moreh_mean_h.cpp:41-44` | — | binds unconditionally `moreh_mean_h.cpp:26`; FIFO under `if constexpr (do_mask_h)` `:42,98` | **`do_mask_h`:** 1P + 1C · **`!do_mask_h`:** 1 toucher (compute, role-free) | 1:1 ✓ · **self-loop** |
  | `c_24` accum_dst | — | — | P `moreh_mean_h.cpp:54` (reduce output) + C `:84,92` (`Accumulate::at`) | 1 toucher | **self-loop** |
  | `c_25` masked_input | — | — | P `moreh_mean_h.cpp:72,76` + C `:81` (reduce input) | 1 toucher | **self-loop** |
  | `c_16` out | — | C `writer_moreh_mean_unary_interleaved_start_id.cpp:29,33` | P `moreh_mean_h.cpp:81,89` (reduce output) | 1P + 1C | plain 1:1 ✓ |

  **`MorehMeanWFactory`** — CBs at `moreh_mean_w_program_factory.cpp:62-115`

  | CB | Reader | Writer | Compute | Census | Disposition |
  |---|---|---|---|---|---|
  | `c_0` in0 | P `reader_moreh_mean_w.cpp:41,44` | — | C `moreh_mean_w.cpp:57,63,76,94,99,120` | 1P + 1C | plain 1:1 ✓ |
  | `c_2` scaler | P `reader_moreh_mean_w.cpp:21` → `generate_mm_scaler.hpp:13,30` | — | C `moreh_mean_w.cpp:36,130` | 1P + 1C | plain 1:1 ✓ |
  | `c_3` mask_w | P only under `DO_MASK_W` — `reader_moreh_mean_w.cpp:25-27` | — | binds unconditionally `moreh_mean_w.cpp:25`; FIFO under `if (do_mask_w)` `:43,128` | **`do_mask_w`:** 1P + 1C · **`!do_mask_w`:** 1 toucher (compute, role-free) | 1:1 ✓ · **self-loop** |
  | `c_24` accum_dst | — | — | P `moreh_mean_w.cpp:67,71` + C `:101,122` | 1 toucher | **self-loop** |
  | `c_25` masked_input | — | — | P `moreh_mean_w.cpp:88,92` + C `:99,120` (via the `cb_input` reassign at `:95`) | 1 toucher | **self-loop** |
  | `c_16` out | — | C `writer_moreh_mean_unary_interleaved_start_id.cpp:29,33` | P `moreh_mean_w.cpp:114,118` | 1P + 1C | plain 1:1 ✓ |

  **`MorehMeanNCFactory`** — CBs at `moreh_mean_nc_program_factory.cpp:68-112`

  | CB | Reader | Writer | Compute | Census | Disposition |
  |---|---|---|---|---|---|
  | `c_0` in0 | P `reader_moreh_mean_nc.cpp:52,55` | — | C `moreh_mean_nc.cpp:43,52` | 1P + 1C | plain 1:1 ✓ |
  | `c_1` zero tile | P `reader_moreh_mean_nc.cpp:31-32` → `moreh_common.hpp:98,110` | — | C `moreh_mean_nc.cpp:34` (`wait_front`, never popped — see anomaly 8) | 1P + 1C | plain 1:1 ✓ |
  | `c_2` scalar 1/N | P `reader_moreh_mean_nc.cpp:35-36` | — | C `moreh_mean_nc.cpp:35` (`wait_front`, never popped) | 1P + 1C | plain 1:1 ✓ |
  | `c_24` intermed0 | — | — | P `moreh_mean_nc.cpp:58,62` + C `:45,54,69,79` | 1 toucher | **self-loop** |
  | `c_16` out | — | C `writer_moreh_mean_nc.cpp:29,33` | P `moreh_mean_nc.cpp:74,78` | 1P + 1C | plain 1:1 ✓ |

  **Hidden-second-writer hunt: negative, with a structural reason.** Every `get_write_ptr()` call in play belongs to a donor producer operating on the CB it is *already* the FIFO producer of, bracketed by its own `reserve_back` / `push_back` (`generate_mm_scaler.hpp:16`, `moreh_common.hpp:110`, `moreh_common.hpp:194`, `reduce_helpers_dataflow.inl:164`) — a producer peeking at its own buffer is one toucher, not two. No `fifo_wr_ptr` / `evil_set_write_ptr` / `evil_set_read_ptr` anywhere. And the semaphore-gated raw co-fill that this face describes is **impossible in this op**: it has no semaphores at all, so there is no coordination channel a hidden co-filler could use.

  **On the two config-dependent self-loops (`c_3`, and `c_25` in the H factory).** Under the unmasked config the compute kernel still *constructs* the DFB object unconditionally (`moreh_mean_h.cpp:26,29`, `moreh_mean_w.cpp:25,29`) while its FIFO calls are compiled out. I counted construction as a touch — in Metal 2.0 the named token has to be bound for the kernel to name it — which makes these **1-toucher self-loops, not dead CBs.** This distinction is load-bearing: reading them as dead and dropping the allocation would break the *masked* config, where the reader genuinely fills `c_3`. The recipe does not cover bind-without-FIFO-access; see Recipe notes 4.

- **Offset base pointers:** **GREEN — no fold, and none possible.** The op computes **no device addresses on the host at all**: `->address()` does not appear anywhere in the directory. Each of the six buffer arguments is passed as a `Buffer*` (base only, framework-patched), and every offset travels as a *separate scalar* consumed on-device as a page index:

  - H reader, `..._h_program_factory.cpp:214` — `(tile_offset / Wt * HtWt) + (tile_offset % Wt)`: arithmetic on a **tile index**, not an address. Lands in `col_start_tile_id` and reaches the NoC only as `{.page_id = curr_id}` (`reader_moreh_mean_h.cpp:58`).
  - W reader, `..._w_program_factory.cpp:219` — `tile_offset` → `{.page_id = i}` (`reader_moreh_mean_w.cpp:42`).
  - NC reader, `..._nc_program_factory.cpp:191-199` — `tile_offset` → `{.page_id = read_tile_id}` (`reader_moreh_mean_nc.cpp:53`).
  - Both writers — `tile_offset` (H/NC) or `tile_offset / out_dim_divider` (W) → `{.page_id = …}`.

  This is the already-split shape the recipe describes as the clean outcome. Not Type 1 (no raw offset arg — no arg is ever used as a NoC address), not Type 2 (no accessor is handed an offset base: all five accessors take a bare `Buffer*`-delivered base), not Type 3 (no `address_offset`), not Type 4 (no `ttnn::narrow`, no interior-base `MeshBuffer::create`). Cross-checked against the offset-base-pointer triage analysis (`2026-07-19_offset_base_pointers.md`), a dated prior: `moreh_mean` is **not** in its tables — consistent with this scan (*no fold, op not in tables → clean*).

- **TensorAccessor 3rd argument:** **GREEN — the subject does not fire.** All five construction sites pass exactly two arguments, so there is no page-size override to classify:

  `reader_moreh_mean_h.cpp:46` · `reader_moreh_mean_w.cpp:34` · `reader_moreh_mean_nc.cpp:39` · `writer_moreh_mean_nc.cpp:21` · `writer_moreh_mean_unary_interleaved_start_id.cpp:20`

  Cross-checked against the 3rd-arg triage analysis (`2026-07-06_tensor_accessor_3rd_arg_triage.md`), a dated prior: it lists `moreh_fold` and `moreh_getitem` from this family but **not** `moreh_mean` — consistent with this scan. No new site has appeared since.

---

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory) — **all Case 1**, all mechanical. In each case the base arrives today as a `Buffer*` pushed into `emplace_runtime_args` (delivery mechanism only) and is consumed exclusively through a `TensorAccessor`:

  | Factory | Binding | Host site (RTA idx 0) | Kernel accessor | Case |
  |---|---|---|---|---|
  | H | `input` | `..._h_program_factory.cpp:214` | `reader_moreh_mean_h.cpp:12,46` | 1 |
  | H | `output` | `..._h_program_factory.cpp:216` | `writer_moreh_mean_unary_interleaved_start_id.cpp:11,20` | 1 |
  | W | `input` | `..._w_program_factory.cpp:219` | `reader_moreh_mean_w.cpp:12,34` | 1 |
  | W | `output` | `..._w_program_factory.cpp:221-222` | `writer_moreh_mean_unary_interleaved_start_id.cpp:11,20` | 1 |
  | NC | `input` | `..._nc_program_factory.cpp:191-199` | `reader_moreh_mean_nc.cpp:13,39` | 1 |
  | NC | `output` | `..._nc_program_factory.cpp:201` | `writer_moreh_mean_nc.cpp:13,21` | 1 |

  No Case 2 (no kernel does raw address arithmetic — every access goes through the accessor), and no borrowed-memory DFB reads (no `set_globally_allocated_address` in the op, so the causal-link gate never applies). Per binding the port replaces the `Buffer*` RTA + the `TensorAccessorArgs(...).append_to(...)` CTA plumbing with one `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)`.

- **TensorParameter relaxation:** expected **none** — the op has no custom hash, and the recipe notes a relaxation co-occurs with one. Sheet cell unverified; confirm.
- **TensorAccessor 3rd arg:** **none** — no site passes one.
- **CB endpoints:** **self-loop** on `H/c_24`, `H/c_25`, `W/c_24`, `W/c_25`, `NC/c_24` (all configs) and on `H/c_3`, `W/c_3` (unmasked config only — 1:1 when masked). All 10 remaining CBs are plain 1:1. No multi-binding flag, no dead-CB drop.
- **Kernel-side, mechanical:** move `get_tile_size(cb_id)` onto the object (`dfb.get_tile_size()`) at the 5 sites listed under Device 2.0; swap each `DataflowBuffer(tt::CBIndex::c_N)` construction for `DataflowBuffer(dfb::name)`.

---

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** **none.** The hidden-second-writer hunt came back negative for a structural reason (no semaphores in the op), and no CB has ≥3 touchers or two kernels locked to one FIFO role.
- **Runtime-selected DFB in the W compute kernel.** `moreh_mean_w.cpp:21,51,95` keeps `cb_input` as a *mutable* variable that switches between `c_0` and `c_25` (`cb_masked_input`) at runtime, and constructs a throwaway `DataflowBuffer(cb_input)` at each use (`:57,63,76,78,94,99,120`). This is not a token-for-token substitution: both DFBs must be bound to the compute kernel, and the variable has to stay `uint32_t`-valued, relying on `DFBAccessor`'s constexpr `operator uint32_t()` (`dataflow_buffer.h:55`). It works — but a porter substituting `dfb::name` for `tt::CBIndex::c_N` mechanically will hit it.
- **Compute-kernel CTA names do not describe their values.** In both the H and W compute kernels, one CTA named after a tile-count actually carries the per-core work split. Since the port *names* CTAs by the variable a kernel unpacks them into, inferring the name from the kernel code here would produce a wrong one. See Misc anomaly 2 for the exact sites.
- **Cross-op / shared kernels:** the op **owns all 8 kernel files** and no other op instantiates any of them (the only external hits are in `ttnn/ttnn.egg-info/SOURCES.txt`, a packaging manifest). No `_metal2` fork exists or is needed; no sunset list; no cross-op coordination cost. The 5 donor **headers** are function-call escapes only — all shapes ✓, no donor rewrite required.
- **RTA varargs:** **none.** Every kernel reads a fixed set of RTAs as distinct fields. `reader_moreh_mean_nc.cpp:12-19` uses a running `i++` counter, but over a **fixed run of 7** reads at the top of the kernel — the recipe's explicit non-signal ("*a fixed run of reads via a running `arg_index++` … dissolves into named args*"). Name all seven.

---

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.** Five donor headers, every consumed function shape ✓ under the per-call table. No donor is pre-Device-2.0, so nothing here feeds the Device 2.0 gate. No `CircularBuffer&`-shaped parameter — the donors have already moved to `DataflowBuffer`.

- No Shape 4 (old-style addr-gen) in any donor.
- No `uint32_t sem_id` / `sem_addr` shapes — the op has no semaphores.
- No Shape 2 (`TensorAccessorArgs<N>`) or Shape 3 (CTA-offset NTTP) donor parameters.
- Every DFB-carrying parameter is either `DataflowBuffer` by value or `uint32_t` dfb-id — both reachable from `dfb::name` (see per-call detail).

**Summary table** — one row per (op kernel, donor file):

| Op kernel | Donor file | Class | Status |
|---|---|---|---|
| `reader_moreh_mean_h.cpp` | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | 3 — `ttnn/cpp/ttnn/kernel/` pool | ✓ |
| `reader_moreh_mean_h.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | 2 — official shared kernel lib | ✓ |
| `reader_moreh_mean_h.cpp` | `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/tensor/noc_traits.h` | 1 — `tt_metal/*` | ✓ no concern |
| `reader_moreh_mean_w.cpp` | `ttnn/cpp/ttnn/kernel/dataflow/generate_mm_scaler.hpp` | 3 | ✓ |
| `reader_moreh_mean_w.cpp` | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | 3 | ✓ (header included; no function consumed under `!DO_MASK_W`) |
| `reader_moreh_mean_w.cpp` | `api/dataflow/*`, `api/tensor/noc_traits.h` | 1 | ✓ no concern |
| `reader_moreh_mean_nc.cpp` | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | 3 | ✓ |
| `reader_moreh_mean_nc.cpp` | `api/debug/dprint.h`, `api/dataflow/*`, `api/tensor/noc_traits.h` | 1 | ✓ no concern (`dprint.h` unused — anomaly 6) |
| `writer_moreh_mean_nc.cpp` | `api/dataflow/*`, `api/tensor/noc_traits.h` | 1 | ✓ no concern |
| `writer_moreh_mean_unary_interleaved_start_id.cpp` | `api/dataflow/*`, `api/tensor/noc_traits.h` | 1 | ✓ no concern |
| `moreh_mean_h.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` | 2 | ✓ |
| `moreh_mean_h.cpp` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | 3 | ✓ |
| `moreh_mean_h.cpp` | `api/compute/*`, `api/dataflow/dataflow_buffer.h` | 1 | ✓ no concern |
| `moreh_mean_w.cpp` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | 3 | ✓ |
| `moreh_mean_w.cpp` | `api/compute/*`, `api/dataflow/dataflow_buffer.h` | 1 | ✓ no concern |
| `moreh_mean_nc.cpp` | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | 3 | ✓ |
| `moreh_mean_nc.cpp` | `api/compute/*`, `api/dataflow/dataflow_buffer.h` | 1 | ✓ no concern |

**Per-call detail** — every consumed donor function, by handle shape. (Included despite an all-✓ roll-up because the shapes are `DataflowBuffer`-typed, a spelling the recipe's table predates — see Recipe notes 2.)

| Donor function | Signature shape | Verdict |
|---|---|---|
| `generate_mask_h(DataflowBuffer, uint32_t)` — `moreh_common.hpp:183` | `DataflowBuffer` **by value** | ✓ — `dfb::name` is a `DFBAccessor`; `DataflowBuffer(DFBAccessor)` is an implicit converting ctor (`dataflow_buffer.h:72`), so `dfb::name` passes straight through |
| `generate_mask_w(DataflowBuffer, uint32_t)` — `moreh_common.hpp:223` | `DataflowBuffer` by value | ✓ same |
| `fill_cb_with_value(DataflowBuffer, uint32_t, int32_t)` — `moreh_common.hpp:98` | `DataflowBuffer` by value | ✓ same |
| `generate_mm_scaler(DataflowBuffer, uint32_t)` — `generate_mm_scaler.hpp:13` | `DataflowBuffer` by value | ✓ same |
| `calculate_and_prepare_reduce_scaler<dfb_id, PoolType, ReduceDim, reduce_factor>()` — `reduce_helpers_dataflow.hpp:88` | `uint32_t dfb_id` as **NTTP** | ✓ — `DFBAccessor`'s constexpr `operator uint32_t()` (`dataflow_buffer.h:55`) covers template-parameter position |
| `compute_kernel_lib::reduce<PoolType, ReduceDim, input_dfb_id, scaler_dfb_id, output_dfb_id, …>(…)` — `reduce_helpers_compute.hpp:392` | three `uint32_t` dfb-ids as **NTTPs** | ✓ same |
| `compute_kernel_lib::Accumulate::at(uint32_t cb, uint32_t iter, uint32_t dst)` — `reduce_helpers_compute.hpp:193` | `uint32_t` cb, **runtime** position | ✓ — the constexpr cast covers runtime position too |
| `pack_tile_with_dt(uint32_t, DataflowBuffer)` — compute `moreh_common.hpp:28` | `DataflowBuffer` by value | ✓ |
| `copy_tile_init_with_dt(DataflowBuffer, uint32_t)` — compute `moreh_common.hpp:35` | `DataflowBuffer` by value | ✓ |
| `add_tiles_init_with_dt(DataflowBuffer, DataflowBuffer)` — compute `moreh_common.hpp:42` | `DataflowBuffer` ×2 by value | ✓ |
| `mul_tiles_bcast_scalar_init_short_with_dt(DataflowBuffer, DataflowBuffer)` — compute `moreh_common.hpp:121` | `DataflowBuffer` ×2 by value | ✓ |

**Borrowed kernel files (file-path instantiation): none.** All 8 `kernel_source` paths point inside `moreh_mean/device/kernels/`. A reverse search for external instantiation of each file found only `ttnn/ttnn.egg-info/SOURCES.txt` (packaging manifest). No `_metal2` fork exists beside any of them, and none is needed — this port converts kernels it solely owns, so the shared-kernel fork convention does not apply and there is no sunset list.

Host-side note: the factories include `ttnn/cpp/ttnn/operations/moreh/moreh_helper_functions.hpp` but consume only the host work-split helper `split_work_to_cores_wt_core_range` (`:65`). They do **not** use that header's legacy `CreateCircularBuffer` helpers (`:137,143`) — all 16 CBs are built as inline `CBDescriptor` literals.

### Relaxation candidates

**None to mine** — the op has no custom hash, so there is no hash logic from which to infer which tensor properties it actually depends on.

### TTNN factory analysis

Sheet-derived facts could not be fetched (see *Readiness-sheet availability*); the following are the auditor's code-verified equivalents.

- **Op-owned tensors:** none. `descriptor` concept throughout; no `WorkloadDescriptor`, so no `buffers` vector exists to populate.
- **MeshWorkload need:** none — not a `WorkloadDescriptor` op, so the genuine-vs-artifact question does not arise.
- **Pybind `create_descriptor`:** absent (`moreh_mean_nanobind.cpp:19-31`).
- **Other risky pybind:** none. The nanobind surface is one function over `&ttnn::moreh_mean` with plain scalar/optional/`MemoryConfig`/`DeviceComputeKernelConfig` arguments; no factory or device-op internals are exposed.
- **Custom hash:** absent — the default hash applies, over the whole of `operation_attributes_t` (`moreh_mean_device_operation.hpp:19-25`; see anomaly 1).
- **`get_dynamic_runtime_args`:** absent.
- **`override_runtime_arguments`:** absent.
- **Target concept:** `ProgramSpecFactoryConcept`, no op-owned tensors, for all three factories.

---

## Misc anomalies  *(team-only, non-gating — route to the ops team; the port does not act on these)*

1. **Dead-but-hashed attribute — `divisor`.** `moreh_mean_device_operation.cpp:23` hard-rejects any value: `TT_FATAL(operation_attributes.divisor.has_value() == false, "divisor not supported yet.")`. The field nonetheless rides `operation_attributes_t` (`moreh_mean_device_operation.hpp:22`) and therefore the default program hash, and is still exposed through pybind (`moreh_mean_nanobind.cpp:28`). It can only ever hold `nullopt`, so it contributes a constant to the cache key — harmless, but it is an attribute that is forced yet hashed.

2. **Compute-kernel CTA names contradict the values passed.** In both cases a CTA named for a tile-count actually carries the per-core work split:
   - H: kernel reads CTA(1) into `Wt` (`moreh_mean_h.cpp:17`) and loops `for (wt < Wt)` (`:46`), but the factory passes `units_per_core_group_1` / `units_per_core_group_2` there (`moreh_mean_h_program_factory.cpp:164,185`). The real `Wt` is never given to the compute kernel.
   - W: kernel reads CTA(0) into `Ht` (`moreh_mean_w.cpp:16`) and loops `for (ht < Ht)` (`:47`), but the factory passes `units_per_core_group_N` there (`moreh_mean_w_program_factory.cpp:167,188`). Here CTA(1) *is* a genuine `Wt`, which makes the mismatch easier to miss.

   Functionally correct today. It matters for the port because Metal 2.0 names arguments after the variable the kernel unpacks them into — so the natural name is the wrong one. Also mirrored in each kernel's comments, which describe reducing over `Wt` / `Ht` tiles when the loop is over per-core units. (Surfaced to the porter as a heads-up.)

3. **Redundant RTA in the NC reader.** `moreh_mean_nc_program_factory.cpp:45-53` computes `input_tile_stride` and `inner_size` with two loops over the *same* index range, differing only in the seed (`HtWt` vs `1`) — so `input_tile_stride == HtWt * inner_size` identically. All three values are then sent as separate RTAs, indices 3, 5 and 6 (`:191-199`; read at `reader_moreh_mean_nc.cpp:16,18,19`). One of the three is derivable on-device from the other two.

4. **Unused destructured value — `packer_l1_acc`.** Bound from `get_compute_kernel_config_args` and never read, in all three factories: `moreh_mean_h_program_factory.cpp:52`, `moreh_mean_w_program_factory.cpp:52`, `moreh_mean_nc_program_factory.cpp:62`.

5. **`fp32_dest_acc_en` handled inconsistently across the three factories.** All three plumb the same `fp32_dest_acc_en` into `ComputeConfigDescriptor`, but the intermediate CB `c_24` is treated three different ways:
   - H: `c_24` allocated at `fp32_dest_acc_en_data_format` **and** `unpack_to_dest_mode[CBIndex::c_24] = UnpackToDestFp32` (`..._h_program_factory.cpp:89-97,154`).
   - W: `c_24` allocated at `fp32_dest_acc_en_data_format` but `unpack_to_dest_mode` left entirely `Default` (`..._w_program_factory.cpp:89-97,160`).
   - NC: `c_24` allocated at plain `cb_data_format`, never widened, though the factory does define `FP32_DEST_ACC_EN` for the kernel (`..._nc_program_factory.cpp:95-103,139`).

   Whether W and NC are under-configured or H is over-configured is a question for the op owner. Worth resolving *before* the port so the ported code doesn't cement an accident.

6. **Dead include.** `reader_moreh_mean_nc.cpp:5` includes `api/debug/dprint.h`; the file contains no `DPRINT`.

7. **`kernel_lib` sentinel DFB id aliases a live buffer.** `reduce_helpers_compute.inl:340-343` constructs the accumulator DFB from a lambda that returns `0` when accumulation is disabled: `DataflowBuffer accum_dfb([&]() -> uint32_t { if constexpr (enable_accumulation) { return accumulate.config.cb_accumulator; } else { return 0; } }())`. Index `0` is not a null sentinel — in all three `moreh_mean` factories it is the live input CB `c_0`. No FIFO op reaches it on that path (guarded at `:193`), so it is harmless today, but a "no accumulator" sentinel that names a real buffer is fragile. Routes to the `kernel_lib` owner, not the ops team.

8. **Scalar CBs waited on but never popped (NC compute).** `moreh_mean_nc.cpp:34-35` does `dfb_in1_obj.wait_front(onetile)` and `dfb_scalar_obj.wait_front(1)` with no matching `pop_front` anywhere in the kernel — unlike the H and W compute kernels, which do pop their scaler (`moreh_mean_h.cpp:100`, `moreh_mean_w.cpp:130`). Benign as written (the tiles are produced once and consumed for the kernel's lifetime), and it does not change the endpoint census. Noted only because it makes the NC factory's `c_1`/`c_2` FIFO usage asymmetric with its siblings'.

---

## Per-DeviceOperation attribution

Not applicable — the directory holds a single `DeviceOperation` (`MorehMeanOperation`). Per-*factory* findings differ only in the CB endpoint census and the tensor-binding table; both are broken out per factory above.

---

## Questions for the user

1. **Readiness-sheet cells (blocking confirmation, not blocking the port's design).** The Google Drive connector could not be authorized in this non-interactive session, so the sheet was not fetched. Please read the three `moreh_mean` factory rows in the *"Operations analysis"* sheet and confirm two cells on each:
   - **`Is safe to port?`** — the outstanding gate conjunct. Expected `yes`: the op never calls `->address()`, and every buffer pointer rides the framework-patched `Buffer*` binding (`..._h_program_factory.cpp:212-216`, `..._w_program_factory.cpp:219-222`, `..._nc_program_factory.cpp:191-201`), so the smuggled-pointer failure mode this column usually catches is structurally absent. But the call is yours/the sheet owner's, not the auditor's.
   - **`TensorParameter relaxation`** — expected `none` (no custom hash). If it reads anything else, that is PORT WORK the brief is currently missing, and the recipe's custom-hash gate would also be in play.

   Also worth a glance: that the sheet has **exactly three** rows for this op, matching `MorehMeanHFactory` / `MorehMeanNCFactory` / `MorehMeanWFactory`.

2. **`fp32_dest_acc_en` asymmetry (anomaly 5).** The three factories configure the `c_24` intermediate CB three different ways for the same flag. This is pre-existing and the port would preserve it verbatim, but if it is a latent bug it is much cheaper to fix before the port than after. Is the divergence intentional?

---

## Recipe notes

1. **No procedure for "readiness sheet unreachable," as distinct from "sheet broken."** `ttnn_op_porting_readiness.md` states the Drive connector *"cannot be authorized from inside a session"* — so any headless, sandboxed, or non-interactive audit run structurally cannot fetch it, and can therefore never close the `Is safe to port?` conjunct. The recipe covers two failure modes — a code-vs-sheet conflict and a missing op row — and routes both to the sheet owner as a GATE. It does not cover *"the auditor could not reach the sheet at all,"* which is a different thing: the data may be perfectly fine and clean. Converting it into a RED would misroute work to the sheet owner for an op with no actual blocker; silently ignoring it would manufacture a GREEN the auditor cannot support. I took a third path — GREEN with the conjunct explicitly marked unverified, a Question to the user, and the corroborating code evidence spelled out — but that was an unacknowledged judgment call. Suggest the recipe name this case and prescribe the outcome. A cheap alternative worth considering: let the launching human paste the op's sheet rows into the prompt when the connector is unavailable.

2. **The out-of-directory shape table has no `DataflowBuffer` row.** Its CB-wrapper row is `CircularBuffer` / `CircularBuffer&` → *"⭐ ⚠ flag … Op-by-op porting + DFB-replaces-CB on the consumer side leaves no clean per-op story today."* On `main`, the shared kernel pools have already completed exactly that replacement: all five donors here take `DataflowBuffer` **by value** (`moreh_common.hpp:98,183,223`, `generate_mm_scaler.hpp:13`, compute `moreh_common.hpp:28,35,42,121`). That is the *clean* case — `dfb::name` is a `DFBAccessor` and `DataflowBuffer(DFBAccessor)` is an implicit converting constructor (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:72`) — but an auditor pattern-matching "CB-wrapper-typed parameter → ⭐ flag" on shape alone would mis-flag it and hand the porter a spurious cross-team discussion. Suggest adding a `DataflowBuffer` / `DataflowBuffer&` row marked ✓, and noting that the `CircularBuffer` row now refers specifically to the older wrapper in `api/dataflow/circular_buffer.h`.

3. **The Device 2.0 Green bullet doesn't say where kernels *already past* Device 2.0 land.** The Device 2.0 migration guide documents the wrapper as `CircularBuffer` from `api/dataflow/circular_buffer.h`; this op's kernels use `DataflowBuffer` from `api/dataflow/dataflow_buffer.h` — the Metal 2.0 successor — via its low-level `uint16_t` constructor. That is strictly *ahead of* the baseline the gate demands, and I scored it GREEN without hesitation. But the Green bullet is phrased in terms of `Noc` / `CircularBuffer` / the sanctioned free functions, so a literal reader could ask whether a non-`CircularBuffer` kernel has "completed Device 2.0." One sentence — *DFB-based kernels satisfy the gate; they are past it, not short of it* — would remove the doubt. (Related: the audit's own guidance that `get_tile_size(cb_id)` stays sanctioned *"as long as Device 2.0 uses it"* was easy to apply and correctly kept this op out of holdover-RED.)

4. **The CB endpoint census has no rule for bind-without-FIFO-access.** Both mask CBs here (`c_3`, and `c_25` in the H factory) sit behind a compile-time-disabled branch in the unmasked config: the compute kernel **constructs** the DFB object unconditionally (`moreh_mean_h.cpp:26,29`; `moreh_mean_w.cpp:25,29`) while its `wait_front` / `pop_front` are compiled out by `if constexpr (do_mask_h)` or a constexpr-false `if (do_mask_w)`. The recipe defines an endpoint as a kernel that FIFO-produces, FIFO-consumes, or accesses by raw pointer — construction alone is none of those. Read strictly, that makes these CBs `(0, 0)` **dead** in the unmasked config, and the Dead-CB section says a confirmed dead CB *must* be dropped. Dropping them would break the masked config, where the reader genuinely fills `c_3`. I counted construction as a touch (in Metal 2.0 the named token must be bound for the kernel to name it), yielding 1-toucher self-loops. Worth an explicit rule — something like *a kernel that binds a DFB counts as a role-free toucher even where its FIFO ops are compile-time-eliminated; classify per config and never drop a CB that is live in a sibling config.* The Dead-CB section's "distrust a `(0,0)` result" instinct pointed the right way, but its enumerated indirection paths (a CTA handed to a helper, an aliased index, an uninspected config) don't include this one — a `#ifdef`/`if constexpr`-gated access in a config you *did* inspect.

5. **Minor, and it worked as intended:** both dated triage docs (`2026-07-19_offset_base_pointers.md`, `2026-07-06_tensor_accessor_3rd_arg_triage.md`) omit `moreh_mean`, and both scans came back independently clean — so the "prior, not authority" framing cost nothing here. Recording the agreement since the recipe asks for disagreements to be noted; silence in the tables was not load-bearing either way.
